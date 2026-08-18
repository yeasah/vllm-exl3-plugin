#!/usr/bin/env python3
"""What is in this checkpoint, and what will it cost to serve — before downloading it.

    tools/checkpoint_survey.py <snapshot-dir-or-hf-repo>
    tools/checkpoint_survey.py <hf-repo> --remote --revision 4.00bpw

Reads safetensors headers only — no weights, no GPU, no vLLM — and sorts every stored
tensor into what this plugin knows how to serve, what it deliberately never loads, and
**what it does not recognize at all**. That last bucket is the point: an unfamiliar tensor
family is a reliable predictor of real work, and finding it before spending the bandwidth
is worth more than finding it afterwards.

`--remote` answers the same question for a checkpoint that is not downloaded, via
`HfApi.get_safetensors_metadata` — the Hub's supported call for reading tensor metadata
without fetching weights.

**It screens storage, not architecture.** A clean report means nothing here is stored in a
way the plugin cannot read. It says nothing about whether vLLM implements the architecture,
whether hybrid or MoE layers need handling, or whether the attention head dim has a flash
path. A dirty report reliably predicts format work; a clean one only rules that class out.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vllm_exl3_plugin import format  # noqa: E402
from tp_preflight import resolve  # noqa: E402

#: Stored for a model this plugin serves text-only; real bytes that never reach VRAM.
NEVER_LOADED = ("vision", "visual", "mm_projector", "multi_modal", "audio_tower", "mtp")

#: Below this, an unrecognized tensor is a curiosity rather than a workload: state-space
#: layers carry a handful of per-head scalars that no quantization format touches. Above
#: it, something substantial is stored in a way this plugin cannot read.
NOTEWORTHY = 1 << 20

#: Tiny and kept at full precision by every format.
SMALL_FRY = (".bias", "_bias", "norm", "router", "gate_inp")

_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1,
                "I16": 2, "I32": 4, "I64": 8, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def local_tensors(d: str) -> dict:
    out = {}
    for path in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            a, b = meta["data_offsets"]
            out[name] = (meta["shape"], meta["dtype"], b - a)
    return out


def remote_tensors(repo: str, revision: str | None) -> dict:
    from huggingface_hub import HfApi

    meta = HfApi().get_safetensors_metadata(repo, revision=revision)
    out = {}
    for f in meta.files_metadata.values():
        for name, t in f.tensors.items():
            n = 1
            for s in t.shape:
                n *= s
            out[name] = (list(t.shape), t.dtype, n * _DTYPE_BYTES.get(t.dtype.upper(), 2))
    return out


def module_of(name: str) -> tuple[str, str | None]:
    for suf in format.EXL3_SUFFIXES + format.BLOCKQ_SUFFIXES + (".weight",):
        if name.endswith(suf):
            return name[: -len(suf)], suf.lstrip(".")
    return name, None


def survey(tensors: dict) -> dict:
    modules: dict[str, dict] = {}
    for name, (shape, dtype, nbytes) in tensors.items():
        key, suffix = module_of(name)
        m = modules.setdefault(key, {"bytes": 0, "suffixes": set(), "shape": None,
                                     "unknown": []})
        m["bytes"] += nbytes
        if suffix:
            m["suffixes"].add(suffix)
            if suffix in ("weight", "bq_q"):
                m["shape"] = shape
        else:
            # No recognized suffix at all. Common and harmless for state-space layers,
            # whose parameters *are* the tensor (`A_log`, `D`, `dt_bias`, `conv1d`),
            # so judge these by size rather than by name.
            m["unknown"].append((name, shape, nbytes))
    buckets = {k: {"bytes": 0, "modules": []} for k in
               ("exl3", "blockq", "dense", "small", "never_loaded", "unrecognized")}

    def put(b, key, nbytes):
        buckets[b]["bytes"] += nbytes
        buckets[b]["modules"].append(key)

    for key, m in modules.items():
        if any(k in key for k in NEVER_LOADED):
            put("never_loaded", key, m["bytes"])
        elif "trellis" in m["suffixes"]:
            put("exl3", key, m["bytes"])
        elif "bq_q" in m["suffixes"]:
            put("blockq", key, m["bytes"])
        elif any(k in key for k in SMALL_FRY) or (m["shape"] and len(m["shape"]) < 2):
            put("small", key, m["bytes"])
        elif m["unknown"] and m["bytes"] >= NOTEWORTHY:
            put("unrecognized", key, m["bytes"])
        elif m["unknown"]:
            put("small", key, m["bytes"])
        elif "weight" in m["suffixes"]:
            put("dense", key, m["bytes"])
        else:
            put("unrecognized", key, m["bytes"])
    return {"modules": modules, "buckets": buckets}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target")
    ap.add_argument("--remote", action="store_true", help="read metadata from the Hub")
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    if args.remote:
        tensors = remote_tensors(args.target, args.revision)
        where = f"{args.target}@{args.revision or 'main'} (hub metadata)"
        config = {}
    else:
        d = resolve(args.target)
        tensors = local_tensors(d)
        where = d
        try:
            config = json.load(open(os.path.join(d, "config.json")))
        except OSError:
            config = {}

    s = survey(tensors)
    G = 2**30
    total = sum(b["bytes"] for b in s["buckets"].values())
    print(f" -- {where}")
    print(f" -- {len(tensors)} tensors, {total / G:.2f} GiB\n")

    labels = {
        "exl3": "EXL3-quantized modules",
        "blockq": "block-quantized embedding",
        "dense": "dense (unquantized) tensors",
        "small": "norms, biases, router gates",
        "never_loaded": "never loaded when serving text",
        "unrecognized": "NOT RECOGNIZED",
    }
    for k, label in labels.items():
        b = s["buckets"][k]
        if not b["modules"]:
            continue
        print(f"{b['bytes'] / G:8.3f} GiB  {len(b['modules']):5d} modules  {label}")
        if k in ("dense", "never_loaded", "unrecognized"):
            for key in sorted(b["modules"], key=lambda x: -s["modules"][x]["bytes"])[:6]:
                print(f"{s['modules'][key]['bytes'] / G:8.3f} GiB          {key}")

    # The embed/head tax this project exists to remove
    embed = next((k for k in s["modules"] if k.endswith("embed_tokens")), None)
    head = next((k for k in s["modules"] if k.endswith("lm_head")), None)
    tied = bool(config.get("tie_word_embeddings",
                           (config.get("text_config") or {}).get("tie_word_embeddings")))
    if embed:
        m = s["modules"][embed]
        packed = "bq_q" in m["suffixes"]
        share = m["bytes"] / total * 100
        print(f"\n embedding: {m['bytes'] / G:.3f} GiB ({share:.1f}% of the checkpoint), "
              f"{'block-quantized already' if packed else 'DENSE'}"
              f"{', tied' if tied else ''}")
        # Only advise on the embedding if the body is something this plugin serves;
        # otherwise it reads as a plan for a checkpoint that will not load at all.
        is_exl3 = bool(s["buckets"]["exl3"]["modules"])
        if not is_exl3:
            print("            -> not an EXL3 body, so the embedding is moot here")
        elif not packed and not tied:
            print("            -> tools/quantize_embedding.py applies "
                  f"(~{m['bytes'] / G * (1 - 4.53 / 16):.2f} GiB recoverable)")
        elif not packed and tied:
            print("            -> served from the quantized lm_head, no tooling needed")
    if head and tied:
        print(f" lm_head:   {s['modules'][head]['bytes'] / G:.3f} GiB, redundant "
              "(tied model; never loaded)")

    families: dict[str, list[int]] = {}
    for m in s["modules"].values():
        for name, shape, nbytes in m["unknown"]:
            fam = name.split(".")[-1]
            e = families.setdefault(fam, [0, 0])
            e[0] += 1
            e[1] += nbytes
    if families:
        print("\n unrecognized tensor families (no suffix this plugin knows):")
        for fam, (n, nbytes) in sorted(families.items(), key=lambda kv: -kv[1][1]):
            note = "" if nbytes >= NOTEWORTHY else "  (negligible)"
            print(f"   {fam:24s} x{n:<5d} {nbytes / 2**20:8.2f} MiB{note}")

    bad = s["buckets"]["unrecognized"]
    print()
    if bad["modules"]:
        print(f" !! {bad['bytes'] / G:.3f} GiB across {len(bad['modules'])} modules is "
              "stored in a way this plugin does not recognize.")
        print("    Expect real work before this checkpoint serves.")
    else:
        print(" -- every stored tensor is recognized. Note this screens *storage* only:")
        print("    architecture support, hybrid/MoE handling and attention head dims are")
        print("    separate questions. See tools/tp_preflight.py for splittability.")


if __name__ == "__main__":
    main()
