#!/usr/bin/env python3
"""How many bytes is a checkpoint's media encoder — before downloading it.

    tools/encoder_census.py <hf-repo>[@revision] ...
    tools/encoder_census.py --defaults

Reads safetensors metadata from the Hub (`HfApi.get_safetensors_metadata`, the
supported call for tensor metadata without weights) and separates the **media
encoder** — vision tower, audio tower, and the projector that feeds their output
to the text model — from everything else.

`tools/checkpoint_survey.py` already screens one checkpoint and has a "never
loaded when serving text" bucket, but it fuses the encoder with MTP and draft
heads. Those are also never loaded for text, and also evictable, but they are not
read *per image*, so counting them together overstates what evicting an encoder
buys and understates how cheap it is.

**Why the encoder specifically.** CPU offload is a last-resort trade nearly
everywhere: an offloaded weight is re-read across PCIe every token. An encoder is
read once per image, and not at all for a text-only request — so evicting it is
close to free across a large fraction of real use, while keeping the capability
that `--language-model-only` throws away. Its size is therefore the whole
question, and HF's tensor viewer answers it one tensor at a time.

Bit depth is reported too: EXL3 stores a trellis as int16 with the bit width in
its last dimension, so a quantized tower's real parameter count is not its
element count. See `docs/media-encoders.md` for what the numbers came to.
"""

from __future__ import annotations

import argparse
import collections
import sys

#: The encoder proper. `image_newline` and friends are tiny but belong with it.
ENCODER = ("vision", "visual", "audio_tower", "mm_projector", "multi_modal_projector",
           "audio_projector", "image_newline", "perceiver", "vqmodel")

#: Also never loaded when serving text, also evictable — but read per *token* when
#: they are read at all, so they are not part of the same trade. Counted apart.
SPECULATIVE = ("mtp", "eagle", "draft")

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "I16": 2,
               "I32": 4, "I64": 8, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}

GIB = 1 << 30


def bucket_of(name: str) -> str:
    low = name.lower()
    if any(k in low for k in ENCODER):
        return "encoder"
    if any(f".{k}." in f".{low}." or low.startswith(f"{k}.") for k in SPECULATIVE):
        return "speculative"
    return "text"


def survey(repo: str, revision: str | None) -> dict:
    from huggingface_hub import HfApi

    meta = HfApi().get_safetensors_metadata(repo, revision=revision)
    byte_total: collections.Counter = collections.Counter()
    enc_by_suffix: collections.Counter = collections.Counter()
    enc_weights = 0          # real parameter count of quantized encoder modules
    enc_quant_bytes = 0
    for fmeta in meta.files_metadata.values():
        for name, t in fmeta.tensors.items():
            n = 1
            for d in t.shape:
                n *= d
            nbytes = n * DTYPE_BYTES.get(t.dtype, 2)
            b = bucket_of(name)
            byte_total[b] += nbytes
            if b != "encoder":
                continue
            enc_by_suffix[name.rsplit(".", 1)[-1]] += nbytes
            if name.endswith(".trellis") and len(t.shape) == 3:
                # EXL3: [in/16, out/16, 16*K] int16 -> K bits per weight, so the
                # element count is 16/K of the weights it encodes.
                d0, d1, d2 = t.shape
                enc_weights += (d0 * 16) * (d1 * 16)
                enc_quant_bytes += nbytes
    return {
        "bytes": byte_total,
        "enc_by_suffix": enc_by_suffix,
        "enc_weights": enc_weights,
        "enc_quant_bytes": enc_quant_bytes,
    }


#: The set behind the table in docs/media-encoders.md. Spans formats deliberately:
#: the encoder is bf16 in all of them, and only the *share* moves.
DEFAULTS = [
    ("Qwen/Qwen3.5-9B", None),
    ("cyankiwi/Qwen3.5-9B-AWQ-4bit", None),
    ("turboderp/Qwen3.5-9B-exl3", "4.00bpw"),
    ("Qwen/Qwen3.8-27B-FP8", None),
    ("turboderp/Qwen3.6-27B-exl3", "5.00bpw"),
    ("turboderp/Qwen3.6-27B-exl3", "3.00bpw"),
    ("Intel/Qwen3.6-35B-A3B-int2-mixed-CT-AutoRound", None),
    ("turboderp/Qwen3.5-35B-A3B-exl3", "2.00bpw"),
    ("turboderp/gemma-4-26B-A4B-it-exl3", "2.54bpw"),
    ("turboderp/Muse-Glimmer-30B-exl3", "2.00bpw"),
    # The vision-first family: one encoder across every model size, so the share is
    # a fixed cost over a shrinking denominator.
    ("turboderp/Qwen3-VL-8B-Instruct-exl3", "3.0bpw"),
    ("turboderp/Qwen3-VL-8B-Instruct-exl3", "6.0bpw"),
    ("turboderp/Qwen3-VL-32B-Instruct-exl3", "3.0bpw"),
    ("turboderp/Qwen3-VL-235B-A22B-Thinking-exl3", "3.00bpw"),
    ("turboderp/gemma-3-27b-it-exl3", "4.0bpw"),
    ("turboderp/Step-3.7-Flash-exl3", "3.05bpw"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("targets", nargs="*", help="hf-repo[@revision]")
    ap.add_argument("--defaults", action="store_true",
                    help="survey the set behind docs/media-encoders.md")
    ap.add_argument("--detail", action="store_true",
                    help="per-suffix breakdown of the encoder's storage")
    args = ap.parse_args()

    targets = [(t.split("@")[0], t.split("@")[1] if "@" in t else None)
               for t in args.targets]
    if args.defaults or not targets:
        targets = DEFAULTS

    print(f"{'checkpoint':46s} {'total':>9} {'encoder':>9} {'share':>7} {'spec':>8}  stored")
    print("-" * 96)
    for repo, rev in targets:
        label = repo if rev is None else f"{repo}@{rev}"
        try:
            r = survey(repo, rev)
        except Exception as exc:  # network, gated repo, non-safetensors
            print(f"{label:46s}  !! {type(exc).__name__}: {str(exc)[:34]}")
            continue
        total = sum(r["bytes"].values())
        enc = r["bytes"].get("encoder", 0)
        spec = r["bytes"].get("speculative", 0)
        if not enc:
            print(f"{label:46s} {total/GIB:8.2f}G {'--':>9} {'--':>7} "
                  f"{spec/GIB:7.2f}G  (no encoder)")
            continue
        if r["enc_weights"]:
            bpw = r["enc_quant_bytes"] * 8 / r["enc_weights"]
            stored = (f"{bpw:.2f}bpw, {r['enc_weights']/1e9:.2f}B params "
                      f"({r['enc_weights']*2/GIB:.2f}G at bf16)")
        else:
            stored = "bf16"
        print(f"{label:46s} {total/GIB:8.2f}G {enc/GIB:8.3f}G "
              f"{enc/total*100:6.2f}% {spec/GIB:7.3f}G  {stored}")
        if args.detail:
            for suf, b in r["enc_by_suffix"].most_common():
                print(f"{'':48s}{suf:12s} {b/2**20:9.1f} MiB ({b/enc*100:5.1f}%)")


if __name__ == "__main__":
    sys.exit(main())
