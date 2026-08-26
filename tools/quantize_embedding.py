#!/usr/bin/env python3
"""Add a block-quantized token embedding to an existing EXL3 checkpoint.

Every published EXL3 checkpoint stores `embed_tokens.weight` at full precision --
a quarter to a half of a mid-size checkpoint, and worst at the low bit rates
where it hurts most (docs/embeddings.md). This rewrites one checkpoint to carry
the block-scaled 4-bit embedding of `vllm_exl3_plugin.blockq` instead, which the
plugin then serves without ever materializing the dense matrix.

    tools/quantize_embedding.py <checkpoint-dir> <output-dir>

Scoped to **untied** models, and that scope is now *enforced* -- it was documented
here and checked nowhere, which is how tied checkpoints got repaired by accident.
A tied model already has a quantized `lm_head` covering the same matrix and is
served from it today with no tooling; repairing one produces a checkpoint that no
serving path currently loads correctly. `--allow-tied` exists to build one on
purpose, for work on that path.

Two properties make this safe to run on a published checkpoint:

  - **It quantizes the checkpoint's own bf16 `embed_tokens.weight`**, which for an
    untied model is the original, untouched matrix. Never a dequantized trellis:
    that is the same matrix ~2% lossy already, and quantizing it again compounds
    the error for no saving (docs/embeddings.md).
  - **It never edits in place.** Shards that do not contain the embedding are
    hardlinked (or copied across filesystems) into the output, so the cost is one
    rewritten shard regardless of model size.

The output is a complete checkpoint directory: same weights, same config, minus
the fp16 embedding, plus three small tensors and a `quantization_config.json`
entry describing them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm_exl3_plugin import blockq, format

EMBED_SUFFIX = ".embed_tokens.weight"


def find_embedding(shard_files):
    """Locate the dense embedding: (file, tensor name, shape, dtype)."""
    for path in shard_files:
        with safe_open(path, framework="pt") as h:
            for key in h.keys():
                if key.endswith(EMBED_SUFFIX):
                    sl = h.get_slice(key)
                    return path, key, tuple(sl.get_shape()), sl.get_dtype()
    raise SystemExit(
        f"no tensor ending {EMBED_SUFFIX} in {len(shard_files)} shard(s); "
        "there is no dense embedding here to quantize (already repaired?)"
    )


def encode_embedding(path, key, chunk):
    """Quantize in row chunks; a vocabulary-sized fp32 temporary is several GiB."""
    with safe_open(path, framework="pt") as h:
        sl = h.get_slice(key)
        vocab, hidden = sl.get_shape()
        format.check_blockq_hidden(hidden)
        parts = []
        for a in range(0, vocab, chunk):
            b = min(a + chunk, vocab)
            parts.append(blockq.encode(sl[a:b].float()))
            print(f"\r  encoding {b}/{vocab} rows", end="", file=sys.stderr)
    print(file=sys.stderr)
    return {
        name: torch.cat([p[name] for p in parts], dim=0)
        for name in ("bq_q", "bq_s", "bq_r")
    }


#: safetensors spells dtypes its own way in headers ("BF16", not "torch.bfloat16").
_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I16": 2, "I32": 4, "I8": 1,
                "U8": 1}


def link_or_copy(src, dst):
    """Hardlink, resolving symlinks first.

    A Hugging Face cache snapshot is a directory of symlinks into `blobs/` with
    relative targets, so linking the link itself yields something that dangles
    from anywhere else. Resolve to the blob and hardlink that.
    """
    src = os.path.realpath(src)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def is_tied(src: str) -> bool | None:
    """Whether the checkpoint's config declares tied embeddings.

    Checked because the dense embedding's *presence* says nothing about it: an
    EXL3 checkpoint stores a dense `embed_tokens.weight` alongside a trellis
    `lm_head` even when tied -- that redundancy is the whole subject of
    docs/embeddings.md -- so `find_embedding` succeeds on a tied checkpoint and
    this tool will happily produce one. Multimodal configs nest the flag, so
    both levels are consulted and either one counts.

    Returns None when there is no config to read, which is not the same as
    False and must not be treated as permission.
    """
    path = os.path.join(src, "config.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        cfg = json.load(f)
    for scope in (cfg, cfg.get("text_config") or {}):
        if scope.get("tie_word_embeddings"):
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="EXL3 checkpoint directory")
    ap.add_argument("output", help="directory to write the repaired checkpoint to")
    ap.add_argument("--chunk", type=int, default=16384, help="rows per encode step")
    ap.add_argument("--device", default="cpu", help="device to encode on")
    ap.add_argument("--allow-tied", action="store_true",
                    help="proceed on a tied checkpoint (see the refusal text)")
    args = ap.parse_args()

    src, dst = args.checkpoint, args.output

    tied = is_tied(src)
    if tied is not False and not args.allow_tied:
        detail = ("declares tied embeddings" if tied
                  else "has no config.json, so tying cannot be ruled out")
        raise SystemExit(
            f"{src} {detail}, and repairing a tied checkpoint produces one that "
            "no serving path currently loads correctly.\n"
            "\n"
            "The dense embedding being present is not evidence of untying: an "
            "EXL3 checkpoint stores one next to a trellis lm_head even when "
            "tied. Repairing anyway yields a checkpoint where the embedding and "
            "the head each claim quantized storage, which the plugin's "
            "predicates treat as mutually exclusive -- it fails late, at logits "
            "time, or silently discards the head's trellis and serves garbage.\n"
            "\n"
            "Pass --allow-tied only to build such a checkpoint deliberately, "
            "for work on that serving path."
        )

    if os.path.exists(dst) and os.listdir(dst):
        raise SystemExit(f"{dst} exists and is not empty")
    os.makedirs(dst, exist_ok=True)

    shards = sorted(
        os.path.join(src, f) for f in os.listdir(src) if f.endswith(".safetensors")
    )
    if not shards:
        raise SystemExit(f"no .safetensors in {src}")

    embed_file, embed_key, shape, dtype = find_embedding(shards)
    prefix = embed_key[: -len(".weight")]
    vocab, hidden = shape
    print(f" -- {embed_key}: {list(shape)} {dtype} in "
          f"{os.path.basename(embed_file)}")

    stored = encode_embedding(embed_file, embed_key, args.chunk)
    if args.device != "cpu":
        stored = {k: v.to(args.device) for k, v in stored.items()}

    # Rewrite only the shard that held the embedding; hardlink the rest.
    for path in shards:
        out = os.path.join(dst, os.path.basename(path))
        if path != embed_file:
            link_or_copy(path, out)
            continue
        with safe_open(path, framework="pt") as h:
            meta = h.metadata()
            keep = {k: h.get_tensor(k) for k in h.keys() if k != embed_key}
        keep.update({f"{prefix}.{n}": t for n, t in stored.items()})
        save_file(keep, out, metadata=meta)

    # Everything else in the checkpoint travels unchanged.
    for name in os.listdir(src):
        if name.endswith(".safetensors") or name == "quantization_config.json":
            continue
        s = os.path.join(src, name)
        if os.path.isfile(s):
            link_or_copy(s, os.path.join(dst, name))

    # Describe the new tensors the way the rest of the checkpoint describes its
    # own, so `EXL3Config` sees them without a special case.
    qc_path = os.path.join(src, "quantization_config.json")
    qc = json.load(open(qc_path)) if os.path.exists(qc_path) else {}
    ts = qc.setdefault("tensor_storage", {})
    ts.pop(prefix, None)
    ts[prefix] = {
        "stored_tensors": {
            f"{prefix}.{n}": {
                "shape": list(t.shape),
                "n_bytes": t.numel() * t.element_size(),
                "dtype": str(t.dtype),
            }
            for n, t in stored.items()
        },
        "quant_format": "exl3_blockq",
        "bits_per_weight": format.BLOCKQ_BITS,
        "block_size": format.BLOCKQ_BLOCK,
    }
    with open(os.path.join(dst, "quantization_config.json"), "w") as f:
        json.dump(qc, f, indent=2)

    was = vocab * hidden * _DTYPE_BYTES.get(dtype, 2)
    now = sum(t.numel() * t.element_size() for t in stored.values())
    print(f" -- embedding {was / 2**30:.3f} GiB -> {now / 2**30:.3f} GiB "
          f"({format.blockq_bpw(hidden):.4f} bpw), saved {(was - now) / 2**30:.3f} GiB")
    print(f" -- wrote {dst}")


if __name__ == "__main__":
    main()
