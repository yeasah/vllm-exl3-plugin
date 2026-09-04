#!/usr/bin/env python3
"""Add a block-quantized token embedding to an existing EXL3 checkpoint.

Every published EXL3 checkpoint stores `embed_tokens.weight` at full precision --
a quarter to a half of a mid-size checkpoint, and worst at the low bit rates
where it hurts most (docs/embeddings.md). This rewrites one checkpoint to carry
the block-scaled 4-bit embedding of `vllm_exl3_plugin.blockq` instead, which the
plugin then serves without ever materializing the dense matrix.

    tools/quantize_embedding.py <checkpoint-dir> <output-dir>

Useful on **tied** models too, which was not always true. A tied model is served
from its quantized `lm_head` with no tooling at all, but through the *trellis*,
which is the wrong encoding for a lookup and costs ~73x the divergence of a
block-quantized embedding (docs/embeddings.md). Repairing a tied checkpoint keeps
the trellis for the logits GEMM and adds `bq_*` for the gather, so each role gets
the encoding built for it.

That arrangement needs a plugin new enough to serve both from one module --
`EXL3BlockQTiedEmbeddingMethod`, added 2026-08-26. **An older plugin loads such a
checkpoint without complaint and serves garbage**, so a repaired tied checkpoint
is not portable backwards; the run below says so when it applies.

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

`--sidecar` writes the same `bq_*` into a shard of their own and leaves every
original shard alone, dense embedding included. Nothing is rewritten, so nothing
is duplicated: the output costs the `bq_*` bytes and nothing else, as long as the
source stays where the hardlinks can reach it. That is the better trade whenever
the original is being kept anyway -- a single-file checkpoint otherwise
duplicates the *whole model* to add a few hundred MiB, since the shard holding
the embedding is the only shard. It is the worse trade if the output has to
stand alone, because it is larger in total and depends on the source.

A sidecar checkpoint carries both encodings, which is also the only way to
compare them on one set of weights: `EXL3_DENSE_EMBED=1` serves the dense
embedding and ignores `bq_*`.
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

EMBED_SUFFIX = format.EMBED_WEIGHT_SUFFIX
INDEX_NAME = "model.safetensors.index.json"


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


def write_index(src, dst_index, weight_map):
    """Write a safetensors index describing the sidecar output.

    Required, not cosmetic. vLLM globs `*.safetensors` and then, if an index
    exists, keeps only the files the index mentions
    (`filter_duplicate_safetensors_files`, added so a checkpoint shipping both
    sharded and consolidated files does not load both). A sidecar missing from
    the map is dropped silently, and the failure surfaces later as `bq_*` that
    never loaded.

    Written even where the source had none, because the alternative -- several
    safetensors files and no index -- is a layout no published checkpoint here
    uses, and one that works only because a particular loader happens to glob.
    """
    meta = {}
    src_index = os.path.join(src, INDEX_NAME)
    if os.path.exists(src_index):
        with open(src_index) as f:
            meta = json.load(f).get("metadata", {}) or {}
    meta["total_size"] = sum(
        os.path.getsize(os.path.join(os.path.dirname(dst_index), name))
        for name in sorted(set(weight_map.values()))
    )
    with open(dst_index, "w") as f:
        json.dump({"metadata": meta, "weight_map": weight_map}, f, indent=2)


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
    ap.add_argument(
        "--sidecar", action="store_true",
        help="add bq_* in their own shard and leave every original shard "
             "alone, instead of rewriting the one holding the embedding. "
             "Costs only the bq_* bytes, but the dense embedding stays in the "
             "output, so this is for keeping the original alongside rather "
             "than replacing it.",
    )
    args = ap.parse_args()

    src, dst = args.checkpoint, args.output

    # Said out loud because the resulting checkpoint is a durable artifact and
    # the requirement it carries is invisible in the file: served by a plugin
    # predating EXL3BlockQTiedEmbeddingMethod, a repaired tied checkpoint loads
    # clean and emits garbage. Nothing downstream can warn about that, so this
    # is the only place it gets said.
    tied = is_tied(src)
    if tied is not False:
        which = "is tied" if tied else "has no config.json, so may be tied"
        print(f" -- {os.path.basename(os.path.normpath(src))} {which}: the "
              f"output keeps the trellis lm_head for logits and adds bq_* for "
              f"the lookup.\n"
              f"    Requires a plugin with EXL3BlockQTiedEmbeddingMethod "
              f"(2026-08-26 or newer). Older ones serve it as garbage.",
              file=sys.stderr)

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

    bq_named = {f"{prefix}.{n}": t for n, t in stored.items()}

    if args.sidecar:
        # Every original shard is left exactly as it is, including the dense
        # embedding, and `bq_*` go in a shard of their own. Nothing is
        # rewritten, so nothing is duplicated: the output costs the `bq_*`
        # bytes plus directory entries, provided the source stays around for
        # the links to point at.
        #
        # The output is always renumbered into the sharded convention with an
        # index, even from a single-file source. Adding a second file beside a
        # bare `model.safetensors` and no index does load here -- vLLM globs --
        # but that layout appears in none of the 69 published checkpoints on
        # this machine, so it is not a shape to emit on the strength of one
        # consumer's loader. `model-0000i-of-0000N` plus an index is what the
        # sharded ones use, and `openbmb/MiniCPM5-1B` shows a single shard
        # named that way with an index too.
        total = len(shards) + 1
        weight_map: dict[str, str] = {}
        for i, path in enumerate(shards, start=1):
            name = f"model-{i:05d}-of-{total:05d}.safetensors"
            link_or_copy(path, os.path.join(dst, name))
            with safe_open(path, framework="pt") as h:
                for k in h.keys():
                    weight_map[k] = name
        side = f"model-{total:05d}-of-{total:05d}.safetensors"
        save_file(bq_named, os.path.join(dst, side), metadata={"format": "pt"})
        weight_map.update({k: side for k in bq_named})
        write_index(src, os.path.join(dst, INDEX_NAME), weight_map)
    else:
        # Rewrite only the shard that held the embedding; hardlink the rest.
        for path in shards:
            out = os.path.join(dst, os.path.basename(path))
            if path != embed_file:
                link_or_copy(path, out)
                continue
            with safe_open(path, framework="pt") as h:
                meta = h.metadata()
                keep = {k: h.get_tensor(k) for k in h.keys() if k != embed_key}
            keep.update(bq_named)
            save_file(keep, out, metadata=meta)

    # Everything else in the checkpoint travels unchanged.
    for name in os.listdir(src):
        if name.endswith(".safetensors") or name == "quantization_config.json":
            continue
        srcf = os.path.join(src, name)
        if not os.path.isfile(srcf):
            continue
        if args.sidecar and name == INDEX_NAME:
            continue  # regenerated above, against the new shard names
        link_or_copy(srcf, os.path.join(dst, name))

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
        # Sidecar output keeps the dense embedding, so both encodings ship and
        # EXL3_DENSE_EMBED can choose between them. Recorded here rather than
        # inferred from the safetensors index, because a single-file checkpoint
        # has no index and that is exactly the shape sidecar mode is best on.
        "dense_embedding_retained": bool(args.sidecar),
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
