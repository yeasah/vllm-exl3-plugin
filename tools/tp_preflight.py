#!/usr/bin/env python3
"""Which tensor-parallel degrees can this checkpoint actually be split at?

EXL3's Hadamard transform is block-diagonal in blocks of 128, so a tensor may
only be cut on a multiple of 128 *per rank*. Whether that holds is a property of
the checkpoint's stored dimensions, which are not the model's: exllamav3 pads
before quantizing (gemma-4-26B stores a 768-wide expert intermediate where the
config says 704). So the answer differs per checkpoint in ways config.json does
not predict, and a bigger model is not a more divisible one.

This reads safetensors headers only -- no GPU, no weights, no vLLM -- so it can
be run before renting anything.

    tools/tp_preflight.py <snapshot-dir-or-hf-repo-name> [tp ...]
    tools/tp_preflight.py <hf-repo> --remote --revision 3.0bpw

`--remote` answers the question for a checkpoint that is not downloaded, via
`HfApi.get_safetensors_metadata`, the Hub's own supported call for reading
tensor metadata without fetching weights. It is deliberately *less* hub traffic
than the alternative: today the only way to learn whether a checkpoint is
TP-viable is to download all of it, so this replaces gigabytes with a few
kilobytes of header reads per model.

It takes one repo per invocation on purpose. Do not wrap it in an enumeration
over the Hub -- vetting a shortlist you already have is a different thing from
crawling, and only the first is intended here. Authentication is whatever
`huggingface_hub` already has (`HF_TOKEN`, or a stored `hf auth login`).
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vllm_exl3_plugin import format  # noqa: E402

#: How each quantized module is split. Column = output dimension (the tensor's
#: `out_features`); row = input dimension. Anything unmatched is reported rather
#: than assumed, since guessing wrong here is exactly the silent-corruption case.
COLUMN = re.compile(
    r"\.(q_proj|k_proj|v_proj|gate_proj|up_proj|in_proj_qkv|in_proj_z|in_proj_qkvz)$"
    r"|(^|\.)lm_head$")
ROW = re.compile(r"\.(o_proj|down_proj|out_proj)$")

#: Stored tensors that vLLM hands over as a *tuple* of output shards, because it
#: merges them into a wider parameter than the checkpoint has (Qwen3.5's linear
#: attention fuses `in_proj_qkv` + `in_proj_z` into `in_proj_qkvz`, so the single
#: stored `in_proj_qkv` covers shards 0, 1 and 2).
#:
#: `EXL3Parameter._load_fused` composes that split with the tensor-parallel one
#: (`format.fused_shard_bounds`), so these no longer block TP -- but the path is
#: only verified offline, never on multi-GPU hardware, so it is still called out
#: rather than passed over silently. It cannot be derived from the safetensors
#: headers, hence a list of known cases.
FUSED_MULTI_SHARD = re.compile(r"\.in_proj_qkv$")


def resolve(target: str) -> str:
    """Pick the snapshot that actually holds weights.

    A repo often has several: fetching one file at a different revision (a
    `processor_config.json`, say) leaves a snapshot directory containing only
    that file. Choosing by name would pick it about half the time, so choose by
    weight count instead.
    """
    if os.path.isdir(target):
        return target
    hits = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--"
        + target.replace("/", "--") + "/snapshots/*")))
    if not hits:
        raise SystemExit(f"no local snapshot for {target!r}")
    weighted = [(len(glob.glob(os.path.join(h, "*.safetensors"))), h) for h in hits]
    n, best = max(weighted)
    if not n:
        raise SystemExit(f"no snapshot of {target!r} contains safetensors")
    return best


def remote_trellis_shapes(repo: str, revision: str | None) -> dict[str, list[int]]:
    """Every `<module>.trellis` shape, read from the Hub without downloading.

    EXL3 repos publish one branch per bit rate and `main` often carries no
    weights at all, so `revision` usually matters.
    """
    from huggingface_hub import HfApi

    meta = HfApi().get_safetensors_metadata(repo, revision=revision)
    shapes = {}
    for f in meta.files_metadata.values():
        for name, t in f.tensors.items():
            if name.endswith(".trellis"):
                shapes[name[: -len(".trellis")]] = list(t.shape)
    return shapes


def trellis_shapes(d: str) -> dict[str, list[int]]:
    """Every `<module>.trellis` shape, from the safetensors headers."""
    shapes = {}
    for path in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n))
        for name, meta in header.items():
            if name.endswith(".trellis"):
                shapes[name[: -len(".trellis")]] = meta["shape"]
    return shapes


def main() -> None:
    argv = sys.argv[1:]
    remote = "--remote" in argv
    revision = None
    if "--revision" in argv:
        i = argv.index("--revision")
        revision = argv[i + 1]
        del argv[i : i + 2]
    argv = [a for a in argv if a != "--remote"]
    target = argv[0]
    degrees = [int(a) for a in argv[1:]] or [2, 4, 8]
    if remote:
        d = f"{target}@{revision or 'main'} (hub metadata)"
        shapes = remote_trellis_shapes(target, revision)
    else:
        d = resolve(target)
        shapes = trellis_shapes(d)
    if not shapes:
        raise SystemExit(f"no EXL3 trellis tensors under {d}")

    # Collapse layer/expert indices: every layer has the same shapes, and a
    # 256-expert model would otherwise print 30k identical lines.
    groups: dict[tuple, list[str]] = {}
    for name, shape in shapes.items():
        generic = re.sub(r"\.\d+\.", ".N.", re.sub(r"\.\d+\.", ".N.", name))
        in_f, out_f = format.dims_from_trellis_shape(shape)
        groups.setdefault((generic, in_f, out_f), []).append(name)

    print(f"{target}\n  {d}\n  {len(shapes)} quantized tensors, "
          f"{len(groups)} distinct shapes\n")
    worst = {}
    fused = []
    for (generic, in_f, out_f), members in sorted(groups.items()):
        if FUSED_MULTI_SHARD.search(generic):
            fused.append((generic, len(members)))
            print(f"  fus {generic:<52} {out_f:>6}  fused output shards, "
                  "TP path unverified on hardware")
            continue
        if COLUMN.search(generic):
            axis, size = "col", out_f
        elif ROW.search(generic):
            axis, size = "row", in_f
        else:
            print(f"  ?   {generic}  in={in_f} out={out_f}  UNCLASSIFIED")
            continue
        cells = []
        for tp in degrees:
            try:
                format.check_tp_split(size, tp, generic)
                cells.append(f"tp{tp}:ok")
            except format.EXL3FormatError:
                cells.append(f"tp{tp}:NO")
                worst[tp] = worst.get(tp, 0) + len(members)
        print(f"  {axis} {generic:<52} {size:>6}  {'  '.join(cells)}")

    print()
    for tp in degrees:
        bad = worst.get(tp, 0)
        if bad:
            verdict = f"BLOCKED ({bad} tensors cannot split)"
        else:
            verdict = "USABLE"
        print(f"  TP={tp}: {verdict}")
    if fused:
        n = sum(c for _, c in fused)
        print(f"\n  {n} tensors carry fused output shards -- a vLLM packing"
              "\n  property, not a dimension one. _load_fused composes that split"
              "\n  with the TP split; the arithmetic is unit-tested but the path"
              "\n  has never run on real multi-GPU hardware. Treat a TP verdict"
              "\n  above as provisional for this checkpoint.")


if __name__ == "__main__":
    main()
