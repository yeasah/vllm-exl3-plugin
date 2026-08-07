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


def resolve(target: str) -> str:
    if os.path.isdir(target):
        return target
    hits = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--"
        + target.replace("/", "--") + "/snapshots/*")))
    if not hits:
        raise SystemExit(f"no local snapshot for {target!r}")
    return hits[-1]


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
    target = sys.argv[1]
    degrees = [int(a) for a in sys.argv[2:]] or [2, 4, 8]
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
    for (generic, in_f, out_f), members in sorted(groups.items()):
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
        verdict = "USABLE" if not bad else f"BLOCKED ({bad} tensors cannot split)"
        print(f"  TP={tp}: {verdict}")


if __name__ == "__main__":
    main()
