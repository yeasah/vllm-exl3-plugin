#!/usr/bin/env python3
"""Attribute a torch CUDA memory snapshot's *peak* to the call sites that formed it.

`torch.cuda.memory._dump_snapshot()` writes every allocation with its stack, but
pytorch.org/memory_viz is built for finding leaks in a flat profile: it cannot exclude the
static weights that dominate the scale, its default filtering hides exactly the small
transients of interest, and at a million entries it stops being usable at all.

What actually matters for sizing a serving margin is narrower: at the moment of peak
usage, what was live, and which code allocated it. This walks the trace once, finds that
moment, and attributes it -- then separately ranks call sites by the largest single
allocation each ever made, which is what a margin has to cover.
"""
import argparse
import pickle
from collections import defaultdict

#: Frames to step over when naming a call site. The snapshot interleaves C++ unwind
#: entries (filename "??") with Python ones, and the C++ ones are always on top -- naming a
#: site by frame 0 labels every allocation in the process
#: `torch::CapturedTraceback::gather`, which is how this first ran. Python frames only.
SKIP = ("torch/_ops.py", "torch/_dynamo", "torch/_inductor", "torch/nn/modules/module.py",
        "torch/_subclasses", "torch/utils/_", "torch/_compile.py", "torch/_library")


def site(frames, depth):
    """First few *Python* frames that are not allocator/dispatch plumbing."""
    out = []
    for f in frames or ():
        fn = f.get("filename", "")
        if not fn.endswith(".py"):
            continue
        if any(s in fn for s in SKIP):
            continue
        short = fn.rsplit("/", 2)
        short = "/".join(short[-2:]) if len(short) > 1 else fn
        out.append(f"{short}:{f.get('line')} {f.get('name')}")
        if len(out) >= depth:
            break
    return " <- ".join(out) or "<no frames>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--depth", type=int, default=2, help="stack frames per call site")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--min-mib", type=float, default=1.0,
                    help="ignore allocations smaller than this when ranking")
    args = ap.parse_args()

    with open(args.snapshot, "rb") as f:
        snap = pickle.load(f)

    trace = snap["device_traces"][0]
    live, cur, peak, peak_i = {}, 0, 0, 0
    biggest = defaultdict(int)   # call site -> largest single allocation
    count = defaultdict(int)
    for i, e in enumerate(trace):
        a = e["action"]
        if a == "alloc":
            live[e["addr"]] = e
            cur += e["size"]
            if cur > peak:
                peak, peak_i = cur, i
            s = site(e.get("frames"), args.depth)
            biggest[s] = max(biggest[s], e["size"])
            count[s] += 1
        elif a in ("free_completed", "free_requested"):
            v = live.pop(e["addr"], None)
            if v is not None:
                cur -= v["size"]

    # replay to the peak to see what was live there
    live, cur = {}, 0
    for e in trace[: peak_i + 1]:
        if e["action"] == "alloc":
            live[e["addr"]] = e
        elif e["action"] in ("free_completed", "free_requested"):
            live.pop(e["addr"], None)
    at_peak = defaultdict(int)
    for v in live.values():
        at_peak[site(v.get("frames"), args.depth)] += v["size"]

    M = 1024 ** 2
    print(f"trace entries      {len(trace):,}")
    print(f"peak live          {peak / 1024**3:.3f} GiB  (entry {peak_i:,})")
    print(f"live blocks there  {len(live):,}\n")

    print(f"=== composition of the peak (top {args.top}) ===")
    for s, b in sorted(at_peak.items(), key=lambda kv: -kv[1])[: args.top]:
        print(f"{b/M:9.1f} MiB  {s}")

    print(f"\n=== largest single allocation per call site (top {args.top}, >= {args.min_mib} MiB) ===")
    rows = [(b, s) for s, b in biggest.items() if b >= args.min_mib * M]
    for b, s in sorted(rows, reverse=True)[: args.top]:
        print(f"{b/M:9.1f} MiB  x{count[s]:<7d} {s}")


if __name__ == "__main__":
    main()
