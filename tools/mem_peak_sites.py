#!/usr/bin/env python3
"""What the profiling run's transient peak is made of: replay an EXL3_MEM_SNAPSHOT allocator trace to its peak
and group the blocks live there by allocating site (first frames outside torch). "kept" is what was still live when
the run ended, so a lazily allocated persistent buffer is told apart from a true transient.

    tools/mem_peak_sites.py SNAPSHOT [DEVICE]
"""
import collections, pickle, sys

snap = pickle.load(open(sys.argv[1], "rb"))
dev = int(sys.argv[2]) if len(sys.argv) > 2 else 0
trace = snap["device_traces"][dev]
SKIP = ("/torch/", "<frozen", "/contextlib.py", "/functools.py", "/_ops.py", "/_contextlib.py", "/utils/_device.py")


def site(frames, depth=3):
    picked = []
    for f in frames:
        fn = f.get("filename", "")
        if any(s in fn for s in SKIP):
            continue
        short = fn.split("/vllm-exl3-plugin/")[-1].split("/vllm-stock/")[-1].split("site-packages/")[-1]
        picked.append(f"{short}:{f.get('line')} {f.get('name')}")
        if len(picked) == depth:
            break
    return " <- ".join(picked) or "?"


live, total, peak, peak_i = {}, 0, 0, -1
for i, e in enumerate(trace):
    a = e["action"]
    if a == "alloc":
        live[e["addr"]] = (e["size"], i); total += e["size"]
        if total > peak:
            peak, peak_i = total, i
    elif a in ("free_completed",) and e["addr"] in live:
        total -= live.pop(e["addr"])[0]
# live set at the peak, and which of those were never freed by the end of the trace
live, at_peak = {}, None
for i, e in enumerate(trace):
    a = e["action"]
    if a == "alloc":
        live[e["addr"]] = (e["size"], i)
    elif a == "free_completed" and e["addr"] in live:
        live.pop(e["addr"])
    if i == peak_i:
        at_peak = dict(live)
still = set(live)
groups = collections.defaultdict(lambda: [0, 0, 0])   # bytes, count, bytes still live at the end
for addr, (size, i) in at_peak.items():
    g = groups[site(trace[i].get("frames", []))]
    g[0] += size; g[1] += 1
    if addr in still:
        g[2] += size
print(f"peak of allocations made during the trace: {peak / 2**30:.2f} GiB at event {peak_i} of {len(trace)}")
print(f"{'GiB':>6} {'kept':>6} {'n':>5}  site")
for s, (b, n, k) in sorted(groups.items(), key=lambda kv: -kv[1][0])[:25]:
    print(f"{b / 2**30:6.3f} {k / 2**30:6.3f} {n:5}  {s}")
