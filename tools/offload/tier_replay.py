"""Replay a routing trace (routing_trace.py) against an expert cache with a disk tier below host RAM.

usage: tier_replay.py TRACE EXPERT_MIB [--vram 0.2,0.3,0.4] [--ram 1.0,0.75,0.5,0.4,0.3]

Three tiers, per layer: VRAM slots (the exclusive cache as routing_replay.py validates it: a miss is read from below
this token and promoted `lag` tokens later, the 32 most-wanted misses per token over all layers), host RAM holding a
fraction of each layer's experts (an LRU, exclusive with VRAM: promoted experts leave it, evicted ones return), and disk
for the rest. An expert used while on disk is read first, then held in RAM. Decode counts the reads each tier serves
per token; prefill counts what each chunk stages from RAM and from disk (it touches most of a layer's experts). Times
come from bandwidths, not measurement: they say whether a disk tier is a disaster or usable, not what it would serve.
EXPERT_MIB is one expert's three projections in one layer, as stored.
"""
import argparse, collections, sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("expert_mib", type=float)
ap.add_argument("--vram", default="0.2,0.3,0.4")
ap.add_argument("--ram", default="1.0,0.75,0.6,0.5,0.4,0.3")
ap.add_argument("--lag", type=int, default=2)
ap.add_argument("--budget", type=int, default=32)
ap.add_argument("--link-gbs", type=float, default=20.0, help="UVA reads and staging copies from host RAM, GB/s")
ap.add_argument("--ssd-gbs", default="2,6", help="disk read rates to bracket, GB/s (random ~1 MiB reads)")
ap.add_argument("--concurrent", default=None, help="e.g. 1,2,4: decode with n requests sharing each step instead")
ap.add_argument("--prefill-bypass", action="store_true",
                help="prefill's disk reads are staged and used but not kept in RAM")
ap.add_argument("--deep-gbs", default="5.4", help="disk rate for prefill staging, which can run deep queues, GB/s")
a = ap.parse_args()

Z = np.load(a.trace + ".npz", allow_pickle=True)
tags, layers, topk = Z["tags"], Z["layers"], Z["topk"]
E = int(max(t.max() for t in topk)) + 1
L = int(layers.max()) + 1
steps, cur = [], None
for tag, l, t in zip(tags, layers, topk):
    if tag == "none":
        continue
    j = int(tag.split(":")[0])
    ph = "prefill" if t.shape[0] > 1 else "decode"
    if cur is None or l <= cur[-1][1] or j != cur[-1][0]:
        cur = []
        steps.append(cur)
    cur.append((j, int(l), ph, t))
dec_tokens = sum(1 for st in steps if st[0][2] == "decode")
chunks = sum(1 for st in steps if st[0][2] == "prefill")
print(f"E={E} L={L} k={topk[0].shape[1]} decode tokens {dec_tokens} prefill chunks {chunks}; "
      f"one expert {a.expert_mib:.2f} MiB, all {E * L * a.expert_mib / 1024:.1f} GiB")


def replay(fv, fr, steps=steps, budget=None, unique=False):
    """(per decode step: vram, ram, disk reads; per prefill chunk: MiB from ram, MiB from disk). With merged steps
    (several requests' tokens a step), `unique` counts an expert once per layer per step, as a batched step reads it."""
    budget = a.budget if budget is None else budget
    n_dec = max(sum(1 for st in steps if st[0][2] == "decode"), 1)
    cv, cr = int(fv * E), int(fr * E)
    vram = [dict() for _ in range(L)]
    ram = [dict() for _ in range(L)]          # experts on the host, not in VRAM; the rest are on disk
    for l in range(L):                        # cold start: the lowest ids in RAM, as a load would leave them
        for i in range(min(cr, E)):
            ram[l][i] = 0
    clock, pending = 0, collections.deque()
    dv = dr = dd = 0
    pr_ram = pr_disk = 0
    full = []                                  # (tokens, MiB from RAM, MiB from disk) for chunks of >= 1536 tokens

    def to_ram(l, i):
        r = ram[l]
        if cr <= 0:
            return
        if len(r) >= cr:
            del r[min(r, key=r.get)]          # its copy goes back to disk, which holds everything
        r[i] = clock

    for st in steps:
        clock += 1
        ph = st[0][2]
        while pending and pending[0][0] <= clock:
            _, l, i = pending.popleft()
            v = vram[l]
            if i in v:
                continue
            if len(v) >= cv:
                out = min(v, key=v.get)
                del v[out]
                to_ram(l, out)                 # exclusive: an evicted expert returns to host rows
            v[i] = clock
            ram[l].pop(i, None)
        want = collections.Counter()
        before = (pr_ram, pr_disk)
        for _, l, _, t in st:
            ids = t.ravel().tolist() if ph == "decode" and not unique else np.unique(t).tolist()
            v, r = vram[l], ram[l]
            for i in ids:
                if i in v:
                    v[i] = clock
                    dv += ph == "decode"
                    continue
                if i in r:
                    r[i] = clock
                    if ph == "decode":
                        dr += 1
                    else:
                        pr_ram += a.expert_mib
                else:
                    if ph == "decode":
                        dd += 1
                    else:
                        pr_disk += a.expert_mib
                    # a prefill-only read streams through the staging window without displacing RAM's decode-hot set
                    if ph == "decode" or not a.prefill_bypass:
                        to_ram(l, i)
                if ph == "decode":
                    want[(l, i)] += 1
        if ph == "decode":
            for l, i in sorted(want, key=lambda k: (-want[k], (k[0] - clock) % L))[:budget]:
                pending.append((clock + a.lag, l, i))
        elif st[0][3].shape[0] >= 1536:
            full.append((st[0][3].shape[0], pr_ram - before[0], pr_disk - before[1]))
    replay.full = full
    return dv / n_dec, dr / n_dec, dd / n_dec, pr_ram / max(chunks, 1), pr_disk / max(chunks, 1)


def batched_steps(n):
    """Decode only: n requests at a time share each step, a new one joining as one ends (routing_replay.py's)."""
    jobs = collections.defaultdict(list)
    cur = {}
    for st in steps:
        if st[0][2] != "decode":
            continue
        jobs[st[0][0]].append(st)
    queue = collections.deque(sorted(jobs))
    active, out = [], []
    while queue or active:
        while queue and len(active) < n:
            active.append(collections.deque(jobs[queue.popleft()]))
        merged = {}
        for q in active:
            for _, l, _, t in q.popleft():
                merged.setdefault(l, []).append(t)
        active = [q for q in active if q]
        out.append([(0, l, "decode", np.vstack(v)) for l, v in sorted(merged.items())])
    return out


if a.concurrent:
    ssd = [float(x) for x in a.ssd_gbs.split(",")]
    print(f"\nconcurrent decode (no prefill): n requests a step, an expert read once per layer per step, promotion "
          f"budget {a.budget} x n; disk time at {' or '.join(f'{x:g}' for x in ssd)} GB/s, per step and per token")
    for fv in [float(x) for x in a.vram.split(",")]:
        print(f"\n-- VRAM {fv:.0%} --")
        print(f"{'RAM':>5} {'n':>2}  {'vram':>6} {'ram':>6} {'disk':>6}  " +
              " ".join(f"{f'ms/step @{x:g}':>13} {f'ms/tok @{x:g}':>12}" for x in ssd) + f"  {'disk/tok vs n=1':>15}")
        for fr in [float(x) for x in a.ram.split(",")]:
            fr = min(fr, 1.0 - fv)
            base = None
            for n in [int(x) for x in a.concurrent.split(",")]:
                v, r, d, _, _ = replay(fv, fr, batched_steps(n), a.budget * n, unique=True)
                ms = [d * a.expert_mib * 2**20 / (x * 1e9) * 1e3 for x in ssd]
                base = base or d
                print(f"{fr:5.0%} {n:2d}  {v:6.1f} {r:6.1f} {d:6.1f}  " +
                      " ".join(f"{m:13.2f} {m / n:12.2f}" for m in ms) + f"  {d / n / base:15.2f}x", flush=True)
    sys.exit(0)

ssd = [float(x) for x in a.ssd_gbs.split(",")]
deep = [float(x) for x in a.deep_gbs.split(",")]
fulls = []
mib = 1024 ** 2
print(f"\nper decode token: reads from VRAM / RAM / disk, and the time the RAM and disk reads add at {a.link_gbs:g} GB/s "
      f"link and {' or '.join(f'{s:g}' for s in ssd)} GB/s disk; per prefill chunk: GiB staged from RAM / disk, seconds")
for fv in [float(x) for x in a.vram.split(",")]:
    print(f"\n-- VRAM {fv:.0%} of each layer's experts --")
    print(f"{'RAM':>5}  {'vram':>6} {'ram':>6} {'disk':>6}  {'+ms ram':>8} " + " ".join(f"{f'+ms @{s:g}':>9}" for s in ssd)
          + f"   {'chunk ram':>9} {'disk':>6} " + " ".join(f"{f's @{s:g}':>7}" for s in ssd))
    for fr in [float(x) for x in a.ram.split(",")]:
        if fv + fr > 1.0 + 1e-9:
            fr = 1.0 - fv
        v, r, d, pr, pd = replay(fv, fr)
        ms_ram = r * a.expert_mib * mib / (a.link_gbs * 1e9) * 1e3
        ms_disk = [d * a.expert_mib * mib / (s * 1e9) * 1e3 for s in ssd]
        s_chunk = [pr * mib / (a.link_gbs * 1e9) + pd * mib / (s * 1e9) for s in ssd]
        print(f"{fr:5.0%}  {v:6.1f} {r:6.1f} {d:6.1f}  {ms_ram:8.2f} " + " ".join(f"{m:9.2f}" for m in ms_disk)
              + f"   {pr / 1024:9.2f} {pd / 1024:6.2f} " + " ".join(f"{t:7.2f}" for t in s_chunk), flush=True)
        if replay.full:
            tok = sum(c[0] for c in replay.full) / len(replay.full)
            gd = sum(c[2] for c in replay.full) / len(replay.full) / 1024
            gr = sum(c[1] for c in replay.full) / len(replay.full) / 1024
            add = [gd * 2**30 / (x * 1e9) for x in deep]
            fulls.append((fv, fr, tok, gr, gd, add))
    if fulls:
        print(f"  full chunks (>= 1536 tokens, n={len(replay.full)}): per chunk GiB from RAM / disk, and the seconds the "
              f"disk part adds at {' or '.join(f'{x:g}' for x in deep)} GB/s")
        for fv_, fr_, tok, gr, gd, add in fulls:
            print(f"    RAM {fr_:4.0%}: {tok:5.0f} tokens, {gr:5.2f} / {gd:5.2f} GiB, +" + " / +".join(f"{t:.2f} s" for t in add))
        fulls.clear()
