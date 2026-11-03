"""Replay cache policies over a topk trace. Each hit = expert read from VRAM; miss = read over the link."""
import sys, numpy as np, collections
Z = np.load(sys.argv[1] + ".npz", allow_pickle=True)
tags, layers, topk = Z["tags"], Z["layers"], Z["topk"]
E = 1 << int(np.ceil(np.log2(int(max(t.max() for t in topk)) + 1)))
L = int(layers.max()) + 1
ev = []  # (job, kind, phase, layer, ids[rows,k])
for tag, l, t in zip(tags, layers, topk):
    if tag == "none": continue
    j, kind = tag.split(":")
    ev.append((int(j), kind, "prefill" if t.shape[0] > 1 else "decode", int(l), t))
print(f"E={E} L={L} calls={len(ev)} rows decode={sum(e[4].shape[0] for e in ev if e[2]=='decode')/L:.0f} "
      f"prefill={sum(e[4].shape[0] for e in ev if e[2]=='prefill')/L:.0f} (tokens)  k={ev[0][4].shape[1]}")
FR = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]
def hist(sel):
    h = np.zeros((L, E))
    for e in ev:
        if sel(e): np.add.at(h[e[3]], e[4].ravel(), 1)
    return h
dec = hist(lambda e: e[2] == "decode"); pre = hist(lambda e: e[2] == "prefill")
def conc(h, name):
    s = -np.sort(-h, axis=1); c = np.cumsum(s, 1) / s.sum(1, keepdims=True)
    row = [c[:, int(f * E) - 1].mean() for f in FR]
    print(f"{name:26s} top-frac share: " + " ".join(f"{f:.0%}->{r:.1%}" for f, r in zip(FR, row)))
    return c
print(f"uniform would be           top-frac share: " + " ".join(f"{f:.0%}->{f:.1%}" for f in FR))
c = conc(dec, "decode (in-sample oracle)"); conc(pre, "prefill (in-sample oracle)")
print("per-layer decode share of top 40% experts:", " ".join(f"{x:.0%}" for x in c[:, int(.4*E)-1]))
top = (-dec).argsort(1)[:, :2]; print("hottest expert share per layer (decode): max %.1f%% median %.1f%%; floor expert share median %.3f%%" % (
    100*(dec.max(1)/dec.sum(1)).max(), 100*np.median(dec.max(1)/dec.sum(1)), 100*np.median(dec.min(1)/dec.sum(1))))
print("never-routed experts per layer in decode: median", int(np.median((dec == 0).sum(1))))
def static_hit(train, test, frac):
    C = int(frac * E); hit = tot = 0
    for l in range(L):
        keep = (-train[l]).argsort()[:C]
        hit += test[l][keep].sum(); tot += test[l].sum()
    return hit / tot
print("\n-- static, ranked on one workload, tested on another (decode reads) --")
kinds = sorted({e[1] for e in ev})
for held in kinds:
    tr = hist(lambda e: e[1] != held); te = hist(lambda e: e[1] == held and e[2] == "decode")
    if te.sum() == 0: continue
    print(f"held out {held:10s} " + " ".join(f"{static_hit(tr, te, f):.1%}" for f in FR))
tr = hist(lambda e: e[0] % 2 == 0); te = hist(lambda e: e[0] % 2 == 1 and e[2] == "decode")
print(f"odd/even job split   " + " ".join(f"{static_hit(tr, te, f):.1%}" for f in FR))
# one uniform-ish cold ranking from prefill only: can a calibration prompt set seed decode?
print(f"prefill -> decode    " + " ".join(f"{static_hit(pre, dec, f):.1%}" for f in FR))
# temporal
seq = collections.defaultdict(list)
for e in ev:
    if e[2] == "decode": seq[e[3]].append((e[0], e[4][0]))
ov = []
for l, s in seq.items():
    for (j1, a), (j2, b) in zip(s, s[1:]):
        if j1 == j2: ov.append(len(set(a.tolist()) & set(b.tolist())) / len(a))
print(f"\ntoken-to-token routed-set overlap (decode, same layer): {np.mean(ov):.1%}")

def batched_steps(n):
    """Decode only: n requests at a time share each forward step, a new one joining as one ends."""
    jobs = collections.defaultdict(list); cur = {}
    for e in ev:
        if e[2] != "decode": continue
        c = cur.get(e[0])
        if c is None or e[3] <= c[-1][3]:
            c = cur[e[0]] = []; jobs[e[0]].append(c)
        c.append(e)
    queue = collections.deque(sorted(jobs)); active = []; steps = []
    while queue or active:
        while queue and len(active) < n:
            active.append(collections.deque(jobs[queue.popleft()]))
        merged = {}
        for a in active:
            for e in a.popleft():
                merged.setdefault(e[3], []).append(e[4])
        active = [a for a in active if a]
        steps.append([(0, "mix", "decode", l, np.vstack(v)) for l, v in sorted(merged.items())])
    return steps

def run(frac, lag=1, budget=None, prefill_touch=True, policy="lru", first=None, steps=None):
    """Decode hits. A miss is served over the link this token; its promotion lands `lag` tokens later,
    at most `budget` promotions per token summed over all layers. Pointer flip only at token boundary."""
    C = int(frac * E); hit = tot = 0; fh = ft = 0; moved = 0
    state = [dict() for _ in range(L)]; clock = 0; pending = collections.deque()
    # group into forward steps
    cur = None
    if steps is None:
        steps = []
        for e in ev:
            if cur is None or e[3] <= cur[-1][3] or e[0] != cur[-1][0]:
                cur = []; steps.append(cur)
            cur.append(e)
    seen_job = {}
    for st in steps:
        clock += 1
        j, ph = st[0][0], st[0][2]
        n = seen_job[j] = seen_job.get(j, 0) + (1 if ph == "decode" else 0)
        while pending and pending[0][0] <= clock:
            _, l, i = pending.popleft()
            s = state[l]
            if i not in s:
                if len(s) >= C:
                    del s[min(s, key=s.get)]
                s[i] = clock; moved += 1
        want = []
        for e in st:
            l, ids = e[3], np.unique(e[4]).tolist() if ph != "decode" else e[4].ravel().tolist()
            s = state[l]
            for i in ids:
                h = i in s
                if ph == "decode":
                    tot += 1; hit += h
                    if first and n <= first: ft += 1; fh += h
                if h: s[i] = clock
                elif ph == "decode" or prefill_touch: want.append((l, i))
        if budget is not None and ph == "decode":
            # most-wanted first, ties spread over layers by a rotating start, so no layer starves
            cnt = collections.Counter(want)
            want = sorted(cnt, key=lambda k: (-cnt[k], (k[0] - clock) % L))[:budget]
        elif budget is not None: want = []
        elif ph != "decode": want = want if prefill_touch else []
        for l, i in want: pending.append((clock + lag, l, i))
    return (hit / tot, moved, (fh / ft if ft else 0))
dt = sum(e[4].shape[0] for e in ev if e[2] == "decode") / L
print(f"\n-- decode hit rate: miss served by link, promoted lag tokens later --")
print(f"{'config':34s}" + " ".join(f"{f:>6.0%}" for f in FR))
for name, kw in (("lag1 unlimited", dict()), ("lag2 unlimited", dict(lag=2)), ("lag4 unlimited", dict(lag=4)),
                 ("lag2 budget 64/tok", dict(lag=2, budget=64)), ("lag2 budget 32/tok", dict(lag=2, budget=32)),
                 ("lag2 budget 16/tok", dict(lag=2, budget=16)), ("lag2 budget 6/tok", dict(lag=2, budget=6)),
                 ("lag2 b32, prefill no-touch", dict(lag=2, budget=32, prefill_touch=False))):
    rs = [run(f, **kw) for f in FR]
    print(f"{name:34s}" + " ".join(f"{r[0]:6.1%}" for r in rs), f" promos/tok @40%: {rs[3][1]/dt:.1f}", flush=True)
rs = [run(f, lag=2, budget=32, first=32) for f in FR]
print(f"{'first 32 tokens of each request':34s}" + " ".join(f"{r[2]:6.1%}" for r in rs))

print("\n-- n concurrent requests sharing one cache, lag 2, promotion budget per step --")
print(f"{'concurrent':34s}" + " ".join(f"{f:>6.0%}" for f in FR))
for n in (1, 2, 4, 8):
    st = batched_steps(n)
    for b in sorted({32, 32 * n}):
        print(f"{f'{n} (budget {b})':34s}" + " ".join(f"{run(f, lag=2, budget=b, steps=st)[0]:6.1%}" for f in FR), flush=True)
