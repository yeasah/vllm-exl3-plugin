"""Phase 0 spikes (b)-(d) on the real exl3_mgemm, synthetic weights, decode shape (1 token).

(b) pointer-table entries flipped between replays of a captured graph: does the replay follow them,
    and is a read over UVA bit-identical to the same bytes read from VRAM?
(c) cost of a captured per-layer topk_ids copy into a pinned host ring.
(d) a side-stream H2D copy overlapping replays: replay slowdown, and does the copy land in time?

usage: graph_spike.py [hidden] [intermediate] [bits] [experts] [layers]
"""
import sys, time, torch
from vllm_exl3_plugin.ops import _exl3_moe_mm

H, I, BITS, E, L = (int(a) for a in (sys.argv[1:] + ["2816", "1408", "4", "128", "30"][len(sys.argv) - 1:]))
TOPK, dev = 8, "cuda"
torch.manual_seed(0)

def trellis(k, n):
    return torch.randint(-32768, 32767, (k // 16, n // 16, 16 * BITS), dtype=torch.int16)

def sign(n):
    return (torch.randint(0, 2, (n,)) * 2 - 1).half()

def expert():
    """gate, up, down as {trellis, suh, svh}, on the CPU."""
    return {"gate": (trellis(H, I), sign(H), sign(I) * 0.05), "up": (trellis(H, I), sign(H), sign(I) * 0.05),
            "down": (trellis(I, H), sign(I), sign(H) * 0.05)}

class Layer:
    """One MoE layer: VRAM slots, a pinned host copy of the same bytes, and pinned alternates."""
    def __init__(self):
        cpu = [expert() for _ in range(E)]
        alt = [expert() for _ in range(E)]
        self.vram = [{p: tuple(t.to(dev) for t in v) for p, v in x.items()} for x in cpu]
        self.host = [{p: tuple(t.pin_memory() for t in v) for p, v in x.items()} for x in cpu]
        self.alt = [{p: (v[0].pin_memory(),) + tuple(t.to(dev) for t in v[1:]) for p, v in x.items()} for x in alt]
        # scales stay resident, as in the design; only trellis entries move
        for a, v in zip(self.alt, self.vram):
            for p in a: a[p] = (a[p][0],) + v[p][1:]
        self.tab = {p: [torch.tensor([x[p][i].data_ptr() for x in self.vram], dtype=torch.long, device=dev)
                        for i in range(3)] for p in ("gate", "up", "down")}

    def point(self, src, ids):
        """Point trellis entries `ids` at `src` ('vram', 'host', 'alt'), in stream."""
        for p in ("gate", "up", "down"):
            new = torch.tensor([getattr(self, src)[e][p][0].data_ptr() for e in ids], dtype=torch.long)
            self.tab[p][0][torch.tensor(ids, device=dev)] = new.to(dev, non_blocking=True)

    def __call__(self, x, ids, w):
        t = self.tab
        return _exl3_moe_mm(x, ids, w, *t["gate"], *t["up"], *t["down"], BITS, BITS, I, E,
                            False, True, "silu")

def section(s): print(f"\n== {s}", flush=True)

section(f"(b) pointer flip under a captured graph  H={H} I={I} K={BITS} E={E}")
lay = Layer()
x = (torch.randn(1, H, device=dev) * 0.5).half()
ids = torch.randperm(E, device=dev)[:TOPK].view(1, TOPK).to(torch.int32)
w = torch.softmax(torch.randn(1, TOPK, device=dev), -1)
routed = ids.view(-1).tolist()
for _ in range(3): lay(x, ids, w)  # autotune outside capture
torch.cuda.synchronize()

def eager(src):
    lay.point(src, list(range(E))); y = lay(x, ids, w).clone(); lay.point("vram", list(range(E))); return y

ref_vram, ref_host, ref_alt = eager("vram"), eager("host"), eager("alt")
print("eager: finite", bool(torch.isfinite(ref_vram).all()), " |y| %.3f" % ref_vram.float().abs().mean().item())
print("eager host == eager vram (bitwise):", torch.equal(ref_host, ref_vram))
print("eager alt differs from vram:", not torch.equal(ref_alt, ref_vram))

g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    lay(x, ids, w)
    with torch.cuda.graph(g): out = lay(x, ids, w)
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()

def replay(src, which):
    lay.point(src, which); g.replay(); torch.cuda.synchronize(); return out.clone()

r = replay("vram", routed); print("replay, slots           == eager vram:", torch.equal(r, ref_vram))
r = replay("host", routed); print("replay, flipped to host == eager vram:", torch.equal(r, ref_vram))
r = replay("alt", routed);  print("replay, flipped to alt  == eager alt: ", torch.equal(r, ref_alt))
r = replay("vram", routed); print("replay, flipped back    == eager vram:", torch.equal(r, ref_vram))
half = routed[: TOPK // 2]; lay.point("alt", half)
lay.point("vram", routed[TOPK // 2:]); y_mix = lay(x, ids, w).clone()
r = replay("alt", half);    print("replay, half flipped    == eager half:", torch.equal(r, y_mix))
# routing changes between replays too, as it will every token
ids.copy_(torch.randperm(E, device=dev)[:TOPK].view(1, TOPK)); lay.point("vram", list(range(E)))
torch.cuda.synchronize(); y_new = lay(x, ids, w).clone()
g.replay(); torch.cuda.synchronize(); print("replay, new topk_ids    == eager:     ", torch.equal(out, y_new))

section(f"(c) captured topk_ids copy, {L} layers in one graph")
# one layer's weights reused, each position routed to its own experts so L2 does not serve repeats
lids = [torch.randperm(E, device=dev)[:TOPK].view(1, TOPK).to(torch.int32) for _ in range(L)]
ring = torch.empty((L, 1, TOPK), dtype=torch.int32).pin_memory()

def forward(log):
    h = x
    for i in range(L):
        if log: ring[i].copy_(lids[i], non_blocking=True)
        h = lay(h, lids[i], w)
    return h

def capture(log):
    gg = torch.cuda.CUDAGraph(); st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        forward(log)
        with torch.cuda.graph(gg): forward(log)
    torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize(); return gg

def timed(gg, n=300, during=None):
    for _ in range(10): gg.replay()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n):
        if during: during()
        gg.replay()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3

g0, g1 = capture(False), capture(True)
t0, t1 = min(timed(g0) for _ in range(3)), min(timed(g1) for _ in range(3))
ring.zero_(); g1.replay(); torch.cuda.synchronize()
ok = all(torch.equal(ring[i], lids[i].cpu()) for i in range(L))
print(f"replay {t0:.3f} ms without, {t1:.3f} ms with ring copies (+{(t1 - t0) * 1e3:.1f} us, "
      f"{(t1 - t0) / t0:+.1%}); ring holds every layer's topk_ids: {ok}")

section("(d) side-stream promotions overlapping replays")
side = torch.cuda.Stream()
src = [lay.host[e][p][0] for e in range(E) for p in ("gate", "up", "down")][:3 * 64]
dst = [torch.empty_like(t, device=dev) for t in src]
expert_mib = sum(t.numel() * 2 for t in src[:3]) / 2**20

def promote(k):
    with torch.cuda.stream(side):
        for j in range(3 * k): dst[j].copy_(src[j], non_blocking=True)
        e = torch.cuda.Event(); e.record(); return e

side.synchronize(); c0 = time.perf_counter(); promote(15); side.synchronize()
print(f"one expert {expert_mib:.2f} MiB; 15 experts copied alone in {(time.perf_counter() - c0) * 1e3:.3f} ms")
for k in (6, 15, 32, 64):
    t = min(timed(g0, during=lambda: promote(k)) for _ in range(3))
    done = 0
    for _ in range(50):
        e = promote(k); g0.replay(); torch.cuda.current_stream().synchronize(); done += e.query()
    side.synchronize()
    print(f"{k:3d} experts/step ({k * expert_mib:5.1f} MiB): replay {t:.3f} ms vs {t0:.3f} alone "
          f"({(t - t0) / t0:+.1%}); copy landed by end of replay {done}/50", flush=True)
