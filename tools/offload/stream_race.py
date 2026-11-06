"""Do two EXL3 GEMMs on different CUDA streams interfere? vLLM runs a MoE layer's shared expert on an
aux stream while the routed experts run on the main one (batches <= 256 tokens), and exllamav3's
GEMM kernels synchronize their split-K reduction through one per-device lock buffer.

Runs a dense exl3_gemm (stream A) and the routed exl3_mgemm path (stream B) concurrently, ITERS
times, and compares each result bit for bit with a serial reference. A watchdog reports a hang.
usage: stream_race.py [uva|vram] [ITERS] [TOKENS]
"""
import os, sys, threading, time, torch
from vllm_exl3_plugin import cpu_offload
from vllm_exl3_plugin.ops import _exl3_moe_mm, ext
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

where = sys.argv[1] if len(sys.argv) > 1 else "uva"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
M = int(sys.argv[3]) if len(sys.argv) > 3 else 2
BITS, H, I, E, TOPK, DI, dev = 2, 2048, 512, 128, 8, 1024, "cuda"
torch.manual_seed(0)

def trellis(*shape_kn):
    *lead, k, n = shape_kn
    return torch.randint(-32768, 32767, (*lead, k // 16, n // 16, 16 * BITS), dtype=torch.int16)

def sign(*s):
    return ((torch.randint(0, 2, s) * 2 - 1).half() * 0.05).to(dev)

# shared expert stand-in: a dense K=H -> N=DI projection, then DI -> H, as the shared MLP's
dense = [(trellis(H, DI).to(dev), sign(H), sign(DI)), (trellis(DI, H).to(dev), sign(DI), sign(H))]

def dense_mm(x):
    for w, su, sv in dense:
        y = torch.empty((x.shape[0], w.shape[1] * 16), dtype=torch.half, device=dev)
        ext().exl3_gemm(x, w, y, su, torch.empty_like(x), sv, -1, False, True, 0)
        x = y
    return x

# routed experts, trellis in VRAM or pinned host behind a UVA view
tabs = {}
for p, (k, n) in {"gate": (H, I), "up": (H, I), "down": (I, H)}.items():
    w = trellis(E, k, n)
    if where == "uva":
        h = cpu_offload._pinned(w.shape, torch.int16); h.copy_(w); w = get_accelerator_view_from_cpu_tensor(h)
    else:
        w = w.to(dev)
    tabs[p] = (w, torch.tensor([w[e].data_ptr() for e in range(E)], device=dev),
               torch.tensor([s.data_ptr() for s in sign(E, k)], device=dev),
               torch.tensor([s.data_ptr() for s in sign(E, n)], device=dev))
scales = {p: (sign(E, k), sign(E, n)) for p, (k, n) in {"gate": (H, I), "up": (H, I), "down": (I, H)}.items()}
for p in tabs:  # keep scale storage alive and point the tables at it
    su, sv = scales[p]
    tabs[p] = (tabs[p][0], tabs[p][1], torch.tensor([su[e].data_ptr() for e in range(E)], device=dev),
               torch.tensor([sv[e].data_ptr() for e in range(E)], device=dev))

def moe(x, ids, w):
    t = tabs
    return _exl3_moe_mm(x, ids, w, *t["gate"][1:], *t["up"][1:], *t["down"][1:], BITS, BITS, I, E,
                        False, True, "silu")

xd = (torch.randn(M, H, device=dev) * 0.5).half()
xm = (torch.randn(M, H, device=dev) * 0.5).half()
ids = torch.stack([torch.randperm(E)[:TOPK] for _ in range(M)]).to(dev, torch.int32)
wt = torch.softmax(torch.randn(M, TOPK, device=dev), -1)
ref_d, ref_m = dense_mm(xd).clone(), moe(xm, ids, wt).clone()
torch.cuda.synchronize()

last = [time.time()]
def watchdog():
    while True:
        time.sleep(5)
        if time.time() - last[0] > 30:
            print(f"HANG: no progress for 30 s ({where}, M={M})", flush=True); os._exit(2)
threading.Thread(target=watchdog, daemon=True).start()

sa, sb = torch.cuda.Stream(), torch.cuda.Stream()
bad_d = bad_m = 0
t0 = time.time()
for i in range(ITERS):
    with torch.cuda.stream(sb): ym = moe(xm, ids, wt)
    with torch.cuda.stream(sa): yd = dense_mm(xd)
    torch.cuda.synchronize()
    bad_d += not torch.equal(yd, ref_d); bad_m += not torch.equal(ym, ref_m)
    last[0] = time.time()
print(f"{where} M={M}: {ITERS} concurrent rounds in {time.time() - t0:.1f} s; dense wrong {bad_d}, "
      f"moe wrong {bad_m}", flush=True)
