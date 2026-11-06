"""Pin the host blocks phase 1 needs: one block per layer, all experts. Defaults are gemma 4.10bpw
(30 x 415 MiB = 12.2 GiB). Checks pin time, whole-block DMA rate, and DMA and UVA readback.

usage: pin_probe.py [blocks] [MiB per block]
"""
import sys, time, torch
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor as view

N, MIB = (int(a) for a in (sys.argv[1:] + ["30", "415"][len(sys.argv) - 1:]))
n = MIB << 18  # int32 elements
print(f"{N} blocks x {MIB} MiB = {N * MIB / 1024:.1f} GiB pinned", flush=True)
blocks = []; t0 = time.perf_counter()
for i in range(N):
    b = torch.empty(n, dtype=torch.int32, pin_memory=True)
    b.fill_(i * 7919 + 1); blocks.append(b)
    if (i + 1) % 10 == 0: print(f"  {i + 1} pinned, {time.perf_counter() - t0:.1f} s", flush=True)
print(f"pinned in {time.perf_counter() - t0:.1f} s; all report pinned: {all(b.is_pinned() for b in blocks)}")

stage = torch.empty(n, dtype=torch.int32, device="cuda")
for p in ("first", "second"):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for b in blocks: stage.copy_(b, non_blocking=True)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    print(f"whole-block DMA, {p} pass: {N * MIB / 1024 / dt * 1.0737:.1f} GB/s, {dt / N * 1e3:.1f} ms per block")

bad = []
for i in (0, N // 2, N - 1):
    stage.copy_(blocks[i]); want = i * 7919 + 1
    dma_ok = bool((stage == want).all())
    uva = view(blocks[i])
    uva_ok = uva.device.type == "cuda" and bool((uva == want).all())
    if not (dma_ok and uva_ok): bad.append((i, dma_ok, uva_ok))
print("readback DMA and UVA correct:", not bad, bad or "")
