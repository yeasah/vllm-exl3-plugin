"""Does this environment let CUDA allocate past physical VRAM, and what does touching it cost?"""
import time, torch
GiB = 1 << 30
def smi():
    f, t = torch.cuda.mem_get_info()
    return f"cudaMemGetInfo free {f/GiB:.2f}/{t/GiB:.2f}"
print(smi())
blocks = []
try:
    for i in range(16):
        t0 = time.perf_counter()
        b = torch.empty(GiB, dtype=torch.uint8, device="cuda")
        b.fill_(i); torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        blocks.append(b)
        print(f"block {i}: total {len(blocks)} GiB, alloc+fill {dt*1e3:.0f} ms, {smi()}", flush=True)
except torch.OutOfMemoryError as e:
    print("OOM at", len(blocks), "GiB:", str(e)[:120])
print("max_memory_allocated", torch.cuda.max_memory_allocated()/GiB)
# Read every block repeatedly: are they all truly resident, or is something paging?
for round in range(3):
    ts = []
    for b in blocks:
        t0 = time.perf_counter()
        s = b.sum(dtype=torch.long); torch.cuda.synchronize(); s.item()
        ts.append(time.perf_counter() - t0)
    print(f"read pass {round}: " + " ".join(f"{t*1e3:.0f}" for t in ts) + " ms/GiB")
