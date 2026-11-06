import time, torch
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor as view
GiB=1<<30
print("torch", torch.__version__, torch.cuda.get_device_name())
# VRAM read baseline, well under capacity
b=[torch.empty(GiB,dtype=torch.uint8,device="cuda").fill_(1) for _ in range(6)]
for rnd in range(2):
    ts=[]
    for x in b:
        t0=time.perf_counter(); s=x.sum(dtype=torch.long); s.item(); ts.append(time.perf_counter()-t0)
    print("VRAM sum(6GiB resident) ms/GiB:", " ".join(f"{t*1e3:.0f}" for t in ts))
del b; torch.cuda.empty_cache()
host=torch.empty(2*GiB,dtype=torch.uint8).fill_(1).pin_memory()
uva=view(host)
print("view device", uva.device)
dst=torch.empty(2*GiB,dtype=torch.uint8,device="cuda")
def bench(fn,size,total=1<<30,rep=3):
    n=max(1,total//size); best=None
    for _ in range(rep):
        t0=time.perf_counter()
        for i in range(n):
            off=(i*size)%(host.numel()-size); fn(off,size)
        torch.cuda.synchronize()
        dt=time.perf_counter()-t0
        rate=n*size/dt/1e9; best=max(best or 0,rate)
    return best
dma=lambda o,s: dst[o:o+s].copy_(host[o:o+s],non_blocking=True)
acc=[]
# int32 loads: a uint8 reduction issues narrow reads, which over UVA measure transaction size, not the link
def read(o,s): acc.append(uva[o:o+s].view(torch.int32).sum(dtype=torch.long))
vram=lambda o,s: acc.append(dst[o:o+s].view(torch.int32).sum(dtype=torch.long))
print(f"{'size':>9} {'DMA GB/s':>9} {'UVAread':>9} {'VRAMread':>9}")
for size in [64<<10,256<<10,512<<10,1<<20,4<<20,16<<20,256<<20]:
    d=bench(dma,size); acc.clear(); u=bench(read,size); acc.clear(); v=bench(vram,size); acc.clear()
    print(f"{size>>10:>7}Ki {d:9.2f} {u:9.2f} {v:9.2f}",flush=True)
t0=time.perf_counter(); z=torch.empty_like(host).copy_(host); print("host memcpy GB/s", host.numel()/(time.perf_counter()-t0)/1e9)
