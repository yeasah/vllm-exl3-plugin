"""Phase 2: prefill-shaped MoE layers with half the experts in pinned host blocks, timed four ways.

resident  every expert in VRAM
uva       offloaded experts read in place over UVA
staged    each layer's host blocks copied into one VRAM window in stream, then the GEMM on it
copy      the staging copies alone

Synthetic weights at gemma-4-26B's expert shape; random top-k routing.
usage: stage_bench.py [bits] [tokens] [layers]   CODEBOOK=mcg for the older codebook (default mul1)
"""
import os, sys, torch
from vllm_exl3_plugin import cpu_offload
from vllm_exl3_plugin.ops import _exl3_moe_mm
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

BITS, M, L = (int(a) for a in (sys.argv[1:] + ["4", "1024", "6"][len(sys.argv) - 1:]))
H, I, E, OFF, TOPK, dev = 2816, 768, 128, 64, 8, "cuda"
MCG = os.environ.get("CODEBOOK", "mul1") == "mcg"
torch.manual_seed(0)
shapes = {"gate": (H, I), "up": (H, I), "down": (I, H)}

def trellis(n, k, m):
    return torch.randint(-32768, 32767, (n, k // 16, m // 16, 16 * BITS), dtype=torch.int16)

class Layer:
    def __init__(self):
        self.res, self.host, self.scales = {}, {}, {}
        for p, (k, n) in shapes.items():
            w = trellis(E, k, n)
            self.res[p] = w.to(dev)
            h = cpu_offload._pinned((OFF, *w.shape[1:]), torch.int16); h.copy_(w[E - OFF:])
            self.host[p] = h
            sign = lambda d: ((torch.randint(0, 2, (E, d)) * 2 - 1).half() * 0.05).to(dev)
            self.scales[p] = (sign(k), sign(n))
        ptrs = lambda t: [t[e].data_ptr() for e in range(E)]
        self.uva_view = {p: get_accelerator_view_from_cpu_tensor(h) for p, h in self.host.items()}
        self.tab = {}
        self.tab["resident"] = {p: ptrs(self.res[p]) for p in shapes}
        self.tab["uva"] = {p: ptrs(self.res[p])[:E - OFF] + [self.uva_view[p][r].data_ptr() for r in range(OFF)]
                           for p in shapes}

    def stage_table(self, window, offs):
        self.tab["staged"] = {}
        for p in shapes:
            row = self.host[p][0].numel() * 2
            self.tab["staged"][p] = self.tab["resident"][p][:E - OFF] + [window.data_ptr() + offs[p] + r * row
                                                                      for r in range(OFF)]
        self.copies = [(window[offs[p]:offs[p] + self.host[p].numel() * 2], self.host[p].view(-1).view(torch.uint8))
                       for p in shapes]
        self.dev_tab = {k: {p: torch.tensor(v, dtype=torch.long, device=dev) for p, v in t.items()}
                        for k, t in self.tab.items()}

    def run(self, mode, x, ids, w):
        if mode in ("staged", "copy"):
            for dst, src in self.copies: dst.copy_(src, non_blocking=True)
            if mode == "copy": return None
        t = self.dev_tab[mode]
        return _exl3_moe_mm(x, ids, w, t["gate"], self.sp["gate"][0], self.sp["gate"][1],
                            t["up"], self.sp["up"][0], self.sp["up"][1],
                            t["down"], self.sp["down"][0], self.sp["down"][1],
                            BITS, BITS, I, E, MCG, not MCG, "silu")

layers = [Layer() for _ in range(L)]
size = sum(-(-h.numel() * 2 // 256) * 256 for h in layers[0].host.values())
window = torch.empty(size, dtype=torch.uint8, device=dev)
for lay in layers:
    offs, o = {}, 0
    for p, h in lay.host.items(): offs[p] = o; o += -(-h.numel() * 2 // 256) * 256
    lay.stage_table(window, offs)
    lay.sp = {p: tuple(torch.tensor([s[e].data_ptr() for e in range(E)], dtype=torch.long, device=dev)
                       for s in lay.scales[p]) for p in shapes}

x = (torch.randn(M, H, device=dev) * 0.5).half()
ids = torch.stack([torch.randperm(E)[:TOPK] for _ in range(M)]).to(dev, torch.int32)
w = torch.softmax(torch.randn(M, TOPK, device=dev), -1)
off_bytes = size * L
print(f"{'mcg' if MCG else 'mul1'} K={BITS} M={M} layers={L}: {off_bytes / 2**20:.0f} MiB offloaded ({OFF}/{E} experts)", flush=True)

ref = {}
for mode in ("resident", "uva", "staged", "copy"):
    for _ in range(2):
        for lay in layers: y = lay.run(mode, x, ids, w)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(3):
        a.record()
        for lay in layers: y = lay.run(mode, x, ids, w)
        b.record(); b.synchronize(); ts.append(a.elapsed_time(b))
    ms = min(ts) / L
    if y is not None: ref[mode] = y.clone()
    extra = f"  = {size / ms / 1e6:.1f} GB/s" if mode == "copy" else ""
    print(f"{mode:9s} {ms:7.2f} ms/layer  {ms * 1e3 / M:6.1f} us/token{extra}", flush=True)
print("uva == resident:", torch.equal(ref["uva"], ref["resident"]),
      " staged == resident:", torch.equal(ref["staged"], ref["resident"]))
