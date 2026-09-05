#!/usr/bin/env python3
"""What does it cost to move scattered KV blocks over PCIe, really?

The granularity curve this project has been designing against was measured on
**CUDA managed memory**: fault-driven migration, where a 16-token granule ran at
4.1 GB/s and the plateau only arrived near 1 MiB -- 1024 consecutive token
positions. That number forced an uncomfortable conclusion, that a residency
policy would have to score *spans* of context rather than blocks, which is
coarse enough to wreck the attention-sink structure the whole idea leans on.

But the design that came out of `blocktable_evict.py` does not fault. It moves
blocks explicitly, and explicit DMA has a completely different cost structure:
page-migration granularity is irrelevant, and what remains is per-transfer
submission overhead. This measures that instead of assuming it, using vLLM's own
primitives rather than a synthetic proxy:

    swap_blocks        one call per layer, a (src, dst) block mapping inside it
    swap_blocks_batch  one driver call for every copy, via cuMemcpyBatchAsync
                       on CUDA 12.8+ (a loop of cudaMemcpyAsync below that)

Both are what a pager would actually call, and the difference between them is
the submission overhead alone -- same bytes, same scatter, same direction.

The geometry is per *layer*, because that is how a KV cache is laid out: one
tensor per attention layer, and a block's data within a layer is contiguous.
Moving one block of context therefore means `layers` separate copies, and the
scatter is across the layer tensors as much as within them. Defaults model
Qwen3.8-27B at fp8 -- 16 attention layers of 64 (the rest are GDN and hold no
pageable KV), 2 KiB per token per layer, so a 16-token block is 32 KiB.

    tools/kv_transport.py sweep [--blocks N] [--layers N] [--block-bytes N]
    tools/kv_transport.py verify

`verify` checks the copies actually land where the mapping says, because a
bandwidth number from a transfer that silently did nothing is the failure this
tool would otherwise report as a triumph.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

# One decode step at the 48.4 t/s measured on this box for the 27B; the unit
# every transfer cost here is ultimately judged against.
STEP_MS = 20.7
#: Fraction of a decode step a pager is willing to spend on transfer. The
#: budget it implies is *absolute* -- so many tokens per step, whatever the
#: context length -- which is the invariant a residency policy has to bound.
LATENCY_BUDGET = 0.05


def _ops():
    from vllm import _custom_ops as ops

    return ops


def make_pools(num_blocks_host, num_blocks_gpu, block_bytes, layers):
    """A host-side block pool and a GPU-side one, per layer.

    Byte tensors rather than a real KV shape: `swap_blocks` takes the block
    size in bytes and treats each tensor as consecutive contiguous blocks, so
    the layout that matters is exactly this one. Host memory is pinned because
    an unpinned source makes the driver stage through its own bounce buffer and
    measures that instead of the link.
    """
    host = [
        torch.empty(num_blocks_host * block_bytes, dtype=torch.uint8,
                    pin_memory=True)
        for _ in range(layers)
    ]
    gpu = [
        torch.empty(num_blocks_gpu * block_bytes, dtype=torch.uint8,
                    device="cuda")
        for _ in range(layers)
    ]
    return host, gpu


def time_cuda(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters       # ms


def block_indices(num_blocks_host, n, granule, seed=0):
    """`n` source blocks drawn in runs of `granule` consecutive blocks.

    Runs rather than singletons because that is the shape a policy produces: a
    residency decision covers a span of context, and the question is how much
    coalescing buys. granule=1 is the pessimal case and the one the design
    needs to know about.
    """
    g = torch.Generator().manual_seed(seed)
    runs = max(1, n // granule)
    starts = torch.randperm(max(1, num_blocks_host // granule), generator=g)[:runs]
    idx = (starts.unsqueeze(1) * granule + torch.arange(granule)).flatten()
    return idx[:n].to(torch.int64)


def arm_swap_blocks(ops, host, gpu, src_idx, dst_idx, block_bytes):
    mapping = torch.stack([src_idx, dst_idx], dim=1).contiguous()

    def run():
        for h, g in zip(host, gpu):
            ops.swap_blocks(h, g, block_bytes, mapping)

    return run


def arm_swap_batch(ops, host, gpu, src_idx, dst_idx, block_bytes):
    """Every copy for every layer submitted in one driver call."""
    src_ptrs, dst_ptrs = [], []
    for h, g in zip(host, gpu):
        src_ptrs.append(h.data_ptr() + src_idx * block_bytes)
        dst_ptrs.append(g.data_ptr() + dst_idx * block_bytes)
    src = torch.cat(src_ptrs).to(torch.int64)
    dst = torch.cat(dst_ptrs).to(torch.int64)
    sizes = torch.full((src.numel(),), block_bytes, dtype=torch.int64)

    def run():
        ops.swap_blocks_batch(src, dst, sizes)

    return run


def arm_coalesced(ops, host, gpu, src_idx, dst_idx, block_bytes, granule):
    """One copy per *run* instead of one per block.

    The granule sweep on its own answers whether locality matters; this answers
    the different and more useful question of whether transfer *size* does. A
    policy that evicts in runs can hand the DMA engine one descriptor for the
    whole run, and the gap between this and the per-block arms is the
    per-transfer overhead a pager would be paying for nothing.

    Runs are only coalescible when they are contiguous in *both* pools, which
    is why the destination is laid out to match; a real pager controls its own
    host-side layout and can arrange exactly that.
    """
    src_ptrs, dst_ptrs, sizes = [], [], []
    runs = range(0, src_idx.numel(), granule)
    for h, g in zip(host, gpu):
        for start in runs:
            n = min(granule, src_idx.numel() - start)
            src_ptrs.append(h.data_ptr() + int(src_idx[start]) * block_bytes)
            dst_ptrs.append(g.data_ptr() + int(dst_idx[start]) * block_bytes)
            sizes.append(n * block_bytes)
    src = torch.tensor(src_ptrs, dtype=torch.int64)
    dst = torch.tensor(dst_ptrs, dtype=torch.int64)
    sz = torch.tensor(sizes, dtype=torch.int64)

    def run():
        ops.swap_blocks_batch(src, dst, sz)

    return run


def arm_contiguous(host, gpu, n, block_bytes):
    """The ceiling: the same bytes as one copy per layer, no scatter at all."""
    nbytes = n * block_bytes

    def run():
        for h, g in zip(host, gpu):
            g[:nbytes].copy_(h[:nbytes], non_blocking=True)

    return run


def verify(args):
    """Prove the copies land where the mapping says before trusting any timing."""
    ops = _ops()
    block_bytes, layers = args.block_bytes, 2
    host, gpu = make_pools(64, 64, block_bytes, layers)
    for i, h in enumerate(host):
        view = h.view(64, block_bytes)
        for b in range(64):
            view[b] = (b * 7 + i) % 251
    for g in gpu:
        g.zero_()
    src = torch.tensor([5, 9, 40], dtype=torch.int64)
    dst = torch.tensor([1, 62, 7], dtype=torch.int64)
    arm_swap_blocks(ops, host, gpu, src, dst, block_bytes)()
    torch.cuda.synchronize()
    ok = True
    for i, (h, g) in enumerate(zip(host, gpu)):
        hv, gv = h.view(64, block_bytes), g.view(64, block_bytes).cpu()
        for s, d in zip(src.tolist(), dst.tolist()):
            if not torch.equal(hv[s], gv[d]):
                print(f"  layer {i}: block {s} -> {d} MISMATCH")
                ok = False
    # And that it moved nothing else.
    untouched = set(range(64)) - set(dst.tolist())
    for i, g in enumerate(gpu):
        gv = g.view(64, block_bytes).cpu()
        if any(gv[b].any() for b in untouched):
            print(f"  layer {i}: wrote outside the mapping")
            ok = False
    print("swap_blocks moves exactly the mapped blocks:", "yes" if ok else "NO")

    batch = arm_swap_batch(ops, host, gpu, src, dst, block_bytes)
    for g in gpu:
        g.zero_()
    batch()
    torch.cuda.synchronize()
    ok2 = all(
        torch.equal(h.view(64, block_bytes)[s], g.view(64, block_bytes).cpu()[d])
        for h, g in zip(host, gpu)
        for s, d in zip(src.tolist(), dst.tolist())
    )
    print("swap_blocks_batch agrees with it:", "yes" if ok2 else "NO")
    return ok and ok2


def sweep(args):
    ops = _ops()
    torch.cuda.init()
    host, gpu = make_pools(args.pool_blocks, args.pool_blocks,
                           args.block_bytes, args.layers)
    # Destination laid out in the same run order as the source, so a run is
    # one descriptor on both sides -- the layout a pager controls anyway.
    dst_idx = torch.arange(args.blocks, dtype=torch.int64)
    bytes_moved = args.blocks * args.block_bytes * args.layers

    rows = []
    contig_ms = time_cuda(arm_contiguous(host, gpu, args.blocks, args.block_bytes))
    for granule in args.granules:
        src_idx = block_indices(args.pool_blocks, args.blocks, granule)
        r = {"granule_blocks": granule,
             "granule_tokens": granule * args.tokens_per_block,
             "granule_bytes": granule * args.block_bytes}
        for name, fn in (
            ("swap_blocks", arm_swap_blocks(ops, host, gpu, src_idx, dst_idx,
                                            args.block_bytes)),
            ("swap_batch", arm_swap_batch(ops, host, gpu, src_idx, dst_idx,
                                          args.block_bytes)),
            ("coalesced", arm_coalesced(ops, host, gpu, src_idx, dst_idx,
                                        args.block_bytes, granule)),
        ):
            ms = time_cuda(fn)
            gbps = bytes_moved / (ms * 1e-3) / 1e9
            r[name] = {"ms": ms, "gbps": gbps,
                       "pct_step": 100 * ms / STEP_MS,
                       "tokens_per_step": tokens_per_step(gbps, args)}
        rows.append(r)

    out = {
        "latency_budget": LATENCY_BUDGET, "step_ms": STEP_MS,
        "blocks": args.blocks, "layers": args.layers,
        "block_bytes": args.block_bytes,
        "tokens_per_block": args.tokens_per_block,
        "bytes_moved": bytes_moved,
        "copies_per_call": args.blocks * args.layers,
        "contiguous": {"ms": contig_ms,
                       "gbps": bytes_moved / (contig_ms * 1e-3) / 1e9,
                       "pct_step": 100 * contig_ms / STEP_MS},
        "rows": rows,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    report(out)


def tokens_per_step(gbps, args):
    """Tokens fetchable per decode step inside the latency budget.

    The number a policy is actually constrained by: a context fraction means
    different things at 32K and 300K, but this does not move.
    """
    per_token = args.block_bytes * args.layers / args.tokens_per_block
    return gbps * 1e9 * (STEP_MS * 1e-3 * LATENCY_BUDGET) / per_token


def report(out):
    print(f"\n{out['blocks']} blocks x {out['layers']} layers = "
          f"{out['copies_per_call']} copies, "
          f"{out['bytes_moved'] / 2**20:.1f} MiB, "
          f"{out['block_bytes'] // 1024} KiB per copy "
          f"({out['tokens_per_block']} tokens/block)")
    c = out["contiguous"]
    print(f"  contiguous ceiling: {c['gbps']:6.1f} GB/s  {c['ms']:.3f} ms  "
          f"{c['pct_step']:.1f}% of a decode step")
    print(f"  per token, all layers: "
          f"{out['block_bytes'] * out['layers'] // out['tokens_per_block'] / 1024:.0f}"
          f" KiB   (tokens/step = what fits in {int(LATENCY_BUDGET * 100)}%"
          f" of a {STEP_MS} ms step)")
    # The whole table collapses to two constants, and saying which two is more
    # useful than the table: a copy costs a fixed submission overhead plus its
    # bytes at link rate, so cost = n_copies * overhead + bytes / bandwidth.
    # Locality is absent from that expression, which is the finding.
    per_block = out["rows"][0]["swap_blocks"]
    per_copy_us = per_block["ms"] * 1e3 / out["copies_per_call"]
    xfer_us = out["block_bytes"] / (c["gbps"] * 1e9) * 1e6
    print(f"  implied cost model: {per_copy_us - xfer_us:.2f} us per copy "
          f"submitted + bytes at {c['gbps']:.0f} GB/s "
          f"({per_copy_us:.2f} us measured for a "
          f"{out['block_bytes'] // 1024} KiB copy, {xfer_us:.2f} of it transfer)")
    print(f"  {'granule':>18}  {'per-block':>21}  {'batched':>21}  "
          f"{'one copy per run':>21}")
    for r in out["rows"]:
        a, b, c = r["swap_blocks"], r["swap_batch"], r["coalesced"]
        print(f"  {r['granule_blocks']:>4} blk /{r['granule_tokens']:>5} tok  "
              f"{a['gbps']:7.1f} GB/s {a['tokens_per_step']:6.0f} tok  "
              f"{b['gbps']:7.1f} GB/s {b['tokens_per_step']:6.0f} tok  "
              f"{c['gbps']:7.1f} GB/s {c['tokens_per_step']:6.0f} tok")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--blocks", type=int, default=512)
    s.add_argument("--layers", type=int, default=16)
    s.add_argument("--block-bytes", type=int, default=32 * 1024)
    s.add_argument("--tokens-per-block", type=int, default=16)
    s.add_argument("--pool-blocks", type=int, default=4096)
    s.add_argument("--granules", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32, 64])
    s.add_argument("--out", default="")
    s.set_defaults(func=sweep)
    v = sub.add_parser("verify")
    v.add_argument("--block-bytes", type=int, default=32 * 1024)
    v.set_defaults(func=verify)
    r = sub.add_parser("report")
    r.add_argument("out", nargs="+")
    r.set_defaults(func=lambda a: [report(json.load(open(f))) for f in a.out])
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
