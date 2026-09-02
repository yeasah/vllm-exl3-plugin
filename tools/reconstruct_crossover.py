#!/usr/bin/env python3
"""Measure where `reconstruct` + cuBLAS overtakes the fused EXL3 kernel.

`RECONSTRUCT_THRESHOLD` defaults to 144, inherited from exllamav3's own
`AUTO_RECONSTRUCT_THRESHOLD`. End-to-end serving confirms the switch matters enormously
-- disabling it costs 40-80% of prefill throughput -- but says nothing about whether 144
is the right place for *our* shapes on *this* card, because a long prompt only ever
presents `chunk_size` rows and never exercises the region the threshold decides.

This measures the crossover directly: real weights from a checkpoint, both paths, across
row counts, with the reconstruct cost inside the timed region because serving pays it on
every call.
"""
import argparse
import glob
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_exl3_plugin import ops  # noqa: E402


def bench(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--layer", default="0")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--rows", default="1,8,16,32,64,96,128,144,192,256,384,512,768,1024")
    args = ap.parse_args()

    from safetensors import safe_open

    shards = sorted(glob.glob(os.path.join(args.checkpoint, "*.safetensors")))
    stems, handles = {}, []
    for f in shards:
        h = safe_open(f, "pt", device="cuda:0")
        handles.append(h)
        for k in h.keys():
            if f".layers.{args.layer}." in k and k.endswith(".trellis"):
                stems[k[: -len(".trellis")]] = h
    if not stems:
        sys.exit(f"no trellis tensors for layer {args.layer}")

    rows = [int(r) for r in args.rows.split(",")]
    print(f"{'module':22s} {'shape':>14s} " + "".join(f"{r:>8d}" for r in rows))
    print(f"{'':22s} {'':>14s} " + "".join(f"{'ratio':>8s}" for _ in rows))

    for stem in sorted(stems, key=lambda s: s.split(".")[-1]):
        h = stems[stem]
        trellis = h.get_tensor(f"{stem}.trellis")
        suh = h.get_tensor(f"{stem}.suh")
        svh = h.get_tensor(f"{stem}.svh")
        keys = set(h.keys())
        mcg = f"{stem}.mcg" in keys
        mul1 = f"{stem}.mul1" in keys
        k = trellis.shape[0] * 16
        n = trellis.shape[1] * 16
        bits = trellis.shape[2] // 16
        name = stem.split(".")[-1]

        out = []
        for r in rows:
            a = torch.randn((r, k), dtype=torch.half, device="cuda:0")

            def fused():
                y = torch.empty((r, n), dtype=torch.half, device="cuda:0")
                a_had = torch.empty_like(a)
                ops.ext().exl3_gemm(a, trellis, y, suh, a_had, svh, -1, mcg, mul1, 0)
                return y

            def recon():
                return ops._reconstruct_mm(a, trellis, suh, svh, mcg, mul1)

            try:
                tf = bench(fused, args.iters)
                tr = bench(recon, args.iters)
                out.append(f"{tf / tr:8.2f}")
            except torch.OutOfMemoryError:
                out.append(f"{'oom':>8s}")
            del a
            torch.cuda.empty_cache()
        print(f"{name:22s} {f'{k}x{n}@{bits}b':>14s} " + "".join(out))
        del trellis, suh, svh
        torch.cuda.empty_cache()

    print("\nratio = fused / reconstruct. >1 means reconstruct+cuBLAS is faster;")
    print("the crossover is where each row first exceeds 1.00.")


if __name__ == "__main__":
    main()
