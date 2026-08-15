# Architecture-conditional code in exllamav3

*Where exllamav3 changes behaviour by GPU architecture. Consult before running on
hardware other than the sm_120 Blackwell most figures were measured on.*

A census, not an analysis. We lost most of a day to one arch-gated barrier
(fixed directly in [our exllamav3 fork](https://github.com/yeasah/exllamav3),
formerly `patches/exllamav3-sm90-barrier.patch`), so this is an index of
*where else*
exllamav3 changes behaviour by GPU architecture — somewhere to look first when
the next baffling thing turns up, and a checklist for when we run on hardware
other than the sm_120 Blackwell everything so far was measured on.

Not exhaustive as a risk register: plenty of divergence is untyped (different
kernel shapes, different occupancy, tuner picking differently). But
`#if __CUDA_ARCH__` and `switch (cc)` are grep-able, and that is the point.

Taken against exllamav3 v1.3.0, the vendored submodule.

## What we actually call

Scoping matters — most of the extension is unreachable from the plugin.
`vllm_exl3_plugin/ops.py` calls exactly four entry points:

    ext().reconstruct     dequantize a trellis (Phase 0 oracle, and the
                          reconstruct-above-threshold path)
    ext().exl3_gemm       dense quantized linear
    e.exl3_mgemm          routed experts, pointer-table batched
    e.had_r_128           Hadamard transform

Everything below is flagged for whether it sits under one of those.

## Compile-time `__CUDA_ARCH__` gates

| file | gate | switches | ours? |
|---|---|---|---|
| `quant/exl3_gemm_kernel.cuh` | `> 890` (3 sites) | hand-rolled `group_barrier` instead of `grid.sync()` | **yes — this is the one that bit us**, now patched to opt-in |
| `quant/exl3_gemm_inner.cuh` | `== 860` | `EXL3_GEMM_H_ACC`, fp32-accumulate HMMA | no (sm_86 only; comment says "unvalidated on other archs") |
| `quant/codebook.cuh` | `== 860` | inline-PTX codebook decode variant | no (sm_86 only) |
| `routing.cu` | `>= 800 && !USE_ROCM` | hardware max-reduce for top-k routing | no — vLLM does its own routing |
| `compat.cuh` | `< 750 \|\| CUDART < 11000` | `tanh_opt` fallback | no (we are sm_80+) |
| `cache/lmq.cuh` | `defined()` only | host/device clamp macro | no (cache code, unused) |

Only four distinct thresholds exist in the whole extension: `> 890`, `== 860`,
`>= 800`, `< 750`. The `== 860` pair is interesting mainly as precedent — it
shows the project does ship narrowly-scoped per-arch code paths that other
architectures never execute, and therefore that never get tested elsewhere.

## Runtime compute-capability branches

`DevCtx::get_cc()` buckets devices (`quant/exl3_devctx.cu:39-43`):

    major >= 10          -> CC_BLACKWELL
    major >= 9           -> CC_HOPPER
    major >= 8, minor 9  -> CC_ADA
    major >= 8           -> CC_AMPERE
    else                 -> CC_OLD

Consumers:

| file | what it selects | ours? |
|---|---|---|
| `quant/exl3_kernel_map.cu` | **kernel shape per cc** — a different heuristic tree for `{OLD,AMPERE}`, `ADA`, and `{HOPPER,BLACKWELL}` | **yes** — every `exl3_gemm`/`exl3_mgemm` call goes through this |
| `quant/exl3_gemv_int8.cu` | int8-GEMV bit threshold (6 on Hopper/Blackwell, 5 elsewhere) | only under `EXL3_INT8_GEMV=1`, which we do not set |
| `quant/exl3_gemv.cu` | GEMV shape heuristics, several `CC_ADA`/`CC_AMPERE` special cases | yes, for small-m dense calls |
| `quant/exl3_moe.cu` | passes cc into kernel selection | no — we use `exl3_mgemm`, not `exl3_moe` |

`exl3_kernel_map.cu` is the significant one: Hopper and Blackwell share a
branch, so **our card is validated by whatever Hopper testing exists**, and
Ampere/Ada take genuinely different shape decisions. Any performance figure in
this repo is therefore a Blackwell figure, and the cloud Ampere box will
exercise a different tree.

## Shared device-global state

Distinct from arch gating, and the actual root cause of the hang we hit: several
kernels synchronize through one per-device buffer, `DevCtx::get_locks(device)`,
allocated once and zeroed once.

    MAX_TILES_C          tile locks for the GEMM reduction
    BARRIER_LOCKS_OFFSET 2 * MAX_BARRIERS ints, sense-reversing barrier counters
    MOE_SCHED_OFFSET     MoE expert-scheduler tickets

Users: `exl3_gemm.cu`, `exl3_gemv.cu`, `exl3_moe.cu`, and the two kernel headers.

**`quant/exl3_moe_kernel.cuh` uses `group_barrier` unconditionally** — five call
sites, no `__CUDA_ARCH__` guard — plus the device-global expert scheduler at
`MOE_SCHED_OFFSET`. That is the same hand-rolled barrier over the same shared
buffer that deadlocked us on sm_90+, except here it is on *every* architecture.

We do not call `exl3_moe` today. It is the obvious next optimization if we ever
want a fully-fused MoE instead of three `exl3_mgemm` calls, so: **if that is
ever attempted, expect this class of bug immediately, on all hardware.**

## Tensor parallelism

`parallel/` (`all_reduce.cu`, `all_reduce_cpu.cu`, `gather.cu`) has **no arch
conditionals** — no `__CUDA_ARCH__`, no `cc` branching — and uses `grid.sync()`
uniformly. It does use cooperative launches.

It is also almost certainly dead code for us: vLLM does its own collectives, and
our TP work shards weights and lets vLLM all-reduce. Recorded so the next person
does not go looking for hazards in it.

## What this predicts for non-Blackwell hardware

- The barrier patch is a **no-op below sm_90** — that path already used
  `grid.sync()`. Applying it on an Ampere box changes nothing, so it is safe to
  keep applied unconditionally.
- `exl3_kernel_map.cu` will select **different kernel shapes**, so throughput
  numbers will not transfer and the autotuner will make different choices.
- `exl3_gemv.cu` has Ampere- and Ada-specific fast paths that our Blackwell runs
  have never executed.

Which means a cloud Ampere/Ada box is not just a TP test — it is the first
exercise of a materially different code path through the same kernels.
