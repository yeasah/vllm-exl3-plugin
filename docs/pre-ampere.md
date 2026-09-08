# Pre-Ampere (Volta, Pascal) support for the EXL3 kernels

*Assessment, 2026-09-08. Sibling to [exllamav3-arch.md](exllamav3-arch.md), which
censuses arch-conditional code from sm_80 up; this one asks what happens below it.*

Motivation: V100 (sm_70, 32 GiB HBM2, 900 GB/s) and P100 (sm_60, 16 GiB HBM2,
732 GB/s) are among the last sources of cheap CUDA VRAM. The bandwidth premise
holds — a V100 matches an RTX 5070 Ti (896 GB/s) — so the question is whether the
kernels can be made to run at all.

**Verdict: Pascal no, Volta yes-but-the-toolchain-is-the-cost, and there is a
zero-kernel-work path worth testing before either.**

**Status (2026-09-08): Pascal ruled out for good. Volta assessed and parked — not
being acted on, recorded so the question does not get reopened cold.** Nothing
below is a task; if it ever becomes one, the entry point is "Recommended order"
and step 1 is the only cheap step.

## The toolchain is the wall, not the kernels

CUDA 13.0 removed offline compilation for Maxwell, Pascal and Volta. Measured on
the toolkit this stack uses:

    $ nvcc --version                 # release 13.3, V13.3.73
    $ nvcc --list-gpu-arch
    compute_75 compute_80 compute_86 compute_87 compute_88 compute_89
    compute_90 compute_100 compute_110 compute_103 compute_120 compute_121

sm_70 and sm_60 cannot be named. This is not a flag we are missing.

It cascades:

| layer | where sm_70 dies |
|---|---|
| nvcc | removed in CUDA 13.0; last support is the 12.x series |
| PyTorch | dropped sm_50/60/70 from cu128+ wheels as of 2.8 ([pytorch#157517](https://github.com/pytorch/pytorch/issues/157517)) |
| vLLM | `CMakeLists.txt:114-132` — `7.0` appears only in the `else()` branch, i.e. CUDA < 12.8 |

So Volta pins CUDA to ≤ 12.6-ish, PyTorch to ≤ 2.7 or a self-built wheel, and
vLLM to a release old enough to still list 7.0. That is a three-project fork that
widens every month, against a submodule we already struggle to keep current
(see [upstream.md](upstream.md)). The 1Cat-vLLM Volta fork
(<https://github.com/1CatAI/1Cat-vLLM>) recommends "CUDA 12.8, PyTorch 2.10,
SM70", which given the table above means they build torch themselves — the cost
is real and they are paying it.

**Pascal is dead on this axis alone**: vLLM's CMake has never listed 6.0 or 6.1
in any branch.

## Kernel work, measured rather than guessed

Compiling all 107 `.cu` files at `-arch=sm_75` (the lowest nvcc 13.3 accepts, and
therefore a lower bound on the sm_70 gap):

    50 fail, 57 build clean.

Every one of the 50 failures is one of exactly **two** PTX features:

- `cp.async` / `cp.async.commit_group` / `cp.async.wait_group` (sm_80+)
- `mma.sync.aligned.m16n8k16` (sm_80+)

They live in three headers — `quant/exl3_gemm_inner.cuh`,
`quant/exl3_gemv_kernel.cuh`, `quant/exl3_gemv_int8_kernel.cuh` — and reach the
48 comp_units plus `exl3_moe_kernel.cuh` by inclusion. Nothing else in the
extension is implicated. `ldmatrix` (sm_75), `__dp4a` (sm_61), `lop3.b32`
(sm_50), `__funnelshift_r`, half2 arithmetic and the `__shfl_*_sync` reductions
all assemble at sm_75.

Clean at sm_75, including everything on the plugin's reconstruct path:
`reconstruct.cu`, `quant/hadamard.cu`, `norm.cu`, `rope.cu`, `activation.cu`,
`attention.cu`, `routing.cu`, `dsa_topk.cu`, `stloader_cu.cu`,
`quant/exl3_devctx.cu`, `quant/exl3_kernel_map.cu`, `quant/pack.cu`,
`quant/quantize.cu`.

### A third blocker that does not show up at compile time

`exl3_gemm.cu:294` and `:614` request `SMEM_MAX` (90 KiB) via
`cudaFuncSetAttribute`, and `cudaLaunchCooperativeKernel` passes `SMEM_MAX` as
the dynamic shared-memory size — unconditionally, for every shape, regardless of
what the shape actually needs. Caps: **Turing 64 KiB, Volta 96 KiB.** So every
GEMM launch fails at runtime on sm_75 and would need the request sized to the
shape; on sm_70 the existing 90 KiB request fits as-is.

That inverts the usual ordering: **on shared memory, Volta is a friendlier target
than Turing.**

## What Volta needs beyond the Turing gap

| requirement | sm_70 | note |
|---|---|---|
| `cp.async` | absent | fall back to `ld.global` + `st.shared` with manual double-buffering — mechanical, costs latency hiding |
| `mma.m16n8k16` | absent | only `mma.m8n8k4`; a quarter the K depth |
| `ldmatrix` | absent (sm_75+) | A fragments must be hand-assembled from shared memory |
| `__dp4a` | present | codebook 2 decode is fine |
| full-rate fp16x2 | present | the whole `decode_3inst_2` chain is fine |
| `__nanosleep`, cooperative launch, ITS | present | `group_barrier` and `grid.sync()` both usable |
| 96 KiB smem/block | present | 90 KiB request fits |

The hard part is not `mma.m8n8k4` itself — `ptx.cuh:19-45` already carries
`ptx_mma_m8n8k4`, written and then abandoned ("emulated on Ampere and later,
don't use"). The hard part is that its operand layout is the Volta quad-pair
layout, which shares nothing with the m16n8k16 layout that `dq4`/`dq2x2`
(`quant/exl3_dq.cuh`) emit directly into `FragB`. Losing `ldmatrix` at the same
time means both the A and B fragment producers get rewritten, not adapted. That
is the bulk of the work, and it is the part that cannot be validated on any
hardware we can rent without first solving the toolchain problem.

## Pascal, specifically

Not a port. There are no tensor cores, so the GEMM becomes a SIMT fp16x2 or fp32
kernel — a different kernel, not a variant of this one. Worse, the two Pascal
dies split the requirements in a way that no single fallback satisfies:

- **P100 (sm_60)**: full-rate fp16x2, but **no `__dp4a`** — codebook 2's decode
  breaks.
- **P40 / GTX 10x0 (sm_61)**: has `__dp4a`, but fp16x2 runs at **1/64 rate** —
  the entire half2 codebook decode collapses.

Recommend dropping Pascal from consideration.

## The path that already exists

`vllm_exl3_plugin/ops.py:497` `_reconstruct_mm` — `reconstruct` + `had_r_128` +
a torch hgemm — is the plugin's correctness-oracle path, and **both kernels it
depends on already compile clean below sm_80** and use nothing newer than sm_60
intrinsics. On a CUDA 12.x / sm_70 build with the GEMM comp_units excluded, EXL3
weights would plausibly serve on a V100 today with zero kernel work.

The trade is bandwidth for capacity. Per forward pass reconstruct reads the
trellis (`k*n*bits/8`), writes a full fp16 weight (`2kn`) and reads it back
(`2kn`) — roughly **2x the traffic of just running the model in fp16**. It is
strictly worse than fp16 on a card where the model already fits, and the only
thing it buys is the case where the model does not fit. For a V100 appliance
that is exactly the case that matters, but it means the fused kernel is the
whole performance story, not a nice-to-have.

## Recommended order, cheapest first

1. **Confirm the reconstruct path on sm_70.** One rented V100 hour, CUDA 12.6, a
   cu126 torch wheel, `EXLLAMA_NOCOMPILE` build with comp_units excluded. This
   decides whether there is a product here at all, and costs no kernel work. Do
   this before anything else.
2. **If pursuing the fused kernel, do Turing (sm_75) first.** It needs *zero*
   toolchain work — nvcc 13.3, current torch and current vLLM all support it — so
   it isolates the kernel problem from the fork problem. Two PTX features and a
   shared-memory sizing fix, all of which are on the path to sm_70. Turing is a
   stepping stone, not a destination: T4 is 320 GB/s and 2080 Ti is 11 GiB, so
   neither is interesting for capacity.
3. **Only then sm_70**, and only if step 1 showed the capacity case is worth it,
   because that is where the three-project version fork starts.

## Correction to the premise

There is no sm_67. The GTX 1650 is **sm_75** (Turing TU117) and the GTX 1050 is
**sm_61** (Pascal GP107). That reframes the anecdote in two ways. The 1650's
partial success was on an architecture vLLM supports and current nvcc targets, so
it says nothing about pre-Ampere. And the 1050 is on an architecture vLLM's CMake
has never listed, so whatever ran there ran through torch-native paths only.

One genuinely odd datapoint survives: the run was **TP=2 across the two cards**,
i.e. heterogeneous tensor parallelism spanning sm_75 and sm_61 with only one of
the two in `CUDA_SUPPORTED_ARCHS`. Not useful — the pair has no capacity story
and the slower card sets the pace — but it is evidence that vLLM's TP path
tolerates a mixed-architecture group further than the build config suggests.
Recorded as an observation, not a direction.
