# Mixture of experts

*Originally "Phase 3 — mixture of experts". `exl3_mgemm` behind vLLM's `FusedMoE`,
the unrecorded Laguna scale factor, and the sm_90+ barrier hang.*

Goal from the feasibility report: **MoE via `exl3_moe`/`exl3_mgemm` adapted to
vLLM's `FusedMoE` interface.**

**Status: working on all three MoE checkpoints.** `gemma-4-26B-A4B-it-exl3`
(9.46 GiB), `Qwen3.5-35B-A3B-exl3` (10.63 GiB, needs the `patches/` change to
load) and `Laguna-XS-2.1-exl3` (256 experts, 2bpw, 8.54 GiB) all answer
correctly, lead with a real confident first token, and stay coherent when
sampled at temperature. Laguna took two fixes rather than one — a scale factor
the checkpoint does not record, and then an fp16 overflow that the first fix
introduced.

That verdict is for TP=1. Under tensor parallelism Laguna is a different story:
`exl3_mgemm` hits a profiled 100-1000x per-launch slowdown at TP=4 on this
checkpoint and not on Qwen3.5, which has an identical expert/layer/shard-width
profile. See [tensor-parallel.md](tensor-parallel.md) "Open problem".

Getting here needed a scale factor that lives in exllamav3's architecture
definition rather than in the checkpoint, care about where that factor is
applied, and one patch to exllamav3 itself — see "The MoE hang" below, which
also cost two wrong theories worth recording.

## `exl3_mgemm` is usable here, unlike merged QKV

Phase 1 established that a merged QKV *cannot* use exllamav3's pointer-table
kernel: EXL3 assigns bit widths per tensor and real checkpoints mix them inside
one layer (q=4, k=5, v=5), while `MultiLinear` asserts equal K.

Experts do not have that problem. Scanning every expert tensor in Laguna-XS
(121,212 tensors) and gemma-4-26B (47,652) found **a single bit width
throughout**, with no mixed-K group anywhere. `process_weights_after_loading`
re-checks per layer rather than trusting it, because a mixed-K checkpoint would
otherwise decode with the wrong K and produce plausible-looking nonsense.

The kernel is built for exactly this shape of problem: one input broadcast to N
experts, int64 pointer tables, expert-range filtering, and a **fused weighted
reduction** that scales each expert's result by its routing weight and sums into
`C[0]`. vLLM's `apply(layer, x, topk_weights, topk_ids, ...)` maps onto it
almost directly.

## Design

**Parameters are collectors, not one per tensor.** Laguna has 256 experts x 3
projections x 39 layers; a parameter per stored tensor would mean ~120,000
`nn.Parameter` objects. Each layer instead registers eight collectors (`w13_*`
and `w2_*` per EXL3 sub-tensor) that accumulate per-expert tensors as vLLM's
loader delivers them. The `w13_`/`w2_` naming is not ours to choose: vLLM's
expert mapping rewrites `experts.{id}.gate_proj.` to `experts.w13_` and keeps
whatever suffix follows.

**Pointer tables address per-expert tensors directly**, rather than a stacked
`[num_experts, ...]` copy. The kernel dereferences each address independently,
so stacking buys nothing and costs a second full copy of every expert weight
while both exist — on a 256-expert model that is enough to exhaust a 16 GB card
mid-load. exllamav3's own `MultiLinear` builds its tables the same way.

## Four bugs this needed

1. **`is_quantized` could not see MoE layers at all.** The module is named
   `...mlp.experts` while the checkpoint only has `...experts.0.gate_proj`, so an
   exact lookup missed and vLLM silently fell back to **dense fp16 experts** —
   an immediate OOM whose only clue was `Using TRITON Unquantized MoE backend`.
   Fixed by checking quantized *ancestors*, precomputed once.
2. **vLLM reads any 3D checkpoint tensor as packed multi-expert weights**
   (`is_fused = loaded_weight.dim() == 3`), but an EXL3 trellis is natively 3D.
   There is no hook — the rank test is the whole decision — so `plugin.py` lifts
   trellis tensors to 4D on the way in and the loader drops the axis again. This
   is the plugin's first monkeypatch, and it is a *format* collision rather than
   the container-format problem GGUF needs patches for.
3. **`exl3_mgemm` refuses expert-range filtering together with the multi-token
   weighted reduction.** `min_index` must be `-1` without expert parallelism;
   passing `0` looks harmless and raises at the first forward.
4. **Gemma 4 splices `.moe.` into expert paths in its own `_weight_iterator`**
   (`.experts.{id}.` becomes `.moe.experts.{id}.`), which `apply_vllm_mapper`
   never sees because it is not part of `hf_to_vllm_mapper`. So the module sits
   at `...layers.N.moe.experts` while `tensor_storage` says
   `...layers.N.experts.0.gate_proj`. `is_quantized` now also tries the prefix
   with any one interior component removed, accepting only a match against a
   known quantized ancestor.

## The activation is not always silu

gemma-4 uses `gelu_tanh`. The activation now comes from `layer.activation`
rather than being assumed, and an unknown one raises rather than silently
substituting.

Worth recording a near-miss: vLLM's `activation.py` describes gated activations
as *"gate x activation(up)"*, which reads backwards from what the kernels do.
`SiluAndMul`'s docstring is explicit — `silu(x[:d]) * x[d:]`, i.e. the
activation applies to the **gate** half — and the kernels agree. We had it
right, but the earlier per-layer MoE oracle could not have caught it either way,
because the reference it compared against made the same assumption. An oracle
only tests what it does not share with the thing under test.

## Checkpoints that fuse output shards (Qwen3.5)

Qwen3.5's linear-attention block makes vLLM merge `in_proj_qkv` and `in_proj_z`
into one `in_proj_qkvz`, and its weights mapper hands the checkpoint's
`in_proj_qkv` over as the shard *tuple* `(0, 1, 2)` — three logical shards
inside a single EXL3 tensor that was quantized as one matrix. vLLM then tries to
split it itself via `_load_fused_module_from_checkpoint`, which needs
`param.output_dim`, and ours raised `AttributeError`.

Adding the attribute would only have moved the failure somewhere quieter: an
EXL3 tensor has no single output dimension in consistent units. The trellis is
tile-granular on dim 1 (offsets would need dividing by 16), `svh` is
element-granular on dim 0, and `suh` must not be split at all — it is the shared
*input* scale.

vLLM does have the right escape hatch — a branch that hands the whole tensor to
`param.load_merged_column_weight` and lets it split itself — but it was gated on
`type(param) in (RowvLLMParameter, BasevLLMParameter)`, an exact type check that
excludes subclasses.

**`patches/vllm-fused-param-capability-check.patch`** replaces it with a declared
capability, `BasevLLMParameter.handles_fused_shards`. Two earlier attempts were
worse and are worth recording:

- `isinstance` is a **regression**: `ModelWeightParameter`, `PackedvLLMParameter`
  and the rest inherit from `BasevLLMParameter` and *do* define `output_dim`, so
  it would divert AWQ/GPTQ/compressed-tensors into the whole-tensor branch.
- `not hasattr(param, "output_dim")` is closer, but `output_dim` is a *proxy* for
  the property the branch cares about, and the logical connection is weak enough
  that a future parameter class could satisfy the test by accident. It also
  changes behaviour for `SharedWeightParameter`, which lacks `output_dim` yet
  cannot handle a fused load either.

The declared flag says exactly what the branch needs. Three declarations
reproduce current behaviour for **every** in-tree class:

| class | `handles_fused_shards` | why |
|---|---|---|
| `BasevLLMParameter` | `True` (default) | no output dimension to narrow along |
| `_ColumnvLLMParameter` | `False` | defines `output_dim`; the caller narrows |
| `SharedWeightParameter` | `False` | supports neither route; keeps today's behaviour rather than trading one exception for another |

Everything else inherits. `ModelWeightParameter` and friends resolve to `False`
through multiple inheritance because `_ColumnvLLMParameter` precedes
`RowvLLMParameter` in the MRO — checked, not assumed. `PerTensorScaleParameter`
inherits `True` but is claimed by an `isinstance` branch above ours at both call
sites, so it never arrives.

The branch also forwards `loaded_shard_id`, which is load-bearing: without it the
parameter has the whole tensor but no idea which of the layer's output shards it
covers — a tuple such as `(0, 1, 2)`, or `None` for all of them.

`EXL3Parameter` declares the flag explicitly rather than inheriting it, so the
intent survives a change to the base default.

`EXL3Parameter._load_fused` then splits the tensor across the shards it covers,
which is the same operation as a tensor-parallel column split and carries the
same 128-boundary constraint.

Under tensor parallelism both cuts apply — once to separate the shards, once to
take this rank's slice of each — and they compose, because both are column
splits. `format.fused_shard_bounds` does that arithmetic: vLLM's
`output_partition_sizes` are already per-rank, so shard `i` spans
`per_rank[i] * tp_size` stored columns and this rank takes the `tp_rank`-th
slice within it.

    stored:  [ shard0 rank0 | shard0 rank1 | shard1 rank0 | shard1 rank1 ]
    rank 0 takes  ^^^^^^^^^^^^                ^^^^^^^^^^^^

One condition secures every boundary: if each per-rank width is a multiple of
`HAD_BLOCK` then so is every span and offset derived from it, so a single check
covers both cuts.

`tests/test_format.py` checks the property that matters — every rank's slices tile
the unsharded shard exactly, with no gap or overlap — plus Hadamard alignment at
each boundary, across TP=1/2/4.

**Since verified on hardware.** This was unit-tested only when written: Qwen3.5 at
TP=2 was blocked by the old `NotImplementedError` while the 2x box was available,
and that box was released before the fix landed. It has since run on the 8x RTX
3090 box — `Qwen3.5-35B-A3B-exl3` @3.00bpw is token-identical to TP=1 at both TP=2
and TP=4; see [tensor-parallel.md](tensor-parallel.md) "What is validated" #7.
`tools/tp_preflight.py` still reports fused-shard checkpoints as provisional,
which now means "verified on one checkpoint, not a class" rather than "never run".

**Result: Qwen3.5-35B-A3B loads (10.63 GiB) and generates correctly.** It hung
during generation when first tried; that turned out to be the autotuner problem
described below, and it no longer reproduces either way.

## Laguna: a scale factor the checkpoint does not record

`Laguna-XS-2.1-exl3` @2.00bpw loaded (8.54 GiB) and ran, but emitted only
whitespace and dashes. The routed-expert output was **128× too small**.

exllamav3 divides Laguna's routed `up_proj` by a constant `interm_div = 128.0`
and multiplies the routing weights by the same constant to compensate, purely
to keep the fp16 intermediate in range. Both halves are needed; we had neither.

The trap is that the two halves live in different places. The **divisor** is
baked into the stored weights — `Linear.load_exl3` ignores its own
`weight_scale`, which only ever applies on the fp16 fallback path, so a
converted checkpoint carries the already-scaled weights. The **compensation** is
a literal in exllamav3's architecture definition
(`architecture/laguna.py`), not in `config.json` and not in
`quantization_config.json`. A consumer reading only the checkpoint sees
`moe_routed_scaling_factor: 2.5`, which is correct for the *original* weights
and wrong by exactly 128 for these.

Measured, not assumed — the scale lands wholly in the up projection's input
scale `suh`, leaving `svh` and the trellis untouched:

| | `mean\|suh_gate\| / mean\|suh_up\|` | dequantized `rms(W_gate)/rms(W_up)` |
|---|---|---|
| Laguna routed experts | 119.7 – 129.7 | 128.4 |
| Laguna **shared** expert (`interm_div=1.0`) | 1.000 | 1.02 |
| gemma-4-26B routed experts | 0.94 – 1.00 | — |

The shared expert is the control that makes this conclusive: it sits in the same
layer, was quantized by the same converter, and shows no such offset.

`format.infer_interm_divisor` therefore recovers the constant from the weights
and snaps it to the nearest power of two, and `EXL3MoEMethod` folds it into the
routing weights — the same place exllamav3 restores the magnitude, after the
fp16 down projection. It **raises rather than guessing** when the measured ratio
is neither ~1 nor near a power of two, because the wrong constant here is a
silent factor-of-N error in every routed expert rather than a crash.

Result: `'The capital of France is **Paris**.'`, stopping on EOS — but see
"Laguna is still broken" below. That answer is real, and it is also hiding a
garbage first token.

Four things were ruled out before this was found, and are worth keeping since
they constrain any future MoE bug: it was **not the MoE layer** (a live
`RoutedExperts` recomputed from dequantized weights gave `max_abs_err = 0.0000`
— the layer is exact *given its inputs*, which is exactly why a scale factor
applied outside it survived the check), **not configuration**, **not prompting**,
and **not shared experts**.

## The MoE hang: an sm_90+ barrier in exllamav3

**Fixed**, carried directly in [our exllamav3
fork](https://github.com/yeasah/exllamav3) (formerly an out-of-tree
`patches/exllamav3-sm90-barrier.patch`, folded into history once the fork
existed to hold it). MoE models hung during generation — GPU pinned at 100%,
no forward progress — on Laguna-XS and Qwen3.5.

`exl3_mgemm_kernel` synchronizes its blocks twice per iteration, and which
barrier it uses depends on the architecture:

```c
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 890)
    group_barrier(blockIdx.z, gridDim.x, barrier_counters_sense);
#else
    grid.sync();
#endif
```

These are not the same thing. `grid.sync()` is cooperative groups' whole-grid
barrier, with per-launch state. `group_barrier` is a hand-rolled sense-reversing
barrier over one z-slice that synchronizes through the **device-global `locks`
buffer** — one allocation per device, shared by every launch, zeroed only once
at allocation. sm_90+ (Hopper, Blackwell) takes the second path; Ampere and Ada
take the first. Building with `grid.sync()` on sm_120 clears every hang.

The fix makes the hand-rolled barrier opt-in (`EXL3_SM90_BARRIER=1` at build
time) rather than automatic above sm_89. It also adds two things worth keeping:
`cuda_check` around `cudaLaunchCooperativeKernel` in the mgemm path, which
previously swallowed launch failures the autotuner path already checked for, and
an `EXL3_DIAG_GRID=1` dump of the launch geometry with the occupancy the kernel
actually achieves.

| cell | before | after |
|---|---|---|
| Laguna eager, long prompt | hang 3/3 | ok, 34.8 tok/s |
| Laguna **graphs** | hang | **ok, 172-175 tok/s** |
| Laguna eager, autotuner on | hang 3/3 | ok, 35.2 tok/s |
| Qwen3.5 eager, autotuner off | hang 3/3 | ok, 25.8 tok/s |
| Qwen3.5 **graphs** | hang (tuner off) | **ok, 123-125 tok/s** |
| gemma-4-26B eager | ok | ok, 39.8 tok/s |

### Two wrong theories, and what killed them

Recording these because both looked convincing and both were wrong.

**"The autotuner causes it."** Disabling it made Laguna stop hanging in 7/7 runs,
so it shipped as the default. It was a coincidence of one model at one context
length: Qwen3.5 turned out to hang with the autotuner *off* and run with it *on*,
the exact inverse, and Laguna's answer flipped on `max_model_len` alone with the
same code and prompt. The autotuner only ever changed which grid geometry got
launched, which changed which barrier state got touched.

**"The grid cannot be co-resident."** The grid is sized to fill the device
exactly — `blocks_per_sm = 1`, `grid = 2 x 35 = 70`, `resident_max = 70` — so a
barrier that requires full residency has zero headroom. Plausible, and false: the
`EXL3_DIAG_GRID` instrumentation showed the grid never exceeded what fits, and
forcing a margin (70 -> 66 blocks) changed nothing.

What did survive was the one observation both theories had to explain:
`CUDA_LAUNCH_BLOCKING=1` made the hang disappear. Serializing launches hides
races over shared mutable state, which is exactly what the device-global barrier
buffer is, and does nothing for an occupancy problem.

### Diagnostic notes

- **The faulthandler stack lies** under an async launch queue. It bottoms out at
  whatever trivial op fills the queue — `router_logits.float()` here — not at the
  stuck kernel.
- `force_num_sms` is **not** a grid override. `num_sms = tiles` overwrites it
  immediately; it only gates whether the autotuner runs.

## CUDA graphs are worth 1.9x to 4.9x

Llama-3.2-1B, identical work: 312 tok/s captured versus 163 eager. Laguna-XS is
starker still — 172 versus 35, a 4.9x gap — because MoE decode is launch-bound.
Until the barrier fix the graph path was unusable for MoE at all, so every MoE
throughput figure recorded before it came from the slow path.

An earlier note here claimed gemma-4-26B could not *start* under graphs. That
was an artifact of the test harness, not a limit of the model, the graphs or the
plugin. vLLM picks `max_num_batched_tokens` by usage context — 8192 for offline
`LLM()`, 2048 for `vllm serve`, on any GPU below 70 GiB — so an offline harness
profiles a batch **4x larger** and reserves correspondingly more activation
memory:

| `max_num_batched_tokens` | available KV | KV tokens |
|---|---|---|
| 8192 (offline default) | 0.77 GiB | — (fails at `max_model_len=4096`) |
| 2560 (serve-like) | 1.91 GiB | 21,022 |

At the serve-like setting gemma-4-26B runs under graphs at
`max_model_len=16384`, `gpu_memory_utilization=0.90`, `max_num_seqs=2`. Worth
remembering when comparing any offline measurement against a served one: they do
not profile the same batch. (2048 exactly is not usable here — gemma's
`max_tokens_per_mm_item` is 2496, and vLLM raises rather than chunking.)

## MoE prefill has no reconstruct fallback, and it shows

Surfaced by `bench/`'s first MoE throughput baseline rather than looked for.
`Qwen3.5-35B-A3B-exl3` @2.00bpw on the 5070 Ti, graphs on, the same workload
shape as [kernels.md](kernels.md) (decode 8x128, prefill 4x~2.2k):

| | decode | prefill | ratio |
|---|---|---|---|
| Llama-3.2-1B @3.0bpw, dense | 2752 tok/s | 34239 tok/s | 12.4x |
| Qwen3.5-35B-A3B @2.0bpw, MoE | 546 tok/s | 961 tok/s | **1.8x** |

Part of that gap is simply model size, but not a 7x difference in *shape*. The
dense path switches to cuBLAS above `RECONSTRUCT_THRESHOLD` rows
([ops.py](../vllm_exl3_plugin/ops.py) `_exl3_mm`), which
[kernels.md](kernels.md) measures as 3.4x on prefill — the cooperative GEMM loses
badly once the multiply is compute-bound. **`fused_moe.py` has no equivalent**,
so MoE prefill stays on `exl3_mgemm` in exactly the regime the dense path
abandons it.

Whether that is worth fixing is genuinely open, and the reason is memory rather
than effort. Reconstructing a dense fp16 weight is cheap for one linear and
transient; for MoE it would mean reconstructing the *active experts*, and during
prefill with many token positions the active set approaches all of them — which
for a 35B at 2bpw is the whole reason the checkpoint fits at all. A partial form
(reconstruct only the top-k experts of a chunk, bounded like
`_EMBED_BLOCK_CHUNK`) is the shape that might work.

Recorded here rather than in TODO because it is an observation with an open
feasibility question, not yet a task. The numbers are now gated
(`bench/run.py perf-check --tier full`), so if anything changes them it will say
so.

## Qwen3.5 now runs

`Qwen3.5-35B-A3B-exl3` loads (10.63 GiB) and answers correctly, 4 runs out of 4
in eager mode with the autotuner either on or off, leading with a real first
token. It needs the `patches/` change to load at all, and its routed experts
carry no `interm_div`. It also runs under CUDA graphs at 123-125 tok/s — the hang
that used to make graphs unusable here was the sm_90+ barrier, fixed above.

## The second Laguna bug: fp16 overflow in the fused reduction

*(Fixed. Kept because the symptom was so thoroughly disguised.)*

After the divisor fix, `Laguna-XS-2.1` still produced an **entirely NaN prefill
hidden state** — `hidden (1, 2048) std=nan nonfinite=2048` at `compute_logits` —
which the quantized `lm_head` turned into all-zero logits. Every decode step after it is
clean (`std = 1.59, 1.63, 1.76`). So exactly one token is wrong, the first:

- **at temperature 0**, `argmax` of a zero tensor is token 0, which in this
  tokenizer is `〈|UNK|〉`. vLLM skips special tokens when detokenizing, so the
  text reads correctly and the damage is invisible.
- **at temperature > 0**, that step samples uniformly from all 100,352 tokens.
  A random token is injected at position 1 and the generation derails, which is
  what "incoherent above temperature 0" turned out to mean.

The tell is in the token ids, not the text: `[0, 785, 9626, ...]`. That leading
`0` was present in the very first successful-looking Laguna run and was read
past — **check `token_ids`, not just decoded text, before calling a model
correct.** Reported logprobs are the other cheap tell: a first-step top-1 of
exactly `-ln(vocab_size)` means a uniform distribution, i.e. constant logits.

Scope, checked across every EXL3 checkpoint on hand:

| model | kind | first token | step-1 top-1 logprob | |
|---|---|---|---|---|
| Llama-3.2-1B | dense | `The` | -0.0 | clean |
| MiniCPM5-1B | dense | `<think>` | -0.0 | clean |
| gemma-4-12B | dense | `The` | -0.0001 | clean |
| gemma-4-26B-A4B | MoE | `The` | -0.0 | clean |
| Qwen3.5-35B-A3B | MoE | `Thinking` | -0.0005 | clean |
| **Laguna-XS-2.1** | MoE | `〈\|UNK\|〉` (id 0) | **-11.5164 = -ln(100352)** | **broken** |

So this is Laguna-specific rather than an MoE-wide or plugin-wide fault, and the
other models' verdicts stand.

Ruled out: it is not vLLM's async scheduling (identical token ids and logprobs
with it on and off), and it is not the logprobs machinery — the sampler faithfully
reports a distribution that really is uniform, because the logits really are zero.

**Leading hypothesis, not yet established:** fp16 range at our kernel boundary.
`exl3_mm` casts activations to fp16, while exllamav3 runs Laguna with
`out_dtype = torch.float` on both attention and MLP and carries an fp32 residual
— a choice it makes for this architecture specifically, alongside the
`interm_div = 128.0` above, which it justifies with "routed-expert activations
overflow fp16". A bf16 residual above 65504 becomes `inf` on that cast and then
NaN. Prefill-only fits: more token positions, more chances at the tail. If that
is right, the divisor discovery and this are two symptoms of the same numerical
pressure rather than unrelated finds.

### What it actually was

A layer-by-layer bisect over the prefill found the residual climbing steadily —
15, 24, 80, 119, ... 804 — and then **layer 38's routed output at 155648 with 16
elements already `inf`**, against an fp16 ceiling of 65504. One `inf` in the
residual takes the whole hidden state to NaN by layer 39.

The amplification was ours. The divisor fix folded `interm_div = 128` into the
routing weights, which is where exllamav3 puts it — but exllamav3 also gives the
routed down projection `out_dtype = torch.float`, so its fused reduction
accumulates in fp32. Ours allocated `out` as fp16 (`c_fp32 = C.dtype() ==
at::kFloat` — the kernel writes whatever dtype it is handed), so the factor of
128 landed inside an fp16 accumulator.

The fix is to apply the divisor *outside* the kernel, which is algebraically
identical — `sum_j (d*w_j) y_j == d * sum_j w_j y_j` — but keeps the reduction
128x smaller and lands the scale in the model's own bf16 dtype, where it is
exact because the divisor is a power of two. Giving the kernel an fp32 `C` works
too and is what exllamav3 does, but costs a second full-size scratch buffer at
max batch.

The clipped value was not close to right: layer 38's true output is **1081344**,
seven times the 155648 that fp16 saturated to.

### Result

First token `The` at logprob -0.0001, `'The capital of France is Paris.'` greedy,
and coherent at temperature — at T=1.0, *"Known as \"the City of Light\" (La
Ville Lumiere), Paris is renowned for its cultural heritage..."*. The
temperature-dependent incoherence is gone, because it was only ever the first
token being sampled from a uniform distribution.

### Why this took two passes

The first fix was correct and incomplete, and the symptom it left behind was
disguised twice over: detokenization hides special tokens, and greedy decoding
returns *something* from a degenerate tensor. Sampling above temperature 0 is
what exposes it. **Check `token_ids`, not decoded text** — the leading `0` was
printed in the very first successful-looking run and read past.
