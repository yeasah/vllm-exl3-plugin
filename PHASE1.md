# Phase 1 — the fused kernels

Goal from the feasibility report: **swap in the real fused GEMM/GEMV kernels
behind a `direct_register_custom_op`, and validate under vLLM's actual serving
loop (continuous batching, CUDA graph capture, prefix caching).**

**Status: done and verified**, on `Llama-3.2-1B-Instruct-exl3` (3.0bpw and
3.5bpw) and `MiniCPM5-1B-exl3` (3.00bpw, mcg codebook, quantized head). 34 tests
pass. One model — `gemma-4-12B-it-exl3` — loads and runs but produces garbage;
see "The gemma-4 failure" below, which is unresolved.

## What changed

`process_weights_after_loading` now keeps the trellis resident instead of
decoding it, and `apply` calls `exl3_mm`, a custom op in our own
`torch.ops.vllm_exl3` namespace wrapping `ext.exl3_gemm`.

The kernel's contract (from `exl3_gemm.cu`):

    A     (m, k) fp16, row-major contiguous
    B     (k//16, n//16, 16*K) int16 trellis
    C     (m, n) fp16 or fp32, need not be zeroed
    suh   optional (k,) fp16 input scales/flips
    A_had scratch, same size and dtype as A, required whenever suh is given
    svh   optional (n,) fp16 output scales/flips
    requires k % 16 == 0 and n % 128 == 0

Both Hadamard transforms happen *inside* the kernel; the caller only supplies
scratch for the transformed activations. exllamav3 aliases that scratch onto A
for its cached batch-1 path, which vLLM cannot do — the activation tensor is a
live residual stream — so we always allocate separately.

Phase 0's dequantize-at-load path is retained behind `VLLM_EXL3_DEQUANTIZE=1`.
It is a transcription of exllamav3's own dequantization, which makes it the
reference the fused path is tested against.

## Results

`Llama-3.2-1B-Instruct-exl3` @ 3.0bpw, RTX 5070 Ti, fp16, prefix caching off.
Decode is 8 concurrent sequences x 128 tokens; prefill is 4 x ~2.2k tokens.

| | Phase 0 (dense fp16) | fused only | **fused + threshold** |
|---|---|---|---|
| weights | 2.35 GiB | 0.86 GiB | **0.86 GiB** |
| decode | 2247 tok/s | 2749 tok/s | **2754 tok/s** |
| prefill | 38834 tok/s | 10009 tok/s | **33938 tok/s** |

Two things worth reading off that table.

**Decode gets 22% *faster*, not slower.** Decode is memory-bandwidth bound, and
the quantized weights are 2.7x smaller. The kernel is not a tax paid for memory
savings; at decode batch sizes it is simply the better choice.

**Prefill needs the reconstruct fallback.** The cooperative GEMM loses badly to
cuBLAS once the multiply is compute-bound — 3.9x, which is the difference
between usable and not. exllamav3 solves this with
`AUTO_RECONSTRUCT_THRESHOLD = 144`: above that many rows it decodes the trellis
to a dense fp16 matrix and calls hgemm. We now do the same
(`VLLM_EXL3_RECONSTRUCT_THRESHOLD`, default 144, 0 disables), which recovers
prefill to within 13% of the dense-weight ceiling. The dense matrix is transient
— one layer's worth, freed on return — so it costs scratch, not the memory
saving.

MiniCPM5-1B, which has a quantized head, goes from **2.12 GiB to 0.79 GiB** and
emits byte-identical greedy output to the Phase 0 dequantized run.

## CUDA graphs and the autotuner

Both were flagged as open hazards in the feasibility report. Both work, at TP=1.

vLLM captures 51 PIECEWISE and 35 FULL graph shapes with `VLLM_COMPILE` and
inductor enabled, and produces output identical to eager. That is not obvious:
`exl3_gemm` launches via `cudaLaunchCooperativeKernel`, and on first use for a
given shape it *benchmarks* candidate tile shapes — which would be fatal inside
a capture. It works because vLLM runs `cudagraph_num_of_warmups` dummy passes at
each shape before capturing it, priming the autotune cache.

The autotune cache is on disk at `~/.cache/exllamav3/autotune/`, overridable
with `EXLLAMAV3_TUNE_CACHE`. Multi-process TP workers racing on it remains an
open Phase 2 question; the env var gives a per-worker override if needed.

Adding the reconstruct threshold cut graph capture from 35s to 2s, because the
large shapes no longer invoke the autotuner at all.

## bfloat16 activations

`get_supported_act_dtypes` now returns fp16 *and* bf16. The kernels are still
fp16 — `exl3_gemm` hard-checks A for `kHalf` — but `exl3_mm` casts at the kernel
boundary, so the residual stream keeps whatever dtype vLLM chose. The cast is
safe because EXL3 regularizes activations into a narrow range: measured max |x|
was 3.7e3 against fp16's 6.5e4 ceiling.

This removes the forced `--dtype float16`, which matters because nearly every
EXL3 repo inherits `bfloat16` from its base model. Verified to give identical
output on MiniCPM5-1B.

## Two bugs the 12B model exposed

Both were latent and would have hit any multi-branch or multimodal repo.

**1. We were reading `quantization_config.json` from the wrong revision.**
vLLM does not pass `revision` to `maybe_update_config` (there is a TODO about it
on the base class), so we defaulted to `main`. EXL3 repos publish one branch per
bit rate and `main` frequently has no `quantization_config.json` at all — for
gemma-4-12B it 404s. We then fell back to "assume every linear is quantized" and
claimed the unquantized BF16 vision tower. Fixed by resolving the revision from
`hf_config._commit_hash`, which transformers sets to the commit it actually
loaded the config from. The fallback now also logs a warning, since it is a
guess rather than a fact.

**2. `tensor_storage` keys needed vLLM's naming, not the checkpoint's.**
`get_quant_method` receives vLLM module prefixes, and multimodal models
restructure heavily: gemma-4 moves `model.language_model.layers.N` to
`language_model.model.layers.N`. Without translation every language-model layer
looked unquantized, and vLLM allocated dense fp16 weights for a 12B model — an
out-of-memory error that points nowhere near the cause. vLLM provides
`apply_vllm_mapper` for exactly this; we now implement it.

## The gemma-4 failure (unresolved)

`turboderp/gemma-4-12B-it-exl3@3.00bpw_mul1` loads cleanly in 6.32 GiB — Phase 0
could never have run it, since dequantized it needs ~24 GiB — and then generates
garbage (`'...........'`).

Ruled out, each by direct measurement rather than inspection:

- **Not the kernel.** `exl3_mm` matches the dequantized reference to ~1e-3 on
  gemma's own layer shapes and `mul1` tensors, at every batch size.
- **Not the weights.** exllamav3 itself, on the same checkpoint files, answers
  correctly ("The capital of France is **Paris**", "2 + 2 = 4").
- **Not fp16 overflow.** Instrumenting every `exl3_mm` call over a full forward
  pass found zero non-finite values in or out; max |activation| was 3694 against
  a 65504 ceiling.
- **Not the residual dtype.** bfloat16 produces the same garbage. (This was the
  leading hypothesis: vLLM refuses fp16 for gemma2/gemma3 as "numerically
  unstable", and exllamav3 carries fp32 residuals through its own gemma4
  architecture. It is still true, just not the cause here.)
- **Not layer classification.** The mapper is lossless — 667 keys in, 667 out,
  all 329 exl3 modules preserved — and the shape checks in
  `process_weights_after_loading` pass for every layer.
- **Not the unusual layer geometry.** Every sixth layer is a "k_eq_v"
  full-attention layer with head_dim 512 instead of 256, q out 8192, k out 512,
  and *no* `v_proj` in the checkpoint. vLLM handles this by duplicating
  `k_proj` into the V slot in `_weight_iterator`, and because that rule matches
  on `"self_attn.k_proj" in name` rather than on a `.weight` suffix, it
  duplicates our `trellis`/`suh`/`svh` correctly too.

So the weights are right, the kernel is right, the classification is right, and
the dtype is not the issue. The next step is a layer-by-layer comparison of
hidden states against exllamav3 running the same checkpoint, which is the only
remaining way to localize it. Until then this should be treated as one model
that does not work, not as a general Phase 1 limitation — the models that do
work are verified thoroughly.

## Development note

vLLM's compile cache is keyed on its own config and version, and cannot see
out-of-tree plugin code. Any edit that changes what `apply()` traces to will
silently reuse a stale compiled graph and fail with a bare `KeyError` on a
parameter name deep inside an AOT-compiled artifact. **Set
`VLLM_DISABLE_COMPILE_CACHE=1` while working on this plugin.** Toggling
`VLLM_EXL3_DEQUANTIZE` forces it automatically, since there the mismatch is
guaranteed.
