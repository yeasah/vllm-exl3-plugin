# The fused kernels

*Originally "Phase 1 — the fused kernels". Keeping weights quantized at runtime:
`exl3_mm`, the reconstruct threshold, CUDA graphs, bf16 activations.*

Goal from the feasibility report: **swap in the real fused GEMM/GEMV kernels
behind a `direct_register_custom_op`, and validate under vLLM's actual serving
loop (continuous batching, CUDA graph capture, prefix caching).**

**Status: done and verified**, on `Llama-3.2-1B-Instruct-exl3` (3.0bpw and
3.5bpw), `MiniCPM5-1B-exl3` (3.00bpw, mcg codebook, quantized head) and
`gemma-4-12B-it-exl3` (3.00bpw, mul1 codebook, multimodal) — a 12B model in
6.32 GiB on a 16 GB card. 34 tests passed at the time.

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

Phase 0's dequantize-at-load path is retained behind `EXL3_DEQUANTIZE=1`.
It is a transcription of exllamav3's own dequantization, which makes it the
reference the fused path is tested against.

## Results

`Llama-3.2-1B-Instruct-exl3` @ 3.0bpw, RTX 5070 Ti, fp16, prefix caching off.
Decode is 8 concurrent sequences x 128 tokens; prefill is 4 x ~2.2k tokens.

**These numbers are gated, not historical.** `bench/run.py perf-check` reruns this
exact shape against a committed baseline and fails on a >10% regression — see
[bench/README.md](../bench/README.md). Re-measured 2026-08-16 on the same card:
decode 2752 tok/s against the 2754 below, prefill 34239 against 33938. Throughput
on this box turns out to be far steadier than expected — ~1% spread within a
process, ~0.5% across fresh ones — which is what makes a gate practical at all.

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
(`EXL3_RECONSTRUCT_THRESHOLD`, default 144, 0 disables), which recovers
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
with `EXLLAMAV3_TUNE_CACHE`. Multi-process TP workers racing on it was left open
here; the env var gives a per-worker override if needed. **Since answered** — the
cache survives two concurrent writers and eight writing it from empty, producing
correct entries either way; see [tensor-parallel.md](tensor-parallel.md) "What is
validated" #3 and #6.

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

## The gemma-4 "failure" — resolved, and it was the test harness

`turboderp/gemma-4-12B-it-exl3@3.00bpw_mul1` loads in **6.32 GiB** — a 12B model
on a 16 GB card, which Phase 0 could never have run — and works correctly:

    USER:  What is the capital of France? Answer in one sentence.
    MODEL: The capital of France is Paris.
    USER:  What is 2 + 2?
    MODEL: 2 + 2 = 4

It produced garbage only because **every prompt in this project's manual testing
was a raw completion string**, and this checkpoint does not prepend a BOS token
to those. Gemma collapses without BOS.

Concretely: `tokenizer_config.json` sets no `add_bos_token`, and the tokenizer's
post-processor does not add one either, so
`hf("The capital of France is")["input_ids"]` returns `[818, 5279, 529, 7001,
563]` — no `2` — *even with* `add_special_tokens=True`. The `<bos>` lives in
`chat_template.jinja` instead. Anything going through the chat template is fine;
raw completions are not.

    prompt_token_ids=[818, 5279, ...]     -> '..-.........'
    prompt_token_ids=[2, 818, 5279, ...]  -> ' capital of France.\nthought\n...'

Llama-3.2-1B and MiniCPM5-1B hid this because their tokenizers do add BOS, so
the same harness worked on them.

### How it was localized

Worth recording, because four plausible hypotheses were wrong and only
measurement distinguished them.

1. **Every quantized linear was verified in situ.** Running the real model with
   forward hooks on all 20 quantized linears across layers 0/4/5/6/11 and
   recomputing each one's output from dequantized weights gave agreement to
   ~5e-4 everywhere — including the k_eq_v layers, whose `[8192, 512, 512]`
   shard layout was the leading suspect. This is the seam the tensor-level
   tests do not cover: shard ordering, trimming and concatenation inside
   `apply()`.
2. **Then the whole forward was compared against exllamav3.** Feeding both
   runtimes the *same token ids* and comparing the hidden state entering each of
   the 48 layers gave cosine similarity >= 0.99999 at every layer, and both
   produced identical top-5 logits (` Berlin` for "...the capital of Germany
   is"). At that point the model was demonstrably correct and the only remaining
   difference was how the prompt became token ids.

Hypotheses ruled out by measurement along the way: fp16 overflow (zero
non-finite values in a full forward; max |activation| 3694 against a 65504
ceiling), residual dtype (bfloat16 produced identical garbage), layer
classification (mapper lossless, 667 keys in and out), and the unusual k_eq_v
geometry.

**Nothing in the plugin, the kernels, or vLLM's Gemma4 implementation was at
fault.** The lesson for the test suite is that raw completion prompts are not a
safe default across model families; the packaging layer should drive models
through their chat templates.

## The `mul1` codebook: +4.3% decode, and free

exllamav3 publishes some checkpoints twice at the same bitrate, once with the `mcg`
codebook and once with `mul1` -- the kernels take the codebook as two booleans and compile
the multiplier constants in (`EXL3LinearMethod.__init__`). `mul1` is described as a
performance improvement; measured 2026-08-26 on `gemma-4-12B-it-exl3` at 3.00bpw, where
both variants are published, it is.

| | `mcg` | `mul1` | |
|---|---|---|---|
| decode tok/s | 583.6 | **608.6** | **+4.28%** |
| prefill tok/s | 3268.9 | 3264.9 | -0.13% |

Same weights, same bitrate. Workload is `bench/perf.py`'s (8 seqs x 128 decode tokens, 4 x
2200 prefill), run interleaved A,B,B,A across four separate loads so ordering and thermal
drift cannot produce the gap: `mul1` won in both its slots (610.1, 606.9) and `mcg` lost in
both (583.3, 586.2), against a within-revision spread of ~0.85%.

**Only decode moves, which is the mechanism working as expected.** The multiplier is
*per-weight* work inside the trellis decode. Decode is bound by exactly that, so it shows;
prefill amortizes the same decode over 2200 tokens of GEMM, so it disappears. A codebook
change that sped up both equally would have been evidence of a measurement artifact rather
than of a faster codebook.

**And it costs nothing in quality**: body-only KLD at 3.00bpw is 0.102867 (`mcg`) against
0.102701 (`mul1`), a difference of 0.000166 -- see docs/embeddings.md, where that pair was
measured to rule out a codebook confound in the layer curve. So prefer `mul1` where a
checkpoint offers both.

**gemma-4-12B is the transition point, so the dual publish is a one-off.** Every EXL3
checkpoint on hand published after it carries `mul1` (Qwen3.6-27B, Qwen3.8-27B,
Muse-Glimmer, Laguna) and every one before still carries `mcg` (Qwen3.5-9B, Qwen3.5-35B-A3B,
MiniCPM5-1B, gemma-4-26B-A4B@2.54bpw). Nothing else publishes both, which is why this
measurement was only possible here. The practical read: the older checkpoints are leaving
~4% of decode on the floor and only a re-quant upstream recovers it -- worth knowing for
gemma-4-26B-A4B in particular. (Older checkpoints again -- Llama-3.2-1B, Qwen3-0.6B -- carry
neither tensor, predating both. So does `Qwen3.8-27B SC_3.00bpw_H4`, which presumably means
the `sc_*` pipeline parameterizes the codebook differently; not chased.)

*One loose end.* Those two have near-identical KLD but perplexity **17.9276** (`mcg`) against
**18.4556** (`mul1`) -- equal divergence from the bf16 reference, differently aimed, and
`mul1`'s happens to land worse against the actual tokens. Ten rows of openwebtext, so this
is a lead rather than a finding, and it is the one thing that would complicate "free" if it
survived a larger sample. Worth resolving before the recommendation above is leaned on
hard.

## Development note

vLLM's compile cache is keyed on its own config and version, and cannot see
out-of-tree plugin code. Any edit that changes what `apply()` traces to will
silently reuse a stale compiled graph and fail with a bare `KeyError` on a
parameter name deep inside an AOT-compiled artifact. **Set
`VLLM_DISABLE_COMPILE_CACHE=1` while working on this plugin.** Toggling
`EXL3_DEQUANTIZE` forces it automatically, since there the mismatch is
guaranteed.
