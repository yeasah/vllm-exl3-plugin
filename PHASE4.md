# Phase 4: quantized embeddings (TODO #3)

Goal: stop paying fp16 for the token embedding. Every EXL3 checkpoint stores it
unquantized, and at the sizes this project targets that is a quarter to a half of the
whole file.

This note records what was established before any of it was built, so the decisions are
auditable and the work is resumable.

## The prize, measured

Embed/head share of real EXL3 checkpoints (bytes off the shard headers, `tools/`-free --
just the census in the commit that added this file):

| checkpoint | total | embed | head | embed+head | tied |
|---|---|---|---|---|---|
| gemma-4-12B-it | 6.49 GiB | 1.88 (28.9%) | 0.70 (10.8%) | **39.7%** | yes |
| Qwen3.5-9B @4.00bpw | 6.69 GiB | 1.89 (28.3%) | 0.71 (10.6%) | 38.9% | no |
| Qwen3.6-27B | 12.87 GiB | 2.37 (18.4%) | 0.89 (6.9%) | 25.3% | no |
| gemma-4-26B-A4B-it (MoE) | 10.92 GiB | 1.38 (12.6%) | 0.52 (4.7%) | 17.3% | yes |
| Qwen3.5-35B-A3B (MoE) | 10.16 GiB | 0.95 (9.3%) | 0.36 (3.5%) | 12.8% | no |
| Laguna-XS-2.1 | 12.28 GiB | 0.38 (3.1%) | 0.14 (1.2%) | 4.3% | no |

The win is concentrated in **dense mid-size models with large vocabularies** -- precisely
the appliance target. MoE checkpoints dilute it (expert weights dominate) and Laguna is an
outlier with a small vocabulary for its size.

## Feasibility: can a row be read without dequantizing the tensor?

This is the gate on the whole idea. If an embedding lookup needs the full dense matrix,
quantized embeddings can never save VRAM at inference time, only on disk.

It doesn't. Working backwards through `ops.dense_weight`, for output row `t`:

```
w = reconstruct(trellis)   # (k=hidden, n=vocab) inner matrix
w = had_left(w)            # block-diagonal along k -- mixes rows, but we want all k anyway
w = w * suh[:, None]       # row scale -- no mixing across columns
w = had_right(w)           # block-diagonal along n, in blocks of 128  <-- the constraint
w = w * svh[None, :]       # column scale -- scalar per output row
W_eff = w.t()              # (vocab, hidden)
```

`had_right` is the only step that mixes output rows, and it is block-diagonal in blocks of
`HAD_BLOCK` (128). So row `t` depends on exactly the 128-row block containing it, and
nothing else. Concretely, for `Qwen3-0.6B-exl3` @4.0bpw (k=1024, n=151936, 6-bit head):

- per-row trellis slab: **96 KiB of 111.3 MiB, 0.084% of the tensor**
- distinct 128-blocks in the vocabulary: 1187

Two further simplifications fall out, both verified:

1. A single column of the Hadamard is a ±1 vector, so `had_right` collapses from a 128×128
   matmul to a matvec.
2. `had_left` (dim 0) commutes with a right-multiply (dim 1), so the transform can be
   applied to the *vector* after the matvec rather than to the (k, 128) *matrix* before it.

Validated against `ops.dense_weight` (the project's existing correctness oracle, itself a
transcription of exllamav3's own dequantization): **bit-identical** (rel err 0.00e+00) for
batched lookups of real trace tokens and of 2048 random ids; 9.4e-06 for the single-token
matvec form, which is fp16 rounding, not a different answer.

## Cost

Batched prototype (decode only the 128-blocks a batch actually touches), naive PyTorch, no
kernel work, on Qwen3-0.6B-exl3 @4.0bpw:

| shape | fp16 gather | quantized | overhead |
|---|---|---|---|
| prefill 2048 | 0.005 ms | 9.688 ms | 9.683 ms |
| prefill 512 | 0.005 ms | 4.150 ms | 4.146 ms |
| decode 32 | 0.005 ms | 0.266 ms | 0.261 ms |
| decode 1 | 0.005 ms | 0.173 ms | 0.169 ms |

Read this as an upper bound, for two reasons. It is a 0.6B model, so the fixed
embedding cost is amortized over the least layer compute it ever will be -- at 12-27B the
same absolute cost sits against ~50x more work per token. And the decode path, which is
the latency-sensitive one, is already only 0.17 ms.

The inherent inefficiency is the 128-row block granularity: a batch touching 971 distinct
blocks decodes 124k rows to use 2048. That is a property of the Hadamard block size, not
of the implementation -- it cannot be optimized away, only amortized (dedupe within a
batch, which the prototype already does, and potentially a small decoded-block cache for
frequent tokens). A fused kernel would remove the PyTorch overhead but not the 128x
read amplification.

## Sequencing: tied models first

**Phase A -- tied models, no new checkpoint format.** A tied model's EXL3 checkpoint
already ships a quantized `lm_head` (this project's quantizer writes one regardless of
tying, TODO.md #2). So the embedding can be served from that existing tensor, and the fp16
`embed_tokens` simply never loaded. No repair tool, no quantizer work, works on published
checkpoints as they are. Applies to gemma-4-12B (**-1.88 GiB, 29% of the checkpoint**),
gemma-4-26B-A4B (-12.6%), Qwen3-0.6B (-52%), Llama-3.2-1B (-54%).

**Phase B -- untied models.** Qwen3.6-27B, Qwen3.5-9B, Laguna, MiniCPM5 have no quantized
embedding anywhere, so one has to be produced: the repair tool (TODO.md #2) or the pipeline
(#4b). Phase A builds and de-risks the entire lookup path Phase B then reuses; only the
source of the tensor differs.

## vLLM integration: the hooks, all sanctioned

The one real obstacle is that **both tied-model patterns skip `lm_head.*` from the
checkpoint entirely**, so the quantized head never reaches the model:

- Qwen3-style: `self.lm_head = self.model.embed_tokens` (one module), loader gets
  `skip_prefixes=["lm_head."]` (`models/qwen3.py:341`)
- gemma4-style: a real `ParallelLMHead` tied via `tie_weights()` (two modules sharing a
  weight), loader gets `skip_substrs=[..., "lm_head."]` (`models/gemma4.py:1751`)

This is solvable without monkeypatching anything, because `AutoWeightsLoader.load_weights`
applies the weights mapper **before** the skip filter (`models/utils.py:418` vs `:421`),
and the quantization config gets to contribute to that mapper (`:414`):

| need | hook |
|---|---|
| get `lm_head.*` past the tied skip | `EXL3Config.get_cache_scale_mapper()` -- rename `lm_head.` to the embedding's prefix. Declared `@staticmethod` on the base but *called on the instance*, so it can be overridden as a normal method and gate on tied-ness. |
| register trellis/suh/svh instead of `weight` | `get_quant_method()` returning an embedding method for `VocabParallelEmbedding` |
| embedding lookup | `quant_method.embedding(layer, ids)` (`vocab_parallel_embedding.py:501`) |
| logits from the same tensor | `quant_method.apply()`, plus `quant_method.tie_weights()` (`:80`) for the gemma4 two-module case |

Note the mapper's target prefix is model-dependent (`model.embed_tokens` vs
`model.language_model.embed_tokens` for multimodal wrappers); `EXL3Config.apply_vllm_mapper`
already deals with exactly this restructuring for `tensor_storage` and is the place to
resolve it.

## Phase A result

Implemented and measured on `turboderp/Qwen3-0.6B-exl3` (tied, quantized `lm_head` on
disk). qbench, openwebtext 8x2048, transformers bf16 reference, dense-vs-quantized
embedding as the only difference between each pair:

| | bpw embed | vram_gb | ppl | KLD | KLD median |
|---|---|---|---|---|---|
| 4.0bpw dense embed | 16.000 | 0.4960 | 30.6353 | 0.043633 | 0.027594 |
| 4.0bpw quant embed | 6.016 | **0.3152** | 30.6332 | 0.044664 | 0.028407 |
| 3.0bpw dense embed | 16.000 | 0.4448 | 35.5235 | 0.192459 | 0.120742 |
| 3.0bpw quant embed | 6.016 | **0.2639** | 35.5409 | 0.193530 | 0.121412 |

- **4.0bpw: -36.5% VRAM for +2.4% KLD** (absolute +0.00103)
- **3.0bpw: -40.7% VRAM for +0.56% KLD** (absolute +0.00107)

The absolute KLD cost is essentially *constant* across bit rates (+0.00103 vs +0.00107),
which is what it should be -- it is the same 6-bit embedding in both cases, contributing a
fixed error independent of how hard the layers are quantized. So as a fraction it shrinks
the more aggressive the quantization, i.e. the trade improves exactly in the regime the
appliance cares about. ppl is unchanged to within noise (30.6353 -> 30.6332, slightly
*lower* with the quantized embedding at 4.0bpw).

Throughput cost, same checkpoint, eager, batch as noted:

| | dense | quant | |
|---|---|---|---|
| prefill 2048x1 | 149,597 tok/s | 156,304 tok/s | +4.5% (noise; less memory traffic) |
| decode 1x256 | 99.1 tok/s | 95.7 tok/s | -3.4% |
| decode 16x128 | 1562.7 tok/s | 1521.6 tok/s | -2.6% |

~3% decode cost, on the model where the fixed per-step embedding work is amortized over
the least layer compute it ever will be, in naive PyTorch with no kernel work.

## Open question: quality at scale

Phase A makes the embedding inherit the head's bit width (`head_bits`, usually 6). That is
a quality question, not just plumbing, and gemma-4 is the most numerically delicate family
here (it already needs fp32 residuals, and is the reason for the flash-attention head-dim
work in TODO #1). Reason for optimism: low-bit GGUFs run gemma embeddings down at Q3_K.

qbench answers this directly now, and the `embed_quant` option already prototyped in
`Exl3Backend` (simulated embedding precision, independent of any real storage format) is
the right instrument for finding the knee before committing to a bit width.
