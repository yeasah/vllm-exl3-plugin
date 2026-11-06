# Quantized embeddings

*Originally "Phase 4: quantized embeddings". Not the feasibility report's Phase 4,
which was packaging. Open work on tied models is TODO `tied-embeddings`.*

*"Phase A" and "Phase B" below are local to this note: A is serving a tied model's
embedding from its existing quantized head (shipped), B is producing an embedding
tensor for untied models (open).*

Goal: stop paying fp16 for the token embedding. Every EXL3 checkpoint stores it
unquantized, and at the sizes this project targets that is a quarter to a half of the
whole file.

This note records what was established before any of it was built, so the decisions are
auditable and the work is resumable.

## Why upstream leaves the embedding dense, in their own words

Worth stating before any of the measurements below, because it changes what the gap
*is*. On the Qwen3.8-27B model card, turboderp writes that the EXL3 quant "contains
2.4 GiB of (unquantized) embeddings, **not counted since they're not loaded into VRAM
during inference**" ([discussion 2](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/discussions/2),
read 2026-08-28).

That is true of their serving path and false of ours. exllamav3 keeps the embedding on
the host and gathers rows there; vLLM's `VocabParallelEmbedding` is a device tensor in
the model graph, so every byte lands in VRAM. So this is not an oversight upstream failed
to notice — it is a decision that is correct for their deployment and simply does not
transfer to a paged, batched, GPU-resident server.

Two things follow. The bpw figures on those model cards are consistent within their own
frame rather than misleading, and **the language elsewhere in this repo about upstream
"mistakes" or "underperformance" on this axis is harsher than the facts warrant**. What
is actually true is narrower and still worth the whole effort: an EXL3 checkpoint carries
a tensor that costs us gigabytes of VRAM and costs them nothing, and nobody upstream has
a reason to fix it.

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

### Against other formats: where the packaging costs more than the algorithm saves

The census above is internal. The reason it matters is competitive, and the comparison
was the original motivation for all of the work below (measured 2026-08-08 from real
safetensors bytes and from the Hub's tensor-metadata viewer -- no downloads).

On `Llama-3.2-1B-Instruct` (128,256 vocab, tied), EXL3 spends **0.673 GiB on embed+head
at every target bpw** -- identical at 2.00, 3.00 and 4.00, because neither tensor
participates in the bit-rate target -- against **0.201 GiB** for bartowski's Q6_K GGUF
and **0.168 GiB** for unsloth's Q5_K. A 3.3-4x gap, from two packaging choices rather
than the quantization algorithm: the embedding is not quantized, and a tie is not
exploited (a second, independently quantized `lm_head` is baked out *in addition to* the
untouched fp16 embedding, where GGUF stores the shared tensor once).

**Both choices are coherent for exllamav3's own deployment, and they are really one
choice.** `Embedding.caps["prefer_cpu"]` is `True` and every loader honors it
unconditionally, so upstream's fp16 embedding lives in system RAM and never occupies
VRAM; there is even a pinned-staging path to hide the per-token upload behind the decode
loop. turboderp [says as much](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/discussions/2)
and excludes the embedding from published checkpoint sizes on exactly that basis. Once
the embedding is fp16 on the host, a tied model *must* also carry a separate quantized
`lm_head` on the GPU, because the host copy cannot serve as one -- the duplication is a
consequence of the residency decision rather than an independent oversight. What makes
it our problem is that vLLM has no CPU-embedding path: the lookup sits inside the
graph-captured forward, and prefill gathers thousands of rows rather than one. We pay in
VRAM what upstream pays in host RAM. The download is paid by everyone either way, and
that is the axis the rest of this section measures.

The consequence is not academic. On real total file size, bartowski's IQ4_XS is smaller
than *every* EXL3 checkpoint tested and beats two of the three on KLD -- the opposite of
what an embed/head-excluded comparison shows.

It persists at target scale. Relative tax shrinks with model size but never disappears,
and tracks embedding-parameter count against package size more than scale per se:

| model | bpw low → high | EXL3 embed share | GGUF embed share |
|---|---|---|---|
| Qwen3.5-35B-A3B (509M embed params) | 2.13 → 4.09 | 9.04% → 5.23% | 1.43% (flat) |
| gemma-4-26B-A4B-it (738M embed params) | 2.10 → 6.10 | 14.40% → 6.28% | 2.86% (flat) |
| Laguna-XS-2.1 (205M embed params) | 2.00 → 6.10 | 4.43% → 1.57% | 0.59% (flat) |

Two readings. **GGUF's share stays flat across quant tiers** because it scales embedding
precision with the target level; EXL3's fixed fp16 means its tax is worst exactly at the
low-bpw end — worst for precisely the VRAM-constrained users who pick a low-bpw
checkpoint in the first place. And **the most "efficiently sized" model is the worst
case, not the best**: gemma-4-26B-A4B-it, the closest of the three to this project's
target shape, spends 14.4% of the entire package on the embedding at 2.10bpw, and 19.79%
on embed+head together once its duplicate head is counted (it is the only tied source of
the three, so GGUF skips the head entirely). In body-layer budget at matched download
size that is ~8.25 of 10.28 GiB for EXL3 against ~9.64 of 9.92 GiB for unsloth's
same-size UD-IQ2_XXS — **~1.4 GiB, 17% more actual weight budget**, for two tensors the
per-layer allocator never touches.

**This is not an EXL3 defect so much as an ecosystem one.** AWQ, GPTQ, AutoRound and
DASHQ checkpoints all ship full fp16/bf16 embeddings too; GGUF is the only widely used
format that gets this right, plausibly because llama.cpp grew up optimizing for consumer
hardware where every byte mattered, while the GPU-serving formats grew up where a few
hundred MB against a 70B+ model was noise. Fixing it puts this project ahead of the
entire GPU-quantization ecosystem outside GGUF, not merely at parity with one competitor.

*Amended 2026-08-24: "GGUF is the only widely used format" is now true of published
checkpoints but no longer of the tooling.* `llm-compressor` can quantize an embedding
(`QuantizationModifier(targets=["Embedding"])`, weight-only and data-free), and vLLM has
served those checkpoints since v0.27.0 — `CompressedTensorsEmbeddingWNA16Int`, a Triton
fused gather+dequant on the same `VocabParallelEmbedding` hook this plugin uses, landed
in PR #44340 within weeks of the work below. So the census above still describes every
checkpoint anyone has published, but the gap is now a question of what producers *do* by
default rather than of what the formats *can* express. What their scheme costs at matched
bytes is measured in ["What llm-compressor's embedding quantization
costs"](#what-llm-compressors-embedding-quantization-costs-and-where-its-menu-is-a-trap).

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
tying -- a pipeline defect, and the reason a repair tool is wanted). So the embedding can
be served from that existing tensor, and the fp16 `embed_tokens` simply never loaded. No
repair tool, no quantizer work, works on published checkpoints as they are.

The saving is **embed minus head**, not embed: vLLM skips `lm_head.*` entirely for a tied
model, so that tensor was *never resident* before, and serving the embedding from it makes
it newly resident. Getting this wrong overstates the win by the size of the head.

| checkpoint | embed (fp16) | head (quantized) | saving | of previously resident |
|---|---|---|---|---|
| gemma-4-12B-it | 1.88 GiB | 0.70 GiB | **1.18 GiB** | -20.4% |
| gemma-4-26B-A4B-it | 1.38 GiB | 0.52 GiB | 0.86 GiB | -8.3% |
| Qwen3-0.6B | 0.29 GiB | 0.11 GiB | 0.18 GiB | -36.5% |
| Llama-3.2-1B | 0.49 GiB | 0.18 GiB | 0.31 GiB | -42% |

Confirmed against gemma-4-12B in practice: KV cache headroom went from 7.32 GiB to
8.47 GiB, i.e. **+1.15 GiB**, against the 1.18 GiB predicted.

**Phase B -- untied models.** Qwen3.6-27B, Qwen3.5-9B, Laguna, MiniCPM5 have no quantized
embedding anywhere, so one has to be produced -- by a repair tool over published
checkpoints, or by the quantizer pipeline itself. Phase A builds and de-risks the entire
lookup path Phase B then reuses; only the source of the tensor differs.

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

## A tied head vLLM used to throw away

*2026-09-09, found by the v0.29.0 bump.* Phase A above measures
`turboderp/Qwen3-0.6B-exl3`, whose `config.json` declares `head_bits: 6`. On
`turboderp/Llama-3.2-1B-Instruct-exl3` the same path had **never run at all**, and
nothing said so.

Three signals tell the plugin a checkpoint stores a quantized head: the safetensors
index, a `quantization_config.json` storage map, and `head_bits`. That checkpoint has
none of them -- one `model.safetensors` with no index (the Hub 404s that path), no
sidecar, and a `quantization_config` block whose keys are only `quant_method`,
`version`, `bits` and `calibration` -- while carrying `lm_head.suh/svh/trellis` on
disk. So `_head_storage_exists()` was False, the tie was declined, and vLLM 0.28
discarded every `lm_head.*` weight for tied models (`skip_prefixes=["lm_head."]`).
The trellis went on the floor, the dense `embed_tokens.weight` loaded in its place,
and every logit was correct.

**The gate recorded the fallback as the reference.** `llama-3.2-1B-3.0bpw-tied` was
blessed at `weight_gib: 0.86`, which is 0.489 GiB of dense fp16 embedding plus ~0.37
GiB of 3bpw layers -- the entry whose stated purpose is that the embedding is *not*
loaded dense. A weight-bytes channel cannot catch a regression that was already
present when the baseline was taken.

vLLM 0.29 replaced the blanket skip with alias detection (`_get_tied_embedding_params`),
which skips only `lm_head.weight` and refuses anything else unclaimed, turning a silent
fallback into a load error. The fix reads the single file's header where there is no
index -- the same question, asked of the only file there is.

| | before | after |
|---|---|---|
| weight bytes | 0.86 GiB | **0.55 GiB** |
| KV headroom | 11.42 GiB | 11.73 GiB |
| greedy continuation | — | unchanged |
| max abs dlogprob | — | 2.911e-01 |
| KL max | — | 1.037e-01 |
| argmax disagreements | — | 1 of 88 |

**None of that deviation is the bump.** Held fixed at the dense embedding,
0.29 reproduces the 0.28 baseline at *exactly* 0.000e+00 on both prompts, weight bytes
included. The deviation is the embedding source, and it is larger than Phase A's
because the shared tensor is coarser: Qwen3-0.6B's head is 6-bit, while this checkpoint
declares no `head_bits` at all, so its head is quantized at the **body rate of 3.0bpw**.
The same mechanism, four bits shallower.

**Taken as-is rather than gated on depth.** A minimum-depth conditional was considered
and rejected: head depth is what the fp8 shared-tensor plan addresses directly (*fp8 as
the shared tensor*, above), and this entry is a baseline rather than something served,
so the value here is having the explanation rather than a threshold nobody measured.

**Declining a head is now an action, not an absence.** The same 0.29 change means an
unclaimed trellis raises, so `get_cache_scale_mapper` has to drop `lm_head.*` explicitly
wherever the config does not serve it. Without that, `EXL3_DENSE_EMBED=1` -- the switch
whose whole purpose is isolating the embedding from every other change -- was the one
setting under which a tied EXL3 checkpoint could not be loaded at all.

## Serving under torch.compile and CUDA graphs

Every measurement above was taken **eager**, and that turned out to be hiding something:
until 2026-08-16, `EXL3EmbeddingMethod` did not work under vLLM's *default* execution
mode at all. `bench/` caught it on its first run — `vllm serve` on a tied EXL3 checkpoint
failed to start, without anyone asking for anything unusual.

Two distinct problems, and the second is the one worth remembering.

**Tracing.** `embed_rows` deduplicates the 128-blocks a batch touches, which means
`torch.unique` (data-dependent output size), a Python loop bounded by
`blocks.numel()`, and a `nonzero` inside it. Dynamo can trace none of it, so the engine
died during startup compilation. Fixed by putting the gather behind an opaque custom op,
the same way `exl3_mm` already handles kernel-level branching.

**Capture.** Opacity was necessary and *not sufficient*, which is the interesting part.
A CUDA graph records a fixed kernel sequence against fixed buffers; here every gather
index descended from the capture-time token ids, so a replayed graph would have returned
capture-time rows. That is a silent-wrong-answer shape. It failed loudly instead —
`cudaErrorStreamCaptureInvalidated`, because reading `blocks.numel()` back to the host is
a synchronization and synchronizing mid-capture is illegal — but the loudness was luck,
not design.

The fix is a second path taken below `EXL3_EMBED_STATIC_MAX` (default 512, sized to cover
vLLM's capture sizes) that chunks over the *token count* rather than over deduplicated
blocks. Its loop bound comes from `token_ids.shape`, which is metadata, so the kernel
sequence depends on batch size alone. It decodes one block per token, giving up
deduplication.

**That trade is nearly free where it applies, and avoided where it is not.** Deduplication
only pays when tokens share a 128-block, which scattered decode-batch ids rarely do; it
pays enormously on a large prefill, where it saturates at the vocabulary's block count
(Qwen3-0.6B: 1187 blocks for 8192 tokens). So the deduplicating path still runs above the
threshold, and prefill is never graph-captured anyway.

Verified three ways: bit-exact against `dense_weight` including duplicate ids (the
existing equality tests, unchanged); a capture-then-replay test that captures on one set
of ids and replays on a *different* set, demanding the second set's rows; and end to end,
where the eager and CUDA-graph bench entries agree with their own baselines exactly.

One measurement worth keeping from that verification: eager and CUDA graphs differ by
**~0.157 nats** on Qwen3-0.6B with the embedding path removed entirely
(`EXL3_DENSE_EMBED=1`). The quantized path differs by *less* than that across the same two
modes, which is how the static path was cleared of introducing error — the cross-mode
floor is a property of vLLM, not of this code. It is also why the two modes get separate
baselines in `bench/`.

## gemma-4-12B: the tax, and where it lives

The Qwen3-0.6B result did not survive contact with a real target-class model unchanged.
On `gemma-4-12B-it-exl3`, dense-embedding (exllamav3 native) vs quantized-embedding
(vLLM, Phase A), openwebtext 10x2048:

| layer bpw | native KLD | quant-embed KLD | tax | vram (native -> quant) |
|---|---|---|---|---|
| 3.00 | 0.10293 | 0.12354 | +0.0206 | 6.39 -> 4.60 GiB |
| 3.50 | 0.07026 | 0.09135 | +0.0211 | 7.03 -> 5.23 GiB |
| 4.00 | 0.02696 | 0.04858 | +0.0216 | 7.66 -> 5.87 GiB |

A flat **+0.021 KLD** for a flat **-1.79 GiB**. Still a dominant trade -- quant-embed at
4.00bpw (5.87 GiB, 0.0486) beats native at 3.50bpw (7.03 GiB, 0.0703) on both axes -- but
an order of magnitude more than the ~0.001-0.002 seen on Qwen3-0.6B, and at 4.00bpw the
embedding contributes nearly as much divergence as every layer combined.

**The tax is one confidence bucket, not a general degradation.** At 4.00bpw, native ->
quant-embed by reference-confidence bucket:

| ref conf | native | quant embed | |
|---|---|---|---|
| [0.00,0.25) | 0.0417 | 0.0434 | +4% |
| [0.25,0.50) | 0.0366 | 0.0392 | +7% |
| [0.50,0.75) | 0.0299 | 0.0315 | +6% |
| [0.75,0.95) | 0.0161 | 0.0166 | +3% |
| **[0.95,1.00)** | **0.0108** | **0.1161** | **+979%** |

That bucket is 19.2% of tokens and accounts for 0.0202 of the 0.0216 tax -- essentially
all of it. Its *median* is unchanged (0.000391 -> 0.000416), so it is a heavy tail, not a
shift. And on the quantized-embedding side that bucket mean is nearly flat across layer
bpw (0.1435 / 0.1318 / 0.1161 for 3.0 / 3.5 / 4.0) while native falls away fast (0.0380 /
0.0280 / 0.0108): an error floor that spending bits on *layers* cannot buy down. It also
explains why ppl barely moves (17.898 -> 17.937) -- perplexity is not tail-sensitive.

### It is not a few broken tokens

The obvious hypothesis -- outlier embedding rows that 6 bits cannot hold -- is **false**.
Measuring the substitution error directly (6-bit dequantized `lm_head` row vs the bf16
`embed_tokens` row, per token, all 262144 of them):

- relative row error: mean 0.0209, median 0.0206, p90 0.0215, **p99 0.0280**, max 0.1417
- tokens with relative error > 0.5: **zero**
- row norm has no predictive power: every norm decile lands on 0.0207, including the top
  decile running to norm 7.13
- the only elevated blocks are in the reserved region, and the worst tokens decode to
  `<unused642>`, `<unused607>`, `<unused618>`, ... -- slots that never appear in text and
  therefore cannot contribute a single count to the KLD

So the substitution is a **uniform ~2.06% perturbation of every row**. There is nothing to
special-case, and no clamp or exception list can help; reducing the tax means more bits,
globally. (Incidentally this is a good result for EXL3's regularization: a 6-bit code over
a 262144x3840 matrix with p99 error 2.8% and no catastrophic rows is the Hadamard doing
exactly its job.)

The heavy tail therefore comes from KL's own nonlinearity rather than from bad rows: where
the reference is near-deterministic, KL ~ -log q(top), so a uniform small perturbation
produces enormous divergence wherever a confident prediction sat on a narrow logit margin,
and little anywhere else. Which is precisely the bucket pattern above.

### Consequence for Phase B

Phase A cannot act on this -- it is locked to whatever `head_bits` the checkpoint used
(6.004 here). Phase B chooses the embedding's bit width, so the knee is worth locating.

More interestingly, the uniformity is an argument *against* using EXL3 format for
embeddings at all. EXL3's advantage is handling ill-conditioned weights with outliers;
this matrix has none (norms clustered near 1.03, error uniform, nothing for the Hadamard
to rescue). A plain per-row integer scheme could plausibly match it at a fraction of the
complexity -- and would drop the 128-block read amplification entirely, since row
extraction becomes a trivial slice instead of a block decode.

### Measured: a naive per-row quantizer beats the trellis by ~89x at equal bits

Swept with `Exl3Backend`'s `embed_quant` (simulated embedding precision, storage-format
independent) on gemma-4-12B @4.00bpw. Everything else is held fixed -- real 4.00bpw
layers, real 6-bit trellis `lm_head` -- so only the input embedding varies:

| embedding | ppl | KLD | tax vs fp16 | vs the trellis |
|---|---|---|---|---|
| fp16 (baseline) | 17.8979 | 0.026963 | — | |
| 8-bit per-row | 17.9019 | 0.027012 | +0.000049 | 440x better |
| **6-bit per-row** | 17.9112 | 0.027206 | **+0.000243** | **89x better** |
| 5-bit per-row | 17.8893 | 0.027535 | +0.000572 | 38x better |
| **4-bit per-row** | 17.9090 | 0.028835 | **+0.001872** | **11.5x better** |
| 3-bit per-row | 26.6644 | 0.512568 | +0.485605 | cliff |
| 6-bit per-*tensor* | 2708219 | 12.750586 | +12.72 | catastrophic |
| **6-bit EXL3 trellis (real)** | 17.9370 | 0.048582 | **+0.021619** | 1x |

The trellis is not merely beatable here, it is *badly* beaten: a min/max per-row integer
quantizer at the **same 6 bits** costs 89x less divergence, and at **4 bits** -- a third
fewer -- still costs 11.5x less. Two supporting readings: the 3-bit cliff is sharp
(4-bit is fine, 3-bit destroys the model), and per-*tensor* granularity at 6 bits is
catastrophic, which confirms per-row scaling is doing the real work and that the harness
is genuinely perturbing what it claims to be.

The likely reason is a mismatch of objective rather than a defect. EXL3 quantized this
tensor as an **output projection**: the quantizer optimizes `x @ W.T` against typical
activations, where error in directions the activations rarely occupy is nearly free. Using
the same tensor as an **embedding** demands something different -- that each individual
*row* be an accurate vector on its own. Nothing in the format was asked to preserve that.

### Consequence: the Phase B menu, for gemma-4-12B

Embedding + head storage, and the KLD tax each option carries:

| option | embed+head | tax |
|---|---|---|
| exllamav3 native (fp16 embed + trellis head) | 2.579 GiB | +0 |
| **Phase A** (one trellis serving both) | **0.704 GiB** | +0.0216 |
| trellis head + per-row 4-bit embed | 1.172 GiB | +0.0019 |
| trellis head + per-row 6-bit embed | 1.407 GiB | +0.00024 |

(Sizes are 262144 x 3840 at the stated depth, head at the checkpoint's 6.004 bits:
fp16 embed 1.875 GiB, trellis head 0.704 GiB. An earlier revision of this table
carried the same four rows ~4.8% low against an inconsistent shape; these agree
with the census at the top and with "The full menu" below.)

Phase A remains the smallest configuration and keeps its place as the extreme-VRAM option.
But a separate per-row embedding buys back essentially the entire tax for 0.47-0.70 GiB --
and at this operating point that is better value than spending the same bytes on layer
bits, since the layer curve has to flatten hard (0.0270 is already only 0.025 above the
0.00176 noise floor, so no amount of layer precision can buy what fixing the embedding
buys).

So Phase B should **not** be "quantize the embedding with exllamav3's quantizer". It should
be a scalar integer scheme (per-row here; block-scaled by the later measurement below) at
4-6 bits: better quality per bit by an order of magnitude,
far simpler, no Hadamard, no 128-block read amplification, and a trivially cheap row
gather.

**Note what does not yet exist.** Every per-row number above comes from `fake_quantize`
simulating a precision level on a resident fp16 tensor -- deliberately, since that
characterizes sensitivity without committing to a packed format. But it means there is no
per-row *storage* format and no per-row *serving* path anywhere: the plugin loads trellis
tensors only, and `ops.embed_rows` is trellis row extraction. Both ends have to be built
before a repair tool's output is loadable at all.

**Trap for whoever builds that: quantize from the original fp16 embedding, never from a
trellis reconstruction.** A tied checkpoint offers two apparent sources for the embedding
matrix — the true fp16 `embed_tokens`, and a dequantization of the quantized `lm_head`,
which is the same logical matrix and is *right there*. The second is already ~2% lossy per
row. Quantizing it again is double quantization and the errors compound visibly, for no
saving: the fp16 original is in the same checkpoint. The two techniques stack (serve a
tied model from its existing head today; emit a per-row tensor in a repair tool) but each
needs its own best-available source.

### The head sweep: the mirror image, and it settles the tied-model question

Same method, `head_quant`, holding the embedding at fp16 and replacing the checkpoint's
quantized head with the (tied) embedding matrix put through the same fake quantizer.
`bits: 16` is the control that isolates the trellis head's own cost.

Control (fp16 embedding, fp16 head): KLD **0.026890**.

| head encoding | tax | |
|---|---|---|
| **trellis, 6.004-bit (real)** | **+0.000072** | 1x |
| per-row 8-bit | +0.000423 | 6x worse |
| per-row 6-bit | +0.004531 | **63x worse** |
| per-row 5-bit | +0.045249 | 625x worse |

**Exactly the mirror image of the embedding result.** At the same 6 bits, the trellis is
63x *better* for the head, where per-row was 89x better for the embedding. Per-row does
not catch the trellis on the head even given two extra bits (8-bit per-row is still 6x
worse than 6-bit trellis).

This is strong confirmation of the objective-mismatch reading rather than any defect in
either scheme: each encoding wins decisively at the job it was designed for. The trellis
optimizes `x @ W.T` against typical activations, which is what a head does. A per-row
min/max integer scheme preserves each row as a vector, which is what an embedding needs.
Neither transfers.

The two error sources are also **independent and additive**: head 6-bit (+0.004531) plus
embedding 6-bit (+0.000243) predicts +0.004774, and sharing one 6-bit per-row tensor for
both roles measures +0.004771. So the two can be chosen separately without interaction.

### The full menu, gemma-4-12B @4.00bpw

| option | embed+head | KLD | tax |
|---|---|---|---|
| exllamav3 native (fp16 embed + trellis head) | 2.579 GiB | 0.026963 | +0 |
| **Phase A: one shared trellis** | **0.704 GiB** | 0.048582 | +0.021619 |
| **one shared per-row 6-bit** | **0.703 GiB** | 0.031734 | +0.004771 |
| one shared per-row 5-bit | 0.586 GiB | 0.072510 | +0.045547 |
| trellis head + per-row 4-bit embed | 1.172 GiB | 0.028835 | +0.001872 |
| trellis head + per-row 6-bit embed | 1.407 GiB | 0.027206 | +0.000243 |

### The measured frontier

Filling in the shared-tensor depths (all measured; the noise floor for this model and test
set is **0.001759**):

| option | embed+head | tax | vs noise floor | |
|---|---|---|---|---|
| shared per-row 5-bit | 0.586 GiB | +0.045547 | 26x | frontier |
| shared per-row 6-bit | 0.703 GiB | +0.004771 | 2.7x | frontier |
| shared trellis 6.004-bit (Phase A) | 0.704 GiB | +0.021619 | 12x | **dominated** |
| **shared per-row 7-bit** | **0.820 GiB** | **+0.001120** | **0.64x** | frontier |
| shared per-row 8-bit | 0.938 GiB | +0.000413 | 0.23x | frontier |
| trellis head + per-row 4-bit embed | 1.172 GiB | +0.001944 | 1.1x | **dominated** |
| trellis head + per-row 6-bit embed | 1.407 GiB | +0.000315 | 0.18x | frontier |
| exllamav3 native | 2.579 GiB | +0 | — | frontier |

**At 7 bits shared, the entire embed+head encoding costs less divergence than the model's
own numerical self-noise**, for 0.820 GiB against native's 2.579 -- a 1.76 GiB saving that
is, in quality terms, free. 8-bit buys a further 2.7x margin for 0.118 GiB.

Splitting is dominated across the whole useful range: shared 7-bit is both 0.35 GiB
*smaller* and 1.7x *better* than trellis-head-plus-4-bit-embed. The reason is structural --
splitting stores the same logical matrix twice, and since the embedding has a hard floor at
4 bits (3-bit is a cliff), splitting cannot go below `head_bits + 4`. Sharing satisfies
both roles at 7. Splitting only re-enters above ~1.4 GiB, where it is paying ~0.47 GiB to
shave 0.00009 off something already well under the noise floor.

Additivity held at every point and was slightly conservative (measured came in 0.000003 to
0.000274 *better* than head-tax + embed-tax predicted), so the two roles can be costed
independently and the cross-product does not need sweeping.

Two conclusions for a **tied** model:

1. **One shared per-row tensor, at 7 bits.** Not the trellis (dominated), and not two
   tensors (also dominated). This is what a repair tool should emit, and it is simpler
   than either alternative: no Hadamard, no block gather, row extraction is a slice.
2. **The head sets the bit depth, not the embedding.** The head is ~60x more
   bit-sensitive, so a shared tensor is priced by what the head needs and the embedding
   rides along above its own requirement. That waste is still far cheaper than a second
   copy of the matrix.

Untied models are unaffected by the sharing question -- genuinely different matrices --
and simply want a per-row embedding at 4-6 bits alongside their existing trellis head,
which the additive model prices directly off the embedding column above.

### The frontier scores bytes and KLD, not serviceability -- and that reorders it

The table above ranks storage against divergence, which is the right way to price what a
repair tool should *emit*. It is not sufficient to decide what to *build first*, because
it does not ask whether either role can still be served from the result.

A tied model's tensor does two jobs. The embedding role is a row gather, which per-row
makes strictly easier -- a slice instead of a 128-block decode. The head role is a GEMM,
and today that is `ops.exl3_mm`, the fused trellis kernel that keeps weights packed and
decodes inside the MMA pipeline. **There is no per-row-integer GEMM.** exllamav3 exposes
four entry points (`reconstruct`, `exl3_gemm`, `exl3_mgemm`, `had_r_128`) and none of them
serves per-row-quantized weights, nor does anything in torch keep them packed at 4-7 bits.

So a shared per-row tensor leaves three options: dequantize to dense fp16 for the head,
which returns 1.875 GiB on gemma-4-12B and destroys the entire saving; write an int-GEMM
kernel, which is the one kind of work this project has deliberately stayed out of; or do
not share. On that axis the ranking inverts:

| option | embed+head | tax | new kernel needed? |
|---|---|---|---|
| shared per-row 7-bit | 0.820 GiB | +0.00112 | **yes** -- head has no GEMM |
| trellis head + per-row 4-bit embed | 1.172 GiB | +0.00194 | no |
| trellis head + per-row 6-bit embed | 1.407 GiB | +0.00032 | no |
| exllamav3 native | 2.579 GiB | +0 | — |

The "dominated" split option costs 0.35 GiB more than sharing and is still 1.4 GiB better
than native, needs no kernel work, and -- decisively -- **is the only shape untied models
can use at all**, since their head really is a different matrix. One implementation
therefore covers untied models completely and tied models at a small penalty.

**Build the split first; treat the shared tensor as a later optimization gated on kernel
work.** Its value is also narrower than it looks: sharing only helps *tied* models, and of
the checkpoints censused here only the gemma-4 family is both tied and mid-size.

*Both halves of that gating have since moved (2026-08-26). The kernel gate is gone -- see
"fp8 as the shared tensor" below. And gemma-4's deployability was recorded as riding on
the flash-attention head-dim-512 question, which was a misdiagnosis: what decides it is the
quantized KV cache, tracked under FUTURE.md `turboquant-sliding-window`, whose walls have so far
reduced to one-line staleness bugs and which covers gemma-4, Laguna and Muse-Glimmer --
every real-world candidate except Qwen. On present evidence that is more likely to be
resolved than not, so the constituency for a shared tensor is real rather than
hypothetical, and the priority demotion this paragraph justified should be unwound with the
misdiagnosis that caused it.*

### fp8 as the shared tensor: the kernel gate is gone

*Measured 2026-08-26, `~/qbench/gemma-tied-fp8.yaml`, same gemma-4-12B @4.00bpw_mul1 and
test set as the sweeps above. Reproduces both recorded anchors exactly -- head per-row-8 at
+0.000423 and shared per-row-8 at +0.000412 against the recorded +0.000423 and +0.000413 --
so it is directly comparable rather than merely similar.*

The section above gates the shared tensor on writing an int-GEMM. fp8-e4m3 is the obvious
way out: it is 8 bits, it is per-channel scaled, and `torch._scaled_mm` already exists, so
the head matmul is a primitive rather than a kernel project. The question is only whether
it is good enough, since e4m3 holds *relative* error roughly constant where an integer arm
holds absolute error constant within a row -- a different failure mode, not a better one.

Head role, against the fp16 control (0.026890), the same convention as the head sweep:

| head encoding | tax | vs trellis |
|---|---|---|
| trellis, 6.004-bit (real) | +0.000073 | 1x |
| per-row int8 | +0.000423 | 5.8x |
| **fp8-e4m3, per-channel** | **+0.000686** | **9.4x** |
| fp8-e4m3, per-tensor | +0.001110 | 15.2x |

**Per-tensor fp8 is 1.6x worse than per-channel for free**, so the channel scale is not
optional. The embedding role is nearly free in either format -- sharing adds only +0.000056
on top of the fp8 head and +0.000062 on top of the int8 head -- which confirms conclusion 2
above from a third direction: **the head is ~91% of the damage**, and the shared tensor is
priced entirely by what the head needs.

Placed on the frontier (tax against native, `embed+head` bytes, noise floor 0.001759):

| option | embed+head | tax | vs floor | new kernel? |
|---|---|---|---|---|
| shared per-row 7-bit | 0.820 GiB | +0.001120 | 0.64x | yes |
| **shared fp8-e4m3** | **0.938 GiB** | **+0.000669** | **0.38x** | **no** |
| shared per-row 8-bit | 0.938 GiB | +0.000412 | 0.23x | yes |
| blockq-32 4-bit embed + trellis head | 1.234 GiB | +0.000297 | 0.17x | no, but does not run yet -- see below |
| trellis head + per-row 6-bit embed | 1.407 GiB | +0.000315 | 0.18x | no |
| exllamav3 native | 2.579 GiB | +0 | — | — |

Two readings, and the second is the one that decides it.

**On quality, fp8 is acceptable.** At 0.38x the noise floor it sits comfortably inside the
region this note already called free, and it is *better* than the 7-bit per-row point the
sweep above recommends as the tied operating point -- so the prior that fp8 would be too
lossy was wrong. It loses to int8 per-row by 1.6x at byte-for-byte parity, exactly as the
Gaussian prior predicted in direction if not in size, but both are far under the floor.

**On value, the comparison depends on which baseline is real -- and for a tied model the
split is not one yet.** The blockq row above is *not* what ships for gemma-4: a tied
checkpoint with a block-quantized embedding still crashes at logits time, because both
predicates in `quantization/config.py` are per-checkpoint while `EXL3TiedLMHeadMethod`
reads a trellis off a module that now holds `bq_*`. That is tracked under TODO
`quantized-embeddings`. So for the one family this question is about, **today's baseline is
native's 2.579 GiB**, and both routes are unbuilt:

| route | gets to | saves vs native | work |
|---|---|---|---|
| fix tied+blockq, keep the trellis head | 1.234 GiB | 1.345 GiB | per-module predicates, rename at the head's own prefix, a guard for the silent trellis loss |
| shared fp8-e4m3 | 0.938 GiB | **1.641 GiB** | fp8 head path, fp8 gather, repair-tool emission |

Measured that way fp8 is the *larger* saving of the two, by the same 0.296 GiB -- so the
frontier's original verdict, that sharing is worth real work, survives contact with fp8
rather than being overturned by it.

**What settles the order is a correctness bug, and it needs no argument from any model's
priority.** The tempting justification -- that the fix must happen anyway because
`gemma4-e2b` needs it -- is the weakest available. E2B/E4B is a development aid rather than
a serving target, and reaching it at all would take an entirely new architecture in the
exl3 pipeline; that is a great deal of work to hang on a curiosity, and a poor thing to
sequence a roadmap around. Its actual value is narrower and real: it is the only checkpoint
on hand that is tied *and* needs per-layer blockq, so it exercises together two paths
nothing else reaches.

**Fixed 2026-08-26**, and the fix is what makes the blockq row above real for a tied
model. It was reachable by the obvious workflow -- which is how it was found.
`tools/quantize_embedding.py` documented itself as scoped to *untied* models and enforced
that nowhere: no `tie_word_embeddings` check, only `key.endswith(".embed_tokens.weight")`,
and a tied EXL3 checkpoint *does* carry a dense embedding -- `gemma-4-12B-it-exl3` ships
`model.language_model.embed_tokens.weight` at `[262144, 3840]` BF16 beside
`lm_head.trellis`. That redundancy is precisely what this note exists to remove, so
naturally it is there. (The `SystemExit` in `find_embedding` said "a tied checkpoint has
nothing for this tool to do" -- prose in a *not-found* branch, false for EXL3, and it reads
like a guard while being none.)

The cause was one assumption: `embedding_is_quantized()` and `embedding_is_blockq()` were
written as mutually exclusive, but they ask about *different modules* -- whether the head's
trellis is being renamed onto the embedding, and whether the embedding has `bq_*` of its
own. A repaired tied checkpoint makes both true and both load-bearing.
`EXL3BlockQTiedEmbeddingMethod` now carries both parameter sets on the one module, lookup
on `bq_*` and logits on the trellis, with no vLLM patch and **no metadata change** --
`tie_word_embeddings` stays `true`, which it is. Serving a tied model this way replaces the
trellis-served lookup (Phase A, +0.021619) with the blockq one (+0.000297): **~73x less
divergence for +0.53 GiB.**

Verified on a repaired tied gemma-4-12B: it holds ~0.53 GiB more weight than the
unrepaired baseline (KV cache 19,176 -> 17,545 tokens), and 150 of 157 prompt logprobs
differ from it, mean |delta| 0.122 -- the gather is executing rather than merely allocated.

**So: fp8 is confirmed viable, the shared tensor is deferred rather than closed, and its
priority has gone up rather than down.** The blocker it was gated on -- no scalar-integer
GEMM -- is genuinely gone, and the second-order gate that demoted the whole tied-model line
was a misdiagnosis (see `turboquant-sliding-window` in [turboquant-kv.md](turboquant-kv.md)), so
the constituency is real. That ordering has now resolved: the crash is fixed, the split
*is* a 1.234 GiB baseline, and fp8's margin is therefore the marginal 0.296 GiB for 2.25x
the divergence rather than a route to a working path. Worth having -- 0.296 GiB is real in
a VRAM-bound appliance and 2.25x of something already 0.38x the noise floor is still under
the floor -- but it is now an optimization on top of something that works, needing an fp8
head path, an fp8 gather and repair-tool emission to collect. The recorded "7 or 8 bit
shared per-row" operating point is **superseded on encoding**: fp8 is the shape to build,
if it is built.



### Shipping it as a sidecar rather than a rewrite

`tools/quantize_embedding.py` rewrites the shard that held the embedding, so the
output duplicates that shard. For a single-file checkpoint that is the whole
model: `Qwen3-8B-exl3` at 6.0bpw spent 5.62 GiB of new data to add 0.33 GiB of
`bq_*`. `--sidecar` instead hardlinks every original shard, writes `bq_*` in a
shard of their own, and renumbers the result into `model-0000i-of-0000N` with an
index. The cost falls to the `bq_*` bytes, provided the source stays where the
links can reach it — measured across the twelve local checkpoints, **67.62 GiB
of unshared data became 5.00 GiB**, and every `bq_*` tensor came out
byte-identical to the rewrite's.

The trade is that a sidecar checkpoint is not self-contained: it depends on the
source, and it carries *both* encodings, since the dense embedding is never
removed. That second property is useful rather than incidental —
`EXL3_DENSE_EMBED=1` then serves the dense copy and ignores `bq_*`, which is the
only way to measure the two against each other on one set of weights. On
MiniCPM5-1B that reads as identical greedy tokens with logprobs 1.5e-1 apart.

Hardlinking is the whole economy, so the tool now says when it could not: across
a filesystem boundary every shard is copied instead and the mode saves nothing.

## Choosing depths: what the repair tool should default to

*Partly superseded by "Is GGUF the right storage format?" below, which changed the storage
layout from per-row to block-scaled. The marginal-KLD-per-byte criterion and the head/layer
curves still hold; the per-model depth calibration this section concludes with does not —
one constant covers every model measured in the block-scaled layout.*

The criterion that makes this well-posed is **equal marginal KLD per byte** -- push a
component's depth until the next byte buys less there than it would elsewhere. Both curves
are measured: the embed/head taxes here, and the layer curve from turboderp's own model
card for `gemma-4-12B-it-exl3` (2.00-8.00bpw), which agrees with our measurements to within
label rounding where they overlap (3.00/3.50/4.00bpw: their 0.107/0.073/0.028 vs our
0.1029/0.0703/0.0270).

**The marginal is a *forward* difference** -- "the marginal at 4.00bpw" means what the
next bit buys, 4.00 -> 5.00, the same convention as "the 3->4 bit marginal is 4.13 KLD/GiB,
the 4->5 is 0.011" below. Stated because reading it as a backward difference makes the
table look 5x wrong at 4.00 and invites a correction that would break it: the measured
3.50 -> 4.00 slope really is 0.068 KLD/GiB while the 4.00 -> 5.00 marginal really is 0.014.
Both are right. The curve is *convex* through this range -- improvement accelerates into
4.00bpw and then collapses -- which is the whole reason a crossover exists at all.

**Confirmed independently (2026-08-26, `~/qbench/gemma-body-curve.yaml`).** The layer curve
above was taken from turboderp's model card; it has now been measured here on a matched
codebook, with `head_quant: 16` so only the body varies:

| revision | body-only KLD |
|---|---|
| 3.00bpw (plain codebook) | 0.102867 |
| 3.00bpw_mul1 | 0.102701 |
| 3.50bpw_mul1 | 0.070143 |
| 4.00bpw_mul1 | 0.026890 |

Reproduces 0.1029/0.0703/0.0270. It also **refutes a codebook confound**: plain and `mul1`
at the same bitrate differ by 0.000166 in body KLD, so a curve mixing the two is still a
bitrate curve. (Unexplained, and not the same question: those two have near-identical KLD
but perplexity 17.9276 against 18.4556 -- equal divergence from the reference, differently
directed. Ten rows, so a lead rather than a finding.)

### When is a block-quantized embedding worth the VRAM, against just buying body bits?

A *tied* model already drops its dense embedding and serves the lookup from the trellis, so
adding `bq_*` **costs** VRAM rather than saving it -- +0.531 GiB on gemma-4-12B -- in
exchange for taking that lookup from +0.021619 to +0.000297. The same 0.531 GiB could
instead buy ~0.42 bpw of body. Which wins is a marginal question, and both sides are now
measured:

blockq delivers `0.021322 / 0.531` = **0.0402 KLD/GiB**. The body beats that where its
forward marginal is larger:

| at body bpw | forward layer marginal | vs 0.0402 | better spend |
|---|---|---|---|
| 3.00 | 0.0514 | 1.28x | body bits |
| 3.50 | 0.0684 | 1.70x | body bits |
| 4.00 | 0.0142 | 0.35x | **blockq** |

**The crossover is essentially 4.00bpw.** Above it, repair the embedding; below it, spend
the bytes on the body and leave the trellis serving the lookup. The intuition is the ratio
rather than the absolute: the trellis-served embedding costs a fixed ~0.0216, which is 1.80x
the body's own damage at 4.00bpw but only 1.21x at 3.00bpw -- a rounding error next to a
body that is already far worse.

Two consequences. **gemma-4-12B at 4.00bpw is a clear yes.** **gemma-4-26B-A4B at 2.54bpw
is a clear no**, and not because it is bigger -- its blockq is actually *cheaper* (0.390
GiB, since hidden is 2816 rather than 3840). It is the bitrate that decides, and down there
the layer marginal is roughly an order of magnitude above the bar.

Caveat on the one number this rests on: the 4.00 -> 5.00 marginal is still the model card's,
not ours. It agreed with our measurements at all three overlapping points, but pinning the
crossover rather than inferring it wants a 4.50 or 5.00bpw run.

Note the leverage: **1 bpw of layers costs 10.8x what 1 bit of embed+head costs** on this
model, which is why the first few embed/head bits are such good value and why they fall off
a cliff so quickly afterwards.

**Tied models (one shared per-row tensor): `depth = body_bpw + 3`.**

| body bpw | layer marginal (KLD/GiB) | optimal depth | ratio | offset |
|---|---|---|---|---|
| 2.00 | 0.5892 | 5 | 2.50x | +3.0 |
| 3.00 | 0.0536 | 6 | 2.00x | +3.0 |
| 4.00 | 0.0142 | 7 | 1.75x | +3.0 |
| 5.00 | 0.0039 | 8 | 1.60x | +3.0 |
| 6.00 | 0.0008 | 8 | 1.33x | +2.0 |

The offset is stable where the ratio is not (2.50x -> 1.33x), so this is an additive law.
"Double the body depth" is a good rule of thumb only near 3bpw, where 2x3 happens to equal
3+3; it over-provisions above that. This also rehabilitates 5-bit: catastrophic at a 4bpw
body (+169%) but the *correct* choice at 2bpw (+6%), because it was never intrinsically bad,
only mismatched.

**Untied models (keep the trellis head, add a per-row embedding): measure per model.**

*Superseded below.* The 4-bit figure derived here holds for gemma-4-12B and does not
generalize -- see "Untied models, measured" for three models spanning a 35x range.

The head is what drives the tied depth. Remove it from the shared tensor and the embedding
alone wants 4 bits at every body depth in shipping range (2.00-4.50bpw), set by the 3-bit
cliff rather than by any tradeoff -- the 3->4 bit marginal is 4.13 KLD/GiB, the 4->5 is
0.011, and the layer curve sits between them across that whole range.

It is cheaper than that framing suggests: the embedding-only tax is **+0.00187 at 4 bits,
which is this model's noise floor (0.00176)**, and +0.00057 at 5 bits, a third of it. Above
5 bits the tax is not resolvable -- the 6/7/8-bit measurements are non-monotone, which is
noise. So: 4 bits, or 5 for margin, and never more.

**The size-budget solver belongs to the full quantizer, not the repair tool.** A repair tool
takes a checkpoint whose layer bpw is already fixed and cannot move bytes into the body, so
it has exactly one free variable and nothing to solve -- the heuristic (or an explicit
override) is the right interface. Trading depth *across* components only becomes a real
optimization in the from-scratch quantizer, where layer bpw is free and the
Lagrangian actually binds.

### Untied models, measured

*The 35x spread below is a property of the per-row scheme, not of the models. See "Is GGUF
the right storage format?" — under the block-scaled layout the same three models are within
a factor of 8 of each other and all fit one default.*

Three checkpoints, embedding varied with the trellis head untouched. Two are genuinely
untied, so this is the first time the embedding is perturbed on models where it is a
*different matrix* from the head; gemma-4-12B is carried over from the sweeps above as
the tied reference point, not as a fourth untied case:

| model | arch | vocab x hidden | body | noise floor | 4-bit tax | 5-bit | 6-bit |
|---|---|---|---|---|---|---|---|
| gemma-4-12B (tied ckpt) | Gemma3 | 262144 x 3840 | 4.00 | 0.001759 | +0.00187 | +0.00057 | +0.00024 |
| MiniCPM5-1B | Llama | 130560 x 1536 | 3.00 | 0.000966 | +0.00568 | +0.00068 | +0.00077 |
| Qwen3.5-9B | Qwen3.5 | 248320 x 4096 | 4.00 | 0.000718 | **+0.06453** | +0.01488 | +0.00173 |

**The flat-4-bit rule is false.** The 4-bit tax spans 35x across models, and the optimal
depth under the marginal criterion is 4 / ~5 / 6 respectively. Qwen3.5-9B at 4 bits costs
+0.065 against a 4.00bpw body of 0.0102 -- a 6x increase in total divergence.

**What does generalize is that the tax is body-independent.** Qwen3.5-9B measured at three
body depths:

| body bpw | body KLD | 5-bit tax | 4-bit tax |
|---|---|---|---|
| 2.00 | 0.212714 | +0.014252 | +0.065794 |
| 4.00 | 0.010221 | +0.014883 | +0.064530 |
| 6.00 | 0.001237 | +0.014252 | +0.064228 |

Constant to three significant figures. So **one sweep at any single body depth
characterizes a model completely**, which keeps per-model calibration cheap even though no
universal constant exists.

**It is not encoding difficulty.** Per-row distribution shape is nearly identical across
all three (crest factor p50 3.5-4.0, kurtosis ~3, i.e. Gaussian), so per-row min/max
produces similar *relative* error everywhere; gemma is even the outlier in the wrong
direction (crest p99 of 31.5) and is the *least* sensitive. The 35x spread is how much
each model's downstream computation cares about a given embedding perturbation, not how
well the perturbation is represented. Mechanism unestablished -- immediate post-embedding
normalization is a plausible candidate and is untested.

Practical consequence for the repair tool: it cannot ship a constant. It needs either a
per-model calibration sweep (cheap, one body depth) or a conservative default -- 6 bits
covers all three models measured, at a cost of over-provisioning gemma by ~2 bits.

## How far this generalizes

Everything above is one model family at one vocabulary size. Two things temper how much
that matters:

- **Tied models are rare in the target range.** Of the checkpoints censused here, only the
  gemma-4 family is both tied and mid-size; Qwen3.5-9B, Qwen3.6-27B, Laguna and MiniCPM5
  are all untied. So the `body + 3` rule governs a narrow slice, and the flat 4-bit untied
  rule governs most of what would actually be deployed.
- **Tied small models still load through this path** (Llama-3.2-1B, Qwen3-0.6B, where the
  embedding is 50%+ of the checkpoint), so Phase A serves them today with no tooling. But
  they do *not* motivate the depth heuristic, and it should not be applied to them: a
  speculative-decoding draft model has its every token verified against the target, so its
  output distribution is the target's no matter how badly it is quantized. Draft quality is
  an acceptance-rate knob -- speed, not fidelity -- and KLD against the draft's own fp16
  self, which is what qbench measures, is simply the wrong metric. The relevant divergence
  would be against the *target* model, where the baseline disagreement is dominated by the
  model-size gap rather than by quantization, so the marginal cost of quantizing a draft
  hard is small. Combined with its small share of total VRAM, allocation decisions there
  barely move anything. Draft models want "quantize aggressively, measure acceptance rate",
  which is a different experiment and a different harness.

The *shape* of the findings should transfer (per-row for embeddings, trellis for heads,
additivity, head-sets-the-depth); the specific constants are gemma-4-12B's and are cheap to
re-derive per family once a tool exists to produce the tensors.

## Quality at scale: asked, and answered

Phase A makes the embedding inherit the head's bit width (`head_bits`, usually 6), which
is a quality question rather than plumbing — and gemma-4 is the most numerically delicate
family here (it already needs fp32 residuals, and is the reason for the flash-attention
head-dim work, retired -- see docs/kernels.md). This was the open question when Phase A shipped.

It has since been answered, by the sweeps above: `Exl3Backend`'s `embed_quant` was indeed
the right instrument, and the knee is *not* a constant. On gemma-4-12B inheriting 6.004
bits of trellis costs +0.0216 KLD, an order of magnitude more than a per-row scheme at the
same depth; across three models the 4-bit tax spans 35x, so depth has to be calibrated per
model. See "Choosing depths" and "Untied models, measured".

What remains open is only the generalization: everything measured is gemma-4-12B,
MiniCPM5-1B and Qwen3.5-9B. The *shape* of the findings should transfer; the constants
are theirs.

## Is GGUF the right storage format? Measured, and no — but its *layout* is

*Tracked in TODO as `gguf-embeddings`, and this settles it. Measured 2026-08-17.*

The question was whether a GGUF quantization type should be the storage format for a
quantized embedding, instead of the bespoke per-row scheme the sweeps above settled on.
It is a fair question: GGUF's k-quants are the same *family* the per-row result pointed
at (block-scaled scalar quantization, not a trellis), matured over years, with encoders,
GPU dequant kernels and an already-written vLLM gather path in the sibling
`vllm-gguf-plugin`.

Answer: **no.** Not because GGUF is worse — at matched bytes it is roughly a tie — but
because the entire measurable advantage comes from a structural property that is ~30
lines to implement ourselves, and none of it comes from the parts that would cost a
dependency.

### Method

Three models, each with its EXL3 checkpoint's embedding replaced and everything else held
fixed, scored against that model's own bf16 reference. Two new `Exl3Backend` hooks make
the arms commensurable: `embed_quant`'s `granularity` gained `block:N` and `kshape:N`, and
a new `embed_file` option substitutes an embedding matrix from a safetensors file — which
is how a *real* GGUF encoder's output gets scored on exactly the same footing as the
simulated precision levels.

The GGUF embeddings are real published ones (unsloth and bartowski quants of the same base
checkpoints the EXL3 quants came from), dequantized offline with `gguf-py`. Only the
`token_embd` tensor was fetched — a GGUF's tensor offsets are in its header, so an HTTP
range request against a sparse local file gets one tensor out of a 5 GiB quant without
downloading the rest.

The schemes, with what each really costs to store:

| scheme | bpw | what it is |
|---|---|---|
| per-row N-bit | N + 32/hidden | fp16 min+max per row — **the bespoke plan** |
| per-blk{32,64,128} N-bit | N + {1, 0.5, 0.25} | fp16 min+max per block, the naive block scheme |
| k-shape N-bit | N + 0.5 | **Q4_K's layout, naive encoder** (see below) |
| GGUF IQ3_S / IQ4_XS / Q4_K / Q5_K | 3.4375 / 4.25 / 4.5 / 5.5 | the real thing |

"k-shape" is the arm that makes the comparison decide something. It is GGUF's own
structure — superblocks of 256, sub-blocks of 32, each with its own (min, scale), those
sub-scalars themselves quantized to 6 bits against one fp16 pair per superblock — encoded
the most naive possible way, from min/max, with no rounding search and no imatrix. At
4 bits it is 4.5 bpw, *byte-identical to Q4_K*. Whatever separates the two is llama.cpp's
encoder, not its format, which is precisely the thing worth knowing before taking on a
dependency to get it.

### The measurements

Tax = KLD above the same checkpoint's fp16-embedding baseline. "x floor" is that tax
against the model's own self-noise floor, i.e. below 1.0 is not resolvable.

**gemma-4-12B-it @4.00bpw** (baseline 0.026963, floor 0.001759):

| bpw | scheme | tax | x floor |
|---|---|---|---|
| 3.0083 | per-row 3-bit | +0.485605 | 276 |
| 3.0000 | per-blk32 2-bit | +0.295291 | 168 |
| **3.5000** | **k-shape 3-bit** | **+0.000690** | 0.39 |
| 4.0000 | per-blk32 3-bit | +0.000742 | 0.42 |
| 4.0083 | per-row 4-bit | +0.001872 | 1.06 |
| **4.5000** | **k-shape 4-bit** | **+0.000330** | 0.19 |
| 4.5000 | per-blk64 4-bit | +0.000399 | 0.23 |
| 5.0083 | per-row 5-bit | +0.000572 | 0.33 |
| 5.5000 | **GGUF Q5_K** | +0.000221 | 0.13 |
| 5.5000 | per-blk64 5-bit | +0.000000 | 0.00 |
| 6.0083 | per-row 6-bit | +0.000243 | 0.14 |

**Qwen3.5-9B @4.00bpw** (baseline 0.013128, floor 0.001350) — the model that resolves the
question, because its embedding taxes run to 90x its noise floor where gemma's sit below
it:

| bpw | scheme | tax | x floor |
|---|---|---|---|
| 3.4375 | **GGUF IQ3_S** | +0.006802 | 5.04 |
| **3.5000** | **k-shape 3-bit** | **+0.003993** | 2.96 |
| 3.5000 | per-blk64 3-bit | +0.007254 | 5.37 |
| 4.0000 | per-blk32 3-bit | +0.004181 | 3.10 |
| 4.0078 | per-row 4-bit | +0.122225 | **90.6** |
| 4.2500 | **GGUF IQ4_XS** | +0.000845 | 0.63 |
| 4.2500 | per-blk128 4-bit | +0.003407 | 2.52 |
| 4.5000 | **GGUF Q4_K** | +0.000308 | 0.23 |
| **4.5000** | **k-shape 4-bit** | **+0.000428** | 0.32 |
| 4.5000 | per-blk64 4-bit | +0.000970 | 0.72 |
| 5.0078 | per-row 5-bit | +0.044159 | 32.7 |
| 6.0078 | per-row 6-bit | +0.004945 | 3.66 |

**MiniCPM5-1B @3.00bpw** (baseline 0.114097, floor 0.000977) — the case that reverses the
*naive* block scheme, and the reason the k-shape arm matters:

| bpw | scheme | tax | x floor |
|---|---|---|---|
| 3.5000 | k-shape 3-bit | +0.008615 | 8.82 |
| 4.0000 | per-blk32 3-bit | +0.009376 | 9.59 |
| 4.0208 | per-row 4-bit | +0.003920 | 4.01 |
| 4.5000 | k-shape 4-bit | +0.001456 | 1.49 |
| 5.0000 | per-blk32 4-bit | +0.001458 | 1.49 |
| 5.0208 | per-row 5-bit | +0.000724 | 0.74 |
| 5.5000 | k-shape 5-bit | +0.000353 | 0.36 |
| 6.0208 | per-row 6-bit | +0.000268 | 0.27 |

### What it says

**1. Per-row min/max is dominated, and not narrowly.** At matched bytes on Qwen3.5-9B,
block-scaled 3-bit costs +0.0042 where per-row 4-bit costs +0.1222 — **29x**, and at
5 bpw it is 134x. The mechanism is visible in reconstruction: per-row's *worst* rows blow
up (row-relative L2 p99 0.61 and max 2.25 on gemma at 4 bits, i.e. rows destroyed
outright) while its median row is fine, because one outlier component sets the scale for
the whole row. A block scale confines that damage to 32 values. Since an embedding row is
one *token*, the tail is exactly what matters.

**2. GGUF's k-quants and a naive encoder in GGUF's layout are a tie.** At byte-identical
4.5 bpw on Qwen3.5-9B: real Q4_K +0.000308 against k-shape +0.000428 — llama.cpp's
encoder is 1.4x better, and both are 3-4x *below* the noise floor, so the margin is not
resolvable. At the low end the ordering reverses: at ~3.44-3.5 bpw our naive encoder is
1.7x **better** than IQ3_S (+0.0040 vs +0.0068), where the difference *is* resolvable. The
one clear GGUF win is IQ4_XS at 4.25 bpw (+0.00085), which our k-shape does not have a
byte-matched arm for and which is also below the floor.

**3. So the win is the layout, not the encoder and not the format.** Hierarchical scales
(6-bit sub-scales against one fp16 pair per superblock) are what buys block granularity at
0.5 bpw instead of the 1.0 bpw a naive fp16 pair per 32 costs — and that overhead is the
whole reason `per-blk32` *loses* to per-row on MiniCPM5-1B while `k-shape` does not. Across
all three models k-shape is never worse than per-row at matched bytes, and is 2.7x better
on gemma and ~30x on Qwen3.5-9B.

**4. Whether block scaling pays off is predictable from the tensor**, which is worth
knowing for a repair tool. Per-row range against the median per-32-block range:

| model | p50 | p90 | p99 | block scaling |
|---|---|---|---|---|
| gemma-4-12B-it | 1.84 | 2.48 | 15.56 | large win |
| Qwen3.5-9B | 1.78 | 1.95 | 2.26 | large win |
| MiniCPM5-1B | 1.63 | 1.81 | 1.98 | a wash |

That statistic explains the *reconstruction* ordering. It does not predict the downstream
magnitude — Qwen3.5-9B has the least heterogeneous rows of the two winners and by far the
biggest downstream effect — because downstream damage is heterogeneity times the model's
own sensitivity to embedding perturbation, and that sensitivity is the 35x spread already
recorded above.

### Decision

**Do not adopt a GGUF quant type. Adopt its layout, with our own encoder.** Concretely:
superblocks of 256, sub-blocks of 32, 6-bit quantized (min, scale) per sub-block against
one fp16 pair per superblock, at a chosen depth of 3-5 bits.

The reasons, in the order they weigh:

- **No quality is given up.** A tie at matched bytes, and a win at the low end.
- **The encoder gap is the deciding practical fact.** `gguf-py` cannot write k-quants —
  every k-quant raises `NotImplementedError`, because those encoders are C in llama.cpp.
  Adopting GGUF means shelling out to `llama-quantize`, binding ggml, or reimplementing
  the encoder anyway. The naive encoder measured here is ~30 lines of torch and is already
  written (it is the `kshape:N` arm).
- **Depth granularity survives.** GGUF offers a fixed menu; this keeps arbitrary depths,
  which the sweeps above show is worth real bytes since the optimum is per-model.
- **Row independence is kept, and it was never GGUF-specific.** Verified on real files:
  every k-quant and IQ type here stores whole blocks per row (`hidden % 256 == 0` on all
  three models), and dequantizing one row's byte slice alone is bit-identical to the
  full-tensor decode. Our own layout has the same property by construction — and unlike
  the trellis, no 128-row block decode.
- **No cross-plugin dependency, and no serving-path question.** Two out-of-tree
  quantization plugins registering different methods through `vllm.general_plugins`
  *should* coexist, but nobody has tried it, and the answer stops mattering: a block-scaled
  gather is a slice, an unpack and a multiply-add.

What is given up is real but small: llama.cpp's rounding search and imatrix weighting,
worth ~1.4x at 4.5 bpw and below the noise floor there. If that ever matters, the encoder
can be improved in place without touching the storage layout or the serving path — which
is the opposite of the situation adopting GGUF would leave us in.

**The hybrid-checkpoint objection dissolves rather than being answered.** A GGUF-embedded
EXL3 checkpoint would still be readable by nothing else (the body is trellis), so GGUF
bought no interoperability to begin with. The sibling `vllm-gguf-plugin` remains worth
reading for its plugin shape and its generic weight adapter — it is the only other example
of this exact architecture — but not worth depending on for this.

### Consequences for the rest of this note

Two earlier conclusions are superseded, both in the direction of *fewer bytes and less
per-model tuning*:

- **"The flat-4-bit rule is false... it cannot ship a constant"** was a per-row finding.
  With the k-shape layout, one setting covers all three models: **4 bits, 4.5 bpw**, at a
  tax of 0.19x / 0.32x / 1.49x the respective noise floors. The per-row scheme needed
  4 / ~5 / 6 bits per model to get there and needed 6 bits (6.02 bpw) as a conservative
  default. That is 1.5 bpw saved on every untied model — 0.18 GiB on each of
  gemma-4-12B's and Qwen3.5-9B's embeddings — plus the calibration sweep no longer being
  required per model.
- **The 3-bit cliff is a per-row artifact, not a property of the embedding.** Per-row
  3-bit destroys gemma (+0.486) and so does per-blk32 2-bit (+0.295, same 3.0 bpw), but
  k-shape 3-bit at 3.5 bpw costs +0.00069 — a third of the noise floor. The usable floor
  is ~3.5 bpw, not 4.

The head sweep is untouched by any of this: it measured that the trellis wins decisively
for the head role, and nothing here is a head encoding.

## Build or adopt: costed, and the answer flipped the cheap-looking option

*Follow-on to the section above, which established that GGUF wins nothing on quality.
That left a fair question — GGUF's encoder, kernels and gather path all exist and are
proven, so adopting them could still be cheaper than writing ours, even at equal quality.
Measured 2026-08-17.*

### A simpler format than Q4_K, measured before committing to either

The `k-shape` arm above replicated Q4_K's structure, including its 6-bit sub-scales packed
against one fp16 pair per 256-element superblock. That packing is the fiddliest part of the
layout, so the first question is whether it is load-bearing. `blockq32` drops it: per-block
(min, scale) for blocks of 32, both **quantized to int8 against one fp16 range per row**.
Same 0.5 bpw of scale overhead, no superblock, and every field byte-aligned — 4-bit values
are nibbles, scales are two `uint8` arrays, and a row carries four fp16 scalars.

Tax over each model's fp16-embedding baseline, at 4.5 bpw for all three schemes:

| model | noise floor | blockq32 4-bit (4.52) | k-shape 4-bit (4.50) | GGUF Q4_K (4.50) |
|---|---|---|---|---|
| Qwen3.5-9B | 0.001350 | **+0.000225** | +0.000428 | +0.000308 |
| gemma-4-12B-it | 0.001759 | **+0.000297** | +0.000330 | +0.000221 (Q5_K @5.5) |
| MiniCPM5-1B | 0.000977 | **+0.001435** | +0.001456 | — |

The simplest of the three is the best of the three on the model that resolves differences,
and a tie elsewhere — all comfortably under the noise floor. Q4_K's scale packing buys
nothing here. Lower depths behave the same way: blockq32 3-bit (3.52 bpw) costs +0.004046
on Qwen3.5-9B against IQ3_S's +0.006802 at 3.4375.

### What adopting GGUF would actually cost, checked rather than assumed

`vllm-gguf-plugin` publishes a prebuilt `cp310-abi3-manylinux_2_28` wheel, so there is no
build step, and its dequant works **today** against this environment's torch: gathering
rows out of a real Q4_K `token_embd` and calling `ops.ggml_dequantize` on the gathered rows
reproduces `gguf-py` to fp16 rounding (max abs diff 6.2e-05 for Q4_K, 3.1e-05 for IQ4_XS,
3.3e-03 for IQ3_S). So the sibling plugin's serving path is real and usable as a library.
Three things turned up alongside that, and they are what decide it.

**1. Installing it monkeypatches vLLM globally.** Its `vllm.general_plugins` entry point
runs on *every* vLLM start, and `register()` patches `EngineArgs.create_model_config`,
`maybe_override_with_speculators` and the diffusers loader — whether or not GGUF is in use.
Entry points activate by installation, not by import, so depending on the package cannot
opt out of that. This project's standing property is that all its hooks are sanctioned
extension points and nothing is monkeypatched; a dependency would import someone else's
patching of engine-argument parsing into every user's process, for one tensor.

**2. Only the kernel is reusable, and the kernel is not the work.** The loader, config
parser, weight adapters and `GGUFEmbeddingMethod` are all built around a checkpoint that
*is* a GGUF. Ours is an EXL3 checkpoint with one GGUF-encoded tensor in it, so none of that
machinery applies: what carries over is `ggml_dequantize` and four lines of `index_select`.
Every remaining piece — checkpoint keys, `create_weights`,
`process_weights_after_loading`, vocab-parallel sharding, capture safety, the repair tool's
CLI — has to be written either way. "All those pieces already exist" is true of the kernel
and false of the integration, and the integration is nearly all of the work.

**3. The kernel is not even a performance argument.** Both paths gather whole rows out of a
`[vocab, row_bytes]` uint8 tensor, so their memory traffic is identical and only the decode
differs. In eager mode the CUDA kernel wins on launch overhead, but under the mode that
actually ships — inductor-compiled inside a CUDA graph — a plain-torch unpack fuses and an
opaque custom op cannot:

| tokens | GGUF + CUDA kernel | blockq32, plain torch | blockq32, compiled |
|---|---|---|---|
| 1 | 4.2 us | 16.6 us | **2.1 us** |
| 8 | 6.2 us | 26.8 us | **2.2 us** |
| 64 | 6.2 us | 22.8 us | **4.2 us** |
| 512 | 14.5 us | 49.4 us | **10.3 us** |

(CUDA-graph replay, RTX 5070 Ti, Qwen3.5-9B's 248320 x 4096 embedding. Eager, the same
comparison runs 13.6 us against 74.2 us at one token — which is the number to quote if the
serving path ever cannot be compiled.)

### Decision: build it

**blockq32, 4-bit, as the default and initially the only depth.** The constant-depth result
above is what makes that scope defensible: 4 bits covers every model measured, so the
encoder and the unpack only ever have to handle nibbles, and 3- and 5-bit packing can wait
until something demands them.

Against adopting GGUF: equal quality (better at 3.5 bpw), faster on the path that ships,
no dependency that monkeypatches vLLM, no vendored CUDA kernel in a project that ships no
compiled extensions today, no second toolchain for encoding (`gguf-py` cannot write
k-quants; emitting real Q4_K bytes would mean binding libggml or implementing its packed
6-bit scales ourselves — strictly more work than implementing our own byte-aligned layout),
and it keeps working anywhere torch does, which matters while exllamav3 has no ROCm support
and this is one of the few paths that would not need a kernel port.

What is genuinely given up: llama.cpp's rounding search and imatrix weighting, worth ~1.4x
at 4.5 bpw against `k-shape` and *negative* against `blockq32` on the model that can resolve
it; and other people's testing of a mature codec. The second is the real one, and it is
bounded by the format being small enough to test exhaustively — an encoder/unpack round-trip
is exactly invertible by construction, which is a stronger test than any of this note's
statistical comparisons.

**The dependency stays worth having as a reference, not as a runtime.** `vocal_embeds.py`
is a good short model of the gather-and-dequantize shape, and this plugin remains the only
other example of this exact out-of-tree architecture.

## Phase B result: the format, built and serving

*Was TODO `quantized-embeddings`, now `tied-embeddings`. Built 2026-08-17, immediately after the
decision above.*

Untied models now serve a quantized embedding. `vllm_exl3_plugin/blockq.py` holds the
format, `tools/quantize_embedding.py` produces it, and `EXL3BlockQEmbeddingMethod` serves
it. [blockq-format.md](blockq-format.md) is the format reference — layout, decode and
encode in full, and what may be assumed about it. In brief, the stored layout is what the
measurements chose:

    <key>.bq_q   uint8   [vocab, hidden // 2]        4-bit values, two per byte
    <key>.bq_s   uint8   [vocab, 2, hidden // 32]    per-block scale codes, then min codes
    <key>.bq_r   fp32    [vocab, 4]                  one affine range per row, for each

A value is `q * scale + min`; the per-block `scale` and `min` are 8-bit codes against
`bq_r`'s per-row ranges. 4.5 bpw of values and scales, plus 128 bits per row for the
ranges — 4.531 bpw at `hidden = 4096`, 4.583 at 1536.

### Measured on two real checkpoints

| | checkpoint | embedding | resident embedding |
|---|---|---|---|
| MiniCPM5-1B @3.00bpw | 771 → 508 MiB | 0.374 → 0.107 GiB | 382 → 109.6 MiB |
| Qwen3.5-9B @4.00bpw | 6.72 → 5.36 GiB | 1.895 → 0.537 GiB | — |

Encoding a 248320 x 4096 embedding and rewriting the checkpoint takes 12 seconds on CPU.

### The end-to-end check that matters

Every number that justified this format came from qbench's `blockq:32` *simulation*, which
packs nothing. So the question the implementation has to answer is not "does it run" but
"is the thing now running the thing that was measured". Both arms through the vllm engine,
so the engine difference cancels in the delta:

| MiniCPM5-1B | dense embed | blockq32 4-bit | tax |
|---|---|---|---|
| simulated (exllamav3 engine) | 0.114097 | 0.115532 | +0.001435 |
| real packed format (vllm engine) | 0.114369 | 0.115928 | **+0.001559** |

Within 9% of the simulated tax, against a noise floor of 0.000977 — engine and
fp16-decode differences, not a scale bug, which would show up as orders of magnitude
rather than percent.

Three things are asserted in tests rather than left to inspection: the shipped encoder
reproduces an independent transcription of the simulated arithmetic; a row's storage
decodes identically whether gathered alone or as part of the whole tensor (the property
the serving path and vocab-parallel TP both rest on); and the decode agrees across eager,
`torch.compile` and CUDA-graph replay — the modes that have caught real defects here
before.

Two floating-point caveats found while writing those tests, both benign and both now
documented where they bite. Encoding is reproducible *per device*, not across devices:
`.round()` breaks ties differently on CPU and GPU, so a handful of codes per vocabulary
land one step apart — a byte-reproducibility caveat for the tool, not a correctness one
for the model. And compiled output differs from eager by up to one fp16 ulp on the odd
element, because inductor may fuse the multiply-add; exactness is required of graph
*replay*, and holds there.

### Two things validation turned up, neither of them in this format

**vLLM only asks for an embedding quant method on about a third of its
architectures.** `qwen3_5.py` constructs its `VocabParallelEmbedding` without
`quant_config`, so `get_quant_method` is never called for it and no quantized
embedding can be served there at all. On vLLM main, **86 of 131** model files that
construct one omit it; `llama.py`, `gemma4.py`, `qwen2.py`, `qwen3_moe.py` and
`deepseek_v2.py` pass it, which is why every model this feature had been tried on
before happened to work.

The failure modes differ in nastiness. A *tied* model silently keeps its dense
embedding — Phase A quietly does nothing, and has been able to quietly do nothing
since it shipped. A *block-quantized* checkpoint fails to load, complaining that
`embed_tokens.bq_q` does not exist, which points nowhere near the cause.
**The fix is one file, not 86.** `VocabParallelEmbedding.__init__` only consults a
config its caller passed, so `vllm-embed-quant-config` defaults it
from the config being built under — `get_current_vllm_config()`, already how
`linear.py` and `logits_processor.py` read ambient construction state. Every
architecture is reached at once, and configs that do not quantize embeddings are
unaffected, because they already return `None` for a `VocabParallelEmbedding` and
land on `UnquantizedEmbeddingMethod` exactly as before. Checked rather than
assumed: AWQ on `qwen3_5` (the very architecture the patch unblocks) still gets
the dense 1940 MiB path, bitsandbytes on Qwen3-0.6B likewise, and the plugin's own
end-to-end suite is unchanged.

Worth upstreaming rather than carrying: `vllm-gguf-plugin` has the identical
limitation — its `GGUFEmbeddingMethod` is unreachable on the same 86
architectures, and nothing in that package works around it — so this is an
ecosystem gap rather than ours.

With the patch, Qwen3.5-9B serves its embedding from 549 MiB instead of 1940, and
the served rows are **bit-identical** to what the encoder wrote — checked through
the whole load path (vLLM's weight loading, vocabulary padding, parameter
registration) rather than inferred from output looking reasonable.

**qbench's `vllm` engine cannot score Qwen3.5-9B**, which is why the end-to-end KLD
validation above is MiniCPM5-1B's rather than the more sensitive model's. Its
dense, unmodified checkpoint measures ppl 248076 through that engine while
generating coherent text through plain `LLM.generate`. Not a hybrid-Mamba
teacher-forcing limitation, which is what it first looked like: scoring through
vLLM's public `prompt_logprobs` API is self-consistent on these very models (see
[qbench.md](qbench.md)), so the fault is in that engine's own scoring path.

### Native exllamav3 can load these checkpoints too

*Added 2026-08-18, prompted by the triattention calibration question.*

exllamav3 originally refused a repaired checkpoint outright — `Required tensor
model.embed_tokens.weight not found` — which also put it out of reach of every `eval/`
harness and of anything else driving a model natively. `Embedding.load()` in the fork now
materializes the dense matrix from the packed tensors when they are present.

The served path never builds that matrix; it decodes only the rows a batch touches. But
the *values* are identical either way, so this is not a fictional configuration: it is
the shipped one, materialized for measurement convenience. The one thing materializing
does change is memory, which is why the size accounting had to learn the format in both
places it lives — `Embedding.stored_bytes` for the streamed engine, and
`safetensors_storage_info`'s suffix table for the checkpoint path, which was silently
omitting the entire embedding from `vram_gb` (bpw_embed 0.0). Both now report 4.5833 bpw
and 0.4859 GiB on MiniCPM5-1B and agree with each other.

**This tightens the end-to-end validation above considerably.** The repaired checkpoint
scored through the *same* engine as the simulation gives KLD 0.115528 against 0.115532 —
a 4e-6 difference, where the vllm engine differs by 4e-4. So the packed format reproduces
the measured scheme essentially exactly, and yesterday's 9% gap was the engine, as
suspected but not then demonstrated.

**One trap worth knowing if anything else ever decodes at load time.** The decode must ask
for its tensors with `no_defer`: under `begin_deferred_load()` a tensor's contents arrive
*after* `load()` returns, so arithmetic at load time reads an unfilled buffer and yields a
plausible-looking wrong matrix rather than an error. Every other module defers safely only
because none of them compute at load. It was caught by comparing against the plugin's own
decoder — the reason to keep two implementations of a format that must agree.

### What this does not cover yet

- **Tensor parallelism is implemented but untested.** All three tensors slice on dim 0, so
  `tp.ROLE_VOCAB` is a row slice with none of the trellis path's 128-row Hadamard
  alignment rule — which is why it is a small amount of code. It still needs a real
  multi-GPU run before it is believed.
- ~~**`bench/` does not gate it.**~~ *Closed 2026-08-24.* The gate now derives the
  checkpoint instead of pulling one: `bench/fixtures.py` runs
  `tools/quantize_embedding.py` on the already-pinned `MiniCPM5-1B-exl3@3.00bpw` (3.1s,
  byte-reproducible) and two `fast`-tier entries serve the result, eager and under CUDA
  graphs. The number that earns them is resident weight bytes — **0.79 GiB against 0.52**
  for the same checkpoint with and without the packed embedding, so a silent fall back to
  dense bf16 fails the gate while every logit stays correct. Deriving rather than
  publishing also puts the producer under the gate, which nothing else did. See
  [bench/README.md](../bench/README.md), "Fixtures".
- **Tied models are unchanged**, and remain served from their existing quantized `lm_head`.
  The shared-tensor optimization stays deferred on the same grounds as before: it needs an
  integer GEMM for the head role, and gemma-4 is nearly its whole constituency.

## On-load encoding, and paying for it once

*Added 2026-09-08, at the request of the appliance project (`vllm-untwisted`).*

On-load encoding, the default since 2026-11-01 (`EXL3_BLOCKQ_ON_LOAD=0` opts out), gets the saving above on a checkpoint nobody has repaired:
the dense `embed_tokens.weight` is encoded into `bq_*` while it loads, and the fp16
matrix is never resident. That removes the derived-artifact step entirely — no
`tools/quantize_embedding.py` run, no second copy of the checkpoint on disk, no
sidecar to keep in sync with the original — which is what makes it the right default
for a system that serves whatever HF checkpoints happen to exist.

Its cache is capped by `EXL3_BLOCKQ_CACHE_MAX_GB`, least recently used out, so a default-on
cache has an owner. Two constraints shape everything below. The encode runs **on CPU**, because `blockq.encode` does not
reproduce across devices — a GPU encode is equally accurate and produces a *different*
model from the same weights, which would mean a model served this way disagreed with
the same model quantized offline. And it runs **per engine start**, which is the cost
this section is about: about 0.6–1.6 s for MiniCPM5-1B's 130560x1536, and
proportionally more for a real vocabulary.

### What the default costs

"Under the noise floor" does not generalize. Simulated `blockq:32` 4-bit on the scored slices, as a
multiple of excess KL, against what the saved bytes would buy in body bits:

| model | wild 4.00 | swe 4.00 | wild 5.00 | swe 5.00 | saved | same bytes in body bits |
|---|---|---|---|---|---|---|
| Ornith-1.5-9B | 1.05 | 1.08 | 1.26 | 1.44 | 1.46 GB | ~6-9x lower |
| Qwen3.8-27B | 1.01 | 1.00 | 1.04 | 1.05 | 1.83 GB | ~2.3x lower |

The 9B's tax is about the size of its noise floor and 4-10x the +0.000225 measured on Qwen3.5-9B above;
why is open. It grows with the rung because a fixed embedding error weighs more against a cleaner body.
Text-answer recall is unchanged within CIs on both models (-0.64 to +0.05 points). So the case for the
default is economic: the bytes are worth far more in the body than the embedding error costs.

Served through vLLM, on-load matches a pre-quantized checkpoint exactly, |dlogprob| 0, at TP=1 and TP=2.

### Why that cost needed removing rather than reducing

Engine startup is not a one-time cost in the workload this is for. A profiling sweep
starts an engine per configuration — per bit rate, per KV dtype, per TP width — and
re-encodes the identical matrix every time, from identical bytes, on the CPU, while
the GPU sits idle. The work is a pure function of the dense tensor, so the answer is
not a faster encoder but a cache.

`~/.cache/vllm-exl3-plugin/blockq-embeddings/` holds one safetensors file per encoded
embedding, in the same three-tensor layout `tools/quantize_embedding.py` writes.
`EXL3_BLOCKQ_CACHE=0` turns it off; `EXL3_BLOCKQ_CACHE_DIR` moves it; deleting the
directory is always safe.

### What an entry is keyed on, and why each part had to be there

- **The resolved commit hash, not the revision that was asked for.** These differ in
  the case this project sees constantly: EXL3 repos publish one branch per bit rate,
  the thing being served is `turboderp/X-exl3@3.00bpw`, and that branch gets
  re-pushed when the quantization is redone. An entry keyed on the branch name would
  outlive the tensor it holds and hand a re-quantized checkpoint the previous
  embedding — silently, since a decoded embedding of the right shape produces
  plausible text. `hf_config._commit_hash` is what transformers actually resolved,
  and it is the only stable name for what is loaded. If no commit resolves, nothing
  is cached: the same refusal `_skip_hub_lookup` makes rather than substituting
  `"main"`.
- **The tensor's name.** One embedding per model is today's shape, not a property of
  the format — multimodal checkpoints already carry more than one embedding matrix,
  and the day this path serves a second one, a per-checkpoint key would hand it the
  first one's rows. Taken from the module prefix and recorded per method rather than
  read off `EXL3Config.embed_prefix`, which holds whichever embedding was constructed
  last.
- **Vocabulary, hidden size, block size and bit width.** Implied by the two above and
  in the key regardless, so that changing `BLOCKQ_BLOCK` or `BLOCKQ_BITS` orphans old
  entries instead of misreading them. The shapes are re-checked on read; the key is
  what makes a mismatch a miss rather than an error.

A **local directory** has no commit to key on, so it is identified by a digest of its
own files' names, sizes and mtimes. Weaker than a hash — `touch` invalidates, and a
rewrite preserving both size and mtime would not — but it is the property a local
checkout actually has, and it fails toward re-encoding rather than toward a wrong
matrix.

### Entries hold the whole vocabulary, and the reader slices

Encoding is row-independent (every reduction in `blockq.encode` is inside a row), so a
rank's shard of the encoding equals the encoding of its shard. Storing the full matrix
therefore makes one entry serve every tensor-parallel width, which is the arrangement
that matters here: sweeping TP=1,2,4 is one encode, not three, and each rank reads only
its own rows back out through safetensors' slicing rather than materializing the
vocabulary.

The cost is that a *cold* load at TP=N has every rank encode the full matrix to write
the same bytes. That is once per checkpoint, and it is why the **uncached** path still
encodes only the local rows — with nothing to amortize there is no reason to do the
other ranks' work.

### Measured

Qwen3.5-9B @4.00bpw, 248320x4096 embedding, one RTX 5070 Ti, `enforce_eager`:

| | model loading | resident weights |
|---|---|---|
| cold (encode + write the entry) | 9.99 s | 5.48 GiB |
| warm (read the entry) | 1.96 s | 5.48 GiB |

The encode was 5.8 s of the cold figure, so it accounts for most but not all of the
8.0 s difference; the rest is the OS page cache being warm on the second run, which
the cache gets credit for only by accident. The resident footprint being *identical*
is the check that matters — a cache that silently fell back to the dense embedding
would also look like a speedup.

Cold and warm serve **the same token ids** (`[11751, 13, 198, 32, 13, 2912, 198,
33]` greedy from the same prompt), checked as ids rather than as rendered text.
Same result on MiniCPM5-1B @3.00bpw, whose smaller 130560x1536 embedding encodes in
0.6–1.6 s.

### Nothing here is allowed to be fatal

A read-only or full `~/.cache`, a truncated entry, an entry written by a layout this
version does not understand: each is a miss and a log line, never a failure to serve.
An appliance that stopped loading models because a cache directory filled up would
have traded a startup cost for an outage.

## What llm-compressor's embedding quantization costs, and where its menu is a trap

*Measured 2026-08-24, after the amendment at the top of this note. Every embedding arm
swept above is **affine** — a (min, max) pair per row or per block. compressed-tensors is
**symmetric**: one scale per group, no min, codes over `[-8, 7]` at 4 bits. That is half
an affine arm's metadata at the same block size (16/N bpw against 32/N), so their default
recipe is 4.25 bpw where `blockq32` is 4.53. Whether the missing min is worth those bytes
was the one axis the sweeps never touched.*

### Method

A `sym:N` / `sym_row` granularity in qbench's `fake_quantize`, scored on the same three
models against the same cached references as every table above, so the numbers drop
straight into the existing frontier.

The arm is not a transcription: it is checked bit-identical against `compressed_tensors`'
own `calculate_qparams` + `fake_quantize` at 3/4/5/8 bits, group and channel, in
fp32/fp16/bf16 (`tools/ct_sym_check.py` in the qbench working dir; `llm-compressor`
itself is not needed, the quantization math lives in `compressed-tensors`, which is
already a vLLM dependency). Two details it would have been easy to get wrong by writing
from the README, both of which the check pins down:

- The scale is `max|v| / ((2^b - 1) / 2)` — denominator 7.5 at 4 bits, not 7. The most
  *positive* value in every group therefore rounds to 8, clips to 7, and comes back ~7%
  low, while the most negative one is exact. That asymmetry is their format.
- The arithmetic stays in the weight dtype. compressed-tensors stores `weight_scale` in
  `params_dtype` (vLLM allocates it that way), so promoting to fp32 the way the affine
  arms do would measure a scheme nobody serves.

`sym:N` is `strategy="group"`, `sym_row` is `strategy="channel"` — the two settings the
README offers.

### The measurements

Only the rows that bear on the comparison; the full tables are above.

**Qwen3.5-9B @4.00bpw** (baseline 0.013128, floor 0.001350) — again the only one of the
three with the dynamic range to resolve anything:

| bpw | scheme | tax | x floor |
|---|---|---|---|
| 3.2500 | **ct group-64 3-bit** | +0.011688 | **8.66** |
| 3.5156 | blockq32 3-bit | +0.004046 | 3.00 |
| 4.0039 | **ct channel 4-bit** | +0.243952 | **180.74** |
| 4.0083 | per-row 4-bit (affine) | +0.122225 | 90.55 |
| 4.2500 | **ct group-64 4-bit** (their default) | +0.004637 | **3.44** |
| 4.2500 | per-blk128 4-bit | +0.003407 | 2.52 |
| 4.2656 | blockq64 4-bit | +0.001567 | 1.16 |
| 4.5000 | **ct group-32 4-bit** | +0.003561 | **2.64** |
| 4.5156 | blockq32 4-bit | +0.000225 | 0.17 |
| 8.0039 | **ct channel 8-bit** | +0.000160 | **0.12** |

**MiniCPM5-1B @3.00bpw** (baseline 0.114097, floor 0.000977):

| bpw | scheme | tax | x floor |
|---|---|---|---|
| 4.0083 | per-row 4-bit (affine) | +0.003920 | 4.01 |
| 4.0104 | **ct channel 4-bit** | +0.006053 | **6.19** |
| 4.2500 | **ct group-64 4-bit** | +0.002903 | **2.97** |
| 4.2917 | blockq64 4-bit | +0.002649 | 2.71 |
| 4.5000 | **ct group-32 4-bit** | +0.002503 | **2.56** |
| 4.5000 | per-blk64 4-bit | +0.002526 | 2.59 |
| 4.5417 | blockq32 4-bit | +0.001435 | 1.47 |
| 8.0104 | **ct channel 8-bit** | +0.000043 | **0.04** |

**gemma-4-12B-it @4.00bpw** (baseline 0.026963, floor 0.001759) — every group arm sits
below its floor, as everything else here does:

| bpw | scheme | tax | x floor |
|---|---|---|---|
| 4.0042 | **ct channel 4-bit** | +0.049119 | **27.93** |
| 4.0083 | per-row 4-bit (affine) | +0.001872 | 1.06 |
| 4.2500 | **ct group-64 4-bit** | +0.000343 | **0.20** |
| 4.2667 | blockq64 4-bit | +0.000383 | 0.22 |
| 4.5000 | **ct group-32 4-bit** | +0.000141 | **0.08** |
| 4.5156 | blockq32 4-bit | +0.000298 | 0.17 |
| 8.0042 | **ct channel 8-bit** | +0.000174 | **0.10** |

### What it says

**1. `strategy: "channel"` is a trap for embeddings, and the README hands it over as a
neutral alternative.** It is worse than the affine per-row scheme this note already
rejected — 2.0x worse on Qwen3.5-9B, 1.5x on MiniCPM5-1B, **26x** on gemma-4-12B, which
is the one place gemma resolves anything at all. Symmetric makes the per-row failure mode
worse rather than better, because a row that needs an offset now cannot have one *and*
loses half its codes to the side of zero that has no mass. "Use `strategy: channel` (and
drop `group_size`) for per-row channel scales" is presented as a size/quality dial; at
4 bits on an embedding it is 28-181x the noise floor and it should not be offered.

**2. At the only resolving datapoint, symmetric costs ~16x at matched bytes.** Qwen3.5-9B,
group-32 4-bit (4.50 bpw) +0.003561 against `blockq32` (4.53 bpw) +0.000225. Their own
default, group-64 at 4.25 bpw, is 3.44x the floor where `blockq64` at 4.27 bpw is 1.16x.
So the missing min is not paid for by the bytes it saves.

**3. And the min outranks block resolution.** Still on Qwen3.5-9B: affine per-blk128 at
4.25 bpw (2.52x floor) beats symmetric group-32 at 4.50 bpw (2.64x). Blocks 4x coarser
and a quarter-bit cheaper, and it still wins — the offset is doing more work than the
scale granularity is.

**4. The direction is model-dependent, and gemma flips it.** There, group-32 symmetric
(0.08x floor) measures *better* than `blockq32` (0.17x) — but both are 6-12x below the
model's own noise floor, so the ordering is not resolvable and neither is a tax. MiniCPM
is a near-tie leaning `blockq` (1.47x vs 2.56x). Only Qwen3.5-9B separates them, and it
separates them by 16x. That is the same shape as every other result in this note: the
model with the sensitive embedding is the one that decides, and it decides for the affine
layout.

**5. Their conservative setting is honest, and `blockq` claims it at 57% of the bits.**
"W8 channel, effectively lossless" holds on all three — 0.12x / 0.04x / 0.10x of the
floor on Qwen3.5-9B / MiniCPM5-1B / gemma-4-12B. It costs 8 bpw, which is still half of
fp16 and would fix most of the tax measured at the top of this note; a checkpoint
published that way would be a real improvement over what everyone ships today. But the
same standard applied to `blockq32` at 4 bits reaches it on two of the three at 4.53 bpw
(0.17x on Qwen3.5-9B, 0.17x on gemma-4-12B, both unresolvable against the floor), and on
all three at 5 bits / 5.54 bpw (0.26x on MiniCPM5-1B, the one model where 4 bits is
resolvable at 1.47x). MiniCPM is the same holdout the flat-4-bit rule already carries at
1.49x, so this adds no new exception — "effectively lossless" is available at 57% of
their bits on the models where the phrase means anything, and at 69% everywhere.

**6. Their accuracy evidence cannot see any of this.** The README validates on
`pythia-1.4b` and reports W4 group-64 at 14.752 wikitext ppl against a 14.733 baseline.
That is the gemma case: a model whose embedding sensitivity is below what the measurement
can resolve, reported as "near-lossless" without a noise floor to compare against. It is
not wrong about pythia; it is evidence about one model in the class that cannot
distinguish schemes, generalized to a default.

### Consequences

**Nothing changes in the format.** `blockq32` stays as measured — this is the third
alternative layout scored against it (GGUF k-quants, k-shape, now compressed-tensors) and
the first two tied where this one loses on the model that resolves.

**It does change what is worth saying upstream.** Finding 1 is not about us: it is a
default in a widely used tool that damages any large-vocabulary model with a sensitive
embedding, it reproduces on three models, and it does not depend on adopting anything of
ours. Finding 2 is the argument for an affine group scheme in compressed-tensors, but it
needs a format change on their side (their `weight_zero_point` exists for linears, and
their embedding kernel simply does not read one) and would still be 0.5 bpw of metadata
at block 32 without the double-quantized scales that make `blockq` cheap. The first is
worth reporting on its own; the second is only worth raising if someone wants it.

**The load-time idea this came from survives.** Their embedding recipe being data-free is
the load-relevant fact — no calibration, so the same is available to any bf16 checkpoint
at startup — and nothing measured here argues against doing it, only against doing it
symmetrically. That remains unbuilt and untracked.

## Somebody else picked the same design point

*Found 2026-08-25. Not a measurement of ours — a convergence worth recording, because it
is the strongest available evidence that the sweep landed somewhere sensible.*

`manjunathshiva/Muse-Glimmer-30B-tq3-g64` is an MLX quantization of a checkpoint this
project also serves. Unservable here, and that is beside the point. Its embedding:

```
embed_tokens.weight  [202048, 832] U32     packed 4-bit
embed_tokens.scales  [202048, 104] BF16    group of 64
embed_tokens.biases  [202048, 104] BF16    group of 64   <- affine, not symmetric

202048 x 6656 = 1344.8M params in 721.4 MiB  ->  4.5000 bpw
```

That is **exactly the `per-blk64 4-bit` arm** swept above: affine, group 64, one scale
and one offset per group, 4 + 32/64 = 4.50 bpw. It measured +0.000970 (0.72x the noise
floor) on Qwen3.5-9B. `blockq32` sits beside it at 4.5312.

Three things follow:

- **The design point is not idiosyncratic.** An unrelated practitioner, in a different
  runtime, with a different body quantizer, chose the same scheme at the same bit rate
  for the same tensor. The sweep's conclusion — block-scaled affine at ~4.5 bpw — is
  where independent work converges.
- **Affine, not symmetric.** They store a `biases` tensor per group. This is the
  distinction llm-compressor's embedding path cannot express and vLLM's embedding kernel
  cannot read, measured above at 2.64x floor against 0.17x. What ships in the wild when
  the format permits it is the affine one.
- **Extras are a first-class concept there.** The config key is literally
  `affine_extras: {bits: 4, group_size: 64}`, applied to embedding, head *and* vision
  tower while the body runs turboquant 3-bit. That is `config_groups` by another name,
  and it is what makes the compressed-tensors route (see [upstream.md](upstream.md)) the
  right shape rather than a compromise.

`blockq32` remains the better encoding at essentially the same bytes — its
double-quantized scales buy block-32 granularity for the metadata cost of block-64 — but
"better than the thing serious people independently ship" is a much more useful claim
than "better than the thing nobody ships."

## blockq on non-EXL3 checkpoints: the quality answer is already in

*Asked 2026-08-25. Splits into two questions with very different answers.*

**Quality generalizes for free, because the tensor is the same.** An embedding is dense
in every format — the encoder census makes the same point for vision towers — and it is
literally the same matrix:

| checkpoint | `embed_tokens.weight` |
|---|---|
| `Qwen/Qwen3.5-9B` (bf16 base) | `[248320, 4096]` BF16 |
| `cyankiwi/Qwen3.5-9B-AWQ-4bit` | `[248320, 4096]` BF16 |
| `turboderp/Qwen3.5-9B-exl3@4.00bpw` | `[248320, 4096]` F16 |

EXL3's is a cast, not a requantization. So every blockq measurement in this note was
taken on a tensor an AWQ or GPTQ checkpoint of the same base also carries, byte for byte.
There is nothing to re-measure: `blockq32` at 4 bits costs what it costs, whatever
quantized the *body*.

**Serving does not generalize, and the obstacle is structural.** vLLM resolves one
`quant_config` per model, and `get_quant_method` dispatches from it per module. Serving a
blockq embedding beside AWQ weights therefore needs one of:

- the other format's config to learn blockq (i.e. this becomes a compressed-tensors-style
  upstream feature, not a plugin);
- a delegating wrapper config, which nothing in vLLM composes today; or
- **no stored format at all** — quantize the dense embedding in memory at load.

The third is why the JIT-at-load framing is the right shape for the cross-format case
rather than an optimisation of the checkpoint one. It needs no new checkpoint format, no
config composition, and no agreement from any other project: it is a hook that replaces a
dense `VocabParallelEmbedding` with a packed one during loading. It is also, as noted
when the idea came up, a separate project from this plugin — the audience is anyone
serving a large-vocabulary model, not anyone serving EXL3.

It still depends on `vllm-embed-quant-config` (86 of 131 architectures never pass
`quant_config` at all), which is the same blocker in the same place. See
[upstream.md](upstream.md).

**There is a fourth option, and it is better than the three above.** compressed-tensors
is *itself* a composing config: one `quant_config` that dispatches per `config_groups`,
with non-uniform recipes a documented and supported feature — mixed precisions, mixed
strategies, even different modifiers per module family, running directly in vLLM. An
embedding scheme expressed there therefore composes with any weight quantization, on any
architecture llm-compressor supports, through a serving path that already exists.

What stands in the way is not composition but *quality*: their embedding kernel is
symmetric-only, which this note measured at 2.64x the noise floor against `blockq32`'s
0.17x. The enabling change is small and is not ours — teach
`CompressedTensorsEmbeddingWNA16Int` the zero point its linear sibling already reads.
Asymmetric group-64 would be ~4.31 bpw, *cheaper* than `blockq32`, and in the affine
league the per-block arms above measured. Filed in [upstream.md](upstream.md) as the
highest-value llm-compressor item.

That reorders the cross-format question: the route is not "carry blockq to other
formats", it is "make the format everyone already uses good enough that carrying is
unnecessary".

## The axis nobody has swept: vector quantization

*Speculative, and deliberately marked so — nothing below is measured, unlike the rest of
this note. Recorded because the last four design reversals all came from adding one arm
to an existing sweep, and this is the arm that is conspicuously missing.*

Every scheme compared here is **scalar**: uniform grids at various granularities (per
tensor, per row, per block of 32/64/128, k-quant-shaped), plus llama.cpp's IQ types,
which are non-uniform *scalar* codebooks with block scales. What has never been on the
axis is **vector quantization** — split a row into subvectors, fit a codebook, store an
index per subvector. Product quantization, in the FAISS sense.

Three things make it the plausible next winner rather than an idle suggestion:

- **Its objective is the one this job actually has.** A codebook fitted over rows
  minimizes reconstruction error of rows *as vectors*, which is the criterion the whole
  note turns on. Every scalar scheme approximates that criterion with a grid; this one
  optimizes it directly.
- **The codebook is free at this scale.** The thing that usually makes VQ awkward — a
  codebook to store and amortize — costs nothing against a 130k-262k row matrix, where
  it is rounding error on the total.
- **Row independence survives.** A lookup stays a gather of codes plus codebook
  indexing, so the serving path, the slice-ability and the vocabulary-parallel TP story
  are unchanged. It would slot into the same three-tensor shape.

**What the prize actually is, and it is not "beat blockq at 4.5 bpw".** There is nothing
left to win there: the tax is already at or below the noise floor on all three models,
so a better encoder buys unmeasurable quality. The prize is **the low end** — making 2.5
to 3 bpw usable, where blockq's tax is resolvable and where an appliance running an
aggressive configuration would spend the saving on layer bits or KV cache.

**What would have to be true for it to be worth it.** Two costs land squarely on things
this project just escaped. The encoder stops being closed-form arithmetic and becomes a
fitting procedure (k-means over a vocabulary-sized matrix), and the codebook is
model-specific, which reintroduces exactly the per-model calibration step the
constant-depth result removed. So the bar is not "wins" but "wins enough at 2.5-3 bpw to
justify a fitted encoder", and it should be held to that.

**Cost to find out: one `embed_quant` granularity and a re-run.** The harness, the
reference logits, the three models and every comparison arm already exist; simulating a
PQ scheme needs no storage format and no serving path, exactly as `blockq:32` needed
none before it earned one.

## The most extreme case found: gemma-4 E2B, 54% embeddings

Measured 2026-08-19 against `google/gemma-4-E2B-it` (BF16, 9.54 GiB).

E2B and E4B carry *two* embedding tensors. The usual `embed_tokens` is
[262144, 1536] at 0.750 GiB; alongside it sits `embed_tokens_per_layer`,
[262144, 8960] — 35 layers × 256, one conditioning vector per layer per token —
at 4.375 GiB.

| subtree | size | share |
|---|---|---|
| `embed_tokens_per_layer` | 4.375 GiB | 45.8% |
| language model (rest) | 3.455 GiB | 36.2% |
| `embed_tokens` | 0.750 GiB | 7.9% |
| audio tower | 0.568 GiB | 5.9% |
| vision tower | 0.312 GiB | 3.3% |
| per-layer input plumbing | 0.077 GiB | 0.8% |
| **embeddings, both** | **5.125 GiB** | **53.7%** |

The model is tied (`tie_word_embeddings: true`, and no `lm_head` tensor in the
checkpoint), so the main embedding is free from the quantized head. The per-layer
tensor has no tied counterpart and no quantized copy anywhere — exactly the case
`blockq` exists for, and 85% of the embedding bytes.

**Why this is the sharpest statement of the tax.** Quantizing only what EXL3
quantizes today leaves both embeddings at BF16, so a 4bpw E2B lands near 6.95 GiB
against 9.54 BF16: a 27% saving on a model sold as 2B. Serving the tied embedding
from the head and the per-layer tensor at blockq 4-bit projects ~3.3 GiB — 2.1×
better than the naive conversion, 2.9× against BF16. Qwen3.5-9B's 6.72 -> 5.36 GiB
is the same argument at a tenth the amplitude.

**Two things the projection assumes and has not measured.** The 4-bit constant was
established on standard token embeddings, whose rows feed the residual stream; the
per-layer tensor is consumed inside each block as conditioning — a different
functional role carrying 85% of the bytes, so it wants its own depth sweep rather
than the inherited constant. And blockq's 32-wide blocks divide the 256-wide
per-layer slice exactly, so no block straddles a layer boundary; that is luck, not
design, and would not hold for a per-layer width that is not a multiple of 32.

None of it is measurable end to end yet — exllamav3 cannot convert the
architecture (FUTURE.md `gemma4-e2b`).


## Moved from TODO.md (2026-11-05)

*The working record of these items as they stood when TODO.md was cut back to open tasks, kept verbatim. Open work on them, if any, is in TODO.md or FUTURE.md under the same slug or the one named.*

### `repair-tool` — Repair tool for existing EXL3 checkpoints

Every existing EXL3 checkpoint carries two packaging choices that cost far more in a
GPU-resident server than they do upstream, enough to more than erase its efficiency
advantage against other formats on total bytes: a separate output head is emitted for
tied models (redundant once the embedding is quantized, and not small), and the
embedding is left at full resolution. Both are rational for exllamav3, which keeps the
embedding in system RAM and counts it against neither VRAM nor the published size --
see [docs/embeddings.md](embeddings.md). A post-processing tool
preserves the very large investment in computation the published EXL3 collection
represents, rather than requiring it be redone.

**Half of this now exists.** `tools/quantize_embedding.py` rewrites one checkpoint's
embedding into the block-scaled 4-bit format the measurements chose, hardlinking every
shard it does not touch — 12 seconds and 1.36 GiB saved on Qwen3.5-9B. What it does
*not* do is the rest of a repair tool:

- **Drop a tied model's redundant `lm_head`.** The other pipeline mistake, untouched.
  The plugin already ignores those bytes at load, so this is purely a file-size fix —
  which is exactly what a downloader cares about.
- **Take a Hub repo rather than a local directory**, and write something publishable:
  a model card noting what changed, and the revision it was derived from.
- **Batch**, since the value is in repairing a collection, not one checkpoint.

Depth stays a shipped constant (4 bits) unless something argues otherwise; an override
is worth having, and a size-budget solver belongs to the full quantizer, where layer
bpw is actually free rather than fixed.

→ [docs/embeddings.md](embeddings.md)

### `quantized-embeddings` — VRAM-efficient use of quantized embeddings

Serving the embedding quantized rather than dequantizing it at load. Dequantizing at
load proves the math and saves file storage and I/O, but leaves VRAM — the thing
this project cares about most — completely unchanged.

**Both shapes now serve, and an untied model no longer needs a derived checkpoint at
all.** Tied models come from the checkpoint's existing quantized `lm_head`, with the
fp16 `embed_tokens` never loaded and no tooling needed. Untied models — which have no
quantized copy to reuse — get one either from `tools/quantize_embedding.py`, whose
`--sidecar` mode adds `bq_*` beside the original shards instead of rewriting one, or
from `EXL3_BLOCKQ_ON_LOAD=1`, which encodes the dense embedding during weight loading
and reproduces the tool's output byte for byte. Either way it is the block-scaled
4-bit format of `vllm_exl3_plugin/blockq.py`, served by `EXL3BlockQEmbeddingMethod`:
Qwen3.5-9B goes 1940 -> 549 MiB resident, at a KLD tax at or below the model's own
noise floor. All hooks are sanctioned vLLM extension points; nothing is monkeypatched.

Three things remain, in rough order of how much they would cost to discover late.

1. **Most architectures never ask for an embedding quant method**, and the fix is
   carried in `vllm-embed-quant-config` (fork commit `c69d01e`) rather than
   upstream. 86 of 131
   vLLM model files omit `quant_config` when constructing their
   `VocabParallelEmbedding`, so neither shape can serve there — silently dense for a
   tied model, a load failure for a block-quantized one. The patch is one file
   (default the config from `get_current_vllm_config()`), verified not to disturb
   configs that do not quantize embeddings.
   **Worth offering upstream**: `vllm-gguf-plugin` is blocked by exactly the same
   thing, so it is an ecosystem fix and not only ours.

   **But not as written — it breaks speculative decoding** (found 2026-08-20). A
   drafter is built under the *target's* `quant_config`, so the patch hands the
   drafter's embedding an EXL3 method and it is asked for a `bq_q` it never had.
   Nothing about that is EXL3-specific; filed as-is it breaks any quantized target
   with a differently quantized or unquantized drafter. Two candidate fixes, and the
   choice decides what gets filed: condition the ambient default on the module
   belonging to the model the config describes (which
   `VocabParallelEmbedding.__init__` cannot know), or fix the drafter's config
   instead so it stops misdescribing what is being built — a cleaner contribution
   than an 86-file workaround, and it makes this patch safe as a side effect.
   The same ad-hoc plumbing fails in the opposite direction too
   (`DFlashQwen3Model.fc`, gemma-4's `vision_adapter.c_fc`): any fix worth filing
   should be judged against both. →
   [docs/format-and-loading.md](format-and-loading.md) "Ambient `quant_config`"

2. **Tensor parallelism is written but unproven.** All three stored tensors slice on
   dim 0, so `tp.ROLE_VOCAB` is a row slice with none of the trellis path's 128-row
   Hadamard alignment rule — which is why it is a handful of lines. It has never run
   on more than one GPU. Needs the `vast` box, alongside `moe-tp` and the TP tier of
   `bench-suite`.
3. **Only 4 bits is packed.** That is deliberate — one depth covers every model
   measured, and nibbles keep both ends byte-aligned — but 3 bits is usable at ~3.5
   bpw and would want the packing if a checkpoint ever calls for it.

**A tied model with a block-quantized embedding crashes** — **FIXED 2026-08-26**
(found 2026-08-19, observed 2026-08-25). The two predicates in
`quantization/config.py` treated the cases as mutually exclusive —
`embedding_is_blockq()` said so in as many words — but they answer questions
about *different modules*: one asks whether a tied model's `lm_head.*` is being
renamed onto the embedding, the other whether the embedding has block-quantized
tensors of its own. A repaired tied checkpoint makes both true and both
load-bearing.

Two silent failures came out of that, and both are gone:

- `EXL3TiedLMHeadMethod` read a trellis off the embedding module, which now held
  `bq_*`. Died late, at logits time.
- The blockq branch returned *before* `self.embed_prefix = prefix`, so the prefix
  stayed at its `"model.embed_tokens"` default while the rename still fired —
  routing 755 MiB of trellis to a path a nested model does not have, dropping it
  without complaint, and serving garbage.

The fix is `EXL3BlockQTiedEmbeddingMethod`: both parameter sets on the one
module — which is the existing design, since vLLM skips a tied model's
`lm_head.*` and those weights are already renamed onto the embedding for the head
to borrow back — with the lookup on `bq_*` and the logits on the trellis.
`embed_prefix` is now recorded before any branch. **No vLLM patch and no metadata
change**: `tie_word_embeddings` stays `true`, which it is.

`apply()` is overridden explicitly rather than inherited, and that was not
cosmetic. The MRO puts the blockq method first (it must, for the gather) and its
`apply` is a stub raising "no matmul path" — right for an untied embedding, wrong
here. The gemma4-style two-module shape never notices because its head reaches
the trellis through its own method; **the Qwen3-style one-module shape would have
raised on the first token.** Caught by an MRO assertion in
`tests/test_tied_blockq_routing.py`, not by running a model.

Verified on a repaired tied `gemma-4-12B-it-exl3@3.00bpw_mul1`: loads clean,
answers correctly, holds ~0.53 GiB more weight than the unrepaired baseline (KV
cache 19,176 → 17,545 tokens) which is `bq_*` loading beside the trellis, and 150
of 157 prompt logprobs differ from that baseline (mean |Δ| 0.122) — so the gather
is executing, not merely allocated. Full suite 118 pass.

`tools/quantize_embedding.py` now *supports* tied checkpoints rather than
producing broken ones, and says at creation time that the output needs a plugin
with this method — an older one loads it clean and serves garbage, which nothing
downstream can detect.

**Still open on this path**: tensor parallelism is unproven (both sets shard on
dim 0, but the combination has never run on more than one GPU), and the
Qwen3-style one-module tied shape is covered only by the MRO test, not
end-to-end. Neither blocks the gemma-4 case.

**The shared tied-model tensor's kernel blocker is gone** (2026-08-26). One tensor
serving both roles needed a scalar-integer GEMM for the head, which exists nowhere.
fp8-e4m3 clears that — `torch._scaled_mm` is a primitive, not a kernel project — and it
is good enough: **+0.000669** against native, 0.38x the noise floor, better than the
7-bit per-row point that was the recorded tied operating point. Per-channel scaling is
required; per-tensor is 1.6x worse for nothing. So "7 or 8 bit shared per-row" is
**superseded on encoding** — fp8 is the shape to build if it is built.

**It stays deferred, and the ordering that deferred it has now resolved.** The crash
above landed first, on correctness grounds rather than because any model demanded it,
and that changed the baseline this has to be measured against. The blockq split is a
real tied-model baseline as of 2026-08-26: **1.234 GiB at +0.000297**, serving gemma-4
from `bq_*` for the lookup and the trellis for the logits.

So fp8's margin is the marginal one after all: **0.296 GiB for 2.25x the divergence**,
not the 1.641 GiB it looked like while the split was unbuilt. That margin is still
worth something — 0.296 GiB is real in a VRAM-bound appliance, and 2.25x of 0.38x the
noise floor is still under the floor — but it is now an optimization on top of a
working path rather than a route to one, and it needs an fp8 head path, an fp8 gather
and repair-tool emission to collect it.

The constituency is no longer hypothetical, which is what keeps this open at all: the
demotion gemma and every tied-model optimization inherited came from the
head-dim-512 misdiagnosis, corrected under `turboquant-sliding-window`.
**Revisit if a tied model ever needs that last 5%**, or if fp8 emission turns out to be
nearly free alongside other repair-tool work.

**Upstream's trellis embedding (exllamav3 1.6.4: `convert.py -eb`, `convert_embedding`, a fused
gather) needs a plugin loader whether or not we adopt it for tied models.** If upstream-direction
quants move to it, they stop carrying a dense table, so on-load blockq has nothing to encode, and
fetching a reference embedding is no clean swap: the rest of such a model was calibrated against the
quantized table. It stores `<key>.trellis` and `<key>.signs`.

**It beats blockq on quality per bit** (simulated on Ornith-1.5-9B: real encode and `dequant_rows`, swapped
into existing quants by qbench's `embed_file`, so later layers are not calibrated against it). K=4 is 4.06 bpw
with table error 0.066 against blockq's 4.52 and 0.078. Excess KL multiples, wild/swe: at 4.00, K=4 1.03/1.07
against blockq 1.05/1.08; at 5.00, K=4 1.17/1.35, K=5 1.03/1.10, K=6 1.01/1.01, against blockq 1.26/1.44.
Text-answer recall is within CIs of dense at K=4 on 4.00 and K=5 on 5.00. K=4 on 5.00 is the exception at -1.70
points [-2.64, -0.88], worse than blockq's -0.64 despite better KL, unexplained. Depth is a free parameter
of the format, so this needs no 5-bit blockq. Encoding is about 45 s on a GPU for 248320x4096, and reproduces
bit for bit across runs, chunk sizes, and a 5060 Ti against a 5070 Ti. Gather ties blockq at chunk sizes on a
synthetic table, uncompiled. Still unmeasured: the 27B, a true `-eb` conversion, the served path. The reproducibility result is GPU against
GPU only: blockq's own problem was CPU against GPU, so the two are not compared like for like.

**Decided 2026-11-01: the pipeline adopts trellis by default; on-load stays blockq.** Start the ladder like the
head's: K=4 base, K=5 at 5 bpw body and above. exllamav3 reads the format natively, so a published quant needs
no dense copy, which is what ruled out emitting `bq_*`. On-load does not move to trellis without a much larger
difference than this: it would tie the serving arm to the encoder and add about 45 s of GPU to a first load,
and blockq remains a high-quality format that is easy to produce on the fly.

**The plugin loader is in** (`vllm_exl3_plugin/trellis_embed.py`, `EXL3TrellisEmbeddingMethod`): plain torch,
bit-exact against `ngram_dequant` at K=1..8. Served against a dense twin holding the same decoded table, eager
is identical at TP=1 and TP=2. Under CUDA graphs it differs by up to 0.72 on tail logprobs, repeatably:
compiled `gather` differs from the kernel in about 6% of elements by 1-2 fp16 ULPs, with rows following ids, and
blockq's own eager-to-graphs gap is the same size. Compiled it is about 25x slower than blockq at 8192 tokens
(6.5 ms against 0.26), which does not reach a served engine: decode 2409 tok/s against 2418, prefill 22021
against 22097 on MiniCPM5-1B, inside run spread. Five `trellis` bench entries are in; `quant.py` passes
the ladder by default (`--embed-bits`, 16 keeps it dense).

**A true conversion matches the simulation** (shipping recipe, YAQA and calibration mix; conversion does calibrate
through the quantized table). Ornith-1.5-9B: 7.31 to 5.79 GB at 4.00, 8.17 to 6.78 GB at 5.00. Excess KL 1.05/1.04
and 1.05/1.11 wild/swe, against 1.03/1.07 and 1.03/1.10 simulated. Text-answer recall -1.06 points [-2.70, +0.47]
and +0.06 [-1.17, +1.35]. MiniCPM5-2B: 1.73 to 1.33 GB and 1.98 to 1.61 GB. Both serve through the plugin.
Qwen3.8-27B at 5.00: 19.90 to 18.16 GB, excess KL 1.01/0.95 wild/swe, against blockq's 1.04/1.05, text-answer recall
+0.88 [-0.10, +1.90]: no cost at all on a model whose embedding tax was already near nil.

**Reconverting a rung can move recall by 4 points, with no embedding involved.** MiniCPM5-2B 5.00 dense, from the
same measurement, reconverted here: text facts 91.5 to 87.4, -4.16 [-6.93, -1.39], against that control `-eb` is
+0.46 [-2.47, +3.24], and a K=5 table on the original body is +0.00. **Conversion is deterministic per host and
differs between hosts:** same inputs and converter tree on a 5060 Ti, 5070 Ti and RTX 3060 give byte-identical repeats
and three different quants (885 of 1266 tensors, 97% of trellis words), and the 3060 reproduces the original exactly.
Same-generation cards differ, so it is not architecture alone. `EXL3_SEED_IDX_OFFSET` selects a rounding draw, for
sampling this spread on one host. Compare recall only within one host, and across draws for small effects.

**Measured over seven equivalent MiniCPM5-2B 5.00 draws:** excess KL varies 0.6% wild, 2.1% swe. Text-facts retention
averages 91.3 with sd 1.72 and range 5.2 (640 questions), PopQA 95.7 with sd 0.47. A quarter of the facts flip between
draws, and the paired CIs in this section cover question sampling, not this. Requiring the reference to hold the answer
at P >= 0.7 takes sd to 0.13 and flips to 0.3%, and the 3.00 rung still reads 87.9, so it keeps seeing real damage; it
saturates at 100 on a good 5.00 rung, so it detects damage and does not rank good quants.

**Recall rows in the calibration help at 3 bpw (MiniCPM5-2B, 2026-11-03).** 15% of the 250 calibration rows replaced by
multi-turn closed-book chats, from questions outside the scored slice (no overlap), no YAQA, one host. Over three seed
draws each at 3.00: kept 74.6 to 77.8 on the slice, and 76.3 to 81.0 (about 2.5 standard errors) on a reworded slice whose
instruction moves into a different system prompt, so it is not the eval's phrasing being learned. Swe excess KL +5.2%,
wild +0.5%; confident errors show no clear change. At 4.00 it does nearly nothing (93.0 / 93.3 / 94.0 kept on the reworded
slice at 0 / 15 / 30%). Calibration seems to shape which of the model's own knowledge survives rounding, not only fit it to
what it will see. Qwen3.8-27B barely responds: +1.08 kept at 2.50 and +0.65 at 3.00, single draws, with swe moving -7% and +20%, so its
ladder ships on the current calibration. Recall calibration stays a lead for small models. The 27B's 2.50 keeps 82.8 with
2.35% confident errors, against 88.6 and 1.27% at 3.00.
A slice without any answer instruction does not work as a format check: the reference answers in sentences and knows 27 of 888.

**The card reports kept, confident errors, and kept by band.** Kept is sum min(P, Q) / sum P: of the correct answers the
reference would give under sampling, the share the quant still gives, per question, so a gain cannot pay for a loss (a plain
sum ratio rated the 27B's 3 bpw above 4 bpw). It stays close to binary retention, and across seven 2B draws with hosts its sd is
0.75 against 1.72. Confident errors are the share of known questions where the quant is at P >= 0.7 on a wrong token: a low
answer probability cannot tell a shrug from a false statement. Bands of reference probability: sure (>= 0.9), firm (0.7-0.9),
leaning, shaky, barely (< 0.3). MiniCPM5-2B at 3 / 4 / 5 bpw: kept 75.9 / 89.5 / 95.6, confident errors 4.87 / 1.62 / 0.35%.
Draw noise at its 3.00: kept sd 2.54, confident errors sd 1.06, band retention at most 2.0x binomial.

| Qwen3.8-27B, 3059 questions | served GB | kept | confident errors |
|---|---|---|---|
| 4.00 one-sided | 14.12 | 96.6 | 0.33% |
| `UD-Q4_K_M` | 16.46 | 93.4 | 0.46% |
| plain `Q4_K_M` | 16.81 | 92.7 | 0.78% |
| 5.00 `-eb` | 17.25 | 96.7 | 0.13% |
| `UD-Q5_K_M` | 19.77 | 95.6 | 0.75% |

YAQA at 4.00 costs kept (92.7 against 96.6) and not confident errors (0.20 against 0.33%): it weakens right answers rather
than replacing them, which is a shrug and not a hallucination. `quant.py recall-slice`
scores the pool (Wikidata plus PopQA, 19,684 questions) with one reference pass, then `eval/recall_select.py` keeps each
model's known non-numeric questions, with PopQA at most 33% and one relation at most 25%, capped apart above and below P 0.7.
It refuses under 300 confident questions before any arm is scored (`!!! RECALL SLICE TOO THIN`, exit 3, `--allow-thin`
overrides). `qbench --slices recall` scores it, and qbench gives `recall_bands`. MiniCPM5-2B: 888 questions, 414 confident,
7 minutes for the pass. Not in the composite. The card shows kept, confident errors and a chart by band with `quant.py`'s
`recall_section`, and shows none of it on a thin slice. Chart bars are Wilson intervals widened 2x: draw noise reached 2.04x
binomial in the worst band, over the 2B at 3.00 and 5.00 and the 9B at 5.00. Confident-error noise is measured on the 2B only.

**It does not serve a tied head.** The earlier hope here, one tensor for head and gather, does not match the
format: rows are rotated by random signs and a Hadamard, built for row accuracy, with no GEMM path. A tied
model carries it beside the head's trellis, the same trade as `EXL3BlockQTiedEmbeddingMethod`, which is the accepted
behaviour until something serves both from one tensor. fp8 was the earlier idea for that, untested.

→ [docs/embeddings.md](embeddings.md)

### `quantize-embeddings-pipeline` — Quantizing embeddings in the pipeline

Stop shipping a full fp16 embedding, and stop emitting a redundant head for tied
models, at conversion time rather than as a repair pass. Forward-looking: it fixes
what gets published from here on, where `repair-tool` rescues what already exists.

**Not "do for every model what is already done for tied models".** That framing
predates the Phase 4 measurements and is wrong — it would emit a trellis, which is
the wrong encoding for an embedding by roughly two orders of magnitude. The
objective mismatch is the reason: the trellis optimizes `x @ W.T` against typical
activations, which is what a *head* needs, while an embedding needs each individual
row accurate as a vector. What to emit instead:

- **Tied models**: one shared block-scaled integer tensor serving both roles.
  Dominates both the shared trellis and any two-tensor split. Depth is set by the
  head, which is ~60x more bit-sensitive; the embedding rides along above its own
  requirement, and that waste is still far cheaper than a second copy of the matrix.
- **Untied models**: a block-scaled integer embedding at a lower depth, alongside the
  trellis head exactly as produced today. The head is already right; only the
  embedding changes.

Embedding depth is no longer per-model: 4 bits (4.5 bpw in the block-scaled layout)
covers every model measured. The head's depth in a shared tensor is a separate
question, and is now **answered**: a budget-neutral sweep on phi-4-mini puts the head
optimum at 5-6 bits, with 7 costing +34% KLD and 8 costing +87% — see
[docs/qbench.md](qbench.md), "Head bitrate: 6 is defensible". Since the shared
tensor's depth *is* the head's, that prices the tied-model plan directly: 5-6, not
higher.

**Not urgent: `repair-tool`'s post-processing covers this today.** All this item adds
is that checkpoints built here would be right from the start instead of needing a pass.
Strictly as a quantization problem the embedding is the least interesting one on the
map — a lookup with a distribution-independent optimum that measurement has already
pinned at a flat 4 bits across every model tested. There is no allocation question here
worth solving; depth is a constant.

**The format question is no longer blocked on upstream.** It previously waited on the
bar for proposing a storage format to exllamav3, which is higher than "serves this
plugin's targets", plus a wrinkle: the recipe is a flat `{tensor: bits}` map, so an
embedding needing a different *encoding* rather than a different depth cannot be
expressed in it. Now that checkpoints are built here rather than consumed, that bar is
just "does it work for us", and the wrinkle is nobody else's design decision to wait on.

**And the serving path is no longer a blocker either — for untied models it is done.**
`blockq` is specified ([docs/blockq-format.md](blockq-format.md)), emitted by
`tools/quantize_embedding.py`, served by `EXL3BlockQEmbeddingMethod`, and gated by three
`bench/` entries (eager, CUDA graphs, and throughput) that derive a fixture on every run
and held at exactly 0.0 across both the v0.28.0 and exllamav3 v1.4.3 bumps. So for the
untied case nothing is missing but doing it at conversion time instead of as a repair
pass, and `repair-tool` already covers the interim.

**The solver is not part of this item, and as of 2026-08-25 is not part of any.**
Embedding depth is fixed at 4, the head measures out at 5-6, and body tensors cannot be
allocated independently at all. Two scalars is a lookup table, not a search space — see
[docs/qbench.md](qbench.md).

**What is still blocked is the tied half, and on a different thing than this item used
to say.** It wanted "one shared block-scaled integer tensor serving both roles", and
`blockq` deliberately has no matmul path — decoding to dense fp16 to multiply would
return the whole saving. That needs a scalar-integer GEMM for the head role, which does
not exist; see `quantized-embeddings`. Tied models therefore continue to be served from
their existing quantized `lm_head`, which is already shipped and costs nothing new.

**The v1.4.3 bump changes nothing here.** Upstream's "new optimization pipeline" was
tagged at `2398c05`, the same commit the 2026-08-23 allocation study was run against, so
the pipeline this item would build on is the one already measured — see
[docs/qbench.md](qbench.md).

→ [docs/embeddings.md](embeddings.md)

### `blockq-on-load` — Default-on since 2026-11-01; baselines still to re-bless

**Decided 2026-11-01: on-load blockq becomes the default, with an LRU cap on its cache** (done:
`EXL3_BLOCKQ_CACHE_MAX_GB`, 16 by default). The cap answers the old blocker, that nothing owned eviction.
Not emitting `bq_*` at conversion: without the dense table a quant is unusable by exllamav3, and with it the
quant is much larger. A checkpoint with a trellis embedding is refused by name
(`format.check_embedding_storage`). **Flipped 2026-11-01** after the served-path soak on `bench/`: flag off
matches every baseline; on-load equals the pre-quantized fixture exactly at TP=1 and TP=2; MoE's TP spread is
no wider than dense. The soak found one bug, fixed: a backend-rebased embedding kept its dense `weight` beside
`bq_*` (Muse-Glimmer, +0.71 GiB where it now saves 1.80). **Still to do: re-bless the untied baselines**, which
needs a committed tree. The cost table below is simulation.

**Judge embedding damage by economics, not an absolute threshold:** weigh it against what the same bytes buy
elsewhere (body bits, trellis embeddings, deeper blockq). BF16 is the reference, not a candidate default: it is
unlikely to be right for any configuration that wants EXL3.

| excess KL multiple, simulated 4-bit blockq | wild 4.00 | swe 4.00 | wild 5.00 | swe 5.00 |
|---|---|---|---|---|
| Ornith-1.5-9B (saves 1.46 GB, worth ~6-9x in body bits) | 1.05 | 1.08 | 1.26 | 1.44 |
| Qwen3.8-27B (saves 1.83 GB, worth ~2.3x in body bits) | 1.01 | 1.00 | 1.04 | 1.05 |

Body-bit worth is log-interpolated over 4.00 to 5.00. Text-answer recall is unchanged within CIs on both models
(-0.64 to +0.05 points). The 9B's tax is 4-10x the docs' +0.000225 for Qwen3.5-9B on the older eval, not yet
explained. Deeper blockq at 9B 5.00: 5-bit takes wild 1.26x to 1.07x and swe 1.44x to 1.08x for 0.127 GB, which
ties body bits on wild and beats them on swe; 6-bit adds 1.01-1.02x, which body bits beat. At 4.00 they beat a
5th bit outright. Smaller models at 5+ bpw are a real use case, since they fit small cards, and whether this
9B's sensitivity is general to them is open. Even so, measure a trellis embedding before any 5-bit blockq work:
4-bit packing is easy to understand and trust, and a second depth would complicate that.

The mechanism is built and opt-in behind `EXL3_BLOCKQ_ON_LOAD=1`: a stock EXL3
checkpoint's dense embedding is encoded during weight loading, so the memory
saving needs no derived artifact. Measured on MiniCPM5-1B, model loading falls
0.79 -> 0.52 GiB, matching the offline tool's own 0.267 GiB saving, and the
served result is token-identical to a pre-quantized checkpoint at
|dlogprob| 0.000000e+00.

What remains is the decision to flip the default, and it wants evidence rather
than a change. The determinism question is settled but constrains it: encoding
happens on CPU deliberately, because `blockq.encode` does not reproduce across
devices -- decoded values disagree by up to 2x a quantization step (measured;
`bq_r`, the fp32 per-row affine range, differs in 52.9% of elements from
reduction order alone). CPU is what makes on-load agree with the tool. A GPU
encode would be faster and equally accurate but would silently produce a
different model from the same weights, so if load time ever becomes the
objection, that is the trade being made.

Before flipping: soak it on the models in `bench/`, and confirm the load-time
cost on a large vocabulary is acceptable -- MiniCPM5's 130560x1536 encodes in
about 2.4 s wall, and the models that matter are several times that. The
fallbacks are already proven by unit test rather than by argument: an
indivisible hidden size, an unknown one, a tied model and an explicit
`EXL3_DENSE_EMBED` all end on the dense path.

**The per-start cost is now paid once per checkpoint** (2026-09-08, asked for by
`vllm-untwisted`, whose profiling starts an engine per configuration). The
encoding is cached under `~/.cache/vllm-exl3-plugin/blockq-embeddings/`, keyed on
the resolved commit hash rather than the bit-rate branch -- branches get
re-pushed, and an entry under a branch name would hand a re-quantized checkpoint
the previous embedding without saying so. `EXL3_BLOCKQ_CACHE=0` disables it.

**The large-vocabulary number the flip was waiting on is in:** Qwen3.5-9B's
248320x4096 encodes in **5.8 s**, taking model loading from 1.96 s to 9.99 s on
a cold start and back to 1.96 s on every warm one, at an identical 5.48 GiB
resident. So the objection is now specifically about *first* loads -- 5.8 s of
CPU before a 9B model serves its first token, once per checkpoint -- rather than
about every start. That is the number the default flip has to be argued on, and
what remains unmeasured is a vocabulary well past 248K.

→ [docs/embeddings.md](embeddings.md)

### `gemma4-e2b` — Quantizing gemma-4 E2B/E4B

**Outcome wanted:** an EXL3 E2B checkpoint serving through vLLM with both of its
embeddings quantized. It is the sharpest available demonstration that the embedding
tax is real: 54% of this model is embeddings, the pipeline as it stands saves 27% on
a model sold as 2B, and the repaired one projects 65%.

**Explicitly not a priority.** The thesis is already proven on Qwen3.5-9B with
measured KLD; E2B makes the argument undeniable rather than more true, and the cost
is architecture work in the exllamav3 fork — the most expensive category here, with
no cheap gate for correctness. What would raise it: the appliance wanting a small
vision-capable model, or the audio question under `multimodal` blocking something.

**vLLM already serves it.** The BF16 model runs and describes images correctly
(2026-08-19), so nothing is blocked on the serving side. The gap is entirely
conversion.

**The blocker chain, in order:**

1. **exllamav3 refuses the architecture.** `Config.from_directory` raises
   `NotImplementedError("Gemma4 per-layer inputs are not implemented yet")` —
   a bare guard in `architecture/gemma4.py`, no partial implementation behind it.
   What it is guarding is small: six tensor patterns
   (`per_layer_model_projection`, `per_layer_projection_norm`, and per-layer
   `per_layer_projection` / `per_layer_input_gate` / `post_per_layer_input_norm`),
   77 MiB, 0.8% of the model.
2. **`tools/quantize_embedding.py` handles one embedding suffix**, and this model
   has two.
3. **Tied + blockq crashes**, which is a live defect recorded under
   `quantized-embeddings` rather than here. E2B is tied *and* needs blockq for its
   per-layer tensor, so it cannot serve until that is fixed.

**Text-only is the version worth building.** E2B's vision tower is a full ViT encoder
and its audio tower a conformer with depthwise conv — both unrelated to anything
exllamav3 handles today, and together only 0.88 GiB. Pass them through dense and
serve with `--language-model-only`. One trap if either is ever quantized: every
linear in both towers carries `input_max`/`input_min`/`output_max`/`output_min`
siblings alongside `X.linear.weight`. Those are QAT ranges, not weights, and have to
be recognized as such.

→ [docs/embeddings.md](embeddings.md) "The most extreme case found"
