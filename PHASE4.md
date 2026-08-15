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
checkpoints as they are.

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
| exllamav3 native (fp16 embed + trellis head) | 2.46 GiB | +0 |
| **Phase A** (one trellis serving both) | **0.672 GiB** | +0.0216 |
| trellis head + per-row 4-bit embed | 1.12 GiB | +0.0019 |
| trellis head + per-row 6-bit embed | 1.34 GiB | +0.00024 |

Phase A remains the smallest configuration and keeps its place as the extreme-VRAM option.
But a separate per-row embedding buys back essentially the entire tax for 0.45-0.67 GiB --
and at this operating point that is better value than spending the same bytes on layer
bits, since the layer curve has to flatten hard (0.0270 is already only 0.025 above the
0.00176 noise floor, so no amount of layer precision can buy what fixing the embedding
buys).

So Phase B should **not** be "quantize the embedding with exllamav3's quantizer". It should
be a per-row integer scheme at 4-6 bits: better quality per bit by an order of magnitude,
far simpler, no Hadamard, no 128-block read amplification, and a trivially cheap row
gather.

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

## Choosing depths: what the repair tool should default to

The criterion that makes this well-posed is **equal marginal KLD per byte** -- push a
component's depth until the next byte buys less there than it would elsewhere. Both curves
are measured: the embed/head taxes here, and the layer curve from turboderp's own model
card for `gemma-4-12B-it-exl3` (2.00-8.00bpw), which agrees with our measurements to within
label rounding where they overlap (3.00/3.50/4.00bpw: their 0.107/0.073/0.028 vs our
0.1029/0.0703/0.0270).

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
optimization in the from-scratch quantizer (TODO.md #4b), where layer bpw is free and the
Lagrangian actually binds.

### Untied models, measured

Three untied checkpoints, embedding varied with the trellis head untouched. This is the
first time the embedding is perturbed on models where it is genuinely a *different matrix*
from the head:

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

## Open question: quality at scale

Phase A makes the embedding inherit the head's bit width (`head_bits`, usually 6). That is
a quality question, not just plumbing, and gemma-4 is the most numerically delicate family
here (it already needs fp32 residuals, and is the reason for the flash-attention head-dim
work in TODO #1). Reason for optimism: low-bit GGUFs run gemma embeddings down at Q3_K.

qbench answers this directly now, and the `embed_quant` option already prototyped in
`Exl3Backend` (simulated embedding precision, independent of any real storage format) is
the right instrument for finding the knee before committing to a bit width.
