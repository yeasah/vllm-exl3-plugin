# YAQA in the EXL3 quantizer

Investigation opened 2026-08-26 against `TODO: yaqa`, measured out over the following
days. Nothing is implemented in the converter; the rounding algorithm, five correctness
guards and six diagnostics live in [tools/yaqa/](../tools/yaqa/).

**Where it landed, so nobody has to read to the end for it.** YAQA works on EXL3's
quantizer and reproduces the paper at matched data budget: **−19% KL in-domain, −16% on
neutral text** on Llama-3.2-1B at 2 bits, against Appendix A.11's −20.4% at the same
2K-sequence budget. It is not a calibration artifact — it survives a domain shift that
[EXL3-SC](qbench.md) failed. Gains grow as bitrate falls, and there is no inference cost.
Against that: a reverse-streaming gradient pass the converter does not have, 2.8x the
Hessian working set (14x on wide MoE), a calibration corpus we do not ship, and one
unexplained pathological layer worth 1-3 points. **The build decision is open** — see
[Where it lands](#where-it-lands). One caveat added 2026-08-31: those numbers were all
measured with `apply_out_scales` off, which is not what the converter ships — see
[The ignored `H_O` is load-bearing](#the-ignored-h_o-is-load-bearing).

Read [Result](#result-matched-data-budget-reproduces-the-paper) onward for what holds.
The Qwen3-0.6B sections before it are a superseded first pass, kept for the hypotheses
they ruled out. Numbers marked *(estimated)* are arithmetic on model configs, not
measurements.

Source: Tseng, Sun, De Sa, *Model-Preserving Adaptive Rounding*, arXiv:2505.22988v3
(03 Jun 2026), CC BY 4.0. **Paper only.** See [Licence hazard](#licence-hazard) —
this is no longer as simple as "don't clone the repo".

## The one-sentence version

EXL3's rounding step minimizes each layer's *immediate activation error*; YAQA
minimizes an estimate of the *whole model's output KL*, by adding a second Hessian
on the output channels and feeding rounding error back along both axes instead of
one.

## Why this deserves more than "large project, likely modest gains"

The TODO entry's framing was written before anyone read Table 1. That table is run
**with incoherence processing and the QTIP quantizer** — which is to say, with
EXL3's quantizer and EXL3's transform. The `LDLQ` row is, near enough, what we ship
today; the `YAQA-B` row is the target.

| model | bits | KL, LDLQ | KL, YAQA-B | change |
|---|---|---|---|---|
| Llama 3.2 3B Inst. | 2 | 0.455 | 0.288 | −37% |
| | 3 | 0.085 | 0.047 | −45% |
| | 4 | 0.021 | 0.014 | −33% |
| Llama 3.1 70B Inst. | 2 | 0.497 | 0.335 | −33% |
| | 3 | 0.138 | 0.094 | −32% |
| | 4 | 0.045 | 0.030 | −33% |

Read as bits rather than percentages, on the 3B: YAQA-B at 3 bits (0.047) lands
between LDLQ at 3 bits (0.085) and LDLQ at 4 bits (0.021). At 2 bits the gap is
wider still. **Call it a third to a half of an effective bit, and more of it at the
low end than the high end** — which is the end the appliance lives at, and the end
where `cpu-offload` was argued for on exactly the same grounds.

Two further claims from the paper worth keeping in view, both unverified by us:

- It is quantizer-agnostic and adds **no inference overhead** — the output format is
  unchanged, so nothing in the plugin, the kernels, or `blockq` is touched. This is
  purely a converter-side change.
- Sketch B at **2K sequences costs ~1 GPU-hour** for an 8B model and still beats
  LDLQ handily (Table 9: Qwen3 8B @ 2 bits, KL 0.227 vs 0.285, GSM8K 62.1 vs 56.3),
  against 1.5 GPU-hours for LDLQ's own calibration. That is the paper's own
  cheapest-configuration claim and it is the one to test first, because it says the
  cost objection may not exist.

## What the algorithm is, in EXL3's terms

EXL3 stores weights row-major as `(k, n) = (in_features, out_features)` — transposed
from the paper's `W ∈ R^{m×n}` — so the paper's `H_I` (input side) is EXL3's `H`,
and the paper's `H_O` (output side) is a **new `n × n` matrix over output channels**.

Today, [`ldlq()`](../deps/exllamav3/exllamav3/modules/quant/exl3_lib/quantize.py#L488)
walks 16-row bands of `k` from the bottom up and quantizes a whole row of 16×16 tiles
at a time, with one feedback term:

```
rows = W[bi:bj] + Lᵀ · (W[bj:] − Ŵ[bj:])
```

YAQA replaces that with three terms — input feedback, output feedback, and the
cross term:

```
Ŵ = Q(W + L_Oᵀ Δ L_I + L_Oᵀ Δ + Δ L_I),    Δ = W − Ŵ
```

and the sweep order changes from *rows of tiles* to *anti-diagonals of tiles*: a tile
at `(i, j)` now depends on everything below it *and* everything to its right, so the
frontier is a wavefront of `k/16 + n/16` steps rather than `k/16`. The paper's
Lemma 3.3 is what buys this — the Kronecker structure keeps the dependency depth at
`m + n` instead of `mn`.

### The parts that fit well

- **`Q` is already a black box of exactly the right shape.**
  [`quantize_tiles()`](../deps/exllamav3/exllamav3/modules/quant/exl3_lib/quantize.py#L66)
  takes a batch of independent 16×16 tiles and returns reconstructions plus indices,
  with no cross-tile state. The trellis search is per-tile. A wavefront just changes
  *which* tiles are in each batch, not what a batch is.
- **The tile geometry already matches.** The paper's `g_x`/`g_y` block-LDL sizes are
  16×16 for us on both axes, and
  [`block_ldl()`](../deps/exllamav3/exllamav3/modules/quant/exl3_lib/quantize.py#L411)
  is already parameterised by block size. It gets called a second time on `H_O` with
  the same `16`.
- **Output-side incoherence processing already exists.** `sv` and `had_n` are already
  applied to the weight's output axis and already shipped as `svh`. Today `H_O = I`,
  so nothing needs transforming; under YAQA, `H_O` needs the same treatment `H` gets
  from `su`/`had_k` in
  [`finalize_capture_H()`](../deps/exllamav3/exllamav3/modules/quant/exl3_lib/quantize.py#L832).
  Mechanically straightforward — but see the traps below.
- **Cost of the rounding step itself is ~2×**, per the paper's SND argument, on a
  step that is not the bottleneck.

So the *rounding* half of YAQA is a contained change to one file.

### The part that does not fit at all

`H_O` (and `H_I` under Sketch B) are built from **∇_y ℓ — the gradient of the KL to
the original model's output logits, with respect to this layer's output.** That
requires backpropagation from the model head down to the layer.

`convert_model.py` cannot do this as written, and not for a small reason. Its whole
design is a **forward-only, one-module-resident stream**:

- it loads one module, runs the calibration rows through it, captures `H` as a
  running `x.T @ x` in
  [`Linear.capture_H()`](../deps/exllamav3/exllamav3/modules/linear.py#L533),
  quantizes, advances `state`, unloads, and moves on;
- `state` is the list of per-row hidden states, held on CPU and overwritten in place
  each module — and it is the *quantized-prefix* state, so error from earlier layers
  is already baked in;
- the full-precision model is never resident, there is no autograd anywhere, and the
  peak footprint is one module.

That property is a large part of why EXL3 can quantize a 70B on one consumer card,
and it is the property YAQA's Hessian stage breaks. This is the project.

## Reverse streaming: the shape a solution probably takes

The naive reading — "you need the whole BF16 model resident with autograd" — is not
forced. The gradient stage is structurally the mirror image of what the converter
already does, and could plausibly be a *second stream running backwards*:

1. Forward once through the **unquantized** model, keeping block-boundary states.
2. Walk modules in reverse. For each: load it, recompute its forward from its saved
   input state with autograd on *within the block only*, backprop the incoming
   `grad_state` through it, accumulate `H_O` (and `H_I` for Sketch B) for each of its
   linears, emit `grad_state` for the block below, unload.
3. Then quantize forward as today, now with both Hessians in hand.

One module resident, autograd scoped to one module, `grad_state` sitting in host RAM
next to `state`. The converter already checkpoints `state` to
`ckpt/state.safetensors` for resume, so the machinery for spilling it exists.

**The cost is storing the forward states.** Keeping every block boundary is out of
reach — 313 GB for Qwen3.8-27B at the default 250×2048 calibration *(estimated)*, and
YAQA-B wants 2K sequences, not 250 rows. Segment checkpointing at √L boundaries
brings that to ~40 GB at the price of one extra forward pass overall, which is the
standard trade and looks like the right one. **This has not been worked through
properly and is the first thing to actually design.**

Note also that step 1 wants states from the *original* model, whereas the existing
pipeline's `state` is the error-propagated quantized one. These are two different
passes over the data, not one.

## Sizing

Hessian working set per transformer layer, fp32, unique matrices only (Q/K/V share
`H_I`; gate/up share `H_I`) — *(estimated from config.json, not measured)*:

| model | `H_I` today | new `H_O` | ratio | worst single tensor |
|---|---|---|---|---|
| Qwen3.5-9B | 0.75 GB | 1.32 GB | 2.8× | gate/up `H_O`, 576 MB |
| Qwen3.8-27B | 1.46 GB | 2.60 GB | 2.8× | gate/up `H_O`, 1.16 GB |
| Muse-Glimmer-30B | 1.88 GB | 3.36 GB | 2.8× | gate/up `H_O`, 1.52 GB |
| Qwen3.5-35B-A3B | 0.34 GB | 4.58 GB | **14.3×** | expert `down_proj` `H_O`, 4.10 GB |

Dense models pay a flat ~2.8×, dominated by `H_O` for `gate_proj`/`up_proj` where
`out_features` is the MLP width. That is not comfortable but it is the same order as
the `down_proj` `H_I` the converter already handles, and `H_swap_device` already
spills `H` to host RAM.

**Wide MoE is the outlier, and it is an outlier by construction.** `H_I` for an
expert `down_proj` is `moe_intermediate²` = 1 MB; `H_O` is `hidden²` = 16.8 MB, and
there are 256 of them per layer. Nothing about that is shareable — every expert has
its own output space. 4.1 GB per layer in host RAM, consumed and freed per layer, is
survivable; it is worth knowing before someone is surprised by it, because
Qwen3.5-35B-A3B is in the blessed bench tier.

Data: the converter defaults to 250 rows × 2048 tokens = 512K tokens. YAQA-B's
cheapest published configuration is 2K sequences × 2048 = 4M tokens, 8× more, each
needing a forward *and* backward on the unquantized model. Whether the paper's
robustness to sequence count extends down to 250 rows is unknown and is a cheap thing
to find out early.

## Traps identified but not resolved

- ~~**EXL3 already has a non-identity `H_O` and ignores it.**~~ **Resolved, and the
  sign was backwards** — measured 2026-08-31, see
  [The ignored `H_O` is load-bearing](#the-ignored-h_o-is-load-bearing). The latent
  `H_O = H_n D_sv² H_nᵀ` is real and the description above is accurate, but restoring
  it makes the model *worse*, by up to +57% KL. Ignoring it is the mechanism by which
  `apply_out_scales` works. Nothing to harvest here.
- **`regularize()` picks output-channel scales after `H` is finalized.**
  [`regularize()`](../deps/exllamav3/exllamav3/modules/quant/exl3_lib/quantize.py#L1125)
  folds a data-dependent `out_channel_scales` into `sv` and divides the weight by it.
  `H_O` must receive the matching congruence transform, applied in the right order
  relative to `su`/`had_n`. Get this wrong and nothing crashes — the model is just
  quietly worse. This wants a synthetic case where the correct `H_O` is known.
  (Related: `apply_out_scales` is a *heuristic* for exactly the non-uniform output
  sensitivity that `H_O` measures directly, so YAQA may subsume it. But it is a
  *load-bearing* heuristic worth up to 66% KL on its own — see
  [The ignored `H_O` is load-bearing](#the-ignored-h_o-is-load-bearing) — so anything
  that displaces it has to clear that bar, not the uncalibrated one.)
- **qmap sharing does not extend to `H_O`.** Q/K/V share one `H_data` because they
  share an input. They do not share an output. `H_O` has to be per-linear, which cuts
  against how `capture_H` is keyed today.
- **Regularization and positive-definiteness.** The paper wants fp32 throughout and
  explicitly reports TF32 as insufficient, with `≈1e-4·tr(H)/n` diagonal damping.
  EXL3's `sigma_reg` default is 0.025 and its Cholesky already has a retry ladder
  plus a capture-on-failure path. Two damping regimes now, on matrices with quite
  different spectra — `H_O` is claimed to be strongly low-rank, which is the *reason*
  YAQA beats LDLQ but also the reason its Cholesky is the one that will fail.
- **Fallback paths.** `q_fallback` and `fallback_quant()` assume one Hessian. Whether
  a missing or degenerate `H_O` should fall back to `H_O = I` (i.e. plain LDLQ) rather
  than to uncalibrated is a decision, and `H_O = I` is exactly what LDLQ is, so the
  degradation path is clean if it is wired that way.

## The ignored `H_O` is load-bearing

Measured 2026-08-31 on Llama-3.2-1B-Instruct with
[`tools/yaqa/outscales.py`](../tools/yaqa/outscales.py); full logs for every run quoted
below are in [docs/data/yaqa-outscales/](data/yaqa-outscales/). This closes the first
entry in [Traps](#traps-identified-but-not-resolved), in the opposite direction to the
one it was written in.

**The claim under test.** `regularize()` with `apply_out_scales` on folds a
per-output-channel scale into `sv` and divides the weight by it. The layer's true output
metric is the identity, so the metric in *quantization* space is `H_O' = H_n D_sv² H_nᵀ`,
which `ldlq()` ignores. Feeding it to the wavefront needs no gradients, no sketch and no
second pass. The Trap entry assumed that was free quality left on the table.

**It is not a corner case.** `convert_model.py`'s `--out_scales` defaults to `always`,
not `auto`, so the skew heuristic in `regularize()` is dead code by default and every
calibrated body tensor in every shipped EXL3 quant carries this. Confirmed directly
against `turboderp/Llama-3.2-1B-Instruct-exl3` at 2 bits: its `svh` tensors have
non-unit magnitude, matching a weights-only prediction of `out_channel_scales` to three
decimals (L0.k_proj 0.3782 vs 0.378, L0.gate_proj 0.1196 vs 0.120, lm_head 0.0979 vs
0.098). Because the block Hadamard is orthogonal, `H_O'`'s eigenvalues are *exactly*
`out_channel_scales²`, so the size of the effect is readable off the weights alone,
with no calibration data: `std/mean` runs 0.23-0.46 on `q_proj`/`k_proj`, 0.05-0.15 on
`down_proj`, up to 29x max/min.

### Result: restoring it makes the model worse

Per-arm KL of the whole model against bf16 with one layer quantized, bootstrap CI over
sequences, `ldlq-dup` reading exactly +0.0% throughout. `ldlq+ho` restores the true
metric; `ldlq-noos` instead makes it exact by turning `apply_out_scales` off.

| tensor | K | eval | `ldlq+ho` | `ldlq-noos` |
|---|---|---|---|---|
| L1.q_proj | 2 | code | +27.2% [+15.6, +40.7] | +52.8% |
| L1.q_proj | 3 | in-domain | +8.0% [+2.9, +13.7] | +19.9% |
| L1.q_proj | 4 | in-domain | −0.2% [−1.5, +1.2] | +5.0% |
| L1.k_proj | 2 | literary | +57.3% [+52.0, +62.7] | +61.8% |
| L1.k_proj | 2 | in-domain | +31.6% [+20.9, +44.9] | +26.7% |
| L1.o_proj | 3 | code | −6.4% [−10.1, −2.2] | −0.2% |

The harm shrinks with bitrate — near-neutral at 4 bits (−0.2 to +4.7% across all three
tensors), largest at 2 — which is the mirror of YAQA proper and for the same reason: at
low bitrate the rounding has more freedom to spend on whatever objective it is given, so
a misspecified one costs more. The ordering is clean on `k_proj` (literary: +57.3%,
+12.0%, +4.7% at K=2/3/4) and holds for 4 bits against the rest on `q_proj`, where K=2
and K=3 are within each other's intervals. `o_proj` is the exception, and it is the
tensor whose output scales are flattest (`sv` std/mean 0.180 against 0.386 for
`q_proj`) and where `apply_out_scales` is worth least (`ldlq-noos` only +2 to +6%).
On `L1.k_proj` at K=2 in-domain the "fix" is *worse than deleting the heuristic
entirely*.

### Why: it un-does what `apply_out_scales` is for

Dividing the weight by its per-output-channel RMS and then rounding with `H_O = I` does
not minimize absolute activation error — it minimizes **relative per-channel** error.
That is the whole content of the heuristic, and it is delivered precisely by dropping
the `D_sv²` factor. `outscales.py` reports the slope of
`log(per-channel relative error)` against `log(out_channel_scales)`: 0 means uniform
relative error, −1 means uniform absolute error.

| arm | slope | KL (L1.q_proj, K=3, in-domain) |
|---|---|---|
| `ldlq`, as shipped | **−0.001** | +0.0% |
| `ldlq+ho`, metric restored | −0.550 | +8.0% |
| `ldlq-noos` | −0.883 | +19.9% |

KL rank-orders with `|slope|` across every tensor and bitrate measured. The correction
works exactly as designed — it lowers the objective it targets, `tr(Δ H_O Δᵀ H_I)`, by
14-17% on `q_proj`/`k_proj` and 4-5% on `o_proj`, stably across four decades of damping
— and buys that by raising `‖Δ‖²` by the same order (~20% and ~2% respectively) and
dragging the error profile back toward uniform-absolute.

### `α = 0` is at the optimum

Generalising to `H_O = B D_sv^{2α} B` — `α = 1` the true metric, `α = 0` the identity
EXL3 rounds with, `α < 0` over-correcting the other way — the shipped behaviour is the
best of the five at K=3, degrading in both directions:

| α | −0.5 | 0 | 0.25 | 0.5 | 1 |
|---|---|---|---|---|---|
| L1.q_proj K=3 in-domain | +4.5% | **+0.5%** | +1.0% | +2.7% | +8.7% |
| L1.q_proj K=3 literary | −0.5% | **−0.8%** | +0.6% | +1.4% | +6.9% |

A −13.3% dip at `α = 0.5`, K=2 in-domain did not reproduce at K=3 or on `o_proj`;
2-bit rounding is a much higher-variance lottery and it is noise. Note that `α = 0` is
*not* bit-identical to `ldlq`: `L_O` comes out ~3e-8 rather than exactly zero, so the
wavefront takes its two-sided arithmetic path and flips tiles at quantizer decision
boundaries. It reads ±1% rather than the +0.0% of `ldlq-dup`, and that is the honest
noise floor for the `α` arms — a second floor worth having, since `ldlq-dup` only
bounds the KL measurement and not the rounding.

### Two things this changes elsewhere in this document

- **Every YAQA number here was measured with the heuristic off.**
  [`probe.py`](../tools/yaqa/probe.py#L288) sets `apply_out_scales = False`, so the
  `ldlq` baseline behind the −19% headline is up to 66% worse than what the converter
  ships (+2 to +6% on `o_proj`, +15 to +66% on `q_proj`/`k_proj`). Whether YAQA's gain
  survives on top of `apply_out_scales` is untested, and it is now the first thing to
  check if the build is picked up — it is cheap, and a −19% measured against the wrong
  baseline is the kind of error [EXL3-SC](qbench.md) made.
- **The composition matters.** Under YAQA *and* out-scales the correct metric is
  `B D_sv H_O_yaqa D_sv B`, which contains the `D_sv²` factor measured harmful here.
  An implementation that "fixes the bookkeeping" while adding YAQA gets that factor
  whether or not it wants it.

### Attempted corroboration, inconclusive

If out-scales' equalization is right, YAQA's own measured `H_O` should say downstream
sensitivity falls like `ocs⁻²`, making the composed metric flat.
[`ho_vs_scales.py`](../tools/yaqa/ho_vs_scales.py) fits
`log diag(H_O_sketchB)` against `log ocs` at 128 sequences. The slopes scatter from
−2.20 to +0.04 with `r` from −0.90 to +0.01: near −2 with strong correlation on
`down_proj` and `o_proj` (where the composed metric does flatten), near 0 on `q_proj`
(where it does not). At this sketch budget the diagnostic is too weak to build on —
this document's own history shows under-sampled Sketch B reversing conclusions — so the
mechanism rests on the error-profile slope above, which is a direct measurement, not on
this. Worth redoing at the paper's data budget if anyone revisits.

## Licence hazard

The TODO says "papers only", on the assumption that the hazard is the reference
repository. **That assumption is wrong in a way that matters.** The arXiv paper is
CC BY 4.0, and its appendices contain Python source:

- **A.6** — a working implementation of the YAQA rounding wavefront, including the
  anti-diagonal indexing.
- **A.8, A.9** — modified PyTorch backward passes for Sketches A and B.

I read A.6 while reading the appendix, before noticing what it was; I have not opened
A.8 or A.9. Whether CC BY 4.0 listings inside the paper count as "the paper" or as
"the reference code" is a call for the project, not for me, and it should be made
before anyone goes further, because A.6 is precisely the part of the algorithm that
is hard to get right from the prose. The prose form — Algorithm 1, the fixed-point
iteration — is unambiguously safe and is what this note is written from.

## The single-layer experiment

Built 2026-08-26 as the cheapest decisive test, before any converter work.
`tools/yaqa/` holds it: `rounding.py` (the wavefront), `test_guard.py` (correctness),
`probe.py` (end-to-end KL), `hessian_fit.py` (how well the sketch fits the true
Hessian). Model is Qwen3-0.6B bf16 on one 5070 Ti; **one** linear is quantized and
everything else left at bf16, so the measured quantity is the whole-model KL against
the original — exactly what YAQA claims to minimize.

Arms share one `regularize()` call and one pair of sign vectors, so the *only*
difference between them is the rounding:

| arm | `H_I` | `H_O` | |
|---|---|---|---|
| `ldlq` | `E[xᵀx]` | `I` | what EXL3 ships |
| `yaqa-o` | `E[xᵀx]` | Sketch B | output feedback only |
| `yaqa` | Sketch B | Sketch B | full YAQA |
| `ldlq-b` | Sketch B | `I` | Sketch B's input factor alone |

Sketch B needs no custom autograd at this scale. Appendix A.9's einsums are just
`H_I += GᵀG`, `H_O += GGᵀ` for the per-*sequence* weight gradient `G` of the
Monte-Carlo-sampled cross entropy at the model's own output — i.e. `.grad`, once per
sequence, with the real Fisher rather than the empirical one.

### The implementation is correct

Two guards, both passing, because a null result is only worth reporting if the thing
being tested works:

- **With `H_O = I` the wavefront reproduces EXL3's `ldlq()` bit-for-bit** — identical
  encoded indices, `max |Δrecon| = 0`. It is a legal reordering: the input-feedback
  term for tile `(i, j)` reads only column-block `j` at rows below, and every such
  tile lands on an earlier wavefront.
- **On synthetic data with a known low-rank `H_O`, YAQA cuts the loss it targets,
  `tr(Δ H_O Δᵀ H_I)`, by 82.6%** — while *increasing* LDLQ's objective by 117% and
  doubling `‖Δ‖²`. That is the right signature. An algorithm that improved every
  metric at once would mean the harness was measuring nothing.

A `ldlq-dup` arm — a bit-identical copy of the `ldlq` weight — is carried through the
KL measurement as a live noise floor. With TF32 off and eager attention it reads
exactly `+0.0%`, so effects of a few percent are real.

### First pass, on a model too small to interpret (Qwen3-0.6B, superseded)

*Kept because the guards, the ruled-out hypotheses and the failure ordering below are
still valid. The magnitudes are not: this ran on a 0.6B, at a fraction of the data
budget, and both were wrong. Skip to "Result" for what actually holds.*

Qwen3-0.6B, K=3, 128 calibration rows and 512 Sketch B sequences at 1024 tokens,
32 held-out eval rows. Each cell is that arm's whole-model KL against `ldlq`:

| layer.proj | `H_O` 50%/90% of trace | `ldlq-b` | `yaqa-o` | `yaqa` |
|---|---|---|---|---|
| L2.mlp.down_proj | 9% / 31% | +1.1% | **+72.7%** | **+60.8%** |
| L2.mlp.gate_proj | 8% / 47% | −2.5% | +2.3% | −0.7% |
| L2.self_attn.o_proj | 21% / 72% | +0.6% | +0.9% | +0.7% |
| L14.mlp.down_proj | 19% / 68% | +0.3% | −6.3% | −6.6% |
| L14.mlp.gate_proj | — | +0.5% | −1.0% | −1.6% |
| L14.self_attn.o_proj | — | +0.0% | −3.7% | −3.6% |
| L26.mlp.down_proj | 20% / 71% | −0.9% | −8.6% | −8.7% |
| L26.mlp.gate_proj | 12% / 58% | −7.6% | +4.9% | −2.0% |
| L26.self_attn.o_proj | 18% / 69% | −1.3% | −12.2% | −12.5% |

Three things to read off it:

1. **Away from layer 2, YAQA wins consistently** — −1.6% to −12.5%, median around
   −6%. The effect is real and reproducible, and it is not noise: the `ldlq-dup`
   arm reads `+0.0%` in every row.
2. **Almost all of it is the output-side feedback.** `yaqa-o` (Sketch B's `H_O`
   with the *activation* `H_I`) tracks full `yaqa` closely, while `ldlq-b`
   (Sketch B's `H_I`, no wavefront) is near zero. If only one half of YAQA were
   ever implemented, it is the `H_O` half that matters.
3. **It is 3–5× short of the paper's 30%**, and one configuration is catastrophically
   worse rather than merely flat.

The failure ordering looked informative: the more low-rank `H_O` is, the worse YAQA
does, which is backwards from Theorem 3.4 where low rank is exactly what makes YAQA's
bound beat LDLQ's. That reads like a numerical problem, and two plausible mechanisms
were tested on L2.mlp.down_proj, the worst case. **Both are ruled out:**

- **Not ill-conditioning.** `max|L_O|` is 0.20–0.32 across damping from 1e-4 to 0.5.
  The feedback is small and the LDL is healthy.
- **Not the codebook operating point.** `regularize()` picks EXL3's global scale by
  test-quantizing the *uncompensated* weight, so two-sided feedback could in principle
  push the tiles outside the range that scale was chosen for. Measured, the values
  actually handed to the trellis have RMS ×1.00–1.02 of the weight for every arm. An
  added `+rs` arm that re-matches the scale to the fed distribution changes nothing
  (+50.1% vs +60.8%, still far worse than `ldlq`).
- **Damping only helps by erasing the algorithm.** Sweeping `sigma_o` walks the result
  from +72.7% at 1e-3 monotonically down to +1.8% at 0.5 — but heavy damping drives
  `H_O` toward `I`, which *is* LDLQ. That is not a fix, it is a measurement of how
  much YAQA has to be switched off to stop it hurting.

So the feedback at layer 2 is well-conditioned, small, and correctly transformed — and
it still steers the rounding the wrong way. That leaves only one explanation: **at that
layer the Kronecker sketch `H_O ⊗ H_I` is simply a bad approximation of the true
Hessian**, which is the quantity Theorem 3.4's bound depends on and the one
`hessian_fit.py` measures directly.

### Why: the sketch is under-sampled, and the theory is behaving exactly as advertised

`hessian_fit.py` measures the paper's own Figure 3 quantity without any quantizer in
the loop. The true Hessian is the Fisher, `H = E_b[vec(G_b) vec(G_b)ᵀ]`, so for any
Kronecker sketch the (unnormalized) alignment is `E_b[tr(G_bᵀ H_O G_b H_I)]`, an exact
expectation over held-out per-sequence gradients. LDLQ is the sketch `(I, E[xᵀx])`, so
the two sit in the same frame. Higher is better; `‖H‖` is common to both and dropped,
as the paper does.

Qwen3-0.6B, 128 calibration rows at 1024 tokens, 48 held-out sequences:

| layer.proj | LDLQ | B@64 | B@256 | B@1024 | best/LDLQ | measured KL |
|---|---|---|---|---|---|---|
| L2.mlp.down_proj | 6.36e5 | 3.24e5 | 5.54e5 | 7.75e5 | **1.22×** | **+72.7%** |
| L2.self_attn.o_proj | 5.80e2 | 7.07e2 | 8.59e2 | 9.33e2 | 1.61× | +0.7% |
| L26.mlp.down_proj | 1.89e3 | 3.12e3 | 3.53e3 | 3.67e3 | 1.94× | −8.7% |
| L26.self_attn.o_proj | 4.83e2 | 8.08e2 | 9.18e2 | 9.45e2 | 1.96× | −12.5% |

**Sketch alignment and end-to-end KL benefit are perfectly rank-correlated.** That is
Theorem 3.4 doing exactly what it claims: the benefit is governed by how close the
Kronecker sketch is to the true Hessian, and nothing else. The mechanism transfers to
EXL3's quantizer intact.

What does not transfer is the *magnitude*, and the reason is visible in the same table:

- **Alignment has not begun to saturate.** Every row is still climbing at 1024
  sequences. L2.mlp.down_proj goes 3.24e5 → 5.54e5 → 7.75e5 and is still rising.
- **At small budgets Sketch B is worse than the Hessian it replaces.** At 64 sequences
  L2.mlp.down_proj reads 3.24e5 against LDLQ's 6.36e5 — 0.51×. Under-sampling does not
  merely weaken YAQA, it inverts it, and that is the whole of the +72.7%.

So the earlier reading was wrong in an instructive way: layer 2 is not unstable, it is
the layer where 512 sequences buys the least alignment. The paper uses 2K–64K sequences
at 2048 tokens; this ran 512 at 1024, one eighth of the tokens of its *cheapest*
published configuration. EXL3's bundled calibration corpus yields only 694 unique rows
at 2048 tokens in total, so matching the paper's budget needs a real corpus, not a
tuning change.

### Confirming the diagnosis: alignment predicts the sign

Re-run at 4x the sketch budget (1024 calibration rows, 2048 Sketch B sequences), against
the alignment measured on the same configuration:

| layer.proj | alignment vs LDLQ | KL @512 seqs | KL @2048 seqs |
|---|---|---|---|
| L2.mlp.down_proj | 1.11x | +72.7% | **+8.4%** |
| L2.self_attn.o_proj | 1.19x | +0.7% | +3.7% |
| L26.mlp.down_proj | 1.79x | −8.7% | −5.9% |
| L26.self_attn.o_proj | 1.38x | −12.5% | −8.4% |

Two things land here. **Quadrupling the sketch budget took the catastrophic case from
+72.7% to +8.4%**, which confirms under-sampling as the cause beyond reasonable doubt.
And **alignment predicts the sign in all four cases** — above ~1.3x it helps, below
~1.2x it hurts — with the rank order right in three of four.

That last point is the most useful engineering result here. Alignment is computed from
gradients the sketch pass already produces, needs no quantizer and no rounding, and is
measured on held-out sequences. **Whether YAQA will help a given layer is knowable
before committing to it**, which turns a risky global switch into a per-layer decision
with a free fallback (`H_O = I` *is* LDLQ).

### Why that pass was not a result

A 0.6B model is smaller than anything the paper tested, and the run used roughly a
twentieth of the paper's smallest data budget. "We measured less" was an observation
about Qwen3-0.6B and about starved calibration, not about YAQA on EXL3. Both were
corrected, and both mattered.

### Llama-3.2-1B at the paper's own budget: the size hypothesis holds

Qwen3-0.6B was smaller than anything the paper tested, which made "we measured less"
uninterpretable. Table 1 covers 1B-70B, and the **1B shows the paper's *largest*
relative gains** (-36.6% / -40.4% / -40.6% at 2/3/4 bits, against -32%/-36%/-32% for
the 8B), so Llama-3.2-1B-Instruct is both the right control and small enough to run on
one 16 GiB card. Run at YAQA-B's published configuration -- 2048 sequences x 2048
tokens, 4M tokens per layer -- K=3, noise floor `+0.0%` on every row:

| layer.proj | `H_O` 50%/90% | `ldlq-b` | `yaqa-o` | `yaqa` |
|---|---|---|---|---|
| L7.mlp.down_proj | 18% / 68% | -1.5% | -17.4% | **-18.4%** |
| L7.self_attn.o_proj | 15% / 61% | -2.3% | -14.5% | **-15.2%** |
| L14.mlp.down_proj | 22% / 73% | -6.2% | -7.1% | **-11.7%** |
| L14.self_attn.o_proj | 21% / 71% | -4.0% | -2.6% | **-5.6%** |
| L1.mlp.down_proj | 10% / 38% | -7.1% | +58.4% | **+56.4%** |
| L1.self_attn.o_proj | 27% / 77% | -8.4% | -7.8% | **-8.0%** |

**Median -13%, against -6% on Qwen3-0.6B -- the effect roughly doubled for a doubling of
model size.** Two points is not a curve, but it is enough to say the 0.6B result was a
small-model artifact and that dismissing scale was wrong.

**Sketch B's input factor also starts earning its keep**: `ldlq-b` runs -1.5% to -8.4%
here, where on Qwen3-0.6B it was worth roughly nothing. Both halves of YAQA contribute at
this scale, not just the output side.

### Bitrate: the gain grows as bits fall

Same model and budget, sweeping K within one Hessian collection (`--bits 2 3 4`; the
collection is ~18 min per layer and bitrate-independent, so it is paid once):

| layer | K=2 | K=3 | K=4 |
|---|---|---|---|
| L7.mlp.down_proj | **-21.3%** | -18.4% | -10.3% |
| L14.mlp.down_proj | **-11.6%** | -11.7% | -10.3% |
| L1.mlp.down_proj | +49.9% | +56.4% | +40.7% |

**The benefit is largest exactly where the appliance lives.** -21.3% at 2 bits on the
best layer, monotone in the right direction on both healthy layers. Same shape as the
paper's own table, and the opposite of what a numerical artifact would look like --
rounding pathologies get *worse* with more headroom, not better.

### The early-layer anomaly is isolated to `H_O`

L1.mlp.down_proj is harmful at every bitrate (+40.7% to +56.4%), across two models and
four separate data budgets. The decomposition says exactly where it lives:

| arm | what it uses | L1.down_proj @ K=2 |
|---|---|---|
| `ldlq-b` | Sketch B's `H_I`, no output feedback | **-10.8%** |
| `yaqa-o` | activation `H_I` + Sketch B's `H_O` | +55.3% |
| `yaqa` | both from Sketch B | +49.9% |

**Sketch B's input factor is not merely fine at this layer, it is the best arm on the
board** -- better than at any healthy layer. The damage is entirely output-side. So this
is not bad gradients, a bad sketch pass or bad data: `H_I` and `H_O` come from the same
gradients in the same loop.

The layer is also the most anisotropic in the model (50% of `H_O`'s trace in 10% of its
eigenvalues, against 18-22% at healthy layers). That suggested the *second-order model
itself* fails there -- YAQA minimizes `vec(D)ᵀ H vec(D)`, and where `H` is this
anisotropic a 2-bit perturbation might sit outside the radius where that quadratic
approximates the KL. **Tested with `secondorder.py`, and refuted.** The prediction was
that YAQA would *lower* the true second-order error while raising the KL. It does the
opposite:

| | `Q`/token (true 2nd-order) | actual KL | |
|---|---|---|---|
| L7 (healthy) `ldlq` | 1.020e-2 | 5.70e-3 | |
| L7 `yaqa` | 6.95e-3 (−32%) | 4.86e-3 (−15%) | lowers both |
| L1 (anomalous) `ldlq` | 3.70e-2 | 7.25e-3 | |
| L1 `yaqa` | 5.14e-2 (**+39%**) | 1.35e-2 (+86%) | raises both |

`Q` predicts the sign of the KL change at both layers, so the quadratic is fine. YAQA
*increases* the true second-order error at L1: the sketch is genuinely misaligned there,
not the theory. That is the better of the two outcomes, since misalignment is in
principle fixable with better Hessian estimation where a broken quadratic would not be.

It also gives a gate that needs no threshold: round both ways, compute `Q` on held-out
gradients, keep the lower. `H_O = I` is LDLQ exactly, so the fallback is free.

### Result: matched data budget reproduces the paper

The gap to the paper turned out to be the corpus, and the arithmetic was stark. EXL3's
bundled calibration text is 942K tokens once `code.utf8` is held out, so a 2048-sequence
sketch recycled 460 rows **4.5x** — which is **4.5x below the smallest configuration the
paper ever reports** (Appendix A.11: 2K sequences of 2K tokens, all unique) and 142x
below its main results. Fresh Monte-Carlo labels on a repeated row cut label-sampling
variance and do nothing for data-sampling variance.

Re-run against 4.19M unique tokens from RedPajama-V2 English, sketch count unchanged at
2048 so the *only* variable is unique-token volume, Llama-3.2-1B-Instruct, K=2, bootstrap
CIs on every delta:

| `yaqa` @K=2 | 942K recycled 4.5x | 4.19M unique |
|---|---|---|
| L7.mlp.down_proj, in-domain | −10.2% | **−18.8%** |
| L7.mlp.down_proj, literary | −12.0% | −15.7% |
| L14.mlp.down_proj, in-domain | −10.0% | **−19.3%** |
| L14.mlp.down_proj, literary | −11.6% | −17.3% |
| L14.mlp.down_proj, code | +0.3% | −4.6% |
| L1.mlp.down_proj, literary | +79.3% | +88.5% |

**−19.0% in-domain against A.11's −20.4% at the same 2K-sequence budget.** That is a
near-exact reproduction of the paper's own smallest published configuration, and it is
the answer to "is something about EXL3 costing us the effect": no. Nothing structural was
hiding. A.11's curve continues (−20.4% at 2K, −26.3% at 16K, extrapolating to ~−28% at
64K), so a production implementation on a real corpus would plausibly sit near −25%.

Neutral text runs consistently ~2.6 points below in-domain, so **−16% is the number to
plan against, not −19%**.

### YAQA is not a calibration artifact — the test EXL3-SC failed

This project has already been burned by a quantization result that existed only on its
own calibration distribution: EXL3-SC measured 1.30x better in-domain and 1.17x worse on
neutral text, a **1.52x swing in standing from the evaluation set alone**
([qbench.md](qbench.md)). YAQA estimates a Fisher, a far higher-variance statistic than
LDLQ's `E[xᵀx]`, so it has strictly *more* capacity to overfit a corpus. Scored only on a
held-out slice of its own mixture, it would have reproduced exactly that mistake.

Scored on three distributions from one sketch collection — a held-out slice of the
calibration corpus, `code.utf8` held out of calibration entirely, and a separate literary
corpus — the healthy layers move by 1–4 points, not 52%. **The SC failure mode is ruled
out.**

The decomposition is mechanically sensible, too: the distribution-bound half is
`ldlq-b`, Sketch B's *input* factor (+4.9% on code, −3.7% in-domain at L14). `H_O`
captures downstream model structure, which is largely distribution-independent; `H_I`
captures input statistics, which are not. `yaqa-o` — EXL3's existing activation `H_I`
plus Sketch B's `H_O` — is the most robust arm on the board and is also the cheaper half
to build.

### The early-layer pathology, and why the paper would not have seen it

One linear is worse than LDLQ everywhere: the **first block's `down_proj`**, in both
models tested (L1 on Llama-3.2-1B, L2 on Qwen3-0.6B), at every bitrate, on every eval
distribution, and at every data budget. More and better data makes it *worse*
(+79.3% → +88.5% neutral), so it is not a starvation artifact. It always has the most
anisotropic `H_O` in the model (50% of trace in 10% of eigenvalues, against 18–23% at
healthy layers), and `secondorder.py` showed YAQA *raises* the true second-order error
there by 39% — so the sketch is genuinely misaligned at that layer, rather than the
quadratic model breaking down. Unexplained.

**But an earlier version of this note over-weighted it badly, and the correction matters.**
Averaging KL across the three layers sampled gave L1 a 44% share and produced "+30.2%
ungated, worse than LDLQ". In a real Llama-3.2-1B that linear is 1 of 16 x 7 = 112, so
its true share is ~1–3%:

| L1's share of model KL | ungated | gated | gate worth |
|---|---|---|---|
| 0.9% (1 of 112, equal weight) | −15.5% | −16.4% | 0.9 pts |
| 2% | −14.3% | −16.4% | 2.1 pts |
| 3% | −13.3% | −16.4% | 3.1 pts |
| 5% | −11.2% | −16.4% | 5.2 pts |

So **the gate is worth 1–3 points, not the difference between failure and success**, and
the "+30.2%" figure was an artifact of a three-layer sample. This also answers why the
paper never mentions such a layer: one bad linear in 112 costs a couple of points against
a −20% aggregate and is invisible in a whole-model KL number. It would only ever show up
in a per-layer sweep, which is not an experiment the paper had reason to run.

A gate exists if wanted, and needs no threshold tuning: round both ways, compute
`vec(Δ)ᵀ H vec(Δ) = E_b[⟨Δ, G_b⟩²]` on held-out gradients, keep the lower. `H_O = I` is
LDLQ exactly, so the fallback is free.

### Where it lands

At the paper's minimum data budget, on Llama-3.2-1B, K=2, single-layer:

- **−16% on neutral text, −19% in-domain**, healthy layers, tight bootstrap CIs.
- **~−14% to −15.5% whole-model ungated**, −16.4% with a per-layer gate.
- Plausibly **~−25% at the paper's full 64K-sequence budget**, per A.11's curve.
- Gains grow as bitrate falls, which is where the appliance lives.
- No inference cost, no format change: the plugin, kernels and `blockq` are untouched.

Against a build cost of: a reverse-streaming gradient pass the converter does not have,
2.8x the Hessian working set on dense models and 14x on wide MoE, a calibration corpus we
do not ship, and — if the last couple of points are wanted — a per-layer selection step.

**The decision is open.** This is one model, three layers, mostly one bitrate, measured a
layer at a time; the numbers are honest but they are not a whole-model conversion. What
would move it: the 8B (~20.5 GiB, one 24 GiB card), a full-budget sketch, or an
explanation of the first-block `down_proj`.


## What differs in EXL3's pipeline

### Incoherence processing is block-diagonal — real, asymmetric, and *not* the explanation

This was the leading candidate. It is a genuine structural difference, it does what the
theory says it should to `μ`, and **it still does not explain the gap.** Recording it in
full because it is the most plausible-sounding hypothesis available and it is now dead.

QuIP# and QTIP incoherence-process with a **full-dimension** randomized Hadamard
transform. EXL3 does not: `had_k = had_n = 128`, and
[`blockwise_preapply_had_*`](../deps/exllamav3/exllamav3/modules/quant/exl3_lib/quantize.py#L360)
applies a Hadamard **independently to each 128-wide block**. The sign flips are global,
the *mixing* is not — a 4096-wide output has 32 blocks that never mix. That block
structure is what makes EXL3 tensor-parallel-shardable at all (see
[tensor-parallel.md](tensor-parallel.md)), so it is not removable in production.

Theorem 3.4's advantage carries `μ_I² μ_O²`, and the weakness is asymmetric in exactly
the way that would matter: LDLQ's `H_O = I` is perfectly incoherent under any orthogonal
transform, so weak output-side mixing costs LDLQ nothing and costs YAQA the whole
output-side term. `probe.py --full-had-out` swaps in a full-width output Hadamard — which
breaks TP, but isolates the effect:

| layer.proj | `μ_O` block-128 | `μ_O` full width | `yaqa` block-128 | `yaqa` full width |
|---|---|---|---|---|
| L2.mlp.down_proj | 7.28 | 5.28 | +8.4% | +13.7% |
| L2.self_attn.o_proj | 5.52 | 5.29 | +3.7% | −7.0% |
| L26.mlp.down_proj | 7.42 | 4.84 | −5.9% | −5.7% |
| L26.self_attn.o_proj | 8.01 | 4.96 | −8.4% | −5.4% |

**`μ_O` improves consistently — roughly 7–8 down to ~5 — and YAQA's advantage does not.**
One layer improves a lot, one degrades a lot, two are flat; the mean change is under a
percent. Whatever is costing us the paper's magnitude, it is not the width of the
Hadamard, and the TP-128 constraint is not the thing standing in the way.

(Note the baselines move too, since changing `had_n` re-processes the weight for every
arm — `ldlq`'s own KL shifts by ±10%. The within-configuration `yaqa` vs `ldlq`
comparison is the meaningful one, and that is what the table reports.)

### Other candidates, none yet tested

- **The quantizer is QTIP-derived, not QTIP.** EXL3's 16×16 tile trellis, its codebook
  variants (`mcg`, `mul1`) and its Viterbi search over 256 elements are not what the
  paper's "QTIP quantizer" row used. How much rounding freedom is worth depends on the
  codebook's local geometry.
- **`H` is transformed with sign-only `su`.** `finalize_capture_H()` applies the initial
  ±1 signs, but `regularize()` then further scales the weight's input channels by
  `in_channel_scales` and by `g_scale` without `H` following. EXL3's `L_I` is therefore
  already an approximation of the correct factor, in both arms.
- ~~**`apply_out_scales` gives EXL3 a latent non-identity `H_O` it ignores.**~~
  Measured and closed: it is not a free win but a free *loss*, and the reason is that
  the omission is what makes `apply_out_scales` work. See
  [The ignored `H_O` is load-bearing](#the-ignored-h_o-is-load-bearing).
- **Production EXL3 calibrates on the quantized-prefix state.** Its `H_I` reflects the
  input distribution the *quantized* model will actually see. That is an upstream
  correction where `H_O` is a downstream one, so they should be complementary rather
  than overlapping — but the single-layer harness does not exercise it at all.

## What the 8B test costs

Measured, not estimated, on Llama-3.1-8B-Instruct geometry (`hidden 4096`,
`intermediate 14336`, 32 layers, `vocab 128256`) instantiated at reduced depth so the
per-layer term could be separated from the constant one:

| term | ctx 1024 | ctx 2048 |
|---|---|---|
| weights (bf16, 8.03B) | 14.96 GiB | 14.96 GiB |
| vocab-sized logit tensors + workspace | ~2.9 GiB | ~5.1 GiB |
| per checkpointed layer | 0.008 GiB | 0.016 GiB |
| `H_I` + `H_O` (`down_proj`, worst case) | 0.83 GiB | 0.83 GiB |
| **peak, worst-case target layer** | **~19.2 GiB** | **~20.5 GiB** |

Three things worth correcting or knowing:

- **It is not 2.8× weights.** That figure is the *converter's* per-layer Hessian working
  set in a production conversion, and it does not apply here. In this experiment the
  Hessians are 0.83 GiB — about 4% of peak. The cost is ~1.37× weights, dominated by
  holding the unquantized model plus the vocabulary-sized logit tensors.
- **The non-weight cost barely depends on model size.** It scales with `vocab × ctx`,
  and Llama-3.1-8B's 128256 vocabulary is *smaller* than Qwen3-0.6B's 151936. Depth is
  nearly free once activations are recomputed.
- **Gradient checkpointing is what makes it fit, and HF silently no-ops it in `eval()`
  mode.** Enabled but in eval, the per-layer term is 0.65 GiB at ctx 2048 — 21 GiB over
  32 layers, pushing peak to ~41 GiB. `model.train()` is required for it to engage, and
  is numerically identical here since Llama and Qwen blocks have no dropout.

**It does not need to fit on one device.** Only the target layer's weight and its two
Hessians must be co-resident (0.83 GiB); `device_map="auto"` shards layers across GPUs
and autograd crosses device boundaries transparently, with no collectives involved.
`--device-map` is wired up for this. So a single 24 GiB card is enough, and 2×16 GiB
would also do.

This workstation cannot run it on three counts: 16 GiB of VRAM against a 20.5 GiB floor,
12 GiB free disk against a ~16 GiB download, and 23 GiB of system RAM. The `vast`
8×3090 box recorded in notes is no longer reachable — its SSH host key has changed, so
the instance has been reassigned and needs re-provisioning.

## If it is picked up again

In the order that answers the most per hour. Nothing here is required reading before a
decision — [Where it lands](#where-it-lands) has the numbers.

0. **Re-run the headline against the baseline the converter actually ships**, i.e.
   `probe.py` with `apply_out_scales = True` instead of the `False` it hardcodes today.
   This is first because it is the only item that can *close* the project rather than
   advance it: `apply_out_scales` is worth up to 66% KL by itself and corrects the same
   axis YAQA does, so the −19% could be largely or entirely subsumed. It is also the
   cheapest thing on this list — same model, same corpus, this workstation, ~18 min per
   layer — because nothing new needs downloading: the RedPajama-V2 shard is already in
   the HF cache and the README's snippet re-extracts the flat `.txt` in seconds.
   Everything below is wasted effort if this comes back neutral.
1. **Llama-3.1-8B-Instruct**, the paper's own model, for a third point on a size trend
   that has been favourable throughout (0.6B → 1.2B roughly doubled the effect). Measured
   floor **~20.5 GiB at ctx 2048** — one 24 GiB card, or sharded, since it does not need
   a single device. Not this workstation. Meta gating was requested; the `unsloth` mirror
   is ungated and its config verifies against spec.
2. **A full-budget sketch.** A.11's curve runs −20.4% at 2K sequences to −26.3% at 16K.
   We ran 2K. The corpus is already downloaded (26.2M tokens of RedPajama-V2 English =
   12,807 unique rows at 2048); only GPU time is missing, and it scales linearly with
   sequence count.
3. **The first block's `down_proj`.** Reproducible in both models tested, worse with more
   data, the most anisotropic `H_O` in the model, and YAQA raises the true second-order
   error there. Worth understanding before shipping, or gating around.
4. Only then the converter work: the reverse stream, then the rounding change, which is
   the easy half and is already written and guarded.

**The cheap side-result was measured, and it is not there.** Restoring the output
metric `H_n D_sv² H_nᵀ` that `regularize()` creates and `ldlq()` drops costs up to +57%
KL rather than saving anything; the omission is load-bearing. See
[The ignored `H_O` is load-bearing](#the-ignored-h_o-is-load-bearing). What it left
behind is item 0 above — the same measurement showed `probe.py` has been scoring YAQA
against a baseline the converter does not use.
