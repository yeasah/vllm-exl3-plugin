# TurboQuant KV cache: page geometry, and what boundary protection buys

*Two subjects that surfaced together and are worth keeping apart. The first is a
correctness fix — a TurboQuant KV cache could not coexist with sliding-window
layers, and now can. The second is an optimization with a much wider blast
radius, applying to every TurboQuant model including dense ones.*

Measured on vLLM 0.28.0, single RTX 5070 Ti (16 GiB), 2026-08-29. Open work is
tracked as `turboquant-sliding-window` and `turboquant-boundary-tax` in
[TODO.md](../TODO.md).

## Part 1 — Why a sliding-window model could not serve

`--kv-cache-dtype turboquant_4bit_nc` with the sliding layers held native by
`--kv-cache-dtype-skip-layers` failed in KV cache group construction: first an
`AssertionError` inside `unify_kv_cache_spec_page_size`, and behind it
`NotImplementedError: page size is not divisible by the maximum page size`.
Three independent bugs, fixed in
[`vllm-tq-01-sliding-window-kv-pages`](../patches.md).

**The load-bearing one is a mispriced page.**
`Platform._align_heterogeneous_kv_block_size` prices the quantized primary
through `backend_cls`, which `_find_non_ssm_backend` defines as the backend of
the *first* attention layer. With skip layers that layer is usually an
unquantized one on FLASH_ATTN, whose `customize_spec` is a no-op for
TurboQuant's packing — so the primary is priced as dense uint8 (`2 * head_dim`
bytes/head) instead of its packed slot (`head_dim + 6`). The shared page is then
computed against a page ~1.9x too large, and because `2*hd / (hd+6)` is never an
integer, the real primary page can never divide it. Invisible from the error
message, and unfixable by any amount of work on the other two.

The remaining two are the ones upstream had already flagged in its own comments:
a **first/last-N sibling** (a full-attention skip layer had no way to pad up to
the shared page — that mechanism existed only for `SlidingWindowSpec`), and
**`page_size_padded` staleness** in `unify`, which scales `block_size` without
scaling the pad (narrowed to `AttentionSpec`, which is where the field is declared —
the base `KVCacheSpec` has no pad, so an unguarded access fails type checking).

Sequence on `Laguna-XS-2.1-exl3@3.00bpw` (40 layers, 10 full / 30 sliding at
window 512, 8 KV heads, head_dim 128):

| state | outcome |
|---|---|
| baseline | `AssertionError` — the `page_size_padded` staleness |
| + backend pricing fix | block 32→64, shared page 65536→**68608**; sliding and turboquant agree; layer 0 alone at 262144 |
| + sibling | all classes 68608 → **serves** |
| staleness one-liner alone | clears the assert, exposes the divisibility wall behind it |

The sibling is gated twice so nothing else moves: only when the native per-token
page is not an integer multiple of the primary's (nvfp4 is exactly 2x and keeps
reconciling by block scaling), and only when some layer is sliding-window or the
model is hybrid. That second gate matters — an all-full-attention model never
reaches `unify` at all (`UniformTypeKVCacheSpecs` takes it first) and packs
differing page sizes more tightly than padding does. Ungated, the sibling cost
**2.0% of KV tokens on dense TurboQuant**, a regression on a path that worked.

Not yet shown on gemma-4 or Muse-Glimmer: neither fits this card. gemma also has
a second geometry problem this fix does not address — `head_dim` 256 with 8 KV
heads on its sliding layers against `global_head_dim` **512** with **one** KV
head on its global ones, so the two types are 8192 and 2048 bytes/token natively.
See [vllm#41403](https://github.com/vllm-project/vllm/issues/41403).

## Part 2 — What boundary protection buys

`TurboQuantConfig.get_boundary_skip_layers` leaves the first and last two layers
on a native KV cache. Its docstring justifies the hard `n=2` with "Empirically
required for aggressive presets (k3v4_nc, 3bit_nc) — without it GSM8K drops ~30
points on Qwen3-4B."

### The claim reproduces exactly

`Qwen/Qwen3-4B` bf16, GSM8K 5-shot completion, greedy, full test set n=1319:

| config | acc | vs auto | KV bytes/token |
|---|---|---|---|
| auto (bf16 KV) | 86.81% | — | 147,456 |
| 4bit_nc, boundary on | 84.53% | −2.27 | 50,688 |
| 4bit_nc, boundary off | 78.01% | −8.79 | 38,592 |
| k3v4_nc, boundary on | 78.70% | −8.11 | 46,592 |
| k3v4_nc, boundary off | 48.90% | −37.91 | 33,984 |
| 3bit_nc, boundary on | 78.17% | −8.64 | 42,496 |
| 3bit_nc, boundary off | 46.25% | −40.56 | 29,376 |

Removing the skip, paired: k3v4_nc **−29.80** (440 lost / 47 gained, p=4e-81)
against the docstring's "~30 points". Not obsolete, not a mis-specified test.
3bit_nc **−31.92** (485/64). And **4-bit is not exempt**: **−6.52** (167/81,
p=5e-08).

### But layer 0 is the entire effect, and the trailing layers buy nothing

Decomposing the stock set `{0,1,34,35}`:

| native layers | 4bit_nc | share of effect | k3v4_nc | share | KV B/token (4bit) |
|---|---|---|---|---|---|
| `{}` | 78.01% | — | 48.90% | — | 38,592 |
| `{0}` | 83.17% | **79%** | 77.41% | **96%** | 41,616 |
| `{0,1}` | 83.62% | 86% | 79.30% | 102% | 44,640 |
| `{34,35}` | 77.63% | **−6%** | 48.67% | **−1%** | 44,640 |
| `{0,35}` (n=1) | 83.55% | 85% | 77.56% | 96% | 44,640 |
| `{0,1,34,35}` (stock) | 84.53% | 100% | 78.70% | 100% | 50,688 |

**The last two layers are a pure tax.** Protecting only them is
indistinguishable from protecting nothing — 4bit p=0.79, k3v4 p=0.92 — while
costing two layers of native cache. Protecting layer 0 alone recovers 79% / 96%
of the effect, and is itself statistically indistinguishable from full stock
protection (4bit `{0}` vs `{0,1,34,35}`: −1.36 points, **p=0.17**) at **18%
fewer bytes**. The likely physics is attention sinks / massive activations in the
first layer, which predicts exactly this asymmetry.

### The frontier, and what it means for the presets

Every measured Qwen3-4B point, `*` = Pareto-optimal:

| KV B/token | acc | | preset | native layers |
|---|---|---|---|---|
| 29,376 | 46.25% | * | 3bit_nc | `{}` |
| 33,984 | 48.90% | * | k3v4_nc | `{}` |
| 37,136 | 77.41% | * | k3v4_nc | `{0}` |
| 38,592 | 78.01% | * | 4bit_nc | `{}` |
| 40,288 | 79.30% | * | k3v4_nc | `{0,1}` |
| 41,616 | 83.17% | * | 4bit_nc | `{0}` |
| 42,496 | 78.17% | | 3bit_nc | `{0,1,34,35}` |
| 44,640 | 83.62% | * | 4bit_nc | `{0,1}` |
| 46,592 | 78.70% | | k3v4_nc | `{0,1,34,35}` |
| 50,688 | 84.53% | * | 4bit_nc | `{0,1,34,35}` |
| 147,456 | 86.81% | * | auto | `{}` |

**Both stock aggressive presets are dominated.** `3bit_nc` as it ships — 78.17%
at 42,496 — is beaten by `4bit_nc` protecting only layer 0 by **five points at
fewer bytes**. `k3v4_nc` as it ships is beaten by the same configuration by 4.5
points at 11% fewer bytes. Whatever byte budget an operator picks the aggressive
presets for, a 4-bit cache with less boundary protection serves it better.

**And none of those better configurations can be expressed.**
`--kv-cache-dtype-skip-layers` only ever adds to the automatic list
(`existing | set(boundary)` in `arg_utils.py`), so there is no invocation that
reduces or retargets boundary protection. `get_boundary_skip_layers(model_config,
n=2)` already takes the parameter; the sole call site hardcodes it.

### Which layer the boundary set actually protects, per model

Boundary protection is `{0, 1, L-2, L-1}`, but on a sliding-window model most of
those are sliding layers the operator is already holding native. What is left is
one full-attention layer, and *which* one differs:

| model | layers | boundary set | full-attn layer protected | position |
|---|---|---|---|---|
| Laguna-XS-2.1 | 40 | `{0,1,38,39}` | `{0}` | **first** |
| gemma-4-12B | 48 | `{0,1,46,47}` | `{47}` | **last** |
| Muse-Glimmer-30B | 52 | `{0,1,50,51}` | `{51}` | **last** |
| Qwen3.8-27B | 64 | `{0,1,62,63}` | `{63}` | **last, and saturated** |

**Qwen3.8-27B adds a third case, where the lever has no resolution at all.** It is hybrid
with `full_attention_interval: 4`, so its KV-bearing layers are 3, 7, 11 ... 63 and the
boundary set grows inward from two ends that are both linear attention. Protected
KV layers by `n`: **n=0 -> none; n=1 -> {63}; n=2 -> {63}; n=3 -> {63}**. The lever
saturates at n=1, so n=1, 2 and 3 are the same configuration and only 0-vs-1 is a real
choice — one layer of sixteen, 6.25% of KV, ~550 tokens on a 9K-token budget and easily
inside page rounding. Confirmed from practice 2026-09-01: sweeping `boundary:0..2` on this
model produced no change in effective KV tokens, which is what the structure predicts for
1 and 2 and very nearly predicts for 0.

**The general form**: on a hybrid model the boundary set is indexed over *all* layers
while only some carry KV, so `n` buys protection in steps of "however many full-attention
layers happen to fall within n of an end" — which for an interval of 4 is one, then zero,
then zero. Any model with `full_attention_interval > 2` will show the same saturation.
Read the intersection, not the set.

Read against the isolation above, that is a per-model recommendation rather than
a general one. On **gemma-4 and Muse the protected layer is the last one** — the
position measured as buying nothing (p=0.79 at 4bit, p=0.92 at k3v4) — and their
layer 0, the position that does carry the effect, is a sliding layer that stays
native regardless. So `boundary:0` on those models is predicted near-free.

Measured on Muse-Glimmer-30B @2.50bpw, `turboquant_4bit_nc`, 4.37 GiB of KV,
full 131,072 context, releasing layer 51:

| | KV tokens | max concurrency at 131k |
|---|---|---|
| stock boundary | 751,263 | 5.73x |
| `boundary:0` | **862,096** | **6.58x** |

**+14.8% for one layer**, at unchanged weights and unchanged KV memory.

### Laguna shows none of this

Same measurement on `Laguna-XS-2.1-exl3@3.00bpw` (chat CoT, n=1319): 4bit
92.49% on / 92.34% off (p=0.89), 3bit **1215/1319 both ways**, 26 flips each
direction (p=1.00). Two candidate explanations were proposed, and the table
above disfavours the first. "Only one layer is at stake" does not explain it,
because on Laguna that layer is **layer 0** — the one carrying 79-96% of the
effect on Qwen3-4B. If the count were the reason, most of the effect should have
survived; none of it did. What remains is that Laguna is a far less perturbed
system (10 of 40 layers ever compressed, against 32 of 36 on Qwen3-4B), or that
the sensitivity is model-specific. Distinguishing those needs a second dense
model, not a second sliding-window one.

### Long context, and weight quantization: the full matrix (2026-08-30)

`Qwen/Qwen3-4B` is 40,960 native with `rope_scaling: null`, so a bf16 reference *and* an
unquantized 32K KV cache both fit on a 16 GiB card and **no YaRN appears anywhere** — the
only axes are the ones being measured. Context length is swept with the shot count (192
tokens/shot on GSM8K), which holds task, metric and scoring fixed; prefix caching pays the
long prefill once. A 3.00bpw EXL3 conversion of the same model (`head_bits 6`, bundled
calibration) supplies the weight axis. First 500 problems throughout, so every column is
paired.

| weights | KV | ~0.7K | 8.2K | 32.8K |
|---|---|---|---|---|
| bf16 | auto (bf16 KV) | 88.0% | 89.0% | 86.8% |
| bf16 | tq4 + protection | 86.2% | 88.2% | 86.4% |
| bf16 | tq4 − protection | 78.2% | 84.0% | 82.6% |
| 3bpw | auto (bf16 KV) | 75.4% | 79.8% | 76.6% |
| 3bpw | tq4 + protection | 73.0% | 77.0% | 74.4% |
| 3bpw | tq4 − protection | 68.2% | 72.2% | 70.0% |

**Boundary protection does not become more valuable at long context — it becomes less.**
Paired McNemar, significant in all six cells (p from 4e-2 to 2e-5):

| | ~0.7K | 8.2K | 32.8K |
|---|---|---|---|
| bf16, cost of dropping protection | **−8.0** | −4.2 | **−3.8** |
| 3bpw, cost of dropping protection | −4.8 | −4.8 | −4.4 |

That contradicts the attention-sink prior stated above, which predicted the first layer
should matter *more* as sequences lengthen. On bf16 the cost halves from 0.7K to 32.8K
(~1.6 sigma, suggestive rather than conclusive since the two prompts differ); on 3bpw it is
flat. Neither shows growth. The long-context worry that was deferring the shipping default
does not reproduce on this task.

**The tq4 cost itself is small and never significant here** (auto → tq4+protection: −0.4
to −2.8 points, p = 0.13 to 0.87 across six cells). All six have the same sign, which a
sign test puts at p=0.03, so the effect is real and just below this n's resolution.

**Compounding is at most mild.** The tq4 cost is consistently larger on 3bpw weights
(−2.2 to −2.8) than on bf16 (−0.4 to −1.8), but no individual comparison is significant.
Consistent with the roughly-additive prediction from the same-direction superposition
result in [qbench.md](qbench.md), with a hint of mild super-additivity that this n cannot
resolve. Weight quantization dominates either way: bf16 → 3bpw costs 9-13 points against
the KV configs' 0.4-8.

**The caveat is now the most important thing in this section.** Many-shot GSM8K's long
context is 175 near-identical exemplars, and *redundancy is exactly what makes a task
robust to losing part of its context*. So the observed shrink may be a property of the
task rather than of length — a model that can answer from any of 175 examples does not
care much which ones the cache damaged. This is the result that most needs a second
instrument, and a needle/RULER-style retrieval probe is the one that would not share the
weakness. **Until that runs, "protection matters less at long context" is a statement
about many-shot prompts, not about long context.**

### Retrieval: the needle probe the many-shot result demanded (2026-08-30)

Many-shot GSM8K is redundant and barely stresses retrieval, so its "protection matters
less at long context" needed an instrument without that weakness. Needle-in-a-haystack
on the same six configurations, wikitext-103 haystack, identical trials across cells.

**Three variants saturated before one bit.** On Qwen3-4B a single needle scores 93-100%
everywhere with no ordering by KV quality (bf16 `auto` at 32K scored 144/150 against
bf16 tq4-*without*-protection at 148/150); 8 needles with distinct keys scored 200/200;
8 same-key needles 197/200. Verbatim recall of rare numeric strings is close to free.
What finally bites is **16 same-key needles at 32K**, where sixteen indistinguishable
spans must be held apart across the full context.

| weights | KV | 8K | 32K |
|---|---|---|---|
| bf16 | auto | 97.5% | 88.4% |
| bf16 | tq4 + protection | 97.0% | 87.5% |
| bf16 | tq4 − protection | 96.2% | 86.2% |
| 3bpw | auto | 97.4% | 86.2% |
| 3bpw | tq4 + protection | 96.7% | 86.9% |
| 3bpw | tq4 − protection | 96.9% | **84.1%** |

100 trials x 16 needles = 1600 paired items per cell. **tq4 itself is free on retrieval**
— `auto -> tq4+protection` is −0.9% to +0.7% and never significant in four cells.
Dropping protection costs −0.8%, −1.3%, +0.2%, −2.8%, and **only the last is significant**
(3bpw @32K, p=0.009) — the cell where both stressors combine.

**This corrects the many-shot reading in the direction the caveat predicted.** There, the
cost of dropping protection *shrank* with length (−8.0 → −3.8). Here it grows (bf16 −0.8 →
−1.3, 3bpw +0.2 → −2.8). The 8K cells sit at 96-97% with little room, so part of that is
ceiling; but where there is headroom, the effect is present rather than absent.

**And retrieval is the less sensitive task, which was not the expectation.** At comparable
headroom (GSM8K ~85%, needle ~86%) dropping protection costs 3.8-8.0 points on GSM8K
against 1.3-2.8 here. The damage layer 0 prevents looks like general degradation of the
computation rather than a failure to reach distant context — consistent with an
attention-sink role, and a caution against assuming retrieval benchmarks are the sensitive
instrument for KV quality.

**A harness bug worth recording, because it nearly became a finding.** The first hard-needle
sweep ended its prompt with "Give the numbers only.", which a base-style completion
*continues* rather than obeys: the model emitted ` Do not include any other text.\nAnswer:`
and then restated every fact in full sentences, so every generation hit the token cap
(`finish_reason: length`). The metric was scoring verbosity against that cap, and it
produced bf16 `auto` at 70.8% against bf16 tq4 at 95.9%, **p=7.7e-69** — strongly
significant and physically impossible. Ending the prompt at `\nAnswer:` and sizing the cap
to the answer moved that same cell to 97.5%. The tell was implausibility, not weak
statistics; all twelve cells of that sweep were discarded.

### Controls and limits

- **Decoding is bit-reproducible.** The same config run twice: 1220/1319 both
  times, **zero** discordant items. So every per-item flip between two
  configurations is the KV change, not batching, and the paired tests are clean.
- **The perturbation is real but unbiased on Laguna.** ~52 items (4%) flip in
  every Laguna comparison, symmetrically (27/25, 26/26). Not invisible —
  directionless.
- **Sensitivity ~1.2 points** on the Laguna nulls: with ~52 discordant items the
  exact test needs a 34/18 split to reach p<0.05.
- **All of this is short context** — ~700-token prompts on Qwen3-4B, ~1.3k on
  Laguna. The reason to compress KV is long context, and both KV damage and any
  first-layer effect plausibly grow with sequence length. No long-context
  measurement exists yet, and it is the one that should decide a shipping default.
- **MiniCPM5-1B is not a usable instrument** and its numbers are excluded: an
  unquantized KV cache scores *below* a 3-bit one there (12.8% vs 16.0%), so it
  has no signal to lose. phi4mini would not load; both Qwen3.x checkpoints are
  hybrids and already exempt from boundary skips.

### Does tq4 compose safely with aggressive weight quantization? (2026-08-30)

The question this work started from, and the one vLLM's own study cannot answer because it
evaluates KV quantization against unquantized weights.

**At 4 bits, composition is additive.** The cost of adding tq4 is the same whether the
weights are bf16 or 3.00bpw, pooled over five task/context combinations (GSM8K at
0.7K/8.2K/32.8K, needle at 8K/32K): difference-of-differences **−0.09%, 95% CI −1.1% to
+0.9%**. Per-cell estimates wobble in both directions with no consistent sign. That is also
what the same-direction superposition result in [qbench.md](qbench.md) predicts — weight
and KV quantization are degradations pointing the same way.

**At the aggressive presets it is not.** Raising the KV perturbation ~4x makes the
interaction appear immediately (GSM8K 5-shot, n=1319, cost against an unquantized cache):

| preset | bf16 weights | 3.00bpw weights | interaction |
|---|---|---|---|
| 4bit_nc | −2.3% | −2.7% | −0.5 [−3.4, +2.5] n.s. |
| k3v4_nc | −8.1% | **−15.0%** | **−6.9 [−10.5, −3.3]** |
| 3bit_nc | −8.6% | **−15.1%** | **−6.4 [−10.0, −2.9]** |

So on a 3bpw model the aggressive presets cost roughly **twice** what they cost on bf16
weights, and the excess is a genuine interaction rather than a sum of parts. Absolute
damage compounds hard: bf16+auto 86.8% → 3bpw+k3v4_nc 59.7%, of which ~12 points is the
weights, ~8 would be additive KV, and ~7 is the interaction.

**The second table is the positive control for the first**, which is what licenses
reading the 4-bit result as additive rather than as a failure to measure. The same
instrument, the same 1319 problems and the same paired test detect compounding
unambiguously when it is there (p<0.001 at both aggressive presets), so its silence at
4 bits is a measurement. Compare the needle probe above, whose first three variants
saturated at 93-100%: those nulls were worth nothing because sensitivity had never been
demonstrated. Any null reported here should carry a control of this shape.

**Two consequences.** First, tq4 is safe to compose with aggressive EXL3 — the reassuring
result, now with a bound rather than an absence of evidence. Second, **published KV-quant
evaluations on unquantized weights understate the risk for quantized deployments**: vLLM's
study recommends avoiding k3v4_nc and 3bit_nc based on bf16-weight numbers, and on a 3bpw
model the penalty is about double what those numbers imply. Anyone serving an EXL3 or
GGUF-quantized model should read published KV-quant results as a lower bound.

It also strengthens the Pareto argument above on exactly the models that matter here: at
3.00bpw, `4bit_nc` *without* boundary protection scores 65.1% against `k3v4_nc` *with* it
at 59.7% — **5.4 points better at 17% fewer bytes**, where on bf16 weights the two were
tied.

### Independent corroboration, and what it leaves open (2026-05-11)

vLLM published their own TurboQuant study
([vllm.ai/blog/2026-05-11-turboquant](https://vllm.ai/blog/2026-05-11-turboquant)) over
Llama-3.3-70B, Qwen3-30B-A3B (Instruct and Thinking) and MiniMax-M2.7 — dense and MoE,
30B to 200B+, against our 4B. It reaches the same split we did the hard way:

- **Long-context retrieval (openai/mrcr) tolerates 4-bit** and breaks only at the
  aggressive presets — k3v4_nc 33.5% AUC against bf16's 45.8% on Qwen3-30B at 256k,
  3bit_nc 31.2%.
- **Reasoning is the sensitive axis** — ~20 point drops on AIME25 and LiveCodeBench-v6
  for k3v4_nc and 3bit_nc on Qwen3-30B-Thinking.
- **Their recommendation**: 4bit_nc viable under memory pressure at 1-4 points;
  **avoid k3v4_nc and 3bit_nc in production**; fp8 remains the better choice wherever 2x
  capacity is enough.

That retrieval-vs-reasoning asymmetry is exactly what the needle and GSM8K instruments
here disagreed about, on a model two orders of magnitude smaller, which is worth more than
either result alone.

**And their study prices the whole approach, which the summary above skipped on first
reading.** Throughput is a major section, not a footnote, and its conclusion is
*"lower KV-cache storage cost does not directly translate into faster serving"*:

| | Qwen3-30B-A3B (2xH100) | Llama-3.3-70B (4xH100) |
|---|---|---|
| fp8 | matches bf16 | matches bf16 |
| `turboquant_k8v4` | 80% of bf16 | 75% |
| `turboquant_4bit_nc` | ~77% | 75% |
| `turboquant_k3v4_nc` | ~75% | ~70% |
| `turboquant_3bit_nc` | 73% | **66%** |

Latency overhead runs to 60% (Qwen3-30B) and 10-68% (Llama-70B) across batch 1/8/32/64,
where fp8 is negligible. The stated mechanism is that **TurboQuant dequantizes to bf16
before attention, and that cost grows with the amount of KV accessed** — so the penalty
scales with exactly the context length the compression is bought for.

**But it is not one-dimensional.** Under burst load on Llama-70B the capacity wins on
tail latency anyway: P99 TTFT is ~17 s at bf16, under 3.5 s for every TurboQuant preset,
and ~1.3 s at fp8. Steady-state throughput and admission under pressure are different
questions, and this project has measured neither locally — every figure in this note is
capacity. See `turboquant-boundary-tax` in [TODO.md](../TODO.md).

**What their study does not cover, and this one does.** Their four models are all
full-attention, so the sliding-window page-geometry failure in Part 1 never arises for
them. Their quick-start notes that first/last layer skipping happens, but carries no
per-layer sensitivity analysis — so the layer-0 result above, and the finding that the
trailing layers buy nothing, is not something their evaluation would have surfaced.

**And the two results compose into the upstream argument.** They recommend avoiding the
aggressive presets. We measure that the byte budget those presets are chosen for is better
served by 4bit_nc with less boundary protection — which their own flag cannot express. So
the ask is not to relax a guard they have just published evidence for; it is to make
reachable the configuration that dominates the branch they are recommending against.

### Not TurboQuant: CJK tokens spliced into long sessions (2026-10-24)

A 135k-token agent session on Ornith-1.5-9B (4.00bpw, `turboquant_4bit_nc`, an RTX 3060) degraded late
into foreign-script tokens spliced mid-word ("the修罗", "are我吗", "ex多7B-exl3") and, at worst, runs of
garbage. Its first regenerations pointed at TurboQuant: fp8 and NVFP4 responses came out clean. They were
one or two samples each. Replaying the server's logged request exactly (`--enable-log-requests` at debug
level gives the JSON body), n=8 per arm, on this host:

| KV cache, exact request | spliced tokens per 1k chars |
|---|---|
| bf16 (no quantization) | 0.19 |
| nvfp4 | 0.18 |
| fp8 | 0.15 |
| turboquant_4bit_nc (two seeds) | 0.13, 0.13 |
| turboquant_4bit_nc, `top_k=20` | 0 |

Independent of the cache; it reproduces on stock vLLM 0.32.0 with the bf16 weights at TP=2, so neither
EXL3 nor our patches either. The cause is sampling: the harness sent no parameters, and the model ships no
`generation_config.json`, so the server sampled at temperature 1.0 with no truncation, which reaches a
tail holding ~0.9% of the mass per step outside the top 20. Ornith's tail is ~25-30% heavier than its
parent Qwen3.5-9B's at both 135k and 1k context (bf16 KV) and splices more often, but Qwen3.5 does it
too; context length moves the tail only ~30%. The card's recommended sampling (`top_k=20`) removes it,
and `quant.py upload` now publishes that as each revision's `generation_config.json`.

**What is TurboQuant's here:** decode speed. At 135k context the session decoded at 15 tok/s under
`turboquant_4bit_nc` against 38 (fp8) and 36 (nvfp4) on the same card, falling steadily from 28 tok/s at
45k. A splice rate is also the wrong instrument for a KV cache: n=1-2 responses per arm cannot separate
0.13 from 0.19 per 1k chars. The record is in `~/tq-conv` and `quant_work/_tqconv`.

## Reproducing

The harness is [tools/gsm8k_kv.py](../tools/gsm8k_kv.py); per-item results for
every run above are in [docs/data/turboquant-kv/](data/turboquant-kv/), so the
tables re-derive without a GPU:

    tools/gsm8k_kv.py report 'docs/data/turboquant-kv/qwen_*.json'
    tools/gsm8k_kv.py report 'docs/data/turboquant-kv/iso_turboquant_4bit_nc_*.json'
    tools/gsm8k_kv.py report 'docs/data/turboquant-kv/full_laguna_*.json'

Boundary control is a replacement of `get_boundary_skip_layers`, which
`EngineArgs.create_engine_config` calls in the driver process before any engine
subprocess exists — so it lands on the real code path, and each run prints the
engine's resulting `kv_cache_dtype_skip_layers` to prove which layers were
skipped. There is no CLI for this; that is the finding.

Qwen3-4B, the dense grid (~2 min/arm):

    export MML=2048 UTIL=0.93
    for kv in turboquant_4bit_nc turboquant_k3v4_nc turboquant_3bit_nc; do
      for b in on off; do
        tools/gsm8k_kv.py run Qwen/Qwen3-4B $kv $b 1319 qwen_${kv}_${b}.json fewshot
      done
    done
    tools/gsm8k_kv.py run Qwen/Qwen3-4B auto on 1319 qwen_auto_on.json fewshot

The layer isolation:

    for spec in layers:0 layers:0,1 layers:34,35 layers:0,35; do
      tools/gsm8k_kv.py run Qwen/Qwen3-4B turboquant_4bit_nc "$spec" 1319 \
          iso_4bit_${spec//[:,]/_}.json fewshot
    done

Laguna, which needs its 30 sliding layers held native and a smaller context to
fit 12.6 GiB of weights on a 16 GiB card (~20 min/arm, near-serial decoding):

    export MML=1280 UTIL=0.93 MAXTOK=448
    export SKIP_SLIDING=$(python -c "print(','.join(str(i) for i in range(40) if i%4))")
    tools/gsm8k_kv.py run ~/ckpt/Laguna-XS-2.1-exl3-3.00bpw-bq \
        turboquant_4bit_nc on 1319 full_laguna_4bit_on.json chat

The determinism control is just the same invocation twice to different output
files. It is worth re-running whenever the sampling path changes: every paired
p-value above assumes it.

### A reproduction model that needs no plugin (2026-08-31)

Every model in this document is EXL3-quantized, which is no use to anyone reproducing the
Part 1 bug from a stock vLLM checkout. **`unsloth/gemma-3-1b-it`** is the smallest clean
reproducer found:

- **1.86 GiB** in bf16, so it fits anywhere, and it is an ungated mirror (google's own
  Gemma repos are gated, as are Cohere's Command-R7B).
- **Text-only**, so it never reaches the multimodal gate that blocks gemma-4 in
  [vllm#41403](https://github.com/vllm-project/vllm/issues/41403).
- **Interleaved sliding/full**: 26 layers, `sliding_window_pattern: 6`, so full attention
  at layers 5, 11, 17, 23 and a 512-token window elsewhere.
- **Uniform head geometry** — `head_dim` 256 on both layer types, unlike gemma-4's
  `global_head_dim` 512 against `head_dim` 256, which is a separate unfixed problem.
- `Gemma3ForCausalLM` is supported by stock vLLM 0.28.

Other candidates and why not: `EXAONE-4.0-32B` is interleaved with ideal uniform geometry
but 59.6 GiB; `Ministral-8B-Instruct-2410` is interleaved at 14.9 GiB, too tight to leave
KV room on 16 GiB; gemma-4-E2B is small and interleaved but multimodal *and* carries the
split head_dim; Qwen2.5, Phi-4-mini, SmolLM3 and AFM-4.5B are all effectively
full-attention.

Reproduction, holding the 22 sliding layers native because TurboQuant cannot serve a
sliding window:

    sliding = [str(i) for i in range(26) if (i + 1) % 6 != 0]
    LLM(model="unsloth/gemma-3-1b-it", kv_cache_dtype="turboquant_4bit_nc",
        kv_cache_dtype_skip_layers=sliding,
        max_model_len=2048, gpu_memory_utilization=0.60, enforce_eager=True)

On stock 0.28.0 that fails in KV cache group construction:

    NotImplementedError: Layer model.layers.5.self_attn.attn: page size is not
    divisible by the maximum page size and cannot be padded.

on layer 5 — a TurboQuant layer — which is the mispriced-primary bug. With
`86e3b1e` in `deps/vllm` it serves, at 264,171 KV tokens.

**Note which half this exercises.** gemma-3-1b's boundary indices `{0,1,24,25}` are all
sliding layers that were already held native, so no *full-attention* layer keeps a native
cache and the sibling hunk never fires. To exercise that half, add a full-attention layer
to the skip list — appending `"5"` makes 23 skip layers and still serves (278,150 tokens),
where it is the sibling padding that reconciles the resulting third page class.

## Where this goes upstream

Two separable pieces, in order.

1. **The page-size fixes.** Policy-free correctness, no default moves.
2. **A lever for boundary protection**, drafted as
   [`vllm-tq-02-boundary-lever`](../patches.md).
   The argument is a reachability gap
   rather than a request to relax a conservative default: the configurations on
   the frontier cannot be expressed today. The mechanism with the least new
   surface is exposing the `n` that already exists, as a keyword in
   `--kv-cache-dtype-skip-layers` — that flag already carries a keyword
   vocabulary (`sliding_window`), matched by plain membership test, and its whole
   purpose is which layers keep a native cache. It also requires fixing the
   `key=int` crash (`sorted(existing | set(boundary), key=int)` raises on the
   documented `sliding_window` keyword), which is a standalone bug.

   Verified end to end on Laguna, where the effective skip list is printed back
   by the engine:

   | invocation | resulting native layers | KV tokens |
   |---|---|---|
   | `sliding_window` | `0, 1, 38, 39, sliding_window` | 1,697 |
   | `sliding_window boundary:1` | `0, 39, sliding_window` | 1,697 |
   | `sliding_window boundary:0` | `sliding_window` | 1,732 |

   `boundary:N` is symmetric on purpose. Asymmetric protection needs no new
   syntax, because layer indices are numbered from the front and so stay portable
   across models: `--kv-cache-dtype-skip-layers 0 boundary:0` resolves to `['0']`
   on any model, which is the best-measured configuration above. A front/back
   `boundary:N,M` form would add a grammar dimension whose only unique
   contribution is a portable spelling for *back*-anchored protection — the half
   measured here as buying nothing. Worth revisiting only if long context shows
   the trailing layers doing something.

   The middle row is the sliding-window case in miniature: `boundary:1` drops
   layers 1 and 38, both of which are sliding and already native, so capacity does
   not move. Only layer 0 is ever at stake on this model, which is the same fact
   that makes Laguna's quality nulls unsurprising in hindsight.

   New presets would be the most invasive option despite feeling like the
   lightest: each costs ~5 registration sites (`TQ_PRESETS`, a `KVQuantMode`
   member, `STR_DTYPE_TO_TORCH_DTYPE`, the backend's `supported_kv_cache_dtypes`,
   the `CacheDType` literal), and a cross-product would enshrine configurations
   measured here at 46-49%.

A default change — `n=1`, or dropping the trailing layers, both of which the
isolation supports — needs long context and more than one model first. Note that
[vllm#41403](https://github.com/vllm-project/vllm/issues/41403) currently
presents monkeypatching `get_boundary_skip_layers` to `[]` as a free gemma
workaround; on a dense model that costs 6.5 points at 4 bits and 30 at k3v4, and
that is worth telling them.

## A worked 27B configuration, and why each awkward flag is there (2026-09-02)

The first configuration to serve `Qwen3.8-27B` at 4.00bpw usefully on a 16 GiB card —
**73728 tokens**, from a starting point where the same checkpoint with a dense embedding
did not load at all. Kept in `~/ckpt/run-qwen3.8-27b.sh`. Three of its flags are
counterintuitive enough that a future reader will be tempted to "fix" them, so the
reasoning is recorded with them.

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True vllm serve \
  Qwen3.8-27B-exl3-4.00bpw-bq \
  --max-num-seqs 1 --max-model-len auto --language-model-only \
  --kv-cache-dtype turboquant_4bit_nc --enable-prefix-caching \
  --performance-mode interactivity \
  --max-num-batched-tokens 512 --cudagraph-capture-sizes 1 2 4 \
  --kv-cache-memory=1577189376 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
```

**The checkpoint is `-bq`, and that is what makes any of this possible.** A
block-quantized embedding takes 4.00bpw from 15.70 GiB of weights to 14.00 — the
embedding is 1.70 GiB at this model's 248320 x 5120, and at 15.92 GiB of card the dense
version does not load. See [qbench.md](qbench.md): the quality cost is ~1/9 of the harness
noise floor.

**`--kv-cache-memory=1641152512` rather than `--gpu-memory-utilization`.** vLLM's memory
profiler is unreliable here: `--max-num-batched-tokens 256` consistently over-allocates
and OOMs under the profiler while 128 profiles fine, though the true difference is 61 MiB
against a much larger margin. Pinning bypasses the profiler. **Do not guess the value** —
a "safe" 1 GiB costs a third of the context. Derive it: take the
`--kv-cache-memory=X` that vLLM *prints* at a configuration which profiles successfully,
then correct by the only term that changes, `(chunk_new - chunk_old) x vocab x 2` bytes.
At vocab 248320 that is 61 MiB per doubling from 128, i.e. ~3.0K tokens.

**`--cudagraph-capture-sizes 1 2 4` is in the command line but is not doing anything**,
and the reasoning that put it there was wrong (corrected 2026-09-02). vLLM already bounds
the capture list by concurrency: `vllm.py:1885` sets
`max_graph_size = min(max_num_seqs * decode_query_len * 2, 512)`, which at
`--max-num-seqs 1` is **2**, so the default list is `[1, 2]`. Passing `1 2 4` therefore
*adds* a capture size rather than trimming seventeen, and cannot have been what let chunk
512 start — the pin dropped 61 MiB in the same change, and that was the real fix. Harmless
to leave, but it buys nothing.

**`--max-num-batched-tokens 512` is a logits-buffer control, not a throughput knob.** The
buffer scales `chunk x vocab`, and at this vocabulary it dominates every other activation
term. Dropping it from the default took usable context from ~9K to ~46K. The cost is
prefill latency — 256 makes a full-context prompt ~300 sequential forward passes — so the
extreme is not the right point. Measured at 512 with the capture list trimmed: **3072 tokens given up (-4%) to take
full-context prefill from 300 sequential passes to 144**.

**That is stale as of vLLM 0.29 — the logits buffer is no longer sized by the chunk
(checked 2026-09-12).** The profile run computes logits only at
`logit_indices = cumsum(num_scheduled_tokens) - 1`, one row per request
([gpu_model_runner.py:6305](../deps/vllm/vllm/v1/worker/gpu_model_runner.py#L6305)), so
the tensor it profiles is `max_num_seqs x vocab` — 1 MiB at `--max-num-seqs 1` — not
`chunk x vocab`. Measured on the 3.00bpw config at `gpu_memory_utilization=0.975`,
2048 -> 512 moves peak activation 0.80 -> 0.77 GiB and KV capacity 227,094 -> 230,169
tokens: **+3,075 tokens, where `chunk x vocab x 2` predicts ~40K.** The arithmetic that
failed is version-independent, so the ~9K -> 46K result above belongs to the version it
was measured on and does not transfer. Treat the knob as a prefill-latency control and
nothing else. Most of that 0.77 GiB turned out not to be activation at all — the same
configuration profiles 0.18 GiB once the torch.compile cache is warm or off — so the
vocabulary was never the term worth chasing here; see
[memory-accounting.md](memory-accounting.md).

**CUDA graphs are on, and they are not the memory hazard they are assumed to be.**
Capture reported **0.04 GiB**, and enabling them *raised* usable context 46K -> 67K:
`enforce_eager` was the more expensive path on this model, presumably via a more
conservative activation profile. Do not "save memory" by going eager here.

**`--max-num-seqs 1`** keeps the linear-attention recurrent state at 0.074 GiB. This is a
hybrid model (`full_attention_interval: 4`, 16 of 64 layers carry KV), and that state is
*per sequence* — 16 concurrent would cost 1.2 GiB, which this budget does not have.
Related: `boundary:N` is inert on this model, see the per-model table above.

**`--language-model-only`** drops the vision and audio towers, which nothing in a
text-serving path reads.

**Measured KV rate: 21,369 bytes/token** (1.528 GiB / 76800), i.e. **5.22 bits/element**
effective across 16 x 2 x 4 x 256 elements — 4-bit payload plus scales. Note this is
*not* the `head_dim + 6` slot width used elsewhere in this note for page-geometry
arithmetic; predicting capacity from that model overestimates by 57%. Use the measured
rate for capacity work.

**Untested, and the most likely remaining failure:** a logprobs request allocates roughly
`positions x vocab` and is absent from the startup profile. With KV pinned it cannot grow
at runtime, so the remaining OOM candidates are all request-shaped. The adversarial test
is cheap and worth running deliberately: fill the KV to capacity, then issue a long
request asking for logprobs.

### The configuration that superseded it: full declared context at 3.00bpw (2026-09-12)

The section above is kept because its reasoning about each flag still holds, but it is no
longer the configuration to copy. At 3.00bpw the same model now serves **its full declared
262144 context** — 264,993 tokens in the cache, auto-fit reporting no reduction — on the
same 16 GiB card:

```
VLLM_DISABLE_COMPILE_CACHE=1 EXL3_BLOCKQ_ON_LOAD=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True vllm serve \
  turboderp/Qwen3.8-27B-exl3 --revision 3.00bpw \
  --max-num-seqs 1 --max-model-len auto --language-model-only \
  --kv-cache-dtype turboquant_4bit_nc --enable-prefix-caching \
  --performance-mode interactivity --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.975 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
```

What changed is not this note's subject — it is that TurboQuant's prefill stopped
allocating anything sized by context, which freed 1580 MiB at the peak. Three things about
the command are worth knowing before editing it, all established in
[memory-accounting.md](memory-accounting.md) "The payoff, and the term that was hiding
behind it":

- **`--gpu-memory-utilization 0.975` is load-bearing and hand-found.** Nothing derives it;
  auto-fit cannot see the consumers that decide whether it is safe. That is TODO
  `kv-budget-margin`.
- **`VLLM_DISABLE_COMPILE_CACHE=1` is load-bearing too**, which is not obvious. A *cold*
  compile cache inflates profiled peak activation by 0.59 GiB and costs 35K tokens of
  declared context; disabling the cache, or having it already warm, both avoid it.
- **`--kv-cache-memory=` is not needed any more.** The pin in the 4.00bpw command above
  existed to bypass an unreliable profiler; the utilization path now reaches the ceiling on
  its own, and **it serves rather than merely declaring**. At 0.975: a 257,549-token prompt
  — 98.2% of the declared context — completes at **665.65 t/s of prefill**
  (`profile-completion.py`, `live` mode, so 387 s to first token). At **0.985**, after the
  reserve declaration and the scratch pooling: **242,420 tokens in 355.1 s**, 92.5% of
  capacity. That matters because
  "profiles successfully, then dies at first inference" is the failure this ground keeps
  producing, and the KV cache being full is when it would happen.
  **The proof is behavioural, not accounted**: no profiled peak exists at that length,
  because `torch.cuda.memory._record_memory_history` OOMs the *host* at that many
  allocations (see [memory-accounting.md](memory-accounting.md), method caveats). What
  supports it besides the completion is that the ~100K profiles were flat — nothing in
  them grew with prompt length once the continuation prefill stopped allocating by
  context. The logprobs adversarial test above is still outstanding and is a different
  shape of risk: request-sized, not context-sized.


## Moved from TODO.md (2026-11-05)

*The working record of these items as they stood when TODO.md was cut back to open tasks, kept verbatim. Open work on them, if any, is in TODO.md or FUTURE.md under the same slug or the one named.*

### `mtp-turboquant` — Zero draft acceptance, and output that does not survive it

The outcome wanted is either a working MTP drafter on a TurboQuant KV cache, or a
refusal at startup. What happens today is the third thing: it runs, drafts at ~90
tok/s, accepts **nothing**, and the output is degenerate.

It unblocks the appliance's only speculative-decoding option on the model it most
wants one for. Qwen3.8-27B at 3.00bpw with a 4-bit KV cache is the configuration the
16 GiB card is built around, and MTP is the one draft method the checkpoint ships.

**Measured 2026-09-10**, serving `turboderp/Qwen3.8-27B-exl3@3.00bpw` with
`--kv-cache-dtype turboquant_4bit_nc` and `--speculative-config
'{"method":"mtp","num_speculative_tokens":3,"revision":"3.00bpw"}'`, one-line prompt:

- **0.0% acceptance across 912 drafted tokens**, per-position 0.000/0.000/0.000, after
  2.9% (0.047/0.034/0.007) in the first window. Position 0 is the draft that depends
  on nothing but the target's own hidden state, so 4.7% falling to 0.000 says the
  drafter is wrong from its first proposal rather than degrading down the chain. (A
  useful MTP head is far higher than that; we have no measured figure of our own for
  this checkpoint, which is itself worth fixing before tuning anything.)
- **Degenerate output**: an unclosed thinking block that generates forever with
  nothing reaching the client. With 0% acceptance a correct implementation would
  reproduce the target model's own output exactly, so *something beyond the drafter
  is affected*. **The draft model is required for it**: dropping `--speculative-config`
  and resizing the KV cache to a comparable size gives a correct short story, same
  model, same `turboquant_4bit_nc`, same context budget (2026-09-10). So the quantized
  KV path alone is fine at this context and the fault is in the draft/verify cycle.
- vLLM says so itself at startup: `Fused multi-step draft decode is not supported by
  attention backend(s) TURBOQUANT; falling back to rebuilding attention metadata
  between draft steps` (`speculator.py`). The backend declares
  `_init_reorder_batch_threshold(1, supports_spec_as_decode=False)`, so a 4-query
  verification step is classified as a *prefill* rather than a decode.

**The 2x2 is complete, and both controls point the same way.** TQ without spec decode
is correct, so the KV path alone is fine; fp8 *with* MTP is correct on the same 0.29
install, so the drafter and the verification machinery are fine. Only the crossing
fails. The question is no longer whether the draft cycle is at fault, nor whether MTP
is broken generally -- it is what TurboQuant does that fp8 does not, during
verification.

**Two readings of the code, one of which kills the obvious hypothesis.** The path a
4-token verification batch takes is the small-continuation branch of
`TurboQuantAttentionImpl.forward`, and it *does* handle causality: each query token is
issued as its own decode request with an incrementing synthetic `seq_len`
(`_arange_cache[cached_len + 1 : seq_len + 1]`), so token `j` attends to exactly
`cached_len + j + 1` keys. "The verification batch is unmasked" was the first guess and
it is wrong. Note the precondition stated in the comment above it, though: the branch
assumes `do_kv_cache_update` has already written all `q_len` tokens to the cache -- so
during verification, *rejected drafts are in the KV cache before they are rejected*.
That is the seam worth probing next. Against it: the store takes a per-token
`slot_mapping` and the metadata region is indexed per slot, so a rejected token's slot
should be cleanly reusable rather than poisoning a block-shared scale -- which is the
mechanism one would reach for first, and it does not obviously hold here.

**Do not take a clean `bench/` capture as evidence either way.** Speculative decoding
is output-preserving by construction, so a broken drafter and a working one produce
identical tokens; `qwen3.8-27B-3.0bpw-blockq-MTP-tq4` captured cleanly for exactly that
reason and its note claimed too much for years. Any measurement here has to read
acceptance rate, which means `disable_log_stats=False`.

*A second finding from the same run, worth separating*: `--speculative-config` on a
hybrid model silently switches Mamba-state checkpointing to **dense**
(`arg_utils.py`: "Hybrid model with EAGLE speculative decoding: defaulting
prefix_cache_retention_interval to dense checkpointing"), because EAGLE's tail-block
drop makes the sparse retention unreachable. That is what puts KV occupancy at 40.6%
for a one-line prompt on a 53,040-token context, and it is a large hidden cost of
turning MTP on for this architecture regardless of whether acceptance is ever fixed.

### `kv-nvfp4` — 4-bit KV without TurboQuant's costs

**Outcome:** NVFP4 KV (`--kv-cache-dtype nvfp4`, FlashInfer with XQA decode on SM12x since vLLM
0.32) measured against `turboquant_4bit_nc` on KL and on capacity, and against TurboQuant's
limitations: the boundary-layer tax, sliding-window models, FA3, TP and spec decode.

**Unblocks:** possibly ~4-bit KV at TurboQuant-like quality on the paths TurboQuant refuses or
taxes. Upstream's own test case is Qwen3.8-27B at TP=2 (GQA 6, head_dim 256), our local config.

**Candidate:** the existing TurboQuant comparison harness (docs/turboquant-kv.md) with one more
arm, on the 0.6B and the 9B; then the 27B at TP=2. UltraQuant (`ultraquant_4bit`, FP4 with
power-of-two group scales and rotated keys) is the ROCm-only sibling; its Triton paths are
portable if NVFP4 disappoints.

### `turboquant-sliding-window` — TurboQuant KV cache for sliding-window models

*Written up in [docs/turboquant-kv.md](turboquant-kv.md), Part 1. What follows is
the diagnostic history, kept because two of the three walls were misdiagnosed on the
way.*

**Status 2026-08-29: Laguna serves, patched.** The three fixes are in
[`vllm-tq-01-sliding-window-kv-pages`](../patches.md)
and the measurement is at the end of this item. Still open: the quality question
(`qbench` on tq4), gemma-4 and Muse-Glimmer (neither fits the local card), and
reporting any of it upstream. The history below is kept because it is what the
diagnosis cost, and because two of the three walls were misdiagnosed on the way.

`TurboQuantAttentionBackend` never overrides `supports_sliding_window`, so it takes
the base class's `return False` and is rejected for any model with a sliding window.
That is **gemma-4, Laguna and Muse-Glimmer** — one reason, three families.
Muse-Glimmer is 39 sliding / 13 full at a 2048 window with `head_dim` 128, so like
Laguna it carries no head-dim triton pin and is blocked on the sliding window alone.

**And the rejection is a fix, not merely an obstacle.** Reported 2026-08-25: Muse
*used* to be accepted with a turboquant cache dtype and is not any more. It could
never have been correct — the model has been sliding-window throughout — so
whatever ran before was serving wrong attention silently. 0.28 refusing it is the
gate upstream should always have had, which is worth remembering when arguing for
the sliding-window support: the ask is to make it work, not to relax the check.

**Bigger than the FA miss it was mistaken for.** Closing the head-dim-512 gap would
have bought gemma-4 a flash-attention path and nothing else (retired, see
[docs/kernels.md](kernels.md)); this buys gemma-4 *and* Laguna a quantized KV
cache, which on a 16 GiB card decides whether the context is usable.

**The economics are better than the layer count suggests**, which is what makes it
worth pursuing rather than filing as impossible. gemma-4 is mostly sliding layers
(12B: 40 sliding / 8 full; 26B-A4B: 25 / 5), so compressing only the full-attention
layers sounds like giving up most of the prize. It is not: a sliding layer's cache is
capped at `sliding_window` (1024) while a full-attention layer grows with context, so
the five-or-eight full layers *dominate* KV bytes exactly where pressure is real —
~62% of KV at 8k on the 26B, ~86% at 32k, ~96% at 128k. Compressing only those
reaches ~65% total KV reduction at 32k against ~75% for compressing everything.

**There is precedent for the approach, and it is bypass rather than support.** The
external `turboquant-vllm` project ([alberto.codes,
2026-03-31](https://alberto.codes/blog/2026-03-31-from-one-model-to-seven-making-turboquant-model-portable))
reached Gemma-2/Gemma-3 by *declining to compress* sliding layers — "compressing a
sliding window layer's cache breaks the eviction semantics" — keeping layer indices
aligned with `None` padding. Not teaching the cache sliding window; detecting sliding
layers and leaving them alone. **vLLM already has that mechanism**:
`--kv-cache-dtype-skip-layers` takes the literal keyword `sliding_window`
(`layers/attention/attention.py:306`) and sets those layers to `kv_cache_dtype="auto"`.

**Measured 2026-08-24 on gemma-4-12B@3.00bpw_mul1, and it does not currently work:**

- **Unforced, the failure looks FA-related and is not.** `TRITON_ATTN is not valid
  ... ['kv_cache_dtype not supported']` — gemma-4 is pinned to triton by head dim
  512, triton rejects the turboquant dtype, and nothing mentions sliding window. This
  is the message that produced the original wrong diagnosis.
- **`--kv-cache-dtype-skip-layers sliding_window` crashes**, and it is one line:
  turboquant merges its own boundary skips with `sorted(existing | set(boundary),
  key=int)` (`engine/arg_utils.py:1985`), so the documented keyword raises
  `ValueError: invalid literal for int() with base 10: 'sliding_window'`. The exact
  invocation implementing the blog's approach cannot be typed.
- **Skipping the same layers by numeric index changes nothing**, because backend
  selection is global: vLLM picks one backend for the model and that backend must
  accept the configuration. Per-layer dtype skipping cannot route around it.
- **Forced to turboquant, three blockers, not one:** `['kv_cache_dtype not
  supported', 'partial multimodal token full attention not supported', 'sliding
  window not supported']`. The third is the known one. The second is
  multimodal-specific and means gemma-4 needs more than sliding-window support. **The
  first is unexplained** — `supports_kv_cache_dtype` returns true for anything
  `turboquant_*` and the full `turboquant_4bit_nc` reaches the backend. Suspicion, not
  a finding: turboquant auto-adds boundary layers to the skip list, those become
  `"auto"`, and global selection then validates turboquant against a native-dtype
  group. **Settle this before reporting anything upstream.**

So the `key=int` collision is a real, small, reportable bug on the path that matters,
but fixing it alone would not get gemma-4 to turboquant — the multimodal blocker sits
behind it.

**Laguna is the better playground, and it gets furthest.** Text-only (no multimodal
blocker), `head_dim` 128 (no triton pin, so no head-dim confound), and 10 full / 30
sliding layers at a 512-token window — half gemma's. That layout makes the bypass
*more* attractive, not less: the ten full-attention layers are 73% of KV bytes at 4k,
84% at 8k, **95.5% at 32k** and 98.8% at 128k, so compressing only them captures ~95%
of the theoretical maximum.

Measured 2026-08-24 on `Laguna-XS-2.1-exl3@3.00bpw`:

- `--kv-cache-dtype turboquant_4bit_nc` alone: `No valid attention backend found for
  cuda` — turboquant is the only backend accepting the dtype and it rejects the
  sliding window, so nothing is left.
- **Skipping the 30 sliding layers by index passes backend selection** and fails much
  deeper, in KV cache group construction:
  `get_kv_cache_groups -> unify_kv_cache_spec_page_size -> assert
  self.page_size_padded >= self.unpadded_page_size_bytes` (`kv_cache_utils.py:1118`,
  `kv_cache_interface.py:207`).

**That first wall is a one-line staleness bug, confirmed by patching it.** The unifier
scales `block_size` by the ratio via `replace(layer_spec, block_size=new_block_size)`
and leaves `page_size_padded` untouched; turboquant sets that field for its packed
`slot_size_aligned` layout, so the recomputed `unpadded_page_size_bytes` overtakes the
stale padded value. Scaling both cleared it.

**Behind it is a second wall that is not small.** With the assert gone, the same run
fails on a full-attention (turboquant) layer:

    NotImplementedError: Layer model.layers.4.self_attn.attn: page size is not
    divisible by the maximum page size and cannot be padded.

Padding is gated on `indexes_kv_by_block_stride`, which is *derived* from the
backend's `get_kv_cache_stride_order()` (`gpu_model_runner.py:7833`) — a property of
the memory layout, not a flag to set. TurboQuant packs K+V into a single interleaved
slot per head per position, so its page size is neither divisible into a native
layer's nor paddable to match it.

**So the accurate statement is narrower than "TurboQuant cannot do sliding window" and
wider than "one function":** TurboQuant's packed KV page layout cannot currently
coexist with native-dtype layers in one cache. That is precisely the bookkeeping the
external project wrote by hand *outside* vLLM, and it is the thing any bypass approach
has to solve — the sliding-window rejection is only the first gate.

**And upstream has already restructured it, after our pin.** vLLM's
`[N/N] KV-Cache Layout Refactor` series has parts 1-3 in v0.27.0 and parts **4 and 5
landing after it**:

- `61874f9842` [4/N] Promote local KV cache specs via a class-changing replace helper
  (#51612) — rewrites `kv_cache_utils.py`, the file holding the unifier.
- `57bd0ed441` [5/N] Backend-published KV packing via `customize_spec` (#51704) —
  touches `turboquant_attn.py` directly, and **deletes `TQFullAttentionSpec`**.

TurboQuant stops being a special spec class carrying an opaque `page_size_padded` and
becomes an ordinary `FullAttentionSpec` publishing `state_content_bytes =
slot_size_aligned` through a backend hook. Page size then computes as
`num_heads * storage_block_size * state_content_size_bytes` — a **per-cell** quantity,
so it scales linearly with `block_size` where the old padded value did not.

**Re-run on v0.28.0, 2026-08-25, and the prediction was half right.** TurboQuant does
stop setting `page_size_padded` — it publishes `state_content_bytes` only. But the
walls did not dissolve, and 0.28 turns out to have built *first-class machinery for
this exact case* with a gap in it:

- **Stages 1 and 2 fail exactly as on 0.27.0** — bare dtype gives no valid backend, the
  `sliding_window` keyword still raises `ValueError: invalid literal for int()`.
- **Stage 3 progresses further, then hits the same assert** (`page_size_padded >=
  unpadded_page_size_bytes`), now reached through cudagraph memory profiling rather
  than direct KV init. The staleness bug moved rather than vanishing: in 0.28 it is the
  *sliding* spec that carries a padded page (`page_size_padded=shared_page`) while the
  unifier's block-scaling branch still leaves it stale. Same one-line fix, different
  spec.
- **With that patched, the divisibility wall returns**, and instrumenting the unifier
  says why — there are **three** page-size classes, not two:

```
layer 0: FullAttentionSpec   block=32 page=131072 padded=None   content=None quant=0
layer 1: SlidingWindowSpec   block=16 page=65536  padded=65536  content=None quant=0
layer 4: FullAttentionSpec   block=32 page=34304  padded=None   content=134  quant=7
max_page_size=131072  distinct=[34304, 65536, 131072]
```

Layer 0 is *native full attention* because TurboQuant auto-adds its own boundary skip
layers (`get_boundary_skip_layers`, first/last N) on top of the operator's list. So the
pool holds native-full, native-sliding and turboquant-full, and 34304 does not divide
131072.

**`--kv-cache-dtype-skip-layers` is a supported configuration in 0.28**, not a hack:
`CacheConfig.skip_page_size_padded` is documented as "the page size of layers skipped
from KV cache quantization ... so unquantized skip layers pad up to the quantized
primary's page", and `Platform._align_..._block_size` bumps `block_size` so the primary
page covers the padded one. It handles **one** padded class. Upstream marked the gap
itself, twice, in that same function:

```python
# To add the first/last-N sibling:
#   padded_pages.append(per_token_page_bytes(<sibling_dtype>, "auto"))
# To add the first/last-N sibling:
#   cache_config.sibling_page_size_padded = shared_page
```

The first/last-N sibling *is* TurboQuant's boundary protection. So what blocks Laguna
is a case upstream has already identified and left unimplemented, plus the block-scaling
staleness bug — two narrow, reportable things rather than a missing capability.

**The boundary-protection question is filed separately** as
`turboquant-boundary-tax` below: the patch here keeps that protection working, so
removing it is an independent optimization across all TurboQuant models rather than
anything this item depends on.

**Answered 2026-08-29, and Laguna serves.** The answer to the bounded question was
yes, but the two known fixes were not the load-bearing one. See
[`vllm-tq-01-sliding-window-kv-pages`](../patches.md)
for all three.

- **The primary page was priced by the wrong backend.**
  `_align_heterogeneous_kv_block_size` prices the quantized primary through
  `backend_cls`, which `_find_non_ssm_backend` defines as the backend of the *first*
  attention layer — with skip layers, an unquantized one on FLASH_ATTN, whose
  `customize_spec` is a no-op for TurboQuant's packing. So the primary was priced as
  dense uint8 (2·hd bytes/head, 2048/token) instead of packed (hd+6, 1072/token). Since
  `2·hd/(hd+6)` is never an integer — 1.91 at hd 128, 1.95 at hd 256 — the shared page
  could never be a multiple of the real primary page, and no amount of sibling work
  would have fixed that. The sibling function `_align_hybrid_block_size` already
  special-cases TurboQuant for this exact reason and says so in its own comment.
  Resolving the backend that serves the primary dtype moves block_size 32 → 64 and the
  shared page 65536 → 68608, which is the packed page exactly.
- **The sibling then does what upstream's comments say**, with two gates: only when the
  native page is not an integer multiple of the primary's (nvfp4 is exactly 2× and
  keeps its block-scaling path), and only when some layer is sliding-window or the model
  is hybrid. The second gate matters — an all-full-attention model never reaches
  `unify` at all (`UniformTypeKVCacheSpecs` takes it first) and packs differing page
  sizes more tightly than padding does. Ungated, the sibling cost **2.0% of KV tokens on
  dense TurboQuant**, which is a real regression on the path that already worked.
- **The `page_size_padded` staleness one-liner is real but no longer on the path.**
  Applied alone against the baseline it clears the assert and exposes the divisibility
  wall behind it, reproducing the history recorded above; with the other two fixes the
  pages reconcile and `unify` returns before the scaling branch.

Measured on `Laguna-XS-2.1-exl3@3.00bpw`, `turboquant_4bit_nc`, 30 sliding layers
skipped by index, RTX 5070 Ti / vLLM 0.28.0. All three page classes reconcile to 68608
— turboquant layers exact at block 64, native full and sliding padded from 65536 at
block 16 (4.5% waste) — and the model generates. **Boundary protection is intact**: the
first/last-N layers keep native KV, so nothing was traded for this. Dense TurboQuant
(MiniCPM5-1B) is unchanged at 909,888 vs 909,920 KV tokens, and both models still serve
with an unquantized cache.

**Not yet shown on the other two families.** gemma-4 and Muse-Glimmer do not fit the
16 GiB card (Muse OOMs during load; its native path additionally fails earlier on an
unrelated `vision_adapter.c_fc.mul1` weight-loading error, and it is served through
`--model-impl transformers` here). Both need the vast box. gemma-4 also still has the
multimodal blocker in front of it, which this does not touch.

The `key=int` crash on the literal `sliding_window` keyword (`arg_utils.py:2022`) is
untouched and still stands — the runs above skip layers by numeric index.

**There is an upstream tracking issue for the gemma-4 half of this**, found 2026-08-29:
[vllm-project/vllm#41403](https://github.com/vllm-project/vllm/issues/41403), "TurboQuant
+ Gemma 4 multimodal: 5-gate blocker stack" (open). Same destination, four gates in
common, and it is worth reading before writing anything upstream.

What it gives us:

- **The multimodal blocker has a workaround.** `--hf-overrides
  '{"text_config":{"use_bidirectional_attention":null}}'` clears
  `partial multimodal token full attention not supported` at the cost of vision
  quality. That is the blocker filed here as sitting behind everything else, and it
  turns out to be steppable for a text-only measurement.
- Two pieces of trivia that will cost an afternoon otherwise: TurboQuant's triton
  kernels need `ninja` on the path, and an installed external `turboquant-vllm` plugin
  collides with the in-tree API (`TQ4FullAttentionSpec.__init__() got an unexpected
  keyword argument 'tq_slot_size'`).
- Their Gate 2 is our boundary-skip question, and their workaround is exactly the one
  considered here: monkeypatch `get_boundary_skip_layers` to return `[]`. So the
  approach has independent users, and our patch is the alternative that keeps the
  protection rather than trading it away.

Where we are ahead: their Gate 5 stops at `NotImplementedError ... cannot unify by
adjusting block_size` and attributes it to gemma's heterogeneous head_dim. That is a
real second cause, but it is not the one that stops Laguna — the mispriced primary page
is, and it is invisible from the error message. Their Gate 2 diagnosis ("needs per-layer
attention backend routing") is also stale for 0.28: per-layer routing already happens
here, FLASH_ATTN for the skip layers and TURBOQUANT for the rest, which is precisely how
the aligner ends up asking the wrong backend for the primary's packing.

**And gemma is harder than Laguna in a way this item had not recorded.** Its full and
sliding layers do not share a KV geometry: `head_dim` 256 with 8 KV heads on the sliding
layers, `global_head_dim` **512** with `num_global_key_value_heads` **1** on the global
ones (plus `attention_k_eq_v: true`). So natively the two layer types are 8192 and 2048
bytes/token — the heterogeneity the issue names, and a *fourth* page class that the fix
above does not address: `padded_pages` computes one native per-token page from
`model_config.get_num_kv_heads()`, which cannot be right for both. The Laguna fix is
necessary but not sufficient for gemma.

It also re-prices gemma's prize, favourably: the global layers are only 2048 bytes/token
against 8192 for sliding, so the fixed sliding cost is ~335 MiB total (capped at the
1024 window) while the global layers are what grows — 2.1 GiB at 128k, which tq4 takes
to ~540 MiB. That is the same number the issue arrives at independently for the 31B.

Worth pairing with the quality question before investing: tq4 on a trellis-quantized
model has field experience and a capability-benchmark lower bound behind it, but no
`qbench` numbers. That is the measurement that decides whether this is *practical*
rather than merely running.

**This item now carries the gemma-4 dependency, and its odds have improved enough to
change downstream priorities** (2026-08-26). Two things moved. First, the blocker was
misidentified: it was filed as a likely-insurmountable flash-attention gap (head dim
512, since retired -- see [docs/kernels.md](kernels.md)), and it is actually this — a KV-cache layout problem whose walls have so
far all reduced to one-line staleness bugs, one of them already cleared by patching.
Second, the scope is three families and not one: gemma-4, Laguna and Muse-Glimmer are
all sliding-window, which is **every real-world candidate except Qwen**. A fix is
therefore load-bearing for most of the candidate pile rather than for one demoted
family, and it is now better read as *more likely to be resolved than not*. (Resolved
for Laguna on 2026-08-29; see the measurement above.)

What that unblocks: gemma-4 goes back to being a genuine serving candidate, so the
shared embed+head tensor deferred under `quantized-embeddings` has a **real
constituency rather than a hypothetical one**. The severe priority demotion that both
gemma and every tied-model optimization inherited was downstream of the misdiagnosis,
and should be unwound with it. The divisibility wall — the remaining unknown when this
was written — turned out to be three narrow bugs and is cleared. What is left is the
quality question below, and whether gemma-4's multimodal blocker yields.

**`bench/` does not cover this, and the 0.29 bump showed what that costs.** Every
TurboQuant entry in the matrix is MiniCPM5-1B, which has no sliding window, so the
page-geometry patch is verified only by hand against the `unsloth/gemma-3-1b-it`
reproducer in the note. That model is the reason to fix it rather than an obstacle:
1.86 GiB, ungated, text-only, `Gemma3ForCausalLM` on stock vLLM, and it fails loudly
on an unpatched tree — so an entry holding the 22 sliding layers native gates the
patch, and appending full-attention layer 5 gates the first/last-N sibling, which
nothing exercises today. Cheap enough for the `fast` tier.

→ [docs/kernels.md](kernels.md)

### `turboquant-boundary-tax` — What TurboQuant's first/last-N protection costs, and when it pays

Split out of `turboquant-sliding-window` on 2026-08-29: that item is a correctness fix
and is answered, this is an optimization across every TurboQuant model including dense
ones, and nothing depends on it — the sibling patch keeps boundary protection working.

**Measurements are written up in
[docs/turboquant-kv.md](turboquant-kv.md)** (Part 2), with the harness in
[tools/gsm8k_kv.py](../tools/gsm8k_kv.py) and per-item results in
[docs/data/turboquant-kv/](data/turboquant-kv). In short, on Qwen3-4B at n=1319:

- The docstring's claim **reproduces exactly** — k3v4_nc loses 29.80 points against its
  "~30 points on Qwen3-4B". 4-bit is not exempt either: −6.52, p=5e-08.
- **Layer 0 is the whole effect.** Protecting `{34,35}` is indistinguishable from
  protecting nothing (p=0.79 / 0.92); protecting `{0}` alone is indistinguishable from
  full stock protection (p=0.17) at 18% fewer bytes.
- **Both aggressive presets are dominated as they ship.** `3bit_nc` is beaten by
  `4bit_nc` protecting only layer 0 by five points *at fewer bytes*.
- **None of the better configurations can be expressed** — the flag only ever adds to
  the automatic skip list.
- Laguna shows none of it, and that anomaly is unexplained.

Open:

- **Long context.** Everything measured is ~700-1300 token prompts. The reason to
  compress KV is long context, and both KV damage and any first-layer effect plausibly
  grow with length. This is the measurement that should decide any default change, and
  it does not exist yet.
- **A second dense model**, to know whether "layer 0 only" is a property of Qwen3-4B or
  of transformers. The attention-sink explanation predicts it generalises.
- **TurboQuant's throughput cost is unmeasured here, and it is large.** Every number in
  this project is KV *capacity*; none is tokens/s. vLLM's own study — already cited in
  the note — measures 66-80% of bf16 throughput depending on preset and model, with
  latency overhead to 68%, because TurboQuant dequantizes to bf16 before attention and
  that cost grows with the KV accessed. It also shows the trade is not one-dimensional:
  P99 TTFT under burst *improves* 5x over bf16. For an appliance that trade is the whole
  decision, and we have no local number on either axis. Cheap to add to the harness.
- **And whether it generalises across *quantizers*, not just models.** If attention
  sinks are the mechanism, the effect is a property of the model and constrains any KV
  compression scheme — which makes the finding worth more than one vLLM patch. E8-lattice
  KV caches claiming ~2 bits are now shipping from at least two other projects
  (see the field notes); none of the public claims say what they hold native at layer 0.
- **Why Laguna is flat** — one full-attention layer at stake versus four, or 30 of 40
  layers native regardless? Cheaply separable by compressing Laguna's sliding layers too
  once that path exists, or by testing a dense model of Laguna's depth.
- **The upstream lever is drafted** as
  [`vllm-tq-02-boundary-lever`](../patches.md): the
  `key=int` fix plus `boundary:N` in `--kv-cache-dtype-skip-layers`, default unchanged
  at 2. Verified end to end. What is left is filing it — as a reachability gap rather
  than a defaults change, with the Qwen3-4B frontier as the motivation.
- **Tell [vllm#41403](https://github.com/vllm-project/vllm/issues/41403)** that
  monkeypatching `get_boundary_skip_layers` to `[]` is not free — it presents that as a
  costless gemma workaround, and on a dense model it costs 6.5 points at 4 bits.

→ [docs/turboquant-kv.md](turboquant-kv.md)
