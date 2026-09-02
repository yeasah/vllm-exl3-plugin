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
[patches/vllm-tq-01-sliding-window-kv-pages.patch](../patches/vllm-tq-01-sliding-window-kv-pages.patch).

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
`vllm-tq-01-sliding-window-kv-pages.patch` it serves, at 264,171 KV tokens.

**Note which half this exercises.** gemma-3-1b's boundary indices `{0,1,24,25}` are all
sliding layers that were already held native, so no *full-attention* layer keeps a native
cache and the sibling hunk never fires. To exercise that half, add a full-attention layer
to the skip list — appending `"5"` makes 23 skip layers and still serves (278,150 tokens),
where it is the sibling padding that reconciles the resulting third page class.

## Where this goes upstream

Two separable pieces, in order.

1. **The page-size fixes.** Policy-free correctness, no default moves.
2. **A lever for boundary protection**, drafted as
   [patches/vllm-tq-02-boundary-lever.patch](../patches/vllm-tq-02-boundary-lever.patch).
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
