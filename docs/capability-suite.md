# Measuring capability through the served path

Evidence and rationale behind TODO `capability-suite`. qbench answers how far a
quantized distribution sits from its reference; this note is about the separate
question of whether the served configuration still *does the work*, and about how such
a comparison has to be built.

The analysis code lives in [capability-suite/](../capability-suite/).

Two eras of evidence. The first half is SWE-bench through `mini-swe`, 2026-08. The
second is GPQA-Diamond through `evalscope` and through `opencode`/harbor, 2026-09,
which is where the instrument itself turned out to be most of the measurement.

## The pilot: 23 problems, and what it could and could not show

*2026-08-21. SWE-bench-Lite **dev** split, 23 problems. Local
`Qwen3.8-27B-exl3-3.00bpw-bq` with `tq4` KV -- 3-bit weights, block-quantized
embedding and 4-bit KV, all three axes at once -- against `qwen3.8-27b` fp8 via
OpenRouter. Both arms completed all 23.*

| | resolved |
|---|---|
| local, 3.00bpw + bq + tq4 | **12 / 23** |
| cloud fp8 baseline | 10 / 23 |

Paired, which is the only view that means anything on identical problems:

| | count |
|---|---|
| both resolved | 9 |
| local only | 3 |
| cloud only | 1 |
| neither | 10 |

**19 of 23 agree.** The margin lives in four discordant pairs split 3-1, exact
McNemar two-sided *p* = 0.625. Local reproduced **9 of the baseline's 10 successes**.
Bootstrap 95% CI on the difference: **[-8.7%, +26.1%]**.

**What that does and does not establish.** It cannot distinguish "equal" from
"modestly better" -- but it *excludes* degradation worse than about 9 percentage
points against a 43% base rate, roughly a 20% relative floor. A configuration actually
broken by quantizing all three axes would show -43pp. Ruling that out is a real
result; the difference is not.

**And the design could not have produced a significant result whatever happened.** At
~17% discordance, 23 problems bought four informative pairs, and even a clean 4-0
split gives *p* = 0.125. The effective sample size of a paired comparison is the
discordant count, not the problem count. That is the single most useful number to plan
against: budget ~150 problems for ~25 discordant pairs, ~300 for ~50.

**The comparator was the weak instrument, not the subject.** "Advertised fp8, 256K"
says nothing about KV dtype, speculative decoding or sampling defaults, and three
instances came from a different provider entirely. That uncontrolled variance
*inflates* disagreement, so 19/23 agreement across two unrelated serving stacks is a
conservative reading rather than a flattering one. A controlled comparison wants the
same model unquantized on the same stack, and Qwen3.8-27B needs ~27 GiB even at fp8 --
so it is a rented-hardware job.

**A behavioural difference pass/fail cannot see.** The baseline's *unresolved*
trajectories ran longer than its resolved ones (91 vs 82 median turns); the quantized
model's ran shorter (77 vs 82). Cloud thrashes before failing, local quits sooner.
Weak at n≈12 per cell, but it is the kind of signal worth a metric: a model that knows
it is stuck behaves differently from one that does not.

**Marginal statistics actively mislead here.** On the same data, aggregate turn
medians said the quantized model used *fewer* turns (77 vs 91) while the paired
comparison said the opposite (+4 median delta, local lower on only 10 of 23). Compare
pairs, never margins.

## The truncated full run: a benchmark that was 93% django

*2026-08-21 to 08-23. Full SWE-bench Lite, cloud arm on a 2x GPU vast rental serving
`Qwen/Qwen3.8-27B-FP8` at TP=2, harness running locally. Killed 31.5 hours in by loss
of the host.*

139 instances attempted, 92 `Submitted`. The 47 failures decompose into three
unrelated causes, which matters because only one of them is about the model's
environment at all:

| cause | n | what it was |
|---|---|---|
| remote API death | 25 | 23 `BadGatewayError` + 2 `APIError`, contiguous from 31.5h |
| local docker | 19 | 12 `TimeoutExpired` + 7 `CalledProcessError`, all `docker run` (exit 125 or 120s startup timeouts), all matplotlib images |
| harness | 1 | `LimitsExceeded`; a test container died and the agent polled a corpse until the turn cap |
| earlier remote blips | 2 | `Timeout`, same HTML-error-page signature as the death, at 16.9h and 22.7h |

**The failure was abrupt, established three ways.** The server log's final ticks show
one request completing normally at 27 tok/s, KV dropping to 0%, a clean idle interval,
and then the host vanishing between two 10-second ticks -- ssh reset in the same
instant. No CUDA error, no NCCL timeout, no engine crash, which is what a GPU leaving
the bus mid-inference would have produced. Afterwards NVML counted two devices and
could not return a handle for one, so the dead card is a consequence or co-symptom of
a host-level event, not the initiating cause. Independently, the completed
trajectories show no behavioural drift: paired turn deltas against the local run on
the same instances are noise across every quartile (median -9, +1, -18, +9, with the
cloud arm exceeding local on roughly half the instances throughout). Turn counts *do*
rise through the run (77 -> 88 -> 106 -> 117 median), but the local control reproduces
the same rise on the same instances, so it is a property of instance ordering rather
than of the hardware.

Consequence: excision is sufficient. No boundary judgement is needed and all 92
completed results are usable.

**The larger problem was not the hardware.** SWE-bench Lite is ordered by repository,
and the run died deep in the django block: the 92 completed instances are 6 astropy
and 86 django. Whatever comes out of it is a statement about django, not about Lite --
a single codebase with a single set of conventions. That limits generalisation far
more than *n* does, and it is entirely avoidable by shuffling the instance list with a
fixed seed shared across arms, so that any prefix is a representative sample.
Truncation is not an edge case: rentals die, budgets end, and runs get stopped by
hand.

Shuffling does not fix the benchmark's own concentration -- Lite is roughly 38% django
and 26% sympy, so two repositories are two-thirds of it. That is a standing argument
for a more diverse suite, against which sits the constraint below.

## Discriminating power peaks near 50% resolved

Discordant pairs carry the entire signal, so a suite on which both arms fail together
is nearly useless regardless of size. At 12/23 the pilot sat almost exactly at the
optimum. A harder and more diverse suite -- `multilingual`, say -- trades coverage
against detectability: if the model resolves 15%, discordance collapses and the
instance count needed for the same number of informative pairs rises several-fold. The
right choice is the most diverse suite on which the model still lands near half, not
the hardest one available.

## Context exhaustion is a harness failure, and needs its own policy

*Amended 2026-09-18 for Q&A benchmarks — see "A runaway on a Q&A item is model
behaviour". What follows holds for agentic benchmarks, where a real harness would
compact and continue; it does not hold where running to the limit is the model failing
to converge.*

A bench with no context management measures itself as much as the model. When a request
outgrows the window, a real agent harness would compact and continue; this one fails, and
the failure is not even labelled consistently -- hitting the limit on the *prompt* raises
an explicit error, while hitting it on *output* is not terminal (the agent is told to be
less chatty and retries) and surfaces later under an unrelated category. Both are the same
event.

Scoring them as model failures is wrong on its own terms, too: spending more of the
context window is a legitimate strategy, and "it should have finished sooner" is not a
claim the harness is entitled to make.

**The policy, for a paired comparison:** if both arms hit context on the same instance,
it is a null result -- drop the pair, which turns it into the increased sampling error it
actually is. If exactly one arm hits it, the pair needs manual inspection and does not
count toward the statistic either way. Both cases must be reported with `n`, since a
benchmark that quietly discards its hardest instances flatters whichever arm ran out
first.

## Comparative runs on rented hardware: a preflight problem, not a procurement one

`bench/` makes the operator name the platform, because a throughput number is a fact
about a machine as well as about a build. The first draft of this note assumed a
capability number inherits that problem and concluded the answer was expensive --
named instance types at a large provider, where hardware is classified and a run
months later is genuinely the same machine.

That is the wrong scope, and `bench/`'s own layout already says so: perf baselines
live under `expected/perf/<platform>/` with an operator-supplied tag, while the
token-and-logprob baselines sit flat in `expected/` with no platform in the path. The
design already asserts that *output* is portable across machines and throughput is
not.

**What can actually move the tokens is a short, checkable list.** GPU architecture and
driver decide which attention and MoE backends vLLM autoselects. GPU *count* decides
the tensor-parallel degree and therefore cross-rank reduction order -- which is the
easy one to miss, because card count reads as a capacity property rather than a
numerical one. VRAM decides KV cache size and so the scheduling that batches requests.
Uncorrected ECC errors corrupt weights silently. Everything else about a host -- PCIe
width, clocks, CPU, RAM, cooling, network -- moves the stopwatch and leaves the tokens
alone.

Every item on that list is either checkable in seconds on contact with a box or fixed
by the container, which is under our control. So the rental market is fine for
capability work: screen on arrival, refuse and re-rent if the box does not match.
Named instance types remain the answer for *perf* comparability, where the host really
is part of the measurement.

**`tools/host_survey.py` is that screen.** Stdlib-only and single-file so it runs on a
bare box before torch or vLLM are installed -- `scp` it and run it, which is exactly
when a bad box is still cheap to reject. It reports the survey split into
output-relevant and throughput-only fields, prints a fingerprint over the
output-relevant subset (equal fingerprints should give equal tokens), and
`--compare box.json` diffs two boxes while *classifying* each difference, so a PCIe
generation change reads as comparable and a GPU count change does not. Exit status is
the machine-readable form: 0 usable, 2 usable with warnings, 1 refuse.

It refuses uncorrected ECC outright and warns on corrected errors, which is the check
that might have caught the card that died mid-run -- corrected counts climbing is what
a degrading card looks like before it takes the host with it. It also warns when the
GPU has no ECC at all, because "no errors reported" and "errors cannot be reported"
are different answers and only one of them is reassuring.

The remaining half of the problem is not comparability but *survival*: a 33-hour run on
a spot rental carries real probability of dying partway. A suite that cannot resume
from where it stopped loses everything, and one whose instance order is unshuffled
loses its representativeness as well.

---

# GPQA-Diamond, 2026-09

*Three `evalscope` runs of 198 items each, `temperature 1.0`, `seed 42`, no chat
template override: `Ornith-1.5-35B-A3B-exl3` @3.00bpw (92.4%), `Qwen3.8-27B-exl3`
@6.00bpw (82.3%), `Qwen3.5-9B-exl3` (85.4%). Plus three `opencode`/harbor runs of the
same dataset. Analysis scripted against the persisted per-item records, not the
summary reports.*

## Three metrics, not one: correctness, cost, indecision

A rigid harness scores as failure things that are not failures. A model that meanders,
needs compacting, and arrives at a good answer beats one that converges fast on a wrong
one. What actually matters is **final correctness** and **wall-clock to get there** —
and wall-clock is best estimated as total tokens in and out, because prefill and decode
rates are facts about hardware as much as about the model.

**A third axis earned its place: indecision.** The long trajectories in these runs are
not degenerate loops. Sampling the three longest from the Ornith run, the text is
grammatical, on-topic and self-aware throughout; one of them *converged* and answered
correctly at 164K tokens. What the worst one does is oscillate — the phrase *"I've been
assuming the mass fraction points to antimony, but let me reconsider whether it might
point to a different element"* appears six times, and it ends "Final answer: B.
Actually, wait. Let me reconsider whether the answer might be D... OK, I'll go with B."
That is a **termination failure, not a generation failure**, and it is invisible to both
other axes: the model is neither wrong-by-ignorance nor slow-by-difficulty.

Counting reconsideration markers separates the two reasons a trajectory is long:

| Ornith item | tokens | restarts | per 1K tok | correct |
|---|---|---|---|---|
| 127 — legitimate enumeration | 259,339 | 1,049 | **4.04** | no |
| 81 — long but converged | 164,083 | 1,243 | 7.58 | yes |
| 147 — oscillation | 149,054 | 2,806 | **18.83** | no |

Density inverts the token ranking, which is the point: the longest item is the least
indecisive.

**As a failure predictor the raw count is marginally better than tokens; density is
not.** AUC, P(metric ranks a wrong item above a right one):

| run | tokens | restarts | density |
|---|---|---|---|
| Ornith-35B 3.0bpw | 0.752 | **0.781** | 0.760 |
| Qwen3.8-27B 6.0bpw | 0.757 | **0.790** | 0.683 |
| Qwen3.5-9B | 0.738 | **0.746** | 0.640 |

So the two uses come apart: **density diagnoses an item, raw count ranks a population.**
Two caveats worth more than the AUC figures. Restarts and tokens are not independent —
more thinking mechanically produces more reconsiderations, and +0.03 is what survives
that overlap. And AUC-for-failure is not the question a comparison asks: what matters is
which metric moves most *between arms*, which needs two arms of one model and is
untested. The marker list is hand-written and phrasing is model-specific, which would
bias cross-model use but not the within-model comparisons this is for.

**The pairing trap from the SWE-bench pilot extends to the cost axis.** Aggregate turn
medians said the quantized model used fewer turns while the paired comparison said the
opposite; token totals fail the same way, because a model that quits early looks cheap.
Compute the cost delta paired, on the **correct-in-both** subset.

## GPQA-Diamond as an instrument: a fixed pool, and saturation

**198 items is the entire dataset.** Unlike SWE-bench there is no "budget more
problems" lever, and the effective sample size is still the discordant count. What that
buys, by exact McNemar:

| discordance | pairs | smallest significant split (p<0.05) |
|---|---|---|
| 10% | 20 | 15/5 |
| 20% | 40 | 27/13 |
| 30% | 59 | 38/21 |
| 40% | 79 | 49/30 |

Roughly a 2:1 asymmetry, easing as discordance rises. A subtler effect is not visible at
198 items with any amount of repeated sampling — repeated sampling reduces per-item
measurement noise, but the between-item variance pairing already removed is where the
leverage was. That is a pool-size problem, and the answer is a different benchmark.

**And it is at or near ceiling for 30B-class reasoning models**: 92.4%, 85.4%, 82.3%.
At 92.4% only fifteen items are wrong, so two similar arms would produce a handful of
discordant pairs and correctness cannot discriminate. **When correctness saturates, the
cost axis carries the signal** — it has enormous dynamic range on the same runs (median
2,827 output tokens, mean 15,700; fourteen items own 51% of all output tokens) and
accuracy falls monotonically with trajectory length:

| output tokens | items | accuracy |
|---|---|---|
| <4K | 112 | 97.3% |
| 4–16K | 44 | 93.2% |
| 16–64K | 28 | 92.9% |
| ≥64K | 14 | **50.0%** |

## Seeded non-greedy, not greedy

Greedy is the cleanest paired instrument — zero sampling noise, so every flip is
attributable to the change under test — and it is the wrong choice anyway. It is not how
models are operated, and it is structurally blind to the failure mode that matters here:
greedy flips only when the argmax changes, so a model whose tail has been damaged
decodes identically under greedy and badly under sampling. Same reason token ids beat
rendered text.

**Seeded sampling with the same seed across arms** keeps the RNG streams in lockstep
until the first genuine divergence, so early flips mean the distributions differed rather
than the draws did. It needs a *per-request* seed; under continuous batching a global one
does not give that.

The cost is symmetric draw-noise flips, which dilute the asymmetry. Writing A for
one-way flips caused by a real effect and M for symmetric noise flips:

| A | M=0 | M=10 | M=20 | M=40 |
|---|---|---|---|---|
| 12 | p<0.001 | 0.017 | 0.050 | 0.126 |
| 18 | p<0.001 | 0.001 | 0.005 | **0.025** |
| 25 | p<0.001 | <0.001 | <0.001 | **0.003** |

A real effect worth ~18 one-way flips survives heavy noise; only a marginal one drowns.
That failure mode is benign, because an effect too small to see past the sampler is
usually too small to act on. Escalate with **seeds, not items** — k seeds gives per-item
success *rates* compared with a bootstrap, which separates capability from draw noise in
a way McNemar on binaries cannot. `evalscope` review records already carry
`generation_index`, so the k>1 path is modelled by the harness.

Match every sampling parameter, and the chat template, to what will actually be served.
A knee measured at one temperature does not transfer to another, and changing it later
invalidates comparisons the same way changing the context limit does.

## The harness is an output-relevant field

`tools/host_survey.py` formalises that GPU count changes the tokens while PCIe width only
changes the stopwatch. **An agent harness is squarely on the tokens side**, and the
2026-09 runs show it dominating the measurement.

**Same model, same items, same default reasoning — 31 points apart.** The `Qwen3.5-9B`
run overlaps `evalscope` on 32 items: **50.0% through `opencode`, 81.2% through
`evalscope`**, discordant **10/0**, every disagreement in the same direction
(McNemar p≈0.002). A capability difference flips both ways; strictly one-sided means a
systematic failure mode. Two were found:

- **Protocol compliance.** Five of sixteen failures are `✗ Error: /app/answer.txt not
  found`. The model reasoned correctly, answered in prose, and never wrote the required
  file — two steps, no tool call. Scored as if it got the science wrong.
- **Reasoning volume, and it is model-dependent.** Median reasoning tokens: the 9B gets
  **1,246** under `opencode` against **10,164** under `evalscope`, an 8.2x *suppression*;
  the 27B gets **3,378** against **2,012**, a 1.7x *increase*. The scaffold does opposite
  things to different models.

**So an agentic harness is not merely overkill for a non-agentic benchmark, it is
actively harmful**, and its absolute scores are not comparable to published figures or to
a simple harness. The blend of science knowledge, tool-use compliance and reasoning
survival differs per model, which is exactly why cross-model rankings through it fail.
`evalscope`'s built-in harness is the better instrument for Q&A, and lighter.

**Paired comparisons through it survive, though.** The `fp8` vs `tq4` KV arms below share
scaffold, compliance and suppression, so all of it cancels in the pairing. That is what
pairing is for; it is the *absolute* number that is uninterpretable.

**A setting present in the config is not a setting that took effect.** vLLM degrades
`reasoning_effort` to a boolean for any chat template that does not understand the graded
form — `chat_completion/protocol.py`: `extra_kwargs["enable_thinking"] =
self.reasoning_effort != "none"`. Only `Qwen3.8` templates carry the graded knob;
`Qwen3.5-9B`, `Qwen3.5-35B-A3B` and `Ornith` carry only `enable_thinking`. So
`reasoning_effort: medium` *capped* the 27B and was silently discarded for Ornith, which
ran uncapped. The token distribution is the check, and it is the only one: the config
states intent, the distribution states behaviour.

**What a harness fingerprint has to carry**, by the same logic as the host one: agent
name and version, harness and version, chat template, every sampling parameter, the
context and output limits, and the effort setting *as it took effect*. Two arms under
different fingerprints are a different experiment, not a continuation. `opencode` is a
moving target in a way `mini-swe` was not.

## A runaway on a Q&A item is model behaviour, not harness failure

The policy above — drop the pair if both arms exhaust context, manual-inspect if one
does — was written for SWE-bench, where a real agent would compact and continue and the
event says nothing about the model. **On a multiple-choice question it says a great
deal**: running to the context limit is a failure to converge, which is the model's, and
it is the behaviour the indecision metric exists to catch. Score it, and report the
**runaway rate per arm as a third number**. If one arm runs away on twelve items and the
other on four, that is the finding, and it lives entirely in the rate — the answers
produced on those items are near chance and carry nothing.

**A hard cap is defensible, and 128K is the right level.** What each cap would cost the
Ornith run, whose tail is the only one that reaches these lengths:

| cap | items truncated | of those, were correct | accuracy |
|---|---|---|---|
| 32K | 27 (13.6%) | **70%** | 92.4% → 82.8% |
| 64K | 14 (7.1%) | 50% | → 88.9% |
| **128K** | **4 (2.0%)** | 50% | → **91.4%** |

At 32K you truncate 27 items of which 70% were *correct* — genuine long deliberation
that pays off. By 128K they are coin flips. The cap belongs above the
productive-deliberation range, and the reason to accept the penalty is not that those
trajectories were doomed — half of them land — but that chance-level items contribute
symmetric flips, which is the term that dilutes a paired comparison. Truncating them
costs ~1pp of absolute score and slightly improves discriminating power. The other two
models lose nothing at 128K; whether a given subject reaches it is worth confirming on
the first arm rather than assuming.

## Pricing it: expected time to a correct answer

The two axes are commensurable, which makes a quantization decision arithmetic rather
than judgement. Expected time to a correct result is `tokens_to_correct x ms_per_token`:
degradation raises the first, offload raises the second. With retry-to-success semantics,
`E[tokens to correct] = tokens_per_attempt / P(correct)`.

Worked against [cpu-offload.md](cpu-offload.md)'s measured throughput, on a 16 GiB card
at Gen5 x16, using the `gemma-4-12B` ladder's actual scaling — body moves with bit rate
(4.097 / 4.778 / 5.460 GB at 3.00 / 3.50 / 4.00bpw) while embed+head stays fixed at
2.874 GB, so a 16 GiB 4bpw model is 12.8 GiB at 3bpw and fits resident:

- 4bpw with 2.9 GiB offloaded: **75 tok/s**
- 3bpw fully resident: **111 tok/s**
- **so 4bpw is worth it only if 3bpw needs >1.48x the tokens to reach a correct answer**

A 9pp correctness drop takes that threshold to 1.18x. In one-shot QA the arithmetic
collapses — correctness dominates and throughput is nearly irrelevant — so the formula is
for agentic use, where a wrong answer costs a retry rather than the task.

**This only works if the harness persists per-item token counts.** `evalscope` does:
`usage.input_tokens` / `output_tokens` / `reasoning_tokens` plus
`perf_metrics.latency`/`ttft`/`tpot`, per item, with a stable `index` for pairing.
`opencode` trajectories carry it too, but `final_metrics.total_completion_tokens`
**excludes reasoning** — the thinking is in `steps[].metrics.extra.reasoning_tokens`, and
reading the obvious field understates output by roughly 8x. A harness configured to log
only pass/fail throws away the half that prices the decision.

## Measured: 4-bit KV costs nothing detectable, on both axes

*`Qwen3.8-27B-exl3` @3.00bpw through `opencode`, 198 paired items, `fp8` KV against
turboquant `tq4`. Both arms at the model's default `xhigh` effort — no setting was
passed — median peak prompt 10,204 tokens and max 66,885, against a 131,072 limit.*

- correctness: 76.3% vs 74.2%; 137 both-right, 37 both-wrong, **24 discordant (14/10)**;
  McNemar exact **p = 0.541**; bootstrap 95% CI **[-2.5, +7.1] pp**
- cost: paired reasoning-token delta **median +1**, and **+33** on the correct-in-both
  subset

**A two-axis null.** It cannot distinguish equal from slightly worse, but it excludes
`tq4` being worse than about 7 points — the same shape of result as the SWE-bench pilot,
at full 198 and through a real agent. Note the mean cost delta (+93) is moved by a single
`fp8` runaway of 45,347 tokens against `tq4`'s 35,768 while the median barely shifts;
use the median.

**The body-bit half of the comparison is unmeasured**, and it is the one that matters:
whatever a bpw ladder produces has to be read against ≤7pp for 4-bit KV on the same
instrument. Reported from practice, and consistent with this: 4-bit TurboQuant KV costs
far less capability than degrading body weights past the knee.

**Provenance caveat.** The `tq4` arm's `config.json` records `job_name:
"2026-09-13__16-21-27"` while the `fp8` arm records the descriptive name, so which arm is
which rests on the directory name alone, and the two ran a day apart. That is the harness
fingerprint gap in miniature, on a result good enough to deserve better.
