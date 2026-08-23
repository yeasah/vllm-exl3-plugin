# Measuring capability through the served path

Evidence and rationale behind TODO `capability-suite`. qbench answers how far a
quantized distribution sits from its reference; this note is about the separate
question of whether the served configuration still *does the work*, and about what two
SWE-bench runs taught us regarding how such a comparison has to be built.

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
