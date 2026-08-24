# bench — the dependency-bump gate

*Does this build still serve what the committed baseline serves?*

```
bench/run.py list                 what the matrix covers, and why
bench/run.py check [--tier fast]  compare a fresh capture to the baseline
bench/run.py bless [--tier fast]  record the current build as the baseline
bench/run.py capture <entry> OUT  one entry, for hand inspection
bench/run.py perf-check  --platform TAG   throughput vs this machine's baseline
bench/run.py perf-bless  --platform TAG   record this machine's throughput
```

`check` reads *what* is served and is blind to how fast; `perf-check` is the
other half. A bump wants both — a change costing 30% of decode throughput passes
every correctness entry cleanly.

Run `check` before and after a vLLM or exllamav3 bump. `bless` only after reading
a failure and deciding the change is intended — blessing is how a real regression
becomes the new normal, so it is a separate verb on purpose.

## Why it captures what it captures

This exists because two silent defects got through everything else we had (see
[docs/transformers-backend.md](../docs/transformers-backend.md)), and they failed
in *different channels*:

| channel | catches | the worked example |
|---|---|---|
| greedy token ids | gross divergence | a dropped embedding norm: **7 of 7** tokens wrong |
| top-k logprobs at every prompt position | **monotonic** error | a dropped logit soft cap: **40 of 40** tokens identical, every logprob off by 5.1× in effective temperature |
| reported weight bytes | VRAM regressions logits cannot see | tied-embedding serving falling back to fp16: 0.28 → 0.46 GiB on Qwen3-0.6B |

The middle row is the reason this is not a token-comparison suite. A gate that
only diffed generated text would have passed the soft-cap bug without a murmur.

The third row is the reason it is not only a logit suite. Serving a tied model's
embedding from its quantized `lm_head` is the project's central VRAM claim, and
breaking it leaves every logit correct.

## Measurement

Teacher-forced: both sides are fed a fixed token sequence and `prompt_logprobs`
scores every position at identical context. Sampled comparison would conflate
numerical error with model confidence, and once two runs diverge they are on
different contexts — comparing different prompts rather than different
arithmetic. The numerics live in [core.py](core.py), shared with
[tools/tp_compare.py](../tools/tp_compare.py), which asks the same question
against a second live capture instead of a committed baseline.

Every entry runs in its own process: vLLM has no supported way to stand up
several engines back to back in one process, and a crashed EngineCore otherwise
leaves the parent deadlocked on a zombie.

## Thresholds

`argmax_disagreements`, the greedy continuation and `weight_gib` are exact.
`dlogprob_max` (0.25 nats) and `kl_max` (5e-2) have headroom, bracketed by two
floors measured on this build rather than guessed:

| floor | value | what it is |
|---|---|---|
| same build, re-run | **exactly 0.0** | teacher-forced decoding at fixed context is deterministic — a `check` that changes nothing reports nothing |
| same weights, different kernels | **~0.157 nats / 0.013 KL** | Qwen3-0.6B eager vs CUDA graphs with the embedding path removed; the closest proxy for benign upstream drift |

The one real defect with numbers moved logprobs by ~15 nats, so 0.25 sits above
the drift floor and ~60× below a bug.

Argmax and greedy stay exact on purpose. A kernel change big enough to flip an
argmax at fixed context is one a human should look at, and it will occasionally
fire on something benign — that cost buys a gate that does not quietly absorb the
next dropped norm.

## Two kinds of baseline, and why they are stored differently

This is the distinction the layout exists to enforce:

- A **correctness** baseline is a fact about this codebase and its dependencies.
  Run the same build anywhere and it should hold. Stored flat in `expected/`.
- A **perf** baseline is that *plus a machine*. Stored under
  `expected/perf/<platform>/`, and `perf-check` refuses to compare across
  platforms rather than reporting a change of computer as a regression.

**The platform tag is the operator's, and it is mandatory.** There is no default
and no attempt to fingerprint the machine, because a machine cannot be identified
from inside it — firmware, host BIOS, thermal headroom, a noisy neighbour on a
shared host and the hypervisor's own scheduling all move throughput and none are
visible. Auto-detection would produce a key that looks authoritative and is not.
So `--platform <tag>` or `BENCH_PLATFORM=<tag>`, at whatever granularity you
need: one box, or `vast-8x3090-a` and `-b` if rentals need telling apart. It only
has to mean the same machine next time.

What the machine *will* admit — GPU name, capability, driver, torch, CUDA — is
recorded as **evidence, not identity**. `perf-check` prints any field that
changed under a tag, because a mislabelled baseline is exactly the failure this
scoping prevents.

### Do not trust the package version string

`vllm.__version__` reports `0.27.1.dev0+ge50f7d369.d20260810` for a checkout
detached at **v0.27.0**. The hash and date are right; the base version is not —
it appears to be whatever was newest when the editable build was made. It is
recorded as `vllm_reported` and should be read as a hint, never as the answer.

The worse problem is invisible rather than wrong: this project applies
`patches/` to vLLM as **unstaged working-tree changes**, and no version string
can see those. Two baselines could carry an identical version field and have been
produced by different patch stacks — which, for a gate whose entire job is
spanning a dependency bump, is the difference most likely to matter.

So each capture records `git` provenance for all three trees that decide what it
means — the plugin, vLLM and exllamav3:

```json
"src.vllm": {"describe": "v0.27.0-dirty", "head": "e50f7d3696",
             "dirty_files": 9, "diff_sha": "a0f2e5c81d51", "untracked": 0}
```

`describe` is the truth the version string missed, and answers only "which
commit" — `dirty_files` and `diff_sha` answer "what is uncommitted". They are
kept apart because `git describe` accepts no pathspec, so its `--dirty` flag
would contradict the two fields beside it (see below).

`diff_sha`, a digest of `git diff HEAD`, is what distinguishes patch stacks. It
identifies a stack without describing it; to see what changed, diff the trees.
The digest covers tracked modifications only, so `untracked` is counted
separately, since an untracked `.py` inside a package does change behaviour.

**The plugin's own provenance excludes `bench/expected/`.** Baselines are this
suite's output and they live inside the tree being described, so without the
exclusion a `bless` describes its own side effects: the first entry writes a
baseline, the second records `dirty_files: 1`, the third `2`, and a full bless
can never record a clean state however clean the checkout was. Provenance is
about the code that produced a measurement, not the measurement.

The exclusion has to reach **every** field that counts uncommitted state, and
until 2026-08-24 it did not: `dirty_files` and `diff_sha` were scoped,
`untracked` was not. That gap is invisible on a bless that overwrites existing
baselines — they are tracked, so nothing is untracked — and appears only on one
that *adds* entries, where each new baseline is untracked at the moment it is
counted (a capture opens its output file before recording the environment). The
blockq pair caught it on their first bless, recording `untracked: 1` and `2`:
numbers describing the run itself, which no clean checkout could ever reproduce,
so those two entries would have warned on every `check` forever.
`tests/test_bench_provenance.py` pins the scoping in both directions — baselines
excluded, an untracked `.py` in the package still counted.

**The three fields are not disjoint**, which matters when reading a baseline.
`dirty_files` counts porcelain status lines, untracked ones included, so it is
the total; `diff_sha` digests tracked modifications only; `untracked` is the
remainder, the part no diff can see. One modified source plus one untracked
source reads `dirty_files: 2, untracked: 1`.

### `verify`

```
bench/run.py verify
```

Do all baselines in a set agree about what produced them? A set is meant to be
one snapshot of one build, and nothing enforces that — `bless` writes entries one
at a time over most of an hour, and anything changing underneath it splits the
set silently.

This is not redundant with the dirty-tree warning `bless` prints. That warning
catches an operator editing mid-run; it could not catch the case above, where the
suite dirtied its own tree, because no operator discipline was involved. Only
comparing the finished set across entries finds that class of problem.

Correctness baselines carry the same record but are **not** scoped by it, and
`check` only warns. Portability is the intent — but "meant to be portable" is not
"is": fp16 accumulation depends on tile shapes, which depend on the GPU, so a
check on different hardware can move logprobs and occasionally an argmax. The
warning is there so that reads as hardware rather than as a regression.

## Throughput

Separate entries, separate baselines, and a different configuration: **CUDA
graphs on**, because perf measured in eager mode would gate a way nobody serves.
The workload deliberately reproduces the shape in
[docs/kernels.md](../docs/kernels.md) — decode 8 concurrent × 128 tokens, prefill
4 × ~2.2k, fp16, prefix caching off — so that note's recorded numbers and these
are the same measurement. It reproduces to ~0.1% on decode.

The gate is **−10%, one-sided**. Throughput is not deterministic the way logprobs
are, but on the dev card it is far steadier than expected:

| | spread |
|---|---|
| repeated runs inside one process | ~1% |
| medians across fresh processes | ~0.5% (decode 2750.9 / 2738.9 / 2741.2 tok/s) |

So −10% is roughly 20× the observed noise and still catches anything worth the
name. Only regressions fail: a large speedup with correct logits is good news,
and work being silently skipped is what the correctness gate is for.

Deliberately few entries. A throughput regression is broad — a kernel or
scheduler change lands on any model exercising that path — so covering every
*path* matters and covering every checkpoint does not.

### What the blockq perf entry guards, and what it cannot

`minicpm5-1B 3.0bpw blockq perf` exists so a served path in the throughput
record actually uses the packed embedding. Its scope is narrower than it looks,
and the measurement that establishes this is worth not re-deriving
(2026-08-24, dev card, this workload):

| gather repeated | decode | vs 1× | prefill | vs 1× |
|---|---|---|---|---|
| 1× | 2998.2 | — | 43070.6 | — |
| 4× | 2990.3 | −0.3% | 43327.8 | +0.6% |
| 16× | 2999.6 | +0.05% | 43325.9 | +0.6% |
| 64× | 2891.9 | −3.5% | 36943.3 | −14.2% |

Running the decode **sixteen times over is invisible**; it takes 64× before
anything clears run noise. The path is ~0.06% of a decode step and ~0.23% of
prefill here, and a *smaller* share on the larger models this project targets.
So the −10% gate resolves roughly a 25× regression in the gather and nothing
finer.

**A same-run dense companion does not fix that**, which is worth recording
because it is the obvious next idea. The ratio should cancel whole-engine
drift, but the two sides are separate engine processes with independent
autotune and allocator state, so their drift compounds instead: across three
pairs the prefill ratio spread 2.50pp (0.9849 / 1.0099 / 1.0093), *worse* than
either absolute number, while decode managed 0.51pp. Both land back at ~25×.

**That argues for keeping the entry, not for dropping it or replacing it with a
microbenchmark.** A regression this instrument cannot see is, by the same
measurement, one nobody serving the model would feel. The condition that *is*
relevant in practice is an interaction between the engine and the embedding
path costing real throughput — unlikely, not impossible, and precisely what the
existing gate catches now that an entry serves blockq. An op-level
microbenchmark would resolve the gather far better and would miss interaction
regressions entirely, which is the likelier direction for this path to break,
blockq being ours to change.

**A note on the dense-embedding entries.** Their value as *gates* has fallen
since blockq shipped: nobody would serve an untied model that way with the
packed embedding available. They remain useful for discrimination — the llama
anchor is what says whether a regression is blockq's or everything's, besides
holding `docs/kernels.md`'s numbers live — but the gate does not need to
discriminate everything. It needs to get attention when something that matters
moves.

## Baselines must not depend on the calendar

Some chat templates inject today's date — Muse-Glimmer via
`strftime_now('%Y-%m-%d')`, Llama 3.x via `date_string`. Left alone, those
entries' prompt ids change at midnight, `check` correctly refuses to compare, and
the gate goes red for a reason that has nothing to do with the build. A gate that
fails on the calendar is one people learn to ignore, which is the same failure
mode the thresholds are shaped to avoid.

`core.PINNED_TEMPLATE_VARS` passes fixed values for every spelling we have met.
There is no common one, so **when adding a model, check
`"strftime_now" in tok.chat_template`** and add its spelling if it is missing.
Templates that do not use these variables ignore them, so passing them
unconditionally is safe.

The two entries this affected — `llama-3.2-1B-3.0bpw-tied` and
`muse-glimmer-30B-2.0bpw-via-transformers-backend` — were re-blessed on
2026-08-17 and now embed the pinned date rather than the day they were recorded.

## Known-broken entries

An entry that cannot currently be captured keeps its `known_broken` reason and
keeps running — deleting it would lose both the coverage and the knowledge. It is
never blessed, and `check` reports it without failing the gate. If it starts
capturing cleanly, `check` fails and tells you to clear the field: that is how a
fix gets noticed.

## Fixtures: entries whose checkpoint nobody publishes

Every entry but two names a `repo@revision` and lets the Hub resolve it. The
block-quantized embedding has no such checkpoint —
`tools/quantize_embedding.py` derives one from a published checkpoint, and no
output of it has been published. Those entries set `fixture="blockq"`, and
`bench/fixtures.py` builds the derived checkpoint into
`~/.cache/vllm-exl3-plugin/bench-fixtures/` (override with `BENCH_FIXTURES`)
before capture.

**Derived rather than published, and the reasons are not close.** Building the
MiniCPM5-1B fixture takes 3.1s, against a ~500 MB download; it needs no account,
so the gate stays runnable by anyone with the repo; and it is byte-reproducible
— the encoder runs on CPU, and two builds a week apart in different processes
agreed on the sha256 of all three tensors. The deciding reason is the last one:
a derived fixture puts **`tools/quantize_embedding.py` under the gate**, which
nothing else does. A published fixture would freeze the producer's output at
upload time and never exercise the producer again — and the producer is what
rewrites real checkpoints.

The tool is run as a subprocess rather than imported, so what is gated is the
command line a user runs.

**Caching is keyed on what decides the contents**: the base `repo@revision`, the
recipe version, and a digest of the encoder's own source (`blockq.py`,
`format.py`, the tool). Edit the encoder and the next run builds a new fixture
rather than serving the old one — the alternative being a gate that passes
while testing a checkpoint the current code would not produce. Builds are staged
and renamed, so an interrupted build cannot leave a half-written checkpoint for
the next run to serve.

**The capture records a content digest of what the recipe added**, and `check`
reports a change to it *before* the logprob comparison. This is the distinction
that makes a fixture entry readable: without it, "the derived checkpoint
changed" and "the build regressed" look identical, and every logprob difference
downstream is a consequence rather than a finding. It is the same separation
`src.*.diff_sha` draws for the patch stack.

### What the blockq pair is for

The weight-bytes gate carries most of the value. `minicpm5-1B-3.0bpw-mcg` and
`minicpm5-1B-3.0bpw-blockq-eager` are the same checkpoint differing only in the
embedding, at **0.79 and 0.52 GiB** resident — so a blockq path that quietly
fell back to loading dense bf16 would serve every logit correctly and give back
the entire 0.27 GiB saving, and only this number would notice. Same shape as the
`llama-3.2-1B-3.0bpw-tied` entry, for the same reason.

The graphs entry is not redundant with the eager one. The decode is written as
plain torch precisely so inductor can fuse it into the surrounding graph, so the
compiled path is a different computation reaching the same answer, and a gather
whose indices got baked into a replayed graph would return the previous batch's
rows while failing nothing else. `tests/test_blockq.py` makes that claim at the
unit level; these entries make it through vLLM.

Eager and graphs differ by ~0.30 nats / 0.028 KL and 3 argmax flips of 75 on
this model, so — as with the Qwen3-0.6B pair — each mode keeps its own baseline
and the difference is execution mode, not the embedding.

### Both guards were watched failing

A guard nobody has seen fire is a comment. Each was proven by reintroducing the
defect it exists for and running the entry against its committed baseline
(2026-08-24):

**The weight gate, against a silent fall back to dense.** Injected the realistic
regression — `process_weights_after_loading` decoding once at load and keeping
the dense table, as a future refactor might. Result:

```
prompt 0: 15 pos, argmax 0, |dlogprob| max 0.000e+00, KL max 0.000e+00, greedy ok
prompt 1: 60 pos, argmax 0, |dlogprob| max 0.000e+00, KL max 0.000e+00, greedy ok
FAIL - weight bytes 0.52 -> 0.9 GiB
```

All 75 positions byte-identical, greedy unchanged, not one argmax flip. The
entire logit half of the suite passes it without a murmur, and the resident-bytes
number is the only thing that notices — which is the argument for the entry.

**The fixture digest, against a changed derived checkpoint.** Flipped one nibble
of one row in the cached fixture (the cache is keyed on the encoder's source, so
a fixture altered underneath it is reused). Result:

```
prompt 0: 15 pos, argmax 0, |dlogprob| max 0.000e+00, KL max 0.000e+00, greedy ok
prompt 1: 60 pos, argmax 0, |dlogprob| max 0.000e+00, KL max 0.000e+00, greedy ok
FAIL - fixture blockq content changed: 2fd4685d5c65139c -> b58e6bc537e54d61
```

The perturbed row was not one the prompts look up, so the served logits are
identical and the digest is the only evidence the checkpoint changed at all.
That is the case the record exists for, and it argues for keeping the digest
even though a *typical* fixture change would also move logprobs.

Restored, both entries return exactly 0.0 — the documented same-build floor.

`bench/fixtures.py`'s cache key is the one part with no runtime guard, because
its failure is silent in the other direction: a key that does not change when
the encoder does yields a gate that passes while serving a checkpoint the
current code would not produce. `tests/test_bench_fixtures.py` pins that
property.

## Tiers

`fast` (~15 min on a 16 GiB card) is the one to run casually: uniform K=3,
mixed-in-layer bit widths, `mcg`, tied and untied, both execution modes, and the
Transformers backend on a text-only model.

`full` adds what does not fit that budget — MoE, the `mul1` codebook and the
gemma4-style tie, and the multimodal Transformers backend. Run it before a
dependency bump.

Not covered at all: **TP**, which needs the 8×3090 box rather than the dev card,
and wants its own tier.

## Pin `model_impl`, do not let it disperse

The two Transformers-backend entries set `model_impl="transformers"` explicitly
rather than relying on dispatch. This matters more than it looks: vLLM gains
native implementations over time — MuseGlimmer gets one in 0.27.2 — and an entry
that depended on auto-resolution would quietly stop testing the backend the
moment that happened, while still passing. Pinning means the coverage is of *our
integration*, not of vLLM's routing, and it survives.

The MiniCPM pair is the useful shape: the same checkpoint on both paths, which
turns "the backend is token-for-token identical to native" from a claim in
[docs/transformers-backend.md](../docs/transformers-backend.md) into something
checked on every run.

## Adding an entry

An entry earns its place by exercising a plugin surface no other entry reaches;
put that in `exercises`, because it is the argument for keeping it when the suite
gets slow and the first thing to read when it fails. If the entry encodes a
*prediction* — as the Muse-Glimmer one does about what 0.27.2 will break — put
that there too. A prediction a gate will test is worth more than one in a doc.
