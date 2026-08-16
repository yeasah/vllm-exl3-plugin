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

## Known-broken entries

An entry that cannot currently be captured keeps its `known_broken` reason and
keeps running — deleting it would lose both the coverage and the knowledge. It is
never blessed, and `check` reports it without failing the gate. If it starts
capturing cleanly, `check` fails and tells you to clear the field: that is how a
fix gets noticed.

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
