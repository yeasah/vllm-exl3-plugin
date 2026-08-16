# bench — the dependency-bump gate

*Does this build still serve what the committed baseline serves?*

```
bench/run.py list                 what the matrix covers, and why
bench/run.py check [--tier fast]  compare a fresh capture to the baseline
bench/run.py bless [--tier fast]  record the current build as the baseline
bench/run.py capture <entry> OUT  one entry, for hand inspection
```

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
