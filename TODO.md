# Open tasks

What is outstanding, our current understanding of it, and the approach we think is
the best candidate today. Nothing else.

<!--
POLICY -- read before adding to this file.

The test: **if it would still be true and worth reading after the item is closed,
it does not belong here.** Measurements, tables, ruled-out hypotheses, bug
post-mortems and chronology all fail that test. They go to the matching note in
docs/ when written -- not here first for migration later, because that migration
never happens.

Each item carries four things and stops:
  1. the outcome wanted;
  2. one or two sentences on what it unblocks;
  3. the current best-candidate approach, and the one-line reason it is the
     candidate;
  4. a pointer to the note holding the evidence.
Roughly a screen per item. If an item outgrows that, the overflow is evidence and
belongs in docs/.

An item enters only if it is an outstanding *task*. Observations, results and
"we noticed that" go straight to the subject note.

Headings carry a **stable slug**, and cross-references from code and docs use the
slug, never the position -- `TODO: repair-tool`, not `TODO #2`. Items get
reordered and evicted; positions rot silently and take every reference with them.

**But slugs are only durable against reordering, not against eviction** -- this
file is the one thing here guaranteed to churn, so prefer pointing at the `docs/`
note instead wherever the sentence works either way. A reader almost always needs
the subject, not the queue position. Reserve a slug reference for the case where
TODO is genuinely the only home for that work (no note covers it yet).

Eviction, in order:
  1. `grep -rn 'TODO.*<slug>' docs vllm_exl3_plugin tools README.md` and repoint
     every hit at the note that now holds the outcome. Do this *before* step 2 --
     "Recently closed" is capped, so it cannot be the long-term landing place for
     a reference.
  2. Delete the item; add one line to "Recently closed" at the foot, keeping the
     slug in that line so a stale reference still lands somewhere during the
     window before pruning.
  3. Prune "Recently closed" to ~10 entries.
  4. Record the outcome in the note, and the status in README.

Never reuse a retired slug for different work.
-->

The biggest win available in the immediate future is `quantized-embeddings`
combined with either `repair-tool` or `quantize-embeddings-pipeline`. Together they
take EXL3 from significantly underperforming competing formats on most checkpoints
(when given a full and honest accounting) to being as competitive as originally
advertised. [docs/qbench.md](docs/qbench.md) has the measurement that establishes
the gap is real on the served path.

## `bench-suite` — A TP tier for the bump gate

`bench/` gates vLLM and exllamav3 bumps on token ids, per-position logprobs and
resident weight bytes, against baselines committed for the current pin. Two tiers
are populated: `fast` (~15 min) covers uniform K=3, mixed-in-layer bit widths,
`mcg`, tied and untied, both execution modes, and the Transformers backend on a
text-only model; `full` adds MoE, `mul1` with the gemma4-style tie, and the
multimodal Transformers backend.

**What remains is TP**, which is the one axis the dev card cannot reach. It wants
its own tier rather than an entry in `full`, because `bless` needs the 8×3090 box
(`vast`) and the baselines are per-degree. TP=2/4/8 are the degrees already
validated by hand in [docs/tensor-parallel.md](docs/tensor-parallel.md), so those
are what a tier should pin; `tools/tp_compare.py` already shares the numerics and
its eager-vs-graphs floor workflow is the model for setting the tolerance.

**Carry a hub-prefetch subcommand with it.** A tier is only runnable on a box that
already holds its checkpoints, and the TP tier's boxes are short-lived cloud
rentals where a `bless` that discovers a missing 14 GiB repo halfway through has
wasted the rental. `bench/run.py deps --tier tp` should print `repo@revision` per
entry, with `--fetch` doing the download. Both forms rather than one: the listing
is what you want before committing to 50 GiB and in any planning or CI context,
the fetch is what you want on a fresh box, and the two differ by a flag. The
entries already carry `model`/`revision`, so this reads the matrix rather than
duplicating it.

Throughput is gated separately (`perf-check`/`perf-bless`), reproducing
`docs/kernels.md`'s workload shape so that note's table stays live. Perf
baselines are per-machine — `bench/expected/perf/<platform>/`, with the tag
supplied by the operator — because a perf number is a fact about a machine as
well as about the build. Only `rtx5070ti-dev` is populated.

A TP tier wants perf entries too, and inherits that: `vast` needs its own bless,
and rentals that are not the same machine need distinct tags. There the
interesting number is also not raw throughput but how it *scales* with degree —
a collective-path regression shows up as a scaling change while each degree's
absolute number still looks plausible against its own baseline. That comparison
is cross-entry rather than against a baseline, so it needs something
`perf-check` does not currently do.

→ [bench/README.md](bench/README.md)

## `fa-head-dim-512` — Flash Attention for head dim 512 on pre-FA4 architectures

vLLM has no FA path for head dim 512, and does not allow mixed attention layers
(over real, demonstrated instability concerns) that could otherwise cover the
majority of layers at dim 256. So the attention backend falls back to triton, which
carries a number of performance costs — not least taking turboquant off the table.

This looks at first glance like a bad complexity-to-payoff trade, and probably is
not: it is close to the difference between an entire model family (Gemma 4 among
them) being on the table or not. Not because those models will not run, but because
they will run badly enough against alternatives of similar capability to be a dead
choice.

**Candidate approach.** llama.cpp/ggml extended their FA implementation to cover
head dim 512 ([PR #20998](https://github.com/ggml-org/llama.cpp/pull/20998)). Unlike
ggml, vLLM uses a forked copy of upstream flash attention
([vllm-project/flash-attention](https://github.com/vllm-project/flash-attention)),
holding a confusing array of implementations across CUDA, ROCm, FA2-4 and Hopper.
The open question is whether any surface in that fork could be extended the way the
llama.cpp changeset extends ggml, for consumer Ampere / Ada / Blackwell.

**Something now hangs off the answer.** gemma-4 is the only tied mid-size family on
hand, so it is effectively the entire constituency for the shared embed+head tensor
deferred under `quantized-embeddings`. If this turns out to be a lost cause and
gemma-4 is not practically deployable, that optimization loses most of its reason to
exist; if it lands, the optimization becomes the natural follow-up here.

## `repair-tool` — Repair tool for existing EXL3 checkpoints

Every existing EXL3 checkpoint is handicapped by two pipeline mistakes, badly enough
to more than erase its efficiency advantage against other formats: a separate output
head is emitted for tied models (entirely redundant, and not small), and the
embeddings are skipped entirely and left at full resolution. The second is
significant at any size and painful at aggressive quantization, where the embedding
alone can exceed the whole rest of the model.

A post-processing tool preserves the very large investment in computation that the
published EXL3 collection represents, rather than requiring it be redone.

**Candidate approach, settled by measurement.** Emit a **block-scaled integer**
tensor, not a re-quantized trellis. As an embedding, scalar quantization beats the
trellis by ~89x at equal bits on gemma-4-12B, because the trellis optimizes `x @ W.T`
for a head rather than per-row accuracy — and it is simpler besides: no Hadamard, and
row extraction is a slice rather than a block decode. Scale per sub-block of 32 rather
than per row (the layout is in `quantized-embeddings`): per-row min/max lets one
outlier component set the scale for a whole token's vector, which costs up to 134x at
matched bytes. For tied models one shared tensor serves both roles and dominates both
the shared trellis and any two-tensor split.

Depth can be a shipped constant, which per-row could not offer: 4 bits (4.5 bpw) lands
at or below the noise floor on all three models measured. An override is still worth
having, and a size-budget solver still belongs to the full quantizer, where layer bpw
is actually free.

Untied models additionally need the embedding produced from scratch, as a
constrained optimization: the rest of the model's quantization decisions are already
made and fixed.

→ [docs/embeddings.md](docs/embeddings.md)

## `quantized-embeddings` — VRAM-efficient use of quantized embeddings

Serving the embedding quantized rather than dequantizing it at load. Dequantizing at
load proves the math and saves file storage and I/O, but leaves VRAM — the thing
this project cares about most — completely unchanged.

Tied models are **shipped**: `EXL3EmbeddingMethod` serves a tied model's embedding
from its existing quantized `lm_head`, with the fp16 `embed_tokens` never loaded.
Works on published checkpoints as they are, no repair tool and no quantizer work.
All hooks are sanctioned vLLM extension points; nothing is monkeypatched.

**Two things are open**, and the second is easy to overlook because the measurements
that motivate it were done without it.

1. **Untied models have no quantized embedding to reuse**, so one must be produced —
   that is `repair-tool` / `quantize-embeddings-pipeline`.
2. **There is no scalar-quantized serving path.** Everything the plugin can load is
   trellis: `stored_tensor_names()` is `trellis`/`suh`/`svh` plus a codebook tensor,
   and `EXL3EmbeddingMethod` serves via `ops.embed_rows`, which is trellis row
   extraction. The block-scaled scheme that the sweeps showed is orders of magnitude
   better as an embedding exists **only as simulation** — qbench's `fake_quantize`
   rounds a resident fp16 tensor to a bits-bit grid and immediately dequantizes it,
   explicitly "in place of dtype/storage changes... without committing to a real
   packed format or a real kernel". Nothing is packed, stored, or loaded.

So the tensor format is decided but unimplemented at both ends. **Both tooling items
are blocked on this**: a repair tool emitting such tensors today would produce
checkpoints nothing can serve.

**Next up, and scoped to untied models: a block-scaled embedding tensor served
alongside the checkpoint's existing trellis head.** Needs a storage layout (packed
values plus the superblock/sub-block scales described below),
`create_weights`/`process_weights_after_loading` registration, and a gather — which
should be markedly simpler than the trellis path, since row extraction becomes a slice
with no Hadamard and no 128-block read amplification. Rows are independent, so
vocab-parallel TP carries none of the 128-block alignment arithmetic the trellis path
needs.

Two reasons this shape rather than the shared tensor the frontier table favours. It is the **only** shape untied models can use, since their head is a
genuinely different matrix. And it needs no new kernel: a shared tensor has to serve
the head role too, and there is no scalar-integer GEMM anywhere — sharing would mean
either dequantizing to dense fp16 (which gives back the whole saving) or a trip to
kernel town. Costs ~0.35 GiB against sharing on gemma-4-12B, and is still ~1.4 GiB
better than native.

It is also where the new value is. Tied models already run with a much improved VRAM
profile via Phase A; what they carry is a **KLD hit we now know is unnecessary**
(+0.0216 from the trellis where a block-scaled tensor costs +0.0003 at 4.5 bpw). Untied
models get nothing at all today.

The shared-tensor optimization is deferred, possibly a long way. It only helps tied
models, and gemma-4 is the only tied mid-size family on hand — so if gemma-4 proves
impractical to deploy, which currently rides on `fa-head-dim-512`, there is almost
nothing left for it to apply to. Natural follow-up to that item rather than an
independent one.

The lookup *plumbing* is built and de-risked by Phase A — the vLLM hooks,
`tie_weights`, the tied-skip mapper — and that part is encoding-agnostic. It is the
decode underneath it that is trellis-only.

**The format is block-scaled, not per-row** — settled by measurement in
`gguf-embeddings`, and this is the one part of the plan above that it changed. Per-row
min/max is dominated at every matched byte count on every model measured, by up to 134x,
because one outlier component sets the scale for a whole row and the rows it ruins are
whole tokens. The layout to emit is GGUF's, with our own encoder: **superblocks of 256,
sub-blocks of 32, 6-bit quantized (min, scale) per sub-block against one fp16 pair per
superblock**, at 3-5 bits. That costs 0.5 bpw of scales rather than the 1.0 bpw a naive
fp16 pair per 32 would, which is the difference between beating per-row everywhere and
losing to it on MiniCPM5-1B.

**Default 4 bits (4.5 bpw), and the per-model calibration sweep is no longer needed.**
One setting covers gemma-4-12B, Qwen3.5-9B and MiniCPM5-1B at 0.19x / 0.32x / 1.49x their
noise floors, where the per-row scheme needed a per-model depth and a conservative default
of 6 bits. Arbitrary depths are still worth keeping (the optimum is per-model, and 3 bits
is now usable where per-row had a cliff), but they are an optimization rather than a
requirement.

Row alignment still does not bite, and for a stronger reason than before: a row is a whole
number of 256-element superblocks in every model measured (3840, 4096 and 1536 are all
multiples of 256), so rows never share a block and every row stays independently
sliceable — verified against real GGUF files, where a single row's byte slice decodes
bit-identically to the full-tensor decode. Assert the divisibility rather than assuming
it; a model whose `hidden` is not a multiple of 256 needs a smaller superblock, which is
also what llama.cpp does.

A fused kernel is the obvious highest-performing answer to the remaining per-token
cost, but is worth weighing against less costly approaches first: it is development
overhead and likely further pinning on CUDA, and exllamav3 has no ROCm support today
while talking about adding it.

→ [docs/embeddings.md](docs/embeddings.md)

## `transformers-backend` — Serving models vLLM does not implement

**Mostly answered: it works.** Serving through vLLM's Transformers backend
(`--model-impl transformers`) is token-for-token identical to the native path on
MiniCPM5-1B, and — with three vLLM patches — to *native exllamav3* on
Muse-Glimmer-30B, a model vLLM has no implementation for, vision tower included.
Model coverage therefore moves from "what vLLM implements natively" to
approximately "what transformers implements".

Text-only models on plain architectures need nothing. Beyond that the backend needs
patching, and the reason is always the same: it runs the model's *base* module
graph, so arithmetic living above it or inside a layer it substitutes gets dropped
— silently, whenever that arithmetic carries no weights.

→ [docs/transformers-backend.md](docs/transformers-backend.md)

**What remains open:**

- **Upstreaming the three patches.** All read as backend gaps rather than EXL3
  special cases, so all are plausible contributions. For
  `vllm-replicated-linear-weight-loader-v2.patch`, check the
  `RowvLLMParameter`-narrowing edge noted in the doc first. For
  `vllm-transformers-backend-embedding-postprocess.patch`, the detection is
  structural (an `nn.Embedding` subclass overriding `forward` with exactly one
  submodule) and upstream may want something declared by transformers instead.
- **Auditing rather than waiting.** Both Muse-Glimmer defects were found by reading
  the transformers modelling file against what the backend substitutes, and both
  would have stayed invisible on any metric short of a token-level comparison
  against another engine. The same read is worth doing per architecture before
  trusting the backend on it, and a cheap generic version — diffing a model's
  `ForCausalLM.forward` against the backend's `compute_logits`, and each
  substituted layer class against its replacement — would catch the whole family.
- **Breadth.** Three architectures tested through the backend, one of them
  multimodal and generating correctly — on *text* prompts only; no image has been
  passed through any of it, see `multimodal`. Still no MoE and no TP through the
  backend.

## `multimodal` — Actually pass an image through, and gate it

**Outcome wanted:** an EXL3 checkpoint demonstrably *understanding an image*
served through vLLM, plus a `bench/` entry that keeps it that way.

The project's default position is that multimodal does not matter yet, and as a
sequencing call that is still right. But the practice contradicts the position —
graphs and screenshots are how people actually hand over context, this project's
users included — so the honest version is "deferred", not "irrelevant".

**Images work.** Verified 2026-08-17 through a real chat client (Jan, chosen over
`vllm chat` because mainstream clients default to tens of thousands of tokens of
tool preamble): **gemma-4-12B, Qwen3.6-9B and Qwen3.8** all describe images
accurately, including fine detail like reading text in the image. That closes the
gap this item was opened for — the vision path is not merely loading.

**Still open, and now specific:**

- **No gate.** All three results are hand-run. `bench/core.py`'s prompts are
  strings, and an image entry needs the fixture committed beside the baseline,
  since a baseline against an image that later changes is worthless.
- **Audio is broken, probably not ours.** gemma-4-12B insists every clip is
  chirping birds. It is a unified model with no audio encoder — sound goes into
  the same token space as text — so there is no EXL3-quantized audio component to
  get wrong, which makes a pipeline fault upstream of us the likely cause. Cheap
  way to settle it rather than assume: the same clip through native exllamav3,
  the instrument that settled Muse's text path.
- **Muse-Glimmer remains the one blocked checkpoint**, for two unrelated reasons
  already characterized: native vLLM cannot serve its quantized vision adapter
  (`vision_adapter.c_fc` is a plain `nn.Linear`, unreachable by any quantization
  plugin), and the Transformers backend route is degraded on vLLM ≥ main by the
  upstream soft-cap gap, which wrecks sampled output while leaving greedy intact.

**The VRAM link, which is the reason this item is not merely nice-to-have.**
Multimodal is where headroom runs out first: Qwen3.8 only fits an image budget at
`--limit-mm-per-prompt '{"image": {"count": 1, "width": 512, "height": 512}}'`,
and its encoder cache allocation is what OOMs otherwise. The embedding this model
carries is ~2 GiB — **larger by itself than the entire post-weights headroom that
test was squeezing into**. So `quantized-embeddings` is not just a size win in the
abstract; on a 16 GiB card it is the difference between usable and unusable image
budgets. That makes multimodal a *beneficiary* of the embedding work rather than a
competitor for attention.

Note also that vLLM's multimodal knobs churn: `--max-num-encoder-input-tokens` has
been removed with no obvious replacement, `--mm-processor-cache-gb 0` does *not*
bound the encoder cache, and `--limit-mm-per-prompt` now carries feature-size as
well as counts. Check flags against the pinned tree rather than from memory.

**A first-quantization target worth noting: gemma-4 E2B/E4B.** Among common
general-purpose models it is one of very few with a genuinely *separate* audio
encoder, and the EXL3 collection skips it entirely. Small, capable for its size,
and structurally different from anything handled so far — a separate encoder
exercises the multimodal path deliberately rather than incidentally, which is
exactly what a first target should do.

**Candidate approach for the parts still open: the instrument that already worked.**
exllamav3 implements
this vision path natively (`MuseGlimmerVisionModel`, `examples/imgdesc.py`), so
the same image and prompt can go through both engines and be compared token for
token. It is the reference the text case proved worth having, and it is the only
one available — there is no fp16 Muse-Glimmer on hand to fall back to.

**Two obstacles, both already characterized rather than guessed:**

- **Native vLLM cannot serve the quantized vision adapter at all.** vLLM main
  builds `vision_adapter.c_fc` as a plain `nn.Linear`, which never reaches
  `get_quant_method`, so no quantization plugin can touch it. `--model-impl
  transformers` is the only route; `--language-model-only` bypasses vision
  entirely and is the reason this stayed invisible.
- **The gate is text-only.** `bench/core.py`'s prompts are strings, and an image
  entry needs a different capture shape — the fixture has to be committed with
  the baseline, since a baseline against an image that later changes is worthless.

→ [docs/transformers-backend.md](docs/transformers-backend.md),
[bench/README.md](bench/README.md)

## `exl3-metadata` — Improving the metadata situation

exllamav3 plays fast and loose with checkpoint metadata; important things are
missing or misleading. Low-hanging fruit, and restoring some trust in the stored
data would be useful — significantly tempered by the fact that existing checkpoints
keep their incomplete metadata regardless, unless we repair and republish them
(license permitting).

## `quantize-embeddings-pipeline` — Quantizing embeddings in the pipeline

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
question and was only ever measured on gemma-4-12B.

Distinct from `repair-tool` in one way that matters: here layer bpw is *free*, so
trading depth across components becomes a real constrained optimization (the
Lagrangian actually binds) rather than a one-variable heuristic. This is where the
size-budget solver belongs.

Depends on `quantized-embeddings` growing a block-scaled serving path — see there.
Nothing can load what this would emit today.

→ [docs/embeddings.md](docs/embeddings.md)

## `yaqa` — YAQA-quality rounding in the quantizer

The quantizer process itself can likely be improved, given its QTIP heritage and
YAQA's further work on QTIP. **Given license incompatibility it is important that
nobody look at the reference code — papers only.**

Large project, likely modest gains. Highly desirable, but as against the other
low-hanging fruit this one is high up the tree and should be scheduled as such.

## `moe-tp` — Finish the job on MoE + TP

Not new, but still outstanding. TP=2/4/8 are validated on hardware, MoE+TP is no
longer categorically unvalidated, and the autotune cache survives eight concurrent
writers.

**What remains:** TP=3, 5, 6, 7 (needs a box with those card counts); a second
checkpoint at TP=8 (needs a download — nothing on hand clears preflight); expert
parallelism (a kernel development gap, not a hardware-coverage one); and the Laguna
TP=4 `exl3_mgemm` performance bug. That last one is narrowed to two exact kernel
instantiations with both autotuners ruled out; closing it needs someone reading
`exl3_mgemm`'s source against Laguna's and Qwen3.5's actual per-tensor quantization
parameters, not more GPU time.

→ [docs/tensor-parallel.md](docs/tensor-parallel.md)

## `cpu-offload` — CPU offload for EXL3 weights

`vllm serve --cpu-offload-gb` silently offloads nothing for EXL3 — 0.01 GiB against
3.63 GiB for the same model as AWQ. Two independent causes, both traced: the UVA
offloader decides eligibility at construction time when our parameters are still
empty CPU placeholders, and `process_weights_after_loading` then replaces those
parameters with objects the offloader never sees.

**Candidate approach.** Register the offload ourselves from
`process_weights_after_loading`, where the finished tensors already exist at final
shape: reach `get_offloader()`, and if it is a `UVAOffloader` with budget left, pin
and wrap the tensor as `uva.py` would, updating its byte counter. Bits-agnostic,
checkpoint-vintage-agnostic, no vLLM changes. Accepts a known cost — it depends on a
vLLM internal that can break on a version bump.

→ [docs/format-and-loading.md](docs/format-and-loading.md) "CPU offload"

## `qbench-noise-floor` — Self-noise-floor support for qbench's `vllm` engine

The `vllm` engine cannot be the `reference` group with `noise_floor` at its default,
because it has no noise injection: vLLM's decoder layers are not at a predictable,
engine-version-stable location the way `TransformersBackend`'s forward-hook approach
needs one. Every cross-format comparison run through the served path therefore has
to borrow a noise floor from another engine.

→ [docs/qbench.md](docs/qbench.md)

## `retire-gemma4-patch` — Retire `patches/vllm-gemma4-transformers-5.15-per-layer.patch`

vLLM landed their own fix for the transformers 5.15 per-layer config break upstream:
[70b84f0](https://github.com/vllm-project/vllm/commit/70b84f0bcbb6d0a35b74b1035673a1c934089dbb)
(PR #49797, hmellor), and did it generically — a real
`ModelArchitectureConfig.from_layers()` / per-layer arch-config plumbing through
`get_num_kv_heads`/`get_num_attention_heads`, not a gemma-4-only patch like ours.

**Already verified, on 2026-08-17.** A `bench/` dry run against a 0.27.2 preview
(`vllm-main` @ `v0.27.2rc0-136-gfdab2b10bc`) carrying **only** the fused-param and
ReplicatedLinear patches — no gemma-4 patch — had `gemma-4-12B 3.0bpw mul1 tied`
compare **bit-identical** to its baseline: `argmax 0`, `|dlogprob| max 0.000e+00`
across 84 scored positions, greedy unchanged. Upstream's generic fix covers what
ours did.

So at the bump this is mechanical: drop the patch, update the README table. The
re-verification is already done and the entry will keep doing it.

## Recently closed

*One line each, newest first. Prune to ~10 when appending.*

- `gguf-embeddings` — decided 2026-08-17, see
  [docs/embeddings.md](docs/embeddings.md) "Is GGUF the right storage format?". No:
  at matched bytes GGUF's k-quants tie a naive encoder in GGUF's own *layout*
  (+0.000308 vs +0.000428 on Qwen3.5-9B at 4.5 bpw, both below the noise floor) and
  lose to it at 3.5 bpw. The win is block-scaled granularity with hierarchical
  scales, which is ~30 lines to encode; adopting GGUF would buy that at the cost of
  an encoder dependency (`gguf-py` cannot write k-quants), a fixed depth menu, and a
  cross-plugin runtime dependency, for no interoperability the hybrid checkpoint
  could use anyway.

- `embed-rows-compile` — done 2026-08-16, see
  [docs/embeddings.md](docs/embeddings.md) "Serving under torch.compile". Tied
  embedding serving did not survive vLLM's *default* execution mode at all;
  `ops.embed_rows` is now an opaque custom op with a capture-safe path below
  `EXL3_EMBED_STATIC_MAX`. Found by `bench/` on its first run.

- `embed-head-depth-study` — done 2026-08-15, see
  [docs/embeddings.md](docs/embeddings.md). Established scalar quantization over
  trellis for embeddings, trellis for heads, additivity of the two, and
  head-sets-the-depth for a shared tensor. Its "no universal depth constant" finding
  was per-row-specific and is superseded by `gguf-embeddings`.
- `tied-embedding-serving` — done 2026-08-15, see
  [docs/embeddings.md](docs/embeddings.md) "Phase A result". Tied models serve their
  embedding from the quantized `lm_head`; Qwen3-0.6B 508 → 323 MiB resident,
  gemma-4-12B +1.15 GiB of KV headroom, ~3% decode cost.
- `qbench-vllm-engine` — done 2026-08-14, see [docs/qbench.md](docs/qbench.md). A
  `vllm` engine for qbench, plus four bugs real usage surfaced and the first
  cross-format comparison on the actually-served path.
