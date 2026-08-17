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
hand, so it is effectively the entire constituency for the shared per-row embed+head
tensor deferred under `quantized-embeddings`. If this turns out to be a lost cause and
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

**Candidate approach, settled by measurement.** Emit a **per-row integer** tensor,
not a re-quantized trellis. As an embedding, per-row beats the trellis by ~89x at
equal bits on gemma-4-12B, because the trellis optimizes `x @ W.T` for a head rather
than per-row accuracy — and it is simpler besides: no Hadamard, and row extraction
is a slice rather than a block decode. For tied models one shared per-row tensor
serves both roles and dominates both the shared trellis and any two-tensor split.

Depth cannot be a shipped constant: the 4-bit tax spans 35x across
gemma-4-12B / MiniCPM5-1B / Qwen3.5-9B. But it is constant in body bpw, so a single
calibration sweep at any one depth characterizes a model. The interface should be
that sweep or an explicit override — a size-budget solver belongs to the full
quantizer, where layer bpw is actually free.

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
2. **There is no per-row serving path.** Everything the plugin can load today is
   trellis: `stored_tensor_names()` is `trellis`/`suh`/`svh` plus a codebook tensor,
   and `EXL3EmbeddingMethod` serves via `ops.embed_rows`, which is trellis row
   extraction. The per-row scheme that the Phase 4 sweeps showed is ~89x better as
   an embedding exists **only as simulation** — qbench's `fake_quantize` rounds a
   resident fp16 tensor to a bits-bit grid and immediately dequantizes it,
   explicitly "in place of dtype/storage changes... without committing to a real
   packed format or a real kernel". Nothing is packed, stored, or loaded.

So the tensor format is decided but unimplemented at both ends. **Both tooling items
are blocked on this**: a repair tool emitting per-row tensors today would produce
checkpoints nothing can serve.

**Next up, and scoped to untied models: a per-row embedding tensor served alongside
the checkpoint's existing trellis head.** Needs a storage layout (packed values plus
per-row scale/zero-point), `create_weights`/`process_weights_after_loading`
registration, and a gather — which should be markedly simpler than the trellis path,
since row extraction becomes a slice with no Hadamard and no 128-block read
amplification. Per-row rows are also independent, so vocab-parallel TP carries none
of the 128-block alignment arithmetic the trellis path needs.

Two reasons this shape rather than the shared per-row tensor the frontier table
favours. It is the **only** shape untied models can use, since their head is a
genuinely different matrix. And it needs no new kernel: a shared tensor has to serve
the head role too, and there is no per-row-integer GEMM anywhere — sharing would mean
either dequantizing to dense fp16 (which gives back the whole saving) or a trip to
kernel town. Costs ~0.35 GiB against sharing on gemma-4-12B, and is still ~1.4 GiB
better than native.

It is also where the new value is. Tied models already run with a much improved VRAM
profile via Phase A; what they carry is a **KLD hit we now know is unnecessary**
(+0.0216 from the trellis where per-row costs +0.0002 at the same depth). Untied
models get nothing at all today.

The shared-tensor optimization is deferred, possibly a long way. It only helps tied
models, and gemma-4 is the only tied mid-size family on hand — so if gemma-4 proves
impractical to deploy, which currently rides on `fa-head-dim-512`, there is almost
nothing left for it to apply to. Natural follow-up to that item rather than an
independent one.

The lookup *plumbing* is built and de-risked by Phase A — the vLLM hooks,
`tie_weights`, the tied-skip mapper — and that part is encoding-agnostic. It is the
decode underneath it that is trellis-only.

**Bit-pack; arbitrary depths 4-8, not just byte-aligned ones.** *Open: whether to
adopt a GGUF quant type instead of a bespoke format — see `gguf-embeddings`, which
should be settled before this gets implemented.* The measured optima
are per-model and mostly not byte-aligned (4 for gemma untied, ~5 for MiniCPM5-1B, 6
for Qwen3.5-9B, 7 for tied gemma shared), so restricting to 4/8 would push most models
to 8 and hand back ~0.24 GiB on Qwen3.5-9B alone — 12-15% of the whole win. Storing a
6-bit value in an 8-bit slot is never the answer either: same bytes as 8-bit
quantization, worse quality.

The alignment worry does not arise. A row is `hidden x depth` bits and `hidden` is a
multiple of 8 in every real model (`num_heads x head_dim`, and EXL3 pads to 128
besides), so rows land on whole bytes at every depth — 3840 x 6 = 2880 B, 4096 x 6 =
3072 B, 1536 x 5 = 960 B. No row padding, every row independently sliceable, which is
the property that made per-row attractive in the first place. Only intra-row
extraction is left. Assert the multiple-of-8 rather than assuming it.

A fused kernel is the obvious highest-performing answer to the remaining per-token
cost, but is worth weighing against less costly approaches first: it is development
overhead and likely further pinning on CUDA, and exllamav3 has no ROCm support today
while talking about adding it.

→ [docs/embeddings.md](docs/embeddings.md)

## `gguf-embeddings` — Adopt a GGUF quant type instead of inventing a format?

**Outcome wanted:** a decision, before `quantized-embeddings` builds a bespoke
per-row format at both ends — is a GGUF quantization type the better storage
format for a quantized embedding?

**Why it is a real question rather than a detour.** Our own sweep concluded that
*block-scaled scalar* quantization beats the trellis by ~89x as an embedding,
because the trellis optimizes `x @ W.T` for a head while an embedding needs each
row accurate as a vector. GGUF's k-quants are that exact family, matured over
years. So this is not a competing bet against the Phase 4 result — it is the
possibility that we already decided what we want and someone has shipped it.

**Independent confirmation of the head/embedding asymmetry, from another
lineage.** `bartowski/Muse-Glimmer-30B-GGUF` @IQ2_M stores `token_embd.weight`
as **IQ3_S** (~3.4 bpw) and `output.weight` as **Q5_K** (~5.5 bpw) — the head
given ~2 more bpw than the embedding, which is what our head sweep measured
independently (the head is ~60x more bit-sensitive). Worth weighing: llama.cpp
arrived at the same shape by different means.

**What is actually available, checked rather than assumed. This is the part that
changes the calculus.** GGUF left vLLM's tree
([RFC #39583](https://github.com/vllm-project/vllm/issues/39583)) but not the
ecosystem: it moved to
[vllm-project/vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin),
an **official out-of-tree quantization plugin with the same architecture as this
one** — `quantization/config.py`, `linear.py`, `fused_moe.py`, a `plugin.py`
entry point. Apache-2.0, same as us, and actively developed (pushed 2026-08-16).

- **It already serves embeddings quantized, gathering rows.** Not dequant-at-load
  — `quantization/vocal_embeds.py` does
  `torch.index_select(qweight, dim=0, index=x_flat)` then
  `ops.ggml_dequantize(...)` on just those rows. **That is precisely the untied
  per-row serving path `quantized-embeddings` is blocked on**, already written.
- **The gather is far simpler than ours**, because GGUF's row-major block layout
  makes rows independently sliceable: no Hadamard, none of the 128-block read
  amplification `embed_rows` has to amortize, and no `torch.unique` — so none of
  the `embed-rows-compile` capture trouble either.
- **The GPU decode is not ours to write.** It ships Triton dequant kernels for
  q2_k..q8_1 and iq1_s..iq4_xs, plus CUDA (`csrc/gguf/dequantize.cuh`).
- **Decoding off-GPU is free too**: `gguf-py` dequantizes every type in pure
  Python, which is enough for the offline comparison below.
- **Encoding is the one real gap.** `gguf-py` quantizes only simple types; every
  k-quant raises `NotImplementedError`, since those encoders are C in llama.cpp.
  A repair tool would have to shell out to `llama-quantize`, bind ggml, or
  reimplement.

**The honest objection, and it may be weaker than it looks.** Using GGUF for the
embedding means a hybrid checkpoint: EXL3 trellis body, GGUF-quantized embedding.
Nothing else would read it — but nothing else would read a bespoke per-row
embedding either, since exllamav3 does not serve one. So the incompatibility is a
cost we are paying under either plan, and the question is only which format the
plugin has to implement.

**The real tension to resolve.** `quantized-embeddings` settled on *bit-pack,
arbitrary depths 4-8* because measured optima are per-model and rarely
byte-aligned. GGUF offers a fixed menu instead — but each type is a **better
encoder at a given size** (hierarchical scales, non-uniform codebooks for the IQ
types). So the trade is depth granularity against quality-per-byte, and the axis
that decides it is **quality at matched bytes**, not depth.

**Candidate approach: settle it by measurement before writing any format code.**
We hold `bartowski/Muse-Glimmer-30B-GGUF` and the EXL3 Muse-Glimmer of the same
base model. Dequantize the GGUF `token_embd` with `gguf-py`, and compare its
reconstruction error and downstream KLD against qbench's `fake_quantize` per-row
simulation at matched bytes — qbench already has the `embed_quant` hook for
exactly this sweep. No kernel work, no llama.cpp, and it answers the question
that decides the format.

**Then a second question, which only arises if the first says GGUF: reuse or
reimplement?** Two Apache-2.0 out-of-tree plugins registering different quant
methods through `vllm.general_plugins` should coexist, but "should" is doing work
there and nobody has tried it. The options are to depend on `vllm-gguf-plugin`
for the embedding path, vendor the pieces we need, or write our own against the
format. Worth noting the sibling is also simply *worth reading* regardless of the
outcome here — it is the only other example of this exact plugin shape, and it has
solved problems we have hit (weight adapters per model family, a generic GGUF
weight adapter rather than an allowlist).

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

- **Tied models**: one shared per-row integer tensor serving both roles. Dominates
  both the shared trellis and any two-tensor split. Depth is set by the head, which
  is ~60x more bit-sensitive; the embedding rides along above its own requirement,
  and that waste is still far cheaper than a second copy of the matrix.
- **Untied models**: a per-row integer embedding at a lower depth, alongside the
  trellis head exactly as produced today. The head is already right; only the
  embedding changes.

Depths are per-model — the 4-bit tax spans 35x across the three models measured —
but constant in body bpw, so one calibration sweep at any single depth
characterizes a model.

Distinct from `repair-tool` in one way that matters: here layer bpw is *free*, so
trading depth across components becomes a real constrained optimization (the
Lagrangian actually binds) rather than a one-variable heuristic. This is where the
size-budget solver belongs.

Depends on `quantized-embeddings` growing a per-row serving path — see there.
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

Next time the vLLM pin moves past that commit: drop our patch, update the README
patch table, and re-verify gemma-4-12B loads clean without it.

## Recently closed

*One line each, newest first. Prune to ~10 when appending.*

- `embed-rows-compile` — done 2026-08-16, see
  [docs/embeddings.md](docs/embeddings.md) "Serving under torch.compile". Tied
  embedding serving did not survive vLLM's *default* execution mode at all;
  `ops.embed_rows` is now an opaque custom op with a capture-safe path below
  `EXL3_EMBED_STATIC_MAX`. Found by `bench/` on its first run.

- `embed-head-depth-study` — done 2026-08-15, see
  [docs/embeddings.md](docs/embeddings.md). Established per-row over trellis for
  embeddings, trellis for heads, additivity of the two, head-sets-the-depth for a
  shared tensor, and that no universal depth constant exists.
- `tied-embedding-serving` — done 2026-08-15, see
  [docs/embeddings.md](docs/embeddings.md) "Phase A result". Tied models serve their
  embedding from the quantized `lm_head`; Qwen3-0.6B 508 → 323 MiB resident,
  gemma-4-12B +1.15 GiB of KV headroom, ~3% decode cost.
- `qbench-vllm-engine` — done 2026-08-14, see [docs/qbench.md](docs/qbench.md). A
  `vllm` engine for qbench, plus four bugs real usage surfaced and the first
  cross-format comparison on the actually-served path.
