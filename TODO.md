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
embeddings are skipped entirely and left at full resolution. A post-processing tool
preserves the very large investment in computation the published EXL3 collection
represents, rather than requiring it be redone.

**Half of this now exists.** `tools/quantize_embedding.py` rewrites one checkpoint's
embedding into the block-scaled 4-bit format the measurements chose, hardlinking every
shard it does not touch — 12 seconds and 1.36 GiB saved on Qwen3.5-9B. What it does
*not* do is the rest of a repair tool:

- **Drop a tied model's redundant `lm_head`.** The other pipeline mistake, untouched.
  The plugin already ignores those bytes at load, so this is purely a file-size fix —
  which is exactly what a downloader cares about.
- **Take a Hub repo rather than a local directory**, and write something publishable:
  a model card noting what changed, and the revision it was derived from.
- **Batch**, since the value is in repairing a collection, not one checkpoint.

Depth stays a shipped constant (4 bits) unless something argues otherwise; an override
is worth having, and a size-budget solver belongs to the full quantizer, where layer
bpw is actually free rather than fixed.

→ [docs/embeddings.md](docs/embeddings.md)

## `quantized-embeddings` — VRAM-efficient use of quantized embeddings

Serving the embedding quantized rather than dequantizing it at load. Dequantizing at
load proves the math and saves file storage and I/O, but leaves VRAM — the thing
this project cares about most — completely unchanged.

**Both shapes now serve.** Tied models come from the checkpoint's existing quantized
`lm_head`, with the fp16 `embed_tokens` never loaded and no tooling needed. Untied
models — which have no quantized copy to reuse — get one from
`tools/quantize_embedding.py` in the block-scaled 4-bit format of
`vllm_exl3_plugin/blockq.py`, served by `EXL3BlockQEmbeddingMethod`: Qwen3.5-9B goes
1940 -> 549 MiB resident and 6.72 -> 5.36 GiB on disk, at a KLD tax at or below the
model's own noise floor. All hooks are sanctioned vLLM extension points; nothing is
monkeypatched.

Four things remain, in rough order of how much they would cost to discover late.

1. **Most architectures never ask for an embedding quant method**, and the fix is
   carried in `patches/vllm-embed-quant-config.patch` rather than upstream. 86 of 131
   vLLM model files omit `quant_config` when constructing their
   `VocabParallelEmbedding`, so neither shape can serve there — silently dense for a
   tied model, a load failure for a block-quantized one. The patch is one file
   (default the config from `get_current_vllm_config()`, as `linear.py` already
   does), verified not to disturb configs that do not quantize embeddings.
   **Worth offering upstream**: `vllm-gguf-plugin` is blocked by exactly the same
   thing, so it is an ecosystem fix and not only ours.
2. **Tensor parallelism is written but unproven.** All three stored tensors slice on
   dim 0, so `tp.ROLE_VOCAB` is a row slice with none of the trellis path's 128-row
   Hadamard alignment rule — which is why it is a handful of lines. It has never run
   on more than one GPU. Needs the `vast` box, alongside `moe-tp` and the TP tier of
   `bench-suite`.
3. **`bench/` does not gate any of it.** The bump gate pulls checkpoints from the Hub
   and no repaired checkpoint is published, so covering this needs either a published
   fixture or a bench step that produces one from a checkpoint it already pulls. Until
   then a vLLM or exllamav3 bump can break the block-quantized path silently.
4. **Only 4 bits is packed.** That is deliberate — one depth covers every model
   measured, and nibbles keep both ends byte-aligned — but 3 bits is usable at ~3.5
   bpw and would want the packing if a checkpoint ever calls for it.

**A tied model with a block-quantized embedding crashes** (found 2026-08-19). The
two predicates in `quantization/config.py` treat the cases as mutually exclusive —
`embedding_is_blockq()` says so in as many words — but a tied checkpoint whose
embedding has been repaired makes both true. It loads without complaint and dies at
logits time: `EXL3TiedLMHeadMethod` reads a trellis off the embedding module, which
now holds `bq_*` instead. Worse, `get_cache_scale_mapper` still fires with
`embed_prefix` at its `"model.embed_tokens"` default, routing 755 MiB of trellis to a
module path that does not exist on a nested model, and nothing objects — so the
silent weight loss wants a guard of its own. The fix is to make both predicates
per-module rather than per-checkpoint, and to point the rename at the head's own
prefix, which still defeats the loader's `lm_head` skip. Not hypothetical:
`gemma4-e2b` is tied *and* needs blockq for its per-layer embedding.

**The shared tied-model tensor stays deferred**, on the same grounds as before: one
tensor serving both roles needs a scalar-integer GEMM for the head, which does not
exist anywhere, and sharing would otherwise mean dequantizing to dense fp16 and giving
back the whole saving. It also only helps tied models, and gemma-4 is nearly its whole
constituency — so it is best read as a follow-up to `fa-head-dim-512` rather than an
independent goal.

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
  the instrument that settled Muse's text path. The other instrument is
  gemma-4 E2B/E4B, the only model on hand with a genuinely *separate* audio
  encoder and so the only one where a quantized audio component could be blamed
  at all — blocked on `gemma4-e2b`.
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

## `gemma4-e2b` — Quantizing gemma-4 E2B/E4B

**Outcome wanted:** an EXL3 E2B checkpoint serving through vLLM with both of its
embeddings quantized. It is the sharpest available demonstration that the embedding
tax is real: 54% of this model is embeddings, the pipeline as it stands saves 27% on
a model sold as 2B, and the repaired one projects 65%.

**Explicitly not a priority.** The thesis is already proven on Qwen3.5-9B with
measured KLD; E2B makes the argument undeniable rather than more true, and the cost
is architecture work in the exllamav3 fork — the most expensive category here, with
no cheap gate for correctness. What would raise it: the appliance wanting a small
vision-capable model, or the audio question under `multimodal` blocking something.

**vLLM already serves it.** The BF16 model runs and describes images correctly
(2026-08-19), so nothing is blocked on the serving side. The gap is entirely
conversion.

**The blocker chain, in order:**

1. **exllamav3 refuses the architecture.** `Config.from_directory` raises
   `NotImplementedError("Gemma4 per-layer inputs are not implemented yet")` —
   a bare guard in `architecture/gemma4.py`, no partial implementation behind it.
   What it is guarding is small: six tensor patterns
   (`per_layer_model_projection`, `per_layer_projection_norm`, and per-layer
   `per_layer_projection` / `per_layer_input_gate` / `post_per_layer_input_norm`),
   77 MiB, 0.8% of the model.
2. **`tools/quantize_embedding.py` handles one embedding suffix**, and this model
   has two.
3. **Tied + blockq crashes**, which is a live defect recorded under
   `quantized-embeddings` rather than here. E2B is tied *and* needs blockq for its
   per-layer tensor, so it cannot serve until that is fixed.

**Text-only is the version worth building.** E2B's vision tower is a full ViT encoder
and its audio tower a conformer with depthwise conv — both unrelated to anything
exllamav3 handles today, and together only 0.88 GiB. Pass them through dense and
serve with `--language-model-only`. One trap if either is ever quantized: every
linear in both towers carries `input_max`/`input_min`/`output_max`/`output_min`
siblings alongside `X.linear.weight`. Those are QAT ranges, not weights, and have to
be recognized as such.

→ [docs/embeddings.md](docs/embeddings.md) "The most extreme case found"

## `exl3-metadata` — Improving the metadata situation

exllamav3 plays fast and loose with checkpoint metadata; important things are
missing or misleading. Low-hanging fruit, and restoring some trust in the stored
data would be useful — significantly tempered by the fact that existing checkpoints
keep their incomplete metadata regardless, unless we repair and republish them
(license permitting).

**Candidate approach: declare rules, not instances.** `tensor_storage` enumerates one
entry per module, which is why it is untrustworthy — absence is ambiguous between "not
quantized" and "we failed to record it", and `Muse-Glimmer-30B-exl3` omits 303
quantized modules with no way to tell from the file. compressed-tensors solves the
same problem with `config_groups`: named groups whose `targets` are regexes over
module names, plus an explicit `ignore` list for exceptions. Two regexes describe a
whole mixed-precision 35B MoE where our map needs 667 entries for gemma-4-12B, and a
rule cannot omit a module because it is not listing them. Worth recording `provider`
and a format name alongside, as that checkpoint does — knowing which quantizer and
version produced a file is free and we do not store it.

**What it unblocks, which is the reason to care.** Per-module bit width becomes
knowable at `create_weights` time, reliably and for every module. That is the exact
precondition `cpu-offload` had to reject the AWQ preallocate-and-mutate pattern over
— see that item, and [docs/format-and-loading.md](docs/format-and-loading.md) "CPU
offload", where the objection is recorded.

**It reaches existing checkpoints too, via `repair-tool`.** A rule map can be
*derived* from a checkpoint's actual tensor shapes at repair time rather than trusted
from its config, so the published collection gains the property as it is repaired
instead of waiting on re-quantization. That is also a cheap consistency check on the
metadata already there: derive the groups, compare against `tensor_storage`, and any
disagreement is a bug in one of them.

→ [docs/format-and-loading.md](docs/format-and-loading.md)

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
3.63 GiB for the same model as AWQ. Two independent causes, both traced: the offloader
decides eligibility at construction time when our parameters are still empty CPU
placeholders, and `process_weights_after_loading` then replaces those parameters with
objects the offloader never sees. **Both causes hit both backends** — `PrefetchOffloader`
fails identically, just louder.

**Why this is worth doing, and it is not capacity.** With `quantized-embeddings` and
`repair-tool` landed, these models now fit 16 GiB — but only at 2.0-2.5bpw, well past
the knee of the KLD curve. Offload trades PCIe bandwidth for bits per weight, which is
a quality lever rather than a fallback. **Sized on 2026-08-20, and the answer splits.** On the
one routed MoE measured, vLLM frees only ~1 GiB of a claimed 8 and the reason is not
yet understood; on dense models it offloads honestly, verified against awq, gptq and
compressed-tensors. So the ceiling is not a general property — but dense is also where
every offloaded byte is re-read every token, so the good ceiling and the good
throughput are on opposite sides. Valuable on the margin either way, not a step up the
bpw curve on its own.

**Candidate approach.** Register the offload ourselves from
`process_weights_after_loading`, where the finished tensors already exist at final
shape: reach `get_offloader()` and hand it the real tensor as the backend would have,
**pinned** — pinning is required both by `PrefetchOffloader`'s own assert and by
PyTorch, which refuses unpinned H2D during CUDA graph capture. Bits-agnostic,
checkpoint-vintage-agnostic, no vLLM changes. Accepts a known cost — it depends on a
vLLM internal that can break on a version bump.

**Scope note: which backend wins depends on access pattern, and MoE inverts the
obvious answer.** Prefetch overlaps transfer with compute, so it suits dense weights
read every pass. But it is routing-blind — it copies every offloaded parameter each
forward pass — while UVA's zero-copy reads touch only what the kernel actually reads.
For routed experts, and for sparsely-read tensors like a vision tower, UVA therefore
wins decisively; measured on 2026-08-20. Only one backend is active per process, so if weights are being
offloaded the tower stays resident; at 0.3-0.6 GiB that is the cheap end of the
trade. Verified 2026-08-19 that the prefetch path is otherwise fully functional with
EXL3 tensors, so pinning is the only outstanding requirement.

**Honor the parameter selectors in that registration.** Both backends take a set of
parameter-name segments and offload *only* what matches — `--cpu-offload-params` for
UVA, `--offload-params` for prefetch, both exact dot-delimited segments
(`f".{param}." in f".{name}."`). Registering our tensors without that check makes the
EXL3 path silently non-selective, which nobody notices until someone tries the
selective form and gets the whole model offloaded. Four lines at the time, an
irritation to retrofit — and selectivity is what makes the sparse case work at all:
`--cpu-offload-params experts` is why a routed model pays PCIe for only the experts it
actually reads.

→ [docs/format-and-loading.md](docs/format-and-loading.md) "CPU offload"

## `qbench-noise-floor` — Self-noise-floor support for qbench's `vllm` engine

The `vllm` engine cannot be the `reference` group with `noise_floor` at its default,
because it has no noise injection: vLLM's decoder layers are not at a predictable,
engine-version-stable location the way `TransformersBackend`'s forward-hook approach
needs one. Every cross-format comparison run through the served path therefore has
to borrow a noise floor from another engine.

→ [docs/qbench.md](docs/qbench.md)

## `capability-suite` — Measuring capability through the served path

**Outcome wanted:** task-level numbers for a configuration as it is actually served —
quantized weights, quantized embedding, quantized KV — so quality claims stop being
divergence figures plus personal impressions.

**Why it is the missing instrument.** qbench answers "how far is this distribution from
the reference", which ranks encodings well and says nothing about whether a 3.0bpw model
with a 4-bit KV cache still writes working code, follows a long agentic trace, or recalls
something from 40K tokens back. Those are the questions the appliance actually ships
against, and the two newest axes — block-quantized embeddings and KV compression — are
precisely where divergence is least informative, since KV damage surfaces as retrieval
failure deep in the window rather than as perplexity at a position.

**Candidate approach: give exllamav3's existing task harnesses a vLLM execution path.**
`eval/` already has `mmlu.py`, `humaneval.py`, `ifbench.py`, `bbeh_mini.py`, `longctx.py`
and `spec_decode.py`, with prompts, graders and datasets solved. But every one drives
native exllamav3 (`model_init.init` / `Generator` / `Job`), so none can see a
block-quantized embedding (exllamav3 cannot load the format) or vLLM-side KV
quantization. This is the same move qbench made for scoring: keep the task definitions,
add an engine. `longctx.py` first — it is the probe for the least understood axis, and it
already works on whole documents with altered variants rather than synthetic needles.
It is also the least sampling-sensitive task in the set, being a retrieval probe, so the
engine path can land greedy-only and the tiering below can wait for `humaneval`/`ifbench`.

**Sampling is an axis of the suite, not a setting.** Greedy is the primary tier, since most
knobs are distribution-relative — `top_p` cuts at a mass threshold and quantization fattens
the tail, so one setting admits more junk from a 3.0bpw model than from the reference, and
part of any reported gap is then the sampler rather than the encoding; penalties are worse,
being cumulative over a trace. But greedy alone is a known blind spot — the soft-cap gap
under `multimodal` wrecks sampled output while leaving greedy intact — so a second tier
runs each model's recommended parameters, which is also the tier that answers what the
appliance ships. There, **snapshot the parameters into the suite config** with the model
card revision and date they came from rather than reading the card at run time: cards get
edited silently, and a baseline that follows one stops meaning what it said. Thinking and
non-thinking are separate entries, carrying different recommendations. Whichever tier
produced a number belongs in the artifact beside the vLLM pin.

→ [docs/qbench.md](docs/qbench.md) (scope: why divergence is deliberately all qbench
measures), [docs/embeddings.md](docs/embeddings.md)

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
