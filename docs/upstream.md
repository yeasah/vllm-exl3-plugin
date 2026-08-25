# Upstream: what we owe other projects, and what we want from them

*The queue of outward-facing work, in one place. Tracked in TODO as
`upstream-queue`, which is a pointer to this note rather than a duplicate of it.*

This project accumulates findings about other people's code faster than it acts on
them, because acting is a different kind of work: a measurement is finished when it is
right, an upstream contribution is finished when someone else agrees. Scattered across
TODO items and subject notes, that queue was invisible as a queue — you could not see
what was ready, what was blocked, or what was worth doing first.

## What belongs here, and in what shape

Three shapes, and confusing them is how good findings get filed badly:

- **Patch** — we carry a diff, it works, and someone else would want it. The bar is
  that it is defensible *as a general change*, not as a thing that unblocks us.
- **Report** — a defect or gap we can demonstrate but should not unilaterally fix,
  usually because the right fix is a design decision that belongs to the maintainers.
- **Question** — we do not understand something well enough to file it yet, and the
  honest first move is to ask rather than to assert.

A fourth category earns its place by exclusion: **carried, not offered** — patches we
depend on that nobody else plausibly wants, which are ours to maintain and not worth
anyone's review time.

**The strongest items share a property**: they cost the recipient nothing to accept
beyond the fix itself. No adoption of our format, no dependency on this plugin, no
agreement with our design taste. Where an item fails that test it is noted, because it
changes both the priority and the framing.

---

## vLLM

### Ready to offer

**`vllm-transformers-backend-logit-softcap` — half a patch.** The Transformers backend
reads only `logit_scale`, never a non-standard spelling (MuseGlimmer's
`output_multiplier`), and `LogitsProcessor` applies its scale *after* the soft cap where
such a model needs it before. Upstream landed `soft_cap=final_logit_softcapping` at
0.28; these two remain. The fix is the fold-into-cap identity, which the existing
`LogitsProcessor` expresses exactly and which reduces to today's behaviour when the
multiplier is 1:

```
tanh(z / (T/m)) · (T/m) · m  ==  T · tanh(z · m / T)
```

*Evidence*: greedy output is unaffected (both transforms are monotonic), but every
reported logprob and every temperature-dependent sampling decision is wrong by 1/0.196
≈ 5.1x in effective temperature. Gated by `bench/` entry
`muse-glimmer-30B-2.0bpw-via-transformers-backend`, which captures at 0.000e+00 with the
patch and is what verified the halving is arithmetically identical to the pre-0.28 form.

**`vllm-replicated-linear-weight-loader-v2`.** `ReplicatedLinear` is the one
`LinearBase` subclass with no `weight_loader_v2` branch, which any quantized multimodal
model on the Transformers backend needs. Untouched upstream through 0.28.
*Before filing*: check the `RowvLLMParameter`-narrowing edge noted in
[transformers-backend.md](transformers-backend.md) — it is the one place this could be
wrong in a way review would catch and we would not.

### Blocked on a design decision that is not ours

**`vllm-embed-quant-config` — the 86-file problem.** 86 of 131 vLLM model files omit
`quant_config` when constructing their `VocabParallelEmbedding`, so no quantized
embedding can be served on those architectures — silently dense for a tied model, a load
failure for a block-quantized one. Our patch defaults it from
`get_current_vllm_config()` in one place.

*Strength*: an ecosystem fix rather than ours alone — `vllm-gguf-plugin` is blocked by
exactly the same gap.

**Reproduced with official tools only, 2026-08-25 — no EXL3, no plugin, no format of
ours.** llm-compressor's embedding example validates on `pythia-1.4b` and states the
result is "ready to be loaded into vLLM". Pythia is `GPTNeoXForCausalLM`, which vLLM
serves from `gpt_neox.py:208` — a file that builds its embedding as
`VocabParallelEmbedding(config.vocab_size, config.hidden_size)`, no `quant_config`. So
following their documented recipe on their own example family produces a checkpoint
vLLM cannot load:

```
# llm-compressor, in its own venv (it wants transformers <= 5.14.1)
QuantizationModifier(config_groups={"embedding": {"targets": ["Embedding"], "weights":
    {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group",
     "group_size": 64}}})
# -> gpt_neox.embed_in.weight_packed / weight_scale / weight_shape, format pack-quantized

# stock vLLM v0.28.0
ValueError: There is no module or parameter named 'embed_in.weight_packed' in
GPTNeoXModel. The available parameters belonging to embed_in
(VocabParallelEmbedding) are: {'embed_in.weight'}
```

With `vllm-embed-quant-config` applied the same checkpoint loads and generates. That is
the whole report: their tool, their example model, their compatibility claim, and a
one-file fix.

**Two details that make it a better report than ours would have been.** The failure is
*loud* here — compressed-tensors' packed tensors have nowhere to land, so it raises,
where our tied-EXL3 case degrades silently. And it needs no argument about whether
embedding quantization is worthwhile: llm-compressor already shipped the feature and
documented the claim.

**Count re-verified on v0.28.0**: 85 of 131 model files constructing a
`VocabParallelEmbedding` omit `quant_config` (was 86 of 131 at v0.27.0). Affected files
include `gpt_neox.py`, `opt.py`, `bloom.py`, `phi.py`, `minicpm.py`.

**The speculative-decoding objection stands — measured 2026-08-25, after being wrongly
withdrawn earlier the same day.** Reverting *only* this patch on v0.28.0 and running the
target with `google/gemma-4-12B-it-assistant` as drafter loads cleanly; with the patch it
fails:

```
ValueError: There is no module or parameter named 'model.embed_tokens.weight' in
Gemma4MTP. The available parameters belonging to model.embed_tokens
(VocabParallelEmbedding) are: {'model.embed_tokens.trellis', 'model.embed_tokens.mul1',
 'model.embed_tokens.svh', 'model.embed_tokens.suh'}
```

The mechanism, now traced rather than inferred. `vocab_parallel_embedding.py:299` reads
`if quant_config is not None: quant_method = quant_config.get_quant_method(...)`, and
`get_draft_quant_config` correctly returns **None** for a drafter with no quantization
config of its own — so upstream gives it `UnquantizedEmbeddingMethod` and its plain
`.weight` loads. **Our patch turns that `None` into the ambient config**, which is the
target's, so the drafter's embedding is built in EXL3's shape and its own weight has
nowhere to land.

*A wrong turn worth recording*, because it is the kind that survives review: the chain
through `config/speculative.py`'s `method == "mtp"` branch — which does copy the target's
quantization onto the draft, with a comment saying it is for drafters living inside the
target checkpoint — looks like an exact fit and is not the cause here. It was refuted by
reverting one patch, which is the test that should have come first.

*The drafter break does not reproduce with official tools, and cannot* — it is caused by
this patch, not by anything upstream ships. That makes it not a second bug report but the
**reviewer's first objection, answered in advance**: here is the gap, here is a
reproduction on your own example model, and here is why the obvious fix breaks
separate-checkpoint drafters, measured rather than supposed.

*So the two candidate fixes from 2026-08-20 stand unchanged*: condition the ambient
default on the module belonging to the model the config describes (which
`VocabParallelEmbedding.__init__` cannot know), or fix the drafter's config so it stops
misdescribing what is being built. The second remains the cleaner contribution.

**And the patch is load-bearing for correctness, not only for loading.** The same run
without it produced `'1111.11.11.1.11.'` from `turboderp/gemma-4-12B-it-exl3` — it loads,
runs, and emits garbage, where the `bench/` entry with the patch captures correct output.
The path is now traced, and it is not the "silently dense" case at all — it is *our own*
predicted silent weight loss, observed for the first time.

Nothing leaked: the tied EXL3 checkpoint really does carry a dense
`model.language_model.embed_tokens.weight` `(262144, 3840)` BF16 *alongside*
`lm_head.trellis`, which is the pipeline defect [embeddings.md](embeddings.md) records —
the quantizer writes a full embedding regardless of tying. So
`UnquantizedEmbeddingMethod` loaded a real tensor and the embedding was fine. The head
was not:

1. with no quant method requested for the embedding, `config.py:566`
   (`self.embed_prefix = prefix`) never runs, so `embed_prefix` keeps its
   `"model.embed_tokens"` default from `:101`;
2. gemma-4 nests the embedding at `model.language_model.embed_tokens`;
3. `get_cache_scale_mapper` therefore renames `lm_head.*` onto a path that does not
   exist (`:406`), and `embedding_is_quantized()` is true so the rename does fire;
4. the trellis lands nowhere, **nothing objects**, and the tied head is left with no
   weights — hence plausible-looking garbage rather than an error.

That is verbatim the hazard TODO `quantized-embeddings` predicted: *"`get_cache_scale_mapper`
still fires with `embed_prefix` at its `model.embed_tokens` default, routing 755 MiB of
trellis to a module path that does not exist on a nested model, and nothing objects — so
the silent weight loss wants a guard of its own."* This run is the first observation of
it, and the argument for building that guard: the failure is completely silent and the
model keeps serving.

### Reports, not patches

**Media encoders cannot be offloaded, on any model, in any format.**
`get_offloader().wrap_modules()` has exactly one call site in vLLM, inside
`make_layers()` — the helper that builds a *text decoder's* `ModuleList`. Vision towers
build their own, so no encoder is ever offered to either offload backend.

*Why it is worth someone's time*: an encoder is the one offload target whose economics
are not a compromise — read once per image, never for a text-only request — so this is
the cheapest headroom in a multimodal deployment and it is unreachable. Sizes are
0.79–3.64 GiB, up to **21% of the package** on `Qwen3-VL-8B @3.0bpw`. Nine of ten
surveyed checkpoints ship a bf16 tower, so no quantization plugin can reach it either;
the fix has to be upstream. Numbers and method in
[media-encoders.md](media-encoders.md), reproducible with `tools/encoder_census.py`.

**TurboQuant and sliding-window models — a cluster, in decreasing confidence.**
Everything here is measured on `Laguna-XS-2.1-exl3@3.00bpw` at v0.28.0; see TODO
`turboquant-sliding-window` for the full reproduction.

1. **The documented `sliding_window` keyword crashes.**
   `--kv-cache-dtype-skip-layers sliding_window` is a supported spelling
   (`layers/attention/attention.py` matches it literally), but combining it with a
   turboquant cache dtype raises `ValueError: invalid literal for int() with base 10:
   'sliding_window'` — turboquant merges its own boundary skips with
   `sorted(existing | set(boundary), key=int)`. One line, and it blocks the exact
   invocation the feature is for. **Highest confidence, smallest fix.**
2. **`page_size_padded` goes stale when `block_size` is scaled.** The unifier's
   block-scaling branch does `replace(layer_spec, block_size=new_block_size)` and leaves
   the padded page alone, so `unpadded_page_size_bytes` overtakes it and the property's
   own assert fires. Confirmed by patching it: the assert clears and the next wall
   appears. In 0.28 it is the *sliding* spec that carries the padded page.
3. **The first/last-N sibling page class is unimplemented — and upstream says so.**
   With (2) patched, a turboquant layer fails as not divisible and not paddable, because
   the pool holds three page classes rather than two: turboquant auto-adds its own
   boundary skips, producing a native *full-attention* spec alongside native *sliding*
   and turboquant *full*. `Platform._align_..._block_size` handles one padded class and
   carries two comments reading `# To add the first/last-N sibling:`. That sibling is
   TurboQuant's boundary protection. **This is the real blocker**, and it is a design
   question rather than a patch we should write.
4. **Unexplained, and must not be filed yet**: turboquant reports
   `kv_cache_dtype not supported` for `turboquant_4bit_nc`, a dtype in its own
   `supported_kv_cache_dtypes`, with the full string reaching the backend. Reproduces on
   both gemma-4 and Laguna, so it is not model-specific. Hypothesis — boundary skips
   give some layers `"auto"` and global backend selection then validates turboquant
   against a native-dtype group — is unverified. **Settle before reporting.**

*Framing note*: `--kv-cache-dtype-skip-layers` is a **supported configuration** in 0.28,
not something we are abusing. `CacheConfig.skip_page_size_padded` is documented for
exactly it. That makes these bug reports against an intended feature rather than feature
requests, which is a much easier conversation.

**`--disable-sliding-window` fails on gemma-4.** vLLM assigns
`hf_text_config.sliding_window = None` (`config/model.py`), but transformers 5.x
declares `sliding_window: int = 512` on `Gemma4TextConfig` as a non-optional annotated
field and rejects it: `TypeError: Field 'sliding_window' expected int, got NoneType`.
A vLLM/transformers contract mismatch; it works on models whose config permits the
assignment, which is why it is not universally broken. Small, and belongs to whichever
side owns the contract.

**Plain `nn.Linear` modules are unreachable by any quantization plugin.** vLLM's native
MuseGlimmer builds its vision adapter as `self.c_fc = nn.Linear(...)`, which never
reaches `get_quant_method`, so a checkpoint with a quantized adapter cannot be served on
the native path at all. `DFlashQwen3Model.fc` is the same shape. Gated as
`bench/` entry `muse-glimmer-30B-2.0bpw-native`, whose `known_broken` reason carries the
measured error. *This is the general form of the `embed-quant-config` problem*, and a
fix worth filing should be judged against both.

### Carried, not offered

**`vllm-fused-param-capability-check`** — lets a parameter declare it splits fused
checkpoint tensors itself. Qwen3.5 will not load without it and `handles_fused_shards`
has no upstream equivalent, but it is a hook shaped around how this plugin loads. No
second consumer is known, so it is ours to maintain until one appears.

---

## llm-compressor

**`strategy: "channel"` is a trap for embeddings, and the docs offer it neutrally.**
The embedding-quantization example presents `"channel"` as a plain alternative to
`"group"`. Measured on three models, symmetric channel 4-bit costs **6.2x / 27.9x /
180.7x** the model's own noise floor (MiniCPM5-1B, gemma-4-12B-it, Qwen3.5-9B) — worse
in every case than the affine per-row scheme this project measured and rejected.

*Why it is a clean offer*: no format change, no adoption of anything of ours, no
argument about affine-vs-symmetric. One documented option that wants a warning or
removal. The arm that produced the numbers is verified bit-identical to
`compressed_tensors`' own encoder (`tools/ct_sym_check.py` in the qbench working dir),
so the measurement is of their scheme and not our transcription.

**The embedding kernel cannot do asymmetric, and does not say so.** This is the item
that matters most, and it is smaller than it looks.

vLLM's *linear* WNA16 scheme handles asymmetry properly: it takes `symmetric`, validates
it against `WNA16_ZP_SUPPORTED_TYPES_MAP`, raises a clear error for unsupported widths,
and passes `zero_points=not self.symmetric`. `CompressedTensorsEmbeddingWNA16Int` does
none of that. Its Triton kernel hardcodes a symmetric offset and reads no zero point:

```python
q = ((packed >> shift) & ((1 << NUM_BITS) - 1)) - (1 << (NUM_BITS - 1))
out = q.to(tl.float32) * scale.to(tl.float32)
```

and its `__init__` reads `num_bits`, `strategy` and `group_size` but never
`weight_quant.symmetric` — so it neither supports asymmetric nor refuses it, where the
linear path next door refuses cleanly.

*The ask, in two parts*: bring the embedding kernel to parity with the linear one
(register `weight_zero_point`, apply it), and in the interim **raise** rather than
silently mis-dequantize, exactly as `compressed_tensors_wNa16.py` already does.

*Why it is worth doing rather than just correct*: symmetric is what makes their current
embedding support mediocre. Measured on Qwen3.5-9B, symmetric group-32 costs 2.64x the
model's noise floor where an affine scheme at comparable bytes costs 0.17x — and their
own default, group-64, is 3.44x. Asymmetric group-64 with a packed zero point would be
**~4.31 bpw, cheaper than `blockq32`'s 4.53**, while landing in roughly the affine
league our per-block arms measured. The fix does not need blockq, or us, or any format
change: the storage exists, the compressor writes it, the linear path reads it.

*And this is the piece that makes composition work.* compressed-tensors is one
`quant_config` that dispatches per `config_groups`, and non-uniform recipes are a
documented, supported feature — mixed precisions, mixed strategies, even different
modifiers per module family (AWQ on `self_attn`, GPTQ on `mlp`), all running directly in
vLLM. So an embedding scheme expressed *there* composes with any weight quantization on
any supported architecture, which is precisely what a plugin-side format cannot do. See
[embeddings.md](embeddings.md), "blockq on non-EXL3 checkpoints".

*The bigger finding is deliberately not offered*: their **group** strategy costs ~16x at
matched bytes against `blockq32` on the model with enough dynamic range to resolve it.
That would need a zero-point their embedding kernel does not read, i.e. a format change,
and it is only worth raising if there is appetite. Evidence in
[embeddings.md](embeddings.md).

---

## exllamav3

*We track a fork (`yeasah/exllamav3`), so "upstream" here means turboderp's tree and the
bar is higher: changes we make for ourselves do not need offering, only ones that are
right for everyone.*

**~~`sc_measure.py` reports a KLD with a constant additive floor~~ — retired 2026-08-25,
unfiled.** The report targeted `util/sc_measure.py` and `sc_optimize`'s power-law fit.
Both are **gone**: upstream's v1.4.3 deprecates `measure.py`/`optimize.py` and replaces
the whole thing with a new pipeline (`conversion/measure_model.py`,
`conversion/optimize_model.py`, `abf4911`), and the `sc_*` tools no longer exist in the
tree at all.

The half of the finding that was about *consequence* is moot with them: the new optimizer
fits no alpha, so there is nothing for a constant floor to bias from 1.791 to 1.996. The
half about *cause* — fp16 logits carrying independent rounding in reference and perturbed
passes — is not obviously fixed or obviously still harmful; `measure_model.kldiv`
upcasts to float for the softmax but the logits reaching it are whatever the model
produced. Deciding whether a version of this applies to the replacement is fresh analysis
against different code, not a rewrite of the old report.

*Recorded rather than deleted because the retirement is the useful part*: this is the
standing check below catching a stale item **before** it was sent, rather than after —
which is the first time round that has happened.

---

## Priority, and the reasoning

Ordered by *value to the recipient per unit of our effort*, not by how much each one
annoys us:

1. **llm-compressor channel strategy.** Finished evidence, clean framing, actively
   harmful default, nothing to negotiate. The only item that is purely someone else's
   benefit, which makes it the easiest to send.
2. **TurboQuant `key=int`.** One line, high confidence, blocks a documented feature.
   Send with (2) and (3) as context, since a maintainer will immediately ask what is
   behind it.
3. **The softcap half-patch.** Working code, gated by a bench entry, small.
4. **Encoder offload.** Largest payoff of the set for the widest audience, but it is a
   report against a design gap and will want a conversation rather than a diff.
5. **exllamav3 KLD floor.** Small and well-evidenced, but the fork relationship means it
   is the least urgent — we already have the fix where we need it.
6. **`embed-quant-config`.** Highest value to us, and deliberately last: it needs a
   design decision, a reproduction we have not built, and a fix judged against two
   other manifestations. Filing it early would waste the good will the others earn.

**Not yet fileable**: the turboquant `kv_cache_dtype` mystery (4), until the hypothesis
is verified or replaced.

## A standing check before anything is sent

- Does it reproduce on a clean checkout of the current release, with a command someone
  else can run?
- Is the fix defensible without reference to this project?
- Have we checked whether upstream already fixed it? *We have now paid this cost four
  times* — see `check-upstream-before-patching-vllm`. The turboquant walls were measured
  against a structure 0.28 had already replaced.
- If it is a report rather than a patch, is the thing we do not understand stated as
  such rather than smoothed over?
