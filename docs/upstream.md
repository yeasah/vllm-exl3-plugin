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

*The speculative-decoding objection is withdrawn.* It was recorded 2026-08-20 as "the
patch hands the drafter's embedding an EXL3 method", and reproducing it on v0.28.0
(below) showed the patch is not involved: `gemma4_mtp.py` passes `quant_config`
**explicitly**, from `get_draft_quant_config(vllm_config)`, so the ambient default never
fires. The breakage is upstream's own and is filed separately as the next item. What
remains before offering the patch is ordinary review work — it should still be judged
against the plain-`nn.Linear` gap below, which is the same problem in a different place.

### Reports, not patches

**A separate-checkpoint drafter inherits the target's quantization.** Reproduced on
v0.28.0, 2026-08-25, and this one is pure upstream — no plugin, no patch, and any
quantization format hits it.

```
vllm serve turboderp/gemma-4-12B-it-exl3 --revision 4.00bpw_mul1 \
  --speculative-config '{"model":"google/gemma-4-12B-it-assistant","num_speculative_tokens":4}'

ValueError: There is no module or parameter named 'model.embed_tokens.weight' in
Gemma4MTP. The available parameters belonging to model.embed_tokens
(VocabParallelEmbedding) are: {'model.embed_tokens.trellis', 'model.embed_tokens.mul1',
 'model.embed_tokens.svh', 'model.embed_tokens.suh'}
```

The chain, all in-tree:

1. `config/speculative.py` rewrites `model_type` `gemma4_unified_assistant` →
   `gemma4_mtp` and sets `architectures: ["Gemma4MTPModel"]`, so the assistant is served
   by the MTP implementation.
2. That routes it into the `method == "mtp"` branch, which copies the target's
   quantization onto the draft: `if not self.quantization: self.quantization =
   self.target_model_config.quantization`. Its own comment gives the assumption —
   *"use the draft model from the same model"* — which is true of real MTP, whose
   weights live inside the target checkpoint.
3. **gemma-4's assistant is a separate checkpoint** with its own plain bf16
   `model.embed_tokens.weight` `[262144, 1024]`, at a hidden size the target does not
   even share (1024 against 3840).
4. `get_draft_quant_config` — whose docstring correctly says *"Draft models should use
   their own quantization config instead of the verifier/target model's"* — then
   faithfully returns the target's, because step 2 stamped it onto the draft config.
5. `gemma4_mtp.py:361` builds the embedding with it, registering `trellis/suh/svh/mul1`,
   and the drafter's own `.weight` has nowhere to land.

*Why it is a good report*: the intent is already written down in the helper's docstring,
and the defect is one conditional away from it. The fix is to not inherit quantization
when the draft model is a different checkpoint from the target — which step 1's own
rewrite is what makes possible to detect. Nothing about it is EXL3-specific: an AWQ,
GPTQ or compressed-tensors gemma-4 with this drafter fails identically.

*Last step before filing*: confirm it reproduces with our patches unapplied. The code
path says they are irrelevant (the config is passed explicitly, not defaulted), but the
report should say "verified" rather than "should not matter".

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

**`sc_measure.py` reports a KLD with a constant additive floor** of ~6.1e-5, measured on
phi-4-mini. Evidence it is a floor rather than curve shape: flat across sensitivity
quintiles spanning 51x, log-log correlation with sensitivity 0.088, and it reproduces at
5.3e-5 in an independent run with different rows, trace and noise levels. Cause is fp16
logits carrying independent rounding in reference and perturbed passes.

*Why it matters to them, not just to us*: subtracting it moves `sc_optimize`'s fitted
alpha from **1.791 to 1.996** — the exact square law theory predicts in the small-error
limit. Fix is to fit `kld = c + s·rfn²`; with alpha pinned at 2.0, two noise levels
solve for both and no extra passes are needed: `c = (4·kld_lo − kld_hi)/3`.
Independent of whether per-tensor allocation is worth doing, which measurement says it
largely is not. Detail in [qbench.md](qbench.md).

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
