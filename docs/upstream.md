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

**`vllm-turboquant-continuation-prefill-copy` — one line each for K and V.**
`TurboQuantAttentionImpl._continuation_prefill` assembles the cached + current K/V as

```python
k_full[:cached_len] = k_cached_trim.to(qdtype)
v_full[:cached_len] = v_cached_trim.to(qdtype)
```

The inverse rotation immediately above emits **fp16** (deliberately -- the comment cites
fp16 tensor cores), while `qdtype = query.dtype` follows the model and is **bf16** on
every model that reaches this path. So `.to(qdtype)` is a real conversion, and being
out-of-place it materializes a second full-context tensor that the slice assignment then
reads once and drops. `Tensor.copy_` converts inside the copy:

```python
k_full[:cached_len].copy_(k_cached_trim)
```

*Evidence*: measured at the real shapes (Hk=4, D=256, cached_len=117,401, fp16 -> bf16
through a transposed view) the assignment peaks at **230.0 MiB** and `copy_` at **0.0**,
bit-identical output. A CUDA memory capture of a 117K-token prefill on Qwen3.8-27B put
the four `_continuation_prefill` buffers at 914 MiB of a 930 MiB peak; this removes one
of them outright and the V-side equivalent at its own peak. Worth ~10K tokens of context
on a 16 GiB card.

*Still present on main* (checked 2026-09-02 at `e3e1241003`; the file has diverged for
the KV-layout refactor but these lines are untouched, and the only recent commit to it is
an unrelated MLA fix).

*Confirmed in situ*: three buffers remain live at the peak (`1080` k_full, `1081`
v_full, `1065` the rotation) at 232 MiB each on a 118,784-token context, where four would
have been 928 MiB. The isolated measurement and the served one agree.

*Not offered, and not pursued* -- the other two candidates were costed and declined:

- **Eliminating `1065`, the rotation temporary** (a third of what is left). Layout is not
  the obstacle: `k_full[:cached_len, h, :]` is a strided view with `ldc = Hk*D`, and
  `torch.mm` writes into one at zero cost, so a per-head loop of four small mms would drop
  the allocation outright. The obstacle is dtype -- the rotation is fp16 by design
  (`_tq_Pi_half = H.to(torch.float16)`) while `k_full` follows the query into bf16, and
  `mm` will not write fp16 inputs into a bf16 out. The clean version makes the dequant
  kernel emit bf16 so the path is single-dtype, which is a change to *their* numerics for
  *our* memory, argued from one card.
- **Chunking the KV dimension** with flash-style accumulation, the only thing that changes
  the shape rather than the slope. That is a rewrite of an attention backend we do not own.

Neither is plausibly monkeypatchable, neither is a patch worth carrying against a moving
backend, and neither matches the shape of ask that has been landing (see *Priority, and
the reasoning*). The asymmetry is the point: one line recovered 232 MiB because it asked
nothing of anyone's design; the remaining 692 MiB would cost a fork.

*The consequence*: TurboQuant's prefill transient stays at **6144 B/token, linear in
cached context** -- an axis the profile run never varies, so vLLM reports the same peak
activation for a 4K session and a 130K one. TQ headroom cannot be validated by a short
prompt, and the `--kv-cache-memory` pin stays load-bearing on tight configs. fp8 has no
equivalent problem: its transient is chunk-scaled, and the chunk is exactly what the
profiler varies.

**Declining to *offer* these is not declining to *have* them, and that distinction became
load-bearing on 2026-09-08**, when `kv-pager` died and KV eviction/offload went off the
table categorically. Compression is the only remaining lever on declared context per card,
so the 692 MiB is back in scope as work we carry -- and it was carried, on 2026-09-12:
see [memory-accounting.md](memory-accounting.md). Both the reserve and the transient are
gone, for 1580 MiB at the peak. Two things reopen at the same time, and both were missed above
because the framing was "what would upstream take":

- **A third option on the rotation temporary that argues with nobody.** The costing went
  straight from "per-head `mm` with `out=`" (dtype-blocked) to "make the dequant emit bf16"
  (their numerics, our memory) without considering *slabs*: rotate a few thousand tokens at
  a time and copy each slab into `k_full`, and the buffer is O(slab) with the dtypes,
  kernels and arithmetic all unchanged. It is a loop bound, not a design change.
- **Chunked-KV is not the rewrite it was priced as.** Both primitives already ship:
  `flash_attn_varlen_func` takes `return_softmax_lse=` and `out=`, and
  `vllm/v1/attention/ops/merge_attn_states.py` is written for precisely this split --
  prefix output/LSE from cache, suffix from the current chunk, with a
  `prefill_tokens_with_context` argument -- and upstream's own FLASH_ATTN backend merges
  that way in five places. The remaining work is causal bookkeeping across chunks.

**And there is a second cost the captures may not have counted.**
`TurboQuantMetadataBuilder._reserve_workspace` reserves the two fp16 dequant buffers at
`round_up(max_model_len - 1, block_size)` whenever chunked prefill is on and
`max_num_batched_tokens > 128` -- 4096 B/token of *declarable* context, ~486 MiB at 128K
on Hk=4/D=256, independent of whether a continuation ever runs. Unlike the transient it
prices the declaration rather than the request, which is the axis the appliance sells. It
is not obviously inside the 0.828 GiB figure in [kernels.md](kernels.md), and which side of
`consumed`/`peak_activation` it lands on has to be measured before it is designed around.

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

**A cold torch.compile costs 35K tokens of declared context, permanently** (measured
2026-09-12). `profile_run()` executes inside `memory_profiling()`
([gpu_worker.py:553](../deps/vllm/vllm/v1/worker/gpu_worker.py#L553)), that context resets
peak stats on entry ([mem_utils.py:291](../deps/vllm/vllm/utils/mem_utils.py#L291)), and
the activation term the KV budget subtracts is `torch_peak - torch_allocated`. On a cold
cache the first forward is also the compile, so inductor's compile-time allocations set
the mark. Measured on `Qwen3.8-27B-exl3@3.00bpw` at `gpu_memory_utilization=0.975`, the
same config reports **0.77 GiB of peak activation cold and 0.18 GiB warm** — a 0.59 GiB
difference, 35K tokens of `max_model_len`, and the difference between auto-fit declaring
the model's full 262144 and clamping it. `consumed` and CUDAGraph memory are identical
across the pair, and a warm *load* of the same artifact costs nothing, which is what makes
it attributable to compilation rather than to the graph.

*Why it is worth someone's time*: every operator's first launch after a version bump or a
flag change is the cold one, and the penalty sticks for the life of that process while
looking like a property of the release. **A downstream deployment already carries a
workaround for it** — the appliance project hit this independently and does warmup runs to
get a warm cache before the run that has to declare context — so this is a report about
removing a workaround people are already writing, not a hypothetical. It also makes any memory measurement silently
cache-dependent — the shape of the problem `bench-cache-control` exists for on our side.
Candidate fix is a reset between compilation and the measured forward, or compiling
outside the window.

**Not fileable yet** by the standing check below: everything above is a stock-vLLM
mechanism with no plugin involvement, but it was measured on the fork with an EXL3 model,
and *what* the compiler allocates is still a suspicion (inductor autotuning, on the
strength of `benchmark_combo_kernel: True` in the config) rather than a measurement. Both
gaps are cheap to close: one stock build, one stock model, one cold-then-warm pair.

### Carried, not offered

**`vllm-fused-param-capability-check`** — lets a parameter declare it splits fused
checkpoint tensors itself. Qwen3.5 will not load without it and `handles_fused_shards`
has no upstream equivalent, but it is a hook shaped around how this plugin loads. No
second consumer is known, so it is ours to maintain until one appears.

### What the 0.29 bump actually cost

Predicted 2026-09-08 against `v0.29.0rc6`, done 2026-09-09 against `v0.29.0`
(`98dff2a81d`, 187 commits past `v0.28.0`). Recorded together because the prediction was
half right in a way worth remembering.

**Right about the patches.** All seven were checked against `v0.29.0` before rebasing and
none had been fixed upstream, so nothing retired. TurboQuant's own code was frozen exactly
as measured -- `_continuation_prefill` and `_tq_Pi_half` byte-identical -- and the rebased
series is line-for-line the 0.28 series apart from two comment lines. Two commits needed
resolution:

- **`81566217e`** (page geometry, was `86e3b1eef`) in `v1/core/kv_cache_utils.py`, which is
  the KV-spec plumbing the prediction named. Upstream replaced
  `layer_spec.indexes_kv_by_block_stride` with `not isinstance(layer_spec, MLAAttentionSpec)`
  and **deleted the attribute from the tree**, so keeping our side of that conflict would
  have been an `AttributeError` at cache construction.
- **`fe1cdc942`** (the embedding `quant_config` default, was `c69d01ed8`) in
  `vocab_parallel_embedding.py`, which the prediction did not look at. Upstream added a
  keyword-only `quant_method` for model-specific preselection and reshaped the dispatch, so
  the fallback moved above it and is now gated on `quant_method is None` too.

**Wrong about where the cost would land.** The patches were the cheap part. The expensive
part was two *plugin* assumptions about how vLLM discards a tied model's `lm_head.*`, both
silent until 0.29 stopped doing it: see
[embeddings.md](embeddings.md) *A tied head vLLM used to throw away*. The lesson generalizes
past this bump -- a survey of what our patches touch does not cover what our plugin
*assumes*, and only the second kind fails silently.

**Also worth knowing for the next bump.** MoE router logits changed on non-SM90 CUDA cards:
`allow_cublas_router_gemm` used to require `allow_specialized_router_gemm` (SM90+, family-120
excluded), so sm120 computed the router GEMM through `F.linear` in bf16 and copied to fp32;
0.29 gates it on `(is_cuda() or is_rocm()) and no_bias`, so the same card now gets the cuBLAS
bf16xbf16->fp32 epilogue with no intermediate rounding. Deterministic, reproducible, and
*more* accurate -- but it moves top-k selection often enough to flip 3 of 63 argmax positions
on `qwen3.5-35B-A3B` at 2.00bpw.

---

## SINQ

**`device_map="auto"` silently produces an unusable model.** A checkpoint saved by SINQ's
own transformers integration, reloaded with `device_map="auto"`, goes through
transformers' native `SinqConfig` quantizer and comes back as
`sinq.sinqlinear_hf.SINQLinear` modules with `ready=False`. Nothing complains at load;
the first forward raises `AssertionError: model was not quantized` from
`sinqlinear_hf.py:433`. With `device_map="cuda:0"` the same file loads through SINQ's own
patched path (`sinq.sinqlinear.SINQLinear`, `ready=True`) and generates correctly.

*Repro*: quantize any model with `SinqConfig(nbits=4, group_size=128)`,
`save_pretrained`, then `from_pretrained(..., device_map="auto")` and generate.
Established 2026-09-01 on `Qwen/Qwen3-0.6B-Base`, transformers 5.15.0, `sinq` 0.2.0.

*Why it is worth their time*: `device_map="auto"` is the single most common way people
load a model, it is what accelerate's own documentation leads with, and the failure
surfaces far from its cause — an assertion about quantization, thrown from a matmul,
against a model that loaded without a word. Either the HF-integration path should mark
the layers ready or it should refuse at load. Costs them nothing to accept: no format
change, no dependency, no agreement with anything of ours.

**2D-tiled checkpoints do not survive a save/load round trip.** `tiling_mode="2D"`
quantizes and generates correctly in process, but after `save_pretrained` the reload
raises `RuntimeError: The size of tensor a (64) must match the size of tensor b (1024) at
non-singleton dimension 2` at the first forward. Isolated both ways on
`Qwen/Qwen3-0.6B-Base`, 4 bits, group 64, 2026-09-01: in-process generation is coherent,
the round trip is not — so the defect is in serialization, not in 2D itself. Same shape
as their `device_map` bug: it fails at a matmul rather than at load.

*Also worth telling them, since it is the reason anyone would care*: counted with its
metadata, 2D costs **0.24 bpw** over 1D at group 64, and on Qwen3-0.6B at 3 bits it is
strictly dominated by 1D — worse KLD at more bits. If that generalizes, the option's
documentation should say what it costs.

**It installs a `sitecustomize.py`, which costs every Python process in the environment
3.3 seconds.** The `sinq` package writes `sitecustomize.py` into site-packages, and CPython
imports that at **every interpreter startup**. Measured 2026-09-01:

| | interpreter start | modules preloaded |
|---|---|---|
| as shipped | **3.31 s** | 106 |
| `SINQ_AUTO_PATCH=0` | **0.02 s** | 2 |

165x, paid by every `python3 -c`, every CLI tool, every git hook in the venv — to import
sinq, gemlite and the whole `gemlite.triton_kernels` tree before the user's script runs.
It also prints `Found gemlite installation, fast SINQ-ference for 4-bit models` to
**stdout** on the way, unconditionally, from `sinqlinear.py:19` and `sinqlinear_hf.py:24`,
which corrupts the output of any program whose stdout is parsed.

**And installing it destroys a pre-existing `sitecustomize.py`, silently.** Reproduced
2026-09-01 in a throwaway venv holding a local customization that sets an env var and
prints a banner:

1. `pip install --no-deps sinq` → **"Successfully installed sinq-0.2.0"**, no warning, no
   mention of the file it just overwrote. The local customization stops taking effect;
   its env var reads `None`.
2. In an environment where torch is not importable, the replacement *fails* — so a
   working customization is swapped for `[SINQ] autopatch failed: No module named 'torch'`
   on **every interpreter start**, permanently.
3. `pip uninstall sinq` → **the file is deleted outright**. The original is not restored,
   because pip has no idea it ever existed. The user is left with no `sitecustomize.py`
   at all and nothing in either command's output that would explain why.

That is unrecoverable data loss from a routine install/uninstall cycle, and it is silent
at every step.

*Why it is worth their time*: four separable defects, each with an easy fix, and none of
them require the maintainers to agree with anything of ours. `sitecustomize` is reserved
for the environment's owner — only one file can hold the name, so **whichever package
installs it last silently wins**. A `.pth` under a package-specific name registering a
lazy hook, or simply documenting the env var, does the same job with no collision and no
destruction. The eager import should be lazy regardless. And the banner belongs on a
logger at debug level, or on stderr at worst.

*Carried locally*: the venv's `sitecustomize.py` is neutered with the default inverted —
`SINQ_AUTO_PATCH=1` restores it — with the original kept beside it as
`sitecustomize.py.sinq-orig`. Only *loading a saved SINQ checkpoint* through
`from_pretrained` needs the patch; quantizing in process is unaffected, which is verified.

*Second, smaller, and separable*: SINQ's weights are attached as plain tensor attributes
rather than parameters or buffers, so `named_parameters()` and `named_buffers()` see
nothing on a quantized model. Any parameter-walking size or memory tool reports only the
norms and embedding, with no error. Registering `W_q` as a buffer would fix it for every
such tool at once. Worth raising as a question rather than a defect — there may be a
reason, and asking is the honest first move.

## llama-cpp-python

**The CUDA pre-built wheels require AVX-512 and do not say so, and the requirement bites
even when the CPU backend is never used.** Established 2026-09-01 on
`llama_cpp_python-0.3.35-py3-none-manylinux_2_35_x86_64.whl` from
`https://abetlen.github.io/llama-cpp-python/whl/cu132`.

Importing works; constructing a model kills the process with **SIGILL** (exit 132, core
dumped). `dmesg` names the library:

    traps: python3[3636098] trap invalid opcode ip:7f7482422956 sp:7ffebdb93240
           error:0 in libggml-cpu.so.0[1e956,7f7482417000+13e000]

Disassembling the shipped `libggml-cpu.so.0.20.0` at that offset:

    1e956:  62 f1 7d 48 6f 05 ...   vmovdqa32 0x1337e0(%rip),%zmm0

An EVEX-encoded `vmovdqa32` into `%zmm0` — AVX-512F — and the symbol it sits in is
**`ggml_cpu_init`**. So it executes during *backend initialization*, before any inference,
and **regardless of `n_gpu_layers`**: a CUDA wheel is unusable on a non-AVX-512 host even
when the GPU is doing all the work. There is no runtime feature dispatch guarding it.

*Host*: Intel Core Ultra 9 285 (Arrow Lake) — `avx`, `avx2`, `avx_vnni`, `f16c`, `fma`
present; `avx512f`, `avx512bw`, `avx512vnni`, `amx_tile`, `bf16` absent. Intel removed
AVX-512 from consumer P+E designs, so this excludes every Alder/Raptor/Arrow Lake desktop
— not an exotic configuration.

*Why it is worth their time*: the documented requirements for the pre-built wheels list
exactly three things — CUDA version, NVIDIA compute capability, and Python version — and
no CPU requirement at all. A user meeting all three published requirements gets a core
dump with no diagnostic beyond `dmesg`. The fixes are cheap and independent: state the CPU
baseline in the wheel documentation; and/or build the bundled CPU backend at a baseline
matching the manylinux tag, which does not imply AVX-512. The runtime dispatch that
llama.cpp already has for CPU kernels apparently does not cover `ggml_cpu_init` itself.

*Carried locally*: pinned back to 0.3.34, whose wheel is CPU-only but runs. GGUF arms in
`qbench` therefore run on CPU; see [qbench.md](qbench.md).

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
limit. Fix is to fit `kld = c + s·rfn²`; with alpha pinned at 2.0, two noise levels solve
for both and no extra passes are needed: `c = (4·kld_lo − kld_hi)/3`. Independent of
whether per-tensor allocation is worth doing, which measurement says it largely is not.
Detail in [qbench.md](qbench.md).

*Still live at v1.4.3, and more so.* This item was briefly retired on 2026-08-25 on the
false belief that the `sc_*` tools had been deleted — they had not; they live in the
repository **root**, not `util/`, and a search for `util/sc_*.py` found nothing and was
mistaken for absence. What v1.4.3 actually deprecates is `util/measure.py` and
`util/optimize.py`, whose new one-line docstrings point at `doc/optimize.md` — the
documentation for the **`sc_*` pipeline itself**. So the tools this report is about are
not going away; they are the replacement. `sc_optimize.py` still fits
`kld(t, K) = S_t · rfn_t(K)^alpha` per tensor, so both halves of the finding — the floor
and the alpha it biases — apply unchanged.

---

### qbench: two defects found using it, 2026-09-10

Both surfaced while measuring whether v1.4.9 cost quality (it did not --
[qbench.md](qbench.md)). Neither is EXL3-specific and both are cheap to fix.

**The results cache is not keyed on the engine's version, so it replays across an
upgrade.** `model_key` in `eval/qbench.py` hashes engine, source, options, source stamp
and noise. Nothing identifies *which build of exllamav3* produced a result, so a score
computed under one version is served to a later run under another, silently and with no
warning. This produced a wrong answer here: a run intended to measure v1.4.9 reported
figures computed nine days earlier under the previous version, and the giveaway was not
in the output at all -- only `manifest.json`'s per-arm `seen` timestamp shows it.
`manifest.json` already records `engine`, so the fix is small; adding the engine version
to the entry and to the key would make the stale case a miss instead of a lie.

**The noise floor crashes on an `exllamav3` reference.** With
`engine: exllamav3` on the reference arm, the floor pass dies in the per-layer noise
injection:

```
RuntimeError: Expected a 'cpu' device type for generator but found 'cuda'
```

A torch generator is being created on CPU and used on a CUDA tensor (or the reverse),
which newer torch refuses. It looks like a path only ever exercised with a
`transformers` reference. Worked around with `noise_floor: false`, which is the right
setting for a quantized reference anyway -- the floor would measure that quant's
self-noise rather than fp16's -- so this only bites projects that use a non-transformers
reference, which is a small set but includes any comparison that cannot afford the bf16
weights.

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
