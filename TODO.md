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
take EXL3 from losing to competing formats on most checkpoints, once total bytes are
counted rather than only the tensors that participate in the bit-rate target, to being
as competitive as that bit-rate target implies. [docs/qbench.md](docs/qbench.md) has the measurement that establishes
the gap is real on the served path.

## `bench-suite` — A TP tier for the bump gate

`bench/` gates vLLM and exllamav3 bumps on token ids, per-position logprobs and
resident weight bytes, against baselines committed for the current pin. Two tiers
are populated: `fast` (~15 min) covers uniform K=3, mixed-in-layer bit widths,
`mcg`, tied and untied, both execution modes, and the Transformers backend on a
text-only model; `full` adds MoE, `mul1` with the gemma4-style tie, and the
multimodal Transformers backend.

**This is the next `vast` trip, and everything else TP rides along with it**
(sequenced 2026-08-26). Nothing TP-shaped outranks the gate tier, because until the
per-degree baselines exist every other TP result is unrepeatable. So the tier goes
first and the rest slot into the same rental rather than earning trips of their own:

| rides along | what it needs there | slug |
|---|---|---|
| blockq embedding at TP>1 | written, never run multi-GPU. All three tensors slice on dim 0, so `tp.ROLE_VOCAB` is a row slice with none of the trellis path's 128-row Hadamard rule | `quantized-embeddings` |
| tied+blockq at TP>1 | `EXL3BlockQTiedEmbeddingMethod` (2026-08-26) carries two parameter sets on one module; both shard on dim 0, but the *combination* is unproven | `quantized-embeddings` |
| MoE+TP remainder | TP=3/5/6/7 (an 8-card box covers these by using fewer cards); a second checkpoint at TP=8, which needs a download arranged **before** the rental starts | `moe-tp` |
| Transformers backend | no MoE and no TP has ever gone through it | `transformers-backend` |
| comparator arm | Qwen3.8-27B unquantized is ~27 GiB and cannot run on the dev card at all | `comparator` |

**Two things to arrange before the meter starts**, both of which otherwise waste
rental time: the second TP=8 checkpoint has to be downloaded ahead (nothing on hand
clears preflight), and `tools/host_survey.py` should run on contact, since GPU
architecture, driver, card count, VRAM and uncorrected ECC are what move tokens and
are all checkable before any real work.

**Explicitly not a `vast` item:** the Laguna TP=4 `exl3_mgemm` performance bug. It is
narrowed to two exact kernel instantiations with both autotuners ruled out, so closing
it needs someone reading kernel source against per-tensor quantization parameters —
more GPU time buys nothing. Listed under `moe-tp` and stays there.

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

**No longer hypothetical.** On 2026-08-20 the local cache lost
`Muse-Glimmer-30B-exl3@2.00bpw`, replaced by `2.50bpw` for unrelated reasons, while
`full` still pins the 2.00 baseline. Nothing broke — `bench` fetches by revision and
the branch is still published — but the next run would have discovered a multi-GiB
download midstream. `deps` is what turns that into a line of output beforehand, and
the failure it prevents is not confined to rental boxes.

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

## `turboquant-sliding-window` — TurboQuant KV cache for sliding-window models

*Written up in [docs/turboquant-kv.md](docs/turboquant-kv.md), Part 1. What follows is
the diagnostic history, kept because two of the three walls were misdiagnosed on the
way.*

**Status 2026-08-29: Laguna serves, patched.** The three fixes are in
[patches/vllm-tq-01-sliding-window-kv-pages.patch](patches/vllm-tq-01-sliding-window-kv-pages.patch)
and the measurement is at the end of this item. Still open: the quality question
(`qbench` on tq4), gemma-4 and Muse-Glimmer (neither fits the local card), and
reporting any of it upstream. The history below is kept because it is what the
diagnosis cost, and because two of the three walls were misdiagnosed on the way.

`TurboQuantAttentionBackend` never overrides `supports_sliding_window`, so it takes
the base class's `return False` and is rejected for any model with a sliding window.
That is **gemma-4, Laguna and Muse-Glimmer** — one reason, three families.
Muse-Glimmer is 39 sliding / 13 full at a 2048 window with `head_dim` 128, so like
Laguna it carries no head-dim triton pin and is blocked on the sliding window alone.

**And the rejection is a fix, not merely an obstacle.** Reported 2026-08-25: Muse
*used* to be accepted with a turboquant cache dtype and is not any more. It could
never have been correct — the model has been sliding-window throughout — so
whatever ran before was serving wrong attention silently. 0.28 refusing it is the
gate upstream should always have had, which is worth remembering when arguing for
the sliding-window support: the ask is to make it work, not to relax the check.

**Bigger than the FA miss it was mistaken for.** Closing the head-dim-512 gap would
have bought gemma-4 a flash-attention path and nothing else (retired, see
[docs/kernels.md](docs/kernels.md)); this buys gemma-4 *and* Laguna a quantized KV
cache, which on a 16 GiB card decides whether the context is usable.

**The economics are better than the layer count suggests**, which is what makes it
worth pursuing rather than filing as impossible. gemma-4 is mostly sliding layers
(12B: 40 sliding / 8 full; 26B-A4B: 25 / 5), so compressing only the full-attention
layers sounds like giving up most of the prize. It is not: a sliding layer's cache is
capped at `sliding_window` (1024) while a full-attention layer grows with context, so
the five-or-eight full layers *dominate* KV bytes exactly where pressure is real —
~62% of KV at 8k on the 26B, ~86% at 32k, ~96% at 128k. Compressing only those
reaches ~65% total KV reduction at 32k against ~75% for compressing everything.

**There is precedent for the approach, and it is bypass rather than support.** The
external `turboquant-vllm` project ([alberto.codes,
2026-03-31](https://alberto.codes/blog/2026-03-31-from-one-model-to-seven-making-turboquant-model-portable))
reached Gemma-2/Gemma-3 by *declining to compress* sliding layers — "compressing a
sliding window layer's cache breaks the eviction semantics" — keeping layer indices
aligned with `None` padding. Not teaching the cache sliding window; detecting sliding
layers and leaving them alone. **vLLM already has that mechanism**:
`--kv-cache-dtype-skip-layers` takes the literal keyword `sliding_window`
(`layers/attention/attention.py:306`) and sets those layers to `kv_cache_dtype="auto"`.

**Measured 2026-08-24 on gemma-4-12B@3.00bpw_mul1, and it does not currently work:**

- **Unforced, the failure looks FA-related and is not.** `TRITON_ATTN is not valid
  ... ['kv_cache_dtype not supported']` — gemma-4 is pinned to triton by head dim
  512, triton rejects the turboquant dtype, and nothing mentions sliding window. This
  is the message that produced the original wrong diagnosis.
- **`--kv-cache-dtype-skip-layers sliding_window` crashes**, and it is one line:
  turboquant merges its own boundary skips with `sorted(existing | set(boundary),
  key=int)` (`engine/arg_utils.py:1985`), so the documented keyword raises
  `ValueError: invalid literal for int() with base 10: 'sliding_window'`. The exact
  invocation implementing the blog's approach cannot be typed.
- **Skipping the same layers by numeric index changes nothing**, because backend
  selection is global: vLLM picks one backend for the model and that backend must
  accept the configuration. Per-layer dtype skipping cannot route around it.
- **Forced to turboquant, three blockers, not one:** `['kv_cache_dtype not
  supported', 'partial multimodal token full attention not supported', 'sliding
  window not supported']`. The third is the known one. The second is
  multimodal-specific and means gemma-4 needs more than sliding-window support. **The
  first is unexplained** — `supports_kv_cache_dtype` returns true for anything
  `turboquant_*` and the full `turboquant_4bit_nc` reaches the backend. Suspicion, not
  a finding: turboquant auto-adds boundary layers to the skip list, those become
  `"auto"`, and global selection then validates turboquant against a native-dtype
  group. **Settle this before reporting anything upstream.**

So the `key=int` collision is a real, small, reportable bug on the path that matters,
but fixing it alone would not get gemma-4 to turboquant — the multimodal blocker sits
behind it.

**Laguna is the better playground, and it gets furthest.** Text-only (no multimodal
blocker), `head_dim` 128 (no triton pin, so no head-dim confound), and 10 full / 30
sliding layers at a 512-token window — half gemma's. That layout makes the bypass
*more* attractive, not less: the ten full-attention layers are 73% of KV bytes at 4k,
84% at 8k, **95.5% at 32k** and 98.8% at 128k, so compressing only them captures ~95%
of the theoretical maximum.

Measured 2026-08-24 on `Laguna-XS-2.1-exl3@3.00bpw`:

- `--kv-cache-dtype turboquant_4bit_nc` alone: `No valid attention backend found for
  cuda` — turboquant is the only backend accepting the dtype and it rejects the
  sliding window, so nothing is left.
- **Skipping the 30 sliding layers by index passes backend selection** and fails much
  deeper, in KV cache group construction:
  `get_kv_cache_groups -> unify_kv_cache_spec_page_size -> assert
  self.page_size_padded >= self.unpadded_page_size_bytes` (`kv_cache_utils.py:1118`,
  `kv_cache_interface.py:207`).

**That first wall is a one-line staleness bug, confirmed by patching it.** The unifier
scales `block_size` by the ratio via `replace(layer_spec, block_size=new_block_size)`
and leaves `page_size_padded` untouched; turboquant sets that field for its packed
`slot_size_aligned` layout, so the recomputed `unpadded_page_size_bytes` overtakes the
stale padded value. Scaling both cleared it.

**Behind it is a second wall that is not small.** With the assert gone, the same run
fails on a full-attention (turboquant) layer:

    NotImplementedError: Layer model.layers.4.self_attn.attn: page size is not
    divisible by the maximum page size and cannot be padded.

Padding is gated on `indexes_kv_by_block_stride`, which is *derived* from the
backend's `get_kv_cache_stride_order()` (`gpu_model_runner.py:7833`) — a property of
the memory layout, not a flag to set. TurboQuant packs K+V into a single interleaved
slot per head per position, so its page size is neither divisible into a native
layer's nor paddable to match it.

**So the accurate statement is narrower than "TurboQuant cannot do sliding window" and
wider than "one function":** TurboQuant's packed KV page layout cannot currently
coexist with native-dtype layers in one cache. That is precisely the bookkeeping the
external project wrote by hand *outside* vLLM, and it is the thing any bypass approach
has to solve — the sliding-window rejection is only the first gate.

**And upstream has already restructured it, after our pin.** vLLM's
`[N/N] KV-Cache Layout Refactor` series has parts 1-3 in v0.27.0 and parts **4 and 5
landing after it**:

- `61874f9842` [4/N] Promote local KV cache specs via a class-changing replace helper
  (#51612) — rewrites `kv_cache_utils.py`, the file holding the unifier.
- `57bd0ed441` [5/N] Backend-published KV packing via `customize_spec` (#51704) —
  touches `turboquant_attn.py` directly, and **deletes `TQFullAttentionSpec`**.

TurboQuant stops being a special spec class carrying an opaque `page_size_padded` and
becomes an ordinary `FullAttentionSpec` publishing `state_content_bytes =
slot_size_aligned` through a backend hook. Page size then computes as
`num_heads * storage_block_size * state_content_size_bytes` — a **per-cell** quantity,
so it scales linearly with `block_size` where the old padded value did not.

**Re-run on v0.28.0, 2026-08-25, and the prediction was half right.** TurboQuant does
stop setting `page_size_padded` — it publishes `state_content_bytes` only. But the
walls did not dissolve, and 0.28 turns out to have built *first-class machinery for
this exact case* with a gap in it:

- **Stages 1 and 2 fail exactly as on 0.27.0** — bare dtype gives no valid backend, the
  `sliding_window` keyword still raises `ValueError: invalid literal for int()`.
- **Stage 3 progresses further, then hits the same assert** (`page_size_padded >=
  unpadded_page_size_bytes`), now reached through cudagraph memory profiling rather
  than direct KV init. The staleness bug moved rather than vanishing: in 0.28 it is the
  *sliding* spec that carries a padded page (`page_size_padded=shared_page`) while the
  unifier's block-scaling branch still leaves it stale. Same one-line fix, different
  spec.
- **With that patched, the divisibility wall returns**, and instrumenting the unifier
  says why — there are **three** page-size classes, not two:

```
layer 0: FullAttentionSpec   block=32 page=131072 padded=None   content=None quant=0
layer 1: SlidingWindowSpec   block=16 page=65536  padded=65536  content=None quant=0
layer 4: FullAttentionSpec   block=32 page=34304  padded=None   content=134  quant=7
max_page_size=131072  distinct=[34304, 65536, 131072]
```

Layer 0 is *native full attention* because TurboQuant auto-adds its own boundary skip
layers (`get_boundary_skip_layers`, first/last N) on top of the operator's list. So the
pool holds native-full, native-sliding and turboquant-full, and 34304 does not divide
131072.

**`--kv-cache-dtype-skip-layers` is a supported configuration in 0.28**, not a hack:
`CacheConfig.skip_page_size_padded` is documented as "the page size of layers skipped
from KV cache quantization ... so unquantized skip layers pad up to the quantized
primary's page", and `Platform._align_..._block_size` bumps `block_size` so the primary
page covers the padded one. It handles **one** padded class. Upstream marked the gap
itself, twice, in that same function:

```python
# To add the first/last-N sibling:
#   padded_pages.append(per_token_page_bytes(<sibling_dtype>, "auto"))
# To add the first/last-N sibling:
#   cache_config.sibling_page_size_padded = shared_page
```

The first/last-N sibling *is* TurboQuant's boundary protection. So what blocks Laguna
is a case upstream has already identified and left unimplemented, plus the block-scaling
staleness bug — two narrow, reportable things rather than a missing capability.

**The boundary-protection question is filed separately** as
`turboquant-boundary-tax` below: the patch here keeps that protection working, so
removing it is an independent optimization across all TurboQuant models rather than
anything this item depends on.

**Answered 2026-08-29, and Laguna serves.** The answer to the bounded question was
yes, but the two known fixes were not the load-bearing one. See
[patches/vllm-tq-01-sliding-window-kv-pages.patch](patches/vllm-tq-01-sliding-window-kv-pages.patch)
for all three.

- **The primary page was priced by the wrong backend.**
  `_align_heterogeneous_kv_block_size` prices the quantized primary through
  `backend_cls`, which `_find_non_ssm_backend` defines as the backend of the *first*
  attention layer — with skip layers, an unquantized one on FLASH_ATTN, whose
  `customize_spec` is a no-op for TurboQuant's packing. So the primary was priced as
  dense uint8 (2·hd bytes/head, 2048/token) instead of packed (hd+6, 1072/token). Since
  `2·hd/(hd+6)` is never an integer — 1.91 at hd 128, 1.95 at hd 256 — the shared page
  could never be a multiple of the real primary page, and no amount of sibling work
  would have fixed that. The sibling function `_align_hybrid_block_size` already
  special-cases TurboQuant for this exact reason and says so in its own comment.
  Resolving the backend that serves the primary dtype moves block_size 32 → 64 and the
  shared page 65536 → 68608, which is the packed page exactly.
- **The sibling then does what upstream's comments say**, with two gates: only when the
  native page is not an integer multiple of the primary's (nvfp4 is exactly 2× and
  keeps its block-scaling path), and only when some layer is sliding-window or the model
  is hybrid. The second gate matters — an all-full-attention model never reaches
  `unify` at all (`UniformTypeKVCacheSpecs` takes it first) and packs differing page
  sizes more tightly than padding does. Ungated, the sibling cost **2.0% of KV tokens on
  dense TurboQuant**, which is a real regression on the path that already worked.
- **The `page_size_padded` staleness one-liner is real but no longer on the path.**
  Applied alone against the baseline it clears the assert and exposes the divisibility
  wall behind it, reproducing the history recorded above; with the other two fixes the
  pages reconcile and `unify` returns before the scaling branch.

Measured on `Laguna-XS-2.1-exl3@3.00bpw`, `turboquant_4bit_nc`, 30 sliding layers
skipped by index, RTX 5070 Ti / vLLM 0.28.0. All three page classes reconcile to 68608
— turboquant layers exact at block 64, native full and sliding padded from 65536 at
block 16 (4.5% waste) — and the model generates. **Boundary protection is intact**: the
first/last-N layers keep native KV, so nothing was traded for this. Dense TurboQuant
(MiniCPM5-1B) is unchanged at 909,888 vs 909,920 KV tokens, and both models still serve
with an unquantized cache.

**Not yet shown on the other two families.** gemma-4 and Muse-Glimmer do not fit the
16 GiB card (Muse OOMs during load; its native path additionally fails earlier on an
unrelated `vision_adapter.c_fc.mul1` weight-loading error, and it is served through
`--model-impl transformers` here). Both need the vast box. gemma-4 also still has the
multimodal blocker in front of it, which this does not touch.

The `key=int` crash on the literal `sliding_window` keyword (`arg_utils.py:2022`) is
untouched and still stands — the runs above skip layers by numeric index.

**There is an upstream tracking issue for the gemma-4 half of this**, found 2026-08-29:
[vllm-project/vllm#41403](https://github.com/vllm-project/vllm/issues/41403), "TurboQuant
+ Gemma 4 multimodal: 5-gate blocker stack" (open). Same destination, four gates in
common, and it is worth reading before writing anything upstream.

What it gives us:

- **The multimodal blocker has a workaround.** `--hf-overrides
  '{"text_config":{"use_bidirectional_attention":null}}'` clears
  `partial multimodal token full attention not supported` at the cost of vision
  quality. That is the blocker filed here as sitting behind everything else, and it
  turns out to be steppable for a text-only measurement.
- Two pieces of trivia that will cost an afternoon otherwise: TurboQuant's triton
  kernels need `ninja` on the path, and an installed external `turboquant-vllm` plugin
  collides with the in-tree API (`TQ4FullAttentionSpec.__init__() got an unexpected
  keyword argument 'tq_slot_size'`).
- Their Gate 2 is our boundary-skip question, and their workaround is exactly the one
  considered here: monkeypatch `get_boundary_skip_layers` to return `[]`. So the
  approach has independent users, and our patch is the alternative that keeps the
  protection rather than trading it away.

Where we are ahead: their Gate 5 stops at `NotImplementedError ... cannot unify by
adjusting block_size` and attributes it to gemma's heterogeneous head_dim. That is a
real second cause, but it is not the one that stops Laguna — the mispriced primary page
is, and it is invisible from the error message. Their Gate 2 diagnosis ("needs per-layer
attention backend routing") is also stale for 0.28: per-layer routing already happens
here, FLASH_ATTN for the skip layers and TURBOQUANT for the rest, which is precisely how
the aligner ends up asking the wrong backend for the primary's packing.

**And gemma is harder than Laguna in a way this item had not recorded.** Its full and
sliding layers do not share a KV geometry: `head_dim` 256 with 8 KV heads on the sliding
layers, `global_head_dim` **512** with `num_global_key_value_heads` **1** on the global
ones (plus `attention_k_eq_v: true`). So natively the two layer types are 8192 and 2048
bytes/token — the heterogeneity the issue names, and a *fourth* page class that the fix
above does not address: `padded_pages` computes one native per-token page from
`model_config.get_num_kv_heads()`, which cannot be right for both. The Laguna fix is
necessary but not sufficient for gemma.

It also re-prices gemma's prize, favourably: the global layers are only 2048 bytes/token
against 8192 for sliding, so the fixed sliding cost is ~335 MiB total (capped at the
1024 window) while the global layers are what grows — 2.1 GiB at 128k, which tq4 takes
to ~540 MiB. That is the same number the issue arrives at independently for the 31B.

Worth pairing with the quality question before investing: tq4 on a trellis-quantized
model has field experience and a capability-benchmark lower bound behind it, but no
`qbench` numbers. That is the measurement that decides whether this is *practical*
rather than merely running.

**This item now carries the gemma-4 dependency, and its odds have improved enough to
change downstream priorities** (2026-08-26). Two things moved. First, the blocker was
misidentified: it was filed as a likely-insurmountable flash-attention gap (head dim
512, since retired -- see [docs/kernels.md](docs/kernels.md)), and it is actually this — a KV-cache layout problem whose walls have so
far all reduced to one-line staleness bugs, one of them already cleared by patching.
Second, the scope is three families and not one: gemma-4, Laguna and Muse-Glimmer are
all sliding-window, which is **every real-world candidate except Qwen**. A fix is
therefore load-bearing for most of the candidate pile rather than for one demoted
family, and it is now better read as *more likely to be resolved than not*. (Resolved
for Laguna on 2026-08-29; see the measurement above.)

What that unblocks: gemma-4 goes back to being a genuine serving candidate, so the
shared embed+head tensor deferred under `quantized-embeddings` has a **real
constituency rather than a hypothetical one**. The severe priority demotion that both
gemma and every tied-model optimization inherited was downstream of the misdiagnosis,
and should be unwound with it. The divisibility wall — the remaining unknown when this
was written — turned out to be three narrow bugs and is cleared. What is left is the
quality question below, and whether gemma-4's multimodal blocker yields.

→ [docs/kernels.md](docs/kernels.md)

## `turboquant-boundary-tax` — What TurboQuant's first/last-N protection costs, and when it pays

Split out of `turboquant-sliding-window` on 2026-08-29: that item is a correctness fix
and is answered, this is an optimization across every TurboQuant model including dense
ones, and nothing depends on it — the sibling patch keeps boundary protection working.

**Measurements are written up in
[docs/turboquant-kv.md](docs/turboquant-kv.md)** (Part 2), with the harness in
[tools/gsm8k_kv.py](tools/gsm8k_kv.py) and per-item results in
[docs/data/turboquant-kv/](docs/data/turboquant-kv/). In short, on Qwen3-4B at n=1319:

- The docstring's claim **reproduces exactly** — k3v4_nc loses 29.80 points against its
  "~30 points on Qwen3-4B". 4-bit is not exempt either: −6.52, p=5e-08.
- **Layer 0 is the whole effect.** Protecting `{34,35}` is indistinguishable from
  protecting nothing (p=0.79 / 0.92); protecting `{0}` alone is indistinguishable from
  full stock protection (p=0.17) at 18% fewer bytes.
- **Both aggressive presets are dominated as they ship.** `3bit_nc` is beaten by
  `4bit_nc` protecting only layer 0 by five points *at fewer bytes*.
- **None of the better configurations can be expressed** — the flag only ever adds to
  the automatic skip list.
- Laguna shows none of it, and that anomaly is unexplained.

Open:

- **Long context.** Everything measured is ~700-1300 token prompts. The reason to
  compress KV is long context, and both KV damage and any first-layer effect plausibly
  grow with length. This is the measurement that should decide any default change, and
  it does not exist yet.
- **A second dense model**, to know whether "layer 0 only" is a property of Qwen3-4B or
  of transformers. The attention-sink explanation predicts it generalises.
- **Why Laguna is flat** — one full-attention layer at stake versus four, or 30 of 40
  layers native regardless? Cheaply separable by compressing Laguna's sliding layers too
  once that path exists, or by testing a dense model of Laguna's depth.
- **The upstream lever is drafted** as
  [patches/vllm-tq-02-boundary-lever.patch](patches/vllm-tq-02-boundary-lever.patch): the
  `key=int` fix plus `boundary:N` in `--kv-cache-dtype-skip-layers`, default unchanged
  at 2. Verified end to end. What is left is filing it — as a reachability gap rather
  than a defaults change, with the Qwen3-4B frontier as the motivation.
- **Tell [vllm#41403](https://github.com/vllm-project/vllm/issues/41403)** that
  monkeypatching `get_boundary_skip_layers` to `[]` is not free — it presents that as a
  costless gemma workaround, and on a dense model it costs 6.5 points at 4 bits.

→ [docs/turboquant-kv.md](docs/turboquant-kv.md)

## `repair-tool` — Repair tool for existing EXL3 checkpoints

Every existing EXL3 checkpoint carries two packaging choices that cost far more in a
GPU-resident server than they do upstream, enough to more than erase its efficiency
advantage against other formats on total bytes: a separate output head is emitted for
tied models (redundant once the embedding is quantized, and not small), and the
embedding is left at full resolution. Both are rational for exllamav3, which keeps the
embedding in system RAM and counts it against neither VRAM nor the published size --
see [docs/embeddings.md](docs/embeddings.md). A post-processing tool
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
   (default the config from `get_current_vllm_config()`), verified not to disturb
   configs that do not quantize embeddings.
   **Worth offering upstream**: `vllm-gguf-plugin` is blocked by exactly the same
   thing, so it is an ecosystem fix and not only ours.

   **But not as written — it breaks speculative decoding** (found 2026-08-20). A
   drafter is built under the *target's* `quant_config`, so the patch hands the
   drafter's embedding an EXL3 method and it is asked for a `bq_q` it never had.
   Nothing about that is EXL3-specific; filed as-is it breaks any quantized target
   with a differently quantized or unquantized drafter. Two candidate fixes, and the
   choice decides what gets filed: condition the ambient default on the module
   belonging to the model the config describes (which
   `VocabParallelEmbedding.__init__` cannot know), or fix the drafter's config
   instead so it stops misdescribing what is being built — a cleaner contribution
   than an 86-file workaround, and it makes this patch safe as a side effect.
   The same ad-hoc plumbing fails in the opposite direction too
   (`DFlashQwen3Model.fc`, gemma-4's `vision_adapter.c_fc`): any fix worth filing
   should be judged against both. →
   [docs/format-and-loading.md](docs/format-and-loading.md) "Ambient `quant_config`"

2. **Tensor parallelism is written but unproven.** All three stored tensors slice on
   dim 0, so `tp.ROLE_VOCAB` is a row slice with none of the trellis path's 128-row
   Hadamard alignment rule — which is why it is a handful of lines. It has never run
   on more than one GPU. Needs the `vast` box, alongside `moe-tp` and the TP tier of
   `bench-suite`.
3. **Only 4 bits is packed.** That is deliberate — one depth covers every model
   measured, and nibbles keep both ends byte-aligned — but 3 bits is usable at ~3.5
   bpw and would want the packing if a checkpoint ever calls for it.

**A tied model with a block-quantized embedding crashes** — **FIXED 2026-08-26**
(found 2026-08-19, observed 2026-08-25). The two predicates in
`quantization/config.py` treated the cases as mutually exclusive —
`embedding_is_blockq()` said so in as many words — but they answer questions
about *different modules*: one asks whether a tied model's `lm_head.*` is being
renamed onto the embedding, the other whether the embedding has block-quantized
tensors of its own. A repaired tied checkpoint makes both true and both
load-bearing.

Two silent failures came out of that, and both are gone:

- `EXL3TiedLMHeadMethod` read a trellis off the embedding module, which now held
  `bq_*`. Died late, at logits time.
- The blockq branch returned *before* `self.embed_prefix = prefix`, so the prefix
  stayed at its `"model.embed_tokens"` default while the rename still fired —
  routing 755 MiB of trellis to a path a nested model does not have, dropping it
  without complaint, and serving garbage.

The fix is `EXL3BlockQTiedEmbeddingMethod`: both parameter sets on the one
module — which is the existing design, since vLLM skips a tied model's
`lm_head.*` and those weights are already renamed onto the embedding for the head
to borrow back — with the lookup on `bq_*` and the logits on the trellis.
`embed_prefix` is now recorded before any branch. **No vLLM patch and no metadata
change**: `tie_word_embeddings` stays `true`, which it is.

`apply()` is overridden explicitly rather than inherited, and that was not
cosmetic. The MRO puts the blockq method first (it must, for the gather) and its
`apply` is a stub raising "no matmul path" — right for an untied embedding, wrong
here. The gemma4-style two-module shape never notices because its head reaches
the trellis through its own method; **the Qwen3-style one-module shape would have
raised on the first token.** Caught by an MRO assertion in
`tests/test_tied_blockq_routing.py`, not by running a model.

Verified on a repaired tied `gemma-4-12B-it-exl3@3.00bpw_mul1`: loads clean,
answers correctly, holds ~0.53 GiB more weight than the unrepaired baseline (KV
cache 19,176 → 17,545 tokens) which is `bq_*` loading beside the trellis, and 150
of 157 prompt logprobs differ from that baseline (mean |Δ| 0.122) — so the gather
is executing, not merely allocated. Full suite 118 pass.

`tools/quantize_embedding.py` now *supports* tied checkpoints rather than
producing broken ones, and says at creation time that the output needs a plugin
with this method — an older one loads it clean and serves garbage, which nothing
downstream can detect.

**Still open on this path**: tensor parallelism is unproven (both sets shard on
dim 0, but the combination has never run on more than one GPU), and the
Qwen3-style one-module tied shape is covered only by the MRO test, not
end-to-end. Neither blocks the gemma-4 case.

**The shared tied-model tensor's kernel blocker is gone** (2026-08-26). One tensor
serving both roles needed a scalar-integer GEMM for the head, which exists nowhere.
fp8-e4m3 clears that — `torch._scaled_mm` is a primitive, not a kernel project — and it
is good enough: **+0.000669** against native, 0.38x the noise floor, better than the
7-bit per-row point that was the recorded tied operating point. Per-channel scaling is
required; per-tensor is 1.6x worse for nothing. So "7 or 8 bit shared per-row" is
**superseded on encoding** — fp8 is the shape to build if it is built.

**It stays deferred, and the ordering that deferred it has now resolved.** The crash
above landed first, on correctness grounds rather than because any model demanded it,
and that changed the baseline this has to be measured against. The blockq split is a
real tied-model baseline as of 2026-08-26: **1.234 GiB at +0.000297**, serving gemma-4
from `bq_*` for the lookup and the trellis for the logits.

So fp8's margin is the marginal one after all: **0.296 GiB for 2.25x the divergence**,
not the 1.641 GiB it looked like while the split was unbuilt. That margin is still
worth something — 0.296 GiB is real in a VRAM-bound appliance, and 2.25x of 0.38x the
noise floor is still under the floor — but it is now an optimization on top of a
working path rather than a route to one, and it needs an fp8 head path, an fp8 gather
and repair-tool emission to collect it.

The constituency is no longer hypothetical, which is what keeps this open at all: the
demotion gemma and every tied-model optimization inherited came from the
head-dim-512 misdiagnosis, corrected under `turboquant-sliding-window`.
**Revisit if a tied model ever needs that last 5%**, or if fp8 emission turns out to be
nearly free alongside other repair-tool work.

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

- **Upstreaming — now half a patch, and the feared cost did not materialize.**
  Settled at the v0.28.0 bump (2026-08-25). Upstream closed most of it: the
  postprocess patch is retired outright, and the softcap patch reduces to its
  `output_multiplier` half plus the fold-into-cap identity, since `causal.py`
  passes `soft_cap` but still reads only `logit_scale` and still applies its
  scale *after* the cap. That half is what remains worth offering.

  **The seam survives**, which was the open worry: upstream does not stop using
  `VocabParallelEmbedding`, it *rebases the model's embedding class onto it*
  (`type(cls.__name__, (cls, _VocabParallelEmbeddingBase), {})`), and
  `replace_embedding_class` passes `quant_config` into
  `VocabParallelEmbedding.__init__` — so a quantized embedding still attaches on
  this path. Confirmed by the gate, not by reading: the two Transformers-backend
  entries capture at 0.000e+00 across the bump.

  `vllm-replicated-linear-weight-loader-v2.patch` remains untouched upstream;
  check the `RowvLLMParameter`-narrowing edge noted in the doc before offering
  it.
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
  since a baseline against an image that later changes is worthless. The fixture
  buys a second thing once it exists: exllamav3 quantizes vision towers with no
  calibration data at all, and the same image-conditioned logprob divergence
  measures a bf16 tower against a quantized one on an otherwise identical
  checkpoint -- no vision benchmark required. See
  [docs/media-encoders.md](docs/media-encoders.md).
- **Audio is broken, probably not ours.** gemma-4-12B insists every clip is
  chirping birds. It is a unified model with no audio encoder — sound goes into
  the same token space as text — so there is no EXL3-quantized audio component to
  get wrong, which makes a pipeline fault upstream of us the likely cause. Cheap
  way to settle it rather than assume: the same clip through native exllamav3,
  the instrument that settled Muse's text path. The other instrument is
  gemma-4 E2B/E4B, the only model on hand with a genuinely *separate* audio
  encoder and so the only one where a quantized audio component could be blamed
  at all — blocked on `gemma4-e2b`.
- **`--language-model-only` is doing more work than it looks.** It is the flag
  reached for whenever headroom runs short, and the reason is now priced: the
  encoder is 0.79-3.64 GiB across every checkpoint surveyed and up to **21% of
  the package** (Qwen3-VL-8B @3.0bpw), worst on exactly the small, low-bpw
  checkpoints picked for small cards. It should not have to be a capability trade —
  evicting the encoder to host memory costs one PCIe pass per image and nothing at
  all for text — except that vLLM cannot offload an encoder at all
  (`upstream-queue`). → [docs/media-encoders.md](docs/media-encoders.md),
  [docs/upstream.md](docs/upstream.md)
- **Muse-Glimmer is usable in practice now, through the Transformers backend.**
  Reported 2026-08-25: fp8 KV works, and vLLM 0.28 ships a reasoning parser
  (`muse_glimmer_reasoning_parser.py`) plus tool-call parsing, so the serving path
  is complete rather than merely functional. Two things it still cannot do:
  turboquant (sliding window -- see `turboquant-sliding-window`, where it is the
  third family), and the **native** implementation, which fails on the quantized
  vision adapter even with `--language-model-only`.
- **Muse-Glimmer remains blocked on the native path**, for two unrelated reasons
  already characterized: native vLLM cannot serve its quantized vision adapter
  (`vision_adapter.c_fc` is a plain `nn.Linear`, unreachable by any quantization
  plugin — the same class of gap as `DFlashQwen3Model.fc`, see
  `quantized-embeddings` item 1), and the Transformers backend route is degraded on vLLM ≥ main by the
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
question, and is now **answered**: a budget-neutral sweep on phi-4-mini puts the head
optimum at 5-6 bits, with 7 costing +34% KLD and 8 costing +87% — see
[docs/qbench.md](docs/qbench.md), "Head bitrate: 6 is defensible". Since the shared
tensor's depth *is* the head's, that prices the tied-model plan directly: 5-6, not
higher.

**Not urgent: `repair-tool`'s post-processing covers this today.** All this item adds
is that checkpoints built here would be right from the start instead of needing a pass.
Strictly as a quantization problem the embedding is the least interesting one on the
map — a lookup with a distribution-independent optimum that measurement has already
pinned at a flat 4 bits across every model tested. There is no allocation question here
worth solving; depth is a constant.

**The format question is no longer blocked on upstream.** It previously waited on the
bar for proposing a storage format to exllamav3, which is higher than "serves this
plugin's targets", plus a wrinkle: the recipe is a flat `{tensor: bits}` map, so an
embedding needing a different *encoding* rather than a different depth cannot be
expressed in it. Now that checkpoints are built here rather than consumed, that bar is
just "does it work for us", and the wrinkle is nobody else's design decision to wait on.

**And the serving path is no longer a blocker either — for untied models it is done.**
`blockq` is specified ([docs/blockq-format.md](docs/blockq-format.md)), emitted by
`tools/quantize_embedding.py`, served by `EXL3BlockQEmbeddingMethod`, and gated by three
`bench/` entries (eager, CUDA graphs, and throughput) that derive a fixture on every run
and held at exactly 0.0 across both the v0.28.0 and exllamav3 v1.4.3 bumps. So for the
untied case nothing is missing but doing it at conversion time instead of as a repair
pass, and `repair-tool` already covers the interim.

**The solver is not part of this item, and as of 2026-08-25 is not part of any.**
Embedding depth is fixed at 4, the head measures out at 5-6, and body tensors cannot be
allocated independently at all. Two scalars is a lookup table, not a search space — see
[docs/qbench.md](docs/qbench.md).

**What is still blocked is the tied half, and on a different thing than this item used
to say.** It wanted "one shared block-scaled integer tensor serving both roles", and
`blockq` deliberately has no matmul path — decoding to dense fp16 to multiply would
return the whole saving. That needs a scalar-integer GEMM for the head role, which does
not exist; see `quantized-embeddings`. Tied models therefore continue to be served from
their existing quantized `lm_head`, which is already shipped and costs nothing new.

**The v1.4.3 bump changes nothing here.** Upstream's "new optimization pipeline" was
tagged at `2398c05`, the same commit the 2026-08-23 allocation study was run against, so
the pipeline this item would build on is the one already measured — see
[docs/qbench.md](docs/qbench.md).

→ [docs/embeddings.md](docs/embeddings.md)

## `yaqa` — YAQA-quality rounding in the quantizer

Round with a second Hessian on the output channels, minimizing the full-model output KL
instead of each layer's immediate activation error. Converter-side only — format, kernels
and plugin untouched — so every model we ship would get better at no inference cost.

**Measured, and the decision is open rather than blocked.** At the paper's minimum data
budget, on its own model family, YAQA gives **−19% KL in-domain and −16% on neutral text**
(Llama-3.2-1B, 2 bits, per-layer), against Appendix A.11's −20.4% at that same budget —
i.e. it reproduces the paper, and nothing structural in EXL3 is costing us the effect. It
survives the domain shift that `EXL3-SC` failed, and the gain grows as bitrate falls.
A.11's curve suggests ~−25% at full budget.

**The cost is the question, not the benefit.** A reverse-streaming gradient pass the
converter does not have (it is forward-only, one module resident, no autograd), 2.8x the
Hessian working set on dense models and 14x on wide MoE, a calibration corpus we do not
ship, and one unexplained pathological layer — the first block's `down_proj`, reproducible
across both models tested — worth 1-3 points if gated around.

**What would move it, cheapest first:** re-run the headline with
`apply_out_scales = True` (see below) — it runs on this workstation and can close the
entry outright; then the 8B (~20.5 GiB, one 24 GiB card, not this workstation); a
full-budget sketch (corpus already in the HF cache, only GPU time missing); or an
explanation of that first-block layer.

**The cheap side-result was measured and is closed (2026-08-31).** EXL3 applies
`out_channel_scales` and then rounds as if the output metric were the identity, which it
no longer is — but restoring it costs up to **+57% KL**, and `α = 0` (the shipped
behaviour) is at the optimum of a five-point sweep. Dropping that factor is *how*
`apply_out_scales` works: it makes the rounding minimize relative per-channel error
rather than absolute. Nothing to harvest.

**A caveat this leaves on the numbers above:** `probe.py` sets `apply_out_scales = False`,
so the −19% is measured against a baseline up to 66% worse than what the converter ships
(`--out_scales` defaults to `always`). Whether the gain survives on top of the heuristic
is untested and is now the cheapest thing to check first.

→ [docs/yaqa.md](docs/yaqa.md), [tools/yaqa/](tools/yaqa/)

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

**A third cause, and it is upstream's.** `get_offloader().wrap_modules()` has exactly
one call site in vLLM, inside `make_layers()` — the helper that builds a *text
decoder's* `ModuleList`. A vision tower builds its own, so **no encoder is offered to
either backend on any model in any format**. This is not reachable from here for the
common case: our `process_weights_after_loading` hook sees only quantized modules, and
nine of ten surveyed checkpoints ship a bf16 tower. Split out as
`upstream-queue`, since the fix that matters is a few lines upstream and helps
every multimodal model in vLLM. Sizes and evidence in
[docs/media-encoders.md](docs/media-encoders.md).

**Scope note: which backend wins depends on access pattern, and MoE inverts the
obvious answer.** Prefetch overlaps transfer with compute, so it suits dense weights
read every pass. But it is routing-blind — it copies every offloaded parameter each
forward pass — while UVA's zero-copy reads touch only what the kernel actually reads.
For routed experts UVA therefore wins decisively; measured on 2026-08-20. *(An earlier
version of this note extended that to "sparsely-read tensors like a vision tower" at
0.3-0.6 GiB. Both halves were wrong: a tower cannot be offered to either backend at
all, and the real sizes are 0.79-3.64 GiB bf16 — 3.64 on Step-3.7-Flash alone — or
0.90 GiB for Muse-Glimmer's 1.92B parameters at 4 bpw.)* Verified 2026-08-19 that the prefetch path is otherwise fully
functional with EXL3 tensors, so pinning is the only outstanding requirement.

**Honor the parameter selectors in that registration.** Both backends take a set of
parameter-name segments and offload *only* what matches — `--cpu-offload-params` for
UVA, `--offload-params` for prefetch, both exact dot-delimited segments
(`f".{param}." in f".{name}."`). Registering our tensors without that check makes the
EXL3 path silently non-selective, which nobody notices until someone tries the
selective form and gets the whole model offloaded. Four lines at the time, an
irritation to retrofit — and selectivity is what makes the sparse case work at all:
`--cpu-offload-params experts` is why a routed model pays PCIe for only the experts it
actually reads.

**What our own registration buys on a quantized tower is bandwidth, not capacity.**
Muse-Glimmer's tower moves 0.89 GiB per image batch instead of the 3.57 GiB the same
weights would be at bf16 — a 4x cut in the per-image cost of having evicted it. That is
the difference between an eviction you tolerate and one you leave in place. It applies
to one checkpoint today and to anything this project quantizes with `--vision_bits`
later.

→ [docs/format-and-loading.md](docs/format-and-loading.md) "CPU offload",
[docs/media-encoders.md](docs/media-encoders.md)

## `upstream-queue` — Findings other projects should hear about

Ten items across vLLM, llm-compressor and exllamav3: two patches ready to offer, one
blocked on a design decision that is not ours, six reports, and one question we cannot
yet phrase honestly. They were scattered across TODO items, patch headers and subject
notes, which made the queue invisible *as* a queue -- what is ready, what is blocked,
what is worth doing first.

**The whole inventory, with evidence, framing and a priority order, is in
[docs/upstream.md](docs/upstream.md).** This item exists so the work has a slug; it
deliberately does not duplicate the contents.

**What is next**, per that note's ordering: llm-compressor's `strategy: "channel"`
default (finished evidence, purely their benefit, nothing to negotiate), then
TurboQuant's `key=int` collision, then the halved softcap patch.

**One reproduction is missing and blocks the highest-value item.**
`vllm-embed-quant-config` cannot be filed until the speculative-decoding breakage it
introduces has a demonstration, and the two MTP entries in `bench/` do not provide one:
an MTP drafter shares the target's embedding, so the mismatch never arises. It wants an
*external* drafter against a quantized target -- `turboderp/Qwen3.6-27B-DFlash-exl3` is
the checkpoint, and it would make a good `full`-tier entry regardless.

→ [docs/upstream.md](docs/upstream.md)

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

**Six design rules, established by two SWE-bench runs and painful to retrofit.**
Evidence for each is in the note; the rules themselves are what shapes the build:

- **Compare paired, never marginal** — on the same data, marginal turn medians said
  the opposite of the paired comparison.
- **The effective sample size is the discordant count**, not the problem count. At
  ~17% discordance, 23 problems bought four informative pairs.
- **Budget ~150 problems minimum, 300 for comfort.** A cloud baseline for full Lite
  is ~$90; wall time binds, not money.
- **Capture turns-to-failure, not only pass/fail.** A model that knows it is stuck
  behaves differently from one that does not, and the pass/fail column cannot see it.
- **Shuffle the instance list with a fixed seed, shared across arms.** Lite is ordered
  by repo, so a truncated run samples one codebase — the killed cloud run's 92 results
  are 93% django. Truncation is not an edge case, and a shuffle also composes with
  resuming.
- **Pick the suite where the model lands nearest 50% resolved**, not the hardest one.
  Discordance carries the signal, so detectability peaks at half — that is the cost
  side of choosing a harder, more diverse set.

**The comparator is the weak instrument, and a controlled one needs `vast`** —
Qwen3.8-27B is ~27 GiB even at fp8, so the same-model-unquantized arm cannot run on
the dev card. **Comparability across rentals is a preflight problem, not a procurement one** —
what can move tokens is GPU architecture, driver, count (via TP degree), VRAM and
uncorrected ECC, all checkable on contact; the rest of the host moves only the
stopwatch. `tools/host_survey.py` screens a box on arrival and classifies differences
against a recorded one. Named instance types stay the answer for *perf*
comparability. What remains genuinely open is survival: a multi-day run on a spot
rental needs to resume from where it stopped.

→ [docs/capability-suite.md](docs/capability-suite.md) (the two runs, the statistics
and the rented-hardware problem), [docs/qbench.md](docs/qbench.md) (scope: why
divergence is deliberately all qbench measures),
[docs/embeddings.md](docs/embeddings.md)

## Recently closed

*One line each, newest first. Prune to ~10 when appending.*

- `head-bits` — answered 2026-08-25, see [docs/qbench.md](docs/qbench.md) "Head bitrate:
  6 is defensible". Budget-neutral sweep on phi-4-mini, five points within 0.041% of each
  other on size: head 5 edges head 6 by 3.4% (inside run noise), while 7 costs +34%, 8
  costs +87% and 4 costs +35%. The hypothesis that the head wants more bits is not
  supported -- it is 16% of quantizable weights, so each head bit costs 0.19 body bits.
  With the body null and the embedding pinned at 4, the allocation solver has two scalars
  left and needs no solver. The remaining sub-item, generalizing the body null to a second
  model, is folded into that note.

- `retire-gemma4-patch` — done 2026-08-25 at the v0.28.0 bump. Upstream landed
  generic per-layer arch config (`ModelArchitectureConfig.from_layers`), so
  `Gemma4Config` reads `model_config.model_arch_config` directly and the patch
  was dropped. `vllm-transformers-backend-embedding-postprocess.patch` retired
  with it, by a better mechanism than ours — upstream rebases the embedding's
  class instead of substituting it. Both retirements confirmed by the gate
  rather than by reading: the entries exercising them capture at 0.000e+00
  against pre-bump baselines. See README "Retired at the 0.28 bump".

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
