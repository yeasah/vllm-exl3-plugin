# KVarN: a low-bit KV cache we ported, measured, and shelved

KVarN is a calibration-free, Sinkhorn variance-normalized KV-cache quantization
offered to vLLM as [PR 46812](https://github.com/vllm-project/vllm/pull/46812).
We ported it, ran it, and found it **strictly worse than fp8 on the axis that
matters to us**. This note is the evidence, including two traps that would cost
the next person a day each.

## Why we looked

TurboQuant's prefill transient is structural: three `_continuation_prefill`
buffers at 2048 B/token each, **linear in cached context**, on an axis vLLM's
profiler never varies (see [turboquant-kv.md](turboquant-kv.md) and
[upstream.md](upstream.md) for why the remaining two fixes were declined). That
makes TQ headroom unvalidatable by a short prompt and keeps `--kv-cache-memory`
load-bearing. An alternative low-bit KV path was worth pricing.

KVarN was the candidate because its PR is **dead**: one commit, last pushed
2026-06-30, `mergeable: dirty` since the day it opened, `review_comments: 0`, and
five issue comments that are two bots, two mergify conflict notices, and a
stranger asking for an ETA the author never answered. Sixty-seven days, no
maintainer engagement. That is a green light rather than a warning -- there is no
risk of duplicating work that is about to land, and the code is Apache-2.0 under
the vLLM CLA, so adopting it is unencumbered.

> **Trap.** The PR's `updated_at` reads two months later than its last push,
> because that field counts comments. Check the head commit's date,
> `mergeable_state`, and `review_comments` instead. We read the wrong field and
> called an abandoned PR "actively maintained".

## The port

`experiment/kvarn` in [our fork](https://github.com/yeasah/vllm/tree/experiment/kvarn),
based on `v0.28.1rc0-235-g2fe5cef35e` — upstream `main`, not the v0.28.0 tag the
appliance branch uses, because the port targets the post-`TQ*Spec` contract.
See [../patches.md](../patches.md).
All 5,242 lines of new code apply unchanged; every vLLM symbol they import still
exists. The work is entirely in the ~300 lines of hooks across 12 files, and two
of those hooks turned out to be **deletions**:

- `TQFullAttentionSpec` / `TQSlidingWindowSpec` and their `tq_slot_size` field are
  gone from vLLM, replaced by `AttentionSpec.state_content_bytes` plus the
  `customize_spec()` / `supported_kv_cache_layouts()` backend hooks. So the PR's
  spec subclass and its manager registration are simply dropped.
- `customize_spec` sets `state_content_bytes = tile_bytes_aligned // group`,
  verified to reproduce the backend's own `get_kv_cache_shape` page size for all
  four presets at head_size 128 and 256.
- The layout is `LBHNC`: vLLM allocates `[blocks, heads, block_size, slot_bytes]`
  where KVarN's kernels index `[blocks, heads, tile_bytes]`. Same bytes, same
  order (`block_size * slot_bytes == tile_bytes`), so folding the trailing pair
  is a free view.

**No build is needed.** KVarN is pure Python and Triton. `wheels.vllm.ai/<sha>/`
publishes a per-commit wheel; extract its extensions into the tree and run with
`PYTHONPATH`. Extract the whole `vllm/` prefix, not just the `.so` files -- the
wheel ships ~1970 vendored Python files (`vllm/third_party/`) that are absent from
git, and adding `_flashmla_extension_C.abi3.so` without them makes the code
believe flashmla is available and then fail to import its interface.

## The bug that cost a day, and the isolation that found it

Output degenerated after the first token. The chain, each step eliminated by
measurement rather than argument:

| step | finding | eliminated |
|---|---|---|
| exact-commit wheel | `auto` correct, KVarN garbage | the platform |
| fused vs `_decode_path_slow` | byte-identical output | the decode kernels |
| store/dequant round-trip in isolation | 10% rel err @ 4 bits, 50% @ 2 bits, both branches | the quantizer |
| cache readback | `|K|max=0.000000, nonzero=0` | **nothing was written** |
| flush probe | `_batched_flush` never called | the trigger, not the write |

Cause: the impl never received `layer_name`, so everything in `kvarn_attn.py`
that scopes state per layer or per KV-cache group found nothing, `flush_block_ids`
stayed empty, and the cache was all zeros. Prefill answered from raw K/V (first
token correct); every later token attended over nothing.

**That was a porting error, not upstream drift.** The upstream `attention.py`
hunk carried this plumbing alongside the `TQSlidingWindowSpec` logic that
`customize_spec` legitimately replaces, and trimming the hunk took both. The
lesson is narrow and worth keeping: *when an upstream hunk mixes concerns,
dropping the part you have replaced can silently take plumbing with it.*

`KVARN_FORCE_SLOW_DECODE=1` is kept in the tree as a bisect lever -- it routes
decode through the fp16 dequant + SDPA fallback, which separates the store side
from the decode side in one run.

## What it costs and what it buys

Qwen3.8-27B at 3.00bpw + bq, RTX 5070 Ti (16 GiB, 896 GB/s), `max_num_seqs=1`,
`max_num_batched_tokens=512`, 9728-token prompt:

| | KV memory | context | prefill | generation |
|---|---|---|---|---|
| `kvarn_k4v2_g128` | 2.82 GiB | 160,849 tok | **355 t/s** | 48.4 t/s |
| `turboquant_4bit_nc` | — | — | — | 48.3 t/s |
| fp8 | 4.53 GiB | 137,882 tok | **1243 t/s** | — |

**+17% context for 3.5x slower prefill.** The static pool costs 1.71 GiB, worth
~97,500 tokens; even recovering all of it only reaches 1.75x fp8 on context, and
prefill does not move -- that is Sinkhorn running over every tile at quantize
time, inherent to the method rather than an implementation defect.

Both rates check out against geometry: KVarN 18,825 B/token against a predicted
16,384 plus ~2,450 of unquantized GDN state amortized per token; fp8 35,277
against 32,768 plus the same ~2,450.

## Trap 1: `k4v4` at head_dim 256 is exactly fp8

`tile_bytes_aligned` rounds the **per-token slot to a power of two** whenever
`head_dim >= 256`. Qwen3.8-27B's k4v4 slot is 272 bytes -- sixteen past a
boundary -- so it rounds to 512:

| preset | D | tile | raw 4-bit | padding | vs fp8 |
|---|---|---|---|---|---|
| `k4v4_g128` | 128 | 17,920 | 16,384 | 9% | 1.83x |
| **`k4v4_g128`** | **256** | **65,536** | **32,768** | **100%** | **1.00x** |
| `k4v2_g128` | 256 | 32,768 | 24,576 | 33% | 2.00x |

So `k4v4` on a head_dim-256 model has *exactly fp8's byte rate* and cannot win on
size no matter what. Our first run used it and looked strictly dominated on every
axis; `k4v2` is the honest comparison.

The rule exists so Gemma-4's heterogeneous head_dims (256 alongside 512) give
integer slot ratios, because vLLM's page-size unification used to grow pages by
*scaling `block_size`*. Current vLLM does the opposite for specs carrying
`state_content_bytes` -- it pads the page and leaves `block_size` fixed -- so on a
uniform-head_dim model the rule buys nothing and costs 88%. Relaxing to 8-byte
alignment would take k4v4 to 1.83x and k4v2 to 2.37x. Untested.

## Trap 2: the pool is sized for a 256-way batch

`pool_slots = max(2*max_num_seqs + ceil(max_num_batched_tokens/group) + 8, 8)`,
allocated per layer as `[pool_slots, group, kv_heads, head_dim]` fp16 for K and
again for V. On Qwen3-4B, measured 41 MiB per slot:

| config | slots | non-torch | KV tokens |
|---|---|---|---|
| default `max_num_seqs` (256) | ~73 | 2.76 GiB | won't fit at util 0.60 |
| `max_num_seqs=1`, mnbt 2048 | 26 | 1.02 GiB | 19,200 |
| `max_num_seqs=1`, mnbt 512 | 14 | 0.54 GiB | 32,000 |

Two things worth knowing. `KVARN_POOL_MEM_FRAC` is a **ceiling, not a target** --
it only caps `max_num_seqs`, so once concurrency is 1 the pool sits at its
structural floor and the knob does nothing (swept 0.5/0.20/0.05, byte-identical).
And at low concurrency the `ceil(mnbt/group)` term dominates `2*S`, so the pool is
sized by the chunk budget rather than by the workload.

The 27B's measured 1.59 GiB is ~14x the structural prediction of 112 MiB, which is
**unexplained**. Either `_max_num_batched_tokens` inside the impl is not what was
passed, or the pool is created for all 64 layers rather than the 16 that attend.
One probe at `_ensure_pool` printing `pool_size`, `_max_num_batched_tokens` and the
impl count would settle it.

Unlike TurboQuant's, this cost is **fixed and profiler-visible** -- it appears in
`consumed memory` at startup rather than as a mid-session OOM. That is the design
property we went looking for, working as intended, even at an unwelcome magnitude.

## Generation speed cannot differ here, and that locates their claim

48.4 vs 48.3 t/s is not a tie between two 4-bit formats; it is a measurement of
something structurally unobservable. Weights are ~10.5 GiB per decode step, so at
48 t/s that is 508 GiB/s -- **61% of the card's 834 GiB/s**, i.e. weight-bound.
The entire KV difference between the two formats is:

| context | TQ4 KV/step | KVarN KV/step | delta as % of weight traffic |
|---|---|---|---|
| 4,096 | 0.082 GiB | 0.072 GiB | 0.09% |
| 65,536 | 1.306 GiB | 1.149 GiB | 1.49% |
| 160,849 | 3.205 GiB | 2.820 GiB | 3.66% |

KV traffic only matches weight traffic at ~527,000 tokens **at batch 1** -- and
that threshold divides by batch size, so at batch 16 it is ~33,000 tokens, which
is ordinary. Their throughput claim is a *serving* claim: many concurrent
sequences with real context, where weights amortize across the batch and KV
traffic does not. Every other design decision agrees -- a pool sized for 256-way
concurrency, an `fa_rows` cap at 262,144 tokens, a fused-vs-materialize crossover
tuned on tokens/sec. It is a throughput project, measured in a regime that is not
a single-stream appliance's.

Prefill is the cost that does **not** amortize with batching, which is why it is
the axis that decides this and the one their design never optimizes for.

> **Recon heuristic, earned here.** Neither "prefill" nor "input" appears once in
> the KVarN README. What a project omits is a design statement. Ask which axis a
> competitor does not talk about before reading the numbers it does report --
> it is free and it sets the expectation correctly.

## Verdict, and what is still open

Not adopted. +17% context (or ~2x with the alignment fix and the pool recovered)
does not pay for 3.5x prefill on an interactive single-stream appliance.

**Untested, and the reason to come back: accuracy.** "Matches FP16 accuracy" is
strong vague language. If it survives a real measurement, there is plausibly a
place for KVarN in *offline batched* work -- where prefill cost amortizes across a
queue and nobody is waiting on a first token, which is exactly the regime its
design targets. The arm costs one string: `tools/gsm8k_kv.py` and
`tools/niah_kv.py` take the KV dtype straight through `build_llm`, trial lists are
seeded identically, so results pair against the existing TQ and fp8 runs and the
McNemar test applies unchanged. See TODO `kvarn-accuracy`.

Pin `KVARN_FUSED_VERIFY=0` when doing it. The fused-verify gate triggers at
`ceil(max_seq_len/group) >= 64`, which at group 128 is **exactly 8192 tokens** --
exactly the middle NIAH rung. Without pinning, one arm silently spans two
implementations, and the `g64` presets move the boundary to 4096, so different
arms would have different crossovers.

## The durable outcome

More valuable than the format: the **out-of-tree door for KV quantization is one
enum away from open**. vLLM now has `customize_spec()`, `kv_quant_mode`,
`page_size_padded`, `KVCacheLayout` and `AttentionBackendEnum.CUSTOM` -- all
generic, all usable from a plugin. The only thing still impossible from outside
the tree is *naming a cache dtype*, because `CacheDType` is a closed `Literal`.

KVarN is the evidence: 5,694 additions across 20 files, of which the entire
in-tree surface is **4 lines appended to `CacheDType` and 1 line in
`AttentionBackendEnum`**, sitting directly above the placeholder commented "for
third-party/custom backends". A complete working KV quantization, unreviewed for
two months, gated on five lines. See [upstream.md](upstream.md).

## Running it again

    git clone -b experiment/kvarn https://github.com/yeasah/vllm.git /tmp/kvarn
    # extensions: wheels.vllm.ai/<sha>/vllm-*-cp38-abi3-manylinux_2_28_x86_64.whl
    # extract the whole vllm/ prefix into the tree, then:
    PYTHONPATH=/tmp/kvarn python -m vllm.entrypoints.openai.api_server \
      --model <model> --kv-cache-dtype kvarn_k4v2_g128 --block-size 128 \
      --max-num-seqs 1 --max-num-batched-tokens 512

`block_size` must equal the preset's group. KVarN asserted that inside
`get_kv_cache_shape`, which the current backend contract never calls, so the
invariant is now unguarded -- it should be re-asserted in `customize_spec`.
