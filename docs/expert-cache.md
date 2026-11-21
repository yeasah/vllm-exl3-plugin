# Expert cache: the offload design for high-bpw MoEs on small cards

*Research and design pass, 2026-11-01, on the low-end target: RTX 3060 12 GiB (Gen4 x16)
shared with a desktop, WSL2, i5-12400F, 23 GiB RAM. All served measurements are
`turboderp/gemma-4-26B-A4B-it-exl3`, 128 experts, top-8, 30 layers, TP=1, graphs on.
Tools are in [tools/offload/](../tools/offload/). Background and older measurements are in
[cpu-offload.md](cpu-offload.md); where the two disagree, this note is newer.*

*Phase 0 follow-up, 2026-11-02, on `cuda`: RTX 5070 Ti 16 GiB (Gen5 x16), bare Linux 6.12,
Ultra 9 285, 23 GiB RAM. Routing traced on three models; see [Phase 0 results](#phase-0-results).
Routing tables below are from the corrected replay.*

**Status: phases 0–2 complete and gated, phase 3 built (2026-11-03); see
[Phase 1](#phase-1-results), [Phase 2](#phase-2-results) and [Phase 3 results](#phase-3-results).** The recommendation is one mechanism that replaces
`register_offload`: a pinned host copy of every expert written at load time, a fixed VRAM
slot arena managed as an LRU between steps, and a whole-layer staging copy for prefill.

**Projected for 4.10bpw on this card:** decode ~19 → ~40 tok/s, prefill ~29 → ~150+ tok/s.
These are projections from a validated cost model and a replay, not measurements of the
built thing. See [Projection](#projection).

---

## What changed since cpu-offload.md

**Bigger-than-VRAM loading was not fixed by vLLM 0.32. WDDM oversubscription is hiding it.**
`EXL3Parameter.store` and `_moe_weight_loader` still `.to(device)` every tensor, and
`load_model` still runs `process_weights_after_loading` after `load_weights`. The load works
here because WSL2's driver model pages VRAM to system memory: `oversub_probe.py` allocated
and filled 16 GiB on this 12 GiB card with `max_memory_allocated` at 16.0 and
`cudaMemGetInfo` reading 0 free from 11 GiB up. Strata documents the same behaviour as a
Windows failure mode ("the driver pages the excess to system RAM… speed collapses").

So the bare-Linux OOM in [cpu-offload.md](cpu-offload.md) should still stand. That is inferred
from the code, and needs one run on a Linux host to confirm. **Any result on this host that
exceeds physical VRAM at any moment is contaminated, including at load**, and it fails
differently from run to run: 7.1 GiB of offload died in `empty_cache()` with
`cudaErrorMemoryAllocation`, while 6 GiB loaded.

**Routing is neither uniform nor stationary, and that reverses "paging buys nothing".**
The earlier argument assumed near-uniform routing, and it has now been measured directly.

**EXL3 has CPU kernels now.** "There is no CPU kernel for it" is out of date: exllamav3
has `cpu/moe_mul1.cpp` with AVX2/AVX-512 paths. It requires mul1-codebook experts and K <= 8.

**Codebook is not a design constraint.** The gemma checkpoints here are `mcg` only because they
are the smallest MoE available for this card. Current-pipeline quants outclass that era, so an
`mcg` quant worth serving is a reason to produce a new one, not to support the old codebook.

**Upstream has not solved this for us, so no bump is needed.** Fork is `v0.32.0+16`;
upstream main is 1383 commits past the merge base, and only one touches
`model_executor/offloader/` (#62407, a small prefetch change). vLLM's MoE offload work (RFC
#38256, PR #37190) was reported unmerged and no such commit is on main. What did land is
PLE/Engram offload, KV offload and MRV2 buffer work, all for other subjects. exllamav3 1.6.6's
offload work is inside its own CPU-MoE runtime, not something this path calls. Bump for models,
not for this.

---

## Measurements

### The link

`link_bench.py`, best of three. **The UVA column is understated.** That version summed `uint8`,
whose narrow loads measure PCIe transaction size rather than the link. With `int32` loads the
5070 Ti reads 36 GB/s at 1 MiB, against 4–9 before, and served decode agrees (see
[Phase 0 results](#phase-0-results)). These rows have not been re-measured.

| transfer | pinned DMA | UVA kernel read |
|---|---|---|
| 256 KiB | 18.8 GB/s | 8.6 GB/s |
| 512 KiB | 21.9 | 9.3 |
| 1 MiB (one 4.10bpw trellis) | 22.8 | 11.0 |
| 16 MiB | 24.2 | 13.2 |

**Bulk DMA is about twice zero-copy on this host**, an advantage the Gen3 x8 host could not
show. It favours copying over reading in place whenever the copy can be planned.

### Routing

`routing_trace.py`: 41 requests, 10.9K decode tokens, 27K prefill tokens. Chat, code, maths,
sciences, law, six languages, long documents. Recorded at 2.10bpw, which fits with no offload.
Hit rate is the share of decode expert reads served from VRAM. The table is at per-layer
capacity C, as a fraction of experts:

| placement | 10% | 20% | 30% | 40% | 50% | 60% | 75% |
|---|---|---|---|---|---|---|---|
| uniform / index spread | 10% | 20% | 30% | 40% | 50% | 60% | 75% |
| static hot set, odd/even cross-validated | 22% | 38% | 51% | 63% | 73% | 80% | 90% |
| static, held-out: languages (worst domain) | 17% | 30% | 42% | 50% | 58% | 64% | 74% |
| updated between requests only | — | 37% | 50% | 62% | 71% | 79% | 89% |
| **LRU, lag 2, 32 promotions/token** | 43% | 70% | 83% | **90%** | 95% | 97% | 99% |
| LRU, lag 2, 16 promotions/token | 42% | 68% | 81% | 89% | 94% | 96% | 98% |
| LRU, lag 2, 6 promotions/token | 45% | 66% | 77% | 85% | 90% | 94% | 97% |
| same (32/token), first 32 tokens of a request | 33% | 55% | 68% | 77% | 85% | 90% | 95% |

Figures for the between-request row come from an earlier 27-job trace at matching scale. The
static rows are from the 41-job trace. The LRU rows are from the same 41 jobs re-traced on `cuda`
with longer documents (4.8K tokens instead of 2.6K), replayed with the corrected promotion order.
The first replay kept the first misses in layer order when the budget ran out, which starved late
layers. That understated small budgets and small capacities, and made the 6/token row look like a
collapse.

- **The structure is mostly temporal.** Token-to-token overlap of the routed set is 45.7%,
  against ~6% for independent draws. The hot set follows the text, so a static ranking gets
  ~63% and an LRU ~90% at the same capacity. Hottest expert median 3.3% of a layer, none
  dominant.
- **In-sample ranking flatters static placement by about 8 points at 40%**, and languages are ~13
  below the mean. Strata's PR #38 reports the same failure at 25–35%.
- **A miss does not stall.** Lag means a miss is served by UVA for that token and promoted
  for a later one, so the policy needs no host sync inside the forward pass.
- **The copy budget degrades gently.** At 40% C, 6/token (the exllamav3 default) costs 5
  points against 32/token, provided promotions go to the most-wanted misses across all layers
  rather than first-come. Steady-state need on gemma is ~15/token.
- **Prefill touches nearly everything:** 87% of experts in a 2048-token chunk, 75% at 1024.

### Served, graphs on

Clean comparison, everything physically resident (2.10bpw, 4096 context, 0.90):

| placement | bytes off | decode | prefill (~2350 tok) |
|---|---|---|---|
| resident baseline | 0 | 56.8 t/s | 233 t/s |
| spread by index, 49% hit | 2.99 GiB | 36.5 | ~57 |
| hot set resident, 70–78% hit | 2.99 GiB | **44.1** | ~53 |

**The cost model holds.** Added decode cost fell from 9.8 to 5.1 ms, a ratio of 0.52, where the
hit rates predict 0.43–0.59. Apparent decode bandwidth is about 20 GB/s, above the UVA
microbenchmark, as overlap with compute would produce. **Prefill ignores placement**, as
expected at 75%+ of experts touched, which is why it needs its own mechanism.

At 4.10bpw (6 GiB offloaded), only the first row is valid:

| placement | KV available | decode | prefill |
|---|---|---|---|
| greedy `experts`, 16 layers | 1.73 GiB | 19.3 t/s | 29 t/s |
| spread by index, all layers | **−0.43** (700 MiB forced) | 15.7 | 24 |
| hot set, all layers | **−0.43** (700 MiB forced) | 12.6 | OOM mid-prefill |

**Rows 2 and 3 are not evidence about placement.** Physical VRAM was exhausted ("0 bytes
free"), so WDDM was paging. They show one thing: **post-load offload cannot express a
spread placement on this checkpoint**, because 2.16 GiB of the freed memory never came back.
The stranding did not appear at 2.10bpw; 1.03 MiB trellises against 540 KiB is the obvious
difference, and the mechanism was not established.

### Phase 0 results

**Routing on three models.** Same 41-job workload, decode hit rate at 40% per-layer capacity,
LRU with lag 2:

| model | experts × layers | token overlap | static, odd/even | LRU, 32/token | LRU, 6/token | first 32 tokens | promotions/token, unlimited |
|---|---|---|---|---|---|---|---|
| gemma-4-26B-A4B 2.10bpw | 128 × 30 | 45.5% | 61% | **90.5%** | 84.6% | 77% | 23 |
| Qwen3.6-35B-A3B 2.00bpw-H5 | 256 × 40 | 37.2% | 63% | **85.3%** | 76.6% | 68% | 46 |
| Laguna-XS-2.1 2.00bpw | 256 × 39 | 36.0% | 52% | **81.1%** | 67.8% | 60% | 61 |

**The LRU result generalizes, a little weaker on 256-expert models.** At 50% capacity the three are
95, 91 and 87%. Those models want two to three times more promotions per token, so their budget
matters more: Qwen at 64/token is 87.4%. Laguna's static placement is weakest of the three
(math held out: 32%), so its gain from the LRU is the largest.

**Concurrency degrades gently, not catastrophically.** Here n decode streams share one cache and
a step merges their routing; promotions are capped per step. At 40% capacity, 32 per step:

| concurrent | gemma | Qwen3.6 | Laguna |
|---|---|---|---|
| 1 | 90.3% | 85.1% | 80.8% |
| 2 | 84.4% | 81.1% | 74.4% |
| 4 | 77.3% | 76.7% | 68.9% |
| 8 | 73.6% | 72.8% | 64.8% |

Scaling the budget with n buys 1–3 points at 40% and more at 60–75% capacity. At 8 streams the
LRU is still at or above single-stream static placement. This workload has no shared-prefix
traffic, so it is close to the worst case.

**Bare Linux OOMs, as inferred.** With an 11 GiB ballast on the 16 GiB card,
Ornith-1.5-9B-exl3 6.00bpw plus `--cpu-offload-gb 6` dies in `EXL3Parameter.store`
(`linear.py:100`). `oversub_probe.py` hard-OOMs at 15 GiB; there is no oversubscription.

**Pinning and DMA pass.** 30 blocks × 415 MiB (12.2 GiB, the 4.10bpw expert set) pinned in
4.3 s under `ulimit -l 8192`; the limit does not apply to CUDA host registration. Whole-block DMA
runs at 53.5 GB/s, 8.1 ms per layer, and UVA readback of the pinned blocks is correct.

**Served on the 5070 Ti,** gemma 2.10bpw, 3 GiB of experts off, graphs on, 1024-token chunks.
Reads/token are the offloaded expert reads predicted from the trace, out of 240:

| placement | reads/token off | decode | prefill (~2350 tok) | effective link |
|---|---|---|---|---|
| resident | 0 | 124.7 t/s | 670 t/s | — |
| spread by index, all layers | 122.5 | 86.4 | 156 | ~53 GB/s |
| cold set off (`hot_selector.py`) | 52.9 | **98.0** | 156 | ~38 GB/s |
| greedy `experts`, 16 whole layers | 128 | 79.3 | 120 | ~42 GB/s |

The effective link figure is offloaded bytes over added time, with the VRAM read they replace
credited back. **UVA inside the real GEMM runs at 40–55 GB/s**, close to DMA, which settles the
microbenchmark anomaly. The cold set is 13% faster than spread on 2.3× fewer link reads; decode
cost tracks link bytes. This was the first clean placement comparison at 4.10-like proportions,
and it agrees with the 3060's 2.10bpw table.

**Prefill over UVA re-reads each expert about 60 times per chunk.** Spread adds 11.5 s over
3 chunks. At ~50 GB/s that is ~190 GB, against 3.2 GB offloaded, so the kernel fetches weights
once per row tile and not once per chunk. Staging the same 3.2 GB once per chunk costs ~60 ms.
Projected prefill with staging is **~600+ t/s against 156**, with no overlap needed. This makes
phase 2 a larger win than projected, and the `rows > T` threshold more important: a few routed
rows already multiply link traffic.

**Projection at 2.10bpw on this card,** 48% slots: LRU hit ~94%, so ~13 reads/token of misses,
about 0.4 ms, plus ~15 promotions/token on the side stream. That gives **~115 t/s decode against
98 cold-static and 86 spread**, and prefill near resident.

**Graph spikes pass** (`graph_spike.py`). The test uses the real `exl3_mgemm`, synthetic 4-bit
mul1 experts (2816 × 1408, 128 per layer, 5.67 MiB each) and the decode shape.

- **(b) Pointer flips are followed under a captured graph, with no recapture.** Replays are
  bit-identical to eager in five cases: all slots in VRAM; routed entries flipped to the host copy
  of the same bytes; flipped to different weights; flipped back; half flipped. They also match
  eager when `topk_ids` change between replays. **A UVA read is bit-identical to a VRAM read**, so
  a miss changes timing and never output.
- **(c) A captured per-layer `topk_ids` copy to a pinned ring** costs 148 µs on a 30-layer graph,
  +4.1% of a 3.6 ms replay of MoE alone. That is roughly 2% of a real step. All 30 layers' IDs are
  correct after replay. If it matters, one device-side ring plus a single D2H per step would cut
  it.
- **(d) Side-stream promotions barely touch the replay while they fit inside it:**

  | experts/step | MiB | replay | landed by end of replay |
  |---|---|---|---|
  | 6 | 34 | +0.6% | 50/50 |
  | 15 | 85 | +1.4% | 50/50 |
  | 32 | 182 | +2.7% | 49/50 |
  | 64 | 363 | copy-bound, 2× | 0/50 |

  **So the promotion budget is time, not count:** about replay time × DMA rate per step, here
  ~180 MiB in 3.6 ms. A copy that has not landed is simply flipped one boundary later. The
  controller should check its event rather than assume completion.

**On the 3060 under WSL, 2026-11-03, everything passes too.**

- **(a) Pinning:** 30 × 415 MiB (12.2 GiB) pinned in 8.6–11.9 s (`pin_probe.py`), with DMA and UVA
  readback correct. Strata's ~1 GB WSL pin limit does not apply here.
- **Link speed varies by day on this host.** Whole-block DMA ran at 14–15 GB/s, `link_bench.py`
  DMA at 17–18 GB/s, and fixed-width UVA reads at ~10 GB/s, all with both machines idle at Gen4
  x16. That is below the first day's 23–24 GB/s DMA, so plan for 15 GB/s here.
- **(b) Pointer flips:** all identity checks pass.
- **(c) Ring copy:** +132 µs, +1.4% of a 9.7 ms replay.
- **(d) Side-stream promotions:**

  | experts/step | MiB | replay | landed by end of replay |
  |---|---|---|---|
  | 6 | 34 | +0.6% | 50/50 |
  | 15 | 85 | +3.6% | 50/50 |
  | 32 | 182 | +3.1% | 50/50 |
  | 64 | 363 | copy-bound | 0/50 |

  The slower card's longer step makes room for the slower link, so ~180 MiB per step fits here
  too.

**Phase 0 is complete. Nothing found changes the design.**

### Phase 1 results

**Built:** placement is decided at store time in `cpu_offload.place` (dense linears) and
`place_expert` (routed experts). An offloaded tensor is copied from the mmapped checkpoint
straight into pinned host memory and never touches the device. Each layer's offloaded experts
sit in one pinned block per projection and component. Which experts go is planned at the first
arrival, over expert ids in order, so the result does not depend on load order. Selectors and
budget are unchanged. `register_offload` remains only for the `EXL3_DEQUANTIZE=1` path.

**Pinned memory comes from the driver** (`cuMemHostAlloc`), at exact size:

- `pin_memory=True` rounds every allocation up to a power of two. That held 9.07 GiB of engine
  RSS for 2.99 GiB offloaded, against 6.39 GiB at exact size. The old per-tensor path paid the
  same rounding: a 1.03 MiB 4.10bpw trellis took 2 MiB.
- `cudaHostRegister` on a plain allocation is exact but read slowly inside a serving process:
  decode −8%, prefill −31%, in two interleaved rounds. A fresh process showed no difference,
  so page backing in a long-lived process is the likely cause; this was not established.

**Gates, gemma 2.10bpw on the 5070 Ti, final code:**

| placement | decode | prefill | device peak in load | engine RSS | output vs resident |
|---|---|---|---|---|---|
| resident | 124.6 t/s | 670 t/s | 11.0 GiB | 3.3 GiB | — |
| spread | 85.7 | 154 | 6.9 GiB | 6.4 GiB | identical |
| cold set off | 96.6 | 152 | 6.9 GiB | 6.4 GiB | identical |
| greedy, 16 layers | 77.0 | 115 | 6.9 GiB | 6.4 GiB | identical |

Speed is within the old path's run-to-run spread (77–86 t/s on spread, interleaved).

**More than the free VRAM loads on bare Linux.**

- **MoE:** with a ballast leaving 7.2 GiB free, gemma 2.10bpw (9.6 GB) with 5 GiB of experts
  off serves at 61 t/s.
- **Dense:** with 4.6 GiB free, Ornith-9B 6.00bpw with 4.8 GiB off serves and answers
  correctly. Yesterday this exact setup OOM'd in `store`.

Two startup steps outside the load needed headroom on those tight settings:

- flashinfer autotune runs a dummy batch after KV sizing, and the plugin's embedding op
  allocates 352 MiB in it that the profile did not reserve;
- inductor compiling the dense model makes a ~1.9 GiB transient, about one vocab × hidden
  matrix.

Neither involves offload, but both decide how small a card can serve a given model. They are
worth an item of their own.

**4.10bpw on the 3060 (WSL), ~6 GiB of experts off, closing the table above:**

| placement | KV available | decode | prefill | device peak in load |
|---|---|---|---|---|
| greedy `experts`, 16 layers | 2.51 GiB | 19.2 t/s | 29 t/s | 10.6 GiB |
| spread by index | **2.52 GiB** (was −0.43) | 20.7 | 30 | 9.9 GiB |
| **cold set off** | **2.52 GiB** (was −0.43) | **27.8** | 30 | 9.8 GiB |

Peaks include the desktop's 1.4 GiB and stay under the card's 12 GiB, so WDDM never pages. No
memory is stranded, and outputs are identical across all three. **Hit-aware placement is worth
+44% decode at 4.10bpw on this card, from phase 1 alone.** Prefill stays at ~30 t/s whatever
the placement, which is phase 2's job. Load takes ~90 s here against 15–25 s on `cuda`.

**Correction, found in phase 2:** the identity check above compared completions of prompts in
gemma-3's chat format, which gemma-4 ends at once, so the 160 compared tokens were template
loops, and every prompt was under 32 tokens. `measure.py` now compares chat completions through
the server's template plus prompt logprobs over a 2350-token, three-chunk prefill. Re-run on
`cuda` and the 3060, every placement still matches resident exactly on both.

### Phase 2 results

**Built:** `cpu_offload.build_staging` runs at the end of load. It allocates one VRAM window the
size of the largest layer's offloaded trellis and gives each hosted layer a second set of pointer
tables into it. In `apply`, a batch over `EXL3_STAGE_MIN_TOKENS` (32) copies the layer's host
blocks into the window in stream and runs on those tables. One window serves all layers because
stream order serializes them. The copy is async and graph-capturable from driver-pinned memory,
checked on both hosts.

**`cuda`, gemma 2.10bpw, 3 GiB off, 1024 batched tokens:**

| placement | prefill (~2350 tok) | decode |
|---|---|---|
| resident | 671 t/s | 126.3 |
| spread, staged | **634** | 86.5–89.0 |
| spread, UVA (staging off) | 158 | 88.8 |
| cold set, staged | 639 | 113.7 |
| greedy 16 layers, staged | 639 | 81.8 |

94% of resident, 4x over UVA. Outputs identical to resident in every row. TTFT over resident
steps with chunks, not tokens:

| prompt tokens | 126 | 507 | 1022 | 1094 | 2045 | 2088 | 3064 | 3986 |
|---|---|---|---|---|---|---|---|---|
| chunks | 1 | 1 | 1 | 2 | 2 | 3 | 3 | 4 |
| added ms | 56 | 69 | 69 | 142 | 128 | 206 | 223 | 275 |

**3060 (WSL):** staging works as designed, and prefill is bound by the MoE kernel.

- **Profiled at 2.10bpw spread:** `exl3_moe_mm` is 90% of GPU time, about 138 ms per layer for
  1016 tokens, the same as resident. Staging copies run at ~20 GB/s and cost ~4% of prefill.
  TTFT then matched resident, 4.58 s against 4.46 s.
- **Noise:** an earlier sweep of the same configuration measured 7.2 s, and greedy and spread
  each measured 45 s then 77 s for consecutive prompts. On WSL, prefill timings need a profile
  or a repeat before they count.
- **4.10bpw cold:** TTFT grows with tokens, at 14.5 s per 1016 (~70 t/s, against 30 over UVA).
  The profile shows `exl3_moe_mm` at 468 ms per layer and staging at 0.35 s per prompt, under 3%.

**The 4.10bpw kernel is slow because the checkpoint is mcg.** `stage_bench.py`, synthetic
gemma-shaped experts at K=4 and 1024 tokens on the 3060:

| codebook | resident | staged | UVA |
|---|---|---|---|
| mcg | 450 ms/layer | 456 | — |
| mul1 | 122 | 135 | 947 |

**Confirmed served, on a mul1 checkpoint:** Laguna-XS 4.00bpw (256 experts, 40 layers) on the
3060, with experts 128–255 of every layer off (7.31 GiB):

| prompt tokens | 127 | 1022 | 1095 | 2089 | 2884 |
|---|---|---|---|---|---|
| TTFT | 0.84 s | 3.04 s | 3.75 s | 6.77 s | 8.71 s |

That is **~335 t/s, against ~70 for gemma 4.10bpw mcg and 29 over UVA**. Profiled per 1022-token
prefill: MoE kernel 2.3 s (59 ms/layer), staging copies 0.53 s (14.7 GB/s, constant per chunk),
the rest 0.28 s. Staged runs at ~84% of resident-equivalent; the copy is the whole difference.

So the projection's ~150–190 t/s at 4.10bpw holds, and is beaten, for a mul1 quant, not for the
mcg gemma checkpoint.
Resident 2.10bpw prefill on the 3060 is 4.4 ms/token, ~225 t/s, and is kernel-bound in the same
way. `exl3_moe_mm` runs one row per (token, expert) slot, re-decoding each expert ~64 times per
chunk. exllamav3's own large-batch tiers avoid that, and are now the larger prefill lever. See
TODO `cpu-offload`.

**Gates.**

- **TTFT steps with chunks, not tokens:** passes. On `cuda` it adds ~65 ms per chunk; on the 3060
  the copy is 0.35 s per chunk at 4.10bpw.
- **At least 50% of resident at ~4bpw:** passes against what resident would be, ~84% on Laguna
  4.00bpw mul1 with 7.3 GiB off. A resident run does not fit this card, so no direct comparison
  exists. Gemma 4.10bpw's absolute rate, ~70 t/s, is the mcg kernel's.

### Phase 3 results

**Built:** `expert_cache.py`.

- **Slots:** `EXL3_EXPERT_SLOTS` (a count, or a fraction of experts) per layer, allocated at end of
  load after the staging window.
- **One pointer table:** every layer's decode pointer table becomes a view of one `[L, 3, E]`
  tensor, so each step's updates are one scatter.
- **Routing:** `apply` copies `topk_ids` for decode-sized batches into a device buffer, which is
  capturable. A wrapper on both model runners' `execute_model` copies that buffer to a pinned ring
  after each step and records an event.
- **The controller, at each boundary:**
  1. points experts whose promotion copies have landed at their slots;
  2. plans from the newest finished step;
  3. points victims back at the host;
  4. issues copies on a side stream that waits for those writes.

  Promotions are ranked most-wanted first, up to `EXL3_PROMOTE_PER_STEP` (32). Under async
  scheduling it reads routing one step older than intended, so the effective lag is 3. The
  replay puts that at 89.7% against 90.5% for lag 2.

**Workload:** `decode_bench.py`, eight prompts of natural long answers, 7–7.2K tokens. Hit rates
are per 500-step window, from `EXL3_CACHE_STATS`. `measure.py`'s 200-token `ignore_eos` runs read
3–6 points low: they are mostly cold starts and degenerate text.

**`cuda`, gemma 2.10bpw, every expert offloaded:**

| configuration | decode | VRAM hit, steady |
|---|---|---|
| resident | 126.5 t/s | — |
| no slots | 66.9 | — |
| **40% slots** | **116.2** (1.74x) | 86–91.5%, ~89% |
| spread placement + 40% slots | 121–124 | 95–98% |

**3060, gemma 4.10bpw, every expert offloaded (11.6 GiB), same session:**

| configuration | decode | VRAM hit, steady | KV left |
|---|---|---|---|
| 40% slots, budget 32 | 29.0 t/s | 82–84% | 3.06 GiB |
| **55% slots, budget 32** | **32.6** | 89–93% | 0.82 GiB |
| 40% slots, budget 12 | 27.8 | 78–84% | 3.06 GiB |

A longer 40% run measured 23.5 t/s on a busier session, with hit rates matching `cuda`'s,
83–94%. Outputs match resident on `cuda` and phase 2's outputs on the 3060, chat and prompt
logprobs.

**Where the 3060's time goes** (profile, 40% slots, ~35 ms/token): the GPU's main stream is busy
98.5% of the time, so the controller is not a CPU bottleneck.

- MoE kernels take 23.5 ms per token. Calls with a miss over the link sit in the tail: median
  159 µs against 294 µs at p90.
- Everything else takes ~11 ms, the same as resident on this card.
- Promotions take 3.3 ms per token of side-stream DMA, overlapped.

The remaining cost is misses read over WSL's ~15 GB/s link, so slots are the lever: 40% to 55%
gave +12%.

**Gates.**

- **Graphs on, output identical:** pass.
- **Hit rate at least 85% at 40% slots on a fresh workload:** pass on `cuda`, ~89% steady. The
  3060 matched it on the long run.
- **At least 1.7x on 19.3 t/s:** pass on `cuda` (1.74x over all-offloaded). **Fails on the 3060
  at 4.10bpw.** Same session, 2000-token `decode_bench`:

  | configuration | offloaded | decode | KV left |
  |---|---|---|---|
  | greedy `experts`, the 19.3 baseline | 6.0 GiB | 22.4 t/s | 1.67 GiB |
  | cold static set | 6.0 GiB | 28.0 | 1.86 GiB |
  | every expert offloaded, 55% slots | 11.6 GiB | 31.8 | 0.89 GiB |
  | cold set + 10 slots per layer | 6.0 GiB | **32.1** | 0.84 GiB |

  That is 1.43x on today's greedy, and 1.15x on the cold static set. At equal VRAM the LRU over
  everything and the cold set plus a small LRU land together, at 93% and 90% VRAM hit.
  20 slots on the cold set left 0.11 GiB of KV and could not serve.

**A mul1 checkpoint gives the same ratio.** Laguna-XS 4.00bpw (256 experts, 39 MoE layers),
same session:

| configuration | offloaded | decode | KV left |
|---|---|---|---|
| greedy `experts` | 9.0 GiB | 30.9 t/s | 2.55 GiB |
| every expert offloaded, 45% slots | 14.6 GiB | **43.0** | 1.26 GiB |

That is 1.39x, with 79–84% VRAM hit, lower than gemma's as the replay predicted for 256
experts. Greedy at 7 GiB left 0.28 GiB of KV and could not serve.

**Why the 3060 falls short.** The projection assumed ~23 ms of base work per token and ~4 ms of
misses. Measured, the base work alone is ~11 ms of non-MoE plus the mcg MoE kernels, and the 7–10%
of misses still cost several ms over WSL's link. More VRAM for experts is not available on this
card without starving KV, so the remaining levers are cheaper misses and a cheaper base:

- a CPU tail for misses (phase 5);
- the embedding table on host (phase 5, ~1 GiB more for slots).

### A real session, replayed (2026-11-04)

`replay_requests.py` replays the JSON bodies a vLLM request log recorded, in order. Here it ran a
logged 26-request agentic session on `cuda`: Laguna-XS 2.00bpw, 131K context, prompts up to ~113K
tokens with ~50% prefix hits, and 12K-token tool outputs arriving mid-session. Seconds per
tool-output turn, over the same eight turns:

| configuration | per turn |
|---|---|
| no offload, nvfp4 or bf16 KV (identical) | 14.4–22.8 |
| autosized (everything fits; offload switches off) | 14.4–22.9 |
| 39% of experts in VRAM, nvfp4, `EXL3_KV_RESERVE=6` | 17.7–28.8 |
| no slots, every expert from host, bf16 | 18.1–28.7 |
| the old post-load offload (`dev`) | 134–165 |

There were no hangs in about 130 replayed requests.

Two traps came up while reading field logs:

- **vLLM hides its stats during a long prefill.** Prompt tokens are counted only when a prefill
  finishes. After two 10-second intervals with zero throughput, `LoggingStatLogger` calls the
  engine idle and drops the stats line to DEBUG. A two-minute prefill then looks like a hang at
  INFO: GPU busy, no stats, nothing finishing. Request logging runs at DEBUG and shows those
  lines, so it appears to "fix" it.
- **"GPU KV cache usage" is a share of capacity.** nvfp4 KV holds ~3.3x more tokens than bf16
  in the same memory, so the same prefill rate moves its percentage ~3.3x more slowly. KV dtype
  made no difference to prefill time here.

### Cost arithmetic for 4.10bpw

One expert is 3 × 1.03 MiB ≈ 3.24 MB. Full routing reads 8 × 30 × 3.24 = 778 MB/token. Greedy
reads 415 MB/token in 16 layers, about 38 ms at 11 GB/s, which with ~12 ms of base cost is the
observed 19 t/s. Prefill at 29 t/s is 34 ms per prompt token.

| 4.10bpw checkpoint part | GiB |
|---|---|
| expert trellis | 11.60 |
| embeddings | 1.38 |
| dense trellis | 1.10 |
| head | 0.52 |
| expert suh + svh | 0.08 |
| vision (skipped by `--language-model-only`) | 1.07 |

### Chunk size against residency (2026-11-09)

RTX 3060 under WSL at 0.87, 131072 `--max-model-len`, nvfp4 KV, one 46917-token prompt, varying only
`--max-num-batched-tokens`. A bigger chunk's activation peak comes out of the expert slots (60 MiB per slot
over 40 layers):

| chunk | slots | in VRAM | prefill | decode |
|---|---|---|---|---|
| 512 | 92 | 43% | 521 t/s | 41 t/s |
| 1024 | 91 | 43% | 782 | 40 |
| 2048 | 90 | 42% | 869 | 40 |
| 4096 | 85 | 40% | 998 | 40 |
| 8192 | 73 | 36% | 552 | 30 |

Prefill nearly doubles from 512 to 4096 at no decode cost, then both collapse once residency drops below the
single-stream target of 0.4 (`target_share`). The startup line switched to "fewer than 1 (single-stream decode
will be slower)" at 8192 and only there, so the warning fires where the knee is. Each doubling costs more slots
than the last (7 slots from 512 to 4096, 12 from 4096 to 8192), so the best chunk is the largest one that keeps
residency at the target, which the slot sizing already knows how to compute.

Not pursued (2026-11-10): with prefill prefetch (docs/moe.md, "Prefetching the next layer's experts") the
copy cost that made chunk size matter is mostly hidden, and vLLM's 2048-token default comes within 11-14% of
4096 on both hosts measured. Close enough to leave the default alone.

### Speculative decoding against residency (2026-11-10)

Qwen3.6-35B-A3B 4.00bpw, MTP with 3 tokens, one stream, graphs on, slots fixed by `EXL3_EXPERT_SLOTS`, two
greedy prompts (prose, code), 512 tokens each. Decode tok/s, no spec → MTP:

| in VRAM | this host (5060 Ti, 28.7 GB/s link) | godel (3060, WSL, 23.7 GB/s) |
|---|---|---|
| 20% | | 39.6, 33.7 → 40.1, 37.9 |
| 30% | | 48.9, 42.5 → 53.8, 50.8 |
| 35% | | 51.2, 45.3 → 59.2, 56.7 |
| 40% | 63.1, 57.5 → 77.0, 75.7 | 52.0, 46.3 → 61.2, 57.8 |
| 45% | 63.8, 59.4 → 79.5, 81.5 | |
| 50% | 64.9, 61.0 → 83.1, 86.8 | |
| 55% | 65.3, 62.4 → 85.7, 92.4 | |

No knee in either range: the drafter pays from 30% up on godel and at every share here, and its gain grows
with residency. The first model, that a verify step routes 1 + k rows and so costs like that many streams,
predicted a loss below ~60% and was wrong. The link was not the separator either (the two hosts differ by
~18%, memcpy and mapped reads alike).

The loss seen on godel in serving (31 against ~40 tok/s) was **nvfp4 KV on Ampere**: FlashInfer, the only
backend for it there, treats a verify batch as a decode only with its TRT-LLM decode kernel, which sm_86 lacks,
so every verify step takes the prefill path and rebuilds its metadata. Godel at 35%: default KV (FLASH_ATTN)
+16%/+25%, nvfp4 −15%/+11%; long context widens it. With nvfp4 and MTP at 40%, KV did not fit one 8192-token
request at 0.87 (FlashInfer's undeclared workspace, see docs/memory-accounting.md). Peak device memory stayed
under the card in every run, Windows' ~1.5 GiB included, so none of this was WSL paging.

**Trading residency for a drafter can pay** (user, 2026-11-10, on the Blackwell host with nvfp4 KV, where
MTP works, and `--max-model-len` pinned at the maximum): with MTP the slots came to 29% of experts in VRAM,
without it 41%; MTP at 29% decoded 62 t/s on average against 58 at 41%. On a fast link residency is worth
little to decode (+5% from 40% to 60% here without spec) while the drafter is worth 20-50%, so giving up
slots for its memory comes out ahead. Whether it does elsewhere depends on the link, the KV backend and the
share together, which is why the TODO item measures the link rather than fixing a rule.

---

### Residency against throughput under WSL2 + Docker (2026-11-14)

The user's series, Qwen3.6-35B-A3B 4.00bpw on a desktop-shared 12 GiB RTX 3060 in the serve image under WSL2
(10.32 GiB of VRAM used; vision offloaded; `--max-num-batched-tokens 2048`, `--max-num-seqs 4`). Residency is the
expert cache's share of experts in VRAM, as sized by `--max-model-len` and the KV dtype; prefill is approximate.

| context / MTP | KV | resident | prefill staging | prefill, t/s | decode, t/s |
|---|---|---|---|---|---|
| 128K | nvfp4 | 42% | dropped (auto) | 923 | 35 |
| 128K | nvfp4 | 38% | forced on | 1385 | 35 |
| 256K | nvfp4 | 34% | on | 1385 | 34 |
| 256K | nvfp4 | 36% | forced off | 923 | 35 |
| 16K | bf16 | 45% | on | 1385 | 36 |
| 16K + MTP | bf16 | 40% | on | 923 | 44 |
| 32K | bf16 | 43% | on | 1385 | 36 |
| 32K + MTP | bf16 | 36% | on | 923 | 41 |
| 64K | bf16 | 39% | on | 923 | 34 |
| 64K + MTP | bf16 | 31% | on | 923 | 39 |
| 128K | bf16 | 31% | on | 923 | 33 |
| 128K + MTP | bf16 | 22% | on | 923 | 34 |
| 192K | bf16 | 23% | on | 692 | 31 |
| 192K + MTP | bf16 | OOM | | | |

- **The second staging window is worth ~50% of prefill** (923 -> 1385 t/s) for the 2-4 points of residency it
  takes, which decode does not notice at these shares. The rule that dropped it below `target_share` gave that up
  for nothing, so the window is now on by default (`EXL3_STAGE_PREFETCH=auto` keeps the old rule).
- **No cliff in residency.** From 45% (as close to resident as this card gets, on the flat of the curve) decode
  loses ~8% at 31% and ~14% at 23%: a more gradual knee than the inclusive cache showed, on this model and content.
- **MTP is a decode win wherever it fits** -- largest at high residency (36 -> 44 t/s at 16K), shrinking toward
  none at 22% -- and never a loss. Its cost is prefill (1385 -> 923 t/s at 16-32K), since its draft head and
  verify rows take residency and staging room.
- The serve image's numbers match earlier bare-WSL2 runs; the image needed `VLLM_WSL2_ENABLE_PIN_MEMORY=1` and a
  prebuilt exllamav3 extension to get here (field notes).

## Prior art, and what to take from each

| source | what it does | take |
|---|---|---|
| Strata | hot experts in VRAM, misses computed on CPU, swaps between decode windows, next layer streams during attention in prefill | window-boundary adaptation; layer streaming; a profile saved across runs |
| llama.cpp `--moe-cache-mib` | per-token LRU; GPU runs cached experts at <=32 rows, batch falls back | independent confirmation of LRU and the ~32 threshold: 2.2x at 89% hit, and little or negative gain under ~45–75% hit |
| vLLM RFC #38256 | `expert_map` slots, LFRU, prediction planned | a Belady bound of only ~1.07x at batch 1 says skip prediction; its eager/piecewise cost is what the lag design avoids |
| exllamav3 CPU-MoE | promote-on-miss from a per-step snapshot, pinned arena, saved stats | the snapshot pattern, the keep-a-host-copy rule. The worker process is not portable to this path |
| WiSP | per-layer sync plus LRU | the rejection stands on mechanism (sync, eager), not on the uniformity argument. Co-occurrence stays unproven and unneeded |
| Strata/Engram tables | deterministic sparse lookup, known before forward, served from host | **applies to our embedding table**, see phase 5 |

**CUDA graphs decide the architecture.** [moe.md](moe.md) measures them at 1.9–4.9x, more than
any cache can return, so anything that syncs per layer is out.

---

## Design

```
load       every expert ──> pinned host block (per layer, gate/up/down adjacent)
           never touches the device
                    │ DMA at the step boundary
                    ▼
decode     VRAM slot arena (C per layer) ◄── pointer tables ──► kernel
           miss: table entry holds the host UVA address, so the read just works
                    ▲ whole-layer copy, in stream
prefill    VRAM staging window (one layer) ◄── a second pointer table
```

**Host block, written in `store()` and `_moe_weight_loader`.** One pinned block per layer, all
experts, packed gate/up/down. Experts never reach the device on load. This is the fix the TODO
already names, and it does four jobs at once: bounded load on bare Linux, no stranded pages, no
dependence on WDDM, and the packing lever. The host copy is permanent, so promotion and eviction
are one-way copies and eviction is free.

**Slots.** One fixed VRAM allocation per layer, allocated once, before KV sizing, so it is
counted honestly and cannot fragment. Addresses are stable.

**Pointer tables are the whole interface.** `exl3_mgemm` takes the table's address and the GPU
reads it at run time (`exl3_gemm.cu:634`, `:715`), so entries can change between graph replays
with no recapture. Kernels need no changes. Scale vectors stay resident, as today.

**Controller, in a wrapper around `GPUModelRunner.execute_model`.** Outside the graph, and in the
same patch style as `plugin.py`. Inside `apply`, a captured copy of `topk_ids` into a pinned
per-layer ring. At the boundary, sampling has already forced the step to finish, so the rings
are readable with no added sync. Then:

1. flip entries whose copies have landed to their slots;
2. take misses up to the budget, and for each victim set its entry back to the host address;
3. issue the copies on a side stream, to land during the next replay.

A slot is written only after its entry has left it, at a moment no kernel is running. That is
the safety argument, and it is why flips happen only at the boundary. Upstream RFC reports
routing observation inside a graph at ~1%; it is affordable here.

**Prefill.** When `rows > T`, copy the whole layer's host block into the staging window, in
stream, and pass a second table. Decided from `x.shape`, so no sync. There is no dynamic data in
it, so it is capturable if a prefill shape is ever captured. 415 MB at 23 GB/s is ~18 ms per
layer: **~0.5 s per chunk, independent of tokens in the chunk**, which is the acceptance
criterion the TODO set. `T` starts at 32 and follows from `E × DMA / (top_k × UVA)`.

| VRAM budget, 4.10bpw, this card | GiB |
|---|---|
| `0.90` of the card | 10.8 |
| non-expert weights + scales | −3.1 |
| non-torch and overheads (measured 1.2 with graphs) | −1.2 |
| staging, one layer | −0.4 |
| KV, ~3K–10K tokens | −0.7 to −1.7 |
| **slots** | **4.4 to 5.4, 38–47% of experts** |

That lands in the 90% region of the table, on the shoulder before the curve flattens. Spending
blockq on the 1.38 GiB embedding table would add ~1 GiB more.

### Projection

Decode: base about 23 ms for 4.10bpw (unmeasured, since it does not fit; 2.10bpw is 17.6 ms),
plus 10% of 778 MB at ~20 GB/s = 4 ms, giving **~37–40 t/s against 19.3**. Prefill: resident
rate ~200 t/s (again a 2.10bpw-derived estimate) plus 0.5 ms/token staging: **~150–190 t/s against 29**.
First-32-token hit rate of 76% costs ~6 extra ms per token for those tokens.

### Risks

- **256-expert models hit lower.** 81–85% at 40% against gemma's 90%; size their slots nearer
  50% and give them a larger promotion budget.
- **Pin limits on WSL.** Strata reports the driver pins only about 1 GB under WSL. 12.2 GiB in 30
  blocks passes on this host; other WSL setups may differ, so phase 1 should fail clearly if
  pinning fails.
- **Reference quality.** A cache changes when weights arrive, never which weights. Output must be
  bit-identical to resident; that is a gate, not an assumption.
- **Decode under load.** Measured in replay: 8 streams lose 13–17 points at 40% capacity. Real
  serving has not been measured.
- **Copy interference.** Copies overlapping a replay may cost more than the model assumes;
  Strata's `ADAPT_OVERLAP` "helped on some cards and hurt on others", and the RFC measured 0.94x
  from staging during attention on GB10.
- **Eval noise.** Hit rates here are from 41 requests.

---

## Plan

Each phase is independently shippable and has a gate that can fail it.

**Phase 0: spikes, out of tree.** Each is a yes/no that could change the design.
(a) Pin 12 GiB in 30 blocks on WSL. (b) Flip a table entry between replays of a captured graph and
confirm the replay follows it. (c) Cost of the captured `topk_ids` copy. (d) Side-stream copy
overlapping a replay, on this card. (e) Confirm the bare-Linux OOM on a Linux host.
(f) Trace one Qwen model. *All done and passed, on both hosts; see
[Phase 0 results](#phase-0-results).*

**Phase 1: load-time host blocks.** *Done, all gates pass; see [Phase 1 results](#phase-1-results).*
Replace `register_offload`; residents chosen statically.
*Gate:* a bigger-than-VRAM checkpoint loads on bare Linux; `reserved - allocated` at the no-offload
residual under both allocators; the 4.10bpw spread and hot comparison re-run clean, closing the table
above. This phase alone is worth having and unblocks the rest.

**Phase 2: prefill staging.** *Done, gates pass; see [Phase 2 results](#phase-2-results). Prefill is
now bound by the MoE kernel.* Top priority as an outcome; needs phase 1's host blocks.
*Gate:* TTFT overhead scales with steps not tokens, measured as `first_token_ts - scheduled_ts`;
at least 50% of resident rate at 4.10bpw.

**Phase 3: slots and the controller.** *Built; gates pass on `cuda`; on the 3060 hit rate and identity pass, speed is 1.43x, not 1.7x; see
[Phase 3 results](#phase-3-results).* *Gate:* graphs stay on; hit rate at least 85% at 40% slots on
a fresh workload; at least 1.7x on the 19.3 t/s baseline; bit-identical output to resident.

**Phase 4: sizing and accounting.** `--cpu-offload-gb` retained as the budget, spread computed rather
than hand-written, reachable-versus-offloaded reporting, `TP>1` semantics, and the bench axis the
TODO asks for. Slots sized so a user is not asked to reason about any of this.

Grown since first planned. In order, each one feeding the next:

- **4a, mixed steps.** Record decode routing in every step, including those that carry a
  prefill chunk, and let the promotion budget grow with decode rows. *Gate:* under staggered
  arrivals, the fraction of steps the controller learns from equals the fraction with decodes.
  Output stays identical.
- **4b, concurrency against slots,** on `cuda` (below). *Gate:* a table, and a rule for slots from
  expected concurrency, or the finding that none holds.
- **4c, sizing.** Experts offloaded, slots and placement are computed from `--cpu-offload-gb` and
  what KV needs at `--max-model-len` × `--max-num-seqs`. Phase 3 found cold set plus small LRU as
  good as LRU over everything, so the simpler of the two wins. Allocated before KV sizing.
  *Gate:* gemma 4.10bpw on the 3060 and gemma 2.10bpw on `cuda` within 5% of the best hand-tuned
  configuration, from flags alone.
- **4d, accounting.** Vision first and reported apart, default selectors, `TP>1` semantics,
  reachable-against-offloaded reporting. *Gate:* a multimodal serve with the tower offloaded
  reports both parts, and experts get exactly what the tower left.
- **4e, vision cost.** Per-image prefill cost with the tower over UVA, and the pin rounding.
  Staging the tower if the cost warrants it.

**4a results.** Decode routing is now recorded in every step: `apply` copies the leading rows of
any batch, and the controller reads as many as the step had decodes. Staged steps teach the LRU
but count toward no hit rate. Under 4 staggered streams, every step with decodes was learned from
(500 of 500 per window). Output is still identical to resident. Staging itself was rare there,
because these chat prompts are under 32 tokens.

**4b results,** `cuda`, gemma 2.10bpw, every expert offloaded, `concurrency_bench.py`. Unrelated
requests ran 100 s per level, staggered 2 s apart, and every request ran to its 600-token cap. Each
cell is aggregate decode t/s, the mean VRAM hit over that level's stats windows, and median t/s
per request:

| server | n=1 | n=2 | n=4 | n=8 | n=16 |
|---|---|---|---|---|---|
| resident | 123 · — · 127 | 196 · — · 101 | 235 · — · 61 | 379 · — · 52 | 419 · — · 30 |
| no slots | 65 · — · 64 | 76 · — · 38 | 80 · — · 20 | 79 · — · 11 | 72 · — · 5 |
| 25%, budget 32 | 103 · 75% · 103 | 135 · 65% · 68 | 157 · 61% · 41 | 173 · 57% · 24 | 165 · 56% · 12 |
| 25%, scaled | 102 · 75% | 128 · 63% | 129 · 51% | 130 · 43% | 142 · 49% |
| 40%, budget 32 | 115 · 90% · 116 | 165 · 84% · 84 | 193 · 78% · 50 | 248 · 76% · 35 | 216 · 74% · 18 |
| 40%, scaled | 116 · 90% | 162 · 83% | 171 · 74% | 164 · 59% | 175 · 63% |
| 55%, budget 32 | 121 · 97% · 121 | 181 · 94% · 94 | 213 · 90% · 55 | 317 · 88% · 43 | 276 · 86% · 24 |
| 55%, scaled | 121 · 97% | 185 · 94% | 206 · 89% | 250 · 82% | 179 · 74% |
| 70%, budget 32 | 123 · 99% · 124 | 194 · 98% · 98 | 226 · 97% · 58 | 339 · 96% · 47 | 343 · 94% · 26 |
| 70%, scaled | 123 · 99% | 194 · 98% | 225 · 97% | 331 · 95% | 298 · 91% |

- **Without slots, aggregate is pinned at the link,** ~80 t/s whatever n.
- **Hit rate degrades gently with concurrency,** as the replay predicted. At 40%, 8 streams lose
  14 points, inside the replay's 13–17.
- **A budget scaled with decode rows (32×n) is worse at every fraction and every n > 1.** More
  promotions per step evict what other streams still need. The fixed budget of 32 stays, and the
  scaling option is removed.
- **The best slot fraction moves consistently with concurrency.** As a share of resident aggregate
  at budget 32:

  | slots | n=1 | n=2 | n=4 | n=8 | n=16 |
  |---|---|---|---|---|---|
  | 25% | 83% | 69% | 67% | 46% | 39% |
  | 40% | 94% | 85% | 82% | 65% | 51% |
  | 55% | 98% | 92% | 91% | 84% | 66% |
  | 70% | 100% | 99% | 96% | 90% | 82% |

  Staying within ~10% of resident takes ~40% slots at n=1, ~55% at n=2–4, and ~70% at n=8. n=16
  wants more than 70%. As a rule, **slots ≈ 0.4 + 0.1·log2(n) of experts**, capped at what VRAM
  allows. More streams also want more KV, so 4c has to split the same leftover VRAM between the
  two, against the expected n.
- **Untested:** a 256-expert model, which hit lower in replay; and long prompts, which make steps
  stage.

**4c results: autosizing.** The cache is on by default whenever experts are offloaded.

**The sizing contract** (stated 2026-11-08; the code follows it from then):

- The cache takes the minimum it needs to operate.
- A `--max-model-len` the user set comes first: it says how much context their application needs.
  The reserve is `EXL3_KV_RESERVE` (default 1.0) such requests, not `--max-num-seqs` of them; the
  multiplier is the concurrency the user wants to plan for. The cache takes the rest.
- `--max-model-len auto` licenses the cache to favour speed: slots grow up to the share that held
  decode within ~10% of resident for that concurrency (4b's rule, `target_share`), the rest goes to
  context, and at least 4096 tokens of context are always kept. Until 2026-11-08, auto let slots take
  everything past those 4096 tokens: on a 16 GiB card, Qwen3.5-35B-A3B 2.00bpw now keeps 103 slots
  (40%) and 6.91 GiB of KV, so its full 262,144-token length fits.
- `--cpu-offload-gb` caps the host memory experts take (below).

*Changed 2026-11-08: `--cpu-offload-gb` now caps what is hosted.* As first built, placement hosted every
matching expert whatever the flag said, and the flag set a floor on VRAM slots ("experts give up at most this
much VRAM"). That is not what vLLM documents ("GiB to offload to CPU"), and on WSL it pinned past the WDDM
budget for GPU-visible system memory (about half of physical RAM, across all processes): Qwen3.6-35B-A3B 4.00bpw
on the 3060 pinned 14.2 GiB on a 31.9 GB host whatever the flag, and a smaller flag only starved KV (114 slots,
0.04 GiB of KV at `--cpu-offload-gb 8`). Now each layer hosts the same share of its experts up to the flag,
after what the towers spent; the rest are resident, and slots cache only the hosted ones. `--cpu-offload-gb 99`
still means everything. The bullets below describe the first version where they mention the flag.

- **Placement.** Every expert matching the selectors goes to the host. If host RAM is short, each
  layer gets the same share (`MemAvailable` less the larger of 4 GiB and 10% of RAM, split
  across local ranks). Hosted experts do not spend vLLM's budget. With no selector, dense EXL3
  tensors stay on the device.
- **Sizing, at vLLM's `determine_available_memory`.** This is after the memory profile, before
  KV and graph capture:
  - KV keeps `EXL3_KV_RESERVE` (1.0) times what one `--max-model-len` request needs, from
    vLLM's own `max_memory_usage_bytes`. With `--max-model-len auto`, the reserve covers 4096
    tokens and vLLM fits the length to what is left.
  - Slots take the rest, up to every hosted expert.
  - `--cpu-offload-gb` caps how much VRAM experts give up, so a smaller value keeps more
    experts in VRAM even past the reserve.

  The arithmetic is `size_slots`, unit-tested.
- **Warm start.** Every slot is filled before graph capture.
- **Holding everything.** When the slots hold every hosted expert, there is no controller and no
  routing capture. Prefill reads the slots too, and the staging window returns to KV.
- **The log line** gives slots, the share of experts in VRAM, the KV kept and the reserve's
  basis. It also says how many concurrent streams 4b's rule says that share serves within ~10%
  of resident.
- **Modes:** `EXL3_EXPERT_SLOTS=0` is the phase 2 behaviour; a number fixes the count.

From flags alone (`--cpu-offload-gb 99 --cpu-offload-params experts`):

| host | model | reserve | slots | KV | decode | against |
|---|---|---|---|---|---|---|
| 3060 | gemma 4.10bpw | 1 x 3000 | 78/layer, 61% in VRAM | 0.66 GiB | **33.7 t/s** | hand-tuned best 32.1 |
| 3060 | gemma 4.10bpw | 3 x 3000 | 64/layer, 50% | 9,118 tokens | 31.5 | 32.1 |
| `cuda` | gemma 2.10bpw | 1 x 4096 | all 128 | 5.84 GiB | **126.5**, prefill 671 | resident 126.5, 671 |
| `cuda` | gemma 2.10bpw | 12 x 4096 | 76/layer, 59% | 49,152 tokens, exactly 12 x | 119.3 | resident 126.5 |
| `cuda` | gemma 2.10bpw | 20 x 4096 | none fit; warned | 70K tokens | 62.9 | no slots, 66.9 |

Output matches resident, or phase 2's greedy on the 3060, in every row. With warm start, the
59% run's first 500-step window already hit 95%.

**On WSL, `--gpu-memory-utilization` must leave room for the Windows desktop.** The same
autosized 3060 configuration decoded at 25.3 t/s while the desktop held 1.2 GiB, and at 33.7 with
0.57 GiB. At 0.90 of 12 GiB, 1.2 GiB more overcommits the card and WDDM pages. Autosizing does not
add to this: vLLM always fills to the utilization, with KV or with slots. But it is the first
thing to check when a WSL number looks low.

**Gate:** pass. The 3060 lands 5% above the hand-tuned best, and `cuda` matches resident when
everything fits.

**4d results: accounting.** `TP>1` closed 2026-11-09 on the 2x5060 Ti host, Qwen3.6-35B-A3B 4.00bpw at TP=2.
`--cpu-offload-gb` is per GPU, as vLLM documents it: at 4, each rank hosts 4.00 of its 7.50 GiB shard
(136 of 256 experts per layer) and sizes its own slots; the ranks promote in lockstep, since routing is
replicated, and reach ~92% VRAM hit. Eager output is token-identical to resident. Single-stream decode:
resident 80-83 t/s; 52 slots (67% in VRAM) 76-77, ~94%; every hosted expert slotted 82, ~100%.

- **Default selectors.** `--cpu-offload-gb` without `--cpu-offload-params` now means towers plus
  experts. `plugin._patch_offload_defaults` wraps `UVAOffloader.wrap_modules`: with no selector,
  tower calls (named by `_mark_tower_model`) go through, and the prefix-less decoder-layer call
  from `make_layers` is skipped. vLLM would otherwise offload every norm and router weight, each
  read by every token. Dense EXL3 tensors need an explicit selector in either cache mode. Any
  explicit selector restores vLLM's behaviour.
- **Vision first, reported apart.** Towers are built before the language model, so they keep first
  claim. The summary now reads "5.80 GiB of EXL3 weights … after 1.07 GiB of non-EXL3 weights
  vLLM offloaded first". When the tower takes the whole budget, it warns that it did, instead of
  reporting that nothing matched.
- **Experts get what the tower left.** On `cuda`, multimodal gemma 2.10bpw ran with
  `--cpu-offload-gb 3` and `EXL3_KV_RESERVE=12`. The experts may free 3 − 1.07 = 1.93 GiB, so slots
  were forced to 85 per layer, exactly as computed. The log warned that KV (7.25 GiB) fell short
  of its 10.35 GiB reserve. Output was identical to the multimodal resident run in every
  multimodal configuration, and the test image was described correctly.

**4e results: the vision tower's cost.** The tower (1.07 GiB, two parts) costs nothing measurable
per image when read over UVA.

| host | tower | image cost (896 px, 279 tokens) | slots | decode |
|---|---|---|---|---|
| `cuda`, 2.10bpw | resident | 389 ms | all fit | 110.0 t/s |
| `cuda`, 2.10bpw | offloaded | 391 ms | all fit | 109.4 |
| 3060, 4.10bpw | in VRAM | 5208 ms | 62/layer | 18.4 |
| 3060, 4.10bpw | **offloaded** | 4584 ms | **74/layer** | **25.0** |

On the 3060, the image's cost is its 279 tokens' prefill through the mcg expert kernel (~70 t/s,
phase 2), not the tower. Offloading the tower costs nothing and buys 12 slots per layer, **+36%
decode**. Vision first is right, and staging the tower is not worth building.

- **Pin rounding on the tower.** Still unmeasured: RSS does not track driver-pinned memory
  reliably here.
- **Multimodal gemma** needs `--max-num-batched-tokens` of at least 2496 (its largest image). The
  bigger activations cost about 4 slots per layer against the text-only runs.

*Slots against concurrency, measured served.* The replay says 8 unrelated streams lose 13–17
points at 40% capacity. Serving has not tested that, and sizing needs it: if the best slot
fraction moves consistently with expected concurrency, `--max-num-seqs` or a stated target
concurrency can choose it.

- **Where:** `cuda`, gemma 2.10bpw, every expert offloaded. There is room for KV and for slots up
  to ~70%. A 256-expert model after that.
- **Grid:** slots {25, 40, 55, 70}% × concurrent unrelated requests {1, 2, 4, 8, 16}, with
  `decode_bench.py`-style prompts staggered so prefills land mid-decode.
- **Record:** windowed VRAM hit, aggregate and per-request decode t/s, and the fraction of steps
  that skip routing.
- **Promotion budget:** run it at 32 and at 32×n, since the replay grew it with streams. *Run;
  results above.*
- **Read off:** the knee in t/s per slot fraction at each concurrency, and whether it tracks n.

Two current behaviours will skew this, and should be fixed or at least counted first:

- **A step over 32 tokens records no routing and stages.** Under continuous batching, any step
  that carries a prefill chunk alongside decodes goes over 32 tokens. It takes the staged path,
  which copies every offloaded expert, and records no routing for the controller. With frequent
  arrivals, that could be most steps. One fix is to record per-layer expert counts, a fixed
  `[L, E]` buffer filled by scatter-add for any batch size, instead of ids. The other is to decide
  staging on decode rows rather than total rows.
- **The budget is fixed per step,** while the misses a step can cause grow with its rows.

*The budget has an external claimant, and it goes first.* Serving a multimodal checkpoint on a
small card will usually mean offloading the vision tower too. Its weights are plain tensors, not
EXL3, so vLLM's `UVAOffloader` handles them. That already happens first:

- `wrap_modules` runs at model construction, before any `store()`;
- it skips every EXL3 placeholder, since they read as already on CPU;
- it counts what it takes in the same `cpu_offload_bytes` that store-time placement reads.

So vision takes budget before any expert, and that ordering should stay. The tower runs once per
image while experts run every token, so a vision byte offloaded costs far less than an expert
byte. Phase 4 has to make the ordering explicit rather than accidental:

- **Plan with vision counted.** Compute expert placement from what `wrap_modules` left, not from
  `--cpu-offload-gb`.
- **Report both.** Report vision and expert bytes separately, so a budget the tower used up does
  not read as "offload did nothing".
- **Never empty selectors.** With no selectors, vLLM offloads every non-EXL3 parameter in
  construction order, including the language model's routers, norms and any unquantized
  embedding. Default to the vision tower plus experts instead.
- **Watch the pin rounding.** vLLM pins with `pin_memory=True`, so the power-of-two rounding
  from phase 1 applies to the tower: up to 2x its bytes in host RAM.
- **Measure the cost per image.** Vision prefill reads the tower over UVA once per image; whether
  that wants staging is open.

Every run so far used `--language-model-only`, so none of this is measured yet.

**Phase 5: optional, each on its own evidence.** Embedding table on host: rows known before
forward, a handful per token, the Engram access pattern on a 1.38 GiB table. Saved slot state across
restarts, for the 76% cold start. Double-buffered staging so copies hide behind compute. A CPU tail
for misses, if measured against a current-pipeline mul1 quant. K <= 8 is the limit that matters
there, since higher bitrates are the point of this work.

**Not doing:** predictive prefetch (Belady bound ~1.07x at batch 1), per-layer sync, a disk tier, KV
residency ([kv-pager.md](kv-pager.md) stands, and Strata disables its equivalent on WSL).

**Why not a disk tier** (2026-11-10). Decode depends on every expert having an address the kernel can
read at any moment, a slot or a pinned host row: that is what makes a miss a slow read rather than a stall,
and what lets decode run with no host sync inside a CUDA graph (4.9x on MoE). An expert only on disk breaks
it; fetching on demand needs the routing before each layer's MoE, which is the per-layer sync above and
gives up graphs. The one mechanism that keeps the invariant is HMM: with the open kernel modules on native
Linux a kernel can read pageable memory, including an mmap of the checkpoint, with the page cache as the
host tier. This host reports `Addressing Mode: HMM`; WSL does not have it, and WSL is where host RAM is
tightest. Left unmeasured: the fault cost against pinned reads, and whether a fault inside exllamav3's
spin-waiting kernels stalls more than its own threads. Not worth the complexity for what it would buy.

**What prompted it: hosted experts in slots are stored twice.** The cache keeps a host copy of every
hosted expert, slot or not, so it can evict; an expert left resident is stored once. Two serves on the 3060
at 0.88 with nvfp4 KV, both at 103 of 256 experts per layer in VRAM (40%): vision offloaded (0.83 GiB),
host RAM capped the quota at 224 hosted, 32 resident, 71 slots; language-only hosted all 256 with 103
slots. The second stored 32 x 40 x 60 MiB = 1.9 GiB more expert bytes in all, and its two windows were
96 MiB larger (sized to the most hosted experts in a layer), which was its whole KV deficit: 70,832 tokens
against 82,016. Equal residency is not equal use, though: placement hosts experts in ascending id order,
so the vision run's 32 resident experts are ids 224-255 whatever they are routed, where all 103 of the
other run's follow traffic. Popularity-chosen residents would save the duplicate host bytes when host RAM
binds, but popularity drifts, so that stays speculative. Windows sized to hosted minus slots (staging only
copies experts not in a slot) would return ~300 MiB here; the per-row path, which can run inside graph
capture and copies every hosted row through static tables, is what stands in the way.

**More offload can mean less KV.** Language-only, same card and settings, `--cpu-offload-gb` 12 against 14:
204 against 238 experts hosted per layer, 52 against 18 resident, 51 against 85 slots, 103 in VRAM either
way. The extra 2 GiB of host RAM bought no residency (the extra hosted experts all ended up in slots,
stored twice) and the windows grew by 102 MiB, which was the KV lost: 100,656 tokens against 82,016. What it
can buy is adaptivity, 85 of the 103 following traffic instead of 51, and it does. Same model and settings
(0.88, nvfp4 KV, 4096-token chunks, `--max-model-len auto`), one stream, graphs on, two greedy prompts of
512 tokens; VRAM hit rate from `EXL3_CACHE_STATS` once warm:

| `--cpu-offload-gb` | slots / residents | VRAM hit (prose, code) | decode (prose, code) | context |
|---|---|---|---|---|
| 10 | 17 / 86 | ~70%, ~59% | 42.6, 36.5 t/s | 134,208 |
| 12 | 51 / 52 | ~89%, ~78% | 49.1, 43.2 | 119,296 |
| 14 | 85 / 18 | ~93%, ~86% | 51.3, 44.2 | 96,928 |
| 15 | 103 / 0 | ~94%, ~88% | 52.0, 46.8 | 89,472 |

All four hold 103 experts per layer in VRAM. Turning residents into slots lifts the hit rate from ~65% to
~91% and decode by 22-28%, with little past 85 slots; at 10 GiB the promotion budget is saturated (2048 per
64 steps), the few slots thrashing. Residents chosen by id are close to dead weight. So the larger offers
trade context for decode, but the context they give up pays only for duplicate copies and larger windows.

**An exclusive cache removes the trade** (the default since 2026-11-10; `EXL3_CACHE_EXCLUSIVE=0` restores
the inclusive one). Every expert would have exactly one home, a slot or a host
row; a promotion swaps a hot host expert with a cold slot through a scratch slot, writing the cold one
back to the row the hot one vacated, each entry always pointing at a complete copy. Host rows needed per
layer would be experts minus slots (153 here, ~9 GiB), windows would stop growing with the offer, all of
VRAM would follow traffic, and offering more than it needs would simply go unused. The costs: a
device-to-host write per eviction (free on two copy engines, serialized with everything else on one), host
rows that move (so staged tables become dynamic), and a single-copy moment in every swap, so a checksum
detector comes first. Between the extremes, a partly duplicated cache could keep host copies only for
experts judged dynamic, so that they keep cheap promotions.

Built as `ExclusiveCache`, with the detector (`EXL3_CACHE_VERIFY=1`: checksums at build, every moved expert's
decode and staged entries and bytes checked after each step boundary) tested first and seen to fire on a
write-back to the wrong row and on a slot overwritten before its write-back. Results:

- **Identity:** Qwen3.6-35B-A3B 4.00bpw, 40 fixed slots, decode plus a 6K-token prefill after swaps: token
  ids identical to inclusive in eager and graph mode, and at TP=2 in eager; the detector silent throughout.
- **Same VRAM, more of it adaptive:** with 86 residents and 40 slots, VRAM hit 94-96% against 90-91%.
- **The offload sweep again** (godel, same settings as the table above): the VRAM hit rate is now the same
  at every offer, ~94% / ~88%, and decode matches the 15 GiB inclusive row at 10 GiB (51.6 / 45.9 t/s;
  a first 10 GiB run at 44.4 / 38.5 did not reproduce, with identical cache stats, and is taken as
  interference on the shared desktop).
- **Refusing the excess.** At `--max-model-len auto` the cache stops at `target_share` in VRAM, so hosting
  stops at the rest plus `EXL3_CACHE_SPARE_ROWS` (16) to swap through (`cpu_offload._exclusive_need`); 17
  spare rows held the hit rate of 103. With a fixed length the slots are left to KV sizing and the offer is
  taken as given. Godel offered 15 GiB: hosted 169 of 256 per layer (9.90 GiB, 5.10 unused), windows
  2 x 254 MiB, **137,936 tokens of context against the inclusive cache's 89,472**, decode 50.6 / 45.2
  against 52.0 / 46.8 (within the run-to-run spread there), hit rates identical.

The write-backs cost nothing measurable at these promotion rates, on one copy engine or two.

---

## A disk tier below host RAM: simulated, then a prototype (2026-11-21)

**The question.** Models like Hy4-flash (220B / 12B active, ~156 B of routed experts) need more host RAM than many
hosts have even below 3 bpw. Would a third tier, experts read from the SSD on use, be a disaster or usable? Answered by
replay before any code: `routing_trace.py` recorded Qwen3.8-Flash-Next 2.05bpw on one card (41 jobs, 10.8K decode and
38.9K prefill tokens; `ROUTING_LLM_KW` passes the offload settings it needs), and `tier_replay.py` replays it through
VRAM (the exclusive cache exactly as `routing_replay.py` validates it, lag 2, 32 promotions a token), a host RAM LRU
holding a fraction of each layer's experts (exclusive with VRAM), and disk for the rest. Its VRAM hit rates match
`routing_replay.py`'s on the same trace (70.5 / 80.9 / 87.5% against 70.5 / 80.9 / 87.6% at 20 / 30 / 40%).

**Decode: usable.** Routing recency is strong enough that a small RAM tier catches nearly everything above it. One
expert is 1.19 MiB (all three projections, 2.05 bpw); 480 expert reads a token:

| VRAM / RAM / disk share | disk reads / token | added @ 2 GB/s | @ 6 GB/s |
|---|---|---|---|
| 30 / 60 / 10 | 2.3 | +1.4 ms | +0.5 ms |
| 30 / 40 / 30 | 9.5 | +5.9 ms | +2.0 ms |
| 30 / 30 / 40 | 15.0 | +9.3 ms | +3.1 ms |
| 30 / 20 / 50 | 22.8 | +14.2 ms | +4.7 ms |
| 40 / 20 / 40 | 15.2 | +9.5 ms | +3.2 ms |

Against ~29 ms a token for this model at TP=1, a 30/30/40 split (8.6 GiB of experts in RAM) costs ~10-25%.

**Prefill: where it costs, and the policy it needs.** A chunk touches most of a layer's experts, so the disk share
comes from disk nearly every chunk. Letting those reads into the RAM tier is wrong: a chunk flushes decode's hot set out of
RAM, and full 2K chunks then read *more* than the disk share (14.8 GiB at 30/30/40, against 11.4 GiB of experts on disk).
Streaming prefill's disk reads through the staging window without caching them (`--prefill-bypass`) leaves decode as it
was (15.2 disk reads a token) and cuts a full 2K chunk's disk reads to:

| VRAM / RAM / disk share | per 2K chunk from disk | added at 5.4 GB/s |
|---|---|---|
| 30 / 60 / 10 | 2.3 GiB | +0.45 s |
| 30 / 40 / 30 | 6.6 GiB | +1.3 s |
| 30 / 30 / 40 | 9.1 GiB | +1.8 s |
| 30 / 20 / 50 | 11.5 GiB | +2.3 s |

A 2K chunk takes ~1.37 s at today's ~1.5K t/s at TP=1. With staging overlapped against compute, as it already is from
RAM, 30/40/30 stays near today's rate and 30/30/40 becomes disk-bound at ~1.1K t/s (~640 without overlap).

**This host's SSD** (`ssd_probe.py`: random expert-sized reads, O_DIRECT, from the HF cache's shards): 1.96 GB/s at
queue depth 1 (0.64 ms a read), 5.44 GB/s at 8. Decode's reads land about one every three layers, so they are depth-1
reads, the 2 GB/s column; prefill's staging is bulk and runs deep. `/home/cache` is a VM disk on the hypervisor's
dm-cache, so O_DIRECT in the guest may still be helped by host caching: a best case for this box.

**Concurrency degrades it gently** (`--concurrent 1,2,4,8`: n requests' decode tokens a step, an expert read once per
layer per step, the promotion budget 32 x n). Disk time per token at 30% VRAM, at the depth-1 rate (2 GB/s):

| RAM (disk) share | n = 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 60% (10%) | 1.65 ms | 1.64 | 1.60 | 1.58 |
| 40% (30%) | 6.3 ms | 6.4 | 7.0 | 8.4 |
| 30% (40%) | 9.9 ms | 10.5 | 12.5 | 15.5 |
| 20% (50%) | 15.0 ms | 16.9 | 21.3 | 27.0 |

Requests share hot experts, and a batched step reads each once, so 2-4 requests cost 2-27% more disk time a token,
nothing more at 60% in RAM. The per-step reads grow (about 80 at n = 4, 30/30/40: nearly two a layer, issued together),
which moves the reads toward the deep-queue rate, so the depth-1 figures overstate higher n. No collapse: concurrency
beats serializing unless time to first token is free.

**Not settled:** whether decode disk reads could be issued ahead (routing is only known per layer), and a measured
prototype; this is a replay with modelled times. If a model justifies building it, the next step is
a measured prototype on one such split, not more simulation.

### Design note: what a disk tier changes (2026-11-21)

**The invariant it breaks.** Today every offloaded expert always has a valid address -- a VRAM slot, or its host row
read over UVA -- and "both addresses hold the same bytes, so output never depends on which one is used"
(`expert_cache.py`). That is what lets decode run inside a CUDA graph with the controller acting only between steps,
two steps behind the routing. An expert on disk has no address until it is read, and the routing that says it is
needed is known only inside the forward. Everything below follows from that.

**Backing store: the checkpoint itself.** EXL3 MoE checkpoints store each expert's projections as separate tensors, so
the shards are the disk tier: no second copy, offsets from the headers (as `_checkpoint_census` reads them). O_DIRECT
wants 4 KiB-aligned offsets and lengths, which safetensors offsets are not, so a read takes the aligned superset into a
pinned bounce buffer and copies the expert out (~1.2 MiB; the copy is cheap next to the read).

**RAM tier: host rows become a cache.** `HostBlock` rows already map expert to row, and the exclusive cache already
swaps rows and slots (`row_of`, `free_rows`). A disk tier adds one state, "on disk" (no row): eviction from RAM drops
the row (disk still holds it), a load preads into a free row. `_host_quota` becomes the RAM tier's size rather than the
set of hosted experts; every routed expert is hosted.

**Prefill: contained.** Prefill already learns its routing on the host each layer: `stage_routed` copies in only the
experts the grouped kernel's per-expert counts name. Disk-resident routed experts are read into the staging window
there, straight through, without entering RAM (the simulation's `--prefill-bypass`: caching them flushed decode's hot
set). `prefetch_next` copies only RAM-resident rows; disk-resident ones are read once the layer's counts are known, with
the next layer's reads queued deep. Prefill runs outside graphs, so nothing there changes how it is captured.

**Decode: the hard part, three ways.**
- **A. A host round trip per MoE layer.** Route, read the routing back, make every routed disk expert addressable (pread
  into a row, point its table entry at it), then run the GEMM. Simple and obviously correct, but it needs the MoE layer
  outside a captured graph: eager decode, or vLLM's breakable capture with the break at the MoE op (itself parked on the
  EXL3-with-GDN capture bug). Eager decode on Qwen4Exp ran ~8 t/s against ~22 with full decode graphs, so this mostly
  measures, and does not serve.
- **B. A wait inside the graph.** The graph copies a layer's routing to mapped host memory, then waits on a host-set
  flag with a stream memory operation (`cuStreamWaitValue32`, capturable into graphs) before the expert GEMM. A host
  service thread polls the routing, makes any routed disk expert addressable, and releases the flag at once when none
  is. The cost when nothing is on disk is a handshake per layer (microseconds; ~48 a token), and graphs stay. It is the
  most intrusive: driver memory operations through ctypes, a latency-critical host thread, the pointer table written
  mid-forward from another stream, and graph capture to validate. Probably the production design if A's numbers hold.
- **A with breakable capture: probably the way.** vLLM's breakable capture is driven by a decorator,
  `eager_break_during_capture` (`vllm/compilation/breakable_cudagraph.py`; attention, MLA and the sparse indexer use
  it): a decorated custom op ends the current graph segment, runs eagerly at replay, and a new segment starts. Wrapping
  the EXL3 MoE op in it when a disk tier is on puts A's round trip in that eager gap while everything else replays as
  graphs -- most of B's benefit with none of its in-graph machinery. Three conditions: it applies only under breakable
  capture, not FULL, which EXL3 Qwen4Exp is forced to today because breakable capture fails for EXL3 with GDN layers
  (docs/upstream.md), so this waits on that fix, which the vLLM bump needs anyway; the op must write into a
  caller-provided output (a fresh tensor changes address each replay), so the MoE apply needs adapting; and an MoE break
  per layer roughly doubles Qwen4Exp's eager gaps, perhaps a few ms a token, to measure against full graphs. B becomes the
  fallback if those breaks measure too slow.
- **C. Skip disk-resident experts.** Drop them from the token's top-k and renormalize: graph-friendly and nearly free to
  build, but output then depends on cache state -- lossy, and different from run to run. It breaks the invariant on
  purpose, and would need its own quality measurement. Not a default.

**Guard first.** Whatever the path, a routed expert must never execute from a stale or missing address: extend
`EXL3_CACHE_VERIFY` to assert that every routed expert's table entry is valid at execution time, and put the failure
back (a deliberately missed load) to see it fire before trusting any number.

**Prototype order.** (1) The RAM-as-cache state and the checkpoint reader, with prefill's bypass reads: contained in
`cpu_offload` and the exclusive cache, and enough to measure prefill against the simulation. (2) Decode path A, eager,
to measure real disk reads and latency per token against the replay's figures -- the check that the simulation holds.
(3) If both hold: A through a breakable MoE op, once the breakable-capture fix lands with the vLLM bump; B only if its
breaks cost too much.

### Step 1 built and measured (2026-11-21)

`EXL3_DISK_TIER=<share>` (exclusive cache only) keeps that share of each layer's experts in host rows and leaves the
rest in the checkpoint as meta tensors, whose null addresses make an unstaged use fault rather than compute on stale
bytes. Each disk expert has a staging-window position past the host rows; routed staging reads the routed ones
through `expert_store` (8 reads deep, into pinned sources, one copy per run of positions). Every batch on a disk-tier
layer stages by routing, the per-row path included (gemma-4's gelu_tanh has no grouped kernel), since the per-row
path's decode entries are null and `stage_all`'s window positions for disk experts are unwritten. So decode works,
eagerly; graph capture is refused, as is TP > 1 (sliced experts are not read whole). `EXL3_CACHE_VERIFY` checks that
no routed expert reaches the kernels through a null entry, and a test leaves a read out to see it fire.

**Identity** (`tools/offload/disk_tier_identity.py`): gemma-4-26B-A4B 2.54bpw, 20% slots, 40% of experts in rows and
60% on disk, against all in RAM with the same slots: bit-identical prompt logprobs and greedy tokens over prompts of
7-4429 tokens plus 16 decode tokens each. With a neighbour's bytes read instead, 6849 differences.

**QFN 2.05bpw, TP=1, VRAM 20% / RAM 30% / disk 50%** (`tools/offload/disk_tier_prefill.py --slots 0.2 --share 0.5`;
30% slots would not fit beside the windows on a 16 GiB card), HF cache NVMe:

| | measured | tier_replay.py |
|---|---|---|
| prefill, per full 2K chunk from disk | 10.78 GiB in 1.94 s (5.98 GB/s) | 11.18 GiB, +2.22 s at 5.4 GB/s |
| prefill wall | 3.81 s a chunk, 537 tok/s | |
| decode disk reads per token | 313.5 (3.42 GB/s; 114 ms) | 22.5 |

Prefill holds the simulation. Decode does not, by construction: the replay moves a disk expert into RAM when decode
uses it, and promotes from there; step 1 does neither, so every routed disk expert is read again each token. That
caching, disk to host row on a decode read and promotion out of it, is step 2's first job, with these 22.5 as the
target.

**Decode admissions, and where a token's time goes (2026-11-21).** A decode read of a disk expert now moves it into a
host row (the least recently used row expert goes back to disk-only; the fill waits on an event for queued copies
of the row). Prefetch had been copying the next layer's whole RAM tier on every decode batch the disk dispatch sent
through the grouped path (~0.5 GiB a layer); it is now for prefill-sized batches only. Decode, ms a token, 20%
slots, `tools/offload/disk_tier_prefill.py`:

| | gemma-4-26B-A4B (30 MoE layers) | QFN 2.05bpw (48) |
|---|---|---|
| graphs, all in RAM | 17.4 | 30.4 |
| eager, all in RAM | 25.9 | 53.1 |
| eager, disk-tier dispatch, nothing on disk (`EXL3_STAGE_ROUTED_ALWAYS=1`) | 35.0 | 74.0 |
| eager, disk tier (gemma 40% / QFN 30% in RAM) | 56-67 | 110-128 |

The disk-tier dispatch alone, readback and routed staging every layer, costs 0.3-0.45 ms a MoE layer over eager;
the reads (staging reads 28-41 ms on QFN, plus row fills) most of the rest. QFN's warm decode reads 76.8 experts a
token from disk (52.1 staged, 24.6 row fills) against the replay's 22.5, after only 256 decode tokens of warming.

**Path A through breakable capture: the breaks are nearly free (gemma, 2026-11-21).** `EXL3_MOE_EAGER_BREAK=1` routes
the MoE through `exl3_moe_mm_out`, which writes into the caller's buffer and is wrapped in vLLM's
`eager_break_during_capture`; under `VLLM_USE_BREAKABLE_CUDAGRAPH=1` with a PIECEWISE decode (FULL skips the breaks),
each MoE op ends its graph segment and runs eagerly at replay, where it can read routing back -- so the disk tier runs
unchanged, its capture guard never tripping. gemma-4-26B-A4B 2.54bpw, 20% slots, decode ms a token (cold / warm):

| | ms a token |
|---|---|
| PIECEWISE graphs, compiled | 17.3 / 17.7 |
| breakable capture (breaks at attention) | 17.8 / 18.9 |
| breakable, plus an eager break at each of the 30 MoE layers | 18.1 / 19.2 |
| the same, 60% of experts on disk | 41.6 / 52.5 |
| eager, 60% on disk | 55.9 / 66.8 |

Thirty extra breaks cost ~0.3 ms a token: A-breakable is the decode path, and B (an in-graph wait) is not needed on
this evidence. With the disk tier on, it saves ~14 ms a token over eager; what is left is the reads themselves.
Identity: both arms under breakable PIECEWISE graphs, disk tier against all in RAM, bit-identical over 64 decode
tokens on each of five prompts (6978 admissions, `EXL3_CACHE_VERIFY` on); with a neighbour's bytes read, 6909
differences. (Against an *eager* all-in-RAM arm decode differs everywhere: graph replay pads batches to capture sizes,
so that comparison is not an identity test.) QFN cannot be measured this way until breakable capture works for EXL3
with GDN layers (docs/upstream.md), which the vLLM bump has to fix anyway.

## Verifying the cache: eager identity, not graph identity (2026-11-09)

"Output identical to resident" holds in **eager mode**, at TP=1 and TP=2, for all-hosted and partial hosting:
every live slot matches its host row byte for byte, table entries point where they should, and the MoE
kernel gives bit-identical results reading an expert from a slot or from host memory, eagerly or replayed
from a captured graph, at every batch size tried. That is the acceptance test.

In graph mode the same comparison can differ by a few tokens in a few hundred, deterministically, without
anything being wrong. The MoE kernel is not batch-invariant (a row's output moves by ~4e-6 with the rows
beside it), so anything that changes the compiled forward's structure changes rounding:

- **Routing capture** gave the router's `topk_ids` a second consumer, and Inductor fused the router
  differently. Fixed: the capture is inside the MoE op, and every EXL3 MoE layer passes a routing buffer,
  cache or not, so cached and resident runs compile the same forward. All-hosted cache is now token-identical
  to resident in graph mode.
- **Prefill staging** compiled one copy per hosted expert row into the forward, so partial hosting (fewer
  rows) compiled a different graph. Fixed 2026-11-09 as a side effect: staging now happens inside the MoE
  op, on demand for prefill (docs/moe.md, "Prefill groups assignments by expert").
- **A residual** (Qwen3.5-35B-A3B 2.00bpw, one prompt at token 113 of 256) separates frozen slots from
  promoting ones although both read identical bytes through an identical kernel. Not in the MoE path, not KV
  capacity; most likely the runs' different allocation histories steering kernel selection elsewhere. Closed
  as numerics.

So graph-mode identity is a bonus where the compiled graph is the same, and eager identity is the test.

**Under speculative decoding, eager identity needs async scheduling off** (2026-11-10, Qwen3.6-35B-A3B
4.00bpw, MTP with 3 tokens, two greedy prompts). With async scheduling on, step timing can change the verify
batch shapes, and the kernel's batch variance then flips greedy near-ties. Promotion copies, link contention
from another offloaded process, and live autotuning all shift that timing. Cache on against cache off diverged
at tokens 45 and 100, deterministically and identically under each of those perturbations, which looked like
a promotion race. It is not one: cache off alone diverges at the same tokens between async on and off, and
the "wrong" output is the one cache off produces with async off. With `async_scheduling=False`, cache on
(alone and contended) is token-identical to cache off. Before this, the controller counted only one-token
requests as decodes, so under spec decoding it learned from almost no steps; counting each decode's
verify rows (`decode_rows`) took eager decode from 49 tok/s (frozen slots) to 88, against 20 with the cache
off, at 61% of experts in VRAM.

## Reproducing

```
python tools/offload/oversub_probe.py                  # does this host oversubscribe?
python tools/offload/link_bench.py
python tools/offload/routing_trace.py MODEL REV t      # records t.npz
python tools/offload/routing_replay.py t               # the policy and concurrency tables
python tools/offload/hot_selector.py t 66              # writes cold.txt for --cpu-offload-params
python tools/offload/measure.py                        # decode/prefill against a server on :8123
python tools/offload/graph_spike.py                    # pointer flips, ring copy, side-stream overlap
python tools/offload/pin_probe.py                      # pin 30 x 415 MiB host blocks, DMA, readback
tools/offload/phase1_check.sh REPO [ballast|main]      # placement gates: identity, speed, tight loads
python tools/offload/prefill_sweep.py [TOKENS ...]     # TTFT across prompt lengths, against :8123
python tools/offload/decode_bench.py [MAX_TOKENS]      # decode on natural long answers, against :8123
python tools/offload/concurrency_bench.py N [SECONDS]  # N staggered streams of unrelated requests
python tools/offload/concurrency_table.py OUT PHASES   # tabulate a concurrency grid's t/s and hit rates
python tools/offload/vision_bench.py [SIZE] [REPS]     # what an image costs, with and without it
EXL3_EXPERT_SLOTS=0.4 EXL3_CACHE_STATS=500 vllm serve ...  # expert cache with windowed hit rates
CODEBOOK=mcg python tools/offload/stage_bench.py 4 1024 # MoE prefill: resident, UVA, staged, copy
```

These are research scripts. The trace's prompts are gemma-formatted, and calibration text comes
from the installed exllamav3.

**Sources:** [Strata](https://github.com/Niko1221/Strata), its
[DETAILS.md](https://raw.githubusercontent.com/Niko1221/Strata/main/docs/DETAILS.md) and
[PR #38](https://github.com/Niko1221/Strata/pull/38);
[vLLM RFC #38256](https://github.com/vllm-project/vllm/issues/38256);
[llama.cpp MoE cache write-up](https://www.developersdigest.tech/blog/llama-cpp-moe-offload-n-cpu-moe-vs-moe-cache-2026);
[WiSP](https://github.com/nokia-applied-research/WiSP).
