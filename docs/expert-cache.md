# Expert cache: the offload design for high-bpw MoEs on small cards

*Research and design pass, 2026-11-01, on the low-end target: RTX 3060 12 GiB (Gen4 x16)
shared with a desktop, WSL2, i5-12400F, 23 GiB RAM. All served measurements are
`turboderp/gemma-4-26B-A4B-it-exl3`, 128 experts, top-8, 30 layers, TP=1, graphs on.
Tools are in [tools/offload/](../tools/offload/). Background and older measurements are in
[cpu-offload.md](cpu-offload.md); where the two disagree, this note is newer.*

**Status: designed, not built.** The recommendation is one mechanism that replaces
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

`link_bench.py`, best of three. `sum()` itself saturates near 18.6 GB/s here, so the UVA
column at large sizes is partly the benchmark's ceiling.

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
| **LRU, lag 2, 32 promotions/token** | 18% | 64% | 82% | **90%** | 95% | 97% | 99% |
| LRU, lag 2, 16 promotions/token | 9% | 39% | 70% | 86% | 93% | 96% | 98% |
| LRU, lag 2, 6 promotions/token | 3% | 11% | 27% | 58% | 81% | 90% | 96% |
| same (32/token), first 32 tokens of a request | 13% | 43% | 63% | 76% | 84% | 89% | 95% |

Figures for the between-request row come from an earlier 27-job trace at matching scale; the
rest are from the 41-job trace.

- **The structure is mostly temporal.** Token-to-token overlap of the routed set is 45.7%,
  against ~6% for independent draws. The hot set follows the text, so a static ranking gets
  ~63% and an LRU ~90% at the same capacity. Hottest expert median 3.3% of a layer, none
  dominant.
- **In-sample ranking flatters static placement by about 8 points at 40%**, and languages are ~13
  below the mean. Strata's PR #38 reports the same failure at 25–35%.
- **A miss does not stall.** Lag means a miss is served by UVA for that token and promoted
  for a later one, so the policy needs no host sync inside the forward pass.
- **Too small a copy budget collapses.** 6/token, the exllamav3 default budget for a
  different purpose, loses a third of the hit rate. 32 is the working number at 40% C, and
  steady-state need is ~15/token.
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

---

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

- **One model.** All routing data is gemma. 256-expert Qwen models should be traced before the
  numbers are quoted; that is an ~11 GB download not yet on this host.
- **Pin limits on WSL.** Strata reports the driver pins only about 1 GB under WSL. This host pinned
  6 GiB fine per-tensor, but 12 GiB in 30 big blocks is unproven.
- **Reference quality.** A cache changes when weights arrive, never which weights. Output must be
  bit-identical to resident; that is a gate, not an assumption.
- **Decode under load.** Replay assumed single stream and one conversation at a time. Concurrent
  requests share slots and should thrash more, though they also touch more unique experts.
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
(f) Trace one Qwen model.

**Phase 1: load-time host blocks.** Replace `register_offload`; residents chosen statically.
*Gate:* a bigger-than-VRAM checkpoint loads on bare Linux; `reserved - allocated` at the no-offload
residual under both allocators; the 4.10bpw spread and hot comparison re-run clean, closing the table
above. This phase alone is worth having and unblocks the rest.

**Phase 2: prefill staging.** Top priority as an outcome; needs phase 1's host blocks.
*Gate:* TTFT overhead scales with steps not tokens, measured as `first_token_ts - scheduled_ts`;
at least 50% of resident rate at 4.10bpw.

**Phase 3: slots and the controller.** *Gate:* graphs stay on; hit rate at least 85% at 40% slots on
a fresh workload; at least 1.7x on the 19.3 t/s baseline; bit-identical output to resident.

**Phase 4: sizing and accounting.** `--cpu-offload-gb` retained as the budget, spread computed rather
than hand-written, reachable-versus-offloaded reporting, `TP>1` semantics, and the bench axis the
TODO asks for. Slots sized so a user is not asked to reason about any of this.

**Phase 5: optional, each on its own evidence.** Embedding table on host: rows known before
forward, a handful per token, the Engram access pattern on a 1.38 GiB table. Saved slot state across
restarts, for the 76% cold start. Double-buffered staging so copies hide behind compute. A CPU tail
for misses, if measured against a current-pipeline mul1 quant. K <= 8 is the limit that matters
there, since higher bitrates are the point of this work.

**Not doing:** predictive prefetch (Belady bound ~1.07x at batch 1), per-layer sync, a disk tier, KV
residency ([kv-pager.md](kv-pager.md) stands, and Strata disables its equivalent on WSL).

---

## Reproducing

```
python tools/offload/oversub_probe.py                  # does this host oversubscribe?
python tools/offload/link_bench.py
python tools/offload/routing_trace.py 2.10bpw t        # records t.npz
python tools/offload/routing_replay.py t               # the policy tables
python tools/offload/hot_selector.py t 66              # writes cold.txt for --cpu-offload-params
```

These are research scripts: the trace hard-codes gemma paths and prompts.

**Sources:** [Strata](https://github.com/Niko1221/Strata), its
[DETAILS.md](https://raw.githubusercontent.com/Niko1221/Strata/main/docs/DETAILS.md) and
[PR #38](https://github.com/Niko1221/Strata/pull/38);
[vLLM RFC #38256](https://github.com/vllm-project/vllm/issues/38256);
[llama.cpp MoE cache write-up](https://www.developersdigest.tech/blog/llama-cpp-moe-offload-n-cpu-moe-vs-moe-cache-2026);
[WiSP](https://github.com/nokia-applied-research/WiSP).
