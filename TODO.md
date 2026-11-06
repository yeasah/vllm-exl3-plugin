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

**Now:** ship Qwen3.8-27B (no YAQA, trellis embeddings, a 2.50 rung). **Next:** the TP>1 part of `cpu-offload`
(an unmerged branch), then the MoE pipeline questions it makes pressing (`recipe-value`, `exl3-1.6.4-defaults`,
`moe-tp`), and the `engram-ple` survey. Lower-priority and speculative items live in [FUTURE.md](FUTURE.md).

## `kv-budget-margin` — Make `gpu_memory_utilization` mean what it says

**Outcome:** a utilization knob that is safe to turn: a declared context the engine keeps on any attention
backend, without the operator finding the ceiling by an OOM at first inference.

**Unblocks:** reachable rather than discoverable long context on small cards. TurboQuant already reaches it
(27B 3.00 + `turboquant_4bit_nc` serves 262144 declared context at 0.985 on 16 GiB); other backends do not.

**Candidate:** (1) the general workspace hook for backends beyond TurboQuant, FlashInfer first (0.385 GiB
it does not declare, and it is the default); (2) stop making the operator retype the ceiling every startup
already prints (`free/total`): `gpu_memory_utilization=auto` from the init snapshot, or the appliance reading
NVML before launch. Out of scope: non-torch growth after the snapshot (JIT loading) and request-shaped
allocations, which are admission control. Evidence: [docs/memory-accounting.md](docs/memory-accounting.md).

## `bench-gaps` — What the bump gate still cannot see

**Outcome:** a gate that covers the paths and the axis the appliance is bounded on. Merged from
`bench-cache-control` (the cache fixture and cold, seeded runs are in) and `bench-suite` (the TP=2 tier is in).

**Unblocks:** dependency bumps decided on the gate alone.

**Candidate, cheapest first:** (1) an entry that chunks its prefill (an `Entry` field for
`max_num_batched_tokens` and a long prompt), since `_continuation_prefill` is unreachable from the gate;
(2) a VRAM profile per entry (peak and per-prefill growth beside `weight_gib`), the axis the gate is blind to;
(3) `bless` by entry name, so adding one entry does not re-bless a green tier; (4) flashinfer's autotune,
the one cache class not frozen, shown to matter or not; (5) TP=4/8 tiers on a rental, with a hub prefetch
before it starts. Evidence: [bench/README.md](bench/README.md).

## `tq-decode-slowdown` — TurboQuant decode at long context: upstream's cost or our slabbing?

**Outcome:** know whether `turboquant_4bit_nc`'s long-context decode slowdown is TurboQuant's own or
ours. In a 135k-token session on Ornith-1.5-9B it decoded at 15 tok/s against fp8's 38 and NVFP4's 36
on the same RTX 3060, falling steadily from 28 tok/s at 45k (docs/turboquant-kv.md, "Not TurboQuant").

**Unblocks:** TurboQuant as a long-context option, and whether our continuation-prefill slabbing
(`4e6e94c5a0`, `b5e5eb4be3`) owes a fix.

**Candidate:** decode tok/s at 50k/100k/135k on the stock vLLM 0.32.0 venv
(`/home/bulk/ypell/venv-vllm-stock`) against the fork, same bf16 model at TP=2 and same request. First
suspect on our side: `b5e5eb4be3` makes `_tq_full_dequant_kv`'s slab offset a `tl.constexpr`, so Triton
compiles a variant per distinct offset (the "JIT compilation during inference" warning seen in these
runs) -- a cost that grows with context, though on the prefill side.

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

## `tp-vocab-alignment` — A quantized head whose vocabulary does not split into whole blocks

**Outcome:** a quantized `lm_head` (and the tied embedding served from it) runs vocab-parallel
at any TP degree. Today a rank's slice has to be a whole number of 128-wide Hadamard blocks, so
the plugin refuses at load whenever it is not: no Qwen3-family checkpoint (151936 = 1187 blocks)
runs at TP=2, and no Llama-3 one (128256 = 1002 blocks) at TP=4.

**Unblocks:** TP for the Qwen3 family and Llama-3 at 4 GPUs, and the bench `tp` tier's Qwen3
entries (dense eager and graphs, and the only mixed-width-in-a-shard checkpoint), which stand
`known_broken` on this.

**Candidate:** pad the vocabulary to a multiple of `128 x tp` (vLLM already pads it, to 64), and
have the plugin fill the rows past the checkpoint's end with zero-valued blocks; those logits are
discarded by vLLM's existing padding mask. The alternative, an uneven split at block boundaries,
fights vLLM's equal-shard vocab layout. Evidence: [docs/tensor-parallel.md](docs/tensor-parallel.md),
*Vocab-parallel `lm_head`* (the rule, and MiniCPM5 as the aligned case).

## `recipe-value` — When is a measurement worth its cost?

**Outcome:** a rule for which models get a measured recipe and which get uniform bits (`-hq` on a
MoE), so the hours of measurement are spent only where they pay. Known so far: recipes win on dense
Qwen-family models (27B -11.5% composite, Ornith-9B, Qwen3-0.6B), lose on gemma-4 (`gemma4-recipe`),
and trade slices on Qwen3.6-35B-A3B. The models where they win scale near rfn^2 on their measurement
rows (Ornith: 1.9 at every depth; 27B: 1.45-1.9). The others don't: the MoE at 0.44-1.45, and gemma
before its chaos was explained.

**Candidates:** (1) the exponent profile as a cheap predictor: a 20-row own-voice measurement, or a few
tensors per depth band, before committing to the full one. (2) Per-tensor or per-family exponents in
`sc_optimize` instead of the pinned `-al 2.0`, tested on the 35B. The near-linear regime there (and
under loud content on the 27B) is what the quadratic law misprices.

## `exl3-1.6.4-defaults` — Three upstream defaults the 1.6.4 bump pinned instead of adopting

**Outcome:** each of exllamav3 1.6.4's changed defaults either adopted or rejected on a
measurement, and its pin in `quant.py` removed or given its reason. Pinned to pre-1.6.4 behaviour:
`convert.py --out_scales` (`always`; upstream `auto` drops output scales on gate/up projections
whose calibration statistics predict lower error without), `sc_optimize -tie` (`k_proj+v_proj,
gate_proj+up_proj`; upstream adds `q_proj` to the k/v tie and GDN `in_proj_qkv+in_proj_z`), and
`sc_measure -g` (`experts`; upstream groups dense q/k/v(/g), gate/up and GDN qkv/z as single
measurement targets too, and a MoE layer's experts as one target where ours keeps three), and
(half steps, the fourth, were measured and adopted: docs/calibration.md "Half-integer rates").

**Unblocks:** whatever each is worth on our models. Upstream chose them for its own runtime (one K
per fused kernel launch); our plugin runs mixed widths within a fused projection, so the kernel
argument does not transfer and only the quality argument is left to test.

**Candidate:** `out_scales` first, as a uniform-bits qbench A/B on the 0.6B and the 9B (it moves
every conversion, recipe or not). The recipe-side two are expected rejects -- both coarsen the
allocation, and per-type sensitivity differences are what a recipe exploits -- so they get one
quick rule-in/out arm together (`-g all` only changes keys if the ties change with it), not a
study.

## `yaqa-probe` — Decide YAQA per model before paying for its Hessians

**Outcome:** a cheap pipeline step that says whether a model is YAQA-friendly. Two-sided rounding against an
own-voice output factor is a large win on some models (MiniCPM5-2B 4.00: -18/-19% excess KL) and a poor trade
on others (Qwen3.8-27B: -3..-8% for up to 2.5 points of kept), so `quant.py` should not default to it blindly.

**Unblocks:** YAQA where it pays, and no 4x48 GB rental where it does not.

**Candidate:** an own-voice factor from 60 rows (about an eighth of the full blend's work), then one 4.00 pair,
no YAQA against the probe, judged on KL and kept against a bar around -10%. 60 rows kept 80-90% of the gain on
the 2B, and the factor reached cosine 0.974 with the full one. Possibly from an FP8 release, which might fit a
single 96 GB card: unknowns are Transformers' backward through FP8 linears and whether FP8 factors rank models
as BF16 ones do. Also open: why the 50/50 blend costs more recall than either half alone on the 27B.
Evidence: [docs/yaqa.md](docs/yaqa.md).

## `allocator-policy` — The plugin, not an import side effect, chooses the allocator under vLLM

**Outcome:** under vLLM the allocator is a deliberate, visible choice per deployment, and the
crashing combination (expandable segments + custom all-reduce, on peer-access multi-GPU boxes)
cannot occur. Today exllamav3's import turns expandable segments on behind vLLM's back.

**Unblocks:** TP serving on peer-access boxes (workstation cards, NVLink pairs, most cloud
multi-GPU) without a crash at CUDA-graph capture; memory numbers that mean the same allocator
everywhere.

**Candidate:** the plugin sets the allocator itself before importing exllamav3 and exports it in
`PYTORCH_CUDA_ALLOC_CONF`, so vLLM's own guards see the truth; when expandable segments are on
and TP > 1, it disables custom all-reduce. Expandable is the likely default where memory is
tight (fragmentation has caused OOMs before); classic plus custom all-reduce where decode latency
matters more. Open: how much NCCL all-reduce costs decode against custom all-reduce on a
peer-access box, and whether the appliance picks per box from VRAM headroom. Evidence:
[docs/tensor-parallel.md](docs/tensor-parallel.md), "Custom all-reduce and expandable segments".

## `cpu-offload` — CPU offload for EXL3 weights

**Implemented, measured, and unit-tested; not gated, not committed.**
`vllm_exl3_plugin/cpu_offload.py` registers finished EXL3 tensors with vLLM's UVA
offloader from `process_weights_after_loading`, with a `re:` extension to vLLM's
parameter selectors. Background, the cost model, backend choice and the measured
numbers are in [docs/cpu-offload.md](docs/cpu-offload.md) — this item is open work
only.

**Design these together, not one at a time.** There are now five open levers, and
each one changes the premises of the others:

- load-time pin-at-store changes where tensors live;
- prefill staging needs VRAM carved out of what offload frees, and a pointer-table
  swap;
- expert packing wants a host layout, and the same pointer-table trick;
- spread placement decides which layers every prefill step must copy;
- the backend question is now "UVA for decode, bulk copy for prefill", which neither
  vLLM backend offers.

One host layout could serve both packing and prefill staging, for example per-layer
contiguous with experts packed inside it. A pass that settles the layout, the staging
buffer and the load path at once is likely cheaper than three sequential reworks.

**Done 2026-09-17** (review pass, `tests/test_cpu_offload.py`, 15 tests, GPU-free):
scale vectors excluded by component policy rather than by selector; null-match
warning; per-layer log demoted to `debug` with a `report_once()` summary carrying
tensors as well as bytes; pointer table rebuilt only when something moved and against
a device captured before offload; matcher extracted to a pure `_matches()`. Every new
guard was verified by reintroducing the defect and watching it fail.

The summary is emitted from a wrapper around the module-level
`process_weights_after_loading` (`plugin._patch_offload_summary`), **not** from
`apply()` — that sits inside the `torch.compile` region, where Dynamo refuses
`logging.Logger` calls and kills the engine at startup (gb0291). Cost one failed
serve to find; a test now pins `apply()` free of it.

**Bench presence needs a new axis, not a new case.** Offload trades VRAM for
throughput, and the gate cannot see VRAM (`bench-suite`), so bolting it on records the
throughput loss as a regression and the capacity gain as nothing.

**Why is `suh`/`svh` worth 2.6% in one configuration and 12.5% in another?** Same
model, same host, same ~4.9 MB/token of expert scale vectors: removing them is
19.1 → 19.6 tok/s when only experts are offloaded, but 8.0 → 9.0 when dense trellis is
offloaded too. Both exceed what their bytes can pay for, which granularity explains;
the 10x gap between the two is not explained by anything yet. The policy is right
either way — they are 1.9% of expert bytes, so excluding them is never worse — but the
mechanism would say whether transfer granularity is a lever worth pulling elsewhere,
e.g. whether batching small reads or reordering them against the trellis stream is
worth anything.

**Report reachable bytes, not just offloaded ones.** `report_once()` says what moved;
it does not say what *could* have. Two accounting errors have already been caught by
hand — `--cpu-offload-gb 10` exceeding the 8.21 GB of eligible experts, and 0.936 GB of
unquantized `weight` counted as offloadable dense bytes when `register_offload` never
sees it. Both looked like physics violations until the checkpoint was consulted. A
by-category reachable-vs-offloaded line would have made each obvious at run time.

**Test the budget-exhaustion path.** `_offload_one` returns 0 once
`cpu_offload_bytes >= cpu_offload_max_bytes`, which is the mechanism behind the
eligibility cap above, and nothing covers it. Needs a fake loader; no GPU.

**Top item: prefill pays offload per token, not per step.**
Found 2026-09-21. `exl3_mgemm` runs one single-row product per (token, expert) slot, so
a P-token prefill reads offloaded experts over PCIe P times. On the Gen3 host a 15K
prefill goes 32.5 → 348 s with 3.75 GiB of experts off, which is +21 ms per prompt
token and exactly the link. On Gen5 a 15K request goes 23.4 → 76 s. That is worse than
the decode tax, and vLLM's logged prompt throughput hides it completely. **Offload is
not usable in practice for long prompts until this is fixed.** That is why it ranks
above bigger-than-VRAM loading: making offload work in more cases buys little while
every long prompt pays this. Measurements and mechanism are in [docs/cpu-offload.md](docs/cpu-offload.md).

**Candidate:** when a step's slot reads exceed one bulk copy of the layer (about 32
tokens at top-8/256), copy the layer's offloaded experts into a VRAM staging buffer and
point the kernel's pointer tables at it. Decode stays on UVA. The projection is ~5 s
instead of 315 s here, and ~0.75 s instead of ~50 s on Gen5.

Open questions:
- the staging buffer's size, and double-buffering, against the freed bytes;
- the side-stream copy under the compiled region and CUDA graphs, especially mixed
  prefill+decode batches;
- whether to copy only the active experts, which needs a `topk_ids` host sync.

**Acceptance:** TTFT measured as `first_token_ts - scheduled_ts`, never the logged
throughput line. Prefill overhead should scale with number of steps, not tokens.

**Second: offload cannot load a model bigger than VRAM.** It is the difference between working and not for a primary goal (medium MoE at 4.0bpw in
16 GiB), where packing is a 1.12-1.27x on something that already works. Registering from
`process_weights_after_loading` runs after the whole checkpoint has landed on the
accelerator, so today's offload delivers KV headroom and cannot deliver weight capacity —
the wrong way round, since 4-bit KV is cheap and body bits are not. Confirmed from a real
OOM 2026-09-18; mechanism, the three steps vLLM uses to stay bounded, and the projected
16-30 GiB band it costs are all in [docs/cpu-offload.md](docs/cpu-offload.md).

**Fix shape:** `linear.py:90` and `fused_moe.py:125` are the *only* two sites where
checkpoint tensors reach the accelerator, and all four `load_*` entry points funnel
through one `EXL3Parameter.store` — pin-and-view there instead of at finalisation.
Bounded peak, and the selector would match real checkpoint names rather than
reconstructed ones. Watch for `_interm_divisor` reading `suh` values (fine, scale vectors
stay resident) and `EXL3_DEQUANTIZE=1` needing the trellis on device. Unverified: whether
anything downstream of `store()` assumes the shard is device-resident, and how it
interacts with TP sharding, which happens in `_take_column`/`_take_row` *before* the
store.

**Acceptance check: the saving has to reach the KV budget under both allocators.** Today's
offload-after-load strands the freed trellis pages. The loss is 0.71 of 4 GiB with the
classic allocator and all 4 GiB with `expandable_segments:True` (2026-09-21, table in
[docs/cpu-offload.md](docs/cpu-offload.md)). "Model loading took" hides this, because it
counts live allocations. Check `reserved - allocated` after load, or the "weights +
non-torch" line, with expandable segments on and off. The rework should bring both to
the no-offload residual (~0.03 GiB).

**Worth trying first:** `EXL3_BLOCKQ_ON_LOAD=1` never materialises the dense embedding,
so only `bq_*` reaches the GPU — on a 248K-vocab model ~1-2 GB traded for ~0.3-0.5 GB.
Did not clear the 2026-09-18 case (that run already had it on), but it is the cheap lever
for a checkpoint whose embedding is still dense.

**Gating test passed 2026-09-17: transfer size is the cap.** Same spread selector
against `2.00bpw` (256 KiB trellis) and `3.00bpw` (uniform 384 KiB, same 30720 tensors)
of the same checkpoint, everything else fixed, both baselines measured (136.7 / 129.1):
**29.41 GB/s against 32.92**. So the cap is per-request overhead, not memory-level
parallelism in the decode kernel, and packing is worth building — but a 1.5x larger
tensor buys only 1.12x, where the microbenchmark implied ~1.20x. Fitting `t = L + S/B`
gives 2.85 µs per transfer and a 43.2 GB/s asymptote.

**Then: pack each expert's three trellis tensors contiguously — worth ~1.55x, untested.**
A correctly spread offload is already saturated, but not against the link: one
expert-projection trellis is exactly 256 KiB, and on the Gen5 host 256 KiB transfers get
22.9 GB/s on the UVA path against 49.9 for 16 MiB. Measured offload runs at 29.4 GB/s,
right where that granularity predicts, on a link that does 52. Gate/up/down are always
read together, so packing them into one 768 KiB buffer is projected at 37.4 GB/s —
**63 → ~71 tok/s** on full expert offload, or ~67 if `down` stays separate. An earlier
version of this item said ~79 on a 1.55x; that used an estimated 3bpw baseline of ~118
tok/s, and the measured 129.1 roughly halves the payoff to **1.12–1.27x**. Unlike
stacking all experts into one `[num_experts, ...]` tensor, which `_pointers` rejects for
doubling peak load memory, this needs one 768 KiB buffer at a time and the pointer table
can hold three views into it — the kernel dereferences per-tensor addresses and does not
care that they are adjacent. A distant third behind prefill and loading, **conditional on the
bpw test above**. Note also that `down` is read after the activation, not with gate/up,
so a 768 KiB buffer may behave as 512 + 256; on the measured curve that is ~1.23x rather
than ~1.55x. And bit widths differ per projection (`exl3_gate_bits` vs
`exl3_down_bits`), so the buffer size is per-group, not three equal thirds.

**Spread across all layers, and correct two claims that were wrong.** The Gen5 sweep
(2026-09-17) settled it: layer *coverage* is the placement variable, not experts per
layer. Varying experts-per-layer 7.5x at constant coverage does nothing — 29.4 ± 1.7
GB/s flat from 13% to 100% of experts, including the `everything` row. So cost is
**linear** in bytes touched, and two earlier claims in this item were wrong: that the
lever decays to zero at full offload, and that cost is superlinear. Both were inferred
from the 1 GiB greedy row, which is slow because it touches 5.3 layers, not because it
offloads every expert in them. Full expert offload on Gen5 x16 is 63.0 tok/s against a
136.7 baseline — a 54% cost, not the ~77% projected. Plain `experts` is only wrong for
*partial* offload, where greedy fill concentrates; at full offload it covers every layer
anyway. The spread expert count is `budget / (num_layers * bytes_per_expert)` and should
be computed rather than hand-written as a regex.

**Re-measure where the link is not the bottleneck.** Everything measured to date is on
a Gen3 x8 host at 6.5 GB/s, where the implementation already runs at 87–98% of the link
and no code change can recover bandwidth. The Gen5 x16 box is ~7.7x. Note
`cpu_offload_bytes` is per-process, so `--cpu-offload-gb N` under TP>1 offloads N GB
*per rank*; the 2026-09-17 table is TP=1, so its labels are honest, but a TP tier will
need the distinction spelled out.

**Spread placement is a ~7% lever and should probably be the default.** Confirmed
2026-09-17: at 1 GiB, naive `experts` (which fills greedily by layer and concentrates
at 8-of-8 routed) gives 66.4 tok/s, while 34 experts spread across all 40 layers
(1.07 of 8) gives 70.9 — **+6.8% for the same offloaded bytes**. Apparent bandwidth
rises from 6.27 to 7.63 GB/s, the latter 17% *above* the pinned-DMA ceiling, so part of
the transfer is overlapping with compute rather than serialising. Nothing measured is
sparser than 1.07/layer and the trend has not flattened, so there may be more.

The expert count that spans every layer is a function of budget and checkpoint
(`budget / (num_layers * bytes_per_expert)`), so it should be computed rather than
written by hand as a regex — the same argument as autosizing any knob against the
threshold it is checked against. Worth checking whether the effect survives
`--enforce-eager`; an overlap explanation predicts it should not change much.

**Routing is NOT uniform, and the placement question is open again.** Two patterns with
identical geometry — 34 experts, 40 layers, 4080 tensors, 1.00 GiB, same concentration —
differ by 2.8%: experts 0–33 give 70.9 tok/s against 69.0 for experts 222–255. The
earlier `experts 0-59` (68.8) vs `experts 200-255` (67.7) pair points the same way,
despite 0–59 carrying the worse concentration. The cost ratio implies experts 222–255
are routed ~8.8% more often than 0–33.

Everything in [docs/cpu-offload.md](docs/cpu-offload.md) that rests on uniform routing —
index invariance, LRU-equals-static — was explicitly conditional on a premise that now
looks false. The original sweep could not have caught it, because its index ranges were
confounded with concentration; it took controlling concentration to expose it.

**One histogram, to rule out a cause we control — not a calibration programme.** The
2.8% has two candidate mechanisms: routing skew, or something physical about which
tensors were offloaded (host allocation order, pinned-page locality, NUMA placement).
The second would be a defect in this code and fixable here; the first would not. Log
`topk_ids` once over a normal workload and count per-expert frequency: if 222–255 is
~8.8% hotter than 0–33, it is routing and the question closes; if the counts are flat,
the 2.8% belongs to the offload path and is worth chasing.

**Out of scope deliberately:** whether the skew is a workload property or a model
property, and frequency-ranked placement built on it. Exploiting it needs per-model
calibration, which is a different project — and one WiSP is already the reference for.
Note this weakens the LRU-equals-static dismissal in the doc: under skew an LRU cache
gets the frequency ranking for free, which is exactly when dynamic paging earns its
keep. 2.8% still does not pay for WiSP's per-layer host syncs and eager-only execution,
but the argument against it was made on a premise that no longer holds. Note also that
0–33 and 222–255 are arbitrary windows rather than the extremes of the distribution, so
2.8% bounds the exploitable spread from *below*.

**Control for the expert window in every future offload A/B.** This is the part that is
actionable here regardless of cause: two patterns that differ only in which expert
indices they select differ by up to ~3%. The original sweep missed the concentration
lever precisely because index and concentration moved together and partly cancelled.
Hold the window fixed, or the next comparison will misattribute the difference.

**Unexplained from 2026-08-20:** UVA's reported offload figure was inflated ~9x on a
routed MoE while honest on dense models across three quantization methods. Mechanism
never established; the obvious name-matching explanation is wrong. Worth pinning down
before relying on the claim, since MoE is where offload is most attractive.

**Reference design, before picking this back up:** exllamav3 1.6.4's CPU-MoE offload (its own
runtime, not vLLM's) -- a pinned arena for zero-copy streamed prefill (#341, ~2x long-context
prefill), hot/cold expert rankings persisted across instances, immediate promotion of missed
experts with spare GPU slots, split-K CPU GEMV with work stealing across L3 domains. Commits
`893199c1`, `c48bb5f8`, `e4a928f9`, `82cebdf1`, `21cfd60f`.

→ [docs/cpu-offload.md](docs/cpu-offload.md), [docs/media-encoders.md](docs/media-encoders.md)

## `engram-ple` — Serving Qwen4Exp (Qwen3.8-Flash-Next) with its n-gram table off the GPU

**Outcome wanted:** a 180B-class PLE/engram MoE served through the plugin on 32 GB or less, the n-gram table on host or
disk, the rest EXL3, with the MoE expert cache where it helps; and the same path ready for Qwen 4, which may require it.

**What exists (2026-11-04).** turboderp's `Qwen3.8-Flash-Next-exl3` is `Qwen4ExpForConditionalGeneration` (48 layers,
512 experts, 10 active, one PLE layer). Its table is 128 shards x 2.5M rows x 160, about 51B parameters, stored in
`1.66bpw_h4_ng3` as one K=3 trellis ring per row (31 int16 words, 62 bytes), about 20 GB, plus a per-head bias (16 heads).
The same codec as `trellis_embed.py` without the rotation; `exllamav3` decodes it with `ngram_dequant` and streams it from
disk natively. vLLM serves it as PLE: our fork (10-23) has CPU offload (pinned host, UVA); upstream main adds `disk_offload`
(rows read with `preadv` from the safetensors, OS page cache for hot rows). Both take BF16, FP8 or NVFP4 rows only, and
`Qwen4ExpPLEEmbeddingMethod` raises on any quant config other than `Fp8Config`, so an EXL3 checkpoint fails at load.
`Qwen/Qwen3.8-Flash-Next-FP8` stores the table as 52.3 GB of FP8 in 33 of its 131 files.

**Plan, cheapest first:** (1) survey Qwen4Exp's layers against the plugin (hyper-connections, `qsa`, `low_latency_gemm`,
MTP, vision): which bypass the quant config. (2) Bump the vLLM fork for `disk_offload`, checking what else moved.
(3) A hybrid: the EXL3 body plus the FP8 table from the FP8 release, the plugin handing the PLE layer to vLLM's own FP8
method through an internal `Fp8Config`, so upstream's offload applies unchanged; then judge whether the model is useful.
(4) Only then a native trellis PLE method, 20 GB against 52 of table; its bit depth (`-ngb`) is the one
question left from the retired `category-bits`.

## `tied-embeddings` — An embedding policy for tied models

**Outcome:** tied models served without the trade they make today. Merged from `quantized-embeddings` and
`quantize-embeddings-pipeline`. Untied models are settled: on-load blockq by default for dense checkpoints,
a trellis table from the pipeline (`-eb`, K=4, K=5 at 5 bpw up). A tied model is served from its head's
trellis (no extra bytes, a KLD tax: +0.0216 on gemma-4-12B), or with a second table beside the head
(`EXL3BlockQTiedEmbeddingMethod`, or `-eb`, since conversion unties).

**Unblocks:** gemma-class tied models at their best quality per byte.

**Candidate:** measure the two-table arrangement against head-served on a tied model, now that `-eb` makes it
the default; then the open idea of one tensor serving both roles (FP8 was the earlier candidate, untested).
Also open: tied plus blockq at TP>1 is unproven, and the embedding quant-config patch
(`vllm-embed-quant-config`) breaks speculative decoding as written, so it cannot be offered upstream yet.
Evidence: [docs/embeddings.md](docs/embeddings.md).

## `vision-serving` — Vision that works on a small card without per-model tuning

**Outcome:** a multimodal EXL3 checkpoint serves with images on 16 GiB with no hand-picked image cap.
Merged from `multimodal` and `encoder-offload`. Images work (gemma-4, Qwen3.6, Qwen3.8 read fine detail);
quantized vision towers are deprecated, so the encoder runs dense and offloaded, outside the plugin.

**Unblocks:** vision as a default capability rather than a per-model tuning exercise.

**Candidate:** the encoder's resident cost, not its weights: the chunking work was built and measured on a
few models, so validate it across the families in use (GLM still re-makes a `torch.cat([q, k])` copy that
`qwen2_5_vl` avoids). Autosizing the image cap against measured headroom is the fallback, built only if
chunking is shown insufficient. A minimal image-conditioned `bench/` entry is the cheap regression guard once
an image fixture is committed. Evidence: [docs/media-encoders.md](docs/media-encoders.md).

## `capability-suite` — Measuring capability through the served path

**Outcome:** task-level numbers for a configuration as it is served (quantized weights, embedding and KV),
so quality claims stop being divergence figures plus impressions.

**Unblocks:** a direct answer to whether body bits change task outcomes, which also decides how much
`cpu-offload`'s higher-rate case is worth.

**Candidate:** the `gemma-4-12B` bpw ladder (3.00 / 3.50 / 4.00, embed and head fixed) on GPQA-Diamond via
evalscope, extremes first: if 3.00 and 4.00 do not separate at 198 items, the question closes for the range.
A cheaper shakedown first: the 27B's reasoning-effort ladder (one set of weights, a known-large effect).
Settled so far: 4-bit TurboQuant KV costs nothing detectable, and agentic harnesses distort Q&A benchmarks.
Design rules and results: [docs/capability-suite.md](docs/capability-suite.md).

## Loose ends

*Small residue of retired items: a line each, done when convenient.*

- The embedding quant-config fork patch (`vllm-embed-quant-config`) needs its drafter fix before it can be offered
  upstream (`tied-embeddings`).
- `calibration-mix`: whether the conversational slice's 3% wild cost is dilution, with a share that takes nothing
  from ctx ([docs/calibration.md](docs/calibration.md)).

## Recently closed

*One line each, newest first. Prune to ~10 when appending; the cut of 2026-11-05 exceeds that for one window, so stale references still land.*

- `blockq-on-load` — closed 2026-11-05: on-load blockq is the default with an LRU-capped cache, and the pipeline emits trellis embeddings (`-eb`); see [docs/embeddings.md](docs/embeddings.md).
- `repair-tool` — closed 2026-11-05: `tools/quantize_embedding.py` repairs published checkpoints (`--sidecar`, `--trellis K`); on-load blockq makes it optional; see [docs/embeddings.md](docs/embeddings.md).
- `exl3-metadata` — closed 2026-11-05: see [docs/format-and-loading.md](docs/format-and-loading.md).
- `calibration-mix` — closed 2026-11-05: the conversational mix is the pipeline default; share tuning left as a lead; see [docs/calibration.md](docs/calibration.md).
- `trace-sampling` — closed 2026-11-05: traces sample with each model's own profile; see [docs/calibration.md](docs/calibration.md).
- `pipeline-throughput` — closed 2026-11-05: see [docs/calibration.md](docs/calibration.md).
- `card-composite` — closed 2026-11-05: cards answer which size and justify a duplicate against references, scored as served; see [docs/qbench.md](docs/qbench.md).
- `category-bits` — closed 2026-11-05: head bits on a static ladder, vision dense; the n-gram table's depth carries on under `engram-ple`; see [docs/qbench.md](docs/qbench.md).
- `upstream-queue` — closed 2026-11-05: not maintained; findings worth sending upstream live in [docs/upstream.md](docs/upstream.md).
- `recall-slice` — closed 2026-11-05: kept, confident errors and kept by band are on the card, from per-model slices that refuse thin models; see [docs/recall.md](docs/recall.md).
- `bench-cache-control` — 2026-11-05: merged into `bench-gaps`.
- `bench-suite` — 2026-11-05: merged into `bench-gaps`.
- `multimodal` — 2026-11-05: merged into `vision-serving`.
- `encoder-offload` — 2026-11-05: merged into `vision-serving`.
- `quantized-embeddings` — 2026-11-05: merged into `tied-embeddings`.
- `quantize-embeddings-pipeline` — 2026-11-05: merged into `tied-embeddings`.
- `yaqa` — 2026-11-05: refocused as `yaqa-probe`.
- `kvarn-accuracy` — 2026-11-05: moved to FUTURE.md.
- `qbench-noise-floor` — 2026-11-05: moved to FUTURE.md.
- `mtp-turboquant` — 2026-11-05: moved to FUTURE.md.
- `kv-nvfp4` — 2026-11-05: moved to FUTURE.md.
- `transformers-backend` — 2026-11-05: moved to FUTURE.md.
- `gemma4-e2b` — 2026-11-05: moved to FUTURE.md.
- `gguf-our-imatrix` — 2026-11-05: moved to FUTURE.md.
- `gemma4-recipe` — 2026-11-05: moved to FUTURE.md.
- `turboquant-sliding-window` — 2026-11-05: moved to FUTURE.md.
- `turboquant-boundary-tax` — 2026-11-05: moved to FUTURE.md.

- `moe-recipe-grouping` — closed 2026-10-26, see [docs/qbench.md](docs/qbench.md) "KL floors, flat
  scaling, and where recipes stop paying". `sc_measure -g experts` measured Qwen3.6-35B-A3B in ~5 h
  (371 targets, not a GPU-week), and the recipe loaded and converted. It did not beat uniform `-hq`:
  floor-fit +14%/-2% and measured-floor +26%/-12% (wild/swe excess). `-hq` stays the MoE default.

- `half-rate-tensors` — closed 2026-10-24, see [docs/calibration.md](docs/calibration.md) "Half-integer
  rates". The plugin serves 1.5/2.5/3.5 bpw (`tests/test_half_widths.py`); half-step recipes beat
  integer ones at equal size on both models measured (0.6B -6..-12%, 9B -4..-6%), so `quant.py` solves
  recipes with `-ks 0.5` and builds native half-rung baselines.

- `turboquant-prefill-transient` — closed 2026-09-12, see
  [docs/memory-accounting.md](docs/memory-accounting.md). TurboQuant's continuation
  prefill allocated five buffers sized by `cached_len` or `max_model_len`: 1024 MiB
  standing plus 774 MiB of growth per 100K prefill, against a 2.65 GiB KV cache. Two
  fixes. A discarded set of profiling builders was setting the workspace high-water mark
  before auto-fit ran, which no amount of correct sizing downstream could undo; and the
  cached prefix does not need to be attended in one piece at all, because every query
  sees all of it and only the current chunk is causal, so it slabs and merges by
  log-sum-exp. 1580 MiB back at the peak for ~1% of prefill throughput, agreement with
  the old path to one bf16 quantum per call, identical greedy output at 100K. The
  feared part — causal bookkeeping across chunks — did not exist; the split point is
  exactly where the mask changes character. It *does* buy context, established
  2026-09-12: full declared 262144 on the 16 GiB card at `gpu_memory_utilization=0.975`,
  against 150,528 at the old 0.86 ceiling. What it does not do is buy it *automatically* — auto-fit still
  cannot see these consumers, so the operator finds 0.975 by hand, which is
  `kv-budget-margin`'s whole job.

- `exl3-149-quality` — answered 2026-09-10, **no**, see [docs/qbench.md](docs/qbench.md)
  "Did the exllamav3 v1.4.9 bump cost quality?". Dense (Qwen3-8B, 3.0 and 4.0 bpw) is
  **bit-identical to full float precision** across the bump, which is the expected result
  once you notice upstream's kernel work was MoE-scoped and a dense model never enters
  those paths. MoE (Qwen3.5-35B-A3B) does change and by 0.011% of mean KLD — four orders
  below the value. So the 12-of-16 gate failure that raised the fork-or-not question was
  selection churn plus a max-over-63-positions statistic, and the one genuinely
  path-scoped change is numerically negligible. No fork, no revert, no pin. Method notes
  matter more than the numbers here: the MoE side needed no bf16 reference (EXL3 2.00bpw
  scored against EXL3 3.00bpw puts both arms through MGEMM), which is what kept a 70 GiB
  download unnecessary, and qbench's result cache is not keyed on engine version so the
  first attempt silently replayed a nine-day-old run.
