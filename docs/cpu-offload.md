# CPU offload

*Consolidated 2026-09-17 from TODO `cpu-offload` and the offload section of
[format-and-loading.md](format-and-loading.md), which had accumulated as parallel
investigation logs. Covers why vLLM's offloaders skip EXL3, what the registration
does about it, and what offload actually costs.*

**Status: implemented, measured, not yet gated.** `vllm_exl3_plugin/cpu_offload.py`
registers finished EXL3 tensors with vLLM's UVA offloader from
`process_weights_after_loading`. It works, it saturates the link it runs on, and it
has no tests and no bench presence. Open work is under TODO `cpu-offload`.

**What offload is for, and it is not capacity.** It trades PCIe bandwidth for bits
per weight, so it is a quality lever rather than a fallback. The ordering against
block-quantized embeddings is settled and offload loses: per byte freed, blockq costs
about a ninth of the harness noise floor in quality and nothing in throughput, while
offload costs a PCIe round trip on every token that touches an offloaded weight.
Blockq is the first thing to spend; offload buys the *next* GiB after it is
exhausted. See [qbench.md](qbench.md) for the bpw curve and
[embeddings.md](embeddings.md) for the format.

---

## Why vLLM's offloaders skip EXL3 entirely

`vllm serve --cpu-offload-gb` silently offloads nothing for EXL3. The log reports
`Total CPU offloaded parameters: 0.01` on an EXL3 checkpoint where the same model as
AWQ offloads 3.63 GiB under otherwise identical settings.

There are two independent causes, and the second is the harder one.

**Construction-time eligibility.** The UVA offloader
(`vllm/model_executor/offloader/uva.py`) decides per-module eligibility at
construction time, before checkpoint weights load, by peeking at
`next(module.parameters()).device`. Our `EXL3Parameter` placeholders are
`Parameter(data=None)` — a default empty *CPU* tensor — so every EXL3-quantized layer
reads as "already on CPU" and gets skipped outright, before any real weight exists.
The 0.01 GiB that does get offloaded is just the ordinary dense params living
directly on decoder-layer submodules outside `EXL3LinearMethod`.

**Parameter replacement.** Even if that check passed,
`process_weights_after_loading` replaces the placeholders with brand-new `Parameter`
objects (`exl3_trellis_N` / `exl3_suh_N` / `exl3_svh_N`, or `exl3_weight` on the
dequantize path) built fresh on-device. The offloader never sees these — it wrapped
modules once, at construction, before the replacement happened. AWQ does not hit this
because it preallocates its weight tensor on-device at construction and mutates it in
place, so the offloader's construction-time view stays attached to the tensor that is
still serving inference later.

Both causes hit both backends. `PrefetchOffloader` fails identically, just louder:
it selects and sizes the same empty placeholders, then asserts at onload time with
`CPU storage for linear_attn.in_proj_qkvz.trellis is not pinned!`.

**Considered and rejected: match AWQ's pattern.** Preallocating the correct final
shape at construction needs the per-tensor bit width `K` known upfront, since trellis
shape depends on it. Checkpoints since ~v0.0.2 carry a `tensor_storage` map in
`quantization_config.json` — but it is incomplete on real checkpoints
(`Muse-Glimmer-30B-exl3` omits 303 quantized modules), does not exist pre-v0.0.2, and
improving it going forward would not retroactively fix what is published.

**Chosen direction, and its known cost.** Register from
`process_weights_after_loading`, where the finished tensors already exist at final
shape: reach `get_offloader()` and do what `_maybe_offload_to_cpu` does — pin,
accelerator-view, count. Bits-agnostic, checkpoint-vintage-agnostic, no vLLM changes.
This reaches into a vLLM internal that is not public API, one more surface that can
break on a version bump — the same standing risk already taken with `exl3_mgemm`'s
call sites.

**A third cause, which was upstream's and is now fixed.** Vision towers were never
offered to either backend, because `wrap_modules()` had one call site inside
`make_layers()`. vLLM 0.29.0 added `supports_tower_offload` and a second call site, so
`--cpu-offload-gb --cpu-offload-params visual` now works — verified at 0.86 GiB.
Evidence in [media-encoders.md](media-encoders.md); remaining work under TODO
`encoder-offload`.

---

## What this design cannot do: load a model bigger than VRAM

**Peak VRAM is the full checkpoint, whatever `--cpu-offload-gb` says.** Registering from
`process_weights_after_loading` is what makes the approach simple — the tensors exist at
final shape, so nothing has to predict them — but it also runs *after* everything has
been materialised on the accelerator. `BaseModelLoader.load_model` completes
`self.load_weights(model, ...)` in full before calling `process_weights_after_loading`,
so every byte of the checkpoint has already landed before the first tensor is offloaded.

**Confirmed from a real OOM, 2026-09-18.** A 4.0bpw 36B MoE on a 16 GiB card fails
inside `load_weights`, in `EXL3Parameter.store` (`linear.py:90`,
`loaded_weight.to(self.exl3_device)`), loading the **lm_head**: 364 MiB wanted, 75 MiB
free, 15.08 GiB already committed. Not fragmentation — 21.36 MiB reserved-but-unallocated
— so `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` does not address it.
`linear.py:90` and `fused_moe.py:125` are the only two sites at which checkpoint tensors
reach the accelerator.

**vLLM's own path avoids this in three steps, and EXL3 misses all three:**

1. `wrap_modules`, at construction, moves each decoder layer's parameters to pinned host
   memory. For an ordinary quant method those parameters exist — allocated on device and
   moved off one layer at a time, so peak stays bounded. EXL3's `EXL3Parameter` is
   `data=None`, i.e. an empty **CPU** tensor, and `_maybe_offload_to_cpu` returns early
   on `device == cpu`. Being on the host is what disqualifies it.
2. Weight loading then writes *into* those host-resident parameters, so the weights never
   all sit on the accelerator. EXL3's placeholders were never offloaded, so the loader's
   tensors land on the target device and `EXL3Parameter.store` keeps them there.
3. `device_loading_context` moves CPU-resident parameters onto the device **one module at
   a time** for processing and back off in its `finally` — its own comment says the scope
   exists "for the case where cpu offloading is used". EXL3's real sub-tensors live in
   `param.shards[index]`, a plain dict attribute that `named_parameters()` never yields,
   so the context finds nothing to move.

**So what offload delivers today is KV cache headroom — which is the resource you would
never buy this way.** A checkpoint that does not fit in VRAM cannot be loaded at any
offload setting, so the one thing offload uniquely provides, *weight* capacity, is
exactly what the design cannot reach.

That ordering matters because KV is cheap and getting cheaper, while body bits are not.
Reported from practice: 4-bit TurboQuant KV costs far less capability than degrading
body weights past the knee of the bpw curve. And on a modern hybrid the memory is barely
a constraint at all — `Muse-Glimmer-30B` has 39 of 52 layers on a 2048-token sliding
window, costing **78 MiB in total regardless of context length**, so only the 13
full-attention layers bill per token: 13 KiB at fp16, 3.2 KiB at 4-bit. Two GiB of 4-bit
KV is ~639K tokens, **4.9x that model's declared 131072 limit**, or ~80K tokens across
8 concurrent sequences. The binding constraint there is `max_position_embeddings`, not
VRAM. See [turboquant-kv.md](turboquant-kv.md).

So the preamble's framing stands and should not be diluted: offload is a lever for
*bits per weight*, not a way to find KV. Spending it on KV is paying PCIe for something
a cheaper quantizer already supplies. That is a real gap for the fast-link case, where
[the measurements above](#what-offload-costs) show a large offload is affordable — 1 GiB
for 12.8% on Gen5 x16 — and where running a model that exceeds the card is exactly the
thing worth doing.

**What the limitation actually costs, projected.** Not one marginal checkpoint — the
whole 16–30 GiB band on a 16 GiB card. Using the fitted decode model (4.55 ms fixed +
resident/700 GB/s + offloaded/PCIe, both bandwidths measured), assuming ~80% of weights
in experts and 2 GiB left for KV:

| model | must offload | Gen5 x16 | Gen3 x8 |
|---|---|---|---|
| 13 GiB | none | 111 tok/s | 111 tok/s |
| 16 GiB | 2.9 GiB | **75** | 39 |
| 20 GiB | 6.9 GiB | **53** | 20 |
| 24 GiB | 10.9 GiB | **40** | 14 |
| 30 GiB | 16.9 GiB | 30 | 9 |

Every row below the first fails at load today, including the 16 GiB one that needs only
2.9 GiB moved and would serve at 75 tok/s. The constants come from a two-point fit and
assume dense and head are read in full each token; treat the table as scoping, not as a
measurement. Per-expert packing compounds with this rather than competing — at 37 GB/s
the 24 GiB row is ~45 tok/s — and the gain grows with offload size, which is a second
reason to move the offload point first.

**Even the KV headroom is only partly delivered, and under expandable segments it can
be none at all.** Offloading after load frees the offloaded trellises from the middle of
allocator segments that still hold resident tensors. An expert trellis is 256–384 KiB,
which puts it in the caching allocator's small pool, interleaved with the `suh`/`svh`
vectors the component policy keeps on device. A segment (classic) or a 2 MiB page
(expandable) that still holds one live tensor cannot go back to the driver. vLLM's
"Model loading took" figure is live allocations, so it reports the full saving. The KV
budget is computed from `cudaMemGetInfo`, which counts the stranded pages as consumed.
Measured 2026-09-21 on this host, `Qwen3.6-35B-A3B-exl3@2.00bpw-H5`, `--cpu-offload-gb 4
--cpu-offload-params experts`, allocator stats after `empty_cache` at the end of load:

| allocator | offload | allocated | reserved | stranded | device used |
|---|---|---|---|---|---|
| classic | 4 GiB | 5.06 | 5.77 | 0.71 | 6.00 |
| classic | none | 9.06 | 9.09 | 0.03 | 9.32 |
| expandable | 4 GiB | 5.05 | **9.07** | **4.02** | 9.31 |
| expandable | none | 9.05 | 9.07 | 0.02 | 9.30 |

With `expandable_segments:True` the 4 GiB offload returned nothing to the card: it paid
the PCIe cost and bought zero KV. The classic allocator returned 3.3 of the 4 GiB. A user
report the same day, on `3.00bpw-H5` (384 KiB trellis) on another host, lost 1.82 GiB of
the 4 to the same switch. It is unexplained why that was a partial loss rather than the
full 4 GiB seen here, but either way the size of the loss depends on the allocator
setting. [turboquant-kv.md](turboquant-kv.md)'s serve recipes set expandable segments, so
following them with offload hits this. The rework below removes the cause, since an
offloaded tensor then never occupies device memory. That is why this is recorded rather
than patched separately.

**The fix has an obvious shape.** All four `load_*` entry points funnel through a single
`EXL3Parameter.store`, so that is one choke point at which to pin and accelerator-view
instead of retaining the device tensor, under the same budget and selectors. Two things
would improve as a side effect: peak becomes bounded by what is in flight rather than by
the checkpoint, and the selector would match the checkpoint's **real** tensor name
instead of the one `_offload_moe` reconstructs, since the name is known at load time.

Complications to expect: `process_weights_after_loading` reads shapes, which is free on a
host-mapped tensor, but `_interm_divisor` reads `suh` *values* — harmless, since the
component policy keeps scale vectors resident anyway. The dequantize path
(`EXL3_DEQUANTIZE=1`) calls `ops.dense_weight(trellis, ...)`, which needs the trellis on
device, so it would have to pull back or opt out. And budget spend stays greedy in load
order, as it is today.

---

## The tensor name grammar is a user-facing interface

`register_offload` matches against a reconstructed name, because the MoE path stores
per-expert tensors in plain lists (`layer._exl3_gate_trellis`) that `named_parameters`
never sees. The reconstruction is **faithful to the checkpoint**, not invented — the
index for the checkpoint benchmarked below really does contain:

    model.language_model.layers.3.mlp.experts.7.gate_proj.trellis
    model.language_model.layers.3.mlp.experts.7.gate_proj.suh
    model.language_model.layers.3.mlp.experts.7.gate_proj.svh

**The tail is checkpoint-faithful; the head is vLLM's post-mapping module path.** Those
differ, and the swap is not arbitrary. HF nests a headless *backbone* inside the
wrapper (`model.language_model.layers.N`), while vLLM nests a complete causal LM —
`self.language_model = Qwen3_5ForCausalLM(...)`, which builds its own `self.model`
internally, giving `language_model.model.layers.N`. vLLM reconciles them at
`models/qwen3_5.py`'s `orig_to_new_prefix={"model.language_model.": "model."}`. Match
on the tail: the head is vLLM-internal naming that a `WeightsMapper` may rewrite on
any bump.

**The `re:` prefix exists for conjunction, not for numeric ranges.** vLLM's
`cpu_offload_params` is an `any()` over dot-delimited substrings — a pure OR — so
`{"experts"}` catches all three components including `suh`/`svh`, and
`{"experts", "trellis"}` also catches every dense trellis. "Expert trellises only" is
inexpressible in the stock mechanism. Patterns go through `re.search`, unanchored, so
`.` is a wildcard: `re:experts.1.` matches more than it looks like it does.

### Where the summary can live, and where it cannot

There is no hook for "every layer has been offered to the offloader":
`wrap_modules` runs at model *construction*, and nothing in vLLM calls back after
the `process_weights_after_loading` loop where `register_offload` does its work.

The obvious place — the first `apply()` — **fails the engine at startup.** `apply()`
runs inside vLLM's `torch.compile` region, and Dynamo refuses `logging.Logger` calls
outright:

    torch._dynamo.exc.Unsupported: logging.Logger method not supported
    for non-export cases                                        [gb0291]

Guarding on `torch.compiler.is_compiling()` would avoid the crash but never fire,
since `apply()` is only reached through the compiled path unless `--enforce-eager`
is set. So the summary hangs off a wrapper around the module-level
`process_weights_after_loading` (`plugin._patch_offload_summary`), which runs once,
after all layers, outside any traced region. `tests/test_cpu_offload.py` asserts
`apply()` stays free of it.

---

## What offload costs

*Measured 2026-09-17. `turboderp/Qwen3.5-35B-A3B-exl3` revision `2.00bpw`,
**TP=1**, on one RTX 5060 Ti. `--max-num-seqs 4`, `--language-model-only`,
`--cpu-offload-gb 10`; only the offload selectors varied between runs. 256 experts,
`num_experts_per_tok=8`, 40 layers, `moe_intermediate_size=512`, `hidden_size=2048`.*

Exact byte counts from the checkpoint's safetensors headers, not estimates:

| category | bytes | offloadable by this path? |
|---|---|---|
| expert `trellis` | 8.053 GB | yes |
| expert `suh` + `svh` | 0.157 GB | yes, but see below |
| **all experts** | **8.210 GB** | |
| dense `trellis` (+ 0.002 `suh`/`svh`) | 0.353 GB | yes |
| dense unquantized `weight` | 0.936 GB | **no** — norms, conv1d, gates |
| embed + head | 1.408 GB | no — blockq embedding and output head are excluded |
| checkpoint total | 10.909 GB | |

**Only 0.353 GB of non-expert weight is reachable**, not the full 1.290 GB of
non-embedding dense bytes. The rest is unquantized `weight` tensors that never pass
through `EXL3LinearMethod`, so `register_offload` never sees them. Getting this wrong
is what made the `everything` row look impossible in an earlier draft of this note.

**`--cpu-offload-gb 10` is more budget than there are experts to spend it on.** Only
8.210 GB is eligible under `experts`, so that row is eligibility-bound rather than
budget-bound — the same shape as the 2026-08-20 finding below, and it has to be
accounted for before the row means anything.

| budget | actually offloaded | pattern | tok/s | Δ ms/token | implied GB/s |
|---|---|---|---|---|---|
| — | — | none | 103.0 | — | — |
Budgets are GiB (`int(cpu_offload_gb * 1024**3)`). Both runs shown: the first
sweep, and a re-run under the trellis-only policy. `e/layer` is how many of each
token's 8 routed experts land in host memory, in the layers the pattern covers.

| budget | pattern | layers | e/layer | tok/s (1st) | tok/s (re-run) | apparent GB/s |
|---|---|---|---|---|---|---|
| 1 GiB | `experts` | 0–5 | 8.00 | 66.3 | 66.4 | 6.27 |
| 1 GiB | `re:[1][0-9]\.mlp\.experts\.` | 10–15 | 8.00 | 66.1 | 66.2 | 6.22 |
| 1 GiB | `re:experts\.[1-2][0-9][0-9]\.` | 0–8 | 4.88 | 66.8 | 65.8 | 6.11 |
| 1 GiB | `re:experts\.2[0-9][0-9]\.` | 0–24 | 1.75 | 68.1 | 67.7 | **6.63** |
| 1 GiB | `re:experts\.([0-9]\|[1-5][0-9])\.` | 0–22 | 1.88 | 68.7 | 68.8 | **6.95** |
| 2 GiB | `experts` | 0–10 | 8.00 | 48.3 | 48.4 | 6.13 |
| 3 GiB | `experts` | 0–16 | 8.00 | 38.1 | 38.3 | 6.14 |
| 3 GiB | `re:experts\.[1-2][0-9][0-9]` | 0–26 | 4.88 | 39.2 | 39.7 | 6.50 |
| 1 GiB | `re:experts\.(22[2-9]\|2[3-5][0-9])\.` | 0–39 | 1.07 | — | 69.0 | **7.01** |
| 1 GiB | `re:experts\.([0-9]\|[1-2][0-9]\|3[0-3])\.` | 0–39 | 1.07 | — | **70.9** | **7.63** |
| 10 GB | **8.21** | `experts` (before the policy) | 19.1 | 42.65 | 6.02 |
| 10 GB | 8.05 | `experts` (trellis-only policy) | 19.6 | 41.31 | 6.09 |
| 10 GB | 8.41 | `trellis` (re-run) | 9.0 | 101.4 | 5.94 |
| 10 GB | 8.56 | *no selector* (re-run) | 8.0 | 115.3 | 5.29 |
| 10 GB | 8.56 | *no selector* (original) | 7.9 | 116.9 | 5.22 |

**It is a bandwidth constant, and placement is noise.** Modelling cost as
`bytes_offloaded x (top_k / num_experts) / bandwidth` fits every expert row —
including the 10 GB one once the eligibility cap is applied — at **5.67–6.45 GB/s**,
against a measured 6.54 GB/s pinned-DMA ceiling on this host.

**That constant holds only at fixed layer coverage**, which every row in the original
sweep happened to share closely enough to hide the effect. Spread the same bytes across
all 40 layers and apparent bandwidth rises to 7.63 GB/s — *above* the link — because
part of the transfer stops being serialised. See "Placement" below. So: no *bandwidth*
headroom to recover at a given placement, but real headroom in the placement itself, in
transfer granularity, and in the latency tax that excluding `suh`/`svh` collects.

Placement has three separable axes. A full spread sweep on the Gen5 x16 host
(2026-09-17) separated two of them that the original sweep had confounded, and the
answer is not the one an earlier draft of this note gave.

**Position — which layers — does not matter.** Identical distribution, layers 0–5
against layers 10–15: **66.4 vs 66.2 tok/s**. Worth 0.3%.

**Layer *coverage* — how much of the forward pass the transfers are spread over — is
the variable that matters, and it is worth up to 3x.** Two configurations on the fast
host, both offloading *every* expert in the layers they touch:

| config | layers touched | tok/s | apparent GB/s |
|---|---|---|---|
| 1 GiB, plain `experts` (greedy fill) | 5.3 | 93.8 | 10.03 |
| 7.50 GiB, plain `experts` (full) | 40 | 63.0 | **29.41** |

Same experts-per-layer, 3x apart in rate. Concentrating transfers into a few layers
stalls them with nothing to overlap; spreading them across the pass gives each one a
layer's worth of surrounding compute to hide behind.

**Experts-per-layer does *not* matter, and an earlier draft of this note said it did.**
The spread sweep varies it 7.5x at constant layer coverage and nothing happens:

| offloaded | % of experts | of 8 routed/layer | tok/s | apparent GB/s |
|---|---|---|---|---|
| 1.00 GiB | 13% | 1.04 | 119.3 | 31.45 |
| 1.96 GiB | 26% | 2.08 | 104.7 | 29.42 |
| 2.93 GiB | 39% | 3.12 | 92.1 | 27.75 |
| 3.90 GiB | 52% | 4.16 | 84.7 | 29.14 |
| 4.86 GiB | 65% | 5.20 | 77.4 | 29.10 |
| 5.83 GiB | 78% | 6.24 | 71.7 | 29.50 |
| 6.80 GiB | 91% | 7.28 | 66.1 | 29.20 |
| 7.50 GiB | 100% | 8.00 | 63.0 | 29.41 |
| 7.83 GiB | 100% + dense | 8.00 | 36.1 | 29.56 |

**Flat at 29.4 ± 1.7 GB/s across the whole range**, including the `everything` row once
dense trellis is priced at full duty. So **cost is linear in bytes touched per token**,
and the earlier claim that it is superlinear — that the lever "decays to zero at full
offload" — was wrong. It was inferred from the 1 GiB greedy row, which is slow because
of *layer coverage*, not because of experts-per-layer.

**Index — which experts — does matter, by 2.8%.** Two patterns with identical geometry —
34 experts, all 40 layers, 4080 tensors, 1.00 GiB — differ by more than noise:
`experts 0-33` gives 70.9 tok/s against 69.0 for `experts 222-255` (Gen3 host). The
earlier `experts 0-59` (68.8) vs `experts 200-255` (67.7) pair points the same way.

**The straightforward reading is that routing is not uniform.** The cost ratio implies
experts 222–255 are routed **~8.8% more often** than 0–33 — roughly 13.9% vs 12.7%
against a uniform 13.28% share. Mild, consistent with aux-loss-free balancing.

**Which means the uniformity premise is false, and what rested on it is only as good as
it was.** Expected-bytes invariance was always conditional. The original sweep could not
have caught the failure, because its index ranges were confounded with layer coverage;
it took controlling coverage to expose it. Throughput is an indirect instrument for
routing frequency, and the direct measurement has not been done — see "Open questions".

**This also qualifies the LRU comparison.**
inherently spread — every layer keeps C of N resident, so offloaded-routed-per-layer is
uniform at `top_k x (1 - C/N)`. It is equivalent to a static placement *at the same
per-layer density*, which the greedy `experts` fill is not. Dynamic paging still buys
nothing over a correctly spread static placement; it does beat a concentrated one.

**Duty cycle is the one axis with real structure.** Experts are touched 3.125% of
steps, dense weights 100% — a **32x difference in cost per offloaded byte** by
construction. 0.353 GB of dense at full duty costs 351 MB/token, more than the 252 MB
that 8.05 GB of experts costs; that is the whole reason the unfiltered rows collapse to
8 tok/s. **Never offload anything densely read**, which is what the selectors exist to
enforce.

**Single-stream is the worst case.** Expected *unique* experts touched per layer per
step is `N x (1 - (1 - top_k/N)^B)`, which saturates slowly, so a transferred expert
serving one token is the floor. With all 8.21 GB of experts offloaded, PCIe cost per
token (modelled, at the measured 6.5 GB/s):

| batch | unique experts/layer | MB/token | ms/token |
|---|---|---|---|
| 1 | 8.0 | 256.6 | 39.5 |
| 4 | 30.5 | 244.8 | 37.7 |
| 8 | 57.4 | 230.2 | 35.4 |
| 32 | 163.3 | 163.7 | 25.2 |
| 128 | 251.6 | 63.0 | 9.7 |

against a 9.71 ms/token resident baseline. Offload is far more viable under
concurrency than any single-stream measurement suggests — which matters, because the
appliance case is not batch 1. Note the runs above set `--max-num-seqs 4`, but B=1 and
B=4 differ by only 5% in bytes per token, so the measurements cannot distinguish them
and nothing above depends on which it was.

**`suh`/`svh` must not be offloaded, and the cost exceeds their bytes — but by how
much depends on what else is offloaded.** Two controlled pairs, same model, same host:

| pair | what changed | tok/s | Δ | bytes would predict |
|---|---|---|---|---|
| experts only | expert `suh`/`svh` removed | 19.1 → **19.6** (+2.6%) | 1.34 ms | 0.75 ms |
| + dense trellis | all `suh`/`svh` removed | 8.0 → **9.0** (+12.5%) | 13.9 ms | 1.06 ms |

The same ~4.9 MB/token of expert scale vectors costs **ten times more** in the second
configuration than the first, and both exceed what bandwidth can explain. The
plausible mechanism is granularity: ~1920 separate `suh`/`svh` reads per token (8
experts x 40 layers x 3 projections x 2 tensors), each a few KB, each a dependent PCIe
round trip that must land before its expert's GEMM can start, where contiguous trellis
streams at link speed. **What that does not explain is why the penalty is 10x larger
when dense trellis is also in flight.** Unresolved — see "Open questions".

The policy decision is unaffected: `suh`/`svh` are 1.9% of expert bytes, so excluding
them costs almost nothing in capacity and is never worse. It is hardcoded in
`_OFFLOADABLE_COMPONENTS` rather than left to a selector.

Excluding them also makes the bandwidth model behave: the trellis-only rows fit at
5.94–6.09 GB/s, inside the 5.67–6.45 band, while the rows carrying scale vectors fall
to 5.22–5.29 — the model prices bytes and cannot see round trips.


## The host link is most of the story

*Measured 2026-09-17; see also the host notes in this session's findings.*

| path | measured |
|---|---|
| pinned DMA H2D, 256 MB | 6.54 GB/s |
| UVA zero-copy kernel read | 6.12 GB/s |
| both GPUs concurrently | 12.96 GB/s aggregate |
| GPU0 -> GPU1 | 6.33 GB/s (no P2P; `can_device_access_peer` is False) |
| host RAM streaming read | 27.9 GB/s |

6.5 GB/s is Gen3 x8. The development host runs the GPUs passed through into a
QEMU/KVM guest on a Skylake-era CPU (PCIe 3.0, 16 CPU lanes), so an x8/x8 pair is
Gen3 each and this is the platform ceiling rather than a misconfiguration.

**Do not diagnose the link from inside the guest.** Everything it reports about PCIe
is emulated and wrong in three directions at once: `current_link_speed` reads
2.5 GT/s *even under load*, `max_link_speed` reads 32 GT/s, and `nvidia-smi
pcie.link.gen.max` reads 3. Measured throughput is the only honest instrument; to
check the real link, read `LnkCap` of the *upstream bridge* on the hypervisor (the
GPU's own `LnkCap` reports the card's capability, not the negotiated minimum).

### Measured on both hosts, and a faster link makes *placement* matter more

*5070 Ti, Gen5 x16, single card, same model and budget. 2026-09-17.*

| host | baseline | 1 GiB greedy (5.3 layers) | 1 GiB spread (40 layers) | spread is worth |
|---|---|---|---|---|
| 5060 Ti, Gen3 x8 | 103.0 | 66.4 (−35.5%) | 69.0 (−33.0%) | 1.12x |
| 5070 Ti, Gen5 x16 | 136.7 | 93.8 (−31.4%) | **119.2 (−12.8%)** | **3.12x** |

Apparent bandwidth goes 6.27 → 7.01 GB/s on the slow host but **10.03 → 31.45 GB/s** on
the fast one. **The naive intuition — that a fast link makes placement matter less — is
backwards.** On Gen3 x8 the link saturates either way, so spreading recovers only a
sliver. On Gen5 x16 the link is nowhere near saturated, so what is left is stall time,
and spreading is what removes it.

**Offload cost cannot be projected by scaling link bandwidth.** An earlier draft
projected full expert offload at ~34% on a Gen5 host by scaling 6.5 GB/s to ~50, then
revised it to ~77% from the greedy 1 GiB row. Both were wrong. Measured: **63.0 tok/s
against a 136.7 baseline, a 54% cost**, because full offload covers all 40 layers and
therefore gets the spread rate, not the greedy one.

### The real ceiling is transfer size, and the format sets it

*Measured on the 5070 Ti, 2026-09-17.*

29.4 GB/s is not an arbitrary number. One EXL3 expert-projection trellis is
`(2048/16, 512/16, 16*2)` int16 = **exactly 256 KiB**, and that is the granularity every
offloaded read happens at. On that host:

| transfer size | pinned DMA | UVA kernel read |
|---|---|---|
| 256 KiB — one trellis | 26.4 GB/s | 22.9 GB/s |
| 512 KiB | 46.0 | 32.0 |
| **768 KiB — one expert, gate+up+down** | **47.2** | **35.3** |
| 1 MiB | 47.9 | 39.5 |
| 16 MiB | 51.1 | 49.9 |

The measured offload rate of 29.4 GB/s sits right where a 256–512 KiB granularity
predicts. **So a correctly spread offload is already saturated — not against the link,
which does 52 GB/s, but against what 256 KiB transfers achieve.** The link has ~1.8x
more to give and the format cannot ask for it.

**Confirmed by varying the format's own granularity.** Running the same spread selector
against `2.00bpw` and `3.00bpw` of the same checkpoint changes trellis size — 256 KiB
against a uniform 384 KiB, 30720 tensors either way — with architecture, layers, routing
and selector all fixed. Both baselines measured, no offload:

| revision | trellis | baseline | offloaded | tok/s | apparent GB/s |
|---|---|---|---|---|---|
| 2.00bpw | 256 KiB | 136.7 | 7.50 GiB (100%) | 63.0 | 29.41 |
| 3.00bpw | 384 KiB | 129.1 | 5.84 GiB (52%) | 73.0 | **32.92** |

**Transfer size is the cap, but the scaling is weaker than the microbenchmark suggests:**
a 1.5x larger tensor buys 1.12x more bandwidth, where the `sum()` curve implied ~1.20x.
Fitting `t = L + S/B` to the two real points gives a **per-transfer latency of 2.85 µs**
and an asymptote of **43.2 GB/s** — against 52 GB/s for large pinned DMA on the same
host.

**So packing is worth 1.12–1.27x, not the 1.55x an earlier draft of this note claimed.**
That figure came from extrapolating the microbenchmark and from an *estimated* 3bpw
baseline of ~118 tok/s; the measured 129.1 roughly halves the projected payoff. Using
the fit:

| packed unit | size | GB/s | vs today | full expert offload @2bpw |
|---|---|---|---|---|
| one trellis (today) | 256 KiB | 29.4 | 1.00x | 63.0 tok/s |
| gate+up only | 512 KiB | 35.0 | 1.19x | 66.8 tok/s |
| whole expert | 768 KiB | 37.4 | 1.27x | 71.2 tok/s |

If `down` does not join the bundle — it is read after the activation, not with gate/up —
the effective figure is **1.12x**, the same as simply moving to 3bpw tensors. The fit is
two points extrapolated 2x beyond their range; treat it as indicative.

**Mechanically it is still cheap.** Unlike stacking all experts into one
`[num_experts, ...]` tensor, which `_pointers` rejects for doubling peak load memory,
per-expert packing needs one buffer at a time, and the pointer table can address views
into it since the kernel dereferences per-tensor addresses and does not care that they
are adjacent. Bit widths differ per projection (`exl3_gate_bits` vs `exl3_down_bits`),
so the buffer is not three equal thirds. Untested.

**A side finding from the two baselines.** 3.00bpw is only 1.059x slower than 2.00bpw
despite reading ~16% more weight bytes per token. Solving `t = F + W/B` across the pair
puts effective VRAM bandwidth near 700 GB/s and **fixed cost at ~4.6 ms of the 7.3 ms
token**, i.e. most of decode at batch 1 is not weight reading at all. That does not
affect the offload model, which prices a marginal cost — but it does mean offload's
*percentage* cost is measured against a baseline dominated by something else.

**Every number here remains host-specific.** Any decision about whether offload is worth
shipping should be taken on the fast host; the code is already saturating the slow one.

---

## Which backend, and why MoE inverts the obvious answer

vLLM has two weight-offload backends. `UVAOffloader` (`--cpu-offload-gb`) is
zero-copy: a weight read stalls on PCIe inline. `PrefetchOffloader`
(`--offload-group-size`) issues async H2D copies ahead on a dedicated stream, so it
can hide transfer behind compute of preceding layers. Only one is active per process.

For densely-read weights prefetch is the correct backend. **For routed experts it is
decisively wrong**, because it is routing-blind: `start_onload_to_static` copies every
whitelisted parameter each forward pass regardless of which experts the router picked.
UVA's laziness is what makes a routed model pay for only what it reads.

*Measured 2026-08-20 on `Intel/Qwen3.6-35B-A3B-int2-mixed-CT-AutoRound` — deliberately
not an EXL3 checkpoint, so that both backends could be compared on one model that fits
the card either way. RTX 5070 Ti, `--max-num-seqs 1 --max-model-len 1024`.*

| config | claims | resident | freed | tok/s |
|---|---|---|---|---|
| baseline | — | 12.02 | — | 198 |
| UVA `experts`, gb=8 | 8.11 | 11.15 | 0.87 | 175 |
| prefetch `experts`, group=1 | 1.01 | 11.08 | 0.94 | 42 |
| UVA all params, gb=8 | — | 10.59 | 1.43 | 47 |

**Access pattern dominates by 4x at matched bytes.** Both `experts` rows settle at
~11.1 GiB resident — the same ~0.94 GiB off the card — and differ by 175 vs 42 tok/s.

**Two caveats from that run that still stand.** UVA's *reported* figure was inflated
~9x on the MoE (claimed 8.11, delivered 0.87; three independent metrics agree on the
real one) while being honest on dense models across awq, gptq and compressed-tensors —
mechanism never established, and worth pinning down before relying on the claim. And
"Model loading took X GiB" is `max_memory_allocated()`, a peak over allocator-managed
memory; a UVA host-mapped tensor is not allocator-managed at all, so it is not a
steady-state resident figure.

Registering from `process_weights_after_loading` should avoid the shortfall by
construction: it runs after all replacement, with the final tensors in hand, and
nothing repacks them afterwards. Unverified against EXL3 at the time of writing.

**That verdict was reached on decode alone, and prefill reverses it.** See the next
section: a prefill step reads nearly every expert, which makes it the densely-read case
this section says belongs to prefetch.

---

## Prefill pays the offload cost once per token, not once per step

*Surfaced 2026-09-21 from serving, on the Gen5 x16 host: a 15K-token prompt went from
23.4 s to 1 m 16 s of request time with 3.75 GiB of experts offloaded, while decode only
went 51 → 40 tok/s and vLLM's log showed "prefill" of ~1450 t/s both times. That logged
figure is not a prefill rate, which is why the loss looked unaccounted for. It is
prompt tokens credited at first-token time divided by the 10 s log interval, so a long
prompt always reads as roughly `prompt_len / 10`. (The mechanism is in the field notes.
Measure prefill as `first_token_ts - scheduled_ts`, or with `max_tokens=1`.)*

**Mechanism: every (token, expert) slot is its own single-row product.**
`_exl3_moe_mm` ([ops.py](../vllm_exl3_plugin/ops.py)) repeats each token's row `top_k`
times into `gathered = [tokens * top_k, 1, hidden]`, and `exl3_mgemm` takes
`size_m = A.size(1) = 1`. Tokens routed to the same expert are never grouped, so each
slot reads its expert's whole trellis. In weight traffic a P-token prefill is P decode
steps. With experts resident, those re-reads come from VRAM and L2. With UVA, every one
of them crosses PCIe, because zero-copy has no device-side copy to reuse. Nothing
amortises over the step, and chunk size cannot change it.

*Measured on the Gen3 x8 host (RTX 5060 Ti, TP=1), `Qwen3.5-35B-A3B-exl3@2.00bpw`,
`max_tokens=1` on random 15000-token prompts, prefix caching off, one warmup then two
timed repeats. Offload is `--cpu-offload-gb 3.75 --cpu-offload-params experts`, which
lands as 3.75 GiB in 15360 tensors across layers 0–19.*

| offload | max-num-batched-tokens | 15K prefill | |
|---|---|---|---|
| none | 2048 | 32.5 s | 462 t/s |
| none | 4096 | 32.6 s | 461 t/s |
| none | 8192 | 32.7 s | 459 t/s |
| 3.75 GiB | 2048 | 346.6 / 350.4 s | 43 t/s |

- **The prediction came before the number.** 3.75 GiB over 20 layers × 256 experts is
  ~786 KB per expert, so 8 routed × 20 layers is ~126 MB per token over the link.
  15K × 126 MB at the 6.1 GB/s zero-copy ceiling predicts ~310 s extra. **Measured:
  +315 s, which is 21.0 ms per prompt token, or 6.0 GB/s.** So prefill is purely
  link-bound on per-token expert reads.
- **Same per-token cost as decode, on this host.** Decode pays about +22.5 ms per token.
  That comes from 128 tokens after a 64-token prompt, less that prompt's own ~1.3 s of
  per-token reads. On the Gen5 host prefill paid about 0.6× decode per token: ~50 s
  extra over 15K tokens against 5.4 ms per decode token. The likely reason is that decode
  on a fast link is latency-bound per transfer (2.85 µs each, fitted above), while
  prefill's many concurrent slots keep more transfers in flight. That is inferred, not
  measured.
- **Chunk size is flat even with no offload** (32.5–32.7 s across 4x). That is the
  same per-slot structure: a bigger step amortises nothing. It is probably also why
  [moe.md](moe.md) measures MoE prefill at only 1.8x decode throughput. Grouping tokens by
  expert would help resident prefill too, but that is a kernel change, separate from and
  larger than the one below.

**Scale:** on a slow link this turns offload from a decode tax into a prefill wall.
10.7x on prefill here (32.5 → 348 s), and about 3x on the whole request on the Gen5 host. The chunk-size
sweep was stopped after the 2048 point with offload, because the mechanism already
predicts the other points (flat).

### Design direction: stage a layer's offloaded experts for prefill

A prefill step already touches almost every expert in a layer. So one bulk copy of the
layer's offloaded bytes into a VRAM staging buffer beats per-slot zero-copy reads once
`tokens × top_k × bytes_per_expert` exceeds the layer's offloaded bytes. That is about
**32 tokens** at 256 experts, top-8. Here that means ~4.0 GB per step (0.62 s at 6.5 GB/s,
~0.1 s at Gen5 rates). A 15K prompt at 2048-token steps would pay ~5 s instead of 315 s,
or ~0.75 s instead of ~50 s on Gen5. Decode stays on UVA, where laziness is what wins.

- **Wiring is already the right shape.** The kernel takes per-expert *pointer tables*
  (`exl3_*_trellis_ptrs`), so a prefill step can swap in a table aimed at the staging
  buffer and nothing else about the call changes. This is the same property the packing
  item relies on.
- **Neither vLLM backend does this.** UVA is lazy and never copies. `PrefetchOffloader`
  copies every forward, decode included, which is the 4x loss above, and only one backend
  is active per process. What is wanted is a per-step switch on token count, inside our
  MoE `apply`.
- **Costs to size:** one layer of offloaded experts is ~190 MB here. Double-buffering, so
  the next layer's copy overlaps this layer's compute, doubles that. The buffer can only
  be carved out of what offload freed, so it eats into the saving.
- **Refinement:** copy only the experts active in the step, from `topk_ids`. At 2048
  tokens that is nearly all of them, so it matters mostly for short prefills and mixed
  batches. It would also need a device-to-host sync to build the copy list, which the
  unconditional copy avoids.
- **Unverified:** how an H2D copy on a side stream behaves inside the piecewise-compiled
  region and under CUDA graph capture. Large prefill steps typically run above the
  capture sizes, but mixed prefill+decode batches do not. It also interacts with every
  other open offload item, which is why the TODO treats them together.

---

## Why llama.cpp's offload knobs look different

llama.cpp expresses MoE offload as tensor-name patterns like
`-ot "\.ffn_(up|down|gate)_exps\.=CPU"`, and the ranges in those recipes do real work
there. **The reason is that `=CPU` means a different operation: compute those experts
on the CPU.** Only activations cross PCIe. Weight streaming, which is what this plugin
does, moves the weights instead.

That distinction has teeth on a slow link. Batch-1 MoE decode is weight-bandwidth-
bound either way, at 257 MB/token for this model with all experts offloaded:

- stream to GPU: 257 MB / 6.5 GB/s = **39.5 ms/token**
- read in place on the CPU: 257 MB / 27.9 GB/s = **9.2 ms/token** floor

**But EXL3's format spends that headroom.** Trellis decode is deliberately ALU-heavy —
fewer bits for more decode work — and there is no CPU kernel for it. At 2bpw, 257 MB of
trellis is ~1.0G params to decode per token; on four Skylake cores with AVX2 and no VNNI
that is plausibly 25–80 ms, so decode becomes the wall well before bandwidth does.
GGUF's K-quants are shaped for cheap SIMD dequant precisely so this path works. This is
the same GGUF-vs-EXL3 tension as the embed/head tax, surfacing somewhere new: EXL3 wins
bits-per-quality assuming a GPU does the decoding, and on a host-compute path that
assumption inverts.

**So the knobs diverged because the operations did.** Numeric ranges are uninteresting
*for weight streaming under uniform routing*, where selection is sampling a measure and
every equal-sized subset is equivalent. For CPU-compute offload they partition *labor*
between two processors that overlap, so the split has a real optimum. Expect a ported
`-ot` recipe to behave nothing like it does in llama.cpp.

---

## Related work: WiSP

[nokia-applied-research/WiSP](https://github.com/nokia-applied-research/WiSP)
(Apache-2.0, arXiv 2606.21868) is a vLLM plugin that pages MoE experts from pinned
host memory under an LRU cache, sharing one VRAM budget with the KV cache. Read
2026-09-17.

**It confirms the bandwidth conclusion independently** — its own limitations section
says single-stream decode is "PCIe-bandwidth-bound, not prediction-bound", and
`ensure_resident`'s docstring concedes "there is no real overlap opportunity at this
point because the caller has already forced a host sync to read `topk_ids`".

**Its headline 2.0x is mostly a weak baseline.** The comparison is vanilla
`--cpu-offload-gb 48`, which offloads indiscriminately — the same duty-cycle effect as
the `everything` row above. The `experts` selector already captures most of it.

**Not worth adopting as code.** It targets vLLM 0.11.2 (we are on 0.29); it does a D2H
sync of `topk_ids` plus Python LRU bookkeeping per MoE layer per step, and has no
CUDA-graph handling at all — both quickstart examples pass `--enforce-eager`, where the
UVA path here is capture-compatible. It pins full expert weights in host DRAM (~80 GiB
for their BF16 model), which does not fit this host's 31 GiB; a 3bpw EXL3 model does.
Its byte-identity guarantee explicitly excludes quantized paths.

**Two ideas worth keeping.** `src/wisp/oracle/cooccur.py` asks whether routing has
*co-occurrence* structure — which experts fire together, so they can be grouped into one
transfer. That survives the uniformity argument above, because uniform marginals do not
imply independent joints. And the KV-expert dual-resize controller is a *sizing* idea
orthogonal to paging: it converts a throughput budget into a context budget. On this
model KV is cheap — `full_attention_interval=4` and `num_key_value_heads=2` give
~20 KB/token, so 1 GiB of reclaimed VRAM is ~52K tokens of context, against ~5.2 ms/token
(~35% of throughput) to evict 1 GiB of experts here, or ~7% on a Gen5 host.

---

## Open questions

Tracked under TODO `cpu-offload`.

**Is routing skewed, or is the offload path?** The 2.8% index effect says routing is not
uniform, but throughput is an indirect instrument. One `topk_ids` histogram discriminates
routing skew from a physical cause in the offload path — allocation order, pinned-page
locality, NUMA placement — and only the second would be a defect here. Exploiting real
skew needs per-model calibration, which is out of scope and is WiSP's territory. Either
way the index window is a **confound to control in every future offload A/B**, worth up
to ~3%.

**Does per-expert trellis packing deliver the ~1.55x?** The microbenchmark says a 768 KiB
read beats three 256 KiB ones by that much on the UVA path, and offload is measurably
saturated at the 256 KiB rate. Untested in the real kernel, where parallelism and
overlap differ from a `sum()`.

**Why is removing `suh`/`svh` worth 2.6% in one configuration and 12.5% in another**, for
the same bytes.

**Does prefill staging hold up under compile and CUDA graphs, and what does its buffer
cost in KV?** The mechanism and the ~60x projection are in the prefill section; the
integration questions are not answered.

**What the 2026-08-20 MoE reporting shortfall was** — UVA's claim inflated ~9x on a
routed MoE while honest on dense models.
