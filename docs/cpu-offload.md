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

**That constant holds only at fixed concentration**, which every row in the original
sweep happened to share closely enough to hide the effect. Spread the same bytes more
thinly and apparent bandwidth rises to 7.63 GB/s — *above* the link — because part of
the transfer stops being serialised. See "Placement" below. So: no *bandwidth* headroom
to recover at a given placement, but real headroom in the placement itself, plus a
latency tax that excluding `suh`/`svh` collects.

Placement has three separable axes, and they do **not** behave the same way.

**Position — which layers — does not matter.** The cleanest control in the table is the
first two rows: identical concentration, layers 0–5 against layers 10–15,
**66.4 vs 66.2 tok/s**. Worth 0.3%.

**But *concentration* does matter, and it is worth ~4%.** This is a different axis, and
the expected-bytes argument above says nothing about it. What moves the number is how
many of each token's 8 routed experts land in host memory *in the layers the pattern
covers* — 8.00 when every expert in a layer is offloaded, 1.75 when only 56 of 256 are.
Sorting the 1 GiB rows by that column orders them almost perfectly, and the effect
survives re-running.

**The sparse rows beat the link, which is the tell.** `experts 200-255` and
`experts 0-59` imply **6.63–6.74 and 6.92–6.95 GB/s**, against a measured pinned-DMA
ceiling of **6.54 GB/s** — in *both* runs. Serialised transfer cannot exceed the link,
so part of it is not serialised. The mechanism is intra-layer overlap: when only ~2 of
a layer's 8 routed experts are offloaded, the other ~6 are resident and their GEMMs
issue while the offloaded reads are in flight. At 8-of-8 there is nothing left to hide
behind, and those rows sit at 6.11–6.27, just under the ceiling, like fully serialised
transfer.

**So there is one placement lever, and it is not the one the sweep was looking for:**
spread a fixed offload budget as thinly as possible across as many layers as possible.
Plain `--cpu-offload-params experts` does the opposite — it fills greedily in layer
order and concentrates at 8-of-8. Expressing the spread form requires the `re:`
extension, which is the instrument for the only placement choice that pays.

**Index — which experts — does matter, by 2.8%, and that was not expected.** Two
patterns with *identical* geometry — 34 experts, all 40 layers, 4080 tensors, 1.00 GiB,
1.07 offloaded of 8 routed per layer — differ by more than noise:

| pattern | experts | tok/s | apparent GB/s |
|---|---|---|---|
| `re:experts\.(22[2-9]\|2[3-5][0-9])\.` | 222–255 | 69.0 | 7.01 |
| `re:experts\.([0-9]\|[1-2][0-9]\|3[0-3])\.` | 0–33 | **70.9** | **7.63** |

Nothing differs but the index range. The earlier pair points the same way:
`experts 200-255` (67.7) against `experts 0-59` (68.8), low index faster *despite*
0–59 carrying the worse concentration. Two independent pairs, same direction.

**The straightforward reading is that routing is not uniform.** The cost ratio implies
experts 222–255 are routed **~8.8% more often** than experts 0–33 — roughly 13.9% vs
12.7% of routings against a uniform 13.28% share for 34 of 256. That is a mild skew,
well within what aux-loss-free (bias-based) balancing produces, and it is enough to
move throughput by 2.8%.

**Which means the uniformity premise above is false, and the conclusions that rested on
it are only as good as it was.** Expected-bytes invariance was always conditional —
"*if* routing is uniform, index is measure-preserving" — and the original sweep could
not have detected the failure, because its index ranges were confounded with
concentration. It took *controlling* concentration to expose the index effect.
Throughput is an indirect instrument for routing frequency, though; the direct
measurement is cheap and has not been done. See "Open questions".

**This also qualifies the LRU comparison.** An LRU cache with C slots *per layer* is
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

**This makes every number above host-specific.** A Gen5 x16 host is ~7.7x this link,
which changes what offload *is*: offloading all 8.21 GB of experts costs ~80% of
throughput here and a projected ~34% there, and combines with batching to near-free. Any decision
about whether offload is worth shipping should be taken on the fast host. The code is
already saturating the slow one.

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

Tracked under TODO `cpu-offload`. The substantive ones: whether routing on real
workloads is actually uniform (everything above assumes it is, and the whole placement
question reopens if not — and **the 2.8% index effect above says it is not**. The
bounded next step is one `topk_ids` histogram, not to characterise the skew but to
discriminate it from a physical cause in the offload path itself: allocation order,
pinned-page locality, NUMA placement. The first is out of scope to exploit — it needs
per-model calibration, which is WiSP's territory — while the second would be a defect
here. Either way the index window is a **confound to control in every future offload
A/B**, worth up to ~3%); how far the concentration effect goes, since 1.07 offloaded of
8 routed per layer is the sparsest measured and still improving; **why removing
`suh`/`svh` is worth 2.6% in one configuration and 12.5% in another**, for the same
bytes; what the 2026-08-20 MoE reporting shortfall
was; and what any of this looks like on a link that is not the bottleneck.
