# Media encoders: how big, and what it takes to evict them

*Census measured 2026-08-24; eviction measured 2026-09-12. Tracked in TODO as
`encoder-offload`.*

A vision or audio encoder is the one weight in a multimodal checkpoint whose offload
economics are not a compromise. Every other offload target is re-read across PCIe **every
token**, which is why `--cpu-offload-gb` is a last-resort trade: it buys capacity with
throughput. An encoder is read **once per image**, and not at all for a text-only request.
Evicting it is close to free across a large fraction of real use, and unlike
`--language-model-only` it keeps the capability.

So the value of doing this is entirely a question of how many bytes an encoder is. HF's
tensor viewer answers that one tensor at a time, which for a 300-module tower is no answer
at all. This note is the census, the structural reason none of it was reachable until
vLLM 0.29.0, and what it actually took to serve vision on a 16 GiB card once it was.

## The census

Safetensors metadata off the Hub, no weights fetched (`tools/encoder_census.py
--defaults` reproduces it). "Encoder" is the tower plus the projector feeding the text
model; MTP and draft heads are counted separately as `spec`, because they are not the
same kind of cost at all — see below.

| checkpoint | total | encoder | share | stored |
|---|---|---|---|---|
| Qwen/Qwen3.5-9B (bf16 base) | 17.98 G | 0.849 G | 4.72% | bf16 |
| Qwen3.5-9B AWQ 4-bit | 8.45 G | 0.849 G | 10.06% | bf16 |
| Qwen3.5-9B exl3 @4.00 | 6.69 G | 0.849 G | **12.69%** | bf16 |
| Qwen3.8-27B FP8 | 28.75 G | 0.858 G | 2.99% | bf16 |
| Qwen3.6-27B exl3 @5.00 | 18.53 G | 0.858 G | 4.63% | bf16 |
| Qwen3.6-27B exl3 @3.00 | 12.87 G | 0.858 G | 6.67% | bf16 |
| Qwen3.6-35B-A3B int2 AutoRound | 13.46 G | 0.832 G | 6.18% | bf16 |
| Qwen3.5-35B-A3B exl3 @2.00 | 10.16 G | 0.832 G | 8.19% | bf16 |
| gemma-4-26B-A4B-it exl3 @2.54 | 10.92 G | 1.067 G | 9.77% | bf16 |
| Muse-Glimmer-30B exl3 @2.00 | 10.22 G | 0.904 G | 8.85% | **4.00 bpw** |
| gemma-3-27b-it exl3 @4.0 | 16.33 G | 0.788 G | 4.82% | bf16 |
| Qwen3-VL-235B-A22B exl3 @3.00 | 84.80 G | 1.074 G | 1.27% | bf16 |
| Qwen3-VL-32B exl3 @3.0 | 14.02 G | 1.109 G | 7.91% | bf16 |
| Qwen3-VL-30B-A3B exl3 @3.00 | 12.36 G | 1.003 G | 8.11% | bf16 |
| Qwen3-VL-8B exl3 @6.0 | 7.52 G | 1.074 G | 14.27% | bf16 |
| **Qwen3-VL-8B exl3 @3.0** | 5.10 G | 1.074 G | **21.05%** | bf16 |
| **Step-3.7-Flash exl3 @3.05** | 75.18 G | **3.643 G** | 4.85% | bf16 |

### What it says

**1. The bytes are a constant; only the share moves.** Qwen3.5-9B's encoder is 0.849 GiB
as a bf16 base model, as AWQ 4-bit, and as EXL3 @4.00 — byte-identical. The share runs
4.72% → 10.06% → 12.69%. This is the embed tax's shape exactly (see
[embeddings.md](embeddings.md)): a tensor nobody quantizes costs most, as a fraction,
in the checkpoints chosen by the people with least VRAM to spare.

**2. Every format in the table ships it bf16.** AWQ, AutoRound int2, FP8, EXL3 — all of
them. This is not an EXL3 defect and there is no format here to be ahead of; it is an
ecosystem-wide default *among GPU-serving formats*.

*Corrected 2026-08-25: "the only quantized tower found anywhere" was too strong.*
`manjunathshiva/Muse-Glimmer-30B-tq3-g64` (MLX, so unservable here) quantizes the vision
tower to **affine 4-bit group-64** — 1.002 GiB of a 12.53 GiB package — alongside a
turboquant 3-bit g64 body and a 4-bit embedding and head. Its config names the idea
outright, `affine_extras: {bits: 4, group_size: 64}`: a first-class notion that the
non-body tensors get their own scheme at their own bit rate. So the practice exists; it
is the *GPU-serving* ecosystem that defaults to bf16, and MLX that does not. exllamav3 is in fact the only pipeline of the group that *offers* the choice
(`--vision_bits`, defaulting to 16); `compile.py` writes the key only when it is not 16,
so **an absent `vision_bits` means the default was taken**, not that the value is unknown.

**And bf16 is not obviously the wrong call**, which is worth stating because the rest of
this note reads like an indictment and is not one. The two largest encoders here belong to
*vision-first* models: nobody reaches for `Qwen3-VL-8B` or `Step-3.7-Flash` unless image
accuracy is the point, and spending a fifth of the package on the encoder may be precisely
the allocation wanted. Sizing that properly is a solver problem with a vision objective —
KLD against a text corpus, the instrument this project uses everywhere else, would measure
nothing relevant to it. Nothing here proposes touching that.

**What `--vision_bits` actually does, and the caveat it carries.** A quantized tower goes
through `quantize_side_model` in exllamav3's `convert_model.py`, whose own comment names
it: *"uncalibrated side models: MTP head, vision tower"*. It is called with `state = None`,
so each linear gets `init_H_data(False)` -- an `H` on the meta device with `count = 0` --
and `finalize_capture_H` reads that as `q_fallback`. There is no third option available:
the calibration corpus is six `.utf8` files tokenized into `input_ids`, and the tower takes
pixels, so it is never in the calibration forward pass at all. Calibrating it would mean a
second pipeline with an image corpus and the licensing that comes with shipping one. No
data is the honest answer here, not a lazy one.

**Most of EXL3 survives that, but not all.** `fallback_quant` is documented as "the same
quantization as `ldlq()` but without an LDL decomposition", and it calls the same
`quantize_tiles_multigpu` over the same 16x16 tensor-core tiles: the trellis codebook, the
Hadamard incoherence rotation, the sign flips and the global scale search are all intact.
What is missing is the Hessian-weighted layer -- LDLQ error feedback, and the
out-channel-scaling heuristic, which under `q_fallback` reverts to `apply_out_scales =
force_out_scales` and so defaults *off*. In QuIP#/QTIP terms this is the
incoherence-processed VQ baseline without the GPTQ-style correction pass: an increment
removed, not the foundation. **One trap for anyone reading a conversion log** -- under
fallback `proxy_err` is set to plain unweighted MSE, where every body tensor's `proxy_err`
is Hessian-weighted relative error. Same column, two different quantities, not comparable.

**The gap that leaves is measurable cheaply, which qualifies the KLD point above.** "KLD
against a text corpus would measure nothing relevant" is right about *sizing* an encoder
against a vision objective, and wrong about detecting whether quantizing one broke it.
Those are different questions, and the second is a self-comparison: same checkpoint, same
body, bf16 tower against quantized tower, on the same per-position logprob divergence
`bench/` already computes -- that instrument does not care that the prompt carries an
image. Cheaper still as a first look, cosine divergence on the adapter output isolates the
tower with no generation at all. Neither needs labels, a vision benchmark, or an absolute
score. `turboderp/Muse-Glimmer-30B-exl3` is the checkpoint that makes the comparison
possible, and the image fixture the `multimodal` gate needs is the same fixture this would
use.

**The offload argument is untouched by any of it, because eviction is lossless.** Whatever
depth an encoder is stored at, moving it to host memory costs zero accuracy, one PCIe pass
per image, and nothing at all for a text-only request. The quantization question is
contested and model-dependent; the eviction question is not, and the two do not trade
against each other. That is why eviction was the half worth pursuing: it serves the
reader who wants bf16 vision quality and the one who does not.

**3. It is a fixed cost over a shrinking denominator, so it is worst on the smallest
model.** The Qwen3-VL family ships essentially one encoder at every size — 1.074 GiB at
8B and 1.074 GiB at 235B, identical — so the share runs 1.27% → 7.91% → 14.27% →
**21.05%** as the model shrinks and the bit rate drops. A fifth of the entire package of
`Qwen3-VL-8B @3.0bpw` is an encoder, on the checkpoint chosen by whoever had the least
VRAM to start with. This is point 1 sharpened: not merely a constant, but a constant
*shared across a family*.

**4. The prize is the same order as the embedding work, and sometimes larger.** `blockq`
recovered 1.18 GiB on gemma-4-12B; an encoder is 0.79–1.11 GiB across most of the table
and **3.64 GiB** on Step-3.7-Flash — more than the entire non-encoder weight of
Qwen3-VL-8B @3.0. On a 16 GiB card that is 5–7% of the whole card in the typical case,
converting directly into KV headroom, and unlike the embedding it needs no format, no
quantizer change and no quality argument, because eviction is lossless.

**5. Unified models have no prize at all.** gemma-4-12B-it is
`Gemma4UnifiedForConditionalGeneration`: it consumes images directly into the text token
space, with no tower to evict. Its EXL3 checkpoint holds `model.vision_embedder` (9
tensors, 0.093 GiB) and nothing else — 1,665 tensors against the 26B-A4B sibling's 47,652.
Any accounting that treats "multimodal" as one class will get this wrong.

### The `spec` column is download, not VRAM

Built-in MTP and draft heads are **never resident** unless speculative decoding is
configured, so their bytes cost disk and bandwidth but not headroom — a different problem
from the encoder's, and not one offload addresses. Verified 2026-08-24 on
`Qwen3_5ForConditionalGeneration`: the model class never constructs an `mtp` submodule
(the only occurrence of the name in the file is the skip), and `AutoWeightsLoader` filters
`skip_prefixes` out of the weight stream before any parameter lookup
(`models/utils.py:421`), so the tensors are never read. Across vLLM the pattern is
consistent — MTP lives in separate model classes (`deepseek_mtp.py`, `gemma4_mtp.py`,
`ernie_mtp.py`) instantiated only when a speculative config asks.

**The skip is unconditional, which is what makes it robust.** No `skip_prefixes=["mtp."]`
anywhere is gated on the speculative config, so the base model never loads those tensors
under *any* configuration — when MTP drafting is enabled it is the separate drafter
instance that loads them. The plausible-sounding failure (exclude the built-in head
because `--speculative-config` is present, then load an external drafter as well, and end
up with both) needs a conditional skip that does not exist. It is also blocked
independently at the architecture level: an external drafter resolves through its own repo
(`registry.py`: `"DFlashDraftModel" -> ("qwen3_dflash", "DFlashQwen3ForCausalLM")`),
which cannot reach the base model's loader. With an external drafter the built-in head is
therefore pure dead download.

**And the failure mode is loud rather than silent.** `AutoWeightsLoader` raises on a
checkpoint tensor matching no parameter, so an implementation that forgot to skip errors
at load; it cannot quietly allocate. That is why this needs no per-model audit.

Do not try to settle this from the memory report. On the run above vLLM logged
`Model loading took 11.24 GiB` for a checkpoint totalling 11.172 GiB — *more than the
whole file* — because that number is a GPU memory delta across loading and includes
allocator reservation and kernel scratch (exl3's `A_had` buffer among them). Expected
if skipped was 10.974 GiB; the ~0.27 GiB difference is overhead, not tensors. The number
cannot separate the two cases and should not be asked to.

Worth knowing anyway, because the download is not small: Intel's
`Qwen3.6-35B-A3B-int2-mixed-CT-AutoRound` carries **1.573 GiB** of it, and
`Qwen/Qwen3.5-9B` 0.453 GiB.

*External* draft models (DFlash, DFlash2) ship as their own repos, so they are opt-in by
construction and strand nothing.

### Muse-Glimmer: what a quantized tower looks like

The one checkpoint found anywhere with a quantized encoder, built with `-vb 4` against the
default. Its 8.85% is what remains *after* a 4x reduction:

```
model.vision_tower.layers.0.attn.q_proj.trellis
   shape [96, 96, 64] I16 -> in=1536 out=1536 K=4 (4.00 bpw)

vision trellis: 1.917B weights in 914.2 MiB = 4.00 bits/weight
the same weights at bf16 would be 3.57 GiB
```

**The tower is 1.92 billion parameters** — 4.2x Qwen's 456M, larger than plenty of
language models — and it is held at 4.00 bpw while the text body of that checkpoint is at
2.00. The tower is served at twice the precision of the model it feeds. Unquantized, the
checkpoint would be ~12.9 GiB and the tower **27.7%** of it, which is presumably why the
flag was reached for on this model and no other.

Read the trellis carefully when sizing one of these: EXL3 stores it as int16 with the bit
width in the last dimension, so element count is `16/K` of the parameters it encodes. A
naive byte-per-element reading reports 16 bpw for everything.

## Nothing could evict any of it — until 0.29.0

*The section below described v0.28.0 and was true of it. Upstream closed the gap in
v0.29.0; the fix is in our fork already, and was found by looking rather than by asking
(2026-09-12). Kept rather than deleted, because the diagnosis is what made the fix
recognisable when it arrived.*

**What was true.** `--cpu-offload-gb` offloaded no encoder on any model, in any format —
not a dtype question, not an EXL3 question, not a selector question:

```
get_offloader().wrap_modules(…)      # vllm/model_executor/models/utils.py:824
```

That was the **only** call site in vLLM, and it sat inside `make_layers()` — the helper
that builds a *text decoder's* `ModuleList`. Vision towers build their own
(`self.blocks = nn.ModuleList([...])`, e.g. `qwen3_vl.py:628`) and were never handed to
the offloader. Both backends were affected identically, since the omission was upstream
of the backend choice.

### What landed

v0.29.0 adds `BaseOffloader.supports_tower_offload` (default `False`) and a **second**
call site, in `SupportsMultiModal._mark_tower_model`
([interfaces.py:380](../deps/vllm/vllm/model_executor/models/interfaces.py#L380)), whose
comment names this note's diagnosis outright: *"Towers are constructed directly, so
`make_layers` never routes them through the offloader."* It wraps at construction, so
offloaded tower weights are never allocated on the device at all. Around 100 model files
carry the marker — `qwen3_vl`, `qwen3_vl_moe`, `qwen3_5`, `gemma3_mm`, `gemma4_mm`,
`step3p7` and `muse_glimmer` among them.

**UVA only.** `PrefetchOffloader` keeps the flag `False` deliberately, because it
schedules prefetches over a circular layer stack. That is the correct half regardless:
a tower is read once per image and not at all for a text-only request, which is the
access pattern UVA wins by 4x (see [format-and-loading.md](format-and-loading.md)).

**Measured 2026-09-12** on `Qwen3.8-27B-exl3@3.00bpw`, a 16 GiB card:

```
--cpu-offload-gb 2 --cpu-offload-params visual
INFO [uva.py:65] Total CPU offloaded parameters: 0.86
```

0.86 GiB, matching the census. **The selector works and matters**: the counter stops at
0.86 under a 2 GiB budget, proving nothing but the tower matched. `wrap_modules` is called
with `prefix=<attr>` and the selector matches dot-delimited segments of
`f".{prefix}{name}."`, so `visual` names the Qwen tower exactly. Without it the budget is
first-come, and the tower wins only by construction order — an accident, not a design.

### The bytes freed are not the bytes that gate multimodal

Offloading the tower is necessary and not sufficient, and the follow-on measurement is the
useful half of this section. Two *other* encoder costs remain resident, and on a 16 GiB
card they are larger than the weights:

| | bytes | scales with |
|---|---|---|
| tower **weights** (offloadable, lossless) | 0.86 G | model |
| encoder **cache** `[tokens, out_hidden]` | 0.156 G | token budget |
| tower **transient**, one forward | ~1.6 G | patches |

The budget is derived, not defaulted:
`encoder_cache_size = max(max_num_batched_tokens, max_tokens_per_mm_item)`, and
`max_tokens_per_mm_item` comes off `preprocessor_config.json` as
`size.longest_edge / (patch_size * merge_size)^2`. Stock Qwen3.8 ships
`longest_edge: 16777216` with patch 16 and merge 2 — **16384 tokens**, 65536 pre-merge
patches, whose largest single buffer is `[65536, 4304]` bf16 = **564,133,888 bytes**,
the exact figure the allocator names when it fails. Left at the default, profiling that
transient starves KV to 0.21 GiB and the engine refuses to start.

**The knob is `mm_processor_kwargs`, not `--limit-mm-per-prompt`.** The latter's `width`
and `height` are `ImageDummyOptions` — profiling-only, read in
`multimodal/processing/dummy_inputs.py` and nowhere in the request path — so they shrink
the budget while the real image sails through at full resolution. `{"max_pixels": N}`
reaches `smart_resize` inside `_get_vision_info`, the same function that computes the
profiling token count, so budget and request move together by construction. The accounting
consequences are in [memory-accounting.md](memory-accounting.md); the trap's general form
and its tell are in the ecosystem field notes.

**Verified end to end, 2026-09-12** — vision serving with the tower on the host, on one
16 GiB card:

```
--gpu-memory-utilization 0.985 --kv-cache-dtype turboquant_4bit_nc
--cpu-offload-gb 2 --cpu-offload-params visual
--mm-processor-kwargs '{"max_pixels": 1048576}' --max-model-len 32768
```

0.86 GiB of tower on the host, 2.3 GiB of KV (90,593 tokens), a 4032x3024 photo described
correctly. At `max_pixels` 4194304 the same config OOMs inside the tower — twice, once
with KV pinned to vLLM's own recommendation — so the cap is doing real work and is not
cosmetic.

### The remaining fix, which still compounds

**Ours: register offload from `process_weights_after_loading`.** Unchanged by the above,
because it answers a different question. The approach proposed under `cpu-offload` reaches
only *quantized* modules, so today it covers exactly one checkpoint. What it buys there is
not capacity but bandwidth: a quantized tower moves 0.89 GiB per image batch instead of
3.57, so the per-image cost of having evicted it drops 4x. That is the difference between
an eviction you tolerate and one you leave in place — or, read the other way, a resident
tower cheap enough that you decline to offload at all.

Upstream made eviction possible; this makes it cheap. Neither helps a unified model,
which has nothing to evict.

**`compile_mm_encoder` is not the cheap way out (measured 2026-09-12).** The obvious
one-flag route — let inductor fuse the activation into the GEMM epilogue so the
full-width intermediate never materialises — does not happen. The flag is genuinely
active on this tower (28 compile passes, one per vision block, despite a docstring
naming only `Qwen2_5_vl` and `mLLaMa4`), and it left peak activation at 0.43 GiB against
0.44 without it: no reduction. What it *did* do is add **1.13 GiB** to consumed
memory (weights + non-torch), where compiled artifacts and autotune workspaces live,
halving the KV cache from 2.28 to 1.14 GiB (90,593 -> 44,333 tokens). The image still
OOMs at `max_pixels` 4194304. Worth noting the figure is larger and more persistent than
the 0.59 GiB cold-compile term [memory-accounting.md](memory-accounting.md) already has
open; same suspect, and this is a second sighting.

**A third lever this measurement exposed, not yet built.** The tower transient is
image-scaled and therefore unbudgetable in the sense
[memory-accounting.md](memory-accounting.md) sets out — it has to stop existing rather
than be reserved for. `Qwen3_VisionMLP.forward` is
`linear_fc2(act_fn(linear_fc1(x)))`, purely pointwise over dim 0, so chunking it over
tokens is bit-for-bit identical and caps the `[patches, 4304]` pair at the chunk. That is
structurally the same move as TurboQuant's slabbed continuation prefill. It raises the
affordable image size; it does not remove the need for a cap at 12 MP.

## GLM-4.1V: where the transient is measurable, and the law it obeys (2026-09-12)

`Qwen3.8-27B` is a bad instrument for the tower transient — its profiled peak is
decoder-dominated, so the encoder's requirement hides underneath a larger number.
`turboderp/GLM-4.1V-9B-Thinking-exl3@5.00bpw` is the opposite and is now the reference
platform for this work: 8.01 GiB of weights on a 15.5 GiB card leaves room to move, and
the profiled peak *is* the tower.

**Tower offload works unchanged on a second architecture.** `--cpu-offload-gb 3
--cpu-offload-params visual` offloads **1.66 GiB**, matching the census's 1.662 G exactly.
GLM names its tower `visual` as Qwen does, so the same selector works. Uncapped, it serves
a 4032x3024 photo correctly at 6045 prompt tokens with no `max_pixels` at all — the
headroom that Qwen3.8 lacked is what buys that.

**The transient is linear in patch rows, measured.** Two runs differing only in which
modality the profiler chose:

| profiled modality | patch rows | peak activation | KV available |
|---|---|---|---|
| video (default) | 48,672 | 1.85 GiB | 5.89 GiB |
| image (`--limit-mm-per-prompt '{"video": 0}'`) | 24,336 | 0.89 GiB | **7.00 GiB** |

2.00x the rows gives 2.08x the peak, and fitting the two points yields **42,357 B/row with
a -72 MiB intercept** — zero floor within the precision of the logged figures. So the peak
is the tower, it is linear, and chunking at C rows should land at `C x 42.4 KB`: 0.32 GiB
at 8192, 0.08 GiB at 2048. That linearity is the property the chunking argument needs, and
it is now measured rather than assumed.

*Recorded because the arithmetic route failed here:* predicting the peak from
`vision_config` alone gave 1.862 GiB against a measured 1.85, which looked like
confirmation and was not — the prediction used 24,336 rows while the profiled video item
has 48,672. Two measurements that differ in one variable beat one measurement that agrees
with a calculation.

**Why GLM is the worst case, in the useful direction.** Its MLP is gated —
`gate_up_proj` emits `[rows, 2I]` before `SiluAndMul` halves it — and `I/H` is 8.92
against the so400m shape's 3.74. At full resolution (12288 tokens, 49152 rows) one MLP
layer would want 3.76 GiB. Everything else in the census sits between these two models.

### Two incidental findings, both usable

**`--limit-mm-per-prompt '{"video": 0}'` is worth 1.11 GiB of KV** (+19% context) on a
video-capable model you only ever send images to. Unlike `width`/`height`, `count` **is**
enforced, so this one does what it says.

**The profiled modality is chosen by `(tokens, name)`, and the name can decide it.**
GLM's image and video budgets are *tied* at 6084 tokens, so `get_modality_with_max_tokens`
picked `video` on an alphabetical tiebreak. Equal tokens do not mean equal work: video's
token divisor includes `temporal_patch_size`, so at the same budget a video item carries
**twice** the patch rows an image does. Profiling as video is therefore ~2x conservative
here, by accident of sort order.

### How video actually reaches the tower

Frame count does not multiply the token budget — the pixel budget is a total across the
clip, so more frames buys lower per-frame resolution at constant tokens. Frames *are*
independent for attention: `prepare_encoder_metadata` builds `cu_seqlens` as
`patches_per_frame` repeated `grid_t` times, one sequence per frame, so no frame attends to
another. But the forward is a single pass over one concatenated `[total_rows, 1176]`
tensor, so the linear layers see every frame at once and the transient scales with the
total. `max_frames_per_batch` only pads `cu_seqlens` for CUDA-graph capture; it does not
split the forward.

**That is what makes chunking obviously safe here rather than merely plausible:** attention
is already frame-independent and the MLP is pointwise, so nothing in the tower requires all
rows resident simultaneously. The concatenation is an implementation choice, not a
constraint.

## The MLP is not the peak: largest allocation and peak live are different questions (2026-09-12)

Everything above pointed at the vision MLP. It was the wrong target, and the way it was
wrong is more useful than the original claim.

**What was built.** The MLP is pointwise over tokens, so slicing its forward into
row-chunks is bit-for-bit identical and bounds a buffer that otherwise scales with patch
count. Implemented behind a byte budget so one setting holds across tower widths, verified
equivalent (`torch.equal`, max abs diff 0.0) and verified to fire in a real serve
(`Chunking vision MLP: 48672 rows in 72 chunks of <=682`).

**What it did to the number that matters: nothing.**

| Qwen3.8 tower, 65536 rows | peak live | largest single allocation |
|---|---|---|
| unchunked | 1.728 GiB | **538.0 MiB** — `activation.py:816` via the MLP |
| chunked | 1.728 GiB | 432.0 MiB — the qkv gemm |

GLM-4.1V behaves identically: 1.856 GiB either way, unmoved down to a 16 MiB budget.

**The distinction that was being missed.** *Largest single allocation* is what an OOM
message reports — whichever request happened to fail once memory was already gone.
*Peak live* is the sum of everything alive at one instant; it is what the profiler
budgets and what sets the KV cache. Chunking removes the 538 MiB buffer — which is
exactly the `564,133,888` bytes the original Qwen OOM named — and the peak does not move,
because that buffer is never live at the same moment as the peak. The OOM named the MLP
because it was the allocation that failed, not the one that filled the card.

**Where the peak actually is, on both architectures: the attention preamble**, holding
several copies of q/k/v at once.

| | Qwen3.8 | GLM-4.1V |
|---|---|---|
| qkv gemm output | 576 MiB | 428 MiB |
| `rearrange(...).contiguous()` | 288 MiB | 428 MiB |
| `apply_rotary` | 288 MiB | 285 MiB |
| positional-embedding interpolate | 144 MiB | 143 MiB |
| norm + flash-attn + patch conv | 320 MiB | 317 MiB |
| **peak live** | **1.728 GiB** | **1.856 GiB** |

The shape is the same in both files. `self.qkv(x)` emits `[rows, 3H]`; the generator
`q, k, v = (rearrange(x, "s b ... -> b s ...").contiguous() for x in (q, k, v))` then
makes three full copies while that output is still live; `torch.cat([q, k])` copies two of
them again; the rotary emits another pair. The same data is materialized about four times
over, ~1.4 GiB of a 1.85 GiB peak.

**So the target is the copy chain, not chunking**, and it is better shaped work: no loop,
no budget knob, no per-architecture tuning, and the same pattern in both model files.
Before assuming the copies can go, check why `.contiguous()` is there — the attention
backend may require contiguous q/k/v, in which case the win is in avoiding the `cat`
rather than the copies.

**The chunking patch is shelved, not discarded**, because it becomes load-bearing the
moment this succeeds: once the attention peak drops below the MLP's 538 MiB single
allocation, the MLP binds. `shelf/mm-encoder-mlp-chunk` in the fork, exported to
[data/shelved/mm-encoder-mlp-chunk.patch](data/shelved/mm-encoder-mlp-chunk.patch); see
[patches.md](../patches.md).

**Method note, since this cost two wrong turns.** Both were the same error: reasoning about
memory from arithmetic and from error messages instead of from a snapshot.
`tools/memprof.py` answered it in one run and existed the whole time. Its distinction
between "composition of the peak" and "largest single allocation per call site" is exactly
the one being conflated — the tool had already been built to make this mistake visible.

## Muse-Glimmer: recomposing a bf16 tower, and why that is also the instrument

`turboderp/Muse-Glimmer-30B-exl3` is the one checkpoint in the census with a quantized
tower (`-vb 4`, 0.904 G for 1.92B parameters). Two separate wants point at the same
artifact: a **usable** checkpoint, and the **control arm** for a measurement that cannot
otherwise be made.

**The usability case.** A quantized tower is unreachable by vLLM's offloader — that path
sees `nn.Parameter`s, and EXL3 stores trellis tensors the offloader never registers — so
today the tower is resident or nothing. With a bf16 tower it can be evicted like any
other, and eviction being lossless means the size stops mattering: 3.57 GiB on the host is
one PCIe pass per image, not a capacity problem. Bigger but offloadable beats smaller but
pinned. It should also unblock native vLLM for this model, whose current blocker is
`vision_adapter.c_fc` being a plain `nn.Linear` no quantization plugin can reach — a bf16
adapter is exactly what that path wants.

**The measurement case, which is the one this note has been missing.** This document has
argued since 2026-08-24 that the quantized tower's quality is cheaply measurable: a
self-comparison of bf16 tower against quantized tower on the same per-position logprob
divergence `bench/` already computes, an instrument indifferent to whether the prompt
carries an image, with adapter-output cosine divergence as a cheaper first look. It named
Muse-Glimmer as "the checkpoint that makes the comparison possible", which was only half
true: a self-comparison needs **both arms**, and only the quantized one exists. The
recomposed checkpoint is the missing arm, so building it for use also builds the
instrument.

**And nothing currently characterises that tower.** qbench cannot: its axis is total
weight bytes and its quality signal is KLD against a *text* corpus, which a vision tower
does not touch. The conversion log cannot either — under `q_fallback` `proxy_err` is plain
unweighted MSE where every body tensor's is Hessian-weighted relative error, the same
column holding two incomparable quantities. So the honest prior is wide: the tower was
quantized with no calibration data (`quantize_side_model` is called with `state = None`;
the corpus is six `.utf8` files tokenized to `input_ids` and the tower takes pixels), and
it sits somewhere between nearly blind and nearly lossless with nothing published either
way. That is not a criticism of whoever ran the conversion — it is what the pipeline
structurally produces, and `--vision_bits` offers no third option.

**Feasibility.** The vision tensors are cleanly separable. The `2.00bpw` index holds 3596
tensors, of which 1718 are vision, under four prefixes — `model.vision_tower.*` (plus
`ln_pre`, `ln_post`, `patch_embedder`), `model.vision_adapter.*`, `model.vision_projection.*`
— and every quantized one carries EXL3's `trellis` / `suh` / `svh` / `mul1` suffixes, so
they are identifiable by name without inspecting shapes. The graft is: drop those, copy the
corresponding bf16 tensors from the reference repo, rewrite
`model.safetensors.index.json`, and delete `vision_bits` from `quantization_config` so the
loader does not expect a quantized tower.

**Two things to verify before trusting the result**, neither yet done: that the reference
and EXL3 checkpoints agree on tower tensor *names* (EXL3 conversion may rename or fuse),
and that the adapter and projection are grafted as a set with the tower — they sit on the
boundary and a mixed-precision seam there is exactly the kind of silent corruption that
[docs/format-and-loading.md](format-and-loading.md) exists to catch. Emitting a checkpoint
nobody else emits also needs the precedent check: whether any published EXL3 checkpoint
mixes a bf16 tower with a quantized body, since "it loads" is a property of one loader,
not of the format.

## Reproducing

```
tools/encoder_census.py --defaults           # the table above
tools/encoder_census.py <repo>[@rev] --detail   # per-suffix storage breakdown
```

`tools/checkpoint_survey.py` answers the adjacent question for a single checkpoint — what
is stored in a way the plugin can read — and has its own "never loaded when serving text"
bucket. That bucket fuses the encoder with MTP and draft heads; this tool separates them,
because they are evictable on completely different terms.
