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

**A third lever this measurement exposed, not yet built.** The tower transient is
image-scaled and therefore unbudgetable in the sense
[memory-accounting.md](memory-accounting.md) sets out — it has to stop existing rather
than be reserved for. `Qwen3_VisionMLP.forward` is
`linear_fc2(act_fn(linear_fc1(x)))`, purely pointwise over dim 0, so chunking it over
tokens is bit-for-bit identical and caps the `[patches, 4304]` pair at the chunk. That is
structurally the same move as TurboQuant's slabbed continuation prefill. It raises the
affordable image size; it does not remove the need for a cap at 12 MP.

## Reproducing

```
tools/encoder_census.py --defaults           # the table above
tools/encoder_census.py <repo>[@rev] --detail   # per-suffix storage breakdown
```

`tools/checkpoint_survey.py` answers the adjacent question for a single checkpoint — what
is stored in a way the plugin can read — and has its own "never loaded when serving text"
bucket. That bucket fuses the encoder with MTP and draft heads; this tool separates them,
because they are evictable on completely different terms.
