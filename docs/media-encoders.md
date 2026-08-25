# Media encoders: how big, and why nothing can evict them

*Measured 2026-08-24. Tracked in TODO as `encoder-offload`.*

A vision or audio encoder is the one weight in a multimodal checkpoint whose offload
economics are not a compromise. Every other offload target is re-read across PCIe **every
token**, which is why `--cpu-offload-gb` is a last-resort trade: it buys capacity with
throughput. An encoder is read **once per image**, and not at all for a text-only request.
Evicting it is close to free across a large fraction of real use, and unlike
`--language-model-only` it keeps the capability.

So the value of doing this is entirely a question of how many bytes an encoder is. HF's
tensor viewer answers that one tensor at a time, which for a 300-module tower is no answer
at all. This note is the census, and the structural reason none of it is reachable today.

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

**The offload argument is untouched by any of it, because eviction is lossless.** Whatever
depth an encoder is stored at, moving it to host memory costs zero accuracy, one PCIe pass
per image, and nothing at all for a text-only request. The quantization question is
contested and model-dependent; the eviction question is not, and the two do not trade
against each other. That is why the upstream fix below is the half worth pursuing: it
serves the reader who wants bf16 vision quality and the one who does not.

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

## Nothing can evict any of it

`--cpu-offload-gb` offloads no encoder on any model, in any format. Not a dtype question,
not an EXL3 question, not a selector question:

```
get_offloader().wrap_modules(…)      # vllm/model_executor/models/utils.py:824
```

That is the **only** call site in vLLM, and it sits inside `make_layers()` — the helper
that builds a *text decoder's* `ModuleList`. Vision towers build their own
(`self.blocks = nn.ModuleList([...])`, e.g. `qwen3_vl.py:628`) and are never handed to the
offloader. Both backends are affected identically, since the omission is upstream of the
backend choice.

This is a **third** cause, alongside the two traced in TODO `cpu-offload` (eligibility
decided at construction against empty placeholders; `process_weights_after_loading`
replacing the parameters afterwards). Unlike those two it is not ours and not
quantization-specific, so it will never show up as an EXL3 anomaly in a cross-format
comparison — every format loses the same bytes.

### Two fixes, and they compound rather than overlap

**Upstream: offer the tower to the offloader.** Small, general, format-agnostic — every
multimodal model in vLLM gains 0.8-3.6 GiB of optional headroom regardless of quantization.
This is the one that matters for the table above, because in nine of ten rows the tower is
bf16 and therefore invisible to any quantization plugin. Tracked as
`upstream-queue`; framing and priority in [upstream.md](upstream.md).

**Ours: register offload from `process_weights_after_loading`.** The approach already
proposed under `cpu-offload` reaches only *quantized* modules, so today it covers exactly
one checkpoint. What it buys there is not capacity but bandwidth: a quantized tower moves
0.89 GiB per image batch instead of 3.57, so the per-image cost of having evicted it drops
4x. That is the difference between an eviction you tolerate and one you leave in place —
or, read the other way, a resident tower cheap enough that you decline to offload at all.

The two are independent and multiply: the first makes eviction possible, the second makes
it cheap. Neither helps a unified model, which has nothing to evict.

## Reproducing

```
tools/encoder_census.py --defaults           # the table above
tools/encoder_census.py <repo>[@rev] --detail   # per-suffix storage breakdown
```

`tools/checkpoint_survey.py` answers the adjacent question for a single checkpoint — what
is stored in a way the plugin can read — and has its own "never loaded when serving text"
bucket. That bucket fuses the encoder with MTP and draft heads; this tool separates them,
because they are evictable on completely different terms.
