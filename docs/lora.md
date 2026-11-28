# LoRA adapters on EXL3 models

Serving existing PEFT LoRA adapters against EXL3 quants through vLLM's `--enable-lora` / `--lora-modules`
(FUTURE `vllm-lora`, first half). Established 2026-11-27 on Ornith-1.5-9B (Qwen3.5 architecture: full attention,
GatedDeltaNet `linear_attn`, dense MLP) with `nchapman/Ornith-1.5-9B-Swift-1.5-LoRA` (r=64, alpha=64, all 200 dense
language-model projections, no embedding or head deltas), on the 2x RTX 5060 Ti host, vLLM 0.34.1.dev12
(`deps/vllm` `69d608635e`).

## Status

Works on dense EXL3 models: eager, CUDA graphs, TP=2, base and adapter requests from one engine. MoE models are
refused (below). Not yet exercised: `vllm serve --lora-modules` end to end (same code path as `LLM(enable_lora=True)`),
several adapters in one batch, adapters with embedding or `lm_head` deltas, EXL3 at TP>2.

## Why it nearly worked unmodified

vLLM's LoRA wrappers (`ColumnParallelLinearWithLoRA` and kin) compute the base output through the base layer's
`quant_method.apply` and add the low-rank delta to it, so any quant method whose `apply` is a plain
`x -> y` composes with LoRA -- `EXL3LinearMethod.apply` is. What failed was setup, in three places, all reading a
weight by name:

- `_get_lora_device` (`vllm/lora/layers/utils.py`) recognises a layer by `weight`, `qweight`, `weight_packed`, ... and
  raises `Unsupported base layer` on EXL3's `exl3_trellis_<n>`.
- `VocabParallelEmbeddingWithLoRA.create_lora_weights` reads `base_layer.weight.device`. The wrapper is built for
  `embed_tokens` whatever the adapter targets, so every model hits it.
- `from_layer_logits_processor` reads `lm_head.weight.dtype` and `.device`.

`stock_compat._lora_on_exl3` covers all three, scoped to layers whose quant method is the plugin's: the device comes
from the layer's own tensors (first one off the CPU, so host-offloaded layers resolve to the GPU), and the embedding
and head carry an empty stand-in `weight` only for the duration of the call that reads it, so the plugin's own
"is this layer dense?" checks never see one. The head's dtype is the adapter dtype: `LogitsProcessorWithLoRA` stores
it and reads it nowhere. Upstream main (2026-11-23) still has all three lookups.

**MoE is refused clearly.** vLLM's `FusedMoEWithLoRA` needs the experts' quant method to expose a modular kernel that
supports LoRA; `EXL3MoEMethod` has its own `apply` over `exl3_mgemm`, so it would fail with "Old-style modular kernel
not supported". The device patch raises `NotImplementedError` naming the reason instead. Since vLLM wraps every
supported module regardless of what the adapter targets, this refuses LoRA on any EXL3 MoE model, attention-only
adapters included. Lifting it means fitting `exl3_mgemm` into vLLM's modular-kernel interface.

## Measurements

**The adapter's logprob shift, against bf16.** Teacher-forced on six bf16+adapter sequences (2,063 scored tokens),
per-token `lora - base`, correlated with the same shift on bf16 at TP=2 (`tests/data/lora-ornith-swift-bf16.json`):

| arm | r | slope | base distance to bf16 (mean \|dlogprob\|) |
|---|---|---|---|
| 6.00bpw | 0.949 | 0.94 | 0.015 |
| 4.00bpw, eager / graphs / TP=2 | 0.90-0.91 | 0.82-0.87 | 0.041 |
| 3.00bpw-H5 | 0.826 | 0.85 | 0.096 |

The adapter's whole effect is ~0.020 per token, half 4.00bpw's distance from bf16: for a fine-tune this small, the
quantization moves the model more than the adapter does. That argues for adapters trained against the EXL3 base
(the FUTURE item's second half), which could also recover some quantization loss.

**Per module.** Every one of the 128 wrappers covering the adapter's 200 modules adds the delta the file defines to
0.0023 relative error (bf16 level), it reaches the output to within 0.0022 of the rounding floor, and the untargeted
wrappers (`in_proj_ba`, `embed_tokens`) add exactly zero.

## The gate, and the one that looked like a gate

`tests/test_lora.py`. **The logprob-shift fit is a smoke test only.** It fails an adapter not applied (slope 0) or
mis-scaled (2x, 0.5x), but passed with all 72 `linear_attn` deltas zeroed in the adapter file (r=0.918, *higher* than
the intact adapter) and with all 32 `down_proj` ones (0.876): the shift is below quantization noise, so losing a
whole projection family hardly moves it. A dropped family is exactly the realistic failure -- packed-module mapping
is where one would get lost -- so the fit cannot be the gate.

**The gate is the per-module check** (`test_per_module`): hooks on every LoRA-wrapped linear, inside the engine,
comparing against the adapter file's A and B in fp32. The delta cannot be read off the output: it is added into a
bf16 output it is far smaller than (layer 3's `o_proj`: mean |delta| 6.5e-5 against |y| 0.023, under one ulp), and
`y - y_base` came out 0.488 off -- exactly the bf16 rounding floor, computed independently. So it checks two things
separately: vLLM's own LoRA kernels, rerun into a zero buffer through the wrapper's stacked weights, slices and
token mapping, must equal `x @ A^T @ B^T * alpha/r` (replay); and the output must be the base output plus that delta
as bf16 rounds it (applied). Plus coverage: every adapter module must reach a wrapper.

Shown to fail with serving-side faults injected after the adapter loads (`tests/test_lora.py --modules REV OUT -
FAULT`): `drop:in_proj_qkvz` (stacked B zeroed) and `drop:down_proj` fail exactly their family at replay error 1.0;
`skip:o_proj` (base output returned) fails applied error at 1.0; `swap:in_proj_qkvz` (first and last packed A
slices exchanged) fails at 1.09 mean. Swapping the first two slices changed nothing: `in_proj_qkvz`'s q, k and v
slices all carry `in_proj_qkv`'s A, so they are identical. `test_per_module_catches_dropped_family` keeps the first
fault in the suite. Broken adapter *files* do not test this check, since both sides read the file.

## Run-to-run variation, two sources

Found because two "identical" eager runs of the base arm disagreed by up to 0.35 in a logprob. Neither is a defect;
both made the harness unreproducible until handled.

**Batch composition, through the multiprocess engine.** `LLM.generate` with several prompts sends them to the
EngineCore over ZMQ, and a step can start before all have arrived, so which sequences share a prefill batch depends
on timing. Localized by elimination: FlashInfer autotune, LoRA, prefix caching, async scheduling and the GDN backend
(FlashInfer and Triton) all varied; the EXL3 GEMM is bit-deterministic in isolation (30 calls per shape, two
processes, six module types at m=1/64/2069); the in-process engine (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, which queues
every request before stepping) was bit-identical in 6 of 6 passes. Forcing batch partitions in-process then reproduced
the multiprocess outcomes bit-exactly, sequence by sequence: all 24 multiprocess passes on the default (FlashInfer)
GDN kernel, five distinct outcomes among them, matched a partition tried. The four left over were the Triton-GDN
runs, a different kernel, themselves two outcomes.

What differs between formats is how often a batch change matters. bf16 at TP=2 was bit-identical across `all`,
`3+3` and a rerun, and moved only the tokens whose batch landed in another size bucket; EXL3 4.00bpw moved 99% of
tokens under every partition change. The EXL3 GEMM picks its partition per row-count bucket ("past 16 rows, buckets
follow the multi-row tile heights", `exl3_gemm.cu`), so nearly any batch is a new reduction order. Magnitude when it
moves is alike: mean |dlogprob| 0.0072 (bf16) vs 0.0075 (EXL3), max 0.13 vs 0.35. For scale, that is ~20% of
4.00bpw's distance from bf16 and ~40% of this adapter's whole effect.

Exposure in our own tools: none outside this test. The bench gate sends one prompt per `generate` (`bench/core.py`),
which fixes the batch; qbench scores through exllamav3, not vLLM. The harness now runs its arms in-process.

**vLLM's LoRA shrink kernel.** With the batch fixed, the base arm repeated bit for bit and the adapter arm did not
(max 0.18, first difference 11-29 tokens in). The shrink kernel splits K 8 or 64 ways and reduces with relaxed
`tl.atomic_add` (`vllm/lora/ops/triton_ops/kernel_utils.py`, configs in `utils.py`); only `VLLM_BATCH_INVARIANT` sets
split-K 1, and batch-invariant mode refuses GDN layers. A `VLLM_TUNED_CONFIG_FOLDER` holding the default shrink
config at `split_k: 1` made the adapter arm bit-identical across processes. The shift tests use one; the per-module
check keeps the default kernels, which is what serving runs.

## Open

- MoE: `exl3_mgemm` behind vLLM's modular-kernel interface with LoRA support.
- Quantized LoRA: adapters trained against the EXL3 base, given the measurement above.
- The TP=1 arms aborted (SIGABRT) mid-load twice in the first full suite run, with no GPU or host-memory contention
  in telemetry, and not in any run since. The harness keeps a failed arm's full log (`test_lora-arm-*.log` in the
  temp directory).
