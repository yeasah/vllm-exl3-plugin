# vllm-exl3-plugin

Experiment in adding EXL3 support to vLLM, as an out-of-tree
`vllm.general_plugins` quantization backend.

EXL3 is [exllamav3](https://github.com/turboderp-org/exllamav3)'s trellis-coded
quantization format: plain safetensors, plain HF `config.json`, MIT-licensed,
with working CUDA kernels already shipped as a Python-callable extension. The
plugin is an adapter onto vLLM's `QuantizationConfig` / `LinearMethodBase`
interfaces — it does not build kernels.

**Status: Phase 1 complete and verified.** Weights stay quantized and the fused
exllamav3 kernels do the multiply, under torch.compile and CUDA graphs. On
Llama-3.2-1B @3bpw that is 2.35 GiB -> 0.86 GiB with decode 22% *faster* than
the dense path; gemma-4-12B at 3bpw runs in 6.32 GiB on a 16 GB card.

Phase 2 (tensor parallelism) is **validated at TP=2** across three models, eager
and with CUDA graphs, reproducing TP=1 token for token — including a
vocab-parallel quantized `lm_head`. See [PHASE2.md](PHASE2.md). TP>2 is
unexercised.

Phase 3 (MoE) works on all three MoE checkpoints tried: gemma-4-26B-A4B (9.46
GiB), Laguna-XS-2.1 (256 experts at 2bpw, 8.54 GiB) and Qwen3.5-35B-A3B (10.63
GiB, needs the `patches/` change to load). Laguna turned up a scale factor that
EXL3 checkpoints carry but do not record, which the plugin recovers by measuring
the weights; and all three needed exllamav3's cooperative-kernel autotuner
turned off, since it re-tunes on every new batch shape under continuous batching
and deadlocks. See [PHASE3.md](PHASE3.md).

See [PHASE1.md](PHASE1.md) for benchmarks, [PHASE0.md](PHASE0.md) for the format
groundwork, and
[VLLM_PLUGIN_FEASIBILITY.md](VLLM_PLUGIN_FEASIBILITY.md) for the background
research and the phased plan.

## Quick start

    pip install --no-deps --no-build-isolation -e deps/exllamav3   # builds the CUDA ext
    # if CUDA_HOME's toolkit does not match torch's, see PHASE2.md "Finishing this"
    pip install --no-deps -e .

    vllm serve turboderp/Llama-3.2-1B-Instruct-exl3 --revision 3.0bpw

## Layout

    vllm_exl3_plugin/format.py     on-disk format arithmetic (no torch)
    vllm_exl3_plugin/ops.py        wrappers over exllamav3_ext
    vllm_exl3_plugin/tp.py         tensor-parallel slicing rules
    vllm_exl3_plugin/quantization/ EXL3Config, linear, lm_head, fused_moe
    deps/exllamav3                 submodule, reference checkout
    tests/                         runnable without a GPU

## Tests

    python -m unittest discover -s tests

48 tests. The format tests need neither torch, vLLM, nor a GPU; the kernel
oracles and the end-to-end generations skip themselves without CUDA and
exllamav3.

The oracle tests import the `exllamav3` package (not just its extension), which
needs a few small extras the plugin itself does not:

    pip install --no-deps kbnf formatron frozendict general-sam

## Gotcha: BOS tokens

Some checkpoints (gemma-4 among them) put `<bos>` in `chat_template.jinja` and
do *not* add it in the tokenizer, even with `add_special_tokens=True`. Gemma
produces garbage without it. Drive models through their chat template
(`llm.chat`, or the server's chat endpoint) rather than raw completion prompts.

## Requirements

- CUDA, compute capability 8.0+ (Ampere). No ROCm — exllamav3 has none.
- float16 or bfloat16 activations. The kernels are fp16; `exl3_mm` casts at the
  kernel boundary, so bf16 models keep a bf16 residual stream.
- TP=1 and TP=2. Higher degrees are implemented and proven arithmetically
  but unexercised, and warn at startup; see [PHASE2.md](PHASE2.md).

## Environment variables

| variable | default | effect |
|---|---|---|
| `VLLM_EXL3_RECONSTRUCT_THRESHOLD` | 144 | rows above which to decode to dense fp16 and use cuBLAS; 0 always uses the fused kernel |
| `VLLM_EXL3_DEQUANTIZE` | 0 | Phase 0 behaviour: dequantize at load. Correctness oracle, no memory saving |
| `VLLM_EXL3_MOE_AUTOTUNE` | 0 | set to 1 to re-enable exllamav3's cooperative-kernel autotuner for routed experts. It deadlocks under continuous batching and buys no throughput; see PHASE3.md |
| `VLLM_DISABLE_COMPILE_CACHE` | 0 | set to 1 while editing this plugin — vLLM's compile cache cannot see plugin code |
