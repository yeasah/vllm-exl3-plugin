# vllm-exl3-plugin

Experiment in adding EXL3 support to vLLM, as an out-of-tree
`vllm.general_plugins` quantization backend.

EXL3 is [exllamav3](https://github.com/turboderp-org/exllamav3)'s trellis-coded
quantization format: plain safetensors, plain HF `config.json`, MIT-licensed,
with working CUDA kernels already shipped as a Python-callable extension. The
plugin is an adapter onto vLLM's `QuantizationConfig` / `LinearMethodBase`
interfaces — it does not build kernels.

**Status: Phase 0, nothing has run yet.** See [PHASE0.md](PHASE0.md) for what is
built, what is verified, and what is next; see
[VLLM_PLUGIN_FEASIBILITY.md](VLLM_PLUGIN_FEASIBILITY.md) for the background
research and the phased plan.

## Layout

    vllm_exl3_plugin/format.py     on-disk format arithmetic (no torch)
    vllm_exl3_plugin/ops.py        wrappers over exllamav3_ext
    vllm_exl3_plugin/quantization/ EXL3Config + EXL3LinearMethod
    deps/exllamav3                 submodule, reference checkout
    tests/                         runnable without a GPU

## Tests

    python -m unittest discover -s tests

The format tests need neither torch, vLLM, nor a GPU.

## Requirements

- CUDA, compute capability 8.0+ (Ampere). No ROCm — exllamav3 has none.
- `--dtype float16`. exllamav3's kernels are fp16 throughout, and most EXL3
  repos inherit `bfloat16` from their base model, which vLLM will reject.
- TP=1 for now.
