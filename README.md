# vllm-exl3-plugin

Experiment in adding EXL3 support to vLLM, as an out-of-tree
`vllm.general_plugins` quantization backend.

EXL3 is [exllamav3](https://github.com/turboderp-org/exllamav3)'s trellis-coded
quantization format: plain safetensors, plain HF `config.json`, MIT-licensed,
with working CUDA kernels already shipped as a Python-callable extension. The
plugin is an adapter onto vLLM's `QuantizationConfig` / `LinearMethodBase`
interfaces — it does not build kernels.

**Status: Phase 0 complete and verified.** EXL3 checkpoints load through vLLM
and generate correct tokens, at uniform and mixed bit widths, with
dequantization bit-identical to exllamav3's own. Phase 0 dequantizes at load
time, so there is no memory saving yet — that is Phase 1. See
[PHASE0.md](PHASE0.md) for what was verified and what is next; see
[VLLM_PLUGIN_FEASIBILITY.md](VLLM_PLUGIN_FEASIBILITY.md) for the background
research and the phased plan.

## Quick start

    pip install --no-deps --no-build-isolation -e deps/exllamav3   # builds the CUDA ext
    pip install --no-deps -e .

    vllm serve turboderp/Llama-3.2-1B-Instruct-exl3 \
        --revision 3.0bpw --dtype float16 --enforce-eager

## Layout

    vllm_exl3_plugin/format.py     on-disk format arithmetic (no torch)
    vllm_exl3_plugin/ops.py        wrappers over exllamav3_ext
    vllm_exl3_plugin/quantization/ EXL3Config + EXL3LinearMethod
    deps/exllamav3                 submodule, reference checkout
    tests/                         runnable without a GPU

## Tests

    python -m unittest discover -s tests

19 tests. The format tests need neither torch, vLLM, nor a GPU; the kernel
oracles and the end-to-end generations skip themselves without CUDA and
exllamav3.

The oracle tests import the `exllamav3` package (not just its extension), which
needs a few small extras the plugin itself does not:

    pip install --no-deps kbnf formatron frozendict general-sam

## Requirements

- CUDA, compute capability 8.0+ (Ampere). No ROCm — exllamav3 has none.
- `--dtype float16`. exllamav3's kernels are fp16 throughout, and most EXL3
  repos inherit `bfloat16` from their base model, which vLLM will reject.
- TP=1 for now.
