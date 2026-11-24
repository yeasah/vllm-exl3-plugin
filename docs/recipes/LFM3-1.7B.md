# [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B)

An extremely small model for its claimed agentic capabilities.

The KV cache requires 12KiB/token unquantized, and at the maximum model length (256K) this will require 3.05GiB unquantized, 1.53GiB for a `kv_cache_dtype` of `fp8`, or 0.90GiB for `turboquant_4bit_nc` or `nvfp4`

Available EXL3 quants include [Trisert/LFM3-1.7B-exl3](https://hf.co/Trisert/LFM3-1.7B-exl3)

## Just about anywhere

```yaml
model: 'Trisert/LFM3-1.7B-exl3'
revision: '4.0bpw-sc'
max_model_len: auto
max_num_seqs: 4
enable_auto_tool_choice: yes
tool_call_parser: qwen3_coder
reasoning_parser: qwen3
kv_cache_dtype: fp8
```
