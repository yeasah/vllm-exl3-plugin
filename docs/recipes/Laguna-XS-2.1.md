# [poolside/Laguna-XS-2.1](https://hf.co/poolside/Laguna-XS-2.1)

poolside's agentic-focused MoE. This is a 33B parameter model. Reasoning and tool calling are supported via the `poolside_v1` parsers.

The KV cache requires 42KiB/token unquantized, and at the maximum model length (256K) this will require 10.53GiB unquantized, 5.26GiB for a `kv_cache_dtype` of `fp8`, or 3.10GiB for `turboquant_4bit_nc` or `nvfp4`

Available EXL3 quants include [turboderp/Laguna-XS-2.1-exl3](https://hf.co/turboderp/Laguna-XS-2.1-exl3)

## 12GiB+

```yaml
model: 'turboderp/Laguna-XS-2.1-exl3'
revision: 4.00bpw
max_model_len: auto
max_num_seqs: 4
enable_auto_tool_choice: yes
tool_call_parser: poolside_v1
reasoning_parser: poolside_v1
kv_cache_dtype: nvfp4
cpu_offload_gb: 16
cpu_offload_params: experts
#speculative_config: '{"model":"poolside/Laguna-XS-2.1-DFlash","num_speculative_tokens":15,"method":"dflash"}'
```
