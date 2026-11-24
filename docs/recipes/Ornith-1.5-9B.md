# [ornith-ai/Ornith-1.5-9B](https://hf.co/ornith-ai/Ornith-1.5-9B)

A Qwen finetune oriented toward agentic work. Reasoning is supported via the `qwen3` reasoning parser, and tool calling via the `qwen3_coder` parser. There are no default sampling parameters defined in the base model, but the EXL3 quant linked below has this corrected so it will include the recommended sampling parameters by default.

The KV cache requires 32KiB/token unquantized, and at the maximum model length (256K) this will require 8.10GiB unquantized, 4.11GiB for a `kv_cache_dtype` of `fp8`, or 2.38GiB for `turboquant_4bit_nc` or `nvfp4`

Available EXL3 quants include [yeasah/Ornith-1.5-9B-exl3](https://hf.co/yeasah/Ornith-1.5-9B-exl3)

Example yaml config (this can be supplied to vLLM via the `--config` argument):

## 8GiB+

Offloaded vision enabled. To enable MTP, uncomment the `speculative_config` line.

```yaml
model: 'yeasah/Ornith-1.5-9B-exl3'
revision: '4.00bpw'
max_model_len: auto
max_num_seqs: 4
enable_auto_tool_choice: yes
tool_call_parser: qwen3_coder
reasoning_parser: qwen3
kv_cache_dtype: nvfp4
cpu_offload_gb: 1
cpu_offload_params: visual
mm_processor_kwargs: '{"max_pixels": 1048576}' 
#speculative_config: '{"method":"mtp","num_speculative_tokens":2,"revision":"4.00bpw"}'
```

## Benchmarks

|Platform|VRAM|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|
|RTX 3060|10.32 GiB|504K|1|1288.08 t/s|45.76 t/s|
|RTX 3060|10.32 GiB|504K|4|3257.29 t/s|92.27 t/s|
|RTX 5060 ti|15.2 GiB|975K|1|2379.11 t/s|64.08 t/s|
|RTX 5060 ti|15.2 GiB|975K|4|6131.44 t/s|140.10 t/s|
|RTX 5070 ti|15.2 GiB|986K|1|4385.20 t/s|101.39 t/s|
|RTX 5070 ti|15.2 GiB|986K|4|11953.88 t/s|228.85 t/s|

## Cat!

<img src="cats/Ornith-1.5-9B.svg" alt="meow">