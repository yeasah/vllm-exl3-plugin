# [Qwen/Qwen3.5-9B](https://hf.co/Qwen/Qwen3.5-9B)

A small, but still usefully sized entry in the Qwen lineup. 9B parameters, with vision.
Reasoning is supported via the `qwen3` reasoning parser, and tool calling via the `qwen3_coder` parser. There are no default sampling parameters defined in the model, so it is either necessary to set these at every client, or override these to the correct defaults via (for example) `--override-generation-config '{ "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0 }'` -- note that there are *four* different sampling parameter configurations recommended on the model card for different scenarios, so make sure you are using the right set.

The KV cache requires 32KiB/token unquantized, and at the maximum model length (256K) this will require 8.10GiB unquantized, 4.11GiB for a `kv_cache_dtype` of `fp8`, or 2.38GiB for `turboquant_4bit_nc` or `nvfp4`

Available EXL3 quants include [turboderp/Qwen3.5-9B-exl3](https://hf.co/turboderp/Qwen3.5-9B-exl3)

Example yaml config (this can be supplied to vLLM via the `--config` argument):

## 8GiB+

Offloaded vision enabled. To enable MTP, uncomment the `speculative_config` line.

```yaml
model: 'turboderp/Qwen3.5-9B-exl3'
revision: '4.00bpw'
max_model_len: auto
max_num_seqs: 4
enable_auto_tool_choice: yes
tool_call_parser: qwen3_coder
reasoning_parser: qwen3
kv_cache_dtype: nvfp4
override_generation_config: '{ "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0 }'
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

<img src="cats/Qwen3.5-9B.svg" alt="meow">