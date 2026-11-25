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

|Platform|VRAM|Host RAM|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|---|
|RTX 3060|10.1 GiB|0.5 GiB|548K|1|1722 t/s|49 t/s|
|RTX 3060|10.1 GiB|0.5 GiB|548K|4|4607 t/s|140 t/s|
|RTX 5060 Ti|15.1 GiB|0.5 GiB|1017K|1|3181 t/s|68 t/s|
|RTX 5060 Ti|15.1 GiB|0.5 GiB|1017K|4|9013 t/s|201 t/s|
|RTX 5070 Ti|15.1 GiB|0.5 GiB|1010K|1|6154 t/s|102 t/s|
|RTX 5070 Ti|15.1 GiB|0.5 GiB|1010K|4|17223 t/s|315 t/s|

## Cat!

The following cat was produced by `turboderp/Qwen3.5-9B-exl3` at `4.00bpw` on a RTX 5060 Ti in 23.29 seconds, using 1575 tokens (including thoughts, if any)

<img src="cats/Qwen3.5-9B.svg" alt="meow">