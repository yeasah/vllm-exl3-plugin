# [Qwen/Qwen3.8-27B](https://hf.co/Qwen/Qwen3.8-27B)

One of the current darlings of the community, a strong coder. Reasoning is supported via the `qwen3` parser, and tool calling is supported by the `qwen3_coder` parser. The sampling parameter situation is simpler than prior entries in the Qwen lineup, with only two configs recommended, one for thinking and one for non-thinking. Normally one would want to use this model in thinking mode, which is the default. However, note that this model supports different thinking effort levels, and the chat template's default is `xhigh`, which many find excessive (I certainly do, and I would recommend `medium` instead)

The KV cache requires 64KiB/token unquantized, and at the maximum model length (256K) this will require 16.32GiB unquantized, 8.33GiB for a `kv_cache_dtype` of `fp8`, or 4.8GiB for `turboquant_4bit_nc` or `nvfp4`

Available EXL3 quants include [yeasah/Qwen3.8-27B-exl3](https://hf.co/yeasah/Qwen3.8-27B-exl3) and [turboderp/Qwen3.8-27B-exl3](https://hf.co/turboderp/Qwen3.8-27B-exl3)

Example yaml config (this can be supplied to vLLM via the `--config` argument):

## 12-16GiB

No vision, 3.5bpw, NVFP4 KV cache. For 12GiB, or for longer contexts, use `2.5bpw-H5` or `3.0bpw-H5` instead.

```yaml
model: 'yeasah/Qwen3.8-27B-exl3'
revision: '3.50bpw-H5'
max_model_len: auto
max_num_seqs: 4
enable_auto_tool_choice: yes
tool_call_parser: qwen3_coder
reasoning_parser: qwen3
kv_cache_dtype: nvfp4
language_model_only: yes
```

## 24GiB+

Offloaded vision, 4.0bpw, NFVP4 KV cache. To enable MTP, uncomment the `speculative_config` line.

```yaml
model: 'yeasah/Qwen3.8-27B-exl3'
revision: '4.00bpw'
max_model_len: auto
max_num_seqs: 4
enable_auto_tool_choice: yes
tool_call_parser: qwen3_coder
reasoning_parser: qwen3
kv_cache_dtype: nvfp4
cpu_offload_gb: 2
cpu_offload_params: visual
mm_processor_kwargs: '{"max_pixels": 1048576}' 
#speculative_config: '{"method":"mtp","num_speculative_tokens":2,"revision":"4.00bpw"}'
```

## Benchmarks

|Platform|VRAM|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|
|RTX 3060|10.8 GiB|38K|1|386.75 t/s|21.99 t/s|
|RTX 3060|10.8 GiB|38K|2|643.38 t/s|22.93 t/s|
|RTX 5060 ti|15.2 GiB|108K|1|714.69 t/s|27.45 t/s|
|RTX 5060 ti|15.2 GiB|108K|4|1880.57 t/s|51.09 t/s|
|RTX 5070 ti|14.9 GiB|105K|1|1293.09 t/s|46.38 t/s|
|RTX 5070 ti|14.9 GiB|105K|4|3468.45 t/s|89.47 t/s|