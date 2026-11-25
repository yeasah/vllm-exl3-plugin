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

|Platform|VRAM|Host RAM|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|---|
|RTX 3060|10.1 GiB|0.5 GiB|541K|1|1689 t/s|55 t/s|
|RTX 3060|10.1 GiB|0.5 GiB|541K|4|4661 t/s|141 t/s|
|RTX 5060 Ti|15.1 GiB|0.5 GiB|1017K|1|3136 t/s|73 t/s|
|RTX 5060 Ti|15.1 GiB|0.5 GiB|1017K|4|8916 t/s|201 t/s|
|RTX 5070 Ti|15.1 GiB|0.5 GiB|999K|1|6097 t/s|117 t/s|
|RTX 5070 Ti|15.1 GiB|0.5 GiB|999K|4|17303 t/s|318 t/s|

## Cat!

The following cat was produced by `yeasah/Ornith-1.5-9B-exl3` at `4.00bpw` on a RTX 5060 Ti in 78.73 seconds, using 5692 tokens (including thoughts, if any)


<img src="cats/Ornith-1.5-9B.svg" alt="meow">