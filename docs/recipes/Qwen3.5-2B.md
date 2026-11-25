# [Qwen/Qwen3.5-2B](https://hf.co/Qwen/Qwen3.5-2B)

A small, but still usefully sized entry in the Qwen lineup. 2B parameters, with vision. Systems that can run this might be better off considering the considerably more capable but still compact [Qwen3.5-4B](Qwen3.5-4B.md). Reasoning is supported via the `qwen3` reasoning parser, and tool calling via the `qwen3_coder` parser. There are no default sampling parameters defined in the base model, but the EXL3 quant linked below has this corrected so it will include the recommended sampling parameters by default.

The KV cache requires 12KiB/token unquantized, and at the maximum model length (256K) this will require 3.04GiB unquantized, 1.54GiB for a `kv_cache_dtype` of `fp8`, or 0.89GiB for `turboquant_4bit_nc` or `nvfp4`

Available EXL3 quants include [yeasah/Qwen3.5-2B-exl3](https://hf.co/yeasah/Qwen3.5-2B-exl3)

Example yaml config (this can be supplied to vLLM via the `--config` argument):

## 4GiB+

Offloaded vision enabled. To enable MTP, uncomment the `speculative_config` line.

```yaml
model: 'yeasah/Qwen3.5-2B-exl3'
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
|RTX 3060|10.1 GiB|0.2 GiB|2334K|1|6056 t/s|156 t/s|
|RTX 3060|10.1 GiB|0.2 GiB|2334K|4|17406 t/s|389 t/s|
|RTX 5060 Ti|15.1 GiB|0.2 GiB|3614K|1|12477 t/s|210 t/s|
|RTX 5060 Ti|15.1 GiB|0.2 GiB|3614K|4|36104 t/s|572 t/s|
|RTX 5070 Ti|15.1 GiB|0.2 GiB|3566K|1|22910 t/s|274 t/s|
|RTX 5070 Ti|15.1 GiB|0.2 GiB|3566K|4|67049 t/s|766 t/s|

## Cat!

The following "cat" was produced by `yeasah/Qwen3.5-2B-exl3` at `4.00bpw` on a RTX 5060 Ti in 7.82 seconds, using 1632 tokens (including thoughts, if any)

<img src="cats/Qwen3.5-2B.svg" alt="meow">