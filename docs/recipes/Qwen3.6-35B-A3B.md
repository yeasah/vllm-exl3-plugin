# [Qwen/Qwen3.6-35B-A3B](https://hf.co/Qwen/Qwen3.6-35B-A3B)

The most recent Qwen MoE available. Reasoning is supported via the `qwen3` reasoning parser, and tool calling via the `qwen3_coder` parser. The model card calls out different sampling parameters for different use cases, so make sure you are using the right settings.

The KV cache requires 20KiB/token unquantized, and at the maximum model length (256K) this will require 5.14GiB unquantized, 2.64GiB for a `kv_cache_dtype` of `fp8`, or 1.51GiB for `turboquant_4bit_nc` or `nvfp4`

Available EXL3 quants include [yeasah/Qwen3.6-35B-A3B-exl3](https://hf.co/yeasah/Qwen3.6-35B-A3B-exl3)

## 12GiB+

Vision enabled + up to 16GiB of host RAM offload. To enable MTP, uncomment the `speculative_config` line.

```yaml
model: 'yeasah/Qwen3.6-35B-A3B-exl3'
revision: 4.00bpw
max_model_len: auto
max_num_seqs: 4
enable_auto_tool_choice: yes
tool_call_parser: qwen3_coder
reasoning_parser: qwen3
kv_cache_dtype: nvfp4
cpu_offload_gb: 16
cpu_offload_params:
 - visual
 - experts
mm_processor_kwargs: '{"max_pixels": 1048576}'
mm_processor_cache_gb: 1
#speculative_config: '{"method":"mtp","num_speculative_tokens":2,"revision":"4.00bpw"}'
```

|Platform|VRAM|Host RAM|Host Disk|KV|Concurrent Req.|Mean Prefill|Mean Output|
|Platform|VRAM|Host RAM|Host Disk|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|---|---|
|RTX 3060|10.1 GiB (39.0%)|10.3 GiB (61.0%)|0.0 GiB (0.0%)|259K|1|2066 t/s|29 t/s|
|RTX 3060|10.1 GiB (39.0%)|10.3 GiB (61.0%)|0.0 GiB (0.0%)|259K|4|5796 t/s|62 t/s|
|RTX 5060 Ti|15.1 GiB (66.0%)|6.3 GiB (34.0%)|0.0 GiB (0.0%)|259K|1|4574 t/s|64 t/s|
|RTX 5060 Ti|15.1 GiB (66.0%)|6.3 GiB (34.0%)|0.0 GiB (0.0%)|259K|4|12877 t/s|138 t/s|
|RTX 5070 Ti|15.1 GiB (62.0%)|6.9 GiB (38.0%)|0.0 GiB (0.0%)|259K|1|6488 t/s|99 t/s|
|RTX 5070 Ti|15.1 GiB (62.0%)|6.9 GiB (38.0%)|0.0 GiB (0.0%)|259K|4|21192 t/s|188 t/s|

MTP Enabled:

|Platform|VRAM|Host RAM|Host Disk|KV|Concurrent Req.|Mean Prefill|Mean Output|Draft|
|---|---|---|---|---|---|---|---|---|
|RTX 3060|10.3 GiB (30.0%)|12.0 GiB (70.0%)|0.0 GiB (0.0%)|259K|1|1894 t/s|45 t/s|2 tok (72.7%)|
|RTX 3060|10.3 GiB (30.0%)|12.0 GiB (70.0%)|0.0 GiB (0.0%)|259K|4|5449 t/s|67 t/s|2 tok (72.5%)|
|RTX 5060 Ti|15.2 GiB (57.0%)|7.9 GiB (43.0%)|0.0 GiB (0.0%)|259K|1|4290 t/s|93 t/s|2 tok (76.1%)|
|RTX 5060 Ti|15.2 GiB (57.0%)|7.9 GiB (43.0%)|0.0 GiB (0.0%)|259K|4|11290 t/s|130 t/s|2 tok (71.1%)|
|RTX 5070 Ti|15.2 GiB (53.0%)|8.5 GiB (47.0%)|0.0 GiB (0.0%)|259K|1|6321 t/s|137 t/s|2 tok (75.2%)|
|RTX 5070 Ti|15.2 GiB (53.0%)|8.5 GiB (47.0%)|0.0 GiB (0.0%)|259K|4|18971 t/s|185 t/s|2 tok (70.5%)|

## Cat!

The following cat was produced by `yeasah/Qwen3.6-35B-A3B-exl3` at `4.00bpw` on a RTX 5060 Ti in 28.89 seconds, using 3025 tokens (including thoughts, if any)

<img src="cats/Qwen3.6-35B-A3B.svg" alt="meow">