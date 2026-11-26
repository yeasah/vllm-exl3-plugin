# [ornith-ai/Ornith-1.5-35B-A3B](https://hf.co/ornith-ai/Ornith-1.5-35B-A3B)

The ornith counterpart to the Qwen MoE. Same settings and caveats as the Qwen MoE on which it is based.

The KV cache requires 20KiB/token unquantized, and at the maximum model length (256K) this will require 5.14GiB unquantized, 2.64GiB for a `kv_cache_dtype` of `fp8`, or 1.51GiB for `turboquant_4bit_nc` or `nvfp4`

Available EXL3 quants include [yeasah/Ornith-1.5-35B-A3B-exl3](https://hf.co/yeasah/Ornith-1.5-35B-A3B-exl3)

## 12GiB+

Vision enabled + up to 16GiB of host RAM offload. To enable MTP, uncomment the `speculative_config` line.

```yaml
model: 'yeasah/Ornith-1.5-35B-A3B-exl3'
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
|---|---|---|---|---|---|---|---|
|RTX 3060|10.1 GiB (39.0%)|10.3 GiB (61.0%)|0.0 GiB (0.0%)|259K|1|2023 t/s|30 t/s|
|RTX 3060|10.1 GiB (39.0%)|10.3 GiB (61.0%)|0.0 GiB (0.0%)|259K|4|5491 t/s|65 t/s|
|RTX 5060 Ti|15.1 GiB (66.0%)|6.3 GiB (34.0%)|0.0 GiB (0.0%)|259K|1|4094 t/s|64 t/s|
|RTX 5060 Ti|15.1 GiB (66.0%)|6.3 GiB (34.0%)|0.0 GiB (0.0%)|259K|4|12374 t/s|132 t/s|
|RTX 5070 Ti|15.1 GiB (62.0%)|6.8 GiB (38.0%)|0.0 GiB (0.0%)|259K|1|6376 t/s|97 t/s|
|RTX 5070 Ti|15.1 GiB (62.0%)|6.8 GiB (38.0%)|0.0 GiB (0.0%)|259K|4|20793 t/s|185 t/s|

## Cat!

The following cat was produced by `yeasah/Ornith-1.5-35B-A3B-exl3` at `4.00bpw` on a RTX 5060 Ti in 36.58 seconds, using 2337 tokens (including thoughts, if any)

## Cat!

<img src="cats/Ornith-1.5-35B-A3B.svg" alt="meow">