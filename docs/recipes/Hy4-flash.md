# [tencent/Hy4-flash](https://hf.co/tencent/Hy4-flash)

**NOTE: This model is not yet supported by a released version of vLLM; additionally, a patch is required for it to work on `sm_120` cards (i.e. consumer Blackwell).**

**NOTE: vision is not currently possible across more than one GPU**

There is a vLLM [Tracking Issue](https://github.com/vllm-project/vllm/issues/69789) which shows the current status of implementation. Currently it is possible to serve, but it requires additional patches beyond what is present in vllm's `main` branch. When a supporting version of vLLM is released, full support will be integrated.

With a combination of host RAM offload, more aggressive quantization, and potentially disk offload it is possible to run this 220B parameter model on a card as low as 16GiB. Reasoning is supported via the `hy_v4` parser, and tool calling is supported by the `hy_v4` parser.

The KV cache requires 53KiB/token unquantized, and at the maximum model length (1M) this will require 51GiB unquantized or 25GiB for a `kv_cache_dtype` of `fp8`. vLLM does not support other KV data types at this time.

Available EXL3 quants include [turboderp/Hy4-flash-exl3](https://hf.co/turboderp/Hy4-flash-exl3)

Example yaml config (this can be supplied to vLLM via the `--config` argument):

## 16-24GiB

No vision, up to 24GiB of host RAM used to host experts. For more context length, set `max_model_len` to a specific value, at the cost of VRAM expert residency (and thus performance)

```yaml
model: 'turboderp/Hy4-flash-exl3'
revision: 'sc_2.00bpw'
max_model_len: 32768
max_num_seqs: 4
enable_auto_tool_choice: yes
reasoning_parser: hy_v4
tool_call_parser: hy_v4
kv_cache_dtype: fp8
cpu_offload_gb: 24
language_model_only: on
cpu_offload_params:
 - experts
```

|Platform|Type|Date|VRAM|Host RAM|Host Disk|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|---|---|---|---|
|RTX 5060 Ti|sc_2.00bpw|2026-11-26|15.6 GiB (12.0%)|24.0 GiB (65.0%)|7.7 GiB (22.0%)|32K|1|609 t/s|13 t/s|
|RTX 5060 Ti|sc_2.00bpw|2026-11-26|15.6 GiB (12.0%)|24.0 GiB (65.0%)|7.7 GiB (22.0%)|32K|4|2012 t/s|17 t/s|

## 32GiB+

Vision-enabled. For more context length, set `max_model_len` to a specific value, at the cost of VRAM expert residency (and thus performance)

```yaml
model: 'turboderp/Hy4-flash-exl3'
revision: 'sc_2.00bpw'
max_model_len: 131072
max_num_seqs: 4
enable_auto_tool_choice: yes
reasoning_parser: hy_v4
tool_call_parser: hy_v4
kv_cache_dtype: fp8
cpu_offload_gb: 24
cpu_offload_params:
 - visual
 - experts
mm_processor_kwargs: '{"max_pixels": 1048576}' 
mm_processor_cache_gb: 1
```

|Platform|Type|Date|VRAM|Host RAM|Host Disk|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|---|---|---|---|
|RTX 5060 Ti x2|sc_2.00bpw|2026-11-26|30.8 GiB (39.0%)|22.6 GiB (61.0%)|0.0 GiB (0.0%)|128K|1|1102 t/s|27 t/s|
|RTX 5060 Ti x2|sc_2.00bpw|2026-11-26|30.8 GiB (39.0%)|22.6 GiB (61.0%)|0.0 GiB (0.0%)|128K|4|3884 t/s|43 t/s|

## SVGs

The following cat was produced by `turboderp/Hy4-flash-exl3` at `sc_2.00bpw` on a RTX 5060 Ti x2 in 175.98 seconds, using 5175 tokens (including thoughts, if any)

<img src="cats/Hy4-flash.svg" alt="Draw a picture of a cute cat using SVG">
