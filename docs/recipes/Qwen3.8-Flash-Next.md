# [Qwen/Qwen3.8-Flash-Next](https://hf.co/Qwen/Qwen3.8-Flash-Next)

A larger MoE model, and the precursor to the anticipated Qwen4 line. With a combination of host RAM offload, more aggressive quantization, and potentially disk offload it is possible to run this 125B parameter model on a card as low as 16GiB. Reasoning is supported via the `qwen3` parser, and tool calling is supported by the `qwen3_coder` parser.

The KV cache requires 24KiB/token unquantized, and at the maximum model length (256K) this will require 6GiB unquantized or 3GiB for a `kv_cache_dtype` of `fp8`. vLLM does not support other KV data types at this time.

Available EXL3 quants include [turboderp/Qwen3.8-Flash-Next-exl3](https://hf.co/turboderp/Qwen3.8-Flash-Next-exl3)

Example yaml config (this can be supplied to vLLM via the `--config` argument):

## 16GiB

No vision, up to 24GiB of host RAM used to host experts. For more context length, set `max_model_len` to a specific value, at the cost of VRAM expert residency (and thus performance)

Note that the 1.66bpw checkpoint here includes a low-quality 3-bit engram table, and as this is streamed from disk this is not a constraint that binds just because of a smaller VRAM card. It is possible to direct the system to use the engram from a different checkpoint, such as the 8-bit table used in the 2.18bpw, by setting the following environment variable:

`EXL3_NGRAM_REVISION='2.18bpw_h6_ng8'`

```yaml
model: 'turboderp/Qwen3.8-Flash-Next-exl3'
revision: '1.66bpw_h4_ng3'
max_model_len: auto
max_num_seqs: 4
enable_auto_tool_choice: yes
tool_call_parser: qwen3_coder
reasoning_parser: qwen3
kv_cache_dtype: fp8
attention_config: '{"indexer_kv_dtype": "fp8"}'
cpu_offload_gb: 24
cpu_offload_params: experts
language_model_only: yes
```

|Platform|VRAM|Host RAM|Host Disk|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|---|---|
|RTX 5060 Ti|15.1 GiB (21.0%)|18.5 GiB (79.0%)|0.0 GiB (0.0%)|259K|1|1690 t/s|35 t/s|
|RTX 5060 Ti|15.1 GiB (21.0%)|18.5 GiB (79.0%)|0.0 GiB (0.0%)|259K|4|4766 t/s|51 t/s|
|RTX 5070 Ti|15.1 GiB (32.0%)|12.5 GiB (52.0%)|3.6 GiB (16.0%)|35K|1|1437 t/s|31 t/s|
|RTX 5070 Ti|15.1 GiB (32.0%)|12.5 GiB (52.0%)|3.6 GiB (16.0%)|35K|4|605 t/s|54 t/s|

## 24GiB+

Vision-enabled. MTP optional, uncomment `speculative_config` to enable. For more context length, set `max_model_len` to a specific value, at the cost of VRAM expert residency (and thus performance)

```yaml
model: 'turboderp/Qwen3.8-Flash-Next-exl3'
revision: '2.18bpw_h6_ng8'
max_model_len: auto
max_num_seqs: 4
enable_auto_tool_choice: yes
tool_call_parser: qwen3_coder
reasoning_parser: qwen3
kv_cache_dtype: fp8
attention_config: '{"indexer_kv_dtype": "fp8"}'
cpu_offload_gb: 24
cpu_offload_params: 
 - visual
 - experts
mm_processor_kwargs: '{"max_pixels": 1048576}' 
mm_processor_cache_gb: 1
#speculative_config: '{"method":"mtp","num_speculative_tokens":2,"revision":"2.68bpw_h6_ng8"}'
```

|Platform|VRAM|Host RAM|Host Disk|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|---|---|
|RTX 5060 Ti x2|29.8 GiB (56.5%)|14.8 GiB (43.5%)|0.0 GiB (0.0%)|259K|1|2064 t/s|44 t/s|
|RTX 5060 Ti x2|29.8 GiB (56.5%)|14.8 GiB (43.5%)|0.0 GiB (0.0%)|259K|4|6165 t/s|81 t/s|

## Cat!

The following cat was produced by `turboderp/Qwen3.8-Flash-Next-exl3` at `2.18bpw_h6_ng8` on a RTX 5060 Ti x2 in 85.71 seconds, using 3724 tokens (including thoughts, if any)

<img src="cats/Qwen3.8-Flash-Next.svg" alt="meow">