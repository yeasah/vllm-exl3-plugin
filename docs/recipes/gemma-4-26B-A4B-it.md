# [google/gemma-4-26B-A4B-it](https://hf.co/google/gemma-4-26B-A4B-it)

Gemma 4 may not be the strongest at coding, but it is quite strong as a conversational model. Reasoning and tool calling are supported via `gemma4` parsers. As compared to `gemma-4-12B-it`, this model has a more traditional multimodal setup with a separate vision encoder and no audio encoder, and is one of the smaller mixture-of-experts models in this class.

The KV cache requires 24KiB/token unquantized, and at the maximum model length (256K) this will require 5.98GiB unquantized, 2.99GiB for a `kv_cache_dtype` of `fp8`, or 1.76GiB for `turboquant_4bit_nc` or `nvfp4`

Available EXL3 quants include [turboderp/gemma-4-26B-A4B-it-exl3](https://hf.co/turboderp/gemma-4-26B-A4B-it-exl3)

## 8GiB+

```yaml
model: 'turboderp/gemma-4-26B-A4B-it-exl3'
revision: '4.10bpw'
max_model_len: auto
max_num_seqs: 4
max_num_batched_tokens: 2496
enable_auto_tool_choice: yes
tool_call_parser: gemma4
reasoning_parser: gemma4
kv_cache_dtype: fp8
cpu_offload_gb: 8
cpu_offload_params:
 - visual
 - experts
mm_processor_kwargs: '{"max_pixels": 1048576}' 
mm_processor_cache_gb: 1
#speculative_config: '{"model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}'
```

|Platform|VRAM|Host RAM|Host Disk|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|---|---|
|RTX 5060 Ti|15.1 GiB (71.0%)|4.8 GiB (29.0%)|0.0 GiB (0.0%)|256K|1|4354 t/s|65 t/s|
|RTX 5060 Ti|15.1 GiB (71.0%)|4.8 GiB (29.0%)|0.0 GiB (0.0%)|256K|4|12369 t/s|107 t/s|
|RTX 5060 Ti|15.1 GiB (19.0%)|10.9 GiB (81.0%)|0.0 GiB (0.0%)|768K|1|4134 t/s|42 t/s|
|RTX 5060 Ti|15.1 GiB (19.0%)|10.9 GiB (81.0%)|0.0 GiB (0.0%)|768K|4|11669 t/s|50 t/s|
|RTX 5070 Ti|15.1 GiB (69.0%)|5.1 GiB (31.0%)|0.0 GiB (0.0%)|256K|1|6129 t/s|66 t/s|
|RTX 5070 Ti|15.1 GiB (69.0%)|5.1 GiB (31.0%)|0.0 GiB (0.0%)|256K|4|20703 t/s|157 t/s|

## Cat!

The following cat was produced by `turboderp/gemma-4-26B-A4B-it-exl3` at `4.10bpw` on a RTX 5060 Ti in 17.56 seconds, using 1128 tokens (including thoughts, if any)

<img src="cats/gemma-4-26B-A4B-it.svg" alt="meow">