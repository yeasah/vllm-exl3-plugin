# [google/gemma-4-26B-A4B-it](https://hf.co/google/gemma-4-26B-A4B-it)

Gemma 4 may not be the strongest at coding, but it is quite strong as a conversational model. Reasoning and tool calling are supported via `gemma4` parsers. As compared to `gemma-4-12B-it`, this model has a more traditional multimodal setup with a separate vision encoder and no audio encoder, and is one of the smaller mixture-of-experts models in this class.

The KV cache requires 24KiB/token unquantized, and at the maximum model length (256K) this will require 5.98GiB unquantized, 2.99GiB for a `kv_cache_dtype` of `fp8`, or 1.76GiB for `turboquant_4bit_nc` or `nvfp4`

Available EXL3 quants include [turboderp/gemma-4-26B-A4B-it-exl3](https://hf.co/turboderp/gemma-4-26B-A4B-it-exl3)

## 8GiB+

```yaml
model: 'yeasah/gemma-4-26B-A4B-it-exl3'
revision: '4.00bpw'
max_model_len: auto
max_num_seqs: 4
max_num_batched_tokens: 2496
enable_auto_tool_choice: yes
tool_call_parser: gemma4
reasoning_parser: gemma4
kv_cache_dtype: fp8
cpu_offload_gb: 1
cpu_offload_params: visual
mm_processor_kwargs: '{"max_pixels": 1048576}' 
#speculative_config: '{"model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}'
```

## Cat!

<img src="cats/gemma-4-26B-A4B-it.svg" alt="meow">