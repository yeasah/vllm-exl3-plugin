# [google/gemma-4-31B-it](https://hf.co/google/gemma-4-31B-it)

Gemma 4 may not be the strongest at coding, but it is quite strong as a conversational model. Reasoning and tool calling are supported via `gemma4` parsers. As compared to `gemma-4-12B-it`, this model is larger, and has a more traditional multimodal setup with a separate vision encoder and no audio encoder.

The KV cache requires 96KiB/token unquantized, and at the maximum model length (256K) this will require 23.92GiB unquantized, 11.96GiB for a `kv_cache_dtype` of `fp8`, or 7.04GiB for `turboquant_4bit_nc` or `nvfp4`

Available EXL3 quants include [turboderp/gemma-4-31B-it-exl3](https://hf.co/turboderp/gemma-4-31B-it-exl3)

Example yaml config (this can be supplied to vLLM via the `--config` argument):

## 24GiB+

```yaml
model: 'turboderp/gemma-4-31B-it-exl3'
revision: '4.00bpw_mul1'
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
mm_processor_cache_gb: 1
#speculative_config: '{"model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}'
```

|Platform|VRAM|Host RAM|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|---|
|RTX 5060 Ti x2|30.0 GiB|0.0 GiB|260K|1|1344 t/s|37 t/s|
|RTX 5060 Ti x2|30.0 GiB|0.0 GiB|260K|4|4002 t/s|99 t/s|

## Cat!

The following cat was produced by `turboderp/gemma-4-31B-it-exl3` at `4.00bpw_mul1` on a RTX 5060 Ti x2 in 25.71 seconds, using 955 tokens (including thoughts, if any)

<img src="cats/gemma-4-31B-it.svg" alt="meow">