# [google/gemma-4-12B-it](https://hf.co/google/gemma-4-12B-it)

Gemma 4 may not be the strongest at coding, but it is quite strong as a conversational model. Reasoning and tool calling are supported via `gemma4` parsers. 

Gemma 4 12B is particularly unusual in that it has a "unified" multimodal architecture, meaning it does not have a separate vision or audio encoder, and so it is not possible to offload vision in any meaningful sense. `--language-model-only` is still useful if no vision is required, but it does not save a substantial amount of VRAM. It is also the only model on this list that can accept audio.

Gemma 4 has some unusual characteristics that make serving a little trickier:

 - Gemma 4 has varying head dimensions across its layers. Except on recent datacenter-class cards, this will result in the following message at startup `Gemma4 model has heterogeneous head dimensions {'sliding_attention': 256, 'full_attention': 512}. FA4 not available, forcing TRITON_ATTN backend.`, and the forced use of triton attention will preclude or complicate the use of KV cache quantization other than `fp8`. TurboQuant is possible in some configurations, with considerable setup.
 - When vision is enabled, `max-num-batched-tokens` must be at least 2496, and if you try to run it with a lower value, the following error will be raised at startup: `Chunked MM input disabled but max_tokens_per_mm_item (2496) is larger than max_num_batched_tokens (512). Please increase max_num_batched_tokens.`
 - When vision is enabled, a load-time OOM may happen if expandable segments are not enabled (which can be done by setting the `PYTORCH_CUDA_ALLOC_CONF` environment variable to the value `expandable_segments:True`)

The KV cache requires 22.8KiB/token unquantized, and at the maximum model length (256K) this will require 5.57GiB unquantized, 2.78GiB for a `kv_cache_dtype` of `fp8`, or 1.68GiB for `turboquant_4bit_nc` or `nvfp4`

Available EXL3 quants include [yeasah/gemma-4-12B-it-exl3](https://hf.co/yeasah/gemma-4-12B-it-exl3)

Example yaml config (this can be supplied to vLLM via the `--config` argument):

## 12GiB+

```yaml
model: 'yeasah/gemma-4-12B-it-exl3'
revision: '4.00bpw'
max_model_len: auto
max_num_seqs: 4
max_num_batched_tokens: 2496
enable_auto_tool_choice: yes
tool_call_parser: gemma4
reasoning_parser: gemma4
kv_cache_dtype: fp8
#speculative_config: '{"model":"google/gemma-4-12B-it-assistant","num_speculative_tokens":4}'
```
