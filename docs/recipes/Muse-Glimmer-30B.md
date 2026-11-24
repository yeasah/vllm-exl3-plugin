# [meta-models/Muse-Glimmer-30B](https://hf.co/meta-models/Muse-Glimmer-30B)

Meta's 30B dense model. Extremely efficient KV, but ironically only trained for 128K of it. It does, however, allow for much greater KV concurrency in vLLM. Reasoning and tool calling are supported by the `muse_glimmer` parsers.

The KV cache requires 14.8KiB/token unquantized, and at the maximum model length (128K) this will require 1.85GiB unquantized, 0.93GiB for a `kv_cache_dtype` of `fp8`, or 0.54GiB for `turboquant_4bit_nc` or `nvfp4`

Available EXL3 quants include [yeasah/Muse-Glimmer-30B-exl3](https://hf.co/yeasah/Muse-Glimmer-30B-exl3) and [turboderp/Muse-Glimmer-30B-exl3](https://hf.co/turboderp/Muse-Glimmer-30B-exl3)

Note that turboderp uses a quantized vision tower, which saves RAM, but may have a vision quality impact.

Example yaml config (this can be supplied to vLLM via the `--config` argument):

## 16GiB

No vision.

```yaml
model: 'yeasah/Muse-Glimmer-30B-exl3'
revision: '3.00bpw-H5'
max_model_len: auto
max_num_seqs: 4
enable_auto_tool_choice: yes
tool_call_parser: muse_glimmer
reasoning_parser: muse_glimmer
kv_cache_dtype: nvfp4
language_model_only: yes
```

## 24GiB+

Offloaded vision, 4bpw.

```yaml
model: 'yeasah/Muse-Glimmer-30B-exl3'
revision: '4.00bpw'
max_model_len: auto
max_num_seqs: 4
enable_auto_tool_choice: yes
tool_call_parser: muse_glimmer
reasoning_parser: muse_glimmer
kv_cache_dtype: nvfp4
cpu_offload_gb: 4
cpu_offload_params: visual
#speculative-config '{"method": "dflash", "model": "turboderp/Muse-Glimmer-30B-assistant-exl3", "revision": "4.00bpw", "num_speculative_tokens": 15}'
```
