# [meta-models/Muse-Glimmer-30B](https://hf.co/meta-models/Muse-Glimmer-30B)

Meta's 30B dense model. Extremely efficient KV, but ironically only trained for 128K of it. It does, however, allow for much greater KV concurrency in vLLM. Reasoning and tool calling are supported by the `muse_glimmer` parsers.

The KV cache requires 14.8KiB/token unquantized, and at the maximum model length (128K) this will require 1.85GiB unquantized, 0.93GiB for a `kv_cache_dtype` of `fp8`, or 0.54GiB for `turboquant_4bit_nc` or `nvfp4`

NOTE: Currently multi-GPU use (tensor-parallel-size > 1) is not possible on this model.

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

|Platform|VRAM|Host RAM|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|---|
|RTX 5060 Ti|15.1 GiB|0.7 GiB|946K|1|944 t/s|29 t/s|
|RTX 5060 Ti|15.1 GiB|0.7 GiB|946K|4|2703 t/s|78 t/s|
|RTX 5070 Ti|15.0 GiB|0.7 GiB|760K|1|1696 t/s|50 t/s|
|RTX 5070 Ti|15.0 GiB|0.7 GiB|760K|4|5215 t/s|130 t/s|

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

## Cat!

The following cat was produced by `yeasah/Muse-Glimmer-30B-exl3` at `3.00bpw-H5` on a RTX 5060 Ti in 40.07 seconds, using 1168 tokens (including thoughts, if any)

<img src="cats/Muse-Glimmer-30B.svg" alt="meow">