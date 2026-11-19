Serve [EXL3](https://github.com/turboderp-org/exllamav3) quantized models with [vLLM](https://github.com/vllm-project/vllm).

See [vllm-exl3-plugin](https://github.com/yeasah/vllm-exl3-plugin/) for usage details.

Example `docker-compose.yml`:

```yaml
services:
  vllm-exl3-serve:
    image: yeasah/vllm-exl3-serve:latest
    ports:
      - ${HOST_PORT:-8000}:8000
    environment:
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - HF_TOKEN=${HF_TOKEN}
    command: --config /root/vllm_config.yml
    volumes:
      - ./vllm_config.yml:/root/vllm_config.yml
      - ~/.cache/huggingface:/root/.cache/huggingface
      - vllm-exl3-serve-cache:/root/.cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  vllm-exl3-serve-cache:
```
A bind mount to the external environment huggingface cache is included, and a volume for all the other cached items that vLLM generates during operation.

Alongside, the file `vllm_config.yml` should contain the vLLM configuration to be used. See example sections below.

Be patient the first time running a new model: vLLM does a considerable amount of compilation and tuning, and it may take a surprising amount of time. Subsequent runs after this is all done will be much faster.

# [Qwen3.6-35B-A3B](http://hf.co/yeasah/Qwen3.6-35B-A3B-exl3)

 4.00bpw, with offloaded vision and expert cache using 16GiB of host RAM. 

|Platform|VRAM|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|
|RTX 3060|10.32 GiB|265K|1|1600.44 t/s|44.60 t/s|
|RTX 3060|10.32 GiB|265K|4|4461.73 t/s|66.46 t/s|
|RTX 5060 ti|15 GiB|265K|1|3678.27 t/s|63.45 t/s|
|RTX 5060 ti|15 GiB|256K|4|9327.13 t/s|132.70 t/s|
|RTX 5070 ti|14.2 GiB|272K|1|5098.35 t/s|101.63 t/s|
|RTX 5070 ti|14.2 GiB|272K|4|15521.81 t/s|173.26 t/s|

```yaml
model: 'yeasah/Qwen3.6-35B-A3B-exl3'
revision: '4.00bpw'
max_model_len: 262144
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
gpu_memory_utilization: 0.86
```

4.00bpw, with a 128K unquantized KV cache, offloaded vision and expert cache using 16GiB of host RAM, and MTP.

|Platform|VRAM|KV|Concurrent Req.|Mean Prefill|Mean Output|Draft Acceptance Rate|
|---|---|---|---|---|---|---|
|RTX 3060|10.32 GiB|133K|1|1632.31 t/s|41.86 t/s|62.33%|
|RTX 3060|10.32 GiB|133K|4|4375.89 t/s|47.53 t/s|62.01%|
|RTX 5060 ti|15 GiB|132K|1|3646.68 t/s|81.14 t/s|77.38%|
|RTX 5060 ti|15 GiB|132K|4|8737.31 t/s|121.22 t/s|75.64%|
|RTX 5070 ti|15.2 GiB|132K|1|4757.50 t/s|98.98 t/s|66.58%|
|RTX 5070 ti|15.2 GiB|132K|4|12503.21 t/s|204.33 t/s|71.36%|

```yaml
model: 'yeasah/Qwen3.6-35B-A3B-exl3'
revision: '4.00bpw'
max_model_len: 131072
max_num_seqs: 4
enable_auto_tool_choice: yes
tool_call_parser: qwen3_coder
reasoning_parser: qwen3
kv_cache_dtype: auto
cpu_offload_gb: 16
cpu_offload_params:
 - visual
 - experts
mm_processor_kwargs: '{"max_pixels": 1048576}'
gpu_memory_utilization: 0.86
speculative_config: '{"method":"mtp","num_speculative_tokens":2,"revision":"4.00bpw"}'
```

# [Qwen3.8-Flash-Next](http://hf.co/turboderp/Qwen3.8-Flash-Next-exl3) 

16GiB card: 1.66bpw with no vision, fp8 quantized kv cache. 1.66bpw may not be as bad as it sounds: it generally matches `IQ2_XS` in GGUF terms, and this larger model may be more resilient to aggressive quantizaion. The thin benchmarks on the model card suggest more limited capability loss than you might expect, i.e. SWE-bench 76% -> 74%, GPQA-D 91.7% -> 86.4%. See the model card for more details. This configuration requires at least 22.3 GiB of free host RAM; on Windows it cannot be more than 50% of total system RAM, so you must have more than double that in total (practically speaking, 48GiB or more)

|Platform|VRAM|Host RAM|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|---|
|RTX 5060 ti|14.8 GiB|22.3 GiB|268K|1|1522.23 t/s|36.73 t/s|
|RTX 5060 ti|14.8 GiB|22.3 GiB|268K|4|3953.38 t/s|54.27 t/s|

```yaml
model: 'turboderp/Qwen3.8-Flash-Next-exl3'
revision: '1.66bpw_h4_ng3'
max_model_len: 262144
max_num_seqs: 4
enable_auto_tool_choice: yes
tool_call_parser: qwen3_coder
reasoning_parser: qwen3
kv_cache_dtype: fp8
language_model_only: yes
gpu_memory_utilization: 0.96
cpu_offload_gb: 24
cpu_offload_params: experts
attention_config: '{"indexer_kv_dtype": "fp8"}'
```

2x 16GiB cards or 1x 32GiB card: 2.68bpw (about halfway between unsloth's `UD3-IQ3_XXS` and `UD3-IQ4_XS` in quality) with vision, fp8 quantized kv cache, and MTP draft head. For 1x card, remove `tensor_parallel_size: 2` and set `cpu_offload_gb: 28`. For no MTP, remove the `speculative_config` line. `num_speculative_tokens` is an optimization parameter; the typical recipes recommend 3, but 2 gives better performance on these configs. This configuration requires at least 29.8 GiB of free host RAM; on Windows it cannot be more than 50% of total system RAM, so you must have more than double that in total (practically speaking, 64GiB or more)

|Platform|VRAM|Host RAM|MTP|KV|Concurrent Req.|Mean Prefill|Mean Output|Draft Acceptance Rate|
|---|---|---|---|---|---|---|---|---|
|RTX 5060 ti x2|29.8 GiB|26.4 GiB|off|267K|1|2043.71 t/s|40.28 t/s||
|RTX 5060 ti x2|29.8 GiB|26.4 GiB|off|267K|4|4985.20 t/s|76.85 t/s||
|RTX 5060 ti x2|29.8 GiB|26.5 GiB|on|267K|1|1703.02 t/s|58.68 t/s|65.51%|
|RTX 5060 ti x2|29.8 GiB|26.5 GiB|on|267K|4|4799.22 t/s|79.93 t/s|70.67%|

```yaml
model: 'turboderp/Qwen3.8-Flash-Next-exl3'
revision: '2.68bpw_h6_ng8'
tensor_parallel_size: 2
max_model_len: 262144
max_num_seqs: 4
enable_auto_tool_choice: yes
tool_call_parser: qwen3_coder
reasoning_parser: qwen3
kv_cache_dtype: fp8
gpu_memory_utilization: 0.96
cpu_offload_gb: 14
cpu_offload_params:
 - visual
 - experts
mm_processor_kwargs: '{"max_pixels": 1048576}'
attention_config: '{"indexer_kv_dtype": "fp8"}'
speculative_config: '{"method":"mtp","num_speculative_tokens":2,"revision":"2.68bpw_h6_ng8"}'

```

# [Ornith-1.5-9B](http://hf.co/yeasah/Ornith-1.5-9B-exl3)

5.00bpw, with offloaded vision using ~1GiB of host RAM.

|Platform|VRAM|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|
|RTX 3060|10.32 GiB|504K|1|1288.08 t/s|45.76 t/s|
|RTX 3060|10.32 GiB|504K|4|3257.29 t/s|92.27 t/s|
|RTX 5060 ti|15.2 GiB|975K|1|2379.11 t/s|64.08 t/s|
|RTX 5060 ti|15.2 GiB|975K|4|6131.44 t/s|140.10 t/s|
|RTX 5070 ti|15.2 GiB|986K|1|4385.20 t/s|101.39 t/s|
|RTX 5070 ti|15.2 GiB|986K|4|11953.88 t/s|228.85 t/s|
 
```yaml
model: 'yeasah/Ornith-1.5-9B-exl3'
revision: '5.00bpw'
max_model_len: auto
max_num_seqs: 4
enable_auto_tool_choice: yes
tool_call_parser: qwen3_coder
reasoning_parser: qwen3
kv_cache_dtype: nvfp4
cpu_offload_gb: 1
cpu_offload_params: visual
mm_processor_kwargs: '{"max_pixels": 1048576}'
gpu_memory_utilization: 0.86
```

# [Qwen3.8-27B](http://hf.co/yeasah/Qwen3.8-27B-exl3)

2.50bpw. Roughly equivalent in quality to unsloth's `UD-IQ3_XXS`, see model card for more info. The RTX 50x0 ti benchmarks are using a revision of `3.50bpw-H5` instead, which is a higher quality quant; roughly equivalent in quality to unsloth's `UD-IQ4_XS`

|Platform|VRAM|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|
|RTX 3060|10.8 GiB|38K|1|386.75 t/s|21.99 t/s|
|RTX 3060|10.8 GiB|38K|2|643.38 t/s|22.93 t/s|
|RTX 5060 ti|15.2 GiB|108K|1|714.69 t/s|27.45 t/s|
|RTX 5060 ti|15.2 GiB|108K|4|1880.57 t/s|51.09 t/s|
|RTX 5070 ti|14.9 GiB|105K|1|1293.09 t/s|46.38 t/s|
|RTX 5070 ti|14.9 GiB|105K|4|3468.45 t/s|89.47 t/s|

```yaml
model: 'yeasah/Qwen3.8-27B-exl3'
revision: '2.50bpw-H5'
max_model_len: auto
max_num_seqs: 4
enable_auto_tool_choice: yes
tool_call_parser: qwen3_coder
reasoning_parser: qwen3
kv_cache_dtype: nvfp4
language_model_only: yes
gpu_memory_utilization: 0.90
```

# [MiniCPM5-2B](http://hf.co/yeasah/MiniCPM5-2B-exl3)

4.00bpw, with an unquantized KV cache. This will run just about anywhere, if slowly.

|Platform|VRAM|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|
|GTX 1650|2.91 GiB|36K|1|70.30 t/s|7.62 t/s|
|GTX 1650|2.91 GiB|36K|4|127.34 t/s|15.29 t/s|
|RTX 3060|10.32 GiB|202K|1|3770.96 t/s|109.76 t/s|
|RTX 3060|10.32 GiB|202K|4|5903.67 t/s|273.05 t/s|
|RTX 5070 ti|15.2 GiB|340K|1|12354.34 t/s|179.95 t/s|
|RTX 5070 ti|15.2 GiB|340K|4|17819.09 t/s|512.40 t/s|

```yaml
model: 'yeasah/MiniCPM5-2B-exl3'
revision: '4.00bpw'
max_model_len: auto
max_num_seqs: 4
enable_auto_tool_choice: yes
tool_call_parser: minicpm5
gpu_memory_utilization: 0.80
```