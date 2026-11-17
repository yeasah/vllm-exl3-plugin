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

Be patient the first time running a new model: vLLM does a considerable amount of compilation and tuning, and it may take a surprising amount of time. If the requested model and revision is not already in the hf cache, it will also take time to download this, and vLLM does not report progress during this time. Subsequent runs after this is all done will be much faster.

# Qwen3.6-35B-A3B

[Qwen3.6-35B-A3B](http://hf.co/yeasah/Qwen3.6-35B-A3B-exl3) at 4.00bpw, with offloaded vision and expert cache using 16GiB of host RAM. 

|Platform|VRAM|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|
|RTX 3060|10.32 GiB|265K|1|1501.68 t/s|40.25 t/s|
|RTX 3060|10.32 GiB|265K|4|2634.26 t/s|82.68 t/s|
|RTX 5070 ti|15.2 GiB|269K|1|5058.51 t/s|100.78 t/s|
|RTX 5070 ti|15.2 GiB|269K|4|8608.08 t/s|214.97 t/s|

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

# Qwen3.6-35B-A3B w/BF16 KV

[Qwen3.6-35B-A3B](http://hf.co/yeasah/Qwen3.6-35B-A3B-exl3) at 4.00bpw, with a 128K unquantized KV cache, offloaded vision and expert cache using 16GiB of host RAM, and MTP.

|Platform|VRAM|KV|Concurrent Req.|Mean Prefill|Mean Output|Draft Acceptance Rate|
|---|---|---|---|---|---|---|
|RTX 3060|10.32 GiB|133K|1|1387.96 t/s|40.88 t/s|49.92%|
|RTX 3060|10.32 GiB|133K|4|3268.76 t/s|50.20 t/s|49.72%|
|RTX 5070 ti|15.2 GiB|132K|1|4974.37 t/s|143.27 t/s|65.03%|
|RTX 5070 ti|15.2 GiB|132K|4|12419.21 t/s|236.00 t/s|56.23%|

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
speculative_config: '{"method":"mtp","num_speculative_tokens":3,"revision":"4.00bpw"}'
```

# Ornith-1.5-9B

[Ornith-1.5-9B](http://hf.co/yeasah/Ornith-1.5-9B-exl3) at 5.00bpw, with offloaded vision using ~1GiB of host RAM.

|Platform|VRAM|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|
|RTX 3060|10.32 GiB|498K|1|1311.86 t/s|44.87 t/s|
|RTX 3060|10.32 GiB|498K|4|2781.77 t/s|116.80 t/s|
|RTX 5070 ti|15.2 GiB|986K|1|4453.80 t/s|102.00 t/s|
|RTX 5070 ti|15.2 GiB|986K|4|7403.76 t/s|307.33 t/s|
 
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

# Qwen3.8-27B

[Qwen3.8-27B](http://hf.co/yeasah/Qwen3.8-27B-exl3) at 2.50bpw. Roughly equivalent in quality to unsloth's `UD-IQ3_XXS`, see model card for more info. The RTX 5070 ti benchmarks are using a revision of `3.50bpw-H5` instead, which is a higher quality quant; roughly equivalent in quality to unsloth's `UD-IQ4_XS`

|Platform|VRAM|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|
|RTX 3060|10.8 GiB|46K|1|371.76 t/s|21.87 t/s|
|RTX 3060|10.8 GiB|46K|4|647.28 t/s|43.20 t/s|
|RTX 5070 ti|15.2 GiB|114K|1|1395.99 t/s|46.47 t/s|
|RTX 5070 ti|15.2 GiB|114K|4|2139.79 t/s|124.83 t/s|

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

# MiniCPM5-2B

[MiniCPM5-2B](http://hf.co/yeasah/MiniCPM5-2B-exl3) at 4.00bpw, with an unquantized KV cache. This will run just about anywhere, if slowly.

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