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

On Windows (Docker Desktop's WSL2 engine), a bind mount reaches the container through a 9P file bridge out of the WSL VM. That's fine for loading, but if `cpu-offload-gb` is too small to hold every offloaded expert, the rest are read from the checkpoint during decode, and every such read crosses the bridge. Measured on Qwen3.6-35B-A3B with 8% of experts on disk: 16 t/s from a bind mount, 21–24 t/s with the cache on the WSL filesystem. If you serve with experts spilling to disk, consider replacing the bind mount with a named volume (`- hf-cache:/root/.cache/huggingface`, plus `hf-cache:` under the volumes section), which will be faster but isn't shared with Windows. The server warns at startup when this applies.

Alongside, the file `vllm_config.yml` should contain the vLLM configuration to be used. See example sections below.

Be patient the first time running a new model: vLLM does a considerable amount of compilation and tuning, and it may take a surprising amount of time. Subsequent runs after this is all done will be much faster.

For additional example configurations including models large and small, see the [recipe index](https://github.com/yeasah/vllm-exl3-plugin/docs/recipes/index.md).

# [Qwen3.6-35B-A3B](http://hf.co/yeasah/Qwen3.6-35B-A3B-exl3)

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
speculative_config: '{"method":"mtp","num_speculative_tokens":2,"revision":"4.00bpw"}'
```
