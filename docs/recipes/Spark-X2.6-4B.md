# [XHToken/Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B)

Small but mighty? A truly ridiculous maximum context for a 4B model.

This model requires an external vLLM plugin to add support: [Spark-plugin](https://github.com/XHToken/Spark-plugin.git). It pins vLLM v0.32, so it is important to not install with dependencies or it will downgrade vLLM. It runs well on the current fork or stock vLLM v0.34.

```
git clone https://github.com/XHToken/Spark-plugin.git
pip install --no-deps Spark-plugin
```

The `vllm-exl3-serve` docker image has this plugin pre-installed as of version 0.12.

The KV cache requires 40KiB/token unquantized, and at the maximum model length (1M) this will require 36.4GiB unquantized, 18.2GiB for a `kv_cache_dtype` of `fp8`, or 10.7GiB for `turboquant_4bit_nc` or `nvfp4`

vLLM builds this model's RoPE caches for the full 1M positions whatever `--max-model-len` is set to: 640 MiB of
VRAM, about a quarter of what the 4.00bpw quant loads. The plugin logs it at startup (`RoPE cos/sin caches`). Today a
shorter `--max-model-len` does not reduce it ([upstream.md](../upstream.md), "RoPE caches at the native length").

Available EXL3 quants include [UnstableLlama/Spark-X2.6-4B-exl3](https://hf.co/UnstableLlama/Spark-X2.6-4B-exl3)

## 4GiB+

Example yaml config (this can be supplied to vLLM via the `--config` argument):

```yaml
model: 'UnstableLlama/Spark-X2.6-4B-exl3'
revision: '4.00bpw'
max_model_len: auto
max_num_seqs: 4
enable_auto_tool_choice: yes
tool_call_parser: spark25
reasoning_parser: qwen3
```

|Platform|VRAM|Host RAM|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|---|
|RTX 3060|10.1 GiB|0.2 GiB|603K|1|2514 t/s|89 t/s|
|RTX 3060|10.1 GiB|0.2 GiB|603K|4|7774 t/s|240 t/s|
|RTX 5060 Ti|15.1 GiB|0.2 GiB|613K|1|5334 t/s|122 t/s|
|RTX 5060 Ti|15.1 GiB|0.2 GiB|613K|4|15146 t/s|340 t/s|
|RTX 5070 Ti|15.1 GiB|0.2 GiB|584K|1|6903 t/s|171 t/s|
|RTX 5070 Ti|15.1 GiB|0.2 GiB|584K|4|29278 t/s|460 t/s|

## Cat!

The following cat was produced by `UnstableLlama/Spark-X2.6-4B-exl3` at `4.00bpw` on a RTX 5060 Ti in 128.38 seconds, using 15056 tokens (including thoughts, if any)

<img src="cats/Spark-X2.6-4B.svg" alt="meow">