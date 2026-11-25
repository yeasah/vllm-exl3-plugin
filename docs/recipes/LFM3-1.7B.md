# [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B)

An extremely small model for its claimed agentic capabilities.

The KV cache requires 12KiB/token unquantized, and at the maximum model length (256K) this will require 3.05GiB unquantized, 1.53GiB for a `kv_cache_dtype` of `fp8`, or 0.90GiB for `turboquant_4bit_nc` or `nvfp4`

Available EXL3 quants include [Trisert/LFM3-1.7B-exl3](https://hf.co/Trisert/LFM3-1.7B-exl3)

## Just about anywhere

```yaml
model: 'Trisert/LFM3-1.7B-exl3'
revision: '4.0bpw-sc'
max_model_len: auto
max_num_seqs: 4
enable_auto_tool_choice: yes
tool_call_parser: qwen3_coder
reasoning_parser: qwen3
kv_cache_dtype: fp8
```

|Platform|VRAM|Host RAM|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|---|
|RTX 3060|10.1 GiB|0.1 GiB|1473K|1|6075 t/s|177 t/s|
|RTX 3060|10.1 GiB|0.1 GiB|1473K|4|18143 t/s|459 t/s|
|RTX 5060 Ti|15.1 GiB|0.1 GiB|1300K|1|13077 t/s|241 t/s|
|RTX 5060 Ti|15.1 GiB|0.1 GiB|1300K|4|36560 t/s|654 t/s|
|RTX 5070 Ti|15.0 GiB|0.1 GiB|1285K|1|22658 t/s|314 t/s|
|RTX 5070 Ti|15.0 GiB|0.1 GiB|1285K|4|69006 t/s|874 t/s|

## Cat!

The following cat was produced by `Trisert/LFM3-1.7B-exl3` at `4.0bpw-sc` on a RTX 5060 Ti in 9.94 seconds, using 2382 tokens (including thoughts, if any)

<img src="cats/LFM3-1.7B.svg" alt="meow">
