# [openbmb/MiniCPM5-2B](https://hf.co/openbmb/MiniCPM5-2B)

A small, but reasonably capable model. No vision.

The KV cache requires 42KiB/token unquantized, and at the maximum model length (256K) this will require 5.25GiB unquantized, 2.63GiB for a `kv_cache_dtype` of `fp8`, or 1.54GiB for `turboquant_4bit_nc` or `nvfp4`

Available EXL3 quants include [yeasah/MiniCPM5-2B-exl3](https://hf.co/yeasah/MiniCPM5-2B-exl3)

## 4GiB+

Example yaml config (this can be supplied to vLLM via the `--config` argument):

```yaml
model: 'yeasah/MiniCPM5-2B-exl3'
revision: '4.00bpw'
max_model_len: auto
max_num_seqs: 4
enable_auto_tool_choice: yes
tool_call_parser: minicpm5
```

## Benchmarks

|Platform|VRAM|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|
|GTX 1650|2.91 GiB|36K|1|70.30 t/s|7.62 t/s|
|GTX 1650|2.91 GiB|36K|4|127.34 t/s|15.29 t/s|
|RTX 3060|10.32 GiB|202K|1|3770.96 t/s|109.76 t/s|
|RTX 3060|10.32 GiB|202K|4|5903.67 t/s|273.05 t/s|
|RTX 5070 ti|15.2 GiB|340K|1|12354.34 t/s|179.95 t/s|
|RTX 5070 ti|15.2 GiB|340K|4|17819.09 t/s|512.40 t/s|
