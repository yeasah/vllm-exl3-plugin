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
kv_cache_dtype: 'fp8'
```

## Benchmarks

|Platform|VRAM|Host RAM|KV|Concurrent Req.|Mean Prefill|Mean Output|
|---|---|---|---|---|---|---|
|RTX 3060|10.1 GiB|0.1 GiB|396K|1|4761 t/s|123 t/s|
|RTX 3060|10.1 GiB|0.1 GiB|396K|4|12942 t/s|340 t/s|
|RTX 5060 Ti|15.0 GiB|0.1 GiB|620K|1|9173 t/s|164 t/s|
|RTX 5060 Ti|15.0 GiB|0.1 GiB|620K|4|24347 t/s|454 t/s|
|RTX 5070 Ti|15.0 GiB|0.1 GiB|612K|1|16854 t/s|200 t/s|
|RTX 5070 Ti|15.0 GiB|0.1 GiB|612K|4|45507 t/s|585 t/s|
|RTX 3060|10.1 GiB|0.1 GiB|396K|1|4761 t/s|123 t/s|
|RTX 3060|10.1 GiB|0.1 GiB|396K|4|12942 t/s|340 t/s|

## Cat!

The following "cat" was produced by `yeasah/MiniCPM5-2B-exl3` at `4.00bpw` on a RTX 5060 Ti in 22.34 seconds, using 3608 tokens (including thoughts, if any)

<img src="cats/MiniCPM5-2B.svg" alt="meow">
