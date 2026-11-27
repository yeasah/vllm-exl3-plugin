# Model Recipe Index

This is not an inclusive list of supported models; rather these are example configurations known to work well.

|Model Name|Type|Total Parameters|Active Parameters|Vision|KV KiB/t|Max Context|Fits Card|
|---|---|---|---|---|---|---|---|
|[Hy4-flash](Hy4-flash.md)|MoE|220|12|✅|53|1M|16 GiB+|
|[Qwen3.8-Flash-Next](Qwen3.8-Flash-Next.md)|MoE|125|6|✅|24|256K|16 GiB+|
|[gemma-4-31B-it](gemma-4-31B-it.md)|Dense|31|31|✅|96|256K|24 GiB+|
|[Muse-Glimmer-30B](Muse-Glimmer-30B.md)|Dense|30|30|✅|15|128K|24 GiB+|
|[Qwen3.8-27B](Qwen3.8-27B.md)|Dense|27|27|✅|64|256K|12 GiB+|
|[Qwen3.6-35B-A3B](Qwen3.6-35B-A3B.md)|MoE|35|3|✅|20|256K|12 GiB+|
|[Ornith-1.5-35B-A3B](Ornith-1.5-35B-A3B.md)|MoE|35|3|✅|20|256K|12 GiB+|
|[Laguna-XS-2.1](Laguna-XS-2.1.md)|MoE|33|3|❌|42|256K|12 GiB+|
|[gemma-4-12B-it](gemma-4-12B-it.md)|Dense|12|12|✅|22|256K|12 GiB+|
|[gemma-4-26B-A4B-it](gemma-4-26B-A4B-it.md)|MoE|26|4|✅|24|256K|8 GiB+|
|[Qwen3.5-9B](Qwen3.5-9B.md)|Dense|9|9|✅|32|256K|8 GiB+|
|[Ornith-1.5-9B](Ornith-1.5-9B.md)|Dense|9|9|✅|32|256K|8 GiB+|
|[Spark-X2.6-4B](Spark-X2.6-4B.md)|Dense|4|4|❌|40|1M|4 GiB+|
|[Qwen3.5-4B](Qwen3.5-4B.md)|Dense|4|4|✅|32|256K|4 GiB+|
|[Qwen3.5-2B](Qwen3.5-2B.md)|Dense|2|2|✅|12|256K|4 GiB+|
|[MiniCPM5-2B](MiniCPM5-2B.md)|Dense|2|2|❌|42|128K|4 GiB+|
|[LFM3-1.7B](LFM3-1.7B.md)|Dense|1.7|1.7|❌|12|256K|any|

## Host memory with vision enabled

Configurations with vision set `mm_processor_cache_gb: 1`. vLLM's default is a 4 GiB cache of preprocessed images, kept once in the API server and once in the engine (8 GiB of host RAM in a single-server setup), to skip repeat preprocessing when the same image comes back, as it does across turns of a chat. On a small host, or one giving most of its RAM to offloaded experts, that is RAM taken from the experts for a CPU-side speedup to the encoder path, which these configurations already deprioritize. 1 GiB still holds dozens of images at the `max_pixels` used here. The plugin counts this cache against host RAM when it sizes the expert host tier, so raising it leaves less room for experts. `--language-model-only` configurations need nothing: no images are processed.
