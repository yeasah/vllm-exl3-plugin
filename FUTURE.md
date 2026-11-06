# Future

Curiosities, lower-priority work and speculative ideas: recorded so they are not lost, kept out of
TODO.md so they do not crowd the active queue. An entry here is a pointer and a reason, nothing more;
the evidence lives in the note it links. When one becomes active, it moves to TODO.md under the same slug.

## Investigations

- `kvarn-accuracy` — Does KVarN actually match FP16 accuracy? A claim worth testing before relying on it. [docs/kvarn.md](docs/kvarn.md)
- `qbench-noise-floor` — Self-noise-floor support for qbench's `vllm` engine, so served-path runs get a floor like the native ones. [docs/qbench.md](docs/qbench.md)
- `mtp-turboquant` — Zero MTP draft acceptance under TurboQuant KV, and output that does not survive it. [docs/turboquant-kv.md](docs/turboquant-kv.md)
- `kv-nvfp4` — 4-bit KV without TurboQuant's costs: NVFP4 KV on cards that support it. [docs/turboquant-kv.md](docs/turboquant-kv.md)
- `transformers-backend` — Serving models vLLM does not implement: it works; what is left is offering the remaining half-patch upstream. [docs/transformers-backend.md](docs/transformers-backend.md)
- `gemma4-e2b` — Quantizing gemma-4 E2B/E4B, tied and PLE-carrying. Shares ground with `engram-ple`. [docs/embeddings.md](docs/embeddings.md)
- `gguf-our-imatrix` — GGUF quantized from our calibration: is the gap to EXL3 the data or the format? [docs/qbench.md](docs/qbench.md)
- `gemma4-recipe` — Recipes wreck gemma-4. The noted next step (BOS) has been tried; no evident lead. [docs/qbench.md](docs/qbench.md)
- `turboquant-sliding-window` — Laguna serves with TurboQuant KV, patched. Left: tq4 quality on qbench, gemma-4 and Muse-Glimmer, and reporting upstream. [docs/turboquant-kv.md](docs/turboquant-kv.md)
- `turboquant-boundary-tax` — TurboQuant's first/last-layer protection: layer 0 is the whole effect on Qwen3-4B. Left: long context, a second dense model, throughput. Nothing depends on it. [docs/turboquant-kv.md](docs/turboquant-kv.md)

## Leads from results

- `recall-calibration` — Recall rows in the calibration lift kept by 3-5 points at 3 bpw on MiniCPM5-2B (three
  draws, holds on reworded prompts) and barely at all on Qwen3.8-27B (+1). A lever for small models at low rates.
  [docs/recall.md](docs/recall.md)
- `recall-pool-growth` — Thin domains (medicine, chemistry, computing, non-numeric history) limit small models'
  recall slices; growing the pool there is cheap now that arms score only each model's slice. [docs/recall.md](docs/recall.md)
- `high-rung-own-voice` — Our lead over SC shrinks with rate fastest on own voice and chat; at 5.00 the gap is real
  (3x the floor). Raising own voice to ~37% of calibration rows on Qwen3.8-27B 5.00 (one draw): wild -5.1%, own
  voice -7.0%, multilingual -3.1%, swe +10.0%. The same chat-for-agentic trade as the calibration mix, not free
  headroom; it would take SC's 5.00 wild edge from 1.12 to about 1.06. [docs/calibration.md](docs/calibration.md)

## New directions

- `vllm-lora` — Loading and serving existing LoRAs against EXL3 quants through vLLM's `--lora-modules`; then
  quantized LoRA.
- `vllm-omni` — EXL3 quantization through vllm-omni, starting with its Transformers-compatible components (often a
  Qwen-family text model); then uncalibrated EXL3 of non-text (diffusers) models.
