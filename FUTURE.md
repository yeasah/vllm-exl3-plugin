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
- `gemma4-e2b` — Quantizing gemma-4 E2B/E4B, tied and PLE-carrying. Shares ground with `engram-ple`. exllamav3 PR #527 adds partial support (open, 2026-11-07). [docs/embeddings.md](docs/embeddings.md)
- `gguf-our-imatrix` — GGUF quantized from our calibration: is the gap to EXL3 the data, the allocation or the format? A measured per-tensor GGUF claims large gains on a 0.8B (2026-11-07). [docs/qbench.md](docs/qbench.md)
- `gemma4-recipe` — Recipes wreck gemma-4. The noted next step (BOS) has been tried; no evident lead. [docs/qbench.md](docs/qbench.md)
- `turboquant-sliding-window` — Laguna serves with TurboQuant KV, patched. Left: tq4 quality on qbench, gemma-4 and Muse-Glimmer, and reporting upstream. [docs/turboquant-kv.md](docs/turboquant-kv.md)
- `turboquant-boundary-tax` — TurboQuant's first/last-layer protection: layer 0 is the whole effect on Qwen3-4B. Left: long context, a second dense model, throughput. Nothing depends on it. [docs/turboquant-kv.md](docs/turboquant-kv.md)
- `sm75-serving` — The plugin serves on Turing: Qwen3-0.6B on a GTX 1650 with an sm_75 image (2026-11-07). Left:
  a card that can run something useful (a T4), KV quantization without FP8, and the README's 8.0+ claim. [docs/pre-ampere.md](docs/pre-ampere.md)
- `yaqa-vram-pr` — exllamav3 PR #395 cuts the YAQA factor pass's VRAM (the 27B's needed 4x48 GB). Try it before it
  merges when the quant pipeline next probes new models. [docs/yaqa.md](docs/yaqa.md)

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

- `metadata-refresh` — Ship a base model's metadata fix (chat template, tokenizer config, generation config) without a
  full re-run. The template shapes everything downstream (calibration, measurements, YAQA rows, eval traces), so a
  representative update is a re-do; the compromise is to copy the newer files into each branch only when the weight and
  `config.json` blobs are identical to the pinned commit, record weights and metadata commits apart, and add a card
  note that the quants were made and measured with the earlier metadata, saying whether rendering changed (eval
  prompts rendered both ways, token ids compared). Files we override
  (`generation_config.json` under a sampling profile's `publish`) need their own check: a base update can fix what the
  override worked around (Ornith-9B gaining a config, the MoE's starting to match its card), so the profile's reason is
  re-tested against the new file rather than the override carried forward. The appliance can offer the same for anyone's quants at serve time.
- `qbench-fp8-fixture` — A simulated FP8 threshold line for models with no FP8 or AWQ release on the Hub (Ornith-1.5-9B
  has neither; the card then has nothing familiar to read the ladder against). Not a stand-in for a missing artifact:
  vLLM's online `--quantization fp8` serves exactly this from a bf16 checkpoint with one flag. Simulate what that path
  does (per-tensor weight scale as far as known; check its current defaults), which qbench's `fake_quantize` already
  has as `fp8_tensor`; the work is a `weight_quant` option applying it to quantizable linears in the transformers
  streaming engine, with embedding, head and norms left out. Validate against Qwen3.6-35B-A3B's real FP8 release,
  already scored. Like the published FP8 lines, it measures weights under bf16 compute. Wait for more cards to want it.
- `qbench-vllm-engine` — Bring qbench's `vllm` engine up to par, so served formats with no Transformers path that
  matches how they serve (NVFP4 runs W4A4 on Blackwell, W4A16 elsewhere) can be scored as served. Three pieces, all
  in docs/qbench.md: a child process per vLLM arm (designed there: engine and reduction in the child, reference
  logits by path, statistics back), which cures the per-run leak that makes multi-arm runs OOM; the Qwen3.5-family
  mis-scoring (KLD 10.26 against 0.013, traced to the patched `compute_topk_scores` path, the self-consistency probe
  as its guard); and a noise floor like the other engines'. Little used today because of the fragility; AWQ through
  Transformers stood in for NVFP4 on the Qwen3.6-35B-A3B card (2026-11-11).
- `small-card-models` — Find models worth publishing for 4-6 GiB cards. A sub-3 rung of a mid-size model is the wrong
  answer there: Ornith-1.5-9B's 2.50 kept 70% of recall facts and still didn't bring a 4 GiB card in reach, while 8 GiB
  runs its 3 bpw fine (2026-11-11). The right one is a smaller model at 3 bpw and up. Candidates include the new LFM3
  models, which need the exllamav3 fork bumped before they can be converted.
- `vllm-lora` — Loading and serving existing LoRAs against EXL3 quants through vLLM's `--lora-modules`; then
  quantized LoRA.
- `vllm-omni` — EXL3 quantization through vllm-omni, starting with its Transformers-compatible components (often a
  Qwen-family text model); then uncalibrated EXL3 of non-text (diffusers) models.
