# TriAttention: a KV eviction method we could not get to run truthfully

TriAttention (MIT/NVIDIA/ZJU, [arXiv 2604.04921](https://arxiv.org/abs/2604.04921),
Apache-2.0) is a training-free, decode-time KV-cache *eviction* method: it scores
cached tokens by a trigonometric importance measure derived from offline per-head
query statistics, keeps the top `budget`, and physically compacts the cache. It
claims 10.7x KV compression and 2.5x throughput on long reasoning "with no
accuracy loss", and NVIDIA merged it into TensorRT-LLM in August 2026.

We investigated it, fixed four defects in its vLLM integration, got compaction
running, and stopped. This note is why, and what to check first if anyone
returns.

## Why we looked

TurboQuant's prefill transient is structural and we declined to fix it (see
[turboquant-kv.md](turboquant-kv.md), [upstream.md](upstream.md)). Eviction is a
different lever: unlike quantization or offload, it changes what a request
*needs* rather than where its KV lives, so it is the one mechanism that can
genuinely decouple declared context from KV bytes. TriAttention also patches
vLLM's context-window check so `max_model_len` can exceed KV capacity, promising
to meet the target by evicting -- exactly the property a 16 GiB appliance wants.

Worth noting for the upstream thread: **eviction is pluggable in vLLM today,
quantization is not.** Eviction rides existing scheduler/worker surfaces through
`vllm.general_plugins`; quantization needs a new cache dtype and hits the closed
`CacheDType` Literal. The door is shut on one enum, not on KV management.

## Three disqualifiers, all visible before any debugging

1. **Prefill cost is unchanged and prefix caching must be off.** The integration
   documents that prefix caching produces incorrect hits on compressed entries.
   For multi-turn serving that is the dominant term: a conversation resends its
   history, and without prefix caching every turn re-prefills it. At the ~1200
   t/s we measured on a 27B, a 40K conversation costs **~33 s of prefill per
   turn**. Compression saves memory; what this spends is latency.
2. **The evaluation is math reasoning only.** `triattention/evaluation/` is
   `eval_math.py`, `grader.py`, `latex2sympy`; the README names AIME and
   MATH-500 and no retrieval benchmark at all. Long *reasoning* means a short
   prompt and a long generated chain of thought -- context is mostly the model's
   own recent output, the friendliest possible regime for eviction. NIAH inverts
   every one of those properties, and is untested.
3. **Neither "prefill" nor "input" appears once in the README.** What a project
   omits is a design statement. Asking which axis a competitor does not discuss
   is free and would have set expectations correctly before any of the below.

## The vLLM integration is five months and 4,709 commits stale

vLLM support was merged when vLLM was at **v0.19.0 (2026-04-02)**. We ran against
v0.28; that is 15 releases and 4,709 commits later. Six instances of upstream
drift, found one at a time, each masking the next:

| # | drift | effect | fixed |
|---|---|---|---|
| 1 | `EngineCore.log_iteration_details` -> `capture_iteration_details` | integration would not load | yes |
| 2 | `AttentionLayerBase.bind_kv_cache` stores a tensor, not a list | resolver skipped every layer -> `no_compactable_groups` | yes |
| 3 | `BlockTable.compute_slot_mappings` gained `num_tokens_padded` | TypeError at first decode | yes |
| 4 | `is_ec_consumer` replaced by a `getattr` for a nonexistent `is_ec_producer` | branch unconditional | no |
| 5 | `scheduler.schedule()` no longer passed `_should_throttle_prefills()` | prefill throttling disabled | no |
| 6 | KV cache layout: K/V packed into the content dim (#44455) | compaction reads `2*head_size` as the head count | no |

Fixes 1-3 are on branch `calibrate-streaming-exl3` in `~/git/triattention`, all
written to tolerate both old and new shapes rather than pin a version. 4 and 5
cannot affect an accuracy measurement on a single node without spec decode. 6 is
the wall.

**v0.25.1 is the layout boundary** -- the newest release before #44455 (2026-07-11,
first shipped in v0.26.0). On v0.25.1 compaction runs; on v0.26+ it cannot.

## Three traps, each worth a day

**`--enable-prefix-caching false` enables prefix caching.** The flag is an
argparse `BooleanOptionalAction` with `nargs=0`, so a following `false` is a
stray positional and the flag is set **True**. `--no-enable-prefix-caching` is
the only spelling that disables it. TriAttention's own docs give the broken form,
so anyone following them runs precisely the configuration they are warned
against -- and only notices once compaction works well enough to corrupt
something.

**vLLM configures only the `vllm` logger tree.** `init_logger("triattention...")`
creates a logger outside it, inheriting root at `WARNING`. Every `logger.info`
vanishes while the one `logger.warning` in the same file prints, which makes a
working module look half-dead. We spent hours concluding "TriAttention is
unwired" when it was running correctly and silently. Fix with
`VLLM_LOGGING_CONFIG_PATH` pointing at a config that adds a `triattention`
logger; one is at `~/ckpt/vllm_logging_triattn.json`. The tell was visible from
the first log line -- a TriAttention message with no `INFO ... [file.py:NNN]`
prefix never went through vLLM's handler.

**Two config files disagree on `protect_prefill`.** `vllm/core/config.py`
defaults it `True`; `vllm/runtime/config.py` -- the one the runtime path reads --
defaults it `False`, and nothing bridges it. With it off the prompt is evictable,
so at a small budget compaction discards the user's actual question. Likewise
`enable_kv_usage_trigger` defaults `False` and is neither bridged nor defaulted,
so nothing watches pool utilization: with only the per-request length trigger, a
pool can fill to 100% with every request below its own threshold. Set
`TRIATTN_RUNTIME_ENABLE_KV_USAGE_TRIGGER=1`.

## Where it actually fails, and the reproducer

On v0.25.1 with the fixes, compaction runs and reclaims blocks correctly:

    compression applied ... before=8318 after=8192 reclaimed_blocks=8
    FREE_BLOCKS: gid=0 freed=8 kept=512 new=0

and generation collapses immediately afterwards.

**126 tokens of 8318 -- 1.5% of the context -- and coherence is gone.** That is
the whole diagnosis. No eviction policy, however badly chosen, makes a model
incoherent by dropping 1.5% of its tokens. Selection is exonerated; the physical
compaction corrupts cache state. Every tuning knob (`kv_budget`,
`protect_prefill`, trigger thresholds) governs *which* tokens are dropped and
therefore cannot help.

Afterwards the cache immediately reports back above budget (`actual_kv=8319`)
and every further attempt is suppressed by `batch-queue dedup`, so it grows
unchecked while the output is already ruined -- the observed "spirals while
continuously skipping".

**The minimal test, if anyone returns**: set the budget just under the context so
exactly one small compaction fires, and check coherence. If a 1-2% eviction
breaks it, the fault is in the relocation, not the policy. Finding it means
instrumenting the compaction to verify surviving tokens land where the block
table and slot mapping claim.

`TRIATTN_DEBUG_GROUP_PIPELINE=1` (added by us) reports group counts, resolved
group ids and the token/budget arithmetic, then names which skip fired.

## Verdict

Not adopted. The three disqualifiers stand on their own for an interactive
appliance; the implementation defect is on top. Recovering the vLLM path means
targeting ~v0.19.0 -- a stack from April that will not load our plugin and that
nobody would deploy. That is archaeology, not a baseline.

**If the accuracy claim matters, ask TensorRT-LLM.** Their integration merged
2026-08-04, is actively maintained, is an independent implementation rather than
a port of this one, and consumes the same calibration `.pt`. "Does trigonometric
eviction preserve accuracy" does not require our stack to answer, and it is the
only remaining question that would change this verdict.

## What survived: the calibration work

Independent of any of the above, since it runs through `transformers`. See the
`calibrate-streaming-exl3` branch. Two bugs fixed in their calibration script:

- **`q_norm` was never applied.** transformers does `q_norm(q_proj(x))` then
  RoPE; the script recovered the pre-norm query. Affects every Qwen3-family
  model -- and **Qwen3-8B is TensorRT-LLM's own worked example**, on a model
  otherwise fully in scope. Measured effect below.
- **Gated attention split wrongly.** With `attn_output_gate`, `q_proj` emits
  query and gate interleaved *per head*, not as a chunk of the flat projection.

And three capability additions: `--stream` materializes one decoder block at a
time (Qwen3-8B bf16 calibrates at **3.5 GiB peak against 16.4 GB of weights**, so
model size no longer bounds a run, and output is bit-identical); EXL3 checkpoints
load directly via `dense_weight`; block-scaled embeddings decode lazily from the
host.

The measurement that answered the original question -- should you calibrate on
bf16 or on the checkpoint you serve -- comparing arms by the **eviction decisions
they cause** rather than by distance between statistics (keep-5% kept-set
overlap, Llama-3.2-1B at 32K):

| source of variation | overlap |
|---|---|
| `q_norm` skipped (the bug) | **39.3%** |
| calibration text: source code | 86.9% |
| calibration text: other prose | 91.1% |
| weights at 2.0bpw | 94.0% |
| same corpus, different 32K window | 97.1% |
| weights at 3.0bpw | 97.6% |

So: **calibrate on the checkpoint you serve** -- at 3 bits quantization perturbs
calibration *less* than reading a different window of the same text. **Control
the domain** of the calibration text, which matters ~6x more than 3-bit
quantization and which the guide ("a Wikipedia article, a book chapter, or a
source code file are all good choices") treats as interchangeable. **And fix
`q_norm`**, which changes six of every ten kept tokens.

An L2 distance between stat files orders these arms correctly but compresses the
scale badly at the top, and the two disagreed on one adjacent pair at 8K that 32K
separates. `scripts/compare_calibrations.py` does the decision-space version.
