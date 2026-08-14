# Ideas for next steps

By far the biggest win here for the immediate future is a combination of #3 and eiher #2 or #4b -- these two combined will take EXL3 from significantly underperforming competing formats/quantizations on most checkpoints (when given a full and honest accounting) to being as highly competitive as originally advertised.

## 0. Further qbench improvements

So far, we have extended qbench in a couple ways: accounting for embeddings in VRAM tests, and automatic pulling from the huggingface hub for reference and test models.

**Done (2026-08-14): a `vllm` engine.** qbench can now run any vllm-servable model -- this project's own EXL3 plugin, AWQ, GPTQ, FP8, etc. -- through the real `vllm.LLM` offline API, under the same KLD/ppl methodology as the other three engines. The point is comparing the actually-*served* path, not a proxy for it: does vllm+vllm-exl3-plugin reproduce native exllamav3's quality, and how does it stack up against AWQ-via-vllm on the same checkpoint, both served the same way a user would actually run them.

The interesting part was getting full-vocab per-token logits out of vLLM at all: its public `prompt_logprobs` API is built for a UI's top-k display, and even at `prompt_logprobs=-1` (full vocab) still builds one Python object per (position, vocab-entry) downstream -- hundreds of millions of them for one 2048-token row, unusable at qbench's scale. Worked around by keeping vLLM's `EngineCore` in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) and hooking `LogprobsProcessor` (the one place, common to every model-runner variant, where the raw tensor gets pythonized) to capture the real tensor. Streams row-by-row (fires qbench's callback the moment each row finishes, not after the whole batch) rather than holding every row's full-vocab tensor in memory at once, which at qbench's usual scale would be tens of GB.

Validated: reconstructed logits cross-checked against a plain transformers forward pass on Qwen3-0.6B (mean KLD ~0.003, backend-kernel-noise scale); bpw/vram accounting cross-checked against a real AWQ checkpoint (landed within 0.01 GiB of vLLM's own logged checkpoint size) and a real EXL3 checkpoint (bpw_embed=16.0, matching the known unquantized-embedding behavior); end-to-end run on Qwen3-0.6B-exl3 @4.0bpw showed native exllamav3 (ppl 4.6599, kld 0.081316) essentially matching the same checkpoint served through vllm+vllm-exl3-plugin (ppl 4.6064, kld 0.080634) -- different-kernel-path scale, not different-model scale.

**Follow-up (same day): two bugs surfaced by real usage** (rows=10, length=2048 -- the smoke test above only used 2 rows of ~50 tokens, which never exercised either).

1. An OOM no memory knob fixed short of manually shrinking `kv_cache_memory_bytes` to 4 GiB, on a *0.6B* model. Root cause: `prompt_logprobs=-1` makes vLLM's own `compute_topk_scores` do `torch.topk(logits, vocab_size)` once per 1024-token chunk of scored prompt -- k this close to n makes `torch.topk` fall back to something close to a full sort, workspace and all (confirmed in isolation: ~7 GiB transient peak at Qwen3's 152k vocab, ~11.7 GiB at Qwen3.5's 256k, vs. under 1 GiB at k=1). That spike happens *after* vLLM's memory profiler already sized the KV cache, so it's invisible to `--gpu-memory-utilization` and every other normal knob. Fixed by not asking vLLM to do that sort at all: patched `compute_topk_scores` (scoped to the prompt-logprobs path only) to grab its raw input tensor directly and request `prompt_logprobs=1` instead of `-1`, so vLLM's own remaining topk call is a cheap top-1. `max_num_seqs` defaulted to 1 so per-request boundaries fall out for free instead of needing to replicate vLLM's chunked-prefill request-splitting arithmetic by hand.
2. `vram_gb`/`bpw_head` overreporting on tied-embedding EXL3 checkpoints for the `vllm` engine specifically, because this project's own EXL3 quantizer writes a full redundant `lm_head` for every tied model regardless (see #2 below) -- bytes present on disk that `vllm_exl3_plugin`'s own `head_is_quantized()` already knows to skip loading for a tied model, but the checkpoint-only accounting had no way to know that without reading `config.json`'s `tie_word_embeddings`. Fixed.

**Follow-up #2 (same day): Exl3Backend's own bpw_head/vram_gb was dead code, not just imprecise.** Chasing down why the fix above made native and vllm's head accounting *agree* at bpw_head=16.0 on `turboderp/Qwen3-0.6B-exl3` led somewhere more interesting: they didn't actually agree, they coincided. `Exl3Backend`'s tied-head check (`self.config.stc.has_tensor(m.key)`, a bare unsuffixed `"lm_head"` lookup) can never succeed -- this codebase only ever stores suffixed keys (`lm_head.trellis`, `lm_head.weight`, ...) -- so it's been silently false for *every* model this engine has ever evaluated, tied or not, always falling back to reporting the embedding's bpw as the head's. Worse: for this specific checkpoint, native exllamav3 doesn't tie at all in practice, despite `tie_word_embeddings: true` -- `Linear.load()` tries the checkpoint's own `lm_head.*` tensors before ever falling back to the embedding, and since this project's quantizer wrote a real one anyway, that succeeds immediately, and native genuinely loads and serves logits through a real, separately-quantized ~6bpw head. So the pre-fix agreement (both engines reporting 16.0) was masking a real behavioral difference: vllm's 16.0 was correct (it really does tie), native's 16.0 was a bug hiding a real head it had just loaded.

   Fixed using `used_alt_key`, the ground truth `Linear.load()` already computes about whether it fell back to the embedding or used its own primary key -- no need to re-derive tensor-group existence from outside the module. Verified against two checkpoints, both now exactly matching the vllm engine's independently-computed number for the same on-disk tensors: `Qwen3-0.6B-exl3` @4.0bpw goes from the dead 16.0 to 6.0157 bpw, `Qwen3.5-9B-exl3` @4.00bpw (genuinely not tied) goes from 16.0 to 6.0040 bpw. Native and vllm now correctly *disagree* on the 0.6B checkpoint's head accounting (6.0157bpw/0.6050 GiB vs. vllm's 16.0bpw/0.4960 GiB) -- accurately, not as a bug: they really do serve that checkpoint's output layer differently.

**Follow-up #3 (same day): the "spotty" OOM was a teardown leak, plus GPTQ/AWQ went unaccounted.** Two more from real bench use, both independent of the prompt-logprobs OOM above.

1. `VllmBackend.close()` freed essentially nothing -- measured 8162 → 8102 MiB, the entire KV cache reservation staying resident, because `del self.llm` doesn't stop the engine's worker (the model and KV cache stay referenced behind module-level distributed state). Any project with *more than one* vllm-engine model therefore failed on the second one, sometimes outright and sometimes as a later fragmentation-dependent OOM -- which is exactly why it presented as intermittent and why dropping `gpu_memory_utilization` to 0.5-0.7 helped without fixing it. Now uses vLLM's own between-models teardown (`engine_core.shutdown()` → drop → `cleanup_dist_env_and_memory()`): 8162 → 400 MiB, and three engines run back to back at the *default* 0.85 where the second previously couldn't start at 0.5.
2. Classic GPTQ/AWQ checkpoints (autogptq/autoawq/auto-round -- as opposed to compressed-tensors) were dropped from accounting entirely, since they name weights `qweight`/`qzeros`/`scales`/`g_idx` and none of the suffix tables knew any of it: `bpw_layer=0.0` and a `vram_gb` covering only the embedding. Fixed by recovering numel from `qweight`'s packed element count (format-agnostic: GPTQ and AWQ pack along different axes but the total is identical) times the bit width from `quantization_config`. AWQ 4bit went 0.2898 → 0.5029 GiB against a 0.5031 GiB file; the four already-correct formats are unchanged.

With those in, the first real cross-format comparison this whole engine existed to enable, Qwen3-0.6B, 2-row smoke trace (so treat the absolute numbers as indicative, not a verdict):

| | layer bpw | vram_gb | ppl | KLD |
|---|---|---|---|---|
| AutoRound 4bit | 4.177 | 0.5040 | 4.8528 | 0.16905 |
| AWQ 4bit | 4.156 | 0.5029 | 4.9696 | 0.20358 |
| EXL3 4.0bpw (vllm) | 4.023 | 0.4960 | 4.6064 | **0.08063** |

EXL3 at less than half the KLD of both, at slightly *smaller* total size -- i.e. the format advantage is real and measurable on the served path, which is what makes the embed/head tax (#2/#3/#4b below) the thing actually standing between that and a competitive appliance.

Known limitation: the `vllm` engine has no noise-injection (self-noise-floor) support, so it can't be the `reference` group with `noise_floor` left at its default; vLLM's decoder layers aren't at a predictable, engine-version-stable location the way TransformersBackend's forward-hook approach needs one. All of the above committed on the `yeasah/exllamav3` fork (submodule pointer not yet bumped here).

## 1. Investigate Flash Attention for head dim 512 on pre-FA4 architectures

llama.cpp/ggml integrated into their implementation of FA a patch that extends it to cover head dimensions of 512, which allows them to support flash attention with i.e. Gemma 4 on "typical" hardware.

The changeset: https://github.com/ggml-org/llama.cpp/pull/20998

vLLM does not have this, and also does not allow the use of mixed attention layers (which could otherwise provide FA coverage for the majority of layers which are dim 256) due to (real, demonstrated) concerns about instability when doing this. So, the attention backend defaults to triton, which has a number of negative performance implications (not the least of which is taking turboquant off the table)

Unlike ggml, vLLM uses a forked copy of the upstream reference flash attention implementation:

https://github.com/vllm-project/flash-attention

Which contains a somewhat confusing array of implementations targeting CUDA, ROCm, FA2-4, and hopper specific kernels. The question here is: is there any surface within this project wherein a flash attention backend compatible with consumer ampere/ada lovelace/blackwell that could be plausibly extended in the manner of the llama.cpp changeset?

This may look at first glance like a bad deal in terms of complexity versus payoff, but it's probably not an exaggeration to say that it is the difference between an entire model family being on the table or not (not because it won't run at all otherwise, but because its performance will be so badly impacted as compared to the alternative models of otherwise similar capability that effectively makes them a dead choice)

## 2. Repair tool for existing EXL3 checkpoints

Every existing EXL3 checkpoint is seriously handicapped by two mistakes, to the point that its otherwise impressive efficiency gains are more than erased as compared to other formats.

 - The quantization pipeline generates a separate output head and embeddings for all tied models (which is entirely redundant content, and not small)
 - The quantization pipeline skips the embeddings entirely and leaves it at full resolution. This is significant at essentially any size, but particularly painful at aggressive quantizations where the size of the embeddings alone is a very significant fraction of the entire quantized model (in some cases larger than the entire rest of the model)

 Constructing a tool that post-processes existing EXL3 checkpoints would allow us to continue to use that otherwise useful investment in computation that the EXL3 collection represents. When processing a model, there will be two possibilities:

  - The original model was tied. In this case the quantized output head is present and can be repurposed as the tied embeddings/head that we otherwise need to generate, the original F16 embeddings discarded, and that's the whole story.
  - Otherwise, the F16 embeddings need to be quantized, leveraging the same quantization tools that were used to originally quantize the rest of the model -- but also recognizing that it takes the form of a constrained optimization problem, where the rest of the model quantization decisions have already been made, and allocation decisions happen following those already-made decisions.

## 3. VRAM-efficient use of the quantized embeddings

Our current explorations with quantized embeddings have involved dequantizing at load, which does prove that they work mathematically, and would save on file storage and I/O, but doesn't change the VRAM situation at all -- it still gets loaded at full resolution. To actually benefit in the way that matters most to the project (VRAM savings), the embeddings need to be loaded quantized and used that way. While the obvious and probably highest performing answer to this is to create new kernels for this, it is worth investigating other less costly approaches that may or may not have acceptable performance without the development overhead (and likely further pinning on CUDA -- exllamav3 currently does not support ROCm, but they talk about it, and adding more blockers to that support should it ever land isn't ideal)

## 4. EXL3 quantization pipeline improvements

This has several components.

### a. Improving the metadata situation

Just that: exllamav3 plays it rather fast and loose with the metadata, and important things are missing or misleading. Correcting this is relatively low hanging fruit, and restoring some trust in the stored data would be useful (significantly tempered by the fact that existing checkpoints will still have incomplete/incorrect metadata, at least unless we repair and republish them -- assuming license allows)

### b. Quantizing embeddings

This is really kind of interchangable with "don't emit an extra copy of the head for tied models", because it's the same data either way. We need to do exactly what is already done for tied models, but for every model, and store it as the embeddings rather than creating a new head. That's really it, unless we want to get fancier with depth allocation for embeddings versus the rest of the model.

### c. YAQA

There's reason to believe that the quantizer process itself can be improved, given its QTIP heritage and the further work of QTIP done in the YAQA project. Given license incompatibility it is important that nobody look at the reference code: just the papers. This is likely to be a large sized project, and also likely to produce modest gains -- not to say they aren't highly desirable, but as compared to the other low-hanging fruit this one is pretty high up in the tree, and should be scheduled as such.

## 5. Finish the job on MoE + TP

Not a new idea in any way, but it's still out there and we should finish the job.

Progress (2026-08-12, on the `vast` 8x3090 box):

- TP=8 validated for the first time on real hardware, not just the offline
  arithmetic -- `gemma-4-31b-it-exl3` @3.00bpw, eager mode, token-for-token
  identical to TP=1 including per-token logprobs.
- MoE + TP is no longer categorically unvalidated: `Qwen3.5-35B-A3B-exl3`
  @3.00bpw is token-identical to TP=1 at both TP=2 and TP=4 -- first real
  hardware run of the fused-shard TP composition, previously unit-tested only.
- The autotune cache survives eight concurrent workers writing it from empty
  (previously only tested at two).
- New open problem: `Laguna-XS-2.1-exl3` at TP=4 doesn't finish in reasonable
  time. `nsys` (ptrace itself is blocked, but CUDA-API-level tracing isn't)
  profiled it down to two exact `exl3_mgemm_kernel` instantiations taking
  0.9-3.8s *per single launch* -- a 100-1000x slowdown, not an autotune loop
  (both exllamav3's own `CACHEDEBUG` and `TRITON_PRINT_AUTOTUNING` showed zero
  output). Not simply about TP=4 or the 128-wide shard it produces, though --
  `Qwen3.5-35B-A3B` has an identical expert/layer/shard-width profile and
  never hits it. Needs someone to read `exl3_mgemm`'s source against the two
  checkpoints' actual per-tensor quantization parameters; not more hardware
  time.
- CUDA graphs at TP=8 also token-identical to TP=1 (`gemma-4-31b-it-exl3`,
  same box) -- both capture passes (51 PIECEWISE + 35 FULL shapes) clean per
  worker.
- The live alignment guard validated, not just unit-tested:
  `Llama-3.2-1B-Instruct-exl3` at TP=8 (its known-illegal 512-KV-channel case)
  raised `EXL3FormatError` on every affected rank with the expected diagnostic,
  before any real compute.
- Checked which other downloaded checkpoints could serve as a second TP=8
  case: none currently can. `gemma-4-12B`, `Qwen3.5-9B`, `Qwen3.6-27B` are each
  blocked by 1-8 tensors per `tools/tp_preflight.py` -- would need a new
  download to close that gap.

Still open: TP=3, 5, 6, 7; a second checkpoint at TP=8 (needs a download);
expert parallelism (kernel development gap, not a hardware-coverage one); and
the Laguna TP=4 `exl3_mgemm` bug above. Details in PHASE2.md.

## 6. CPU offload for EXL3 weights (vLLM's UVA offloader currently skips them entirely)

`vllm serve --cpu-offload-gb` silently offloads nothing for EXL3. Log shows
`Total CPU offloaded parameters: 0.01` on an EXL3 checkpoint where the same
model as AWQ offloads 3.63 GiB under otherwise identical settings.

Root cause, traced through vLLM's offloader and our own `linear.py`: the UVA
offloader (`vllm/model_executor/offloader/uva.py`) decides per-module
eligibility at construction time, before checkpoint weights load, by peeking
at `next(module.parameters()).device`. Our `EXL3Parameter` placeholders
(`create_weights`, `linear.py:184`) are `Parameter(data=None)` -- a default
empty *CPU* tensor -- so every EXL3-quantized layer reads as "already on CPU"
and the offloader skips it outright, before any real weight exists. The 0.01
GiB that does get offloaded is just the ordinary dense params that live
directly on decoder-layer submodules outside `EXL3LinearMethod` (RMSNorm
weights, etc).

That construction-time check isn't the only problem, or even the main one:
even if it passed, `process_weights_after_loading` (`linear.py:272`) replaces
the placeholders with brand-new `Parameter` objects (`exl3_trellis_N` /
`exl3_suh_N` / `exl3_svh_N`, or `exl3_weight` on the dequantize path) built
fresh on-device. The offloader never sees these -- it wrapped modules once, at
construction, before this replacement happened. AWQ doesn't hit this because
it preallocates its weight tensor on-device at construction and mutates it in
place the whole way through, so the offloader's construction-time CPU/UVA view
stays attached to the same tensor that's still serving inference later.

Considered and rejected: preallocating the correct final shape at construction
time (needs the per-tensor bit width `K` known upfront, since trellis shape
depends on it) so loading could happen in place, matching AWQ's pattern.
Checkpoints since ~v0.0.2 carry a `tensor_storage` map in
`quantization_config.json` that already records bit width per module (we
already fetch this in `config.py`'s `_load_tensor_storage`, just not for this
purpose) -- but it's already known to be incomplete on real checkpoints
(`Muse-Glimmer-30B-exl3` omits 303 quantized modules from it), doesn't exist
pre-v0.0.2 at all, and improving it going forward wouldn't retroactively fix
what's already published. More to the point: we're moving toward detaching
from directly-served upstream EXL3 checkpoints anyway (see #2) in favor of
repaired-only versions -- other than tied models, which qualify for a direct
tensor swap -- so betting on upstream checkpoint metadata quality isn't where
the value is regardless.

Chosen direction instead: don't try to make EXL3 loading conform to
"preallocate + mutate in place." `process_weights_after_loading` already has
the finished trellis/suh/svh tensors, with real final shape, at exactly the
moment it builds them. Reach into vLLM's offloader singleton there
(`get_offloader()`, `vllm/model_executor/offloader/base.py:111`), and if it's
a `UVAOffloader` with remaining `cpu_offload_max_bytes` budget, do what
`uva.py`'s `_maybe_offload_to_cpu` does ourselves: pin the tensor, wrap it with
`get_accelerator_view_from_cpu_tensor`, and register that instead of a plain
on-device `Parameter` -- updating the offloader's own byte counter so later
layers' budget accounting stays correct. Bits-agnostic, checkpoint-vintage-
agnostic, no vLLM changes, no plugin storage-path refactor. Only covers the
UVA backend (not `PrefetchOffloader`, a different per-layer streaming design)
-- acceptable since UVA is close to universal on the CUDA-only stack this
project already requires.

Known cost, going in with eyes open: this reaches into a vLLM internal
(`get_offloader()` / `UVAOffloader` are not public API), one more surface that
can silently break on a vLLM version bump -- the same standing risk already
taken on with `exl3_mgemm`'s call sites across the exllamav3 fork transition.
Worth it here: this is the only way to make CPU offload work for EXL3 at all,
and the feature matters enough to accept the maintenance exposure.

## Housekeeping

- **Retire `patches/vllm-gemma4-transformers-5.15-per-layer.patch`.** vLLM landed
  their own fix for the transformers 5.15 per-layer config break upstream:
  [70b84f0](https://github.com/vllm-project/vllm/commit/70b84f0bcbb6d0a35b74b1035673a1c934089dbb)
  (PR #49797, hmellor), and did it generically -- a real
  `ModelArchitectureConfig.from_layers()` / per-layer arch-config plumbing
  through `get_num_kv_heads`/`get_num_attention_heads`, not a gemma-4-only
  patch like ours. Next time we bump the vLLM pin past that commit, drop our
  patch, update the README patch table, and re-verify gemma-4-12B loads clean
  without it.
