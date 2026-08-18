# qbench: measuring quality across formats

*Extracted from TODO.md, where this accumulated as a work log. qbench lives in the
[`yeasah/exllamav3`](https://github.com/yeasah/exllamav3) fork; this note records
what was added for this project and what it cost to get right.*

The point of the work below is a single comparison this project could not make
before: does vLLM + vllm-exl3-plugin reproduce native exllamav3's quality, and how
does EXL3 stack up against AWQ/GPTQ/AutoRound **on the same checkpoint, served the
same way a user would actually run it** — not against a proxy for the served path.

Two earlier extensions preceded it: accounting for embeddings in VRAM tests, and
automatic pulling from the Hugging Face hub for reference and test models.

## A `vllm` engine (2026-08-14)

qbench can now run models through the real `vllm.LLM` offline API, under the same
KLD/ppl methodology as the other three engines: this project's own EXL3 plugin,
plus the quantization paths vLLM handles natively — AWQ, GPTQ, AutoRound, FP8,
compressed-tensors.

**Not "anything vLLM can serve", though.** GGUF via `vllm-gguf-plugin` is not
measurable through this engine today. So the engine's reach is base vLLM's own
quantization support plus this plugin, which is enough for the EXL3-vs-AWQ/GPTQ
comparison it was built for, but is not a general "serve it and measure it" tool.

The interesting part was getting full-vocab per-token logits out of vLLM at all.
Its public `prompt_logprobs` API is built for a UI's top-k display, and even at
`prompt_logprobs=-1` (full vocab) still builds one Python object per (position,
vocab-entry) downstream — hundreds of millions of them for one 2048-token row,
unusable at qbench's scale. Worked around by keeping vLLM's `EngineCore`
in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) and hooking `LogprobsProcessor`
(the one place, common to every model-runner variant, where the raw tensor gets
pythonized) to capture the real tensor. It streams row-by-row — firing qbench's
callback the moment each row finishes rather than after the whole batch — because
holding every row's full-vocab tensor at qbench's usual scale would be tens of GB.

Validated three ways: reconstructed logits cross-checked against a plain
transformers forward pass on Qwen3-0.6B (mean KLD ~0.003, backend-kernel-noise
scale); bpw/vram accounting cross-checked against a real AWQ checkpoint (within
0.01 GiB of vLLM's own logged checkpoint size) and a real EXL3 checkpoint
(`bpw_embed=16.0`, matching the known unquantized-embedding behavior); and an
end-to-end run on Qwen3-0.6B-exl3 @4.0bpw where native exllamav3 (ppl 4.6599, kld
0.081316) essentially matched the same checkpoint served through vllm + the plugin
(ppl 4.6064, kld 0.080634) — different-kernel-path scale, not different-model
scale.

## Four bugs that only real usage surfaced

The smoke test above used 2 rows of ~50 tokens. At rows=10, length=2048, four
separate faults appeared.

**1. An OOM no memory knob could fix, on a 0.6B model** — short of manually
shrinking `kv_cache_memory_bytes` to 4 GiB. Root cause: `prompt_logprobs=-1` makes
vLLM's own `compute_topk_scores` call `torch.topk(logits, vocab_size)` once per
1024-token chunk of scored prompt. With k that close to n, `torch.topk` falls back
to something close to a full sort, workspace and all — confirmed in isolation at
~7 GiB transient peak at Qwen3's 152k vocab and ~11.7 GiB at Qwen3.5's 256k,
against under 1 GiB at k=1. That spike happens *after* vLLM's memory profiler has
already sized the KV cache, so it is invisible to `--gpu-memory-utilization` and
every other normal knob.

Fixed by not asking vLLM to do the sort at all: `compute_topk_scores` is patched
(scoped to the prompt-logprobs path only) to grab its raw input tensor directly,
and qbench requests `prompt_logprobs=1` so vLLM's own remaining topk is a cheap
top-1. `max_num_seqs` defaults to 1 so per-request boundaries fall out for free
instead of needing vLLM's chunked-prefill request-splitting arithmetic replicated
by hand.

**2. `vram_gb`/`bpw_head` overreported on tied-embedding EXL3 checkpoints** for the
`vllm` engine specifically, because this project's own EXL3 quantizer writes a full
redundant `lm_head` for every tied model regardless (see
[embeddings.md](embeddings.md)). Those
bytes are present on disk, and `vllm_exl3_plugin`'s `head_is_quantized()` already
knows to skip loading them for a tied model — but checkpoint-only accounting had no
way to know that without reading `config.json`'s `tie_word_embeddings`. Fixed.

**3. `Exl3Backend`'s own `bpw_head`/`vram_gb` was dead code, not merely imprecise.**
Chasing why the previous fix made native and vllm agree at `bpw_head=16.0` on
`turboderp/Qwen3-0.6B-exl3` led somewhere more interesting: they did not agree,
they coincided.

`Exl3Backend`'s tied-head check (`self.config.stc.has_tensor(m.key)`, a bare
unsuffixed `"lm_head"` lookup) can never succeed — that codebase only ever stores
suffixed keys (`lm_head.trellis`, `lm_head.weight`, …) — so it has been silently
false for *every* model this engine has ever evaluated, tied or not, always falling
back to reporting the embedding's bpw as the head's.

Worse, for this specific checkpoint native exllamav3 does not tie at all in
practice, despite `tie_word_embeddings: true`: `Linear.load()` tries the
checkpoint's own `lm_head.*` tensors before falling back to the embedding, and
since this project's quantizer wrote a real one anyway, that succeeds immediately.
Native genuinely loads and serves logits through a real, separately-quantized
~6bpw head. So the pre-fix agreement at 16.0 was masking a real behavioral
difference: vllm's 16.0 was correct (it really does tie), native's 16.0 was a bug
hiding a real head it had just loaded.

Fixed using `used_alt_key` — the ground truth `Linear.load()` already computes
about whether it fell back to the embedding or used its primary key, so there is no
need to re-derive tensor-group existence from outside the module. Verified against
two checkpoints, both now exactly matching the vllm engine's independently-computed
number for the same on-disk tensors: `Qwen3-0.6B-exl3` @4.0bpw goes from the dead
16.0 to 6.0157 bpw, `Qwen3.5-9B-exl3` @4.00bpw (genuinely untied) from 16.0 to
6.0040. Native and vllm now correctly *disagree* on the 0.6B checkpoint
(6.0157bpw/0.6050 GiB vs. 16.0bpw/0.4960 GiB) — accurately, not as a bug. They
really do serve that checkpoint's output layer differently.

**4. A teardown leak, which is what made the OOM look "spotty".**
`VllmBackend.close()` freed essentially nothing — measured 8162 → 8102 MiB, the
entire KV cache reservation staying resident — because `del self.llm` does not stop
the engine's worker; the model and KV cache stay referenced behind module-level
distributed state. Any project with *more than one* vllm-engine model therefore
failed on the second, sometimes outright and sometimes as a later
fragmentation-dependent OOM, which is exactly why it presented as intermittent and
why dropping `gpu_memory_utilization` to 0.5-0.7 helped without fixing it. Now uses
vLLM's own between-models teardown (`engine_core.shutdown()` → drop →
`cleanup_dist_env_and_memory()`): 8162 → 400 MiB, and three engines run back to
back at the *default* 0.85 where the second previously could not start at 0.5.

**Also: classic GPTQ/AWQ checkpoints went entirely unaccounted.** autogptq /
autoawq / auto-round (as opposed to compressed-tensors) name weights
`qweight`/`qzeros`/`scales`/`g_idx`, and none of the suffix tables knew any of it —
`bpw_layer=0.0` and a `vram_gb` covering only the embedding. Fixed by recovering
numel from `qweight`'s packed element count (format-agnostic: GPTQ and AWQ pack
along different axes but the total is identical) times the bit width from
`quantization_config`. AWQ 4bit went 0.2898 → 0.5029 GiB against a 0.5031 GiB file;
the four already-correct formats are unchanged.

## The first cross-format comparison

Qwen3-0.6B, 2-row smoke trace — treat the absolute numbers as indicative, not a
verdict:

| | layer bpw | vram_gb | ppl | KLD |
|---|---|---|---|---|
| AutoRound 4bit | 4.177 | 0.5040 | 4.8528 | 0.16905 |
| AWQ 4bit | 4.156 | 0.5029 | 4.9696 | 0.20358 |
| EXL3 4.0bpw (vllm) | 4.023 | 0.4960 | 4.6064 | **0.08063** |

EXL3 at less than half the KLD of both, at slightly *smaller* total size. The
format advantage is real and measurable on the served path — which is what makes
the embed/head tax the thing standing between that and a competitive appliance.
See [embeddings.md](embeddings.md).

## Scope: what the size axis means, and what it deliberately excludes

qbench's size/vram number means **total real stored bytes across the checkpoint** — every
tensor, embedding and head included. That inclusion is the point of the accounting fixes
above: when these plots appear on model cards next to a download link, the audience reads
the axis as "how big is this file" regardless of how the caption scopes it, so excluding
the embedding was a real defect rather than a documented simplification.

**KV cache, activation memory, and batching/offload tradeoffs are deliberately out of
scope, and should stay out.** Two reasons. They are independent of the model and
quantization being compared — KV quant choice, batch size and offload strategy are the
user's variables, not the checkpoint's. And the audience comparing community quantizations
already treats KV cache as a separate, well-tooled budgeting step, with dedicated
calculators; folding it in would make the comparison *less* legible to exactly the people
who handle that axis competently.

The boundary matters because it will be tempting to cross later. Full "will this fit and
run" capacity planning — weights plus KV cache at a target context length plus batching —
belongs to the packaged appliance, whose users are precisely the ones who do *not* bring
their own calculator. Keep the two apart: **qbench answers "how big is this weight file",
the appliance answers "will this configuration run on this hardware".** Capacity planning
is a separate component, not a qbench flag.

## A third accounting bug, of the kind this file keeps finding

`safetensors_storage_info` buckets a tensor whose suffix it does not recognize under its
own full name and then drops it, which its docstring correctly warns "undercounts rather
than crashing". Block-quantized embeddings (`bq_q`/`bq_s`/`bq_r`) were the first format to
exercise that: the `vllm` engine reported `bpw_embed = 0.0` and a `vram_gb` missing the
entire embedding — 0.3789 GiB where the truth is 0.4859. Fixed by extending the suffix
table, as the docstring instructs.

Worth noting the pattern rather than just the fix: this is the third time storage
accounting has been quietly wrong (after the dead `bpw_head` fallback and the unaccounted
classic GPTQ/AWQ checkpoints), and all three failed silently in the direction of a
*plausible* number.

So there is now a standing guard: `check_against_disk` compares the counted bytes against
the checkpoint's actual on-disk tensor bytes and reports `accounted_share`, warning when a
bucket is impossible (zero) or the shortfall is too large for a legitimate exclusion. It is
the formalization of what has always been done by hand here — go look at the file sizes on
the hub — and it needs no second implementation to compare against, which is what makes it
applicable to every path rather than only the two that happen to compute the same figure
twice.

Two thresholds, because the callers differ: a checkpoint-header scan counts everything and
should land near 1.0, while the streamed engine walks the text model alone and legitimately
omits whole vision towers. And it guards *under*-counting only — whether a tensor that
exists on disk is one the engine will really load is a separate, policy question, which the
scope section below addresses.

It was checked against the bug that motivated it rather than assumed to work: with the
`bq_*` suffixes removed from the table again, a repaired MiniCPM5-1B reports `vram_gb`
0.3789 against a true 0.4859, `bpw_embed` 0.0, 78% accounted, and the warning fires.

## Known limitations, and what closing them would unlock

**No noise floor.** The `vllm` engine has no noise-injection (self-noise-floor)
support, so it cannot be the `reference` group with `noise_floor` left at its
default. vLLM's decoder layers are not at a predictable, engine-version-stable
location the way `TransformersBackend`'s forward-hook approach needs one. Tracked
as TODO `qbench-noise-floor`.

**No GGUF through `vllm-gguf-plugin`**, as above.

**The `vllm` engine mis-scores Qwen3.5.** Qwen3.5-9B's unmodified EXL3 checkpoint
measures ppl 248076 / KLD 10.26 through it, against ppl 12.15 / KLD 0.0131 for the
same checkpoint through the `exllamav3` engine, while generating coherent text
through plain `LLM.generate`.

**It is this engine's own scoring path, not teacher forcing on hybrid-Mamba
models**, which is what it first looked like. Teacher-forced scoring through
vLLM's *public* `prompt_logprobs` API is self-consistent on exactly these models:
generate greedily, re-score prompt+continuation, and every generated token comes
back top-1 at its position -- 16/16 on Qwen3.5-9B and 16/16 on Qwen3.5-35B-A3B,
matching a non-Mamba control (Llama-3.2-1B), and still 8/8 at 512 and 1024 tokens.

So the fault lies in what this engine does differently at its own scale: 2048-token
rows, `max_num_seqs=1`, and the `compute_topk_scores` patch it installs to dodge
the full-vocabulary `torch.topk` blowup. That blowup is real and worth noting on
its own -- the public API OOMs at 2048 positions on a 248320 vocabulary, since the
logprobs tensor alone is 2.0 GiB -- so the patched path is load-bearing rather than
optional, and is the first place to look.

The self-consistency probe above is the cheap guard, needs no reference model, and
is worth running against any engine change here.

Closing both, in that order, would turn this engine into something qualitatively
different rather than merely more complete: every measurement could run inside one
engine, against a reference produced by that same engine. Worth being clear about
what that would then be measuring, because it is easy to over-read.

It would **stop being a comparison of engines and become strictly a comparison of
model representations as interpreted by one engine.** That is a narrower claim
than qbench's current cross-engine setup makes, and in some ways a cleaner one --
engine-to-engine kernel differences drop out entirely, so what remains is
attributable to the format. But it cannot answer "is vLLM as good as llama.cpp at
serving this", which the cross-engine arrangement can.

The obvious audience for that is **not this project**: it is the `vllm-gguf-plugin`
developers, for whom "how does our GGUF path compare against native llama.cpp on
the same checkpoint" is a first-order question and currently an awkward one to
answer. If they do not already have such a tool, this would be the useful thing to
hand them. Worth noting as a possible contribution rather than a roadmap item.
