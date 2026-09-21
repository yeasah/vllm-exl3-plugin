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

**RETIRED as a comparison (2026-09-01)** — see "gemma-4-12B: the ordering reverses"
below. Qwen3-0.6B sits far enough into the damaged regime that its *ordering* does not
survive a change of model, so nothing in this section or the later 0.6B tables should be
cited for how formats rank. They stay because the wrong version is what gets remembered,
and because the wiring, the accounting fixes and the sub-3-bit cliff they established all
hold. The cross-format exemplar is now **Qwen3-8B** (`~/qbench/qwen3-8b.yaml`), chosen for
being dull: dense uniform attention, untied embeddings, a 151936 vocab, and published
EXL3 / AWQ / GGUF coverage. gemma stays in the regression matrix, where harsh is a
feature.

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

**This is now a standing merge policy for the exllamav3 fork, not a one-off
resolution.** Upstream disagrees: as of v1.4.9 its GGUF walk excludes
`token_embd.weight` (and `per_layer_token_embd`) from the size accounting outright. That
is a legitimate choice for a comparison of *decoder* quantization, and it is the wrong
one here for the reason above — the embed+head tax is the finding, and a number that
omits it cannot show it. **Never adopt the exclusion on a merge.** The divergence is
deliberate, permanent until the argument above changes, and recorded in
`eval/qbench/engines.py`'s own docstring so it is visible at the point a merge would
undo it.

Two things make this worth stating as policy rather than leaving to judgement each time.
It is a *silent* change — adopting upstream's version alters what the size axis means
while every plot, table and CSV keeps rendering, so nothing fails and the numbers simply
stop meaning what the surrounding documents say they mean. And it would break
comparability with every figure already recorded in this note and in
[embeddings.md](embeddings.md), which were all taken with the embedding counted. A merge
that quietly re-baselines the axis invalidates the archive rather than extending it.

The boundary matters because it will be tempting to cross later. Full "will this fit and
run" capacity planning — weights plus KV cache at a target context length plus batching —
belongs to the packaged appliance, whose users are precisely the ones who do *not* bring
their own calculator. Keep the two apart: **qbench answers "how big is this weight file",
the appliance answers "will this configuration run on this hardware".** Capacity planning
is a separate component, not a qbench flag.

## Did the exllamav3 v1.4.9 bump cost quality? No (2026-09-10)

The bump's regression gate failed 12 of 16 entries with two changed greedy
continuations, which reads as a kernel catastrophe and was nearly escalated to a
fork-or-not decision. It was not one. `bench/` compares a build to the previous build,
so it detects *change* and cannot distinguish better from worse; this is the measurement
that can.

**Dense: bit-identical.** Qwen3-8B, the cross-format exemplar, scored against an HF BF16
reference before and after the bump:

| arm | ppl | KLD | median | p90 | x noise floor |
|---|---|---|---|---|---|
| HF BF16 reference | 15.3779 | — | — | — | — |
| noise floor | 15.4057 | 0.000992 | 0.000729 | 0.001956 | 1x |
| EXL3 3.0bpw | 16.1595 | 0.058331 | 0.024434 | 0.119763 | 58.8x |
| EXL3 4.0bpw | 15.5905 | 0.014256 | 0.006184 | 0.029606 | 14.4x |

Both EXL3 arms came back **identical to full float precision on every field** — not
within tolerance, the same doubles. That is the expected result rather than a surprising
one: upstream's kernel work was `MGEMM: Add sliced mode for better scheduling in mixed-N
bundles` and MoE/CPU-MoE range filtering, and a dense model never enters those paths, so
the executed kernel is unchanged and floating point is deterministic. (4.0bpw at 14.4x
the floor also matches the ~15x this note cites elsewhere for a healthy model, so the
harness is behaving.)

**MoE: changed, and negligibly.** Qwen3.5-35B-A3B, which does exercise MGEMM:

| | v1.4.3-32 | v1.4.9 | delta |
|---|---|---|---|
| ppl | 12.2393 | 12.2364 | -0.0029 (0.02%) |
| KLD mean | 0.157762 | 0.157779 | **+0.000017 (+0.011%)** |
| median | 0.090098 | 0.090131 | +0.00003 |
| p90 | 0.302266 | 0.303359 | +0.0011 (0.36%) |

Not bit-identical, which is correct — this is the path that changed. The magnitude is
four orders below the KLD itself.

**This resolves the apparent contradiction with the gate**, and the resolution is the
general lesson. `bench/` reported 0.635 nats and KL 0.332 on this same model: the
*maximum over 63 scored positions*. qbench reports the *mean over ~20,480 tokens*. A
handful of near-ties flipping produces a large max and leaves the aggregate untouched.
Both instruments were right about different quantities, and only one of them is a
quality claim.

### Two things about the method, because both nearly produced a wrong answer

**qbench caches results per model and the key does not include the engine version.**
`model_key` hashes engine, source, options, source stamp and noise — so a result computed
under one exllamav3 build is served to a later run under a different one, silently. The
first "v1.4.9" numbers taken here were a replay of a 2026-09-01 run, and the tell was
weak in a specific way: the run took 69 minutes and printed plausible figures, but that
time was the *reference* logits being recomputed after a cache eviction while both EXL3
arms came off disk. `manifest.json` records the `seen` timestamp per arm and is the only
thing that says which. Invalidate by moving `results_<key>.json` aside; the manifest key
is in the same file.

**The MoE comparison needs no bf16 reference at all**, which is what made it affordable —
the matching reference is a 70 GiB download that was never pulled. Scoring EXL3 2.00bpw
against EXL3 3.00bpw of the same model puts *both* arms through MGEMM, and the question
is only whether the pair moves across the bump. The absolute KLD from such a run is a
distance between two lossy models and must never be quoted as a quality figure; the
project file says so at the top. Its one blind spot: a change shifting both arms
identically would cancel, which is unlikely across two bitrates with different shapes but
is a real limit. A bf16 reference is worth downloading only once such a screen says
something moved.

Project files: `~/qbench/qwen3-8b-exl3ver.yaml` and `~/qbench/qwen35-moe-exl3ver.yaml`.

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

So there is now a standing guard: `check_against_disk` compares the tally against the
checkpoint's actual on-disk tensor bytes. It is the formalization of what has always been
done by hand here — go look at the file sizes on the hub — and it needs no second
implementation to compare against, which is what makes it applicable to every path rather
than only the two that happen to compute the same figure twice.

**It is deliberately not a ratio against a threshold**, which was the first design and is
worse than it looks. The models with the most legitimately-absent bytes — a 50-layer vision
tower, an MTP head — are exactly the ones where a real gap has the most room to hide in the
slack, so any threshold loose enough not to fire on them is loose enough to miss a dropped
embedding. Calibrating the threshold on real models makes it worse, not better.

Instead every on-disk tensor is classified: **counted** (its module key is one the caller
tallied), **expected absent** (a multimodal tower, an MTP head, a norm, a bias, a router
gate, or a tied model's redundant `lm_head`), or **unexplained**. Only the last matters,
and it should be exactly zero on any checkpoint however much apparatus the model carries —
so there is nothing to calibrate. The warning names the offending module keys, which turns
"some number looks off" into "these tensors were dropped".

Demonstrated both ways rather than assumed. With the `bq_*` suffixes removed from the table
again, a repaired MiniCPM5-1B reports `vram_gb` 0.3789 against a true 0.4859, `bpw_embed`
0.0, and **0.107 GiB unexplained**, naming `model.embed_tokens.bq_q/bq_s/bq_r`. And on
Muse-Glimmer, whose vision tower puts 0.90 GiB legitimately out of scope, silently dropping
the embedding surfaces as **2.505 GiB unexplained** naming
`model.language_model.embed_tokens` — where the ratio version would have read ~75%, at the
threshold, and would have missed it entirely on a model with more apparatus.

## In-domain calibration does not survive contact with other data (2026-08-23)

exllamav3's `dev` branch added a per-tensor bitrate pipeline (`doc/optimize.md`) that
measures each tensor's sensitivity and solves for an allocation at a size budget. The
published `Qwen3.8-27B-exl3` card plots the resulting `EXL3-SC` quants against plain
EXL3, GGUF, NVFP4 and FP8 -- and scores every arm on a **self-generated in-domain
trace**, which is the distribution `EXL3-SC` alone was calibrated on. No other arm was
offered the same treatment, although llama.cpp's `imatrix` is the exact analogous
mechanism and accepts an arbitrary corpus.

This measures what that is worth. Three arms, one reference (Qwen's official FP8, since
a bf16 27B does not fit here), scored on two evaluation sets.

| arm | head | their in-domain trace | openwebtext |
|---|---|---|---|
| noise floor | -- | 0.000505 | 0.001295 |
| EXL3 3.00bpw (uniform) | 6-bit | 0.037624 | 0.046738 |
| **SC body + 6-bit head** (built here) | 6-bit | **0.028910** | **0.054788** |
| EXL3-SC 3.00bpw H4 (as published) | 4-bit | 0.029702 | 0.056793 |

**The middle row is the controlled comparison.** `SC_3.00bpw_H4` differs from plain in
three ways at once -- body allocation, calibration data, and a 6->4 head demotion -- so
its head was replaced with plain's 6-bit head by rewriting one shard and hardlinking the
rest. At 12.82 GiB against plain's 12.87 it is also matched on size to within 0.4%. What
remains is the body recipe alone.

Against plain, that body is **1.30x better in-domain and 1.17x worse on neutral text** --
a **1.52x swing** in relative standing from nothing but the evaluation distribution. All
six points sit 36-75x above their floors, so none of it is measurement slop.

**The head demotion is a mild pessimization in both directions**, not the story: the
6-bit head beats the 4-bit one on in-domain (0.0289 vs 0.0297) *and* on neutral (0.0548
vs 0.0568). It buys size rather than quality.

**Validation of the substitution.** The in-domain ratio measured here is 1.30x where the
published pair (0.0332 -> 0.0257) is 1.29x -- reproduced to within 2% despite an FP8
reference instead of bf16 and a 16-of-24-row subset of their trace (the longest rows
exceed 16 GiB of VRAM through the linear-attention path). A mismatched reference is
common-mode across arms, so ratios survive it and absolute values do not: plain 3.00bpw
reads 0.0467 here against their published 0.112 on openwebtext.

**What this licenses.** On this model at this bitrate, the self-calibrated recipe is a
fit to its calibration distribution rather than a general improvement -- a real gain for
deployments resembling that trace, and a real loss elsewhere. It does *not* establish
anything about other bitrates or models, nor whether the *allocation* would still win if
calibrated on neutral data; separating allocation from calibration needs a recipe built
on the bundled corpus, which only upstream can produce.

**Two things worth carrying elsewhere.** Their eval trace has a perplexity of 1.41
against openwebtext's 11.2 -- a very low-entropy distribution, which compresses the
dynamic range every arm on that chart is scored in. And the trace's own metadata records
that it was generated by an EXL3 6.00bpw quant, not by the bf16 model, which is one more
asymmetry the non-EXL3 arms do not get.

Projects and raw results: `~/qbench/qwen38-27b-sc-{neutral,indomain}.yaml`.

## Head bitrate: 6 is defensible, and the lever does not want pulling (2026-08-25)

*Tracked as `head-bits`, which this closes.* The one allocation question the composability
result did not kill: the head is a single tensor traded against a uniform body, a 1-D sweep
with no superposition assumption anywhere, so it could be answered by converting at each
head bitrate and scoring.

Budget-neutral by construction on phi-4-mini: quantizable weights are 3221M body + 615M
head, so `615·H + 3221·B` is held constant; the dense embedding (1.145 GiB, 42% of the
checkpoint) is excluded because neither knob moves it. **Verified after conversion rather
than assumed** — all five points land within **0.041%** of each other (max 688 KiB of
1.67 GiB), against a signal in the third decimal place.

| head / body | KLD | x floor | vs head 6 |
|---|---|---|---|
| 4 / 3.382 | 0.133103 | 25.3 | +34.6% |
| **5 / 3.191** | **0.095567** | 18.2 | **-3.4%** |
| 6 / 3.000 *(default)* | 0.098907 | 18.8 | — |
| 7 / 2.809 | 0.132526 | 25.2 | +34.0% |
| 8 / 2.618 | 0.185293 | 35.2 | +87.3% |

Noise floor 0.005258, so every point sits 18-35x above it and the differences are
resolvable by a wide margin.

**The hypothesis is not supported.** The item expected the answer not to be 6, on the
grounds that `lm_head` measures 15x more sensitive than any body tensor at matched
injected error. It does — but sensitivity per tensor is the wrong currency. The head is
16% of quantizable weights, so each head bit costs 0.19 body bits spread across a far
larger tensor, and the trade turns sharply negative in both directions: +34% at head 7,
+87% at head 8, +35% at head 4. The optimum is 5-6 and the default is defensible.

**Caveats, because the margin at 5 is small.** One model at one budget; head 5 beats
head 6 by 3.4% against roughly 1% run-to-run variation in this harness, and no point was
repeated, so "5 is better" and "5 and 6 are indistinguishable" are not currently
separable. What *is* separable is everything outside 5-6.

**Consequence: the allocation solver has nothing left to solve.** Body tensors cannot be
allocated independently (above); the embedding is a flat 4 bits across every model
measured ([embeddings.md](embeddings.md)); the head is 5-6 here. That leaves two scalars,
and two scalars are a lookup table rather than a search space. Note what this is *not*: a
claim that bit allocation is impossible in general. The body result is specific to EXL3's
sequential error compensation, which is what makes independently measured deltas cancel —
a quantizer without it might well compose.

### Head share is the hidden variable, and MoE models sit far outside the tested range (2026-09-18)

The sweep above is budget-neutral, so its verdict is an *exchange rate*: the head
was 16% of phi-4-mini's quantizable weights, which made each head bit cost 0.19
body bits, which is why 7 cost +34% and 8 cost +87%. Nothing about "5-6" is
intrinsic to a head — it is what a 16% head buys. Head share across the models
in play here:

| model | head params | head / (head + body) | cost of one head bit |
|---|---|---|---|
| Phi-4-mini-instruct *(the tested model)* | 0.615B | **16.0%** | 0.19 body bits |
| Ornith-1.5-9B | 1.017B | 11.8% | 0.13 body bits |
| Qwen3.6-35B-A3B, Ornith-1.5-35B-A3B | 0.509B | **1.4%** | 0.015 body bits |

A 35B-A3B MoE is **13x cheaper per head bit** than the model the sweep was run
on, because 35B of experts sit in the denominator. The penalties that closed the
question at 7 and 8 bits were the *body's* loss, and at 1.4% share the body
barely loses anything: head 5 -> 8 on these models is ~191 MiB, about 1.1% of a
4 bpw checkpoint. So the 5-6 verdict should not be carried onto them — if
anything the optimum is expected to move up, and `-hb 5` is the conservative end
of an untested range. A dense 9B at 11.8% is close enough to the tested regime
that 5-6 transfers with much less strain.

This is `category-bits` question 1 stated concretely, and it is the one that
gates publishing an MoE checkpoint. Composing it needs a donor that already
carries the head at another bit width, which for a self-converted model means a
second conversion rather than a free rearrangement.

**A composed head carries its source body's calibration context.** Conversion is
sequential — `advance_state_parallel` advances the calibration state "through the
(re-quantized) module", which is the same error compensation the body null above
attributes its result to — so `lm_head`'s Hessian is captured on activations
already degraded by that checkpoint's *body*. A head lifted from a 2 bpw arm onto
a 6 bpw body is therefore not a clean "head at K=3" point: it is a head fitted to
2 bpw activations and serving 6 bpw ones. The diagonal (each head on its own body)
is exact and free, since it is the converted checkpoint; the confound lives
entirely off-diagonal, which is where a head sweep needs to go.

Two things make this tractable rather than disqualifying. The mismatch is
expected to be *pessimistic* for the higher head bitrates — a head fitted to
cleaner activations than it ends up serving — so a composed high-bit head that
still wins is a safe conclusion, and one that loses is ambiguous. And a sweep
setting `head = min(body + 1, 6)` produces **two or more arms sharing head K=6
once body >= 5**, whose heads differ only in the body they were fitted to.
Composing those onto one common body isolates the calibration-context effect with
K held fixed, and is the control that says whether the rest of the matrix can be
read at face value. It only exists if the sweep actually reaches body >= 5.

## Head bitrate against body bitrate: the 5x4 grid (2026-09-20)

`ornith-ai/Ornith-1.5-35B-A3B`, body 2/3/4/5/6 bpw against head 3/4/5/6, with a
genuine noise floor at KLD **0.014147**. This is the free-promotion framing — `-hb`
is an `aux_target` outside `max_bits`, so a head bit buys size rather than trading
against the body — which is the question you answer when choosing a `-hb`, where
the phi-4-mini sweep answered the size-constrained one.

**Six of the twenty arms are conversions; the rest are composed.** The natives
trace `head = min(body + 1, 6)` — (2,H3), (3,H4), (4,H5), (5,H6), (6,H6) — plus
(3,H6) at the default. Everything else was built with
`tools/compose_checkpoint.py`, so the calibration-context caveat in the section
above applies off-diagonal: a composed head was fitted to its *source* body's
activations. That is handled below rather than waved at.

Excess KLD over the floor, and what each extra head bit buys:

| body | H3 | H4 | H5 | H6 | H4->5 | H5->6 |
|---|---|---|---|---|---|---|
| 2 | 0.180878 | 0.171750 | 0.169600 | 0.168661 | 1.25% | 0.55% |
| 3 | 0.064397 | 0.052585 | 0.049836 | 0.049727 | 5.23% | 0.22% |
| 4 | 0.033386 | 0.020768 | 0.017533 | 0.016837 | 15.58% | **3.97%** |
| 5 | 0.021285 | 0.008640 | 0.005550 | 0.004775 | 35.76% | **13.97%** |
| 6 | 0.017515 | 0.004679 | 0.002095 | 0.001288 | 55.22% | **38.51%** |

**The head's importance scales with body quality, and steeply.** At body 2 the
entire H3-H6 range is 7%; at body 6 it is **13.6x**. At low bitrate the body error
dominates and the head is not the bottleneck; at high bitrate the head is most of
what is left. "6 is defensible" was not wrong, it was a single point on this
surface.

**Byte-matched, since a head bit costs 0.0592 GB here.** Priced against what the
same VRAM buys on the body (local log-slope of excess KLD from the neighbouring
body arm):

| body | bar to clear | H5->H6 buys | verdict | native arm | bias favours |
|---|---|---|---|---|---|
| 2 | 1.83% | 0.55% | H5 | neither | — |
| 3 | 1.62% | 0.22% | H5 | H6 | **against the winner** |
| 4 | 1.89% | 3.97% | **H6** | H5 | **against the winner** |
| 5 | 1.96% | 13.97% | **H6** | H6 | with the winner |
| 6 | 1.96% | 38.51% | **H6** | H6 | with the winner |

**Head 5 below body 4, head 6 at and above** — and the crossover is where the
composition confound is least able to have caused it. In both rows that decide it,
the native arm is the one that *lost*: at body 3 the native H6 still failed to
justify itself against a composed H5, and at body 4 a composed H6 beat the native
H5 anyway. Whatever advantage a native conversion carries is pushing against each
verdict, and neither flips. Rows 5 and 6 have the bias pointing the same way as
the result, but at 7x and 20x the bar.

**H4 at body 2 is the one open cell.** H4->H5 buys 1.25% against a 1.83% bar, so
strict byte-matching prefers H4 — but 1.25% sits at this harness's ~1% run-to-run
variation and both arms are composed, so they are not separable here. Head 5
across both low rows is the simpler rule and costs nothing measurable. H3 is never
defensible: at body 6 it is 13.6x the H6 arm to save 0.18 GB.

**The grid is truncated at H6, and it should not be read as finding an optimum
there.** At body 6 the last measured step still buys 38.51% against a 1.96% bar —
nowhere near flat. For a model whose head is 1.4% of quantizable weights the
section above predicts exactly this, and nothing here tests H7 or H8. The rule is
"6 is enough to capture most of it at body >= 4", not "6 is the optimum".

**The control that would settle the composition question has not been run.** Three
native H6 arms exist, at body 3, 5 and 6, whose heads differ only in the body they
were fitted to. Composing those onto one common body isolates calibration context
with K held fixed, which is the measurement that says how far the off-diagonal
cells can be trusted.

Raw results: `~/qbench/results/ornith-1.5-35b-a3b/`, project
`~/qbench/ornith-1.5-35b-a3b.yaml`.

**Re-checked on the chat template, 2026-09-21: the verdicts hold.** H6 at body >= 4
on both Ornith and Qwen3.6. Body 3 moves from a clear H5 to a tie, which H5 still wins
on bytes. See the next section.

## Scoring through the chat template: what off-template rows did (2026-09-21)

Before `template: render`, the pipeline benched with `template: true`, which put every
row off-template: inside an open `<think>` block for Qwen3.6 and Ornith. The two modes
score **identical tokens**; only the prefix differs (18 vs 20 tokens). So every
comparison below is paired row-for-row, over qbench's cached per-token KLD vectors,
with 20k bootstrap resamples over rows and 90% intervals.

**The size of the effect depends on the model.**

| model | noise floor, off-template → render | quant arms |
|---|---|---|
| Qwen3.6-35B-A3B | 0.0077 → 0.0042 (0.55x) | 0.61-0.85x |
| Ornith-1.5-35B-A3B | 0.0141 → 0.0123 (0.87x) | excess KLD within ±3% at 2-4 bpw |
| Ornith 9B (dense) | — | under 1%, run elsewhere by the user |

**It is not confidence.** On Qwen3.6 the reference's confidence distribution barely
moves: bucket shares agree within ~1 point, and PPL goes 10.60 → 10.10. Yet KLD drops
~40% *within every confidence bucket*. The model is more **stable** on-template, meaning
the same perturbation moves its output less, and that holds from the noise floor to
2 bpw.

**Off-template instability grows with depth into the block, on Qwen3.6 only.** Noise
floor KLD by token position:

| | 0-32 | 32-256 | 256-1024 | 1024-2048 |
|---|---|---|---|---|
| Qwen3.6 off-template | 0.0117 | 0.0048 | 0.0055 | **0.0098** |
| Qwen3.6 render | 0.0112 | 0.0043 | 0.0039 | 0.0043 |
| Ornith off-template | 0.0295 | 0.0175 | 0.0150 | 0.0123 |
| Ornith render | 0.0361 | 0.0172 | 0.0124 | 0.0103 |

Ornith is a post-trained Qwen MoE, and it has the same shape in both modes. The
plausible reading is that its training corpus exercised long off-template text. Its
floor is also 2-3x Qwen3.6's in absolute terms, so it gave up some on-template
stability too. The explanation is a hypothesis; the position profiles are measured.

**The curve changed shape; the ordering did not.** Qwen3.6, paired, render step factor
divided by off-template step factor:

- 2->3 bpw: 1.32 [1.06, 1.67]
- 3->4: 0.83 [0.60, 1.11]
- 4->5: 1.26 [1.08, 1.51]
- 5->6: 1.00

Off-template, the curve had a bump at 3 bpw. PPL was non-monotone (+0.03% at 4 bpw,
+0.25% at 5) and went *below* the base at 6 bpw (-0.09%). Under render, KLD is smooth
and ΔPPL is monotone: 7.61 / 1.07 / 0.25 / 0.07 / 0.00%. Those off-template features
were amplified noise, not properties of the quantization.

**Render is the quieter instrument.** Within-mode step-factor intervals are ~±4% under
render, against ~±25% off-template, at the same 10 rows. That is why rows stay at 10.

**Head bitrate re-checked under render.** Composed arms (`tools/compose_checkpoint.py
--take head`) give 3.00bpw-H6 and 4.00bpw-H5 against the native 3.00bpw-H5 and
4.00bpw. The bar is the byte-matched rule from the grid above: a head bit is 0.0592 GB,
priced at the local log-slope of excess KLD to the next body arm.

| | body 3, H5->H6 | bar | P(H6 wins) | body 4, H5->H6 | bar | P(H6 wins) |
|---|---|---|---|---|---|---|
| Qwen3.6 | 2.06% [1.13, 2.93] | 1.87% | 0.64 | 8.27% [6.81, 9.98] | 1.91% | 1.00 |
| Ornith | 2.00% [1.23, 2.92] | 1.67% | 0.75 | 3.16% [1.57, 4.74] | 1.70% | 0.93 |
| Ornith grid, off-template | 0.22% | 1.62% | — | 3.97% | 1.89% | — |

On Ornith the body-3 head gain rose ~10x. That comparison also flips which arm is
composed: the grid's native arm at body 3 was H6, here it is H5. Native arms have tended
to score better, so the flip biases *against* the larger H6 gain, not toward it.
Policy unchanged: H5 below body 4, H6 at and above.

**What to distrust.** Any off-template qbench conclusion that rests on differences of a
few percent between arms at >= 4 bpw, on a model that behaves like Qwen3.6. That is the
regime where the amplified noise exceeded the arm-to-arm differences. The same-day
Qwen3.6 numbers above are the check to repeat before relying on one. The category-bits
and promotion sections were measured before render and have not been re-checked.

**Knob.** `quant.py qbench --chat-template {auto,render,none}`. `auto` renders when the
base repo ships a template (a standalone file or inline in `tokenizer_config.json`), and
uses raw text when it does not, as for base models like `meta-llama/Llama-3.2-3B`, where
`render` raises. Raw text is the right instrument there.

## Per-tensor bit allocation does not compose (2026-08-23)

*Survives the exllamav3 v1.4.3 bump unrevisited, and provably.* The study ran from
`~/git/exllamav3-dev`, cloned at `2398c05` on 2026-08-23 and never fetched since — its
reflog holds a single `clone:` entry. Upstream then tagged **v1.4.3 at that same
commit**, so the "new optimization pipeline" that release ships is byte-identically the
code these measurements were taken against. Nothing below needs re-running, including the
`sc_optimize` alpha of 1.791 that [upstream.md](upstream.md) reports as biased by the
fp16 KLD floor.


The section above separated `EXL3-SC`'s two changes and found the *calibration* half to be
distribution-bound. This one tests the other half on its own: **does per-tensor allocation
help when calibration is held constant?** Both arms below draw calibration from the same
bundled corpus mix, target 3.0 bpw with a 6-bit head, and differ only in whether bits are
uniform or assigned by a solved recipe. phi-4-mini is small enough that the reference is
genuine bf16, so these are absolute KLD figures.

| arm | bpw | KLD | ppl |
|---|---|---|---|
| noise floor | -- | 0.005258 | 13.967 |
| uniform | 4.00 | 0.029408 | 14.230 |
| **uniform** | **3.00** | **0.098907** | **15.065** |
| recipe (`sc_optimize`, defaultmix) | 3.00 | 0.097198 | 14.892 |
| recipe from measured marginal deltas | 3.00 | 0.103185 | 15.088 |
| uniform | 2.00 | 0.440018 | 20.401 |

**The solved recipe is worth 1.7%** against a solver prediction of 13.3%, and a second
recipe built from a strictly better measurement is *worse than uniform*. The rest of this
section is why, because most of the obvious explanations are wrong.

### What the ceiling actually is

With reconstruction error near-constant across tensors (see below), optimal allocation
reduces to replacing the size-weighted arithmetic mean of per-parameter sensitivity with
its geometric mean. For this model that ratio is **21.8%** -- so the null is not "there is
nothing to gain".

### Four explanations that were measured and rejected

**Calibration size.** 50 -> 250 trace rows and 40 -> 200 Hessian-capture rows moved the
sensitivity ranking by Spearman **0.987**, magnitude by 1.05x. Not it.

**Per-tensor error anchors.** `sc_rfn_probe` against the real 3.0 bpw checkpoint gives
measured rfn spanning 0.1441-0.1790 across all 224 body tensors -- an interquartile width
of 1.8%. `sc_optimize`'s default global anchor (`2:0.292`, 1.96/bit) already predicts
0.1490 against a measured median of 0.1483. There was never differentiating signal on the
error side for anchors to supply.

**The error model at low K.** Predicted from the shortfall arithmetic that demotions must
cost ~2x more than modelled, implying rfn(K=2) ~ 0.388. Converting a real 2.0 bpw
checkpoint and probing it gives **0.2942** against a modelled 0.2907, with a per-tensor
K=2/K=3 error ratio of **1.985** against the assumed 1.96. Refuted. Measured rfn by K:

| K | 2 | 3 | 4 | 6 (head) |
|---|---|---|---|---|
| median rfn | 0.2942 | 0.1483 | 0.0751 | 0.0224 |

**The shaped-noise surrogate.** Built a probe that substitutes each tensor's *real*
dequantized K=2 weight one at a time (`sc_realsens.py`), giving sensitivities directly
comparable to injected noise at rfn 0.29. Agreement is good -- median ratio 1.05, Spearman
0.959 -- and, decisively, **feeding the real measured sensitivities to the same solver
still predicts a 20.1% gain**. The surrogate was never the problem.

### An fp16 measurement floor, worth fixing regardless

Every KLD reading in `sc_measure` carries a constant additive floor of **~6.1e-5**: flat at
5.85-6.31e-5 across quintiles spanning 51x in sensitivity (log-log correlation with
sensitivity 0.088), and reproducible at 5.3e-5 in an independent run with different rows,
trace and noise levels. It is not a restart artifact -- the tool's own control asserts an
exact-zero unperturbed KLD and passes. It is the model computing logits in fp16: reference
and perturbed logits each carry independent rounding, so their difference has a noise
component that does not shrink as the perturbation does. Caching in fp32 would not help;
the rounding happens inside the forward pass.

The consequence is a biased exponent. Subtracting the floor moves `sc_optimize`'s fitted
alpha from **1.791 to 1.996** -- the exact square law that theory predicts in the
small-error limit. Correcting it changes 20 of 224 assignments and moves the predicted gain
from 16.2% to 17.6%, so it is a real methodology bug but not the explanation.

### The actual finding: deltas do not compose

Measuring one tensor at a time against an *otherwise-clean* model systematically
under-counts what it costs to push that tensor deep, because the surrounding tensors are
not quantized. The fix is to measure marginally -- in the model as it will actually be. So:
materialise the 3.0 bpw checkpoint as plain fp16 (`dequantize.py`, validated at KLD
0.098579 against the trellis checkpoint's 0.098907), use it as the base, and measure each
tensor's whole-model KLD delta when moved to K=2 or K=4 in context (`sc_marginal.py`).

That confirmed the under-count: summed over all tensors, demotion costs **1.23x** more
in-context than the clean baseline predicts, with per-tensor ratios spanning 0.50-1.80.
29 of 224 tensors get *worse* when given an extra bit.

The recipe solved from those measured deltas scored **0.103185** -- worse than uniform,
against a predicted 0.0663. Assembling the identical allocation by mixing dequantized
weights from the uniform-2/3/4 checkpoints (`mix_recipe.py`), which is exactly what the
per-tensor framework assumes a recipe *is*, scores **0.105270**. So the conversion process
is not to blame; the deltas themselves do not superpose:

| treatment | sum-of-deltas | measured | error |
|---|---|---|---|
| promote all 224 to K=4 | 38.4% | 29.7% | -23% |
| demote all 224 to K=2 | 462.5% | 444.9% | -4% |
| **marginal recipe (29 down, 42 up)** | **64.7%** | **106.8%** | **+65%** |

(as a fraction of the K=3 baseline, so the two evaluation sets are comparable.)

**Superposition holds reasonably when every tensor moves the same direction and collapses
when they move in opposite directions.** That is the regime every allocation solver
operates in, and it is why the objective being minimised -- a sum of independently measured
per-tensor terms -- has little relationship to the KLD that results.

### The same effect, seen from outside: fractional bitrates are penalised (2026-08-30)

If mixed-direction moves superpose badly, then **every fractional bpw target should sit
above the trend through its integer neighbours** — a fractional target is nothing but a
mixture of K levels, which is the mixed-direction regime by construction. turboderp's
pre-SC `Muse-Glimmer-30B` card carries two fractional points on an **openwebtext** eval
(no in-domain calibration confound, no SC arm), and both do:

| bpw | measured | log-interp of neighbours | bump | additivity alone | residual |
|---|---|---|---|---|---|
| 2.50 | 0.123 | 0.0900 | **+37%** | +27% | +8% |
| 3.50 | 0.030 | 0.0230 | **+31%** | +22% | +7% |

Two effects, and the split matters. A 50/50 mix of K levels gives the *arithmetic* mean of
the endpoint KLDs while the log-trend is the *geometric* mean, so `(1+r)/2*sqrt(r)`
predicts +27% and +22% from additivity alone — no allocation failure required, and it
depends only on the local steepness of the curve. **What is left is +8% and +7% on two
independent points**, which is this section's superposition penalty measured from the
outside.

It is easy to miss because it is small on either axis: +31% at KLD 0.030 is 4% of a linear
plot dominated by the 2.0 bpw point, and 0.12 decades on a log one. The in-domain eval of
the later SC card magnifies it — 1.63x and 1.62x inflation on the 3.00 and 4.00 points
against **1.80x on 3.50** — which is what turns the +31% into the +45% visible there, and
why the effect first surfaced on the confounded chart.

**Not yet tested, and cheap:** assemble a 50/50 K3/K4 mixture from the existing phi-4-mini
`uniform-3.0` and `uniform-4.0` checkpoints with `mix_recipe.py` and score it. Geometric
(no penalty) is 0.0539, additivity alone 0.0642, additivity plus a 7% residual **0.0686**.
If it lands near the last, fractional EXL3 bitrates carry a structural ~25-35% penalty and
**integer K is the only efficient place on the curve** — which is not how bpw targets are
currently chosen, here or anywhere.

### What the null is about, and what the pipeline actually does (2026-09-20)

Asked directly: is the tax specific to *per-tensor* variation, or does per-layer
variation pay it too? And what stops us targeting a lower bitrate and selectively
promoting the important tensors? Reading
`exllamav3/conversion/allocation.py` answers both, and corrects a framing this
file carried for two days.

**exl3 has a budget independent of any recipe.** `-b/--bits` sets
`max_bits = int(bpw * sum_numel)`; `-rcp/--recipe` replaces the budgeted path
entirely ("used in place of the budgeted allocation"). They are alternatives.

**The budgeted allocator never demotes.** `base_bpw = floor(bpw)`, every budgeted
Linear starts there, and the loop only ever calls `increase_1()`, spending the
fractional remainder on whole qgroups in priority order (then by distance to the
nearer end of the forward pass). So **"target lower and selectively promote" is
not an alternative to what exl3 does — it is exactly what exl3 does.** Every
fractional EXL3 checkpoint is that scheme: 2.54 bpw means everything at K=2 with
some groups promoted to K=3 until the budget runs out.

**Which is why it is already measured, and already taxed.** The fractional points
in the section above *are* selective-promotion checkpoints, and they land +37% and
+31% above the trend through their integer neighbours. So the answer to "what is
stopping us" is: nothing mechanical. It is the default, it has always shipped, and
the measurement says it costs.

**The dominant term is Jensen, not an allocation failure.** A mixture of two K
levels gives the *arithmetic* mean of their KLDs where the interpolated trend gives
the *geometric* mean, and arithmetic >= geometric. That is +27% and +22% on the two
points, it depends only on the local steepness of the curve, and **no choice of
which tensors to promote can avoid it** — it is a property of mixing K levels at
all. Only the +8% and +7% residual is superposition, and that is the part a clever
recipe could in principle recover. The recipe experiment is what says it does not:
42 tensors up, 29 down, scored 0.103185 against uniform's 0.098907.

**So granularity is not the axis.** Per-layer variation is not exempt — it is a
mixture, so it pays Jensen like any other. Nor is "promotion-only" the exempting
property, which is where this file was wrong on 2026-09-18: the budgeted allocator
*is* promotion-only from the floor and is taxed anyway. The property that actually
exempts something is **not being budget-matched**.

**`-hq` is the case that is not budget-matched.** The MoE boost
(`select_hq_bits`, 2 bits on attention and shared experts for MoE architectures)
is applied *after* the budget loop and is never checked against `max_bits`:

```python
while sum_bits < max_bits:   # budget loop -- promotions only
    ...
if hq:                        # after, and unchecked
    t.clamp_min()
```

`final_bits` is then recomputed post-clamp and returned as the model's actual bpw,
so `-hq` overshoots the requested target by design. Visible in the artifacts:
`Laguna-XS-2.1-exl3` is published as **`3.00bpw` and records `bits: 3.01`**, which
is the 0.24% of trellis bytes its `shared_expert` boost costs;
`Qwen3.5-35B-A3B-exl3`, which has no boost, records exactly `3.0`.

So `-hq`'s KLD win was measured against a *smaller* model — the same target
without the boost — and needs no superposition argument to explain. **The
budget-matched comparison has never been run**: give the same 0.24% to the routed
experts instead and see which buys more. That is the experiment `category-bits`
should be framed around, not "promotion vs reallocation".

**One more consequence, for the head.** `sum_numel` and `sum_bits` accumulate only
for `qbits_key == "bits"`; head, MTP and vision are `aux_targets` and sit outside
`max_bits`. **`-hb` does not come out of the body budget.** The head sweep above
imposed budget-neutrality *manually* (holding `615·H + 3221·B` constant), so
"head 6 is defensible" answers "if the head must be paid for from the body" — the
right question when size-constrained, but not the question "what should I pass to
`-hb`". At fixed `-b`, a higher `-hb` is a pure promotion that simply makes the
file bigger.

### What this licenses

**Superseded in scope as of 2026-09-20** — see "Extra-budget promotion does
compose" below. Everything here is budget-neutral reallocation, and that turns out
to be the load-bearing condition rather than an incidental one. The null holds
exactly as stated for reallocation; it does not extend to promotion.

On this model at this bitrate, per-tensor allocation is worth ~1.7% at best, and the
apparent 13-22% available to a first-order solver is an artifact of assuming
independence. This says nothing about larger models, other bitrates, or allocation schemes
that optimise the combined objective directly rather than a sum of parts -- which is the
only direction these results suggest is worth taking. It also does not touch the *head*,
which the solver never allocates and which measurement puts 15x above any body tensor in
sensitivity.

Project and raw results: `~/qbench/phi4mini-alloc.yaml`. Measurement JSONs in
`~/qbench/sens/`, recipes in `~/qbench/recipes/`, and the tools built for this in
`~/qbench/tools/` (all derived from exllamav3 `dev`; the pinned fork is untouched).

The converted checkpoints themselves are gone -- 42 GiB of inputs to a finished
study -- but their bit allocations are kept in
[data/qbench/phi4mini-allocations.json](data/qbench/phi4mini-allocations.json):
`bits`, `head_bits`, calibration shape, and the per-module assignment for each
of the nine converted variants. That last part is the one worth keeping, because
it is solved rather than chosen: `recipe-3.0` differs from `uniform-3.0` on 88
of 225 modules and spans 1 to 6 bits, so re-deriving it means re-running the
solver, where re-converting from a recorded recipe is merely slow.
`microsoft/Phi-4-mini-instruct` is the source model. The two unquantized
variants (`deq-3.0`, `mixed-marginal`) were dequantized rather than converted
and carry no allocation, which is itself the thing to know about them.

## Extra-budget promotion does compose, and the null was about opposed moves (2026-09-20)

The allocation null above was measured only where bits taken from one tensor were
given to another. Every escalation within it pushed toward *better predictions*,
which is pressure toward *more* reallocation — `recipe-3.0` moved 88 of 225 modules
(59 up, 29 down, **1711 opposed pairs**, three modules driven to K=1) and
`marginal-3.0` moved 71 (42 up, 29 down, 1218 opposed pairs). Nobody ever pushed
the other way: fewer moves, gentler moves, or no demotions at all. Three axes, and
every tested point sat at the extreme of all three.

Doing that on Qwen3-0.6B, with `tools/compose_checkpoint.py` building each arm in
seconds from the native 2/3/4/5/6 bpw conversions, gives a different answer.

**Method.** A 4.00 bpw base, with one tensor role at a time lifted from the 5.00 or
6.00 bpw arm — a pure promotion, nothing demoted. Scored against the log-trend
through the uniform arms, which is what the same bytes would have bought spent on
the whole body. Noise floor 0.001334; body slope 1.2587/bpw. Composed arms carry
the calibration-context bias of [the head grid](#head-bitrate-against-body-bitrate-the-5x4-grid-2026-09-20):
a tensor converted at 6 bpw was fitted to 6 bpw activations and serves 4 bpw ones,
which is **pessimistic**, so a win here is safe and a narrow loss is ambiguous.

### The bar a promotion has to clear

Promoting a fraction `f` of body weights by `Δ` bits beats spending the same bytes
uniformly only if the promoted set's per-parameter sensitivity exceeds

    slope·Δ / (1 - e^(-slope·Δ))

as `f -> 0`. Two things fall out that are worth stating before the data. The bar is
**independent of how small the promoted set is** — cost and benefit shrink together,
so being selective does not make the economics easier; it is a sensitivity
threshold, not a count threshold. And it rises steeply with `Δ`: **1.76x at +1 bit,
2.74x at +2, 3.86x at +3** on this model.

### The dense body is not flat

Each role promoted +2 bits, alone, from the 6.00 bpw arm:

| tensor | % of body | sens / avg | bar | vs trend | ppl | |
|---|---|---|---|---|---|---|
| **k_proj** | 6.67 | **3.96x** | 2.52x | **-10.4%** | -21.4% | **beats** |
| v_proj | 6.67 | 1.95x | 2.52x | +4.1% | +3.8% | loses |
| down_proj | 20.00 | 0.91x | 2.15x | +37.8% | +35.6% | loses |
| up_proj | 20.00 | 0.83x | 2.15x | +40.2% | +40.4% | loses |
| o_proj | 13.33 | 0.75x | 2.33x | +27.1% | +23.6% | loses |
| gate_proj | 20.00 | 0.59x | 2.15x | +47.6% | +42.2% | loses |
| q_proj | 13.33 | 0.44x | 2.33x | +32.3% | +26.2% | loses |

**`sum(sens x fraction) = 1.018`.** The seven roles are the whole body and their
shares sum to unity, which the additivity model requires and does not get for free
— the strongest available check that the per-tensor sensitivities mean what they
are being read to mean.

The spread is **8.9x**, which contradicts the reading of "the sensitivity gradient
is flat" taken from the 21.8% ceiling earlier in this file. That number is the
geometric-vs-arithmetic gap of the *size-weighted mean*, dominated by the bulk, and
it barely moves when one 6.67% tensor is 4x. The tail is what a promotion spends
against, and the tail is fat.

**k against q is the striking pair**: symmetric in the dot product, 9x apart per
parameter. GQA accounts for 2x of it (8 KV heads serving 16 query heads, so each k
parameter feeds twice the attention scores); the rest is unexplained here, and
q_proj having twice the width and more redundancy to spend is a candidate rather
than an answer. **The mechanism is not needed to use the result** — this is exactly
what an imatrix measures directly.

### Promotions superpose, to under 1%

Predicting multi-tensor arms from the single-tensor sensitivities above:

| arm | bpw | predicted | measured | error | vs trend |
|---|---|---|---|---|---|
| k +1 | 4.0896 | 0.033420 | 0.033111 | **-0.9%** | -13.1% |
| k +2 | 4.1563 | 0.031382 | 0.031381 | -0.0% | -10.4% |
| **k +1, v +1** | 4.1563 | 0.029470 | 0.029261 | **-0.7%** | **-16.5%** |
| k +2, v +1 | 4.2229 | 0.027432 | 0.027476 | **+0.2%** | -14.7% |
| k +2, v +2 | 4.2896 | 0.026429 | 0.026604 | **+0.7%** | -10.2% |

Maximum error 0.9%. Against **+65%** for the budget-neutral marginal recipe, the
conclusion is not subtle: **the collapse was never about granularity, count, or
measurement quality. It was about opposed moves.** Remove the demotions and the
deltas superpose almost exactly.

### Spread the bits, do not concentrate them

At identical cost (bpw 4.1563), `k+1, v+1` scores 0.029261 against `k+2` at
0.031381 — **6.8% better for the same bytes**. The bar explains it: the +1 bar is
1.76x where the +2 bar is 2.74x, so a second bit on an already-promoted tensor is a
much harder sell than a first bit on the next one. v_proj shows both sides, since
1.95x sits between the two bars: adding v **+1** to `k+2` gains 4.3 points
(-10.4% -> -14.7%), while adding v **+2** instead loses 4.5 points (-> -10.2%).

**Consequence for `-hq`**: `select_hq_bits = 2` is the wrong default on this
evidence. A tensor has to be 2.74x to justify the second bit and only 1.76x to
justify the first, so the same bytes spread over more tensors at +1 should win.
Untested on an MoE, and the shared-expert case may differ — the duty-cycle gap
there is large enough that both bits could clear.

### What this licenses, and what it does not

Per-tensor allocation is **not** dead; budget-neutral per-tensor allocation is.
A conservative, promotion-only recipe is worth **-16.5% against the uniform
alternative** on a dense model at 4 bpw, from two tensor roles guessed at without
any importance data, with the composition bias running pessimistic throughout.

What is not established: one model, one bitrate pair, dense, and the arms are
composed rather than converted, so the numbers are a floor rather than the real
thing. Whether the ranking transfers across models or bitrates is untested — and
the ranking is the part an imatrix would supply, which now has **seven ground-truth
points to be validated against before any of it is built**.

Raw results and arms: `quantization/work/Qwen3-0.6B-exl3/main/qb_results.json`.

## What identifies a promotable tensor: not an imatrix, and not a fixed role (2026-09-20)

Promotion works (above) and the bar is arithmetic, so the remaining question is how
to find the tensors worth promoting without measuring every one. **Three shortcuts
were tested and all three failed**, and the failures are worth keeping because each
will be proposed again: a GGUF imatrix, a fixed tensor role, and a published
per-tensor sensitivity table. They fail for one shared reason — sensitivity has to be
measured in the model as it will be served, and none of the three is.

### A GGUF imatrix does not contain the ranking

Sources, both legacy (pre-GGUF) imatrix format — `int32 n_entries`, then per entry
`int32 name_len / name / int32 ncall / int32 nval / float32[nval]`, with a trailing
`int32 ncall_total / int32 len / dataset`:

- [unsloth/Qwen3-0.6B-GGUF](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/imatrix_unsloth.dat)
  — 196 entries, ncall 688, dataset `unsloth_calibration_Qwen3-0.6B.txt` (note:
  *model-specific* calibration, which given the 1.52x in-domain/neutral swing above
  is not a neutral instrument)
- [bartowski/Qwen_Qwen3-0.6B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B.imatrix)
  — 196 entries, ncall 137, dataset `/training_dir/calibration_datav3.txt`

Both parse clean (196 = 28 layers x 7 roles, `nval` matching each role's input dim)
and agree closely with each other. Neither correlates with measured sensitivity:

| reduction | Spearman rho | ordering, most to least important |
|---|---|---|
| `sum_j d_j ||W[:,j]||^2` (the planned proxy) | **+0.036** | q, k, v, gate, down, up, o |
| relative (normalized by `||W||^2`) | +0.250 | k, q, v, gate, down, up, o |
| activation energy alone | +0.357 | k, q, v, down, gate, up, o |
| `||W||^2` alone, no imatrix | **-0.571** | gate, q, up, v, down, k, o |
| **measured** | +1.000 | **k, v, down, up, o, gate, q** |

**The tell is consistent: q_proj ranks 1st or 2nd under every imatrix reduction and
is last in truth (0.44x)** — an 8x error in the worst direction. o_proj is last
under every proxy and mid-pack in truth.

**Why, and it is structural rather than a bad reduction.** An imatrix records mean
squared activation per *input* channel. That is exactly right for what llama.cpp
uses it for — weighting the quantizer's local reconstruction objective — but
allocation sensitivity is an *output*-side property: how far the model's output
moves per unit perturbation of this tensor's output. q_proj produces ample output
energy and then feeds a softmax, shift-invariant and behind a per-head QK-norm that
removes scale, so those errors are damped. No input-side statistic can see that.
It is also why `sc_realsens` ranked correctly at Spearman 0.959: it measured output
deltas. The expensive measurement is expensive because it measures the right thing.

**What survives**: at *role* granularity a proxy is not needed. Seven roles is seven
compose arms, minutes on a small model, and exact rather than correlated. A proxy
would only earn its keep at per-tensor granularity, which the budget-neutral null
says is not worth chasing anyway. The MoE case is unaffected — `-hq`'s boundary is
categorical and needs no ranking at all.

### A published per-tensor sensitivity table does not pick a paying set

`turboderp/Qwen3.8-27B-exl3` ships a
[kld_table.json](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/blob/main/kld_table.json):
per-tensor measured `S` for all 400 budgeted tensors plus a geometric KLD model over
K=1..8 (`alpha` 1.7403, `bit_ratio` 1.96, so a flat 3.2257x per bit). This is the
output-side signal the imatrix lacks, and the K-model is not a guess — phi-4-mini
measured a K=2/K=3 error ratio of **1.985** against the assumed 1.96.

**The role ordering cross-validates.** Ranking by KLD/size reproduces, on a different
model and from an entirely separate pipeline, the shape measured above:

| role | Qwen3.8-27B (table) | Qwen3-0.6B (measured here) |
|---|---|---|
| v_proj | 5.85x | 1.95x |
| k_proj | 2.68x | 3.96x |
| q_proj | **0.51x** | **0.44x** |
| mlp.* | 0.85-1.15x | 0.59-0.91x |

k and v high, q lowest, MLP near average. So per-role sensitivity is a real and
reproducible property — which is what made the table worth testing.

**It still does not work.** Promoting the top 60 by KLD/size from K=4 to K=6 costs
+0.1890 bpw (the table predicted +0.1892) and is predicted to cut total KLD by
**26.5%**:

| | KLD reduction |
|---|---|
| `kld_table.json` predicts | **26.5%** |
| measured, bf16 reference | **4.4%** |
| measured, FP8 reference | 1.5% |

A **6x** over-prediction. The arm is faithful — 60 modules verified at K=6, 348 still
at K=4, lm_head untouched — so the sensitivities are what failed, not the
composition.

**And no reading of the curve rescues it.** The comparison above is base-relative, so
no slope enters. Separately, the arm cut excess 4.4% for +0.189 bpw, which breaks
even only if the body curve is flatter than **0.24/bpw**; the flattest segment
measured anywhere on this model is 0.790, **3.3x steeper**. Across the full observed
slope range the arm lands **+11% to +29% against trend**.

**Why this does not contradict "promotion composes".** Additivity held to under 1%
on Qwen3-0.6B — but those sensitivities were backed out of composed arms measured on
the same instrument, so they were in-context by construction. `sc_measure` measures
tensors in isolation, which is exactly the regime "deltas do not compose" describes.
The framework is sound; **its inputs have to be measured in the model as it will be
served.** Published per-tensor sensitivities are not that, and this is the third
shortcut to die on the same point.

### Published bitrate ladders are not a controlled series (2026-09-20)

A byproduct of the above, and a caution that reaches further. Qwen3.8-27B's uniform
arms do not lie on a log-linear curve, and the deviation is not one bad point: a
5.00bpw arm was added specifically to break the tie, with both hypotheses registered
in the project file first, and **it missed both low** (0.004501 against predictions
of 0.005268 and 0.006175). Adjacent slopes alternate rather than drift.

The control is the two models converted here in a single batch with one settings set:

| model | converted by | adjacent slopes | spread |
|---|---|---|---|
| Qwen3-0.6B | one batch, here | 1.794 -> 1.402 -> 1.321 -> 1.196 | **1.50x**, monotone |
| Ornith-1.5-35B-A3B | one batch, here | 1.221 -> 1.180 -> 1.170 -> 1.082 | **1.13x**, monotone |
| Qwen3.8-27B | turboderp, published | 0.790 -> **1.581** -> 0.951 | **2.00x**, alternating |

Both of ours decline smoothly and monotonically; the published ladder does not, at
comparable signal-to-floor. The likely reason is that **each published revision is an
independent conversion run**, so anything unrecorded that differs between them
(calibration draw, seed, host, patch level within a version) makes each arm a sample
rather than a point on one curve.

**What to take from it**: a `vs trend` figure computed against a chord through
independently-published arms is unsound, and may be why both registered predictions
missed. Every such figure elsewhere in this file is computed against arms converted
here in one batch, which this comparison suggests is the reason they behave. Note
also the reference matters: an FP8 reference put the 6.00bpw arm at 1.99x the noise
floor with non-monotone perplexity and mutually inconsistent slopes (0.958 from 4->6
against 0.674 from 3->4). Projects: `~/qbench/q38-kldtable{,-bf16}.yaml`, kept as a
pair for that contrast.

### k_proj is not the `-hq` of dense models

The obvious follow-up to k_proj measuring 3.96x on Qwen3-0.6B: is it a portable
rule? Three points, each the same construction (4 bpw base, k_proj lifted +2 bits,
priced against the uniform trend):

| model | family | GQA | k share | sensitivity | bar (+2) | vs trend |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | Qwen3 | 2 | 6.67% | **3.96x** | 2.52x | **-10.4%** |
| Llama-3.2-3B | Llama | 3 | 3.12% | **1.30x** | 2.86x | +5.0% |
| Qwen3-8B | Qwen3 | 4 | 2.17% | 2.44-2.52x | 2.94-3.52x | +0.9 to +2.5% |

**Not monotone in GQA and not monotone in scale.** Only Qwen3-0.6B clears its bar.
The GQA-fanout story that seemed to explain k's advantage is refuted outright: GQA
rose 2 -> 4 and concentration *fell*. Llama corroborates on both metrics (ppl +2.5%
vs trend) and its 6 bpw arm is not saturated, unlike both Qwen runs.

**One pattern, held loosely.** Both elevated models carry QK-norm; the flat one does
not. Per-head RMSNorm on q and k changes how weight error reaches attention scores,
so it is mechanically plausible — but family and QK-norm are perfectly confounded
across three points and scale moves too. A Llama-family model with QK-norm, or
gemma-4 at GQA=2, would separate them. Treat as a hypothesis, not a finding.

**Caveat on the Llama point**: its gap to trend is 0.98x the noise floor, so the
direction is safe and the magnitude is not — 1.3x +/- 0.3. Under 2.86x either way.

**So there is no portable dense rule.** The promotion framework stands: extra-budget
promotion composes to under 1%, the bar is `slope*delta / (1 - e^(-slope*delta))`,
and `-hq` clears it by 4.8x on an MoE. What does not transfer is *which* tensor to
promote. With the imatrix shortcut dead too, the only thing that identifies a
promotable dense tensor is measuring it. MoE has a categorical boundary worth
exploiting; dense does not.

Raw results: `~/qbench/results/{qwen8b,llama3b}-kproj/`, `~/qbench/qwen8b-kproj.yaml`,
`~/qbench/llama3b-kproj.yaml`.

## `-hq` beats its budget-matched control, and duty cycle is why (2026-09-20)

The companion to the section above, in the opposite regime: where the dense body
had to be searched for a tensor worth promoting, an MoE has a *categorical*
boundary that upstream already exploits. Whether it earned its keep was open,
because every published comparison was against a checkpoint at the same nominal
target *without* the boost — a smaller model, so the win needed no explanation
beyond more bytes.

`ornith-ai/Ornith-1.5-35B-A3B`, converted twice at `-b 4 -hb 6` with the same
pipeline, commit and calibration, differing only in `-hq`. Noise floor 0.014147;
body slope 1.1697/bpw from the 4->5 arms.

| arm | bpw | excess KLD | x floor |
|---|---|---|---|
| 4.00 non-`-hq` | 4.0469 | 0.028253 | 2.00 |
| 4.00 **`-hq`** | 4.1304 | **0.015719** | 1.11 |

**For +0.0835 bpw (+0.327 GiB) it cuts excess KLD by 44.4%.** Priced against the
uniform alternative — the same bytes spread over the whole body, 0.025623 — `-hq`
comes in at **-38.7%**. So it is not merely a bigger model: it beats the
budget-matched control decisively, which is the comparison that had never been run.

**The concentration is enormous.** The promoted set is 4.18% of body weights and
carries **49.1% of the KLD**, which is **11.8x** average per-parameter sensitivity
against a bar of 2.47x — clearing by 4.8x. Nothing in the dense body of
Qwen3-0.6B came within a factor of three of this; the best role there was k_proj at
3.96x.

**Duty cycle accounts for it, as predicted before the arm landed.** Ornith routes
**top-8 of 256 experts**, so a routed expert sees 3.1% of tokens while attention and
the shared expert see all of them — a **32x** gap. Measured concentration is 11.8x,
the same order and comfortably inside the ceiling that gap allows. The prediction
was made with the falsifier attached (near break-even would have killed the duty-cycle
story) and it survived.

**Consequence for `select_hq_bits = 2`.** The dense result argues for spreading bits
at +1 rather than concentrating at +2, because the +1 bar is 1.76x where the +2 bar
is 2.74x. That argument **does not transfer here**: at 11.8x both bits clear with
room to spare, and concentration is correct precisely because the sensitivity is
concentrated. The default is right for MoE and suspect for dense.

**Two caveats.** The `-hq` arm sits at 1.11x the noise floor against non-`-hq`'s
2.00x, so the ratio is well resolved but the derived 11.8x is the softest number
here — +/-5% on the hq excess moves it ~6%. And **perplexity cannot corroborate
this one**: the `-hq` arm reads 11.6300 against a floor of 11.6350, i.e. below it,
so ppl is saturated at 4 bpw on this model and KLD carries the result alone. Every
other finding in this sequence had two metrics agreeing; this one does not.

Raw results: `quantization/work/Ornith-1.5-35B-A3B-exl3/main/qb_results.json`.

## Known limitations, and what closing them would unlock

**No noise floor.** The `vllm` engine has no noise-injection (self-noise-floor)
support, so it cannot be the `reference` group with `noise_floor` left at its
default. vLLM's decoder layers are not at a predictable, engine-version-stable
location the way `TransformersBackend`'s forward-hook approach needs one. Tracked
as TODO `qbench-noise-floor`.

**No GGUF through `vllm-gguf-plugin`**, as above.

**The `llamacpp` engine has always run on CPU**, and silently. The installed
`llama-cpp-python` (0.3.34) is a CPU-only wheel — `libggml-cpu.so`, no `libggml-cuda.so`,
zero CUDA symbols — so the engine's `n_gpu_layers = 999` default is accepted and ignored:
no error, no warning, no offload. Established 2026-09-01 from `nvidia-smi` reading 0%
against 2218% CPU, and confirmed by `llama_cpp.llama_supports_gpu_offload()` returning
False. Every GGUF arm this project has run was therefore CPU-bound, which is the whole
explanation for those arms feeling slow.

**No recorded result is affected** — qbench has no time axis, and llama.cpp's CPU and CUDA
paths agree on output; if anything the CPU path is the better reference, for the same
reason the SINQ arms take its PyTorch path over gemlite's autotuned kernel. What it would
affect is the cross-engine idea sketched below: comparing `vllm-gguf-plugin` against
"native llama.cpp on the same checkpoint" is meaningless on *throughput* while one side is
on CPU by accident, though still sound on quality. Building with CUDA needs a source
install (`CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-binary llama-cpp-python
llama-cpp-python`), and the engine should assert `llama_supports_gpu_offload()` rather
than trust the flag.

**`options.quantize` cannot reach a model larger than the box**, which is what stops SINQ
arms being added to the gemma-4-12B project. It quantizes *after* an ordinary
`from_pretrained`, so the full bf16 model has to be resident first: 22.3 GiB of weights
against 16.3 GiB of VRAM and ~20 GiB of RAM. `device_map="auto"` does not rescue it —
accelerate assigns the overflow to meta and SINQ's `_patch_other` does `layer.to(device)`
on it, so it fails in 4 s with `NotImplementedError: Cannot copy out of meta tensor`
rather than OOMing. Neither does disk offload: SINQ patches by walking modules, not
through forward, so accelerate's materialize-on-access hooks never fire.

*Two adjacent bugs, both fixed, both of the silent kind.* `streaming: true` combined with
`quantize` ignored the spec and scored the **unquantized** model against the unquantized
reference — KLD ~0, a column of zeroes indistinguishable from a lossless quantizer. It
raises now. And `BaseQuantizeConfig`'s own default `method` is **`"dual"`, not `"sinq"`**
— a variant keeping fp16 metadata, 4.5104 bpw against sinq's 4.2761 at nbits=4/group 64 —
so a project file omitting `method` silently mixed two quantizers into one sweep. The
option now defaults to the named method and prints the effective config.

*The fix, when it is wanted*: quantize per block during a streaming materialization,
never holding the whole bf16 model. `sinq.sinqlinear.SINQLinear(linear_layer, cfg,
del_orig=True, ...)` is a per-`nn.Linear` constructor that frees the original as it goes,
so the shape is: walk the decoder blocks in checkpoint order, materialize one from the
shards, replace its `nn.Linear` children, keep the (small) quantized result resident, move
on. Peak is one bf16 block plus the accumulating quantized model — ~8.5 GiB for
gemma-4-12B at 4 bits, comfortably resident — after which scoring uses the ordinary
non-streaming path. The existing streaming machinery already materializes per module for
the reference pass; what differs is keeping the result instead of returning it to meta.

**The `vllm` engine cannot be isolated from the rest of a project run**, and the design
for fixing it is recorded here rather than queued, because the auto-sized
`gpu_memory_utilization` (2026-09-01) turned the symptom from a failure into a
degradation. Revisit if a project starts failing again, or if a vLLM crash costs a long
run.

*The cheap version is closed.* `VLLM_ENABLE_V1_MULTIPROCESSING=0` is load-bearing:
the capture patches `compute_topk_scores`, which runs **in the worker**, so with
multiprocessing on the worker is a subprocess that never sees the monkeypatch and the
full-vocabulary capture silently stops being the code that runs. Installing it there via
a `vllm.general_plugins` entry point (which does load in worker processes) does not help
either — the captured tensor would then be in the worker's address space and the consumer
that finalizes the row is in the frontend, at 1.16 GiB per row.

*Which is what says where the boundary goes:* not the engine, but **the engine and the
reduction together**. The capture needs worker, frontend and callback sharing an address
space, so all three go in the child and only statistics come back. The surface is the
nine-line per-model block in `qbench.py`: in go `mspec`, `max_len`, `device`, `ids`,
`ranges`, `vocab_size` and `ref_store` — which is a **path**, the load-bearing detail,
since the child then loads reference logits off disk and no logits ever cross the pipe;
back come `stats.results()`, `stats.kl_vector()` (~80 KB at 10x2048) and `backend.info`.
A wrapper implementing the existing four-member backend contract, ~100-150 lines, with no
change to any engine and none to the rest of `qbench.py`.

*What it would buy beyond the leak*: `options.env` becomes genuinely isolated instead of
save-and-restore, and a vLLM crash stops taking a whole project run with it. *What it
costs*: an interpreter plus torch/vLLM import per vllm arm (~15-30 s on top of engine
init), care that a child traceback does not vanish, and `noise_eps`, which does not
survive the boundary — though this engine has no noise injection anyway.

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

## Adding SINQ as a comparator arm: what it would actually cost (2026-09-01)

`huawei-csl/SINQ` (Apache-2.0, ICML 2026) — Sinkhorn-Normalized Quantization, dual
row/column scales with iterative Sinkhorn-style normalization to even out variance
across quantization groups. **Calibration-free**, 2/3/4/5/6/8 bits at group size 64 or
128, weight-only, symmetric/asymmetric plus an NF4 variant. Pre-quantized checkpoints
exist under the `huawei-csl` org.

**Reconnaissance says this is a couple of edits, not a project.**

- **`SinqConfig` is already in the installed transformers (5.15.0)**, native since
  Feb 2026, so a pre-quantized checkpoint loads through the ordinary
  `AutoModelForCausalLM.from_pretrained` path. The runtime `sinq` package is a separate
  `pip install` — transformers ships only the config class.
- **The streaming allowlist does not bite.** `TransformersBackend` raises
  `"Streaming does not support this quantization_config"` for anything outside
  {compressed-tensors, modelopt, fp8, mxfp4}, but `streaming` defaults to **False**, and
  the default `device_map` load has no such gate.
- **One real edit, in the place this file keeps finding bugs.** `_quant_bits()` reads
  `bits` / `w_bit` / `weight_bits`; `SinqConfig.to_dict()` emits **`nbits`** (verified
  against the installed class, alongside `group_size`, `tiling_mode`, `method`, and
  `quant_method: sinq`). None of the three keys match, so the size axis comes back
  `None` — the fourth instance of the accounting-bug class enumerated above.
- **Storage suffixes are the unknown.** `_KNOWN_SUFFIXES` covers CT, EXL3 and classic
  GPTQ/AWQ; SINQ's dual-scale tensors are named something else, and an unrecognized
  suffix is mis-bucketed rather than rejected. **`check_against_disk` is what makes this
  cheap**: it classifies every on-disk tensor as counted, expected-absent or
  *unexplained*, so a wrong suffix list surfaces as a number rather than as a plausible
  point on a plot. The apparatus built after the first three bugs is precisely what makes
  the fourth format a small job.

**What the arm would and would not show.** SINQ's headline is *speed* — ~21 s for
Qwen3-14B, claimed ~2x HQQ and ~31x AWQ/GPTQ — and qbench has **no time axis**, so the
thing SINQ is actually selling is invisible here. What the plot would answer is the
question worth asking anyway: how far calibration-free dual-scaling gets at matched
total bytes against EXL3's calibrated trellis. Quantization wall-clock is a real
operational number for the appliance, and if it is wanted it belongs beside the plot as
a recorded fact, not as a qbench axis.

**Same lab as KVarN, and the same idea.** KVarN's KV cache is Hadamard rotation plus
"iterative Sinkhorn-like variance normalization"; SINQ is that normalization applied to
weights. They are one research program in two places, so a SINQ result at matched bytes
is also weak evidence about the KV claims — see the ecosystem field notes.

### The arm exists: first SINQ numbers, and three accounting gaps it exposed

Quantizing locally rather than pulling a prequant was the right call — it gave a known
configuration and it is *fast*: `Qwen/Qwen3-0.6B-Base` at 4 bits took **2.3 s for g64 and
1.8 s for g128**, load included, on one 5070 Ti. The speed claim is not marketing.

`~/qbench/qwen-0.6b-sinq.yaml`, 10 rows x 2048, `openwebtext10k`, reference
`Qwen3-0.6B-Base` bf16:

| | bpw_layer | vram_gb | ppl | KLD |
|---|---|---|---|---|
| HF BF16 (reference) | 16.000 | 1.3999 | 18.2025 | — |
| Noise floor | 16.000 | 1.3999 | 18.2059 | 0.00159 |
| AWQ 4bit (vLLM) | 4.156 | 0.5029 | 34.2096 | 0.63316 |
| **SINQ 4bit g128** | 4.143 | 0.5022 | 21.5154 | **0.16827** |
| **SINQ 4bit g64** | 4.276 | 0.5090 | 20.6813 | **0.13145** |

SINQ g128 lands within 0.013 bpw and 0.7 MiB of the AWQ arm — as close to matched bytes
as two independently-produced checkpoints get — at **3.8x lower KLD**. Calibration-free.

**Read it as a smoke trace, not a verdict**, for two reasons beyond the 0.6B model and
ten rows. The arms **differ in engine as well as format** (SINQ through `transformers`,
AWQ through `vllm`), which is the exact confound the EXL3 / EXL3-vLLM split exists to
keep visible. And the AWQ checkpoint is one community quantization of unknown care, not
a controlled AWQ baseline. What the run does establish is that the arm works end to end
and the axes are trustworthy.

**Three accounting gaps, each reporting a plausible number rather than an error** — the
pattern this file keeps finding, now four and five and six:

1. `_quant_bits` knew `bits` / `w_bit` / `weight_bits`; SINQ spells it **`nbits`**.
2. Storage suffixes. `W_q` is int-packed exactly like `qweight` and shares its numel
   math, so it feeds the same slot. Its sidecars needed full dotted paths, because
   **SINQ quantizes its own scales and zeros** and the leaf names of that second-order
   metadata are the single letters `m`, `s`, `x`. Bare, they would match anything.
3. **The one worth carrying elsewhere: SINQ's weights are invisible to a parameter
   walk.** `W_q` is a plain tensor attribute and the scales are a plain Python dict, so
   neither `named_parameters()` nor `named_buffers()` yields them. The live-module path
   in `TransformersBackend` therefore saw only norms and the embedding and reported
   `bpw_layer 0.0` with a `vram_gb` covering the embedding alone — silently. **Any tool
   that measures a model by summing `p.numel()` has the same blind spot on this format**,
   including every memory profiler and every `sum(p.numel() for p in model.parameters())`
   in a README. Fixed by falling back to `safetensors_storage_info`, which is the better
   source regardless: it is what `check_against_disk` validates, and it cannot be fooled
   by how a loader chooses to attach its tensors.

Gaps 1 and 2 were caught by `check_against_disk` immediately and by name
(`bpw_layer is 0.0, so that bucket matched no tensor at all; 0.219 GiB ... matched no
bucket`, naming `model.layers.0.mlp.down_proj.W_q`). Gap 3 was **not** — that path
computes its own numbers and never consults the disk check, which is why it survived a
fix that made the standalone function correct. The guard was real and the gap was
outside it.

**A fourth, in SINQ itself, and it is a bug worth reporting.** A checkpoint saved by
SINQ's own transformers integration **cannot be loaded with `device_map="auto"`**: that
routes through transformers' native `SinqConfig` quantizer, which builds
`sinq.sinqlinear_hf.SINQLinear` modules and never marks them ready, and the failure
arrives as `AssertionError: model was not quantized` at the **first forward**, not at
load. An explicit device (`device_map="cuda:0"`) reaches SINQ's own patched loader
(`sinq.sinqlinear.SINQLinear`, `ready=True`) and works. Two loaders selected by an
argument that has nothing to do with which one is wanted. qbench already exposes
`device_map` as an option, so the project file sets it and no harness change was needed.

**Results are cached per model** (`_logit_cache/qbench/results_*.json`, keyed on data,
reference, model spec and `METRICS_VERSION`), and the cache stores `backend.info`
alongside the metrics. A harness fix to the *accounting* therefore does not invalidate
anything — a rerun replays the stale numbers and looks like the fix failed. Delete the
matching `results_*.json` (the manifest maps hashes to labels) when changing how storage
is measured.

### The full cross-format table, and which thirds of it are comparable (2026-09-01)

SINQ folded into `qwen-0.6b.yaml`. The result looks, at first glance, like SINQ beating
EXL3 by 3x. It is not that, and the reason is worth being precise about.

| | engine | bpw_layer | bpw_head | bpw_embed | vram_gb | ppl | KLD |
|---|---|---|---|---|---|---|---|
| HF BF16 (reference) | transformers | 16.00 | 16.00 | 16.00 | 1.3999 | 18.20 | — |
| Noise floor | transformers | 16.00 | 16.00 | 16.00 | 1.3999 | 18.21 | 0.00159 |
| **BF16 via vLLM** (control) | vllm | 16.00 | 16.00 | 16.00 | 1.3999 | 18.20 | **0.00150** |
| SINQ 4bit g64 | transformers | 4.28 | 16.00 | 16.00 | 0.5090 | 20.68 | 0.13145 |
| SINQ 4bit g128 | transformers | 4.14 | 16.00 | 16.00 | 0.5022 | 21.52 | 0.16827 |
| AutoRound 4bit | vllm | 4.18 | 16.00 | 16.00 | 0.5040 | 31.31 | 0.54395 |
| AWQ 4bit | vllm | 4.16 | 16.00 | 16.00 | 0.5029 | 34.21 | 0.63316 |
| EXL3 4.00 bpw | vllm | 4.02 | 6.02 | 6.02 | 0.3152 | 29.98 | 0.50243 |
| EXL3 3.50 bpw | vllm | 3.52 | 6.02 | 6.02 | 0.2896 | 31.88 | 0.55904 |
| EXL3 3.00 bpw | vllm | 3.02 | 6.02 | 6.02 | 0.2639 | 34.49 | 0.64493 |
| EXL3 2.75 bpw | vllm | 2.77 | 5.02 | 5.02 | 0.2329 | 37.54 | 0.72497 |
| Q4_K_M | llamacpp | 4.78 | 6.56 | 4.50 | 0.4452 | 30.67 | 0.53008 |
| IQ3_M | llamacpp | 3.67 | 6.56 | 3.44 | 0.3694 | 35.18 | 0.66610 |
| IQ2_M | llamacpp | 2.76 | 5.50 | 3.44 | 0.3032 | 74.72 | 1.40830 |

**First suspicion, and it was wrong.** Every non-`transformers` arm scores KLD ≥ 0.50 and
every `transformers` arm ≤ 0.17, with nothing in between, across three engines and five
formats from 2.75 to 4.78 bpw. A gap that lands exactly on engine boundaries rather than
on a bitrate axis is the signature of an engine artifact, and SINQ shares its engine with
the reference — which would have handed it a free advantage.

**The control refutes that.** The same unquantized bf16 weights through vLLM score
**KLD 0.00150 / ppl 18.199** against the transformers reference — *below* the reference's
own noise floor of 0.00159. The engines agree to four decimal places on ppl. Whatever
separates these arms, it is not the engine, and no cross-engine correction is warranted.
`~/qbench/qwen-0.6b-enginectl.yaml`; **this control should exist in every cross-engine
project file**, because it costs one arm and it is the only thing standing between a
format claim and an engine claim.

**So the table is real — but it contains one clean comparison, one other clean
comparison, and one that is not a comparison at all.**

*Clean, and the headline:* **SINQ vs AWQ vs AutoRound.** All four arms carry a bf16 head
*and* a bf16 embedding, and land within 1.3% of each other on total bytes
(0.5022–0.5090 GiB). Everything but the body quantizer is held fixed. SINQ is **3.2x
better than AutoRound and 3.8x better than AWQ**, calibration-free, on a 1.8-second
quantization. That result stands as measured.

*Clean:* **EXL3 vs GGUF.** Both quantize head and embedding, so both are honest on the
vram axis. EXL3 4.00 bpw beats Q4_K_M at 0.502 vs 0.530 KLD using **29% fewer bytes**.

*Not a comparison:* **SINQ vs EXL3.** It is confounded twice, both ways favouring SINQ.
EXL3 carries a **6.02-bit head and embedding** against SINQ's bf16 — and the head is the
tensor that produces the logits KLD is computed on, at 26% of this model's weights
(151936 x 1024 of 596M). And SINQ occupies **0.5022 GiB against EXL3-vLLM's 0.3152 —
59% more memory**. The EXL3 arm at SINQ's footprint would sit well above 4 bpw, off the
top of the measured range. Read down the `vram_gb` column rather than the `bpw_layer`
column and the two are not near each other at all.

That EXL3 at 4.02 bpw *already beats* AWQ at 4.16 (0.502 vs 0.633) **while also
quantizing its head to 6 bits**, where AWQ pays nothing for its bf16 head, is the
measurement that shows how large the handicap is.

**What a controlled sweep needs.** The interesting question — how does calibration-free
dual-scaling compare with a calibrated trellis at matched *total bytes* — is untouched by
this table. The cheap way to reach it is to stop excluding SINQ's embedding: this run
passed `modules_to_not_convert=["lm_head"]`, and on a tied model that leaves 311 MiB of
bf16 embedding, **57% of the checkpoint**. Quantize it and SINQ lands near EXL3's
footprint, where the comparison means something. The alternative — EXL3 arms converted at
`head_bits 16` — answers the same question from the other side and costs a conversion per
point. Prefer the first: total bytes is the axis the appliance cares about, and it is the
axis this file exists to keep honest.

Two notes for whoever runs it. The head/embed treatment splits by *format family*, not by
bitrate, so any table mixing families needs the `bpw_head` and `bpw_embed` columns
visible or it will be misread exactly the way this one was. And [the head-bitrate
study](#head-bitrate-6-is-defensible-and-the-lever-does-not-want-pulling-2026-08-25) does
not transfer: it was budget-neutral, trading head bits against body bits at constant
bytes, so it says the optimum trade is 5-6 — not what a bf16 head buys at a fixed body
bitrate, which is the quantity that matters here and is unmeasured.

### Isolating the body quantizer: the head was not the confound, and the advantage is bitrate-local (2026-09-01)

Two corrections to the section above, both from measurement.

**The head/embedding confound was the wrong explanation.** `head_quant: {bits: 16}` on
the exllamav3 arms replaces the checkpoint's quantized head with the dense tied
embedding, and that engine already keeps the embedding at bf16 — so every arm below
carries a bf16 head *and* a bf16 embedding, and only the body quantizer varies. (Note
the direction: `embed_quant` / `head_quant` / `embed_file` are **`Exl3Backend`-only
options**, so normalization has to run toward bf16 rather than pushing SINQ down to a
6-bit head. Giving the transformers engine the same knobs is the missing piece if the
other direction is ever wanted.)

Removing EXL3's 6-bit head changed **nothing**:

| EXL3 arm | 6-bit head (vLLM) | bf16 head | Δ |
|---|---|---|---|
| 4.00 bpw | 0.50243 | 0.50170 | −0.0007 |
| 3.50 bpw | 0.55904 | 0.55757 | −0.0015 |
| 3.00 bpw | 0.64493 | 0.64263 | −0.0023 |
| 2.75 bpw | 0.72497 | 0.72436 | −0.0006 |

All four inside the ~1% run-to-run variation this harness shows. **The 6-bit head costs
EXL3 essentially nothing on this model** — independent corroboration of the head-bitrate
result above, arriving from the opposite direction and without a budget-neutral trade.
The confound named in the previous section was real in principle and empty in practice;
what remains of it is the byte axis, not the quality axis.

**And then the sweep, which is the actual finding.** Every arm bf16 head and embedding:

| | bpw_layer | vram_gb | ppl | KLD |
|---|---|---|---|---|
| Noise floor | 16.00 | 1.3999 | 18.21 | 0.00159 |
| SINQ 4bit g64 | 4.28 | 0.5090 | 20.68 | **0.13145** |
| SINQ 4bit g128 | 4.14 | 0.5022 | 21.52 | **0.16827** |
| AutoRound 4bit | 4.18 | 0.5040 | 31.31 | 0.54395 |
| AWQ 4bit | 4.16 | 0.5029 | 34.21 | 0.63316 |
| EXL3 4.00 bpw | 4.02 | 0.7858 | 29.97 | 0.50170 |
| EXL3 3.50 bpw | 3.52 | 0.7602 | 31.82 | 0.55757 |
| SINQ 3bit g64 | 3.48 | 0.4680 | 37.61 | **0.73513** |
| SINQ 3bit g128 | 3.34 | 0.4613 | 54.89 | **1.11322** |
| EXL3 3.00 bpw | 3.02 | 0.7346 | 34.44 | 0.64263 |
| EXL3 2.75 bpw | 2.77 | 0.7216 | 37.52 | 0.72436 |
| SINQ 2bit g64 | 2.28 | 0.4065 | **269,529** | **9.63180** |
| SINQ 2bit g128 | 2.14 | 0.3997 | **2,550,682** | **11.82632** |

**The two curves cross between 4 and 3 bits, and below that SINQ does not degrade — it
fails.** At 4 bits SINQ is 3.8x better than EXL3. At ~3.4 bits it is already *worse* than
EXL3 at 3.02 (0.735 vs 0.643), and SINQ 3bit g128 at 3.34 bpw is beaten by EXL3 at
**2.75** bpw. At 2 bits SINQ produces perplexities in the hundreds of thousands and
millions — not a degraded model, a destroyed one — where EXL3 at 2.77 still sits at
0.724.

That is the textbook signature of the two families: **scalar RTN with good normalization
is excellent where the grid is dense enough and falls off a cliff when it is not, while a
calibrated trellis degrades gracefully.** EXL3 moves only 0.502 → 0.724 across
4.02 → 2.77 bpw; SINQ moves 0.131 → 11.8 across 4.28 → 2.28. The low-bitrate regime is
precisely what the QuIP#/QTIP lineage exists for, and this is what that looks like
measured.

**What this means for the project.** The 4-bit result is real and should not be dismissed
— at 4 bits, calibration-free dual scaling beats a calibrated trellis by 3.8x here, on a
1.5-second quantization, and that is worth understanding. But the operating range this
project cares about is 2–4 bpw, and SINQ is not a competitor there at any bitrate below
about 3.5. Two caveats before generalizing: this is a **0.6B model**, where every format's
low-bitrate behaviour is at its worst and the crossover point is likely to move down on a
larger one; and on **total bytes** SINQ still carries the bf16 embedding (57% of its
checkpoint), so its `vram_gb` column is not a deployment figure.

The larger-model sweep is still worth running — but to locate *where* the crossing is,
not to ask whether there is one.

### SINQ's remaining knobs: A-SINQ is free, 2D is not, and neither moves the crossover (2026-09-01)

Both variants needed a harness change to reach at all. **2D-tiled checkpoints quantize
and serve correctly but do not survive `save_pretrained` -> `from_pretrained`** (a shape
mismatch at the first forward; verified in-process generation is fine, so it is the
serialization that is broken). **A-SINQ is refused outright by the transformers
integration**, which points at the official repo. `TransformersBackend` therefore gained
`options.quantize`, which runs SINQ's own `quantize_model()` on the freshly loaded bf16
model — also just cheaper for a sweep, at ~1.5 s and no checkpoint per point. Its storage
accounting agrees with the on-disk figure to four decimals on the configs that can be
saved (4.2761 / 4.1432).

All arms bf16 head and embedding, group 64:

| | bpw_layer | vram_gb | ppl | KLD | vs 1D SINQ |
|---|---|---|---|---|---|
| SINQ 4bit 1D | 4.28 | 0.5090 | 20.67 | 0.13125 | — |
| SINQ 4bit 2D | 4.52 | 0.5213 | 20.73 | 0.12958 | −1.3% for **+5.6% bits** |
| A-SINQ 4bit 1D | 4.28 | 0.5090 | 20.57 | 0.12448 | **−5.2% at equal bits** |
| A-SINQ 4bit 2D | 4.52 | 0.5213 | 20.47 | 0.11952 | −8.9% for +5.6% bits |
| SINQ 3bit 1D | 3.48 | 0.4680 | 37.62 | 0.73542 | — |
| SINQ 3bit 2D | 3.72 | 0.4803 | 38.96 | 0.76999 | **+4.7% for +6.9% bits** |
| A-SINQ 3bit 1D | 3.48 | 0.4680 | 36.56 | 0.71145 | **−3.3% at equal bits** |
| A-SINQ 3bit 2D | 3.72 | 0.4803 | 36.19 | 0.69214 | −5.9% for +6.9% bits |
| EXL3 4.00 bpw | 4.02 | 0.7858 | 29.97 | 0.50170 | |
| EXL3 3.00 bpw | 3.02 | 0.7346 | 34.44 | 0.64263 | |

**A-SINQ is a free 3-5%.** Identical bitrate, strictly better at both widths. Use it;
it changes no conclusion.

**2D tiling costs 0.24 bpw and does not repay it.** Counted honestly it buys 1.3% of KLD
for 5.6% more bits at 4 bits, and at 3 bits with plain SINQ it is *strictly dominated* —
worse KLD **and** more bits. Only in combination with A-SINQ does it come close to
paying, and the comparison it needs (1D at the same 4.52 bpw) is not in this table. The
bpw column is the whole reason this is visible; on a bits-nominal axis 2D looks like a
free win.

**The crossover does not move.** The best 3-bit configuration available — A-SINQ 2D at
0.69214 — is still worse than EXL3 at 3.00 bpw (0.64263) **while spending 23% more bits**
(3.72 vs 3.02). At 4 bits the advantage instead grows slightly: A-SINQ 1D is 4x better
than EXL3 at 4.00 with 6% more bits.

So the parameter axis is exhausted, and it did not extend SINQ's strong range downward at
all. **What remains open is the model-size axis**, which is the sweep worth running: the
question is whether a larger model moves the crossing below 3 bpw, and nothing measured
here bears on it.

### gemma-4-12B: the ordering reverses, and 0.6B was the wrong instrument (2026-09-01)

The SINQ arms now reach a 12B model (block-wise quantization, above). The result inverts
the Qwen3-0.6B finding completely.

| | group | bpw_layer | bpw_head | bpw_embed | vram_gb | ppl | KLD |
|---|---|---|---|---|---|---|---|
| HF BF16 (reference) | — | 16.00 | 16.0 | 16.0 | 24.057 | 17.687 | — |
| Noise floor | — | 16.00 | 16.0 | 16.0 | 24.150 | 17.681 | 0.00176 |
| **EXL3 4.00 bpw** | exllamav3 | 4.01 | 6.0 | 16.0 | 7.662 | 17.898 | **0.02696** |
| **EXL3 4.00 bpw** | vLLM | 4.05 | 6.0 | 6.0 | **5.868** | 17.937 | **0.04858** |
| EXL3 3.50 bpw | exllamav3 | 3.51 | 6.0 | 16.0 | 7.027 | 18.149 | 0.07026 |
| Q4_K_XL | GGUF | 4.88 | 5.5 | 5.5 | 6.843 | 18.098 | 0.07132 |
| EXL3 3.50 bpw | vLLM | 3.56 | 6.0 | 6.0 | 5.234 | 18.156 | 0.09135 |
| EXL3 3.00 bpw | exllamav3 | 3.01 | 6.0 | 16.0 | 6.393 | 18.456 | 0.10293 |
| EXL3 3.00 bpw | vLLM | 3.06 | 6.0 | 6.0 | 4.599 | 18.473 | 0.12354 |
| A-SINQ 4bit | transformers | 4.27 | 16.0 | 16.0 | 7.296 | 19.984 | 0.15861 |
| SINQ 4bit | transformers | 4.27 | 16.0 | 16.0 | 7.296 | 20.432 | 0.18576 |
| A-SINQ 3bit | transformers | 3.47 | 16.0 | 16.0 | 6.281 | 45.154 | 1.01148 |
| SINQ 3bit | transformers | 3.47 | 16.0 | 16.0 | 6.281 | 46.696 | 1.04736 |

**EXL3 at 4.00 bpw is 6.9x better than SINQ at 4 bits** (0.02696 against 0.18576) — and
the handicap runs *against* EXL3, which carries a 6-bit head where SINQ keeps bf16.
EXL3 at **3.00 bpw** beats SINQ at 4 bits on quality (0.10293 vs 0.18576) while using
**12% fewer bytes**, and on the vLLM path with its embedding served from the quantized
head it does so at 4.599 GiB against SINQ's 7.296 — **37% smaller**. GGUF's Q4_K_XL also
beats SINQ 4bit, by 2.6x. A-SINQ's calibration is worth more here than at 0.6B (15%
rather than 5%) and does not change the ordering.

At 3 bits SINQ collapses exactly as it did on the small model — ppl 45-47 against EXL3's
18.5 — so the cliff is not a small-model artifact even though the ranking above it was.

**The methodological finding is the bigger one: Qwen3-0.6B was not a valid instrument for
ranking formats, and the earlier section that used it should be read with that in mind.**
On the 0.6B model *every* arm sat in the badly-damaged regime — EXL3 4.00 bpw measured
KLD **0.50**, against **0.027** here. That is an 18x difference in what "4-bit EXL3" means,
and it is the model, not the format. SINQ moved the other way, 0.131 to 0.186. A
comparison run entirely inside a regime where the best available option is already 300x
its noise floor ranks the *damage patterns of a broken model*, not the formats.

**The tell to keep**: check where the best arm sits relative to the noise floor before
believing an ordering. On gemma-12B the best arm is 15x the floor and the spread is
resolvable; on Qwen3-0.6B it was 300x, and everything above it was compressed into a band
where the ordering did not survive a change of model. A format comparison needs a model
big enough that a good quantizer is *nearly lossless* on it, or it is measuring something
else.

What survives from the small-model work: SINQ quantizes extremely fast (30 s for 12B), it
is calibration-free, A-SINQ is a free improvement, 2D tiling costs more than it returns,
and the sub-3-bit cliff is real. What does not survive is any claim about how it ranks.

## Qwen3-8B: the first exemplar-grade cross-format table (2026-09-01)

The replacement for the retired 0.6B comparison. Dense uniform attention, untied
embeddings, official AWQ, published EXL3 ladder, bartowski GGUFs, SINQ quantized in
process. Noise floor **0.000992** — the lowest of any exemplar here, so there is real
resolvable range beneath every arm.

| | group | bpw_l | head | emb | vram_gb | ppl | KLD |
|---|---|---|---|---|---|---|---|
| HF BF16 (reference) | — | 16.00 | 16.0 | 16.0 | 15.256 | 15.378 | — |
| Noise floor | — | 16.00 | 16.0 | 16.0 | 15.256 | 15.406 | 0.00099 |
| **EXL3 4.0 bpw** | exllamav3 | 4.01 | 6.0 | 16.0 | 4.834 | 15.590 | **0.01426** |
| EXL3 4.0 bpw | vLLM | 4.01 | 6.0 | 16.0 | 4.834 | 15.616 | 0.01464 |
| **Q4_K_M** | GGUF | 4.79 | 6.6 | **4.5** | 4.676 | 15.343 | **0.02454** |
| **EXL3 3.5 bpw** | exllamav3 | 3.51 | 6.0 | 16.0 | 4.429 | 15.737 | **0.03326** |
| EXL3 3.5 bpw | vLLM | 3.51 | 6.0 | 16.0 | 4.429 | 15.769 | 0.03355 |
| A-SINQ 4bit | transformers | 4.27 | 16.0 | 16.0 | 5.770 | 15.628 | 0.04579 |
| AWQ 4bit | vLLM | 4.16 | 16.0 | 16.0 | 5.679 | 15.903 | 0.04925 |
| SINQ 4bit | transformers | 4.27 | 16.0 | 16.0 | 5.770 | 15.540 | 0.05147 |
| EXL3 3.0 bpw | exllamav3 | 3.01 | 6.0 | 16.0 | 4.025 | 16.160 | 0.05833 |
| EXL3 3.0 bpw | vLLM | 3.01 | 6.0 | 16.0 | 4.025 | 16.173 | 0.05853 |
| **IQ3_M** | GGUF | 3.58 | 6.6 | **3.4** | 3.622 | 15.495 | **0.08272** |
| EXL3 2.5 bpw | exllamav3 | 2.51 | 6.0 | 16.0 | 3.621 | 16.871 | 0.15215 |
| EXL3 2.5 bpw | vLLM | 2.51 | 6.0 | 16.0 | 3.621 | 16.894 | 0.15262 |

**The engine control holds across the whole ladder.** All four EXL3 rungs agree between
`exllamav3` and `vllm` to **≤0.0005 KLD**, across a 10x range of damage (0.0143 to 0.152).
Engine is not a confound in this table. (The bf16-through-vLLM control is separate, in
`qwen3-8b-enginectl.yaml`; it is not needed for this conclusion.)

**SINQ and AWQ are dominated, decisively.** EXL3 3.5 bpw is better than A-SINQ — the best
of the three — at **1.34 GiB fewer bytes**: 0.0333 against 0.0458 at 4.429 GiB against
5.770. That is 27% better quality for 23% less memory. SINQ's "3.8x better than AWQ"
finding from the 0.6B table does not survive either: here the three cluster inside 12%,
with A-SINQ marginally ahead of AWQ and plain SINQ marginally behind.

**Perplexity would have ranked this table wrong.** Q4_K_M scores ppl **15.343**, *below*
the bf16 reference's 15.378, while carrying 25x the noise floor in KLD. IQ3_M is +0.12 ppl
over reference against EXL3 4.0's +0.21 — and 5.8x worse in KLD. A quantizer can keep the
argmax well-ranked while substantially reshuffling the tail; ppl only asks about the
target token. On a ppl plot IQ3_M looks competitive with EXL3 at 4 bits and it is not
close.

### The embedding tax, finally measured on a fair fight

**Qwen3-8B is untied, and every EXL3 arm carries its embedding at 16 bpw** — 622M
parameters, **1.159 GiB**, sitting inside `vram_gb` and contributing nothing to
`bpw_layer`. GGUF quantizes it (4.5 bpw for Q4_K_M, 3.4 for IQ3_M). That single difference
is most of why the GGUF arms look competitive:

- **IQ3_M vs EXL3 2.5 bpw is a dead heat on bytes** — 3.622 against 3.621 GiB — and IQ3_M
  wins on quality by **1.8x** (0.0827 vs 0.1522). At the low end GGUF is genuinely ahead
  *as shipped*.
- Q4_K_M lands on the Pareto frontier between EXL3 3.5 and 4.0, at 4.676 GiB.

### Measured, not projected: what a block-quantized embedding buys (2026-09-01)

`tools/quantize_embedding.py` rewrites the embedding at **4.5312 bpw** in **5.9 seconds**,
hardlinking every shard it does not touch — 1.159 GiB down to 0.328, **0.831 GiB saved** —
on each of the 4.0 and 3.0 bpw checkpoints. Served through the vLLM plugin, which is what
understands the format; the `exllamav3` engine does not.

**The quality cost is below the noise floor.** Same weights, dense embedding vs blockq:

| | KLD dense | KLD + bq | cost | vs noise floor (0.000992) |
|---|---|---|---|---|
| 4.0 bpw | 0.014638 | 0.014749 | **+0.000111** | 11% of it |
| 3.0 bpw | 0.058529 | 0.058890 | **+0.000361** | 36% of it |

Quantizing 622M embedding parameters from 16 bpw to 4.53 costs, at 4 bits, about one ninth
of what this harness can resolve. Perplexity moved the *other* way at 4.0 bpw (15.5854
against 15.6156), which at this magnitude says the same thing. So the whole 0.831 GiB is
saving, and the byte axis landed where predicted — 4.003 GiB measured against 3.983
projected, 3.194 against 3.174.

**Both GGUF wins reverse.** The Pareto frontier on `vram_gb` is the two bq arms and
nothing else:

| | vram_gb | KLD | |
|---|---|---|---|
| **EXL3 3.0 bpw + bq** | **3.194** | **0.05889** | frontier |
| IQ3_M | 3.622 | 0.08272 | **dominated** — 13% larger *and* 1.40x worse |
| EXL3 2.5 bpw | 3.621 | 0.15215 | dominated |
| **EXL3 4.0 bpw + bq** | **4.003** | **0.01475** | frontier |
| EXL3 3.0 bpw | 4.025 | 0.05833 | dominated |
| EXL3 3.5 bpw | 4.429 | 0.03326 | dominated |
| Q4_K_M | 4.676 | 0.02454 | **dominated** — 17% larger *and* 1.66x worse |
| EXL3 4.0 bpw | 4.834 | 0.01426 | best KLD, at 0.83 GiB for 0.0005 |
| AWQ 4bit | 5.679 | 0.04925 | dominated |
| A-SINQ / SINQ 4bit | 5.770 | 0.04579 / 0.05147 | dominated |

The section above said the embedding tax was "most of why the GGUF arms look competitive".
That is now measured rather than argued: **IQ3_M's low-end win was entirely the
embedding**, and once EXL3 stops carrying 1.159 GiB of fp16 lookup table it is smaller
*and* better at both ends of the range. Every arm here that is not an EXL3+bq arm is
Pareto-dominated by one of the two.

### Which arms may be fitted, and which may only be plotted (2026-09-01)

A convention, because the two uses of this table pull in opposite directions and the
fractional-bitrate result above decides between them.

**Fractional EXL3 rungs stay in the cross-format arms.** 2.5 and 3.5 bpw are published,
downloadable checkpoints someone may actually choose, and the size axis exists precisely
because a reader treats it as "how big is this file". On a frontier plot the question is
*what can I run in N GiB*, and a 3.5bpw checkpoint answers it whether or not it sits on a
trend. They also carry the engine control: four paired `exllamav3`/`vllm` rungs span a 10x
range of damage, where the two integer rungs alone would span 4x and sit entirely in the
low-damage region — and that control has twice been the thing that distinguished a format
result from an engine artifact.

**But no curve may be fitted through them.** A fractional target is a mixture of K levels,
so it sits ~31% above the trend through its integer neighbours by construction — of which
+22-27% is the arithmetic-versus-geometric mean of the endpoints and only +7-8% is the
superposition penalty. Either way a fit over 2.5/3.0/3.5/4.0 is biased by a known offset
on half its samples. **Any slope, trend or extrapolation uses integer rungs only** — which
is why the blockq curve is built at 2.0/3.0/4.0/6.0 and deliberately skips 2.5 and 3.5,
despite those being the two cheapest checkpoints to add.

Stated as a rule: *plot every real operating point; fit only the unmixed ones.*

### The curve that rule buys (2026-09-01)

Four unmixed rungs, every one with the embedding block-quantized at 4.53 bpw, so nothing
in the series is carrying a 1.159 GiB fp16 lookup table and the bytes axis means the same
thing at every point:

| rung | vram_gb | ppl | KLD | x noise floor | KLD ratio per bit |
|---|---|---|---|---|---|
| 2.0 bpw + bq | 2.386 | 18.112 | 0.30599 | 308x | — |
| 3.0 bpw + bq | 3.194 | 16.185 | 0.05889 | 59x | **5.20x** |
| 4.0 bpw + bq | 4.003 | 15.585 | 0.01475 | 15x | **3.99x** |
| 6.0 bpw + bq | 5.620 | 15.355 | 0.00236 | **2.4x** | **2.50x** |

**It saturates, and the saturation is the useful part.** Each bit is worth ~5x at the
bottom of the range, 4x in the middle and 2.5x at the top — a curve bending toward the
noise floor rather than a straight line, which is what it must do since nothing can score
below the floor. Two consequences fall out:

- **6.0 bpw is effectively lossless here** — 2.4x the floor, and a perplexity *below* the
  bf16 reference (15.355 against 15.378). There is almost nothing left to recover above
  it, so bits spent past ~4-5 bpw buy progressively less.
- **2.0 bpw is a floor, not a configuration.** 308x the noise floor and +2.7 ppl. It marks
  where the curve turns vertical, which is the number an appliance needs when deciding how
  far down it can push, not a setting to serve.

The middle of the range is where the bits are worth most: **3.0 -> 4.0 bpw costs 0.81 GiB
and buys 4x**, which is the single best trade in the table — and it is very nearly the
same 0.83 GiB that the embedding was wasting for free. Put the other way: **quantizing the
embedding is worth about as much as a whole extra bit of body precision**, at 1/9th of the
noise floor in quality instead of 4x.

This is the first series in the project that may legitimately be fitted: no mixed-K
targets, one embedding treatment, one engine, one model.

**Consequence for `quantized-embeddings` and `repair-tool`:** the case is no longer
inferential. 0.831 GiB is larger than the gap between adjacent EXL3 rungs on this model,
it costs a ninth of the noise floor, and it takes six seconds per checkpoint. On a tied
model the same saving needs no tooling at all.
