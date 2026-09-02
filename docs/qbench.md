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

### What this licenses

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

At the block-quantized 4.25 bpw that `tools/quantize_embedding.py` already produces, that
embedding costs 0.308 GiB instead of 1.159 — **0.851 GiB off every EXL3 arm**:

| | measured | with a blockq embedding | KLD |
|---|---|---|---|
| EXL3 4.0 bpw | 4.834 | **3.983** | 0.01426 |
| EXL3 3.5 bpw | 4.429 | **3.578** | 0.03326 |
| EXL3 3.0 bpw | 4.025 | **3.174** | 0.05833 |
| EXL3 2.5 bpw | 3.621 | **2.770** | 0.15215 |
| Q4_K_M (already quantized) | 4.676 | 4.676 | 0.02454 |
| IQ3_M (already quantized) | 3.622 | 3.622 | 0.08272 |

EXL3 4.0 would beat Q4_K_M by **1.7x on quality at 15% fewer bytes**, and EXL3 3.0 would
beat IQ3_M at 12% fewer bytes. Everything above is a projection from the measured
embedding size, not a measured arm — but it is arithmetic on a tensor whose size is known
exactly, and the format work to realise it is already written. **This is the clearest
statement the project has of what `quantized-embeddings` is worth**, and it is worth more
than the body-bitrate choices being argued over elsewhere: 0.851 GiB is larger than the
entire gap between adjacent EXL3 rungs.
