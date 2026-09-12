# Where the bytes went: VRAM accounting on a card with none spare

How vLLM's memory budget is computed, which consumers it cannot see, and why
`gpu_memory_utilization` stops being a safe knob above ~0.85 on this box. Measured
2026-09-11 on the 16 GiB dev card (RTX 5070 Ti, 16303 MiB board) against
`turboderp/Qwen3.8-27B-exl3@3.00bpw`, `max_num_batched_tokens=512`,
`max_num_seqs=1`, `max_model_len=-1` (auto-fit from a declared 262144).

**Updated 2026-09-12.** The ~0.85 ceiling in that sentence *was* TurboQuant's prefill
transient. With it gone, 0.975 serves the full declared 262144 on the same model and card
("The payoff" below). What is still not safe is that 0.975 has to be found by hand, and
that a cold torch.compile cache moves the answer by 35K tokens.

The short version: **the budget's model of "everything that is not KV cache" is
backend-independent, and the truth is not.** All three attention backends report
`consumed memory (weights + non-torch) = 10.46 GiB`, byte-identical. What is actually
resident before the KV allocation ranges from 10.27 to 11.27 GiB, and the difference is
static attention-backend workspace that the profiling window never sees.

## Method, and the caveats that come with it

Six runs, driven by `~/ckpt/profile-completion.py`, logs in `~/ckpt/logs/`. Two classes:

- **At-ceiling**: `gpu_memory_utilization` set to the highest value that does not OOM
  for that backend/KV dtype — TurboQuant `4bit_nc` at 0.86, Triton fp8 at 0.98,
  FlashInfer fp8 at 0.96. Snapshot taken twice: after engine construction, and around a
  single `llm.chat()` with a ~100K-token prompt.
- **Forced OOM**: `kv_cache_memory_bytes=5882946304` (5.48 GiB), far more than fits.
  This is the controlled experiment. That path **skips memory profiling entirely**
  ([gpu_worker.py:527](../deps/vllm/vllm/v1/worker/gpu_worker.py#L527)), so the
  allocator snapshot taken at the OOM shows each backend's pre-KV baseline with nothing
  inferred and no profiler in the loop.

Instruments per snapshot: `torch.cuda.device_memory_used(0)`, `nvidia-smi`, and a
`torch.cuda.memory._snapshot()` trace summarized two ways — cumulative bytes per
allocation site for long-lived sites, and live bytes per site at the moment of peak
allocation.

Two caveats on the instrument itself:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set before the torch import. The
  ~0.1 GiB unexplained residual below is therefore the expandable-segments figure and
  would likely be larger without it.
- The long-lived summary reports *cumulative* bytes per site, not live bytes. Harmless
  for sites that allocate once (it reads 1.001 GiB where the peak summary reads 1.000),
  but it would overstate any site that resizes repeatedly.
- **The site-attribution instrument has a context ceiling, and it is well below the one
  we now serve.** `torch.cuda.memory._record_memory_history` OOMs the *host* at the
  allocation count a ~256K-token prefill produces, so every per-site number in this note
  is from a ~100K prompt. Runs at capacity can be observed (they complete, and
  `profile-completion.py`'s `live` mode reports rate) but not attributed. Anything claimed
  about where bytes go at full context is therefore an extrapolation from the shape at
  100K — defensible now that nothing in the prefill path is sized by context, and not
  before.

## The two external instruments agree — on headroom, not on "used"

`torch.cuda.device_memory_used` reads consistently ~421 MiB higher than `nvidia-smi`'s
usage column, which is what first looked like an instrument bug. It is not. The board is
16303 MiB (15.92 GiB) but CUDA offers vLLM only 15.51 GiB; the 421 MiB difference is
driver reserve. `device_memory_used` counts that reserve as used and measures against
the board total. `nvidia-smi`'s column and vLLM both work in the 15.51 GiB CUDA frame.

Subtract each from its own total and they agree to within 1 MiB on **all nine
snapshots**:

| run | snapshot | `device_memory_used` | headroom vs 15.92 | nvidia-smi | headroom vs 15.51 |
|---|---|---|---|---|---|
| TQ 4bit_nc 0.86 | startup | 14.875 | 1071 MiB | 14810 MiB | 1072 MiB |
| TQ 4bit_nc 0.86 | post-chat | 15.631 | 297 | 15584 | 298 |
| Triton fp8 0.98 | startup | 15.728 | 198 | 15684 | 198 |
| Triton fp8 0.98 | post-chat | 15.849 | 74 | 15808 | 74 |
| FlashInfer fp8 0.96 | startup | 15.769 | 156 | 15726 | 156 |
| FlashInfer fp8 0.96 | post-chat | 15.908 | 13 | 15868 | 14 |
| TQ 4bit_nc | OOM | 12.098 | 3916 | 11966 | 3916 |
| Triton fp8 | OOM | 11.103 | 4934 | 10948 | 4934 |
| FlashInfer fp8 | OOM | 11.473 | 4555 | 11326 | 4556 |

**Track headroom, never "used".** The two instruments disagree only about where zero is,
and neither is wrong. Any comparison that mixes them by absolute occupancy inherits a
421 MiB error.

That table also states the operating problem plainly: FlashInfer at its "safe" 0.96
finished a single chat **14 MiB from the wall**. "The highest utilization that does not
OOM" is not a stable property of a configuration — one more JIT-compiled kernel and it
is gone.

## The controlled experiment: what each backend costs before KV

From the three forced-OOM runs, where no profiling happens at all. The common base —
weights, rope tables, linear scratch, and a 0.031 GiB exl3 reconstruct buffer live at
the capture — is **10.272 GiB** in all three:

| backend | live peak at OOM | over common base | free at OOM (vLLM's own message) | vs Triton |
|---|---|---|---|---|
| Triton | 10.411 | +0.123 (transients) | 4.82 GiB | — |
| FlashInfer | 10.659 | **+0.385** | 4.45 | −0.37 |
| TurboQuant | 11.274 | **+1.000** | 3.82 | −1.00 |

Two independent instruments — the allocator trace and the free-memory figure in vLLM's
own OOM exception — agree to within 15 MiB. The sites:

- **TurboQuant, 1.000 GiB**: `worker/workspace.py:207 _ensure_workspace_size`, reached
  from [`TurboQuantMetadataBuilder._reserve_workspace`](../deps/vllm/vllm/v1/attention/backends/turboquant_attn.py#L227).
- **FlashInfer, 0.385 GiB**: `flashinfer.py:1037 _get_workspace_buffer <- attn_utils.py:148 init_attn_backend`.
- **Triton, none.** Its +0.123 is genuine transient activity (GDN, exl3 reconstruct) plus
  a 0.031 GiB deep_gemm buffer.

Meanwhile every at-ceiling run reports `consumed memory (weights + non-torch) = 10.46
GiB`. The same number for a backend carrying an extra gigabyte as for one carrying
nothing.

## FlashInfer is the control that proves it is a *window* problem

FlashInfer allocates a 0.385 GiB workspace **twice**, and the profiler sees exactly one:

- `_get_workspace_buffer <- init_attn_backend` — after the profiling window. Invisible.
- `_get_trtllm_workspace_buffer <- flashinfer.py:2362 forward` — allocated during the
  dummy run, so it lands in **peak activation: 0.56 GiB against 0.18 for the other two
  backends**. The delta is 0.38 against an actual 0.385. It is also why FlashInfer's
  first CUDA-graph capture reports 0.46 GiB where the others report 0.07.

Same backend, same buffer, same size, counted or not purely by which side of the
profiling window it was allocated on. **The profiler is not mismeasuring anything. It is
measuring a window that closes before the backends have finished allocating**, and no
amount of care inside that window can fix a consumer that appears after it.

## The budget arithmetic is exact — and two apparent overshoots are not real

Ruled out, because both were previously treated as evidence of drift:

**The printed sum balances to the printed precision, every time.**

| run | consumed | + activation | + KV in use | = | requested |
|---|---|---|---|---|---|
| TQ 0.86 | 10.46 | 0.18 | 2.7 | 13.34 | 13.34 |
| Triton 0.98 | 10.46 | 0.18 | 4.56 | 15.20 | 15.20 |
| FlashInfer 0.96 | 10.46 | 0.56 | 3.86 | 14.88 | 14.89 |

It balances because [gpu_worker.py:600](../deps/vllm/vllm/v1/worker/gpu_worker.py#L600)
folds the CUDA-graph *estimate* into `peak_activation_memory`. Adding the separately
printed *actual* graph figure therefore double-counts it. That is the source of the
"0.11 GiB past what was asked for" recorded at `gpu_memory_utilization=0.88` on
2026-09-10 — an artifact of reading the printout, not a measurement.

**The `--kv-cache-memory=` recommendation is not an instrument for overshoot.** It sits
~0.18 GiB below what was actually allocated in every run, which looked like the engine
confessing how far it had overrun. It is not:
[gpu_worker.py:818](../deps/vllm/vllm/v1/worker/gpu_worker.py#L818) subtracts a
deliberate 150 MiB `redundancy_buffer_memory` plus the actual CUDA-graph memory. Both
are constants of the printing. The line reads its own offset and will do so on every
startup forever.

Once the backend workspaces are accounted for, the residual is **0.08–0.12 GiB** across
all three runs — allocator and context slack, and close to what that 150 MiB buffer
exists to cover. Which means `--kv-cache-memory=<recommended>` is genuinely safe, while
`gpu_memory_utilization` is not: the utilization path omits both the buffer and the
workspaces.

## The profiled window is 512 tokens wide

Worth stating explicitly because it is easy to assume the profile run exercises the
declared context. It does not.
[gpu_worker.py:529](../deps/vllm/vllm/v1/worker/gpu_worker.py#L529) says so in a
comment: `profile_run()` "compiles the model for max_num_batched_tokens", which is 512
in these runs. Nothing in the profiled pass ever executes at max context.

That is the second half of the problem. TurboQuant's runtime prefill transient is
~6144 B/token of *cached* context (0.569 GiB live at 99,923 tokens, against 0.756 GiB of
actual card growth once the allocator's retained segments are counted). In a 512-token
window that is about 3 MB. Invisible for a structurally different reason than the
workspace: not "allocated after the window closed" but "the window never varies the axis
it scales on".

Chat-time growth, same three runs:

| run | chat growth | live peak during chat |
|---|---|---|
| TQ 4bit_nc 0.86 | 774 MiB | 0.612 GiB |
| Triton fp8 0.98 | 124 MiB | 0.105 |
| FlashInfer fp8 0.96 | 142 MiB | 0.105 |

FlashInfer's and Triton's chat peaks are identical — FlashInfer's whole extra cost is
static workspace. TQ's is six times either.

**TQ's row is now 154 MiB and no longer scales** ([`740dd8b6a`](../patches.md)); the
window being 512 tokens wide stopped mattering for this backend because nothing in the
path varies with the axis the window cannot vary. The section stands as the general
statement of the trap, not as a live defect.

## TurboQuant's reserve is sized by declared context, not served context

The two TQ runs settle this without reading the code. The at-ceiling run auto-fits
`max_model_len` from 262144 down to **150528**; the OOM run keeps **262144**. Both
reserve exactly **1.000 GiB**. Sized by the served length, the first would be 0.574 GiB.

1.000 GiB is 4096 B/token × 262144 exactly, which is worth one caveat: the reservation is
`2 × round_up(max_model_len − 1, block_size) × Hk × D × 2`, and with the logged attention
block size of 3072 that rounds 262143 up to 264192 and predicts 1.0078 GiB, not 1.000.
The observed figure implies `kv_cache_spec.block_size` divides 262144 and is therefore
not the 3072 in the log. The exact 4096 B/token was not reconciled against the spec's
`head_size` (the code notes it is a padded `effective_head_size`, not the model's
head_dim), and nothing here depends on it — the load-bearing claim is which *input* the
reservation is sized from, and two runs with different auto-fit outcomes and identical
1.000 GiB settle that.

`_auto_fit_max_model_len` does mutate
`model_config.max_model_len`
([kv_cache_utils.py:2176](../deps/vllm/vllm/v1/core/kv_cache_utils.py#L2176)), so the
reserve reads a value that is later revised down by 43% and never revisits it.

Two independent defects that compound rather than cancel:

- **Over-sized**: 1.000 GiB reserved against a served 150528 tokens needing 0.574. Some
  0.43 GiB is dead outright.
- **Uncounted**: the full 1.000 GiB is absent from every field of the budget.

Note the shape that is *not* what happens here: a context-scaled buffer allocated
*inside* the window would be counted at the declared length and overstate the final
cost. That is a real failure mode, but no backend in this dataset exhibits it —
FlashInfer's counted buffer is not context-scaled, and TurboQuant's context-scaled
buffer is not counted.

Only 2 of the 5 buffers the prefill path uses are covered by the reserve at all. The
three `_continuation_prefill` allocations (0.189, 0.190, 0.190 GiB at ~100K) are fresh
on top of it.

### The over-sizing was a high-water mark, and it is fixed

Settled 2026-09-12 with `VLLM_DEBUG_WORKSPACE=1`, no code change. The whole startup
logs exactly **two** resizes, and both land a second before auto-fit:

```
08:20:34 Resized workspace from 'turboquant_attn.py:243:_reserve_workspace': 0.00 -> 0.76 MB
08:20:34 Resized workspace from 'turboquant_attn.py:259:_reserve_workspace': 0.76 -> 1024.00 MB
08:20:35 Auto-fit max_model_len: reduced from 262144 to 150528
```

Neither hypothesis this note recorded is right on its own, and the fix is neither
deferring a call nor solving a fixed point. The builder is constructed **twice**:

- **First by the CUDA-graph memory profiler**, which stands up a full set of builders
  against a minimal KV cache, measures capture, and throws them away. It runs *after*
  the memory-profiling window closes and *before* auto-fit, so it reserves at 262144.
- **Then by the real `initialize_kv_cache`**, which reads the fitted 150528 and asks for
  588 MB. That read is **fresh, not stale** — and it logs no resize at all, because
  `_ensure_workspace_size` only ever grows.

So the gigabyte is a *discarded* builder's high-water mark. The profiling teardown
clears `attn_groups` and `kv_caches` and never touches the workspace, and growth is
one-way by design: reaching the runtime maximum before `lock_workspace` is exactly what
warmup is for.

**Fixed** in [`881c7345b`](../patches.md): bracket the profiling phase with the
workspace sizes taken on entry and shrink back to them in its teardown, so only growth
that phase caused is given back.

The first version simply released the workspace, which measures the same and is wrong.
`lock_workspace` says in its own comment that the maximum "should have been captured
during warmup/profiling" — growth is one-way *on purpose*, and `profile_run` runs before
this phase and inside the memory-profiling window. Dropping to zero discards its mark
too, and anything that grows the workspace during the profile pass but not during
capture would then grow it after the lock, which is an assertion at first inference
rather than a wrong number. No configuration on this card was found that actually does
that — TurboQuant's mark on entry is zero, and a Qwen3.5-35B-A3B MoE run never touches
the workspace at all — so the narrowing is defensive, taken because the failure it
avoids is a crash and the cost of avoiding it is one list of integers.

Measured A/B on this configuration, same session, patch stashed and unstashed:

| | before | after |
|---|---|---|
| reserve after auto-fit | 1024.00 MB | 588.00 MB |
| card occupancy at startup | 14810 MiB | 14370 MiB (**−440**) |
| card occupancy after a 99,923-token prefill | 15584 MiB | 15164 MiB (**−420**) |
| `max_model_len` | 150528 | 150528 |
| greedy continuation (64 ids) | — | identical |

Two things it does **not** do, both load-bearing for what comes next:

- **It does not buy context, only headroom.** `max_model_len` is unchanged, because
  auto-fit never saw the reservation on either side of the fix. Converting the 440 MiB
  into declared context is `kv-budget-margin`'s hook, not this.
- **It does not change the per-token rate.** 588 MB over 150528 tokens is the same
  4096 B/token as 1.000 GiB over 262144 — the reserve still prices *declared* context,
  it is now merely declared-after-fitting. The table below is unchanged, and removing
  the tax itself was step 3, below.

### And then the sizing itself went away

Step 1 fixed *which* `max_model_len` the reserve reads. It left the 4096 B/token rate
alone, and left the larger term — the ~6144 B/token of runtime transient — untouched.
[`740dd8b6a`](../patches.md) removes both, by not needing the cached context in one
piece at all.

Every query in a continuation chunk attends to *all* of the cached prefix: the causal
mask only starts biting inside the chunk. So the prefix can be cut anywhere and its
partial attentions merged by log-sum-exp, and the chunk is plain causal self-attention
against its own K/V. Both primitives were already in the tree —
`flash_attn_varlen_func(return_softmax_lse=True)` and `merge_attn_states` — and the
dequant kernels needed one argument to cover a slab. The buffers become the slab, the
partials become the chunk, and nothing is sized by the context.

Same model, card and prompt as above:

| | before step 1 | after step 1 | after this |
|---|---|---|---|
| reserve | 1024 MiB | 588 | **96** (a budget, not a shape) |
| occupancy at startup | 14810 MiB | 14370 | **13850** |
| occupancy after a 99,923-token prefill | 15584 | 15164 | **14004** |
| growth during that prefill | 774 MiB | 794 | **154**, and no longer scaling |

1580 MiB back at the peak against the original, on a card whose whole KV cache is
2.65 GiB. Prefill costs ~1% for it: 103.45s against 104.53s, best of three at 100K —
less than the 2–4% the attention alone measures, because the path being replaced also
allocated ~800 MiB per continuation step.

**The decomposition is exact; the arithmetic is not, and the difference is the floor.**
Per call the two implementations agree to 1.0–1.1 bf16 quanta — one representable step,
which is the least a restructuring that outputs bf16 can differ by. Through 20 layers
that reaches 1.06 nats on a *top-20 tail* token, while the top-1 moves 2.2e-03 and the
KL 2.6e-04, with no argmax flips and an identical greedy continuation at 100K. Worth
noting for the gate: `|dlogprob|` max over top-k is dominated by tail tokens and says
almost nothing about distribution here, where KL lands two orders under threshold. The
fast tier does not move at all, because no entry's prompt is long enough to chunk.

### What the one-quantum difference costs in accuracy

Logprobs on one prompt do not answer that, so it was measured on the two tasks
the TurboQuant survey uses, Qwen3-4B, `turboquant_4bit_nc`, chunked prefill at
`max_num_batched_tokens=2048` and prefix caching off so every item re-prefills.
Both tools now record how many calls each implementation served, because the
first attempt at this returned a clean null with the path never executed.

**Both controls -- the same configuration run twice -- came back at 0 discordant
pairs and 100% per-item agreement.** Nothing below means anything without that:
it is what makes a flip attributable to the decomposition rather than to the
harness.

**NIAH**, 32K context, 12 needles x 40 trials = 480 paired outcomes:

| arm | continuation calls | acc | lost | gain | p |
|---|---|---|---|---|---|
| monolithic | 19456 mono | 92.71% | — | — | — |
| monolithic, repeated | 19456 mono | 92.71% | 0 | 0 | control |
| slab @ 1 MiB (~320 merges) | 19520 chunked | 92.50% | 11 | 10 | 1.00 |
| slab @ 96 MiB (~3 merges) | 14400 chunked, 5120 mono | 92.50% | 21 | 20 | 1.00 |

**GSM8K**, 42 shots (8218 tokens), monolithic against the worst-case slab:

| sample | acc | lost | gain | p |
|---|---|---|---|---|
| first 200 problems | 90.50% -> 85.50% | 12 | 2 | 0.013 |
| **200 fresh problems** | **88.50% -> 88.50%** | **6** | **6** | **1.00** |
| pooled 400 | 89.50% -> 87.00% | 18 | 8 | 0.076 |

**The reading.** Chunking reshuffles which marginal items come out right --
4-9% of them -- without changing how many. Every comparison but one has flips
balanced to within a single item.

The exception is worth recording because it looked real and was not. The first
200 GSM8K problems showed -5.00 points at p=0.013, directional at 12 lost
against 2 gained. Three things retired it. A dose-response series found nothing
at 2, 10 and 20 merges, so the loss appeared only at the extreme; the numerical
error is *flat in slab count* (1-2 bf16 quanta from 1 slab to 192 -- the fp32
accumulator and the log-sum-exp weighting make the per-partial bf16 roundings
average rather than accumulate), so there is no mechanism for a merge-count
threshold; and on 200 problems it had never been measured on, the effect was
exactly zero with 6 flips each way. The hypothesis was generated on the first
sample, so the replication is the test.

**What this does not say.** These bound the cost, they do not show it is zero:
+-1.9 points on NIAH and +-2.5 on GSM8K at these sample sizes. Tightening that
is more items, not a different task. And both tasks detect the perturbation
easily at item level against a zero noise floor -- what they cannot resolve is
an accuracy effect smaller than a couple of points.

**Navigation trap, since it cost a wrong patch here.** The live model runner is
`vllm/v1/worker/gpu/model_runner.py` and its helpers under `vllm/v1/worker/gpu/`; the
older `vllm/v1/worker/gpu_model_runner.py` is a still-selectable fallback
(`VLLM_USE_V2_MODEL_RUNNER`, ROCm architectures, no Triton) that reads as the obvious
file and executes in none of our runs. They log as `model_runner.py:` and
`gpu_model_runner.py:` respectively, which is the cheapest way to tell from a log which
one ran. Both carry the fix.

### What it costs, in the units the appliance sells

| config | B/token KV | B/token workspace | effective |
|---|---|---|---|
| TQ 4bit_nc, before | 18,824 | 4,096 | 22,920 |
| **TQ 4bit_nc, now** | **18,824** | **0** | **18,824** |
| fp8 (Triton) | 35,003 | 0 | 35,003 |
| fp8 (FlashInfer) | 35,461 | 0 | 35,461 |

TurboQuant's gross saving over fp8 is 16,179 B/token. The reserve used to spend 4,096 of
it back — **a quarter of the value proposition, going into a buffer nobody was
accounting for**, before any runtime transient was counted. Since
[`740dd8b6a`](../patches.md) the reserve is a 96 MiB constant and the per-token column
is zero, so the whole 16,179 survives to the operator. The constant is still not in the
budget; that is `kv-budget-margin`'s declaration hook, and a constant is the easy case
for it.

## What is accountable and what is not

The two consumers fail for different reasons, and only one of them is a measurement
problem:

- **Context-scaled startup statics** (TQ 1.000 GiB, FlashInfer 0.385 GiB) are
  declarable. No profiling window can catch an allocation that happens after it, so the
  fix is a declaration, not a better measurement.
- **Context-scaled runtime transients** (TQ's ~6144 B/token of cached context) are not
  accountable at any acceptable price. Reserving headroom for them means setting aside
  8 KB/token × 262144 ≈ 2 GiB against a prefill that may never run — worse than the
  disease on a 16 GiB card. This one has to be engineered away rather than budgeted for.

**It has been**, for TurboQuant, in [`740dd8b6a`](../patches.md): the cached prefix is
attended in slabs and merged by log-sum-exp, so neither the reserve nor the transient
is sized by context any more. Both bullets above therefore now describe TurboQuant only
as history — what is left of it is a 96 MiB constant, which the first bullet's
declaration hook handles trivially. The shape of the argument stands for any backend
that grows one: a context-scaled transient cannot be budgeted for, so it has to stop
existing.

### The search this needs already exists

The natural objection to a declaration hook is that the backend would have to learn how
its own footprint scales with context, or that the sizer would have to profile at
several candidate lengths and hope the result interpolates. Neither is required.

`_reserve_workspace` **already computes the scaling function**. It is
`2 × round_up(max_model_len − 1, block_size) × Hk × D × 2` — linear in context with a
rounding step, written down, in the backend, today. Exposing it as a queryable
`get_workspace_bytes(vllm_config)` gives the sizer the coefficient; the backend gains no
knowledge, it only stops spending it silently. Note that `_reserve_workspace` makes *two*
`get_simultaneous` calls and only the second is context-scaled — the first sizes decode
split buffers from `max_num_seqs`/`num_heads`/`max_num_splits`. So the hook returns a
fixed part and a per-token part, not a single number.

And the consumer for that hook is already built.
[`estimate_max_model_len`](../deps/vllm/vllm/v1/core/kv_cache_utils.py#L901) and
[`_estimate_max_model_len_from_groups`](../deps/vllm/vllm/v1/core/kv_cache_utils.py#L2087)
binary-search over candidate context lengths right now — the auto-fit line in every
startup log is their output. They search by temporarily setting `max_model_len` and
*asking for a size* (`max_memory_usage_bytes`), not by running anything: roughly twenty
iterations of arithmetic, no forward passes, no interpolation, no linearity assumption.
They simply only ask the **KV cache spec**, never the attention backend.

Summing a backend hook into `_max_memory_usage_bytes_from_groups` puts backends inside a
search that already runs on every startup, and because the search evaluates the real
expression at each candidate, the `round_up` step comes along for free — which is
exactly what interpolating between profiled points would have got wrong. Lifting the
150 MiB `redundancy_buffer_memory` into the utilization path as well would make
`gpu_memory_utilization` mean what it says on all three backends.

The closed form is available too, if a search is unwanted: with both terms linear,
`L = (budget − fixed) / (b_workspace + b_kv)`, which for TQ is `budget / 22,920`.

## The payoff, and the term that was hiding behind it (2026-09-12)

`turboderp/Qwen3.8-27B-exl3@3.00bpw` with `turboquant_4bit_nc` now serves **the model's
full declared 262144 context on the 16 GiB card** — the cache holds 264,993 tokens and
auto-fit prints *"full model context length 262144 fits in available GPU memory"* instead
of a reduction, for the first time on this model. That is the transient work's dividend in
the units the appliance sells, and it needed `gpu_memory_utilization=0.975`: a number the
operator has to arrive at by hand, which is `kv-budget-margin`'s entire case.

**And it serves, not just declares**: a 257,549-token prompt — 98.2% of the declared
context — completes on this configuration at 665.65 t/s of prefill, 387 s to first token.
That is the check this ground exists to fail, since a budget that profiles cleanly and
then dies at first inference is exactly what `kv-budget-margin` is about; it passes at
capacity, with the caveat in the method notes that no *profiled peak* can be taken at that
length.

The full command is in [turboquant-kv.md](turboquant-kv.md) "The configuration that
superseded it"; it is the run script's verified line with the utilization raised, plus
`--max-num-batched-tokens 512` and `EXL3_BLOCKQ_ON_LOAD=1`, which loads weights at
10.24 GiB.

Reproducing it from that script turned up a consumer nobody had priced: **a cold
torch.compile cache costs 0.59 GiB of budget, which is 35K tokens of declared context.**

| run | compile cache | engine | peak activation | KV | auto-fit | KV tokens |
|---|---|---|---|---|---|---|
| A | cold (compile+save) | child | 0.80 GiB | 3.88 GiB | 262144 → 224256 | 227,094 |
| B | cold (compile+save) | child | 0.77 | 3.91 | 262144 → 227328 | 230,169 |
| C | disabled | in-process | **0.18** | **4.48** | **full 262144 fits** | **264,993** |
| D | disabled | child | 0.18 | 4.48 | full fits | 264,993 |
| E | **warm** (`Directly load`) | child | 0.18 | 4.48 | full fits | 264,993 |

All rows `--gpu-memory-utilization 0.975`; B–E at `--max-num-batched-tokens 512`, A at the
default 2048. C reproduces the operator's run byte-for-byte. In every row `consumed memory
(weights + non-torch)` is 10.45–10.46 GiB and CUDAGraph memory is 0.04 GiB, so **nothing
but the activation term moves.**

Three findings, largest first:

- **A cold-cache launch declares 35K less context than a warm one, and keeps it for the
  life of the process.** E is the control that makes this attributable: the *loaded*
  artifact costs nothing, so the 0.59 GiB is a high-water mark left by compiling and
  saving. It is charged because `profile_run()` runs *inside* the profiling window
  ([gpu_worker.py:553](../deps/vllm/vllm/v1/worker/gpu_worker.py#L553)), that window
  resets peak stats on entry
  ([mem_utils.py:291](../deps/vllm/vllm/utils/mem_utils.py#L291)), and the activation term
  is `torch_peak - torch_allocated` — so inductor's compile-time benchmarking (this config
  runs with `benchmark_combo_kernel: True`) sets the mark that sizes the KV cache. **Same
  failure mode as the TQ reserve's discarded builders**: machinery that is not inference
  setting the high-water mark that auto-fit then reads. The operator-visible version is
  that the first launch after any version or flag change silently serves less context than
  every launch after it — and that `VLLM_DISABLE_COMPILE_CACHE=1`, which is set in this
  box's `~/.bashrc` and looks like a debugging leftover, is load-bearing for the
  milestone above.

  **Downstream already pays for this, and the dev box is the reason we did not know.** The
  appliance project hit it independently and works around it with warmup runs; its managed
  environment does not set `VLLM_DISABLE_COMPILE_CACHE`, so it meets the cold path the way
  an operator does, while every measurement taken here has been on the cheap path. So the
  finding is not "nobody has seen this" — it is that the workaround lives downstream and
  the accounting lives here, and neither side could see the other's half. Two consequences:
  a warmup run is the correct mitigation today for anything that must declare context
  reliably, and the dev environment's exemption should be treated as a blind spot in this
  note's method, not a convenience.
- **`VLLM_ENABLE_V1_MULTIPROCESSING` does not affect the budget at all** — C against D is
  byte-identical, in-process against child `EngineCore`.
- **`--max-num-batched-tokens` barely affects it either**: A against B is +3,075 tokens
  for a 4x smaller chunk, because logits are profiled one row per request, not per batched
  token. The older `chunk x vocab` claim is corrected in
  [turboquant-kv.md](turboquant-kv.md).

For `kv-budget-margin` this is a term no backend declaration can cover, because no backend
owns it: the candidate fix is to stop the profiler reading a mark set by the compiler —
reset peak stats after compilation and before the measured forward, or compile outside the
window. It is upstream's to take, and unlike the workspace hooks it is a bug rather than a
missing feature.

## The ceiling is predictable, which is what makes the item closable (2026-09-12)

The operator's benchmark is `gpu_memory_utilization = free/total` at startup — 15.28/15.51
= **0.9852** on this card — because vLLM sizes the budget as `total x util`, so that ratio
is the largest request that does not promise memory the card never had. Every backend's
ceiling then follows from one term:

    util_max(backend) = (free_at_startup - unseen_post_window) / total

where `unseen_post_window` is what that backend allocates after the profiling window
closes. Checked against all three backends on this model:

| backend | unseen | predicted | observed |
|---|---|---|---|
| Triton fp8 | none | 0.9852 | **starts at 0.985**: KV 4.64 GiB, 141,020 tokens |
| TurboQuant tq4 | 96 MiB reserve | 0.9791 | 0.975 serves, 0.98 OOMs at startup |
| FlashInfer fp8 | 0.385 GiB workspace | 0.9568 | 0.96 serves, **0.975 OOMs** |

The FlashInfer failure at 0.975 is the sharpest of the three because it names its own
mechanism: `allocate_kv_cache` asks for 4.07 GiB with 4.03 free
([utils.py:411](../deps/vllm/vllm/v1/worker/utils.py#L411)), and the process holds 11.47
GiB before KV against the 11.03 the budget assumed — a 0.44 GiB under-measurement, its
0.385 workspace plus slack. **The budget was not wrong about arithmetic; it was wrong
about the backend.**

**The model's own residual is ≤0.05 GiB, which is the number that decides the item.** That
is 0.3 utilization points, so the answer is not "leave a few hundred MiB of slack": a
per-backend declaration plus a margin in the tens of MiB reaches the benchmark on all
three backends. It also demotes the cheapest candidate in `kv-budget-margin` — the 150 MiB
`redundancy_buffer_memory` over-reserves for TurboQuant by 54 MiB, under-reserves for
FlashInfer by 235, and costs Triton 150 MiB it does not need. It is a fallback for
backends that decline to declare, not the fix.

Two classes stay outside any workspace hook, and they bound how literally "no OOM in any
configuration" can be read:

- **Non-torch growth after the snapshot.** The gap between `free_at_startup` and what the
  allocator later reports as capacity is context, not tensors — JIT kernel loading is the
  suspect, and FlashInfer loads the most. A hook returning tensor bytes cannot see it;
  pre-warming kernels before the snapshot could, and a small margin covers it meanwhile.
- **Request-shaped allocations.** The profiled window is 512 tokens wide, so a logprobs
  request at capacity allocates against headroom nobody reserved. That is admission
  control, not a budget term, and it is the one part of the ideal outcome the profiler
  cannot deliver.

## Declaring the reserve: what it fixed, and the term it uncovered (2026-09-12)

[`6ac849972`](../patches.md) gives `AttentionBackend` a
`get_reserved_workspace_bytes(vllm_config, kv_cache_spec)`, default zero, and
subtracts it from the KV budget before auto-fit runs. TurboQuant implements it by
pricing the *same* reservation sets its builder hands to the workspace manager — one
expression, two callers — because a declaration that drifts from the allocation is worse
than none: the budget would be confidently wrong rather than merely blind.

It declares **96.00 MiB**, and `VLLM_DEBUG_WORKSPACE` shows the builder taking exactly
that (`0 -> 0.76 -> 96.00 MB`, the max over the two sets, not their sum).

| util | before | after |
|---|---|---|
| 0.975 | 4.48 GiB KV, 264,993 tok, serves | 4.39 GiB KV, context clamped just under full |
| **0.98** | **OOM at startup** | **4.46 GiB KV, full 262144, 264,993 tok, 80,793-token prompt served in 79.7 s** |
| 0.985 | OOM in startup warmup, 23 MiB free | starts, full context, 267,842 tok — **first inference OOMs** |

**The safe ceiling moved 0.975 → 0.98 and the declared context did not change**: 264,993
tokens either way, so the 96 MiB comes out of headroom nobody was using rather than out of
context. The knob is now honest one step further up.

**It is not a double count**, which was the first thing to rule out since
[`881c7345b`](../patches.md) has the profiling builders allocate and then shrink back.
Control: peak activation is 0.18 GiB whether the reserve is 96 MiB or the monolithic
path's 1032 MiB (`tq_prefill_workspace_mib=0`), so the builders' mark never reaches the
measured peak — and in that configuration the declaration follows the reserve to 1032 MiB
and auto-fit cuts context to 199,680, which is the hook demonstrating it tracks the
allocation rather than a constant.

**What the benchmark now waits on is ours, not vLLM's.** At 0.985 the engine starts, sizes
the full context, and then dies in the *plugin's* reconstruct path —
`torch.empty((k, tile_n), half)` at [ops.py:533](../vllm_exl3_plugin/ops.py#L533), 30 MiB
wanted against 39 MiB free. That is a per-call runtime transient in a kernel wrapper, not
a startup static, so no `get_reserved_workspace_bytes` can declare it; the ways out are
pooling the buffer (it is the same shape every call for a given layer), reusing the
workspace manager the attention backends already share, or reconstructing into a
pre-allocated view. Until then **0.98 is the ceiling for this configuration and the
remaining 0.5 points of utilization are a plugin question**.

## Open

- **Whether FlashInfer's two workspaces are both necessary**, or whether the
  `init_attn_backend` one is dead once the trtllm path allocates its own.
- **Whether the 0.08–0.12 GiB residual is constant across models**, or scales with
  something. It is stable across three backends and three utilizations on one model.
- **What exactly the cold-compile 0.59 GiB is.** Inductor's autotuning/combo-kernel
  benchmarking is the suspect on the strength of the config, not of a measurement, and the
  figure is from one model at one chunk size. Worth knowing before offering upstream a
  reset, because "reset peak stats after compile" is only correct if nothing the compiler
  allocates is still live when the measured forward runs.
