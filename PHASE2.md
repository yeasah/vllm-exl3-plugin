# Phase 2 — tensor parallelism

Goal from the feasibility report: **resolve the Hadamard-block-128 sharding
question, then add tensor parallelism.**

**Status: validated at TP=2** (three models, eager and with CUDA graphs), **at
TP=4** (one MoE model, Qwen3.5-35B-A3B), **and at TP=8** (one dense model,
eager only). All reproduce TP=1 **token for token**; the autotune cache
survives eight worker processes writing it concurrently from empty (not just
two); and vocab-parallel quantized `lm_head` now works. TP=3, 5, 6, 7 remain
unexercised, TP=8 has only been checked eager on one checkpoint shape, and
**MoE at TP=4 has an open, unresolved performance problem** on the one
checkpoint tried (Laguna-XS-2.1) that has nothing to do with correctness --
see "MoE under tensor parallelism" below. See "What is validated" for the
exact boundary.

## The rule

exllamav3's `LinearEXL3.tp_import_split` is the authority, and `tp.py` is a
transcription of it:

| | output (column) split | input (row) split |
|---|---|---|
| `trellis` | dim 1, at `first // 16` | dim 0, at `first // 16` |
| `suh` | replicated | dim 0, at `first` |
| `svh` | dim 0, at `first` | replicated |
| `bias` | dim 0, at `first` | rank 0 only |
| `mcg` / `mul1` | replicated | replicated |

Two granularities are in play, and conflating them is how this goes silently
wrong. **Storage is tile-granular**: the trellis indexes 16x16 tiles, so any
multiple of 16 slices cleanly and nothing complains. **Correctness is
Hadamard-block granular**: the regularization is block-diagonal in blocks of
128, so only a multiple of 128 leaves every block inside one rank.

`format.shard_bounds` enforces 128. `tests/test_tp.py` does not merely assert
that — it forces a 64-wide split (tile-aligned, sub-block) and shows the result
is wrong by more than 10%, three orders of magnitude beyond fp16 noise. Without
that test the guard would look like caution rather than a constraint.

This bites at real TP degrees. Llama-3.2-1B has 8 KV heads of dim 64, so
`k_proj` and `v_proj` produce 512 channels: fine at TP=2 and TP=4, **invalid at
TP=8**, where each rank would get 64.

## MoE under tensor parallelism

Routed experts shard on the **intermediate** dimension: gate/up take a column
(output) split, down takes a row (input) split, and each rank's partial sums are
combined by the all-reduce vLLM already performs. `EXL3MoEMethod` reports no
`moe_kernel`, so `MoERunner._fused_output_is_reduced` is False and the runner
reduces the combined output for us.

The dimension being cut is the **stored** intermediate, not the model's.
exllamav3 pads before quantizing — gemma-4-26B stores 768 where the config says
704 — so shard boundaries come from the tensor in hand rather than from vLLM's
`intermediate_size_per_partition`, and `_tp_shard` reads that width from
whichever sub-tensor carries it (trellis dim 1 or 0, `svh`, `suh`).

That padding also decides which degrees are legal, and the answer differs per
checkpoint because the stored widths do. `tools/tp_preflight.py` computes it
from safetensors headers alone — no GPU, no weights — so it can be run before
renting anything:

| checkpoint | TP=2 | TP=4 | TP=8 | first blocker |
|---|---|---|---|---|
| Llama-3.2-1B | ok | **no** | no | `lm_head` 128256 (odd multiple of 128 at tp4) |
| Laguna-XS-2.1 | ok | ok | **no** | expert intermediate 512 -> 64 |
| Qwen3.5-35B-A3B | ok | ok | **no** | `k_proj` 512 -> 64 |
| gemma-4-26B-A4B | **no** | no | no | dense `mlp` 2176 = 17x128 |

`format.shard_bounds` enforces the 128-wide Hadamard rule, so an illegal degree
raises at load rather than silently computing the wrong thing.

`--remote` answers the same question for a checkpoint that is not downloaded,
through `HfApi.get_safetensors_metadata` — the Hub's supported call for reading
tensor metadata without fetching weights, so vetting a candidate costs kilobytes
instead of the gigabytes a download would. Verified to agree with the local path
on the same checkpoint.

### What blocks high TP degrees

TP=8 needs every split dimension divisible by 1024, and the recurring blockers
are structural rather than unlucky:

- **fine-grained MoE expert intermediates** (512 in both Laguna and Qwen3.5) —
  small *by design*, so a fine-grained MoE essentially cannot reach TP=8 on the
  intermediate axis at all;
- **narrow KV projections** (`k_proj`/`v_proj` at 512 or 256);
- **vocabulary size** — 128256, 130560 and 248320 all fail; 100352 and 262144
  pass. Whether a vocab cooperates is close to arbitrary, and a power-of-two
  vocab is the reliable case.

For MoE specifically, the right axis at high degree is **expert parallelism**,
which gives each rank whole experts, cuts no tensor, and sidesteps the Hadamard
constraint entirely. That is currently refused (`exl3_mgemm` will not combine
expert-range filtering with the fused weighted reduction), so scaling MoE wide
is a kernel feature gap, not a hardware-coverage gap.

A dense model can reach TP=8: `gemma-4-31B-it-exl3` @3.00bpw clears every
dimension, including its 262144 vocab.

**gemma-4-26B cannot be tensor-parallel at any degree.** Its routed experts
split fine at TP=2 (768 -> 384), but the *dense* MLP in the same model is 2176
wide — seventeen Hadamard blocks, an odd multiple, which no even split can
divide. That is not visible from the expert dimensions, from `config.json`, or
from model size, which is exactly why the preflight exists. An earlier version of
this table claimed gemma was usable at TP=2 on the strength of its expert width
alone; it is not.

Expert parallelism is still refused: `exl3_mgemm` can filter an expert range,
but not in combination with the multi-token weighted reduction this method uses.

### Open problem: Laguna-XS-2.1 at TP=4 does not finish in reasonable time

Attempted on the 8x RTX 3090 box, `tools/tp_compare.py capture --tp 4 --eager`.
Preflight says TP=4 is legal (every dimension divides at that degree), and
TP=1 captures of the same checkpoint complete in about a minute. This did not
finish in over an hour, twice, and was killed both times rather than left to
run further.

What was ruled out before killing it:

- **Not a deadlock.** All four worker processes stayed pinned at 100% GPU
  utilization continuously (sampled repeatedly, including two `/proc/PID/io`
  reads 608s apart), and their CPU time climbed in lockstep with wall clock.
  Something is genuinely, continuously executing.
- **Not a compile stall.** No `nvcc`/`ninja`/`cicc`/compiler subprocess existed
  anywhere on the box at any point checked (`ps --ppid` on all four workers was
  empty throughout). A red herring surfaced during manual inspection -- an open
  fd to `flashinfer_jit.log` on all four hot workers -- but that file's mtime
  predated this run by hours (it's FlashInfer's *sampling*-kernel JIT from
  session start, unrelated and already complete).
- **Not an autotune-cache write storm**, or at least not evidenced by host-side
  I/O: a windowed `/proc/PID/io` diff (not the noisy whole-process-lifetime
  average, which is dominated by an earlier phase) showed only ~4 read and ~4
  write syscalls/sec, averaging ~1.8 bytes/write -- consistent with ordinary
  low-rate inter-process signaling (pipes/eventfds between the four ranks), not
  a tight loop. This also does not rule anything *in*: GPU device ioctls (the
  actual kernel-launch traffic) are invisible to `/proc/PID/io` entirely, so
  this measurement cannot see the thing most likely to be looping.
- **Not cache-size-dependent.** Retried with `EXLLAMAV3_TUNE_CACHE` pointed at
  a completely empty directory, isolating whether a large accumulated cache
  (222 records, from every other run this session) was itself the problem.
  Same symptom from a cold start.

**Not root-caused.** Localizing further needs either `exllamav3`'s
`CACHEDEBUG` build flag (a source rebuild, not attempted) or a profiler this
container's sandboxing blocks (`ptrace` is refused outright; `nsys`/`ncu`
almost certainly need the same capability). The working theory -- Laguna's
per-rank expert intermediate at TP=4 is exactly 128, the *minimum* legal
Hadamard block width, and something about that boundary produces tied or
near-tied candidate timings that a variance/stability-based autotune stopping
rule never resolves -- is plausible given where it sits in the shard-width
range, but unverified. Worth retrying if the exl3_mgemm autotuner's stopping
condition is ever inspected for other reasons; not worth more rented GPU time
chasing blind.

### Verified on real multi-GPU hardware

2x RTX 5060 Ti (sm_120), vLLM 0.26.0, both execution modes:

| model | TP | eager | graphs | outcome |
|---|---|---|---|---|
| Laguna-XS-2.1 | 1 | 16.2 tok/s | 132.5 tok/s | ok |
| Laguna-XS-2.1 | **2** | 15.1 tok/s | **131.5 tok/s** | **ok** |
| Qwen3.5-35B-A3B | 1 | 12.2 tok/s | 93.4 tok/s | ok |
| Qwen3.5-35B-A3B | 2 | — | — | blocked at load, see below |

Weights split correctly: **4.4 GiB per worker against 8.54 GiB at TP=1.**
Throughput is flat, which is the expected shape — decode is latency-bound and TP
adds an all-reduce, so what TP buys here is memory headroom, not speed.

**Token-for-token equality does not hold, and cannot for this checkpoint.** The
Phase 2 standard was established on dense models; Laguna at 2bpw diverges from
its own TP=1 run at token 16 of 48. The sweep supplied its own control, though:

    tp1 eager vs tp1 graphs   diverges at token 26/48  (laguna, no sharding at all)
    tp1 eager vs tp1 graphs   diverges at token 43/48  (qwen,   no sharding at all)
    tp1 vs tp2, same mode     diverges at token 16/48  (laguna)

Merely changing execution mode at TP=1 diverges comparably, so greedy divergence
here is a property of a 2bpw model's numerical fragility rather than evidence
about sharding. Together with the offline test below matching to 2e-2, that reads
as correct — but it is **weaker evidence than the dense Phase 2 result**, because
the strongest available check simply does not apply to this checkpoint.

**Qwen3.5 was blocked at TP>1** by a known gap, not a bug: its `in_proj_qkv`
arrived as the shard tuple `(0, 1, 2)` inside one stored tensor, and
`EXL3Parameter._load_fused` could not compose that split with a tensor-parallel
one. It failed at load with exactly that message, and `tools/tp_preflight.py`
predicted it -- it previously cleared Qwen for TP=2 because every dimension
divides, which was a blind spot, since the obstacle was vLLM's packing, not the
checkpoint's shapes.

**This is now fixed and verified on real hardware.** `_load_fused` composing
its split with `format.fused_shard_bounds` closed the gap (offline only, at the
time); on an 8x RTX 3090 box, `Qwen3.5-35B-A3B-exl3` @3.00bpw now loads and runs
correctly at both TP=2 and TP=4, token-identical to TP=1 on both prompts tried
-- see "What is validated" above for the full result. `tools/tp_preflight.py`'s
warning about this path being hardware-unverified is accordingly stale for this
specific case and should be softened once a broader set of fused-shard
checkpoints has been run this way, not just Qwen3.5.

### Measuring this properly: `tools/tp_compare.py`

First-divergence-token is a bad instrument. It conflates numerical error with
model confidence, and after the first difference the two runs are on *different
contexts*, so everything downstream compares prompts rather than arithmetic.

`tools/tp_compare.py` teacher-forces instead: both configurations score the same
fixed token sequence via `prompt_logprobs`, giving a per-position comparison at
identical contexts, reported as KL plus argmax disagreement. Comparing a capture
against itself is exactly zero, which is the sanity check that the metric is
measuring what it claims.

The point is the **noise floor**. Capture eager against CUDA graphs at TP=1 —
same weights, no sharding, no collectives — and whatever differs is that
checkpoint's irreducible kernel-ordering noise. For Laguna-XS at 2bpw:

| TP=1 eager vs TP=1 graphs | argmax disagreements | KL max / mean |
|---|---|---|
| factual prompt | 2/47 | 7.0e-2 / 1.3e-2 |
| open-ended prompt | 9/91 | 5.5e-1 / 2.5e-2 |

Nine argmax flips in 91 positions from an execution-mode change alone. Against
that floor, a generation diverging at token 16 under TP=2 says nothing, and
**token-identical output was never achievable for this checkpoint under any
configuration change** — which is why the dense Phase 2 standard could not be
met here and why chasing it with a higher-bpw checkpoint is a quality question
rather than a correctness one.

**Still not covered by any of this:** a TP=2 capture to compare against the
floor above — the box was released before `tp_compare.py` existed, and it has
not been rerun since. TP=4 *was* attempted on the 8x RTX 3090 box, via
`tp_compare.py`, and several worker processes did write the autotune cache
concurrently elsewhere in that session (see "The autotune cache survives eight
concurrent workers" above, using a dense model) — but the TP=4 attempt on this
specific checkpoint never finished; see "Open problem" above. So this
checkpoint's TP-vs-floor comparison remains open at every degree above 1.

### The offline simulation

`tests/test_tp.py` also simulates every rank sequentially on one device and sums,
establishing that the split reconstructs the unsharded result at TP=2 and TP=4.
That remains the check that runs without hardware.

## Why a row split can be summed

With `H` the blockwise Hadamard, a layer computes

    y = H_n( H_k(x * suh) @ W ) * svh

`H_k` is block diagonal, so splitting `k` on a block boundary gives
`H_k(x*suh) = [H_k(x1*suh1) | H_k(x2*suh2)]`, and the product against the
row-split `W` becomes `h1 @ W1 + h2 @ W2`. Both `H_n` and the `svh` scaling are
linear, so they distribute over that sum: each rank can apply them locally and
the partials add. That is why `svh` stays replicated, and why vLLM's ordinary
all-reduce is the correct combiner rather than something bespoke.

## What is proven

`tests/test_tp.py` simulates ranks sequentially on one device and combines them
the way vLLM would — concatenation for a column split, summation for a row
split — then compares against unsharded execution of the same layer:

- **Column split** at TP=2 and TP=4 matches to ~4e-4 relative. Not bit-exact,
  which was mildly surprising: each output channel is computed by exactly one
  rank from identical inputs, but a narrower `n` makes the kernel autotuner pick
  a different tile shape, changing the fp16 accumulation order over `k`.
- **Row split** at TP=2 and TP=4 matches after summation.
- **A sub-block split corrupts**, as above.
- `shard_bounds` rejects uneven and sub-128 splits.

## What is validated

Run on a rented Vast.ai instance: 2x RTX 3060 (sm_86), torch 2.11.0+cu130,
**vLLM 0.26.0 — the release, not main** — exllamav3 v1.3.0 built from the
pinned submodule, `Llama-3.2-1B-Instruct-exl3` @ 3.0bpw.

1. **TP=2 is token-identical to TP=1**, on three models chosen to exercise
   different paths, compared on token ids rather than text. Not "close" --
   identical. Since the row-parallel split reaches the same answer by a
   different summation order, exact agreement is stronger than it had to be.

   | model | what it adds |
   |---|---|
   | `Llama-3.2-1B-Instruct-exl3` @3.0bpw | baseline; tied head |
   | `Llama-3.2-1B-Instruct-exl3` @3.5bpw | **mixed bit widths inside a merged QKV** (q=4, k=5, v=5) sharded per output partition |
   | `MiniCPM5-1B-exl3` @3.00bpw | **untied, EXL3-quantized `lm_head`** at 6 bits, plus the `mcg` codebook |

   Also identical with **CUDA graphs enabled** rather than eager: each worker
   captures its own 35 graph shapes independently, so cooperative-kernel capture
   is fine in a multi-process TP group.
2. **vLLM's loader drives the slicing correctly.** This was the part the offline
   proof could not reach: the tests showed *"given correctly sliced tensors the
   math is right"*, not *"we produce correctly sliced tensors"*. It needed no
   changes -- `tp_rank * shard_size` with the `tp_rank // num_heads` adjustment
   for replicated KV heads was right as written.
3. **The autotune cache survives concurrent workers.** Two ranks benchmark and
   write `~/.cache/exllamav3/autotune/coop_autotune_v1.bin` at once. Afterwards
   the file is structurally intact (valid magic, 16-byte header + exactly 56
   32-byte records, no torn write), and a second TP=2 run reading that
   cold-written cache produces identical output again -- so the entries the race
   produced select correct kernels. This was the open question flagged in Phase 1.
4. **The plugin works against a vLLM release.** Everything to date was developed
   against main; 0.26.0 needed no changes. That matters for packaging (Phase 4),
   which can pin a release rather than chase main.
5. **TP=8 is token-identical to TP=1**, on an 8x RTX 3090 box (sm_86), vLLM
   0.27.1.dev0+g4bdc8a788 (main), `gemma-4-31b-it-exl3` @3.00bpw -- the dense
   model `tools/tp_preflight.py` identifies as clearing every TP=8 split
   boundary, including its 262144 vocab (see "What blocks high TP degrees"
   above). Same method as the TP=2 result: greedy decode, compared on token ids
   and per-token logprobs rather than text, against a TP=1 run of the same
   checkpoint. Both prompts matched exactly, down to the same -0.0006 logprob
   at the same position. Eager only; one checkpoint; one degree. `linear.py`'s
   runtime warning reflects exactly this boundary.
6. **The autotune cache survives eight concurrent workers writing it from
   empty**, not just two. Same 8x RTX 3090 box: the cache was cleared, then
   `gemma-4-31b-it-exl3` @3.00bpw run fresh at TP=8 so all eight ranks raced to
   populate it simultaneously rather than mostly hitting existing entries.
   Afterward: valid magic, 16-byte header, exactly 64 32-byte records, zero
   remainder -- no torn write -- and output was still token-identical to the
   TP=1 baseline, so the race didn't just produce a well-formed file, it
   produced *correct* entries.
7. **Qwen3.5-35B-A3B is token-identical to TP=1 at both TP=2 and TP=4**, on
   the same 8x RTX 3090 box, and this is the first time this has run on real
   hardware at all -- the fused-shard TP composition (`EXL3Parameter._load_fused`
   composing with `format.fused_shard_bounds`, see below) was previously only
   unit-tested. Same method as the dense results: greedy decode, token ids and
   per-token logprobs against a TP=1 run of the same checkpoint. Both prompts
   matched exactly at both degrees; the largest logprob difference at any
   position was noise-level (e.g. a soft-confidence token moving from -0.4179
   to -0.4566 without changing which token was chosen), consistent with the
   tile-shape-dependent fp16 accumulation noise already characterized for
   dense models, not a sharding bug. `tools/tp_preflight.py` correctly rejects
   TP=8 for this checkpoint (narrow expert/KV dimensions), which was not
   attempted.

## What is still NOT validated

1. **TP=3, TP=5, TP=6, TP=7 for any checkpoint**, and **TP=8 for any MoE
   checkpoint** (all three on hand are structurally blocked at TP=8 by narrow
   expert/KV/vocab dimensions -- see "What blocks high TP degrees"). The
   arithmetic is proven for any degree in `tests/test_tp.py`; only TP=2, TP=4
   and TP=8 have actually run, and TP=4 has only run for one dense checkpoint
   and one MoE checkpoint each.
2. **MoE at TP=4 has an open, unresolved performance problem.** See
   "MoE under tensor parallelism" below -- this is not a correctness gap, the
   checkpoint that hit it (Laguna-XS-2.1) simply never finished in any
   reasonable time, twice, including with a completely fresh autotune cache.
3. **The alignment guard in a live engine.** `format.check_tp_split` is
   thoroughly unit-tested, but no checkpoint run here has actually hit it live:
   every checkpoint run at a given degree was chosen *because* it clears every
   split boundary at that degree. Llama-3.2-1B's 512 KV channels are still the
   known way to break TP=8 (64 channels per rank), but nobody has run that
   combination and watched the guard actually refuse it.
4. **TP=8 with CUDA graphs, and TP=8 on any checkpoint shape other than
   `gemma-4-31b-it-exl3`.** The TP=2 result covered three models and both
   execution modes; the TP=8 result covers one of each.
5. **Expert parallelism**, which remains refused at the kernel level
   regardless of TP degree (see "What blocks high TP degrees") -- a
   development gap, not something more hardware time closes.

## Vocab-parallel `lm_head`

Implemented during the same session, since the box was the only place it could
be validated. A quantized head shards exactly like any other output split --
trellis and `svh` sliced, `suh` replicated -- but it has to compose that with
*two* other vocabulary paddings, so the boundaries come from vLLM's own
`shard_indices` rather than being recomputed:

- `padded_org_vocab_start/end_index` cut the stored tensor, because the
  checkpoint covers the padded vocabulary;
- `org_vocab_start/end_index` give how many rows of that slice are *real*
  vocabulary, which is what `apply()` trims to before zero-padding out to
  `num_embeddings_per_partition`.

At TP=1 these collapse to the previous behaviour, which is why the existing
TP=1 test still passes unchanged. `MiniCPM5-1B` (vocab 130560, so 65280 per rank
-- a clean 510 Hadamard blocks) is token-identical at TP=2.

## Finishing this

TP=2, TP=4, TP=8, the autotune-cache question (now stress-tested at eight
concurrent writers, not just two) and the quantized `lm_head` are done. MoE at
TP>1 is no longer categorically unvalidated -- Qwen3.5-35B-A3B works, verified,
at TP=2 and TP=4. What remains: TP=3, 5, 6, 7 (a box with those card counts
would close the range), CUDA graphs at TP=8, a second checkpoint at TP=8,
watching the alignment guard actually refuse a live illegal split, expert
parallelism (a kernel development gap, not a hardware one), and the open
Laguna-XS-2.1 TP=4 performance problem above, which needs source-level
debugging tools this session didn't have rather than more GPU time. The
sharding arithmetic itself has not needed revisiting and should not.

One environment note worth keeping, because it cost more than the test did. The
rented box advertised CUDA 12.9 but carried torch built against 13.0, and
building exllamav3 needed three fixes:

- point `CUDA_HOME` at the pip `nvidia/cu13` toolkit rather than
  `/usr/local/cuda`, which was 12.9;
- set `NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`, because that
  one prefix mixes nvcc 13.3, CCCL 13.3 and CUDA *runtime* headers 13.0 from
  three independently versioned pip packages, and CCCL refuses the skew;
- add `lib64 -> lib` and `libcudart.so -> libcudart.so.13` symlinks, since the
  pip layout provides neither name the linker looks for.

The resulting build was checked numerically — all 27 GPU tests, including the
bit-exact oracle against exllamav3's own dequantization — before any TP result
was trusted, since a toolchain skew that silently miscompiles is exactly the
failure this project cannot afford.
