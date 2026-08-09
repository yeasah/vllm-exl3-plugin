# Phase 2 — tensor parallelism

Goal from the feasibility report: **resolve the Hadamard-block-128 sharding
question, then add tensor parallelism.**

**Status: validated at TP=2**, across three models, eager and with CUDA graphs.
TP=2 reproduces TP=1 **token for token** every time; the autotune cache survives
two worker processes writing it concurrently; and vocab-parallel quantized
`lm_head` now works. TP>2 remains unexercised for want of hardware. See "What is
validated" for the exact boundary.

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

**gemma-4-26B cannot be tensor-parallel at any degree.** Its routed experts
split fine at TP=2 (768 -> 384), but the *dense* MLP in the same model is 2176
wide — seventeen Hadamard blocks, an odd multiple, which no even split can
divide. That is not visible from the expert dimensions, from `config.json`, or
from model size, which is exactly why the preflight exists. An earlier version of
this table claimed gemma was usable at TP=2 on the strength of its expert width
alone; it is not.

Expert parallelism is still refused: `exl3_mgemm` can filter an expert range,
but not in combination with the multi-token weighted reduction this method uses.

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

**Qwen3.5 is blocked at TP>1** by a known gap, not a bug: its `in_proj_qkv`
arrives as the shard tuple `(0, 1, 2)` inside one stored tensor, and
`EXL3Parameter._load_fused` cannot compose that split with a tensor-parallel one.
It fails at load with exactly that message. `tools/tp_preflight.py` now predicts
it; it previously cleared Qwen for TP=2 because every dimension divides, which
was a blind spot — the obstacle is vLLM's packing, not the checkpoint's shapes.

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

**Still not covered by any of this:** TP>2 on real hardware, several worker
processes writing the exllamav3 autotune cache at once, and a TP=2 capture to
compare against the floor above — the box was released before `tp_compare.py`
existed.

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

## What is still NOT validated

1. **TP=4 and above.** Two GPUs is what was rented. The arithmetic is proven for
   any degree in `tests/test_tp.py`, but only TP=2 has run. `create_weights`
   warns once above TP=2 rather than at every degree.
2. **MoE at TP>1**, which raises (Phase 3 is itself unfinished).
3. **The alignment guard in a live engine.** `format.check_tp_split` is
   thoroughly unit-tested, but no TP degree available here misaligns
   Llama-3.2-1B: 512 KV channels need TP=8 to break, and the box has two cards.

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

TP=2, the autotune-cache question and the quantized `lm_head` are done. What
remains: TP=4 on a four-card box, and MoE once Phase 3 works. The sharding
arithmetic has not needed revisiting and should not.

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
