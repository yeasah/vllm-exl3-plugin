# Phase 2 — tensor parallelism

Goal from the feasibility report: **resolve the Hadamard-block-128 sharding
question, then add tensor parallelism.**

**Status: validated at TP=2.** On a rented 2x RTX 3060 box, TP=2 reproduces
TP=1 **token for token** on every prompt tried, and the autotune cache survives
two worker processes writing it concurrently. TP>2 remains unexercised for want
of hardware, and a quantized `lm_head` at TP>1 still raises. See "What is
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

1. **TP=2 is token-identical to TP=1.** Three prompts, greedy, compared on
   token ids rather than text. Not "close" -- identical. Since the row-parallel
   split reaches the same answer by a different summation order, exact agreement
   is a stronger result than it had to be.
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
2. **Quantized `lm_head` at TP>1**, which still raises `NotImplementedError`.
   The test model ties embeddings, so vLLM skips `lm_head` entirely and this
   path was never even reached. Vocab-dimension sharding has to compose a
   128-boundary split with vLLM's own vocab padding and the trim in `apply()`.
3. **MoE at TP>1**, which also raises (Phase 3 is itself unfinished).
4. **The alignment guard in a live engine.** `format.check_tp_split` is
   thoroughly unit-tested, but no TP degree available here misaligns
   Llama-3.2-1B: 512 KV channels need TP=8 to break, and the box has two cards.

## Finishing this

TP=2 and the autotune-cache question are done. What remains, in order: TP=4 on a
four-card box; a model with untied embeddings to force the quantized `lm_head`
path (and then implement vocab sharding); MoE once Phase 3 works. The arithmetic
has not needed revisiting and should not.

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
