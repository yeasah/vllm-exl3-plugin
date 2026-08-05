# Phase 2 — tensor parallelism (partial: proven, not validated)

Goal from the feasibility report: **resolve the Hadamard-block-128 sharding
question, then add tensor parallelism.**

**Status: the sharding is implemented and its arithmetic is proven. It has
never run on more than one GPU, because this machine has only one.** Read the
"Not validated" section before trusting TP>1 output.

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

## What is NOT validated

This machine has one GPU; vLLM maps TP ranks to distinct devices, so TP>1
cannot start here at all. Untested, in rough order of how likely they are to be
wrong:

1. **vLLM's loader driving these slices at TP>1.** The tests prove *"given
   correctly sliced tensors, the math is right"*. They cannot prove *"we produce
   correctly sliced tensors from the loader's calls"*. `EXL3Parameter` now
   follows vLLM's convention — the whole checkpoint tensor arrives and the
   parameter narrows at `tp_rank * shard_size` — including the
   `tp_rank // num_heads` adjustment for replicated KV heads, but none of that
   has executed.
2. **The autotune cache under multiple worker processes.** exllamav3 keeps one
   on-disk cache (`~/.cache/exllamav3/autotune/`, overridable with
   `EXLLAMAV3_TUNE_CACHE`) and benchmarks kernel shapes on first use. Several
   TP workers racing to write it is an open question. Anything that packages
   this (Phase 4) should keep that variable per-worker overridable rather than
   assuming a single shared file.
3. **Collectives.** vLLM owns the all-reduce, so there is nothing of ours in
   that path, but the assumption is untested.
4. **Quantized `lm_head` at TP>1**, which still raises `NotImplementedError`.
   Vocab-dimension sharding has to compose a 128-boundary split with vLLM's own
   vocab padding and the trim in `apply()`, and blind-coding that seemed worse
   than declining it. Tied-embedding models are unaffected.

Because of (1), `create_weights` emits a one-time warning at TP>1 saying the
path is unvalidated. The alignment check is a hard error — that one is real
regardless of hardware.

## Finishing this

On a multi-GPU machine, in order: run any working model at TP=2 and diff the
output against TP=1 (they should agree to fp16 noise); check the autotune cache
for corruption or races across workers; then TP=4; then decide whether the
`lm_head` case is worth doing. The arithmetic should not need revisiting — it is
pinned by tests that run anywhere.
