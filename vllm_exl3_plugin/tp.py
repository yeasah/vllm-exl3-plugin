"""Tensor-parallel slicing of EXL3 tensors.

This is a transcription of exllamav3's own `LinearEXL3.tp_import_split`, which
is the authority on how an EXL3 weight may be cut:

    output (column) split      input (row) split
    -----------------------    -----------------------
    trellis  dim 1, first//16  trellis  dim 0, first//16
    svh      dim 0, first      suh      dim 0, first
    suh      replicated        svh      replicated
    bias     dim 0, first      bias     rank 0 only
    mcg/mul1 replicated        mcg/mul1 replicated

Two different granularities are in play and it is important not to confuse
them. *Storage* is tile-granular: the trellis is indexed in 16x16 tiles, so any
multiple of 16 can be sliced out of it. *Correctness* is Hadamard-block
granular: the regularization applies a block-diagonal transform in blocks of
128, so only a split on a multiple of 128 leaves each block wholly inside one
rank. A split at, say, 64 slices cleanly and produces wrong numbers silently.
`format.shard_bounds` enforces the 128 rule; `tests/test_tp.py` demonstrates
that violating it really does corrupt the result rather than merely being
disallowed on principle.

Why summation reconstructs a row-parallel split exactly: with H the blockwise
Hadamard, a layer computes

    y = H_n( H_k(x * suh) @ W ) * svh

`H_k` is block diagonal, so splitting k on a block boundary gives
`H_k(x*suh) = [H_k(x1*suh1) | H_k(x2*suh2)]`, and the product against the
correspondingly row-split W becomes `h1 @ W1 + h2 @ W2`. Both `H_n` and the
`svh` scaling are linear, so they distribute over that sum -- each rank can
apply them locally and the partial results add. This is why `svh` is
replicated rather than split, and why an all-reduce is the correct combiner.
"""

from __future__ import annotations

import torch

from .format import TILE

#: Roles an EXL3 sub-tensor can play, which is what decides how it slices.
ROLE_TRELLIS = "trellis"
ROLE_SUH = "suh"
ROLE_SVH = "svh"
ROLE_SCALAR = "scalar"  # mcg / mul1, replicated everywhere
#: The block-quantized embedding's tensors (`blockq.py`). All three are indexed by
#: vocabulary on dim 0 and slice identically, which is the whole reason that format
#: was chosen: a vocabulary-parallel split is a row slice with no alignment rule to
#: satisfy, where the trellis needs whole 128-row Hadamard blocks.
ROLE_VOCAB = "vocab"

#: Storage suffixes that carry `ROLE_VOCAB`.
_VOCAB_TENSORS = ("bq_q", "bq_s", "bq_r")


def role_of(name: str) -> str:
    if name in (ROLE_TRELLIS, ROLE_SUH, ROLE_SVH):
        return name
    if name in _VOCAB_TENSORS:
        return ROLE_VOCAB
    return ROLE_SCALAR


def shard_column(role: str, t: torch.Tensor, first: int, last: int) -> torch.Tensor:
    """Take this rank's slice for an output-dimension (column-parallel) split."""
    if role == ROLE_VOCAB:
        return t[first:last].contiguous()
    if role == ROLE_TRELLIS:
        return t[:, first // TILE : last // TILE, :].contiguous()
    if role == ROLE_SVH:
        return t[first:last].contiguous()
    # suh indexes input channels, which a column split leaves whole.
    return t


def shard_row(role: str, t: torch.Tensor, first: int, last: int) -> torch.Tensor:
    """Take this rank's slice for an input-dimension (row-parallel) split."""
    if role == ROLE_VOCAB:
        return t[first:last].contiguous()
    if role == ROLE_TRELLIS:
        return t[first // TILE : last // TILE].contiguous()
    if role == ROLE_SUH:
        return t[first:last].contiguous()
    # svh indexes output channels, which a row split leaves whole -- and it
    # must stay whole for the partial sums to add correctly (see module docs).
    return t
