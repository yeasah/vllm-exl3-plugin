"""Serving exllamav3's trellis-quantized token embedding (`convert.py --embed_bits`).

A checkpoint stores `<key>.trellis` and `<key>.signs`. Each row is zero-padded to
a multiple of 256 and split into 256-wide groups; each group is a tail-biting
trellis ring over the mul1 codebook with an fp16 scale, packed as `1 + 16 * K`
little-endian words:

    group = signs[g] * hadamard(codebook[states] * scale)

The Hadamard and the random signs make the elements the trellis sees near
Gaussian whatever the table's own statistics. That is built for row accuracy, so
this format has no GEMM path and cannot serve a tied head.

Plain torch for the reason `blockq.py` is: the kernel sequence does not depend on
token values, so a CUDA graph can hold it, and inductor fuses it into the
surrounding graph. Transcribed from exllamav3's `ngram_codec.py` rather than
imported, so serving does not reach into the conversion side; the result is
bit-exact against its `ngram_dequant` kernel at every depth
(tests/test_trellis_embed.py).
"""

from __future__ import annotations

import torch

from . import format

#: Width of one ring: a group of a row.
GROUP_DIM = 256
_MUL1 = 0x83DCD12D
#: fp16 bit patterns of the codebook's inverse gain and bias.
_K_INV, _K_BIAS = 0x1EEE, 0xC931
_HADAMARD_STAGES = 8


def depth_from_shape(row_words: int, groups: int) -> int:
    """K, from a table's words per row and its group count."""
    words = row_words // groups if groups else 0
    k = (words - 1) // 16
    if not groups or row_words % groups or words != 1 + 16 * k or not 1 <= k <= 8:
        raise format.EXL3FormatError(
            f"trellis embedding: {row_words} words per row over {groups} groups "
            "is not a whole number of rings at K=1..8"
        )
    return k


def codebook(device: torch.device | str | None = None) -> torch.Tensor:
    """All 65536 decoded mul1 values as fp16: the kernel's table, built once."""
    s = torch.arange(65536, dtype=torch.int64, device=device)
    prod = (s * _MUL1) & 0xFFFFFFFF
    total = (
        (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255)
    )
    bits = torch.tensor([_K_INV, _K_BIAS], dtype=torch.uint16)
    k_inv, k_bias = bits.view(torch.float16).float().tolist()
    return ((1024 + total).float() * k_inv + k_bias).to(torch.float16)


def _unpack(rings: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`(N, words)` int16 rings to `(N, 256)` 16-bit states and `(N,)` fp16 scales."""
    device = rings.device
    words = rings.shape[1] - 1
    scales = rings[:, 0].contiguous().view(torch.float16)
    stream = rings[:, 1:].contiguous().view(torch.uint16).to(torch.int64)
    # Each position's symbol, read through a two-word window
    bit = torch.arange(GROUP_DIM, device=device) * k
    window = stream[:, bit >> 4] | (stream[:, ((bit >> 4) + 1) % words] << 16)
    symbols = (window >> (bit & 15)) & ((1 << k) - 1)
    # A state stacks its own symbol and the preceding ones, low bits first
    states = torch.zeros_like(symbols)
    for j in range((15 + k) // k):
        states = states | (torch.roll(symbols, j, dims=1) << (j * k))
    return states & 0xFFFF, scales


def _hadamard(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Walsh-Hadamard over the last dimension, in the kernel's order."""
    n = x.shape[0]
    for stage in range(_HADAMARD_STAGES):
        h = 1 << stage
        v = x.view(n, -1, 2, h)
        x = torch.stack((v[:, :, 0] + v[:, :, 1], v[:, :, 0] - v[:, :, 1]), dim=2)
        x = x.view(n, GROUP_DIM)
    return x / (GROUP_DIM**0.5)


def decode(
    packed: torch.Tensor,
    signs: torch.Tensor,
    book: torch.Tensor,
    hidden: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """`(N, groups * words)` packed rows to `(N, hidden)`."""
    n = packed.shape[0]
    groups = signs.shape[0]
    k = depth_from_shape(packed.shape[1], groups)
    states, scales = _unpack(packed.view(n * groups, -1), k)
    values = book[states].float() * scales.float().unsqueeze(1)
    values = _hadamard(values) * signs.float().repeat(n, 1)
    # Through fp16, as the kernel rounds: a target dtype of bf16 must not see fp32
    return values.half().view(n, -1)[:, :hidden].to(out_dtype)


def gather(
    ids: torch.Tensor,
    table: torch.Tensor,
    signs: torch.Tensor,
    book: torch.Tensor,
    hidden: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Embedding lookup: `ids`' rows out of storage, decoded, shape-preserving."""
    flat = ids.reshape(-1)
    rows = decode(torch.index_select(table, 0, flat), signs, book, hidden, out_dtype)
    return rows.view(*ids.shape, hidden)
