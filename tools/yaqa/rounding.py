"""YAQA rounding for the EXL3 trellis quantizer.

EXL3's `ldlq()` rounds a weight with linear feedback along the *input* channels
only: the loss it descends is `tr(Δᵀ H_I Δ)`, the immediate activation error.
YAQA (arXiv:2505.22988) generalizes this to a Kronecker-factored Hessian
`H_O ⊗ H_I` that approximates the full-model output KL, which adds feedback along
the output channels as well.

In EXL3's layout a weight is `(k, n) = (in_features, out_features)` -- transposed
from the paper's `W ∈ R^{m×n}`. Transposing the paper's Equation 6 gives

    Ŵ = Q(W + L_Iᵀ Δ L_O + L_Iᵀ Δ + Δ L_O),    Δ = W - Ŵ

where `L_I` is `(k, k)` from `H_I` and `L_O` is `(n, n)` from `H_O`, both in the
strictly-block-lower-triangular form `block_ldl()` already returns (EXL3 zeroes
the unit diagonal after the decomposition, so these are the paper's `L - I`
directly). Setting `L_O = 0` recovers EXL3's existing single-sided update, and
`equivalent_to_ldlq()` below asserts that it does so bit-for-bit.

Tile `(i, j)` now depends on every tile below it, every tile to its right, and
every tile down-and-right, so the sweep order becomes anti-diagonal wavefronts
running from the bottom-right corner: `k/16 + n/16` steps instead of `k/16`.
Tiles within a wavefront are mutually independent and are quantized in one
batched `quantize_tiles()` call, which is the same shape of work the existing
row sweep hands the quantizer.
"""

from __future__ import annotations

import torch

from exllamav3.modules.quant.exl3_lib.quantize import (
    quantize_tiles_multigpu,
    tensor_core_perm,
    tensor_core_perm_i,
)


def yaqa_round(
    weight: torch.Tensor,
    L_I: torch.Tensor,
    L_O: torch.Tensor | None,
    quant_args: dict,
    progress: bool = False,
    stats: dict | None = None,
):
    """Round `weight` (k, n) with input feedback `L_I` and output feedback `L_O`.

    `L_O = None` means `H_O = I`, i.e. plain LDLQ. Returns (weight_q, encoded),
    matching `ldlq()`'s return signature.

    `stats`, if given, is filled with the RMS of the values actually handed to the
    quantizer against the RMS of the weight itself. EXL3 picks its global codebook
    scale by test-quantizing the *uncompensated* weight, so a feedback term that
    inflates the tiles pushes them outside the range that scale was chosen for.
    """
    device = L_I.device
    size_k, size_n = weight.shape
    assert size_k % 16 == 0 and size_n % 128 == 0
    tiles_k, tiles_n = size_k // 16, size_n // 16

    weight = weight.to(device, torch.float)
    weight_q = torch.zeros_like(weight)
    encoded = torch.zeros((tiles_k, tiles_n, 256), dtype = torch.short, device = device)

    perm, perm_i = tensor_core_perm(device), tensor_core_perm_i(device)

    # Δ is maintained explicitly and stays zero on tiles not yet visited, so the
    # triangular slices below only ever read quantized tiles (see module docstring).
    delta = torch.zeros_like(weight)

    # Wavefronts from the bottom-right corner. d = (tiles_k-1 - i) + (tiles_n-1 - j)
    num_waves = tiles_k + tiles_n - 1
    for d in range(num_waves):
        coords = []
        for i in range(tiles_k - 1, -1, -1):
            j = (tiles_n - 1) - (d - (tiles_k - 1 - i))
            if 0 <= j < tiles_n:
                coords.append((i, j))
        if not coords:
            continue

        batch = torch.empty((len(coords), 16, 16), dtype = torch.float, device = device)
        for b, (i, j) in enumerate(coords):
            i_lo, j_lo = i * 16, j * 16
            # T = L_Iᵀ Δ restricted to this tile's 16 rows and to columns >= j_lo.
            # Its first 16 columns are the pure input-feedback term; with H_O = I
            # that is the only term, so only those columns are worth forming.
            lI = L_I[i_lo:, i_lo : i_lo + 16].T
            if L_O is None:
                comp = lI @ delta[i_lo:, j_lo : j_lo + 16]
            else:
                lO = L_O[j_lo:, j_lo : j_lo + 16]
                T = lI @ delta[i_lo:, j_lo:]
                comp = T[:, :16] + T @ lO + delta[i_lo : i_lo + 16, j_lo:] @ lO
            batch[b] = weight[i_lo : i_lo + 16, j_lo : j_lo + 16] + comp

        if stats is not None:
            stats["fed_sq"] = stats.get("fed_sq", 0.0) + batch.square().sum().item()
            stats["fed_n"] = stats.get("fed_n", 0) + batch.numel()
            stats["fed_max"] = max(stats.get("fed_max", 0.0), batch.abs().max().item())
        tiles = batch.reshape(len(coords), 256)[:, perm]
        quant_w, quant_i = quantize_tiles_multigpu(tiles, quant_args)
        quant_w = quant_w[:, perm_i].reshape(len(coords), 16, 16)

        for b, (i, j) in enumerate(coords):
            i_lo, j_lo = i * 16, j * 16
            weight_q[i_lo : i_lo + 16, j_lo : j_lo + 16] = quant_w[b]
            delta[i_lo : i_lo + 16, j_lo : j_lo + 16] = \
                weight[i_lo : i_lo + 16, j_lo : j_lo + 16] - quant_w[b]
            encoded[i, j] = quant_i[b]

        if progress and d % 32 == 0:
            print(f"    wave {d}/{num_waves}", flush = True)

    return weight_q, encoded
