"""Thin wrappers over exllamav3's compiled extension.

Every exllamav3 import in here is deliberately lazy. `register()` runs in the
vLLM frontend process as well as in each worker, and importing `exllamav3`
loads/JITs its CUDA extension -- we only want that to happen on a process that
is actually going to touch weights.

Phase 0 uses exactly two entry points into exllamav3: `ext.reconstruct`, which
turns a trellis into a dense fp16 matrix, and the Hadamard matrix from
`util.hadamard`. The fused GEMM/GEMV kernels (`exl3_gemm`, `exl3_gemv`,
`exl3_mgemm`) are Phase 1 and are not called here.
"""

from __future__ import annotations

import math

import torch

from .format import HAD_BLOCK

_EXT = None


def ext():
    """The exllamav3 pybind extension module, imported on first use."""
    global _EXT
    if _EXT is None:
        from exllamav3.ext import exllamav3_ext

        _EXT = exllamav3_ext
    return _EXT


def exllamav3_version() -> str:
    from exllamav3.version import __version__

    return __version__


def _hadamard(n: int, device, dtype):
    from exllamav3.util.hadamard import get_hadamard_dt

    # Orthonormal scaling, matching the quantizer's own `preapply_had_*` calls.
    return get_hadamard_dt(n, device, dtype, 1 / math.sqrt(n))


def had_left(x: torch.Tensor, block: int = HAD_BLOCK) -> torch.Tensor:
    """Blockwise Hadamard along dim 0 of a (k, n) matrix.

    Mirrors `preapply_had_l` in exllamav3's quantizer: the transform is applied
    independently to each contiguous block of `block` rows, i.e. it is
    block-diagonal. This is the property that makes tensor-parallel slicing on
    `block` boundaries exact.
    """
    k, n = x.shape
    assert k % block == 0, f"{k} is not a multiple of the Hadamard block size {block}"
    orig_dtype = x.dtype
    x = x.to(torch.float)
    had = _hadamard(block, x.device, x.dtype)
    return (had @ x.view(-1, block, n)).view(k, n).to(orig_dtype)


def had_right(x: torch.Tensor, block: int = HAD_BLOCK) -> torch.Tensor:
    """Blockwise Hadamard along dim 1 of a (k, n) matrix. See `had_left`."""
    k, n = x.shape
    assert n % block == 0, f"{n} is not a multiple of the Hadamard block size {block}"
    orig_dtype = x.dtype
    x = x.to(torch.float)
    had = _hadamard(block, x.device, x.dtype)
    return (x.view(k, -1, block) @ had).view(k, n).to(orig_dtype)


def reconstruct(trellis: torch.Tensor, bits: int, mcg: bool, mul1: bool) -> torch.Tensor:
    """Decode a trellis into the dense *inner* weight, shape (in, out), fp16.

    This is the quantized matrix before regularization is undone -- the input and
    output Hadamard transforms and the suh/svh scale vectors still have to be
    applied. See `dense_weight`.
    """
    in_features = trellis.shape[0] * 16
    out_features = trellis.shape[1] * 16
    w = torch.empty(
        (in_features, out_features), dtype=torch.half, device=trellis.device
    )
    ext().reconstruct(w, trellis, bits, mcg, mul1)
    return w


def dense_weight(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    bits: int,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    """Fully dequantize one EXL3 tensor to a torch.nn.functional.linear weight.

    Returns shape (out_features, in_features), fp16.

    This undoes EXL3's regularization in the same order and the same precision as
    exllamav3's own `LinearEXL3.get_weight_tensor`, so the result should match it
    bit for bit. That equality is the cheapest available correctness oracle and
    the first thing to check once there is a GPU to run on.

    The identity being relied on, with H the (symmetric, orthonormal) blockwise
    Hadamard:

        exllamav3 forward:  y = H_n( H_k(x * suh) @ W_inner ) * svh
        this function:      W_eff = (H_k(W_inner) * suh[:, None]) H_n * svh[None, :]
                            y = x @ W_eff

    which are equal because H is symmetric and the row scaling commutes out.
    """
    w = reconstruct(trellis, bits, mcg, mul1)
    w = had_left(w)
    w = w * suh.unsqueeze(1)
    w = had_right(w)
    w = w * svh.unsqueeze(0)
    return w.t().contiguous()
