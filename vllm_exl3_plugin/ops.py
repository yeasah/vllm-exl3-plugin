"""Thin wrappers over exllamav3's compiled extension.

This module depends on exllamav3's *compiled extension* (`exllamav3_ext`, a
top-level module) and deliberately not on the `exllamav3` Python package.
Importing the package runs `exllamav3/__init__.py`, which pulls in the model,
tokenizer, cache and generator stack along with formatron, kbnf, marisa_trie and
flash-linear-attention -- none of which a vLLM worker has any use for, and all
of which are dependency-resolution risk against vLLM's own pins. The extension
alone is enough.

The import is lazy regardless: `register()` runs in the vLLM frontend process as
well as in each worker, and only a process that actually touches weights should
pay for loading a CUDA extension.

Two paths live here:

- `dense_weight`, which decodes a trellis into a dense fp16 matrix via
  `ext.reconstruct`. This is Phase 0's strategy and remains the correctness
  oracle, since it is a transcription of exllamav3's own dequantization.
- `exl3_mm`, a `torch.compile`-safe custom op wrapping `ext.exl3_gemm`, which
  keeps weights quantized and is what Phase 1 actually serves with.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache

import torch
from torch.library import Library

from .format import HAD_BLOCK, TILE

_EXT = None


def ext():
    """The exllamav3 pybind extension module, imported on first use."""
    global _EXT
    if _EXT is None:
        try:
            # How exllamav3's setup.py installs it: a top-level extension
            # module, no package import required.
            import exllamav3_ext
        except ImportError:
            # JIT-compiled fallback (EXLLAMA_NOCOMPILE installs); this one does
            # go through the package.
            from exllamav3.ext import exllamav3_ext
        _EXT = exllamav3_ext
    return _EXT


@lru_cache(maxsize=8)
def _hadamard(n: int, device, dtype):
    """Normalized Sylvester-Hadamard of order `n`, matching exllamav3's.

    exllamav3 builds order 128 by recursive Sylvester doubling from the stored
    1x1 base case (`util/hadamard_data/hadamard_1.txt`), so this reproduces it
    exactly rather than approximating it -- entries are +/-1, exact in both fp16
    and fp32, scaled to orthonormal. `tests/test_kernels.py` asserts equality
    against `exllamav3.util.hadamard.get_hadamard` on a machine that has it.

    Only powers of two are reachable this way, which covers HAD_BLOCK; the
    Paley constructions exllamav3 also carries are for orders EXL3 never uses
    for this transform.
    """
    if n & (n - 1):
        raise ValueError(f"Hadamard order {n} is not a power of two")
    had = torch.ones((1, 1), dtype=dtype, device=device)
    while had.shape[0] < n:
        d = had.shape[0]
        nxt = torch.empty((d * 2, d * 2), dtype=dtype, device=device)
        nxt[:d, :d] = had
        nxt[:d, d:] = had
        nxt[d:, :d] = had
        nxt[d:, d:] = -had
        had = nxt
    # Orthonormal scaling, matching the quantizer's own `preapply_had_*` calls.
    return had * (1 / math.sqrt(n))


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


# ---------------------------------------------------------------------------
# Fused path: exl3_gemm behind an opaque custom op
# ---------------------------------------------------------------------------
#
# The kernel's own contract (exl3_gemm.cu):
#
#   A     (m, k) fp16, row-major contiguous
#   B     (k//16, n//16, 16*K) int16 trellis
#   C     (m, n) fp16 or fp32, need not be zeroed
#   suh   optional (k,) fp16 input scales/flips
#   A_had scratch the same size and dtype as A, required whenever suh is given
#   svh   optional (n,) fp16 output scales/flips
#   requires k % 16 == 0 and n % 128 == 0
#
# So the Hadamard transforms happen *inside* the kernel; the caller only has to
# supply somewhere to put the transformed activations. exllamav3 aliases that
# scratch onto A itself for its cached batch-1 path, which we cannot do: vLLM's
# activation tensor is live (a residual stream feeds later layers), so we always
# allocate a separate buffer.
#
# Everything is wrapped in one custom op because the kernel picks between four
# dispatch paths (int8 GEMV, fp16 GEMV, autotuned cooperative GEMM) based on
# batch size at launch time. Behind an opaque op, torch.compile and CUDA-graph
# capture see one stable call rather than Python branching on a shape.

_EXL3_LIB = Library("vllm_exl3", "FRAGMENT")

#: Above this many rows, decode the trellis to a dense fp16 matrix and hand the
#: multiply to cuBLAS instead of running the fused kernel. exllamav3 makes the
#: same switch at the same threshold (`AUTO_RECONSTRUCT_THRESHOLD`), and the
#: reason shows up plainly in benchmarks: the cooperative GEMM wins at decode
#: batch sizes, where the smaller weight footprint is what matters, and loses
#: badly to cuBLAS once prefill makes the multiply compute-bound.
#:
#: The dense matrix is transient -- one layer's worth, freed on return -- so
#: this costs scratch space rather than the memory saving. Set
#: VLLM_EXL3_RECONSTRUCT_THRESHOLD=0 to disable and always use the kernel.
RECONSTRUCT_THRESHOLD = int(
    os.environ.get("VLLM_EXL3_RECONSTRUCT_THRESHOLD", "144")
)


def _out_features(trellis: torch.Tensor) -> int:
    return trellis.shape[1] * TILE


def _exl3_mm(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    n = _out_features(trellis)
    k = trellis.shape[0] * TILE
    out_shape = x.shape[:-1] + (n,)
    out_dtype = x.dtype

    rows = x.numel() // x.shape[-1]
    if rows == 0:
        return torch.empty(out_shape, dtype=out_dtype, device=x.device)

    a = x.reshape(rows, x.shape[-1])
    if a.shape[1] < k:
        # exllamav3 pads both dimensions to a multiple of 128 before quantizing,
        # so a model whose hidden size is not a multiple of 128 arrives narrower
        # than the weight. The padded weight rows only ever multiply these
        # zeros, which is why zero-extending is exact rather than approximate.
        a = torch.nn.functional.pad(a, (0, k - a.shape[1]))
    # The kernels are fp16 throughout (exl3_gemm hard-checks A for kHalf), but
    # the *model* need not be: casting only at this boundary keeps the residual
    # stream in whatever dtype vLLM chose. That matters for the Gemma family,
    # which is numerically unstable end-to-end in fp16 -- vLLM refuses fp16 for
    # gemma2/gemma3 outright, and exllamav3 carries fp32 residuals for gemma4.
    # The cast itself is safe: EXL3 regularizes activations into a narrow range
    # (measured max |x| ~3.7e3 on gemma-4-12B, against fp16's 6.5e4 ceiling).
    if a.dtype != torch.half:
        a = a.to(torch.half)
    a = a.contiguous()

    if RECONSTRUCT_THRESHOLD and rows > RECONSTRUCT_THRESHOLD:
        y = _reconstruct_mm(a, trellis, suh, svh, mcg, mul1)
    else:
        y = torch.empty((rows, n), dtype=torch.half, device=x.device)
        a_had = torch.empty_like(a)
        ext().exl3_gemm(a, trellis, y, suh, a_had, svh, -1, mcg, mul1, 0)

    if out_dtype != torch.half:
        y = y.to(out_dtype)
    return y.view(out_shape)


def _reconstruct_mm(
    a: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    """Decode to dense fp16, then let cuBLAS do the multiply.

    Mirrors `LinearEXL3.reconstruct_hgemm`. The regularization that the fused
    kernel folds into its epilogue has to be applied explicitly here: the input
    Hadamard with `suh` pre-scaling, then the matmul against the *inner*
    weight, then the output Hadamard with `svh` post-scaling.
    """
    e = ext()
    bits = trellis.shape[2] // TILE
    a_had = torch.empty_like(a)
    e.had_r_128(a, a_had, suh, None, 1.0)
    w = reconstruct(trellis, bits, mcg, mul1)  # (k, n), inner weight
    y = a_had @ w
    e.had_r_128(y, y, None, svh, 1.0)
    return y


def _exl3_mm_fake(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    del suh, svh, mcg, mul1
    return torch.empty(
        x.shape[:-1] + (_out_features(trellis),), dtype=x.dtype, device=x.device
    )


def _register_ops() -> None:
    """Register into our own torch.ops namespace, never vLLM's."""
    from vllm.utils.torch_utils import direct_register_custom_op

    if hasattr(torch.ops.vllm_exl3, "exl3_mm"):
        return
    direct_register_custom_op(
        op_name="exl3_mm",
        op_func=_exl3_mm,
        fake_impl=_exl3_mm_fake,
        target_lib=_EXL3_LIB,
    )


def exl3_mm(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    """x @ dequant(trellis, suh, svh), without ever materializing the weight."""
    return torch.ops.vllm_exl3.exl3_mm(x, trellis, suh, svh, mcg, mul1)


# Registered at import rather than on first use: registering a custom op while
# torch.compile is tracing is not safe, and `register()` runs early in every
# process that will need it.
_register_ops()
