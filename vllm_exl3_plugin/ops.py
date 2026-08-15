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

from . import env
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


#: Distinct 128-row blocks decoded per pass in `embed_rows`. Bounds the
#: intermediate at roughly `in_features * 128 * chunk` elements (fp32 while
#: transformed): 256 is ~500 MiB at a 4096-wide model, small enough not to
#: surprise vLLM's memory profiler and large enough that the per-chunk Python
#: and launch overhead stays amortized.
#:
#: Changing it can perturb the last bit of a result. `had_right` is a *batched*
#: matmul whose batch count is the chunk size, and cuBLAS selects accumulation
#: order by shape, so a different chunking occasionally rounds an fp32
#: intermediate across an fp16 boundary -- measured at 14 elements in 8.2M, ~1
#: ulp, between chunk sizes 7 and 256. Mathematically the blocks are
#: independent and the chunking is exact; this is float, not logic. The default
#: is bit-exact against `dense_weight`, which is what the tests pin.
_EMBED_BLOCK_CHUNK = int(env.get("EMBED_BLOCK_CHUNK", "256"))


def embed_rows(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    bits: int,
    mcg: bool,
    mul1: bool,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Gather rows of an EXL3 tensor without dequantizing the whole thing.

    Returns (tokens, in_features), the same rows `dense_weight(...)[token_ids]`
    would give, exactly -- this is how a quantized token embedding is served, so
    the fp16 embedding never has to exist in VRAM.

    The whole idea rests on `dense_weight`'s chain touching output rows in
    exactly one place: `had_right` is block-diagonal in blocks of `HAD_BLOCK`,
    so row `t` depends on the 128-row block containing it and nothing else.
    Everything after it (`svh`) is a per-row scalar and everything before it
    (`had_left`, `suh`) either leaves output rows independent or acts along the
    input dimension, which is wanted whole anyway. So the decode is one
    `[k/16, HAD_BLOCK/16, 16*bits]` trellis slab per distinct block -- for a
    151936-row vocabulary that is 96 KiB out of 111 MiB, 0.08% of the tensor.

    Cost therefore scales with the number of *distinct* 128-blocks a batch
    touches, not with the batch size, which is why the blocks are deduplicated
    before decoding. The 128x read amplification for a single row is inherent to
    the Hadamard block size and cannot be optimized away, only amortized.

    Decoding is chunked because the intermediate is what gets big, not the
    result: decoding B blocks at once materializes `(in_features, 128 * B)`,
    which `had_left`/`had_right` then promote to fp32. A large prefill on a
    large-vocabulary model can touch every block there is -- gemma-4-12B at
    262144 vocabulary is 2048 blocks, i.e. a 3840x262144 intermediate, ~2 GiB in
    fp16 and ~4 GiB while transformed. That allocation is invisible to vLLM's
    memory profiler (its profiling run does not see a realistic spread of token
    ids), so leaving it unbounded shows up as an out-of-memory that only appears
    at high `max_num_seqs` and does not respond to the usual memory knobs.

    See docs/embeddings.md for the measurements and the derivation.
    """
    if token_ids.numel() == 0:
        return torch.empty(
            (0, trellis.shape[0] * TILE), dtype=torch.half, device=trellis.device
        )

    tiles_per_block = HAD_BLOCK // TILE
    blocks, inverse = torch.unique(token_ids // HAD_BLOCK, return_inverse=True)
    local = token_ids % HAD_BLOCK

    out = torch.empty(
        (token_ids.numel(), trellis.shape[0] * TILE),
        dtype=torch.half,
        device=trellis.device,
    )
    arange_tiles = torch.arange(tiles_per_block, device=trellis.device)
    arange_cols = torch.arange(HAD_BLOCK, device=trellis.device)

    for start in range(0, blocks.numel(), _EMBED_BLOCK_CHUNK):
        chunk = blocks[start : start + _EMBED_BLOCK_CHUNK]
        tile_index = (chunk.unsqueeze(1) * tiles_per_block + arange_tiles).flatten()
        w = reconstruct(trellis[:, tile_index, :].contiguous(), bits, mcg, mul1)

        w = had_left(w)
        w = w * suh.unsqueeze(1)
        # Blockwise along dim 1 in HAD_BLOCK-wide blocks, which is exactly the
        # block layout just gathered -- each decoded block transforms
        # independently, so chunking cannot change the result.
        w = had_right(w)
        columns = (chunk.unsqueeze(1) * HAD_BLOCK + arange_cols).flatten()
        w = w * svh[columns].unsqueeze(0)

        # Rows whose block landed in this chunk, and where inside it.
        wanted = (inverse >= start) & (inverse < start + chunk.numel())
        rows = wanted.nonzero(as_tuple=True)[0]
        if rows.numel():
            out[rows] = w[
                :, (inverse[rows] - start) * HAD_BLOCK + local[rows]
            ].t()
        del w

    return out


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
#: EXL3_RECONSTRUCT_THRESHOLD=0 to disable and always use the kernel.
RECONSTRUCT_THRESHOLD = int(
    env.get("RECONSTRUCT_THRESHOLD", "144")
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
    direct_register_custom_op(
        op_name="exl3_moe_mm",
        op_func=_exl3_moe_mm,
        fake_impl=_exl3_moe_mm_fake,
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




# ---------------------------------------------------------------------------
# MoE: exl3_mgemm behind a second custom op
# ---------------------------------------------------------------------------
#
# `exl3_mgemm` is built for exactly this shape of problem. From its own docs:
#
#   B, suh and svh are int64 tensors of *device addresses*, one per quantized
#   matrix, rather than the matrix data. A is contiguous fp16 [a_batches, m, k]
#   and C is [c_batches, m, n]. Slot j selects matrix q = indices[j]. With
#   `weights` supplied, each result is scaled by weights[j] and all active
#   slots are summed into C[0] -- a fused weighted MoE reduction.
#
# That maps onto vLLM's `apply(layer, x, topk_weights, topk_ids, ...)` almost
# directly: topk_ids becomes `indices`, topk_weights becomes `weights`.
#
# Unlike merged QKV, this kernel is usable here because every expert in a layer
# shares one bit width -- verified across all 121,212 tensors of Laguna-XS and
# 47,652 of gemma-4-26B, and re-checked at load time.
#
# Note the `mcg`/`mul1` arguments: `exl3_mgemm` declares them `uint32_t`
# multipliers but forwards them into `exl3_mgemm_gr`, which takes `bool`. The
# implicit conversion makes any nonzero value `true`, so they are booleans in
# practice, exactly as for `exl3_gemm`.
#
# The trailing `None, None` on every call below are `size_n_list`/`c_ptrs`,
# added upstream for per-matrix output widths (each slot writing to its own
# address with its own N) -- irrelevant here, since every expert in a layer
# already shares one shape. `exl3_mgemm`'s pybind binding declares no
# `py::arg` defaults, so both must be passed explicitly even though the C++
# signature defaults them to `{}`; passing None reproduces that default and
# takes the plain single-N/shared-C path exl3_gemm.cu already special-cases
# for it (size_n_list's absence is what gates c_ptrs off too).


def _exl3_moe_mm(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_trellis: torch.Tensor,
    gate_suh: torch.Tensor,
    gate_svh: torch.Tensor,
    up_trellis: torch.Tensor,
    up_suh: torch.Tensor,
    up_svh: torch.Tensor,
    down_trellis: torch.Tensor,
    down_suh: torch.Tensor,
    down_svh: torch.Tensor,
    gate_bits: int,
    down_bits: int,
    intermediate: int,
    num_experts: int,
    mcg: bool,
    mul1: bool,
    activation: str,
) -> torch.Tensor:
    e = ext()
    out_dtype = x.dtype
    num_tokens, hidden = x.shape
    top_k = topk_ids.shape[1]
    bszm = num_tokens * top_k

    if num_tokens == 0:
        return torch.empty((0, hidden), dtype=out_dtype, device=x.device)

    xh = x if x.dtype == torch.half else x.to(torch.half)
    # Slot j must hold the activations for (token j // top_k, expert
    # indices[j]), so each token's row is repeated top_k times. That ordering
    # matches topk_ids.reshape(1, -1) exactly.
    gathered = xh.repeat_interleave(top_k, dim=0).unsqueeze(1).contiguous()
    indices = topk_ids.reshape(1, -1).to(torch.long).contiguous()
    weights = topk_weights.reshape(1, -1).to(torch.half).contiguous()

    # A_had must not alias A: the kernel stages the rotated input there, and
    # the autotuner relaunches the kernel on first use.
    a_had = torch.empty_like(gathered)
    interm_g = torch.empty((bszm, 1, intermediate), dtype=torch.half, device=x.device)
    interm_u = torch.empty_like(interm_g)

    # min_index < 0 disables expert-range filtering. That filtering exists for
    # expert-parallel shards, where a rank owns a contiguous slice of experts,
    # and the kernel refuses to combine it with the multi-token weighted
    # reduction we rely on. With no expert parallelism there is nothing to
    # filter, so it stays off.
    lo, hi = -1, num_experts - 1
    # force_shape_idx = -1 and force_num_sms = 0 leave exllamav3 free to pick
    # the kernel shape and autotune the grid, which is its normal behaviour.
    sms = 0
    e.exl3_mgemm(gathered, gate_trellis, interm_g, gate_suh, a_had, gate_svh,
                 indices, None, gate_bits, -1, mcg, mul1, lo, hi, sms, num_tokens,
                 None, None)
    e.exl3_mgemm(gathered, up_trellis, interm_u, up_suh, a_had, up_svh,
                 indices, None, gate_bits, -1, mcg, mul1, lo, hi, sms, num_tokens,
                 None, None)

    interm_a = _gated_activation(activation, interm_g, interm_u)

    # interm_g is dead once interm_a exists, so it doubles as the down
    # projection's scratch -- the same reuse exllamav3 makes.
    out = torch.empty((bszm, 1, hidden), dtype=torch.half, device=x.device)
    e.exl3_mgemm(interm_a, down_trellis, out, down_suh, interm_g, down_svh,
                 indices, weights, down_bits, -1, mcg, mul1, lo, hi, sms, num_tokens,
                 None, None)

    # With `weights` given, the kernel reduces the top_k partials per token;
    # rows 0..num_tokens-1 hold the results.
    y = out[:num_tokens].squeeze(1)
    return y.to(out_dtype) if out_dtype != torch.half else y


#: `silu_and_mul` and friends compute `act(x[:d]) * x[d:]` -- the activation
#: applies to the *gate* half. vLLM's own comment says "gate x activation(up)",
#: which reads the other way round; `SiluAndMul`'s docstring and the kernels are
#: the authority.
_GATED_ACTIVATIONS = {
    "silu": lambda g: torch.nn.functional.silu(g),
    "gelu": lambda g: torch.nn.functional.gelu(g),
    "gelu_tanh": lambda g: torch.nn.functional.gelu(g, approximate="tanh"),
}


def _gated_activation(name: str, gate: torch.Tensor, up: torch.Tensor):
    fn = _GATED_ACTIVATIONS.get(name)
    if fn is None:
        raise NotImplementedError(
            f"EXL3 MoE does not implement the {name!r} activation; known: "
            f"{sorted(_GATED_ACTIVATIONS)}"
        )
    return fn(gate) * up


def _exl3_moe_mm_fake(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_trellis: torch.Tensor,
    gate_suh: torch.Tensor,
    gate_svh: torch.Tensor,
    up_trellis: torch.Tensor,
    up_suh: torch.Tensor,
    up_svh: torch.Tensor,
    down_trellis: torch.Tensor,
    down_suh: torch.Tensor,
    down_svh: torch.Tensor,
    gate_bits: int,
    down_bits: int,
    intermediate: int,
    num_experts: int,
    mcg: bool,
    mul1: bool,
    activation: str,
) -> torch.Tensor:
    return torch.empty_like(x)


def exl3_moe_mm(*args) -> torch.Tensor:
    """Routed-expert MLP over EXL3 weights, without materializing any of them."""
    return torch.ops.vllm_exl3.exl3_moe_mm(*args)


# Registered at import rather than on first use: registering a custom op while
# torch.compile is tracing is not safe, and `register()` runs early in every
# process that will need it.
_register_ops()
