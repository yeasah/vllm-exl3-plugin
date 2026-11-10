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

from typing import Optional

import torch

from . import cpu_offload
from torch.library import Library

from . import env
from .format import HAD_BLOCK, TILE, bits_from_trellis_shape

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


def reconstruct(trellis: torch.Tensor, bits: int | float, mcg: bool, mul1: bool) -> torch.Tensor:
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
    bits: int | float,
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


def _embed_rows(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    bits: float,
    mcg: bool,
    mul1: bool,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Gather rows of an EXL3 tensor without dequantizing the whole thing.

    Returns (tokens, in_features), the same rows `dense_weight(...)[token_ids]`
    would give, exactly -- this is how a quantized token embedding is served, so
    the fp16 embedding never has to exist in VRAM.

    Dispatches: small batches take `_embed_rows_static`, which is what a CUDA
    graph can capture. The deduplicating implementation below is the large-batch
    path, and cannot be captured at any size -- it synchronizes. The rest of this
    docstring describes that path.

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

    `token_ids` is 1-D; callers flatten first, and the registered fake impl
    derives the output shape from `shape[0]` on that basis.

    See docs/embeddings.md for the measurements and the derivation.
    """
    if token_ids.shape[0] == 0:
        return torch.empty(
            (0, trellis.shape[0] * TILE), dtype=torch.half, device=trellis.device
        )

    # Branching on `shape` is host-side metadata, not a read of the tensor, so
    # it costs no synchronization and is legal mid-capture. Everything below
    # this line is not: see `_embed_rows_static`.
    if token_ids.shape[0] <= _EMBED_STATIC_MAX:
        return _embed_rows_static(
            trellis, suh, svh, bits, mcg, mul1, token_ids
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


#: Batch sizes at or below this use the capture-safe path. It must cover every
#: CUDA-graph capture size vLLM might use, because that path is the only one a
#: graph can hold: above it the deduplicating path runs, and that one
#: synchronizes. vLLM's default maximum capture size is 512.
#:
#: Above the threshold the choice is a performance one and goes the other way.
#: Deduplication saturates at the number of 128-blocks in the vocabulary, so a
#: large prefill decodes each block once (Qwen3-0.6B: 1187 blocks) where the
#: static path decodes one per token (8192). At decode batch sizes the two are
#: near-identical instead, because scattered token ids rarely share a block --
#: which is what makes the split nearly free rather than a trade.
_EMBED_STATIC_MAX = int(env.get("EMBED_STATIC_MAX", "512"))


def _embed_rows_static(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    bits: int | float,
    mcg: bool,
    mul1: bool,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """`_embed_rows` with a kernel sequence that does not depend on token values.

    Same arithmetic per block, but one block decode per token and chunking over
    the token count rather than over deduplicated blocks. Nothing here reads a
    tensor's contents back to the host: the loop bound comes from
    `token_ids.shape`, which is metadata, so the sequence of kernels is fixed by
    the batch size alone. That is what CUDA-graph capture requires, and what the
    deduplicating path cannot offer -- `torch.unique` has a data-dependent
    output size, and asking for it is a synchronization.

    A full chunk hands `reconstruct`/`had_left` exactly the same shape the
    deduplicating path does (`_EMBED_BLOCK_CHUNK * HAD_BLOCK` columns), so the
    intermediate is bounded identically and the fp32 rounding matches.
    """
    tiles_per_block = HAD_BLOCK // TILE
    n = token_ids.shape[0]
    out = torch.empty(
        (n, trellis.shape[0] * TILE), dtype=torch.half, device=trellis.device
    )
    arange_tiles = torch.arange(tiles_per_block, device=trellis.device)
    arange_cols = torch.arange(HAD_BLOCK, device=trellis.device)

    for start in range(0, n, _EMBED_BLOCK_CHUNK):
        stop = min(start + _EMBED_BLOCK_CHUNK, n)
        ids = token_ids[start:stop]
        block = ids // HAD_BLOCK
        local = ids % HAD_BLOCK

        tile_index = (block.unsqueeze(1) * tiles_per_block + arange_tiles).flatten()
        w = reconstruct(trellis[:, tile_index, :].contiguous(), bits, mcg, mul1)

        w = had_left(w)
        w = w * suh.unsqueeze(1)
        # Blockwise along dim 1 in HAD_BLOCK-wide blocks, which is the layout
        # just gathered -- one block per token, in token order.
        w = had_right(w)
        columns = (block.unsqueeze(1) * HAD_BLOCK + arange_cols).flatten()
        w = w * svh[columns].unsqueeze(0)

        # Token i's block occupies slot i, so its row is at that slot's offset
        # plus the row's position inside the block.
        slot = torch.arange(stop - start, device=trellis.device)
        out[start:stop] = w[:, slot * HAD_BLOCK + local].t()
        del w

    return out


def _embed_rows_fake(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    bits: float,
    mcg: bool,
    mul1: bool,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    del suh, svh, bits, mcg, mul1
    return torch.empty(
        (token_ids.shape[0], trellis.shape[0] * TILE),
        dtype=torch.half,
        device=trellis.device,
    )


def embed_rows(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    bits: int | float,
    mcg: bool,
    mul1: bool,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Gather rows of an EXL3 tensor, behind an opaque op.

    The implementation's control flow is driven by token *values* -- a
    `torch.unique` whose output size is data-dependent, a Python loop bounded by
    that size, and a `nonzero` inside it. Dynamo cannot trace any of it, so
    without this wrapper the whole engine fails to start whenever vLLM compiles
    the model, which is its default.

    Opacity is necessary but **not sufficient**, and the difference matters:
    see `EXL3EmbeddingMethod` and TODO `embed-rows-compile`. A CUDA graph
    captures a fixed kernel sequence, and every index tensor here comes from the
    capture-time token ids, so a replayed graph would gather capture-time rows.
    Tracing is what this fixes; capture is handled where the decision about
    graph capture actually lives.
    """
    return torch.ops.vllm_exl3.exl3_embed_rows(
        trellis, suh, svh, bits, mcg, mul1, token_ids
    )


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

#: Scratch budget in MiB for the dense weight inside the reconstruct path.
#: `reconstruct` decodes the whole (k, n) weight at once: on a 27B MLP that is
#: 5120x17408 fp16 = 170 MiB held live across the multiply, which measured
#: larger than every other dynamic allocation in a prefill step combined.
#: Decoding one column tile at a time and letting cuBLAS write into the matching
#: slice of the output bounds that scratch at this budget instead. The cost is
#: one extra read of the activations per tile -- small against the weight
#: traffic, which is paid either way. Set EXL3_RECONSTRUCT_TILE_MB=0 to decode
#: the weight whole.
RECONSTRUCT_TILE_MB = int(env.get("RECONSTRUCT_TILE_MB", "32"))

#: Reuse the reconstruct path's scratch buffers instead of allocating per call.
#: Set EXL3_SCRATCH_POOL=0 to go back to per-call allocation, which is what makes
#: the two comparable inside one build.
SCRATCH_POOL = int(env.get("SCRATCH_POOL", "1"))


#: Reusable scratch for the reconstruct path, one buffer per slot per device.
#:
#: Both transients it needs -- the decoded weight tile and the Hadamard-rotated
#: activations -- are fully overwritten before they are read, and the path runs
#: one layer at a time on one stream, so a single buffer per slot serves every
#: call. Re-requesting them per call was 46 MiB of the 107 MiB transient peak
#: measured during a 30K prefill, and that is the expensive 46 MiB: it is
#: allocated after the memory profiler's window closes, so the KV budget cannot
#: see it and the operator pays for it in utilization headroom instead
#: (docs/memory-accounting.md). Held once, the high-water mark is reached during
#: the profile run -- which runs at `max_num_batched_tokens`, the same width as
#: the widest real chunk -- so it lands inside the profiled activation and the
#: budget accounts for it.
_SCRATCH: dict[tuple[str, torch.device, int], torch.Tensor] = {}


def _scratch(
    slot: str, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """A reusable buffer for a transient that is written before it is read.

    Grows to the high-water mark and stays there. Callers must not hold the
    returned view past the point where the next call to the same slot could run,
    and must not let it escape: the tensor returned to the caller of a kernel
    wrapper is never a scratch view.

    Allocating while a CUDA graph is capturing would put the buffer in that
    graph's private pool, which is freed with the graph -- so a later replay, or
    any eager call afterwards, would be writing into memory the allocator has
    handed out again. Capture is rare on this path (it runs above
    `RECONSTRUCT_THRESHOLD` rows, and capture sizes are decode-shaped), so the
    guard simply declines to pool rather than trying to pool per graph.
    """
    nbytes = math.prod(shape) * dtype.itemsize
    if not SCRATCH_POOL or torch.cuda.is_current_stream_capturing():
        return torch.empty(shape, dtype=dtype, device=device)
    # Keyed by stream as well as slot: under dual-batch overlap two ubatches run
    # concurrently on different streams, and sharing one buffer between them
    # would be a data race that corrupts quietly rather than failing. One buffer
    # per stream is the same answer the attention workspace reaches with lanes.
    key = (slot, device, torch.cuda.current_stream(device).cuda_stream)
    buf = _SCRATCH.get(key)
    if buf is None or buf.numel() < nbytes:
        buf = torch.empty(nbytes, dtype=torch.uint8, device=device)
        _SCRATCH[key] = buf
    return buf[:nbytes].view(dtype).view(shape)


def _out_features(trellis: torch.Tensor) -> int:
    return trellis.shape[1] * TILE


#: Batches at least this large, outside graph capture, run grouped by expert (`_moe_grouped`)
_GROUPED_MIN_TOKENS = int(os.environ.get("EXL3_MOE_GROUPED_MIN", "48"))  # crossover on Qwen3.6-35B-A3B (256 x top-8)

_FORWARD_STREAM: list = [None, None]   # (forward context, raw stream) of the last launch


def _one_stream(x: torch.Tensor) -> None:
    """Refuse an exllamav3 GEMM or MoE launch on a second stream within one forward.

    Both kernels spin-wait across blocks (`group_barrier` in the MoE kernel, the split-K
    `barrier_acquire` lock in the GEMM) and size their grids to what the runtime reports as
    resident on an otherwise idle GPU. Two of them in flight at once can each hold the SMs the
    other needs to make progress, and both spin forever: GPU at 100%, VRAM traffic near zero,
    the host waiting on the step's output event. vLLM overlaps work on side streams by design
    (shared experts on `aux_stream()`, 2026-11-08 on a 3060 under WSL); the plugin switches off
    the overlaps it knows of (`plugin._patch_kernel_overlap`), and this catches any it does not,
    at graph capture or the first eager step rather than hours into a session. Graph replays
    never reach here, and capture covers every decode batch size."""
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not x.is_cuda or not is_forward_context_available():
        return
    ctx = get_forward_context()
    stream = torch._C._cuda_getCurrentRawStream(x.device.index)
    last_ctx, last_stream = _FORWARD_STREAM
    if last_ctx is ctx and last_stream != stream:
        raise RuntimeError(
            "exllamav3 kernels were launched on two CUDA streams within one forward pass. They "
            "spin-wait across blocks and deadlock when they overlap (GPU at 100%, no progress), so "
            "vLLM features that run model compute on a side stream cannot be used with EXL3. The "
            "plugin disables the ones it knows (shared-expert overlap, microbatching); this one is "
            "new. See docs/moe.md, \"Overlapped kernels deadlock\".")
    _FORWARD_STREAM[0], _FORWARD_STREAM[1] = ctx, stream


def _exl3_mm(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    _one_stream(x)
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
    # Never `shape[2] // TILE`: a half-integer width (3.5 bpw is 56 words per tile) would truncate
    # to the integer K below it and decode plausible garbage
    bits = bits_from_trellis_shape(trellis.shape, mul1)
    k = trellis.shape[0] * TILE
    n = _out_features(trellis)

    a_had = _scratch("a_had", tuple(a.shape), a.dtype, a.device)
    e.had_r_128(a, a_had, suh, None, 1.0)

    tile_n = _reconstruct_tile_n(k, n)
    if tile_n >= n:
        w = reconstruct(trellis, bits, mcg, mul1)  # (k, n), inner weight
        y = a_had @ w
    else:
        y = torch.empty((a.shape[0], n), dtype=torch.half, device=a.device)
        # `reconstruct_slice` takes the tile's row stride from the width of the
        # tensor it is handed, so a narrower view of a wider buffer decodes
        # wrong. Keeping every tile the same width lets one buffer serve the
        # whole loop: the last tile is pulled back to end at `n` instead of
        # being narrowed, recomputing the overlap rather than reallocating.
        # cuBLAS writes each product straight into the output slice --
        # `torch.mm` honours the stride, with no staging copy.
        w = _scratch("w", (k, tile_n), torch.half, a.device)
        off = 0
        while off < n:
            start = min(off, n - tile_n)
            e.reconstruct_slice(w, trellis, bits, mcg, mul1, start)
            torch.mm(a_had, w, out=y[:, start : start + tile_n])
            off = start + tile_n

    e.had_r_128(y, y, None, svh, 1.0)
    return y


def _reconstruct_tile_n(k: int, n: int) -> int:
    """Column tile width for the reconstruct path, or `n` to decode whole.

    `reconstruct_slice` requires the tile width and its offset both to be
    multiples of `HAD_BLOCK`, which EXL3's padding already guarantees for `n`
    itself; the guard is for anything that arrives unpadded.
    """
    if RECONSTRUCT_TILE_MB <= 0 or n % HAD_BLOCK:
        return n
    budget = (RECONSTRUCT_TILE_MB * 1024 * 1024) // (k * 2)
    if budget >= n:
        return n
    # Split into the fewest tiles the budget allows, then even them out: tiles
    # of `ceil(n / count)` waste far less on the pulled-back final tile than
    # tiles at the budget ceiling would. A single 128-column tile that already
    # exceeds the budget is taken anyway -- it is the narrowest the kernel
    # decodes.
    count = -(-n // max(budget, HAD_BLOCK))
    tile = -(-(-(-n // count)) // HAD_BLOCK) * HAD_BLOCK
    return min(max(tile, HAD_BLOCK), n)


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
        mutates_args=["route"],
        fake_impl=_exl3_moe_mm_fake,
        target_lib=_EXL3_LIB,
    )
    direct_register_custom_op(
        op_name="exl3_embed_rows",
        op_func=_embed_rows,
        fake_impl=_embed_rows_fake,
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


def _moe_per_row(x, topk_ids, topk_weights, gate_trellis, gate_suh, gate_svh, up_trellis, up_suh, up_svh,
                 down_trellis, down_suh, down_svh, gate_bits, down_bits, intermediate, num_experts, mcg, mul1,
                 activation):
    """Every (token, expert) assignment as its own one-row product (`exl3_mgemm`). Capturable, and the
    right shape for decode; at prefill each expert's trellis is decoded once per row it serves."""
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


#: exl3_moe's activation indices (exl3_moe_common.cuh); others take the per-row path
_MOE_ACT = {"silu": 0, "gelu": 1}
_MOE_SCRATCH: dict = {}


def _moe_fused_buffers(device, rows: int, hidden: int, intermediate: int):
    """exl3_moe's per-launch staging, sized for its row capacity at the kernel's concurrency: shared by
    every layer of a shape, as exllamav3 shares them (g_tensor_cache)."""
    key = (device, rows, hidden, intermediate)
    if key not in _MOE_SCRATCH:
        c = ext().exl3_moe_max_concurrency(device.index)
        _MOE_SCRATCH[key] = (torch.empty((c, rows, hidden), dtype=torch.half, device=device),
                             torch.empty((c, rows, hidden), dtype=torch.half, device=device),
                             torch.empty((c, rows, intermediate), dtype=torch.half, device=device),
                             torch.empty((c, rows, intermediate), dtype=torch.half, device=device))
    return _MOE_SCRATCH[key]


def _moe_recon(gate_trellis, gate_suh, gate_svh, up_trellis, up_suh, up_svh, down_trellis, down_suh, down_svh,
               hidden, intermediate, gate_bits, down_bits, mcg, mul1, activation, device):
    """exllamav3's batched reconstruct tier (`moe_batch_recon.BatchReconLayer`) over our pointer tables:
    hot experts are dequantized once per group and run as strided-batched GEMMs. Built per call (a few
    attributes); its folded mode wants per-expert scale addresses, which are exactly our suh/svh tables."""
    from exllamav3.modules import moe_batch_recon as mbr

    r = mbr.BatchReconLayer.__new__(mbr.BatchReconLayer)
    r.gated = True
    r.dims_g = r.dims_u = (hidden, intermediate, gate_bits)
    r.dims_d = (intermediate, hidden, down_bits)
    r.cb_g = r.cb_u = r.cb_d = (mcg, mul1)
    r.act_call = mbr._ACT_CALLS_GATED[activation]
    r.act_limit = 0.0
    r.device = device
    r.cap = mbr.batch_cap(r.dims_g, r.dims_u, r.dims_d)
    r.max_rows = mbr.max_group_rows(r.dims_u, r.dims_d)
    r.ptr_tables = (gate_trellis, up_trellis, down_trellis)
    r.folded = True
    r.interm_fp32 = False
    r.scales = {}
    r.scale_ptrs = {"g": (gate_suh, gate_svh), "u": (up_suh, up_svh), "d": (down_suh, down_svh)}
    return r


def _moe_grouped(x, topk_ids, topk_weights, gate_trellis, gate_suh, gate_svh, up_trellis, up_suh, up_svh,
                 down_trellis, down_suh, down_svh, gate_bits, down_bits, intermediate, num_experts, mcg, mul1,
                 activation, stage: int = -1):
    """Assignments grouped by expert, as exllamav3's own runtime runs prefill (`BlockSparseMLP.forward`):
    one readback of the per-expert counts, then `exl3_moe` over every expert within its row capacity, in
    16/32/64-row tiles, so each expert's trellis is decoded per tile rather than per row (4-11x faster per
    layer at 256-2048 tokens on Qwen3.6-35B-A3B). Deterministic: each assignment gets a scratch slot and
    one gather sums a token's slots in a fixed order. Experts over the fused capacity (256 rows), common
    at real prefill (routing is skewed: 500-800 rows for the hottest at 2.7K tokens), take exllamav3's
    batched reconstruct tier into the same slots; only an expert past that tier's 16K-row cap takes the
    per-row path. Not capturable (the readback), so used only outside graph capture."""
    from exllamav3.modules.block_sparse_mlp import FUSED_ROWS_WIDE, MTILE, MTILE_T1, MTILE_T2, TEMP_ROWS_FUSED
    from exllamav3.modules.moe_batch_recon import plan_groups

    e = ext()
    M, H = x.shape
    K = topk_ids.shape[1]
    E = num_experts
    dev = x.device
    xh = x if x.dtype == torch.half else x.to(torch.half)
    wide = bool(MTILE) and mul1
    R = FUSED_ROWS_WIDE if wide else TEMP_ROWS_FUSED
    tsg, tsu, tig, tiu = _moe_fused_buffers(dev, R, H, intermediate)

    if stage >= 0:
        # before the readback below, so the next layer's copy is already queued behind this layer's work
        cpu_offload.prefetch_next(stage)
    flat_e = topk_ids.reshape(-1).to(torch.long)
    flat_w = topk_weights.reshape(-1).to(torch.half)
    flat_t = torch.arange(M, device=dev).repeat_interleave(K)
    order = flat_e.argsort()
    token_sorted = flat_t[order]
    weight_sorted = flat_w[order]
    count = torch.bincount(flat_e, minlength=E + 1)
    counts = count.tolist()
    if stage >= 0:
        # offloaded experts: only the routed ones not already in VRAM are copied in (cpu_offload.stage_routed)
        gate_trellis, up_trellis, down_trellis = cpu_offload.stage_routed(
            stage, counts, (gate_trellis, up_trellis, down_trellis))

    recon = _moe_recon(gate_trellis, gate_suh, gate_svh, up_trellis, up_suh, up_svh, down_trellis, down_suh,
                       down_svh, H, intermediate, gate_bits, down_bits, mcg, mul1, activation, dev)
    heavy = [i for i, c in enumerate(counts[:E]) if R < c <= recon.max_rows]
    groups = plan_groups(heavy, lambda i: counts[i], recon.cap) if heavy else []

    base = [0] * (E + 1)
    kind = [0] * (E + 1)
    starts = [0] * (E + 1)
    n_slots = acc = 0
    for i, c in enumerate(counts):
        starts[i] = acc
        acc += c
        if i < E and 0 < c <= R:
            base[i], kind[i] = n_slots, 1
            n_slots += c
    group_slots = []
    for grp in groups:
        cmax = max(counts[i] for i in grp)
        group_slots.append(n_slots)
        for b, i in enumerate(grp):
            base[i], kind[i] = n_slots + b * cmax, 2
        n_slots += len(grp) * cmax
    tables = torch.tensor([base, starts, kind], dtype=torch.long).pin_memory().to(dev, non_blocking=True)
    inv_order = torch.empty_like(order).scatter_(0, order, torch.arange(order.numel(), device=dev))
    scratch = torch.empty((max(n_slots, 1), H), dtype=torch.float, device=dev)
    out = torch.zeros((M, H), dtype=torch.float, device=dev)
    act = _MOE_ACT[activation]

    def run(num_active, lo, hi, m_tile):
        e.exl3_moe(xh, out, count, token_sorted, weight_sorted, tsg, tsu, tig, tiu, act,
                   gate_bits, gate_bits, down_bits, gate_trellis, gate_suh, gate_svh, up_trellis, up_suh, up_svh,
                   down_trellis, down_suh, down_svh, mcg, mul1, mcg, mul1, mcg, mul1, 0.0, num_active,
                   scratch, tables[0], lo, hi, m_tile)

    fused = [c for c in counts[:E] if 0 < c <= R]
    if fused:
        t1 = sum(1 for c in fused if MTILE_T1 < c <= MTILE_T2)
        t2 = sum(1 for c in fused if c > MTILE_T2)
        if wide and (t1 or t2):
            if t2:
                run(t2, MTILE_T2 + 1, R, 64)
            if t1:
                run(t1, MTILE_T1 + 1, MTILE_T2, 32)
            if len(fused) - t1 - t2:
                run(len(fused) - t1 - t2, 1, MTILE_T1, 16)
        else:
            run(len(fused), 1, R, 16)
    if groups:
        y_ext = torch.cat([xh, xh.new_zeros((1, H))])
        tok_ext = torch.cat([token_sorted, token_sorted.new_full((1,), M)])
        w_ext = torch.cat([weight_sorted, weight_sorted.new_zeros((1,))])
        for grp, slot in zip(groups, group_slots):
            cmax = max(counts[i] for i in grp)
            recon.run_group(y_ext, out, tok_ext, w_ext, grp, [starts[i] for i in grp], [counts[i] for i in grp],
                            out_slab=scratch[slot: slot + len(grp) * cmax])
    if fused or groups:
        e.exl3_moe_gather(out, scratch, flat_e, inv_order, tables[1, :E], tables[0, :E], tables[2, :E],
                          weight_sorted)

    for i in (i for i, c in enumerate(counts[:E]) if c > recon.max_rows):
        rows = slice(starts[i], starts[i] + counts[i])
        toks = token_sorted[rows]
        part = _moe_per_row(xh.index_select(0, toks), topk_ids.new_full((counts[i], 1), i),
                            weight_sorted[rows].unsqueeze(1), gate_trellis, gate_suh, gate_svh, up_trellis, up_suh,
                            up_svh, down_trellis, down_suh, down_svh, gate_bits, down_bits, intermediate, E, mcg,
                            mul1, activation)
        out.index_add_(0, toks, part.float())
    return out.to(x.dtype)


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
    gate_bits: float,
    down_bits: float,
    intermediate: int,
    num_experts: int,
    mcg: bool,
    mul1: bool,
    activation: str,
    route: Optional[torch.Tensor] = None,
    stage: int = -1,
) -> torch.Tensor:
    _one_stream(x)
    if route is not None:
        # The expert cache's routing capture (expert_cache.py), done here rather than in `apply`: a
        # second consumer of `topk_ids` in the compiled forward changed how Inductor fused the router,
        # so a cached run's numbers drifted from resident at the ULP level in graph mode
        rows = min(x.shape[0], route.shape[0])
        route[:rows].copy_(topk_ids[:rows])
    num_tokens = x.shape[0]
    if num_tokens == 0:
        return torch.empty((0, x.shape[1]), dtype=x.dtype, device=x.device)
    args = (x, topk_ids, topk_weights, gate_trellis, gate_suh, gate_svh, up_trellis, up_suh, up_svh, down_trellis,
            down_suh, down_svh, gate_bits, down_bits, intermediate, num_experts, mcg, mul1, activation)
    if (num_tokens >= _GROUPED_MIN_TOKENS and activation in _MOE_ACT
            and not torch.cuda.is_current_stream_capturing()):
        return _moe_grouped(*args, stage=stage)
    if stage >= 0 and num_tokens > cpu_offload.STAGE_MIN_TOKENS:
        staged = cpu_offload.stage_all(stage)
        if staged is not None:
            args = (*args[:3], staged[0], *args[4:6], staged[1], *args[7:9], staged[2], *args[10:])
    return _moe_per_row(*args)



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
    gate_bits: float,
    down_bits: float,
    intermediate: int,
    num_experts: int,
    mcg: bool,
    mul1: bool,
    activation: str,
    route: Optional[torch.Tensor] = None,
    stage: int = -1,
) -> torch.Tensor:
    return torch.empty_like(x)


def exl3_moe_mm(*args) -> torch.Tensor:
    """Routed-expert MLP over EXL3 weights, without materializing any of them."""
    return torch.ops.vllm_exl3.exl3_moe_mm(*args)


# Registered at import rather than on first use: registering a custom op while
# torch.compile is tracing is not safe, and `register()` runs early in every
# process that will need it.
_register_ops()
