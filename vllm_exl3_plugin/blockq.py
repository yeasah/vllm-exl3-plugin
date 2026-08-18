"""Block-scaled integer storage for a token embedding.

EXL3 leaves `embed_tokens` at fp16 in every published checkpoint, which is a
quarter to a half of a mid-size checkpoint (docs/embeddings.md). This module is
the storage format that fixes it for *untied* models, where there is no
quantized `lm_head` to serve the lookup from.

Why this layout rather than the trellis, a per-row scheme, or GGUF -- all three
were measured, and the note has the tables:

  - The trellis optimizes `x @ W.T` against typical activations, which is what a
    head needs. An embedding needs each individual *row* accurate as a vector,
    and a scalar scheme beats the trellis ~89x at equal bits on that job.
  - Per-row min/max is dominated at every matched byte count, by up to 134x: one
    outlier component sets the scale for a whole row, and a row is one token.
  - GGUF's k-quants tie this at matched bytes and lose to it at 3.5 bpw, while
    costing an encoder we cannot write (`gguf-py` cannot emit k-quants) and a
    dequant kernel that is *slower* here, because an opaque custom op cannot fuse
    into the graph the way the plain-torch decode below does.

docs/blockq-format.md is the format reference, written for someone implementing
against it from outside this codebase. Layout, per embedding, alongside the
checkpoint's existing tensors:

    <key>.bq_q   uint8   [vocab, hidden // 2]   two 4-bit values per byte
    <key>.bq_s   uint8   [vocab, 2, hidden // BLOCK]   scale codes, then min codes
    <key>.bq_r   fp32    [vocab, 4]   (scale_lo, scale_step, min_lo, min_step)

A value is reconstructed as `q * scale + min`, where the per-block `scale` and
`min` are themselves 8-bit codes against one affine range per row -- 16 bits of
scale metadata per 32 values, the same overhead a GGUF k-quant carries, with
every field byte-aligned and no superblock. Row `t` depends on nothing outside
row `t` of these three tensors, so a lookup is a slice and vocab-parallel TP is a
row split; contrast `ops.embed_rows`, which has to decode a whole 128-row
Hadamard block to read one row.

`bq_r` is fp32 rather than fp16 deliberately: it is 16 bytes per row (4 MB on a
262144-row vocabulary, 0.03 bpw) and it is the one place where precision is
shared across a whole row, so bf16's 8 mantissa bits would be a real loss for no
meaningful saving.
"""

from __future__ import annotations

import torch

from . import format

#: Values per scale block. Measured at 32, 64 and 128; 32 is the operating point
#: (docs/embeddings.md, "Build or adopt"). Changing it changes the file format.
BLOCK = 32

#: Only 4-bit values are packed today. The constant-depth result is what makes
#: that enough -- one depth covers every model measured, at or below its noise
#: floor -- and it keeps both the encoder and the decode on nibble boundaries.
#: 3- and 5-bit packing straddle bytes and wait until something demands them.
BITS = 4


def encode(weight: torch.Tensor, block: int = BLOCK) -> dict[str, torch.Tensor]:
    """Quantize an fp16/fp32 `[vocab, hidden]` embedding into the stored tensors.

    Deliberately identical, arithmetic and all, to qbench's `blockq:N` simulation
    that the quality measurements were made with -- including that the *quantized*
    scale and min are what the values are then quantized against, so a scale
    rounded down clips its block rather than silently rescaling it. Encoding
    against the unquantized scale and reconstructing with the quantized one would
    measure better here and worse in the model.

    Reproducible for a given device, not across devices: `.round()` breaks ties on
    a computed float, and CPU and GPU disagree on a handful of codes per
    vocabulary. The decoded values still agree to fp16 rounding, so this is a
    byte-reproducibility caveat for a tool, not a correctness one for a model.
    """
    if weight.dim() != 2:
        raise format.EXL3FormatError(f"embedding must be 2-D, got {list(weight.shape)}")
    vocab, hidden = weight.shape
    format.check_blockq_hidden(hidden, block)

    levels = 2**BITS - 1
    w = weight.float()
    v = w.view(vocab, hidden // block, block)

    lo = v.amin(dim=-1)                                    # [vocab, nblk]
    sc = (v.amax(dim=-1) - lo).clamp_min(1e-12) / levels

    def code8(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """8-bit codes for one per-block scalar array, against one range per row."""
        t_lo = t.amin(dim=-1, keepdim=True)
        step = (t.amax(dim=-1, keepdim=True) - t_lo).clamp_min(1e-12) / 255.0
        codes = ((t - t_lo) / step).round().clamp(0, 255)
        return codes.to(torch.uint8), t_lo.squeeze(-1), step.squeeze(-1)

    sc_codes, sc_lo, sc_step = code8(sc)
    lo_codes, lo_lo, lo_step = code8(lo)

    # Reconstruct exactly what the decoder will see, and quantize against that.
    sc_q = sc_codes.float() * sc_step.unsqueeze(-1) + sc_lo.unsqueeze(-1)
    lo_q = lo_codes.float() * lo_step.unsqueeze(-1) + lo_lo.unsqueeze(-1)
    q = ((v - lo_q.unsqueeze(-1)) / sc_q.unsqueeze(-1).clamp_min(1e-12))
    q = q.round().clamp(0, levels).to(torch.uint8).view(vocab, hidden)

    return {
        "bq_q": pack(q),
        "bq_s": torch.stack([sc_codes, lo_codes], dim=1).contiguous(),
        "bq_r": torch.stack([sc_lo, sc_step, lo_lo, lo_step], dim=-1).contiguous(),
    }


def pack(q: torch.Tensor) -> torch.Tensor:
    """`[rows, hidden]` of 4-bit values into `[rows, hidden // 2]` bytes.

    Even index in the low nibble, which is the order llama.cpp and every other
    nibble format uses; there is no reason to differ and one reason not to.
    """
    rows, hidden = q.shape
    v = q.view(rows, hidden // 2, 2)
    return (v[..., 0] | (v[..., 1] << 4)).contiguous()


def unpack(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of `pack`, as `[rows, hidden]` uint8."""
    rows, half = packed.shape
    return torch.stack([packed & 0x0F, packed >> 4], dim=-1).view(rows, half * 2)


def decode(
    bq_q: torch.Tensor,
    bq_s: torch.Tensor,
    bq_r: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Reconstruct `[rows, hidden]` from already-gathered rows of the storage.

    Plain torch on purpose: inductor fuses this into the surrounding graph, which
    measured faster than calling out to a hand-written dequant kernel that cannot
    be fused (docs/embeddings.md). Scale reconstruction is done in fp32 because
    the two affine ranges are shared across a whole row.
    """
    rows, half = bq_q.shape
    hidden = half * 2
    nblk = bq_s.shape[-1]
    block = hidden // nblk

    q = unpack(bq_q).view(rows, nblk, block).float()
    scale = bq_s[:, 0].float() * bq_r[:, 1:2] + bq_r[:, 0:1]
    minv = bq_s[:, 1].float() * bq_r[:, 3:4] + bq_r[:, 2:3]
    out = q * scale.unsqueeze(-1) + minv.unsqueeze(-1)
    return out.view(rows, hidden).to(out_dtype)


def gather(
    ids: torch.Tensor,
    bq_q: torch.Tensor,
    bq_s: torch.Tensor,
    bq_r: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Embedding lookup: take `ids`' rows out of storage and decode only those.

    Shape-preserving like the `F.embedding` it replaces -- `[*ids.shape, hidden]`
    -- so a caller never has to know that the decode works on flattened rows.
    """
    flat = ids.reshape(-1)
    rows = decode(
        torch.index_select(bq_q, 0, flat),
        torch.index_select(bq_s, 0, flat),
        torch.index_select(bq_r, 0, flat),
        out_dtype,
    )
    return rows.view(*ids.shape, rows.shape[-1])
