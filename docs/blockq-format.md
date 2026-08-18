# The `blockq` embedding format

A small, self-contained storage format for a quantized **token embedding**, used by
`vllm-exl3-plugin` to serve the `embed_tokens` matrix of an untied EXL3 checkpoint
without ever materializing it densely. Roughly 4.53 bits per weight, against 16 for the
fp16 tensor it replaces.

This note is the format reference: what is stored, how to decode it, how to produce it,
and what may be assumed about it. *Why* it looks like this — including why it is not
per-row, not a trellis, and not a GGUF k-quant — is measured out in
[embeddings.md](embeddings.md); none of that is repeated here.

Both reference implementations below were written from this description and check out
byte-identically against the plugin's own code, the decoder on a real 248320 x 4096
checkpoint. If you implement from this page, that is the bar you should be able to hit.

## The shape of the idea

One embedding row is one token's vector, and quantizing it well means keeping *that
row* accurate. Rows are quantized independently, and within a row, values are grouped
into **blocks of 32** that each get their own `(min, scale)` pair — so one outlier
component spoils 32 values rather than the whole row.

Those per-block scalars are themselves quantized, to 8 bits each, against one affine
range per row. That is what keeps the metadata cheap: 16 bits per 32 values (0.5 bpw)
rather than the 32 bits a raw fp16 pair per block would cost.

## What is stored

Three tensors per embedding, named by appending to the module key (so
`model.embed_tokens` yields `model.embed_tokens.bq_q`, and so on):

| tensor | dtype | shape | meaning |
|---|---|---|---|
| `bq_q` | `uint8` | `[vocab, hidden // 2]` | 4-bit values, two per byte |
| `bq_s` | `uint8` | `[vocab, 2, hidden // 32]` | per-block codes: `[:, 0]` scales, `[:, 1]` mins |
| `bq_r` | `float32` | `[vocab, 4]` | per-row `(scale_lo, scale_step, min_lo, min_step)` |

Constants: **block size 32**, **4-bit values**. Neither is stored, because neither
varies — the block size is fixed by the format and 4 bits is the only width packed (one
depth covered every model measured, and nibbles keep both ends byte-aligned). `hidden`
is recoverable as `bq_q.shape[1] * 2`, and the block count as `bq_s.shape[2]`.

**Nibble order**: value `2i` is the low nibble of byte `i`, value `2i+1` the high
nibble. Same convention as llama.cpp and everything else that packs nibbles.

## Decoding

For row `t`, block `b`, position `j` within the block:

```
scale[t, b] = bq_s[t, 0, b] * bq_r[t, 1] + bq_r[t, 0]
min  [t, b] = bq_s[t, 1, b] * bq_r[t, 3] + bq_r[t, 2]
value[t, b * 32 + j] = q[t, b * 32 + j] * scale[t, b] + min[t, b]
```

Reference decoder, numpy, complete:

```python
BLOCK = 32

def decode_rows(bq_q, bq_s, bq_r, rows):
    """bq_q [V, H/2] u8, bq_s [V, 2, H/32] u8, bq_r [V, 4] f32 -> [len(rows), H] f32"""
    q8 = bq_q[rows]
    n, half = q8.shape
    hidden = half * 2
    q = np.empty((n, hidden), dtype=np.uint8)
    q[:, 0::2] = q8 & 0x0F                      # even index: low nibble
    q[:, 1::2] = q8 >> 4                        # odd index: high nibble
    s = bq_s[rows].astype(np.float32)
    r = bq_r[rows].astype(np.float32)
    scale = s[:, 0] * r[:, 1:2] + r[:, 0:1]     # [n, nblk]
    minv  = s[:, 1] * r[:, 3:4] + r[:, 2:3]
    nblk = s.shape[2]
    out = q.reshape(n, nblk, BLOCK).astype(np.float32)
    return (out * scale[:, :, None] + minv[:, :, None]).reshape(n, hidden)
```

Do the scale reconstruction in fp32 even if the model runs in bf16: the two affine
ranges are shared across a whole row, and bf16's 8 mantissa bits are not enough for
them. Cast the *result*, not the arithmetic.

## Encoding

Per row: take each block's `min` and `max`, derive its scale, quantize the two
resulting per-block arrays to 8 bits, and only then quantize the values.

**The one subtlety worth transcribing carefully**: values are quantized against the
*reconstructed* (already 8-bit-quantized) scale and min, not the exact ones. A scale
that rounded down therefore clips its block, which is correct — encoding against the
exact scale and reconstructing with the quantized one measures better in isolation and
worse in the model.

```python
BLOCK, BITS = 32, 4
LEVELS = 2**BITS - 1

def code8(x):
    """8-bit codes for a per-row array, plus the (lo, step) that decode them."""
    lo = x.min(axis=-1, keepdims=True)
    step = np.maximum(x.max(axis=-1, keepdims=True) - lo, 1e-12) / 255.0
    codes = np.clip(np.rint((x - lo) / step), 0, 255).astype(np.uint8)
    return codes, lo[..., 0], step[..., 0]

def encode(w):                                   # w: [vocab, hidden] float
    v = w.astype(np.float32).reshape(w.shape[0], -1, BLOCK)
    lo = v.min(axis=-1)
    sc = np.maximum(v.max(axis=-1) - lo, 1e-12) / LEVELS
    sc_codes, sc_lo, sc_step = code8(sc)
    lo_codes, lo_lo, lo_step = code8(lo)
    sc_q = sc_codes.astype(np.float32) * sc_step[:, None] + sc_lo[:, None]
    lo_q = lo_codes.astype(np.float32) * lo_step[:, None] + lo_lo[:, None]
    q = np.clip(np.rint((v - lo_q[..., None]) / np.maximum(sc_q[..., None], 1e-12)),
                0, LEVELS).astype(np.uint8).reshape(w.shape)
    return {"bq_q": q[:, 0::2] | (q[:, 1::2] << 4),
            "bq_s": np.stack([sc_codes, lo_codes], axis=1),
            "bq_r": np.stack([sc_lo, sc_step, lo_lo, lo_step], axis=-1).astype(np.float32)}
```

Quantize from the **original** fp16/bf16 embedding. For an untied EXL3 checkpoint that
is the checkpoint's own `embed_tokens.weight`, untouched by the quantizer. Never
quantize a dequantized trellis: it is the same matrix already ~2% lossy, and the errors
compound for no saving.

## Constraints and invariants

- `hidden % 32 == 0`. True of every real model (`num_heads x head_dim`); assert it
  rather than accommodating it. `hidden` is even follows from this, which is what
  nibble packing needs.
- **Rows are independent.** Nothing in row `t`'s decode reads any other row. This is the
  property everything else rests on: a lookup is a slice, vocabulary-parallel tensor
  parallelism is a row split with no alignment rule, and a repair tool can encode a row
  range and get bytes identical to slicing the whole tensor's output.
- **A zero row decodes to zeros**, so unused vocabulary slots and any padding rows an
  engine appends cost nothing and need no special case.
- **Encoding is reproducible per device, not across devices.** `np.rint`/`.round()`
  breaks ties on a computed float, and CPU and GPU disagree on a handful of codes per
  vocabulary. Decoded values still agree to fp16 rounding. It is a byte-reproducibility
  caveat for tooling, not a correctness one for a model.

## Size

```
bpw = 4  +  2 * 8 / 32  +  4 * 32 / hidden
    = values + per-block scale metadata + per-row ranges
```

| model | vocab x hidden | bpw | stored | fp16 was |
|---|---|---|---|---|
| Qwen3.5-9B | 248320 x 4096 | 4.5312 | 549.4 MiB | 1940.0 MiB |
| MiniCPM5-1B | 130560 x 1536 | 4.5833 | 109.6 MiB | 382.5 MiB |

## Working with it

**Produce a checkpoint** (untied models; a tied model needs none of this, being served
from its existing quantized `lm_head`):

```
tools/quantize_embedding.py <checkpoint-dir> <output-dir>
```

It quantizes the checkpoint's own embedding, drops the dense tensor, hardlinks every
shard it does not touch, and records the result in `quantization_config.json`:

```json
"model.embed_tokens": {
  "stored_tensors": { "model.embed_tokens.bq_q": {"shape": [...], "n_bytes": ..., "dtype": "torch.uint8"}, ... },
  "quant_format": "exl3_blockq",
  "bits_per_weight": 4,
  "block_size": 32
}
```

**Detection** keys off either that `quant_format` or the presence of `*.bq_q` in the
safetensors index, whichever the checkpoint offers — single-file checkpoints have no
index, so the storage map is sometimes the only evidence.

**In this codebase**: `vllm_exl3_plugin/blockq.py` is the format (encode / pack / unpack
/ decode / gather), `format.py` holds the torch-free shape rules, and
`EXL3BlockQEmbeddingMethod` in `quantization/embedding.py` serves it — a row gather plus
the decode above, in plain torch so that inductor fuses it into the surrounding graph.
`tests/test_blockq.py` covers the arithmetic, row independence, and the eager /
compiled / CUDA-graph-replay paths.

**Serving it elsewhere** needs no kernel: the decode is a gather, a nibble split and a
multiply-add. If you are writing one, the two things to get right are the nibble order
and quantizing against the reconstructed scales.

## What it is not

- **Not GGUF-compatible.** It borrows the *idea* of hierarchical block scales from
  k-quants and nothing else; the layout is its own, byte-aligned throughout, with no
  superblock and no packed 6-bit scale fields.
- **Not for an output head.** There is no matmul path — decoding to dense fp16 to
  multiply would return the entire saving. Heads stay on the EXL3 trellis, which is
  measurably the right encoding for that job.
- **Not a general-purpose weight format.** It is deliberately narrow: one tensor role,
  one bit width, one block size.
