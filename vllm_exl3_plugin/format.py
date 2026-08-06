"""Knowledge of the EXL3 on-disk format, expressed without torch.

Everything in this module is pure arithmetic over shapes and tensor names, so it
is testable on a machine with no GPU, no torch and no vLLM. The rest of the
plugin depends on it for every shape decision it makes.

Reference: exllamav3 v1.3.0, `exllamav3/modules/quant/exl3.py` and
`exllamav3/modules/linear.py`.

Per quantized linear, an EXL3 checkpoint stores:

    <key>.trellis   int16   [in_features // 16, out_features // 16, 16 * K]
    <key>.suh       fp16    [in_features]
    <key>.svh       fp16    [out_features]
    <key>.bias      fp16    [out_features]        (optional)
    <key>.mcg       uint32  [1]                   (optional, codebook selector)
    <key>.mul1      uint32  [1]                   (optional, codebook selector)
    <key>.su        int16   packed signs          (legacy, pre-v0.0.2)
    <key>.sv        int16   packed signs          (legacy, pre-v0.0.2)

K is the integer bit width of that one tensor, 1..8. Weights are quantized in
16x16 tiles, and 256 weights * K bits packs into 16*K int16 values, which is why
K is recoverable from the trellis shape alone.
"""

from __future__ import annotations

import math

# Weights are trellis-coded in 16x16 tiles; the trellis tensor is tile-granular
# on both dimensions.
TILE = 16

# Regularization applies a Hadamard transform in fixed blocks of this size along
# both the input and the output dimension (`had_k`/`had_n` in exllamav3's
# quantizer, `had_r_128` at inference time). A slice of either dimension is only
# mathematically self-contained on a multiple of this.
HAD_BLOCK = 128

# exllamav3's `Linear` pads both dimensions up to a multiple of this before
# quantizing (`pad_to = 128`), so stored dims can exceed the model's real dims.
PAD_TO = 128

MIN_BITS = 1
MAX_BITS = 8

#: Suffixes that belong to an EXL3-quantized linear rather than to the model.
EXL3_SUFFIXES = (
    ".trellis",
    ".suh",
    ".svh",
    ".mcg",
    ".mul1",
    ".su",
    ".sv",
)

#: Subset that must be present for a module to count as EXL3 storage.
REQUIRED_SUFFIXES = (".trellis",)


class EXL3FormatError(ValueError):
    """Raised when a checkpoint does not match the EXL3 layout we understand."""


def pad_dim(n: int, pad_to: int = PAD_TO) -> int:
    """Round a feature count up the way exllamav3's `Linear` does before quantizing."""
    return (n + pad_to - 1) // pad_to * pad_to


def bits_from_trellis_shape(shape) -> int:
    """Recover the per-tensor bit width K from a trellis tensor's shape."""
    if len(shape) != 3:
        raise EXL3FormatError(f"trellis must be 3-dimensional, got shape {list(shape)}")
    last = shape[2]
    if last % TILE != 0:
        raise EXL3FormatError(
            f"trellis last dim {last} is not a multiple of {TILE}; "
            "cannot recover bit width"
        )
    bits = last // TILE
    if not MIN_BITS <= bits <= MAX_BITS:
        raise EXL3FormatError(f"implausible bit width {bits} from trellis shape {list(shape)}")
    return bits


def dims_from_trellis_shape(shape) -> tuple[int, int]:
    """Recover (in_features, out_features) from a trellis tensor's shape."""
    if len(shape) != 3:
        raise EXL3FormatError(f"trellis must be 3-dimensional, got shape {list(shape)}")
    return shape[0] * TILE, shape[1] * TILE


def trellis_shape(in_features: int, out_features: int, bits: int) -> tuple[int, int, int]:
    """The trellis shape a tensor of these dimensions and bit width must have."""
    if in_features % TILE or out_features % TILE:
        raise EXL3FormatError(
            f"EXL3 dimensions must be multiples of {TILE}, "
            f"got in={in_features} out={out_features}"
        )
    return in_features // TILE, out_features // TILE, TILE * bits


def is_had_aligned(n: int) -> bool:
    """Whether a dimension can be split at `n` without straddling a Hadamard block."""
    return n % HAD_BLOCK == 0


def check_tp_split(dim_size: int, tp_size: int, what: str) -> None:
    """Validate that splitting `dim_size` across `tp_size` ranks stays exact.

    EXL3's Hadamard transform is block-diagonal in blocks of `HAD_BLOCK`, so an
    even split lands on block boundaries -- and is therefore exactly equal to the
    unsplit computation -- if and only if the per-rank width is a multiple of
    `HAD_BLOCK`. exllamav3's own tensor-parallel planner enforces the same rule
    (`TPAllocation(channel_width = 128)` in `modules/linear.py`).
    """
    if tp_size <= 1:
        return
    if dim_size % tp_size:
        raise EXL3FormatError(
            f"{what}: {dim_size} does not divide evenly across {tp_size} ranks"
        )
    per_rank = dim_size // tp_size
    if not is_had_aligned(per_rank):
        raise EXL3FormatError(
            f"{what}: tensor-parallel shard would be {per_rank} channels wide, "
            f"which is not a multiple of the EXL3 Hadamard block size "
            f"({HAD_BLOCK}). Splitting here would cut a Hadamard block in half "
            f"and silently produce wrong results."
        )


#: How far the measured gate/up scale ratio may sit from an exact power of two,
#: in log2 units. 0.15 is ~11%, which comfortably covers the spread introduced by
#: per-channel calibration (Laguna's per-expert ratios run 119..130 around a true
#: 128) while staying far from the next power of two.
_DIVISOR_LOG2_TOLERANCE = 0.15


def infer_interm_divisor(ratio: float) -> float:
    """Recover the constant folded into a routed-expert up projection.

    exllamav3 scales some architectures' routed `up_proj` down by a fixed
    `interm_div` and multiplies the routing weights by the same constant to
    compensate, purely to keep the fp16 intermediate in range (Laguna-XS:
    `interm_div = 128.0`). For an EXL3 checkpoint the scale is *baked into the
    stored weights* -- `Linear.load_exl3` ignores its own `weight_scale`, which
    only ever applies on the fp16 fallback path -- so the compensating factor is
    the only half of the pair a consumer can see, and it lives in exllamav3's
    architecture definition rather than anywhere in the checkpoint.

    That makes the constant unrecoverable by reading config.json: the model's
    `moe_routed_scaling_factor` is the *unscaled* value, correct for the original
    weights and wrong by exactly `interm_div` for these. It is recoverable from
    the weights themselves, because the scale lands wholly in the up projection's
    input scale `suh`, leaving `svh` and the trellis untouched.

    `ratio` is a robust estimate of `mean|suh_gate| / mean|suh_up|` across
    experts. Per-channel calibration puts a few percent of noise on it, so the
    result is snapped to the nearest power of two -- exllamav3's constants are
    powers of two, and snapping recovers the exact value rather than a
    slightly-off measurement.

    >>> infer_interm_divisor(0.9731)          # gemma-4-26B: no divisor
    1.0
    >>> infer_interm_divisor(127.58)          # Laguna-XS
    128.0

    Raises rather than guessing when the ratio is neither ~1 nor near a power of
    two, since applying the wrong constant here is a silent factor-of-N error in
    every routed expert.
    """
    if ratio <= 0.0 or not math.isfinite(ratio):
        raise EXL3FormatError(
            f"routed-expert gate/up scale ratio is {ratio!r}, which cannot be a "
            "scale factor; the up projection's suh is probably not loaded"
        )
    exponent = math.log2(ratio)
    nearest = round(exponent)
    if nearest == 0:
        return 1.0
    if nearest < 0 or abs(exponent - nearest) > _DIVISOR_LOG2_TOLERANCE:
        raise EXL3FormatError(
            f"routed-expert up projection is scaled by {ratio:.4g} relative to "
            f"the gate projection, which is neither ~1 nor close to a power of "
            f"two. exllamav3 folds a power-of-two `interm_div` into these "
            f"weights and compensates in the routing weights; this plugin "
            f"recovers it by measurement, and cannot here. Refusing to guess, "
            f"because the wrong constant is a silent factor-of-N error in every "
            f"routed expert."
        )
    return float(2**nearest)


def shard_bounds(
    dim_size: int, tp_rank: int, tp_size: int, what: str = "dimension"
) -> tuple[int, int]:
    """Half-open [first, last) of `dim_size` belonging to `tp_rank`.

    Validates the split first, so an unsafe tensor-parallel degree fails here
    rather than producing quietly wrong numbers.
    """
    check_tp_split(dim_size, tp_size, what)
    per_rank = dim_size // tp_size
    return tp_rank * per_rank, (tp_rank + 1) * per_rank


def module_key_for_tensor(name: str) -> str | None:
    """Map a checkpoint tensor name to its owning module key, or None.

    >>> module_key_for_tensor("model.layers.0.self_attn.q_proj.trellis")
    'model.layers.0.self_attn.q_proj'
    >>> module_key_for_tensor("model.layers.0.input_layernorm.weight") is None
    True
    """
    for suffix in EXL3_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def quantized_module_keys(tensor_names) -> set[str]:
    """The set of module keys that carry EXL3 storage, given all checkpoint names."""
    keys: set[str] = set()
    for name in tensor_names:
        for suffix in REQUIRED_SUFFIXES:
            if name.endswith(suffix):
                keys.add(name[: -len(suffix)])
    return keys
