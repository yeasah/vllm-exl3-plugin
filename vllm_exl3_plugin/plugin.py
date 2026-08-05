"""Plugin entry point.

vLLM loads `vllm.general_plugins` entry points once per process -- the frontend,
the engine core, and every worker -- so `register()` must be idempotent.
`register_quantization_config` is itself idempotent (it logs and overwrites
rather than raising), so a bare call is safe, but the guard keeps the import of
the config module off the hot path on repeat calls.
"""

from __future__ import annotations

from functools import wraps

_REGISTERED = False


def register() -> None:
    """Register the out-of-tree EXL3 quantization backend."""
    global _REGISTERED
    if _REGISTERED:
        return

    from vllm.model_executor.layers.quantization import register_quantization_config

    from .quantization.config import EXL3Config

    register_quantization_config("exl3")(EXL3Config)
    _patch_moe_trellis_rank()
    _REGISTERED = True


def _patch_moe_trellis_rank() -> None:
    """Stop vLLM reading an EXL3 trellis as a packed multi-expert tensor.

    `RoutedExperts.load_weights` decides whether a checkpoint tensor holds all
    experts at once with `is_fused = loaded_weight.dim() == 3`. That is a fine
    heuristic for formats whose per-expert weight is a 2D matrix, but an EXL3
    trellis is *natively* 3D -- `(k/16, n/16, 16*K)` -- so every per-expert
    tensor trips it and gets `chunk(2)`'d by expert id, which raises IndexError
    on any model with more than two experts and would silently corrupt one with
    two.

    There is no hook to opt out: the rank test is the whole decision. So the
    tensors are lifted to 4D on the way in, which the heuristic ignores, and
    `_moe_weight_loader` drops the added axis again. Guarded on the `.trellis`
    suffix, which no other quantization format uses.

    This is the first monkeypatch in the plugin, and it is worth noting that it
    is a *format* collision rather than a container-format problem of the kind
    GGUF needs patches for. If vLLM ever keys `is_fused` on something better
    than tensor rank, this can go.
    """
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    if getattr(RoutedExperts, "_exl3_trellis_rank_patched", False):
        return

    original = RoutedExperts.load_weights

    def _lift(weights):
        for name, weight in weights:
            if name.endswith(".trellis") and weight.dim() == 3:
                weight = weight.unsqueeze(0)
            yield name, weight

    @wraps(original)
    def load_weights(self, weights):
        return original(self, _lift(weights))

    RoutedExperts.load_weights = load_weights
    RoutedExperts._exl3_trellis_rank_patched = True
