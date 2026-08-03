"""Plugin entry point.

vLLM loads `vllm.general_plugins` entry points once per process -- the frontend,
the engine core, and every worker -- so `register()` must be idempotent.
`register_quantization_config` is itself idempotent (it logs and overwrites
rather than raising), so a bare call is safe, but the guard keeps the import of
the config module off the hot path on repeat calls.
"""

from __future__ import annotations

_REGISTERED = False


def register() -> None:
    """Register the out-of-tree EXL3 quantization backend."""
    global _REGISTERED
    if _REGISTERED:
        return

    from vllm.model_executor.layers.quantization import register_quantization_config

    from .quantization.config import EXL3Config

    register_quantization_config("exl3")(EXL3Config)
    _REGISTERED = True
