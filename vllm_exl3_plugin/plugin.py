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
    _patch_offload_summary()
    _patch_offload_defaults()
    _patch_kernel_overlap()
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


def _patch_offload_summary() -> None:
    """Emit the CPU-offload summary at the end of weight loading.

    There is no hook for "every layer has been offered to the offloader".
    `wrap_modules` runs at model *construction*, and nothing in vLLM calls back
    after the `process_weights_after_loading` loop, which is where
    `cpu_offload.register_offload` does its work.

    The obvious place -- the first `apply()` -- is wrong, and expensively so:
    `apply()` runs inside vLLM's `torch.compile` region, and Dynamo refuses
    `logging.Logger` calls outright (`gb0291`, "logging.Logger method not
    supported for non-export cases"), taking the engine down at startup rather
    than degrading. Guarding on `torch.compiler.is_compiling()` would avoid the
    crash but never fire, since `apply()` is only ever reached through the
    compiled path unless `--enforce-eager` is set.

    So wrap the module-level `process_weights_after_loading` instead. It is
    called exactly once, from `BaseModelLoader.load_model`, after every module's
    own `process_weights_after_loading` has run -- and the call site resolves the
    name from module globals, so patching the attribute works whatever the import
    order. `TensorizerLoader` and `ModelExpressLoader` override `load_model` and
    will not report; they are not paths EXL3 takes, and the cost of missing it is
    one log line.
    """
    from vllm.model_executor.model_loader import base_loader

    if getattr(base_loader, "_exl3_offload_summary_patched", False):
        return

    original = base_loader.process_weights_after_loading

    @wraps(original)
    def process_weights_after_loading(*args, **kwargs):
        result = original(*args, **kwargs)
        from .cpu_offload import build_staging, report_once

        build_staging()
        report_once()
        return result

    base_loader.process_weights_after_loading = process_weights_after_loading
    base_loader._exl3_offload_summary_patched = True


def _patch_kernel_overlap() -> None:
    """Never run an EXL3 MoE layer's shared experts on vLLM's side stream.

    For batches up to ``VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`` (256, so every decode
    step) vLLM runs the shared experts on ``aux_stream()`` while the routed experts run on
    the main stream. exllamav3's MoE and GEMM kernels spin-wait across blocks and assume the
    GPU to themselves, so the two can deadlock: found 2026-11-08 as a decode hang on a 3060
    under WSL, rare per step and so only deep into long decodes, and confirmed by
    ``VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`` curing it. Per layer, so other quantizations
    keep their overlap; `ops._one_stream` refuses any overlap this does not cover.
    """
    try:
        from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
            SharedExperts,
            SharedExpertsOrder,
        )
    except ImportError:
        return
    if getattr(SharedExperts, "_exl3_overlap_patched", False):
        return
    original = SharedExperts._determine_shared_experts_order

    @wraps(original)
    def _determine_shared_experts_order(self, hidden_states):
        # `_layer` is the shared-expert MLP, not the MoE layer: refuse the overlap if it runs
        # EXL3 linears, which is what would launch our GEMM on the side stream. Decided once.
        exl3 = getattr(self, "_exl3_shared", None)
        if exl3 is None:
            from .quantization.linear import EXL3LinearMethod

            exl3 = any(isinstance(getattr(m, "quant_method", None), EXL3LinearMethod)
                       for m in self._layer.modules())
            self._exl3_shared = exl3
        if exl3:
            return SharedExpertsOrder.NO_OVERLAP
        return original(self, hidden_states)

    SharedExperts._determine_shared_experts_order = _determine_shared_experts_order
    SharedExperts._exl3_overlap_patched = True


def _patch_offload_defaults() -> None:
    """Give ``--cpu-offload-gb`` without ``--cpu-offload-params`` a safe default:
    multimodal towers and routed experts, nothing else.

    With no selector, vLLM's ``UVAOffloader`` offloads every parameter of every
    module ``wrap_modules`` is handed. That is called twice over: once per tower
    (``_mark_tower_model``, with the tower's name as `prefix`) and once for the
    decoder layers (``make_layers``, with no prefix). For an EXL3 model the
    decoder layers' plain tensors are what is left after the quantized ones --
    norms, router weights -- a few KiB each and read by every token, which over
    the link cost far more than their bytes (see `cpu_offload._OFFLOADABLE_COMPONENTS`
    for the same effect measured on scale vectors).

    So an empty selector lets tower calls through and skips the prefix-less
    decoder-layer call. Towers are built first in the models that have them, so
    they keep first claim on the budget; `cpu_offload` gives experts what is
    left, and leaves dense EXL3 tensors alone without a selector. Any explicit
    selector restores vLLM's behaviour unchanged.
    """
    from vllm.model_executor.offloader.uva import UVAOffloader

    if getattr(UVAOffloader, "_exl3_defaults_patched", False):
        return
    original = UVAOffloader.wrap_modules

    @wraps(original)
    def wrap_modules(self, modules_generator, prefix: str = ""):
        if not self.cpu_offload_params and not prefix:
            return list(modules_generator)
        return original(self, modules_generator, prefix)

    UVAOffloader.wrap_modules = wrap_modules
    UVAOffloader._exl3_defaults_patched = True
