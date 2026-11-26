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


#: RoPE caches below this go unlogged: ordinary contexts cost tens of MiB
_ROPE_LOG_MIN_BYTES = 64 << 20


def _log_rope_caches(model, model_config) -> None:
    """Say what vLLM's RoPE cos/sin caches cost, and for how many positions they were built.

    They are built at model init for the model's maximum position -- usually the config's, whatever --max-model-len
    says, and under --max-model-len auto before the length is settled -- so at a 1M native context they are memory
    nobody can account for from the flags: 640 MiB of Spark-X2.6-4B's 2.63 GiB, 1 GiB of HY4 Flash's (docs/upstream.md,
    "RoPE caches at the native length"). A line, not a fix: nothing here changes the caches."""
    try:
        import torch

        from .log import init_logger

        logger = init_logger(__name__)
        seen, rows, total = set(), [], 0
        for module in model.modules():
            cache = getattr(module, "cos_sin_cache", None)
            if not isinstance(cache, torch.Tensor) or not cache.is_cuda or cache.data_ptr() in seen:
                continue
            seen.add(cache.data_ptr())
            nbytes = cache.numel() * cache.element_size()
            total += nbytes
            rows.append((cache.shape[0], nbytes))
        if total < _ROPE_LOG_MIN_BYTES:
            return
        auto = getattr(model_config, "original_max_model_len", None) == -1
        length = "auto: the final length is set after warmup" if auto else f"--max-model-len {model_config.max_model_len}"
        logger.info("RoPE cos/sin caches: %.0f MiB in %d tensor(s), built for %s positions (%s)", total / (1 << 20),
                    len(rows), "/".join(sorted({f"{r:,}" for r, _ in rows})), length)
    except Exception:   # a diagnostic: never fail a load over it
        pass


def register() -> None:
    """Register the out-of-tree EXL3 quantization backend."""
    global _REGISTERED
    if _REGISTERED:
        return

    _leave_allocator_alone()
    _disk_tier_graph_defaults()
    from vllm.model_executor.layers.quantization import register_quantization_config

    from .quantization.config import EXL3Config

    register_quantization_config("exl3")(EXL3Config)
    # the vital fixes, so a stock vLLM serves EXL3 too (docs/vllm-patches.md)
    from . import stock_compat

    stock_compat.apply()
    _patch_moe_trellis_rank()
    _patch_offload_summary()
    _patch_offload_defaults()
    _patch_kernel_overlap()
    _patch_allocator_check()
    _patch_mm_encoder_dp_default()
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
        from .quantization.linear import check_uneven_mlps

        # Before anything serves: an uneven MLP split is exact only for a channel-pairwise activation
        model = args[0] if args else kwargs.get("model")
        if hasattr(model, "named_modules"):
            check_uneven_mlps(model)
        build_staging()
        if _serving_exl3():
            from .cpu_offload import decide_throughput_offload

            decide_throughput_offload()
            from .allocator_safety import check

            check("after loading the model")
            from .wddm_budget import start_monitor

            start_monitor()
        report_once()
        return result

    base_loader.process_weights_after_loading = process_weights_after_loading
    base_loader._exl3_offload_summary_patched = True

    # A speculative drafter is a second load into the same process. vLLM builds it from its own
    # ModelConfig with runner type "draft" (an MTP head included), so that is what marks it.
    load_model = base_loader.BaseModelLoader.load_model

    @wraps(load_model)
    def load_model_marked(self, vllm_config, model_config, *args, **kwargs):
        from .cpu_offload import drafting

        import os

        import torch

        # EXL3_MEM_SNAPSHOT: the load too (`<base>.load.rank<N>`), so buffers built at model init show by site
        snap = os.environ.get("EXL3_MEM_SNAPSHOT")
        if snap:
            torch.cuda.memory._record_memory_history(max_entries=2_000_000, stacks="python")
        try:
            with drafting(getattr(model_config, "runner_type", None) == "draft"):
                model = load_model(self, vllm_config, model_config, *args, **kwargs)
            _log_rope_caches(model, model_config)
            return model
        finally:
            if snap:
                from .expert_cache import _dump_snapshot

                rank = getattr(vllm_config.parallel_config, "rank", 0)
                _dump_snapshot(f"{snap}.load.rank{rank}", "model load")

    base_loader.BaseModelLoader.load_model = load_model_marked


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


def _serving_exl3() -> bool:
    """Whether the model being built is EXL3. The plugin is installed for every model in the
    environment, and an FP8 model offloaded with no selector must get vLLM's own behaviour: the
    default below once left the Qwen3.6-35B-A3B FP8 release with nothing offloaded at any budget."""
    try:
        from vllm.config import get_current_vllm_config

        quant = get_current_vllm_config().quant_config
    except Exception:
        return True   # cannot tell: keep the plugin's default, which is what EXL3 needs
    return quant is not None and quant.get_name() == "exl3"


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
        if not self.cpu_offload_params and not prefix and _serving_exl3():
            return list(modules_generator)
        return original(self, modules_generator, prefix)

    UVAOffloader.wrap_modules = wrap_modules
    UVAOffloader._exl3_defaults_patched = True


def _wants_encoder_dp(model_config, parallel_config) -> bool:
    """Whether an EXL3 multimodal model should replicate its vision encoder instead of sharding it.

    Vision towers have widths no Hadamard-block split respects: Qwen4Exp's 16 heads of 72 put 576 channels on each
    of two ranks, and its 4304-wide MLP is not a whole number of blocks. Sharded, they fail at load (format.check_tp_split);
    replicated (`mm_encoder_tp_mode="data"`), each rank runs the tower exactly as TP=1 does, for ~0.1 GiB of EXL3
    weights per rank. Only for vLLM's default mode: "weights" is indistinguishable from unset, so this is the one
    choice a user cannot pin, and the sharded mode it replaces cannot load these towers anyway.
    """
    mm = getattr(model_config, "multimodal_config", None)
    info = getattr(model_config, "_model_info", None)
    return (getattr(model_config, "quantization", None) == "exl3" and mm is not None
            and not getattr(mm, "language_model_only", False) and mm.mm_encoder_tp_mode == "weights"
            and getattr(parallel_config, "tensor_parallel_size", 1) > 1
            and bool(getattr(info, "supports_multimodal_encoder_tp_data", False)))


def _patch_mm_encoder_dp_default() -> None:
    """Replicate an EXL3 model's vision encoder across tensor-parallel ranks (`_wants_encoder_dp`)."""
    from vllm.config import VllmConfig

    if getattr(VllmConfig, "_exl3_encoder_dp_patched", False):
        return
    original = VllmConfig.__post_init__

    @wraps(original)
    def __post_init__(self, *args, **kwargs):
        mc = getattr(self, "model_config", None)
        if mc is not None and _wants_encoder_dp(mc, getattr(self, "parallel_config", None)):
            from .log import init_logger

            mc.multimodal_config.mm_encoder_tp_mode = "data"
            init_logger(__name__).info(
                "EXL3: vision encoder replicated on each rank (mm_encoder_tp_mode=data); its widths do not split "
                "on Hadamard blocks")
        return original(self, *args, **kwargs)

    VllmConfig.__post_init__ = __post_init__
    VllmConfig._exl3_encoder_dp_patched = True


def _disk_tier_graph_defaults() -> None:
    """When experts will spill to disk -- or are offloaded under ``--performance-mode throughput``, where decode batches
    over 4 tokens run grouped (`cpu_offload.decide_throughput_offload`) -- CUDA graphs that leave the routed-expert op
    eager: breakable capture with the
    op as an eager break (`ops.moe_eager_break`, on once anything spilled) and PIECEWISE decode graphs, since a FULL
    decode graph captures across the breaks. The disk tier reads routing back on the host and refuses capture. Whether
    anything spills is decided at load, so this predicts it from the config (`cpu_offload.spill_expected`); load
    spills only into a graph path that can run it, and otherwise keeps the rest resident. Each is a default: an
    explicit VLLM_USE_BREAKABLE_CUDAGRAPH or cudagraph_mode stands. QFN 2.05bpw, 50% on disk: 67.5-77.5 ms a token,
    against 110-128 eager (docs/expert-cache.md, "QFN under breakable capture")."""
    from vllm.config import VllmConfig
    from vllm.config.compilation import CUDAGraphMode

    if getattr(VllmConfig, "_exl3_disk_tier_graphs_patched", False):
        return
    original = VllmConfig.__post_init__

    @wraps(original)
    def __post_init__(self, *args, **kwargs):
        mc, cc = getattr(self, "model_config", None), getattr(self, "compilation_config", None)
        if mc is not None and cc is not None and getattr(mc, "quantization", None) == "exl3" \
                and not mc.enforce_eager:
            from .cpu_offload import spill_expected, throughput_offload_expected

            if spill_expected(self) or throughput_offload_expected(self):
                import os

                os.environ.setdefault("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
                if cc.cudagraph_mode is None:
                    cc.cudagraph_mode = CUDAGraphMode.PIECEWISE
        return original(self, *args, **kwargs)

    VllmConfig.__post_init__ = __post_init__
    VllmConfig._exl3_disk_tier_graphs_patched = True


def _leave_allocator_alone() -> None:
    """Keep torch's allocator as vLLM and the user configured it.

    Importing the `exllamav3` package turns on expandable segments unless told not to (`exllamav3/__init__.py`), and
    the plugin imports it lazily -- the MoE and engram paths -- so under vLLM a MoE model switched allocators partway
    through the process and a dense one never did. Expandable segments are not safe everywhere: with torch's
    unpatched libc10_cuda.so (PyTorch #196258, open through 2.14.0) freed pages are unmapped across GPUs, so at TP > 1
    another GPU's queued kernels write into memory that is gone or reused. So the plugin leaves the setting alone; a
    user who wants them sets PYTORCH_CUDA_ALLOC_CONF, which exllamav3 also defers to.
    """
    import os

    os.environ.setdefault("EXL3_EXPANDABLE_SEGMENTS", "0")


def _patch_allocator_check() -> None:
    """Re-check expandable-segment safety once warmup and graph capture have allocated (`allocator_safety`), for an
    EXL3 model; anything else gets stock vLLM's behaviour, as every plugin patch leaves it."""
    try:
        from vllm.v1.worker import gpu_worker
    except ImportError:
        return
    cls = gpu_worker.Worker
    if getattr(cls, "_exl3_allocator_check", False):
        return
    original = cls.compile_or_warm_up_model

    @wraps(original)
    def compile_or_warm_up_model(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        quant = getattr(getattr(self, "vllm_config", None), "quant_config", None)
        if quant is not None and quant.get_name() == "exl3":
            from .allocator_safety import check

            check("after warmup")
        return result

    cls.compile_or_warm_up_model = compile_or_warm_up_model
    cls._exl3_allocator_check = True


