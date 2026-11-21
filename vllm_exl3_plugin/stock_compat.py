"""The vLLM fixes the plugin cannot work without, applied as monkeypatches.

Patch policy (docs/vllm-patches.md): what the plugin needs to load and serve correctly is *vital* and lives here, so
the plugin runs on a stock vLLM of a compatible version; everything else the fork carries (VRAM work, non-essential
fixes, behaviour improvements) is *optimization* and stays in the fork. Each patch here is scoped to EXL3 -- an EXL3
parameter, or a model built under an EXL3 config -- so it leaves every other quantization method and model exactly as
stock vLLM has them, and is harmless where vLLM already does the right thing. All are idempotent: vLLM loads general
plugins once per process, but `register()` may be reached more than once.
"""

from __future__ import annotations

import functools
import inspect

import torch


def apply() -> None:
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, ReplicatedLinear
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

    _embedding_quant_config_default(VocabParallelEmbedding)
    _fused_shard_loading(MergedColumnParallelLinear, QKVParallelLinear)
    _replicated_linear_loading(ReplicatedLinear)


def _is_exl3(quant_config) -> bool:
    return quant_config is not None and getattr(quant_config, "get_name", lambda: None)() == "exl3"


def _embedding_quant_config_default(VocabParallelEmbedding) -> None:
    """An embedding built without a `quant_config` takes the EXL3 config it is built under.

    Most model files construct `VocabParallelEmbedding` without passing `quant_config`, so a quantized embedding is
    unreachable on them: a load error for a block-quantized table, and for a tied model a silently dense embedding
    beside a quantized head. `ParallelLMHead` goes through the same `__init__`. A model that preselects
    `quant_method`, or passes its own config, is left alone, as is any model not under an EXL3 config.
    """
    original = VocabParallelEmbedding.__init__
    if getattr(original, "_exl3_patched", False):
        return
    sig = inspect.signature(original)

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        if bound.arguments.get("quant_config") is None and bound.arguments.get("quant_method") is None:
            from vllm.config import get_current_vllm_config

            try:
                ambient = get_current_vllm_config().quant_config
            except Exception:
                ambient = None
            if _is_exl3(ambient):
                bound.arguments["quant_config"] = ambient
        return original(*bound.args, **bound.kwargs)

    __init__._exl3_patched = True
    VocabParallelEmbedding.__init__ = __init__


def _fused_shard_loading(MergedColumnParallelLinear, QKVParallelLinear) -> None:
    """A fused QKV or gate_up checkpoint tensor reaches an EXL3 parameter whole, with the shard id it covers.

    The merged and QKV loaders narrow a fused tensor along an output dimension, or hand it whole only to
    `type(param) in (RowvLLMParameter, BasevLLMParameter)` -- never to a subclass. An EXL3 tensor has no output
    dimension in consistent units (trellis tile-granular on dim 1, svh element-granular on dim 0, suh not splittable),
    so the parameter splits it itself (`EXL3Parameter.load_merged_column_weight` / `load_qkv_weight`). Qwen3.5
    checkpoints do not load without this.
    """
    from .quantization.linear import EXL3Parameter

    merged = MergedColumnParallelLinear.weight_loader_v2
    if not getattr(merged, "_exl3_patched", False):
        @functools.wraps(merged)
        def merged_loader(self, param, loaded_weight, loaded_shard_id=None):
            if isinstance(param, EXL3Parameter) and (loaded_shard_id is None or isinstance(loaded_shard_id, tuple)):
                self.validate_shard_id(loaded_shard_id, param)
                param.load_merged_column_weight(loaded_weight=loaded_weight, shard_id=loaded_shard_id)
                return
            return merged(self, param, loaded_weight, loaded_shard_id)

        merged_loader._exl3_patched = True
        MergedColumnParallelLinear.weight_loader_v2 = merged_loader

    qkv = QKVParallelLinear.weight_loader_v2
    if not getattr(qkv, "_exl3_patched", False):
        @functools.wraps(qkv)
        def qkv_loader(self, param, loaded_weight, loaded_shard_id=None):
            if isinstance(param, EXL3Parameter) and loaded_shard_id is None:
                self.validate_shard_id(loaded_shard_id, param)
                param.load_qkv_weight(loaded_weight=loaded_weight, shard_id=loaded_shard_id)
                return
            return qkv(self, param, loaded_weight, loaded_shard_id)

        qkv_loader._exl3_patched = True
        QKVParallelLinear.weight_loader_v2 = qkv_loader


def _replicated_linear_loading(ReplicatedLinear) -> None:
    """An EXL3 parameter in a `ReplicatedLinear` loads through its own row loader, whole.

    `ReplicatedLinear` is the only linear layer with no `weight_loader_v2` branch: its v1 loader shape-checks and
    copies into preallocated data, which an EXL3 parameter does not have (it records what it is handed). Any quantized
    model reaching one -- the Transformers backend builds them -- failed to load. A replicated layer partitions
    nothing, so the tensor goes to `load_row_parallel_weight` whole, where the EXL3 parameter applies its own policy.
    """
    from .quantization.linear import EXL3Parameter

    original = ReplicatedLinear.weight_loader
    if getattr(original, "_exl3_patched", False):
        return

    @functools.wraps(original)
    def weight_loader(self, param, loaded_weight: torch.Tensor):
        if isinstance(param, EXL3Parameter):
            if loaded_weight.dim() == 0:   # a scalar off disk, as AutoFP8 stores scales
                loaded_weight = loaded_weight.reshape(1)
            param.load_row_parallel_weight(loaded_weight=loaded_weight)
            return
        return original(self, param, loaded_weight)

    weight_loader._exl3_patched = True
    ReplicatedLinear.weight_loader = weight_loader
