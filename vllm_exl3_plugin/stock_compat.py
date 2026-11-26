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
    try:
        from vllm.model_executor.models.transformers.causal import CausalMixin
    except ImportError:   # a vLLM without the Transformers backend's causal mixin
        CausalMixin = None
    if CausalMixin is not None:
        _transformers_logit_multiplier(CausalMixin)
    try:   # the model maker's out-of-tree plugin (vllm-spark2-5-plugin), when installed
        from vllm_spark2_5_plugin.spark2_5 import Spark2_5ForCausalLM
    except Exception:
        Spark2_5ForCausalLM = None
    if Spark2_5ForCausalLM is not None:
        _spark_exl3_names(Spark2_5ForCausalLM)
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    if hasattr(LogitsProcessor, "_apply_head"):   # vLLM after 0.34: head_dtype honoured in one place
        _exl3_head_dtype(LogitsProcessor)
    try:   # HY4 Flash (upstream main after 0.34)
        from vllm.models.hy_v4_flash.nvidia.model import HYV4FlashModel
    except Exception:
        HYV4FlashModel = None
    if HYV4FlashModel is not None:
        _hy4_flash_engram_names(HYV4FlashModel)
        from vllm.models.hy_v4_flash.nvidia import engram as hy4_engram_module

        from . import hy4_engram

        hy4_engram.patch(hy4_engram_module)


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
                _validate_shard_id(self, loaded_shard_id, param)
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
                _validate_shard_id(self, loaded_shard_id, param)
                param.load_qkv_weight(loaded_weight=loaded_weight, shard_id=loaded_shard_id)
                return
            return qkv(self, param, loaded_weight, loaded_shard_id)

        qkv_loader._exl3_patched = True
        QKVParallelLinear.weight_loader_v2 = qkv_loader


def _validate_shard_id(layer, shard_id, param) -> None:
    """vLLM's own shard-id check, as the running version spells it: v0.34 passes the parameter too, v0.32 does not."""
    validate = getattr(layer, "validate_shard_id", None)
    if validate is None:
        return
    if len(inspect.signature(validate).parameters) >= 2:
        validate(shard_id, param)
    else:
        validate(shard_id)


def _replicated_linear_loading(ReplicatedLinear) -> None:
    """An EXL3 parameter in a `ReplicatedLinear` loads through its own row loader, whole.

    `ReplicatedLinear` is the only linear layer with no `weight_loader_v2` branch: its v1 loader shape-checks and
    copies into preallocated data, which an EXL3 parameter does not have (it records what it is handed). Any quantized
    model reaching one -- the Transformers backend builds them -- failed to load. A replicated layer partitions
    nothing, so the tensor is stored whole on every rank.
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
            # Whole on every rank. Not through load_row_parallel_weight: the parameter takes tp_size from the TP group
            # whatever its layer does, so at TP=2 rank 1 sliced rows [k, 2k) of a k-row tensor and stored nothing
            # (HY4 Flash's q_a_proj); at TP=1 the two are the same call
            param.store(loaded_weight)
            return
        return original(self, param, loaded_weight)

    weight_loader._exl3_patched = True
    ReplicatedLinear.weight_loader = weight_loader


def _transformers_logit_multiplier(CausalMixin) -> None:
    """A model served through the Transformers backend that spells its logit multiplier `output_multiplier`
    (MuseGlimmer) gets it, applied before the soft cap.

    The backend reads only `logit_scale`, and its LogitsProcessor applies a scale after the cap, where such a model
    needs it before. Without this, every logprob and temperature-dependent choice is off, and greedy output changes
    too: the bench's Muse Transformers-backend entry on v0.34 flipped 28 of 106 argmaxes and its greedy continuation,
    bit-identical with this. The multiplier folds into the cap by the identity LogitsProcessor already expresses,
        tanh(z / (T/m)) * (T/m) * m  ==  T * tanh(z * m / T),
    which reduces to the backend's own behaviour when there is no multiplier. turboderp's Muse checkpoints quantize
    the vision adapter, which the native implementation builds as a plain nn.Linear, so this backend is the only way
    to serve them; only models under an EXL3 config are touched.
    """
    original = CausalMixin.__init__
    if getattr(original, "_exl3_patched", False):
        return

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        text_config = getattr(self, "text_config", None)
        processor = getattr(self, "logits_processor", None)
        multiplier = getattr(text_config, "output_multiplier", None)
        if processor is None or multiplier in (None, 1, 1.0) or getattr(text_config, "logit_scale", None) is not None:
            return
        from vllm.config import get_current_vllm_config

        try:
            ambient = get_current_vllm_config().quant_config
        except Exception:
            ambient = None
        if not _is_exl3(ambient):
            return
        from vllm.model_executor.layers.logits_processor import LogitsProcessor

        cap = getattr(text_config, "final_logit_softcapping", None)
        self.logits_processor = LogitsProcessor(
            text_config.vocab_size,
            scale=float(multiplier),
            soft_cap=None if cap is None else float(cap) / float(multiplier),
        )

    __init__._exl3_patched = True
    CausalMixin.__init__ = __init__


def _spark_exl3_names(Spark2_5ForCausalLM) -> None:
    """Spark-X2.6 (`Spark2_5ForCausalLM`, served by its maker's plugin) loads an EXL3 checkpoint under its names.

    exllamav3 stores the fused `q_k_v_proj` as the split `q_proj`, `k_proj`, `v_proj` it quantized
    (`exllamav3/architecture/spark2_5.py`); the plugin's loader maps only `gate_proj`/`up_proj` into `gate_up_proj`, so
    q/k/v found no parameter. Under an EXL3 config the instance's mapper also maps them, by shard id, into the plugin's
    own `QKVParallelLinear`; the class mapper, and so the reference checkpoint, is untouched. exllamav3's other rename,
    `model.embedding` -> `model.embed_tokens`, the EXL3 config undoes for any model (`_embedding_leaf_rename`). The head needs nothing: exllamav3 emits it untied
    (`tie_word_embeddings: false`), so the plugin builds a `ParallelLMHead` under the EXL3 config.
    """
    original = Spark2_5ForCausalLM.__init__
    if getattr(original, "_exl3_patched", False):
        return

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        vllm_config = kwargs.get("vllm_config")
        if vllm_config is None or not _is_exl3(getattr(vllm_config, "quant_config", None)):
            return
        from vllm.model_executor.models.utils import WeightsMapper

        self.hf_to_vllm_mapper = type(self).hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_stacked={".q_proj.": (".q_k_v_proj.", "q"), ".k_proj.": (".q_k_v_proj.", "k"),
                                 ".v_proj.": (".q_k_v_proj.", "v")},
        )

    __init__._exl3_patched = True
    Spark2_5ForCausalLM.__init__ = __init__



#: exllamav3's Engram tensors that stack what HY4 Flash's checkpoint keeps per hyper-connection stream or hash head
#: (`exllamav3/modules/ple.py`: alt keys concatenated, `branch_scales` stacked, all in index order), and the
#: per-part name each row block goes back to. Verified against tencent/Hy4-flash on layer 1: every block equals its
#: source tensor after the bf16 -> fp16 cast exllamav3 stores it in.
_HY4_ENGRAM_STACKED = {
    "branch_scales": "unified_per_hc_scales.{i}",
    "key_proj.weight": "key_projs.{i}.weight",
    "key_proj.bias": "key_projs.{i}.bias",
    "norm_key.weight": "norm_key.{i}.weight",
    "norm_query.weight": "norm_query.{i}.weight",
    "head_norm.weight": "lookup_per_head_norms.{i}.weight",
}


def _hy4_engram_unstack(weights, hc_mult: int):
    """The weight stream with each stacked Engram tensor replaced by its parts, under their checkpoint names."""
    for name, value in weights:
        head, sep, leaf = name.partition("engram_layer.")
        part = _HY4_ENGRAM_STACKED.get(leaf) if sep else None
        if part is None:
            yield name, value
            continue
        # head_norm stacks one row per hash head; the rest one row block per stream
        n = value.shape[0] if leaf == "head_norm.weight" else hc_mult
        if value.shape[0] % n:
            raise ValueError(f"EXL3 Engram tensor {name} of shape {tuple(value.shape)} does not split into {n}")
        for i, block in enumerate(value.reshape(n, value.shape[0] // n, *value.shape[1:]).unbind(0)):
            if leaf in ("branch_scales", "norm_key.weight", "norm_query.weight", "head_norm.weight"):
                block = block.reshape(value.shape[1:])
            yield f"{head}engram_layer.{part.format(i=i)}", block


_HY4_EXL3_EXPERT = None


def _hy4_exl3_experts(model, weights, loaded: set):
    """Load EXL3 routed-expert tensors the way HY4 Flash loads its own, and pass every other tensor on.

    HY4 Flash's expert branch matches only unquantized and ModelOpt suffixes (`.weight`, `.weight_packed`, `_scale`,
    `.input_scale`) and builds the target as `weight{suffix}`; an EXL3 expert tensor (`.trellis`, `.suh`, `.svh`,
    `.mcg`, `.mul1`) fell through to "Unknown Flash checkpoint weight". Same mapping, same `weight_loader` call and the
    same part record as that branch, with the EXL3 leaf as the parameter suffix (`EXL3MoEMethod` registers
    `w13_<leaf>` / `w2_<leaf>`)."""
    import re

    global _HY4_EXL3_EXPERT
    if _HY4_EXL3_EXPERT is None:
        _HY4_EXL3_EXPERT = re.compile(r"(.*)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(trellis|suh|svh|mcg|mul1)")
    mapping = {w: (p, e, s) for p, w, e, s in model.get_expert_mapping()}
    parameters = dict(model.named_parameters())
    for name, value in weights:
        m = _HY4_EXL3_EXPERT.fullmatch(name)
        if m is None:
            yield name, value
            continue
        prefix, expert_id, projection, leaf = m.groups()
        param_name, mapped_expert, shard = mapping[f"experts.{expert_id}.{projection}."]
        target = f"{prefix}.{param_name}{leaf}"
        if target not in parameters:
            raise ValueError(f"EXL3 expert tensor {name} has no parameter {target}")
        parameter = parameters[target]
        if parameter.weight_loader(parameter, value, target, shard_id=shard, expert_id=mapped_expert,
                                   return_success=True):
            model._record_part(target, (mapped_expert, shard))
            loaded.add(target)


def _hy4_flash_engram_names(HYV4FlashModel) -> None:
    """HY4 Flash loads an EXL3 checkpoint: exllamav3's Engram tensors and its routed experts.

    vLLM's `FlashEngram.load_weight` accepts the checkpoint's per-stream and per-head tensors and records each part
    it fills; exllamav3 stores them stacked (`_HY4_ENGRAM_STACKED`), which it refuses (`branch_scales`). Splitting
    happens in the weight stream, ahead of the model's own loop, so every part is recorded as from a reference
    checkpoint. Routed-expert tensors go through `_hy4_exl3_experts`. The n-gram table itself
    (`multi_head_embedding`, trellis-coded in an EXL3 checkpoint) is not this.
    """
    original_init = HYV4FlashModel.__init__
    if getattr(original_init, "_exl3_patched", False):
        return
    original_load = HYV4FlashModel.load_weights

    @functools.wraps(original_init)
    def __init__(self, *args, vllm_config=None, **kwargs):
        original_init(self, *args, vllm_config=vllm_config, **kwargs)
        self._exl3_engram_hc = (
            vllm_config.model_config.hf_text_config.hc_mult
            if vllm_config is not None and _is_exl3(getattr(vllm_config, "quant_config", None)) else None
        )
        if self._exl3_engram_hc:
            from . import hy4_engram

            hy4_engram.attach(self, vllm_config)

    @functools.wraps(original_load)
    def load_weights(self, weights):
        hc = getattr(self, "_exl3_engram_hc", None)
        if not hc:
            return original_load(self, weights)
        loaded: set[str] = set()
        # The tables stay on disk (hy4_engram): their parts are filled by staging, not by a checkpoint tensor.
        # Once per model: vLLM calls load_weights again for later shards, and a second record is a duplicate
        if not getattr(self, "_exl3_tables_recorded", False):
            for table in getattr(getattr(self, "engram_hash", None), "exl3_tables", None) or ():
                name = f"layers.{table.layer_id - 1}.engram_layer.multi_head_embedding.embedding.weight"
                self._record_part(name, None)
                loaded.add(name)
            self._exl3_tables_recorded = True
        rest = _hy4_exl3_experts(self, _hy4_engram_unstack(weights, hc), loaded)
        return original_load(self, rest) | loaded

    __init__._exl3_patched = True
    HYV4FlashModel.__init__ = __init__
    HYV4FlashModel.load_weights = load_weights


def _exl3_head_dtype(LogitsProcessor) -> None:
    """An EXL3 head honours a `head_dtype` other than the activations' with exllamav3's fp16 logits, cast late.

    vLLM refuses a head_dtype different from the model dtype for any quantized head, and HY4 Flash's config sets one
    (`enable_lm_head_fp32` surfaces as head_dtype float32). The EXL3 kernel accumulates in fp32 and writes fp16, and
    exllamav3 serves those (its HY4 Flash head is an fp16 `Linear` fed fp16). Fed the bf16 activations, `exl3_mm` casts
    its output to bf16 and three mantissa bits go before any cast to fp32, so the head is fed fp16 here. Through
    `_get_logits` the logits also cross the TP gather in fp16 and are cast once, after it: gathered in fp32 they were
    2 x 472 MiB of warmup's reservation peak on HY4 Flash at TP=2 (2026-11-26). Every other head goes to vLLM's own
    path, check included; `_apply_head`'s other callers still get head_dtype.
    """
    original_apply = LogitsProcessor._apply_head
    if getattr(original_apply, "_exl3_patched", False):
        return
    original_get = LogitsProcessor._get_logits
    from .quantization.lm_head import EXL3LMHeadMethod

    def exl3_cast(self, lm_head, hidden_states):
        head_dtype = getattr(self, "head_dtype", None)
        if (isinstance(getattr(lm_head, "quant_method", None), EXL3LMHeadMethod)
                and head_dtype is not None and head_dtype != hidden_states.dtype):
            return head_dtype
        return None

    def fp16_logits(lm_head, hidden_states, embedding_bias):
        bias = embedding_bias.half() if embedding_bias is not None else None
        return lm_head.quant_method.apply(lm_head, hidden_states.half(), bias=bias)

    @functools.wraps(original_apply)
    def _apply_head(self, lm_head, hidden_states, embedding_bias):
        head_dtype = exl3_cast(self, lm_head, hidden_states)
        if head_dtype is None:
            return original_apply(self, lm_head, hidden_states, embedding_bias)
        return fp16_logits(lm_head, hidden_states, embedding_bias).to(head_dtype)

    @functools.wraps(original_get)
    def _get_logits(self, hidden_states, lm_head, embedding_bias, skip_gather=False):
        head_dtype = exl3_cast(self, lm_head, hidden_states)
        if head_dtype is None:
            return original_get(self, hidden_states, lm_head, embedding_bias, skip_gather)
        logits = fp16_logits(lm_head, hidden_states, embedding_bias)
        if skip_gather:
            return logits.to(head_dtype)
        if lm_head.tp_size > 1:
            logits = self._gather_logits(logits)
        if logits is None:
            return None
        return logits[..., : self.org_vocab_size].to(head_dtype)

    _apply_head._exl3_patched = True
    LogitsProcessor._apply_head = _apply_head
    LogitsProcessor._get_logits = _get_logits
