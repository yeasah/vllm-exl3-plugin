"""`QuantizationConfig` for EXL3 checkpoints.

An EXL3 model is an ordinary HF repo: standard safetensors, standard
`config.json`, tensor keys matching the original model 1:1, and a
`quantization_config` block whose `quant_method` is `"exl3"`. vLLM's normal
config detection therefore finds us with no loader or config-parser patching --
this is the part of the GGUF plugin that EXL3 simply does not need.

Models quantized since ~v0.0.2 additionally ship `quantization_config.json`,
which carries a per-module `tensor_storage` map (shapes, dtypes, bit width,
codebook). It is not part of the HF config, so vLLM does not read it; we fetch
it ourselves in `maybe_update_config` and use it to tell quantized modules from
untouched ones. Older checkpoints lack it, and there we fall back to assuming
every linear in the model is quantized -- which is what exllamav3's converter
actually does.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

from vllm.logger import init_logger

from ..format import EXL3FormatError

logger = init_logger(__name__)

#: Extra per-tensor codebook selector shipped by each codebook variant. The
#: kernels only read whether the tensor is present, not its value -- the
#: multiplier constants are compiled in (`ext.reconstruct(..., bool mcg, bool
#: mul1)`).
_CODEBOOK_TENSORS = {
    "mul1": "mul1",
    "mcg": "mcg",
    "3inst": None,
}


class EXL3Config(QuantizationConfig):
    """Config class for EXL3."""

    def __init__(
        self,
        bits: float | None = None,
        head_bits: int | None = None,
        codebook: str | None = None,
        version: str | None = None,
        out_scales: str | None = None,
    ) -> None:
        super().__init__()
        self.bits = bits
        self.head_bits = head_bits
        self.codebook = codebook or "3inst"
        self.version = version
        self.out_scales = out_scales
        # Populated by maybe_update_config when quantization_config.json exists.
        self.tensor_storage: dict[str, Any] | None = None
        self._ancestors: set[str] | None = None
        # Also from maybe_update_config, which is the only hook that sees
        # hf_config. None means "not yet known".
        self.tie_word_embeddings: bool | None = None
        # Phase 0's dequantize-at-load path, kept as a correctness oracle: it
        # is a transcription of exllamav3's own dequantization, so serving the
        # same prompts both ways isolates a kernel bug from a plumbing bug.
        # Costs the entire memory saving, so it is opt-in.
        self.dequantize = os.environ.get("VLLM_EXL3_DEQUANTIZE", "0") == "1"
        if self.dequantize:
            self._disable_stale_compile_cache()

        if self.codebook not in _CODEBOOK_TENSORS:
            raise EXL3FormatError(
                f"unknown EXL3 codebook {self.codebook!r}; this plugin knows "
                f"{sorted(_CODEBOOK_TENSORS)}"
            )

    def __repr__(self) -> str:
        return (
            f"EXL3Config(bits={self.bits}, head_bits={self.head_bits}, "
            f"codebook={self.codebook!r}, version={self.version!r})"
        )

    @staticmethod
    def _disable_stale_compile_cache() -> None:
        """Stop vLLM reusing a compiled graph traced from the *other* path.

        vLLM's compile-cache key is built from its own config objects and its
        own version. It cannot see out-of-tree plugin code, so switching
        `VLLM_EXL3_DEQUANTIZE` -- which changes what `apply()` traces to, one
        `F.linear` versus one `exl3_mm` per shard -- reuses the previously
        compiled artifact and dies with a bare `KeyError` on a parameter name
        deep inside an AOT-compiled graph.

        The same hazard applies to *any* edit to this plugin that changes the
        traced structure, so `VLLM_DISABLE_COMPILE_CACHE=1` is worth setting
        while developing on it. Here we only force it for the debug flag, where
        the mismatch is guaranteed rather than merely possible.
        """
        import os as _os

        if _os.environ.get("VLLM_DISABLE_COMPILE_CACHE"):
            return
        _os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
        logger.warning(
            "VLLM_EXL3_DEQUANTIZE is set, so VLLM_DISABLE_COMPILE_CACHE=1 has "
            "been forced: vLLM's compile cache does not key on plugin code, "
            "and would otherwise reuse a graph traced from the fused path."
        )

    def get_name(self) -> QuantizationMethods:
        return "exl3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # exllamav3's kernels are fp16 (exl3_gemm hard-checks A for kHalf), but
        # the model around them does not have to be: `exl3_mm` casts at the
        # kernel boundary, so the residual stream keeps vLLM's chosen dtype.
        #
        # bfloat16 is not a convenience here. The Gemma family is numerically
        # unstable end-to-end in fp16 -- vLLM refuses fp16 for gemma2/gemma3
        # outright, and exllamav3 carries fp32 residuals through gemma4 for the
        # same reason. Running those models at all requires a wider residual.
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # exllamav3's kernels are inline-PTX mma.sync + cp.async; sm_80 floor,
        # and there is no ROCm support at all.
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        # config.json always carries quantization_config, so vLLM never needs a
        # fallback file. quantization_config.json is fetched separately in
        # maybe_update_config because it is supplementary, not authoritative.
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> EXL3Config:
        return cls(
            bits=config.get("bits"),
            head_bits=config.get("head_bits"),
            codebook=config.get("codebook"),
            version=config.get("version"),
            out_scales=config.get("out_scales"),
        )

    def stored_tensor_names(self) -> tuple[str, ...]:
        """Tensor suffixes every EXL3 linear in this checkpoint carries.

        Registering a parameter vLLM will never find in the checkpoint is fatal
        (`default_loader` rejects any unloaded parameter), so this list has to be
        exactly right -- hence keying it off the codebook rather than
        registering the optional tensors speculatively.
        """
        names = ["trellis", "suh", "svh"]
        codebook_tensor = _CODEBOOK_TENSORS[self.codebook]
        if codebook_tensor is not None:
            names.append(codebook_tensor)
        return tuple(names)

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: Any = None,
        revision: str | None = None,
    ) -> None:
        if revision is None and hf_config is not None:
            # vLLM does not pass `revision` here (there is a TODO about it on
            # the base class), and defaulting to "main" is actively wrong: EXL3
            # repos publish one branch per bit rate, and `main` frequently has
            # no quantization_config.json at all. transformers records the
            # commit it actually resolved the config from, which is exactly the
            # revision being served.
            revision = getattr(hf_config, "_commit_hash", None)
        self._load_tensor_storage(model_name, revision)
        if hf_config is not None:
            self.tie_word_embeddings = bool(
                getattr(hf_config, "tie_word_embeddings", False)
            )

    def _load_tensor_storage(self, model_name: str, revision: str | None) -> None:
        from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

        try:
            extra = get_hf_file_to_dict(
                "quantization_config.json", model_name, revision or "main"
            )
        except Exception:
            extra = None
        if extra:
            self.tensor_storage = extra.get("tensor_storage")
            self._ancestors = None
            self.codebook = extra.get("codebook", self.codebook)
        else:
            # The fallback is "assume every linear is quantized", which holds
            # for the text-only checkpoints that predate this file but is wrong
            # for anything with an unquantized vision or audio tower. Worth
            # saying out loud rather than failing later on a missing parameter.
            logger.warning(
                "No quantization_config.json for %s@%s; assuming every linear "
                "layer is EXL3-quantized. This is correct for older text-only "
                "checkpoints but will fail on a model with unquantized towers.",
                model_name,
                revision or "main",
            )

    def head_is_quantized(self) -> bool:
        """Whether `lm_head` needs the EXL3 method rather than vLLM's default.

        Getting this wrong in either direction is fatal, not degraded:
        registering parameters the checkpoint never fills makes
        `default_loader` reject the model, and failing to register them leaves
        `lm_head.trellis` unclaimed.

        Tied embeddings settle it immediately. vLLM still *constructs* a
        `ParallelLMHead` for a tied model and only then ties it, but it skips
        every `lm_head.*` weight, so any parameter registered here would never
        be loaded.
        """
        if self.tie_word_embeddings:
            return False
        if self.tensor_storage is not None:
            head = self.tensor_storage.get("lm_head")
            return bool(head) and head.get("quant_format") == "exl3"
        # No storage map (pre-v0.0.2-era repos): `head_bits` in config.json's
        # quantization_config is the remaining signal. Checkpoints that quantize
        # the head always set it; the ones that do not, omit it entirely.
        return self.head_bits is not None

    def apply_vllm_mapper(self, hf_to_vllm_mapper) -> None:
        """Translate `tensor_storage` keys into vLLM's module naming.

        `tensor_storage` is keyed by checkpoint names, but `get_quant_method`
        is handed vLLM's module prefixes, and multimodal models restructure
        heavily between the two (Gemma 4 moves `model.language_model.layers.N`
        under a different parent entirely). Without this, every language-model
        layer looks unquantized and vLLM allocates dense fp16 weights for the
        whole model -- which on a 12B checkpoint means an out-of-memory error
        rather than anything that points at the real cause.

        vLLM calls this from `SupportsQuant.__new__`, before any layer is
        constructed, and passes the *unstacked* mapper so that constituent
        names like `q_proj` survive instead of being folded into `qkv_proj`.
        """
        if self.tensor_storage is not None:
            self.tensor_storage = hf_to_vllm_mapper.apply_dict(self.tensor_storage)
            self._ancestors = None

    def _quantized_ancestors(self) -> set[str]:
        """Every prefix that has at least one EXL3 module beneath it.

        A routed-experts module is named `...mlp.experts` while the checkpoint
        only ever mentions `...mlp.experts.{id}.gate_proj`, so an exact lookup
        finds nothing and the layer looks unquantized. Precomputed once because
        `tensor_storage` runs to ~30k entries on a 256-expert model.
        """
        if self._ancestors is None:
            ancestors: set[str] = set()
            for key, entry in (self.tensor_storage or {}).items():
                if entry.get("quant_format") != "exl3":
                    continue
                parts = key.split(".")
                for i in range(1, len(parts)):
                    ancestors.add(".".join(parts[:i]))
            self._ancestors = ancestors
        return self._ancestors

    @staticmethod
    def _drop_one_component(prefix: str):
        """Variants of `prefix` with any one interior component removed.

        Some models insert a wrapper module the checkpoint has no name for.
        Gemma 4 is the case in point: its MoE block lives at
        `...layers.N.moe.experts`, but the checkpoint says
        `...layers.N.experts.{id}.gate_proj` and the `.moe.` is spliced in by
        the model's own `_weight_iterator` -- which `apply_vllm_mapper` never
        sees, because it is not part of `hf_to_vllm_mapper`.

        Only used as a fallback, and only accepted when the result names a
        known quantized ancestor, so a spurious match is not possible.
        """
        parts = prefix.split(".")
        for i in range(len(parts) - 1):
            yield ".".join(parts[:i] + parts[i + 1 :])

    def is_quantized(self, prefix: str) -> bool:
        """Whether the module at `prefix` has EXL3 storage in the checkpoint."""
        if self.tensor_storage is None:
            return True
        candidates = self._unfuse(prefix)
        if any(
            self.tensor_storage.get(key, {}).get("quant_format") == "exl3"
            for key in candidates
        ):
            return True
        ancestors = self._quantized_ancestors()
        if any(key in ancestors for key in candidates):
            return True
        return any(
            variant in ancestors
            for key in candidates
            for variant in self._drop_one_component(key)
        )

    def _unfuse(self, prefix: str) -> list[str]:
        """Expand a vLLM merged-module prefix back to its checkpoint keys.

        `...self_attn.qkv_proj` exists only in vLLM; the checkpoint has
        `q_proj`/`k_proj`/`v_proj`. `packed_modules_mapping` is filled in by the
        model class as it initializes.
        """
        for fused, parts in self.packed_modules_mapping.items():
            if prefix.endswith(fused):
                stem = prefix[: -len(fused)]
                return [stem + part for part in parts]
        return [prefix]

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            VocabParallelEmbedding,
        )

        from .linear import EXL3LinearMethod
        from .lm_head import EXL3LMHeadMethod

        if isinstance(layer, LinearBase):
            if not self.is_quantized(prefix):
                return UnquantizedLinearMethod()
            return EXL3LinearMethod(self)
        # ParallelLMHead subclasses VocabParallelEmbedding, so it must be
        # tested first.
        if isinstance(layer, ParallelLMHead):
            return EXL3LMHeadMethod(self) if self.head_is_quantized() else None
        if isinstance(layer, VocabParallelEmbedding):
            # EXL3 never quantizes the input embedding: `embed_tokens.weight`
            # is stored dense in every checkpoint inspected.
            return None

        from vllm.model_executor.layers.fused_moe import RoutedExperts

        if isinstance(layer, RoutedExperts):
            from .fused_moe import EXL3MoEMethod

            if not self.is_quantized(prefix):
                return None
            return EXL3MoEMethod(self, layer.moe_config)
        return None
