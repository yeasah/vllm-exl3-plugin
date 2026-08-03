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

from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

from ..format import EXL3FormatError

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

    def get_name(self) -> QuantizationMethods:
        return "exl3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # exllamav3's kernels are fp16 throughout; `LinearEXL3` only ever
        # chooses between an fp16 and an fp32 *output*. Most EXL3 repos inherit
        # `torch_dtype: bfloat16` from their base model, so vLLM will default to
        # bfloat16 and reject it here -- serving needs an explicit
        # `--dtype float16`. Failing loudly beats silently downcasting.
        return [torch.half]

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
        self._load_tensor_storage(model_name, revision)
        self._reject_unsupported_head(hf_config)

    def _load_tensor_storage(self, model_name: str, revision: str | None) -> None:
        from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

        try:
            extra = get_hf_file_to_dict(
                "quantization_config.json", model_name, revision or "main"
            )
        except Exception:
            # Purely supplementary: a missing or unreachable file leaves us on
            # the "everything is quantized" fallback, which is correct for every
            # checkpoint exllamav3's converter produces.
            extra = None
        if extra:
            self.tensor_storage = extra.get("tensor_storage")
            self.codebook = extra.get("codebook", self.codebook)

    def _reject_unsupported_head(self, hf_config: Any) -> None:
        """Fail at startup rather than at the first wrong token.

        EXL3 quantizes `lm_head` too (at `head_bits`). vLLM's `ParallelLMHead` is
        a `VocabParallelEmbedding`, not a `LinearBase`, so this plugin does not
        yet provide a method for it. Models with tied embeddings are fine --
        vLLM skips `lm_head.*` entirely -- which is why the Phase 0 test target
        is a tied model.
        """
        if self.tensor_storage is None or hf_config is None:
            return
        head = self.tensor_storage.get("lm_head")
        if head is None or head.get("quant_format") != "exl3":
            return
        tied = getattr(hf_config, "tie_word_embeddings", False)
        if not tied:
            raise NotImplementedError(
                "This EXL3 checkpoint has a quantized lm_head and untied word "
                "embeddings. vllm-exl3-plugin cannot load a quantized lm_head "
                "yet (it needs a VocabParallelEmbedding method, not a linear "
                "one). Use a checkpoint with tied embeddings for now."
            )

    def is_quantized(self, prefix: str) -> bool:
        """Whether the module at `prefix` has EXL3 storage in the checkpoint."""
        if self.tensor_storage is None:
            return True
        candidates = self._unfuse(prefix)
        return any(
            self.tensor_storage.get(key, {}).get("quant_format") == "exl3"
            for key in candidates
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
        from .linear import EXL3LinearMethod

        if isinstance(layer, LinearBase):
            if not self.is_quantized(prefix):
                return UnquantizedLinearMethod()
            return EXL3LinearMethod(self)
        # Embeddings, LM heads and MoE layers fall through to vLLM's defaults.
        # Quantized MoE is Phase 3; a quantized lm_head is rejected above.
        return None
