"""`QuantizationConfig` for EXL3 checkpoints.

An EXL3 model is an ordinary HF repo: standard safetensors, standard
`config.json`, tensor keys matching the original model 1:1, and a
`quantization_config` block whose `quant_method` is `"exl3"`. vLLM's normal
config detection therefore finds us with no loader or config-parser patching --
this is the part of the GGUF plugin that EXL3 simply does not need.

Models quantized since ~v0.0.2 additionally ship `quantization_config.json`,
which carries a per-module `tensor_storage` map (shapes, dtypes, bit width,
codebook). It is not part of the HF config, so vLLM does not read it; we fetch
it ourselves in `maybe_update_config`.

**`tensor_storage` is metadata, not the checkpoint, and the two can disagree.**
`turboderp/Muse-Glimmer-30B-exl3` quantizes its 50-layer vision tower, adapter
and projection -- 303 modules -- and lists none of them, recording only the 416
language-model modules and `lm_head`. So what actually decides whether a module
is quantized is `model.safetensors.index.json`, which names every tensor that
exists; the storage map is consulted for the things only it knows (codebook,
bit widths) and as a fallback for single-file repos that have no index.

Checkpoints with neither fall back to assuming every linear is quantized --
which is what exllamav3's converter did before it recorded anything.
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


from .. import env, format
from ..format import EXL3FormatError
from ..log import init_logger

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
        # Module keys carrying a block-quantized embedding, from the safetensors
        # index when there is one.
        self._blockq_modules: set[str] = set()
        # Modules the *checkpoint* quantizes, from the safetensors index. Takes
        # precedence over tensor_storage, which can omit some (see
        # _load_index_modules). None means "no index was readable".
        self.quantized_modules: set[str] | None = None
        self._ancestors: set[str] | None = None
        # Also from maybe_update_config, which is the only hook that sees
        # hf_config. None means "not yet known".
        self.tie_word_embeddings: bool | None = None
        # Phase 0's dequantize-at-load path, kept as a correctness oracle: it
        # is a transcription of exllamav3's own dequantization, so serving the
        # same prompts both ways isolates a kernel bug from a plumbing bug.
        # Costs the entire memory saving, so it is opt-in.
        self.dequantize = env.flag("DEQUANTIZE")
        if self.dequantize:
            self._disable_stale_compile_cache()
        # Escape hatch for the quantized-embedding path; see
        # `embedding_is_quantized`.
        self._dense_embed = env.flag("DENSE_EMBED")
        # Where the token embedding lives in vLLM's module tree. Multimodal
        # wrappers nest it (`model.language_model.embed_tokens`), so this is
        # resolved from the model's own weights mapper in `apply_vllm_mapper`
        # rather than assumed; this is the flat-text-model default.
        self.embed_prefix = "model.embed_tokens"

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
        `EXL3_DEQUANTIZE` -- which changes what `apply()` traces to, one
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
            "EXL3_DEQUANTIZE is set, so VLLM_DISABLE_COMPILE_CACHE=1 has "
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

    def _load_index_modules(self, model_name: str, revision: str | None) -> None:
        """Modules that actually carry a trellis, from the safetensors index.

        `tensor_storage` is *metadata about* the checkpoint and can disagree
        with it. `turboderp/Muse-Glimmer-30B-exl3` is the case that forced this:
        it quantizes the whole 50-layer vision tower, its adapter and its
        projection -- 303 modules -- and lists none of them, recording only the
        416 language-model modules and `lm_head`. Believing the metadata there
        means handing 303 quantized modules to `UnquantizedLinearMethod`, which
        allocates a dense `weight` the checkpoint does not contain.

        The index is ground truth: it names every tensor that exists. Where the
        two disagree, this wins.
        """
        from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

        try:
            index = get_hf_file_to_dict(
                "model.safetensors.index.json", model_name, revision or "main"
            )
        except Exception:
            index = None
        weight_map = (index or {}).get("weight_map")
        if not weight_map:
            # Single-file checkpoints have no index. `tensor_storage` remains
            # the only map we have, so leave this unset and fall back to it.
            return
        self.quantized_modules = format.quantized_module_keys(weight_map)
        self._blockq_modules = format.blockq_module_keys(weight_map)
        self._ancestors = None

    def _load_tensor_storage(self, model_name: str, revision: str | None) -> None:
        from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

        self._load_index_modules(model_name, revision)
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
            missing = self._modules_missing_from_storage()
            if missing:
                logger.warning(
                    "quantization_config.json for %s@%s omits %d module(s) that "
                    "the checkpoint quantizes anyway (e.g. %s). Trusting the "
                    "safetensors index instead; the storage map is metadata, "
                    "not the checkpoint.",
                    model_name, revision or "main", len(missing),
                    ", ".join(sorted(missing)[:2]),
                )
        elif self.quantized_modules is not None:
            # No storage map, but the index still says exactly what is
            # quantized, which is all `is_quantized` needs.
            pass
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
        if self.quantized_modules is not None:
            return "lm_head" in self.quantized_modules
        if self.tensor_storage is not None:
            head = self.tensor_storage.get("lm_head")
            return bool(head) and head.get("quant_format") == "exl3"
        # No storage map (pre-v0.0.2-era repos): `head_bits` in config.json's
        # quantization_config is the remaining signal. Checkpoints that quantize
        # the head always set it; the ones that do not, omit it entirely.
        return self.head_bits is not None

    def _head_storage_exists(self) -> bool:
        """Whether the checkpoint physically stores a quantized `lm_head`.

        Deliberately *not* `head_is_quantized`, which answers a different
        question -- whether vLLM's `lm_head` module should load one -- and is
        false for every tied model precisely because vLLM skips those weights.
        Here the tying is the reason to look.
        """
        if self.quantized_modules is not None:
            return "lm_head" in self.quantized_modules
        if self.tensor_storage is not None:
            head = self.tensor_storage.get("lm_head")
            return bool(head) and head.get("quant_format") == "exl3"
        return self.head_bits is not None

    def embedding_is_quantized(self) -> bool:
        """Whether the token embedding is served from EXL3 storage.

        True only for a tied model whose checkpoint carries a quantized
        `lm_head`: that tensor *is* the embedding, so it can serve the lookup
        and the dense `embed_tokens.weight` need never be loaded. Untied models
        have no quantized copy of the embedding anywhere and stay dense until
        something produces one (docs/embeddings.md, Phase B).

        Opt out with `EXL3_DENSE_EMBED=1`, which is the first thing to try
        if a model looks numerically wrong: it isolates the embedding from every
        other change, since nothing else about the model differs between the two
        paths.
        """
        if self._dense_embed:
            return False
        return bool(self.tie_word_embeddings) and self._head_storage_exists()

    def embedding_is_blockq(self) -> bool:
        """Whether the checkpoint carries a block-quantized `embed_tokens`.

        This is the *untied* answer to the same problem `embedding_is_quantized`
        solves for tied models, and the two are mutually exclusive: a tied model
        is served from its `lm_head` and needs no new tensors, while an untied
        model has no quantized copy of its embedding until
        `tools/quantize_embedding.py` makes one.

        Both sources are consulted for the same reason `is_quantized` consults
        both: the safetensors index names every tensor that exists and is ground
        truth, but single-file checkpoints have no index, leaving the storage map
        as the only evidence.
        """
        stored = bool(self._blockq_modules) or any(
            entry.get("quant_format") == "exl3_blockq"
            for entry in (self.tensor_storage or {}).values()
        )
        if stored and self._dense_embed:
            # EXL3_DENSE_EMBED exists to isolate a *tied* model's embedding from
            # everything else by loading the dense copy instead. A repaired
            # checkpoint has no dense copy -- removing it is the point -- so
            # honoring the flag here would fail later, on a missing
            # `embed_tokens.weight`, far from the cause.
            logger.warning(
                "EXL3_DENSE_EMBED is set, but this checkpoint stores a "
                "block-quantized embedding and no dense one. Serving it "
                "quantized; the flag only applies to tied models."
            )
        return stored

    def get_cache_scale_mapper(self):
        """Route a tied model's `lm_head.*` onto the embedding module.

        This hook is named for KV-cache scales, but it is the one place a
        quantization config may rewrite the weight stream, and
        `AutoWeightsLoader` applies it *before* the skip filter
        (`models/utils.py:418` vs `:421`). That ordering is what makes this
        possible at all: every tied model drops `lm_head.*` on the floor
        (`skip_prefixes` for Qwen3-style, `skip_substrs` for gemma4-style), so
        renaming those tensors first is what gets them to the embedding instead
        of requiring the loader to be patched.

        Declared a `@staticmethod` on the base class but invoked on the
        instance, so overriding it as a normal method is safe and is what lets
        the rename depend on this checkpoint being tied.
        """
        mapper = super().get_cache_scale_mapper()
        if not self.embedding_is_quantized():
            return mapper

        import re

        from vllm.model_executor.models.utils import WeightsMapper

        return mapper | WeightsMapper(
            # Drop the dense embedding unread -- this is the actual saving, and
            # it has to go because the module no longer has a `weight` to put it
            # in. Matched on the checkpoint's own name: regex rules run before
            # the prefix rules below (`models/utils.py:101` vs `:121`), so this
            # sees `...embed_tokens.weight` and never the renamed `lm_head.*`,
            # which end in `.trellis`/`.suh`/`.svh`.
            orig_to_new_regex={re.compile(r"(^|\.)embed_tokens\.weight$"): None},
            orig_to_new_prefix={"lm_head.": f"{self.embed_prefix}."},
        )

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
        if self.quantized_modules is not None:
            # apply_dict works on a dict, so round-trip through one.
            self.quantized_modules = set(
                hf_to_vllm_mapper.apply_dict(
                    dict.fromkeys(self.quantized_modules, True)
                )
            )
            self._ancestors = None

    def _exl3_modules(self) -> set[str]:
        """Every module known to carry EXL3 storage, index first."""
        if self.quantized_modules is not None:
            return self.quantized_modules
        return {
            key
            for key, entry in (self.tensor_storage or {}).items()
            if entry.get("quant_format") == "exl3"
        }

    def _modules_missing_from_storage(self) -> set[str]:
        """Quantized modules the storage map fails to mention."""
        if self.quantized_modules is None or self.tensor_storage is None:
            return set()
        listed = {
            key
            for key, entry in self.tensor_storage.items()
            if entry.get("quant_format") == "exl3"
        }
        return self.quantized_modules - listed

    def _quantized_ancestors(self) -> set[str]:
        """Every prefix that has at least one EXL3 module beneath it.

        A routed-experts module is named `...mlp.experts` while the checkpoint
        only ever mentions `...mlp.experts.{id}.gate_proj`, so an exact lookup
        finds nothing and the layer looks unquantized. Precomputed once because
        `tensor_storage` runs to ~30k entries on a 256-expert model.
        """
        if self._ancestors is None:
            ancestors: set[str] = set()
            for key in self._exl3_modules():
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
        if self.tensor_storage is None and self.quantized_modules is None:
            return True
        candidates = self._unfuse(prefix)
        known = self._exl3_modules()
        if any(key in known for key in candidates):
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
        from .embedding import EXL3EmbeddingMethod, EXL3TiedLMHeadMethod

        # ParallelLMHead subclasses VocabParallelEmbedding, so it must be
        # tested first.
        if isinstance(layer, ParallelLMHead):
            if self.embedding_is_quantized():
                # Tied, and served from the quantized lm_head this head's own
                # weights were renamed onto. Owns no storage; see embedding.py.
                return EXL3TiedLMHeadMethod(self)
            return EXL3LMHeadMethod(self) if self.head_is_quantized() else None
        if isinstance(layer, VocabParallelEmbedding):
            # Where the rename in `get_cache_scale_mapper` has to send
            # `lm_head.*`. Taken from the module itself rather than guessed:
            # multimodal wrappers nest the embedding (gemma-4 puts it under
            # `language_model`), and construction runs before any weight is
            # loaded, so this is known in time.
            #
            # Recorded before any branch. It used to be set only on the path
            # below, so a checkpoint taking the blockq branch left it at its
            # `"model.embed_tokens"` default while the rename still fired --
            # routing 755 MiB of trellis to a module path a nested model does
            # not have, dropping it silently, and serving garbage.
            self.embed_prefix = prefix

            blockq_embed = self.embedding_is_blockq()
            # `embedding_is_quantized` answers a question about the *head's*
            # storage -- whether a tied model's `lm_head.*` is being renamed
            # onto this module -- so it is not exclusive with the embedding
            # having block-quantized tensors of its own. A repaired tied
            # checkpoint makes both true, and both are then load-bearing.
            tied_head_here = self.embedding_is_quantized()

            if blockq_embed:
                from .embedding import (
                    EXL3BlockQEmbeddingMethod,
                    EXL3BlockQTiedEmbeddingMethod,
                )

                if tied_head_here:
                    # Tied *and* repaired: this module owns the `bq_*` tensors
                    # for the lookup and receives the head's trellis for the
                    # logits matmul. Each role gets the encoding built for it.
                    return EXL3BlockQTiedEmbeddingMethod(self)
                # Untied. Nothing is renamed onto this module: the tensors are
                # its own, and the head has its own method.
                return EXL3BlockQEmbeddingMethod(self)

            # EXL3 never quantizes the input embedding -- `embed_tokens.weight`
            # is dense in every checkpoint inspected -- but a *tied* model ships
            # a quantized lm_head covering the same matrix, which can serve as
            # the embedding so the dense copy is never loaded at all.
            if not tied_head_here:
                return None
            return EXL3EmbeddingMethod(self)

        from vllm.model_executor.layers.fused_moe import RoutedExperts

        if isinstance(layer, RoutedExperts):
            from .fused_moe import EXL3MoEMethod

            if not self.is_quantized(prefix):
                return None
            return EXL3MoEMethod(self, layer.moe_config)
        return None
