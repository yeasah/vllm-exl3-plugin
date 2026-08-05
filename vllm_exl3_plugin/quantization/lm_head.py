"""EXL3 method for a quantized `lm_head`.

EXL3 quantizes the output projection too, at `head_bits` (usually 6). That is
invisible on tied-embedding models -- vLLM skips `lm_head.*` outright -- but
every EXL3 repo from ~v0.0.12 onward sets `head_bits`, and models above ~3B
rarely tie, so this covers most of the modern checkpoint ecosystem.

`ParallelLMHead` subclasses `VocabParallelEmbedding`, not `LinearBase`, so it
needs its own method class. Two things differ from the linear case:

1. **The weight loader.** `VocabParallelEmbedding` passes its own v1-style
   `weight_loader(param, loaded_weight)` rather than `weight_loader_v2`, and
   that loader assumes a preallocated tensor it can `narrow` along a vocab
   dimension -- which an EXL3 trellis (tile-granular, with its own padding) is
   not. We substitute a loader that just stores, the same way the GGUF plugin
   substitutes `_gguf_embedding_weight_loader`.

2. **Two independent vocab paddings.** vLLM pads the vocabulary up to a multiple
   of 64 (`num_embeddings_padded`); exllamav3 padded the output dimension up to
   a multiple of 128 before quantizing. Both can exceed `org_vocab_size`, and
   they need not agree. EXL3's padded rows hold quantization *noise*, not zeros,
   so they are trimmed off and the result is zero-padded to whatever vLLM asked
   for -- matching the convention `VocabParallelEmbedding.weight_loader` uses
   for unquantized heads, where padding rows are explicitly zero-filled.
"""

from __future__ import annotations

import torch
from torch.nn import Parameter

from .. import format, ops
from .linear import EXL3LinearMethod, EXL3Parameter


def _lm_head_weight_loader(param: EXL3Parameter, loaded_weight: torch.Tensor) -> None:
    """`VocabParallelEmbedding` calls weight loaders with two positional args."""
    param.store(loaded_weight)


class EXL3LMHeadMethod(EXL3LinearMethod):
    """Quantized output projection. Shares the linear method's dequantization."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size

        if getattr(layer, "tp_size", 1) > 1:
            raise NotImplementedError(
                "A quantized lm_head cannot be tensor-parallel yet. Linear "
                "layers shard (see tp.py), but the head has to combine a "
                f"vocab-dimension split on {format.HAD_BLOCK}-channel "
                "boundaries with vLLM's own vocab padding and the trim in "
                "apply(), and that interaction is unverified. Models with "
                "tied embeddings are unaffected -- vLLM skips lm_head there."
            )
        # The layer's own weight_loader cannot drive EXL3's layout; see module
        # docstring. Deliberately dropped rather than wrapped.
        extra_weight_attrs.pop("weight_loader", None)

        device = torch.empty(0).device
        for name in self.quant_config.stored_tensor_names():
            layer.register_parameter(
                name,
                EXL3Parameter(
                    num_shards=1,
                    device=device,
                    weight_loader=_lm_head_weight_loader,
                ),
            )

        layer.exl3_input_size = input_size_per_partition
        layer.exl3_output_sizes = list(output_partition_sizes)
        layer.exl3_params_dtype = params_dtype

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        names = self.quant_config.stored_tensor_names()
        params = {name: getattr(layer, name) for name in names}

        trellis = self._shard(params, "trellis", 0, layer)
        suh = self._shard(params, "suh", 0, layer)
        svh = self._shard(params, "svh", 0, layer)

        bits = format.bits_from_trellis_shape(trellis.shape)
        stored_in, stored_out = format.dims_from_trellis_shape(trellis.shape)
        # The checkpoint is sized against the real vocabulary, not against
        # whatever vLLM padded it to.
        self._check_shapes(
            layer, 0, stored_in, stored_out, layer.org_vocab_size, suh, svh
        )

        target = layer.num_embeddings_per_partition
        if layer.org_vocab_size > target:
            raise format.EXL3FormatError(
                f"lm_head has {layer.org_vocab_size} real vocabulary rows but "
                f"vLLM allocated only {target}"
            )

        if self.quant_config.dequantize:
            weight = ops.dense_weight(
                trellis, suh, svh, bits, "mcg" in params, "mul1" in params
            )
            weight = weight[: layer.org_vocab_size, : layer.exl3_input_size]
            if weight.shape[0] < target:
                # Zero rows produce zero logits, which `_get_logits` then slices
                # off at org_vocab_size anyway.
                weight = torch.nn.functional.pad(
                    weight, (0, 0, 0, target - weight.shape[0])
                )
            layer.register_parameter(
                "exl3_weight",
                Parameter(
                    weight.to(layer.exl3_params_dtype).contiguous(),
                    requires_grad=False,
                ),
            )
        else:
            self._store_quantized(layer, [(trellis, suh, svh, bits)])

        for param in params.values():
            param.release()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.quant_config.dequantize:
            return torch.nn.functional.linear(x, layer.exl3_weight, bias)

        logits = ops.exl3_mm(
            x,
            layer.exl3_trellis_0,
            layer.exl3_suh_0,
            layer.exl3_svh_0,
            self.mcg,
            self.mul1,
        )
        # The kernel always writes exllamav3's padded vocabulary width. vLLM
        # expects its own (`num_embeddings_per_partition`), and the two round
        # differently: 128 vs 64. Trim or zero-extend to match, keeping the
        # rows beyond the real vocabulary at zero rather than leaving EXL3's
        # padding noise in place.
        target = layer.num_embeddings_per_partition
        real = layer.org_vocab_size
        if logits.shape[-1] != target or real < target:
            logits = logits[..., :real]
            if real < target:
                logits = torch.nn.functional.pad(logits, (0, target - real))
        if bias is not None:
            logits = logits + bias
        return logits
