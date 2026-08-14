"""EXL3 method for a quantized token embedding.

No EXL3 checkpoint stores a quantized `embed_tokens`: the quantizer leaves the
input embedding at fp16 in every checkpoint inspected. At the sizes this project
targets that is a quarter to a half of the whole file, which is the single
largest remaining gap between EXL3's advertised efficiency and what it actually
costs to serve (PHASE4.md has the census).

What makes this fixable *today*, with no new checkpoint format and no quantizer
work, is that a **tied** model already ships a quantized `lm_head` covering
exactly the same matrix -- exllamav3 writes one for every tied model regardless
of the tying (TODO.md #2). So the embedding can be served from that tensor and
the fp16 copy simply never loaded. gemma-4-12B-it sheds 1.88 GiB this way, 29%
of the checkpoint.

Two things have to happen for that, and neither is a hack:

1. **The checkpoint's `lm_head.*` has to reach this module at all.** Every tied
   model's `load_weights` skips it -- Qwen3-style via `skip_prefixes`, gemma4
   via `skip_substrs` -- because for an unquantized model those bytes are
   genuinely redundant. `AutoWeightsLoader` applies the weights mapper *before*
   its skip filter, and the quantization config may contribute to that mapper,
   so `EXL3Config.get_cache_scale_mapper` renames them onto this module's prefix
   and the skip never matches. See `config.py`.

2. **A lookup must not dequantize the tensor.** `ops.embed_rows` does the gather
   directly out of the trellis, decoding only the 128-row blocks a batch
   actually touches; see its docstring for why that is exact.

The two tied-model shapes vLLM builds are handled differently:

- **One module** (Qwen3-style, `self.lm_head = self.model.embed_tokens`): this
  method serves both roles -- `embedding()` for the lookup and the inherited
  `apply()` for the logits matmul, off the same stored trellis.
- **Two modules** (gemma4-style, a real `ParallelLMHead` tied via
  `tie_weights()`): the head gets `EXL3TiedLMHeadMethod` below, which owns no
  storage and forwards to the embedding's.
"""

from __future__ import annotations

import torch

from .. import format, ops
from ..log import init_logger
from .lm_head import EXL3LMHeadMethod

logger = init_logger(__name__)


class EXL3EmbeddingMethod(EXL3LMHeadMethod):
    """Token embedding served straight from EXL3 storage.

    Weight creation, loading and shape checking are entirely inherited:
    `ParallelLMHead` *is* a `VocabParallelEmbedding`, the two get identical
    `create_weights` arguments, and the tensor being loaded is in fact an
    lm_head. Only the lookup is new.
    """

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        """Row gather, replacing `F.embedding(input_, layer.weight)`.

        `input_` is already this rank's local row index: at TP=1 that is the
        token id, and above it `VocabParallelEmbedding.forward` has subtracted
        the shard offset and masked out-of-range ids (which it then zeroes on
        the way out). That matches what the loader stored -- this rank's column
        slice of the tensor -- so no further index arithmetic is needed here.
        """
        if self.quant_config.dequantize:
            # Phase 0's reference path: a real fp16 matrix exists, so this is
            # the ordinary lookup and stays available as a correctness oracle.
            return torch.nn.functional.embedding(input_, layer.exl3_weight)

        flat = input_.reshape(-1)
        rows = ops.embed_rows(
            layer.exl3_trellis_0,
            layer.exl3_suh_0,
            layer.exl3_svh_0,
            format.bits_from_trellis_shape(layer.exl3_trellis_0.shape),
            self.mcg,
            self.mul1,
            flat,
        )
        # exllamav3 pads the input dimension up to a multiple of 128; the model
        # wants its real hidden size. The padded columns only ever multiplied
        # zeros, so trimming is exact -- the same trim the linear path applies.
        rows = rows[:, : layer.exl3_input_size]
        return rows.view(*input_.shape, rows.shape[-1]).to(layer.exl3_params_dtype)

    def tie_weights(self, layer: torch.nn.Module, embed_tokens: torch.nn.Module):
        """Point a separate `ParallelLMHead` at this embedding's storage.

        Reached only in the gemma4-style two-module shape, and only when the
        head itself was given this method (see `EXL3Config.get_quant_method`).
        The base class does `layer.weight = embed_tokens.weight`, which cannot
        work here: there is no `.weight`, and the tensors it would alias do not
        exist yet -- tying happens at construction, long before loading.
        Recording the source module instead defers the question to `apply()`.
        """
        layer.exl3_tied_source = embed_tokens
        return layer


class EXL3TiedLMHeadMethod(EXL3EmbeddingMethod):
    """A tied `ParallelLMHead` that owns no storage of its own.

    gemma4-style tying builds a real head module and then ties it. Its weights
    were renamed onto the embedding (see module docstring), so there is nothing
    for this module to allocate or load, and `process_weights_after_loading`
    would have no shards to find. It exists only to compute logits, which it
    does off the embedding's tensors.
    """

    def create_weights(self, layer: torch.nn.Module, *args, **kwargs) -> None:
        # Deliberately allocates nothing. Registering EXL3Parameters here would
        # make `default_loader` reject the model for weights that were, by
        # design, routed to the embedding instead.
        layer.exl3_tied_source = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "exl3_tied_source", None) is None:
            raise format.EXL3FormatError(
                "tied lm_head was never tied to an embedding; "
                "vLLM built it in a shape this plugin does not recognize"
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().apply(layer.exl3_tied_source, x, bias)
