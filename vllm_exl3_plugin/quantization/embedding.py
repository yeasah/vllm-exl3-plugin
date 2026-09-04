"""EXL3 method for a quantized token embedding.

No EXL3 checkpoint stores a quantized `embed_tokens`: the quantizer leaves the
input embedding at fp16 in every checkpoint inspected. At the sizes this project
targets that is a quarter to a half of the whole file, which is the single
largest remaining gap between EXL3's advertised efficiency and what it actually
costs to serve (docs/embeddings.md has the census).

What makes this fixable *today*, with no new checkpoint format and no quantizer
work, is that a **tied** model already ships a quantized `lm_head` covering
exactly the same matrix -- exllamav3 writes one for every tied model regardless
of the tying -- a pipeline defect. So the embedding can be served from that tensor and
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

from torch.nn import Parameter
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

from .. import blockq, env, format, ops, tp
from ..log import init_logger
from .linear import EXL3Parameter
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


def _blockq_weight_loader(param, loaded_weight: torch.Tensor) -> None:
    """`VocabParallelEmbedding` calls weight loaders with two positional args.

    Every block-quantized tensor is indexed by vocabulary on dim 0, so one rule
    covers all three: take this rank's rows. Compare the trellis path, where the
    same split has to respect 128-row Hadamard blocks and each sub-tensor slices
    on a different axis.
    """
    param.store(param._take_row(loaded_weight, param.row_shard_size, param.tp_rank))


def _encode_on_load(dense: torch.Tensor) -> dict[str, torch.Tensor]:
    """Encode a dense embedding into the three stored tensors, in row chunks.

    On CPU deliberately, and that is a correctness choice rather than a
    performance one. `blockq.encode` does not reproduce across devices -- the
    per-row affine ranges come out of `amin`/`amax` reductions whose order
    differs, and decoded values disagree by up to 2x a quantization step
    (measured; see `blockq.encode`). Neither encoding is worse, but only one of
    them matches what `tools/quantize_embedding.py` writes by default, and a
    model served this way should not disagree with the same model served from a
    checkpoint someone quantized offline.

    Chunked because the encode needs the rows in fp32: a vocabulary-sized
    temporary is several GiB, and the whole point here is not to materialize the
    dense embedding. Every reduction in `encode` is inside a row, so chunking by
    rows is exact rather than approximate.
    """
    chunk = max(1, int(env.get("BLOCKQ_ENCODE_CHUNK", "16384")))
    names = tuple(n.removeprefix(".") for n in format.BLOCKQ_SUFFIXES)
    parts = []
    for start in range(0, dense.shape[0], chunk):
        block = dense[start : start + chunk].to(device="cpu", dtype=torch.float32)
        parts.append(blockq.encode(block))
    return {n: torch.cat([p[n] for p in parts], dim=0) for n in names}


def _make_on_load_loader(layer: torch.nn.Module):
    """Receive the dense embedding and leave `bq_*` behind in its place.

    The three parameters are filled exactly as a checkpoint would have filled
    them -- same `store()`, same shards -- so everything downstream, including
    the shape checks and the vocabulary padding in
    `process_weights_after_loading`, is the path already exercised by
    pre-quantized checkpoints rather than a parallel one.
    """

    def load(param, loaded_weight: torch.Tensor) -> None:
        rows = param._take_row(loaded_weight, param.row_shard_size, param.tp_rank)
        for name, tensor in _encode_on_load(rows).items():
            getattr(layer, name).store(tensor)

    return load


class EXL3BlockQEmbeddingMethod(QuantizeMethodBase):
    """Token embedding served from block-scaled 4-bit storage (`blockq.py`).

    This is the untied-model answer, and the one that needed a format: a tied
    model is served from its existing quantized `lm_head` by
    `EXL3EmbeddingMethod` above, with no new tensors at all, but an untied
    model's head is a genuinely different matrix and there is nothing to reuse.
    The three tensors this loads are produced by `tools/quantize_embedding.py`.

    Nothing here is a custom op. The decode is plain torch so that inductor can
    fuse it into the surrounding graph, which is what made it *faster* than
    calling a hand-written dequant kernel (docs/embeddings.md, "Build or adopt").
    """

    def __init__(self, quant_config):
        self.quant_config = quant_config

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
        # For an embedding vLLM passes the vocabulary as the output dimension and
        # the hidden size as the input: (embedding_dim, [rows_per_partition],
        # embedding_dim, num_embeddings_padded).
        del input_size, output_size

        indices = layer.shard_indices
        rows_per_rank = (
            indices.padded_org_vocab_end_index - indices.padded_org_vocab_start_index
        )

        # The layer's own loader cannot drive this layout; dropped as elsewhere.
        extra_weight_attrs.pop("weight_loader", None)

        device = torch.empty(0).device
        for name in format.BLOCKQ_SUFFIXES:
            suffix = name.removeprefix(".")
            layer.register_parameter(
                suffix,
                EXL3Parameter(
                    num_shards=1,
                    device=device,
                    weight_loader=_blockq_weight_loader,
                    role=tp.role_of(suffix),
                    row_shard_size=rows_per_rank,
                ),
            )

        if self.quant_config.blockq_is_on_load():
            # Nothing in the checkpoint fills `bq_*` here, so a parameter has to
            # exist for the dense tensor to arrive on. Sized zero: it is a
            # landing point for a loader, never storage. The config renames
            # `embed_tokens.weight` onto this name.
            layer.register_parameter(
                "bq_src",
                EXL3Parameter(
                    num_shards=1,
                    device=device,
                    weight_loader=_make_on_load_loader(layer),
                    role=tp.role_of("bq_q"),
                    row_shard_size=rows_per_rank,
                ),
            )

        layer.exl3_hidden_size = input_size_per_partition
        layer.exl3_params_dtype = params_dtype
        layer.exl3_real_rows = (
            indices.org_vocab_end_index - indices.org_vocab_start_index
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        names = [n.removeprefix(".") for n in format.BLOCKQ_SUFFIXES]
        params = {n: getattr(layer, n) for n in names}
        stored = {}
        for name, param in params.items():
            if 0 not in param.shards:
                raise format.EXL3FormatError(
                    f"{getattr(layer, 'prefix', '<embedding>')}: no '{name}' "
                    "tensor was loaded for the block-quantized embedding"
                )
            stored[name] = param.shards[0]

        hidden = format.blockq_hidden_from_shape(stored["bq_q"].shape)
        if hidden != layer.exl3_hidden_size:
            raise format.EXL3FormatError(
                f"block-quantized embedding stores hidden size {hidden}, but the "
                f"model's is {layer.exl3_hidden_size}"
            )
        rows = stored["bq_q"].shape[0]
        for name, want in format.blockq_shapes(rows, hidden).items():
            if tuple(stored[name].shape) != tuple(want):
                raise format.EXL3FormatError(
                    f"block-quantized embedding tensor '{name}' has shape "
                    f"{list(stored[name].shape)}, expected {list(want)}"
                )

        # vLLM pads the vocabulary partition (to a multiple of 64) beyond the rows
        # a checkpoint stores. Those ids are masked before the lookup and zeroed
        # after it, but the gather still indexes with them, so the storage has to
        # cover them. Zero rows decode to zero vectors, which is what the masking
        # would have produced anyway.
        target = layer.num_embeddings_per_partition
        if rows > target:
            raise format.EXL3FormatError(
                f"block-quantized embedding has {rows} rows but vLLM allocated "
                f"only {target}"
            )
        if rows < target:
            # F.pad counts dimensions from the last backwards, so padding dim 0
            # means zeros for every trailing dimension first.
            stored = {
                name: torch.nn.functional.pad(
                    t, (0,) * (2 * (t.dim() - 1)) + (0, target - rows)
                )
                for name, t in stored.items()
            }

        for name, t in stored.items():
            layer.register_parameter(name, Parameter(t, requires_grad=False))
        for param in params.values():
            param.release()
        if hasattr(layer, "bq_src"):
            # Its whole job was to catch the dense tensor; keeping it would
            # leave a zero-size parameter on the layer forever.
            layer.bq_src.release()
            del layer.bq_src

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        """Row gather, replacing `F.embedding(input_, layer.weight)`.

        `input_` is already this rank's local row index -- `VocabParallelEmbedding`
        has subtracted the shard offset and masked out-of-range ids -- which is
        exactly what the loader stored, so no index arithmetic is needed here.
        """
        return blockq.gather(
            input_, layer.bq_q, layer.bq_s, layer.bq_r, layer.exl3_params_dtype
        )

    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        # Reached only if something routed a linear through the embedding's
        # method. An untied model's lm_head has its own.
        raise format.EXL3FormatError(
            "the block-quantized embedding has no matmul path: it stores an "
            "embedding, not a weight matrix a GEMM can consume"
        )


class EXL3BlockQTiedEmbeddingMethod(EXL3BlockQEmbeddingMethod, EXL3EmbeddingMethod):
    """A *tied* model whose embedding has also been block-quantized.

    This is the best-measured arrangement for a tied model and the one the
    predicates above originally treated as impossible. A tied checkpoint that
    `tools/quantize_embedding.py` has repaired holds both encodings of the same
    logical matrix -- `bq_*` for the embedding and the trellis `lm_head.*` for
    the head -- and each is the right encoding for its own role: a row gather
    wants per-row scales, a GEMM wants the trellis. Serving the embedding from
    the trellis instead (what a tied model gets without this) costs +0.0216 KLD
    against +0.0003 here, ~73x worse, because it is the wrong encoding for a
    lookup (docs/embeddings.md, "The head sweep").

    **Both tensor sets live on this one module**, which looks odd and is in fact
    the existing design: vLLM skips a tied model's `lm_head.*` outright, so
    `get_cache_scale_mapper` renames those weights onto the embedding's prefix
    and `EXL3TiedLMHeadMethod` -- owning no storage -- reads them back from here
    to compute logits. Nothing about that changes; this class only adds the
    block-quantized tensors beside them, and points the *lookup* at those
    instead of at the trellis.

    So the split of responsibilities is:

    - `embedding()` -> `bq_*`, inherited from `EXL3BlockQEmbeddingMethod`.
    - `apply()` -> the trellis, inherited from `EXL3EmbeddingMethod`, and reached
      only via the tied head's `exl3_tied_source`.
    """

    def __init__(self, quant_config):
        # Not `EXL3BlockQEmbeddingMethod.__init__`, which records only the
        # config: the inherited `apply()` needs the codebook flags that
        # `EXL3LinearMethod.__init__` derives.
        EXL3EmbeddingMethod.__init__(self, quant_config)

    def create_weights(self, layer: torch.nn.Module, *args, **kwargs) -> None:
        """Allocate both sets. Safe to run back to back: the parameter names are
        disjoint (`bq_*` against `trellis`/`suh`/`svh`), and the two layer
        attributes they both set -- `exl3_params_dtype` and `exl3_real_rows` --
        are computed from the same arguments and agree by construction."""
        EXL3EmbeddingMethod.create_weights(self, layer, *args, **kwargs)
        EXL3BlockQEmbeddingMethod.create_weights(self, layer, *args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        EXL3EmbeddingMethod.process_weights_after_loading(self, layer)
        EXL3BlockQEmbeddingMethod.process_weights_after_loading(self, layer)

    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        """Logits off the trellis, stated explicitly because the MRO gets this
        wrong in a way that only one of the two tied shapes notices.

        `EXL3BlockQEmbeddingMethod` comes first here -- it has to, so the lookup
        resolves to the block-quantized gather -- and it defines `apply` as a
        stub that raises "no matmul path". That stub is correct for an untied
        embedding, which genuinely has no weight matrix a GEMM can consume, and
        wrong here, where the module also holds the head's trellis.

        The gemma4-style shape hides this: its head is a separate module whose
        own `EXL3TiedLMHeadMethod.apply` reaches the trellis through
        `EXL3EmbeddingMethod`, never through this class. The Qwen3-style shape
        does not -- one module serves both roles, so this method *is* the logits
        path, and inheriting the stub would raise on the first token.
        """
        return EXL3EmbeddingMethod.apply(self, layer, *args, **kwargs)
