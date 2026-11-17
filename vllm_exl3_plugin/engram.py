"""Hashed n-gram embedding tables (Qwen4Exp's PLE) for EXL3 checkpoints, kept off the GPU.

An EXL3 Qwen4Exp checkpoint (turboderp's Qwen3.8-Flash-Next-exl3) stores its n-gram table -- about
51B parameters -- outside the indexed shards, in its own file, as one K-bit trellis ring per row plus
a per-head bias (exllamav3 `util/convert_ngram.py`). vLLM's Qwen4Exp builds the table as a parameter
(on the device, or pinned in host memory under `engram_config.cpu_offload`) and accepts only BF16,
FP8 or NVFP4 rows: `Qwen4ExpPLEEmbeddingMethod.from_quant_config` raises on any other config.

Here the table stays on disk. Rows are read on the host each step and staged into fixed device
buffers before the forward, which only decodes from those buffers, so CUDA graphs replay it --
the shape upstream vLLM gave its own disk offload (`b44a48c`, after our fork). The reads use
exllamav3's own `RowTable`, which streams this format natively.

Two patches route an EXL3 checkpoint here: `from_quant_config` returns `EXL3PLEEmbeddingMethod`
for our config, and the table classes `Qwen4ExpNGramEmbedding.__init__` picks from dispatch to
`EXL3DiskPLEEmbedding` when that method is in use, so nothing ever allocates the table.

Milestone 1 (this file, so far): the table is a stub returning zero rows, to prove the EXL3 body
loads and runs on the fork. TODO.md `engram-ple`.
"""

from __future__ import annotations

import torch

from .log import init_logger

logger = init_logger(__name__)


def _ple_classes():
    from vllm.models.qwen4_exp.common import ngram_embedding as common

    return common


def make_method_class():
    common = _ple_classes()

    class EXL3PLEEmbeddingMethod(common.Qwen4ExpPLEEmbeddingMethod):
        """An EXL3 n-gram table: no parameter (the rows stay on disk), rows decoded to the
        activation dtype before they reach `dequantize`."""

        def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                           input_size, output_size, params_dtype, **extra_weight_attrs):
            del input_size, output_size, extra_weight_attrs
            # Empty: the rows stay on disk. Qwen4ExpNGramEmbedding reads `weight` to log its dtype and device
            layer.register_parameter("weight", torch.nn.Parameter(
                layer.allocate_embedding_weight(0, input_size_per_partition, params_dtype), requires_grad=False))
            layer.exl3_row_dim = input_size_per_partition
            layer.exl3_rows = sum(output_partition_sizes)
            layer.exl3_params_dtype = params_dtype

        def dequantize(self, layer, embeddings, output_dtype):
            del layer
            return embeddings.to(output_dtype)

    return EXL3PLEEmbeddingMethod


def make_embedding_class():
    common = _ple_classes()

    class EXL3DiskPLEEmbedding(common.Qwen4ExpPLEEmbedding):
        """The table on disk, staged per step into fixed device buffers. Milestone 1: zero rows."""

        supports_prefetch = False

        def __init__(self, num_embeddings, embedding_dim, *, params_dtype, padding_size, prefix,
                     embedding_method, num_ngram_heads=1, max_total_tokens=0, data_parallel_rank=0,
                     reduce_results=True):
            super().__init__(num_embeddings, embedding_dim, params_dtype=params_dtype,
                             padding_size=padding_size, prefix=prefix, embedding_method=embedding_method,
                             num_ngram_heads=num_ngram_heads, max_total_tokens=max_total_tokens,
                             data_parallel_rank=data_parallel_rank, reduce_results=reduce_results)
            if self.etp_data_parallel_size != 1:
                raise NotImplementedError("EXL3 n-gram table: ETP spanning DP is not supported")
            self.num_ngram_heads = num_ngram_heads
            # Fixed for the life of the engine, so a captured graph reads the same address every replay
            self._rows = torch.zeros(max(1, max_total_tokens), num_ngram_heads, embedding_dim,
                                     dtype=params_dtype)
            logger.info("EXL3 n-gram table %s: %d rows x %d, staged per step (stub: zero rows)",
                        prefix, num_embeddings, embedding_dim)

        def allocate_embedding_weight(self, num_embeddings, embedding_dim, dtype):
            return torch.empty(0, embedding_dim, dtype=dtype)

        def start_prefetch(self, ngram_ids):
            return None

        def finalize_prefetch(self, total_tokens):
            raise RuntimeError("EXL3 n-gram table: rows are staged before the forward, not prefetched")

        def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
            return self._rows[: ngram_ids.shape[0]]

    return EXL3DiskPLEEmbedding


def patch_qwen4_exp() -> None:
    """Route EXL3 checkpoints' n-gram tables to `EXL3DiskPLEEmbedding`. A no-op where vLLM has no
    Qwen4Exp."""
    try:
        common = _ple_classes()
        from vllm.models.qwen4_exp.nvidia import ngram_embedding as nvidia
    except ImportError:
        return
    if getattr(common.Qwen4ExpPLEEmbeddingMethod, "_exl3_patched", False):
        return
    from .quantization.config import EXL3Config

    Method, Table = make_method_class(), make_embedding_class()
    original = common.Qwen4ExpPLEEmbeddingMethod.from_quant_config

    def from_quant_config(quant_config, prefix, embedding_dtype=None):
        if isinstance(quant_config, EXL3Config) and embedding_dtype is None:
            return Method()
        return original(quant_config, prefix, embedding_dtype)

    common.Qwen4ExpPLEEmbeddingMethod.from_quant_config = staticmethod(from_quant_config)
    common.Qwen4ExpPLEEmbeddingMethod._exl3_patched = True

    # `Qwen4ExpNGramEmbedding.__init__` picks the device or pinned-host class from these module
    # names; either allocates the whole table. A subclass that hands an EXL3 method's table to ours
    # keeps every isinstance check on the originals working.
    for name in ("Qwen4ExpPLEDeviceEmbedding", "Qwen4ExpPLEPinnedHostEmbedding"):
        cls = getattr(nvidia, name)

        def __new__(klass, *args, _cls=cls, **kwargs):
            if isinstance(kwargs.get("embedding_method"), Method):
                table = Table.__new__(Table)
                table.__init__(*args, **kwargs)
                return table
            return super(klass, klass).__new__(klass)

        setattr(nvidia, name, type(name, (cls,), {"__new__": __new__, "__module__": cls.__module__}))
    nvidia._EXL3Method, nvidia._EXL3Table = Method, Table
