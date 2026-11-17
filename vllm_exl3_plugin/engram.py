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

The per-step staging needs the step's n-gram ids, which depend on the token the previous step
sampled on the GPU: one stream sync per step, as upstream's disk path also takes. It runs from
`Qwen4ExpModelState.prepare_inputs`, outside any CUDA graph; graph capture goes through
`prepare_dummy_inputs`, which never stages. TODO.md `engram-ple`.
"""

from __future__ import annotations

import os

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
        """The table on disk, staged per step into fixed device buffers by `stage`; `forward` only reads them."""

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
            self.exl3_prefix = prefix
            # Fixed for the life of the engine, so a captured graph reads the same address every replay
            self._rows = torch.zeros(max(1, max_total_tokens), num_ngram_heads, embedding_dim,
                                     dtype=params_dtype)
            self._table = None

        def bind(self, model_dir: str) -> None:
            """Open an n-gram table for streaming: exllamav3's own loader and RowTable, on a table file
            (`util/convert_ngram.py` output) or a directory holding one (`_ngram_source`)."""
            from exllamav3.loader.safetensors import SafetensorsCollection
            from exllamav3.modules.quant.exl3_lib.ngram_codec import ROW_DIM, words_per_row
            from exllamav3.modules.row_table import RowTable

            stc = SafetensorsCollection(model_dir)
            # The checkpoint names the layer as exllamav3 does; vLLM's prefix shares its tail from "layers."
            tail = self.exl3_prefix[self.exl3_prefix.index("layers."):]
            bases = sorted({k.rsplit(".", 1)[0].split(".shard_")[0] for k in stc.tensor_file_map
                            if k.endswith(".trellis") and k.rsplit(".", 1)[0].split(".shard_")[0].endswith(tail)})
            if len(bases) != 1:
                raise ValueError(f"EXL3 n-gram table for {self.exl3_prefix}: expected one trellis table ending in "
                                 f"{tail!r} under {model_dir}, found {bases}")
            key = bases[0]
            table = RowTable.find(stc, key, "trellis")
            get = lambda n: stc.get_tensor(f"{key}.{n}", "cpu", allow_bf16=True, no_defer=True)
            offsets, bias = get("head_offsets").long().contiguous(), get("head_bias")
            self._hash_params = {"offsets": offsets, "sizes": get("head_vocab_sizes").long(),
                                 "multipliers": get("layer_multipliers").long()}
            if offsets.shape[0] != self.num_ngram_heads or bias.shape[0] != self.num_ngram_heads:
                raise ValueError(f"{key}: {offsets.shape[0]} offsets and {bias.shape[0]} biases for "
                                 f"{self.num_ngram_heads} heads")
            table.open(stc, True)
            K = (table.row_words - 1) * 16 // ROW_DIM
            if table.row_words != words_per_row(K) or self.embedding_dim != ROW_DIM:
                raise ValueError(f"{key}: {table.row_words} words per row is not a K-bit {ROW_DIM}-wide ring")
            self._stc, self._table, self._K = stc, table, K
            self._head_offsets = offsets
            self._head_bias = bias.half().contiguous().to(self._rows.device)
            logger.info("EXL3 n-gram table %s: %d rows at K=%d (%d bytes a row) streamed from %s",
                        key, table.num_rows, K, table.row_words * 2, model_dir)

        def check_layout(self, module) -> None:
            """vLLM hashes tokens to global row ids itself; its hash must be the one the table was built for,
            or every lookup reads the wrong rows and nothing fails"""
            mine = {"offsets": module.ngram_heads_offsets, "sizes": module.ngram_heads_vocab_sizes,
                    "multipliers": module.layer_multipliers}
            for name, theirs in self._hash_params.items():
                if not torch.equal(mine[name].cpu().long(), theirs):
                    raise ValueError(f"{self.exl3_prefix}: vLLM's n-gram {name} differ from the checkpoint "
                                     f"table's ({mine[name].tolist()[:4]} against {theirs.tolist()[:4]})")

        def stage(self, ngram_ids: torch.Tensor) -> None:
            """This step's rows, read from disk, decoded on the device into the fixed buffer. ngram_ids:
            (tokens, heads) global row ids on the device, after the stream sync that makes them valid."""
            from exllamav3.ext import exllamav3_ext as ext

            T = ngram_ids.shape[0]
            ids = ngram_ids.reshape(-1).to("cpu", torch.int64).contiguous()
            n = ids.numel()
            if n == 0:
                return
            if int(ids.max()) >= self._table.num_rows or int(ids.min()) < 0:
                raise IndexError(f"{self.exl3_prefix}: n-gram row out of range for {self._table.num_rows} rows")
            offsets, H = self._head_offsets, self.num_ngram_heads

            def resolve(ids, pin):
                uids, inverse = torch.unique(ids, sorted=True, return_inverse=True)
                U = uids.numel()
                pin.uids[:U].copy_(uids)
                pin.inverse[:n].copy_(inverse)
                pin.heads[:U].copy_((torch.searchsorted(offsets, uids, right=True) - 1).clamp_(0, H - 1))
                return pin.uids[:U], False

            def decode(packed, pin, U):
                heads = pin.heads[:U].to(packed.device, non_blocking=True)
                rows = torch.empty((U, self.embedding_dim), dtype=torch.half, device=packed.device)
                ext.ngram_dequant(packed, self._K, heads, self._head_bias, rows, False)
                return rows

            out = self._table.lookup(ids, n, resolve, decode, self._rows.device)
            self._rows[:T].view(n, self.embedding_dim).copy_(out)
            return ids.view(T, -1), out.view(T, -1, self.embedding_dim)

        def allocate_embedding_weight(self, num_embeddings, embedding_dim, dtype):
            return torch.empty(0, embedding_dim, dtype=dtype)

        def start_prefetch(self, ngram_ids):
            return None

        def finalize_prefetch(self, total_tokens):
            raise RuntimeError("EXL3 n-gram table: rows are staged before the forward, not prefetched")

        def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
            return self._rows[: ngram_ids.shape[0]]

        def close(self) -> None:
            if self._table is not None:
                self._table.close()
                self._table = None

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
    _patch_model_state(nvidia, Table)


NGRAM_FILE = "ngram_embedding.safetensors"


def ngram_source(model: str, revision: str | None) -> str:
    """The n-gram table to serve: the checkpoint's own, or another one chosen by the user.

    The table's bit depth trades host memory and disk or link bandwidth, not VRAM, so it need not follow the
    body's (turboderp publishes each rate with its own and endorses mixing them). `EXL3_NGRAM_REVISION=<branch>`
    takes that branch's table from the same repo, fetching only that file; `EXL3_NGRAM_FILE=<path>` takes a local
    one (`util/convert_ngram.py` output). Any K works (it is read from the row width), and `check_layout` refuses
    a table hashed for another model.

    Called twice: by `EXL3Config.maybe_update_config` in the frontend, so a download happens there, before the
    engine starts and vLLM's startup watchdog (VLLM_ENGINE_READY_TIMEOUT_S, 600 s) runs -- a table is 18-30 GiB --
    and by `bind`'s caller in the engine, where it then resolves from the cache."""
    path = os.environ.get("EXL3_NGRAM_FILE")
    if path:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"EXL3_NGRAM_FILE={path}: no such file")
        return path
    override = os.environ.get("EXL3_NGRAM_REVISION")
    if override:
        import re

        from huggingface_hub import hf_hub_download

        repo = model
        if os.path.isdir(repo):
            m = re.search(r"/models--([^/]+?)--([^/]+)/snapshots/", os.path.abspath(repo) + "/")
            if not m:
                raise ValueError(f"EXL3_NGRAM_REVISION needs a Hub model id to fetch from; {repo} is a local "
                                 "directory outside the HF cache (use EXL3_NGRAM_FILE)")
            repo = f"{m.group(1)}/{m.group(2)}"
        return hf_hub_download(repo, NGRAM_FILE, revision=override)
    if os.path.isdir(model):
        own = os.path.join(model, NGRAM_FILE)
        return own if os.path.isfile(own) else model
    from huggingface_hub import hf_hub_download

    # The body's own revision: EXL3 repos keep each rate on a branch, and `main` carries no table
    return hf_hub_download(model, NGRAM_FILE, revision=revision)


def _patch_model_state(nvidia, Table) -> None:
    """Stage each step's n-gram rows after vLLM prepares the step's inputs, as upstream's disk path does
    (`stage_checkpoint_rows`): the ids hash tokens the previous step sampled on the GPU, so one sync first."""
    from vllm.models.qwen4_exp.nvidia import model_state

    State = model_state.Qwen4ExpModelState
    if getattr(State, "_exl3_staging_patched", False):
        return
    original = State.prepare_inputs

    def tables(self):
        found = getattr(self, "_exl3_tables", None)
        if found is None:
            found = []
            for module in self.model.modules():
                table = getattr(module, "ngram_embedding", None)
                if isinstance(table, Table):
                    if table._table is None:
                        from vllm.config import get_current_vllm_config

                        mc = get_current_vllm_config().model_config
                        table.bind(ngram_source(mc.model, mc.revision))
                        table.check_layout(module)
                    found.append((module, table))
            self._exl3_tables = found
        return found

    def prepare_inputs(self, input_batch, req_states):
        model_inputs = original(self, input_batch, req_states)
        found = tables(self) if getattr(self, "uses_ngram_embedding", False) else []
        if found:
            ids = [module.compute_ngram_ids(input_batch.input_ids, model_inputs["query_start_loc"],
                                            model_inputs["ngram_context"]) for module, _ in found]
            torch.cuda.current_stream().synchronize()
            for (_, table), step_ids in zip(found, ids):
                staged = table.stage(step_ids)
                dump = os.environ.get("EXL3_NGRAM_DUMP")
                if dump and staged is not None and input_batch.num_reqs == 1 and not os.path.exists(dump) \
                        and input_batch.input_ids[: int(model_inputs["query_start_loc"][1])].unique().numel() > 8:
                    # Debug: one single-request step's tokens, ids and decoded rows, for a check against exllamav3
                    qsl = model_inputs["query_start_loc"]
                    torch.save({"tokens": input_batch.input_ids[: int(qsl[1])].cpu(),
                                "context": model_inputs["ngram_context"][0].cpu(),
                                "ids": staged[0][: int(qsl[1])], "rows": staged[1][: int(qsl[1])].cpu()}, dump)
        return model_inputs

    State.prepare_inputs = prepare_inputs
    State._exl3_staging_patched = True
