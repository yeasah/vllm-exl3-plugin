"""HY4 Flash's Engram n-gram tables for EXL3 checkpoints, read from disk a step at a time.

An EXL3 HY4 Flash checkpoint (turboderp's Hy4-flash-exl3) stores its four n-gram tables -- 63.4B parameters, 660M
rows of 96 -- outside the indexed shards, in `ngram_embedding.safetensors`, as one K-bit trellis ring per row plus a
per-head bias (exllamav3's own format, `ngram_codec`). vLLM's `FlashEmbedding` would hold each table as a dense
BF16/FP8 parameter, pinned in host memory under `engram_config.cpu_offload` (about 127 GB for these), and its loader
never reads that file.

Here the tables stay on disk. vLLM computes each step's buckets in `FlashNgramHashState.prepare_hashes_with_output`,
an eager break under breakable capture. Right after it, under the same break, the plugin reads the rows this rank's
heads need through exllamav3's `RowTable`, decodes them with `ngram_dequant` into one fixed device buffer per table,
and rewrites the buckets in place into indices of that buffer, keeping the -1 of an empty bucket. `FlashEmbedding`
then reports the buffer as its table, with the bucket range `[0, rows)`, so vLLM's `lookup_norm`, projections and
`gated_update` run unchanged: the first reads the buffer, the last only tests the buckets' sign.

Before the first step each table's layout is checked against vLLM's hash (head sizes, offsets, multipliers): a table
hashed for another layout would serve wrong rows for every token and fail nothing.

A FULL CUDA graph runs the eager break inside the capture, where a host read would be recorded once and replayed
stale; staging refuses to run under capture rather than do that.
"""

from __future__ import annotations

import functools

import torch

from .log import init_logger

logger = init_logger(__name__)


def _key(layer_id: int) -> str:
    """The checkpoint's table key for an Engram layer (1-based, as `engram_layer_ids`)."""
    return f"model.layers.{layer_id - 1}.engram_layer.multi_head_embedding.embedding"


class DiskTable:
    """One layer's table on disk, staged per step into `rows`, a buffer that never moves."""

    def __init__(self, emb, layer_id: int, index: int, max_tokens: int, device: torch.device):
        self.layer_id, self.index = layer_id, index
        self.head_start, self.head_end = emb.head_start, emb.head_end
        self.dim = emb.dim
        self.num_rows = emb.num_embeddings
        heads = self.head_end - self.head_start
        # Every (token, local head) of the largest step gets its own row; fixed for a captured graph's sake
        self.rows = torch.zeros(max(1, max_tokens * heads), self.dim, dtype=torch.half, device=device)
        self._table = None
        self._checked = False

    def bind(self, path: str) -> None:
        from exllamav3.loader.safetensors import SafetensorsCollection
        from exllamav3.modules.quant.exl3_lib.ngram_codec import words_per_row
        from exllamav3.modules.row_table import RowTable

        stc = SafetensorsCollection(path)
        key = _key(self.layer_id)
        table = RowTable.find(stc, key, "trellis")
        if table is None:
            raise ValueError(f"EXL3 n-gram table: no trellis table {key} under {path}")
        get = lambda n: stc.get_tensor(f"{key}.{n}", "cpu", allow_bf16=True, no_defer=True)
        self._offsets = get("head_offsets").long().contiguous()
        self._sizes = get("head_vocab_sizes").long()
        self._multipliers = get("layer_multipliers").long()
        table.open(stc, True)
        # The ring is as wide as the row (96 here; the codec's ROW_DIM, 160, is QFN's), K from the words it takes
        K = (table.row_words - 1) * 16 // self.dim
        if not 1 <= K <= 8 or table.row_words != words_per_row(K, self.dim):
            raise ValueError(f"{key}: {table.row_words} words a row is not a K-bit {self.dim}-wide ring")
        if table.num_rows != self.num_rows:
            raise ValueError(f"{key}: {table.num_rows} rows in the table, {self.num_rows} in vLLM's layout")
        self._stc, self._table, self._K = stc, table, K
        self._head_bias = get("head_bias").half().contiguous().to(self.rows.device)
        logger.info("EXL3 n-gram table %s: %d rows at K=%d, heads %d-%d staged on this rank, streamed from %s",
                    key, table.num_rows, K, self.head_start, self.head_end, path)

    def check_layout(self, state) -> None:
        """vLLM hashes tokens to buckets itself; its layout must be the one this table was built for."""
        layout = state.layout
        mine = {"sizes": torch.tensor([p for order in layout.primes[self.index] for p in order]),
                "offsets": torch.as_tensor(layout.offsets[self.index]).long(),
                "multipliers": torch.as_tensor(layout.multipliers[self.index]).long()}
        theirs = {"sizes": self._sizes, "offsets": self._offsets, "multipliers": self._multipliers}
        for name in mine:
            if not torch.equal(mine[name].cpu(), theirs[name]):
                raise ValueError(f"{_key(self.layer_id)}: vLLM's Engram {name} differ from the checkpoint table's "
                                 f"({mine[name].tolist()[:4]} against {theirs[name].tolist()[:4]})")
        self._checked = True

    def stage(self, buckets: torch.Tensor) -> None:
        """Read and decode this step's rows, and point `buckets` (`[T, heads]`, this rank's columns, in place) at
        them. Empty buckets (-1) stay -1."""
        from exllamav3.ext import exllamav3_ext as ext

        ids = buckets.to("cpu", torch.int64)
        valid = ids >= 0
        flat = ids[valid].contiguous()
        n = flat.numel()
        if n == 0:
            return
        if n > self.rows.shape[0]:
            raise RuntimeError(f"{_key(self.layer_id)}: {n} buckets in one step, room for {self.rows.shape[0]}")
        if int(flat.max()) >= self.num_rows:
            raise IndexError(f"{_key(self.layer_id)}: bucket out of range for {self.num_rows} rows")
        offsets = self._offsets
        H = offsets.shape[0]

        def resolve(ids, pin):
            uids, inverse = torch.unique(ids, sorted=True, return_inverse=True)
            U = uids.numel()
            pin.uids[:U].copy_(uids)
            pin.inverse[:n].copy_(inverse)
            pin.heads[:U].copy_((torch.searchsorted(offsets, uids, right=True) - 1).clamp_(0, H - 1))
            return pin.uids[:U], False

        def decode(packed, pin, U):
            heads = pin.heads[:U].to(packed.device, non_blocking=True)
            out = torch.empty((U, self.dim), dtype=torch.half, device=packed.device)
            ext.ngram_dequant(packed, self._K, heads, self._head_bias, out, False)
            return out

        self.rows[:n].copy_(self._table.lookup(flat, n, resolve, decode, self.rows.device))
        remap = torch.full(ids.shape, -1, dtype=torch.int32)
        remap[valid] = torch.arange(n, dtype=torch.int32)
        buckets.copy_(remap.to(buckets.device, non_blocking=True))

    def close(self) -> None:
        if self._table is not None:
            self._table.close()
            self._table = None


def attach(model, vllm_config) -> None:
    """Give an EXL3 HYV4FlashModel's Engram layers disk tables: called from its `__init__` (stock_compat)."""
    state = getattr(model, "engram_hash", None)
    if state is None:
        return
    max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    device = torch.device("cuda", torch.cuda.current_device())
    tables = []
    for index, layer_id in enumerate(state.layout.layer_ids):
        layer = model.layers[layer_id - 1]
        emb = getattr(getattr(layer, "engram_layer", None), "multi_head_embedding", None)
        if emb is None:     # a layer this pipeline rank does not hold
            continue
        table = DiskTable(emb, layer_id, index, max_tokens, device)
        emb.exl3_table = table
        # vLLM's lookup reads `[vocab_start, vocab_end)` of `tables()`: the buffer's rows, which staging points at
        emb.vocab_start, emb.vocab_end = 0, table.rows.shape[0]
        tables.append(table)
    state.exl3_tables = tables
    state.exl3_source = (vllm_config.model_config.model, vllm_config.model_config.revision)
    logger.info("EXL3 n-gram tables: %d layers on disk, %.0f MiB of staging buffers", len(tables),
                sum(t.rows.numel() * 2 for t in tables) / 2**20)


def _stage_all(state, output: torch.Tensor) -> None:
    tables = getattr(state, "exl3_tables", None)
    if not tables:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("EXL3 n-gram tables are read from disk each step, which a captured graph cannot replay; "
                           "serve with breakable or piecewise CUDA graphs, not FULL")
    if tables[0]._table is None:
        from .engram import ngram_source

        path = ngram_source(*state.exl3_source)
        for t in tables:
            t.bind(path)
    if not tables[0]._checked:
        for t in tables:
            t.check_layout(state)
    torch.cuda.current_stream().synchronize()   # the buckets come from this step's tokens, on the device
    for t in tables:
        t.stage(output[t.index, :, t.head_start:t.head_end])


def patch(engram_module) -> None:
    """Route an EXL3 HY4 Flash's Engram tables to disk (`engram_module`: vllm's hy_v4_flash.nvidia.engram)."""
    State, Embedding = engram_module.FlashNgramHashState, engram_module.FlashEmbedding
    if getattr(State, "_exl3_patched", False):
        return
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture

    decorated = State.prepare_hashes_with_output
    original = getattr(decorated, "__wrapped__", decorated)

    @eager_break_during_capture
    @functools.wraps(original)
    def prepare_hashes_with_output(self, input_ids, lookback, layer_name, output):
        original(self, input_ids, lookback, layer_name, output)
        _stage_all(self, output)

    original_tables = Embedding.tables

    @functools.wraps(original_tables)
    def tables(self):
        table = getattr(self, "exl3_table", None)
        return (table.rows, None) if table is not None else original_tables(self)

    State.prepare_hashes_with_output = prepare_hashes_with_output
    State._exl3_patched = True
    Embedding.tables = tables
