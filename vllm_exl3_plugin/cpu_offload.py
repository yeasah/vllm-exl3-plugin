"""CPU offload of EXL3-quantized weights.

vLLM's weight offloaders decide *what* to offload at construction time, before
checkpoint weights load:

- ``UVAOffloader`` (``--cpu-offload-gb``) builds its per-module whitelist by
  peeking ``next(module.parameters()).device`` in ``wrap_modules`` and skips any
  module whose first parameter is already on CPU.
- ``PrefetchOffloader`` (``--offload-group-size``) does the same, and additionally
  snapshots each selected parameter's storage immediately (Stage 2).

EXL3's construction-time view is useless. Its ``EXL3Parameter`` placeholders are
empty CPU tensors, so every quantized layer reads as "already on CPU" and is
skipped before any real weight exists, and the trellis/suh/svh tensors the layers
end up holding are never registered with the offloader. So ``--cpu-offload-gb``
reports ``0.01 GiB`` for an EXL3 checkpoint that offloads ``3.63 GiB`` as AWQ.

So placement is decided here, at the moment each tensor is stored: ``place`` for
dense linears and ``place_expert`` for routed experts copy an offloaded tensor
straight from the mmapped checkpoint into pinned host memory, and only resident
tensors go to the device. A checkpoint larger than VRAM therefore loads: peak
device use during load is the resident set, not the whole checkpoint. Offloading
after load instead (the earlier design) needed the whole checkpoint on the device
first, which bare Linux refuses, and stranded the freed pages in the caching
allocator. The budget and selectors are still vLLM's (``--cpu-offload-gb``,
``--cpu-offload-params``), read from its offloader singleton. No vLLM change.

Routed experts go into one pinned block per layer, projection and component, so a
layer's offloaded experts are one contiguous range for a staging copy, and a load
pins tens of blocks instead of thousands of tensors.

Prefill stages them: a batch over ``STAGE_MIN_TOKENS`` copies the layer's blocks
into one shared VRAM window and runs on a second set of pointer tables into it
(`build_staging`, `expert_tables`). Read over UVA instead, a 1024-token chunk
fetches each offloaded expert ~60 times, since exl3_moe_mm runs one row per
(token, expert) slot; staged, it is fetched once.

UVA only (``uva_offloading`` on); ``NoopOffloader`` means offload was not requested.
The ``EXL3_DEQUANTIZE=1`` reference path still offloads after load, through
``register_offload``, since the tensor it would offload only exists then.

See docs/expert-cache.md for the design this is phase 1 of, and docs/cpu-offload.md
for what offload costs and how to choose what to give it.
"""

from __future__ import annotations

import collections
import contextlib
import ctypes
import functools
import math
import os
import re
import time

import numpy
import torch
from vllm.model_executor.offloader import get_offloader
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from . import format
from .log import init_logger

logger = init_logger(__name__)


# Which EXL3 components may go to host memory, regardless of what the user's
# selectors match.
#
# `suh`/`svh` are deliberately absent. They are ~1.9% of expert bytes, so excluding
# them costs almost nothing in offload capacity and is never worse; measured on
# Qwen3.5-35B-A3B @2.00bpw, 2026-09-17, it is worth between 2.6% and 12.5% of
# throughput depending on what else is offloaded:
#
#   experts only:  19.1 -> 19.6 tok/s   (+2.6%; 1.34 ms, vs 0.75 ms of bytes)
#   + dense trellis: 8.0 ->  9.0 tok/s  (+12.5%; 13.9 ms, vs 1.06 ms of bytes)
#
# Same ~4.9 MB/token of scale vectors, ten times the cost. Both figures exceed what
# their bytes can explain -- they are thousands of few-KB dependent PCIe round trips
# per token, each blocking an expert's GEMM until it lands, where contiguous trellis
# streams at link speed -- but the gap between the two configurations is not
# understood. See TODO `cpu-offload`. Either way this is a property of the tensors,
# not a user preference, so it belongs here rather than in a selector.
#
# Deliberately a *subset* of `quant_config.stored_tensor_names()`: that list is
# what a checkpoint carries, this one is what is worth moving.
_OFFLOADABLE_COMPONENTS = frozenset({"trellis", "weight"})

_STATS: dict[str, int | bool] = {}


def _reset_stats() -> None:
    _STATS.clear()
    _STATS.update(requested=False, selective=False, bytes=0, tensors=0, layers=set(),
                  expert_bytes=0, moe_layers=set(), quota=None, disk_experts=0)


def cache_mode() -> str:
    """The expert cache's mode, from ``EXL3_EXPERT_SLOTS``: "auto" when unset (or
    "auto"), "off" for 0, "fixed" for a count or fraction. With the cache on, every
    eligible expert goes to the host and slots bring back what VRAM allows; off, the
    budget is spent greedily as before."""
    spec = os.environ.get("EXL3_EXPERT_SLOTS", "auto").strip() or "auto"
    if spec == "auto":
        return "auto"
    return "off" if float(spec) == 0 else "fixed"


def declare_moe_layer(prefix: str) -> None:
    """Count a routed-expert layer at construction, before any weight arrives, so the
    first expert block can be planned knowing how many layers will follow. A drafter's layers
    are not counted: they stay resident (`drafting`)."""
    if not _DRAFTING:
        _STATS["moe_layers"].add(prefix)


_reset_stats()


def _matches(params, full: str) -> bool:
    """Does this tensor's name satisfy the user's offload selectors?

    `params` is vLLM's set of parameter-name segments; empty means "everything is
    eligible". A ``re:`` prefix makes the remainder an unanchored `re.search`
    pattern, which is the only way to express a *conjunction* -- vLLM's own
    matching is an OR over dot-delimited substrings, so "expert trellises only"
    cannot be said with it. Note that `.` is a regex wildcard: ``re:experts.1.``
    matches more than it appears to.

    `full` is the reconstructed dotted name, wrapped in dots at both ends so a
    plain segment matches whole components only. Its tail is faithful to the
    checkpoint (``...experts.7.gate_proj.trellis``); its head is vLLM's
    post-mapping module path, which a `WeightsMapper` may rewrite on any bump --
    match on the tail. See docs/cpu-offload.md.
    """
    if not params:
        return True
    for pattern in params:
        if pattern.startswith("re:"):
            if re.search(pattern[3:], full):
                return True
        elif f".{pattern}." in full:
            return True
    return False


def layer_prefix(layer) -> str:
    """The module path selectors match against; routed experts carry it as `layer_name`."""
    return getattr(layer, "prefix", None) or getattr(layer, "layer_name", "")


#: Set while a speculative drafter loads (`plugin._patch_offload_summary`): nothing of it is offloaded
_DRAFTING = False


@contextlib.contextmanager
def drafting(on: bool):
    """Keep every EXL3 weight loaded under this resident.

    A drafter runs k times per step and is small, so its weights are what the VRAM is bought for; and
    the expert cache is sized for the target model's layers. Offloading an MTP head's experts used to
    give it a second staging window and hand the cache its one layer in place of the target's 40,
    leaving every target expert read over the link (2026-11-09, Qwen3.6-35B-A3B)."""
    global _DRAFTING
    saved, _DRAFTING = _DRAFTING, on
    try:
        yield
    finally:
        _DRAFTING = saved


def _uva_loader():
    """vLLM's offloader if it is UVA with acceleration available, else None."""
    if _DRAFTING:
        return None
    try:
        loader = get_offloader()
    except Exception:  # pragma: no cover - defensive; offloader is normally set
        return None

    name = loader.__class__.__name__
    if name == "NoopOffloader":
        # No --cpu-offload-gb / --offload-group-size: offload was not requested.
        return None
    if name != "UVAOffloader":
        logger.debug_once("cpu-offload: %s backend unsupported, ignoring", name)
        return None
    if not getattr(loader, "uva_offloading", False):
        # UVA is configured but UVA acceleration is unavailable; vLLM's own
        # functional_call fallback hits the same construction-time gap EXL3 does,
        # so there is nothing for us to do.
        return None

    _STATS["requested"] = True
    if getattr(loader, "cpu_offload_params", None):
        _STATS["selective"] = True
    return loader


def _count(loader, prefix: str, nbytes: int, tensors: int = 1, budget: bool = True) -> None:
    """Record an offloaded tensor. With `budget` False (experts under the cache) it does
    not spend vLLM's counter: `_host_quota` already holds the hosted experts to the
    budget, once, from what the towers left."""
    if budget:
        loader.cpu_offload_bytes += nbytes
    else:
        _STATS["expert_bytes"] += nbytes
    _STATS["bytes"] += nbytes
    _STATS["tensors"] += tensors
    _STATS["layers"].add(prefix)


def _pinned(shape, dtype) -> torch.Tensor:
    """Page-locked host memory of exactly this size, for the life of the process.

    Not ``pin_memory=True``: PyTorch's caching host allocator rounds every
    allocation up to a power of two, so a 33 MiB block holds 64 MiB of locked RAM
    and a 1.03 MiB trellis holds 2 MiB -- up to twice the offloaded bytes, on the
    hosts that have least to spare (gemma-4-26B 2.10bpw, 2.99 GiB offloaded:
    9.07 GiB engine RSS against 6.50).

    Not ``cudaHostRegister`` on a plain allocation either, though it pins exactly
    what it uses. In a serving process the GPU reads registered memory slowly:
    measured on a 5070 Ti, the same placement decoded 8% and prefilled 31% slower
    than from driver-allocated memory, while a fresh process showed no
    difference. The likely cause is page backing -- a long-running process with
    the checkpoint in page cache hands out scattered 4 KiB pages, and the
    kernel's scattered tile reads pay for each translation.

    So the driver allocates it, at exactly this size. Never freed: weights live
    as long as the process, and `_PINNED` keeps the tensors that address it.
    """
    nbytes = math.prod(shape) * torch.empty(0, dtype=dtype).element_size()
    if not nbytes:
        return torch.empty(shape, dtype=dtype, device="cpu")
    # The driver API acts on the calling thread's current context; cudaSetDevice
    # makes the primary context current, which the loader thread may not yet have.
    torch.cuda.set_device(torch.cuda.current_device())
    ptr = ctypes.c_void_p()
    rc = _libcuda().cuMemHostAlloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes), 0)
    if rc != 0:
        raise RuntimeError(
            f"cpu-offload: could not allocate {nbytes >> 20} MiB of page-locked host "
            f"memory (CUresult {rc}). Lower --cpu-offload-gb or free host RAM."
        )
    raw = (ctypes.c_uint8 * nbytes).from_address(ptr.value)
    t = torch.from_numpy(numpy.ctypeslib.as_array(raw)).view(dtype).view(shape)
    _PINNED.append(t)
    return t


_PINNED: list[torch.Tensor] = []


def free_pinned(t: torch.Tensor) -> int:
    """Return one `_pinned` allocation to the driver; returns its bytes, 0 if `t` is not one. The caller must have
    dropped every other tensor that addresses it, and nothing queued may still read it."""
    for i, held in enumerate(_PINNED):
        if held.data_ptr() == t.data_ptr():
            nbytes = held.numel() * held.element_size()
            del _PINNED[i]
            rc = _libcuda().cuMemFreeHost(ctypes.c_void_p(t.data_ptr()))
            if rc != 0:
                raise RuntimeError(f"cpu-offload: cuMemFreeHost failed (CUresult {rc})")
            return nbytes
    return 0


@functools.cache
def _libcuda():
    # The driver library, which every CUDA host has under this name; the runtime's
    # file name depends on how torch was installed.
    return ctypes.CDLL("libcuda.so.1")


def place(param, t: torch.Tensor) -> torch.Tensor:
    """Where one dense-linear sub-tensor lives: pinned host memory behind a UVA
    view if the selectors and budget take it, otherwise the device.

    `param` carries `exl3_prefix`, `exl3_name`, `exl3_device` and
    `exl3_offloadable` (False on the dequantize path). Mirrors
    ``UVAOffloader._maybe_offload_to_cpu`` for one tensor: the budget is checked
    *before* the tensor, as ``uva.py`` does, so the whole budget is spent.
    """
    if getattr(param, "exl3_host", False):
        # the embedding table bound for host (`embed_on_host`): kept on CPU until its method pads it and
        # pins it once, so it never passes through VRAM. A copy, not the loader's mmapped view
        return t.contiguous().clone()
    loader = (
        _uva_loader()
        if param.exl3_offloadable and param.exl3_name in _OFFLOADABLE_COMPONENTS
        else None
    )
    if (
        loader is None
        # only an explicit selector offloads dense tensors: each of their bytes costs every
        # token, where experts are a fraction per token and have the cache to hide behind.
        # The empty-selector default is towers and experts; see plugin._patch_offload_defaults
        or not loader.cpu_offload_params
        or not _matches(loader.cpu_offload_params, f".{param.exl3_prefix}.{param.exl3_name}.")
        or loader.cpu_offload_bytes >= loader.cpu_offload_max_bytes
    ):
        return t.to(param.exl3_device).contiguous()

    host = _pinned(t.shape, t.dtype)
    _host_copy(host, t)
    _count(loader, param.exl3_prefix, host.numel() * host.element_size())
    return get_accelerator_view_from_cpu_tensor(host)


#: ``EXL3_EMBED_HOST`` (default on; ``0`` keeps it in VRAM): the token embedding's table (trellis
#: ``et_table``, blockq ``bq_*``) lives in pinned host memory behind a UVA view, as llama.cpp keeps its input
#: layer on the CPU. A lookup reads only the rows it gathers -- a few KiB a token -- where in VRAM the whole
#: vocabulary sits idle (0.30 GiB of a tied 2B's 1.46). Measured at no cost: decode and prefill unchanged on the
#: 27B at TP=2, MiniCPM5-1B at batch 1 and a 3060 under WSL, logits bit-identical (docs/cpu-offload.md, "Phase 5:
#: the embedding table on host"). Independent of ``--cpu-offload-gb``: it spends no offload budget.
def embed_on_host() -> bool:
    return os.environ.get("EXL3_EMBED_HOST", "1").strip() != "0"


#: A hosted table's rows, padded to this many bytes. A row gather over UVA moves ~23 GB/s on aligned rows and
#: 2.7 on the 27B's 2600-byte trellis rows: unaligned, `index_select` loses its vector loads, and across the
#: link every narrow load is a round trip (8192 tokens: 7.8 ms against 0.9 padded). In VRAM it never showed.
HOST_ROW_ALIGN = 64


def host_resident(t: torch.Tensor) -> torch.Tensor:
    """`t` (on CPU, row-indexed) copied once into pinned host memory, returned as the UVA view kernels index.
    Rows come back padded to `HOST_ROW_ALIGN` bytes: a gather cuts what it selects back to the stored width."""
    t = t.contiguous()
    if t.dim() == 2:
        row = t.shape[1] * t.element_size()
        pad = (-row % HOST_ROW_ALIGN) // t.element_size() if row > HOST_ROW_ALIGN and HOST_ROW_ALIGN % t.element_size() == 0 else 0
        if pad:
            t = torch.nn.functional.pad(t, (0, pad))
    host = _pinned(t.shape, t.dtype)
    _host_copy(host, t)
    drain_host_copies()
    return get_accelerator_view_from_cpu_tensor(host)


def place_embedding(tensors: dict[str, torch.Tensor], label: str) -> dict[str, torch.Tensor]:
    """An embedding's row-indexed tensors (on CPU, as `place` kept them), placed: on host the ones whose rows are
    wider than `HOST_ROW_ALIGN`, in VRAM the rest. A narrow row (blockq's 16-byte ranges, 4 MiB in all) costs a
    link round trip per token for no saving; the wide ones (the table, blockq's 30-76 MiB of scales) move at the
    link's rate. Measured both ways: keeping the scales in VRAM too saved 1-10 us a lookup (2026-11-16)."""
    device = torch.device("cuda", torch.cuda.current_device())
    out, hosted, kept = {}, [], []
    for name, t in tensors.items():
        if t[0].numel() * t.element_size() > HOST_ROW_ALIGN:
            out[name] = host_resident(t)
            hosted.append((name, t.numel() * t.element_size()))
        else:
            out[name] = t.to(device)
            kept.append(name)
    nbytes = sum(n for _, n in hosted)
    _STATS["embed_host_bytes"] = _STATS.get("embed_host_bytes", 0) + nbytes
    if hosted:
        logger.info(
            "Token embedding %s on host: %.2f GiB pinned (%s), read over UVA%s (EXL3_EMBED_HOST=0 keeps it in VRAM)",
            label, nbytes / (1 << 30), ", ".join(n for n, _ in hosted),
            f"; {', '.join(kept)} in VRAM" if kept else "",
        )
    return out


class HostBlock:
    """Pinned host rows for one MoE layer's offloaded experts, for one projection
    and one component. `rows` maps expert id to row; `view` is the UVA view the
    pointer tables address, `host` the CPU side the loader writes. With the disk
    tier, `disk` holds the offloaded experts that have no row: they stay in the
    checkpoint and are read on use (expert_store)."""

    def __init__(self, rows: dict[int, int], shape, dtype, disk: frozenset = frozenset()):
        self.rows = rows
        self.disk = disk
        self.host = _pinned((len(rows), *shape), dtype)
        self.view = get_accelerator_view_from_cpu_tensor(self.host)


def disk_tier() -> float | str | None:
    """``EXL3_DISK_TIER``: what of the experts host RAM cannot hold stays in the checkpoint, read on use
    (docs/expert-cache.md, "A disk tier below host RAM"). Under the exclusive cache only, whose rows already move
    between experts. Unset or "auto" (the default): host rows are what --cpu-offload-gb and host RAM allow
    (`_host_quota`, margin included), the expert cache takes VRAM by its own policy, and only the rest spills to disk
    -- nothing when the host holds every offloaded expert. "0" or "off": no disk tier; what the host cannot hold stays
    resident in VRAM. A share in (0, 1]: at most that share of each layer in host rows, still within the host quota."""
    spec = os.environ.get("EXL3_DISK_TIER", "").strip().lower()
    if spec in ("", "auto"):
        return "auto"
    if spec in ("0", "off"):
        return None
    share = float(spec)
    if not 0 < share <= 1:
        raise ValueError(f"EXL3_DISK_TIER={spec}: auto, off, or a share of each layer's experts in host rows, in (0, 1]")
    return share


def _graph_path_ready() -> bool:
    """Whether this engine can run disk-tier layers: eager, or breakable capture with PIECEWISE decode graphs, in which
    the MoE op is an eager break (`plugin._disk_tier_graph_defaults`). Anything else captures across the routing
    readback the disk tier needs."""
    try:
        from vllm.config import get_current_vllm_config
        from vllm.config.compilation import CUDAGraphMode

        config = get_current_vllm_config()
        if config.model_config.enforce_eager:
            return True
        cc = config.compilation_config
        return bool(getattr(cc, "use_breakable_cudagraph", False)) and cc.cudagraph_mode in (
            CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE)
    except Exception:
        return False


def _spill_ready() -> str | None:
    """None if this engine can spill experts to disk, else why not; decided once."""
    if "spill_blocker" not in _STATS:
        from . import expert_cache

        why = None
        if cache_mode() == "off" or not expert_cache.EXCLUSIVE:
            why = "it needs the exclusive expert cache"
        elif not _graph_path_ready():
            why = ("CUDA graphs here would capture the MoE op (the disk tier needs breakable capture with PIECEWISE "
                   "decode graphs, or enforce_eager)")
        else:
            try:
                _model_root()
            except Exception as e:
                why = f"the checkpoint is not in a local directory ({e})"
        _STATS["spill_blocker"] = why
    return _STATS["spill_blocker"]


def _offloads_experts(config) -> bool:
    uva = config.offload_config.uva
    params = uva.cpu_offload_params or set()
    return uva.cpu_offload_gb > 0 and (not params or any("expert" in p for p in params))


def _multi_token_decode(config) -> bool:
    """Whether a decode step can carry more tokens than the per-row kernel handles well over host rows (4: its UVA cliff
    is at 5, `ops._LEAN_MAX_TOKENS`): max_num_seqs x (1 + speculative drafts) tokens a step, or ``--performance-mode
    throughput``. MTP with 2 drafts at 4 streams runs 12-token steps (Qwen3.6-35B-A3B 4.00bpw: 101 -> 118 t/s on the
    grouped path); plain decoding past 4 sequences reaches the cliff on its own."""
    if getattr(config, "performance_mode", None) == "throughput":
        return True
    spec = getattr(config, "speculative_config", None)
    drafts = (getattr(spec, "num_speculative_tokens", 0) or 0) if spec is not None else 0
    seqs = getattr(getattr(config, "scheduler_config", None), "max_num_seqs", 1) or 1
    return seqs * (1 + drafts) > 4


def throughput_offload_expected(config) -> bool:
    """At config time: wide decode batches (`_multi_token_decode`) with experts offloaded, which takes the graph path
    that lets decode batches over 4 tokens run grouped (`throughput_offload`)."""
    try:
        return _multi_token_decode(config) and _offloads_experts(config)
    except Exception:
        return False


def decide_throughput_offload() -> None:
    """At the end of load, while the config is current: whether decode batches over 4 tokens on offloaded layers take
    the grouped path's routed staging (one copy of each routed expert into VRAM) rather than the per-row kernel, which
    reads host rows over UVA and falls off a cliff past ~40 assignments: QFN 2.05bpw all in RAM, 386 ms a step at 8
    tokens against 172 grouped. Only where decode batches are wide by configuration (``--performance-mode
    throughput``, or speculative decoding), with experts offloaded, and in a graph path where the MoE op is an eager
    break: the eager break costs ~0.4 ms a token at batch 1, worth it only where batches are wide."""
    try:
        from vllm.config import get_current_vllm_config

        config = get_current_vllm_config()
        _STATS["throughput_offload"] = (_multi_token_decode(config) and _STATS.get("expert_bytes", 0) > 0
                                        and _graph_path_ready())
    except Exception:
        _STATS["throughput_offload"] = False


def throughput_offload() -> bool:
    return bool(_STATS.get("throughput_offload"))


def spill_expected(config) -> bool:
    """At config time, whether the automatic disk tier will spill: the routed experts' bytes exceed what
    --cpu-offload-gb and host RAM (less the margin) can hold, with a fifth to spare, since load decides at the first
    expert, when MemAvailable sits below what it is now (loading's own transient memory). Erring toward spilling
    costs the graph path a few percent (breakable PIECEWISE graphs and an eager break per MoE layer); erring the other
    way leaves load to keep the rest resident, as without the disk tier."""
    try:
        mode = disk_tier()
        if mode is None:
            return False
        if isinstance(mode, float):
            return True
        uva = config.offload_config.uva
        if not _offloads_experts(config):
            return False
        mc = config.model_config
        census = _checkpoint_census(mc.model, mc.revision, getattr(config.load_config, "download_dir", None))
        if not census or not census.get("experts"):
            return False
        ranks = max(config.parallel_config.tensor_parallel_size, 1)
        want = census["experts"] / ranks
        free = (_mem_available() - max(_HOST_MARGIN, _mem_total() // 10)) / ranks
        room = min(free, uva.cpu_offload_gb * (1 << 30))
        return want > 0.8 * room
    except Exception:
        return False


def _plan_block(param, t: torch.Tensor, label: str) -> HostBlock | None:
    """Which of this layer's experts go to the host, for one projection.

    Decided once, at the first expert to arrive, over every expert id in
    ascending order, so the outcome does not depend on the order the checkpoint
    delivers them. Without the cache, experts whose names match the selectors are
    taken while the budget lasts, each checked before it is taken, as for dense
    tensors. With it, every matching expert is taken, up to the same share of each
    layer that host RAM allows.
    """
    if param.exl3_name not in _OFFLOADABLE_COMPONENTS:
        return None
    loader = _uva_loader()
    if loader is None:
        return None
    nbytes = t.numel() * t.element_size()
    cached = cache_mode() != "off"
    mode = disk_tier()
    if isinstance(mode, float):
        from . import expert_cache

        if not cached or not expert_cache.EXCLUSIVE:
            raise ValueError("EXL3_DISK_TIER needs the exclusive expert cache (EXL3_CACHE_EXCLUSIVE on, slots not 0)")
    elif not cached:
        mode = None
    if mode is not None:
        explicit = isinstance(mode, float)
        quota = _host_quota(param.exl3_num_experts, nbytes)
        taken = [e for e in range(param.exl3_num_experts)
                 if _matches(loader.cpu_offload_params, f".{param.exl3_prefix}.{e}.{label}_proj.{param.exl3_name}.")]
        n = min(math.ceil(mode * param.exl3_num_experts), quota) if explicit else quota
        if taken and n < len(taken) and (explicit or _STATS.get("host_limited")):
            blocker = _spill_ready()
            if blocker is None:
                # the host's rows, then every other matching expert to disk; the expert cache's slots fill from the rows
                # and the rows refill from disk (`ExclusiveCache._backfill`), so only what neither holds stays there
                rows = taken[:n]
                for _ in rows:
                    _count(loader, param.exl3_prefix, nbytes, budget=False)
                if label == "gate":   # once per expert, not once per projection
                    _STATS["disk_experts"] += len(taken) - len(rows)
                return HostBlock({e: row for row, e in enumerate(rows)}, t.shape, t.dtype, frozenset(taken[len(rows):]))
            if explicit:
                raise ValueError(f"EXL3_DISK_TIER={mode}: cannot spill to disk: {blocker}")
            if not _STATS.get("spill_warned"):
                _STATS["spill_warned"] = True
                logger.warning("cpu-offload: not spilling experts to disk, so what host RAM cannot hold stays resident "
                               "in VRAM: %s", blocker)
    quota = _host_quota(param.exl3_num_experts, nbytes) if cached else param.exl3_num_experts
    taken = []
    for e in range(param.exl3_num_experts):
        if len(taken) >= quota or (
            not cached and loader.cpu_offload_bytes >= loader.cpu_offload_max_bytes
        ):
            break
        name = f".{param.exl3_prefix}.{e}.{label}_proj.{param.exl3_name}."
        if _matches(loader.cpu_offload_params, name):
            taken.append(e)
            _count(loader, param.exl3_prefix, nbytes, budget=not cached)
    if not taken:
        return None
    return HostBlock({e: row for row, e in enumerate(taken)}, t.shape, t.dtype)


_UNPLANNED = object()

#: Host RAM left free when sizing the expert cache's host copy, at least.
_HOST_MARGIN = 4 << 30


#: Host rows the exclusive cache keeps free to swap through, beyond those it must hold. 17 per layer kept a
#: 3060's hit rate and decode level with 103 (Qwen3.6-35B-A3B, docs/expert-cache.md).
_SPARE_ROWS = int(os.environ.get("EXL3_CACHE_SPARE_ROWS", "16"))


_ROUTED = re.compile(r"\.experts\.\d+\.")


def _checkpoint_root(model: str, revision: str | None, download_dir: str | None) -> str:
    """The local directory holding a checkpoint's shards; raises when they are not all on disk."""
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model, revision=revision, cache_dir=download_dir, local_files_only=True,
                             allow_patterns=["*.safetensors", "*.safetensors.index.json"])


@functools.lru_cache(maxsize=4)
def _checkpoint_census(model: str, revision: str | None, download_dir: str | None) -> dict[str, int] | None:
    """Stored bytes by part (routed experts, mtp, visual, embed, other), from the local shards' headers.

    By the time an expert weight arrives every shard is on disk, so this reads nothing remote. None when the
    shards cannot be found, which the callers take as "not known"."""
    import json
    import struct

    try:
        root = _checkpoint_root(model, revision, download_dir)
        index = os.path.join(root, "model.safetensors.index.json")
        if os.path.isfile(index):
            with open(index) as f:
                files = sorted(set(json.load(f)["weight_map"].values()))
        else:
            files = ["model.safetensors"]
        parts = collections.Counter()
        for name in files:
            with open(os.path.join(root, name), "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(n))
            for key, info in header.items():
                if key == "__metadata__":
                    continue
                a, b = info["data_offsets"]
                part = ("mtp" if key.startswith("mtp.") else "visual" if ".visual." in key or key.startswith("visual.")
                        else "experts" if _ROUTED.search(key) else "embed" if "embed_tokens" in key else "other")
                parts[part] += b - a
        return dict(parts)
    except Exception as e:  # pragma: no cover - defensive; the caller falls back
        logger.debug("cpu-offload: no checkpoint census (%s)", e)
        return None


def _census():
    try:
        from vllm.config import get_current_vllm_config

        config = get_current_vllm_config()
        mc = config.model_config
        return _checkpoint_census(mc.model, mc.revision, getattr(config.load_config, "download_dir", None)), config
    except Exception:
        return None, None


def _rank_expert_bytes() -> int | None:
    """This rank's routed-expert bytes, as stored (projections and layers need not match in size)."""
    census, config = _census()
    if not census or not census.get("experts"):
        return None
    widths, rank = _uneven_widths()
    if widths is not None:
        return census["experts"] * widths[rank] // sum(widths)
    return census["experts"] // max(config.parallel_config.tensor_parallel_size, 1)


#: VRAM the exclusive cap leaves to vLLM beyond the weights and MoE activations it can count: the CUDA context,
#: graphs, load-time buffers (Qwen4Exp's indexer takes 0.75 GiB) and the KV reserve. Generous on purpose:
#: overstating it only hosts more, and the slots take back what is left at KV sizing; understating it is what ran a
#: 16 GiB card out of memory (2026-11-18)
_VRAM_ALLOWANCE = 2 << 30


def _moe_activation_bytes(config) -> int:
    """The routed MoE's activation peak at the batched-token budget: two fp32 buffers of tokens x top-k x this
    rank's intermediate width (one is the 800 MiB profiling allocation that failed at 8192 tokens, top-10, 640)."""
    try:
        text = config.model_config.hf_text_config
        tokens = config.scheduler_config.max_num_batched_tokens
        topk = getattr(text, "num_experts_per_tok", None) or getattr(text, "moe_topk", None) or 8
        width = getattr(text, "moe_intermediate_size", None) or getattr(text, "intermediate_size", 0)
        widths, rank = _uneven_widths()
        width = widths[rank] if widths is not None else width // max(config.parallel_config.tensor_parallel_size, 1)
        return 2 * tokens * topk * width * 4 * 2
    except Exception:
        return 0


def _vram_for_experts(loader) -> int | None:
    """Bytes of this rank's VRAM the routed experts can have at most, resident rows, slots and staging windows
    together; None when it cannot be estimated."""
    census, config = _census()
    if not census or not torch.cuda.is_available():
        return None
    tp = max(config.parallel_config.tensor_parallel_size, 1)
    params = set(getattr(loader, "cpu_offload_params", None) or ())
    other = census.get("other", 0)
    if not embed_on_host():
        other += census.get("embed", 0)
    if config.speculative_config is not None:
        other += census.get("mtp", 0)   # the drafter stays resident (`drafting`)
    mm = getattr(config.model_config, "multimodal_config", None)
    if mm is not None and not getattr(mm, "language_model_only", False) and "visual" not in params:
        other += census.get("visual", 0)
    total = torch.cuda.mem_get_info()[1]
    return max(_usable_vram() - other // tp - max(_VRAM_ALLOWANCE, total // 10) - _moe_activation_bytes(config), 0)


def _usable_vram() -> int:
    """This device's memory times --gpu-memory-utilization."""
    from vllm.config import get_current_vllm_config

    return int(torch.cuda.mem_get_info()[1] * gpu_utilization(get_current_vllm_config().cache_config))


def gpu_utilization(cache_config) -> float:
    """The effective `gpu_memory_utilization`. Since v0.34 the field stays None when unset and the default depends on
    the sizing mode (1.0 with the extensible KV cache, 0.9 without), which `resolved_gpu_memory_utilization` applies;
    reading the field crashed offload planning on any config that left it unset (Laguna-XS-2.1, 2026-11-22)."""
    resolved = getattr(cache_config, "resolved_gpu_memory_utilization", None)
    if resolved is not None:
        return float(resolved)
    value = getattr(cache_config, "gpu_memory_utilization", None)
    return float(value) if value is not None else 0.9


def _exclusive_need(experts: int, loader) -> int | None:
    """Experts per layer worth hosting under the exclusive cache, when that is known at load: with
    `--max-model-len auto`, `target_share` of the experts can load resident, so host rows are needed only for the
    rest plus a few spare to swap through; the late slots then grow over those rows as far as VRAM allows
    (`expert_cache._build_late`: slots first, since 2026-11-26). The staging windows are sized to the rows. None when it is not known (a fixed length leaves the slots to KV sizing), or
    when selectors may host projections unevenly, which falls back to the inclusive cache."""
    from . import expert_cache

    if cache_mode() != "auto" or not expert_cache.EXCLUSIVE:
        return None
    params = getattr(loader, "cpu_offload_params", None) or ()
    if any(p.startswith("re:") or "proj" in p for p in params):
        return None
    try:
        from vllm.config import get_current_vllm_config

        if getattr(get_current_vllm_config().model_config, "original_max_model_len", None) != -1:
            return None
    except Exception:
        return None
    streams = float(os.environ.get("EXL3_KV_RESERVE", "1.0"))
    in_vram = expert_cache.target_in_vram(experts, streams)
    need = max(0, min(experts, experts - in_vram + _SPARE_ROWS))
    # The target is a decode-speed goal and assumes VRAM can hold it; a 16 GiB card cannot hold 40% of 23 GiB of
    # experts beside the rest, and resident rows are fixed once loaded. The cap is only a saving (smaller staging
    # windows), and the VRAM it is checked against is an estimate that has missed by 1.2 GiB (load-time buffers),
    # so it applies only with a quarter of usable VRAM to spare; otherwise everything offered is hosted and the
    # slots take what VRAM is left at KV sizing (Qwen3.8-Flash-Next 1.66bpw on one card: OOM -> 119K tokens of KV)
    rank_bytes, room = _rank_expert_bytes(), _vram_for_experts(loader)
    if rank_bytes is None or room is None:
        return None
    L = max(len(_STATS["moe_layers"]), 1)
    row = rank_bytes / (L * experts)   # one expert's three projections in one layer
    used = row * (L * (experts - need + _SPARE_ROWS) + planned_windows(L) * need)
    if used + _usable_vram() // 4 > room:
        logger.info(
            "cpu-offload: not capping hosting at the exclusive cache's target: its %.2f GiB of experts in VRAM "
            "would not fit with a margin (~%.2f GiB for experts); hosting what --cpu-offload-gb allows",
            used / (1 << 30), room / (1 << 30))
        return None
    return need


def _uneven_widths():
    """(every rank's expert intermediate width, this rank) under an uneven expert split, else (None, None)"""
    try:
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

        from .quantization.fused_moe import rank_widths

        tp_size = get_tensor_model_parallel_world_size()
        widths = rank_widths(tp_size) if tp_size > 1 else None
        return (widths, get_tensor_model_parallel_rank()) if widths else (None, None)
    except Exception:
        return None, None


def _host_quota(experts: int, nbytes: int) -> int:
    """How many experts per layer and projection the host holds, decided once.

    As many as --cpu-offload-gb allows, after what the towers took (they are offloaded
    first and spend the same budget), and as host RAM allows; the same share of every
    layer, so placement stays even. The rest stay resident. The flag is a cap on host
    memory, as vLLM documents it: on WSL that memory is pinned against a system-wide WDDM
    budget, so hosting more than asked is not free. Assumes the three projections'
    trellises are the same size, which holds for gate, up and down (k x n either way)."""
    if _STATS["quota"] is None:
        layers = max(len(_STATS["moe_layers"]), 1)
        # From the shards' headers: the first tensor times three overstated 1.66bpw's 22.71 GiB as 28.12, whose
        # projections differ in size (2026-11-18)
        want = _rank_expert_bytes() or layers * experts * 3 * nbytes
        # The margin protects the host, so it is taken once, not once per rank: per rank it reserved 8 GiB of a
        # 40 GiB host at TP=2 and left Qwen4Exp's 3.05bpw body short of host rows (2026-11-17)
        free = (_mem_available() - max(_HOST_MARGIN, _mem_total() // 10)) // _local_ranks()
        loader = _uva_loader()
        budget = (max(loader.cpu_offload_max_bytes - loader.cpu_offload_bytes, 0)
                  if loader is not None else want)
        room = max(min(free, budget), 0)
        share = 1.0 if want <= room else room / want
        widths, rank = _uneven_widths()
        if widths is not None:
            # An uneven expert split (fused_moe.rank_widths): the ranks' expert bytes differ, so an equal budget
            # each would leave the widest rank the most in VRAM -- 60% of Qwen4Exp's at TP=2, which ran rank 0
            # out of VRAM. The same total (the budget, or host RAM, times the ranks) is shared so that every
            # rank keeps the same expert bytes in VRAM: the wider rank hosts more, the narrower less
            wants = [want * w / widths[rank] for w in widths]
            resident = max((sum(wants) - room * len(widths)) / len(widths), 0)
            hosted = min(max(want - resident, 0), want)
            share = hosted / want if want else 1.0
        limited = share < 1.0
        _STATS["quota"] = math.floor(share * experts)
        need = _exclusive_need(experts, loader)
        capped = need is not None and need < _STATS["quota"]
        if capped:
            _STATS["quota"] = need
        # host RAM or the budget, not the cache's cap, is what leaves experts out: the automatic disk tier takes them
        _STATS["host_limited"] = limited and not capped
        if capped or limited:
            # One line, after every decision: each used to log as it was made, and later ones contradicted earlier
            q = _STATS["quota"]
            split = f" (rank {rank}, {widths[rank]} of {sum(widths)} wide)" if widths is not None else ""
            if capped:
                why = (f"the exclusive expert cache keeps {experts - q + _SPARE_ROWS} in VRAM at --max-model-len auto "
                       f"and swaps through {_SPARE_ROWS} spare rows")
                if widths is not None:
                    why += ", with every rank spending the same bytes on it"
            else:
                why = "{} allows {:.2f} GiB a rank{}".format(
                    "--cpu-offload-gb" if budget <= free else "host RAM", room / (1 << 30),
                    ", shared so every rank keeps the same expert bytes in VRAM" if widths is not None else "")
            spills = not capped and disk_tier() is not None and _spill_ready() is None
            (logger.info if capped or spills else logger.warning)(
                "cpu-offload: hosting %d of %d experts per layer%s, %.2f of %.2f GiB; %.2f GiB %s: %s",
                q, experts, split, want * q / experts / (1 << 30), want / (1 << 30),
                want * (experts - q) / experts / (1 << 30),
                "spill to disk, less what the expert cache's slots take" if spills else "stay resident", why)
    return _STATS["quota"]


def _meminfo(key: str) -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) << 10
    return 0


def _mem_available() -> int:
    return _meminfo("MemAvailable")


def _mem_total() -> int:
    return _meminfo("MemTotal")


def _local_ranks() -> int:
    """Workers on this host, which share its RAM."""
    try:
        from vllm.distributed import parallel_state

        return max(parallel_state.get_tensor_model_parallel_world_size(), 1)
    except Exception:
        return 1


def place_expert(param, t: torch.Tensor, label: str, expert_id: int) -> torch.Tensor:
    """Where one routed expert's sub-tensor lives: a row of this projection's
    host block if it was chosen for offload, otherwise the device.

    `param` is the layer's collector; it carries `exl3_prefix`,
    `exl3_num_experts`, `exl3_device`, and `exl3_blocks`, keyed by projection
    label, holding each projection's plan once made.
    """
    block = param.exl3_blocks.get(label, _UNPLANNED)
    if block is _UNPLANNED:
        block = param.exl3_blocks[label] = _plan_block(param, t, label)
    if block is not None and expert_id in block.disk:
        # read on use from the checkpoint: no storage here, and a meta tensor's address is 0, so a pointer table
        # entry for it is null and an unstaged read faults rather than computing on someone else's bytes
        return torch.empty(t.shape, dtype=t.dtype, device="meta")
    row = block.rows.get(expert_id) if block is not None else None
    if row is None:
        return t.to(param.exl3_device).contiguous()
    if t.shape != block.host.shape[1:]:
        raise format.EXL3FormatError(
            f"{param.exl3_prefix}: expert {expert_id} {label} {param.exl3_name} has "
            f"shape {list(t.shape)}, others {list(block.host.shape[1:])}; exl3_mgemm "
            "takes one shape per call"
        )
    _host_copy(block.host[row], t)
    return block.view[row]


#: Host copies in flight: (future, src, dst), the tensors held until the bytes land
_COPIES: collections.deque = collections.deque()
_COPY_POOL = None


def _host_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    """dst.copy_(src) into pinned host memory, as memmoves on the plugin's own copy threads.

    Not torch's copy_: it parallelizes a CPU copy this size over its OpenMP pool, whose threads spin-wait
    between copies (libgomp's default). At one copy per expert a dozen-plus threads sat at ~90% CPU and
    starved the loader: each 0.5 MiB copy took ~640 us instead of ~46 (Qwen3.6-35B-A3B 4.00bpw, hosted
    copies 32-37 s of a 62-68 s load). OMP_WAIT_POLICY=PASSIVE cures that, but libgomp reads it once, as
    torch loads, before the plugin registers. A single-threaded memmove fixed the fast host and slowed a
    WSL host (2 -> 3 s per layer): first touches of the mapped checkpoint are dear there, and parallelism
    is what paid. So: memmove (it releases the GIL) on EXL3_HOST_COPY_THREADS threads that sleep when
    idle, asynchronous, drained by `drain_host_copies` before anything reads the rows."""
    global _COPY_POOL
    if src.device.type != "cpu" or src.dtype != dst.dtype or not src.is_contiguous() or src.numel() != dst.numel():
        dst.copy_(src)
        return
    if _COPY_POOL is None:
        from concurrent.futures import ThreadPoolExecutor

        n = int(os.environ.get("EXL3_HOST_COPY_THREADS", str(min(8, os.cpu_count() or 1))))
        _COPY_POOL = ThreadPoolExecutor(max_workers=max(1, n), thread_name_prefix="exl3-host-copy")
    fut = _COPY_POOL.submit(ctypes.memmove, dst.data_ptr(), src.data_ptr(), src.numel() * src.element_size())
    _COPIES.append((fut, src, dst))
    # bounded: a loader faster than the copies must not hold an unbounded backlog of source tensors
    while len(_COPIES) > 64 * _COPY_POOL._max_workers:
        _land(_COPIES.popleft())


def _land(entry) -> None:
    """Wait for one queued copy, holding its tensors until it has landed. Not `popleft()[0].result()`:
    that drops the popped tuple, and with it the last reference to the source, before the wait, so a copy
    still queued read freed memory -- about 2% of rows when stressed (2026-11-10)."""
    fut, src, dst = entry
    fut.result()
    del src, dst


def drain_host_copies() -> None:
    """Wait for every host copy in flight; called wherever offloaded rows are first read."""
    while _COPIES:
        _land(_COPIES.popleft())


#: Batches of more than this many tokens stage a layer's offloaded experts into
#: VRAM before its GEMM, rather than reading them over UVA. exl3_moe_mm runs one
#: row per (token, expert) slot, so UVA traffic grows with tokens x top_k while
#: the staging copy is one layer's offloaded bytes whatever the batch.
STAGE_MIN_TOKENS = int(os.environ.get("EXL3_STAGE_MIN_TOKENS", "32"))

#: Grouped (prefill) batches copy the next layer's offloaded experts into a second window on a side
#: stream while this layer computes. Copies are a fifth to a third of offloaded prefill on a 3060 under
#: WSL (2026-11-10: 2.07 s of 10.2 at 4096-token chunks, 3.65 of 12.6 at 2048), since each chunk copies
#: most of every layer again. Copies run on the copy engine, not SMs, so they cannot meet exllamav3's
#: spin-waiting kernels the way an overlapped kernel did (plugin._patch_kernel_overlap).
#: ``EXL3_STAGE_PREFETCH``: "1" (default) keeps both windows; "auto" lets the expert cache drop the
#: second when keeping it would take residency below its target (`drop_prefetch`); "0" never allocates it.
#: On by default since 2026-11-14: the window is worth ~50% of prefill, and the few points of residency it
#: costs move decode by little at these shares (Qwen3.6-35B-A3B on a 12 GiB 3060, 128K nvfp4 KV: 42% resident
#: without it, 38% with it, 923 -> 1385 t/s prefill and 35 t/s decode either way; docs/expert-cache.md).
_PREFETCH_ENV = os.environ.get("EXL3_STAGE_PREFETCH", "1").strip().lower() or "1"
PREFETCH_MODE = {"1": "on", "0": "off"}.get(_PREFETCH_ENV, _PREFETCH_ENV)
PREFETCH = PREFETCH_MODE != "off"

#: MoE layers with host blocks, waiting for `build_staging` at the end of load.
_HOSTED: list = []

#: Offsets within the staging window, to keep each block's rows aligned.
_ALIGN = 256


def register_experts(layer, blocks: dict) -> None:
    """Note a MoE layer's trellis host blocks, by projection label, for staging."""
    layer.exl3_stage = None
    layer.exl3_route = None  # the layer's own buffer is set after this; the cache's views replace it
    blocks = {label: b for label, b in blocks.items() if b is not None}
    if blocks:
        layer.exl3_host_blocks = blocks
        _HOSTED.append(layer)


def _layer_number(layer) -> int:
    """The decoder layer a MoE layer belongs to, from its module path, so staging indices follow the
    forward; checkpoint order (which the loader follows) need not."""
    m = re.search(r"layers\.(\d+)\.", layer_prefix(layer))
    return int(m.group(1)) if m else 0


def planned_windows(layers: int) -> int:
    """Prefill staging windows `build_staging` allocates for this many hosted layers."""
    return 2 if PREFETCH and layers > 1 else 1


def build_staging() -> None:
    """Allocate a VRAM staging window, sized to the largest layer's offloaded
    experts, and give each hosted layer a second set of trellis pointer tables
    into it.

    One window serves every layer: a layer's copy is issued in stream after the
    previous layer's GEMM, so stream order alone keeps them from overlapping.
    With ``EXL3_STAGE_PREFETCH`` there are two, alternating by layer, so the next
    layer's copy can run beside this one's kernels (`prefetch_next`). Layers are
    indexed in forward order (by layer number) for that. Called at the end of
    load, before vLLM sizes the KV cache, so the windows are counted as model
    memory. Tables are static, since the windows never move, so a captured graph
    can hold them.
    """
    drain_host_copies()
    if not _HOSTED:
        return
    _HOSTED.sort(key=_layer_number)
    plans, size = [], 0
    for layer in _HOSTED:
        offsets, off = {}, 0
        for label, block in layer.exl3_host_blocks.items():
            offsets[label] = off
            # a disk expert has a window position past the host rows, where it is read to when routed
            off += -(-(block.host.shape[0] + len(block.disk)) * _row_bytes(block) // _ALIGN) * _ALIGN
        plans.append((layer, offsets))
        size = max(size, off)
    if any(b.disk for layer in _HOSTED for b in layer.exl3_host_blocks.values()):
        _open_disk_tier(_HOSTED)
    device = next(iter(_HOSTED[0].exl3_host_blocks.values())).view.device
    count = planned_windows(len(_HOSTED))
    windows = [torch.empty(size, dtype=torch.uint8, device=device) for _ in range(count)]

    for layer, offsets in plans:
        w = (len(STAGED)) % count
        window, base = windows[w], windows[w].data_ptr()
        copies, tables, plan = [], [], {}
        for label in ("gate", "up", "down"):
            table = getattr(layer, f"exl3_{label}_trellis_ptrs")
            block = layer.exl3_host_blocks.get(label)
            if block is not None:
                nbytes = block.host.numel() * block.host.element_size()
                row_bytes = _row_bytes(block)
                dst = window[offsets[label]:offsets[label] + (block.host.shape[0] + len(block.disk)) * row_bytes]
                copies.append((dst[:nbytes], block.host.view(-1).view(torch.uint8)))
                plan[label] = (dst, row_bytes)
                table = table.cpu()
                for e, row in block.rows.items():
                    table[e] = base + offsets[label] + row * row_bytes
                for e, pos in getattr(layer, "exl3_disk_pos", {}).items():
                    table[e] = base + offsets[label] + pos * row_bytes
                table = table.to(device)
            tables.append(table)
        layer.exl3_stage = (copies, tuple(tables))
        layer.exl3_stage_plan = plan
        layer.exl3_stage_window = w
        layer.exl3_stage_idx = len(STAGED)
        STAGED.append(layer)
    # Strong references: the staged tables hold raw addresses into them
    _STATS["windows"] = windows
    logger.info(
        "cpu-offload: %s%.0f MiB prefill staging window%s; batches over %d tokens copy a layer's "
        "offloaded experts in rather than reading them over UVA%s",
        "2 x " if count == 2 else "", size / (1 << 20), "s" if count == 2 else "", STAGE_MIN_TOKENS,
        ", the next layer's while this one computes" if count == 2 else "",
    )
    from . import expert_cache

    expert_cache.prepare(list(_HOSTED))
    _HOSTED.clear()


#: Hosted layers by staging index (`exl3_stage_idx`), which the MoE op is passed: the op stages,
#: not the compiled forward, so the copies can depend on the batch's routing and the cache's state
STAGED: list = []


def stage_all(idx: int):
    """Copy all of a layer's offloaded experts into the window and return the tables into it, or
    None if the layer no longer stages (the cache holds every hosted expert). For batches the
    grouped path does not take: on the current stream, no sync, capturable."""
    layer = STAGED[idx]
    if layer.exl3_stage is None:
        return None
    _settle(layer, idx)
    copies, tables = layer.exl3_stage
    for dst, src in copies:
        dst.copy_(src, non_blocking=True)
    return tables


#: Prefetches issued and not yet consumed, by window: (staging index, event, rows moved per label)
_PENDING_STAGE: list = [None, None]
_SIDE: dict = {}


def _settle(layer, idx: int):
    """Before anything stages into this layer's window: a prefetch pending there is consumed if it is
    this layer's, and waited for and dropped if it is another's (layers ran out of the order the
    indices assume), so the window is never written twice at once. Returns the prefetch's moved rows,
    or None if this layer must stage for itself."""
    w = getattr(layer, "exl3_stage_window", 0)
    pending = _PENDING_STAGE[w] if w < len(_PENDING_STAGE) else None
    if pending is None:
        return None
    _PENDING_STAGE[w] = None
    torch.cuda.current_stream().wait_event(pending[1])
    if pending[0] == idx:
        return pending[2]
    _STATS["prefetch_misordered"] = _STATS.get("prefetch_misordered", 0) + 1
    if _STATS["prefetch_misordered"] == 1:
        logger.warning("cpu-offload: a layer staged out of the forward order prefetch assumes "
                       "(layer %d after a prefetch for %d); correct, but that copy is not hidden", idx, pending[0])
    return None


def _wanted_rows(layer, want) -> dict:
    """Per label, the (host row, expert) pairs to copy: experts `want` accepts that are not in VRAM. Under
    the exclusive expert cache an expert's row moves, so the layer's `exl3_row_of` (-1 in VRAM) is the
    authority; it changes at step boundaries, and within a forward only for the layer being staged, before its own
    staging reads it (a decode read's admission, `admit_at_read`)."""
    row_of = getattr(layer, "exl3_row_of", None)
    if row_of is not None:
        pairs = sorted((int(row_of[e]), int(e)) for e in numpy.flatnonzero(row_of >= 0) if want(int(e)))
        return {label: pairs for label in layer.exl3_host_blocks}
    from . import expert_cache

    in_vram = expert_cache.in_vram(layer)
    return {label: sorted((row, e) for e, row in block.rows.items()
                          if want(e) and not (in_vram is not None and in_vram[e]))
            for label, block in layer.exl3_host_blocks.items()}


def _copy_rows(layer, moved: dict) -> None:
    """Copy the given host rows to their window offsets, in coalesced runs, on the current stream."""
    for label, rows in moved.items():
        if not rows:
            continue
        dst, row_bytes = layer.exl3_stage_plan[label]
        src = layer.exl3_host_blocks[label].host.view(-1).view(torch.uint8)
        i = 0
        while i < len(rows):
            k = i
            while k + 1 < len(rows) and rows[k + 1][0] == rows[k][0] + 1:
                k += 1
            a, b = rows[i][0] * row_bytes, (rows[k][0] + 1) * row_bytes
            dst[a:b].copy_(src[a:b], non_blocking=True)
            i = k + 1


def _point_at_window(layer, moved: dict, tables: tuple) -> tuple:
    """The decode tables with the moved experts pointed at the window. The rest keep their decode
    entries, which are right whether they point at a slot or at the host."""
    out = []
    for j, label in enumerate(("gate", "up", "down")):
        rows = moved.get(label)
        if not rows:
            out.append(tables[j])
            continue
        mask = torch.zeros(tables[j].numel(), dtype=torch.bool)
        mask[[e for _, e in rows]] = True
        mask = mask.pin_memory().to(tables[j].device, non_blocking=True)
        out.append(torch.where(mask, layer.exl3_stage[1][j], tables[j]))
    return tuple(out)


def stage_routed(idx: int, counts: list[int], tables: tuple) -> tuple:
    """Copy into the window only the offloaded experts this batch routes to that are not already in
    VRAM, and return the decode tables with those experts pointed at the window.

    `counts` is the grouped path's per-expert readback. Copying every hosted expert per prefill step
    cost a fixed ~15 GB on Qwen3.6-35B-A3B 4.00bpw whatever the chunk, which, once the MoE kernel was
    grouped, made 512-token chunks link-bound. Experts in an expert-cache slot or resident keep their
    VRAM addresses; unrouted ones are never read. Rows go to their usual window offsets in coalesced
    runs, so the staged tables (`exl3_stage`) give their addresses. A layer `prefetch_next` already
    copied in uses that copy instead."""
    layer = STAGED[idx]
    if layer.exl3_stage is None:
        return tables
    moved = _settle(layer, idx)
    if moved is None:
        moved = _wanted_rows(layer, lambda e: counts[e] > 0)
        _copy_rows(layer, moved)
    from . import expert_cache

    cache = expert_cache.CACHE
    # a decode-sized batch admits its disk experts as it reads them; prefill bypasses RAM, as the routing ring does
    admit = (cache if cache is not None and hasattr(cache, "admit_at_read") and _READ_ADMIT
             and sum(counts) <= expert_cache._ROWS * cache.top_k else None)
    read = _read_disk(layer, lambda e: counts[e] > 0, admit)
    if read:
        moved = {label: moved.get(label, []) + read for label in layer.exl3_host_blocks}
    out = _point_at_window(layer, moved, tables)
    if expert_cache.VERIFY:
        _verify_routed(idx, counts, out)
        if cache is not None and hasattr(cache, "verify_staged_bytes"):
            cache.verify_staged_bytes(layer, [e for e, c in enumerate(counts[:len(out[0])]) if c > 0], out)
    return out


def stage_disk_decode(idx: int, ids: list[int], experts: int, tables: tuple) -> tuple:
    """A decode batch's tables on a disk-tier layer: the decode tables themselves, which point row experts at their host
    rows (read over UVA) and slotted ones at VRAM, once the routed disk experts are read. Under the exclusive cache
    those are admitted as read, straight into host rows, so their decode entries point there too; only one that could
    not be admitted is read through the pinned sources into its window position, and the returned tables -- a copy
    -- point it there. No copies of row experts into the window, no masks over every expert."""
    layer = STAGED[idx]
    routed = {e for e in ids if 0 <= e < experts}
    pos = getattr(layer, "exl3_disk_pos", None)
    out = tables
    from . import expert_cache

    cache = expert_cache.CACHE
    if pos and any(e in pos for e in routed):
        admit = cache if cache is not None and hasattr(cache, "admit_at_read") and _READ_ADMIT else None
        bounced = _read_disk(layer, routed.__contains__, admit, window_rows=False)
        if bounced and layer.exl3_stage is not None:
            index = torch.tensor([e for _, e in bounced], dtype=torch.long).pin_memory().to(tables[0].device,
                                                                                            non_blocking=True)
            out = []
            for j, t in enumerate(tables):
                t = t.clone()
                t.index_copy_(0, index, layer.exl3_stage[1][j].index_select(0, index))
                out.append(t)
            out = tuple(out)
    if expert_cache.VERIFY:
        counts = [0] * (experts + 1)
        for e in ids:
            if 0 <= e < experts:
                counts[e] += 1
        _verify_routed(idx, counts, out)
        if cache is not None and hasattr(cache, "verify_staged_bytes"):
            cache.verify_staged_bytes(layer, sorted(routed), out)
    return out


def has_disk(idx: int) -> bool:
    """Whether a staged layer has experts on the disk tier, which only `stage_routed` can read."""
    return bool(getattr(STAGED[idx], "exl3_disk_pos", None))


def _verify_routed(idx: int, counts: list[int], tables: tuple) -> None:
    """EXL3_CACHE_VERIFY: every routed expert's entry in the tables the kernels get is live. A disk expert's
    decode entry is null, so one that `stage_routed` failed to read and point at the window shows here."""
    routed = [e for e, c in enumerate(counts[:len(tables[0])]) if c > 0]
    for j, label in enumerate(("gate", "up", "down")):
        entries = tables[j].cpu()[routed]
        dead = [e for e, v in zip(routed, entries.tolist()) if v == 0]
        if dead:
            raise RuntimeError(f"expert cache: staged layer {idx} routes {label} of experts {dead[:8]} "
                               f"through null entries (on disk and not read)")


#: The disk tier's reader and its pinned sources (`_open_disk_tier`); empty when no expert is on disk
_DISK: dict = {}


def _row_bytes(block) -> int:
    return math.prod(block.host.shape[1:]) * block.host.element_size()


def _store_key(layer, store) -> str:
    """The checkpoint stem of a layer's experts. vLLM's module path need not be the checkpoint's (a weight
    mapper can rename the trunk), so match on the part from `layers.` on; a drafter's `mtp.` layers share
    those numbers and are never hosted, so they are left out."""
    mine = layer_prefix(layer)
    tail = mine[mine.find("layers."):] if "layers." in mine else mine
    hits = {stem for stem, _ in store.index if not stem.startswith("mtp.")
            and (stem == tail or stem.endswith("." + tail))}
    if len(hits) != 1:
        raise RuntimeError(f"disk tier: {mine} matches {sorted(hits) or 'no'} expert stems in the checkpoint")
    return hits.pop()


def _model_root() -> str:
    """The served checkpoint's local directory."""
    from vllm.config import get_current_vllm_config

    config = get_current_vllm_config()
    mc = config.model_config
    return _checkpoint_root(mc.model, mc.revision, getattr(config.load_config, "download_dir", None))


#: ``EXL3_DROP_CHECKPOINT_CACHE``: "auto" (default) releases the checkpoint's page cache once the weights are
#: loaded under WSL only; "1" everywhere, "0" never. vLLM's lazy loader reads every tensor through the page cache
#: (safetensors' get_tensor copies out of its mapping), so a loaded checkpoint stays cached at about its size. On
#: bare Linux that cache is free and makes the next start warm. Under WSL it is not: the guest counts its cache as
#: available while Windows sees the VM's whole working set, and a VM sized near the host's free memory pages
#: (godel, 24 GiB VM on 22.5 GiB free, 13 GiB of a 19 GiB checkpoint cached after load). Released, the pages are
#: free in the guest, and WSL's free page reporting returns them to Windows.
def release_checkpoint_cache() -> None:
    mode = os.environ.get("EXL3_DROP_CHECKPOINT_CACHE", "auto").strip().lower()
    if mode == "0":
        return
    if mode != "1":
        from vllm.platforms.interface import in_wsl

        if not in_wsl():
            return
    try:
        from vllm.config import get_current_vllm_config
        from vllm.model_executor.model_loader.weight_utils import drop_checkpoint_cache

        config = get_current_vllm_config()
        mc = config.model_config
        drop_checkpoint_cache(mc.model, mc.revision, getattr(config.load_config, "download_dir", None))
    except Exception as exc:   # an older vLLM, or no local checkpoint: the cache stays, as it always did
        logger.info("checkpoint page cache kept: %s", exc)


def _open_disk_tier(layers) -> None:
    """The reader for experts left on disk, each layer's window positions for them, and one pinned source per
    projection sized for the layer with the most. A read is one span per expert (expert_store); under TP each rank
    reads the whole expert and keeps its slice of the intermediate dimension, as loading does (gate and up by tile
    columns, which is strided in the file, down by tile rows), so every rank's read costs what TP=1's does."""
    from concurrent.futures import ThreadPoolExecutor

    from .expert_store import ExpertStore

    root = _model_root()
    store = ExpertStore(root)
    most = {}
    for layer in layers:
        blocks = layer.exl3_host_blocks
        disk = next(iter(blocks.values())).disk
        if any(b.disk != disk for b in blocks.values()):
            raise RuntimeError(f"disk tier: {layer_prefix(layer)}'s projections put different experts on disk")
        if not disk:
            continue
        key = _store_key(layer, store)
        rows = next(iter(blocks.values())).host.shape[0]
        layer.exl3_disk_pos = {e: rows + k for k, e in enumerate(sorted(disk))}
        layer.exl3_store_key = key
        parts = store.index[(key, min(disk))][3]
        layer.exl3_disk_slices = _rank_slices(parts)
        for label, block in blocks.items():
            _, _, dt, shape = parts[label]
            rshape = layer.exl3_disk_slices[label][1] if layer.exl3_disk_slices else shape
            got = math.prod(rshape) * torch.empty(0, dtype=_STORE_DTYPES[dt]).element_size()
            if got != _row_bytes(block):
                raise RuntimeError(f"disk tier: {layer_prefix(layer)} {label} holds {_row_bytes(block)} bytes an "
                                   f"expert, its slice of the checkpoint's {got}")
            most[label] = max(most.get(label, 0), len(disk) * _row_bytes(block))
    _DISK.update(store=store, pinned={label: _pinned((n,), torch.uint8) for label, n in most.items()},
                 pool=ThreadPoolExecutor(int(os.environ.get("EXL3_DISK_QUEUE", "8")), "exl3-disk"), done=None)
    logger.info("expert cache disk tier: %d experts read from %s on use, through %.0f MiB of pinned sources",
                _STATS["disk_experts"], root, sum(most.values()) / (1 << 20))
    fs = _filesystem_type(root)
    if fs in _SLOW_FILESYSTEMS:
        logger.warning(
            "expert cache disk tier: the checkpoint is on a %s mount (%s), a file bridge out of the VM (WSL2's DrvFs, "
            "and Docker Desktop's bind mounts on WSL2): every disk-tier read crosses it. On a WSL2 host decode with 8%% "
            "of experts on disk went 16 t/s from such a mount against 21-24 from the WSL filesystem. Put the "
            "Hugging Face cache on a native filesystem (the WSL distro's disk, or a Docker named volume), or give "
            "--cpu-offload-gb enough host RAM that nothing spills.", fs, root)


def _rank_slices(parts) -> dict | None:
    """Under tensor parallelism, per projection: (this rank's view of a stored trellis, the view's shape). Routed
    experts shard on the intermediate dimension -- tile columns of gate and up, tile rows of down -- at the bounds
    loading uses (`quantization.fused_moe._tp_shard`, uneven where the width demands). None at TP=1."""
    try:
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

        size, rank = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
    except Exception:
        return None
    if size <= 1:
        return None
    from .format import TILE, shard_bounds

    out = {}
    for label, (_, _, _, shape) in parts.items():
        column = label in ("gate", "up")
        dim = shape[1 if column else 0] * TILE
        first, last = shard_bounds(dim, rank, size, f"MoE intermediate ({label})", uneven=True)
        a, b = first // TILE, last // TILE
        fn = (lambda t, a=a, b=b: t[:, a:b]) if column else (lambda t, a=a, b=b: t[a:b])
        out[label] = (fn, tuple(fn(torch.empty(shape, device="meta")).shape))
    return out


def read_expert(layer, expert: int, dests: dict | None = None) -> dict:
    """One disk expert's trellises from the checkpoint: into `dests` (label -> any contiguous tensor holding exactly
    this rank's bytes, such as a host row or a pinned source slot), or into new tensors. A rank's slice under TP."""
    store = _DISK["store"]
    key = layer.exl3_store_key
    slices = getattr(layer, "exl3_disk_slices", None)
    out = None
    if dests is not None:
        out = {}
        for label, (_, _, dt, shape) in store.index[(key, expert)][3].items():
            rshape = slices[label][1] if slices else shape
            out[label] = dests[label].reshape(-1).view(torch.uint8).view(_STORE_DTYPES[dt]).view(rshape)
    return store.read(key, expert, out, {label: fn for label, (fn, _) in slices.items()} if slices else None)


#: Filesystems that bridge files in from outside the VM: 9P (WSL2's DrvFs under /mnt/c, Docker Desktop's bind mounts
#: on its WSL2 engine) and FUSE. Disk-tier reads through them cost several times a native read.
_SLOW_FILESYSTEMS = {"9p", "v9fs", "drvfs", "fuse"}


def _filesystem_type(path: str) -> str | None:
    """The filesystem type of the mount holding `path`, from /proc/mounts (the longest mount point that prefixes it)."""
    try:
        real = os.path.realpath(path)
        best, fs = "", None
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                point = parts[1].replace("\\040", " ")
                if (real == point or real.startswith(point.rstrip("/") + "/")) and len(point) > len(best):
                    best, fs = point, parts[2]
        if fs is not None and fs.startswith("fuse"):
            return "fuse"
        return fs
    except OSError:
        return None


#: ``EXL3_DISK_READ_ADMIT`` (default 1): decode reads a disk expert straight into the host row it is admitted to,
#: once; 0 restores reading it to a pinned bounce buffer and again into its row a step later (`_admit`/`_fill_rows`).
_READ_ADMIT = os.environ.get("EXL3_DISK_READ_ADMIT", "1") == "1"


def _read_disk(layer, want, admit=None, window_rows=True) -> list:
    """Read the layer's disk experts `want` accepts into their window positions; returns (position, expert)
    pairs. Synchronous: the reads run on a thread pool, a deep queue being what an SSD needs, then one copy a
    run of positions on the current stream. The pinned sources are reused by the next batch's reads, so
    those wait for this batch's copies to land first. They hold a layer's disk experts as loaded; while an
    admission is pending (expert_cache, `_admit`) a layer has one more on disk per admission, so the reads
    go in batches of what the sources hold.

    With `admit` (the exclusive cache, a decode batch), experts it admits (`admit_at_read`) are read straight into
    their new host rows and copied to the window from there, at the row's window position; the rest as above. With
    `window_rows` false the admitted ones are not copied and not returned: a decode kernel reads them from their rows."""
    pos = getattr(layer, "exl3_disk_pos", None)
    if not pos:
        return []
    es = sorted((e for e in pos if want(e)), key=pos.__getitem__)   # by position, so runs coalesce
    if not es:
        return []
    store, key, pinned = _DISK["store"], layer.exl3_store_key, _DISK["pinned"]
    plan = layer.exl3_stage_plan
    parts = store.index[(key, es[0])][3]
    into_rows = []
    if admit is not None:
        rows = admit.admit_at_read(layer, es, want)
        if rows:
            into_rows = sorted((r, x) for x, r in rows.items())
            es = [e for e in es if e not in rows]

            def into_row(item):
                r, x = item
                read_expert(layer, x, {label: block.host[r] for label, block in layer.exl3_host_blocks.items()})

            t = time.perf_counter()
            list(_DISK["pool"].map(into_row, into_rows))
            _STATS["disk_read_s"] = _STATS.get("disk_read_s", 0.0) + time.perf_counter() - t
            if window_rows:
                _copy_rows(layer, {label: into_rows for label in layer.exl3_host_blocks})
            else:
                into_rows = []
            _STATS["disk_reads"] = _STATS.get("disk_reads", 0) + len(rows)
            _STATS["disk_admitted"] = _STATS.get("disk_admitted", 0) + len(rows)
            _STATS["disk_read_admitted"] = _STATS.get("disk_read_admitted", 0) + len(rows)
    if not es:
        return into_rows
    rank_bytes = {label: _row_bytes(block) for label, block in layer.exl3_host_blocks.items()}
    cap = min(pinned[label].numel() // nb for label, nb in rank_bytes.items())

    def one(k, e):
        read_expert(layer, e, {label: pinned[label][k * nb:(k + 1) * nb] for label, nb in rank_bytes.items()})

    for b in range(0, len(es), cap):
        batch = es[b:b + cap]
        t0 = time.perf_counter()
        if _DISK["done"] is not None:
            _DISK["done"].synchronize()
        t1 = time.perf_counter()
        list(_DISK["pool"].map(one, range(len(batch)), batch))
        t2 = time.perf_counter()
        # where a disk-tier step's time goes: waiting for earlier copies out of the pinned sources, then reading
        _STATS["disk_wait_s"] = _STATS.get("disk_wait_s", 0.0) + t1 - t0
        _STATS["disk_read_s"] = _STATS.get("disk_read_s", 0.0) + t2 - t1
        for label, (dst, row_bytes) in plan.items():
            i = 0
            while i < len(batch):
                k = i
                while k + 1 < len(batch) and pos[batch[k + 1]] == pos[batch[k]] + 1:
                    k += 1
                a = pos[batch[i]] * row_bytes
                dst[a:a + (k - i + 1) * row_bytes].copy_(pinned[label][i * row_bytes:(k + 1) * row_bytes],
                                                         non_blocking=True)
                i = k + 1
        done = torch.cuda.Event()
        done.record()
        _DISK["done"] = done
    _STATS["disk_reads"] = _STATS.get("disk_reads", 0) + len(es)
    return [(pos[e], e) for e in es] + into_rows


_STORE_DTYPES = {"I16": torch.int16, "F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32,
                 "I32": torch.int32}


def prefetch_window_bytes() -> int:
    """Bytes of the second staging window, which `drop_prefetch` can give back; 0 if there is none or
    prefetch was forced on."""
    windows = _STATS.get("windows") or []
    return windows[1].numel() if len(windows) == 2 and PREFETCH_MODE == "auto" else 0


def drop_prefetch() -> int:
    """Fold every layer back onto the first window, free the second, and stop prefetching; returns the
    bytes freed. Only before graph capture: captured tables would keep the old addresses."""
    global PREFETCH
    windows = _STATS.get("windows") or []
    if len(windows) < 2:
        return 0
    first, second = windows
    delta = first.data_ptr() - second.data_ptr()
    for layer in STAGED:
        if getattr(layer, "exl3_stage_window", 0) != 1 or layer.exl3_stage is None:
            continue
        def rebase(dst):
            off = dst.data_ptr() - second.data_ptr()
            return first[off:off + dst.numel()]

        for label, (dst, row_bytes) in layer.exl3_stage_plan.items():
            layer.exl3_stage_plan[label] = (rebase(dst), row_bytes)
        copies, tables = layer.exl3_stage
        copies = [(rebase(dst), src) for dst, src in copies]
        tables = list(tables)
        for j, label in enumerate(("gate", "up", "down")):
            block = layer.exl3_host_blocks.get(label)
            if block is None:
                continue   # that entry is the decode table itself, not a staged copy of it
            table = tables[j].cpu()
            for e in (*block.rows, *block.disk):
                table[e] += delta
            tables[j] = table.to(tables[j].device)
        layer.exl3_stage = (copies, tuple(tables))
        layer.exl3_stage_window = 0
    _STATS["windows"] = [first]
    freed = second.numel()
    del second, windows
    PREFETCH = False
    torch.cuda.empty_cache()
    return freed


def prefetch_next(idx: int) -> None:
    """With ``PREFETCH``, start copying the next layer's offloaded experts (all of them not in VRAM: its
    routing is not known yet) into its window on a side stream, behind everything already enqueued on
    this one -- which includes the last layer to use that window -- so the copy runs beside this layer's
    kernels. Called by a grouped MoE op after its count readback and table uploads, just before its
    kernels: with one copy engine every transfer queues behind this copy, the readback included."""
    if not PREFETCH or idx + 1 >= len(STAGED):
        return
    nxt = STAGED[idx + 1]
    w = getattr(nxt, "exl3_stage_window", 0)
    if nxt.exl3_stage is None or w == getattr(STAGED[idx], "exl3_stage_window", 0) or _PENDING_STAGE[w] is not None:
        return
    main = torch.cuda.current_stream()
    side = _SIDE.get(main.device)
    if side is None:
        side = _SIDE[main.device] = torch.cuda.Stream(main.device)
    ready = torch.cuda.Event()
    ready.record(main)
    moved = _wanted_rows(nxt, lambda e: True)
    with torch.cuda.stream(side):
        side.wait_event(ready)
        _copy_rows(nxt, moved)
        done = torch.cuda.Event()
        done.record(side)
    _PENDING_STAGE[w] = (idx + 1, done, moved)


def register_offload(layer) -> int:
    """Offload a dequantize-path layer's finished `exl3_weight`.

    Only the ``EXL3_DEQUANTIZE=1`` reference path needs this: everywhere else
    `place` and `place_expert` decided placement at load. Returns the bytes
    handed to the host.
    """
    loader = _uva_loader()
    if loader is None:
        return 0
    return _offload_dequantize(layer, layer_prefix(layer), loader)


def report_once() -> None:
    """Emit the one-line offload summary, once weight loading is complete.

    Called from the ``process_weights_after_loading`` wrapper installed by
    ``plugin._patch_offload_summary``, which is the only point at which every
    layer is known to have been offered. **Not** from ``apply()``: that runs
    inside vLLM's ``torch.compile`` region, where Dynamo refuses
    ``logging.Logger`` calls and fails the engine at startup. See that patch for
    the full reasoning.

    Reports tensors as well as bytes, because the number that distinguishes
    "offload did not help" from "offload did not happen" is the count -- a
    selector with a typo in it matches nothing, offloads nothing, logs nothing
    and serves at full speed, which in a placement sweep reads as a result.
    """
    drain_host_copies()
    if _STATS.get("reported"):
        return
    _STATS["reported"] = True

    if not _STATS["requested"]:
        return

    # vLLM's construction-time pass (wrap_modules) offloads plain tensors, such as a vision
    # tower, before any EXL3 tensor is stored, and counts them in the same total; so the rest
    # of that total is theirs, and they had first claim on the budget.
    loader = _uva_loader()
    # (experts hosted under the expert cache are not in that total; see `_count`)
    spent = _STATS["bytes"] - _STATS["expert_bytes"]
    other = max(loader.cpu_offload_bytes - spent, 0) if loader is not None else 0
    budget = loader.cpu_offload_max_bytes if loader is not None else 0

    if _STATS["tensors"] == 0:
        if other and other >= budget:
            logger.warning(
                "cpu-offload: the --cpu-offload-gb budget (%.2f GiB) was used up by "
                "%.2f GiB of non-EXL3 weights vLLM offloaded first (such as a vision "
                "tower), so no EXL3 weights were offloaded. Raise the budget, or narrow "
                "--cpu-offload-params.", budget / (1 << 30), other / (1 << 30),
            )
        elif _STATS["selective"]:
            logger.warning(
                "cpu-offload: offload was requested but no EXL3 tensor matched "
                "--cpu-offload-params. Nothing was offloaded and nothing will be "
                "slower; check the selector against the checkpoint's tensor names "
                "(see docs/cpu-offload.md). Patterns are dot-delimited segments, "
                "or 're:<regex>'."
            )
        else:
            logger.warning(
                "cpu-offload: offload was requested but no EXL3 tensor was "
                "eligible. Nothing was offloaded."
            )
        return

    logger.info(
        "cpu-offload: %.2f GiB of EXL3 weights in %d tensors across %d layers "
        "offloaded to host memory%s",
        _STATS["bytes"] / (1 << 30), _STATS["tensors"], len(_STATS["layers"]),
        f", after {other / (1 << 30):.2f} GiB of non-EXL3 weights vLLM offloaded first"
        if other else "",
    )


def _offload_dequantize(layer, prefix, loader) -> int:
    """The dequantize path (``EXL3_DEQUANTIZE=1``): a single ``exl3_weight``,
    which exists only after load, so it is moved there. Leaves the Parameter in
    place and swaps ``param.data``, so ``apply()`` still reads it."""
    param = getattr(layer, "exl3_weight", None)
    if param is None or "weight" not in _OFFLOADABLE_COMPONENTS:
        return 0
    if not _matches(loader.cpu_offload_params, f".{prefix}.weight."):
        return 0
    if getattr(param, "_vllm_is_uva_offloaded", False):
        return 0
    if loader.cpu_offload_bytes >= loader.cpu_offload_max_bytes:
        return 0

    nbytes = param.data.numel() * param.data.element_size()
    param.data = get_accelerator_view_from_cpu_tensor(param.data.to("cpu").pin_memory())
    param._vllm_is_uva_offloaded = True
    _count(loader, prefix, nbytes)
    return nbytes
