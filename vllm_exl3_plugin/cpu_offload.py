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
                  expert_bytes=0, moe_layers=set(), quota=None)


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


class HostBlock:
    """Pinned host rows for one MoE layer's offloaded experts, for one projection
    and one component. `rows` maps expert id to row; `view` is the UVA view the
    pointer tables address, `host` the CPU side the loader writes."""

    def __init__(self, rows: dict[int, int], shape, dtype):
        self.rows = rows
        self.host = _pinned((len(rows), *shape), dtype)
        self.view = get_accelerator_view_from_cpu_tensor(self.host)


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
        want = layers * experts * 3 * nbytes
        free = _mem_available() // _local_ranks() - max(_HOST_MARGIN, _mem_total() // 10)
        loader = _uva_loader()
        budget = (max(loader.cpu_offload_max_bytes - loader.cpu_offload_bytes, 0)
                  if loader is not None else want)
        room = max(min(free, budget), 0)
        share = 1.0 if want <= room else room / want
        _STATS["quota"] = math.floor(share * experts)
        if share < 1.0:
            logger.warning(
                "cpu-offload: %s holds %.2f of %.2f GiB of routed experts; offloading %d of %d "
                "per layer for the expert cache, the rest stay in VRAM",
                "--cpu-offload-gb" if budget <= free else "host RAM", room / (1 << 30),
                want / (1 << 30), _STATS["quota"], experts,
            )
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
#: ``EXL3_STAGE_PREFETCH``: "auto" (default) allocates both windows and lets the expert cache drop the
#: second when keeping it would take residency below its target (`drop_prefetch`); "1" keeps it
#: whatever that costs, "0" never allocates it.
PREFETCH_MODE = {"1": "on", "0": "off"}.get(
    os.environ.get("EXL3_STAGE_PREFETCH", "auto").strip().lower() or "auto",
    os.environ.get("EXL3_STAGE_PREFETCH", "auto").strip().lower() or "auto")
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
            off += -(-block.host.numel() * block.host.element_size() // _ALIGN) * _ALIGN
        plans.append((layer, offsets))
        size = max(size, off)
    device = next(iter(_HOSTED[0].exl3_host_blocks.values())).view.device
    count = 2 if PREFETCH and len(_HOSTED) > 1 else 1
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
                row_bytes = nbytes // block.host.shape[0]
                dst = window[offsets[label]:offsets[label] + nbytes]
                copies.append((dst, block.host.view(-1).view(torch.uint8)))
                plan[label] = (dst, row_bytes)
                table = table.cpu()
                for e, row in block.rows.items():
                    table[e] = base + offsets[label] + row * row_bytes
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
    authority; it changes only at step boundaries, never within a forward."""
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
    return _point_at_window(layer, moved, tables)


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
        moved = {}
        for label, (dst, row_bytes) in layer.exl3_stage_plan.items():
            off = dst.data_ptr() - second.data_ptr()
            moved[id(dst)] = first[off:off + dst.numel()]
            layer.exl3_stage_plan[label] = (moved[id(dst)], row_bytes)
        copies, tables = layer.exl3_stage
        copies = [(moved[id(dst)], src) for dst, src in copies]
        tables = list(tables)
        for j, label in enumerate(("gate", "up", "down")):
            block = layer.exl3_host_blocks.get(label)
            if block is None:
                continue   # that entry is the decode table itself, not a staged copy of it
            table = tables[j].cpu()
            for e in block.rows:
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
