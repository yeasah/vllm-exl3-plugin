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
    _STATS.update(requested=False, selective=False, bytes=0, tensors=0, layers=set())


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


def _uva_loader():
    """vLLM's offloader if it is UVA with acceleration available, else None."""
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


def _count(loader, prefix: str, nbytes: int, tensors: int = 1) -> None:
    loader.cpu_offload_bytes += nbytes
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
        or not _matches(loader.cpu_offload_params, f".{param.exl3_prefix}.{param.exl3_name}.")
        or loader.cpu_offload_bytes >= loader.cpu_offload_max_bytes
    ):
        return t.to(param.exl3_device).contiguous()

    host = _pinned(t.shape, t.dtype)
    host.copy_(t)
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
    delivers them. Experts whose names match the selectors are taken while the
    budget lasts, each checked before it is taken, as for dense tensors.
    """
    if param.exl3_name not in _OFFLOADABLE_COMPONENTS:
        return None
    loader = _uva_loader()
    if loader is None:
        return None
    nbytes = t.numel() * t.element_size()
    taken = []
    for e in range(param.exl3_num_experts):
        if loader.cpu_offload_bytes >= loader.cpu_offload_max_bytes:
            break
        name = f".{param.exl3_prefix}.{e}.{label}_proj.{param.exl3_name}."
        if _matches(loader.cpu_offload_params, name):
            taken.append(e)
            _count(loader, param.exl3_prefix, nbytes)
    if not taken:
        return None
    return HostBlock({e: row for row, e in enumerate(taken)}, t.shape, t.dtype)


_UNPLANNED = object()


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
    block.host[row].copy_(t)
    return block.view[row]


#: Batches of more than this many tokens stage a layer's offloaded experts into
#: VRAM before its GEMM, rather than reading them over UVA. exl3_moe_mm runs one
#: row per (token, expert) slot, so UVA traffic grows with tokens x top_k while
#: the staging copy is one layer's offloaded bytes whatever the batch.
STAGE_MIN_TOKENS = int(os.environ.get("EXL3_STAGE_MIN_TOKENS", "32"))

#: MoE layers with host blocks, waiting for `build_staging` at the end of load.
_HOSTED: list = []

#: Offsets within the staging window, to keep each block's rows aligned.
_ALIGN = 256


def register_experts(layer, blocks: dict) -> None:
    """Note a MoE layer's trellis host blocks, by projection label, for staging."""
    layer.exl3_stage = None
    layer.exl3_route = None  # set by expert_cache.build when slots are on
    blocks = {label: b for label, b in blocks.items() if b is not None}
    if blocks:
        layer.exl3_host_blocks = blocks
        _HOSTED.append(layer)


def build_staging() -> None:
    """Allocate one VRAM staging window, sized to the largest layer's offloaded
    experts, and give each hosted layer a second set of trellis pointer tables
    into it.

    One window serves every layer: a layer's copy is issued in stream after the
    previous layer's GEMM, so stream order alone keeps them from overlapping.
    Called at the end of load, before vLLM sizes the KV cache, so the window is
    counted as model memory. Tables are static, since the window never moves, so
    a captured graph can hold them.
    """
    if not _HOSTED:
        return
    plans, size = [], 0
    for layer in _HOSTED:
        offsets, off = {}, 0
        for label, block in layer.exl3_host_blocks.items():
            offsets[label] = off
            off += -(-block.host.numel() * block.host.element_size() // _ALIGN) * _ALIGN
        plans.append((layer, offsets))
        size = max(size, off)
    device = next(iter(_HOSTED[0].exl3_host_blocks.values())).view.device
    window = torch.empty(size, dtype=torch.uint8, device=device)
    base = window.data_ptr()

    for layer, offsets in plans:
        copies, tables = [], []
        for label in ("gate", "up", "down"):
            table = getattr(layer, f"exl3_{label}_trellis_ptrs")
            block = layer.exl3_host_blocks.get(label)
            if block is not None:
                nbytes = block.host.numel() * block.host.element_size()
                row_bytes = nbytes // block.host.shape[0]
                dst = window[offsets[label]:offsets[label] + nbytes]
                copies.append((dst, block.host.view(-1).view(torch.uint8)))
                table = table.cpu()
                for e, row in block.rows.items():
                    table[e] = base + offsets[label] + row * row_bytes
                table = table.to(device)
            tables.append(table)
        layer.exl3_stage = (copies, tuple(tables))
    # Strong reference: the staged tables hold raw addresses into it
    _STATS["window"] = window
    logger.info(
        "cpu-offload: %.0f MiB prefill staging window; batches over %d tokens "
        "copy a layer's offloaded experts in rather than reading them over UVA",
        size / (1 << 20), STAGE_MIN_TOKENS,
    )
    from . import expert_cache

    expert_cache.build(list(_HOSTED))
    _HOSTED.clear()


def expert_tables(layer, tokens: int):
    """The gate, up and down trellis pointer tables for a batch of `tokens`.

    Over the threshold, issues this layer's staging copies on the current stream
    and returns the tables into the window; the GEMM that follows in the same
    stream reads them after they land. No sync, and the decision is on the
    batch's shape, so it holds for a whole captured graph.
    """
    if layer.exl3_stage is None or tokens <= STAGE_MIN_TOKENS:
        return (layer.exl3_gate_trellis_ptrs, layer.exl3_up_trellis_ptrs,
                layer.exl3_down_trellis_ptrs)
    copies, tables = layer.exl3_stage
    for dst, src in copies:
        dst.copy_(src, non_blocking=True)
    return tables


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
    if _STATS.get("reported"):
        return
    _STATS["reported"] = True

    if not _STATS["requested"]:
        return

    if _STATS["tensors"] == 0:
        if _STATS["selective"]:
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
        "offloaded to host memory",
        _STATS["bytes"] / (1 << 30), _STATS["tensors"], len(_STATS["layers"]),
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
