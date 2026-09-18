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

This module works around both problems by re-offloading at the moment the finished
tensors exist. ``register_offload`` is a sibling of the dense / quantized stores in
``process_weights_after_loading`` -- it reaches vLLM's offloader singleton and hands
it the real tensor as ``_maybe_offload_to_cpu`` would: pin, accelerator-view, count.
No vLLM change.

Stage 1 is UVA only (``uva_offloading`` on). Prefetch is Stage 2; ``NoopOffloader``
means offload was not requested at all.

See docs/cpu-offload.md for what offload costs and how to choose what to give it.
"""

from __future__ import annotations

import re

import torch
from vllm.model_executor.offloader import get_offloader
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

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
    _STATS.update(requested=False, selective=False, bytes=0, tensors=0, layers=0)


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


def register_offload(layer) -> int:
    """Offload this layer's finished EXL3 tensors to the active backend.

    Returns the number of bytes handed to the host. No-op unless the active
    backend is UVA with UVA acceleration available; every other backend leaves
    the layer untouched.
    """
    try:
        loader = get_offloader()
    except Exception:  # pragma: no cover - defensive; offloader is normally set
        return 0

    name = loader.__class__.__name__
    if name == "NoopOffloader":
        # No --cpu-offload-gb / --offload-group-size: offload was not requested.
        return 0
    if name != "UVAOffloader":
        logger.debug_once("cpu-offload: %s backend unsupported, ignoring", name)
        return 0
    if not getattr(loader, "uva_offloading", False):
        # UVA is configured but UVA acceleration is unavailable; vLLM's own
        # functional_call fallback hits the same construction-time gap EXL3 does,
        # so there is nothing for us to do.
        return 0

    _STATS["requested"] = True
    if getattr(loader, "cpu_offload_params", None):
        _STATS["selective"] = True

    # routed experts don't have prefix, but do have layer_name
    prefix = getattr(layer, "prefix", getattr(layer, "layer_name", ""))

    total = 0
    total += _offload_quantized(layer, prefix, loader)
    total += _offload_dequantize(layer, prefix, loader)
    total += _offload_moe(layer, prefix, loader)

    if total:
        _STATS["layers"] += 1
        # Per-layer detail is debug: a 40-layer MoE emits one line per layer and
        # the number anybody wants is the total. report_once() carries that.
        logger.debug(
            "cpu-offload: %d KiB of EXL3 tensors offloaded from layer %s",
            total >> 10, prefix or "<layer>",
        )
    return total


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
        _STATS["bytes"] / (1 << 30), _STATS["tensors"], _STATS["layers"],
    )


def _offload_one(param, name, prefix, loader) -> int:
    """Offload one on-device tensor to UVA host memory, or 0 if ineligible.

    Mirrors ``UVAOffloader._maybe_offload_to_cpu`` for a single parameter:
    select by the offload params, pin (UVA requires pinned memory and CUDA-graph
    capture refuses unpinned H2D), accelerate-view, and count. Leaves the original
    Parameter object in place and only swaps ``param.data``; the accelerator view
    keeps the same object identity so ``apply()`` still reads it.
    """
    if not _matches(getattr(loader, "cpu_offload_params", None), f".{prefix}.{name}."):
        return 0

    # Idempotent: a tensor the context already reoffloaded (or a prior call) is
    # skipped rather than moved twice.
    if getattr(param, "_vllm_is_uva_offloaded", False):
        return 0

    # Honor the byte budget the offloader was sized with at construction.
    # UVA checks this *before* each parameter (uva.py:88); match that so we
    # spend the whole budget rather than leaving it.
    if loader.cpu_offload_bytes >= loader.cpu_offload_max_bytes:
        return 0

    nbytes = param.data.numel() * param.data.element_size()

    if not param.data.is_pinned():
        param.data = param.data.to(device="cpu").pin_memory()
    param.data = get_accelerator_view_from_cpu_tensor(param.data)
    param._vllm_is_uva_offloaded = True
    loader.cpu_offload_bytes += nbytes
    _STATS["bytes"] += nbytes
    _STATS["tensors"] += 1
    return nbytes


def _offload_quantized(layer, prefix, loader) -> int:
    """The quantized path: ``exl3_trellis_N`` / ``exl3_suh_N`` / ``exl3_svh_N``."""
    total = 0
    for index in range(len(getattr(layer, "exl3_output_sizes", []))):
        for attr, component in (
            ("exl3_trellis", "trellis"),
            ("exl3_suh", "suh"),
            ("exl3_svh", "svh"),
        ):
            if component not in _OFFLOADABLE_COMPONENTS:
                continue
            param = getattr(layer, f"{attr}_{index}", None)
            if param is None:
                continue
            total += _offload_one(param, component, prefix, loader)
    return total


def _offload_dequantize(layer, prefix, loader) -> int:
    """The dequantize path (``EXL3_DEQUANTIZE=1``): a single ``exl3_weight``."""
    param = getattr(layer, "exl3_weight", None)
    if param is None:
        return 0
    return _offload_one(param, "weight", prefix, loader)


def _offload_moe(layer, prefix, loader) -> int:
    """Routed experts: offload each expert's trellis and rebuild the pointer
    table, since ``exl3_mgemm`` dereferences device addresses, not the
    per-expert tensors themselves.

    Reconstruct tensor name for matching by enumerating the tensors.
    """
    total = 0
    for label in ("gate", "up", "down"):
        for component in ("trellis", "suh", "svh"):
            if component not in _OFFLOADABLE_COMPONENTS:
                continue
            tensors = getattr(layer, f"_exl3_{label}_{component}", None)
            if not tensors:
                continue

            # Capture the device *before* offloading. Afterwards these are UVA
            # accelerator views, and the pointer table is only correct because
            # such a view reports the accelerator rather than the host -- an
            # invariant of vLLM's, not ours, and not one to depend on silently.
            device = tensors[0].device

            moved = 0
            for i, param in enumerate(tensors):
                moved += _offload_one(
                    param, f"{i}.{label}_proj.{component}", prefix, loader
                )
            total += moved
            if not moved:
                # Nothing left the card, so the addresses are unchanged. Skip the
                # rebuild: it would be ~360 needless small H2D syncs over a load.
                continue

            ptrs = getattr(layer, f"exl3_{label}_{component}_ptrs", None)
            if ptrs is not None:
                # The kernel reads device addresses; the offloaded accelerator
                # views expose the (host-mapped) device address via data_ptr.
                setattr(
                    layer,
                    f"exl3_{label}_{component}_ptrs",
                    torch.tensor(
                        [t.data_ptr() for t in tensors],
                        dtype=torch.long,
                        device=device,
                    ),
                )
    return total
