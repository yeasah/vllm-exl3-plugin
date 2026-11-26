"""Report when Windows starts paging this process's VRAM (WSL only).

Under WSL, CUDA allocations go through WDDM, whose video memory manager gives each process a budget: dedicated VRAM
less what everything else on the card holds, the Windows desktop and its apps included. Past that budget nothing
fails -- `cudaMemGetInfo` reads 0 free and allocations still succeed -- and the manager demotes the overflow to system
memory instead. Measured on an RTX 3060 (2026-11-26): a sweep over every allocation held ran 327 GB/s under the
budget, 282 at 0.46 GiB over and 98 at 1.96 over; a serving bench lost 3x on prefill.

The budget also moves on its own as Windows rebalances (0.25 GiB during one idle test), and any Windows app can claim
VRAM at any time, so sizing at startup cannot settle it. Holding headroom back is a policy only the operator can set
(`gpu_memory_utilization` is the instrument); what the server can do is know when it happens. Demand paging then
degrades instead of OOMing, and says so.

WSL exposes the budget to Linux processes through DXCore (`/usr/lib/wsl/lib/libdxcore.so`, mounted into GPU
containers too): `AdapterMemoryBudget` on the local segment group, whose `currentUsage` counts this process's CUDA
allocations exactly. WSL gives CUDA and DXCore different LUIDs for the same card, so the adapter is identified by the
one thing that is certain: it is the one carrying this process's allocations.
"""

from __future__ import annotations

import ctypes as C
import logging
import os
import threading
import time
import uuid

from .log import init_logger

logger = init_logger(__name__)

_LIB = "/usr/lib/wsl/lib/libdxcore.so"
GIB = 1 << 30
#: Seconds between polls; one ioctl each. `EXL3_WDDM_MONITOR=0` turns the monitor off.
_INTERVAL = float(os.environ.get("EXL3_WDDM_MONITOR_INTERVAL", "1.0"))
#: While over budget, warn again when the overshoot has grown by this much, or this long after the last warning.
_REWARN_BYTES = GIB // 4
_REWARN_SECONDS = 300.0


class _GUID(C.Structure):
    _fields_ = [("d1", C.c_uint32), ("d2", C.c_uint16), ("d3", C.c_uint16), ("d4", C.c_ubyte * 8)]

    @classmethod
    def of(cls, s: str) -> "_GUID":
        u = uuid.UUID(s)
        g = cls(u.fields[0], u.fields[1], u.fields[2])
        g.d4[:] = list(u.bytes[8:])
        return g


class _Budget(C.Structure):
    _fields_ = [("budget", C.c_uint64), ("currentUsage", C.c_uint64),
                ("availableForReservation", C.c_uint64), ("currentReservation", C.c_uint64)]


class _NodeSegment(C.Structure):
    _fields_ = [("nodeIndex", C.c_uint32), ("segmentGroup", C.c_uint32)]


def _vcall(obj, index, restype, *argtypes):
    """Call slot `index` of a COM-style interface's vtable (WSL's DXCore has no virtual destructor slots)."""
    vtbl = C.cast(C.cast(obj, C.POINTER(C.c_void_p))[0], C.POINTER(C.c_void_p))
    return C.CFUNCTYPE(restype, C.c_void_p, *argtypes)(vtbl[index])


class DXCore:
    """The compute adapters DXCore lists, and their local-memory budget for this process."""

    _IID_FACTORY = "78ee5945-c36e-4b13-a669-005dd11c0f06"
    _IID_LIST = "526c7776-40e9-459b-b711-f32ad76dfc28"
    _IID_ADAPTER = "f0db4c7f-fe5a-42a2-bd62-f2a6cf6fc83e"
    _ATTR_COMPUTE = "248e2800-a793-4724-abaa-23a6de1be090"
    _STATE_MEMORY_BUDGET = 1
    _PROP_DESCRIPTION = 2

    def __init__(self, path: str = _LIB):
        lib = C.CDLL(path)
        self._keep = []  # GUIDs and interface pointers must outlive the calls that reference them
        factory = C.c_void_p()
        self._check(lib.DXCoreCreateAdapterFactory(C.byref(self._guid(self._IID_FACTORY)), C.byref(factory)))
        adapters = C.c_void_p()
        self._check(_vcall(factory, 3, C.c_int32, C.c_uint32, C.c_void_p, C.c_void_p, C.c_void_p)(
            factory, 1, C.byref(self._guid(self._ATTR_COMPUTE)), C.byref(self._guid(self._IID_LIST)),
            C.byref(adapters)))
        self.adapters = []
        for i in range(_vcall(adapters, 4, C.c_uint32)(adapters)):
            a = C.c_void_p()
            self._check(_vcall(adapters, 3, C.c_int32, C.c_uint32, C.c_void_p, C.c_void_p)(
                adapters, i, C.byref(self._guid(self._IID_ADAPTER)), C.byref(a)))
            if _vcall(a, 8, C.c_bool, C.c_uint32)(a, self._STATE_MEMORY_BUDGET):
                self.adapters.append(a)
        self._keep += [factory, adapters]

    def _guid(self, s):
        g = _GUID.of(s)
        self._keep.append(g)
        return g

    @staticmethod
    def _check(hr):
        if hr != 0:
            raise OSError(f"DXCore call failed: HRESULT {hr & 0xFFFFFFFF:#010x}")

    def describe(self, adapter) -> str:
        buf = C.create_string_buffer(256)
        self._check(_vcall(adapter, 6, C.c_int32, C.c_uint32, C.c_size_t, C.c_void_p)(
            adapter, self._PROP_DESCRIPTION, 256, buf))
        return buf.value.decode(errors="replace")

    def budget(self, adapter) -> tuple[int, int]:
        """(budget, current usage) of local memory for this process, in bytes."""
        inp, out = _NodeSegment(0, 0), _Budget()
        self._check(_vcall(adapter, 9, C.c_int32, C.c_uint32, C.c_size_t, C.c_void_p, C.c_size_t, C.c_void_p)(
            adapter, self._STATE_MEMORY_BUDGET, C.sizeof(inp), C.byref(inp), C.sizeof(out), C.byref(out)))
        return out.budget, out.currentUsage


def pick_adapter(usages: list[int], held: int) -> int | None:
    """Index of the adapter carrying this process's allocations: the only one whose usage covers what torch holds.
    None when no adapter or more than one does -- a guess could watch the wrong card, which is worse than no
    monitor."""
    covering = [i for i, u in enumerate(usages) if u >= held > 0]
    return covering[0] if len(covering) == 1 else None


class Watch:
    """Turns (budget, usage) samples into log events; no I/O, so it can be tested without a GPU."""

    def __init__(self):
        self.over_since: float | None = None
        self.peak_over = 0
        self.reported_over = 0
        self.last_warned = 0.0
        self.episodes = 0
        self.seconds_over = 0.0
        self.worst_over = 0
        self.least_headroom: int | None = None

    def observe(self, budget: int, usage: int, now: float) -> list[tuple[int, str]]:
        events = []
        over = usage - budget
        self.least_headroom = -over if self.least_headroom is None else min(self.least_headroom, -over)
        if over > 0:
            if self.over_since is None:
                self.over_since, self.peak_over, self.reported_over, self.last_warned = now, over, over, now
                self.episodes += 1
                events.append((logging.WARNING,
                               f"WDDM budget exceeded: this process holds {usage / GIB:.2f} GiB of VRAM against a "
                               f"budget of {budget / GIB:.2f} GiB ({over / GIB:+.2f} GiB over). Windows is paging the "
                               "overflow to system memory, so throughput degrades until it recedes: something else "
                               "on the GPU (a Windows app, another process) claimed memory, or the budget moved. "
                               "A lower gpu_memory_utilization leaves room for it."))
            else:
                self.peak_over = max(self.peak_over, over)
                if over >= self.reported_over + _REWARN_BYTES or now - self.last_warned >= _REWARN_SECONDS:
                    self.reported_over, self.last_warned = over, now
                    events.append((logging.WARNING,
                                   f"WDDM budget still exceeded after {now - self.over_since:.0f} s: "
                                   f"{usage / GIB:.2f} GiB held against {budget / GIB:.2f} GiB "
                                   f"({over / GIB:+.2f} GiB over, worst {self.peak_over / GIB:+.2f})."))
            self.worst_over = max(self.worst_over, self.peak_over)
        elif self.over_since is not None:
            lasted = now - self.over_since
            self.seconds_over += lasted
            events.append((logging.INFO,
                           f"WDDM budget recovered after {lasted:.0f} s (worst {self.peak_over / GIB:+.2f} GiB over): "
                           f"{usage / GIB:.2f} GiB held against {budget / GIB:.2f} GiB. Over budget {self.episodes} "
                           f"time(s) so far, {self.seconds_over:.0f} s in all."))
            self.over_since = None
        return events


_STARTED = False
_LOCK = threading.Lock()


def enabled() -> bool:
    if os.environ.get("EXL3_WDDM_MONITOR", "1") == "0" or not os.path.exists(_LIB):
        return False
    from vllm.platforms.interface import in_wsl

    return in_wsl()


def start_monitor() -> None:
    """Start the monitor in this (worker) process, once; WDDM's budget is per process, so it has to run where the
    CUDA allocations are. A no-op off WSL."""
    global _STARTED
    with _LOCK:
        if _STARTED or not enabled():
            return
        _STARTED = True
    try:
        import torch

        dx = DXCore()
        held = torch.cuda.memory_reserved()
        samples = [dx.budget(a) for a in dx.adapters]
        index = pick_adapter([u for _, u in samples], held)
        if index is None:
            logger.info("WDDM budget monitor off: no single adapter's usage covers the %.2f GiB this process holds "
                        "(%d adapters)", held / GIB, len(samples))
            return
    except Exception as exc:  # DXCore absent or changed shape: the monitor is a report, never a reason to fail
        logger.info("WDDM budget monitor off: %s", exc)
        return
    adapter = dx.adapters[index]
    budget, usage = samples[index]
    logger.info("WDDM budget monitor on %s: budget %.2f GiB, this process holds %.2f GiB (%.2f GiB headroom); "
                "warns when Windows starts paging this process's VRAM", dx.describe(adapter), budget / GIB,
                usage / GIB, (budget - usage) / GIB)
    threading.Thread(target=_run, args=(dx, adapter), name="exl3-wddm-budget", daemon=True).start()


def _run(dx: DXCore, adapter) -> None:
    watch = Watch()
    while True:
        try:
            budget, usage = dx.budget(adapter)
        except Exception as exc:
            logger.warning("WDDM budget monitor stopped: %s", exc)
            return
        for level, message in watch.observe(budget, usage, time.monotonic()):
            logger.log(level, message)
        time.sleep(_INTERVAL)
