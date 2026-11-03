"""VRAM slots for offloaded routed experts, managed as an LRU between steps.

Phase 3 of docs/expert-cache.md. Each MoE layer with offloaded experts gets ``C``
VRAM slots. An offloaded expert's decode pointer-table entry holds either its host
row's UVA address (a miss: the kernel reads it over the link) or a slot's address
(a hit). The kernel reads the table at run time, so changing an entry needs no
graph recapture, and both addresses hold the same bytes, so output never depends
on which one is used.

The controller runs at step boundaries, outside the forward and outside any graph:

1. entries whose promotion copies have landed are pointed at their slots;
2. routing from the latest finished step updates the LRU, and the most-wanted
   misses are taken up to a per-step budget; each one's victim slot has its
   entry pointed back at the host first;
3. the promotions are copied on a side stream that waits for those table writes,
   so every kernel that could still read a victim slot has finished first.

Routing reaches the host without a sync: `apply` copies each layer's `topk_ids`
into one device buffer (capturable), and after the step the wrapper copies that
buffer into a pinned ring and records an event. A ring entry is read only once its
event has completed, which holds under async scheduling, where the CPU runs a step
ahead. So decisions lag the routing by about two steps, which docs/expert-cache.md
measured as nearly free.

Off unless ``EXL3_EXPERT_SLOTS`` is set: a count of slots per layer, or a fraction
of the layer's experts if it contains a dot. ``EXL3_PROMOTE_PER_STEP`` (32) bounds
promotions per step across all layers, ``EXL3_CACHE_STATS`` logs hit rates every
that many steps.
"""

from __future__ import annotations

import os

import numpy
import torch

from .log import init_logger

logger = init_logger(__name__)

_LABELS = ("gate", "up", "down")
#: Decode batches up to this many tokens record routing; larger ones are staged.
_ROWS = int(os.environ.get("EXL3_STAGE_MIN_TOKENS", "32"))
_RING = 4

#: The live cache, once built; None when slots are off or nothing is offloaded.
CACHE: "ExpertCache | None" = None


def _slots_per_layer(experts: int) -> int:
    spec = os.environ.get("EXL3_EXPERT_SLOTS", "")
    if not spec:
        return 0
    return int(float(spec) * experts) if "." in spec else int(spec)


class ExpertCache:
    def __init__(self, layers: list, slots: int, budget: int):
        self.layers = layers
        self.budget = budget
        L, E = len(layers), layers[0].exl3_num_experts
        self.E = E
        first = next(iter(layers[0].exl3_host_blocks.values()))
        self.device = first.view.device
        top_k = None

        # One table for every layer's decode pointers, so a step's updates are one scatter.
        # The layers' tables become views of it, before any graph captures their addresses.
        self.table = torch.empty((L, 3, E), dtype=torch.long, device=self.device)
        self.flat = self.table.view(-1)
        # Per (layer, expert): what each address-holding label needs. host[l][e] is a list of
        # (label index, host address); `slot_addr` gives a slot's address per label.
        self.host: list[dict[int, list[tuple[int, int]]]] = []
        self.slot_tensors: list[dict[int, torch.Tensor]] = []
        self.capacity = numpy.zeros(L, dtype=numpy.int64)
        self.offloaded = numpy.zeros((L, E), dtype=bool)
        for l, layer in enumerate(layers):
            for j, label in enumerate(_LABELS):
                self.table[l, j].copy_(getattr(layer, f"exl3_{label}_trellis_ptrs"))
                setattr(layer, f"exl3_{label}_trellis_ptrs", self.table[l, j])
            hosted: dict[int, list[tuple[int, int]]] = {}
            for j, label in enumerate(_LABELS):
                block = layer.exl3_host_blocks.get(label)
                if block is None:
                    continue
                for e, row in block.rows.items():
                    hosted.setdefault(e, []).append((j, block.view[row].data_ptr()))
            self.host.append(hosted)
            self.offloaded[l, list(hosted)] = True
            cap = min(slots, len(hosted))
            self.capacity[l] = cap
            self.slot_tensors.append({
                j: torch.empty((cap, *layer.exl3_host_blocks[label].host.shape[1:]),
                               dtype=layer.exl3_host_blocks[label].host.dtype, device=self.device)
                for j, label in enumerate(_LABELS) if label in layer.exl3_host_blocks
            })
            top_k = top_k or getattr(layer, "top_k", None)  # RoutedExperts.top_k

        # Slot state. slot_of[l, e] is the slot an expert occupies or is being copied into
        # (-1 if none); owner[l][s] the expert in slot s. live[l, e] is True once its entry
        # points at the slot. last_use is the routing step that last touched it.
        self.slot_of = numpy.full((L, E), -1, dtype=numpy.int64)
        self.owner = [numpy.full(c, -1, dtype=numpy.int64) for c in self.capacity]
        self.live = numpy.zeros((L, E), dtype=bool)
        self.last_use = numpy.full((L, E), -1, dtype=numpy.int64)
        self.pending: list[tuple[torch.cuda.Event, list[tuple[int, int]]]] = []

        # Routing: one device buffer the layers write into, a pinned ring per step.
        self.top_k = top_k or 8
        self.route = torch.zeros((L, _ROWS, self.top_k), dtype=torch.int32, device=self.device)
        for l, layer in enumerate(layers):
            layer.exl3_route = self.route[l]
        self.ring = torch.zeros((_RING, L, _ROWS, self.top_k), dtype=torch.int32).pin_memory()
        # launched steps whose routing is not read yet: ring slot, event, tokens, the live
        # mask they ran with (for stats), and the step number
        self.steps: list[tuple[int, torch.cuda.Event, int, numpy.ndarray | None, int]] = []
        self.step = 0
        self.clock = 0
        self.side = torch.cuda.Stream(self.device)

        self.stats_every = int(os.environ.get("EXL3_CACHE_STATS", "0"))
        self.hits = self.misses = self.resident_hits = self.promoted = 0
        self.planned = self.lag = 0

    # ------------------------------------------------------------ addresses

    def _slot_addr(self, l: int, j: int, s: int) -> int:
        t = self.slot_tensors[l][j]
        return t.data_ptr() + s * t[0].numel() * t.element_size()

    def _write(self, updates: list[tuple[int, int]]) -> None:
        """Set flat table entries, in stream order on the current stream."""
        if not updates:
            return
        idx, val = zip(*updates)
        idx = torch.tensor(idx, dtype=torch.long).pin_memory().to(self.device, non_blocking=True)
        val = torch.tensor(val, dtype=torch.long).pin_memory().to(self.device, non_blocking=True)
        self.flat.index_copy_(0, idx, val)

    def _entry(self, l: int, j: int, e: int) -> int:
        return (l * 3 + j) * self.E + e

    # ------------------------------------------------------------ the step

    def step_begin(self) -> None:
        """At a boundary, before this step's forward is enqueued."""
        # 1. flip landed promotions to their slots. Written before planning, which may pick
        # one of them as a victim and must see its entry already on the slot.
        flips = []
        while self.pending and self.pending[0][0].query():
            _, done = self.pending.pop(0)
            for l, e in done:
                s = self.slot_of[l, e]
                flips += [(self._entry(l, j, e), self._slot_addr(l, j, s)) for j, _ in self.host[l][e]]
                self.live[l, e] = True
        self._write(flips)

        # 2. plan from the newest finished step's routing; older finished ones are dropped
        routed = None
        while self.steps and self.steps[0][1].query():
            idx, _, tokens, live, step_no = self.steps.pop(0)
            routed = (idx, tokens, live, step_no)
        if routed is not None:
            self._plan(*routed)

    def _plan(self, idx: int, tokens: int, live_then: numpy.ndarray | None, step_no: int) -> None:
        self.clock += 1
        ids = self.ring[idx % _RING, :, :tokens].numpy().reshape(len(self.layers), -1)
        L = len(self.layers)
        rows = numpy.repeat(numpy.arange(L), ids.shape[1])
        flat_ids = ids.reshape(-1)
        off = self.offloaded[rows, flat_ids]
        if self.stats_every:
            hit = live_then[rows, flat_ids]
            self.hits += int((off & hit).sum())
            self.misses += int((off & ~hit).sum())
            self.resident_hits += int((~off).sum())
            self.planned += 1
            self.lag += self.step - step_no
        self.last_use[rows[off], flat_ids[off]] = self.clock

        # misses: offloaded, not in a slot or on the way to one; most-wanted first, ties
        # spread over layers by a rotating start so no layer starves
        want = off & (self.slot_of[rows, flat_ids] < 0)
        keys, counts = numpy.unique(rows[want] * self.E + flat_ids[want], return_counts=True)
        order = sorted(range(len(keys)), key=lambda i: (-counts[i], (keys[i] // self.E - self.clock) % L))

        updates, issued = [], []
        for i in order:
            if len(issued) >= self.budget:
                break
            l, e = divmod(int(keys[i]), self.E)
            s = self._victim(l)
            if s is None:
                continue
            old = self.owner[l][s]
            if old >= 0:  # point the victim back at the host before its slot is overwritten
                updates += [(self._entry(l, j, old), addr) for j, addr in self.host[l][old]]
                self.slot_of[l, old] = -1
                self.live[l, old] = False
            self.owner[l][s] = e
            self.slot_of[l, e] = s
            issued.append((l, e))
        if issued:
            self._promote(issued, updates)

    def _victim(self, l: int) -> int | None:
        owners = self.owner[l]
        if not len(owners):
            return None
        free = numpy.flatnonzero(owners < 0)
        if len(free):
            return int(free[0])
        # least recently used among slots whose copies have landed (in flight ones are not
        # live) and that were not routed in the step being planned from
        best, best_use = None, None
        for s, e in enumerate(owners):
            if not self.live[l, e]:
                continue
            use = self.last_use[l, e]
            if use >= self.clock:
                continue
            if best is None or use < best_use:
                best, best_use = s, use
        return best

    def _promote(self, issued: list[tuple[int, int]], victim_updates) -> None:
        # victims' entries move to the host first, on the main stream, after every kernel
        # already enqueued; the copies wait for that
        self._write(victim_updates)
        main = torch.cuda.current_stream(self.device)
        ready = torch.cuda.Event()
        ready.record(main)
        self.side.wait_event(ready)
        with torch.cuda.stream(self.side):
            for l, e in issued:
                s = self.slot_of[l, e]
                layer = self.layers[l]
                for j, label in enumerate(_LABELS):
                    block = layer.exl3_host_blocks.get(label)
                    if block is None or e not in block.rows:
                        continue
                    self.slot_tensors[l][j][s].copy_(block.host[block.rows[e]], non_blocking=True)
        done = torch.cuda.Event()
        done.record(self.side)
        self.pending.append((done, issued))
        self.promoted += len(issued)

    def step_end(self, tokens: int) -> None:
        """After this step's forward is enqueued: move its routing to the host ring."""
        self.step += 1
        if tokens <= _ROWS:
            slot = self.step % _RING
            self.ring[slot].copy_(self.route, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            # keep at most _RING - 1 unread, so a ring slot is never overwritten while queued
            if len(self.steps) >= _RING - 1:
                self.steps.pop(0)
            self.steps.append((slot, ev, tokens, self.live.copy() if self.stats_every else None,
                               self.step))
        if self.stats_every and self.step % self.stats_every == 0:
            # over the window since the last report, so warm-up does not linger in it
            off = self.hits + self.misses
            total = off + self.resident_hits
            logger.info(
                "expert cache: step %d; last %d: VRAM hit %.1f%% of routed, slot hit %.1f%% "
                "of offloaded, %d promotions, routing read %.1f steps after launch",
                self.step, self.stats_every, 100 * (total - self.misses) / max(total, 1),
                100 * self.hits / max(off, 1), self.promoted, self.lag / max(self.planned, 1),
            )
            self.hits = self.misses = self.resident_hits = self.promoted = 0
            self.planned = self.lag = 0


def _patch_runners() -> None:
    """Run the controller around every real step of either model runner.

    Patched here, in the worker, once the cache exists, rather than at plugin
    registration: the runner modules are heavy imports, and other processes never
    step a model. Dummy runs (profiling, graph capture) are left alone.
    """
    import importlib
    from functools import wraps

    for name in ("vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu.model_runner"):
        try:
            cls = importlib.import_module(name).GPUModelRunner
        except (ImportError, AttributeError):
            continue
        if getattr(cls, "_exl3_cache_patched", False):
            continue
        original = cls.execute_model

        @wraps(original)
        def execute_model(self, scheduler_output, *args, _original=original, **kwargs):
            cache = CACHE
            if cache is None or kwargs.get("dummy_run") or (args and args[1:2] == (True,)):
                return _original(self, scheduler_output, *args, **kwargs)
            tokens = scheduler_output.total_num_scheduled_tokens
            if tokens:
                cache.step_begin()
            out = _original(self, scheduler_output, *args, **kwargs)
            if tokens:
                cache.step_end(tokens)
            return out

        cls.execute_model = execute_model
        cls._exl3_cache_patched = True


def build(layers: list) -> None:
    """Called at the end of load, after the staging window, before KV sizing."""
    global CACHE
    if not layers or not os.environ.get("EXL3_EXPERT_SLOTS"):
        return
    slots = _slots_per_layer(layers[0].exl3_num_experts)
    if slots <= 0:
        return
    CACHE = ExpertCache(layers, slots, int(os.environ.get("EXL3_PROMOTE_PER_STEP", "32")))
    _patch_runners()
    nbytes = sum(t.numel() * t.element_size() for d in CACHE.slot_tensors for t in d.values())
    logger.info(
        "expert cache: %d slots per layer over %d layers, %.0f MiB; up to %d promotions "
        "per step", slots, len(layers), nbytes / (1 << 20), CACHE.budget,
    )
