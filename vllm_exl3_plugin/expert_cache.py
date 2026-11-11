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

Routing reaches the host without a sync: `apply` copies the leading rows of each
layer's `topk_ids` into one device buffer (capturable), and after the step the
wrapper copies that buffer into a pinned ring and records an event. vLLM orders
decode rows first, so the leading rows are the step's decodes even when a prefill
chunk shares the step; the controller reads as many as the step had. A ring entry is read only once its
event has completed, which holds under async scheduling, where the CPU runs a step
ahead. So decisions lag the routing by about two steps, which docs/expert-cache.md
measured as nearly free.

Sizing (phase 4c). On by default whenever experts are offloaded; ``EXL3_EXPERT_SLOTS``
is "auto" (the default), 0 for off, or a fixed count per layer (a fraction of the
layer's experts if it contains a dot). Auto sizes at vLLM's ``determine_available_memory``,
after the memory profile and before KV and graph capture: KV keeps
``EXL3_KV_RESERVE`` (1.0) times what one ``--max-model-len`` request needs, and slots
take the rest: a length the user set is what their application needs, so it comes
first. ``EXL3_KV_RESERVE`` is also the concurrency the cache sizes for. With
``--max-model-len auto`` the user has left the trade to us: the reserve is a floor of
4096 tokens, slots stop at the share of experts in VRAM that measured within ~10% of
resident decode for that concurrency (`target_share`), and the rest goes to context,
which vLLM fits the length to. ``--cpu-offload-gb`` caps what is hosted
(`cpu_offload._host_quota`). See `size_slots`. ``EXL3_PROMOTE_PER_STEP`` (32) bounds
promotions per step across all layers, whatever the batch: growing it with decode rows
thrashed, measured at 2-16 streams (docs/expert-cache.md, 4b). ``EXL3_CACHE_STATS``
logs hit rates every that many steps.
"""

from __future__ import annotations

import math
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
    return int(float(spec) * experts) if "." in spec else int(spec)


def target_share(streams: float) -> float:
    """Share of a layer's experts in VRAM that kept decode within ~10% of resident at this
    many concurrent streams: ~0.4 at 1, +0.1 per doubling (docs/expert-cache.md, 4b)."""
    return min(1.0, 0.4 + 0.1 * math.log2(max(streams, 1.0)))


def size_slots(available: int, reserve: int, hosted: int, slot_bytes: int,
               layers: int, per_layer_max: int, cap: int | None = None) -> int:
    """Slots per layer for the auto mode.

    `available` is the KV memory vLLM measured with every hosted expert off the device,
    `reserve` the KV to keep, `hosted` the hosted experts' bytes, and `slot_bytes` one slot
    across its projections.

    Slots take what the reserve leaves: nothing beyond the hosted experts, and nothing
    beyond what is available. --cpu-offload-gb is not a floor here; it already capped what
    is hosted (`cpu_offload._host_quota`), and experts it left out are resident. `cap`, per
    layer, stops slots at a target (--max-model-len auto), leaving the rest to KV.
    """
    want = max(available - reserve, 0)
    target = min(want, hosted, available)
    if cap is not None:
        return min(per_layer_max, cap, target // max(slot_bytes * layers, 1))
    return min(per_layer_max, target // max(slot_bytes * layers, 1))


#: Tokens the reserve covers when --max-model-len is auto, before the multiplier.
_AUTO_LEN_TOKENS = 4096


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
        # launched steps whose routing is not read yet: ring slot, event, decode rows, the
        # live mask they ran with (for stats; None if staged, where every expert was in VRAM),
        # and the step number
        self.steps: list[tuple[int, torch.cuda.Event, int, numpy.ndarray | None, int]] = []
        self.step = 0
        self.clock = 0
        self.side = torch.cuda.Stream(self.device)

        self.stats_every = int(os.environ.get("EXL3_CACHE_STATS", "0"))
        self.hits = self.misses = self.resident_hits = self.promoted = 0
        self.planned = self.lag = 0
        self.window_steps = self.decode_steps = self.staged_steps = 0
        self._fill()

    def _fill(self) -> None:
        """Start warm: fill every slot before the first step, so a cache that holds all of a
        layer's hosted experts never misses and one that holds some starts with hits instead
        of a cold climb at the promotion budget. Before graph capture, so synchronous."""
        updates = []
        for l, layer in enumerate(self.layers):
            for s, e in enumerate(sorted(self.host[l])[: self.capacity[l]]):
                for j, label in enumerate(_LABELS):
                    block = layer.exl3_host_blocks.get(label)
                    if block is not None and e in block.rows:
                        self.slot_tensors[l][j][s].copy_(block.host[block.rows[e]], non_blocking=True)
                self.owner[l][s] = e
                self.slot_of[l, e] = s
                self.live[l, e] = True
                updates += [(self._entry(l, j, e), self._slot_addr(l, j, s)) for j, _ in self.host[l][e]]
        self._write(updates)
        torch.cuda.synchronize(self.device)

    @property
    def holds_everything(self) -> bool:
        return all(self.capacity[l] >= len(self.host[l]) for l in range(len(self.layers)))

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
        if self.stats_every and live_then is not None:
            hit = live_then[rows, flat_ids]
            self.hits += int((off & hit).sum())
            self.misses += int((off & ~hit).sum())
            self.resident_hits += int((~off).sum())
        if self.stats_every:
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

    def step_end(self, tokens: int, decodes: int) -> None:
        """After this step's forward is enqueued: move its decode rows' routing to the host
        ring. A step over the staging threshold ran every expert from VRAM, so its rows
        teach the LRU but count toward no hit rate."""
        self.step += 1
        staged = tokens > _ROWS
        self.window_steps += 1
        self.staged_steps += staged
        rows = min(decodes, _ROWS)
        if rows:
            self.decode_steps += 1
            slot = self.step % _RING
            self.ring[slot].copy_(self.route, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            # keep at most _RING - 1 unread, so a ring slot is never overwritten while queued
            if len(self.steps) >= _RING - 1:
                self.steps.pop(0)
            live = self.live.copy() if self.stats_every and not staged else None
            self.steps.append((slot, ev, rows, live, self.step))
        if self.stats_every and self.step % self.stats_every == 0:
            # over the window since the last report, so warm-up does not linger in it
            off = self.hits + self.misses
            total = off + self.resident_hits
            logger.info(
                "expert cache: step %d; last %d: VRAM hit %.1f%% of routed, slot hit %.1f%% "
                "of offloaded, %d promotions, routing read %.1f steps after launch; "
                "%d steps had decodes, %d learned from, %d staged",
                self.step, self.stats_every, 100 * (total - self.misses) / max(total, 1),
                100 * self.hits / max(off, 1), self.promoted, self.lag / max(self.planned, 1),
                self.decode_steps, self.planned, self.staged_steps,
            )
            self.hits = self.misses = self.resident_hits = self.promoted = 0
            self.planned = self.lag = 0
            self.window_steps = self.decode_steps = self.staged_steps = 0


#: Every expert in exactly one place (`ExclusiveCache`), the default; EXL3_CACHE_EXCLUSIVE=0 keeps the inclusive
#: cache, which holds a host copy of every hosted expert and leaves the rest resident by id.
EXCLUSIVE = os.environ.get("EXL3_CACHE_EXCLUSIVE", "1") != "0"
#: EXL3_CACHE_VERIFY=1: after every step boundary, synchronize and check that each expert that moved has
#: its table entries on a copy of its own bytes (checksums taken at build); a debug detector, not a mode.
VERIFY = os.environ.get("EXL3_CACHE_VERIFY", "0") == "1"
_WEIGHTS: dict = {}


def _checksum(t: torch.Tensor) -> int:
    """Position-weighted sum of a tensor's bytes as int16, so a permutation or a shifted copy differs."""
    v = t.reshape(-1).view(torch.int16)
    if v.device.type != "cuda":
        v = v.to("cuda", non_blocking=False)
    w = _WEIGHTS.get((v.numel(), v.device))
    if w is None:
        w = _WEIGHTS[(v.numel(), v.device)] = torch.arange(v.numel(), device=v.device, dtype=torch.int64) % 65521 + 1
    return int((v.to(torch.int64) * w).sum())


class ExclusiveCache(ExpertCache):
    """Every expert in exactly one place: a VRAM slot or a host row.

    The inclusive cache keeps a host copy of every hosted expert, slot or not, and leaves the rest resident
    by expert id. Here residents are ordinary slots (their own tensors), filling the new slots frees their
    experts' host rows, and a promotion is a swap: the cold expert Y in slot s is written back to a free
    host row f, and then the hot one X is copied from its row h into s, which frees h. Two step boundaries:

    1. Y is copied s -> f on the side stream, behind an event on the forward stream; Y stays readable at s.
    2. Once landed, Y's entries flip to f on the forward stream; then, behind an event after that flip, X
       is copied h -> s. X stays readable at h.
    3. Once landed, X's entries flip to s, and h joins the free rows.

    Every entry points at a complete copy at every moment, and no copy writes memory that any entry or
    queued kernel could read. Staged prefill tables are kept beside the decode tables (an expert on host
    stages from its current row), and staging reads `exl3_row_of`, which follows the moves. Measured on a
    3060 under WSL at 103 of 256 experts in VRAM: residents chosen by id were close to dead weight, and the
    host copies of slotted experts cost 1.9 GiB of host RAM and, through larger windows, ~45K tokens of KV
    (docs/expert-cache.md, "More offload can mean less KV").
    """

    def _fill(self) -> None:
        L, E = len(self.layers), self.E
        self.row_of = numpy.full((L, E), -1, dtype=numpy.int64)
        self.free_rows: list[list[int]] = []
        self.views: list[list[tuple]] = []      # per layer, per slot: one device view per label
        self.busy = numpy.zeros((L, E), dtype=bool)   # Y being written back
        self.reserved: list[dict[int, int]] = []      # per layer: slot -> incoming X
        self.pending_d: list = []
        self.pending_p: list = []
        self.stage = None
        if self.layers[0].exl3_stage is not None:
            self.stage = torch.empty((L, 3, E), dtype=torch.long, device=self.device)
            self.stage_flat = self.stage.view(-1)
        owners, updates, stage_updates = [], [], []
        for l, layer in enumerate(self.layers):
            blocks = [layer.exl3_host_blocks[label] for label in _LABELS]
            rows = blocks[0].rows
            resident = [e for e in range(E) if e not in rows]
            views = [tuple(getattr(layer, f"_exl3_{label}_trellis")[e] for label in _LABELS) for e in resident]
            cap = int(self.capacity[l])
            views += [tuple(self.slot_tensors[l][j][s] for j in range(3)) for s in range(cap)]
            owner = numpy.full(len(views), -1, dtype=numpy.int64)
            for s, e in enumerate(resident):
                owner[s] = e
                self.slot_of[l, e] = s
                self.live[l, e] = True
            hosted = sorted(rows)
            free = []
            for k, e in enumerate(hosted[:cap]):
                s = len(resident) + k
                for j, block in enumerate(blocks):
                    views[s][j].copy_(block.host[rows[e]], non_blocking=True)
                owner[s] = e
                self.slot_of[l, e] = s
                self.live[l, e] = True
                free.append(rows[e])
            for e in hosted[cap:]:
                self.row_of[l, e] = rows[e]
            self.free_rows.append(free)
            self.views.append(views)
            self.reserved.append({})
            owners.append(owner)
            self.capacity[l] = len(views)
            layer.exl3_row_of = self.row_of[l]
            if self.stage is not None:
                for j in range(3):
                    self.stage[l, j].copy_(layer.exl3_stage[1][j])
                layer.exl3_stage = (layer.exl3_stage[0], tuple(self.stage[l, j] for j in range(3)))
            for e in hosted[:cap]:
                s = self.slot_of[l, e]
                updates += [(self._entry(l, j, e), self._view_addr(l, s, j)) for j in range(3)]
                stage_updates += [(self._entry(l, j, e), self._view_addr(l, s, j)) for j in range(3)]
        self.owner = owners
        self.offloaded = self.row_of >= 0
        self._write(updates)
        self._write_stage(stage_updates)
        torch.cuda.synchronize(self.device)
        self.ref = None
        if VERIFY:
            self.ref = numpy.zeros((L, 3, E), dtype=object)
            for l in range(L):
                for e in range(E):
                    for j in range(3):
                        self.ref[l, j, e] = _checksum(self._home(l, j, e))
            self._verify(range(L), full=True)

    @property
    def holds_everything(self) -> bool:
        return False

    # ------------------------------------------------------------ addresses

    def _view_addr(self, l: int, s: int, j: int) -> int:
        return self.views[l][s][j].data_ptr()

    def _host_addr(self, l: int, j: int, row: int) -> int:
        return self.layers[l].exl3_host_blocks[_LABELS[j]].view[row].data_ptr()

    def _window_addr(self, l: int, j: int, row: int) -> int:
        dst, row_bytes = self.layers[l].exl3_stage_plan[_LABELS[j]]
        return dst.data_ptr() + row * row_bytes

    def _home(self, l: int, j: int, e: int) -> torch.Tensor:
        row = self.row_of[l, e]
        if row >= 0:
            return self.layers[l].exl3_host_blocks[_LABELS[j]].host[row]
        return self.views[l][self.slot_of[l, e]][j]

    def _write_stage(self, updates) -> None:
        if self.stage is None or not updates:
            return
        idx, val = zip(*updates)
        idx = torch.tensor(idx, dtype=torch.long).pin_memory().to(self.device, non_blocking=True)
        val = torch.tensor(val, dtype=torch.long).pin_memory().to(self.device, non_blocking=True)
        self.stage_flat.index_copy_(0, idx, val)

    def _to_host(self, l: int, e: int, row: int):
        self.row_of[l, e] = row
        self.slot_of[l, e] = -1
        self.live[l, e] = False
        self.offloaded[l, e] = True
        d = [(self._entry(l, j, e), self._host_addr(l, j, row)) for j in range(3)]
        st = [(self._entry(l, j, e), self._window_addr(l, j, row)) for j in range(3)] if self.stage is not None else []
        return d, st

    def _to_slot(self, l: int, e: int, s: int):
        self.row_of[l, e] = -1
        self.slot_of[l, e] = s
        self.live[l, e] = True
        self.offloaded[l, e] = False
        self.owner[l][s] = e
        d = [(self._entry(l, j, e), self._view_addr(l, s, j)) for j in range(3)]
        return d, list(d)

    # ------------------------------------------------------------ the step

    def step_begin(self) -> None:
        moved, updates, stage_updates = set(), [], []
        # 3. landed promotions: X onto its slot, and its old row is free
        while self.pending_p and self.pending_p[0][0].query():
            _, done = self.pending_p.pop(0)
            for l, x, s, h in done:
                d, st = self._to_slot(l, x, s)
                updates += d
                stage_updates += st
                self.free_rows[l].append(h)
                self.reserved[l].pop(s, None)
                moved.add((l, x))
        # 2. landed write-backs: Y onto its row; X's copy into the slot follows, behind these flips
        follow = []
        while self.pending_d and self.pending_d[0][0].query():
            _, done = self.pending_d.pop(0)
            for l, x, y, s, f in done:
                d, st = self._to_host(l, y, f)
                updates += d
                stage_updates += st
                self.busy[l, y] = False
                follow.append((l, x, s, int(self.row_of[l, x])))
                moved.add((l, y))
        self._write(updates)
        self._write_stage(stage_updates)
        if follow:
            self._copy_in(follow)
        if self.ref is not None:
            self._verify_tick = getattr(self, "_verify_tick", 0) + 1
            if self._verify_tick % 512 == 0:
                self._verify(range(len(self.layers)), full=True)
            elif moved:
                self._verify(sorted({l for l, _ in moved}), experts=moved)
        routed = None
        while self.steps and self.steps[0][1].query():
            idx, _, tokens, live, step_no = self.steps.pop(0)
            routed = (idx, tokens, live, step_no)
        if routed is not None:
            self._plan(*routed)

    def _copy_in(self, follow) -> None:
        main = torch.cuda.current_stream(self.device)
        ready = torch.cuda.Event()
        ready.record(main)
        self.side.wait_event(ready)
        with torch.cuda.stream(self.side):
            for l, x, s, h in follow:
                layer = self.layers[l]
                for j, label in enumerate(_LABELS):
                    self.views[l][s][j].copy_(layer.exl3_host_blocks[label].host[h], non_blocking=True)
        done = torch.cuda.Event()
        done.record(self.side)
        self.pending_p.append((done, follow))

    def _plan(self, idx: int, tokens: int, live_then, step_no: int) -> None:
        self.clock += 1
        L = len(self.layers)
        ids = self.ring[idx % _RING, :, :tokens].numpy().reshape(L, -1)
        rows = numpy.repeat(numpy.arange(L), ids.shape[1])
        flat_ids = ids.reshape(-1)
        if self.stats_every and live_then is not None:
            hit = live_then[rows, flat_ids]
            self.hits += int(hit.sum())
            self.misses += int((~hit).sum())
        if self.stats_every:
            self.planned += 1
            self.lag += self.step - step_no
        self.last_use[rows, flat_ids] = self.clock
        want = (self.row_of[rows, flat_ids] >= 0) & (self.slot_of[rows, flat_ids] < 0)
        keys, counts = numpy.unique(rows[want] * self.E + flat_ids[want], return_counts=True)
        order = sorted(range(len(keys)), key=lambda i: (-counts[i], (keys[i] // self.E - self.clock) % L))
        issued = []
        for i in order:
            if len(issued) >= self.budget:
                break
            l, x = divmod(int(keys[i]), self.E)
            if not self.free_rows[l]:
                continue
            s = self._victim(l)
            if s is None:
                continue
            y = int(self.owner[l][s])
            f = self.free_rows[l].pop()
            self.busy[l, y] = True
            self.reserved[l][s] = x
            self.slot_of[l, x] = s          # reserved, so it is not planned again; still read from its row
            issued.append((l, x, y, s, f))
        if issued:
            self._write_back(issued)

    def _victim(self, l: int) -> int | None:
        best, best_use = None, None
        for s, e in enumerate(self.owner[l]):
            if e < 0 or s in self.reserved[l] or not self.live[l, e] or self.busy[l, e]:
                continue
            use = self.last_use[l, e]
            if use >= self.clock:
                continue
            if best is None or use < best_use:
                best, best_use = s, use
        return best

    def _write_back(self, issued) -> None:
        main = torch.cuda.current_stream(self.device)
        ready = torch.cuda.Event()
        ready.record(main)
        self.side.wait_event(ready)
        with torch.cuda.stream(self.side):
            for l, x, y, s, f in issued:
                layer = self.layers[l]
                for j, label in enumerate(_LABELS):
                    layer.exl3_host_blocks[label].host[f].copy_(self.views[l][s][j], non_blocking=True)
        done = torch.cuda.Event()
        done.record(self.side)
        self.pending_d.append((done, issued))
        self.promoted += len(issued)

    # ------------------------------------------------------------ the detector

    def _verify(self, layers, experts=None, full: bool = False) -> None:
        """Synchronize, then check that each expert's decode (and staged) entries point at its current home
        and that the home holds its bytes. Raises on the first mismatch: a broken swap loses the only copy."""
        torch.cuda.synchronize(self.device)
        table = self.table.cpu().numpy()
        stage = self.stage.cpu().numpy() if self.stage is not None else None
        todo = [(l, e) for l in layers for e in range(self.E)] if full else sorted(experts)
        for l, e in todo:
            row = self.row_of[l, e]
            for j in range(3):
                if row >= 0:
                    want, want_stage = self._host_addr(l, j, row), self._window_addr(l, j, row) if stage is not None else None
                else:
                    s = self.slot_of[l, e]
                    if s < 0 or self.owner[l][s] != e:
                        raise RuntimeError(f"expert cache: layer {l} expert {e} has no home (slot {s})")
                    want = want_stage = self._view_addr(l, s, j)
                if table[l, j, e] != want:
                    raise RuntimeError(f"expert cache: layer {l} expert {e} {_LABELS[j]} entry {table[l, j, e]:#x} "
                                       f"is not its home {want:#x} (row {row})")
                if stage is not None and stage[l, j, e] != want_stage:
                    raise RuntimeError(f"expert cache: layer {l} expert {e} {_LABELS[j]} staged entry is wrong")
                got = _checksum(self._home(l, j, e))
                if got != self.ref[l, j, e]:
                    raise RuntimeError(f"expert cache: layer {l} expert {e} {_LABELS[j]} holds the wrong bytes "
                                       f"(row {row}, slot {self.slot_of[l, e]})")


def decode_rows(scheduler_output) -> int:
    """How many of the step's leading rows are decodes (vLLM sorts decodes first; see the module
    docstring). Under speculative decoding a decode is its token plus the drafts being verified,
    all routed through the target model, so a request counts as many rows as it scheduled. Counting
    only one-token requests there read every step as decode-free, and the cache never learned."""
    spec = getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
    return sum(n for r, n in scheduler_output.num_scheduled_tokens.items()
               if n == 1 + len(spec.get(r, ())))


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
                cache.step_end(tokens, decode_rows(scheduler_output))
            return out

        cls.execute_model = execute_model
        cls._exl3_cache_patched = True


_PENDING: list = []
#: A cache that holds every hosted expert, kept alive with no controller.
_STATIC: "ExpertCache | None" = None


def in_vram(layer):
    """Per expert of a hosted layer, whether its decode entry points into VRAM (a slot), as a numpy bool
    array; None when there is no cache (every hosted expert is read from the host)."""
    if CACHE is None:
        return None
    idx = getattr(CACHE, "_index", None)
    if idx is None:
        idx = CACHE._index = {id(l): i for i, l in enumerate(CACHE.layers)}
    i = idx.get(id(layer))
    return None if i is None else CACHE.live[i]


def prepare(layers: list) -> None:
    """Called at the end of load with the layers that hold host blocks. The cache is built
    at vLLM's `determine_available_memory`, once the memory profile says what is free."""
    from . import cpu_offload

    if not layers or cpu_offload.cache_mode() == "off":
        return
    _PENDING[:] = layers
    _patch_worker()


def _patch_worker() -> None:
    from functools import wraps

    from vllm.v1.worker import gpu_worker

    cls = gpu_worker.Worker
    if getattr(cls, "_exl3_cache_sizing", False):
        return
    original = cls.determine_available_memory

    @wraps(original)
    def determine_available_memory(self):
        available = original(self)
        if not _PENDING:
            return available
        layers = list(_PENDING)
        _PENDING.clear()
        return available - build(layers, available, self)

    cls.determine_available_memory = determine_available_memory
    cls._exl3_cache_sizing = True


def _kv_reserve(worker) -> tuple[int, str]:
    """Bytes of KV to keep, and how they were arrived at, for the log."""
    mult = float(os.environ.get("EXL3_KV_RESERVE", "1.0"))
    config = worker.vllm_config
    model = config.model_config
    if getattr(model, "original_max_model_len", None) == -1:
        saved, model.max_model_len = model.max_model_len, _AUTO_LEN_TOKENS
        try:
            need = _kv_needed(config, worker.get_kv_cache_spec(), mult)
        finally:
            model.max_model_len = saved
        return need, f"{mult:g} x {_AUTO_LEN_TOKENS} tokens (--max-model-len auto)"
    need = _kv_needed(config, worker.get_kv_cache_spec(), mult)
    return need, f"{mult:g} x --max-model-len {model.max_model_len}"


def _kv_needed(config, specs: dict, mult: float) -> int:
    """KV bytes for `mult` requests at the current max_model_len, by the arithmetic vLLM's
    own admission check uses (`get_kv_cache_configs`): KV cache groups, which pad hybrid
    models' layers to uniform group sizes, at block granularity, plus the null block the
    block pool holds back. The plain per-layer sum (`max_memory_usage_bytes`) falls short of
    that on hybrid models -- Qwen3.6 with nvfp4 KV missed by ~600 tokens of 131072 and
    refused to start. Single worker only, which is what this plugin sizes for.
    """
    from dataclasses import replace

    from vllm.v1.core import kv_cache_utils as kv
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    try:
        merged = dict(specs)
        extra = max(0, config.num_prefill_lookahead_tokens - 1)
        for name, spec in merged.items():
            if isinstance(spec, SlidingWindowSpec):
                merged[name] = replace(spec, extra_retained_tokens=extra)
        groups = kv.get_kv_cache_groups(config, merged)
        groups = [g for g in groups if not g.host_resident]
        if not groups:
            return 0
        return (int(mult * kv._max_memory_usage_bytes_from_groups(config, groups))
                + kv._pool_bytes_per_block(config, groups))
    except (AttributeError, TypeError, ValueError) as e:  # vLLM internals moved
        logger.warning_once(
            "expert cache: could not size KV the way vLLM checks it (%s); using the per-layer "
            "sum with a 2%% margin", e,
        )
        return int(1.02 * mult * kv.max_memory_usage_bytes(config, specs.values()))


def _uniform_hosting(layers) -> bool:
    """The exclusive cache moves an expert's three projections together, so each layer must host all three
    for the same experts at the same rows (true of the default selectors; a projection-specific regex is not)."""
    for layer in layers:
        blocks = layer.exl3_host_blocks
        if set(blocks) != set(_LABELS) or any(blocks[label].rows != blocks["gate"].rows for label in _LABELS):
            logger.warning_once("expert cache: projections are hosted unevenly; EXL3_CACHE_EXCLUSIVE needs all "
                                "three hosted for the same experts, so the inclusive cache is used")
            return False
    return True


def build(layers: list, available: int, worker) -> int:
    """Size and allocate the slots; returns the bytes taken from what KV would have had."""
    global CACHE
    from . import cpu_offload

    E = layers[0].exl3_num_experts
    first = layers[0]
    slot_bytes = sum(b.host[0].numel() * b.host.element_size()
                     for b in first.exl3_host_blocks.values())
    per_layer_max = max(max(len(b.rows) for b in l.exl3_host_blocks.values()) for l in layers)
    hosted = cpu_offload._STATS["expert_bytes"]
    given_back = 0   # bytes freed here that KV would otherwise not have had
    if cpu_offload.cache_mode() == "fixed":
        slots, how = min(_slots_per_layer(E), per_layer_max), "fixed by EXL3_EXPERT_SLOTS"
        reserve = None
    else:
        reserve, how = _kv_reserve(worker)
        cap = None
        if getattr(worker.vllm_config.model_config, "original_max_model_len", None) == -1:
            # auto: slots up to the target share, counting the experts that stayed resident
            streams = float(os.environ.get("EXL3_KV_RESERVE", "1.0"))
            cap = max(math.ceil(target_share(streams) * E) - (E - per_layer_max), 0)
            how += f"; slots stop at {100 * target_share(streams):.0f}% of experts for {streams:g} stream(s)"
        slots = size_slots(available, reserve, hosted, slot_bytes, len(layers), per_layer_max, cap)
        extra = cpu_offload.prefetch_window_bytes()
        if extra:
            # the second staging window (prefill prefetch) is dropped only when it is what takes
            # residency below the target; under it either way, prefill keeps its 20-39%
            target = target_share(float(os.environ.get("EXL3_KV_RESERVE", "1.0")))
            freed_slots = size_slots(available + extra, reserve, hosted, slot_bytes, len(layers), per_layer_max, cap)
            kept, freed = (E - per_layer_max + slots) / E, (E - per_layer_max + freed_slots) / E
            if kept < target <= freed:
                dropped = cpu_offload.drop_prefetch()
                available += dropped
                given_back += dropped
                slots = freed_slots
                logger.info(
                    "expert cache: prefill prefetch off; its second staging window (%.0f MiB) would have left "
                    "%.0f%% of experts in VRAM, under the %.0f%% target (EXL3_STAGE_PREFETCH=1 keeps it)",
                    dropped / (1 << 20), 100 * kept, 100 * target,
                )
    if slots <= 0:
        logger.warning(
            "expert cache: no VRAM left for slots after the KV reserve (%s); experts are "
            "read from host every token. Lower EXL3_KV_RESERVE or --max-model-len.", how,
        )
        return -given_back
    cls = ExpertCache
    if EXCLUSIVE and slots < per_layer_max and _uniform_hosting(layers):
        cls = ExclusiveCache
    cache = cls(layers, slots, int(os.environ.get("EXL3_PROMOTE_PER_STEP", "32")))
    used = sum(t.numel() * t.element_size() for d in cache.slot_tensors for t in d.values())
    if cache.holds_everything:
        # Every hosted expert has a slot for good: no controller, no routing, and prefill
        # reads the slots too, so the staging window goes back to KV.
        global _STATIC
        _STATIC = cache
        windows = cpu_offload._STATS.pop("windows", None) or []
        for layer in layers:
            layer.exl3_stage = None
            layer.exl3_route = layer.exl3_route_own   # nothing reads it now, but the graph keeps its shape
        freed = sum(w.numel() for w in windows)
        del windows
        torch.cuda.empty_cache()  # back to the driver, so KV's allocation can use it
        logger.info(
            "expert cache: every offloaded expert fits in VRAM (%d slots per layer, %.0f MiB); "
            "they are served from there, KV keeps %.2f GiB, and the host copy goes unused. "
            "Offload was not needed at this --max-model-len and EXL3_KV_RESERVE.",
            slots, used / (1 << 20), (available - used + freed) / (1 << 30),
        )
        return used - freed - given_back
    CACHE = cache
    _patch_runners()
    in_vram = (E - per_layer_max + slots) / E
    streams = 2 ** ((in_vram - 0.4) / 0.1) if in_vram >= 0.4 else 0
    logger.info(
        "expert cache: %d slots per layer over %d layers (%.0f MiB), %.0f%% of experts in "
        "VRAM; KV keeps %.2f GiB (reserve: %s). Measured on gemma-4-26B, that holds "
        "decode within ~10%% of resident for %s concurrent streams.",
        slots, len(layers), used / (1 << 20), 100 * in_vram,
        (available - used) / (1 << 30), how,
        "more than 16" if streams > 16 else f"~{streams:.0f}" if streams >= 1
        else "fewer than 1 (single-stream decode will be slower)",
    )
    return used - given_back
