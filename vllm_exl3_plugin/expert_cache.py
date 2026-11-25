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
import time

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


def target_in_vram(experts: int, streams: float) -> int:
    """Experts per layer this rank keeps in VRAM at `target_share`, resident and slots together.

    Under an uneven expert split (fused_moe.rank_widths) every rank spends the same expert bytes, as
    `cpu_offload._host_quota` does for what stays resident: KV is sized to the fullest rank, and an equal count on
    Qwen4Exp's 384 + 256 split left rank 0 2.3 GiB fuller than rank 1 at TP=2 (2026-11-17). The bytes counted are
    the L layers' VRAM rows plus the k prefill staging windows, each a layer's hosted rows (E - n + spare): equal
    totals at the mean width's share give n_r = (w_mean/w_r * (L n + k (E - n + s)) - k (E + s)) / (L - k), which is
    the share itself when the widths agree. The wider rank covers less (32% against 51% there)."""
    from . import cpu_offload

    n = target_share(streams) * experts
    widths, rank = cpu_offload._uneven_widths()
    if widths is not None:
        L = len(cpu_offload._STATS["moe_layers"])
        k, s = cpu_offload.planned_windows(L), cpu_offload._SPARE_ROWS
        scale = sum(widths) / len(widths) / widths[rank]
        if L > k:
            n = (scale * (L * n + k * (experts - n + s)) - k * (experts + s)) / (L - k)
        else:   # layers not declared yet: VRAM rows alone
            n *= scale
    return max(0, min(experts, math.ceil(n)))


def slot_row_bytes(layers: list) -> int:
    """One slot's bytes per layer, the mean over the layers as they store their experts, rounded up so slots never
    outgrow what was measured. Sized from the first layer alone, 1.66bpw Qwen3.8-Flash-Next (its first layer one of
    eight wider ones) costed a slot at 56 MiB across the layers against 45, leaving 21% of the slots' VRAM to KV."""
    total = sum(b.host[0].numel() * b.host.element_size() for l in layers for b in l.exl3_host_blocks.values())
    return -(-total // max(len(layers), 1))


def size_slots(available: int, reserve: int, hosted: int, slot_bytes: int,
               layers: int, per_layer_max: int, cap: int | None = None) -> int:
    """Slots per layer for the auto mode.

    `available` is the KV memory vLLM measured with every hosted expert off the device,
    `reserve` the KV to keep, `hosted` the hosted experts' bytes, and `slot_bytes` one slot's mean per layer
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


#: Tokens the reserve covers when --max-model-len is auto, before the multiplier: the context guaranteed before slots
#: take VRAM (``EXL3_AUTO_LEN_TOKENS``). 4096 let the residency target take a 16 GiB card and leave QFN 1.66bpw 6,272
#: tokens; a floor this side of 16-32K is not a useful default.
_AUTO_LEN_TOKENS = int(os.environ.get("EXL3_AUTO_LEN_TOKENS", "32768"))


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


#: Disk experts decode may move into host rows a step (``EXL3_DISK_ADMIT_PER_STEP``); 0, the default, admits every
#: routed one, as tier_replay.py models it. Each admission is one more checkpoint read, at the next step's start.
_ADMIT_BUDGET = int(os.environ.get("EXL3_DISK_ADMIT_PER_STEP", "0"))

#: Promotions a step, across all layers (``EXL3_PROMOTE_PER_STEP``). Past a few, more only churn experts that are about
#: equally hot, and their copies compete with decode's reads over the link: Qwen3.6-35B-A3B 4.00bpw at 59% in VRAM
#: decoded 157 t/s at 4 streams with 32, 170 with 8, at the same hit rate; gemma-4-26B 4.10bpw at 10% of experts in
#: slots lost hit rate to it (51% at 4, 43% at 32) and decoded 35.9 ms a token at 32 against 30.3 at 8. 0 leaves the
#: placement's hit rate (60% there) and halves 4-stream decode; 8 is within a few percent of the best everywhere.
_PROMOTE_PER_STEP = int(os.environ.get("EXL3_PROMOTE_PER_STEP", "8"))

#: `row_of` for an expert with no host row: the disk tier's (cpu_offload.disk_tier), read from the checkpoint on use
DISK = -2


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
        self.pending_a: list = []                     # admissions from disk: (event, [(l, x, z, r)])
        self.admitting: list[set] = [set() for _ in self.layers]
        self.evicting: list[set] = [set() for _ in self.layers]   # pending admissions' Z, at a row's window position
        self.read_admitted: list = []                 # (l, e) moved by admit_at_read, verified at the next step
        self.admitted = 0
        self.disk_on = any(getattr(layer, "exl3_disk_pos", None) for layer in self.layers)
        self.stage = None
        if self.layers[0].exl3_stage is not None:
            self.stage = torch.empty((L, 3, E), dtype=torch.long, device=self.device)
            self.stage_flat = self.stage.view(-1)
        owners, updates, stage_updates = [], [], []
        for l, layer in enumerate(self.layers):
            blocks = [layer.exl3_host_blocks[label] for label in _LABELS]
            rows, disk = blocks[0].rows, blocks[0].disk
            resident = [e for e in range(E) if e not in rows and e not in disk]
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
            for e in disk:
                self.row_of[l, e] = DISK
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
        self._backfill()
        self.ref = None
        if VERIFY:
            self.ref = numpy.zeros((L, 3, E), dtype=object)
            for l in range(L):
                for e in range(E):
                    if self.row_of[l, e] == DISK:
                        # from the checkpoint, so an admission's fill is checked against bytes it did not read itself
                        from . import cpu_offload
                        got = cpu_offload.read_expert(self.layers[l], e)
                        for j in range(3):
                            self.ref[l, j, e] = _checksum(got[_LABELS[j]])
                        continue
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
        # 4. admissions whose rows nothing queued can read any more: X read from disk into its row
        while self.pending_a and self.pending_a[0][0].query():
            _, done = self.pending_a.pop(0)
            d, st = self._fill_rows(done)
            updates += d
            stage_updates += st
            moved.update((l, x) for l, x, _, _ in done)
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
        if self.read_admitted:
            moved.update(self.read_admitted)
            self.read_admitted = []
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
        if self.disk_on:
            self._admit(rows, flat_ids)

    # ------------------------------------------------------------ late slots

    def grow(self, n: int) -> int:
        """Add up to `n` slots to every layer, filled from host rows; returns the bytes allocated.

        Late slots (TODO `late-slots`): placed once the KV cache has its final size, so they take what is actually
        left rather than a prediction of it. Graph-safe because decode reaches every expert through `self.table`,
        whose address graphs captured and whose contents only ever change by scatter: a new slot is a new entry
        value, as a promotion is. New slots start filled -- the lowest-id experts still in host rows, as `_fill`
        chooses -- since the promotion search never picks an empty slot. In-flight moves are drained first: warmup
        steps run the controller too."""
        if n <= 0:
            return 0
        torch.cuda.synchronize(self.device)
        for _ in range(64):
            if not (self.pending_d or self.pending_p or self.pending_a):
                break
            self.step_begin()
            torch.cuda.synchronize(self.device)
        else:
            raise RuntimeError("expert cache: in-flight moves did not drain before growing the slots")
        added, moves = 0, []
        for l, layer in enumerate(self.layers):
            candidates = [int(e) for e in numpy.flatnonzero((self.row_of[l] >= 0) & (self.slot_of[l] < 0))
                          if not self.busy[l, e]]
            take = candidates[:n]
            if not take:
                continue
            chunk = [torch.empty((len(take), *layer.exl3_host_blocks[label].host.shape[1:]),
                                 dtype=layer.exl3_host_blocks[label].host.dtype, device=self.device)
                     for label in _LABELS]
            added += sum(t.numel() * t.element_size() for t in chunk)
            first = len(self.views[l])
            self.views[l] += [tuple(chunk[j][k] for j in range(3)) for k in range(len(take))]
            self.owner[l] = numpy.concatenate([self.owner[l], numpy.full(len(take), -1, dtype=numpy.int64)])
            self.capacity[l] = len(self.views[l])
            for k, e in enumerate(take):
                row = int(self.row_of[l, e])
                for j, label in enumerate(_LABELS):
                    self.views[l][first + k][j].copy_(layer.exl3_host_blocks[label].host[row], non_blocking=True)
                moves.append((l, e, first + k, row))
        torch.cuda.synchronize(self.device)
        updates, stage_updates = [], []
        for l, e, s, row in moves:
            d, st = self._to_slot(l, e, s)
            updates += d
            stage_updates += st
            self.free_rows[l].append(row)
        self._write(updates)
        self._write_stage(stage_updates)
        torch.cuda.synchronize(self.device)
        self._backfill()
        if self.ref is not None:
            self._verify(range(len(self.layers)), full=True)
        return added

    def release_host_tier(self, worker) -> int:
        """When the slots hold every expert -- nothing in a host row, nothing on disk -- the host rows hold nothing and
        are returned to the driver, prefill staging off with them. Each is one pinned allocation, so only all of it can
        go (a partly idle host tier stays, reported by `report_tiers`). Not when a captured CUDA graph could copy out
        of them: late growth runs after capture, and a graph over `STAGE_MIN_TOKENS` tokens captured `stage_all`'s
        copies from the rows. Returns the bytes freed."""
        from . import cpu_offload

        if (self.row_of != -1).any():
            return 0
        config = worker.vllm_config
        sizes = getattr(config.compilation_config, "cudagraph_capture_sizes", None) or []
        if not config.model_config.enforce_eager and sizes and max(sizes) > cpu_offload.STAGE_MIN_TOKENS:
            logger.info("expert cache: every expert is in VRAM, but the host rows stay: CUDA graphs up to %d tokens may "
                        "copy out of them (prefill staging above %d)", max(sizes), cpu_offload.STAGE_MIN_TOKENS)
            return 0
        torch.cuda.synchronize(self.device)
        freed = 0
        for layer in self.layers:
            layer.exl3_stage = None
            for block in layer.exl3_host_blocks.values():
                host = block.host
                block.host = block.view = torch.empty((0, *host.shape[1:]), dtype=host.dtype)
                freed += cpu_offload.free_pinned(host)
                del host
        if freed:
            logger.info("expert cache: every expert is in VRAM; %.2f GiB of host rows returned", freed / (1 << 30))
        return freed

    def report_tiers(self) -> None:
        """One line on where the experts ended up, once placement is final: VRAM (resident and slots), host rows, disk,
        each as bytes and share of all routed experts, and the pinned host memory behind the rows -- allocated once at
        load and never returned, so rows the slots emptied stay pinned (idle) unless the disk tier refills them."""
        from . import cpu_offload

        gib = 1 << 30
        vram = held = disk = pinned = spare = 0.0
        n_vram = n_held = n_disk = 0
        for l, layer in enumerate(self.layers):
            blocks = layer.exl3_host_blocks
            rb = sum(cpu_offload._row_bytes(b) for b in blocks.values())
            held_rows = next(iter(blocks.values())).host.shape[0] if blocks else 0
            spare += min(len(self.free_rows[l]), cpu_offload._SPARE_ROWS, held_rows) * rb
            v, h, d = int((self.slot_of[l] >= 0).sum()), int((self.row_of[l] >= 0).sum()), int((self.row_of[l] == DISK).sum())
            n_vram, n_held, n_disk = n_vram + v, n_held + h, n_disk + d
            vram, held, disk = vram + v * rb, held + h * rb, disk + d * rb
            pinned += sum(cpu_offload._row_bytes(b) * b.host.shape[0] for b in blocks.values())
        total = max(n_vram + n_held + n_disk, 1)
        logger.info("expert placement: VRAM %.2f GiB (%.0f%%), host RAM %.2f GiB (%.0f%%) in %.2f GiB pinned "
                    "(%.2f GiB of spare rows to swap through, %.2f GiB idle), disk %.2f GiB (%.0f%%)", vram / gib,
                    100 * n_vram / total, held / gib, 100 * n_held / total, pinned / gib, spare / gib,
                    max(pinned - held - spare, 0) / gib, disk / gib, 100 * n_disk / total)

    # ------------------------------------------------------------ the disk tier

    def _backfill(self) -> None:
        """Refill free host rows from disk, keeping the spares promotions swap through (`cpu_offload._SPARE_ROWS`).
        Filling slots from rows frees rows; under the disk tier the host should still hold all its budget allows, so
        what stays on disk is only what neither VRAM nor host RAM holds. Only while nothing runs: at build, and after
        late slots grow, both synchronized."""
        if not self.disk_on:
            return
        from . import cpu_offload

        store, spare = cpu_offload._DISK.get("store"), cpu_offload._SPARE_ROWS
        if store is None:
            return
        todo = []
        for l, layer in enumerate(self.layers):
            disk = [int(e) for e in numpy.flatnonzero(self.row_of[l] == DISK)]
            while len(self.free_rows[l]) > spare and disk:
                todo.append((l, disk.pop(0), self.free_rows[l].pop()))
        if not todo:
            return

        def one(item):
            l, x, r = item
            layer = self.layers[l]
            cpu_offload.read_expert(layer, x, {label: b.host[r] for label, b in layer.exl3_host_blocks.items()})

        list(cpu_offload._DISK["pool"].map(one, todo))
        updates, stage_updates = [], []
        for l, x, r in todo:
            self.layers[l].exl3_disk_pos.pop(x)
            d, st = self._to_host(l, x, r)
            updates += d
            stage_updates += st
        self._write(updates)
        self._write_stage(stage_updates)
        torch.cuda.synchronize(self.device)
        left = int((self.row_of == DISK).sum())
        cpu_offload._STATS["disk_experts"] = left
        logger.info("expert cache: %d disk experts refilled freed host rows; %d stay on disk (%.1f per layer)",
                    len(todo), left, left / max(len(self.layers), 1))

    def _admit(self, rows, flat_ids) -> None:
        """Decode's routed disk experts move into host rows, the most-routed first (up to `_ADMIT_BUDGET`): each evicts
        the least recently used expert in a row (Z), which goes back to disk-only -- the checkpoint holds it, so
        nothing is written back. Z flips now: later ops read it from disk, staged at the vacated row's window
        position. X is read into the row only once everything already queued, which could still be copying Z's
        row into the window, has run (`_fill_rows`, from step_begin); until then it is read per use, as before.
        Prefill routing never reaches here (the routing ring holds decode rows), so prefill bypasses RAM."""
        want = self.row_of[rows, flat_ids] == DISK
        keys, counts = numpy.unique(rows[want] * self.E + flat_ids[want], return_counts=True)
        issued, updates, stage_updates = [], [], []
        for i in numpy.argsort(-counts, kind="stable"):
            if _ADMIT_BUDGET and len(issued) >= _ADMIT_BUDGET:
                break
            l, x = divmod(int(keys[i]), self.E)
            if x in self.admitting[l]:
                continue
            z = self._row_victim(l)
            if z is None:
                continue
            r = int(self.row_of[l, z])
            self.row_of[l, z] = DISK
            self.offloaded[l, z] = False
            self.layers[l].exl3_disk_pos[z] = r     # the row's window position, unused until X lands there
            self.evicting[l].add(z)
            updates += [(self._entry(l, j, z), 0) for j in range(3)]
            stage_updates += [(self._entry(l, j, z), self._window_addr(l, j, r)) for j in range(3)]
            self.admitting[l].add(x)
            issued.append((l, x, z, r))
        if not issued:
            return
        self._write(updates)
        self._write_stage(stage_updates)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        self.pending_a.append((ready, issued))

    def _row_victim(self, l: int, routed=None) -> int | None:
        """The least recently used expert in a host row that no promotion has reserved, and not used this step; nor,
        given `routed` (a mask over experts), routed by the batch being staged, whose row copies may still be queued:
        `last_use` lags routing by a step or more."""
        row_of, slot_of, use = self.row_of[l], self.slot_of[l], self.last_use[l]
        ok = (row_of >= 0) & (slot_of < 0) & (use < self.clock)
        if routed is not None:
            ok &= ~routed
        cand = numpy.flatnonzero(ok)
        if not len(cand):
            return None
        return int(cand[numpy.argmin(use[cand])])

    def layer_index(self, layer) -> int | None:
        index = getattr(self, "_layer_index", None)
        if index is None:
            index = self._layer_index = {id(m): i for i, m in enumerate(self.layers)}
        return index.get(id(layer))

    def verify_staged_bytes(self, layer, experts, tables) -> None:
        """EXL3_CACHE_VERIFY: the bytes at each routed expert's staged address are that expert's, by checksum --
        after synchronizing, so every queued copy has landed. Entries can be right while the bytes behind them are
        not: a host row overwritten before its window copy ran left a live entry pointing at another expert."""
        l = self.layer_index(layer)
        if l is None or self.ref is None:
            return
        torch.cuda.synchronize(self.device)
        for j, label in enumerate(_LABELS):
            entries = tables[j].cpu()
            dst, row_bytes = layer.exl3_stage_plan[label]
            base, end = dst.data_ptr(), dst.data_ptr() + dst.numel()
            for e in experts:
                a = int(entries[e])
                if base <= a < end:
                    got = dst[a - base:a - base + row_bytes]
                elif self.slot_of[l, e] >= 0 and a == self._view_addr(l, int(self.slot_of[l, e]), j):
                    got = self.views[l][int(self.slot_of[l, e])][j]
                elif self.row_of[l, e] >= 0 and a == self._host_addr(l, j, int(self.row_of[l, e])):
                    got = layer.exl3_host_blocks[label].host[int(self.row_of[l, e])]
                else:
                    raise RuntimeError(f"expert cache: layer {l} expert {e} {label} is staged at {a:#x}, none of its "
                                       f"homes")
                if _checksum(got) != self.ref[l, j, e]:
                    raise RuntimeError(f"expert cache: layer {l} expert {e} {label}: the staged bytes are not its own")

    def admit_at_read(self, layer, xs, routed) -> dict:
        """Decode's routed disk experts `xs` take host rows now, read straight into them (TODO `disk-tier`): one read
        an admission, where `_admit` reads each again a step later. Called inside the layer's MoE op once routing has
        been read back -- which synchronized the forward stream -- and before its staging, so the only readers of host
        rows still in flight are promotions (row -> slot, whose experts hold a reserved slot and are never victims) and
        write-backs (slot -> free row, never an owned row). The victim Z, unrouted by this batch, goes to disk and takes
        X's window position; X's entries point at the row (decode) and the row's window position (staged). Returns
        {X: row}; the caller reads each X into its row and copies it to the window from there."""
        l = self.layer_index(layer)
        if l is None:
            return {}
        pos = layer.exl3_disk_pos
        mask = numpy.zeros(self.E, dtype=bool)
        mask[[e for e in range(self.E) if routed(e)]] = True
        got, updates, stage_updates = {}, [], []
        for x in xs:
            if _ADMIT_BUDGET and len(got) >= _ADMIT_BUDGET:
                break
            # a pending admission's X, or its Z -- whose window position is the row X is about to fill -- waits for it
            if x in self.admitting[l] or x in self.evicting[l] or self.row_of[l, x] != DISK:
                continue
            z = self._row_victim(l, mask)
            if z is None:
                break
            r = int(self.row_of[l, z])
            self.row_of[l, z] = DISK
            self.offloaded[l, z] = False
            pos[z] = pos.pop(x)
            updates += [(self._entry(l, j, z), 0) for j in range(3)]
            stage_updates += [(self._entry(l, j, z), self._window_addr(l, j, pos[z])) for j in range(3)]
            d, st = self._to_host(l, x, r)
            updates += d
            stage_updates += st
            self.last_use[l, x] = self.clock      # not a victim again before its routing is planned
            got[x] = r
            self.read_admitted.append((l, x))
            self.read_admitted.append((l, z))
        if got:
            self._write(updates)
            self._write_stage(stage_updates)
            self.admitted += len(got)
        return got

    def _fill_rows(self, done):
        """Read each admitted X into its row, then flip it there; Z takes X's window position on disk."""
        from . import cpu_offload

        store = cpu_offload._DISK["store"]

        def one(item):
            l, x, _, r = item
            layer = self.layers[l]
            cpu_offload.read_expert(layer, x, {label: b.host[r] for label, b in layer.exl3_host_blocks.items()})

        t = time.perf_counter()
        list(cpu_offload._DISK["pool"].map(one, done))
        cpu_offload._STATS["disk_fill_s"] = cpu_offload._STATS.get("disk_fill_s", 0.0) + time.perf_counter() - t
        updates, stage_updates = [], []
        for l, x, z, r in done:
            pos = self.layers[l].exl3_disk_pos
            pos[z] = pos.pop(x)
            stage_updates += [(self._entry(l, j, z), self._window_addr(l, j, pos[z])) for j in range(3)]
            d, st = self._to_host(l, x, r)
            updates += d
            stage_updates += st
            self.admitting[l].discard(x)
            self.evicting[l].discard(z)
        self.admitted += len(done)
        cpu_offload._STATS["disk_admitted"] = cpu_offload._STATS.get("disk_admitted", 0) + len(done)
        return updates, stage_updates

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
            if row == DISK:
                # no home but the checkpoint: its entries stay null, so an unstaged use faults, and its staged
                # entries point at its window position, where routed staging reads it
                if any(table[l, j, e] != 0 for j in range(3)):
                    raise RuntimeError(f"expert cache: layer {l} expert {e} is on disk but has a live entry")
                pos = self.layers[l].exl3_disk_pos[e]
                if stage is not None and any(stage[l, j, e] != self._window_addr(l, j, pos) for j in range(3)):
                    raise RuntimeError(f"expert cache: layer {l} expert {e} is on disk but its staged entry is not "
                                       f"its window position {pos}")
                continue
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

    # Late slots, phase B: after the estimate-sized KV cache's warmup, or once an extensible one is extended
    warm = cls.compile_or_warm_up_model

    @wraps(warm)
    def compile_or_warm_up_model(self, *args, **kwargs):
        result = warm(self, *args, **kwargs)
        if _LATE and not getattr(self.cache_config, "enable_extensible_kv_cache", False):
            grow_late(self, extensible=False)
        return result

    cls.compile_or_warm_up_model = compile_or_warm_up_model
    extend = getattr(cls, "extend_kv_cache", None)
    if extend is not None:
        @wraps(extend)
        def extend_kv_cache(self, *args, **kwargs):
            result = extend(self, *args, **kwargs)
            if _LATE:
                grow_late(self, extensible=True)
            return result

        cls.extend_kv_cache = extend_kv_cache
    cls._exl3_cache_sizing = True


#: Warmup's allocated peak over the profile's: 1.03-1.05x measured under late slots (2026-11-25). It was 1.4, from torch's
#: reservation growth (1.08-1.36x on 2026-11-21), which the keep now covers separately at 1.1x its measurement.
_EXTENSIBLE_GROWTH = 1.05


def _extensible_keep(worker) -> int | None:
    """The extensible KV cache's keep, predicted from the profiled peak by its own formula; None when not extensible."""
    if not getattr(worker.vllm_config.cache_config, "enable_extensible_kv_cache", False):
        return None
    try:
        from vllm.v1.worker import extensible_kv_cache as ext
    except ImportError:
        return None
    transient = int(int(getattr(worker, "peak_activation_memory", 0) or 0) * _EXTENSIBLE_GROWTH)
    floor_of = getattr(ext, "kv_cache_margin_floor_bytes", None)   # the fork scales it to the device
    floor = floor_of(worker.device) if floor_of is not None else ext.KV_CACHE_MARGIN_FLOOR_BYTES
    keep_of = getattr(ext, "kv_cache_keep_bytes", None)
    if keep_of is not None:
        reserve_of = getattr(ext, "kv_cache_non_torch_reserve_bytes", None)
        nt = reserve_of(worker.device) if reserve_of is not None else ext.KV_CACHE_NON_TORCH_RESERVE_BYTES
        return keep_of(transient, transient, floor, ext.KV_CACHE_MARGIN_FRACTION, nt)
    return transient + max(floor, int(transient * ext.KV_CACHE_MARGIN_FRACTION))


def _extensible_withholding(worker) -> int:
    """What the extensible KV cache keeps free beyond the profiled peak the estimate is built on (up-front slots)."""
    keep = _extensible_keep(worker)
    return 0 if keep is None else max(keep - int(getattr(worker, "peak_activation_memory", 0) or 0), 0)


def _late_slot_loss(worker, spill: bool = False) -> int:
    """What the slot region planned at --max-model-len auto loses once warmup has run, so late slots plan around it.

    The extensible KV cache keeps the larger of the allocated peak plus its margin and torch's reservation growth
    (`kv_cache_keep_bytes`); measured, warmup's peak came within 3-5% of the profile's, and the peak branch decided.
    On top of that the slot region gives up CUDA graph memory (~0.14 GiB on every 16 GiB run) and, when slots grow in
    rounds -- rows refilling from disk between them -- allocator rounding on each round's chunks (~0.27 GiB). Under-
    predicting it left ~10% of experts on disk (Qwen3.6-35B-A3B 2.00bpw: 1.29 GiB predicted, 2.02 lost); over-predicting
    costs a little KV past the floor. 0 when the cache is not extensible."""
    keep = _extensible_keep(worker)
    if keep is None:
        return 0
    return keep + _GRAPH_ALLOWANCE + (_ROUNDS_ALLOWANCE if spill else 0)


#: CUDA graph memory the slot region gives up after planning (0.13-0.14 GiB measured on 16 GiB cards)
_GRAPH_ALLOWANCE = 160 << 20
#: Allocator rounding when late slots grow in rounds, the rows refilling from disk between them (~0.27 GiB measured)
_ROUNDS_ALLOWANCE = 320 << 20


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

    try:  # upstream moved KV planning out of kv_cache_utils after 0.33
        from vllm.v1.core import kv_cache_planning as kv
    except ImportError:
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
        if hasattr(kv, "get_global_kv_cache_groups") and hasattr(kv, "_pool_bytes_per_block_from_builder"):
            # v0.34's admission check, exactly: the builder's groups, their per-group block counts (hybrid models pad
            # their layers to uniform groups), and the null block BlockPool holds back, at the builder's pool bytes per
            # block. The per-layer sum below left QFN with fp8 KV and an fp8 indexer 0.11 GiB short at 262144 tokens
            builder = kv.get_kv_cache_config_builder(config)
            groups = kv._project_kv_cache_groups_to_worker(kv.get_global_kv_cache_groups(config, [dict(specs)]),
                                                           dict(specs))
            groups = [g for g in groups if not g.host_resident]
            if not groups:
                return 0
            block = kv._pool_bytes_per_block_from_builder(builder, config, groups)
            return int(mult * kv._max_memory_usage_bytes_from_groups(builder, config, groups)) + block
        if kv.__name__.endswith("kv_cache_planning"):
            # upstream main's `check_enough_kv_cache_memory` before v0.34: the per-layer sum, against what is
            # left after the null block
            # specs as vLLM has them: they carry their own retention there, which the fork's
            # extra_retained_tokens adjustment above would understate
            groups = [g for g in kv.get_kv_cache_groups(config, dict(specs)) if not g.host_resident]
            return (int(mult * kv.max_memory_usage_bytes(config, specs.values()))
                    + (kv._pool_bytes_per_block(config, groups) if groups else 0))
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


#: Late slots, waiting for the KV cache's final size: the cache and how its growth is to be budgeted
_LATE: dict = {}


def _late_slots(worker, layers) -> bool:
    """Whether the slots are placed after the KV cache is sized (TODO `late-slots`): the auto slot count, with the
    exclusive cache, which can grow. At a fixed --max-model-len the slots take everything the KV reserve leaves; under
    --max-model-len auto, the KV cache is planned at what the target share's slots leave and the slots take what is
    left once it is final. ``EXL3_LATE_SLOTS=0`` places them up front everywhere."""
    from . import cpu_offload

    if os.environ.get("EXL3_LATE_SLOTS", "1") == "0" or cpu_offload.cache_mode() == "fixed" or not EXCLUSIVE:
        return False
    return _uniform_hosting(layers)


def _build_late(layers: list, available: int, worker) -> int:
    """Phase A of late slots: an exclusive cache with no slots, and the KV cache planned at exactly its reserve, so
    what is left stays unallocated through warmup and graph capture; `grow_late` fills it once the KV cache's size
    is final. Returns the bytes kept from the KV cache's plan."""
    global CACHE
    reserve, how = _kv_reserve(worker)
    cache = ExclusiveCache(layers, 0, _PROMOTE_PER_STEP)
    CACHE = cache
    _patch_runners()
    slot_bytes = slot_row_bytes(layers)
    _LATE.update(cache=cache, reserve=reserve, slot_bytes=slot_bytes, layers=len(layers),
                 peak=int(getattr(worker, "peak_activation_memory", 0) or 0))
    kept = max(available - reserve, 0)
    if getattr(worker.vllm_config.model_config, "original_max_model_len", None) == -1:
        # auto: KV is satisfied first -- one request at the model's native length (x EXL3_KV_RESERVE) -- and every
        # byte past that goes to slots; vLLM fits the length to the plan, and the slots take what is actually free once
        # the KV cache is final, so a misjudged warmup lands on them. A residency target ("40% is enough") was right
        # only for small experts: Qwen3.6-35B-A3B 4.00bpw on 16 GiB had KV for 3.1 native-length requests at 39% of
        # experts in VRAM, and one request's worth at 57% decoded 11% (1 stream) and 27% (4) faster. Where VRAM cannot
        # hold a native-length request, KV takes it all and vLLM fits the length, as it would without the cache.
        from . import cpu_offload

        native = _kv_needed(worker.vllm_config, worker.get_kv_cache_spec(),
                            float(os.environ.get("EXL3_KV_RESERVE", "1.0")))
        native_len = worker.vllm_config.model_config.max_model_len
        kept = max(min(available - native, available - reserve), 0)
        why = f"one {native_len}-token request ({native / (1 << 30):.2f} GiB)"
        # With experts on disk every slot takes one off it (the rows it empties refill from disk), so the slots may take
        # KV down to its floor until the disk is empty: a disk read costs far more than KV past the floor buys
        per_layer_max = max(max(len(b.rows) + len(b.disk) for b in l.exl3_host_blocks.values()) for l in layers)
        on_disk = max(max(len(b.disk) for b in l.exl3_host_blocks.values()) for l in layers)
        withheld = _late_slot_loss(worker, spill=bool(on_disk))
        if on_disk:
            empty = min(on_disk + cpu_offload._SPARE_ROWS, per_layer_max) * slot_bytes * len(layers) + withheld
            if empty > kept:
                kept = max(min(empty, available - reserve), 0)
                why = f"the floor, {how}, as far as emptying the disk tier ({on_disk} experts a layer on disk) needs"
        logger.info("expert cache: slots are placed after the KV cache is sized; --max-model-len auto plans KV at "
                    "%.2f GiB, for %s, and %.2f GiB stay free for slots (%.2f GiB of them for what the extensible KV "
                    "cache keeps after warmup)", (available - kept) / (1 << 30), why, kept / (1 << 30),
                    withheld / (1 << 30))
        return kept
    logger.info("expert cache: slots are placed after the KV cache is sized; KV is planned at its reserve, %.2f GiB "
                "(%s), and %.2f GiB stay free for slots", reserve / (1 << 30), how, kept / (1 << 30))
    return kept


def grow_late(worker, extensible: bool) -> None:
    """Phase B of late slots: grow the slots into what the final KV cache leaves.

    With the extensible KV cache, what it keeps free is measured, and kept exactly: the larger of the allocated
    transient peak and torch's reservation growth, plus its margin (max(floor, fraction x that), the floor scaled to
    the device where the fork does so). Its torch memory cap was set from free memory before this, so slots
    allocated through torch come out of that allowance and leave it the reserve and margin it meant to keep.
    Without it, the slots stay inside gpu_memory_utilization x total less the profiled peak, as the estimate-based
    sizing reasons."""
    cache = _LATE.pop("cache", None)
    if cache is None:
        return
    from . import cpu_offload

    torch.cuda.synchronize(cache.device)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(cache.device)
    if extensible:
        from vllm.v1.worker import extensible_kv_cache as ext

        kv = getattr(worker.model_runner, "extensible_kv_cache", None)
        peak = getattr(kv, "reserved_headroom_bytes", 0)
        growth = getattr(kv, "torch_reserve_growth_bytes", 0)
        floor_of = getattr(ext, "kv_cache_margin_floor_bytes", None)
        floor = floor_of(cache.device) if floor_of is not None else ext.KV_CACHE_MARGIN_FLOOR_BYTES
        keep_of = getattr(ext, "kv_cache_keep_bytes", None)   # the fork's: margin on the peak, growth with headroom
        if keep_of is not None:
            reserve_of = getattr(ext, "kv_cache_non_torch_reserve_bytes", None)
            nt = reserve_of(cache.device) if reserve_of is not None else ext.KV_CACHE_NON_TORCH_RESERVE_BYTES
            keep = keep_of(peak, growth, floor, ext.KV_CACHE_MARGIN_FRACTION, nt)
        else:   # stock vLLM: the margin on the larger of the two
            transient = max(peak, growth)
            keep = transient + max(floor, int(transient * ext.KV_CACHE_MARGIN_FRACTION))
        how = f"the extensible KV cache keeps {keep / (1 << 30):.2f} GiB (transient and margin)"
    else:
        util = cpu_offload.gpu_utilization(worker.cache_config)
        keep = int(total * (1 - util)) + _LATE["peak"]
        how = f"{util:g} utilization and the {_LATE['peak'] / (1 << 30):.2f} GiB profiled peak stay free"
    budget = max(free - keep, 0)
    n = int(budget // max(_LATE["slot_bytes"] * _LATE["layers"], 1))
    # slots fill from host rows, and the rows refill from disk (`_backfill`), so a small host tier takes several
    # rounds: QFN 1.66bpw with 4 GiB of rows stopped at their 90 experts a layer with 1.2 GiB of VRAM unused
    added = 0
    while n > 0:
        before = min(len(v) for v in cache.views)
        got = cache.grow(n)
        grown = min(len(v) for v in cache.views) - before
        added += got
        if not got or grown <= 0:
            break
        n -= grown
    E = cache.E
    in_vram = int(numpy.mean([len(v) for v in cache.views]))
    logger.info(
        "expert cache: %d slots per layer over %d layers placed after the KV cache (%.0f MiB), %.0f%% of experts in "
        "VRAM; %s, %.2f GiB free after.", in_vram - (E - max(len(b.rows) + len(b.disk) for b in cache.layers[0].exl3_host_blocks.values())),
        _LATE["layers"], added / (1 << 20), 100 * in_vram / E, how, (free - added) / (1 << 30))
    if hasattr(cache, "release_host_tier"):
        cache.release_host_tier(worker)
    cache.report_tiers()


def build(layers: list, available: int, worker) -> int:
    """Size and allocate the slots; returns the bytes taken from what KV would have had."""
    global CACHE
    from . import cpu_offload

    E = layers[0].exl3_num_experts
    if _late_slots(worker, layers):
        return _build_late(layers, available, worker)
    withheld = _extensible_withholding(worker)
    if withheld:
        available = max(available - withheld, 0)
        logger.info("expert cache: the extensible KV cache will keep %.2f GiB more free than the profile shows "
                    "(reservation growth and its margin); slots size against what is left", withheld / (1 << 30))
    slot_bytes = slot_row_bytes(layers)
    # off the device per layer: host rows, and with the disk tier the experts left in the checkpoint
    per_layer_max = max(max(len(b.rows) + len(b.disk) for b in l.exl3_host_blocks.values()) for l in layers)
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
            in_vram = target_in_vram(E, streams)
            cap = max(in_vram - (E - per_layer_max), 0)
            how += f"; slots stop at {100 * in_vram / E:.0f}% of experts for {streams:g} stream(s)"
        slots = size_slots(available, reserve, hosted, slot_bytes, len(layers), per_layer_max, cap)
        extra = cpu_offload.prefetch_window_bytes()
        if extra:
            # EXL3_STAGE_PREFETCH=auto only (the default keeps the window): the second staging window is
            # dropped only when it is what takes residency below the target
            target = target_in_vram(E, float(os.environ.get("EXL3_KV_RESERVE", "1.0"))) / E
            freed_slots = size_slots(available + extra, reserve, hosted, slot_bytes, len(layers), per_layer_max, cap)
            kept, freed = (E - per_layer_max + slots) / E, (E - per_layer_max + freed_slots) / E
            if kept < target <= freed:
                dropped = cpu_offload.drop_prefetch()
                available += dropped
                given_back += dropped
                slots = freed_slots
                logger.info(
                    "expert cache: prefill prefetch off; its second staging window (%.0f MiB) would have left "
                    "%.0f%% of experts in VRAM, under the %.0f%% target (EXL3_STAGE_PREFETCH=auto; 1 keeps it)",
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
    if cls is not ExclusiveCache and any(b.disk for l in layers for b in l.exl3_host_blocks.values()):
        raise RuntimeError("EXL3_DISK_TIER: experts are on disk but the exclusive cache cannot run (hosting is "
                           "uneven across projections, or the slots would hold every expert); unset it")
    cache = cls(layers, slots, _PROMOTE_PER_STEP)
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
        # and the host rows, which nothing reads now and no graph has captured yet
        host_freed = 0
        for layer in layers:
            for block in layer.exl3_host_blocks.values():
                host = block.host
                block.host = block.view = torch.empty((0, *host.shape[1:]), dtype=host.dtype)
                host_freed += cpu_offload.free_pinned(host)
                del host
        if host_freed:
            logger.info("expert cache: %.2f GiB of host rows returned", host_freed / (1 << 30))
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
    logger.info(
        "expert cache: %d slots per layer over %d layers (%.0f MiB), %.0f%% of experts in VRAM; KV keeps %.2f GiB "
        "(reserve: %s).", slots, len(layers), used / (1 << 20), 100 * in_vram, (available - used) / (1 << 30), how)
    if hasattr(cache, "report_tiers"):
        cache.report_tiers()
    return used - given_back
