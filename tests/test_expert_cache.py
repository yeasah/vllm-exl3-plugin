"""The exclusive expert cache, driven directly: fake MoE layers with real pinned host rows and device tensors,
random routing fed into its ring, and its own detector (EXL3_CACHE_VERIFY) checking after every move that each
expert's decode and staged entries point at its current home and that the home holds its bytes."""

import types

import numpy
import pytest
import torch

from vllm_exl3_plugin import cpu_offload, expert_cache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

E, HOSTED, LABELS = 16, 12, ("gate", "up", "down")


def _layer(n: int, staged: bool):
    """Experts 0..HOSTED-1 on host rows 0..HOSTED-1, the rest resident; contents unique per (layer, label, expert)."""
    g = torch.Generator().manual_seed(1000 + n)
    layer = types.SimpleNamespace(exl3_num_experts=E, top_k=2, prefix=f"m.layers.{n}.mlp.experts")
    blocks, plan, stage_tables = {}, {}, []
    window = torch.zeros(3 * HOSTED * 8 * 64 * 2, dtype=torch.uint8, device="cuda") if staged else None
    for j, label in enumerate(LABELS):
        data = [torch.randint(-30000, 30000, (8, 64), generator=g, dtype=torch.int16) for _ in range(E)]
        block = cpu_offload.HostBlock({e: e for e in range(HOSTED)}, (8, 64), torch.int16)
        for e in range(HOSTED):
            block.host[e].copy_(data[e])
        tensors = [block.view[e] for e in range(HOSTED)] + [data[e].cuda() for e in range(HOSTED, E)]
        setattr(layer, f"_exl3_{label}_trellis", tensors)
        setattr(layer, f"exl3_{label}_trellis_ptrs", torch.tensor([t.data_ptr() for t in tensors], device="cuda"))
        blocks[label] = block
        if staged:
            row_bytes = 8 * 64 * 2
            dst = window[j * HOSTED * row_bytes:(j + 1) * HOSTED * row_bytes]
            plan[label] = (dst, row_bytes)
            t = torch.tensor([t.data_ptr() for t in tensors])
            for e in range(HOSTED):
                t[e] = dst.data_ptr() + e * row_bytes
            stage_tables.append(t.cuda())
    layer.exl3_host_blocks = blocks
    layer.exl3_stage = ([], tuple(stage_tables)) if staged else None
    layer.exl3_stage_plan = plan
    layer._window = window
    return layer


def _cache(monkeypatch, staged=True, slots=3, layers=3, budget=8):
    monkeypatch.setattr(expert_cache, "VERIFY", True)
    return expert_cache.ExclusiveCache([_layer(n, staged) for n in range(layers)], slots, budget)


def _drive(cache, steps: int, seed: int = 0, busy=None):
    rng = numpy.random.default_rng(seed)
    hot = rng.permutation(E)
    for step in range(steps):
        cache.step_begin()
        if busy is not None:
            busy()   # keep the forward stream occupied, so the side-stream copies really are asynchronous
        rows = int(rng.integers(1, 5))
        # skewed routing that drifts, so slots turn over
        p = numpy.exp(-numpy.arange(E) / 3.0)
        ids = numpy.stack([rng.choice(numpy.roll(hot, step // 40), size=2, replace=False, p=p / p.sum())
                           for _ in range(rows)])
        r = torch.as_tensor(ids, dtype=torch.int32, device="cuda")
        for l in range(len(cache.layers)):
            cache.route[l, :rows].copy_(r)
        cache.step_end(rows, rows)
    torch.cuda.synchronize()
    for _ in range(4):   # let the last swaps land
        cache.step_begin()
        torch.cuda.synchronize()


def test_swaps_keep_every_expert_on_its_own_bytes(monkeypatch):
    cache = _cache(monkeypatch)
    big = torch.randn(2048, 2048, device="cuda", dtype=torch.half)
    _drive(cache, 300, busy=lambda: [big @ big for _ in range(3)])
    assert cache.promoted > 50, "routing should have forced swaps"
    cache._verify(range(len(cache.layers)), full=True)
    for l in range(len(cache.layers)):
        on_host = int((cache.row_of[l] >= 0).sum())
        in_vram = int((cache.slot_of[l] >= 0).sum())
        assert on_host + in_vram == E and on_host == E - len(cache.views[l]), "every expert exactly one home"
        assert len(cache.free_rows[l]) == HOSTED - on_host, "free rows are the hosted rows nobody holds"


def test_residents_are_slots_and_filled_rows_are_free(monkeypatch):
    cache = _cache(monkeypatch, slots=3)
    for l, layer in enumerate(cache.layers):
        assert len(cache.views[l]) == (E - HOSTED) + 3
        assert sorted(cache.free_rows[l]) == [0, 1, 2], "the first three hosted experts moved into the new slots"
        assert layer.exl3_row_of is cache.row_of[l] or numpy.shares_memory(layer.exl3_row_of, cache.row_of)


def test_staging_reads_moved_rows(monkeypatch):
    """Prefill staging follows the swaps: what it would copy is each on-host expert's current row."""
    cache = _cache(monkeypatch)
    _drive(cache, 120, seed=3)
    for l, layer in enumerate(cache.layers):
        moved = cpu_offload._wanted_rows(layer, lambda e: True)
        want = sorted((int(cache.row_of[l, e]), e) for e in range(E) if cache.row_of[l, e] >= 0)
        assert moved["gate"] == want


def test_detector_fires_on_a_wrong_row(monkeypatch):
    """Put a real bug back: a write-back that lands an expert's entries on the wrong row."""
    cache = _cache(monkeypatch)
    real = expert_cache.ExclusiveCache._to_host

    def wrong_row(self, l, e, row):
        free = self.free_rows[l]
        return real(self, l, e, free[0] if free else (row + 1) % HOSTED)

    monkeypatch.setattr(expert_cache.ExclusiveCache, "_to_host", wrong_row)
    with pytest.raises(RuntimeError, match="expert cache"):
        _drive(cache, 200)


def test_detector_fires_on_an_unordered_copy(monkeypatch):
    """And the ordering bug: copying the hot expert into the slot before the cold one has been written out of it."""
    cache = _cache(monkeypatch)
    real_write_back = expert_cache.ExclusiveCache._write_back

    def eager(self, issued):
        with torch.cuda.stream(self.side):
            for l, x, y, s, f in issued:
                for j, label in enumerate(LABELS):
                    self.views[l][s][j].copy_(self.layers[l].exl3_host_blocks[label].host[self.row_of[l, x]],
                                              non_blocking=True)
        real_write_back(self, issued)

    monkeypatch.setattr(expert_cache.ExclusiveCache, "_write_back", eager)
    with pytest.raises(RuntimeError, match="expert cache"):
        _drive(cache, 200)


ROWS, DISK = 8, (8, 9, 10, 11)   # disk-tier layers: experts 0-7 in rows, 8-11 in the checkpoint, 12-15 resident


def _disk_layers(monkeypatch, tmp_path, layers=3):
    """Layers as `build_staging` leaves them with EXL3_DISK_TIER: disk experts have null decode entries and staged
    entries at window positions past the rows, and a real checkpoint behind them for admissions to read."""
    from concurrent.futures import ThreadPoolExecutor

    from safetensors.torch import save_file

    from vllm_exl3_plugin.expert_store import ExpertStore

    out, tensors = [], {}
    row_bytes = 8 * 64 * 2
    for n in range(layers):
        g = torch.Generator().manual_seed(2000 + n)
        layer = types.SimpleNamespace(exl3_num_experts=E, top_k=2, prefix=f"m.layers.{n}.mlp.experts")
        window = torch.zeros(3 * (ROWS + len(DISK)) * row_bytes, dtype=torch.uint8, device="cuda")
        blocks, plan, stage_tables = {}, {}, []
        layer.exl3_disk_pos = {e: ROWS + k for k, e in enumerate(DISK)}
        layer.exl3_store_key = f"model.layers.{n}.mlp.experts"
        for j, label in enumerate(LABELS):
            data = [torch.randint(-30000, 30000, (8, 64), generator=g, dtype=torch.int16) for _ in range(E)]
            for e in range(E):
                tensors[f"model.layers.{n}.mlp.experts.{e}.{label}_proj.trellis"] = data[e]
            block = cpu_offload.HostBlock({e: e for e in range(ROWS)}, (8, 64), torch.int16, frozenset(DISK))
            for e in range(ROWS):
                block.host[e].copy_(data[e])
            decode = ([block.view[e].data_ptr() for e in range(ROWS)] + [0] * len(DISK)
                      + [data[e].cuda() for e in range(ROWS + len(DISK), E)])
            keep = [t for t in decode if isinstance(t, torch.Tensor)]
            ptrs = [t.data_ptr() if isinstance(t, torch.Tensor) else t for t in decode]
            setattr(layer, f"_exl3_{label}_trellis",
                    [block.view[e] for e in range(ROWS)] + [torch.empty((8, 64), dtype=torch.int16, device="meta")
                                                            for _ in DISK] + keep)
            setattr(layer, f"exl3_{label}_trellis_ptrs", torch.tensor(ptrs, device="cuda"))
            dst = window[j * (ROWS + len(DISK)) * row_bytes:(j + 1) * (ROWS + len(DISK)) * row_bytes]
            plan[label] = (dst, row_bytes)
            t = torch.tensor(ptrs)
            for e in range(ROWS):
                t[e] = dst.data_ptr() + e * row_bytes
            for e, pos in layer.exl3_disk_pos.items():
                t[e] = dst.data_ptr() + pos * row_bytes
            stage_tables.append(t.cuda())
            blocks[label] = block
        layer.exl3_host_blocks = blocks
        layer.exl3_stage = ([], tuple(stage_tables))
        layer.exl3_stage_plan = plan
        layer._window = window
        out.append(layer)
    for label in LABELS:   # the checkpoint also carries scales between trellises, as real ones do
        for n in range(layers):
            for e in range(E):
                tensors[f"model.layers.{n}.mlp.experts.{e}.{label}_proj.suh"] = torch.ones(16, dtype=torch.half)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    monkeypatch.setattr(cpu_offload, "_DISK", {"store": ExpertStore(str(tmp_path)), "done": None,
                                               "pool": ThreadPoolExecutor(4)})
    return out


def _disk_cache(monkeypatch, tmp_path, slots=3):
    monkeypatch.setattr(expert_cache, "VERIFY", True)
    return expert_cache.ExclusiveCache(_disk_layers(monkeypatch, tmp_path), slots, 8)


def test_disk_admissions_keep_every_expert_on_its_own_bytes(monkeypatch, tmp_path):
    """Decode's routed disk experts move into host rows, evicting the coldest row expert to disk; the detector,
    whose references for disk experts come from the checkpoint, checks each filled row and every entry."""
    cache = _disk_cache(monkeypatch, tmp_path)
    big = torch.randn(2048, 2048, device="cuda", dtype=torch.half)
    _drive(cache, 300, busy=lambda: [big @ big for _ in range(3)])
    assert cache.admitted > 20, "routing should have pulled disk experts into rows"
    assert cache.promoted > 20
    cache._verify(range(len(cache.layers)), full=True)
    for l, layer in enumerate(cache.layers):
        on_disk = set(numpy.flatnonzero(cache.row_of[l] == expert_cache.DISK).tolist())
        assert len(on_disk) == len(DISK), "admission swaps one expert in for one out"
        assert set(layer.exl3_disk_pos) == on_disk
        assert sorted(layer.exl3_disk_pos.values()) == [ROWS + k for k in range(len(DISK))], \
            "once admissions land, disk experts hold the window positions past the rows"
        on_host = int((cache.row_of[l] >= 0).sum())
        in_vram = int((cache.slot_of[l] >= 0).sum())
        assert on_host + in_vram + len(on_disk) == E
        assert len(cache.free_rows[l]) == ROWS - on_host


def test_detector_fires_on_an_admission_that_reads_the_wrong_expert(monkeypatch, tmp_path):
    """The failure put back: an admitted row filled with another expert's bytes."""
    cache = _disk_cache(monkeypatch, tmp_path)
    store = cpu_offload._DISK["store"]
    real = store.read
    monkeypatch.setattr(store, "read", lambda key, e, out=None, slices=None: real(key, e ^ 1, out, slices) if out is not None
                        and out["gate"].is_pinned() else real(key, e, out))
    with pytest.raises(RuntimeError, match="wrong bytes"):
        _drive(cache, 200)


def test_late_slots_grow_a_zero_slot_cache_and_keep_every_expert_on_its_bytes(monkeypatch):
    """Late slots: a cache built with no slots (residents only) grows after the KV cache is sized. New slots take the
    lowest-id experts still in host rows, their rows become free for write-backs, and swaps then run through them."""
    cache = _cache(monkeypatch, slots=0)
    assert [len(v) for v in cache.views] == [E - HOSTED] * 3
    added = cache.grow(3)
    assert added == 3 * len(cache.layers) * 3 * 8 * 64 * 2   # three slots a layer, three projections each
    for l in range(len(cache.layers)):
        assert len(cache.views[l]) == (E - HOSTED) + 3 and cache.capacity[l] == len(cache.views[l])
        assert sorted(cache.free_rows[l]) == [0, 1, 2]
        assert all(cache.slot_of[l, e] >= 0 and cache.row_of[l, e] == -1 for e in (0, 1, 2))
    cache._verify(range(len(cache.layers)), full=True)
    big = torch.randn(2048, 2048, device="cuda", dtype=torch.half)
    _drive(cache, 300, busy=lambda: [big @ big for _ in range(3)])
    assert cache.promoted > 20, "the grown slots should take part in swaps"
    cache._verify(range(len(cache.layers)), full=True)


def test_late_slots_stop_at_the_experts_left_in_rows(monkeypatch):
    cache = _cache(monkeypatch, slots=0)
    cache.grow(100)
    assert all(len(cache.views[l]) == E for l in range(len(cache.layers))), "one slot an expert at most"
    assert cache.grow(5) == 0


def test_detector_fires_on_a_late_slot_filled_from_the_wrong_row(monkeypatch):
    """The failure put back: a grown slot copied from a neighbour's row."""
    cache = _cache(monkeypatch, slots=0)
    real = torch.Tensor.copy_
    calls = {"n": 0}

    def shifted(self, src, non_blocking=False):
        calls["n"] += 1
        if calls["n"] == 2 and src.dim() == 2:          # the second row copy of the growth reads a neighbour
            src = src.roll(1, dims=0)
        return real(self, src, non_blocking=non_blocking)

    monkeypatch.setattr(torch.Tensor, "copy_", shifted)
    with pytest.raises(RuntimeError, match="wrong bytes"):
        cache.grow(2)


class _GrowRecorder:
    """Stands in for an ExclusiveCache in grow_late: records how many slots it was asked for."""

    def __init__(self):
        self.asked = []
        self.E = 256
        self.views = [[0] * 100]
        block = types.SimpleNamespace(rows=[0] * 236, disk=[])
        self.layers = [types.SimpleNamespace(exl3_host_blocks={"w": block})]
        self.device = "cuda:0"

    def grow(self, n):
        self.asked.append(n)
        return 0

    def report_tiers(self):
        pass


def _grow_late_slots(monkeypatch, requested, init_free, free=5 << 30):
    from vllm_exl3_plugin import expert_cache as ec

    from vllm.v1.worker import extensible_kv_cache as ext

    gib = 1 << 30
    cache = _GrowRecorder()
    monkeypatch.setattr(ext, "kv_cache_margin_floor_bytes", lambda device: 3 * gib // 10)
    monkeypatch.setattr(ext, "kv_cache_non_torch_reserve_bytes", lambda device: gib // 8)
    monkeypatch.setattr(ec.torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(ec.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(ec.torch.cuda, "mem_get_info", lambda *a: (free, 12 * gib))
    ec._LATE.update(cache=cache, slot_bytes=gib // 100, layers=1, peak=0)
    kv = types.SimpleNamespace(reserved_headroom_bytes=gib // 4, torch_reserve_growth_bytes=gib // 4)
    worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(extensible_kv_cache=kv),
                                   requested_memory=requested, init_snapshot=types.SimpleNamespace(free_memory=init_free))
    ec.grow_late(worker, extensible=True)
    return cache.asked[0] if cache.asked else 0


def test_late_slots_stay_inside_an_explicit_utilization(monkeypatch):
    gib = 1 << 30
    # 10 GiB requested, 6 consumed since the snapshot (11 free then, 5 now), 0.25 transient: 3.75 GiB for slots,
    # where free less the extensible keep would allow more
    assert _grow_late_slots(monkeypatch, requested=10 * gib, init_free=11 * gib) == 375


def test_late_slots_unbounded_utilization_keeps_the_device_bound(monkeypatch):
    gib = 1 << 30
    from vllm.v1.worker import extensible_kv_cache as ext

    # unset utilization resolves to the whole card: the requested bound (5.75 GiB) is looser than free less the keep
    keep = ext.kv_cache_keep_bytes(gib // 4, gib // 4, 3 * gib // 10, ext.KV_CACHE_MARGIN_FRACTION, gib // 8)
    assert _grow_late_slots(monkeypatch, requested=12 * gib, init_free=11 * gib) == ((5 << 30) - keep) // (gib // 100)
