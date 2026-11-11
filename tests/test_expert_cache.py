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
