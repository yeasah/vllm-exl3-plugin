"""Offload selector grammar and the component policy.

No GPU: `_matches` is pure, and placement is tested with pinning and the UVA view
stubbed out, so what is covered is the decision -- which tensors go to the host,
under which budget, into which rows -- not the CUDA mechanism.
"""

from __future__ import annotations

import json
import os
import re

import pytest

from vllm_exl3_plugin import cpu_offload


def full(prefix: str, name: str) -> str:
    """The string `place` and `place_expert` match against."""
    return f".{prefix}.{name}."


def expert(layer: int, index: int, proj: str = "gate", comp: str = "trellis") -> str:
    """A routed-expert tensor name as `place_expert` reconstructs it."""
    return full(
        f"language_model.model.layers.{layer}.mlp.experts", f"{index}.{proj}_proj.{comp}"
    )


# --------------------------------------------------------------- the grammar

def test_empty_selector_matches_everything():
    assert cpu_offload._matches(set(), expert(3, 7))
    assert cpu_offload._matches(None, expert(3, 7))


def test_plain_segments_match_whole_components_only():
    assert cpu_offload._matches({"experts"}, expert(3, 7))
    assert cpu_offload._matches({"trellis"}, expert(3, 7))
    # a substring of a segment is not a segment
    assert not cpu_offload._matches({"expert"}, expert(3, 7))
    assert not cpu_offload._matches({"trell"}, expert(3, 7))


def test_plain_segments_are_an_or_not_an_and():
    """Why `re:` exists: vLLM's own matching cannot express a conjunction."""
    dense = full("language_model.model.layers.3.self_attn.qkv_proj", "trellis")
    # "experts or trellis" also catches every dense trellis --
    # "expert trellises only" is unsayable without a regex.
    assert cpu_offload._matches({"experts", "trellis"}, dense)
    assert cpu_offload._matches({"re:experts\\..*\\.trellis"}, expert(3, 7))
    assert not cpu_offload._matches({"re:experts\\..*\\.trellis"}, dense)


@pytest.mark.parametrize(
    "pattern,hits,misses",
    [
        # layer ranges
        (r"re:[1][0-9]\.mlp\.experts\.", [expert(10, 0), expert(19, 255)],
                                        [expert(3, 0), expert(20, 0), expert(9, 0)]),
        # expert index ranges -- the 2026-09-17 sweep used exactly these
        (r"re:experts\.2[0-9][0-9]\.", [expert(0, 200), expert(3, 255)],
                                       [expert(0, 199), expert(0, 20), expert(0, 2)]),
        (r"re:experts\.[1-2][0-9][0-9]\.", [expert(0, 100), expert(0, 255)],
                                           [expert(0, 99), expert(0, 9)]),
        # 0-59, and the alternation must not spill into 100-159
        (r"re:experts\.([0-9]|[1-5][0-9])\.", [expert(0, 0), expert(0, 9),
                                               expert(0, 59)],
                                              [expert(0, 60), expert(0, 159),
                                               expert(0, 100)]),
    ],
)
def test_regex_selectors_from_the_placement_sweep(pattern, hits, misses):
    for name in hits:
        assert cpu_offload._matches({pattern}, name), f"{pattern} should match {name}"
    for name in misses:
        assert not cpu_offload._matches({pattern}, name), \
            f"{pattern} should not match {name}"


def test_unescaped_dot_is_a_wildcard():
    """Documented footgun: patterns are regexes, so `.` matches anything."""
    assert cpu_offload._matches({"re:experts.1."}, expert(0, 1))
    assert cpu_offload._matches({"re:experts.1."}, expert(0, 17))  # surprising
    assert not cpu_offload._matches({r"re:experts\.1\."}, expert(0, 17))


def test_a_typo_matches_nothing():
    """The failure report_once() exists to catch."""
    assert not cpu_offload._matches({"re:expert\\.[0-9]+\\."}, expert(3, 7))
    assert not cpu_offload._matches({"exprts"}, expert(3, 7))


# ------------------------------------------------------- the component policy

def test_scale_vectors_are_never_offloadable():
    """suh/svh cost 12.5% of throughput for 1.1% of the bytes -- see the
    module comment. Policy, not preference, so no selector can opt back in."""
    assert "trellis" in cpu_offload._OFFLOADABLE_COMPONENTS
    assert "weight" in cpu_offload._OFFLOADABLE_COMPONENTS
    assert "suh" not in cpu_offload._OFFLOADABLE_COMPONENTS
    assert "svh" not in cpu_offload._OFFLOADABLE_COMPONENTS


class _Loader:
    """The three attributes placement reads from vLLM's UVAOffloader."""

    def __init__(self, params=(), max_bytes=1 << 40):
        self.cpu_offload_params = set(params)
        self.cpu_offload_bytes = 0
        self.cpu_offload_max_bytes = max_bytes


@pytest.fixture
def offload(monkeypatch):
    """Route placement to a fake loader, with plain CPU tensors standing in for
    pinned memory and their UVA views. Returns a function that installs a loader."""
    import torch

    monkeypatch.setattr(cpu_offload, "_pinned", lambda shape, dtype: torch.empty(shape, dtype=dtype))
    monkeypatch.setattr(cpu_offload, "get_accelerator_view_from_cpu_tensor", lambda t: t)
    # budget-driven placement, the cache-off mode; cache-mode tests set it themselves
    monkeypatch.setenv("EXL3_EXPERT_SLOTS", "0")
    cpu_offload._reset_stats()

    def install(loader):
        monkeypatch.setattr(cpu_offload, "_uva_loader", lambda: loader)
        return loader

    yield install
    cpu_offload._reset_stats()


class _Collector:
    """The attributes `place_expert` reads from an EXL3MoEParameter."""

    def __init__(self, name="trellis", experts=8, prefix="m.layers.3.mlp.experts"):
        import torch

        self.exl3_name = name
        self.exl3_prefix = prefix
        self.exl3_num_experts = experts
        self.exl3_device = torch.device("cpu")
        self.exl3_blocks = {}


def _expert_tensor(e):
    import torch

    return torch.full((4, 2, 32), e, dtype=torch.int16)  # 512 bytes


def _on_host(param, label, t):
    block = param.exl3_blocks.get(label)
    return block is not None and any(t.data_ptr() == block.host[r].data_ptr()
                                     for r in block.rows.values())


def test_host_copies_land_before_their_sources_are_released(offload):
    """Placement copies to host rows asynchronously, and the drain (or the backlog bound) must hold a
    copy's source until it lands. `popleft()[0].result()` dropped it first, so a copy still queued read
    freed memory: ~2% of rows corrupt under this load, starting at the allocator's free-list pointer.
    Fresh sources each time and a drain mid-flight are what expose it."""
    import torch

    offload(_Loader({"experts"}))
    bad = 0
    for trial in range(100):
        cpu_offload._reset_stats()
        params = []
        for n in range(4):
            param = _Collector(prefix=f"m.layers.{n}.mlp.experts")
            for e in range(8):
                cpu_offload.place_expert(param, torch.full((4, 2, 32), e + 10 * n, dtype=torch.int16), "gate", e)
            params.append(param)
        cpu_offload.drain_host_copies()
        for n, param in enumerate(params):
            block = param.exl3_blocks["gate"]
            bad += sum(not bool((block.host[r] == e + 10 * n).all()) for e, r in block.rows.items())
    assert bad == 0, f"{bad} of 3200 host rows were corrupted by copies that outlived their sources"


def test_scale_vectors_never_reach_the_loader(offload):
    import torch

    def explode():  # pragma: no cover - only on failure
        raise AssertionError("policy let a scale vector reach the loader")

    offload(None)
    cpu_offload._uva_loader = explode  # restored by monkeypatch
    for comp in ("suh", "svh"):
        param = _Collector(comp)
        t = torch.ones(16, dtype=torch.half)
        assert torch.equal(cpu_offload.place_expert(param, t, "gate", 0), t)
        assert param.exl3_blocks == {"gate": None}


def test_expert_plan_takes_matching_ids_in_order_under_the_budget(offload):
    # Matches 1, 3, 5, 7; budget 2.5 tensors, checked *before* each, so three fit.
    loader = offload(_Loader({r"re:experts\.[1357]\.gate"}, max_bytes=1280))
    param = _Collector()
    # deliver in reverse: the plan must not depend on arrival order
    placed = {e: cpu_offload.place_expert(param, _expert_tensor(e), "gate", e)
              for e in reversed(range(8))}
    block = param.exl3_blocks["gate"]
    assert block.rows == {1: 0, 3: 1, 5: 2}
    assert [e for e in range(8) if _on_host(param, "gate", placed[e])] == [1, 3, 5]
    for e, t in placed.items():
        assert bool((t == e).all()), f"expert {e} holds the wrong contents"
    assert loader.cpu_offload_bytes == 3 * 512
    assert cpu_offload._STATS["tensors"] == 3
    assert cpu_offload._STATS["layers"] == {"m.layers.3.mlp.experts"}


def test_projections_plan_independently_and_share_the_budget(offload):
    loader = offload(_Loader({"experts"}, max_bytes=8 * 512 + 1))
    param = _Collector()
    for label in ("gate", "up"):
        for e in range(8):
            cpu_offload.place_expert(param, _expert_tensor(e), label, e)
    # gate takes all 8 and leaves 1 byte, which still admits one up expert
    assert len(param.exl3_blocks["gate"].rows) == 8
    assert param.exl3_blocks["up"].rows == {0: 0}
    assert loader.cpu_offload_bytes == 9 * 512


def test_exhausted_budget_keeps_everything_on_the_device(offload):
    offload(_Loader({"experts"}, max_bytes=0))
    param = _Collector()
    t = cpu_offload.place_expert(param, _expert_tensor(2), "down", 2)
    assert param.exl3_blocks == {"down": None}
    assert bool((t == 2).all())
    assert cpu_offload._STATS["tensors"] == 0


def test_mismatched_expert_shape_is_refused(offload):
    import torch

    from vllm_exl3_plugin import format

    offload(_Loader({"experts"}))
    param = _Collector()
    cpu_offload.place_expert(param, _expert_tensor(0), "gate", 0)
    with pytest.raises(format.EXL3FormatError):
        cpu_offload.place_expert(param, torch.zeros((4, 2, 48), dtype=torch.int16), "gate", 1)


def _staged_layer(offload, monkeypatch):
    import types

    import torch

    monkeypatch.setattr(cpu_offload, "STAGED", [])
    offload(_Loader({r"re:experts\.[135]\.gate", "down_proj"}))
    params = {"gate": _Collector(), "down": _Collector()}
    placed = {label: [cpu_offload.place_expert(p, _expert_tensor(e), label, e)
                      for e in range(8)] for label, p in params.items()}
    layer = types.SimpleNamespace(exl3_up_trellis_ptrs=torch.arange(8))
    for label, tensors in placed.items():
        setattr(layer, f"exl3_{label}_trellis_ptrs",
                torch.tensor([t.data_ptr() for t in tensors]))
    cpu_offload.register_experts(layer, {"gate": params["gate"].exl3_blocks["gate"],
                                         "up": None,
                                         "down": params["down"].exl3_blocks["down"]})
    cpu_offload.build_staging()
    window = cpu_offload._STATS["windows"][0]
    assert window.numel() == 3 * 512 + 8 * 512
    window.zero_()
    rows = {label: params[label].exl3_blocks[label].rows for label in params}
    return layer, window, rows


def _check_staged(table, resident, rows, window, label, e):
    import torch

    off = int(table[e]) - window.data_ptr()
    assert 0 <= off and off % 256 == 0 and off + 512 <= window.numel()
    assert bool((window[off:off + 512].view(torch.int16) == e).all()), \
        f"staged {label} {e} points at the wrong rows"


def test_full_staging_addresses_each_offloaded_expert_in_the_window(offload, monkeypatch):
    layer, window, rows = _staged_layer(offload, monkeypatch)
    gate, up, down = cpu_offload.stage_all(layer.exl3_stage_idx)
    assert up is layer.exl3_up_trellis_ptrs, "a projection with no host block changed"
    for label, table in (("gate", gate), ("down", down)):
        resident = getattr(layer, f"exl3_{label}_trellis_ptrs")
        for e in range(8):
            if e not in rows[label]:
                assert table[e] == resident[e], f"resident {label} {e} was redirected"
            else:
                _check_staged(table, resident, rows[label], window, label, e)


def test_routed_staging_copies_only_routed_experts_not_in_vram(offload, monkeypatch):
    """The grouped path's staging: of the hosted experts, only the ones this batch routes to and the
    cache does not already hold in a slot are copied in and redirected; every other entry is left as
    the decode table had it, and nothing else lands in the window."""
    import numpy
    import torch

    from vllm_exl3_plugin import expert_cache

    layer, window, rows = _staged_layer(offload, monkeypatch)
    slotted = numpy.zeros(8, dtype=bool)
    slotted[5] = True                                   # gate 5 sits in a slot
    monkeypatch.setattr(expert_cache, "in_vram", lambda l: slotted)
    counts = [0, 0, 1, 2, 0, 3, 0, 0, 0]                # experts 2, 3 and 5 routed
    decode = (layer.exl3_gate_trellis_ptrs, layer.exl3_up_trellis_ptrs, layer.exl3_down_trellis_ptrs)
    gate, up, down = cpu_offload.stage_routed(layer.exl3_stage_idx, counts, decode)
    assert up is layer.exl3_up_trellis_ptrs
    want = {"gate": {3}, "down": {2, 3}}                # gate hosts 1, 3, 5; down hosts all eight
    for label, table, dec in (("gate", gate, decode[0]), ("down", down, decode[2])):
        for e in range(8):
            if e in want[label]:
                _check_staged(table, dec, rows[label], window, label, e)
            else:
                assert table[e] == dec[e], f"{label} {e} was redirected"
    copied = int((window.view(torch.int16) != 0).sum())
    nonzero_experts = sum(len(v - {0}) for v in want.values())  # expert 0's rows are zeros anyway
    assert copied == nonzero_experts * 256, "something other than the routed, non-VRAM rows was copied"


def _prefetch_layers(offload, monkeypatch, order):
    """Hosted layers registered in `order` (layer numbers), staged with prefetch on: two windows."""
    import types

    import torch

    monkeypatch.setattr(cpu_offload, "STAGED", [])
    monkeypatch.setattr(cpu_offload, "PREFETCH", True)
    monkeypatch.setattr(cpu_offload, "_PENDING_STAGE", [None, None])
    offload(_Loader({"experts"}))
    layers = {}
    for n in order:
        param = _Collector(prefix=f"m.layers.{n}.mlp.experts")
        placed = [cpu_offload.place_expert(param, _expert_tensor(e + 10 * n), "gate", e) for e in range(8)]
        layer = types.SimpleNamespace(prefix=param.exl3_prefix,
                                      exl3_gate_trellis_ptrs=torch.tensor([t.data_ptr() for t in placed]),
                                      exl3_up_trellis_ptrs=torch.arange(8), exl3_down_trellis_ptrs=torch.arange(8))
        cpu_offload.register_experts(layer, {"gate": param.exl3_blocks["gate"], "up": None, "down": None})
        layers[n] = layer
    cpu_offload.build_staging()
    for w in cpu_offload._STATS["windows"]:
        w.zero_()
    return layers


def _decode(layer):
    return (layer.exl3_gate_trellis_ptrs, layer.exl3_up_trellis_ptrs, layer.exl3_down_trellis_ptrs)


def _assert_staged_all(layer, table, n):
    """Every expert of layer `n` points into its window at that layer's own rows (contents e + 10n)."""
    import torch

    window = cpu_offload._STATS["windows"][layer.exl3_stage_window]
    for e in range(8):
        off = int(table[e]) - window.data_ptr()
        assert 0 <= off < window.numel(), f"expert {e} does not point into its layer's window"
        got = window[off:off + 512].view(torch.int16)
        host = layer.exl3_host_blocks["gate"].host[layer.exl3_host_blocks["gate"].rows[e]]
        assert bool((got == e + 10 * n).all()), (f"expert {e} holds the wrong rows: window {got.unique().tolist()[:6]}, "
                                                 f"host row {host.unique().tolist()[:6]}")


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="streams and events need CUDA")
def test_staging_indices_follow_the_forward_and_windows_alternate(offload, monkeypatch):
    """Checkpoint order is whatever the loader met; prefetch needs layer order, and neighbours in
    different windows."""
    layers = _prefetch_layers(offload, monkeypatch, [2, 0, 3, 1])
    assert [layers[n].exl3_stage_idx for n in range(4)] == [0, 1, 2, 3]
    assert [layers[n].exl3_stage_window for n in range(4)] == [0, 1, 0, 1]
    assert len(cpu_offload._STATS["windows"]) == 2


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="streams and events need CUDA")
def test_prefetched_layer_uses_the_copy_made_while_the_last_one_ran(offload, monkeypatch):
    """Layer 0's op starts layer 1's copy; layer 1 then stages nothing itself and every hosted expert,
    routed or not, points at the prefetched rows."""
    import torch

    layers = _prefetch_layers(offload, monkeypatch, [0, 1])
    cpu_offload.prefetch_next(0)
    assert cpu_offload._PENDING_STAGE[1] is not None
    copies = []
    monkeypatch.setattr(cpu_offload, "_copy_rows", lambda layer, moved: copies.append(layer))
    gate, _, _ = cpu_offload.stage_routed(1, [1] + [0] * 8, _decode(layers[1]))
    torch.cuda.synchronize()
    assert copies == [], "a prefetched layer copied its experts again"
    assert cpu_offload._PENDING_STAGE == [None, None]
    _assert_staged_all(layers[1], gate, 1)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="streams and events need CUDA")
def test_out_of_order_staging_drops_the_prefetch_and_stays_correct(offload, monkeypatch):
    """If another layer stages into the window a prefetch is pending in, it waits for that copy, drops
    it and stages its own rows; the prefetched layer then stages for itself. Correct either way, only
    not hidden."""
    import torch

    layers = _prefetch_layers(offload, monkeypatch, [0, 1, 2, 3])
    cpu_offload.prefetch_next(0)                         # layer 1's rows into window 1
    gate3, _, _ = cpu_offload.stage_routed(3, [1] * 9, _decode(layers[3]))   # window 1, out of turn
    assert cpu_offload._STATS.get("prefetch_misordered") == 1
    assert cpu_offload._PENDING_STAGE == [None, None]
    torch.cuda.synchronize()
    _assert_staged_all(layers[3], gate3, 3)
    gate1, _, _ = cpu_offload.stage_routed(1, [1] * 9, _decode(layers[1]))
    torch.cuda.synchronize()
    _assert_staged_all(layers[1], gate1, 1)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="streams and events need CUDA")
def test_dropping_prefetch_folds_every_layer_onto_one_window(offload, monkeypatch):
    """When the expert cache gives the second window's bytes to slots, odd layers' staged tables and
    copy plans move to the first window, and staging still lands each layer's own rows where its tables
    point. Before graph capture, so only these tables ever held the old addresses."""
    import torch

    monkeypatch.setattr(cpu_offload, "PREFETCH_MODE", "auto")
    layers = _prefetch_layers(offload, monkeypatch, [0, 1, 2, 3])
    assert cpu_offload.prefetch_window_bytes() > 0
    freed = cpu_offload.drop_prefetch()
    assert freed > 0 and len(cpu_offload._STATS["windows"]) == 1
    assert not cpu_offload.PREFETCH and cpu_offload.prefetch_window_bytes() == 0
    window = cpu_offload._STATS["windows"][0]
    for n in range(4):
        assert layers[n].exl3_stage_window == 0
        window.zero_()   # each path must fill the window itself: the copy list and the copy plan both moved
        gate, _, _ = cpu_offload.stage_all(layers[n].exl3_stage_idx)
        torch.cuda.synchronize()
        _assert_staged_all(layers[n], gate, n)
        window.zero_()
        gate, _, _ = cpu_offload.stage_routed(layers[n].exl3_stage_idx, [1] * 9, _decode(layers[n]))
        torch.cuda.synchronize()
        _assert_staged_all(layers[n], gate, n)


def test_forced_prefetch_is_not_droppable(offload, monkeypatch):
    monkeypatch.setattr(cpu_offload, "PREFETCH_MODE", "on")
    _prefetch_layers(offload, monkeypatch, [0, 1])
    assert len(cpu_offload._STATS["windows"]) == 2
    assert cpu_offload.prefetch_window_bytes() == 0


def test_cache_mode_hosts_matching_experts_without_spending_the_counter(offload, monkeypatch):
    """With the cache on and a budget that covers them, every matching expert is hosted, and
    the hosted bytes are tracked apart from vLLM's counter (the quota already held them to it)."""
    monkeypatch.setenv("EXL3_EXPERT_SLOTS", "auto")
    monkeypatch.setattr(cpu_offload, "_mem_available", lambda: 1 << 40)
    loader = offload(_Loader({r"re:experts\.[1357]\."}))
    cpu_offload.declare_moe_layer("m.layers.3.mlp.experts")
    param = _Collector()
    for e in range(8):
        cpu_offload.place_expert(param, _expert_tensor(e), "gate", e)
    assert param.exl3_blocks["gate"].rows == {1: 0, 3: 1, 5: 2, 7: 3}
    assert loader.cpu_offload_bytes == 0
    assert cpu_offload._STATS["expert_bytes"] == 4 * 512


class UVAOffloader(_Loader):
    """Named as vLLM's, so the real `_uva_loader` accepts it."""

    uva_offloading = True


def test_drafter_weights_stay_resident(offload, monkeypatch):
    """A speculative drafter (an MTP head) loads into the same process after the target. None of
    it is hosted and its layers are not counted toward the per-layer quota: offloading it once
    gave it its own staging window and handed the expert cache its one layer in place of the
    target's 40."""
    monkeypatch.setenv("EXL3_EXPERT_SLOTS", "auto")
    monkeypatch.setattr(cpu_offload, "_mem_available", lambda: 1 << 40)
    loader = UVAOffloader({"experts"})
    monkeypatch.setattr(cpu_offload, "get_offloader", lambda: loader)
    cpu_offload.declare_moe_layer("m.layers.3.mlp.experts")
    target = _Collector()
    for e in range(8):
        cpu_offload.place_expert(target, _expert_tensor(e), "gate", e)
    with cpu_offload.drafting(True):
        cpu_offload.declare_moe_layer("mtp.layers.0.mlp.experts")
        draft = _Collector(prefix="mtp.layers.0.mlp.experts")
        placed = [cpu_offload.place_expert(draft, _expert_tensor(e), "gate", e) for e in range(8)]
    assert len(target.exl3_blocks["gate"].rows) == 8
    assert draft.exl3_blocks == {"gate": None}
    assert all(bool((t == e).all()) for e, t in enumerate(placed))
    assert cpu_offload._STATS["moe_layers"] == {"m.layers.3.mlp.experts"}
    assert not cpu_offload._DRAFTING, "drafting must end with the drafter's load"


def test_drafter_load_is_marked():
    """The loader hook marks a load whose ModelConfig has runner type "draft", and only that."""
    from types import SimpleNamespace

    from vllm.model_executor.model_loader import base_loader

    from vllm_exl3_plugin import plugin

    saved_pwal = base_loader.process_weights_after_loading
    saved_flag = getattr(base_loader, "_exl3_offload_summary_patched", False)
    saved_load = base_loader.BaseModelLoader.load_model
    seen = []
    base_loader.BaseModelLoader.load_model = lambda self, vc, mc, *a, **k: seen.append(cpu_offload._DRAFTING)
    base_loader._exl3_offload_summary_patched = False
    try:
        plugin._patch_offload_summary()
        for runner in ("generate", "draft"):
            base_loader.BaseModelLoader.load_model(None, None, SimpleNamespace(runner_type=runner))
        assert seen == [False, True]
        assert not cpu_offload._DRAFTING
    finally:
        base_loader.BaseModelLoader.load_model = saved_load
        base_loader.process_weights_after_loading = saved_pwal
        base_loader._exl3_offload_summary_patched = saved_flag


def test_decode_rows_count_speculative_verify_rows():
    """Under speculative decoding a decode schedules its token plus its drafts, and every one of
    them is routed through the target; counting one-token requests only read each such step as
    decode-free, so the cache never learned. A prefill chunk is still not a decode."""
    from types import SimpleNamespace

    from vllm_exl3_plugin import expert_cache

    plain = SimpleNamespace(num_scheduled_tokens={"a": 1, "b": 1, "p": 512},
                            scheduled_spec_decode_tokens={})
    assert expert_cache.decode_rows(plain) == 2
    spec = SimpleNamespace(num_scheduled_tokens={"a": 4, "b": 4, "p": 512},
                           scheduled_spec_decode_tokens={"a": [1, 2, 3], "b": [4, 5, 6]})
    assert expert_cache.decode_rows(spec) == 8


def _two_layer_cache(offload, monkeypatch, loader):
    monkeypatch.setenv("EXL3_EXPERT_SLOTS", "auto")
    monkeypatch.setattr(cpu_offload, "_mem_available", lambda: 1 << 40)
    offload(loader)
    for l in (3, 4):
        cpu_offload.declare_moe_layer(f"m.layers.{l}.mlp.experts")
    param = _Collector()
    for e in range(8):
        cpu_offload.place_expert(param, _expert_tensor(e), "gate", e)
    return param.exl3_blocks["gate"].rows


def test_cache_mode_budget_caps_what_is_hosted(offload, monkeypatch):
    """--cpu-offload-gb is a cap on host memory, as vLLM documents it: 2 layers x 8 experts x 3
    projections x 512 bytes = 24 KiB of experts against a 12 KiB budget hosts half of each layer,
    and the rest stay resident. It used to host everything and treat the flag as a floor on VRAM
    slots, which on WSL pinned past the WDDM budget whatever the flag said."""
    rows = _two_layer_cache(offload, monkeypatch, _Loader({"experts"}, max_bytes=12 << 10))
    assert rows == {0: 0, 1: 1, 2: 2, 3: 3}


def test_cache_mode_budget_rounds_down(offload, monkeypatch):
    """A share that is not a whole expert never hosts more than the budget."""
    rows = _two_layer_cache(offload, monkeypatch, _Loader({"experts"}, max_bytes=(12 << 10) - 1))
    assert rows == {0: 0, 1: 1, 2: 2}


def test_cache_mode_towers_spend_the_budget_first(offload, monkeypatch):
    """vLLM offloads towers before any expert, into the same counter: experts get what is left."""
    loader = _Loader({"experts"}, max_bytes=18 << 10)
    loader.cpu_offload_bytes = 6 << 10
    rows = _two_layer_cache(offload, monkeypatch, loader)
    assert rows == {0: 0, 1: 1, 2: 2, 3: 3}


def test_cache_mode_spreads_what_host_ram_allows(offload, monkeypatch):
    monkeypatch.setenv("EXL3_EXPERT_SLOTS", "auto")
    monkeypatch.setattr(cpu_offload, "_HOST_MARGIN", 0)
    monkeypatch.setattr(cpu_offload, "_mem_total", lambda: 0)
    # 2 layers x 8 experts x 3 projections x 512 bytes = 24 KiB wanted; half fits
    monkeypatch.setattr(cpu_offload, "_mem_available", lambda: 12 << 10)
    offload(_Loader({"experts"}))
    for l in (3, 4):
        cpu_offload.declare_moe_layer(f"m.layers.{l}.mlp.experts")
    param = _Collector()
    for e in range(8):
        cpu_offload.place_expert(param, _expert_tensor(e), "gate", e)
    assert param.exl3_blocks["gate"].rows == {0: 0, 1: 1, 2: 2, 3: 3}


def test_cache_mode_leaves_dense_tensors_alone_without_a_selector(offload, monkeypatch):
    import torch

    class Param:
        exl3_device = torch.device("cpu")
        exl3_name = "trellis"
        exl3_prefix = "m.layers.0.self_attn.o_proj"
        exl3_offloadable = True

    monkeypatch.setenv("EXL3_EXPERT_SLOTS", "auto")
    loader = offload(_Loader())
    cpu_offload.place(Param(), torch.ones((8, 8, 8), dtype=torch.int16))
    assert loader.cpu_offload_bytes == 0
    loader.cpu_offload_params = {"o_proj"}
    cpu_offload.place(Param(), torch.ones((8, 8, 8), dtype=torch.int16))
    assert loader.cpu_offload_bytes == 1024, "an explicit selector still offloads dense"


@pytest.mark.parametrize(
    "available,reserve,hosted,expect",
    [
        (100, 30, 200, 7),   # slots take what the reserve leaves: 70 // (2 * 5)
        (100, 30, 40, 4),    # never more than the hosted experts: 40 // 10
        (100, 150, 200, 0),  # the reserve takes everything
        (50, 0, 200, 5),     # never more than what is available
    ],
)
def test_size_slots(available, reserve, hosted, expect):
    from vllm_exl3_plugin import expert_cache

    got = expert_cache.size_slots(available, reserve, hosted, slot_bytes=5, layers=2,
                                  per_layer_max=100)
    assert got == expect


def test_size_slots_auto_cap_leaves_the_rest_to_kv():
    """--max-model-len auto: slots stop at the target and the remainder is KV's, however much
    VRAM there is; without the cap they would take everything past the reserve."""
    from vllm_exl3_plugin import expert_cache

    args = dict(available=1000, reserve=30, hosted=2000, slot_bytes=5, layers=2, per_layer_max=100)
    assert expert_cache.size_slots(**args) == 97
    assert expert_cache.size_slots(**args, cap=40) == 40
    assert expert_cache.size_slots(**{**args, "available": 300}, cap=40) == 27, "the cap is a ceiling, not a floor"


@pytest.mark.parametrize("streams,share", [(1, 0.4), (0.5, 0.4), (2, 0.5), (8, 0.7), (64, 1.0)])
def test_target_share_follows_the_measured_rule(streams, share):
    from vllm_exl3_plugin import expert_cache

    assert expert_cache.target_share(streams) == pytest.approx(share)


def test_dense_place_respects_opt_in_component_and_budget(offload):
    import torch

    class Param:
        exl3_device = torch.device("cpu")

        def __init__(self, name, offloadable=True):
            self.exl3_name = name
            self.exl3_prefix = "m.layers.0.self_attn.o_proj"
            self.exl3_offloadable = offloadable

    loader = offload(_Loader({"o_proj"}, max_bytes=1000))
    t = torch.ones((8, 8, 8), dtype=torch.int16)  # 1024 bytes

    cpu_offload.place(Param("trellis", offloadable=False), t)
    assert loader.cpu_offload_bytes == 0, "a param that did not opt in was offloaded"
    cpu_offload.place(Param("suh"), torch.ones(8, dtype=torch.half))
    assert loader.cpu_offload_bytes == 0, "a scale vector was offloaded"
    host = cpu_offload.place(Param("trellis"), t)
    cpu_offload.drain_host_copies()  # host copies are asynchronous; the load path drains before reading
    assert loader.cpu_offload_bytes == 1024 and torch.equal(host, t)
    assert host.data_ptr() != t.data_ptr(), "offload must copy out of the checkpoint mmap"
    cpu_offload.place(Param("trellis"), t)
    assert loader.cpu_offload_bytes == 1024, "budget is checked before each tensor"


# ------------------------------------------- the grammar against a checkpoint

_CKPT = os.environ.get("EXL3_TEST_CHECKPOINT")


@pytest.mark.skipif(not _CKPT, reason="set EXL3_TEST_CHECKPOINT to a checkpoint dir")
def test_reconstructed_names_exist_in_the_checkpoint():
    """Pin the grammar to an external source, not to itself.

    `place_expert` rebuilds `<i>.<proj>_proj.<component>` from the loader's
    expert id and shard id, which is not the checkpoint's name. That reconstruction is only useful if it
    reproduces what the checkpoint actually calls them.
    """
    index = json.load(open(os.path.join(_CKPT, "model.safetensors.index.json")))
    keys = set(index["weight_map"])
    # `.experts.` rather than `.mlp.experts.`: gemma-4 keeps its experts on the layer (`layers.N.experts.`)
    tails = {k.split(".experts.", 1)[1] for k in keys if ".experts." in k}
    assert tails, "no routed-expert tensors in this checkpoint"

    for proj in ("gate", "up", "down"):
        reconstructed = f"7.{proj}_proj.trellis"
        assert reconstructed in tails, (
            f"{reconstructed!r} is not a name this checkpoint uses; "
            f"the reconstruction in place_expert has drifted"
        )


# ------------------------------------------------------ the summary call site

def test_apply_does_not_log():
    """`apply()` runs inside vLLM's torch.compile region, where Dynamo refuses
    `logging.Logger` calls and takes the engine down at startup (gb0291). The
    summary must not be reachable from there."""
    import inspect

    from vllm_exl3_plugin.quantization.fused_moe import EXL3MoEMethod
    from vllm_exl3_plugin.quantization.linear import EXL3LinearMethod

    for method in (EXL3LinearMethod, EXL3MoEMethod):
        src = inspect.getsource(method.apply)
        assert "report_once" not in src, (
            f"{method.__name__}.apply calls report_once; Dynamo will refuse it"
        )


def test_summary_fires_at_the_post_load_boundary():
    from vllm.model_executor.model_loader import base_loader

    from vllm_exl3_plugin import plugin

    saved = base_loader.process_weights_after_loading
    saved_flag = getattr(base_loader, "_exl3_offload_summary_patched", False)
    saved_report = cpu_offload.report_once
    saved_load = base_loader.BaseModelLoader.load_model

    inner, reports = [], []
    base_loader.process_weights_after_loading = lambda *a, **k: inner.append(a)
    base_loader._exl3_offload_summary_patched = False
    cpu_offload.report_once = lambda: reports.append(1)
    try:
        plugin._patch_offload_summary()
        base_loader.process_weights_after_loading("model", "cfg", "dev")
        assert inner == [("model", "cfg", "dev")], "the wrapped loader must still run"
        assert reports == [1], "the summary must fire once loading is complete"
    finally:
        cpu_offload.report_once = saved_report
        base_loader.process_weights_after_loading = saved
        base_loader.BaseModelLoader.load_model = saved_load
        base_loader._exl3_offload_summary_patched = saved_flag


def test_patching_is_idempotent():
    from vllm.model_executor.model_loader import base_loader

    from vllm_exl3_plugin import plugin

    plugin._patch_offload_summary()
    first = base_loader.process_weights_after_loading
    plugin._patch_offload_summary()
    assert base_loader.process_weights_after_loading is first


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, msg, *args):
        self.lines.append(("info", msg % args))

    def warning(self, msg, *args):
        self.lines.append(("warning", msg % args))


def test_report_counts_what_vllm_offloaded_first(offload, monkeypatch):
    """A vision tower vLLM offloads at construction shares the budget; the summary must say
    so, both when experts got the rest and when the tower used it all."""
    log = _Log()
    monkeypatch.setattr(cpu_offload, "logger", log)
    loader = offload(_Loader(max_bytes=3 << 30))
    loader.cpu_offload_bytes = 1 << 30  # the tower, before any EXL3 store
    cpu_offload._STATS.update(requested=True, bytes=2 << 30, tensors=4, layers={"a"})
    loader.cpu_offload_bytes += 2 << 30
    cpu_offload.report_once()
    assert log.lines[-1][0] == "info" and "after 1.00 GiB of non-EXL3" in log.lines[-1][1]

    # experts hosted under the cache do not spend the budget, so they must not be subtracted
    cpu_offload._reset_stats()
    loader.cpu_offload_bytes = 1 << 30
    cpu_offload._STATS.update(requested=True, bytes=5 << 30, expert_bytes=5 << 30, tensors=4,
                              layers={"a"})
    cpu_offload.report_once()
    assert "after 1.00 GiB of non-EXL3" in log.lines[-1][1]

    cpu_offload._reset_stats()
    loader.cpu_offload_bytes = 3 << 30  # the tower took everything
    cpu_offload._STATS.update(requested=True, selective=True)
    cpu_offload.report_once()
    assert log.lines[-1][0] == "warning" and "used up by" in log.lines[-1][1]


def test_empty_selector_offloads_towers_but_not_decoder_layers():
    """vLLM hands towers to wrap_modules with their name as prefix and decoder layers with
    none. With no selector only the towers may go; with one, vLLM decides as before."""
    import torch
    from vllm.model_executor.offloader.uva import UVAOffloader

    from vllm_exl3_plugin import plugin

    plugin._patch_offload_defaults()
    seen = []
    saved = UVAOffloader._maybe_offload_to_cpu
    UVAOffloader._maybe_offload_to_cpu = lambda self, m, prefix="": seen.append(prefix) or m
    try:
        off = UVAOffloader.__new__(UVAOffloader)
        off.cpu_offload_bytes = 0
        off.cpu_offload_params = set()
        layer = torch.nn.Linear(2, 2)
        assert off.wrap_modules(iter([layer])) == [layer] and seen == []
        off.wrap_modules(iter([layer]), prefix="vision_tower")
        assert seen == ["vision_tower."]
        off.cpu_offload_params = {"norm"}
        off.wrap_modules(iter([layer]))
        assert seen == ["vision_tower.", ""]
    finally:
        UVAOffloader._maybe_offload_to_cpu = saved


def test_report_is_emitted_only_once():
    cpu_offload._reset_stats()
    cpu_offload._STATS.update(requested=True, bytes=1 << 30, tensors=7, layers={"a", "b"})
    cpu_offload.report_once()
    assert cpu_offload._STATS["reported"] is True
    cpu_offload.report_once()  # must not raise or re-emit
    cpu_offload._reset_stats()


def _vllm_config(monkeypatch, original_max_model_len):
    import types

    import vllm.config

    cfg = types.SimpleNamespace(model_config=types.SimpleNamespace(original_max_model_len=original_max_model_len))
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: cfg)


def test_exclusive_cache_hosts_only_what_it_can_use(offload, monkeypatch):
    """At --max-model-len auto the exclusive cache stops at target_share in VRAM (103 of 256 for one
    stream), so it needs rows for the other 153 plus spare ones to swap through; an offer beyond that buys
    nothing and costs KV through the windows, so the quota stops there."""
    from vllm_exl3_plugin import expert_cache

    monkeypatch.setenv("EXL3_EXPERT_SLOTS", "auto")
    monkeypatch.setattr(expert_cache, "EXCLUSIVE", True)
    monkeypatch.setattr(cpu_offload, "_mem_available", lambda: 1 << 40)
    _vllm_config(monkeypatch, -1)
    loader = offload(_Loader({"experts"}))
    assert cpu_offload._exclusive_need(256, loader) == 256 - 103 + cpu_offload._SPARE_ROWS
    cpu_offload.declare_moe_layer("m.layers.3.mlp.experts")
    assert cpu_offload._host_quota(256, 512) == 256 - 103 + cpu_offload._SPARE_ROWS


@pytest.mark.parametrize("case", ["fixed length", "regex selector", "inclusive"])
def test_exclusive_cap_only_when_the_need_is_known(offload, monkeypatch, case):
    from vllm_exl3_plugin import expert_cache

    monkeypatch.setenv("EXL3_EXPERT_SLOTS", "auto")
    monkeypatch.setattr(expert_cache, "EXCLUSIVE", case != "inclusive")
    _vllm_config(monkeypatch, 32768 if case == "fixed length" else -1)
    loader = offload(_Loader({r"re:experts\..*gate"} if case == "regex selector" else {"experts"}))
    assert cpu_offload._exclusive_need(256, loader) is None
