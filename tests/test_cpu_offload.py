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


def test_staged_tables_address_each_offloaded_expert_in_the_window(offload):
    import types

    import torch

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
    window = cpu_offload._STATS["window"]
    assert window.numel() == 3 * 512 + 8 * 512
    window.zero_()

    small = cpu_offload.expert_tables(layer, cpu_offload.STAGE_MIN_TOKENS)
    assert all(a is b for a, b in zip(
        small, (layer.exl3_gate_trellis_ptrs, layer.exl3_up_trellis_ptrs,
                layer.exl3_down_trellis_ptrs))), "small batches must read in place"
    assert not window.any(), "a small batch staged"

    gate, up, down = cpu_offload.expert_tables(layer, cpu_offload.STAGE_MIN_TOKENS + 1)
    assert up is layer.exl3_up_trellis_ptrs, "a projection with no host block changed"
    for label, table in (("gate", gate), ("down", down)):
        resident = getattr(layer, f"exl3_{label}_trellis_ptrs")
        rows = params[label].exl3_blocks[label].rows
        for e in range(8):
            off = int(table[e]) - window.data_ptr()
            if e not in rows:
                assert table[e] == resident[e], f"resident {label} {e} was redirected"
                continue
            assert 0 <= off and off % 256 == 0 and off + 512 <= window.numel()
            assert bool((window[off:off + 512].view(torch.int16) == e).all()), \
                f"staged {label} {e} points at the wrong rows"


def test_cache_mode_hosts_every_matching_expert_outside_the_budget(offload, monkeypatch):
    """With the cache on, slots give VRAM back, so the budget caps what is freed rather than
    what is hosted: every matching expert goes to the host and none of it spends the budget."""
    monkeypatch.setenv("EXL3_EXPERT_SLOTS", "auto")
    monkeypatch.setattr(cpu_offload, "_mem_available", lambda: 1 << 40)
    loader = offload(_Loader({r"re:experts\.[1357]\."}, max_bytes=512))
    cpu_offload.declare_moe_layer("m.layers.3.mlp.experts")
    param = _Collector()
    for e in range(8):
        cpu_offload.place_expert(param, _expert_tensor(e), "gate", e)
    assert param.exl3_blocks["gate"].rows == {1: 0, 3: 1, 5: 2, 7: 3}
    assert loader.cpu_offload_bytes == 0
    assert cpu_offload._STATS["expert_bytes"] == 4 * 512


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
    "available,reserve,hosted,may_free,expect",
    [
        (100, 30, 200, 1000, 7),  # slots take what the reserve leaves: 70 // (2 * 5)
        (100, 30, 40, 1000, 4),   # never more than the hosted experts: 40 // 10
        (100, 30, 200, 120, 8),   # the budget frees at most 120, so slots >= 80 even past the reserve
        (100, 150, 200, 1000, 0),  # the reserve takes everything
        (50, 0, 200, 0, 5),       # a budget floor beyond what is available is clipped to it
    ],
)
def test_size_slots(available, reserve, hosted, may_free, expect):
    from vllm_exl3_plugin import expert_cache

    got = expert_cache.size_slots(available, reserve, hosted, may_free, slot_bytes=5, layers=2,
                                  per_layer_max=100)
    assert got == expect


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
