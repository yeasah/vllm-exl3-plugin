"""Offload selector grammar and the component policy.

No GPU: `_matches` is pure, and the component policy short-circuits before any
tensor is touched. The actual offload (`_offload_one`) needs CUDA and is not
covered here.
"""

from __future__ import annotations

import json
import os
import re

import pytest

from vllm_exl3_plugin import cpu_offload


def full(prefix: str, name: str) -> str:
    """The string `_offload_one` matches against."""
    return f".{prefix}.{name}."


def expert(layer: int, index: int, proj: str = "gate", comp: str = "trellis") -> str:
    """A routed-expert tensor name as `_offload_moe` reconstructs it."""
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


class _ExplodingLoader:
    """Any attribute access means the policy failed to short-circuit."""
    cpu_offload_params = set()

    def __getattr__(self, name):  # pragma: no cover - only on failure
        raise AssertionError(f"policy let a scale vector reach the loader ({name})")


def test_moe_offload_skips_scale_vectors_without_touching_them():
    import torch

    class Layer:
        pass

    layer = Layer()
    # suh/svh present, trellis absent: nothing is eligible.
    for label in ("gate", "up", "down"):
        for comp in ("suh", "svh"):
            setattr(layer, f"_exl3_{label}_{comp}", [torch.zeros(4)])
            setattr(layer, f"exl3_{label}_{comp}_ptrs", torch.zeros(1))

    assert cpu_offload._offload_moe(layer, "m.layers.0.mlp.experts",
                                    _ExplodingLoader()) == 0
    # and the pointer tables were left alone
    for label in ("gate", "up", "down"):
        for comp in ("suh", "svh"):
            assert getattr(layer, f"exl3_{label}_{comp}_ptrs").numel() == 1


# ------------------------------------------- the grammar against a checkpoint

_CKPT = os.environ.get("EXL3_TEST_CHECKPOINT")


@pytest.mark.skipif(not _CKPT, reason="set EXL3_TEST_CHECKPOINT to a checkpoint dir")
def test_reconstructed_names_exist_in_the_checkpoint():
    """Pin the grammar to an external source, not to itself.

    `_offload_moe` rebuilds `<i>.<proj>_proj.<component>` from tensors that
    `named_parameters` cannot see. That reconstruction is only useful if it
    reproduces what the checkpoint actually calls them.
    """
    index = json.load(open(os.path.join(_CKPT, "model.safetensors.index.json")))
    keys = set(index["weight_map"])
    tails = {k.split(".mlp.experts.", 1)[1] for k in keys if ".mlp.experts." in k}
    assert tails, "no routed-expert tensors in this checkpoint"

    for proj in ("gate", "up", "down"):
        reconstructed = f"7.{proj}_proj.trellis"
        assert reconstructed in tails, (
            f"{reconstructed!r} is not a name this checkpoint uses; "
            f"the reconstruction in _offload_moe has drifted"
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


def test_report_is_emitted_only_once():
    cpu_offload._reset_stats()
    cpu_offload._STATS.update(requested=True, bytes=1 << 30, tensors=7, layers=2)
    cpu_offload.report_once()
    assert cpu_offload._STATS["reported"] is True
    cpu_offload.report_once()  # must not raise or re-emit
    cpu_offload._reset_stats()
