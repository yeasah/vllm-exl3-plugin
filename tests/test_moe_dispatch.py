"""Which MoE path a batch takes: grouped by expert (`ops._moe_grouped`) for prefill-sized batches run
eagerly, one row per assignment (`ops._moe_per_row`) for small batches, activations the fused kernel
lacks, and anything captured into a CUDA graph (the grouped path reads per-expert counts back to the
host, which capture cannot do). The grouped path is 4-11x faster per layer at 256-2048 tokens."""
import pytest
import torch


@pytest.fixture
def paths(monkeypatch):
    from vllm_exl3_plugin import ops

    taken = []
    monkeypatch.setattr(ops, "_moe_grouped", lambda *a, **k: taken.append("grouped") or a[0])
    monkeypatch.setattr(ops, "_moe_per_row", lambda *a, **k: taken.append("per_row") or a[0])
    monkeypatch.setattr(ops, "_one_stream", lambda x: None)

    def run(tokens, activation="silu"):
        x = torch.zeros(tokens, 8)
        ids = torch.zeros(tokens, 2, dtype=torch.int32)
        w = torch.zeros(tokens, 2)
        t = torch.zeros(4, dtype=torch.long)
        ops._exl3_moe_mm(x, ids, w, t, t, t, t, t, t, t, t, t, 2.0, 2.0, 16, 4, False, True, activation)
        return taken.pop()

    return ops, run


def test_prefill_sized_batches_run_grouped(paths):
    ops, run = paths
    assert run(ops._GROUPED_MIN_TOKENS) == "grouped"
    assert run(4096) == "grouped"


def test_small_batches_run_per_row(paths):
    ops, run = paths
    assert run(1) == "per_row"
    assert run(ops._GROUPED_MIN_TOKENS - 1) == "per_row"


def test_activation_without_a_fused_kernel_runs_per_row(paths):
    _, run = paths
    assert run(4096, activation="relu2") == "per_row"


def test_graph_capture_runs_per_row(paths, monkeypatch):
    _, run = paths
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert run(4096) == "per_row"
