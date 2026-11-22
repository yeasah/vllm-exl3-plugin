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

    def run(tokens, activation="silu", stage=-1):
        x = torch.zeros(tokens, 8)
        ids = torch.zeros(tokens, 2, dtype=torch.int32)
        w = torch.zeros(tokens, 2)
        t = torch.zeros(4, dtype=torch.long)
        ops._exl3_moe_mm(x, ids, w, t, t, t, t, t, t, t, t, t, 2.0, 2.0, 16, 4, False, True, activation,
                         stage=stage)
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


def test_disk_tier_layers_run_grouped_whatever_the_batch(paths, monkeypatch):
    """Experts on the disk tier are read only by the grouped path's routed staging; a decode-sized batch taking
    the per-row path would follow their null entries, and a mid-sized one stage_all's unwritten window."""
    from vllm_exl3_plugin import cpu_offload

    ops, run = paths
    monkeypatch.setattr(cpu_offload, "has_disk", lambda idx: True)
    assert run(1, stage=0) == "grouped"
    assert run(ops._GROUPED_MIN_TOKENS - 1, stage=0) == "grouped"
    staged = []
    monkeypatch.setattr(cpu_offload, "stage_routed", lambda idx, counts, tables: staged.append(counts) or tables)
    assert run(4096, activation="relu2", stage=0) == "per_row"   # no fused kernel: staged by routing all the same
    assert staged and staged[0][0] == 2 * 4096
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="graph capture"):
        run(1, stage=0)


def test_padding_rows_reach_no_path_with_a_negative_expert(paths, monkeypatch):
    """v0.34 routers give padding rows expert id -1 (VLLM_MOE_SKIP_PADDING); every path must see a valid id, and the
    padding rows must contribute nothing."""
    ops, _ = paths
    seen = []
    monkeypatch.setattr(ops, "_moe_grouped", lambda *a, **k: seen.append((a[1].clone(), a[2].clone())) or a[0])
    x = torch.zeros(64, 8)
    ids = torch.randint(0, 4, (64, 2), dtype=torch.int32)
    ids[40:] = -1
    w = torch.rand(64, 2)
    t = torch.zeros(4, dtype=torch.long)
    route = torch.zeros(64, 2, dtype=torch.int32)
    ops._exl3_moe_mm(x, ids, w, t, t, t, t, t, t, t, t, t, 2.0, 2.0, 16, 4, False, True, "silu", route)
    got_ids, got_w = seen[0]
    assert int(got_ids.min()) >= 0 and int(route.min()) >= 0
    assert torch.equal(got_w[40:], torch.zeros_like(got_w[40:])), "padding rows contribute exactly nothing"
    assert torch.equal(got_ids[:40], ids[:40]) and torch.equal(got_w[:40], w[:40]), "real rows are untouched"
