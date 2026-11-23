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


def test_disk_tier_layers_stage_by_routing_whatever_the_batch(paths, monkeypatch):
    """Experts on the disk tier are read only by routed staging: a decode-sized batch reads its routed disk experts
    first (`stage_disk_decode`) and runs per-row on the tables that returns, never on the raw decode tables, whose
    disk entries are null; a larger one takes the grouped path's routed staging, never stage_all's unwritten window."""
    from vllm_exl3_plugin import cpu_offload

    ops, run = paths
    monkeypatch.setattr(cpu_offload, "has_disk", lambda idx: True)
    decoded = []
    monkeypatch.setattr(cpu_offload, "stage_disk_decode",
                        lambda idx, ids, experts, tables: decoded.append(ids) or tables)
    assert run(1, stage=0) == "per_row" and decoded, "decode reads its disk experts before the per-row kernel"
    assert run(cpu_offload.STAGE_MIN_TOKENS + 1, stage=0) == "grouped"
    monkeypatch.setattr(ops, "_LEAN_DECODE", False)
    assert run(1, stage=0) == "grouped"
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


def test_eager_break_does_nothing_at_capture_time(monkeypatch):
    """Breakable capture runs an eager break once while capturing, after the segments before it were captured but not
    run, so the router's ids are garbage (QFN: an id past every expert, a 16 GiB bincount). The MoE op must not read
    them then; replay calls it again on real routing."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    from vllm_exl3_plugin import ops

    calls = []
    monkeypatch.setattr(ops, "_exl3_moe_mm", lambda *a, **k: calls.append(a) or torch.ones(2, 4))
    out = torch.zeros(2, 4)
    args = [torch.zeros(2, 4), torch.full((2, 2), 1 << 40), torch.ones(2, 2)] + [None] * 9 + [2.0, 2.0, 8, 4, True,
                                                                                            False, "silu"]
    monkeypatch.setattr(BreakableCUDAGraphCapture, "is_active", classmethod(lambda cls: True))
    ops._exl3_moe_mm_out(out, *args)
    assert not calls and not out.any(), "the capture-time call read garbage routing"
    monkeypatch.setattr(BreakableCUDAGraphCapture, "is_active", classmethod(lambda cls: False))
    ops._exl3_moe_mm_out(out, *args)
    assert len(calls) == 1 and out.eq(1).all(), "replay computes"


def test_disk_tier_eager_break_follows_what_spilled(monkeypatch):
    """The MoE op is an eager break exactly when experts were spilled to disk at load; EXL3_MOE_EAGER_BREAK forces it."""
    from vllm_exl3_plugin import cpu_offload, ops

    monkeypatch.setattr(ops, "_EAGER_BREAK_ENV", None)
    monkeypatch.setitem(cpu_offload._STATS, "disk_experts", 0)
    assert not ops.moe_eager_break()
    monkeypatch.setitem(cpu_offload._STATS, "disk_experts", 12)
    assert ops.moe_eager_break()
    monkeypatch.setattr(ops, "_EAGER_BREAK_ENV", "0")
    assert not ops.moe_eager_break()


def test_disk_tier_setting(monkeypatch):
    from vllm_exl3_plugin import cpu_offload

    for spec, want in (("", "auto"), ("auto", "auto"), ("0", None), ("off", None), ("0.25", 0.25)):
        monkeypatch.setenv("EXL3_DISK_TIER", spec)
        assert cpu_offload.disk_tier() == want
    monkeypatch.setenv("EXL3_DISK_TIER", "1.5")
    with pytest.raises(ValueError):
        cpu_offload.disk_tier()


def test_spill_is_predicted_from_expert_bytes_against_host_room(monkeypatch):
    """The graph path is chosen before load, so the spill is predicted: experts beyond 80% of what --cpu-offload-gb
    and host RAM (less the margin) hold."""
    import types

    from vllm_exl3_plugin import cpu_offload

    gib = 1 << 30
    monkeypatch.delenv("EXL3_DISK_TIER", raising=False)
    monkeypatch.setattr(cpu_offload, "_mem_available", lambda: 36 * gib)
    monkeypatch.setattr(cpu_offload, "_mem_total", lambda: 40 * gib)
    experts = {"experts": 20 * gib}
    monkeypatch.setattr(cpu_offload, "_checkpoint_census", lambda *a: experts)

    def config(gb, params=("experts",), tp=1):
        return types.SimpleNamespace(
            offload_config=types.SimpleNamespace(uva=types.SimpleNamespace(cpu_offload_gb=gb, cpu_offload_params=set(params))),
            model_config=types.SimpleNamespace(model="m", revision=None), load_config=types.SimpleNamespace(),
            parallel_config=types.SimpleNamespace(tensor_parallel_size=tp))

    assert not cpu_offload.spill_expected(config(64)), "host RAM holds 20 GiB of experts"
    assert cpu_offload.spill_expected(config(8)), "an 8 GiB budget does not"
    assert not cpu_offload.spill_expected(config(0)), "no offload, nothing to spill"
    assert not cpu_offload.spill_expected(config(8, params=("visual",))), "experts not offloaded"
    experts["experts"] = 40 * gib
    assert cpu_offload.spill_expected(config(64)), "more experts than host RAM less the margin"
    monkeypatch.setenv("EXL3_DISK_TIER", "off")
    assert not cpu_offload.spill_expected(config(8))
