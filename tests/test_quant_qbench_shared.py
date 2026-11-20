"""quant.qbench_parallel with a card listed twice: two scorers share it, each with a fixed share of the row-state
budget a lone scorer would take (half the card's free VRAM), so the two cannot both claim the same half; GGUF and the
reference pass are unchanged (2026-11-20)."""

import json
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "quantization"))

import quant  # noqa: E402


def _harness(tmp_path, monkeypatch, gguf_size=0.0, fail_on_one_card=False):
    calls = []
    def run(cmd, log, env=None, echo=True):
        yml = cmd[2]
        labels = json.loads(open(yml).read())["arms"]
        calls.append((os.path.basename(yml), env.get("CUDA_VISIBLE_DEVICES"), env.get("QBENCH_STATE_BUDGET_GB"), labels))
        if fail_on_one_card and labels == ["gguf"] and "," not in env.get("CUDA_VISIBLE_DEVICES"):
            return 1
        rows = [{"label": "HF BF16", "group": "reference"}] + [{"label": l, "kld": 1.0} for l in labels]
        json.dump(rows, open(yml.replace("qbench-", "qb_").replace(".yaml", ".json"), "w"))
        return 0
    monkeypatch.setattr(quant, "run_logged", run)
    monkeypatch.setattr(quant, "qbench_project", lambda job, args, sl, arms, results=None: {"arms": [l for l, _ in arms]})
    monkeypatch.setattr(quant.yaml, "safe_dump", lambda obj, f, sort_keys=False: f.write(json.dumps(obj)))
    monkeypatch.setattr(quant, "gpu_free_gib", lambda: {0: 20.0, 1: 16.0})
    monkeypatch.setattr(quant, "gguf_gib", lambda path: gguf_size)
    monkeypatch.delenv("QBENCH_STATE_BUDGET_GB", raising=False)
    job = types.SimpleNamespace(main=str(tmp_path), log=lambda n: str(tmp_path / "logs" / f"{n}.log"))
    return calls, job


def test_card_sized_gguf_runs_beside_the_rest(tmp_path, monkeypatch):
    arms = [(f"q{i}", f"/m/q{i}") for i in range(2)] + [("gguf", "/m/x.gguf")]
    calls, job = _harness(tmp_path, monkeypatch, gguf_size=3.5)      # fits half of 16 GiB / 2 scorers
    assert quant.qbench_parallel(job, None, "wild", arms, ["0", "0", "1", "1"], "qbench.py")
    g = [c for c in calls if c[3] == ["gguf"]]
    assert len(g) == 1 and g[0][1] in ("0", "1")                    # one card
    assert [r["label"] for r in json.load(open(tmp_path / "qb_wild.json"))] == ["HF BF16", "q0", "q1", "gguf"]


def test_big_gguf_runs_on_every_card(tmp_path, monkeypatch):
    calls, job = _harness(tmp_path, monkeypatch, gguf_size=16.0)
    arms = [("q0", "/m/q0"), ("gguf", "/m/x.gguf")]
    assert quant.qbench_parallel(job, None, "wild", arms, ["0", "1"], "qbench.py")
    assert [c[1] for c in calls if c[3] == ["gguf"]] == ["0,1"]


def test_a_gguf_failing_on_one_card_is_retried_on_all(tmp_path, monkeypatch):
    calls, job = _harness(tmp_path, monkeypatch, gguf_size=3.0, fail_on_one_card=True)
    arms = [("q0", "/m/q0"), ("gguf", "/m/x.gguf")]
    assert quant.qbench_parallel(job, None, "wild", arms, ["0", "1"], "qbench.py")
    g = [c[1] for c in calls if c[3] == ["gguf"]]
    assert len(g) == 2 and "," not in g[0] and g[1] == "0,1"
    assert [r["label"] for r in json.load(open(tmp_path / "qb_wild.json"))] == ["HF BF16", "q0", "gguf"]


def test_two_scorers_per_card(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(quant, "gguf_gib", lambda path: 100.0)      # the GGUF arm here is a big one: all cards
    def run(cmd, log, env=None, echo=True):
        yml = cmd[2]
        res = yml.replace("qbench-", "qb_").replace(".yaml", ".json")
        labels = json.loads(open(yml).read())["arms"]
        calls.append((os.path.basename(yml), env.get("CUDA_VISIBLE_DEVICES"), env.get("QBENCH_STATE_BUDGET_GB"), labels))
        rows = [{"label": "HF BF16", "group": "reference"}] + [{"label": l, "kld": 1.0} for l in labels]
        json.dump(rows, open(res, "w"))
        return 0
    monkeypatch.setattr(quant, "run_logged", run)
    monkeypatch.setattr(quant, "qbench_project", lambda job, args, sl, arms, results=None: {"arms": [l for l, _ in arms]})
    monkeypatch.setattr(quant.yaml, "safe_dump", lambda obj, f, sort_keys=False: f.write(json.dumps(obj)))
    monkeypatch.setattr(quant, "gpu_free_gib", lambda: {0: 20.0, 1: 16.0})
    monkeypatch.delenv("QBENCH_STATE_BUDGET_GB", raising=False)
    job = types.SimpleNamespace(main=str(tmp_path), log=lambda n: str(tmp_path / "logs" / f"{n}.log"))
    arms = [(f"q{i}", f"/m/q{i}") for i in range(6)] + [("gguf", "/m/x.gguf")]
    assert quant.qbench_parallel(job, None, "wild", arms, ["0", "0", "1", "1"], "qbench.py")
    ref, *parts, gguf = calls
    assert ref[0] == "qbench-wild-ref.yaml" and ref[1] == "0" and ref[2] is None          # alone, unbudgeted
    assert sorted((p[1], p[2]) for p in parts) == [("0", "5.00"), ("0", "5.00"), ("1", "4.00"), ("1", "4.00")]
    assert sorted(l for p in parts for l in p[3]) == [f"q{i}" for i in range(6)]
    assert gguf[1] == "0,1" and gguf[3] == ["gguf"]                                       # each card once
    merged = [r["label"] for r in json.load(open(tmp_path / "qb_wild.json"))]
    assert merged == ["HF BF16"] + [l for l, _ in arms]


def test_one_scorer_per_card_is_unbudgeted(tmp_path, monkeypatch):
    seen = []
    def run(cmd, log, env=None, echo=True):
        seen.append(env.get("QBENCH_STATE_BUDGET_GB"))
        yml = cmd[2]
        json.dump([{"label": "HF BF16", "group": "reference"}] + [{"label": l} for l in json.loads(open(yml).read())["arms"]],
                  open(yml.replace("qbench-", "qb_").replace(".yaml", ".json"), "w"))
        return 0
    monkeypatch.setattr(quant, "run_logged", run)
    monkeypatch.setattr(quant, "qbench_project", lambda job, args, sl, arms, results=None: {"arms": [l for l, _ in arms]})
    monkeypatch.setattr(quant.yaml, "safe_dump", lambda obj, f, sort_keys=False: f.write(json.dumps(obj)))
    monkeypatch.setattr(quant, "gpu_free_gib", lambda: {0: 20.0, 1: 16.0})
    job = types.SimpleNamespace(main=str(tmp_path), log=lambda n: str(tmp_path / "logs" / f"{n}.log"))
    assert quant.qbench_parallel(job, None, "wild", [(f"q{i}", f"/m/q{i}") for i in range(4)], ["0", "1"], "qbench.py")
    assert seen == [None, None, None]


def test_gguf_size_counts_every_shard(tmp_path):
    for i in (1, 2, 3):
        (tmp_path / f"m-Q4_K_M-{i:05d}-of-00003.gguf").write_bytes(b"x" * 1024)
    (tmp_path / "single.gguf").write_bytes(b"x" * 2048)
    assert quant.gguf_gib(str(tmp_path / "m-Q4_K_M-00001-of-00003.gguf")) * 2**30 == 3 * 1024
    assert quant.gguf_gib(str(tmp_path / "single.gguf")) * 2**30 == 2048
