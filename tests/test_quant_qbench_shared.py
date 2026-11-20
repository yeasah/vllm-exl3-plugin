"""quant.qbench_parallel with a card listed twice: two scorers share it, each with a fixed share of the row-state
budget a lone scorer would take (half the card's free VRAM), so the two cannot both claim the same half; GGUF and the
reference pass are unchanged (2026-11-20)."""

import json
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "quantization"))

import quant  # noqa: E402


def test_two_scorers_per_card(tmp_path, monkeypatch):
    calls = []
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
