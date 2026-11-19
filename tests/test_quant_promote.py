"""quant.py promote: a --suffix variant becomes the ladder's rung, the rung it replaces moves aside, and every scored row
follows its quant. A half-relabelled qb_*.json would render one rung's scores under the other's name."""

import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "quantization"))

import quant  # noqa: E402


def _job(root):
    job = types.SimpleNamespace(dir=str(root), main=str(root / "main"), name=root.name, base_repo="org/model")
    job.revdir = lambda rev: os.path.join(job.dir, rev)
    job.revisions = lambda: quant.Job.revisions(job)
    job.is_complete = lambda rev: os.path.isfile(os.path.join(job.revdir(rev), "quantization_config.json"))
    return job


def _rung(root, rev, marker, complete=True):
    d = root / rev
    d.mkdir()
    (d / "marker").write_text(marker)
    if complete:
        (d / "quantization_config.json").write_text("{}")


@pytest.fixture
def job(tmp_path, monkeypatch):
    root = tmp_path / "Model-exl3"
    root.mkdir()
    for rev, marker in (("2.50bpw-H5", "plain"), ("2.50bpw-H5-yaqa", "yaqa"), ("3.00bpw-H5", "three")):
        _rung(root, rev, marker)
    (root / "main" / "_qbench_parts").mkdir(parents=True)
    rows = [{"label": "HF BF16", "group": "reference"}, {"label": "2.50bpw-H5", "kld": 1.0},
            {"label": "2.50bpw-H5-yaqa", "kld": 2.0}, {"label": "3.00bpw-H5", "kld": 3.0}]
    for name in ("qb_wild.json", "qb_recall.json", "_qbench_parts/qb_wild-p0.json"):
        (root / "main" / name).write_text(json.dumps(rows))
    for d, names in (("_recipes", ("2.50bpw-H5.yaml", "2.50bpw-H5-yaqa.yaml", "2.50bpw-H5-yaqa.measurement.json")),
                     ("logs", ("2.50bpw-H5.log", "2.50bpw-H5-yaqa.log", "2.50bpw-H5-yaqa-recipe.log",
                               "baseline-2.50bpw-H5.log"))):
        (root / d).mkdir(exist_ok=True)
        for n in names:
            (root / d / n).write_text(n)
    monkeypatch.setattr(quant, "job_in_use", lambda job: [])
    return _job(root)


def _labels(job, name):
    with open(os.path.join(job.main, name)) as f:
        return {r["label"]: r.get("kld") for r in json.load(f)}


def test_dry_run_changes_nothing(job):
    before = sorted(os.listdir(job.dir))
    assert quant.do_promote(job, types.SimpleNamespace(suffix="yaqa", aside="prev", yes=False))
    assert sorted(os.listdir(job.dir)) == before
    assert _labels(job, "qb_wild.json")["2.50bpw-H5"] == 1.0


def test_promotion_swaps_rungs_scores_and_files(job):
    assert quant.do_promote(job, types.SimpleNamespace(suffix="yaqa", aside="prev", yes=True))
    read = lambda rev: open(os.path.join(job.dir, rev, "marker")).read()
    assert read("2.50bpw-H5") == "yaqa" and read("2.50bpw-H5-prev") == "plain"
    assert not os.path.exists(os.path.join(job.dir, "2.50bpw-H5-yaqa"))
    for name in ("qb_wild.json", "qb_recall.json", "_qbench_parts/qb_wild-p0.json"):
        assert _labels(job, name) == {"HF BF16": None, "2.50bpw-H5": 2.0, "2.50bpw-H5-prev": 1.0, "3.00bpw-H5": 3.0}
    recipes, logs = (sorted(os.listdir(os.path.join(job.dir, d))) for d in ("_recipes", "logs"))
    assert recipes == ["2.50bpw-H5-prev.yaml", "2.50bpw-H5.measurement.json", "2.50bpw-H5.yaml"]
    assert logs == ["2.50bpw-H5-prev.log", "2.50bpw-H5-recipe.log", "2.50bpw-H5.log", "baseline-2.50bpw-H5.log"]
    assert open(os.path.join(job.dir, "_recipes", "2.50bpw-H5.yaml")).read() == "2.50bpw-H5-yaqa.yaml"


def test_refusals(job, tmp_path):
    args = types.SimpleNamespace(suffix="yaqa", aside="prev", yes=True)
    _rung(tmp_path / "Model-exl3", "2.50bpw-H5-prev", "taken")
    assert quant.do_promote(job, args) is False                 # the aside name is taken
    assert quant.do_promote(job, types.SimpleNamespace(suffix="nope", aside="prev", yes=True)) is False
    _rung(tmp_path / "Model-exl3", "3.00bpw-H5-half", "x", complete=False)
    assert quant.do_promote(job, types.SimpleNamespace(suffix="half", aside="old", yes=True)) is False
    assert open(os.path.join(job.dir, "2.50bpw-H5", "marker")).read() == "plain"   # nothing moved


def test_busy_job_is_refused(job, monkeypatch):
    monkeypatch.setattr(quant, "job_in_use", lambda job: [(123, "qbench ...")])
    assert quant.do_promote(job, types.SimpleNamespace(suffix="yaqa", aside="prev", yes=True)) is False
    assert open(os.path.join(job.dir, "2.50bpw-H5", "marker")).read() == "plain"
