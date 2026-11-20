"""probe.py's arms, as the quantize commands it would run: names, suffixes and controls. An --arm whose suffix did not
make <rate>-probe-TAG would be built under one name and looked for under another (caught by reading, 2026-11-19)."""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "quantization"))

import probe  # noqa: E402
import quant  # noqa: E402


def _plan(tmp_path, monkeypatch, *, measured, arms=(), low_bits=0.0, yaqa=False):
    if measured:
        cats = tmp_path / "_measure" / "categories"
        cats.mkdir(parents=True)
        (cats / "self.0-10.json").write_text("{}")
    monkeypatch.setattr(probe, "job_dir", lambda model: str(tmp_path))
    lanes, reported = [], {}
    monkeypatch.setattr(probe, "run_jobs", lambda devs, model, jobs: lanes.extend(jobs))
    monkeypatch.setattr(probe, "run", lambda cmd, env=None: None)
    monkeypatch.setattr(probe, "report", lambda W, slices, draws, arms, low=None: reported.update(arms=arms, low=low))
    a = types.SimpleNamespace(model="org/M", bits=4.0, draws=2, yaqa=yaqa, arm=[probe.parse_arm(x) for x in arms],
                              low_bits=low_bits, slices="wild", device="0,1,2", yaqa_devices=None, trace_args="",
                              layers_per_run=None, reserve_vram=None, quant_parallel=1)
    probe.main(a)
    cmds = {}
    for _, _, cmd, env in lanes:
        if cmd[0] != "quantize":
            continue
        name = quant.bit_specs(cmd[cmd.index("-b") + 1], cmd[cmd.index("--suffix") + 1])
        cmds[name[0][0] if "--draws" not in cmd else "draws"] = (cmd, env)
    return cmds, reported


def _last(cmd, on, off):
    """Which of a --x/--no-x pair is in force: argparse takes the last."""
    hits = [x for x in cmd if x in (on, off)]
    return hits[-1] == on if hits else None


def test_every_arm_is_one_component_over_the_plain_draws(tmp_path, monkeypatch):
    cmds, rep = _plan(tmp_path, monkeypatch, measured=True, arms=["hq2=--hq 2", "rcp=--recipe"], low_bits=2.5, yaqa=True)
    names = ["4.00bpw-probe-recipe", "4.00bpw-probe-yaqa", "4.00bpw-probe-hq2", "4.00bpw-probe-rcp"]
    assert set(cmds) == {"draws", "2.50bpw-H5-probe-low", *names}
    assert rep == {"arms": names, "low": "2.50bpw-H5-probe-low"}
    # (recipe, yaqa) in force per rung: the draws and the low draw are calibration alone, each arm adds one thing
    want = {"draws": (False, False), "2.50bpw-H5-probe-low": (False, False), "4.00bpw-probe-recipe": (True, False),
            "4.00bpw-probe-yaqa": (False, True), "4.00bpw-probe-hq2": (False, False), "4.00bpw-probe-rcp": (True, False)}
    for name, (rcp, yq) in want.items():
        cmd, env = cmds[name]
        assert (_last(cmd, "--recipe", "--no-recipe"), _last(cmd, "--yaqa", "--no-yaqa") is not False and
                "--no-yaqa" not in cmd) == (rcp, yq), name
        assert name == "draws" or env == {"EXL3_SEED_IDX_OFFSET": "0"}
    hq = cmds["4.00bpw-probe-hq2"][0]
    assert hq[hq.index("--suffix") + 1] == "probe-hq2" and hq[hq.index("--hq") + 1] == "2"


def test_no_measurement_no_recipe_arm(tmp_path, monkeypatch):
    cmds, rep = _plan(tmp_path, monkeypatch, measured=False, arms=["hq2=--hq 2"])
    assert set(cmds) == {"draws", "4.00bpw-probe-hq2"} and rep["arms"] == ["4.00bpw-probe-hq2"]


@pytest.mark.parametrize("spec", ["x=-b 3", "d0=--hq 2", "low=--hq 2", "recipe=--hq 2", "a b=--hq 2", "nosep"])
def test_refused_arms(spec):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        probe.parse_arm(spec)


def test_scheduler_with_more_devices_than_jobs(monkeypatch):
    """Two GPUs listed twice (--quant-parallel 2) and three single-GPU jobs: once the last is placed the free slots
    must not be offered a job that does not exist (it raised IndexError)."""
    ran = []
    monkeypatch.setattr(probe, "run", lambda cmd, env=None: ran.append(cmd[cmd.index("-d") + 1]))
    jobs = [(lambda: False, 1, ["quantize", "--suffix", f"j{i}"], None) for i in range(3)]
    probe.run_jobs(["0", "1", "0", "1"], "org/M", jobs)
    assert sorted(ran) == ["0", "0", "1"]
