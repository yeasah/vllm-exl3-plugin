"""probe.py's arms, as the quantize commands it would run: names, suffixes and controls. An --arm whose suffix did not
make <rate>-probe-TAG would be built under one name and looked for under another (caught by reading, 2026-11-19)."""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "quantization"))

import probe  # noqa: E402
import quant  # noqa: E402


def _plan(tmp_path, monkeypatch, *, measured, arms=(), low_bits=0.0):
    if measured:
        cats = tmp_path / "_measure" / "categories"
        cats.mkdir(parents=True)
        (cats / "self.0-10.json").write_text("{}")
    monkeypatch.setattr(probe, "job_dir", lambda model: str(tmp_path))
    lanes, reported = [], {}
    monkeypatch.setattr(probe, "run_lanes", lambda model, ls: lanes.extend(ls))
    monkeypatch.setattr(probe, "run", lambda cmd, env=None: None)
    monkeypatch.setattr(probe, "report", lambda W, slices, draws, uniform, yaqa, low=None, extra=None:
                        reported.update(uniform=uniform, extra=extra))
    a = types.SimpleNamespace(model="org/M", bits=4.0, draws=2, yaqa=False, arm=[probe.parse_arm(x) for x in arms],
                              low_bits=low_bits, slices="wild", device="0,1,2", yaqa_devices=None, trace_args="",
                              layers_per_run=None, reserve_vram=None)
    probe.main(a)
    cmds = {}
    for lane in lanes:
        for _, cmd, env in lane:
            name = quant.bit_specs(cmd[cmd.index("-b") + 1], cmd[cmd.index("--suffix") + 1])
            cmds[name[0][0] if "--draws" not in cmd else "draws"] = (cmd, env)
    return cmds, reported


def test_moe_without_a_recipe_has_no_uniform_arm(tmp_path, monkeypatch):
    cmds, rep = _plan(tmp_path, monkeypatch, measured=False, arms=["hq2=--hq 2"])
    assert set(cmds) == {"draws", "4.00bpw-probe-hq2"}
    cmd, env = cmds["4.00bpw-probe-hq2"]
    assert cmd[cmd.index("--suffix") + 1] == "probe-hq2" and "--hq" in cmd and env == {"EXL3_SEED_IDX_OFFSET": "0"}
    assert rep == {"uniform": None, "extra": {"4.00bpw-probe-hq2": "4.00bpw-probe-d0"}}


def test_with_a_recipe_no_recipe_arms_compare_against_uniform(tmp_path, monkeypatch):
    cmds, rep = _plan(tmp_path, monkeypatch, measured=True, arms=["hq2=--no-recipe --hq 2", "os=--out-scales never"],
                      low_bits=2.5)
    assert {"draws", "4.00bpw-probe-uniform", "4.00bpw-probe-hq2", "4.00bpw-probe-os", "2.50bpw-H5-probe-low"} == set(cmds)
    assert rep["extra"] == {"4.00bpw-probe-hq2": "4.00bpw-probe-uniform", "4.00bpw-probe-os": "4.00bpw-probe-d0"}


@pytest.mark.parametrize("spec", ["x=-b 3", "d0=--hq 2", "low=--hq 2", "a b=--hq 2", "nosep"])
def test_refused_arms(spec):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        probe.parse_arm(spec)
