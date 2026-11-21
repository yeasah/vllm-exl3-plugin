"""probe.py's arms, as the quantize commands it would run: names, suffixes and controls. An --arm whose suffix did not
make <rate>-probe-TAG would be built under one name and looked for under another (caught by reading, 2026-11-19)."""

import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "quantization"))

import probe  # noqa: E402
import quant  # noqa: E402


def _plan(tmp_path, monkeypatch, *, measured, arms=(), low_bits=0.0, yaqa=False, arm_draws=()):
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
                              layers_per_run=None, reserve_vram=None, quant_parallel=1, qbench_parallel=1,
                              arm_draws=list(arm_draws))
    probe.main(a)
    cmds = {}
    for _, _, cmd, env in lanes:
        if cmd[0] != "quantize":
            continue
        name = quant.bit_specs(cmd[cmd.index("-b") + 1], cmd[cmd.index("--suffix") + 1])[0][0]
        cmds[name if "--draws" not in cmd else name + f" x{cmd[cmd.index('--draws') + 1]}"] = (cmd, env)
    return cmds, reported


def _last(cmd, on, off):
    """Which of a --x/--no-x pair is in force: argparse takes the last."""
    hits = [x for x in cmd if x in (on, off)]
    return hits[-1] == on if hits else None


def test_every_arm_is_one_component_over_the_plain_draws(tmp_path, monkeypatch):
    cmds, rep = _plan(tmp_path, monkeypatch, measured=True, arms=["hq2=--hq 2", "rcp=--recipe"], low_bits=2.5, yaqa=True)
    names = {"recipe": ["4.00bpw-probe-recipe-d0"], "yaqa": ["4.00bpw-probe-yaqa-d0"],
             "hq2": ["4.00bpw-probe-hq2-d0"], "rcp": ["4.00bpw-probe-rcp-d0"]}
    assert set(cmds) == {"4.00bpw-probe x2", "2.50bpw-H5-probe-low-d0", *[n[0] for n in names.values()]}
    assert rep == {"arms": names, "low": ["2.50bpw-H5-probe-low-d0"]}
    # (recipe, yaqa) in force per rung: the draws and the low draw are calibration alone, each arm adds one thing
    want = {"4.00bpw-probe x2": (False, False), "2.50bpw-H5-probe-low-d0": (False, False),
            "4.00bpw-probe-recipe-d0": (True, False), "4.00bpw-probe-yaqa-d0": (False, True),
            "4.00bpw-probe-hq2-d0": (False, False), "4.00bpw-probe-rcp-d0": (True, False)}
    for name, (rcp, yq) in want.items():
        cmd, env = cmds[name]
        assert (_last(cmd, "--recipe", "--no-recipe"), _last(cmd, "--yaqa", "--no-yaqa") is not False and
                "--no-yaqa" not in cmd) == (rcp, yq), name
        assert name.endswith(" x2") or env == {"EXL3_SEED_IDX_OFFSET": "0"}       # singletons at offset 0
    hq = cmds["4.00bpw-probe-hq2-d0"][0]
    assert hq[hq.index("--suffix") + 1] == "probe-hq2-d0" and hq[hq.index("--hq") + 1] == "2"


def test_no_measurement_no_recipe_arm(tmp_path, monkeypatch):
    cmds, rep = _plan(tmp_path, monkeypatch, measured=False, arms=["hq2=--hq 2"])
    assert set(cmds) == {"4.00bpw-probe x2", "4.00bpw-probe-hq2-d0"} and list(rep["arms"]) == ["hq2"]


def test_arm_draws(tmp_path, monkeypatch):
    """Three recipe draws are one quantize --draws 3 (offsets 0..2, names -d0..-d2), so a later raise keeps -d0."""
    cmds, rep = _plan(tmp_path, monkeypatch, measured=True, arm_draws=[("recipe", 3)])
    cmd, env = cmds["4.00bpw-probe-recipe x3"]
    assert cmd[cmd.index("--suffix") + 1] == "probe-recipe" and env is None           # quant.py sets the offsets
    assert rep["arms"]["recipe"] == [f"4.00bpw-probe-recipe-d{k}" for k in range(3)]
    with pytest.raises(SystemExit, match="no such arm"):
        _plan(tmp_path / "x", monkeypatch, measured=False, arm_draws=[("recipe", 2)])


def test_a_singleton_from_before_is_adopted_as_d0(tmp_path):
    W = tmp_path
    (W / "4.00bpw-probe-recipe").mkdir()
    (W / "main").mkdir()
    (W / "logs").mkdir()
    (W / "logs" / "4.00bpw-probe-recipe.log").write_text("x")
    (W / "main" / "qb_swe.json").write_text(json.dumps([{"label": "4.00bpw-probe-recipe", "kld": 1.0},
                                                       {"label": "4.00bpw-probe-recipe2", "kld": 2.0}]))
    probe.adopt_singleton(str(W), "4.00bpw-probe-recipe", "4.00bpw-probe-recipe-d0")
    assert (W / "4.00bpw-probe-recipe-d0").is_dir() and not (W / "4.00bpw-probe-recipe").exists()
    assert (W / "logs" / "4.00bpw-probe-recipe-d0.log").exists()
    labels = [r["label"] for r in json.load(open(W / "main" / "qb_swe.json"))]
    assert labels == ["4.00bpw-probe-recipe-d0", "4.00bpw-probe-recipe2"]                # exact labels only


def test_report_averages_an_arms_draws(tmp_path, capsys):
    (tmp_path / "main").mkdir()
    rows = [{"label": "Noise floor", "kld": 0.0}] + \
           [{"label": f"4.00bpw-probe-d{i}", "kld": v, "vram_gb": 10.0} for i, v in enumerate((1.0, 1.1, 0.9))] + \
           [{"label": f"4.00bpw-probe-recipe-d{i}", "kld": v, "vram_gb": 10.0} for i, v in enumerate((0.8, 0.9))]
    (tmp_path / "main" / "qb_swe.json").write_text(json.dumps(rows))
    probe.report(str(tmp_path), ["swe"], [f"4.00bpw-probe-d{i}" for i in range(3)],
                 {"recipe": ["4.00bpw-probe-recipe-d0", "4.00bpw-probe-recipe-d1"]})
    out = json.load(open(tmp_path / "main" / "probe.json"))
    assert abs(out["arms"]["recipe"]["swe"] - 0.85) < 1e-9 and out["arms"]["recipe"]["draws"] == 2
    assert "recipex2" in capsys.readouterr().out
    # sd 10% of the mean; two means of 2 and 3 draws: 2 x 0.1 x sqrt(1/2 + 1/3) = 18%, so 0.85 is within it


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


def test_size_matched_projection(capsys):
    """Bits from the low draw's size: 4.00 at 10 GiB and 2.50 at 6.8 make 2.133 GiB a bit, so an arm 0.2 GiB smaller is
    0.094 bpw lighter. Quality from a bracket: the rule's 2x per half bit, and the low draw's own rate (5x over 1.5
    bits = 1.71x per half bit, under the cap)."""
    col = lambda t: t
    excess = {"wild": {"draws": 1.0, "r": 1.04, "low": 5.0}}
    sizes = {"draws": 10.0, "r": 9.8, "low": 6.8}
    out = {"arms": {"r": {}}}
    probe.size_matched(out, excess, sizes, "4.00bpw-probe-d0", {"r": 1}, ["2.50bpw-H5-probe-low-d0"], col)
    b = -0.2 / (3.2 / 1.5)
    assert abs(out["arms"]["r"]["bits_over_draws"] - b) < 1e-9
    lo, hi = out["arms"]["r"]["size_matched"]["wild"]
    own = 5.0 ** (1 / 3)
    assert abs(lo - 1.04 * 2 ** (2 * b)) < 1e-9 and abs(hi - 1.04 * own ** (2 * b)) < 1e-9
    # past the knee (16x over 1.5 bits = 2.52x per half bit) the rate is capped at the rule: one value
    out = {"arms": {"r": {}}}
    probe.size_matched(out, {"wild": {**excess["wild"], "low": 16.0}}, sizes, "4.00bpw-probe-d0", {"r": 1},
                       ["2.50bpw-H5-probe-low-d0"], col)
    lo, hi = out["arms"]["r"]["size_matched"]["wild"]
    assert lo == hi
    # no low draw: bits from size / rate
    out = {"arms": {"r": {}}}
    probe.size_matched(out, {"wild": {"draws": 1.0, "r": 1.04}}, {"draws": 10.0, "r": 9.8}, "4.00bpw-probe-d0",
                       {"r": 1}, None, col)
    assert abs(out["arms"]["r"]["bits_over_draws"] - (-0.2 / 2.5)) < 1e-9
    # an arm the same size gets no projection
    out = {"arms": {"r": {}}}
    probe.size_matched(out, excess, {**sizes, "r": 10.01}, "4.00bpw-probe-d0", {"r": 1}, None, col)
    assert "size_matched" not in out["arms"]["r"]


