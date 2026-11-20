"""A work directory resumes only into the rung it was started as. quantize resumed with -w -r alone, so convert.py
finished whatever the directory began: the LFM2.5-8B-A1B probe's draws began as recipe rungs, resumed after the probe
changed to uniform bits, and came out recipe rungs every arm was measured against (2026-11-20)."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "quantization"))

import quant  # noqa: E402

UNIFORM = quant.conversion_intent(None, "4.0", "6", 1, None, "always", 4)
RECIPE = quant.conversion_intent("r.yaml", "4.0", "6", 1, None, "always", 4)


def _workdir(tmp_path, intent=None, args=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "args.json").write_text(json.dumps(args or {}))
    if intent is not None:
        (tmp_path / quant.INTENT_FILE).write_text(json.dumps(intent))
    return str(tmp_path)


def test_a_recipe_start_does_not_resume_as_uniform_bits(tmp_path):
    w = _workdir(tmp_path, RECIPE)
    assert quant.intent_mismatch(w, UNIFORM) == {"recipe": (True, False)}


def test_older_work_directories_are_read_from_args_json(tmp_path):
    # the LFM draws: started with -rcp before the intent was recorded
    w = _workdir(tmp_path / "old", args={"recipe": "work/x/_recipes/4.00bpw-probe-d0.yaml", "out_scales": "always",
                                         "embed_bits": 4, "hq": 0})
    assert "recipe" in quant.intent_mismatch(w, UNIFORM)
    w = _workdir(tmp_path / "old2", args={"bits": 4.0, "head_bits": 6.0, "hq": 1, "out_scales": "always", "embed_bits": 4})
    assert quant.intent_mismatch(w, UNIFORM) == {}


def test_the_same_settings_resume(tmp_path):
    assert quant.intent_mismatch(_workdir(tmp_path, UNIFORM), UNIFORM) == {}


def test_a_changed_hq_or_factor_is_caught(tmp_path):
    w = _workdir(tmp_path, UNIFORM)
    assert quant.intent_mismatch(w, quant.conversion_intent(None, "4.0", "6", 2, None, "always", 4)) == {"hq": (1, 2)}
    yq = quant.conversion_intent(None, "4.0", "6", 1, "work/x/_hess/probe", "always", 4)
    assert quant.intent_mismatch(w, yq) == {"hessians": (None, "work/x/_hess/probe")}
