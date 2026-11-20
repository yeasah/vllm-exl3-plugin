"""quant.yaqa_reserve: the --reserve-vram a user would pick for the factor pass over several GPUs. The tool's own 8 GiB a
card left LFM2.5-8B-A1B (16.6 GB in bf16) no room on 2x16 GB; 7 fit (2026-11-20)."""

import json
import os
import struct
import subprocess
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "quantization"))

import quant  # noqa: E402

GiB = 2**30


def _model(tmp_path, gib_bf16, layers=24, extra=None):
    """A checkpoint whose header holds `gib_bf16` GiB worth of bf16 elements (stored as FP8, to check the 2x), only the
    header written: the estimate never reads the data."""
    per = int(gib_bf16 * GiB / 2 / layers)             # spread over the layers, as a checkpoint has it
    tensors = {f"model.layers.{i}.w": {"dtype": "F8_E4M3", "shape": [per], "data_offsets": [i * per, (i + 1) * per]}
               for i in range(layers)}
    tensors["mtp.layers.0.w"] = {"dtype": "BF16", "shape": [10**9], "data_offsets": [layers * per, layers * per + 2 * 10**9]}
    tensors.update(extra or {})
    head = json.dumps(tensors).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(head)) + head)
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": layers, "vocab_size": 65536}))
    rows = tmp_path / "rows.safetensors"
    save_file({"input_ids": torch.zeros((2, 2048), dtype=torch.int16)}, str(rows))
    return str(tmp_path), str(rows)


def _cards(monkeypatch, *free_gib):
    out = "\n".join(f"{i}, {int(g * 1024)}" for i, g in enumerate(free_gib))
    monkeypatch.setattr(quant.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=out, stderr=""))


def test_lfm_on_two_16gb_cards(tmp_path, monkeypatch):
    path, rows = _model(tmp_path, 15.8)
    _cards(monkeypatch, 15.5, 15.5)
    reserve, how = quant.yaqa_reserve(path, rows, ["0", "1"])
    assert reserve == 6.5, how            # 6 and 7 both fit its dry run; the mtp tensor is not counted


def test_small_model_keeps_the_tool_default(tmp_path, monkeypatch):
    path, rows = _model(tmp_path, 1.4)
    _cards(monkeypatch, 15.5, 15.5)
    assert quant.yaqa_reserve(path, rows, ["0", "1"])[0] == 8.0


def test_too_big_comes_out_non_positive_and_quietly(tmp_path, monkeypatch, capsys):
    path, rows = _model(tmp_path, 51.0)
    _cards(monkeypatch, 15.5, 15.5)
    assert quant.yaqa_reserve(path, rows, ["0", "1"])[0] <= 0
    assert "!!" not in capsys.readouterr().out    # collect_hout refuses it; no headroom warning about a negative


def test_listed_cards_only(tmp_path, monkeypatch):
    path, rows = _model(tmp_path, 15.8)
    _cards(monkeypatch, 2.0, 15.5, 15.5)      # card 0 busy, not listed
    assert quant.yaqa_reserve(path, rows, ["1", "2"])[0] == 6.5


def _collect(tmp_path, monkeypatch, device, reserve_vram, gib):
    import types
    path, rows = _model(tmp_path, gib)
    _cards(monkeypatch, 15.5, 15.5)
    ran = []
    monkeypatch.setattr(quant, "run_logged", lambda cmd, log, env=None, echo=True: ran.append(cmd) or 1)
    monkeypatch.setattr(quant, "yaqa_layers_per_run", lambda *a: 24)
    job = types.SimpleNamespace(log=lambda name: str(tmp_path / "logs" / f"{name}.log"))
    args = types.SimpleNamespace(layers_per_run=None, exllamav3dir="x", device=device, reserve_vram=reserve_vram)
    quant.collect_hout(job, args, path, rows, str(tmp_path / "out"), "probe")
    return ran


def test_collect_passes_the_estimate_and_respects_an_explicit_one(tmp_path, monkeypatch):
    rv = lambda cmd: cmd[cmd.index("-rv") + 1] if "-rv" in cmd else None
    assert rv(_collect(tmp_path, monkeypatch, "0,1", None, 15.8)[0]) == "6.5"
    assert rv(_collect(tmp_path, monkeypatch, "0,1", "5", 15.8)[0]) == "5"
    assert rv(_collect(tmp_path, monkeypatch, "0", None, 15.8)[0]) is None       # one card keeps no reserve
    assert _collect(tmp_path, monkeypatch, "0,1", None, 51.0) == []               # refused before running


def _fit(tmp_path, monkeypatch, outcomes, totals=(15.5, 15.5)):
    """yaqa_layers_per_run against scripted dry runs: outcomes maps a layer count to (exit code, log text)."""
    import types
    tried = []
    def run(cmd, log, env=None, echo=True):
        step = int(cmd[cmd.index("-l") + 1].split(":")[1])
        tried.append(step)
        rc, text = outcomes.get(step, (0, " -- Peak VRAM: cuda:0 9.0 GB, cuda:1 8.0 GB"))
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "a") as f:
            f.write(text + "\n")
        return rc
    monkeypatch.setattr(quant, "run_logged", run)
    out = "\n".join(f"{i}, {int(g * 1024)}" for i, g in enumerate(totals))
    monkeypatch.setattr(quant.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=out, stderr=""))
    job = types.SimpleNamespace(log=lambda name: str(tmp_path / "logs" / f"{name}.log"))
    args = types.SimpleNamespace(exllamav3dir="x", device="0,1")
    return quant.yaqa_layers_per_run(job, args, "m", "rows", "6.5", 24, str(tmp_path / "out")), tried


def test_fit_halves_on_oom_and_on_a_peak_too_close_to_capacity(tmp_path, monkeypatch):
    step, tried = _fit(tmp_path, monkeypatch, {
        24: (1, "torch.OutOfMemoryError: CUDA out of memory"),
        12: (0, " -- Peak VRAM: cuda:0 14.9 GB, cuda:1 11.0 GB")})        # within 1 GiB of 15.5
    assert (step, tried) == (6, [24, 12, 6])
    assert open(tmp_path / "out" / ".layers_per_run").read() == "6"
    assert _fit(tmp_path, monkeypatch, {}) == (6, [])                   # a resumed job keeps its ranges


def test_fit_refuses_when_one_layer_does_not_fit_and_stops_on_other_errors(tmp_path, monkeypatch):
    oom = (1, "torch.OutOfMemoryError: CUDA out of memory")
    assert _fit(tmp_path / "a", monkeypatch, {n: oom for n in (24, 12, 6, 3, 1)}) == (None, [24, 12, 6, 3, 1])
    assert _fit(tmp_path / "b", monkeypatch, {24: (1, "KeyError: 'model.layers.0'")}) == (None, [24])
