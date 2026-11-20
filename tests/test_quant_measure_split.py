"""quant.measure_split: the --gpu-split a measurement uses when none is given -- equal headroom on every card after the
weights (and the logits, on the last), kept only if one short check run of sc_measure passes; else one card, streamed
as before. Ornith-1.5-9B on 2x16 GB: 9.4,8.2, checked in 1.8 min (2026-11-20)."""

import json
import os
import struct
import subprocess
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "quantization"))

import quant  # noqa: E402

GiB = 2**30


def _model(tmp_path, gib, layers=32, vocab=248320):
    per = int(gib * GiB / 2 / layers)
    side = int(per ** 0.5)
    tensors = {f"model.language_model.layers.{i}.mlp.down_proj.weight":
               {"dtype": "BF16", "shape": [side, per // side], "data_offsets": [2 * i * per, 2 * (i + 1) * per]}
               for i in range(layers)}
    head = json.dumps(tensors).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(head)) + head)
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"vocab_size": vocab}}))
    return str(tmp_path)


def _split(tmp_path, monkeypatch, gib, rc=0, gpus=("0", "1"), free=(15.5, 15.5)):
    path = _model(tmp_path, gib)
    ran = []
    def run(cmd, log, env=None, echo=True):
        ran.append(cmd[cmd.index("-gs") + 1])
        return rc
    monkeypatch.setattr(quant, "run_logged", run)
    out = "\n".join(f"{i}, {int(g * 1024)}" for i, g in enumerate(free))
    monkeypatch.setattr(quant.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=out, stderr=""))
    job = types.SimpleNamespace(measure=str(tmp_path / "measure"), log=lambda n: str(tmp_path / "logs" / f"{n}.log"))
    os.makedirs(job.measure, exist_ok=True)
    args = types.SimpleNamespace(exllamav3dir="x", boundary_cpu=True)
    rows = [[1] * 8] * 2
    split = quant.measure_split(job, args, path, list(gpus), rows, torch.ones((4, 8), dtype=torch.long), "none")
    return split, ran, job


def test_ornith_on_two_16gb_cards(tmp_path, monkeypatch):
    split, ran, job = _split(tmp_path, monkeypatch, 17.5)
    caps = [float(c) for c in split.split(",")]
    assert ran == [split] and abs(sum(caps) - 17.5) < 0.2 and caps[0] > caps[1]    # the last card keeps the logits
    assert open(os.path.join(job.measure, ".gpu_split")).read() == split
    assert quant.measure_split(job, None, None, [], [], None, None) == split          # recorded: no check again


def test_small_model_stays_on_one_card(tmp_path, monkeypatch):
    split, ran, _ = _split(tmp_path, monkeypatch, 1.4)
    assert (split, ran) == (None, [])


def test_failed_check_falls_back_and_is_remembered(tmp_path, monkeypatch):
    split, ran, job = _split(tmp_path, monkeypatch, 17.5, rc=1)
    assert split is None and len(ran) == 1
    assert open(os.path.join(job.measure, ".gpu_split")).read() == ""
    assert quant.measure_split(job, None, None, [], [], None, None) is None


def test_a_card_listed_twice_is_one_card(tmp_path, monkeypatch):
    split, _, _ = _split(tmp_path, monkeypatch, 17.5, gpus=("0", "0", "1", "1"))
    assert len(split.split(",")) == 2


def test_no_room_falls_back_without_a_check(tmp_path, monkeypatch):
    split, ran, _ = _split(tmp_path, monkeypatch, 28.0)
    assert (split, ran) == (None, [])
