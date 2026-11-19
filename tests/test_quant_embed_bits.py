"""quant.embed_bits_of, which the baseline stage compares against the ladder's depth to find stale rungs.

It read the safetensors manifest, which a rung need not have, and assumed K * hidden / 16 + 8 words a trellis row,
which holds only at hidden 2048: every Qwen3.8-27B baseline read as K=None and stale (2026-11-19).
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "quantization"))

import quant  # noqa: E402


def _rung(tmp_path, hidden, stored):
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"hidden_size": hidden}}))
    storage = {"model.language_model.embed_tokens": {"stored_tensors": stored},
               "mtp.embed_tokens": {"stored_tensors": {"mtp.embed_tokens.weight": {"shape": [8, hidden]}}}}
    (tmp_path / "quantization_config.json").write_text(json.dumps({"tensor_storage": storage}))
    return str(tmp_path)


def test_trellis_depth_from_the_row_layout(tmp_path):
    # 27B: hidden 5120 is 20 groups of 256, each 1 + 16 K words; 1300 words is K=4 (the old formula said 4.04)
    rung = _rung(tmp_path, 5120, {"model.language_model.embed_tokens.trellis": {"shape": [248320, 1300]},
                                  "model.language_model.embed_tokens.signs": {"shape": [20, 256]}})
    assert quant.embed_bits_of(rung) == 4


def test_k5_and_hidden_2048(tmp_path):
    rung = _rung(tmp_path, 2048, {"model.language_model.embed_tokens.trellis": {"shape": [1000, 8 * 81]}})
    assert quant.embed_bits_of(rung) == 5


def test_dense_without_a_manifest(tmp_path):
    rung = _rung(tmp_path, 4096, {"model.language_model.embed_tokens.weight": {"shape": [1000, 4096]}})
    assert not os.path.exists(os.path.join(rung, "model.safetensors.index.json"))
    assert quant.embed_bits_of(rung) == 16


def test_unrecognizable_is_none(tmp_path):
    rung = _rung(tmp_path, 5120, {"model.language_model.embed_tokens.trellis": {"shape": [1000, 1301]}})
    assert quant.embed_bits_of(rung) is None
