"""Tests for the tied-checkpoint guard in `tools/quantize_embedding.py`.

The tool was documented as scoped to untied models and enforced that nowhere, so
running it on a tied checkpoint succeeded and produced an artifact that corrupts
at serve time. The trap is that a tied EXL3 checkpoint *does* carry a dense
`embed_tokens.weight` -- next to a trellis `lm_head` -- so the tensor's presence
looks like evidence of untying and is not. Found 2026-08-26, from a repaired
tied `gemma-4-12B-it-exl3`.

The refusal runs before shard discovery, so a directory holding nothing but a
`config.json` is enough to exercise it: a checkpoint that gets past the guard
fails later on "no .safetensors", which is exactly how the permitted case is
distinguished from the refused one here.
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "quantize_embedding.py")


def _load():
    spec = importlib.util.spec_from_file_location("quantize_embedding", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


qe = _load()


def _ckpt(tmp, config):
    """A checkpoint directory carrying only a config -- no shards."""
    if config is not None:
        with open(os.path.join(tmp, "config.json"), "w") as f:
            json.dump(config, f)
    return tmp


def _run(src, *extra):
    with tempfile.TemporaryDirectory() as out:
        dst = os.path.join(out, "repaired")
        return subprocess.run(
            [sys.executable, TOOL, src, dst, *extra],
            capture_output=True, text=True,
        )


class IsTiedTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_flat_tied_config(self):
        self.assertIs(qe.is_tied(_ckpt(self.tmp, {"tie_word_embeddings": True})), True)

    def test_flat_untied_config(self):
        self.assertIs(qe.is_tied(_ckpt(self.tmp, {"tie_word_embeddings": False})), False)

    def test_nested_tied_config(self):
        """Multimodal configs nest it; gemma-4 sets it in both places."""
        cfg = {"text_config": {"tie_word_embeddings": True}}
        self.assertIs(qe.is_tied(_ckpt(self.tmp, cfg)), True)

    def test_either_level_counts(self):
        cfg = {"tie_word_embeddings": False, "text_config": {"tie_word_embeddings": True}}
        self.assertIs(qe.is_tied(_ckpt(self.tmp, cfg)), True)

    def test_absent_flag_is_untied(self):
        self.assertIs(qe.is_tied(_ckpt(self.tmp, {"hidden_size": 8})), False)

    def test_missing_config_is_unknown_not_untied(self):
        """None must not be mistaken for permission -- the caller checks
        `is not False`, so an unreadable checkpoint refuses rather than runs."""
        self.assertIsNone(qe.is_tied(_ckpt(self.tmp, None)))


class GuardTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_tied_checkpoint_is_refused(self):
        r = _run(_ckpt(self.tmp, {"tie_word_embeddings": True}))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("declares tied embeddings", r.stderr)

    def test_refusal_explains_the_dense_embedding_trap(self):
        """The message has to defuse the inference that produced the bug."""
        r = _run(_ckpt(self.tmp, {"tie_word_embeddings": True}))
        self.assertIn("not evidence of untying", r.stderr)

    def test_unknown_tiedness_is_refused(self):
        r = _run(_ckpt(self.tmp, None))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("cannot be ruled out", r.stderr)

    def test_allow_tied_overrides(self):
        """The escape hatch exists so the serving path can be worked on. It must
        get *past* the guard -- proven by failing on the missing shards instead."""
        r = _run(_ckpt(self.tmp, {"tie_word_embeddings": True}), "--allow-tied")
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("declares tied embeddings", r.stderr)
        self.assertIn("no .safetensors", r.stderr)

    def test_untied_checkpoint_is_not_blocked(self):
        """The guard must not touch the path it was added to protect."""
        r = _run(_ckpt(self.tmp, {"tie_word_embeddings": False}))
        self.assertNotIn("declares tied embeddings", r.stderr)
        self.assertIn("no .safetensors", r.stderr)


if __name__ == "__main__":
    unittest.main()
