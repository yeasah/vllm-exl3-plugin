"""Tests for the tied-checkpoint guard in `tools/quantize_embedding.py`.

The tool was documented as scoped to untied models and enforced that nowhere, so
running it on a tied checkpoint produced an artifact that corrupted at serve
time. The trap is that a tied EXL3 checkpoint *does* carry a dense
`embed_tokens.weight` -- next to a trellis `lm_head` -- so the tensor's presence
looks like evidence of untying and is not. Found 2026-08-26, from a repaired
tied `gemma-4-12B-it-exl3`.

Tied checkpoints are now supported rather than refused: the serving path landed
the same day. What remains is that the output carries an invisible requirement --
a plugin new enough to have `EXL3BlockQTiedEmbeddingMethod` -- and older ones
serve it as garbage without complaint. `is_tied` is what drives saying so.

The notice runs before shard discovery, so a directory holding nothing but a
`config.json` is enough to exercise it: every case then fails on "no
.safetensors", and what distinguishes them is what was said on the way there.
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


class TiedNoticeTest(unittest.TestCase):
    """Tied checkpoints are now *permitted* -- `EXL3BlockQTiedEmbeddingMethod`
    serves them -- but the output carries a requirement nothing downstream can
    see, so the tool has to say it. An older plugin loads such a checkpoint
    without complaint and emits garbage."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_tied_checkpoint_is_allowed_through(self):
        """Proven by reaching the *next* failure -- missing shards -- rather
        than being turned away at the door."""
        r = _run(_ckpt(self.tmp, {"tie_word_embeddings": True}))
        self.assertIn("no .safetensors", r.stderr)

    def test_tied_checkpoint_warns_about_plugin_version(self):
        r = _run(_ckpt(self.tmp, {"tie_word_embeddings": True}))
        self.assertIn("EXL3BlockQTiedEmbeddingMethod", r.stderr)
        self.assertIn("is tied", r.stderr)

    def test_unknown_tiedness_warns_too(self):
        """None is not False: a checkpoint that cannot be read must not be
        quietly treated as untied."""
        r = _run(_ckpt(self.tmp, None))
        self.assertIn("may be tied", r.stderr)

    def test_untied_checkpoint_gets_no_notice(self):
        """The common path stays quiet."""
        r = _run(_ckpt(self.tmp, {"tie_word_embeddings": False}))
        self.assertNotIn("EXL3BlockQTiedEmbeddingMethod", r.stderr)
        self.assertIn("no .safetensors", r.stderr)


if __name__ == "__main__":
    unittest.main()
