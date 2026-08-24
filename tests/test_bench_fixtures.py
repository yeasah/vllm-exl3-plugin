"""Tests for the fixture cache key in `bench/fixtures.py`.

No torch, no vLLM, no GPU, and no checkpoint -- the key is pure, and it is the
only part of the fixture machinery whose failure is *silent*. Everything else
announces itself: a build that fails raises, a fixture whose contents drift is
caught by the digest recorded in the capture. But a key that fails to change
when the encoder changes produces a gate that passes while serving a checkpoint
the current code would not produce, and nothing anywhere would say so.

So what these pin is the staleness property specifically, not the recipe.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import fixtures

MODEL = "turboderp/MiniCPM5-1B-exl3"
REVISION = "3.00bpw"


class TestFixtureKey(unittest.TestCase):
    def test_key_is_stable_across_calls(self):
        self.assertEqual(
            fixtures.key("blockq", MODEL, REVISION),
            fixtures.key("blockq", MODEL, REVISION),
        )

    def test_key_separates_base_checkpoints(self):
        """Two revisions of one repo must not share a derived checkpoint."""
        self.assertNotEqual(
            fixtures.key("blockq", MODEL, REVISION),
            fixtures.key("blockq", MODEL, "4.00bpw"),
        )
        self.assertNotEqual(
            fixtures.key("blockq", MODEL, REVISION),
            fixtures.key("blockq", "turboderp/Qwen3.5-9B-exl3", REVISION),
        )

    def test_key_is_filesystem_safe(self):
        """It becomes a directory name, and the repo id carries a slash."""
        key = fixtures.key("blockq", MODEL, REVISION)
        self.assertNotIn("/", key)
        self.assertEqual(key, os.path.basename(key))

    def test_key_changes_when_the_encoder_changes(self):
        """The property the cache rests on.

        Edit the encoder and the next run must build a new fixture rather than
        serving the one the old code produced. Exercised by perturbing the
        source list rather than the files, which is the same input the digest
        reads and leaves the tree untouched.
        """
        before = fixtures.key("blockq", MODEL, REVISION)
        original = fixtures.BLOCKQ_SOURCES
        try:
            fixtures.BLOCKQ_SOURCES = original + ("tools/quantize_embedding.py",)
            after = fixtures.key("blockq", MODEL, REVISION)
        finally:
            fixtures.BLOCKQ_SOURCES = original
        self.assertNotEqual(before, after)
        self.assertEqual(fixtures.key("blockq", MODEL, REVISION), before)

    def test_key_changes_with_recipe_version(self):
        before = fixtures.key("blockq", MODEL, REVISION)
        original = fixtures.RECIPE_VERSION
        try:
            fixtures.RECIPE_VERSION = original + 1
            self.assertNotEqual(fixtures.key("blockq", MODEL, REVISION), before)
        finally:
            fixtures.RECIPE_VERSION = original

    def test_unknown_recipe_is_refused(self):
        """Rather than producing a plausible key for a recipe nobody implements."""
        with self.assertRaises(SystemExit):
            fixtures.key("trellis", MODEL, REVISION)


class TestFixtureRoot(unittest.TestCase):
    def test_cache_root_is_outside_the_repo(self):
        """Derived checkpoints are large; `bench/expected/` is deliberately the
        only thing a run writes in-tree."""
        root = os.path.abspath(fixtures.cache_root())
        self.assertFalse(root.startswith(fixtures.ROOT + os.sep))

    def test_cache_root_is_overridable(self):
        original = os.environ.get("BENCH_FIXTURES")
        try:
            os.environ["BENCH_FIXTURES"] = "/tmp/somewhere-else"
            self.assertEqual(fixtures.cache_root(), "/tmp/somewhere-else")
        finally:
            if original is None:
                del os.environ["BENCH_FIXTURES"]
            else:
                os.environ["BENCH_FIXTURES"] = original


if __name__ == "__main__":
    unittest.main()
