"""Tests for `bench/core.py`'s source provenance, specifically its scoping.

A capture records the provenance of the tree it was produced from, and that tree
*contains* `bench/expected/`, where the capture is about to be written. So every
field that counts uncommitted state has to exclude the baselines, or blessing
describes its own output and bakes into a baseline a number no clean checkout
can reproduce.

`dirty_files` and `diff_sha` were scoped from the start; `untracked` was not,
and the gap only shows on a bless that *adds* entries rather than overwriting
them -- new baselines are untracked at the moment they are counted, so the count
climbs 1, 2, ... across them. Found 2026-08-24.

These run against a throwaway repo rather than the project's own tree, so they
neither depend on nor disturb its working state.
"""

import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import core


def _git(path, *args):
    subprocess.run(["git", "-C", path, *args], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _write(path, rel, text=""):
    full = os.path.join(path, rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(text)


class ProvenanceScopeTest(unittest.TestCase):
    """A repo shaped like this one: a package, and a baselines directory."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = self._tmp.name
        _git(self.repo, "init", "-q")
        _git(self.repo, "config", "user.email", "t@example.invalid")
        _git(self.repo, "config", "user.name", "test")
        _write(self.repo, "pkg/mod.py", "x = 1\n")
        _write(self.repo, "bench/expected/entry.json", "{}\n")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-qm", "initial")

    def tearDown(self):
        self._tmp.cleanup()

    def prov(self):
        return core.source_provenance(self.repo, core._PROVENANCE_EXCLUDE)

    def test_clean_tree_reports_clean(self):
        p = self.prov()
        self.assertEqual(p["dirty_files"], 0)
        self.assertEqual(p["untracked"], 0)
        self.assertIsNone(p["diff_sha"])

    def test_rewritten_baseline_is_not_dirt(self):
        """Blessing over an existing entry: the tracked-modification case."""
        _write(self.repo, "bench/expected/entry.json", '{"changed": true}\n')
        p = self.prov()
        self.assertEqual(p["dirty_files"], 0)
        self.assertIsNone(p["diff_sha"])

    def test_new_baselines_are_not_untracked_dirt(self):
        """Blessing a *new* entry -- the case that was wrong.

        Two new baselines, as a bless adding two entries writes, must leave the
        count at zero rather than at one and then two.
        """
        _write(self.repo, "bench/expected/new_a.json", "{}\n")
        self.assertEqual(self.prov()["untracked"], 0)
        _write(self.repo, "bench/expected/new_b.json", "{}\n")
        self.assertEqual(self.prov()["untracked"], 0)

    def test_untracked_source_is_still_counted(self):
        """The exclusion must not swallow what the field is for.

        An untracked `.py` inside the package changes behaviour and is exactly
        what no version string can see.
        """
        _write(self.repo, "pkg/extra.py", "y = 2\n")
        self.assertEqual(self.prov()["untracked"], 1)

    def test_modified_source_is_still_dirt(self):
        _write(self.repo, "pkg/mod.py", "x = 2\n")
        p = self.prov()
        self.assertEqual(p["dirty_files"], 1)
        self.assertIsNotNone(p["diff_sha"])

    def test_dirty_files_is_the_total_and_untracked_is_a_breakdown(self):
        """The three fields are not disjoint, and reading them as if they were
        misreads a baseline.

        `dirty_files` counts porcelain status lines, which include untracked
        ones, so it is the *total* uncommitted count. `diff_sha` digests tracked
        modifications only, and `untracked` names the remainder -- the part no
        diff can see. One modified source plus one untracked source is therefore
        `dirty_files: 2, untracked: 1`, not 1 and 1.
        """
        _write(self.repo, "pkg/mod.py", "x = 3\n")
        _write(self.repo, "pkg/extra.py", "y = 2\n")
        p = self.prov()
        self.assertEqual(p["dirty_files"], 2)
        self.assertEqual(p["untracked"], 1)
        self.assertIsNotNone(p["diff_sha"])

    def test_source_and_baseline_changes_are_separated(self):
        """Both at once: only the source half may register."""
        _write(self.repo, "pkg/mod.py", "x = 3\n")
        _write(self.repo, "bench/expected/entry.json", '{"changed": true}\n')
        _write(self.repo, "bench/expected/new.json", "{}\n")
        p = self.prov()
        self.assertEqual(p["dirty_files"], 1)
        self.assertEqual(p["untracked"], 0)

    def test_unscoped_call_still_sees_everything(self):
        """The exclusion is the caller's choice, not baked into the function --
        vLLM and exllamav3 are recorded with no exclusion at all."""
        _write(self.repo, "bench/expected/new.json", "{}\n")
        self.assertEqual(core.source_provenance(self.repo)["untracked"], 1)

    def test_non_repo_returns_none(self):
        with tempfile.TemporaryDirectory() as plain:
            self.assertIsNone(core.source_provenance(plain))


if __name__ == "__main__":
    unittest.main()


class BlessGuardTest(unittest.TestCase):
    """`bless` refuses from a tree with uncommitted tracked changes.

    This used to be a warning, and it looked at the plugin tree only -- a dirty
    vLLM was normal back when the patches lived in its working tree as unstaged
    edits. Both dependencies are pinned submodules now, so a dirty one means
    somebody edited a submodule without committing, which is the drift pinning
    them was meant to end. A baseline blessed from that names a state no
    checkout can return to.

    The distinction these pin down is `diff_sha` versus `dirty_files`: only the
    former means *tracked* modifications, and keying on the latter refuses on a
    tree whose only sin is a scratch file.
    """

    def setUp(self):
        from unittest import mock

        self._tmps = [tempfile.TemporaryDirectory() for _ in range(2)]
        self.plugin, self.dep = (t.name for t in self._tmps)
        for repo, rel in ((self.plugin, "pkg/mod.py"), (self.dep, "vllm/mod.py")):
            _git(repo, "init", "-q")
            _git(repo, "config", "user.email", "t@example.invalid")
            _git(repo, "config", "user.name", "test")
            _write(repo, rel, "x = 1\n")
            _write(repo, "bench/expected/entry.json", "{}\n")
            _git(repo, "add", "-A")
            _git(repo, "commit", "-qm", "initial")

        from bench import run
        self.run = run
        self._patch = mock.patch.object(
            core, "_source_trees",
            lambda: {"plugin": self.plugin, "vllm": self.dep},
        )
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        for t in self._tmps:
            t.cleanup()

    def guard(self, allow_dirty=False):
        """Return code, with the guard's own output swallowed."""
        import contextlib, io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = self.run.refuse_if_dirty(allow_dirty)
        return rc, buf.getvalue()

    def test_clean_trees_allow_bless(self):
        rc, _ = self.guard()
        self.assertEqual(rc, 0)

    def test_modified_plugin_refuses(self):
        _write(self.plugin, "pkg/mod.py", "x = 2\n")
        rc, out = self.guard()
        self.assertEqual(rc, 1)
        self.assertIn("plugin", out)

    def test_modified_dependency_refuses(self):
        """The flip: a dirty submodule is an anomaly now, not the normal state."""
        _write(self.dep, "vllm/mod.py", "x = 2\n")
        rc, out = self.guard()
        self.assertEqual(rc, 1)
        self.assertIn("vllm", out)

    def test_untracked_only_reports_but_allows(self):
        """`dirty_files` counts untracked lines too; `diff_sha` is the real test."""
        _write(self.dep, "vllm/scratch.txt", "notes\n")
        self.assertTrue(core.source_provenance(self.dep)["dirty_files"])
        self.assertIsNone(core.source_provenance(self.dep)["diff_sha"])
        rc, out = self.guard()
        self.assertEqual(rc, 0)
        self.assertIn("untracked", out)

    def test_allow_dirty_overrides(self):
        _write(self.dep, "vllm/mod.py", "x = 2\n")
        self.assertEqual(self.guard()[0], 1)
        rc, out = self.guard(allow_dirty=True)
        self.assertEqual(rc, 0)
        self.assertIn("--allow-dirty", out)

    def test_blessed_baselines_are_not_dirt(self):
        """The exclusion still holds: writing a baseline must not block a bless."""
        _write(self.plugin, "bench/expected/entry.json", '{"new": true}\n')
        _write(self.plugin, "bench/expected/added.json", "{}\n")
        rc, _ = self.guard()
        self.assertEqual(rc, 0)
