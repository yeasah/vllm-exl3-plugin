"""Which n-gram table `engram.ngram_source` resolves: the body's own revision unless overridden.

An earlier version let the EXL3_NGRAM_REVISION lookup overwrite the `revision` argument, so with the override unset
every Hub model's table was fetched from `main`, which holds none (a 404). These tests record what reaches
`hf_hub_download` instead of touching the network.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_exl3_plugin import engram


class NgramSourceTest(unittest.TestCase):
    def _resolve(self, model, revision, env):
        calls = []

        def fake(repo, filename, revision=None, **kwargs):
            calls.append((repo, filename, revision))
            return f"/cache/{repo}/{revision}/{filename}"

        with mock.patch.dict(os.environ, env, clear=False), mock.patch("huggingface_hub.hf_hub_download", fake):
            for k in ("EXL3_NGRAM_REVISION", "EXL3_NGRAM_FILE"):
                if k not in env:
                    os.environ.pop(k, None)
            engram.ngram_source(model, revision)
        return calls

    def test_follows_the_body_revision(self):
        calls = self._resolve("org/model-exl3", "2.05bpw_h4_ng4", {})
        self.assertEqual(calls, [("org/model-exl3", engram.NGRAM_FILE, "2.05bpw_h4_ng4")])

    def test_override_wins(self):
        calls = self._resolve("org/model-exl3", "2.05bpw_h4_ng4", {"EXL3_NGRAM_REVISION": "1.66bpw_h4_ng3"})
        self.assertEqual(calls, [("org/model-exl3", engram.NGRAM_FILE, "1.66bpw_h4_ng3")])

    def test_local_file_needs_no_hub(self):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            calls = self._resolve("org/model-exl3", "2.05bpw_h4_ng4", {"EXL3_NGRAM_FILE": f.name})
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
