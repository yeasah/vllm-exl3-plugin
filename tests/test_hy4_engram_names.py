"""exllamav3's stacked HY4 Flash Engram tensors go back to the per-stream and per-head names vLLM's loader takes.

The split is a reshape in index order (exllamav3/modules/ple.py stacks them that way); checked here by stacking known
parts as exllamav3 does and splitting them back, with everything else in the stream passed through untouched.
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_exl3_plugin.stock_compat import _hy4_engram_unstack


class Hy4EngramUnstackTest(unittest.TestCase):
    H, D, W, HEADS, HD = 4, 12, 6, 16, 5

    def setUp(self):
        g = torch.Generator().manual_seed(0)
        r = lambda *s: torch.randn(*s, generator = g)
        self.scales = [r(self.D) for _ in range(self.H)]
        self.kw = [r(self.D, self.W) for _ in range(self.H)]
        self.kb = [r(self.D) for _ in range(self.H)]
        self.nk = [r(self.D) for _ in range(self.H)]
        self.nq = [r(self.D) for _ in range(self.H)]
        self.hn = [r(self.HD) for _ in range(self.HEADS)]
        p = "layers.3.engram_layer."
        self.stream = [
            (p + "branch_scales", torch.stack(self.scales)),
            (p + "key_proj.weight", torch.cat(self.kw)),
            (p + "key_proj.bias", torch.cat(self.kb)),
            (p + "norm_key.weight", torch.stack(self.nk)),
            (p + "norm_query.weight", torch.stack(self.nq)),
            (p + "head_norm.weight", torch.stack(self.hn)),
            (p + "value_proj.weight", r(self.D, self.W)),
            (p + "lookup_table", torch.arange(7)),
            ("layers.3.mlp.gate.weight", r(3, 3)),
        ]

    def test_parts_come_back_under_checkpoint_names(self):
        out = dict(_hy4_engram_unstack(iter(self.stream), self.H))
        p = "layers.3.engram_layer."
        for i in range(self.H):
            self.assertTrue(torch.equal(out[p + f"unified_per_hc_scales.{i}"], self.scales[i]))
            self.assertTrue(torch.equal(out[p + f"key_projs.{i}.weight"], self.kw[i]))
            self.assertTrue(torch.equal(out[p + f"key_projs.{i}.bias"], self.kb[i]))
            self.assertTrue(torch.equal(out[p + f"norm_key.{i}.weight"], self.nk[i]))
            self.assertTrue(torch.equal(out[p + f"norm_query.{i}.weight"], self.nq[i]))
        for h in range(self.HEADS):
            self.assertTrue(torch.equal(out[p + f"lookup_per_head_norms.{h}.weight"], self.hn[h]))
        for stacked in ("branch_scales", "key_proj.weight", "key_proj.bias", "norm_key.weight", "head_norm.weight"):
            self.assertNotIn(p + stacked, out)

    def test_everything_else_passes_through(self):
        out = dict(_hy4_engram_unstack(iter(self.stream), self.H))
        for name, value in self.stream[6:]:
            self.assertIs(out[name], value)

    def test_refuses_a_stack_that_does_not_split(self):
        bad = [("layers.1.engram_layer.key_proj.weight", torch.zeros(10, 3))]
        with self.assertRaisesRegex(ValueError, "does not split into 4"):
            list(_hy4_engram_unstack(iter(bad), 4))


if __name__ == "__main__":
    unittest.main()
