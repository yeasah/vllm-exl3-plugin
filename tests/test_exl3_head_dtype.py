"""An EXL3 head under a head_dtype other than the activations' serves exllamav3's fp16 logits, cast once.

Fed bf16, exl3_mm would round its fp16 output to bf16 before any cast to fp32; the head is fed fp16 instead, and
_get_logits casts after the (TP) gather and the vocabulary trim. Checked with a stand-in head whose output is exactly
representable in fp16 but not in bf16, so a bf16 detour would show.
"""

import os
import sys
import types
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm.model_executor.layers.logits_processor import LogitsProcessor

from vllm_exl3_plugin import stock_compat
from vllm_exl3_plugin.quantization.lm_head import EXL3LMHeadMethod


class _Head(EXL3LMHeadMethod):
    def __init__(self):        # no quant config: only apply is used
        self.seen = []

    def apply(self, layer, x, bias=None):
        self.seen.append(x.dtype)
        # 1 + 2**-9: exact in fp16 (10 mantissa bits), rounds to 1.0 in bf16 (7)
        return torch.full((x.shape[0], 8), 1 + 2 ** -9, dtype=x.dtype)


class Exl3HeadDtypeTest(unittest.TestCase):
    def setUp(self):
        stock_compat._exl3_head_dtype(LogitsProcessor)
        self.lp = object.__new__(LogitsProcessor)
        torch.nn.Module.__init__(self.lp)
        self.lp.head_dtype, self.lp.org_vocab_size = torch.float32, 6
        self.head = _Head()
        self.lm_head = types.SimpleNamespace(quant_method=self.head, tp_size=1)
        self.hidden = torch.zeros(3, 4, dtype=torch.bfloat16)

    def test_get_logits_feeds_fp16_and_casts_once(self):
        out = self.lp._get_logits(self.hidden, self.lm_head, None)
        self.assertEqual(self.head.seen, [torch.float16])
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(tuple(out.shape), (3, 6))
        self.assertTrue(torch.all(out == 1 + 2 ** -9), "logits went through bf16")

    def test_apply_head_keeps_head_dtype_for_other_callers(self):
        out = self.lp._apply_head(self.lm_head, self.hidden, None)
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.all(out == 1 + 2 ** -9))


if __name__ == "__main__":
    unittest.main()
