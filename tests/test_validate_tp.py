"""The TP split guard checks only the dimension a layer actually splits.

A replicated layer splits nothing; it keeps the world's tp_size all the same, so the guard used to read its full width
as one rank's share and refuse HY4 Flash's 576-wide kv_a_proj_with_mqa at TP=2. The same width on a layer that does
split its output must still be refused: 576 / 2 cuts a 128-wide Hadamard block.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear

from vllm_exl3_plugin.format import EXL3FormatError
from vllm_exl3_plugin.quantization.linear import EXL3LinearMethod


def _layer(cls, tp_size):
    layer = object.__new__(cls)          # no distributed init: only the type and tp_size are read
    layer.tp_size = tp_size
    return layer


class ValidateTPTest(unittest.TestCase):
    def test_replicated_layer_is_not_split(self):
        EXL3LinearMethod._validate_tp(_layer(ReplicatedLinear, 2), 3072, [576])

    def test_column_split_that_cuts_a_block_is_refused(self):
        with self.assertRaises(EXL3FormatError):
            EXL3LinearMethod._validate_tp(_layer(ColumnParallelLinear, 2), 3072, [288])

    def test_row_split_checks_the_input(self):
        EXL3LinearMethod._validate_tp(_layer(RowParallelLinear, 2), 4096, [3072])
        with self.assertRaises(EXL3FormatError):
            EXL3LinearMethod._validate_tp(_layer(RowParallelLinear, 2), 288, [3072])


if __name__ == "__main__":
    unittest.main()


class ReplicatedLoadingTest(unittest.TestCase):
    """A replicated layer's EXL3 tensor reaches every rank whole, rank 1 included."""

    def test_every_rank_stores_the_whole_tensor(self):
        import torch

        from vllm_exl3_plugin import stock_compat
        from vllm_exl3_plugin.quantization.linear import EXL3Parameter

        stock_compat._replicated_linear_loading(ReplicatedLinear)
        t = torch.arange(3072 * 4, dtype=torch.int16).reshape(192, 64, 1)
        for rank in (0, 1):
            param = torch.Tensor._make_subclass(EXL3Parameter, torch.empty(0))   # a tensor subclass: no vLLM init
            stored = []
            param.store = lambda w, shard_id=0, _s=stored: _s.append(w)
            param.tp_size, param.tp_rank, param.row_shard_size = 2, rank, 3072
            param.role = "trellis"
            ReplicatedLinear.weight_loader(None, param, t)
            self.assertEqual(len(stored), 1)
            self.assertTrue(torch.equal(stored[0], t), f"rank {rank} stored {tuple(stored[0].shape)}")
