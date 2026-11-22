"""ops._cast_output: the kernels' fp16 output converted to the model's dtype in place, chunk by chunk."""
import pytest
import torch

from vllm_exl3_plugin import ops


@pytest.mark.parametrize("numel", [7, ops._CAST_CHUNK, 3 * ops._CAST_CHUNK + 5])
def test_in_place_cast_gives_what_to_gives(numel):
    g = torch.Generator().manual_seed(0)
    y = (torch.randn(numel, generator=g) * 1000).half().reshape(1, -1)
    want = y.to(torch.bfloat16)
    got = ops._cast_output(y.clone(), torch.bfloat16)
    assert got.dtype == torch.bfloat16 and got.shape == y.shape
    assert torch.equal(got, want)


def test_large_outputs_reuse_their_storage():
    y = torch.zeros(2, ops._CAST_CHUNK, dtype=torch.half)
    got = ops._cast_output(y, torch.bfloat16)
    assert got.data_ptr() == y.data_ptr(), "a full second copy is what this avoids"


def test_other_dtypes_fall_back_to_to():
    y = torch.ones(3, ops._CAST_CHUNK, dtype=torch.half)
    got = ops._cast_output(y, torch.float32)
    assert got.dtype == torch.float32 and torch.equal(got, y.float())
