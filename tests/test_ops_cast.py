"""ops._cast_output: the kernels' fp16 output converted to the model's dtype in place, in its own storage."""
import pytest
import torch

from vllm_exl3_plugin import ops

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("numel", [7, 1 << 22, 3 * (1 << 22) + 5, 8192 * 6656])
def test_in_place_cast_gives_what_to_gives(device, numel):
    """A fully overlapping dtype-converting copy_ is not documented behaviour; this pins it, on the device the
    plugin relies on it."""
    g = torch.Generator(device=device).manual_seed(0)
    y = (torch.randn(numel, generator=g, device=device) * 1000).half().reshape(1, -1)
    want = y.to(torch.bfloat16)
    got = ops._cast_output(y.clone(), torch.bfloat16)
    assert got.dtype == torch.bfloat16 and got.shape == y.shape
    assert torch.equal(got, want)


def test_large_outputs_reuse_their_storage():
    y = torch.zeros(2, 1 << 22, dtype=torch.half)
    got = ops._cast_output(y, torch.bfloat16)
    assert got.data_ptr() == y.data_ptr(), "a full second copy is what this avoids"


def test_other_dtypes_fall_back_to_to():
    y = torch.ones(3, 1 << 20, dtype=torch.half)
    got = ops._cast_output(y, torch.float32)
    assert got.dtype == torch.float32 and torch.equal(got, y.float())
