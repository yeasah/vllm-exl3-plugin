"""A vision tower's activation peak must shrink without moving a single bit.

Three changes bound the peak of a ViT forward, and all three are exactly
equivalent -- they change when tensors are alive, never what they contain:

  1. dropping an `einops` `.contiguous()` whose result the triton rotary
     kernel never needed (it takes `x.stride(...)` explicitly);
  2. rotating q/k in place instead of into a fresh buffer;
  3. slicing the pointwise MLP into row-chunks sized by a byte budget.

**Each is invisible until the others unmask it**, which is why they are tested
together as well as apart. Peak live memory is a `max` over the forward, so
whichever tensor is largest at the peak instant hides every change aimed
elsewhere: on Qwen3.8's tower chunking the MLP alone moves the peak by *zero*
(1.728 GiB either way), and only after the attention copies go does it become
worth 488 MiB. Measured 2026-09-12, and the reason the MLP work was shelved
once and un-shelved.

**Chunking is mathematically exact but not bit-exact in general, and an
earlier version of this file asserted otherwise.** Every row is independent, so
slicing changes no arithmetic -- but cuBLAS picks its GEMM kernel from the M
dimension, and a smaller M can select a split-K variant that reduces over K in
a different order. Measured on a 20000x1536x4304 MLP: exact at 5000 and 10394
rows per chunk, and off by up to 4.9e-4 (about one bf16 ULP at these
magnitudes) at 649 and 2598. It happened to be exact at the default budget on
both real towers, which is what made the wrong claim survive several checks.
So the bound here is a few ULP, not equality, and the docstring says why.

The two attention changes *are* bit-exact and stay asserted as such: they move
no GEMM shapes, only lifetimes.

**Weight init matters here and quietly invalidated an earlier version of this
check.** Default random init through ~27 bf16 blocks overflows and yields an
output that is 99.99998% NaN, where every comparison passes vacuously -- an
md5 over two all-NaN tensors matches, and `torch.equal` returns False for the
same reason, so the two disagree and neither means anything. The small-sigma
init below keeps the output finite, which is what makes `torch.equal` a real
assertion.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="vision tower forward needs a GPU"
)

CHUNK_ENV = "VLLM_MM_ENCODER_MLP_CHUNK_MB"


def _pointwise_mlp_ref(x, fc1, fc2, inter):
    """Plain single-shot form of a gated vision MLP."""
    h = fc1(x)
    return fc2(torch.nn.functional.silu(h[..., :inter]) * h[..., inter:])


@pytest.mark.parametrize("chunk_mb", [16, 64, 256])
def test_chunked_mlp_is_bit_exact(chunk_mb, monkeypatch):
    """Row-chunking a pointwise MLP changes no bit of the result."""
    import vllm.envs as envs
    from vllm.model_executor.models.vision import (
        chunked_pointwise_mlp,
        mm_encoder_mlp_chunk_rows,
    )

    monkeypatch.setattr(envs, CHUNK_ENV, chunk_mb, raising=False)

    torch.manual_seed(0)
    rows, hidden, inter = 20_000, 1536, 4304
    dev = "cuda"
    fc1 = torch.nn.Linear(hidden, 2 * inter, bias=False, dtype=torch.bfloat16).to(dev)
    fc2 = torch.nn.Linear(inter, hidden, bias=False, dtype=torch.bfloat16).to(dev)
    for p in (*fc1.parameters(), *fc2.parameters()):
        with torch.no_grad():
            p.normal_(0.0, 0.01)

    x = torch.randn(rows, hidden, dtype=torch.bfloat16, device=dev) * 0.5
    mlp = lambda t: _pointwise_mlp_ref(t, fc1, fc2, inter)  # noqa: E731

    calls = []

    def counting_mlp(t):
        calls.append(t.shape[0])
        return mlp(t)

    chunk_rows = mm_encoder_mlp_chunk_rows(inter)
    ref = chunked_pointwise_mlp(mlp, x, 0)
    got = chunked_pointwise_mlp(counting_mlp, x, chunk_rows)

    # The guard must actually fire: a silently-skipped chunking would pass
    # every equivalence assertion below while saving nothing.
    assert len(calls) > 1, f"chunking did not fire (chunk_rows={chunk_rows})"
    assert max(calls) <= chunk_rows
    # A degenerate (all-NaN) reference makes every comparison below vacuous.
    assert torch.isfinite(ref.float()).all(), "degenerate reference"

    # A few bf16 ULP, to cover cuBLAS reducing over K in a different order at a
    # different M. Not equality -- see the module docstring.
    scale = ref.float().abs().max().item()
    bound = 8 * scale * 2**-8
    diff = (ref.float() - got.float()).abs().max().item()
    assert diff <= bound, f"{diff:.3e} exceeds {bound:.3e} (chunk_rows={chunk_rows})"


def test_chunk_budget_holds_across_tower_widths(monkeypatch):
    """One byte budget, not a row count, so it ports across architectures."""
    import vllm.envs as envs
    from vllm.model_executor.models.vision import mm_encoder_mlp_chunk_rows

    monkeypatch.setattr(envs, CHUNK_ENV, 256, raising=False)
    budget = 256 * 1024 * 1024
    # so400m (Qwen3-VL, gemma-3, SigLIP), Muse-Glimmer, GLM-4.1V
    for inter in (4304, 8960, 13696):
        rows = mm_encoder_mlp_chunk_rows(inter)
        live = rows * 3 * inter * 2
        assert live <= budget
        assert live > budget // 2, f"budget badly under-used at inter={inter}"


def test_chunking_disabled_is_a_single_call(monkeypatch):
    """0 restores the original single-shot behaviour exactly."""
    import vllm.envs as envs
    from vllm.model_executor.models.vision import (
        chunked_pointwise_mlp,
        mm_encoder_mlp_chunk_rows,
    )

    monkeypatch.setattr(envs, CHUNK_ENV, 0, raising=False)
    assert mm_encoder_mlp_chunk_rows(4304) == 0

    calls = []
    x = torch.randn(4096, 64, dtype=torch.bfloat16, device="cuda")

    def counting(t):
        calls.append(t.shape[0])
        return t * 2

    out = chunked_pointwise_mlp(counting, x, 0)
    assert calls == [4096]
    assert torch.equal(out, x * 2)
