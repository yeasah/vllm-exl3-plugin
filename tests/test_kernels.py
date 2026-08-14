"""Correctness oracles against exllamav3 itself.

Requires a CUDA device and an installed exllamav3. Everything here is skipped
otherwise, so `python -m unittest discover -s tests` still works on a machine
with neither.

The checkpoint tests need `turboderp/Llama-3.2-1B-Instruct-exl3`; set
`EXL3_TEST_MODEL` / `EXL3_TEST_REVISION` to point at something else.
"""

from __future__ import annotations

import os
import unittest

try:
    import torch

    HAVE_CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    torch = None
    HAVE_CUDA = False

try:
    import exllamav3_ext  # noqa: F401

    HAVE_EXT = True
except ImportError:  # pragma: no cover
    HAVE_EXT = False

MODEL = os.environ.get("EXL3_TEST_MODEL", "turboderp/Llama-3.2-1B-Instruct-exl3")
REVISION = os.environ.get("EXL3_TEST_REVISION", "3.0bpw")

requires_gpu = unittest.skipUnless(
    HAVE_CUDA and HAVE_EXT, "needs CUDA and exllamav3's compiled extension"
)


def _load_layer(key: str, device="cuda:0"):
    """Pull one quantized linear's tensors straight out of the checkpoint."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = snapshot_download(MODEL, revision=REVISION)
    tensors = {}
    for name in sorted(os.listdir(path)):
        if not name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(path, name), framework="pt", device=device) as f:
            for k in f.keys():
                if k.startswith(key + "."):
                    tensors[k[len(key) + 1 :]] = f.get_tensor(k)
    if "trellis" not in tensors:
        raise unittest.SkipTest(f"{key} not found in {MODEL}@{REVISION}")
    return tensors


@requires_gpu
class TestHadamard(unittest.TestCase):
    def test_matches_exllamav3(self):
        """Our Sylvester construction must be exllamav3's matrix, not merely *a*
        Hadamard matrix -- the checkpoint was quantized with theirs."""
        from exllamav3.util.hadamard import get_hadamard

        from vllm_exl3_plugin.ops import _hadamard

        import math

        for n in (16, 32, 64, 128):
            with self.subTest(n=n):
                theirs = get_hadamard(n).cuda().float()  # entries are +/-1
                ours = _hadamard(n, torch.device("cuda:0"), torch.float32)
                # Compare the sign pattern exactly -- that is the part that has
                # to agree with the quantizer. Rescaling by sqrt(n) to undo the
                # normalization is not an exact round trip in fp32, so testing
                # the scale separately keeps this an equality check.
                torch.testing.assert_close(torch.sign(ours), theirs, rtol=0, atol=0)
                self.assertTrue(
                    torch.all(ours.abs() == torch.full_like(ours, 1 / math.sqrt(n)))
                )


@requires_gpu
class TestDequantOracle(unittest.TestCase):
    """`ops.dense_weight` must reproduce exllamav3's own dequantization."""

    KEYS = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.mlp.down_proj",
    ]

    def _linear(self, t, in_features, out_features):
        from exllamav3.modules.quant.exl3 import LinearEXL3

        return LinearEXL3(
            config=None,
            in_features=in_features,
            out_features=out_features,
            suh=t["suh"],
            svh=t["svh"],
            trellis=t["trellis"],
            mcg=t.get("mcg"),
            mul1=t.get("mul1"),
            bias=t.get("bias"),
        )

    def test_matches_get_weight_tensor(self):
        from vllm_exl3_plugin import format, ops

        for key in self.KEYS:
            with self.subTest(key):
                t = _load_layer(key)
                in_f, out_f = format.dims_from_trellis_shape(t["trellis"].shape)
                bits = format.bits_from_trellis_shape(t["trellis"].shape)

                ours = ops.dense_weight(
                    t["trellis"],
                    t["suh"],
                    t["svh"],
                    bits,
                    "mcg" in t,
                    "mul1" in t,
                )
                theirs = self._linear(t, in_f, out_f).get_weight_tensor()

                self.assertEqual(tuple(ours.shape), (out_f, in_f))
                torch.testing.assert_close(ours, theirs.t(), rtol=0, atol=0)

    def test_matches_forward(self):
        """The stronger check: our dense weight against the real fused kernel.

        `get_weight_tensor` and `forward` are separate code paths in exllamav3;
        agreeing with the one that actually runs during inference is what
        matters. fp16 accumulation order differs, so this is a tolerance check,
        not equality.
        """
        from vllm_exl3_plugin import format, ops

        for key in self.KEYS:
            with self.subTest(key):
                t = _load_layer(key)
                in_f, out_f = format.dims_from_trellis_shape(t["trellis"].shape)
                bits = format.bits_from_trellis_shape(t["trellis"].shape)

                torch.manual_seed(0)
                x = torch.randn((4, in_f), dtype=torch.half, device="cuda:0")

                linear = self._linear(t, in_f, out_f)
                theirs = linear.forward(x, {})

                w = ops.dense_weight(
                    t["trellis"], t["suh"], t["svh"], bits, "mcg" in t, "mul1" in t
                )
                ours = torch.nn.functional.linear(x, w)

                torch.testing.assert_close(ours, theirs, rtol=2e-2, atol=2e-2)



@requires_gpu
class TestFusedKernel(unittest.TestCase):
    """`exl3_mm` must agree with the dequantized reference.

    This is the Phase 1 correctness claim: the fused kernel never materializes
    a weight, so its only check is that it produces what multiplying by the
    dequantized weight would have. `dense_weight` is itself pinned bit-exact
    against exllamav3 by TestDequantOracle, which makes it a valid reference.

    The batch sizes straddle every dispatch boundary the kernel has: a GEMV
    path for small m, the autotuned cooperative GEMM above it, and exllamav3's
    own reconstruct threshold at 144 (which this plugin deliberately does not
    follow -- reconstructing would defeat the memory saving).
    """

    BATCHES = (1, 2, 4, 17, 144, 145, 512)

    CASES = [
        # (repo, revision, key) -- covers the default codebook and mcg
        (
            "turboderp/Llama-3.2-1B-Instruct-exl3",
            "3.0bpw",
            "model.layers.0.self_attn.q_proj",
        ),
        (
            "turboderp/MiniCPM5-1B-exl3",
            "3.00bpw",
            "model.layers.0.mlp.down_proj",
        ),
    ]

    def test_matches_dense_reference(self):
        from tests.remote_tensors import fetch_module_tensors
        from vllm_exl3_plugin import format, ops

        for repo, revision, key in self.CASES:
            try:
                t = fetch_module_tensors(repo, revision, key)
            except OSError as e:
                self.skipTest(f"could not fetch {repo}@{revision}: {e}")

            in_f, _ = format.dims_from_trellis_shape(t["trellis"].shape)
            bits = format.bits_from_trellis_shape(t["trellis"].shape)
            mcg, mul1 = "mcg" in t, "mul1" in t
            weight = ops.dense_weight(
                t["trellis"], t["suh"], t["svh"], bits, mcg, mul1
            )

            for m in self.BATCHES:
                with self.subTest(key=key, m=m):
                    torch.manual_seed(0)
                    x = torch.randn(
                        (m, in_f), dtype=torch.half, device="cuda:0"
                    ) * 0.1
                    ref = torch.nn.functional.linear(x, weight)
                    got = ops.exl3_mm(
                        x, t["trellis"], t["suh"], t["svh"], mcg, mul1
                    )
                    self.assertEqual(got.shape, ref.shape)
                    # fp16 accumulation in a different order, not a different
                    # computation: differences are last-bit noise.
                    torch.testing.assert_close(got, ref, rtol=2e-2, atol=5e-3)

    def test_preserves_leading_dims(self):
        """vLLM hands linears 2D activations, but the op should not assume it."""
        from tests.remote_tensors import fetch_module_tensors
        from vllm_exl3_plugin import format, ops

        repo, revision, key = self.CASES[0]
        try:
            t = fetch_module_tensors(repo, revision, key)
        except OSError as e:
            self.skipTest(f"could not fetch {repo}@{revision}: {e}")
        in_f, out_f = format.dims_from_trellis_shape(t["trellis"].shape)

        x = torch.randn((3, 5, in_f), dtype=torch.half, device="cuda:0") * 0.1
        y = ops.exl3_mm(x, t["trellis"], t["suh"], t["svh"], "mcg" in t, "mul1" in t)
        self.assertEqual(tuple(y.shape), (3, 5, out_f))
        flat = ops.exl3_mm(
            x.reshape(15, in_f), t["trellis"], t["suh"], t["svh"],
            "mcg" in t, "mul1" in t,
        )
        torch.testing.assert_close(y.reshape(15, out_f), flat, rtol=0, atol=0)

    def test_empty_batch(self):
        """Zero-token forwards happen; the kernel must not be launched."""
        from tests.remote_tensors import fetch_module_tensors
        from vllm_exl3_plugin import format, ops

        repo, revision, key = self.CASES[0]
        try:
            t = fetch_module_tensors(repo, revision, key)
        except OSError as e:
            self.skipTest(f"could not fetch {repo}@{revision}: {e}")
        in_f, out_f = format.dims_from_trellis_shape(t["trellis"].shape)

        x = torch.empty((0, in_f), dtype=torch.half, device="cuda:0")
        y = ops.exl3_mm(x, t["trellis"], t["suh"], t["svh"], "mcg" in t, "mul1" in t)
        self.assertEqual(tuple(y.shape), (0, out_f))


@requires_gpu
class TestEmbedRows(unittest.TestCase):
    """`embed_rows` must return exactly what indexing the dense weight would.

    This is the Phase 4 correctness claim, and it is an equality claim rather
    than a tolerance one: gathering a row decodes a strict subset of the tensor
    (the row's own 128-block), but every arithmetic step it performs is the same
    step `dense_weight` performs on the same values, so the results should agree
    bit for bit. A tolerance here would hide exactly the bug this is for --
    picking up the wrong block, or applying `had_right` across a block boundary.
    """

    CASES = [
        # An lm_head, which is what Phase A actually serves an embedding from,
        # plus an ordinary linear to show nothing about this is head-specific.
        ("turboderp/Llama-3.2-1B-Instruct-exl3", "3.0bpw", "lm_head"),
        (
            "turboderp/Llama-3.2-1B-Instruct-exl3",
            "3.0bpw",
            "model.layers.0.self_attn.q_proj",
        ),
    ]

    def test_matches_dense_reference(self):
        from tests.remote_tensors import fetch_module_tensors
        from vllm_exl3_plugin import format, ops

        for repo, revision, key in self.CASES:
            with self.subTest(key=key):
                try:
                    t = fetch_module_tensors(repo, revision, key)
                except OSError as e:
                    self.skipTest(f"could not fetch {repo}@{revision}: {e}")
                bits = format.bits_from_trellis_shape(t["trellis"].shape)
                _, out_f = format.dims_from_trellis_shape(t["trellis"].shape)
                mcg, mul1 = "mcg" in t, "mul1" in t

                dense = ops.dense_weight(
                    t["trellis"], t["suh"], t["svh"], bits, mcg, mul1
                )
                # Block boundaries either side, a within-block run, duplicates
                # (which must dedupe to one decode without changing the answer),
                # and the last row -- the one a padded vocabulary gets wrong.
                ids = torch.tensor(
                    [0, 1, 127, 128, 129, 255, 256, 7, 7, 128, out_f - 1],
                    device="cuda:0",
                )
                got = ops.embed_rows(
                    t["trellis"], t["suh"], t["svh"], bits, mcg, mul1, ids
                )
                torch.testing.assert_close(got, dense[ids], rtol=0, atol=0)

    def test_single_row_and_empty(self):
        """The decode path (one token) and zero-token forwards both happen."""
        from tests.remote_tensors import fetch_module_tensors
        from vllm_exl3_plugin import format, ops

        repo, revision, key = self.CASES[0]
        try:
            t = fetch_module_tensors(repo, revision, key)
        except OSError as e:
            self.skipTest(f"could not fetch {repo}@{revision}: {e}")
        bits = format.bits_from_trellis_shape(t["trellis"].shape)
        in_f, _ = format.dims_from_trellis_shape(t["trellis"].shape)
        mcg, mul1 = "mcg" in t, "mul1" in t
        dense = ops.dense_weight(t["trellis"], t["suh"], t["svh"], bits, mcg, mul1)

        one = torch.tensor([9707], device="cuda:0")
        got = ops.embed_rows(t["trellis"], t["suh"], t["svh"], bits, mcg, mul1, one)
        torch.testing.assert_close(got, dense[one], rtol=0, atol=0)

        empty = torch.empty((0,), dtype=torch.long, device="cuda:0")
        got = ops.embed_rows(t["trellis"], t["suh"], t["svh"], bits, mcg, mul1, empty)
        self.assertEqual(tuple(got.shape), (0, in_f))

    def test_decodes_only_the_needed_blocks(self):
        """The point of the exercise: cost tracks distinct 128-blocks, not batch
        size. Asserted through `reconstruct`, since a version that decoded the
        whole tensor would still return the right rows."""
        from unittest import mock

        from tests.remote_tensors import fetch_module_tensors
        from vllm_exl3_plugin import format, ops

        repo, revision, key = self.CASES[0]
        try:
            t = fetch_module_tensors(repo, revision, key)
        except OSError as e:
            self.skipTest(f"could not fetch {repo}@{revision}: {e}")
        bits = format.bits_from_trellis_shape(t["trellis"].shape)
        mcg, mul1 = "mcg" in t, "mul1" in t

        # 300 tokens drawn from 2 distinct blocks -> 2 blocks' worth decoded.
        ids = torch.cat(
            [
                torch.arange(0, 128, device="cuda:0").repeat(2),
                torch.full((44,), 5000, device="cuda:0"),
            ]
        )
        real = ops.reconstruct
        seen = []

        def spy(trellis, *a, **kw):
            seen.append(trellis.shape)
            return real(trellis, *a, **kw)

        with mock.patch.object(ops, "reconstruct", spy):
            ops.embed_rows(t["trellis"], t["suh"], t["svh"], bits, mcg, mul1, ids)

        self.assertEqual(len(seen), 1, "should be a single fused decode")
        tiles = seen[0][1]
        self.assertEqual(
            tiles,
            2 * (format.HAD_BLOCK // format.TILE),
            f"decoded {tiles} tiles for 2 distinct blocks",
        )
        self.assertLess(tiles, t["trellis"].shape[1] / 10)


if __name__ == "__main__":
    unittest.main()
