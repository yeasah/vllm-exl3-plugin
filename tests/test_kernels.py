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


if __name__ == "__main__":
    unittest.main()
