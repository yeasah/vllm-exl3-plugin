"""Half-integer EXL3 bit widths (1.5 / 2.5 / 3.5 bpw, exllamav3 >= 1.6.4, mul1 codebook only).

A half-integer tensor alternates KA- and (KA + 1)-bit trellis steps, so its tile is 16 * K
uint16 words -- 56 for 3.5 -- which is not a multiple of 16. Anything that recovers K as
`shape[2] // 16` would decode it as the integer width below and produce plausible garbage, so
every path the plugin serves a linear through is checked against exllamav3 here: the dense
oracle bit-exact, the fused GEMM and the reconstruct path at row counts on both sides of
RECONSTRUCT_THRESHOLD, the embedding gather, and the routed-expert op with mixed widths.

Tensors are random, as in exllamav3's own tests (tests/modules/linear/test_linear_exl3_paths.py):
every trellis bit pattern is a valid encoding, so no quantizer or checkpoint is needed. Each case
carries an integer mul1 control (K = 4), so a broken harness shows up as a broken control.

Requires a CUDA device and exllamav3's compiled extension; skipped otherwise.
"""

from __future__ import annotations

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
    try:
        from exllamav3.ext import exllamav3_ext  # noqa: F401

        HAVE_EXT = True
    except ImportError:
        HAVE_EXT = False

requires_gpu = unittest.skipUnless(
    HAVE_CUDA and HAVE_EXT, "needs CUDA and exllamav3's compiled extension"
)

WIDTHS = (1.5, 2.5, 3.5, 4)  # 4: the integer mul1 control
DEV = "cuda:0"


def _rand_linear(k, n, K, seed):
    """trellis / suh / svh / mul1 marker for one random mul1 EXL3 linear."""
    from exllamav3.modules.quant.exl3_lib.quantize import codebook_mul1_mult

    g = torch.Generator(device="cpu").manual_seed(seed)
    words = int(16 * K)
    trellis = torch.randint(-(2 ** 15), 2 ** 15, (k // 16, n // 16, words), generator=g,
                            dtype=torch.int32).to(torch.int16)
    # Scales of either sign around unit magnitude, the shape of real suh/svh
    def scale(m):
        s = torch.rand(m, generator=g) * 0.5 + 0.75
        return (s * torch.where(torch.rand(m, generator=g) < 0.5, -1.0, 1.0)).half()
    mul1 = torch.tensor(codebook_mul1_mult, dtype=torch.uint32).view(torch.int)
    return {"trellis": trellis.to(DEV), "suh": scale(k).to(DEV), "svh": scale(n).to(DEV),
            "mul1": mul1.to(DEV)}


def _rel_rms(a, b) -> float:
    """fp16 accumulation-order differences stay small; a wrong K or a broken tile is O(1)"""
    a, b = a.float(), b.float()
    return ((a - b).square().mean().sqrt() / b.square().mean().sqrt()).item()


@requires_gpu
class TestHalfWidthLinear(unittest.TestCase):
    K_IN, N_OUT = 1024, 768

    def _theirs(self, t):
        from exllamav3.modules.quant.exl3 import LinearEXL3

        return LinearEXL3(config=None, in_features=self.K_IN, out_features=self.N_OUT,
                          suh=t["suh"], svh=t["svh"], trellis=t["trellis"], mcg=None,
                          mul1=t["mul1"], bias=None)

    def test_width_recovered(self):
        from vllm_exl3_plugin import format

        for K in WIDTHS:
            with self.subTest(K=K):
                t = _rand_linear(self.K_IN, self.N_OUT, K, seed=1)
                self.assertEqual(format.bits_from_trellis_shape(t["trellis"].shape, True), K)

    def test_dense_weight_matches_exllamav3(self):
        """The oracle every other path is held to: bit-exact against exllamav3's own decode."""
        from vllm_exl3_plugin import ops

        for K in WIDTHS:
            with self.subTest(K=K):
                t = _rand_linear(self.K_IN, self.N_OUT, K, seed=2)
                ours = ops.dense_weight(t["trellis"], t["suh"], t["svh"], K, False, True)
                theirs = self._theirs(t).get_weight_tensor()
                torch.testing.assert_close(ours, theirs.t(), rtol=0, atol=0)

    def test_exl3_mm_both_paths(self):
        """Fused kernel below RECONSTRUCT_THRESHOLD, reconstruct + cuBLAS above it."""
        from vllm_exl3_plugin import ops

        self.assertTrue(0 < 33 < ops.RECONSTRUCT_THRESHOLD < 256,
                        "row counts no longer straddle the reconstruct threshold")
        for K in WIDTHS:
            t = _rand_linear(self.K_IN, self.N_OUT, K, seed=3)
            w = ops.dense_weight(t["trellis"], t["suh"], t["svh"], K, False, True)
            for rows in (1, 8, 33, 256):
                with self.subTest(K=K, rows=rows):
                    torch.manual_seed(rows)
                    x = torch.randn((rows, self.K_IN), dtype=torch.half, device=DEV) * 0.5
                    ref = (x.float() @ w.float().t()).half()
                    got = ops.exl3_mm(x, t["trellis"], t["suh"], t["svh"], False, True)
                    self.assertLess(_rel_rms(got, ref), 1e-2)

    def test_embed_rows(self):
        from vllm_exl3_plugin import ops

        for K in WIDTHS:
            with self.subTest(K=K):
                # An embedding stores [hidden, vocab]: rows are gathered along the output dim
                t = _rand_linear(self.K_IN, self.N_OUT, K, seed=4)
                dense = ops.dense_weight(t["trellis"], t["suh"], t["svh"], K, False, True)
                ids = torch.tensor([0, 5, 127, 128, 400, self.N_OUT - 1], device=DEV)
                got = ops.embed_rows(t["trellis"], t["suh"], t["svh"], K, False, True, ids)
                torch.testing.assert_close(got, dense[ids], rtol=0, atol=2e-3)


@requires_gpu
class TestHalfWidthMoE(unittest.TestCase):
    """Routed experts with gate/up and down at different widths, one of them half-integer."""

    E, H, I, TOKENS, TOP_K = 8, 512, 256, 6, 2

    def _experts(self, gate_K, down_K):
        return [{"gate": _rand_linear(self.H, self.I, gate_K, seed=10 + e),
                 "up": _rand_linear(self.H, self.I, gate_K, seed=50 + e),
                 "down": _rand_linear(self.I, self.H, down_K, seed=90 + e)}
                for e in range(self.E)]

    def test_matches_dense_reference(self):
        from vllm_exl3_plugin import ops

        for gate_K, down_K in ((3.5, 2.5), (2.5, 4), (4, 1.5), (4, 4)):
            with self.subTest(gate=gate_K, down=down_K):
                ex = self._experts(gate_K, down_K)

                def ptrs(p, name):
                    return torch.tensor([e[p][name].data_ptr() for e in ex], dtype=torch.long,
                                        device=DEV)

                torch.manual_seed(0)
                x = torch.randn((self.TOKENS, self.H), dtype=torch.half, device=DEV) * 0.5
                topk_ids = torch.stack([torch.randperm(self.E, device=DEV)[: self.TOP_K]
                                        for _ in range(self.TOKENS)]).to(torch.int32)
                topk_w = torch.softmax(torch.randn((self.TOKENS, self.TOP_K), device=DEV), -1)

                got = ops.exl3_moe_mm(
                    x, topk_ids, topk_w,
                    ptrs("gate", "trellis"), ptrs("gate", "suh"), ptrs("gate", "svh"),
                    ptrs("up", "trellis"), ptrs("up", "suh"), ptrs("up", "svh"),
                    ptrs("down", "trellis"), ptrs("down", "suh"), ptrs("down", "svh"),
                    gate_K, down_K, self.I, self.E, False, True, "silu",
                )

                dense = [{p: ops.dense_weight(e[p]["trellis"], e[p]["suh"], e[p]["svh"],
                                              gate_K if p != "down" else down_K, False, True).float()
                          for p in ("gate", "up", "down")} for e in ex]
                ref = torch.zeros((self.TOKENS, self.H), device=DEV)
                for i in range(self.TOKENS):
                    xi = x[i].float()
                    for j in range(self.TOP_K):
                        d = dense[int(topk_ids[i, j])]
                        a = torch.nn.functional.silu(d["gate"] @ xi) * (d["up"] @ xi)
                        ref[i] += topk_w[i, j].float() * (d["down"] @ a)
                self.assertLess(_rel_rms(got, ref.half()), 2e-2)


if __name__ == "__main__":
    unittest.main()
