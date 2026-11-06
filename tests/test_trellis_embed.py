"""The trellis embedding's decode, against exllamav3's own.

**Bit-exact against the kernel is the claim.** A served model that disagreed with
native exllamav3 in the last bit would be a different model, and nothing above
this file could tell. `ngram_codec.dequant_rows` is the CPU reference and runs
anywhere; `ngram_dequant` needs a GPU, and that comparison skips without one.
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_exl3_plugin import format, trellis_embed

ROWS, HIDDEN = 64, 384  # two groups, the second padded


def table(k: int, device="cpu"):
    """A packed table of random symbols and plausible scales, with real sign vectors."""
    g = torch.Generator().manual_seed(k)
    groups = (HIDDEN + 255) // 256
    words = 1 + 16 * k
    packed = torch.randint(
        -(2**15), 2**15, (ROWS * groups, words), generator=g, dtype=torch.int32
    ).to(torch.int16)
    scales = (torch.rand(ROWS * groups, generator=g) + 0.1).to(torch.float16)
    packed[:, 0] = scales.view(torch.int16)
    signs = (torch.randint(0, 2, (groups, 256), generator=g) * 2 - 1).to(torch.float16)
    return packed.view(ROWS, -1).to(device), signs.to(device)


class CodecTest(unittest.TestCase):
    def test_the_codebook_is_exllamav3s(self):
        from exllamav3.modules.quant.exl3_lib.ngram_codec import mul1_codebook

        self.assertTrue(torch.equal(trellis_embed.codebook("cpu"), mul1_codebook("cpu")))

    def test_decode_equals_the_cpu_reference(self):
        from exllamav3.modules.quant.exl3_lib.ngram_codec import (
            dequant_rows,
            mul1_codebook,
        )

        for k in range(1, 9):
            with self.subTest(k=k):
                packed, signs = table(k)
                n = packed.shape[0]
                want = dequant_rows(
                    packed.view(n * signs.shape[0], -1),
                    k,
                    mul1_codebook("cpu"),
                    signs=signs.repeat(n, 1),
                ).half().view(n, -1)[:, :HIDDEN]
                got = trellis_embed.decode(
                    packed, signs, trellis_embed.codebook("cpu"), HIDDEN, torch.half
                )
                self.assertTrue(torch.equal(got, want))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_decode_equals_the_kernel(self):
        from exllamav3.ext import exllamav3_ext as ext

        for k in range(1, 9):
            with self.subTest(k=k):
                packed, signs = table(k, "cuda")
                n, groups = packed.shape[0], signs.shape[0]
                out = torch.empty(
                    (n * groups, 256), dtype=torch.half, device="cuda"
                )
                ext.ngram_dequant(packed.view(n * groups, -1), k, None, signs, out, True)
                got = trellis_embed.decode(
                    packed, signs, trellis_embed.codebook("cuda"), HIDDEN, torch.half
                )
                self.assertTrue(torch.equal(got, out.view(n, -1)[:, :HIDDEN]))

    def test_a_zero_row_decodes_to_zero(self):
        """vLLM pads the vocabulary with zero rows, and the gather still indexes them."""
        packed, signs = table(4)
        packed[3] = 0
        got = trellis_embed.decode(
            packed, signs, trellis_embed.codebook("cpu"), HIDDEN, torch.half
        )
        self.assertEqual(got[3].abs().max().item(), 0)

    def test_gather_preserves_shape(self):
        packed, signs = table(4)
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        book = trellis_embed.codebook("cpu")
        got = trellis_embed.gather(ids, packed, signs, book, HIDDEN, torch.half)
        self.assertEqual(got.shape, (2, 3, HIDDEN))
        self.assertTrue(
            torch.equal(
                got[1, 2],
                trellis_embed.decode(packed[6:7], signs, book, HIDDEN, torch.half)[0],
            )
        )

    def test_a_row_that_is_not_whole_rings_is_refused(self):
        with self.assertRaises(format.EXL3FormatError):
            trellis_embed.depth_from_shape(100, 3)
        self.assertEqual(trellis_embed.depth_from_shape(2 * 65, 2), 4)
        with self.assertRaises(format.EXL3FormatError):
            trellis_embed.depth_from_shape(2 * (1 + 16 * 9), 2)


if __name__ == "__main__":
    unittest.main()
