"""The token embedding's table on host (`EXL3_EMBED_HOST`): pinned, behind a UVA view, rows padded.

**Bit-exact against the same table in VRAM is the claim.** Only where the bytes live changes, so a lookup that
differed in any bit would be a bug in the padding or the trim, not a numerical effect. Needs a GPU: a UVA view is
a CUDA tensor.
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_exl3_plugin import blockq, cpu_offload, trellis_embed

from test_trellis_embed import HIDDEN, ROWS, table


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class EmbedHostTest(unittest.TestCase):
    def test_trellis_rows_are_padded_and_gather_exactly(self):
        """Two groups at K=4 are 260-byte rows: not 64-aligned, so the host copy is padded (the 27B's case)."""
        packed, signs = table(4)
        row_words = packed.shape[1]
        hosted = cpu_offload.host_resident(packed, "test")
        self.assertEqual(hosted.device.type, "cuda")
        self.assertEqual(hosted.shape[1] * hosted.element_size() % cpu_offload.HOST_ROW_ALIGN, 0)
        self.assertGreater(hosted.shape[1], row_words)
        self.assertTrue(torch.equal(hosted[:, :row_words].cpu(), packed))

        signs, book = signs.cuda(), trellis_embed.codebook("cuda")
        ids = torch.randint(0, ROWS, (3, 17), device="cuda")
        want = trellis_embed.gather(ids, packed.cuda(), signs, book, HIDDEN, torch.bfloat16)
        got = trellis_embed.gather(ids, hosted, signs, book, HIDDEN, torch.bfloat16, row_words)
        self.assertTrue(torch.equal(got, want))

    def test_blockq_gathers_exactly(self):
        stored = blockq.encode(torch.randn(ROWS, 512) * 0.02)
        names = ("bq_q", "bq_s", "bq_r")
        widths = tuple(stored[n].shape[-1] for n in names)
        hosted = [cpu_offload.host_resident(stored[n], n) for n in names]
        self.assertEqual(hosted[2].shape, stored["bq_r"].shape)  # a narrow row is never padded
        ids = torch.randint(0, ROWS, (5, 9), device="cuda")
        want = blockq.gather(ids, *(stored[n].cuda() for n in names), torch.bfloat16)
        got = blockq.gather(ids, *hosted, torch.bfloat16, widths)
        self.assertTrue(torch.equal(got, want))

    def test_an_aligned_row_is_left_alone(self):
        t = torch.randint(0, 255, (ROWS, 128), dtype=torch.uint8)
        self.assertEqual(cpu_offload.host_resident(t, "test").shape, t.shape)


if __name__ == "__main__":
    unittest.main()
