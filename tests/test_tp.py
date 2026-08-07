"""Tensor-parallel sharding: the math, proven offline.

This machine has one GPU, so vLLM's distributed path cannot run here at all.
What *can* be established without a second device is the part that is actually
novel and risky: whether cutting an EXL3 weight across ranks reconstructs the
unsharded result. Everything below simulates the ranks sequentially on one
device and combines their partial outputs the way vLLM would -- concatenation
for a column split, summation (vLLM's all-reduce) for a row split.

What this does NOT cover, and what still needs real multi-GPU hardware:
vLLM's loader driving these slices at TP>1, NCCL collectives, and the
exllamav3 autotune cache being written by several worker processes at once.
"""

from __future__ import annotations

import unittest

from vllm_exl3_plugin import format

try:
    import torch

    HAVE_CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    HAVE_CUDA = False

try:
    import exllamav3_ext  # noqa: F401

    HAVE_EXT = True
except ImportError:  # pragma: no cover
    HAVE_EXT = False

requires_gpu = unittest.skipUnless(
    HAVE_CUDA and HAVE_EXT, "needs CUDA and exllamav3's compiled extension"
)

REPO, REVISION = "turboderp/Llama-3.2-1B-Instruct-exl3", "3.0bpw"
# in=2048 out=2048, so both dimensions divide by 128 at tp 2 and 4.
SQUARE = "model.layers.0.self_attn.q_proj"


class TestShardBounds(unittest.TestCase):
    """Pure arithmetic -- runs anywhere."""

    def test_even_split(self):
        self.assertEqual(format.shard_bounds(2048, 0, 2, "q"), (0, 1024))
        self.assertEqual(format.shard_bounds(2048, 1, 2, "q"), (1024, 2048))

    def test_tp1_is_whole_tensor(self):
        self.assertEqual(format.shard_bounds(512, 0, 1, "k"), (0, 512))

    def test_rejects_sub_hadamard_shard(self):
        # 512 channels over 8 ranks is 64 each: half a Hadamard block.
        with self.assertRaises(format.EXL3FormatError):
            format.shard_bounds(512, 0, 8, "k_proj")

    def test_rejects_uneven(self):
        with self.assertRaises(format.EXL3FormatError):
            format.shard_bounds(2048, 0, 3, "q_proj")


@requires_gpu
class TestShardedMathMatches(unittest.TestCase):
    """Sharded execution must reproduce the unsharded result."""

    def _layer(self):
        from tests.remote_tensors import fetch_module_tensors

        try:
            return fetch_module_tensors(REPO, REVISION, SQUARE)
        except OSError as e:
            self.skipTest(f"could not fetch {REPO}@{REVISION}: {e}")

    def test_column_split_concatenates(self):
        """Output split: each rank owns a slice of the output channels."""
        from vllm_exl3_plugin import ops, tp

        t = self._layer()
        in_f, out_f = format.dims_from_trellis_shape(t["trellis"].shape)
        mcg, mul1 = "mcg" in t, "mul1" in t
        torch.manual_seed(0)
        x = torch.randn((8, in_f), dtype=torch.half, device="cuda:0") * 0.1
        whole = ops.exl3_mm(x, t["trellis"], t["suh"], t["svh"], mcg, mul1)

        for tp_size in (2, 4):
            with self.subTest(tp_size=tp_size):
                parts = []
                for rank in range(tp_size):
                    first, last = format.shard_bounds(out_f, rank, tp_size, "out")
                    parts.append(
                        ops.exl3_mm(
                            x,
                            tp.shard_column("trellis", t["trellis"], first, last),
                            tp.shard_column("suh", t["suh"], first, last),
                            tp.shard_column("svh", t["svh"], first, last),
                            mcg,
                            mul1,
                        )
                    )
                got = torch.cat(parts, dim=-1)
                # Not bit-identical, despite each output channel being computed
                # by exactly one rank from the same inputs: a narrower `n` makes
                # the kernel autotuner pick a different tile shape, which
                # changes the order of the fp16 accumulation over k. The
                # measured difference is ~4e-4 relative, i.e. rounding.
                torch.testing.assert_close(got, whole, rtol=2e-2, atol=5e-3)

    def test_row_split_sums(self):
        """Input split: ranks compute partial sums that vLLM all-reduces."""
        from vllm_exl3_plugin import ops, tp

        t = self._layer()
        in_f, _ = format.dims_from_trellis_shape(t["trellis"].shape)
        mcg, mul1 = "mcg" in t, "mul1" in t
        torch.manual_seed(0)
        x = torch.randn((8, in_f), dtype=torch.half, device="cuda:0") * 0.1
        whole = ops.exl3_mm(x, t["trellis"], t["suh"], t["svh"], mcg, mul1)

        for tp_size in (2, 4):
            with self.subTest(tp_size=tp_size):
                total = None
                for rank in range(tp_size):
                    first, last = format.shard_bounds(in_f, rank, tp_size, "in")
                    part = ops.exl3_mm(
                        x[:, first:last].contiguous(),
                        tp.shard_row("trellis", t["trellis"], first, last),
                        tp.shard_row("suh", t["suh"], first, last),
                        tp.shard_row("svh", t["svh"], first, last),
                        mcg,
                        mul1,
                    )
                    total = part.float() if total is None else total + part.float()
                # Summing in a different order than the unsharded kernel, so
                # fp16 rounding differs; the transform itself is exact.
                torch.testing.assert_close(
                    total.half(), whole, rtol=2e-2, atol=5e-3
                )


@requires_gpu
class TestHadamardRuleIsReal(unittest.TestCase):
    """The 128 rule is a correctness constraint, not a convention.

    `shard_bounds` refuses a split finer than a Hadamard block. It would be
    easy to assume that guard is merely cautious -- the trellis slices cleanly
    at any multiple of 16, and nothing errors. This test forces the unsafe
    split anyway and shows the answer really is wrong, which is what makes the
    guard worth keeping.
    """

    def test_sub_block_split_corrupts(self):
        from tests.remote_tensors import fetch_module_tensors
        from vllm_exl3_plugin import ops, tp

        try:
            t = fetch_module_tensors(REPO, REVISION, SQUARE)
        except OSError as e:
            self.skipTest(f"could not fetch: {e}")

        in_f, out_f = format.dims_from_trellis_shape(t["trellis"].shape)
        mcg, mul1 = "mcg" in t, "mul1" in t
        torch.manual_seed(0)
        x = torch.randn((8, in_f), dtype=torch.half, device="cuda:0") * 0.1
        whole = ops.exl3_mm(x, t["trellis"], t["suh"], t["svh"], mcg, mul1)

        # A 64-wide row split: tile-aligned (multiple of 16) but half a
        # Hadamard block, so `shard_bounds` would reject it.
        with self.assertRaises(format.EXL3FormatError):
            format.shard_bounds(in_f, 0, in_f // 64, "in")

        step = 64
        total = None
        for first in range(0, in_f, step):
            last = first + step
            part = ops.exl3_mm(
                x[:, first:last].contiguous(),
                tp.shard_row("trellis", t["trellis"], first, last),
                tp.shard_row("suh", t["suh"], first, last),
                tp.shard_row("svh", t["svh"], first, last),
                mcg,
                mul1,
            )
            total = part.float() if total is None else total + part.float()

        err = (total.half() - whole).abs().max().item()
        scale = whole.abs().max().item()
        # Not a rounding difference -- a different computation.
        self.assertGreater(
            err / scale, 0.1,
            "sub-block split unexpectedly agreed; the 128 rule may be "
            "mis-stated, which would change how Phase 2 must shard",
        )


if __name__ == "__main__":
    unittest.main()


MOE_REPO, MOE_REVISION = "turboderp/Laguna-XS-2.1-exl3", "main"
#: Stored intermediate is 512, so it splits cleanly at tp 2 and 4 but not 8.
MOE_EXPERTS = 4


@requires_gpu
class TestMoEShardedMathMatches(unittest.TestCase):
    """Routed experts sharded on the intermediate dimension.

    Same argument as the dense case, one level in: gate/up take a column split
    and down takes a row split, so each rank computes a partial sum over its
    slice of the intermediate and vLLM's all-reduce adds them. Simulated here by
    running every rank sequentially on the one device and summing.
    """

    def _experts(self):
        """Per-expert tensors, from the local HF cache if it has them.

        A 256-expert checkpoint is usually already on disk when anyone is
        working on this, and range-fetching 36 tensors over HTTPS is both slow
        and a network dependency in a test that otherwise needs none.
        """
        local = self._from_cache()
        if local is not None:
            return local
        from tests.remote_tensors import fetch_module_tensors

        out = []
        try:
            for e in range(MOE_EXPERTS):
                base = f"model.layers.5.mlp.experts.{e}"
                out.append({
                    p: fetch_module_tensors(MOE_REPO, MOE_REVISION, f"{base}.{p}_proj")
                    for p in ("gate", "up", "down")
                })
        except OSError as e:
            self.skipTest(f"could not fetch {MOE_REPO}: {e}")
        return out

    @staticmethod
    def _from_cache():
        import glob
        import json
        import os

        snaps = glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--turboderp--Laguna-XS-2.1-exl3/"
            "snapshots/*/model.safetensors.index.json"))
        if not snaps:
            return None
        try:
            from safetensors import safe_open
        except ImportError:
            return None
        d = os.path.dirname(snaps[0])
        idx = json.load(open(snaps[0]))["weight_map"]
        handles = {}

        def get(key):
            f = idx[key]
            if f not in handles:
                handles[f] = safe_open(os.path.join(d, f), framework="pt",
                                       device="cuda:0")
            return handles[f].get_tensor(key)

        out = []
        for e in range(MOE_EXPERTS):
            base = f"model.layers.5.mlp.experts.{e}"
            proj = {}
            for p in ("gate", "up", "down"):
                names = [k.rsplit(".", 1)[1] for k in idx
                         if k.startswith(f"{base}.{p}_proj.")]
                proj[p] = {n: get(f"{base}.{p}_proj.{n}") for n in names}
            out.append(proj)
        return out

    @staticmethod
    def _run(experts, x, topk_ids, topk_weights):
        """One rank: pointer tables over whatever slices it was given."""
        import torch

        from vllm_exl3_plugin import format, ops

        def ptrs(proj, name):
            ts = [e[proj][name] for e in experts]
            return torch.tensor([t.data_ptr() for t in ts], dtype=torch.long,
                                device=ts[0].device)

        gate0 = experts[0]["gate"]["trellis"]
        bits = format.bits_from_trellis_shape(gate0.shape)
        interm = gate0.shape[1] * format.TILE
        return ops.exl3_moe_mm(
            x, topk_ids, topk_weights,
            ptrs("gate", "trellis"), ptrs("gate", "suh"), ptrs("gate", "svh"),
            ptrs("up", "trellis"), ptrs("up", "suh"), ptrs("up", "svh"),
            ptrs("down", "trellis"), ptrs("down", "suh"), ptrs("down", "svh"),
            bits,
            format.bits_from_trellis_shape(experts[0]["down"]["trellis"].shape),
            interm, len(experts), False, True, "silu",
        )

    def test_intermediate_split_sums(self):
        import torch

        from vllm_exl3_plugin.quantization.fused_moe import _tp_shard

        experts = self._experts()
        hidden = experts[0]["gate"]["suh"].shape[0]
        torch.manual_seed(0)
        x = torch.randn((8, hidden), dtype=torch.half, device="cuda:0") * 0.1
        top_k = 2
        topk_ids = torch.randint(0, MOE_EXPERTS, (8, top_k), device="cuda:0")
        topk_weights = torch.rand((8, top_k), dtype=torch.float, device="cuda:0")

        whole = self._run(experts, x, topk_ids, topk_weights)

        for tp_size in (2, 4):
            with self.subTest(tp_size=tp_size):
                total = torch.zeros_like(whole, dtype=torch.float32)
                for rank in range(tp_size):
                    sharded = [
                        {
                            proj: {
                                n: _tp_shard(n, t, shard_id, rank, tp_size)
                                for n, t in e[proj].items()
                            }
                            for proj, shard_id in
                            (("gate", "w1"), ("up", "w3"), ("down", "w2"))
                        }
                        for e in experts
                    ]
                    total += self._run(sharded, x, topk_ids, topk_weights).float()
                # Same tolerance rationale as the dense case: a narrower
                # intermediate makes the kernel accumulate in a different order.
                torch.testing.assert_close(
                    total, whole.float(), rtol=2e-2, atol=5e-3
                )

    def test_rejects_sub_hadamard_moe_split(self):
        """512 // 8 = 64 cuts a Hadamard block, and must be refused."""
        with self.assertRaises(format.EXL3FormatError):
            format.shard_bounds(512, 0, 8, "MoE intermediate")
