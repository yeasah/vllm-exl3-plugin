"""Tests for the pure-arithmetic half of the plugin.

These run anywhere -- no torch, no vLLM, no GPU -- which is the point: the shape
and naming rules are where a checkpoint-format misunderstanding turns into
silently wrong output, and they should be pinned down before there is hardware
to test on.

Fixtures are real shapes, read out of the safetensors headers of
`turboderp/Llama-3.2-1B-Instruct-exl3` at 3.0bpw and 3.5bpw.
"""

import unittest

from vllm_exl3_plugin import format

# (name, trellis shape, expected in, expected out, expected bits)
LLAMA_1B_3BPW = [
    ("self_attn.q_proj", [128, 128, 48], 2048, 2048, 3),
    ("self_attn.k_proj", [128, 32, 48], 2048, 512, 3),
    ("self_attn.v_proj", [128, 32, 48], 2048, 512, 3),
    ("self_attn.o_proj", [128, 128, 48], 2048, 2048, 3),
    ("mlp.gate_proj", [128, 512, 48], 2048, 8192, 3),
    ("mlp.up_proj", [128, 512, 48], 2048, 8192, 3),
    ("mlp.down_proj", [512, 128, 48], 8192, 2048, 3),
    ("lm_head", [128, 8016, 96], 2048, 128256, 6),
]


class TestTrellisShapes(unittest.TestCase):
    def test_real_checkpoint_shapes(self):
        for name, shape, in_f, out_f, bits in LLAMA_1B_3BPW:
            with self.subTest(name):
                self.assertEqual(format.bits_from_trellis_shape(shape), bits)
                self.assertEqual(format.dims_from_trellis_shape(shape), (in_f, out_f))
                self.assertEqual(
                    list(format.trellis_shape(in_f, out_f, bits)), list(shape)
                )

    def test_roundtrip_all_bit_widths(self):
        for bits in range(format.MIN_BITS, format.MAX_BITS + 1):
            shape = format.trellis_shape(4096, 1024, bits)
            self.assertEqual(format.bits_from_trellis_shape(shape), bits)
            self.assertEqual(format.dims_from_trellis_shape(shape), (4096, 1024))

    def test_rejects_wrong_rank(self):
        with self.assertRaises(format.EXL3FormatError):
            format.bits_from_trellis_shape([128, 128])

    def test_rejects_implausible_bit_width(self):
        # 16*9 = 144 -> 9 bits, outside the 1..8 EXL3 range.
        with self.assertRaises(format.EXL3FormatError):
            format.bits_from_trellis_shape([128, 128, 144])

    def test_rejects_unaligned_dims(self):
        with self.assertRaises(format.EXL3FormatError):
            format.trellis_shape(2048, 100, 4)


class TestMixedBitWidths(unittest.TestCase):
    """A merged linear cannot assume one bit width across its shards."""

    def test_qkv_may_disagree(self):
        # Observed in Llama-3.2-1B-Instruct-exl3 @ 3.5bpw, every layer.
        q = format.trellis_shape(2048, 2048, 4)
        k = format.trellis_shape(2048, 512, 5)
        self.assertNotEqual(q[2], k[2])
        self.assertEqual(format.bits_from_trellis_shape(q), 4)
        self.assertEqual(format.bits_from_trellis_shape(k), 5)


class TestPadding(unittest.TestCase):
    def test_pad_dim(self):
        self.assertEqual(format.pad_dim(2048), 2048)
        self.assertEqual(format.pad_dim(2880), 2944)  # gpt-oss hidden size
        self.assertEqual(format.pad_dim(1), 128)
        self.assertEqual(format.pad_dim(128256), 128256)


class TestTPAlignment(unittest.TestCase):
    def test_tp1_always_ok(self):
        format.check_tp_split(512, 1, "k_proj")

    def test_llama_1b_kv_breaks_at_tp8(self):
        # 8 KV heads x head_dim 64 = 512 output channels. TP=8 gives 64 per
        # rank, which cuts a 128-wide Hadamard block in half.
        format.check_tp_split(512, 2, "k_proj")
        format.check_tp_split(512, 4, "k_proj")
        with self.assertRaises(format.EXL3FormatError):
            format.check_tp_split(512, 8, "k_proj")

    def test_uneven_split_rejected(self):
        with self.assertRaises(format.EXL3FormatError):
            format.check_tp_split(2048, 3, "q_proj")

    def test_alignment_predicate(self):
        self.assertTrue(format.is_had_aligned(256))
        self.assertFalse(format.is_had_aligned(64))


class TestTensorNames(unittest.TestCase):
    def test_module_key_extraction(self):
        self.assertEqual(
            format.module_key_for_tensor("model.layers.0.self_attn.q_proj.trellis"),
            "model.layers.0.self_attn.q_proj",
        )
        self.assertEqual(
            format.module_key_for_tensor("model.layers.0.mlp.down_proj.suh"),
            "model.layers.0.mlp.down_proj",
        )

    def test_non_exl3_tensors_ignored(self):
        for name in (
            "model.layers.0.input_layernorm.weight",
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.bias",
        ):
            self.assertIsNone(format.module_key_for_tensor(name))

    def test_quantized_module_keys(self):
        names = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.trellis",
            "model.layers.0.self_attn.q_proj.suh",
            "model.layers.0.self_attn.q_proj.svh",
            "model.layers.0.input_layernorm.weight",
        ]
        self.assertEqual(
            format.quantized_module_keys(names),
            {"model.layers.0.self_attn.q_proj"},
        )


class TestIntermDivisor(unittest.TestCase):
    """Recovering exllamav3's `interm_div` from the stored weights.

    Ratios below are measured `mean|suh_gate| / mean|suh_up|` over real
    checkpoints: gemma-4-26B-A4B (no divisor) and Laguna-XS-2.1 (128).
    """

    GEMMA_RATIOS = [0.9731, 0.9951, 0.9943, 0.9365]
    LAGUNA_RATIOS = [127.584, 127.473, 129.705, 119.655]

    def test_no_divisor_is_recognised(self):
        for ratio in self.GEMMA_RATIOS:
            self.assertEqual(format.infer_interm_divisor(ratio), 1.0, ratio)

    def test_laguna_divisor_recovered_exactly(self):
        # Every per-expert ratio has to land on 128 even at the 119.7 extreme,
        # because the value is applied as a multiplier on every routed expert.
        for ratio in self.LAGUNA_RATIOS:
            self.assertEqual(format.infer_interm_divisor(ratio), 128.0, ratio)

    def test_exact_powers_of_two(self):
        for k in range(1, 10):
            self.assertEqual(format.infer_interm_divisor(2.0**k), 2.0**k)

    def test_rejects_non_power_of_two(self):
        # 3x and 100x are both far enough from a power of two that guessing
        # would be a silent factor-of-N error in every routed expert.
        for ratio in (3.0, 100.0, 20.0):
            with self.assertRaises(format.EXL3FormatError):
                format.infer_interm_divisor(ratio)

    def test_rejects_up_larger_than_gate(self):
        # exllamav3 only ever scales the up projection *down*.
        with self.assertRaises(format.EXL3FormatError):
            format.infer_interm_divisor(0.5)

    def test_rejects_degenerate_ratios(self):
        for ratio in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaises(format.EXL3FormatError):
                format.infer_interm_divisor(ratio)


class TestFusedShardBounds(unittest.TestCase):
    """Composing a fused-shard split with a tensor-parallel one.

    Qwen3.5's `in_proj_qkv` carries shards 0, 1 and 2 of the merged
    `in_proj_qkvz`, so under TP the stored tensor is cut twice.
    """

    # Per-rank widths, as vLLM supplies them.
    SIZES = [1024, 512, 512, 256]

    def test_tp1_spans_are_consecutive(self):
        got = format.fused_shard_bounds(self.SIZES, [0, 1, 2])
        self.assertEqual(got, [(0, 0, 1024), (1, 1024, 1536), (2, 1536, 2048)])

    def test_rank_slices_tile_the_unsharded_shard(self):
        """The union of every rank's slice must be exactly the TP=1 span."""
        for tp_size in (2, 4):
            with self.subTest(tp_size=tp_size):
                # At TP=N vLLM reports per-rank sizes, i.e. 1/N of TP=1's.
                sizes = [s // tp_size for s in self.SIZES]
                whole = format.fused_shard_bounds(self.SIZES, [0, 1, 2])
                per_shard = {i: [] for i, _, _ in whole}
                for rank in range(tp_size):
                    for i, lo, hi in format.fused_shard_bounds(
                        sizes, [0, 1, 2], rank, tp_size
                    ):
                        per_shard[i].append((lo, hi))
                for (i, lo, hi) in whole:
                    covered = sorted(per_shard[i])
                    # contiguous, no gaps or overlaps, and spanning the whole
                    self.assertEqual(covered[0][0], lo)
                    self.assertEqual(covered[-1][1], hi)
                    for (_, a), (b, _) in zip(covered, covered[1:]):
                        self.assertEqual(a, b)

    def test_every_boundary_is_hadamard_aligned(self):
        for tp_size in (1, 2, 4):
            sizes = [s // tp_size for s in self.SIZES]
            for rank in range(tp_size):
                for _, lo, hi in format.fused_shard_bounds(
                    sizes, [0, 1, 2], rank, tp_size
                ):
                    self.assertEqual(lo % format.HAD_BLOCK, 0)
                    self.assertEqual(hi % format.HAD_BLOCK, 0)

    def test_rejects_shard_that_cuts_a_hadamard_block(self):
        # 512 across 8 ranks is 64 per rank, half a Hadamard block.
        with self.assertRaises(format.EXL3FormatError):
            format.fused_shard_bounds([64, 64, 64], [0, 1, 2], 0, 8)

    def test_rejects_unknown_shard_index(self):
        with self.assertRaises(format.EXL3FormatError):
            format.fused_shard_bounds([1024, 512], [0, 1, 2])

    def test_rejects_missing_partition_sizes(self):
        with self.assertRaises(format.EXL3FormatError):
            format.fused_shard_bounds([], [0])


if __name__ == "__main__":
    unittest.main()
