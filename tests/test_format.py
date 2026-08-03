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


if __name__ == "__main__":
    unittest.main()
