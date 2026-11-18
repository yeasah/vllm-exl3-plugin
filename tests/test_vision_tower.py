"""EXL3 vision towers: exllamav3's q/k/v split, and the replicated-encoder default.

Shapes from `turboderp/Qwen3.8-Flash-Next-exl3` @2.05bpw: each vision block stores quantized
`attn.{q,k,v}_proj` *and* the converter's leftover dense `attn.qkv.{weight,bias}`, while vLLM's tower has a single
fused `attn.qkv`.
"""
from __future__ import annotations

import types
import unittest

from vllm_exl3_plugin import format

INDEX = {
    f"visual.blocks.0.attn.{name}": "a"
    for name in ("q_proj.trellis", "q_proj.suh", "q_proj.svh", "q_proj.mul1", "q_proj.bias",
                 "k_proj.trellis", "k_proj.suh", "k_proj.svh", "k_proj.mul1", "k_proj.bias",
                 "v_proj.trellis", "v_proj.suh", "v_proj.svh", "v_proj.mul1", "v_proj.bias",
                 "proj.trellis", "proj.suh", "proj.svh", "proj.mul1", "proj.bias",
                 "qkv.weight", "qkv.bias")
}


def _config(index=INDEX):
    from vllm_exl3_plugin.quantization.config import EXL3Config

    c = EXL3Config(bits=2.05, head_bits=6, codebook="mul1")
    c.quantized_modules = format.quantized_module_keys(index)
    c.tie_word_embeddings = False
    return c


def _tower_mapper(config):
    """The plugin's rules composed with the tower's own, in the order AutoWeightsLoader applies them."""
    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer

    return Qwen3_VisionTransformer.hf_to_vllm_mapper | config.get_cache_scale_mapper()


class TestVisionQKVSplit(unittest.TestCase):
    def test_fused_qkv_is_quantized(self):
        c = _config()
        self.assertTrue(c.is_quantized("visual.blocks.0.attn.qkv"))
        self.assertTrue(c.is_quantized("visual.blocks.0.attn.proj"))

    def test_split_tensors_stack_into_qkv(self):
        mapper = _tower_mapper(_config())
        for part in "qkv":
            for suffix in ("trellis", "mul1", "bias"):
                self.assertEqual(
                    mapper._map_name_with_shard(f"visual.blocks.0.attn.{part}_proj.{suffix}"),
                    (f"visual.blocks.0.attn.qkv.{suffix}", part))

    def test_leftover_dense_qkv_is_dropped(self):
        mapper = _tower_mapper(_config())
        self.assertIsNone(mapper._map_name("visual.blocks.0.attn.qkv.weight"))
        self.assertIsNone(mapper._map_name("visual.blocks.0.attn.qkv.bias"))

    def test_dense_tower_keeps_its_qkv(self):
        """A tower stored dense (our pipeline's policy) has no split, so its fused tensors are the only copy."""
        mapper = _tower_mapper(_config({"visual.blocks.0.attn.qkv.weight": "a", "lm_head.trellis": "b"}))
        self.assertEqual(mapper._map_name("visual.blocks.0.attn.qkv.weight"), "visual.blocks.0.attn.qkv.weight")

    def test_language_model_attention_is_untouched(self):
        mapper = _config().get_cache_scale_mapper()
        name = "language_model.model.layers.3.self_attn.q_proj.trellis"
        self.assertEqual(mapper._map_name(name), name)


class TestEncoderDPDefault(unittest.TestCase):
    def _args(self, *, quant="exl3", tp=2, mode="weights", lm_only=False, supports=True):
        mm = types.SimpleNamespace(mm_encoder_tp_mode=mode, language_model_only=lm_only)
        mc = types.SimpleNamespace(quantization=quant, multimodal_config=mm,
                                   _model_info=types.SimpleNamespace(supports_multimodal_encoder_tp_data=supports))
        return mc, types.SimpleNamespace(tensor_parallel_size=tp)

    def test_exl3_multimodal_tp_replicates(self):
        from vllm_exl3_plugin.plugin import _wants_encoder_dp

        self.assertTrue(_wants_encoder_dp(*self._args()))

    def test_left_alone_otherwise(self):
        from vllm_exl3_plugin.plugin import _wants_encoder_dp

        for kw in ({"quant": "fp8"}, {"tp": 1}, {"mode": "data"}, {"lm_only": True}, {"supports": False}):
            self.assertFalse(_wants_encoder_dp(*self._args(**kw)), kw)
        mc, pc = self._args()
        mc.multimodal_config = None
        self.assertFalse(_wants_encoder_dp(mc, pc))


if __name__ == "__main__":
    unittest.main()
