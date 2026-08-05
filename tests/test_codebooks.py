"""The `mcg` and `mul1` codebooks, and the `quantization_config.json` path.

Every checkpoint the earlier tests use predates EXL3's newer procedural
codebooks, so `reconstruct` had only ever been exercised with both flags false
and `stored_tensor_names()` only in its three-tensor form. The repos that do use
them are large -- and under Phase 0's dequantize-at-load they would need ~24 GB
of VRAM -- so the tensor-level checks here pull a single layer over HTTP range
requests instead of downloading a model.

The JSON checks need no GPU at all.
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

# (label, repo, revision, module key)
CODEBOOK_CASES = [
    (
        "mcg",
        "turboderp/MiniCPM5-1B-exl3",
        "3.00bpw",
        "model.layers.0.self_attn.q_proj",
    ),
    (
        "mul1",
        "turboderp/gemma-4-12B-it-exl3",
        "3.00bpw_mul1",
        "model.language_model.layers.0.self_attn.q_proj",
    ),
]


@requires_gpu
class TestCodebooks(unittest.TestCase):
    """`ops.dense_weight` must match exllamav3 under every codebook."""

    def test_dequant_matches_exllamav3(self):
        from exllamav3.modules.quant.exl3 import LinearEXL3

        from tests.remote_tensors import fetch_module_tensors
        from vllm_exl3_plugin import ops

        for label, repo, revision, key in CODEBOOK_CASES:
            with self.subTest(codebook=label):
                try:
                    t = fetch_module_tensors(repo, revision, key)
                except OSError as e:  # network flake, not a code failure
                    self.skipTest(f"could not fetch {repo}@{revision}: {e}")

                self.assertIn(label, t, f"expected a '{label}' tensor in {repo}")
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
                theirs = LinearEXL3(
                    config=None,
                    in_features=in_f,
                    out_features=out_f,
                    suh=t["suh"],
                    svh=t["svh"],
                    trellis=t["trellis"],
                    mcg=t.get("mcg"),
                    mul1=t.get("mul1"),
                ).get_weight_tensor()

                torch.testing.assert_close(ours, theirs.t(), rtol=0, atol=0)

    def test_codebook_flag_actually_changes_decode(self):
        """Guard against the flags being silently ignored.

        `reconstruct` takes mcg/mul1 as booleans -- the multiplier constants are
        compiled into the kernel -- so passing the wrong one is not an error,
        just wrong numbers. If decoding the same trellis both ways produced the
        same weight, these tests would prove nothing.
        """
        from tests.remote_tensors import fetch_module_tensors
        from vllm_exl3_plugin import ops

        label, repo, revision, key = CODEBOOK_CASES[0]
        try:
            t = fetch_module_tensors(repo, revision, key)
        except OSError as e:
            self.skipTest(f"could not fetch {repo}@{revision}: {e}")

        bits = format.bits_from_trellis_shape(t["trellis"].shape)
        with_mcg = ops.reconstruct(t["trellis"], bits, True, False)
        without = ops.reconstruct(t["trellis"], bits, False, False)
        self.assertFalse(torch.equal(with_mcg, without))


class TestStoredTensorNames(unittest.TestCase):
    """The parameter set registered per layer must match the checkpoint exactly.

    vLLM treats a registered-but-never-loaded parameter as fatal, so an extra
    name is as bad as a missing one.
    """

    def _config(self, **kw):
        from vllm_exl3_plugin.quantization.config import EXL3Config

        return EXL3Config.from_config({"quant_method": "exl3", **kw})

    def test_default_codebook(self):
        self.assertEqual(
            self._config(bits=3.0).stored_tensor_names(), ("trellis", "suh", "svh")
        )

    def test_mcg(self):
        self.assertEqual(
            self._config(bits=3.0, codebook="mcg").stored_tensor_names(),
            ("trellis", "suh", "svh", "mcg"),
        )

    def test_mul1(self):
        self.assertEqual(
            self._config(bits=3.0, codebook="mul1").stored_tensor_names(),
            ("trellis", "suh", "svh", "mul1"),
        )

    def test_unknown_codebook_rejected(self):
        with self.assertRaises(format.EXL3FormatError):
            self._config(bits=3.0, codebook="something-new")


class TestQuantizedHeadDetection(unittest.TestCase):
    """Whether `lm_head` gets the EXL3 method.

    Wrong in either direction is fatal rather than degraded: registering
    parameters the checkpoint never fills makes vLLM's loader reject the model,
    and failing to register them leaves `lm_head.trellis` unclaimed.
    """

    def _config_for(self, repo, revision, **kw):
        from vllm_exl3_plugin.quantization.config import EXL3Config

        cfg = EXL3Config.from_config({"quant_method": "exl3", "bits": 3.0, **kw})
        try:
            cfg._load_tensor_storage(repo, revision)
        except Exception as e:  # pragma: no cover - network
            self.skipTest(f"could not fetch config for {repo}: {e}")
        return cfg

    def test_untied_quantized_head_detected(self):
        cfg = self._config_for("turboderp/MiniCPM5-1B-exl3", "3.00bpw")
        if cfg.tensor_storage is None:
            self.skipTest("quantization_config.json unavailable")
        self.assertEqual(cfg.codebook, "mcg")
        cfg.tie_word_embeddings = False
        self.assertTrue(cfg.head_is_quantized())

    def test_tied_head_is_never_quantized(self):
        """vLLM constructs a ParallelLMHead for tied models and then skips
        every lm_head.* weight, so claiming it would strand our parameters."""
        cfg = self._config_for("turboderp/MiniCPM5-1B-exl3", "3.00bpw")
        if cfg.tensor_storage is None:
            self.skipTest("quantization_config.json unavailable")
        cfg.tie_word_embeddings = True
        self.assertFalse(cfg.head_is_quantized())

    def test_head_bits_is_the_fallback_signal(self):
        """Repos with no quantization_config.json still declare head_bits."""
        from vllm_exl3_plugin.quantization.config import EXL3Config

        untied = EXL3Config.from_config(
            {"quant_method": "exl3", "bits": 3.0, "head_bits": 6}
        )
        untied.tie_word_embeddings = False
        self.assertTrue(untied.head_is_quantized())

        no_head = EXL3Config.from_config({"quant_method": "exl3", "bits": 3.0})
        no_head.tie_word_embeddings = False
        self.assertFalse(no_head.head_is_quantized())

    def test_missing_file_falls_back(self):
        """The 0.0.0-era repos have no quantization_config.json at all."""
        cfg = self._config_for("turboderp/Llama-3.2-1B-Instruct-exl3", "3.0bpw")
        self.assertIsNone(cfg.tensor_storage)
        # With no storage map, everything is assumed quantized.
        self.assertTrue(cfg.is_quantized("model.layers.0.self_attn.qkv_proj"))


if __name__ == "__main__":
    unittest.main()
