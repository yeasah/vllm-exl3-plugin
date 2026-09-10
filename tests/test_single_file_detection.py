"""Finding a quantized `lm_head` in a checkpoint that declares nothing.

Three signals tell the plugin which modules carry EXL3 storage: the safetensors
index, a `quantization_config.json` storage map, and `head_bits` in
`config.json`. A checkpoint can have none of them —
`turboderp/Llama-3.2-1B-Instruct-exl3` ships one `model.safetensors` with
`lm_head.suh/svh/trellis` in it, no index (the Hub 404s that path), no sidecar,
and a `quantization_config` block whose keys are only `quant_method`, `version`,
`bits` and `calibration`. Every module then looks unquantized.

For a *tied* model that is not a cosmetic miss. The quantized `lm_head` is the
embedding, so failing to see it means the dense `embed_tokens.weight` is loaded
instead and the whole saving is given back — silently, with every logit still
correct. vLLM 0.28 hid this by skipping every `lm_head.*` weight on a tied
model (`skip_prefixes=["lm_head."]`); 0.29 skips only genuinely aliased
parameters, so the unclaimed trellis became a load error and exposed it.

The fix reads the one header, which is the same question the index answers.
"""

import json
import os
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_exl3_plugin.quantization.config import EXL3Config


def _write_single_file_checkpoint(dirpath: str, tensor_names: list[str]) -> None:
    """A minimal but structurally real single-file safetensors checkpoint."""
    header, offset = {}, 0
    for name in tensor_names:
        header[name] = {"dtype": "F16", "shape": [2], "data_offsets": [offset, offset + 4]}
        offset += 4
    blob = json.dumps(header).encode()
    with open(os.path.join(dirpath, "model.safetensors"), "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\0" * offset)


class SingleFileDetectionTest(unittest.TestCase):
    def _config_for(self, tensor_names: list[str], *, tied: bool = True) -> EXL3Config:
        """A config with no head_bits and no tensor_storage: the header decides."""
        cfg = EXL3Config(bits=3.0)
        cfg.tie_word_embeddings = tied
        with tempfile.TemporaryDirectory() as d:
            _write_single_file_checkpoint(d, tensor_names)
            # No index exists for such a checkpoint; mocked so the test never
            # reaches the network to find that out.
            with mock.patch(
                "vllm.transformers_utils.repo_utils.get_hf_file_to_dict",
                return_value=None,
            ):
                cfg._load_index_modules(d, None)
        return cfg

    def test_quantized_head_is_found_without_any_declaration(self):
        cfg = self._config_for(
            ["lm_head.suh", "lm_head.svh", "lm_head.trellis", "model.embed_tokens.weight"]
        )
        self.assertEqual(cfg.quantized_modules, {"lm_head"})
        # Tied plus head storage on disk: the embedding is served from it.
        self.assertTrue(cfg.embedding_is_quantized())

    def test_layers_are_found_too(self):
        cfg = self._config_for(
            [
                "lm_head.trellis",
                "model.layers.0.self_attn.q_proj.trellis",
                "model.layers.0.self_attn.q_proj.suh",
            ]
        )
        self.assertEqual(
            cfg.quantized_modules, {"lm_head", "model.layers.0.self_attn.q_proj"}
        )

    def test_dense_checkpoint_is_not_claimed(self):
        """The negative control: no trellis anywhere means nothing is claimed.

        Without this the fix could 'pass' by asserting storage that is not
        there, which registers parameters the checkpoint never fills — the
        failure `head_is_quantized`'s docstring calls fatal in the other
        direction.
        """
        cfg = self._config_for(["lm_head.weight", "model.embed_tokens.weight"])
        self.assertIsNone(cfg.quantized_modules)
        self.assertFalse(cfg.embedding_is_quantized())

    def test_untied_checkpoint_does_not_serve_embedding_from_head(self):
        cfg = self._config_for(["lm_head.trellis", "lm_head.suh"], tied=False)
        self.assertEqual(cfg.quantized_modules, {"lm_head"})
        self.assertFalse(cfg.embedding_is_quantized())

    def test_implausible_header_length_is_refused(self):
        """A truncated or corrupt file must not become a huge allocation."""
        cfg = EXL3Config(bits=3.0)
        cfg.tie_word_embeddings = True
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "model.safetensors"), "wb") as fh:
                fh.write(struct.pack("<Q", 1 << 40))  # 1 TiB of "header"
                fh.write(b"{}")
            with mock.patch(
                "vllm.transformers_utils.repo_utils.get_hf_file_to_dict",
                return_value=None,
            ):
                cfg._load_index_modules(d, None)
        self.assertIsNone(cfg.quantized_modules)


class DeclinedHeadIsDroppedTest(unittest.TestCase):
    """A quantized head nobody claims must be dropped from the weight stream.

    Through vLLM 0.28 a tied model discarded every `lm_head.*` weight, so a
    head this config declined to serve vanished on its own. 0.29 skips only
    genuinely aliased parameters, so the trellis arrives at a `ParallelLMHead`
    that has nowhere to put it and the load fails. `EXL3_DENSE_EMBED=1` is the
    documented way to isolate the embedding from every other change, and it is
    exactly the path that declines the head — so without an explicit drop it
    becomes the one setting under which a tied checkpoint cannot be loaded.
    """

    def _config(self, *, dense_embed: bool) -> EXL3Config:
        cfg = EXL3Config(bits=3.0)
        cfg.tie_word_embeddings = True
        cfg.quantized_modules = {"lm_head"}
        cfg._dense_embed = dense_embed
        return cfg

    def test_declined_head_is_dropped(self):
        mapper = self._config(dense_embed=True).get_cache_scale_mapper()
        for suffix in (".trellis", ".suh", ".svh"):
            self.assertIsNone(
                mapper._map_name("lm_head" + suffix),
                f"lm_head{suffix} must be dropped when the head is not served",
            )

    def test_served_head_is_renamed_not_dropped(self):
        """The control: when the tie *is* served, the same tensors must survive."""
        mapper = self._config(dense_embed=False).get_cache_scale_mapper()
        mapped = mapper._map_name("lm_head.trellis")
        self.assertIsNotNone(mapped, "a served head must not be dropped")
        self.assertNotEqual(mapped, "lm_head.trellis", "it should be renamed")
        self.assertIn("embed_tokens", mapped)


if __name__ == "__main__":
    unittest.main()
