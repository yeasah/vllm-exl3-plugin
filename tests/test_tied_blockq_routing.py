"""Routing for a *tied* checkpoint whose embedding has also been repaired.

`embedding_is_quantized()` and `embedding_is_blockq()` were written as mutually
exclusive -- the docstring of the second says so -- but they answer questions
about two different modules. The first asks whether a tied model's `lm_head.*`
is being renamed onto the embedding; the second asks whether the embedding has
block-quantized tensors of its own. A repaired tied checkpoint makes both true,
and both are then load-bearing: the trellis serves the logits GEMM and the
`bq_*` tensors serve the row gather, each the right encoding for its role.

Two regressions are pinned here, both silent in the field:

  - Routing to `EXL3BlockQEmbeddingMethod` (which has no matmul path) instead of
    the combined method, so the tied head finds no trellis and dies at logits.
  - `embed_prefix` left at its `"model.embed_tokens"` default because the blockq
    branch returned before the assignment. The rename then targets a path a
    nested model does not have, 755 MiB of trellis is dropped without complaint,
    and the model serves garbage. Found 2026-08-26 on a repaired
    `gemma-4-12B-it-exl3`, whose embedding lives at
    `model.language_model.embed_tokens`.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_exl3_plugin.quantization.config import EXL3Config

GEMMA_PREFIX = "model.language_model.embed_tokens"


def _config(*, tied: bool, blockq: bool) -> EXL3Config:
    """A config shaped like gemma-4's: head_bits present, no tensor_storage."""
    cfg = EXL3Config(bits=3.0, head_bits=6)
    cfg.tie_word_embeddings = tied
    if blockq:
        cfg._blockq_modules = {GEMMA_PREFIX}
    return cfg


class _FakeEmbedding:
    """Enough of `VocabParallelEmbedding` for `isinstance` to be the only test."""


class RoutingTest(unittest.TestCase):
    def setUp(self):
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            VocabParallelEmbedding,
        )

        self.embed = mock.Mock(spec=VocabParallelEmbedding)
        self.head = mock.Mock(spec=ParallelLMHead)

    def method_for_embedding(self, cfg):
        return cfg.get_quant_method(self.embed, GEMMA_PREFIX)

    def test_tied_and_blockq_gets_the_combined_method(self):
        from vllm_exl3_plugin.quantization.embedding import (
            EXL3BlockQTiedEmbeddingMethod,
        )

        cfg = _config(tied=True, blockq=True)
        self.assertTrue(cfg.embedding_is_quantized())
        self.assertTrue(cfg.embedding_is_blockq())
        self.assertIsInstance(
            self.method_for_embedding(cfg), EXL3BlockQTiedEmbeddingMethod
        )

    def test_untied_blockq_is_unchanged(self):
        """The path this class must not disturb: an untied repaired checkpoint
        still gets the plain blockq method, with no trellis expected."""
        from vllm_exl3_plugin.quantization.embedding import (
            EXL3BlockQEmbeddingMethod,
            EXL3BlockQTiedEmbeddingMethod,
        )

        cfg = _config(tied=False, blockq=True)
        method = self.method_for_embedding(cfg)
        self.assertIsInstance(method, EXL3BlockQEmbeddingMethod)
        self.assertNotIsInstance(method, EXL3BlockQTiedEmbeddingMethod)

    def test_tied_unrepaired_is_unchanged(self):
        """Phase A: served from the head's trellis, no blockq tensors at all."""
        from vllm_exl3_plugin.quantization.embedding import EXL3EmbeddingMethod

        cfg = _config(tied=True, blockq=False)
        method = self.method_for_embedding(cfg)
        self.assertIsInstance(method, EXL3EmbeddingMethod)

    def test_embed_prefix_is_recorded_on_the_blockq_branch(self):
        """The silent one. The rename fires whenever `embedding_is_quantized()`
        is true, so the prefix must be recorded no matter which branch returns."""
        cfg = _config(tied=True, blockq=True)
        self.assertEqual(cfg.embed_prefix, "model.embed_tokens")  # the default
        self.method_for_embedding(cfg)
        self.assertEqual(cfg.embed_prefix, GEMMA_PREFIX)

    def test_rename_targets_the_nested_prefix(self):
        """End of the same thread: the mapper must send `lm_head.*` to the
        module that actually exists."""
        cfg = _config(tied=True, blockq=True)
        self.method_for_embedding(cfg)
        mapper = cfg.get_cache_scale_mapper()
        self.assertEqual(
            mapper.orig_to_new_prefix.get("lm_head."), f"{GEMMA_PREFIX}."
        )


class CombinedMethodTest(unittest.TestCase):
    def test_lookup_is_blockq_and_matmul_is_trellis(self):
        """The whole point of the class, asserted on the MRO rather than by
        running a GPU: the gather must resolve to the block-quantized
        implementation and `apply` must NOT resolve to the blockq stub that
        raises 'no matmul path'."""
        from vllm_exl3_plugin.quantization.embedding import (
            EXL3BlockQEmbeddingMethod,
            EXL3BlockQTiedEmbeddingMethod,
            EXL3EmbeddingMethod,
        )

        cls = EXL3BlockQTiedEmbeddingMethod
        self.assertIs(cls.embedding, EXL3BlockQEmbeddingMethod.embedding)
        # `EXL3TiedLMHeadMethod.apply` reaches the trellis through
        # `EXL3EmbeddingMethod`, so the embedding module must carry a real one.
        self.assertIsNot(cls.apply, EXL3BlockQEmbeddingMethod.apply)
        self.assertIs(cls.tie_weights, EXL3EmbeddingMethod.tie_weights)


if __name__ == "__main__":
    unittest.main()
