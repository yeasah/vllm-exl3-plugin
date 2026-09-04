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


class NeverQueryMainTest(unittest.TestCase):
    """The config must not invent a revision for a Hub metadata lookup.

    Asking for `main` when the revision is unknown is not a harmless default:
    EXL3 repos keep weights on per-bit-rate branches, so `main` 404s, but the
    commit is resolved before the file is missed and huggingface_hub leaves
    `refs/main` pointing at a snapshot it never fetched. A dangling ref makes
    `scan_cache_dir` discard the entire repo, so every cache tool goes blind to
    all of it -- the checkpoint looks lost while every byte is on disk.

    A local directory is exempt: no revision to invent, no request to make.
    """

    def setUp(self):
        self.cfg = EXL3Config(bits=3.0, head_bits=6)

    def test_repo_id_without_revision_is_skipped(self):
        self.assertTrue(self.cfg._skip_hub_lookup("turboderp/Qwen3-8B-exl3", None))

    def test_repo_id_with_revision_proceeds(self):
        self.assertFalse(self.cfg._skip_hub_lookup("turboderp/Qwen3-8B-exl3", "3.0bpw"))

    def test_local_directory_proceeds_without_a_revision(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(self.cfg._skip_hub_lookup(d, None))

    def test_no_hub_call_is_made_without_a_revision(self):
        """The predicate is not the point; not reaching the network is."""
        from unittest import mock

        calls = []

        def spy(filename, model, revision):
            calls.append((filename, model, revision))
            return None

        with mock.patch(
            "vllm.transformers_utils.repo_utils.get_hf_file_to_dict", spy
        ):
            self.cfg._load_tensor_storage("turboderp/Qwen3-8B-exl3", None)
        self.assertEqual(calls, [], f"queried the Hub anyway: {calls}")

    def test_a_revision_does_reach_the_lookup(self):
        """The guard must not be so broad that it disables the feature."""
        from unittest import mock

        calls = []

        def spy(filename, model, revision):
            calls.append((filename, model, revision))
            return None

        with mock.patch(
            "vllm.transformers_utils.repo_utils.get_hf_file_to_dict", spy
        ):
            self.cfg._load_tensor_storage("turboderp/Qwen3-8B-exl3", "3.0bpw")
        self.assertTrue(calls, "the lookup never happened even with a revision")
        self.assertTrue(all(r == "3.0bpw" for _, _, r in calls),
                        f"a call used a substituted revision: {calls}")


class BlockQOnLoadTest(unittest.TestCase):
    """`EXL3_BLOCKQ_ON_LOAD` encodes a dense embedding while it loads.

    The decision has to be made where the method is chosen, not at load time.
    Every refusal below ends on the dense path -- exactly what would have
    happened without the flag -- because the alternative is a layer built for
    `bq_*` that nothing can fill, failing far from the cause.
    """

    def _cfg(self, *, on_load, hidden=1536, tied=False, stored=False):
        cfg = EXL3Config(bits=3.0, head_bits=6)
        cfg._blockq_on_load = on_load
        cfg._hidden_size = hidden
        cfg.tie_word_embeddings = tied
        if stored:
            cfg._blockq_modules = {GEMMA_PREFIX}
        return cfg

    def test_flag_off_leaves_a_stock_checkpoint_dense(self):
        cfg = self._cfg(on_load=False)
        self.assertFalse(cfg.embedding_is_blockq())
        self.assertFalse(cfg.blockq_is_on_load())

    def test_flag_on_encodes_an_untied_checkpoint(self):
        cfg = self._cfg(on_load=True)
        self.assertTrue(cfg.embedding_is_blockq())
        self.assertTrue(cfg.blockq_is_on_load())

    def test_incompatible_hidden_size_falls_back_to_dense(self):
        """BLOCKQ_BLOCK must divide the hidden size; 1000 is not divisible by 32."""
        from vllm_exl3_plugin import format

        with self.assertRaises(format.EXL3FormatError):
            format.check_blockq_hidden(1000)
        cfg = self._cfg(on_load=True, hidden=1000)
        self.assertFalse(cfg.embedding_is_blockq(),
                         "a model that cannot use the format must serve dense")

    def test_unknown_hidden_size_falls_back_to_dense(self):
        cfg = self._cfg(on_load=True, hidden=None)
        self.assertFalse(cfg.embedding_is_blockq())

    def test_a_stored_checkpoint_is_not_an_on_load_one(self):
        """Pre-quantized checkpoints keep reading bq_*, flag or not."""
        cfg = self._cfg(on_load=True, stored=True)
        self.assertTrue(cfg.embedding_is_blockq())
        self.assertFalse(cfg.blockq_is_on_load(),
                         "would re-encode over tensors the checkpoint already has")

    def test_dense_embed_flag_still_wins(self):
        cfg = self._cfg(on_load=True)
        cfg._dense_embed = True
        self.assertFalse(cfg.embedding_is_blockq())

    def test_the_dense_tensor_is_routed_not_dropped(self):
        """On-load needs the dense embedding to arrive; every other mode drops it."""
        cfg = self._cfg(on_load=True)
        mapper = cfg.get_cache_scale_mapper()
        name = "model.embed_tokens.weight"
        mapped = mapper._map_name(name)
        self.assertEqual(mapped, "model.embed_tokens.bq_src",
                         "the encoder never receives the tensor it encodes")

    def test_a_stored_checkpoint_still_drops_the_dense_tensor(self):
        cfg = self._cfg(on_load=True, stored=True)
        mapper = cfg.get_cache_scale_mapper()
        self.assertIsNone(mapper._map_name("model.embed_tokens.weight"))
