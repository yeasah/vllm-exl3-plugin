"""The on-load encode cache: identity, slicing, and every way it declines.

Two properties carry the weight here.

**A hit must equal a miss.** The cache exists to skip work, so the only failure
that matters is one where the skipped work would have produced something else.
`test_warm_load_matches_cold_load` drives the actual loader twice over the same
dense matrix and compares the stored shards byte for byte, at TP=1 and TP=3 --
including the ragged last rank, whose nominal span runs off the end of the
vocabulary.

**A wrong entry must be a miss, not an answer.** Every way of getting the key
wrong is exercised by poisoning a real entry and watching the read decline it,
rather than by asserting that two key strings differ.

Runs on CPU, in a temporary directory; no GPU, no checkpoint.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_exl3_plugin import blockq, blockq_cache, format
from vllm_exl3_plugin.quantization import embedding as embed_mod

VOCAB, HIDDEN = 320, 128


def dense(seed=0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(VOCAB, HIDDEN, generator=g, dtype=torch.float32)


class _Param:
    """The three fields `_row_span` and the loaders read off an EXL3Parameter."""

    def __init__(self, tp_size=1, tp_rank=0, row_shard_size=None):
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.row_shard_size = row_shard_size


class _Slot:
    def __init__(self):
        self.stored = None

    def store(self, tensor):
        self.stored = tensor.contiguous()


class _Layer:
    def __init__(self):
        for name in format.BLOCKQ_SUFFIXES:
            setattr(self, name.removeprefix("."), _Slot())

    def shards(self):
        return {
            n.removeprefix("."): getattr(self, n.removeprefix(".")).stored
            for n in format.BLOCKQ_SUFFIXES
        }


class _Config:
    def __init__(self, model_name="turboderp/Fake-exl3", commit_hash="deadbeef"):
        self.model_name = model_name
        # The branch the user asked for, which the cache deliberately ignores.
        self.revision = "3.00bpw"
        self.commit_hash = commit_hash


class _Method:
    def __init__(self, config=None, tensor="model.embed_tokens.weight"):
        self.quant_config = config or _Config()
        self.embed_tensor = tensor


class CacheDirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = mock.patch.dict(
            os.environ, {"EXL3_BLOCKQ_CACHE_DIR": self.tmp.name}, clear=False
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def entries(self):
        return sorted(
            n for n in os.listdir(self.tmp.name) if n.endswith(".safetensors")
        )


class IdentityTest(CacheDirTest):
    def test_a_hub_model_is_identified_by_its_commit(self):
        self.assertEqual(
            blockq_cache.checkpoint_identity("org/model-exl3", "abc123"),
            "org/model-exl3@abc123",
        )

    def test_a_hub_model_without_a_commit_is_not_cacheable(self):
        """The same refusal `_skip_hub_lookup` makes: an unresolved revision
        cannot be told apart from any other commit of the same repo."""
        self.assertIsNone(blockq_cache.checkpoint_identity("org/model-exl3", None))

    def test_a_local_directory_is_identified_by_its_files(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "config.json"), "w").write("{}")
            open(os.path.join(d, "model.safetensors"), "wb").write(b"x" * 16)
            first = blockq_cache.checkpoint_identity(d, None)
            self.assertIsNotNone(first)
            # A revision is ignored for a local path: the files are the truth.
            self.assertEqual(blockq_cache.checkpoint_identity(d, "main"), first)

            open(os.path.join(d, "model.safetensors"), "wb").write(b"x" * 32)
            self.assertNotEqual(
                blockq_cache.checkpoint_identity(d, None), first,
                "a rewritten shard left the identity unchanged",
            )

    def test_an_empty_directory_is_not_cacheable(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(blockq_cache.checkpoint_identity(d, None))

    def test_the_knob_disables_it(self):
        self.assertTrue(blockq_cache.enabled())
        with mock.patch.dict(os.environ, {"EXL3_BLOCKQ_CACHE": "0"}):
            self.assertFalse(blockq_cache.enabled())


class KeyTest(CacheDirTest):
    def paths(self, **kwargs):
        args = dict(
            identity="org/m@rev1",
            tensor="model.embed_tokens.weight",
            rows=VOCAB,
            hidden=HIDDEN,
        )
        args.update(kwargs)
        return blockq_cache.entry(**args).path

    def test_every_key_component_separates_entries(self):
        base = self.paths()
        for label, kwargs in [
            ("revision", dict(identity="org/m@rev2")),
            ("repo", dict(identity="other/m@rev1")),
            ("tensor", dict(tensor="model.embed_tokens_1.weight")),
            ("rows", dict(rows=VOCAB + 64)),
            ("hidden", dict(hidden=HIDDEN * 2)),
        ]:
            with self.subTest(component=label):
                self.assertNotEqual(base, self.paths(**kwargs))

    def test_the_format_constants_are_in_the_key(self):
        """Changing the block size or bit width must orphan old entries rather
        than let them be read as the new format."""
        base = self.paths()
        with mock.patch.object(format, "BLOCKQ_BLOCK", 64):
            self.assertNotEqual(base, self.paths())
        with mock.patch.object(format, "BLOCKQ_BITS", 3):
            self.assertNotEqual(base, self.paths())

    def test_the_version_orphans_rather_than_misreads(self):
        base = self.paths()
        with mock.patch.object(blockq_cache, "CACHE_VERSION", 2):
            self.assertNotEqual(base, self.paths())

    def test_the_filename_names_the_model_not_the_hash(self):
        """`ls` on the pile should say which checkpoints are in it, so the slug
        is truncated from the right -- a 40-character commit hash otherwise
        pushes the repository name out of the name entirely."""
        long_hash = "a" * 40
        name = os.path.basename(
            self.paths(identity=f"turboderp/MiniCPM5-1B-exl3@{long_hash}")
        )
        self.assertTrue(name.startswith("turboderp-MiniCPM5-1B-exl3-"), name)

    def test_metadata_records_what_the_entry_is(self):
        entry = blockq_cache.entry("org/m@rev1", "a.weight", VOCAB, HIDDEN)
        self.assertEqual(entry.meta["checkpoint"], "org/m@rev1")
        self.assertEqual(entry.meta["tensor"], "a.weight")
        self.assertTrue(all(isinstance(v, str) for v in entry.meta.values()))


class RoundTripTest(CacheDirTest):
    def setUp(self):
        super().setUp()
        self.w = dense()
        self.encoded = blockq.encode(self.w)
        self.entry = blockq_cache.entry("org/m@rev", "e.weight", VOCAB, HIDDEN)

    def test_written_then_read_is_identical(self):
        self.assertTrue(blockq_cache.write(self.entry, self.encoded))
        got = blockq_cache.read(
            self.entry, rows=VOCAB, hidden=HIDDEN, row_start=0, row_stop=VOCAB
        )
        self.assertIsNotNone(got)
        for name, want in self.encoded.items():
            self.assertTrue(torch.equal(got[name], want), name)

    def test_a_slice_matches_the_same_rows_of_the_whole(self):
        blockq_cache.write(self.entry, self.encoded)
        got = blockq_cache.read(
            self.entry, rows=VOCAB, hidden=HIDDEN, row_start=64, row_stop=192
        )
        for name, want in self.encoded.items():
            self.assertTrue(torch.equal(got[name], want[64:192]), name)

    def test_the_returned_rows_outlive_the_file(self):
        """safetensors hands back owned tensors, not a view into a mapping that
        the `with` block closes -- checked rather than assumed, because a stale
        mapping would read as correct until something reused the pages."""
        blockq_cache.write(self.entry, self.encoded)
        got = blockq_cache.read(
            self.entry, rows=VOCAB, hidden=HIDDEN, row_start=0, row_stop=VOCAB
        )
        os.unlink(self.entry.path)
        for name, want in self.encoded.items():
            self.assertTrue(torch.equal(got[name], want), name)

    def test_a_missing_entry_is_a_miss(self):
        self.assertIsNone(
            blockq_cache.read(
                self.entry, rows=VOCAB, hidden=HIDDEN, row_start=0, row_stop=VOCAB
            )
        )

    def test_a_truncated_entry_is_a_miss_and_not_an_error(self):
        blockq_cache.write(self.entry, self.encoded)
        with open(self.entry.path, "r+b") as f:
            f.truncate(64)
        self.assertIsNone(
            blockq_cache.read(
                self.entry, rows=VOCAB, hidden=HIDDEN, row_start=0, row_stop=VOCAB
            )
        )

    def test_an_entry_with_the_wrong_shape_is_a_miss(self):
        """The key should already have prevented this; the shape check is what
        makes a collision or a hand-edited pile a miss rather than a wrong
        matrix, so it is verified by putting the wrong tensors there."""
        blockq_cache.write(self.entry, blockq.encode(dense(seed=1)[: VOCAB // 2]))
        self.assertIsNone(
            blockq_cache.read(
                self.entry, rows=VOCAB, hidden=HIDDEN, row_start=0, row_stop=VOCAB
            )
        )

    def test_an_entry_missing_a_tensor_is_a_miss(self):
        partial = {k: v for k, v in self.encoded.items() if k != "bq_r"}
        blockq_cache.write(self.entry, partial)
        self.assertIsNone(
            blockq_cache.read(
                self.entry, rows=VOCAB, hidden=HIDDEN, row_start=0, row_stop=VOCAB
            )
        )

    def test_an_unwritable_pile_is_survivable(self):
        with mock.patch.dict(
            os.environ, {"EXL3_BLOCKQ_CACHE_DIR": "/proc/nonexistent/cache"}
        ):
            entry = blockq_cache.entry("org/m@rev", "e.weight", VOCAB, HIDDEN)
            self.assertFalse(blockq_cache.write(entry, self.encoded))
            self.assertIsNone(
                blockq_cache.read(
                    entry, rows=VOCAB, hidden=HIDDEN, row_start=0, row_stop=VOCAB
                )
            )

    def test_a_failed_write_leaves_no_temporary_behind(self):
        with mock.patch(
            "safetensors.torch.save_file", side_effect=RuntimeError("disk full")
        ):
            self.assertFalse(blockq_cache.write(self.entry, self.encoded))
        self.assertEqual(os.listdir(self.tmp.name), [])


class LoaderTest(CacheDirTest):
    """The loader itself, which is where a cache bug would actually land."""

    def run_load(self, w, *, tp_size=1, tp_rank=0, method=None):
        layer = _Layer()
        method = method or _Method()
        loader = embed_mod._make_on_load_loader(layer, method)
        shard = -(-VOCAB // tp_size) if tp_size > 1 else None
        loader(_Param(tp_size, tp_rank, shard), w)
        return layer.shards()

    def assert_same(self, a, b, note=""):
        for name in a:
            self.assertTrue(torch.equal(a[name], b[name]), f"{name} {note}")

    def test_warm_load_matches_cold_load(self):
        w = dense()
        for tp_size in (1, 3):
            for rank in range(tp_size):
                with self.subTest(tp_size=tp_size, rank=rank):
                    cold = self.run_load(w, tp_size=tp_size, tp_rank=rank)
                    self.assertTrue(self.entries(), "nothing was cached")
                    warm = self.run_load(w, tp_size=tp_size, tp_rank=rank)
                    self.assert_same(cold, warm, f"tp={tp_size} rank={rank}")

    def test_one_entry_serves_every_tp_arrangement(self):
        """The reason entries hold the whole vocabulary: a TP sweep is one
        encode, not one per arrangement."""
        w = dense()
        self.run_load(w, tp_size=1)
        self.assertEqual(len(self.entries()), 1)
        with mock.patch.object(
            embed_mod, "_encode_on_load", side_effect=AssertionError("re-encoded")
        ):
            for tp_size in (2, 4):
                for rank in range(tp_size):
                    self.run_load(w, tp_size=tp_size, tp_rank=rank)
        self.assertEqual(len(self.entries()), 1)

    def test_shards_partition_the_cached_encoding(self):
        w = dense()
        whole = blockq.encode(w)
        tp_size = 3
        for rank in range(tp_size):
            shard = -(-VOCAB // tp_size)
            got = self.run_load(w, tp_size=tp_size, tp_rank=rank)
            lo, hi = min(rank * shard, VOCAB), min((rank + 1) * shard, VOCAB)
            for name, want in whole.items():
                self.assertTrue(torch.equal(got[name], want[lo:hi]), f"{name}@{rank}")

    def test_a_different_commit_does_not_reuse_the_entry(self):
        """The whole point of keying on the commit rather than the branch: an
        EXL3 bit-rate branch gets re-pushed, and the two pushes have different
        embeddings. Both configs below name the same `revision`."""
        first, second = dense(seed=0), dense(seed=1)
        a = self.run_load(first, method=_Method(_Config(commit_hash="rev-a")))
        b = self.run_load(second, method=_Method(_Config(commit_hash="rev-b")))
        self.assertEqual(len(self.entries()), 2)
        self.assertFalse(torch.equal(a["bq_q"], b["bq_q"]))
        self.assert_same(b, blockq.encode(second), "second revision")

    def test_a_second_embedding_tensor_gets_its_own_entry(self):
        """Models with more than one embedding matrix exist; one key per
        checkpoint would hand the second one the first one's rows."""
        first, second = dense(seed=0), dense(seed=2)
        cfg = _Config()
        a = self.run_load(first, method=_Method(cfg, "model.embed_tokens.weight"))
        b = self.run_load(second, method=_Method(cfg, "model.embed_audio.weight"))
        self.assertEqual(len(self.entries()), 2)
        self.assert_same(b, blockq.encode(second), "second tensor")
        self.assertFalse(torch.equal(a["bq_q"], b["bq_q"]))

    def test_disabled_writes_nothing_and_still_serves(self):
        w = dense()
        with mock.patch.dict(os.environ, {"EXL3_BLOCKQ_CACHE": "0"}):
            got = self.run_load(w)
        self.assertEqual(self.entries(), [], "an entry was written with the knob off")
        self.assert_same(got, blockq.encode(w), "cache disabled")

    def test_disabled_at_tp_encodes_only_the_local_rows(self):
        """The uncached path must not inherit the whole-vocabulary encode that
        only exists so an entry can be shared."""
        w = dense()
        seen = []
        real = embed_mod._encode_on_load

        def spy(t):
            seen.append(t.shape[0])
            return real(t)

        with mock.patch.dict(os.environ, {"EXL3_BLOCKQ_CACHE": "0"}):
            with mock.patch.object(embed_mod, "_encode_on_load", spy):
                self.run_load(w, tp_size=4, tp_rank=1)
        self.assertEqual(seen, [-(-VOCAB // 4)])

    def test_an_uncacheable_checkpoint_still_serves(self):
        w = dense()
        got = self.run_load(w, method=_Method(_Config(commit_hash=None)))
        self.assertEqual(self.entries(), [])
        self.assert_same(got, blockq.encode(w), "no commit hash")


if __name__ == "__main__":
    unittest.main()
