"""Tests for the block-scaled embedding storage format.

The load-bearing test here is `test_matches_measured_scheme`: every quality
number in docs/embeddings.md that justified building this format rather than
adopting GGUF was produced by qbench's `blockq:32` *simulation*, which never
packs anything. If the real encoder diverges from that arithmetic, the shipped
format is not the one that was measured, and the measurements stop describing
it. The reference below is written out independently rather than imported, so
the two implementations have to agree rather than share a bug.

Runs on CPU; no GPU, no vLLM, no checkpoint.
"""

import unittest

import torch

from vllm_exl3_plugin import blockq, format


def reference_blockq(w: torch.Tensor, bits: int = 4, block: int = 32) -> torch.Tensor:
    """qbench's `blockq:N` granularity, transcribed from eval/qbench/engines.py."""
    levels = 2**bits - 1
    xf = w.float()
    v = xf.reshape(*xf.shape[:-1], xf.shape[-1] // block, block)
    lo = v.amin(dim=-1)
    sc = (v.amax(dim=-1) - lo).clamp_min(1e-12) / levels

    def q_row(t):
        tlo = t.amin(dim=-1, keepdim=True)
        st = (t.amax(dim=-1, keepdim=True) - tlo).clamp_min(1e-12) / 255
        return ((t - tlo) / st).round().clamp(0, 255) * st + tlo

    sc, lo = q_row(sc), q_row(lo)
    q = ((v - lo.unsqueeze(-1)) / sc.unsqueeze(-1).clamp_min(1e-12))
    q = q.round().clamp(0, levels)
    return (q * sc.unsqueeze(-1) + lo.unsqueeze(-1)).reshape(xf.shape)


def sample_embedding(vocab=512, hidden=256, seed=0) -> torch.Tensor:
    """Gaussian rows with a heavy-tailed minority, which is what real embeddings
    look like -- and the tail is the whole reason this format beats per-row."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(vocab, hidden, generator=g)
    outliers = torch.randint(0, vocab, (vocab // 16,), generator=g)
    w[outliers, torch.randint(0, hidden, (vocab // 16,), generator=g)] *= 40.0
    return w


class TestPacking(unittest.TestCase):
    def test_pack_unpack_is_exact(self):
        g = torch.Generator().manual_seed(1)
        q = torch.randint(0, 16, (64, 128), generator=g, dtype=torch.uint8)
        self.assertTrue(torch.equal(blockq.unpack(blockq.pack(q)), q))

    def test_nibble_order_is_low_first(self):
        q = torch.tensor([[1, 2, 3, 4]], dtype=torch.uint8)
        packed = blockq.pack(q)
        self.assertEqual(packed.tolist(), [[0x21, 0x43]])

    def test_packed_size_is_half(self):
        q = torch.zeros(8, 512, dtype=torch.uint8)
        self.assertEqual(list(blockq.pack(q).shape), [8, 256])


class TestEncodeDecode(unittest.TestCase):
    def test_matches_measured_scheme(self):
        """The shipped encoder must reproduce the simulation that was measured."""
        w = sample_embedding()
        stored = blockq.encode(w)
        got = blockq.decode(
            stored["bq_q"], stored["bq_s"], stored["bq_r"], torch.float32
        )
        want = reference_blockq(w)
        self.assertLess((got - want).abs().max().item(), 1e-5)

    def test_roundtrip_error_is_bounded(self):
        """A broken scale path still round-trips shapes; it fails on magnitude."""
        w = sample_embedding()
        stored = blockq.encode(w)
        got = blockq.decode(
            stored["bq_q"], stored["bq_s"], stored["bq_r"], torch.float32
        )
        rel = (got - w).norm() / w.norm()
        # ~0.08 measured on real embeddings at 4 bits; anything near 1.0 means
        # the scales are not being applied at all.
        self.assertLess(rel.item(), 0.15)

    def test_stored_shapes_match_the_format(self):
        w = sample_embedding(vocab=64, hidden=128)
        stored = blockq.encode(w)
        want = format.blockq_shapes(64, 128)
        for name, shape in want.items():
            self.assertEqual(list(stored[name].shape), list(shape), name)
        self.assertEqual(stored["bq_q"].dtype, torch.uint8)
        self.assertEqual(stored["bq_s"].dtype, torch.uint8)
        self.assertEqual(stored["bq_r"].dtype, torch.float32)

    def test_constant_and_zero_rows_survive(self):
        """Unused vocabulary slots are exactly zero in real checkpoints, and a
        zero row makes every scale zero -- the one place this arithmetic divides
        by something it computed itself."""
        w = torch.zeros(4, 64)
        w[1] = 3.5
        w[2, ::2] = -1.0
        stored = blockq.encode(w)
        got = blockq.decode(
            stored["bq_q"], stored["bq_s"], stored["bq_r"], torch.float32
        )
        self.assertTrue(torch.isfinite(got).all())
        self.assertTrue(torch.equal(got[0], torch.zeros(64)))
        self.assertLess((got[1] - w[1]).abs().max().item(), 1e-4)


class TestRowIndependence(unittest.TestCase):
    """The property the serving path and vocab-parallel TP both rest on."""

    def test_gather_equals_full_decode_indexed(self):
        w = sample_embedding()
        stored = blockq.encode(w)
        full = blockq.decode(
            stored["bq_q"], stored["bq_s"], stored["bq_r"], torch.float32
        )
        ids = torch.tensor([[7, 0], [511, 300]])
        got = blockq.gather(ids, stored["bq_q"], stored["bq_s"], stored["bq_r"],
                            torch.float32)
        self.assertEqual(list(got.shape), [2, 2, 256])
        self.assertTrue(torch.equal(got.view(4, -1), full[ids.reshape(-1)]))

    def test_encoding_a_row_slice_matches_the_whole(self):
        """A TP rank encodes its own vocabulary slice; it must get the same bytes
        as slicing the full tensor's storage."""
        w = sample_embedding()
        whole = blockq.encode(w)
        part = blockq.encode(w[128:256])
        for name in ("bq_q", "bq_s"):
            self.assertTrue(torch.equal(part[name], whole[name][128:256]), name)
        self.assertTrue(torch.allclose(part["bq_r"], whole["bq_r"][128:256]))


class TestShapeRules(unittest.TestCase):
    def test_rejects_hidden_that_splits_a_block(self):
        with self.assertRaises(format.EXL3FormatError):
            format.check_blockq_hidden(100)

    def test_rejects_odd_hidden(self):
        with self.assertRaises(format.EXL3FormatError):
            format.check_blockq_hidden(33, block=1)

    def test_real_hidden_sizes_are_all_fine(self):
        for hidden in (1536, 2048, 3072, 3840, 4096, 5120, 6656):
            format.check_blockq_hidden(hidden)

    def test_hidden_recovered_from_packed_shape(self):
        self.assertEqual(format.blockq_hidden_from_shape((262144, 1920)), 3840)

    def test_bpw_accounts_for_both_scale_levels(self):
        # 4 bits + 16 bits per 32 values + 4 fp32 per row
        self.assertAlmostEqual(format.blockq_bpw(4096), 4.5 + 128 / 4096, places=6)

    def test_module_keys_from_names(self):
        names = ["model.embed_tokens.bq_q", "model.embed_tokens.bq_s",
                 "model.layers.0.self_attn.q_proj.trellis"]
        self.assertEqual(format.blockq_module_keys(names), {"model.embed_tokens"})


if __name__ == "__main__":
    unittest.main()


HAVE_CUDA = torch.cuda.is_available()
requires_gpu = unittest.skipUnless(HAVE_CUDA, "needs CUDA")


@requires_gpu
class TestServingPath(unittest.TestCase):
    """The decode has to survive the two execution modes the plugin ships in.

    An eager-only result says nothing about the compiled/captured path, which is
    where this format is actually served -- and being fusible there rather than an
    opaque custom op is the reason it was built instead of adopting GGUF's kernel.
    """

    def setUp(self):
        self.w = sample_embedding(vocab=1024, hidden=512).cuda()
        self.stored = {k: v.cuda() for k, v in blockq.encode(self.w).items()}
        self.ids = torch.tensor([0, 5, 1023, 700], device="cuda")

    def _gather(self, ids):
        return blockq.gather(ids, self.stored["bq_q"], self.stored["bq_s"],
                             self.stored["bq_r"], torch.float16)

    def test_matches_cpu_encoder(self):
        """Not bit-identical, and should not be asserted as such: `.round()` on a
        computed float breaks ties differently on the two devices, so a handful of
        codes out of a vocabulary land one step apart. What must hold is that the
        served values agree to fp16 rounding."""
        cpu = blockq.decode(
            *[blockq.encode(self.w.cpu())[k][self.ids.cpu()]
              for k in ("bq_q", "bq_s", "bq_r")], torch.float16
        )
        got = self._gather(self.ids).cpu()
        torch.testing.assert_close(got, cpu, rtol=1e-3, atol=2e-3)

    def test_compiled_matches_eager(self):
        """Inductor is free to fuse the multiply-add, which moves the odd element
        by one fp16 ulp. Exactness is required of graph *replay*, not of the two
        backends against each other."""
        compiled = torch.compile(self._gather)
        torch.testing.assert_close(
            compiled(self.ids), self._gather(self.ids), rtol=1e-3, atol=2e-3
        )

    def test_survives_cuda_graph_capture(self):
        static = self.ids.clone()
        eager = self._gather(static).clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._gather(static)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self._gather(static)
        # Replay with different ids written into the same input buffer, which is
        # how vLLM drives a captured graph.
        static.copy_(torch.tensor([1, 2, 3, 4], device="cuda"))
        graph.replay()
        self.assertTrue(torch.equal(out, self._gather(static)))
        static.copy_(self.ids)
        graph.replay()
        self.assertTrue(torch.equal(out, eager))


class TestSidecarOutput(unittest.TestCase):
    """`tools/quantize_embedding.py --sidecar` writes a loadable checkpoint.

    The failure this exists for is silent. vLLM globs `*.safetensors` and then,
    if an index is present, keeps only the files that index names
    (`filter_duplicate_safetensors_files`). A sidecar left out of the map is
    dropped without a word, and the symptom arrives much later as `bq_*` that
    never loaded.

    The output is always renumbered into `model-0000i-of-0000N` with an index,
    including from a single-file source. Several safetensors files with no
    index does load under vLLM, which globs, but it is a layout that appears in
    none of the published checkpoints surveyed and so is not one to emit.
    """

    def setUp(self):
        import subprocess, sys, tempfile, os

        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.tool = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tools", "quantize_embedding.py",
        )
        self._run = lambda *a: subprocess.run(
            [sys.executable, self.tool, *a], capture_output=True, text=True
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _checkpoint(self, name, sharded):
        """A minimal EXL3-shaped checkpoint: an embedding plus one other tensor."""
        import json, os
        from safetensors.torch import save_file

        src = os.path.join(self.root, name)
        os.makedirs(src)
        embed = {"model.embed_tokens.weight": sample_embedding(vocab=256, hidden=256)}
        other = {"model.layers.0.mlp.up_proj.trellis": torch.zeros(4, 4, dtype=torch.int16)}
        if sharded:
            save_file(embed, os.path.join(src, "model-00001-of-00002.safetensors"))
            save_file(other, os.path.join(src, "model-00002-of-00002.safetensors"))
            index = {"metadata": {"total_size": 0}, "weight_map": {
                **{k: "model-00001-of-00002.safetensors" for k in embed},
                **{k: "model-00002-of-00002.safetensors" for k in other}}}
            with open(os.path.join(src, "model.safetensors.index.json"), "w") as f:
                json.dump(index, f)
        else:
            save_file({**embed, **other}, os.path.join(src, "model.safetensors"))
        with open(os.path.join(src, "config.json"), "w") as f:
            json.dump({"tie_word_embeddings": False}, f)
        return src

    def _sidecar(self, name, sharded):
        import os

        src = self._checkpoint(name, sharded)
        dst = os.path.join(self.root, name + "-bq")
        r = self._run(src, dst, "--sidecar")
        self.assertEqual(r.returncode, 0, r.stderr)
        return dst

    def _index(self, dst):
        import json, os

        with open(os.path.join(dst, "model.safetensors.index.json")) as f:
            return json.load(f)

    def test_index_names_the_sidecar(self):
        """Missing here, vLLM drops the file and bq_* silently never load."""
        for sharded in (True, False):
            with self.subTest(sharded=sharded):
                dst = self._sidecar(f"idx{sharded}", sharded)
                weight_map = self._index(dst)["weight_map"]
                for suffix in format.BLOCKQ_SUFFIXES:
                    name = f"model.embed_tokens{suffix}"
                    self.assertIn(name, weight_map)
                # ... and the dense embedding survives, in its original shard.
                self.assertIn("model.embed_tokens.weight", weight_map)

    def test_output_uses_the_sharded_convention(self):
        """Even from a single-file source: no bare model.safetensors, always an index."""
        import os

        for sharded, n in ((False, 2), (True, 3)):
            with self.subTest(sharded=sharded):
                dst = self._sidecar(f"conv{sharded}", sharded)
                files = sorted(f for f in os.listdir(dst) if f.endswith(".safetensors"))
                self.assertEqual(
                    files, [f"model-{i:05d}-of-{n:05d}.safetensors"
                            for i in range(1, n + 1)])
                self.assertTrue(os.path.exists(
                    os.path.join(dst, "model.safetensors.index.json")))

    def test_every_tensor_is_indexed_and_every_file_used(self):
        import os
        from safetensors import safe_open

        dst = self._sidecar("complete", sharded=True)
        weight_map = self._index(dst)["weight_map"]
        on_disk = {}
        for f in os.listdir(dst):
            if not f.endswith(".safetensors"):
                continue
            with safe_open(os.path.join(dst, f), framework="pt") as h:
                for k in h.keys():
                    on_disk[k] = f
        self.assertEqual(weight_map, on_disk)

    def test_sidecar_hardlinks_every_original_shard(self):
        import os

        dst = self._sidecar("links", sharded=True)
        news = [f for f in os.listdir(dst) if f.endswith(".safetensors")
                and os.stat(os.path.join(dst, f)).st_nlink == 1]
        self.assertEqual(news, ["model-00003-of-00003.safetensors"],
                         "only the sidecar should be new data")

    def test_sidecar_records_that_the_dense_copy_survives(self):
        """The flag that lets EXL3_DENSE_EMBED choose, on checkpoints with no index."""
        import json, os

        src = self._checkpoint("marker", sharded=False)
        for mode, expected in ((["--sidecar"], True), ([], False)):
            dst = os.path.join(self.root, f"marker-bq-{expected}")
            self.assertEqual(self._run(src, dst, *mode).returncode, 0)
            with open(os.path.join(dst, "quantization_config.json")) as f:
                storage = json.load(f)["tensor_storage"]
            entry = next(v for v in storage.values()
                         if v.get("quant_format") == "exl3_blockq")
            self.assertEqual(entry["dense_embedding_retained"], expected)

    def test_rewrite_mode_still_removes_the_dense_embedding(self):
        import os
        from safetensors import safe_open

        src = self._checkpoint("rewrite", sharded=False)
        dst = os.path.join(self.root, "rewrite-bq")
        self.assertEqual(self._run(src, dst).returncode, 0)
        with safe_open(os.path.join(dst, "model.safetensors"), framework="pt") as h:
            keys = set(h.keys())
        self.assertNotIn("model.embed_tokens.weight", keys)
        self.assertIn("model.embed_tokens.bq_q", keys)
