"""Tests for `tools/compose_checkpoint.py`.

Composition is the operation with the worst failure mode in this repo: every
mistake it can make produces a checkpoint that loads. A trellis paired with the
other conversion's `suh` is not a degraded model, it is noise, and no loader
checks for it -- so the guards are the deliverable and these tests exist to
watch them fire, not to confirm the happy path.

Three things are pinned here:

  - **Module grouping**, against the real module-key spellings of every EXL3
    family in the local collection. Grouping is derived from tensor names rather
    than read from `tensor_storage`, because `tensor_storage` omits modules --
    Muse-Glimmer lists none of its 303 vision modules -- and those are exactly
    the ones worth moving.
  - **Selector categories**, against the same spellings. `experts` in
    particular has to mean the *routed* experts under two different namings
    while never matching `shared_expert`, which is the distinction the MoE
    question turns on.
  - **The write path**, on a synthetic two-shard checkpoint: that untouched
    shards are hardlinked rather than rewritten, that the index is regenerated
    against what is actually on disk, and that a module whose donor stores a
    different codebook suffix moves whole.

The synthetic fixture uses real EXL3 shapes at toy sizes, so it runs in
milliseconds with no GPU and no model download.
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "compose_checkpoint.py")
sys.path.insert(0, ROOT)


def _load():
    spec = importlib.util.spec_from_file_location("compose_checkpoint", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cc = _load()

try:
    import torch
    from safetensors.torch import save_file
    HAVE_TORCH = True
except ImportError:  # the grouping and selector tests do not need it
    HAVE_TORCH = False


#: Module keys as the local collection actually spells them, one per family.
#: Written out rather than read from the cache so the test pins the spellings
#: even on a machine that has none of these checkpoints.
REAL_MODULES = [
    # (module key, source checkpoint)
    ("lm_head", "every EXL3 checkpoint"),
    ("model.embed_tokens", "Qwen3-0.6B-exl3"),
    ("model.layers.0.self_attn.q_proj", "Qwen3-0.6B-exl3"),
    ("model.layers.0.mlp.down_proj", "Qwen3-0.6B-exl3"),
    ("model.layers.0.input_layernorm", "Qwen3-0.6B-exl3"),
    ("model.language_model.layers.0.self_attn.k_proj", "Qwen3.8-27B-exl3"),
    ("model.language_model.layers.0.linear_attn.in_proj_qkv", "Qwen3.8-27B-exl3"),
    ("mtp.layers.0.mlp.gate_proj", "Qwen3.8-27B-exl3"),
    ("mtp.fc", "Qwen3.8-27B-exl3"),
    ("model.vision_tower.layers.0.attn.q_proj", "Muse-Glimmer-30B-exl3"),
    ("model.vision_adapter.fc1", "Muse-Glimmer-30B-exl3"),
    ("model.vision_projection", "Muse-Glimmer-30B-exl3"),
    ("model.language_model.layers.0.experts.0.gate_proj", "gemma-4-26B-A4B-it-exl3"),
    ("model.language_model.layers.0.router.proj", "gemma-4-26B-A4B-it-exl3"),
    ("model.language_model.layers.0.router.scale", "gemma-4-26B-A4B-it-exl3"),
    ("model.language_model.layers.0.mlp.experts.7.up_proj", "Qwen3.5-35B-A3B-exl3"),
    ("model.language_model.layers.0.mlp.shared_expert.up_proj", "Qwen3.5-35B-A3B-exl3"),
    ("model.layers.0.mlp.experts.3.down_proj", "Laguna-XS-2.1-exl3"),
    ("model.layers.0.mlp.shared_expert.down_proj", "Laguna-XS-2.1-exl3"),
]


class TestModuleGrouping(unittest.TestCase):
    """`module_key` must reproduce what `tensor_storage` groups together."""

    def test_exl3_suffixes_strip_to_the_module(self):
        for suffix in (".trellis", ".suh", ".svh", ".mcg", ".mul1", ".bias"):
            with self.subTest(suffix):
                self.assertEqual(
                    cc.module_key(f"model.layers.0.self_attn.q_proj{suffix}"),
                    "model.layers.0.self_attn.q_proj",
                )

    def test_dense_weight_strips_to_the_module(self):
        self.assertEqual(
            cc.module_key("model.layers.0.input_layernorm.weight"),
            "model.layers.0.input_layernorm",
        )
        self.assertEqual(
            cc.module_key("model.embed_tokens.weight"), "model.embed_tokens"
        )

    def test_blockq_suffixes_strip_to_the_module(self):
        for suffix in (".bq_q", ".bq_s", ".bq_r"):
            with self.subTest(suffix):
                self.assertEqual(
                    cc.module_key(f"model.embed_tokens{suffix}"), "model.embed_tokens"
                )

    def test_unrecognized_trailing_component_is_its_own_module(self):
        """`router.scale`'s tensor name *is* its module key.

        gemma-4-26B-A4B stores 30 of these. Stripping an unrecognized trailing
        component would fold the router's scale into a module called `router`,
        which does not exist, and the two would then move as one.
        """
        name = "model.language_model.layers.0.router.scale"
        self.assertEqual(cc.module_key(name), name)

    def test_a_quantized_linear_groups_into_exactly_one_module(self):
        names = [
            "model.layers.0.self_attn.q_proj.trellis",
            "model.layers.0.self_attn.q_proj.suh",
            "model.layers.0.self_attn.q_proj.svh",
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.self_attn.q_proj.mcg",
        ]
        groups = cc.modules_of(names)
        self.assertEqual(list(groups), ["model.layers.0.self_attn.q_proj"])
        self.assertEqual(sorted(groups["model.layers.0.self_attn.q_proj"]),
                         sorted(names))


class TestSelectors(unittest.TestCase):
    """Categories, against the real spellings of every family in the collection."""

    def setUp(self):
        self.modules = [m for m, _ in REAL_MODULES]
        # Everything except the norms, the embedding and `router.scale` carries
        # a trellis in the checkpoints these names come from.
        self.quantized = {
            m for m in self.modules
            if not m.endswith(("layernorm", "embed_tokens", "router.scale"))
        }

    def sel(self, s):
        return cc.select(s, self.modules, self.quantized)

    def test_head(self):
        self.assertEqual(self.sel("head"), {"lm_head"})

    def test_embed(self):
        self.assertEqual(self.sel("embed"), {"model.embed_tokens"})

    def test_vision_covers_tower_adapter_and_projection(self):
        self.assertEqual(self.sel("vision"), {
            "model.vision_tower.layers.0.attn.q_proj",
            "model.vision_adapter.fc1",
            "model.vision_projection",
        })

    def test_mtp(self):
        self.assertEqual(self.sel("mtp"),
                         {"mtp.layers.0.mlp.gate_proj", "mtp.fc"})

    def test_experts_means_routed_only(self):
        """The distinction the MoE bit-allocation question turns on.

        Routed experts are numbered; the shared expert is not. `experts` must
        match the first under both namings (`layers.N.experts.N.` and
        `layers.N.mlp.experts.N.`) and must never match `shared_expert`, or
        `--take all --keep experts` silently boosts nothing.
        """
        self.assertEqual(self.sel("experts"), {
            "model.language_model.layers.0.experts.0.gate_proj",
            "model.language_model.layers.0.mlp.experts.7.up_proj",
            "model.layers.0.mlp.experts.3.down_proj",
        })

    def test_shared_experts_are_disjoint_from_routed(self):
        shared = self.sel("shared-experts")
        self.assertEqual(shared, {
            "model.language_model.layers.0.mlp.shared_expert.up_proj",
            "model.layers.0.mlp.shared_expert.down_proj",
        })
        self.assertEqual(shared & self.sel("experts"), set())

    def test_body_excludes_head_embed_vision_and_mtp(self):
        body = self.sel("body")
        self.assertNotIn("lm_head", body)
        self.assertNotIn("model.embed_tokens", body)
        for m in self.sel("vision") | self.sel("mtp"):
            self.assertNotIn(m, body)
        self.assertIn("model.layers.0.self_attn.q_proj", body)

    def test_body_and_quantized_never_include_dense_modules(self):
        for sel in ("body", "quantized"):
            with self.subTest(sel):
                self.assertNotIn("model.layers.0.input_layernorm", self.sel(sel))
                self.assertNotIn("model.language_model.layers.0.router.scale",
                                 self.sel(sel))

    def test_glob_and_regex(self):
        self.assertEqual(self.sel("model.layers.0.mlp.*"), {
            "model.layers.0.mlp.down_proj",
            "model.layers.0.mlp.experts.3.down_proj",
            "model.layers.0.mlp.shared_expert.down_proj",
        })
        self.assertEqual(self.sel("re:^mtp\\."),
                         {"mtp.layers.0.mlp.gate_proj", "mtp.fc"})

    def test_exact_module_name(self):
        self.assertEqual(self.sel("mtp.fc"), {"mtp.fc"})

    def test_unknown_selector_is_refused(self):
        """A typo must not silently select nothing and compose a no-op."""
        with self.assertRaises(SystemExit):
            self.sel("shared_experts")   # the real spelling is shared-experts
        with self.assertRaises(SystemExit):
            self.sel("heads")


# ---------------------------------------------------------------------------
# End to end, on a synthetic two-shard checkpoint
# ---------------------------------------------------------------------------

#: Toy dimensions that are still legal EXL3: multiples of 16 on both axes.
IN_F, OUT_F, VOCAB = 128, 128, 256


def _linear(name, bits, codebook, fill):
    t = cc.format.trellis_shape(IN_F, OUT_F, bits)
    return {
        f"{name}.trellis": torch.full(t, fill, dtype=torch.int16),
        f"{name}.suh": torch.full((IN_F,), float(fill), dtype=torch.float16),
        f"{name}.svh": torch.full((OUT_F,), float(fill), dtype=torch.float16),
        f"{name}.{codebook}": torch.tensor([fill], dtype=torch.int32),
    }


def _head(bits, codebook, fill):
    t = cc.format.trellis_shape(IN_F, VOCAB, bits)
    return {
        "lm_head.trellis": torch.full(t, fill, dtype=torch.int16),
        "lm_head.suh": torch.full((IN_F,), float(fill), dtype=torch.float16),
        "lm_head.svh": torch.full((VOCAB,), float(fill), dtype=torch.float16),
        f"lm_head.{codebook}": torch.tensor([fill], dtype=torch.int32),
    }


def _make_ckpt(d, *, body_bits, head_bits, codebook, fill):
    """A two-shard checkpoint with an index, the shape every sharded EXL3 uses.

    Shard 1 holds the layers, shard 2 holds the head and the embedding -- so a
    head swap must rewrite shard 2 and hardlink shard 1, which is the economy
    the tool is for and the thing worth asserting.
    """
    os.makedirs(d, exist_ok=True)
    s1, s2 = {}, {}
    for i in range(2):
        s1.update(_linear(f"model.layers.{i}.self_attn.q_proj", body_bits,
                          codebook, fill))
        s1.update(_linear(f"model.layers.{i}.mlp.experts.0.up_proj", body_bits,
                          codebook, fill))
        s1[f"model.layers.{i}.input_layernorm.weight"] = torch.full(
            (IN_F,), float(fill), dtype=torch.float16)
    s2.update(_head(head_bits, codebook, fill))
    s2["model.embed_tokens.weight"] = torch.full((VOCAB, IN_F), float(fill),
                                                 dtype=torch.float16)
    names = {}
    for shard, blob in (("model-00001-of-00002.safetensors", s1),
                        ("model-00002-of-00002.safetensors", s2)):
        save_file(blob, os.path.join(d, shard), metadata={"format": "pt"})
        names.update({k: shard for k in blob})
    with open(os.path.join(d, cc.INDEX_NAME), "w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": names}, f)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"architectures": ["ToyForCausalLM"], "model_type": "toy",
                   "hidden_size": IN_F, "vocab_size": VOCAB}, f)
    with open(os.path.join(d, "quantization_config.json"), "w") as f:
        json.dump({"quant_method": "exl3", "version": "1.4.3",
                   "bits": body_bits, "head_bits": head_bits,
                   "codebook": codebook,
                   "tensor_storage": {
                       m: {"quant_format": "exl3", "bits_per_weight":
                           head_bits if m == "lm_head" else body_bits,
                           "stored_tensors": {t: {} for t in ts}}
                       for m, ts in cc.modules_of(list(names)).items()}}, f)
    return d


def _run(*argv):
    return subprocess.run([sys.executable, TOOL, *argv], capture_output=True,
                          text=True)


@unittest.skipUnless(HAVE_TORCH, "needs torch and safetensors")
class TestComposeEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = _make_ckpt(os.path.join(self.tmp.name, "base"),
                               body_bits=3, head_bits=6, codebook="mcg", fill=1)
        self.donor = _make_ckpt(os.path.join(self.tmp.name, "donor"),
                                body_bits=4, head_bits=5, codebook="mul1", fill=2)
        self.out = os.path.join(self.tmp.name, "out")

    def _tensors(self, d):
        from safetensors import safe_open
        out = {}
        for shard in sorted(os.listdir(d)):
            if not shard.endswith(".safetensors"):
                continue
            with safe_open(os.path.join(d, shard), framework="pt") as h:
                for k in h.keys():
                    out[k] = (shard, h.get_tensor(k))
        return out

    def test_head_swap_moves_the_whole_module_and_nothing_else(self):
        r = _run(self.base, self.out, "--from", self.donor, "--take", "head")
        self.assertEqual(r.returncode, 0, r.stderr)
        got = self._tensors(self.out)

        # Every head tensor came from the donor: fill 2, K=5, and the donor's
        # codebook suffix -- so the base's `.mcg` must be gone, not left behind
        # beside the donor's `.mul1`.
        self.assertIn("lm_head.mul1", got)
        self.assertNotIn("lm_head.mcg", got)
        self.assertEqual(
            cc.format.bits_from_trellis_shape(list(got["lm_head.trellis"][1].shape)), 5)
        for k in ("lm_head.trellis", "lm_head.suh", "lm_head.svh"):
            self.assertTrue((got[k][1] == 2).all(), k)

        # Everything else is still the base's.
        for k, (_, t) in got.items():
            if not k.startswith("lm_head"):
                self.assertTrue((t == 1).all(), k)

    def test_dense_embedding_survives_a_head_take(self):
        """`--take head` must not disturb the stored bf16 embedding.

        Every EXL3 checkpoint ships a dense `embed_tokens.weight` even when
        tied, and `tools/quantize_embedding.py` reads exactly that tensor to
        build `bq_*`. So the one thing a head swap must leave alone is the
        input to blockq generation -- and on a tied model, where `lm_head`
        backs the lookup on the serving path, that is easy to conflate.
        """
        r = _run(self.base, self.out, "--from", self.donor, "--take", "head")
        self.assertEqual(r.returncode, 0, r.stderr)
        got = self._tensors(self.out)
        self.assertIn("model.embed_tokens.weight", got)
        shard, t = got["model.embed_tokens.weight"]
        self.assertTrue((t == 1).all())          # the base's, not the donor's
        self.assertEqual(t.dtype, torch.float16)
        self.assertEqual(tuple(t.shape), (VOCAB, IN_F))

    def test_untouched_shard_is_hardlinked(self):
        """The economy of the tool: a head swap must not rewrite the body.

        Asserted on the inode, because a copy passes every content check a
        rewritten shard would.
        """
        r = _run(self.base, self.out, "--from", self.donor, "--take", "head")
        self.assertEqual(r.returncode, 0, r.stderr)
        untouched = "model-00001-of-00002.safetensors"
        rewritten = "model-00002-of-00002.safetensors"
        self.assertEqual(os.stat(os.path.join(self.base, untouched)).st_ino,
                         os.stat(os.path.join(self.out, untouched)).st_ino)
        self.assertNotEqual(os.stat(os.path.join(self.base, rewritten)).st_ino,
                            os.stat(os.path.join(self.out, rewritten)).st_ino)

    def test_index_is_regenerated_against_what_is_on_disk(self):
        """Copying the base's index would name `lm_head.mcg`, which is not there.

        vLLM keeps only the files an index mentions and loads only the tensors
        it maps, so an index describing tensors that do not exist is a load
        failure, and one omitting tensors that do is a silent drop.
        """
        r = _run(self.base, self.out, "--from", self.donor, "--take", "head")
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(os.path.join(self.out, cc.INDEX_NAME)) as f:
            index = json.load(f)
        self.assertEqual(set(index["weight_map"]), set(self._tensors(self.out)))
        for name, shard in index["weight_map"].items():
            self.assertEqual(self._tensors(self.out)[name][0], shard)
        self.assertGreater(index["metadata"]["total_size"], 0)

    def test_keep_reverts_a_take(self):
        """`--take all --keep experts` is the MoE non-routed-boost composition."""
        r = _run(self.base, self.out, "--from", self.donor,
                 "--take", "all", "--keep", "experts")
        self.assertEqual(r.returncode, 0, r.stderr)
        got = self._tensors(self.out)
        for k, (_, t) in got.items():
            expected = 1 if ".experts." in k else 2
            self.assertTrue((t == expected).all(),
                            f"{k} should have come from "
                            f"{'base' if expected == 1 else 'donor'}")

    def test_quantization_config_records_the_swap_and_flags_what_is_stale(self):
        r = _run(self.base, self.out, "--from", self.donor, "--take", "head")
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(os.path.join(self.out, "quantization_config.json")) as f:
            qc = json.load(f)
        self.assertEqual(qc["head_bits"], 5)              # corrected
        self.assertEqual(qc["bits"], 3)                   # inherited
        self.assertIn("bits", qc["composition"]["stale"])
        self.assertIn("codebook", qc["composition"]["stale"])
        self.assertEqual(qc["composition"]["modules_replaced"], 1)
        # The moved module's entry followed it rather than being recomputed.
        self.assertEqual(qc["tensor_storage"]["lm_head"]["bits_per_weight"], 5)
        self.assertIn("lm_head.mul1",
                      qc["tensor_storage"]["lm_head"]["stored_tensors"])

    def test_dry_run_writes_nothing(self):
        r = _run(self.base, self.out, "--from", self.donor, "--take", "head",
                 "--dry-run")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse(os.path.exists(self.out))


@unittest.skipUnless(HAVE_TORCH, "needs torch and safetensors")
class TestGuards(unittest.TestCase):
    """Each guard, with the failure it exists for put back in front of it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = _make_ckpt(os.path.join(self.tmp.name, "base"),
                               body_bits=3, head_bits=6, codebook="mcg", fill=1)
        self.out = os.path.join(self.tmp.name, "out")

    def test_dimension_mismatch_is_refused(self):
        """The failure that motivates the tool's existence as a guarded thing.

        A donor of the same architecture but a different width composes into a
        checkpoint that loads and emits noise. It must not write.
        """
        global IN_F, OUT_F
        wide = os.path.join(self.tmp.name, "wide")
        IN_F, OUT_F = 256, 256
        try:
            _make_ckpt(wide, body_bits=3, head_bits=6, codebook="mcg", fill=2)
        finally:
            IN_F, OUT_F = 128, 128
        r = _run(self.base, self.out, "--from", wide, "--take", "head")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("dimensions differ", r.stderr)
        self.assertFalse(os.path.exists(self.out))

    def test_force_overrides_the_refusal(self):
        global IN_F, OUT_F
        wide = os.path.join(self.tmp.name, "wide")
        IN_F, OUT_F = 256, 256
        try:
            _make_ckpt(wide, body_bits=3, head_bits=6, codebook="mcg", fill=2)
        finally:
            IN_F, OUT_F = 128, 128
        r = _run(self.base, self.out, "--from", wide, "--take", "head", "--force")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("dimensions differ", r.stderr)   # still said

    def test_taking_a_module_the_base_lacks_is_refused(self):
        """Composition replaces; it cannot add.

        A donor with modules the base has no slot for would produce weights
        nothing loads, and silently so.
        """
        donor = _make_ckpt(os.path.join(self.tmp.name, "donor"),
                           body_bits=4, head_bits=5, codebook="mcg", fill=2)
        from safetensors.torch import save_file as sf
        extra = os.path.join(donor, "model-00003-of-00002.safetensors")
        sf(_linear("model.layers.9.self_attn.q_proj", 4, "mcg", 2), extra,
           metadata={"format": "pt"})
        idx = os.path.join(donor, cc.INDEX_NAME)
        with open(idx) as f:
            index = json.load(f)
        with open(idx, "w") as f:
            json.dump({"metadata": index["metadata"], "weight_map": {
                **index["weight_map"],
                **{k: os.path.basename(extra)
                   for k in _linear("model.layers.9.self_attn.q_proj", 4, "mcg", 2)}
            }}, f)
        r = _run(self.base, self.out, "--from", donor, "--take",
                 "model.layers.9.*")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("the base does not have", r.stderr + r.stdout)

    def test_codebook_change_is_reported_not_refused(self):
        """Per-tensor and benign, but the output mixes codebooks and must say so."""
        donor = _make_ckpt(os.path.join(self.tmp.name, "donor"),
                           body_bits=3, head_bits=6, codebook="mul1", fill=2)
        r = _run(self.base, self.out, "--from", donor, "--take", "head",
                 "--dry-run")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("change codebook", r.stderr)

    def test_version_mismatch_is_reported(self):
        """Unrecorded conversion conventions are the risk no shape check sees."""
        donor = _make_ckpt(os.path.join(self.tmp.name, "donor"),
                           body_bits=3, head_bits=5, codebook="mcg", fill=2)
        qp = os.path.join(donor, "quantization_config.json")
        with open(qp) as f:
            q = json.load(f)
        q["version"] = "0.0.1"
        with open(qp, "w") as f:
            json.dump(q, f)
        r = _run(self.base, self.out, "--from", donor, "--take", "head",
                 "--dry-run")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("versions differ", r.stderr)

    def test_multi_shard_without_an_index_is_refused(self):
        """A layout no publisher emits, and one this tool will not consume.

        Without an index there is nothing that says which shards belong to the
        model, so composing from it would be guessing.
        """
        broken = os.path.join(self.tmp.name, "broken")
        _make_ckpt(broken, body_bits=3, head_bits=6, codebook="mcg", fill=2)
        os.remove(os.path.join(broken, cc.INDEX_NAME))
        r = _run(self.base, self.out, "--from", broken, "--take", "head")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no model.safetensors.index.json", r.stderr + r.stdout)

    def test_no_rules_is_refused(self):
        r = _run(self.base, self.out)
        self.assertNotEqual(r.returncode, 0)

    def test_rules_selecting_nothing_are_refused(self):
        """`--take head --keep head` changes nothing; writing a copy would mislead."""
        donor = _make_ckpt(os.path.join(self.tmp.name, "donor"),
                           body_bits=4, head_bits=5, codebook="mcg", fill=2)
        r = _run(self.base, self.out, "--from", donor, "--take", "head",
                 "--keep", "head")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("selected no modules", r.stderr + r.stdout)

    def test_existing_output_is_not_overwritten(self):
        donor = _make_ckpt(os.path.join(self.tmp.name, "donor"),
                           body_bits=4, head_bits=5, codebook="mcg", fill=2)
        os.makedirs(self.out)
        with open(os.path.join(self.out, "something"), "w") as f:
            f.write("x")
        r = _run(self.base, self.out, "--from", donor, "--take", "head")
        self.assertNotEqual(r.returncode, 0)


def _make_dense_ckpt(d, *, fill, vision=True, out_f=OUT_F, in_f=IN_F):
    """An unquantized checkpoint: the model an EXL3 one was converted from."""
    os.makedirs(d, exist_ok=True)
    blob = {}
    for i in range(2):
        blob[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.full(
            (out_f, in_f), float(fill), dtype=torch.bfloat16)
        blob[f"model.layers.{i}.mlp.experts.0.up_proj.weight"] = torch.full(
            (OUT_F, IN_F), float(fill), dtype=torch.bfloat16)
        blob[f"model.layers.{i}.input_layernorm.weight"] = torch.full(
            (IN_F,), float(fill), dtype=torch.bfloat16)
    if vision:
        blob["model.vision_tower.layers.0.attn.q_proj.weight"] = torch.full(
            (out_f, in_f), float(fill), dtype=torch.bfloat16)
    blob["lm_head.weight"] = torch.full((VOCAB, IN_F), float(fill),
                                        dtype=torch.bfloat16)
    save_file(blob, os.path.join(d, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"architectures": ["ToyForCausalLM"], "model_type": "toy",
                   "hidden_size": IN_F, "vocab_size": VOCAB}, f)
    return d


def _add_vision(d, bits, codebook, fill):
    """Give a synthetic EXL3 checkpoint a quantized vision module."""
    from safetensors import safe_open
    shard = os.path.join(d, "model-00001-of-00002.safetensors")
    with safe_open(shard, framework="pt") as h:
        blob = {k: h.get_tensor(k) for k in h.keys()}
    blob.update(_linear("model.vision_tower.layers.0.attn.q_proj", bits,
                        codebook, fill))
    save_file(blob, shard, metadata={"format": "pt"})
    idx = os.path.join(d, cc.INDEX_NAME)
    with open(idx) as f:
        index = json.load(f)
    for k in _linear("model.vision_tower.layers.0.attn.q_proj", bits, codebook, fill):
        index["weight_map"][k] = os.path.basename(shard)
    with open(idx, "w") as f:
        json.dump(index, f)
    qp = os.path.join(d, "quantization_config.json")
    with open(qp) as f:
        q = json.load(f)
    q["vision_bits"] = bits
    with open(qp, "w") as f:
        json.dump(q, f)


@unittest.skipUnless(HAVE_TORCH, "needs torch and safetensors")
class TestRestoreDense(unittest.TestCase):
    """`--restore`: replacing a quantized module with the original BF16.

    The publish-grade case is putting a BF16 vision tower back into a checkpoint
    that quantized one. Only two model families on the Hub do that
    (`Muse-Glimmer-30B-exl3` and `DeepSeek-V4-Flash-Vision-Exp-exl3`, 11
    revisions between them), but the operation is a format change rather than a
    bit-width change, so it gets its own rule and its own checks.

    **The numerical guard is proven on real checkpoints, not here.** A synthetic
    trellis is random and decodes to nothing, so these tests pass `--verify 0`.
    The guard itself was exercised on `turboderp/gemma-4-12B-it-exl3@3.00bpw_mul1`
    against `google/gemma-4-12B-it`: a genuine restore measures a relative error
    of **0.1674** at K=3, matching the 0.1441-0.1790 band docs/qbench.md records
    for that bit width, while the same tensor with its rows permuted measures
    **1.4128**. The threshold of 0.6 sits in that 8.4x gap.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = _make_ckpt(os.path.join(self.tmp.name, "base"),
                               body_bits=3, head_bits=6, codebook="mcg", fill=1)
        _add_vision(self.base, 6, "mcg", 1)
        self.orig = _make_dense_ckpt(os.path.join(self.tmp.name, "orig"), fill=7)
        self.out = os.path.join(self.tmp.name, "out")

    def _tensors(self, d):
        from safetensors import safe_open
        out = {}
        for shard in sorted(os.listdir(d)):
            if not shard.endswith(".safetensors"):
                continue
            with safe_open(os.path.join(d, shard), framework="pt") as h:
                for k in h.keys():
                    out[k] = (shard, h.get_tensor(k))
        return out

    def test_vision_restore_drops_every_exl3_tensor(self):
        """The whole point: nothing of the trellis may survive.

        A leftover `suh` would keep `format.quantized_module_keys` counting the
        module as EXL3 storage, and the loader would then look for a trellis
        that is not there.
        """
        r = _run(self.base, self.out, "--from", self.orig, "--restore", "vision",
                 "--verify", "0")
        self.assertEqual(r.returncode, 0, r.stderr)
        got = self._tensors(self.out)
        mod = "model.vision_tower.layers.0.attn.q_proj"
        self.assertIn(f"{mod}.weight", got)
        for suffix in (".trellis", ".suh", ".svh", ".mcg", ".mul1"):
            self.assertNotIn(f"{mod}{suffix}", got)
        self.assertTrue((got[f"{mod}.weight"][1] == 7).all())
        # and the module no longer reads as quantized
        self.assertNotIn(mod, cc.format.quantized_module_keys(got))

    def test_vision_bits_is_dropped_once_nothing_vision_is_quantized(self):
        r = _run(self.base, self.out, "--from", self.orig, "--restore", "vision",
                 "--verify", "0")
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(os.path.join(self.out, "quantization_config.json")) as f:
            qc = json.load(f)
        self.assertNotIn("vision_bits", qc)
        self.assertEqual(qc["composition"]["modules_restored_to_dense"], 1)

    def test_tensor_storage_entry_becomes_dense(self):
        """It must stop claiming `quant_format: exl3` for a dense module."""
        r = _run(self.base, self.out, "--from", self.orig, "--restore", "vision",
                 "--verify", "0")
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(os.path.join(self.out, "quantization_config.json")) as f:
            ts = json.load(f)["tensor_storage"]
        entry = ts["model.vision_tower.layers.0.attn.q_proj"]
        self.assertNotIn("quant_format", entry)
        self.assertIn("model.vision_tower.layers.0.attn.q_proj.weight",
                      entry["stored_tensors"])

    def test_index_matches_disk_after_a_restore(self):
        r = _run(self.base, self.out, "--from", self.orig, "--restore", "vision",
                 "--verify", "0")
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(os.path.join(self.out, cc.INDEX_NAME)) as f:
            index = json.load(f)
        self.assertEqual(set(index["weight_map"]), set(self._tensors(self.out)))

    def test_restoring_an_unquantized_module_is_refused(self):
        r = _run(self.base, self.out, "--from", self.orig, "--restore",
                 "model.layers.0.input_layernorm", "--verify", "0")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no *quantized* base module", r.stderr + r.stdout)

    def test_donor_missing_the_module_is_refused(self):
        """Names not lining up is the fusion/split case, and must be loud."""
        thin = _make_dense_ckpt(os.path.join(self.tmp.name, "thin"), fill=7,
                                vision=False)
        r = _run(self.base, self.out, "--from", thin, "--restore", "vision",
                 "--verify", "0")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("do not exist in", r.stderr + r.stdout)

    def test_shape_that_does_not_pad_to_the_trellis_is_refused(self):
        """Catches a transposed or differently-shaped dense original."""
        wide = _make_dense_ckpt(os.path.join(self.tmp.name, "wide"), fill=7,
                                out_f=OUT_F * 4, in_f=IN_F)
        r = _run(self.base, self.out, "--from", wide, "--restore", "vision",
                 "--verify", "0")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("does not pad to the trellis", r.stderr + r.stdout)

    def test_skipping_verification_says_so(self):
        r = _run(self.base, self.out, "--from", self.orig, "--restore", "vision",
                 "--verify", "0", "--dry-run")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--verify 0", r.stderr)

    def test_keep_reverts_a_restore(self):
        r = _run(self.base, self.out, "--from", self.orig, "--restore", "vision",
                 "--keep", "vision", "--verify", "0")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("selected no modules", r.stderr + r.stdout)


if __name__ == "__main__":
    unittest.main()
