"""expert_store: a routed expert's trellises read from the checkpoint shards, as the disk tier will read them."""

import json
import os

import torch
from safetensors.torch import save_file

from vllm_exl3_plugin.expert_store import ExpertStore


def _checkpoint(tmp_path):
    """Two shards; each expert's three trellises separated by small scale tensors, so no offset is page-aligned."""
    g = torch.Generator().manual_seed(0)
    shards, weight_map = [{}, {}], {}
    for layer in range(2):
        for e in range(3):
            for lab in ("down", "gate", "up"):
                base = f"model.layers.{layer}.mlp.experts.{e}.{lab}_proj"
                shard = shards[layer]
                shard[base + ".suh"] = torch.randn(37, generator=g).half()
                shard[base + ".trellis"] = torch.randint(-30000, 30000, (5, 7, 32), generator=g, dtype=torch.int16)
                weight_map[base + ".suh"] = weight_map[base + ".trellis"] = f"model-0000{layer + 1}-of-00002.safetensors"
    # an expert missing a projection is not indexed (nothing to read it as)
    shards[1]["model.layers.1.mlp.experts.9.gate_proj.trellis"] = torch.zeros((5, 7, 32), dtype=torch.int16)
    weight_map["model.layers.1.mlp.experts.9.gate_proj.trellis"] = "model-00002-of-00002.safetensors"
    for i, sh in enumerate(shards):
        save_file(sh, str(tmp_path / f"model-0000{i + 1}-of-00002.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return shards


def test_reads_match_the_stored_tensors(tmp_path):
    shards = _checkpoint(tmp_path)
    st = ExpertStore(str(tmp_path))
    assert sorted(st.index) == [(f"model.layers.{l}.mlp.experts", e) for l in range(2) for e in range(3)]
    for (prefix, e) in st.index:
        layer = int(prefix.split(".")[2])
        got = st.read(prefix, e)
        for lab in ("gate", "up", "down"):
            assert torch.equal(got[lab], shards[layer][f"{prefix}.{e}.{lab}_proj.trellis"]), (prefix, e, lab)
    assert st.reads == 6


def test_reads_into_given_tensors(tmp_path):
    shards = _checkpoint(tmp_path)
    st = ExpertStore(str(tmp_path))
    out = {lab: torch.empty((5, 7, 32), dtype=torch.int16) for lab in ("gate", "up", "down")}
    got = st.read("model.layers.0.mlp.experts", 2, out)
    assert all(got[lab] is out[lab] for lab in out)
    assert torch.equal(out["up"], shards[0]["model.layers.0.mlp.experts.2.up_proj.trellis"])
