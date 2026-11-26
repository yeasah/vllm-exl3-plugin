"""The vital vLLM fixes as plugin monkeypatches (stock_compat), on stand-ins for the vLLM classes they patch: each
acts for EXL3 only and hands everything else to vLLM's own code, untouched."""
import types

import pytest
import torch

from vllm_exl3_plugin import plugin, stock_compat
from vllm_exl3_plugin.quantization.linear import EXL3Parameter


class _Exl3Config:
    def get_name(self):
        return "exl3"


class _OtherConfig:
    def get_name(self):
        return "fp8"


def _embedding_class():
    class Embedding:
        def __init__(self, num_embeddings, embedding_dim, params_dtype=None, org_num_embeddings=None,
                     padding_size=64, quant_config=None, prefix="", *, disable_tp=False, quant_method=None):
            self.quant_config, self.quant_method = quant_config, quant_method

    stock_compat._embedding_quant_config_default(Embedding)
    return Embedding


@pytest.mark.parametrize("ambient, wanted", [(_Exl3Config(), "exl3"), (_OtherConfig(), None), (None, None)])
def test_embedding_takes_the_ambient_config_only_under_exl3(monkeypatch, ambient, wanted):
    import vllm.config

    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: types.SimpleNamespace(quant_config=ambient))
    e = _embedding_class()(100, 8, prefix="model.embed_tokens")
    assert (e.quant_config.get_name() if e.quant_config else None) == wanted


def test_embedding_keeps_what_the_model_passed(monkeypatch):
    import vllm.config

    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: types.SimpleNamespace(quant_config=_Exl3Config()))
    Embedding = _embedding_class()
    own = _OtherConfig()
    assert Embedding(100, 8, quant_config=own).quant_config is own
    assert Embedding(100, 8, 64, quant_method=object()).quant_config is None, "a preselected method wins"


def _param():
    # built without __init__, which asks for the TP rank: the patches only test the type
    return EXL3Parameter.__new__(EXL3Parameter)


class _Merged:
    calls: list

    def validate_shard_id(self, shard_id, param):
        pass

    def weight_loader_v2(self, param, loaded_weight, loaded_shard_id=None):
        self.calls.append(("vllm", loaded_shard_id))


class _Qkv(_Merged):
    pass


def test_fused_shards_reach_an_exl3_parameter_whole(monkeypatch):
    stock_compat._fused_shard_loading(_Merged, _Qkv)
    seen = []
    monkeypatch.setattr(EXL3Parameter, "load_merged_column_weight", lambda self, loaded_weight, **kw: seen.append(kw))
    monkeypatch.setattr(EXL3Parameter, "load_qkv_weight", lambda self, loaded_weight, **kw: seen.append(kw))
    m, q = _Merged(), _Qkv()
    m.calls, q.calls = [], []
    t = torch.zeros(4)
    m.weight_loader_v2(_param(), t, (0, 1))
    m.weight_loader_v2(_param(), t, None)
    q.weight_loader_v2(_param(), t, None)
    assert seen == [{"shard_id": (0, 1)}, {"shard_id": None}, {"shard_id": None}]
    m.weight_loader_v2(_param(), t, 1)                       # one shard: vLLM's own path narrows nothing here
    q.weight_loader_v2(_param(), t, "k")
    m.weight_loader_v2(torch.nn.Parameter(t), t, None)       # not EXL3: vLLM's
    assert m.calls == [("vllm", 1), ("vllm", None)] and q.calls == [("vllm", "k")]


def test_replicated_linear_hands_an_exl3_parameter_the_whole_tensor(monkeypatch):
    class Replicated:
        def weight_loader(self, param, loaded_weight):
            self.v1 = loaded_weight

    stock_compat._replicated_linear_loading(Replicated)
    seen = []
    # stored whole: not through load_row_parallel_weight, which slices by tp_rank (test_validate_tp.py, rank 1)
    monkeypatch.setattr(EXL3Parameter, "store", lambda self, loaded_weight, shard_id=0: seen.append(loaded_weight))
    r = Replicated()
    r.weight_loader(_param(), torch.tensor(3.0))
    assert seen[0].shape == (1,), "a scalar off disk is given a dimension"
    plain = torch.ones(2)
    r.weight_loader(torch.nn.Parameter(torch.zeros(2)), plain)
    assert r.v1 is plain


def test_patches_wrap_once():
    class Replicated:
        def weight_loader(self, param, loaded_weight):
            pass

    stock_compat._replicated_linear_loading(Replicated)
    first = Replicated.weight_loader
    stock_compat._replicated_linear_loading(Replicated)
    assert Replicated.weight_loader is first


def test_fused_shards_check_the_shard_id_as_the_running_vllm_spells_it(monkeypatch):
    """v0.32's validate_shard_id takes the shard id alone, v0.34's the parameter too; calling the new form on the
    old one broke every fused EXL3 load on the 0.32 fork."""
    class Old(_Merged):
        def validate_shard_id(self, shard_id):
            self.validated = shard_id

    class New(_Merged):
        def validate_shard_id(self, shard_id, param):
            self.validated = (shard_id, type(param).__name__)

    class Qkv(_Merged):
        pass

    monkeypatch.setattr(EXL3Parameter, "load_merged_column_weight", lambda self, loaded_weight, **kw: None)
    for cls, want in ((Old, (0, 1)), (New, ((0, 1), "EXL3Parameter"))):
        stock_compat._fused_shard_loading(cls, Qkv)
        layer = cls()
        layer.calls = []
        layer.weight_loader_v2(_param(), torch.zeros(4), (0, 1))
        assert layer.validated == want


def test_importing_exllamav3_under_the_plugin_leaves_the_allocator_alone(monkeypatch):
    """The lazy `import exllamav3` in the MoE path used to switch on expandable segments mid-process."""
    import importlib
    import sys

    monkeypatch.delenv("EXL3_EXPANDABLE_SEGMENTS", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    plugin._leave_allocator_alone()
    import os

    assert os.environ["EXL3_EXPANDABLE_SEGMENTS"] == "0"
    calls = []
    monkeypatch.setattr(torch._C, "_accelerator_setAllocatorSettings", lambda s: calls.append(s), raising=False)
    sys.modules.pop("exllamav3", None)
    importlib.import_module("exllamav3")
    assert calls == [], "exllamav3 changed the allocator settings"
    monkeypatch.setenv("EXL3_EXPANDABLE_SEGMENTS", "1")
    sys.modules.pop("exllamav3", None)
    importlib.import_module("exllamav3")
    assert calls == ["expandable_segments:True"], "the probe must see exllamav3's setter when it is allowed"


def test_transformers_backend_takes_an_output_multiplier_before_the_cap(monkeypatch):
    """MuseGlimmer spells its logit multiplier output_multiplier, which the backend's processor never applies, and
    applies a scale only after the cap; under an EXL3 config it is folded into the cap instead."""
    import vllm.config
    import vllm.model_executor.layers.logits_processor as lp

    class LogitsProcessor:   # vLLM's reads the current config when built; the arithmetic is what is tested
        def __init__(self, vocab_size, scale=1.0, soft_cap=None):
            self.scale, self.soft_cap = scale, soft_cap

    monkeypatch.setattr(lp, "LogitsProcessor", LogitsProcessor)

    class Causal:
        def __init__(self, text_config):
            self.text_config = text_config
            self.logits_processor = LogitsProcessor(text_config.vocab_size, scale=1.0,
                                                    soft_cap=text_config.final_logit_softcapping)

    stock_compat._transformers_logit_multiplier(Causal)
    muse = types.SimpleNamespace(vocab_size=8, output_multiplier=0.5, final_logit_softcapping=30.0)
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: types.SimpleNamespace(quant_config=_Exl3Config()))
    c = Causal(muse)
    z = torch.linspace(-80, 80, 9)
    want = 30.0 * torch.tanh(z * 0.5 / 30.0)
    got = torch.tanh(z / c.logits_processor.soft_cap) * c.logits_processor.soft_cap * c.logits_processor.scale
    assert torch.allclose(got, want)
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: types.SimpleNamespace(quant_config=_OtherConfig()))
    assert Causal(muse).logits_processor.scale == 1.0, "only under EXL3"
    plain = types.SimpleNamespace(vocab_size=8, final_logit_softcapping=30.0)
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: types.SimpleNamespace(quant_config=_Exl3Config()))
    assert Causal(plain).logits_processor.soft_cap == 30.0, "no multiplier, nothing changes"




def _map(mapper, name):
    """(name, shard id), from whichever of WeightsMapper's two spellings the running vLLM has."""
    with_shard = getattr(mapper, "_map_name_with_shard", None)
    return with_shard(name) if with_shard is not None else mapper._map_name(name)


def test_embedding_routes_follow_the_models_module_name():
    """exllamav3 names the embedding `embed_tokens` whatever the model calls it; the config's routes (dense, the
    on-load blockq source, a stored blockq or trellis table) end on the module the model built -- Spark-X2.6's
    `model.embedding` -- and are left alone where the names agree."""
    import types

    from vllm.model_executor.models.utils import WeightsMapper

    from vllm_exl3_plugin.quantization.config import EXL3Config

    def mapped(embed_prefix, routes, name):
        cfg = types.SimpleNamespace(embed_prefix=embed_prefix, _EMBED_LEAVES=EXL3Config._EMBED_LEAVES)
        m = EXL3Config._embedding_leaf_rename(cfg, WeightsMapper(orig_to_new_suffix=routes))
        return _map(m, name)[0]

    onload = {".embed_tokens.weight": ".embed_tokens.bq_src"}
    assert mapped("model.embedding", onload, "model.embed_tokens.weight") == "model.embedding.bq_src"
    assert mapped("model.embedding", {}, "model.embed_tokens.weight") == "model.embedding.weight"
    assert mapped("model.embedding", {}, "model.embed_tokens.bq_q") == "model.embedding.bq_q"
    assert mapped("model.embedding", {}, "model.layers.0.mlp.up_proj.trellis") == "model.layers.0.mlp.up_proj.trellis"
    assert mapped("model.embed_tokens", onload, "model.embed_tokens.weight") == "model.embed_tokens.bq_src"


def test_spark_exl3_names_map_split_qkv_onto_the_fused_projection():
    """Spark-X2.6 under its maker's plugin: an EXL3 checkpoint's split q/k/v land on the fused `q_k_v_proj` by shard
    id; the reference checkpoint's names, and any model not under an EXL3 config, are left as the plugin maps them."""
    import types

    from vllm.model_executor.models.utils import WeightsMapper

    class Spark:
        hf_to_vllm_mapper = WeightsMapper(orig_to_new_stacked={".gate_proj": (".gate_up_proj", 0)})

        def __init__(self, *, vllm_config, prefix=""):
            pass

    stock_compat._spark_exl3_names(Spark)
    exl3 = types.SimpleNamespace(get_name=lambda: "exl3")
    m = Spark(vllm_config=types.SimpleNamespace(quant_config=exl3)).hf_to_vllm_mapper
    assert _map(m, "model.layers.3.self_attn.k_proj.trellis") == ("model.layers.3.self_attn.q_k_v_proj.trellis", "k")
    assert _map(m, "model.layers.3.self_attn.q_k_v_proj.weight") == ("model.layers.3.self_attn.q_k_v_proj.weight", None)
    assert _map(m, "model.layers.3.mlp.gate_proj.trellis")[1] == 0
    plain = Spark(vllm_config=types.SimpleNamespace(quant_config=None))
    assert plain.hf_to_vllm_mapper is Spark.hf_to_vllm_mapper
