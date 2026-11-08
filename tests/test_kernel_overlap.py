"""exllamav3's GEMM and MoE kernels must never overlap: they spin-wait across blocks and deadlock.

Found 2026-11-08 as a decode hang on an RTX 3060 under WSL: vLLM ran an EXL3 MoE layer's shared
experts on its side stream while the routed experts ran on the main stream. Three defences, each
tested here: the shared-expert overlap is switched off for EXL3 layers (`plugin._patch_kernel_overlap`),
microbatch overlap is refused at load (`config._refuse_microbatching`), and any other overlap is
refused at the first launch on a second stream within a forward (`ops._one_stream`).
"""
import types

import pytest
import torch


def _shared_experts_with(quant_method):
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts

    mlp = torch.nn.Module()
    mlp.down_proj = torch.nn.Module()
    mlp.down_proj.quant_method = quant_method
    fake = SharedExperts.__new__(SharedExperts)
    torch.nn.Module.__init__(fake)
    object.__setattr__(fake, "_layer", mlp)
    return fake


def test_exl3_shared_experts_never_overlap():
    """The shared experts' `_layer` is the shared MLP, not the MoE layer: an EXL3 linear in it
    is what would put our GEMM on the side stream, so its presence decides."""
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
        SharedExpertsOrder,
    )

    from vllm_exl3_plugin import plugin
    from vllm_exl3_plugin.quantization.linear import EXL3LinearMethod

    plugin._patch_kernel_overlap()
    fake = _shared_experts_with(EXL3LinearMethod.__new__(EXL3LinearMethod))
    order = SharedExperts._determine_shared_experts_order(fake, torch.empty(1, 8))
    assert order == SharedExpertsOrder.NO_OVERLAP


def test_microbatching_is_refused(monkeypatch):
    import vllm.config

    from vllm_exl3_plugin.quantization import config

    for ubatching in (False, True):
        monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda u=ubatching: types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(use_ubatching=u)))
        if ubatching:
            with pytest.raises(ValueError, match="microbatch"):
                config._refuse_microbatching()
        else:
            config._refuse_microbatching()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device for real streams")
def test_second_stream_in_one_forward_is_refused(monkeypatch):
    import vllm.forward_context as fc

    from vllm_exl3_plugin import ops

    ctx = [object()]
    monkeypatch.setattr(fc, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(fc, "get_forward_context", lambda: ctx[0])
    monkeypatch.setattr(ops, "_FORWARD_STREAM", [None, None])
    x = torch.empty(1, device="cuda")

    ops._one_stream(x)
    ops._one_stream(x)                      # same forward, same stream: fine
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        with pytest.raises(RuntimeError, match="two CUDA streams"):
            ops._one_stream(x)
        ctx[0] = object()                   # a new forward on another stream (graph capture): fine
        ops._one_stream(x)
