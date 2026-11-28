"""LoRA adapters on EXL3 models through vLLM's `--enable-lora`.

Unit (CPU): vLLM's LoRA wrappers build on EXL3 layers, which have no `.weight` (`stock_compat._lora_on_exl3`): the
device comes from the layer's own tensors, the embedding and head carry a stand-in weight only for the call that reads
it, other quantization methods still get vLLM's own lookup, and EXL3 routed experts are refused clearly.

End to end (GPU, opt-in by having the models cached), two checks:

- Per module (test_per_module): every LoRA-wrapped linear, through vLLM's own LoRA kernels, stacked weights, slices
  and token mapping, must add exactly the delta the adapter file defines (bf16 lands at ~0.0023 relative), that
  delta must reach the output, wrappers the adapter does not target must add nothing, and every adapter module must
  reach a wrapper. This is the gate: a dropped, swapped or unapplied projection family fails it (shown with injected
  serving faults, `--modules ... FAULT`; test_per_module_catches_dropped_family keeps one).
- Logprob shift (test_eager, test_cuda_graphs, test_tp2): the adapter must move EXL3's logprobs the way it moves
  bf16's (TP=2), teacher-forced on bf16+adapter sequences (tests/data/lora-ornith-swift-bf16.json); the shift,
  lora - base per token, cancels the base's quantization error. Measured 2026-11-27 at 4.00bpw: r~0.91, slope ~0.85,
  eager, CUDA graphs and TP=2 alike (6.00bpw r=0.95, 3.00bpw-H5 0.83). A smoke test only: it catches an adapter not
  applied or mis-scaled, but passed (r=0.92) with all 72 linear_attn deltas zeroed and (0.88) with all 32 down_proj
  ones -- Swift-1.5's whole effect (~0.02 per token) is below 4.00bpw's distance from bf16 (~0.04).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "lora-ornith-swift-bf16.json")
MODEL, REVISION = "yeasah/Ornith-1.5-9B-exl3", "4.00bpw"
ADAPTER = "nchapman/Ornith-1.5-9B-Swift-1.5-LoRA"
MIN_CORR, SLOPE = 0.85, (0.7, 1.2)


def _patched():
    from vllm.lora import model_manager
    from vllm.lora.layers import utils
    from vllm.lora.layers.vocab_parallel_embedding import VocabParallelEmbeddingWithLoRA

    from vllm_exl3_plugin import stock_compat

    stock_compat._lora_on_exl3(utils, VocabParallelEmbeddingWithLoRA, model_manager)
    return utils, VocabParallelEmbeddingWithLoRA, model_manager


def _layer(method, **attrs):
    """A module computing through `method`, holding one EXL3-named tensor on the meta device."""
    layer = torch.nn.Module()
    layer.quant_method = method
    layer.register_parameter("exl3_trellis_0", torch.nn.Parameter(torch.empty(4, 4, device="meta"),
                                                                  requires_grad=False))
    for k, v in attrs.items():
        setattr(layer, k, v)
    return layer


def _exl3(cls):
    return object.__new__(cls)   # only the type is consulted


class LoraSetupTest(unittest.TestCase):
    def setUp(self):
        self.utils, self.Embedding, self.manager = _patched()
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        from vllm_exl3_plugin.quantization.embedding import EXL3BlockQEmbeddingMethod
        from vllm_exl3_plugin.quantization.fused_moe import EXL3MoEMethod
        from vllm_exl3_plugin.quantization.lm_head import EXL3LMHeadMethod
        from vllm_exl3_plugin.quantization.linear import EXL3LinearMethod

        self.linear, self.head = _exl3(EXL3LinearMethod), _exl3(EXL3LMHeadMethod)
        self.blockq, self.moe = _exl3(EXL3BlockQEmbeddingMethod), _exl3(EXL3MoEMethod)
        self.other = _exl3(UnquantizedLinearMethod)
        self.lora_config = types.SimpleNamespace(max_lora_rank=4, lora_dtype=torch.bfloat16)

    def test_linear_device_from_its_tensors(self):
        from vllm.lora.layers import base_linear, fused_moe

        for module in (self.utils, base_linear, fused_moe):
            self.assertEqual(module._get_lora_device(_layer(self.linear)).type, "meta", module.__name__)

    def test_other_methods_keep_vllms_lookup(self):
        with self.assertRaisesRegex(ValueError, "Unsupported base layer"):
            self.utils._get_lora_device(_layer(self.other))

    def test_exl3_experts_refused(self):
        runner = torch.nn.Module()
        runner.routed_experts = _layer(self.moe)
        with self.assertRaisesRegex(NotImplementedError, "EXL3 routed experts"):
            self.utils._get_lora_device(runner)

    def test_embedding_stand_in_is_scoped(self):
        base = _layer(self.blockq, org_vocab_size=10, embedding_dim=6)
        wrapper = self.Embedding(base)
        wrapper.create_lora_weights(2, self.lora_config)
        self.assertEqual(wrapper.lora_a_stacked.device.type, "meta")
        self.assertEqual(tuple(wrapper.lora_b_stacked.shape), (2, 1, 6, 4))
        self.assertFalse(hasattr(base, "weight"), "stand-in weight outlived the call")

    def test_head_stand_in_is_scoped(self):
        lm_head = _layer(self.head, embedding_dim=6, get_sharded_to_full_mapping=lambda: None)
        processor = types.SimpleNamespace(vocab_size=10)
        from vllm.lora.layers import logits_processor

        with mock.patch.object(logits_processor, "get_tensor_model_parallel_world_size", lambda: 1), \
                mock.patch.object(logits_processor, "get_tensor_model_parallel_rank", lambda: 0):
            wrapper = self.manager.from_layer_logits_processor(processor, lm_head, 2, self.lora_config)
        self.assertEqual(wrapper.device.type, "meta")
        self.assertEqual(wrapper.dtype, torch.bfloat16)
        self.assertFalse(hasattr(lm_head, "weight"), "stand-in weight outlived the call")

    def test_idempotent(self):
        first = self.utils._get_lora_device
        _patched()
        self.assertIs(self.utils._get_lora_device, first)


def shift_fit(delta, reference):
    """(correlation, slope through the origin) of per-token logprob shifts against the reference's."""
    x = torch.tensor([v for row in reference for v in row[1:]], dtype=torch.float64)
    y = torch.tensor([v for row in delta for v in row[1:]], dtype=torch.float64)
    corr = torch.corrcoef(torch.stack([x, y]))[0, 1].item()
    return corr, (x @ y / (x @ x)).item()


class ShiftFitTest(unittest.TestCase):
    """The criterion fails the ways a broken application would look."""

    def setUp(self):
        self.ref = json.load(open(FIXTURE))["delta"]

    def test_reference_passes_itself(self):
        corr, slope = shift_fit(self.ref, self.ref)
        self.assertAlmostEqual(corr, 1.0)
        self.assertAlmostEqual(slope, 1.0)

    def test_adapter_not_applied(self):
        flat = [[None] + [0.0] * (len(r) - 1) for r in self.ref]
        _, slope = shift_fit(flat, self.ref)
        self.assertLess(slope, SLOPE[0])

    def test_scale_wrong(self):   # alpha/r applied twice, or not at all
        for k in (2.0, 0.5):
            _, slope = shift_fit([[None] + [k * v for v in r[1:]] for r in self.ref], self.ref)
            self.assertFalse(SLOPE[0] <= slope <= SLOPE[1], k)

    def test_shift_unrelated(self):
        g = torch.Generator().manual_seed(0)
        noise = [[None] + (torch.randn(len(r) - 1, generator=g) * 0.02).tolist() for r in self.ref]
        corr, _ = shift_fit(noise, self.ref)
        self.assertLess(corr, MIN_CORR)


def _cached(repo, revision=None):
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo, revision=revision, local_files_only=True)
    except Exception:
        return None


def _have_gpu():
    try:
        import exllamav3_ext  # noqa: F401
        import vllm  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def _deterministic_shrink_config() -> str:
    """A `VLLM_TUNED_CONFIG_FOLDER` holding vLLM's default LoRA shrink config with split_k=1, for this GPU.

    The file is indexed max_loras -> num_slices -> m -> k -> n, nearest key on all but num_slices, so one entry per
    slice count covers every call. Expand has no split and keeps its defaults (vLLM warns that it found no file)."""
    folder = tempfile.mkdtemp(prefix="test_lora-shrink-")
    name = torch.cuda.get_device_name().replace(" ", "_").replace("-", "_")
    config = {"block_m": 32, "block_n": 16, "block_k": 32, "split_k": 1, "num_warps": 4, "num_ctas": 1,
              "group_size_m": 8, "num_stages": 2, "max_nreg": None}
    with open(os.path.join(folder, f"{name}_SHRINK.json"), "w") as f:
        json.dump({"1": {str(s): {"1": {"1": {"1": config}}} for s in range(1, 9)}}, f)
    return folder


def run_arm(revision: str, eager: bool, tp: int, out: str, adapter: str | None = None) -> None:
    """One engine: per-token logprobs of the fixture's sequences, without and with the adapter, written to `out`."""
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # In-process engine: every request is queued before the first step, so the batch is the same every run. Through
    # the multiprocess engine requests reach the core over ZMQ and a step can start with only some of them; EXL3's
    # GEMM picks its partition by row count, so a different batch moved every token (mean |dlogprob| ~0.008) and the
    # correlation wandered 0.901-0.910 between identical runs. bf16 is invariant across most of those batches.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # And the LoRA shrink kernel at split-K 1: by default it splits K 8 or 64 ways and reduces with atomic_add, so the
    # adapter arm differed run to run (max |dlogprob| ~0.18 from token ~10 on) with the base arm bit-identical. Only
    # here: test_per_module checks the default kernels.
    os.environ["VLLM_TUNED_CONFIG_FOLDER"] = _deterministic_shrink_config()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    fixture = json.load(open(FIXTURE))
    llm = LLM(model=MODEL, revision=revision, tensor_parallel_size=tp, enable_lora=True, max_lora_rank=64,
              max_loras=1, max_model_len=4096, enforce_eager=eager, gpu_memory_utilization=0.85,
              limit_mm_per_prompt={"image": 0, "video": 0})
    lora = LoRARequest("swift", 1, adapter or _cached(ADAPTER))
    params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0)
    prompts = [TokensPrompt(prompt_token_ids=s) for s in fixture["sequences"]]
    result = {}
    for name, request in (("base", None), ("lora", lora)):
        outputs = llm.generate(prompts, params, lora_request=request)
        result[name] = [[None] + [d[t].logprob for d, t in zip(o.prompt_logprobs[1:], s[1:])]
                        for o, s in zip(outputs, fixture["sequences"])]
    json.dump(result, open(out, "w"))


_MODULE_CHECK = {"on": False, "adapter": None, "scale": 1.0, "packed": {}, "stats": {}, "fault": None}


def _module_hooks(model):
    """Forward hooks on every LoRA-wrapped linear, checking two things against the adapter file's A and B in fp32.

    The delta cannot be read off the output: it is added into a bf16 output it is far smaller than (Swift-1.5 on
    o_proj: mean |delta| 6.5e-5 against |y| 0.023, under one ulp), so y - y_base is mostly rounding -- 0.488 relative
    error on layer 3's o_proj, exactly the bf16 floor. Instead:
      - replay: vLLM's own LoRA kernels rerun into a zero buffer, through the wrapper's stacked weights, slices and
        token mapping, must equal x @ A^T @ B^T * alpha/r (`rel_err`);
      - applied: the output must be y_base plus that delta as bf16 rounds it (`applied_err`, against the same floor);
        a wrapper that skipped the add scores 1.
    Untargeted wrappers must add exactly nothing. Runs inside the engine (`apply_model`), so it reads the check's
    state from module globals."""
    from vllm.lora.layers import BaseLayerWithLoRA
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

    state = _MODULE_CHECK
    state["packed"] = dict(getattr(model, "packed_modules_mapping", {}))

    def rel(a, b):
        return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()

    def hook(name, wrapper):
        def f(module, args, out):
            if not state["on"]:
                return
            x = args[0]
            y = out[0] if isinstance(out, tuple) else out
            base = wrapper.base_layer
            leaf = name.rsplit(".", 1)[-1]
            if isinstance(base, VocabParallelEmbedding):   # untargeted here: must add exactly nothing
                state["stats"][name] = {"family": leaf, "targeted": False,
                                        "max_abs": (y - base(x)).float().abs().max().item()}
                return
            y_base = base.quant_method.apply(base, x)
            if getattr(base, "bias", None) is not None and not getattr(base, "skip_bias_add", False):
                y_base = y_base + base.bias
            parts = state["packed"].get(leaf, [leaf])
            site = name.split("layers.", 1)[-1].rsplit(".", 1)[0]   # "<n>.<block>"
            pieces = []
            for part in parts:
                ab = state["adapter"].get(f"{site}.{part}")
                pieces.append(None if ab is None else (x.float() @ ab[0].T) @ ab[1].T * state["scale"])
            if all(p is None for p in pieces):
                state["stats"][name] = {"family": leaf, "targeted": False,
                                        "max_abs": (y - y_base).float().abs().max().item()}
                return
            if any(p is None for p in pieces):
                raise AssertionError(f"{name}: adapter covers only part of {parts}")
            expected = torch.cat(pieces, dim=-1)
            replay = torch.zeros_like(y)
            wrapper.punica_wrapper.add_lora_linear(replay, x, wrapper.lora_a_stacked, wrapper.lora_b_stacked, 1.0,
                                                   wrapper.output_slices)
            floor = (y_base.float() + replay.float()).to(y.dtype).float() - y_base.float()
            state["stats"][name] = {"family": leaf, "targeted": True, "parts": len(parts),
                                    "rel_err": rel(replay.float(), expected),
                                    "applied_err": rel((y - y_base).float(), floor)}
        return f

    fault, _, family = (state.get("fault") or "").partition(":")

    def inject(name, wrapper):
        """Test-only serving faults (`--fault`), for proving the check fails: `drop` zeroes a family's stacked B
        after the adapter loads, `swap` exchanges a packed family's A slices, `skip` returns the base output."""
        done = []

        def pre(module, args):
            if not state["on"] or done:
                return
            done.append(True)
            if fault == "drop":
                for b in wrapper.lora_b_stacked:
                    b.zero_()
            elif fault == "swap":   # first and last: in_proj_qkvz's q, k and v slices share in_proj_qkv's A
                a = wrapper.lora_a_stacked
                first = a[0].clone()
                a[0].copy_(a[-1])
                a[-1].copy_(first)

        def skip(module, args, out):
            if state["on"]:
                base = wrapper.base_layer
                y = base.quant_method.apply(base, args[0])
                return (y, *out[1:]) if isinstance(out, tuple) else y

        if fault in ("drop", "swap"):
            wrapper.register_forward_pre_hook(pre)
        elif fault == "skip":
            wrapper.register_forward_hook(skip)

    for name, module in model.named_modules():
        if isinstance(module, BaseLayerWithLoRA) and isinstance(getattr(module, "base_layer", None),
                                                                 (LinearBase, VocabParallelEmbedding)):
            if fault and name.rsplit(".", 1)[-1] == family:
                inject(name, module)
            module.register_forward_hook(hook(name, module))
    return 0


def _module_stats(model):
    return dict(_MODULE_CHECK["stats"])


def run_module_check(revision: str, out: str, adapter: str | None = None, fault: str | None = None) -> None:
    """Per LoRA-wrapped linear, at TP=1: the delta the wrapper adds, against x @ A^T @ B^T * alpha/r from the adapter
    file, over one fixture sequence's prefill. Untargeted wrappers must add exactly nothing."""
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from safetensors.torch import load_file
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    path = adapter or _cached(ADAPTER)
    config = json.load(open(os.path.join(path, "adapter_config.json")))
    assert not config.get("use_rslora") and not config.get("use_dora"), "plain LoRA scaling only"
    weights = load_file(os.path.join(path, "adapter_model.safetensors"), device="cuda")
    grouped = {}
    for key, value in weights.items():
        module, _, which = key.rpartition(".lora_")
        grouped.setdefault(module.split("layers.", 1)[-1], {})[which.split(".")[0]] = value.float()
    _MODULE_CHECK["adapter"] = {site: (ab["A"], ab["B"]) for site, ab in grouped.items()}
    _MODULE_CHECK["scale"] = config["lora_alpha"] / config["r"]
    _MODULE_CHECK["fault"] = fault

    llm = LLM(model=MODEL, revision=revision, enable_lora=True, max_lora_rank=64, max_loras=1, max_model_len=4096,
              enforce_eager=True, gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 0, "video": 0},
              enable_prefix_caching=False)
    llm.apply_model(_module_hooks)
    sequence = json.load(open(FIXTURE))["sequences"][0]
    _MODULE_CHECK["on"] = True
    llm.generate([TokensPrompt(prompt_token_ids=sequence)], SamplingParams(temperature=0, max_tokens=1),
                 lora_request=LoRARequest("swift", 1, path))
    _MODULE_CHECK["on"] = False
    json.dump(llm.apply_model(_module_stats)[0], open(out, "w"))


def _spawn_arm(revision, eager, tp, adapter=None):
    """`run_arm` in its own process: one engine per process, so each arm starts from a clean device."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "arm.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--arm", revision, "eager" if eager else "graphs", str(tp),
               out] + ([adapter] if adapter else [])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if proc.returncode:
            # kept: TP=1 arms aborted (SIGABRT) mid-load twice on 2026-11-27 and not in four runs after
            fd, log = tempfile.mkstemp(prefix="test_lora-arm-", suffix=".log")
            with os.fdopen(fd, "w") as f:
                f.write(proc.stdout + proc.stderr)
            raise AssertionError(f"arm failed ({proc.returncode}), full log {log}:\n{proc.stderr[-4000:]}")
        result = json.load(open(out))
    return [[None] + [lo - b for b, lo in zip(bs[1:], ls[1:])] for bs, ls in zip(result["base"], result["lora"])]


def _spawn_modules(revision, fault=None):
    """`run_module_check` in its own process; the per-wrapper stats."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "modules.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--modules", revision, out, "-"] + ([fault] if fault else [])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if proc.returncode:
            fd, log = tempfile.mkstemp(prefix="test_lora-modules-", suffix=".log")
            with os.fdopen(fd, "w") as f:
                f.write(proc.stdout + proc.stderr)
            raise AssertionError(f"module check failed ({proc.returncode}), full log {log}:\n{proc.stderr[-4000:]}")
        return json.load(open(out))


#: replay vs the adapter file, and output vs base + delta, both relative; bf16 lands at ~0.0023 and <0.0022
MODULE_TOL = 0.01


def module_failures(stats, adapter_modules):
    """What the per-module check objects to: wrong deltas, unapplied deltas, leakage, adapter modules never seen."""
    bad = [f"{n}: replay rel_err {v['rel_err']:.3g}" for n, v in stats.items() if v["targeted"] and v["rel_err"] > MODULE_TOL]
    bad += [f"{n}: applied_err {v['applied_err']:.3g}" for n, v in stats.items()
            if v["targeted"] and v["applied_err"] > MODULE_TOL]
    bad += [f"{n}: untargeted but added {v['max_abs']:.3g}" for n, v in stats.items() if not v["targeted"] and v["max_abs"]]
    covered = sum(v["parts"] for v in stats.values() if v["targeted"])
    if covered != adapter_modules:
        bad.append(f"{covered} of {adapter_modules} adapter modules reached a wrapper")
    return bad


@unittest.skipUnless(_have_gpu() and _cached(MODEL, REVISION) and _cached(ADAPTER),
                     f"needs CUDA, exllamav3, vLLM and {MODEL}@{REVISION} + {ADAPTER} in the HF cache")
class LoraEndToEndTest(unittest.TestCase):
    def setUp(self):
        self.ref = json.load(open(FIXTURE))["delta"]

    def check(self, delta):
        corr, slope = shift_fit(delta, self.ref)
        print(f"\n  corr={corr:.3f} slope={slope:.3f}", file=sys.stderr)
        self.assertGreaterEqual(corr, MIN_CORR)
        self.assertTrue(SLOPE[0] <= slope <= SLOPE[1], slope)

    def test_eager(self):
        self.check(_spawn_arm(REVISION, True, 1))

    def test_cuda_graphs(self):
        self.check(_spawn_arm(REVISION, False, 1))

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "needs two GPUs")
    def test_tp2(self):
        self.check(_spawn_arm(REVISION, False, 2))

    def _adapter_modules(self):
        return len(json.load(open(os.path.join(_cached(ADAPTER), "adapter_config.json")))["target_modules"])

    def test_per_module(self):
        """Every adapter module's delta, exactly as the file defines it, on every wrapper; nothing elsewhere."""
        bad = module_failures(_spawn_modules(REVISION), self._adapter_modules())
        self.assertEqual(bad, [], "\n".join(bad[:20]))

    def test_per_module_catches_dropped_family(self):
        """The check fails on a serving-side fault the logprob-shift fit passes (r=0.92 with all of linear_attn's
        deltas gone): in_proj_qkvz's stacked B zeroed after the adapter loads."""
        bad = module_failures(_spawn_modules(REVISION, "drop:in_proj_qkvz"), self._adapter_modules())
        self.assertEqual(len(bad), 24, "\n".join(bad[:30]))
        self.assertTrue(all("in_proj_qkvz" in b for b in bad))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--modules":
        # --modules REVISION OUT [ADAPTER|-] [FAULT]
        rest = sys.argv[4:]
        run_module_check(sys.argv[2], sys.argv[3], None if not rest or rest[0] == "-" else rest[0],
                         rest[1] if len(rest) > 1 else None)
    elif len(sys.argv) > 1 and sys.argv[1] == "--arm":
        revision, mode, tp, out = sys.argv[2:6]
        run_arm(revision, mode == "eager", int(tp), out, *(sys.argv[6:7]))
    else:
        unittest.main()
