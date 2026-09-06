"""Every place this reaches into vLLM, in one file on purpose.

Three of the five things this needs are documented extension points and are
registered normally. Two are patches, and they live here rather than scattered
through the code that uses them, so that "what does this plugin actually
monkeypatch" has a one-file answer -- for review, for the eventual move to its
own repo, and for noticing when upstream makes one of them unnecessary.

    registered      KVCacheSpecRegistry.register  spec and manager
    registered      register_backend(...)         the view, in a builder
    registered      vllm.general_plugins          startup

    patched         Attention.get_kv_cache_spec   choose the paged spec
    patched         GPUModelRunner.prepare_attn   sync the worker's row

**`Attention.get_kv_cache_spec`.** `customize_spec` looks like the intended
hook and is even called on the full-attention path -- but only to measure a
page size, after which the layer builds and returns a plain
`FullAttentionSpec` regardless. Its docstring calls itself "a temporary
compatibility API" and says the end state is for the backend to build the spec
directly (vllm#42449). So this patch has an upstream expiry date; check before
carrying it forward.

**`GPUModelRunner.prepare_attn`.** The worker's block table has to equal the
manager's logical mapping before `compute_slot_mappings` reads it positionally.
It does not, because restored blocks reach the worker through the *append*
channel while the manager places them at their index. There is no seam for
this: the scheduler-to-worker protocol can say "here are more blocks" and
cannot say "this block now lives at index i". Closing it properly means a field
on `SchedulerOutput`, which is the same protocol gap a query-aware policy will
need anyway.
"""

from __future__ import annotations

import os
from dataclasses import fields, replace

from .manager import build_manager_class, make_spec_class, register
from .policy import POLICIES


class Config:
    """How a deployment turns this on, without editing code.

    Environment rather than engine kwargs because a plugin loaded through an
    entry point has no argument to receive: it is constructed by vLLM before
    anything of ours runs.
    """

    PREFIX = "VLLM_KVPAGER_"

    def __init__(self, budget=0, sink=2, policy="recency", host_slots=1024,
                 verify=True):
        self.budget = budget
        self.sink = sink
        self.policy = policy
        self.host_slots = host_slots
        self.verify = verify

    @classmethod
    def from_env(cls, env=None) -> "Config":
        env = os.environ if env is None else env

        def get(name, default, cast):
            raw = env.get(cls.PREFIX + name)
            return default if raw is None or raw == "" else cast(raw)

        cfg = cls(
            budget=get("BUDGET", 0, int),
            sink=get("SINK", 2, int),
            policy=get("POLICY", "recency", str),
            host_slots=get("HOST_SLOTS", 1024, int),
            verify=get("VERIFY", True, lambda x: x not in ("0", "false", "no")),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(
                f"unknown policy {self.policy!r}; have {sorted(POLICIES)}")
        if self.budget < 0 or self.sink < 0:
            raise ValueError("budget and sink must be non-negative")
        if self.budget and self.sink > self.budget:
            raise ValueError(
                f"sink ({self.sink}) cannot exceed budget ({self.budget})")
        if self.host_slots <= 0:
            raise ValueError("host_slots must be positive")

    @property
    def enabled(self) -> bool:
        """A budget of zero is the control arm: wired up, evicting nothing."""
        return True

    def __repr__(self) -> str:
        return (f"Config(budget={self.budget}, sink={self.sink}, "
                f"policy={self.policy!r}, host_slots={self.host_slots}, "
                f"verify={self.verify})")


def patch_spec(config: Config):
    """Make full-attention layers ask for the paged spec. Returns the original.

    Sliding-window and other non-full specs are passed through untouched: their
    kernels rebuild key position from the block's index in the row, which
    permuting or compacting breaks -- measured, not assumed.
    """
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    paged_cls = make_spec_class()
    register(paged_cls, build_manager_class())
    original = Attention.get_kv_cache_spec

    def hooked(self, vllm_config):
        spec = original(self, vllm_config)
        if type(spec) is not FullAttentionSpec:
            return spec
        common = {f.name: getattr(spec, f.name) for f in fields(spec)}
        return paged_cls(**common, budget_blocks=config.budget,
                         sink_blocks=config.sink, policy_name=config.policy)

    hooked._kvpager_original = original
    Attention.get_kv_cache_spec = hooked
    return original


def unpatch_spec(original) -> None:
    from vllm.model_executor.layers.attention import Attention

    Attention.get_kv_cache_spec = original


def enable(config: Config | None = None):
    """Register everything and install the patches. The plugin entry point."""
    config = config or Config.from_env()
    original = patch_spec(config)
    return config, original


__all__ = ["Config", "enable", "patch_spec", "unpatch_spec", "replace"]
