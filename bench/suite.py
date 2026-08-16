"""The matrix: which engine configurations the gate covers, and what each is for.

An entry earns its place by exercising a plugin surface no other entry reaches.
`exercises` is not decoration -- it is the argument for keeping the entry when
the suite gets slow, and the first thing to read when an entry fails.

Every entry runs in its own process. vLLM has no supported way to stand up
several engines back to back in one process, and today's session added a second
reason: a crashed EngineCore leaves the parent deadlocked on a zombie, which
process isolation plus a timeout contains.

The `full` tier is not yet populated -- MoE, `mul1`, the Transformers backend and
TP all need entries, and they get added against a gate that is already working
rather than as part of standing it up. See TODO `bench-suite`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Entry:
    label: str
    model: str
    revision: str
    exercises: str
    tier: str = "fast"
    model_impl: str = "auto"
    enforce_eager: bool = True
    tensor_parallel_size: int = 1
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.85
    #: `EXL3_*` switches applied only while this entry's engine is built.
    env: dict = field(default_factory=dict)
    #: Per-entry threshold overrides; see bench/run.py for the defaults and the
    #: reasoning behind them.
    tolerance: dict = field(default_factory=dict)
    #: Why this configuration cannot currently be captured. An entry is kept and
    #: kept running rather than deleted -- deleting it loses the coverage and the
    #: knowledge -- but it cannot be blessed, and `check` reports it as a known
    #: failure instead of failing the gate. Clearing this field is how the fix
    #: gets verified.
    known_broken: str | None = None

    @property
    def name(self) -> str:
        """Filesystem-safe identity, and the baseline filename."""
        return self.label.replace(" ", "-").replace("/", "-").replace(",", "")


#: Both execution modes for one checkpoint, because an eager-only result says
#: nothing about the CUDA-graph path -- capture and replay is where a change to
#: vLLM's graph handling would show up, and only there.
ENTRIES: list[Entry] = [
    Entry(
        label="qwen3-0.6B 3.0bpw eager",
        model="turboderp/Qwen3-0.6B-exl3",
        revision="3.0bpw",
        exercises="uniform K=3, the simplest possible target; the baseline "
        "against which the other Qwen3-0.6B entries isolate one variable each",
    ),
    Entry(
        label="qwen3-0.6B 3.0bpw graphs",
        model="turboderp/Qwen3-0.6B-exl3",
        revision="3.0bpw",
        enforce_eager=False,
        exercises="the CUDA-graph capture/replay path, identical weights to the "
        "eager entry so any difference is execution mode alone",
        known_broken="EXL3EmbeddingMethod does not survive torch.compile. "
        "`ops.embed_rows` calls torch.unique (data-dependent output shape) and "
        "then loops with a data-dependent bound (`range(0, blocks.numel(), ...)`), "
        "neither of which dynamo can trace, so EngineCore dies during startup. "
        "Qwen3-0.6B is tied, so the quantized-embedding path is on by default. "
        "Isolated: the same entry with EXL3_DENSE_EMBED=1 captures fine. This is "
        "vLLM's *default* execution mode, and docs/embeddings.md's tied-path "
        "measurements were all taken eager. See TODO `embed-rows-compile`.",
    ),
    Entry(
        label="qwen3-0.6B 3.5bpw mixed",
        model="turboderp/Qwen3-0.6B-exl3",
        revision="3.5bpw",
        exercises="bit widths mixed *inside* a layer (q=4, k=5, v=5), so it is "
        "the entry that fails if merged-linear shards get concatenated or "
        "assumed to share a bit width",
    ),
    Entry(
        label="llama-3.2-1B 3.0bpw tied",
        model="turboderp/Llama-3.2-1B-Instruct-exl3",
        revision="3.0bpw",
        exercises="tied embeddings: EXL3EmbeddingMethod serves the embedding "
        "from the quantized lm_head and never loads fp16 embed_tokens. The "
        "weight-bytes gate is the point of this entry -- a broken tie_weights "
        "or tied-skip mapper costs VRAM while leaving logits correct",
    ),
    Entry(
        label="minicpm5-1B 3.0bpw mcg",
        model="turboderp/MiniCPM5-1B-exl3",
        revision="3.00bpw",
        exercises="the mcg codebook, an untied model, and a separately "
        "quantized 6-bit head -- none of which the Qwen3/Llama entries reach",
    ),
]


def by_tier(tier: str | None) -> list[Entry]:
    if tier in (None, "all"):
        return list(ENTRIES)
    return [e for e in ENTRIES if e.tier == tier]


def by_name(name: str) -> Entry:
    for e in ENTRIES:
        if e.name == name:
            return e
    raise SystemExit(f"no such entry: {name!r}")
