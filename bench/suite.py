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
        "eager entry so any difference is execution mode alone. This entry is "
        "why `ops.embed_rows` has a capture-safe path at all: it found that "
        "serving a tied model's embedding did not survive torch.compile, on "
        "vLLM's default execution mode. Its baseline is its own -- eager and "
        "graphs differ by ~0.157 nats on this model for reasons that have "
        "nothing to do with the plugin",
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
        model_impl="auto",
        exercises="the mcg codebook, an untied model, and a separately "
        "quantized 6-bit head -- none of which the Qwen3/Llama entries reach. "
        "Paired with the transformers-backend entry below: together the two "
        "are the claim that the backend is token-for-token identical to the "
        "native implementation",
    ),
    Entry(
        label="minicpm5-1B 3.0bpw via transformers backend",
        model="turboderp/MiniCPM5-1B-exl3",
        revision="3.00bpw",
        model_impl="transformers",
        exercises="the Transformers backend on a text-only model, which needs "
        "no patches. `model_impl` is pinned rather than left to dispatch, so "
        "this keeps testing the backend no matter what vLLM later implements "
        "natively -- the coverage is of our integration, not of vLLM's routing",
    ),
    # ---- full tier: the surfaces the fast tier cannot reach on a 16 GiB card
    # in a couple of minutes. Same gate, run before a bump rather than casually.
    Entry(
        label="qwen3.5-35B-A3B 2.0bpw MoE",
        model="turboderp/Qwen3.5-35B-A3B-exl3",
        revision="2.00bpw",
        tier="full",
        gpu_memory_utilization=0.92,
        exercises="the MoE path: exl3_mgemm behind FusedMoE, the w13/w2 "
        "expert-tensor mapping, and the recovered scale factor. Nothing in the "
        "fast tier touches fused_moe at all",
    ),
    Entry(
        label="gemma-4-12B 3.0bpw mul1 tied",
        model="turboderp/gemma-4-12B-it-exl3",
        revision="3.00bpw_mul1",
        tier="full",
        gpu_memory_utilization=0.92,
        exercises="the mul1 codebook, and the gemma4-style two-module tied "
        "shape where a separate ParallelLMHead is pointed at the embedding's "
        "storage -- a different tie path from Llama-3.2-1B's. Also the only "
        "entry needing patches/vllm-gemma4-transformers-5.15-per-layer.patch, "
        "so it is what verifies TODO `retire-gemma4-patch` when the pin moves",
    ),
    Entry(
        label="muse-glimmer-30B 2.0bpw via transformers backend",
        model="turboderp/Muse-Glimmer-30B-exl3",
        revision="2.00bpw",
        tier="full",
        model_impl="transformers",
        gpu_memory_utilization=0.92,
        exercises="the multimodal backend path, and the only exerciser of "
        "three things: the quantized vision tower, the safetensors index as "
        "ground truth for is_quantized (this checkpoint's tensor_storage omits "
        "all 303 vision-tower modules), and "
        "patches/vllm-replicated-linear-weight-loader-v2.patch, which 154 "
        "unsharded submodules here depend on.\n\n"
        "Two predictions this entry exists to test at the 0.27.2 bump, both "
        "from docs/transformers-backend.md. vLLM PR #51247 fixes the dropped "
        "embed_norm by a *different* mechanism than our patch (rebasing the "
        "subclass onto VocabParallelEmbedding rather than wrapping it), so "
        "logprobs should be unchanged -- if they move, the two implementations "
        "disagree. PR #52173 applies the soft cap but reads only `logit_scale`, "
        "never MuseGlimmer's `output_multiplier`, and applies its scale after "
        "the cap rather than before; so this entry is *expected to fail on "
        "logprobs* after the bump. That failure is the point: it converts an "
        "analysis into a checked claim. MuseGlimmer goes native in 0.27.2, "
        "which is exactly why model_impl is pinned here",
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
