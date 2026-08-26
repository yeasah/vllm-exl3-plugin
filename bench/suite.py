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
    #: KV cache dtype, e.g. "turboquant_4bit_nc" or "fp8". None leaves vLLM's
    #: default ("auto"), which is what every entry used before these landed.
    kv_cache_dtype: str | None = None
    #: Speculative decoding, e.g. {"method": "mtp", "num_speculative_tokens": 3}.
    #: An MTP drafter is a *second* model instance whose modules go through
    #: `get_quant_method` independently -- coverage no single-model entry has.
    speculative_config: dict | None = None
    #: Skip the vision/audio tower. Not a shortcut: it is how these checkpoints
    #: are actually served on a 16 GiB card (see ~/ckpt/run-*.sh), and it is the
    #: only way some of them fit at all alongside a KV cache.
    language_model_only: bool = False
    #: Recipe for a checkpoint the Hub does not hold, derived from `model`
    #: at `revision` before capture. Only "blockq" exists: a block-quantized
    #: token embedding produced by `tools/quantize_embedding.py`. See
    #: bench/fixtures.py for why these are derived rather than published.
    fixture: str | None = None
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
    Entry(
        label="minicpm5-1B 3.0bpw blockq eager",
        model="turboderp/MiniCPM5-1B-exl3",
        revision="3.00bpw",
        fixture="blockq",
        exercises="the block-quantized embedding end to end: "
        "EXL3BlockQEmbeddingMethod gathering rows out of the packed table "
        "without ever materializing it. Same base checkpoint as the mcg entry "
        "above, so the pair isolates the embedding -- and the weight-bytes "
        "gate is what makes it worth running, because a blockq path that "
        "silently fell back to loading dense bf16 would serve every logit "
        "correctly while giving back the entire saving.\n\n"
        "It also gates tools/quantize_embedding.py, which nothing else does: "
        "the fixture is derived from the published checkpoint and rebuilt "
        "whenever the producer's own source changes, so a producer that starts "
        "writing a different (or unloadable) checkpoint fails here rather than "
        "in someone's repaired model",
    ),
    Entry(
        label="minicpm5-1B 3.0bpw blockq graphs",
        model="turboderp/MiniCPM5-1B-exl3",
        revision="3.00bpw",
        fixture="blockq",
        enforce_eager=False,
        exercises="the same packed embedding under torch.compile and CUDA "
        "graph replay. Not redundant with the eager entry: the decode is plain "
        "torch precisely so inductor can fuse it into the surrounding graph, "
        "which means the compiled path is a different computation reaching the "
        "same answer, and a gather whose indices got baked into a replayed "
        "graph would return the previous batch's rows while failing nothing "
        "else. tests/test_blockq.py covers this at the unit level; this is the "
        "same claim through vLLM's own graph handling",
    ),
    Entry(
        label="minicpm5-1B 3.0bpw tq4",
        model="turboderp/MiniCPM5-1B-exl3",
        revision="3.00bpw",
        kv_cache_dtype="turboquant_4bit_nc",
        exercises="a quantized KV cache under EXL3 weights, which nothing else "
        "in the matrix touches -- every other entry runs the default `auto` "
        "dtype. Differs from `minicpm5-1B-3.0bpw-mcg` by the KV dtype alone, so "
        "a divergence is attributable to turboquant rather than to the "
        "checkpoint.\n\n"
        "It earns fast-tier placement because turboquant is the linchpin of "
        "this project's low end -- it is what makes a long context fit beside a "
        "2-3bpw model on 16 GiB -- and because this model can host it at all: "
        "MiniCPM5-1B has no sliding window, which is the one thing TurboQuant "
        "cannot serve (see TODO `turboquant-sliding-window`, where gemma-4 and "
        "Laguna are both blocked on exactly that)",
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
        "entry that needed patches/vllm-gemma4-transformers-5.15-per-layer.patch, "
        "which it verified as retirable at the 0.28 bump: upstream landed "
        "generic per-layer arch config, the patch was dropped, and this entry "
        "captured at 0.000e+00 against its pre-bump baseline",
    ),
    Entry(
        label="gemma-4-12B 3.0bpw mul1 tied blockq embed",
        model="turboderp/gemma-4-12B-it-exl3",
        revision="3.00bpw_mul1",
        tier="full",
        fixture="blockq",
        gpu_memory_utilization=0.92,
        exercises="EXL3BlockQTiedEmbeddingMethod -- a tied checkpoint whose "
        "embedding has also been block-quantized, so one module carries both "
        "encodings: bq_* for the lookup and the renamed lm_head trellis for the "
        "logits. Deliberately the same repo and revision as the entry above, so "
        "the pair differs only in the fixture and any drift is attributable to "
        "this path rather than to the model.\n\n"
        "Nothing else covers it. Every other blockq fixture is MiniCPM5-1B, "
        "which is untied, and the untied path routes to a different method that "
        "owns no trellis. Until this entry existed the combination had unit "
        "tests and one hand-run comparison, which is the kind of evidence this "
        "suite exists to replace.\n\n"
        "The two failures it stands against were both silent: predicates that "
        "treated tied and blockq as mutually exclusive (died late, at logits "
        "time) and an embed_prefix left at its default (755 MiB of trellis "
        "routed to a path a nested model does not have, dropped without "
        "complaint, garbage served). Neither announced itself at load, so a "
        "logprob baseline is the instrument that would catch a recurrence.\n\n"
        "Not an endorsement of the configuration at this bitrate: blockq beats "
        "spending the same bytes on body bits only above ~4.00bpw "
        "(docs/embeddings.md). 3.00bpw_mul1 is chosen to match its sibling.",
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
        "Two predictions this entry exists to test at the bump, both from "
        "docs/transformers-backend.md, and both now read against v0.28.0 "
        "(2026-08-25) rather than the 0.27.2 that never came.\n\n"
        "**Both still expect logprobs to be UNCHANGED**, which is the opposite "
        "of what this entry said before the patches were triaged -- read that "
        "carefully, because a failure here is now a real regression rather than "
        "a predicted one.\n\n"
        "PR #51247 fixes the dropped embed_norm by a *different* mechanism than "
        "our patch: upstream stopped substituting the embedding and instead "
        "rebases its class (`_rebase_on_vocab_parallel` builds "
        "`type(cls.__name__, (cls, _VocabParallelEmbeddingBase), {})`), so the "
        "model's own forward survives by construction. Our patch is retired. If "
        "logprobs move, the two implementations disagree.\n\n"
        "PR #52173 applies the soft cap but reads only `logit_scale`, never "
        "MuseGlimmer's `output_multiplier`, and applies its scale after the cap "
        "rather than before. That *would* have made this entry fail -- which is "
        "what it previously predicted -- except the softcap patch was halved "
        "rather than retired: the `output_multiplier` alias and the "
        "fold-into-cap trick are kept, reproducing the same arithmetic. So this "
        "entry now tests **our halving** instead of upstream's omission, and a "
        "logprob failure means the fold is wrong.\n\n"
        "MuseGlimmer went native in 0.28 (`registry.py` maps both its "
        "architectures to `muse_glimmer`), which is exactly why model_impl is "
        "pinned here -- without the pin this entry would have silently stopped "
        "testing the backend at this bump",
    ),
    Entry(
        label="muse-glimmer-30B 2.0bpw native",
        model="turboderp/Muse-Glimmer-30B-exl3",
        revision="2.00bpw",
        tier="full",
        model_impl="auto",
        gpu_memory_utilization=0.92,
        known_broken="vLLM's native MuseGlimmer builds the vision adapter as a "
        "plain `nn.Linear` (`muse_glimmer.py`: `self.c_fc = nn.Linear(...)`), "
        "which never reaches `get_quant_method`, so the checkpoint's trellis "
        "tensors have nowhere to land. Measured at v0.28.0: `ValueError: There "
        "is no module or parameter named 'vision_adapter.c_fc.mul1' in "
        "MuseGlimmerForCausalLM. The available parameters belonging to "
        "vision_adapter.c_fc (Linear) are: {'vision_adapter.c_fc.weight'}`.",
        exercises="the native implementation of a checkpoint the matrix already "
        "serves through the Transformers backend -- the pair the MiniCPM "
        "entries make for a text-only model, which nothing makes for a "
        "multimodal one.\n\n"
        "It is kept as a known failure rather than dropped because the failure "
        "is the coverage: the day upstream builds that adapter as a vLLM linear, "
        "this entry starts capturing and `check` says so. That is how the fix "
        "gets noticed, and it is cheaper than remembering to retest.\n\n"
        "**`--language-model-only` does not bypass it** -- an obvious-looking "
        "workaround, tested 2026-08-25 and reported still failing, so the entry "
        "is not merely un-run for want of the right flag. An earlier version of "
        "this text offered that flag as a second way the entry might start "
        "capturing; it is not one",
    ),
    Entry(
        label="qwen3.8-27B 3.0bpw blockq tq4",
        model="turboderp/Qwen3.8-27B-exl3",
        revision="3.00bpw",
        tier="full",
        fixture="blockq",
        kv_cache_dtype="turboquant_4bit_nc",
        language_model_only=True,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        exercises="the configuration this project is actually served with. It "
        "is not a synthetic combination: it mirrors the live command line in "
        "`~/ckpt/run-qwen3.8-27b.sh` -- blockq embedding, trellis weights, "
        "turboquant KV, language-model-only -- so the gate covers the stack as "
        "deployed rather than one axis at a time.\n\n"
        "Three lossy schemes compose here and nothing else exercises the "
        "combination. It also reaches a hybrid attention layout (48 "
        "linear-attention layers, 16 full) where TurboQuant disables its "
        "boundary skips, which is the case that works where Laguna's does not",
    ),
    Entry(
        label="qwen3.8-27B 3.0bpw blockq MTP fp8",
        model="turboderp/Qwen3.8-27B-exl3",
        revision="3.00bpw",
        tier="full",
        fixture="blockq",
        kv_cache_dtype="fp8",
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
        language_model_only=True,
        # Tight on a 16 GiB card, and the requirement is mostly a *constant*:
        # halving max_model_len 4096 -> 2048 moved it only 0.93 -> 0.88 GiB,
        # because 48 of this model's 64 layers are linear-attention whose state
        # does not scale with context. So headroom comes from utilization, not
        # from a shorter context -- the same 0.95 the tight-fit line in
        # ~/ckpt/run-qwen3.8-27b.sh uses.
        max_model_len=2048,
        gpu_memory_utilization=0.95,
        exercises="the fp8 KV cache, which no other entry uses. Originally "
        "paired with MTP on the belief that MTP could not use turboquant; that "
        "belief was wrong (see the sibling), so the MTP half is redundant and "
        "the fp8 half is what this entry is for.\n\n"
        "**It is also the entry that found a silent environment dependency.** "
        "vLLM selects FlashInfer for fp8 KV on this model (head_dim 256, "
        "hybrid), and FlashInfer needs either the `flashinfer_cubin` package or "
        "`nvcc` to JIT. Neither was reachable -- `flashinfer` 0.6.16 was "
        "installed but `flashinfer-cubin` was not -- so this failed with "
        "`RuntimeError: FlashInfer backend is not available` on a build where "
        "it had previously served.\n\n"
        "Resolved by installing the version vLLM itself pins: "
        "`pip install flashinfer-cubin==0.6.16.post3 --extra-index-url "
        "https://flashinfer.ai/whl/` (it is absent from PyPI past 0.6.13, which "
        "is why the extra index is in `requirements/cuda.txt`). Prebuilt cubins "
        "proved byte-identical to nvcc-JIT here -- 0.000e+00 against a baseline "
        "captured the other way -- so the route does not affect the numbers. "
        "`run.py` keeps an nvcc fallback for a machine that has the toolkit but "
        "not the package, and every entry now records the backend actually "
        "selected, so the next such change is reported rather than mysterious.\n\n"
        "The drafter is the real prize. It is a *second model instance*, built "
        "by vLLM's speculative machinery from the same checkpoint, whose "
        "modules go through `get_quant_method` independently of the main "
        "model's -- and this checkpoint's MTP head is genuinely EXL3-quantized "
        "(8 trellis modules, 202.5 MiB). Nothing else in the matrix serves two "
        "models in one engine, and a plugin that mis-handles the second would "
        "fail nowhere else",
    ),
    Entry(
        label="qwen3.8-27B 3.0bpw blockq MTP tq4",
        model="turboderp/Qwen3.8-27B-exl3",
        revision="3.00bpw",
        tier="full",
        fixture="blockq",
        kv_cache_dtype="turboquant_4bit_nc",
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
        language_model_only=True,
        # Tight on a 16 GiB card, and the requirement is mostly a *constant*:
        # halving max_model_len 4096 -> 2048 moved it only 0.88 -> 0.83 GiB,
        # because 48 of this model's 64 layers are linear-attention whose state
        # does not scale with context. So headroom comes from utilization, not
        # from a shorter context -- the same 0.95 the tight-fit line in
        # ~/ckpt/run-qwen3.8-27b.sh uses.
        max_model_len=2048,
        gpu_memory_utilization=0.95,
        exercises="an MTP drafter, which is the coverage nothing else has: a "
        "*second model instance* built by vLLM's speculative machinery from the "
        "same checkpoint, whose modules go through `get_quant_method` "
        "independently of the main model's. This checkpoint's MTP head is "
        "genuinely EXL3-quantized (8 trellis modules, 202.5 MiB), so a plugin "
        "that mishandled a secondary model would fail here and nowhere else.\n\n"
        "It also stacks four lossy schemes -- blockq embedding, trellis "
        "weights, turboquant KV, and speculative drafting -- on top of the "
        "deployed `--language-model-only` shape.\n\n"
        "**This entry was added as `known_broken` on the belief that MTP and "
        "TurboQuant do not combine. They do.** The first two runs failed on KV "
        "cache sizing (0.88 GiB needed against 0.28 available) rather than on "
        "any spec conflict, and given utilization headroom it captures cleanly. "
        "Worth remembering when the next `known_broken` is written from "
        "recollection rather than from a message",
    ),
]


#: Throughput entries, kept separate from the correctness matrix rather than
#: flagged within it, because they want a different configuration: CUDA graphs
#: **on**, since perf measured in eager mode would gate a way nobody serves,
#: while most correctness entries force eager for the cleaner comparison.
#:
#: Deliberately few. Each one costs a model load plus six workload repetitions,
#: and throughput regressions are broad -- a kernel or scheduler change shows up
#: on any model that exercises the path, so covering every checkpoint buys
#: little over covering every *path*.
PERF_ENTRIES: list[Entry] = [
    Entry(
        label="llama-3.2-1B 3.0bpw perf",
        model="turboderp/Llama-3.2-1B-Instruct-exl3",
        revision="3.0bpw",
        enforce_eager=False,
        max_model_len=4096,
        exercises="the anchor. Same checkpoint, card and workload shape as the "
        "table in docs/kernels.md, so its recorded 2754 tok/s decode / 33938 "
        "prefill stay a live reference rather than history. Covers the fused "
        "kernel at decode and the reconstruct-threshold fallback at prefill, "
        "which are the two paths a throughput regression would land on",
    ),
    Entry(
        label="minicpm5-1B 3.0bpw blockq perf",
        model="turboderp/MiniCPM5-1B-exl3",
        revision="3.00bpw",
        fixture="blockq",
        enforce_eager=False,
        exercises="a served path that actually uses the packed embedding, so "
        "the throughput gate covers blockq at all. What it guards is narrow "
        "and deliberately so: an interaction between the engine and the "
        "embedding path large enough to matter in practice.\n\n"
        "It does not resolve the decode itself, and no whole-engine entry "
        "could -- measured 2026-08-24, repeating the gather 16x per lookup is "
        "invisible (+0.05% decode, +0.6% prefill, both inside run noise) and it "
        "takes 64x before anything clears it. The path is ~0.06% of a decode "
        "step and ~0.23% of prefill here, and a smaller share on the larger "
        "models this project targets. A same-run dense companion does not "
        "rescue it either: the two sides are separate engine processes, so "
        "their drift compounds rather than cancelling (prefill ratio spread "
        "2.5pp across three pairs, worse than either absolute number).\n\n"
        "That is a reason to keep the entry rather than to drop it. A "
        "regression this instrument cannot see is, by the same measurement, "
        "one nobody serving the model would feel -- while the condition that "
        "*is* relevant in practice, an engine/blockq interaction costing real "
        "throughput, is exactly what the -10% gate catches. Fine-grained "
        "protection of the gather belongs to a microbenchmark, which would "
        "miss interaction regressions entirely and is the likelier direction "
        "for this path to break, blockq being ours to change",
    ),
    Entry(
        label="qwen3.5-35B-A3B 2.0bpw MoE perf",
        model="turboderp/Qwen3.5-35B-A3B-exl3",
        revision="2.00bpw",
        tier="full",
        enforce_eager=False,
        gpu_memory_utilization=0.92,
        exercises="MoE throughput, which is its own regime: exl3_mgemm rather "
        "than exl3_gemm, and the path where CUDA graphs matter most -- the "
        "sm_90+ barrier fix is worth 35 -> 172 tok/s on Laguna-XS. A "
        "correctness gate cannot see any of that",
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


def perf_by_tier(tier: str | None) -> list[Entry]:
    if tier in (None, "all"):
        return list(PERF_ENTRIES)
    return [e for e in PERF_ENTRIES if e.tier == tier]


def perf_by_name(name: str) -> Entry:
    for e in PERF_ENTRIES:
        if e.name == name:
            return e
    raise SystemExit(f"no such perf entry: {name!r}")
