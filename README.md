# vllm-exl3-plugin

Experiment in adding EXL3 support to vLLM, as an out-of-tree
`vllm.general_plugins` quantization backend.

EXL3 is [exllamav3](https://github.com/turboderp-org/exllamav3)'s trellis-coded
quantization format: plain safetensors, plain HF `config.json`, MIT-licensed,
with working CUDA kernels already shipped as a Python-callable extension. The
plugin is an adapter onto vLLM's `QuantizationConfig` / `LinearMethodBase`
interfaces — it does not build kernels.

**Status: dense, MoE and tensor-parallel serving all work, plus quantized
embeddings for tied models.** Weights stay quantized and the fused exllamav3
kernels do the multiply, under torch.compile and CUDA graphs — the embedding
gather included, as of 2026-08-16. On Llama-3.2-1B @3bpw that is
2.35 GiB -> 0.86 GiB with decode 22% *faster* than the dense path;
gemma-4-12B at 3bpw runs in 6.32 GiB on a 16 GB card. See
[docs/kernels.md](docs/kernels.md).

**Tensor parallelism** is validated on hardware at TP=2 (three dense models, eager
and with CUDA graphs), TP=4 (Qwen3.5-35B-A3B, MoE) and TP=8 (gemma-4-31B, dense,
both execution modes) — every one reproducing TP=1 token for token, including a
vocab-parallel quantized `lm_head`. Routed experts shard too (8.54 GiB -> 4.4 GiB
per worker at TP=2). TP=3, 5, 6 and 7 remain unexercised and warn at startup, and
`tools/tp_preflight.py` reports which degrees a given checkpoint even admits — the
answer is checkpoint-specific and often surprising. See
[docs/tensor-parallel.md](docs/tensor-parallel.md).

**MoE** works on all three MoE checkpoints tried: gemma-4-26B-A4B (9.46 GiB),
Qwen3.5-35B-A3B (10.63 GiB, needs a `deps/vllm` commit to load) and Laguna-XS-2.1
(256 experts at 2bpw, 8.54 GiB). Getting there needed a scale factor that EXL3
checkpoints carry but do not record, which the plugin recovers by measuring the
weights; and care about where that factor is applied, since inside the kernel it
overflows fp16.

MoE also needed a fix to exllamav3 itself: above sm_89 it swaps cooperative
groups' `grid.sync()` for a hand-rolled barrier that synchronizes through a
device-global buffer shared by every launch, and it deadlocks under vLLM. We
carry that fix directly in [our exllamav3
fork](https://github.com/yeasah/exllamav3) (formerly an out-of-tree patch;
folded into history once the fork existed to hold it), making that path
opt-in, which clears the hangs on Hopper/Blackwell and unlocks CUDA graphs for
MoE — Laguna-XS goes from 35 to 172 tok/s. See [docs/moe.md](docs/moe.md).

**Quantized embeddings** are served for both tied and untied models. A tied model
needs no tooling: the fp16 `embed_tokens` is never loaded and lookups come from the
checkpoint's existing quantized `lm_head` — Qwen3-0.6B goes 508 -> 323 MiB resident,
gemma-4-12B gains 1.15 GiB of KV headroom, for ~3% decode cost. An untied model has
no quantized copy to reuse, so `tools/quantize_embedding.py` adds one in a
block-scaled 4-bit format (~4.53 bpw, at or below the model's own noise floor):
Qwen3.5-9B's embedding goes 1940 -> 549 MiB resident and its checkpoint 6.72 -> 5.36
GiB. See [docs/embeddings.md](docs/embeddings.md).

    tools/quantize_embedding.py <checkpoint-dir> <output-dir>

[docs/exllamav3-arch.md](docs/exllamav3-arch.md) indexes where exllamav3
changes behaviour by GPU architecture — worth consulting first when something
inexplicable turns up, or before running on hardware other than Blackwell.

## Notes

`docs/` holds one note per subject — the durable record of what was measured, what
was ruled out, and why each design went the way it did. Open tasks live in
[TODO.md](TODO.md), which is the only document that carries sequencing.

| note | subject |
|---|---|
| [format-and-loading.md](docs/format-and-loading.md) | on-disk format, driving vLLM's loader, quantized `lm_head`, CPU offload |
| [kernels.md](docs/kernels.md) | fused kernels, reconstruct threshold, CUDA graphs, bf16, benchmarks |
| [tensor-parallel.md](docs/tensor-parallel.md) | Hadamard-block-128 sharding, what each TP degree admits, hardware results |
| [moe.md](docs/moe.md) | `exl3_mgemm` behind `FusedMoE`, the Laguna scale factor, the sm_90+ barrier hang |
| [embeddings.md](docs/embeddings.md) | quantized embeddings, block-scaled vs. trellis, storage format, depth selection, serving under torch.compile |
| [blockq-format.md](docs/blockq-format.md) | the block-scaled embedding format itself: layout, decode/encode reference, invariants, how to produce and consume it |
| [qbench.md](docs/qbench.md) | quality measurement across formats on the served path |
| [yaqa.md](docs/yaqa.md) | YAQA-quality rounding in the quantizer: what it buys, what it costs, why the converter's forward-only stream is the obstacle |
| [transformers-backend.md](docs/transformers-backend.md) | serving architectures vLLM has no implementation for |
| [kvarn.md](docs/kvarn.md) | a low-bit KV cache ported, measured and shelved: +17% context for 3.5x prefill, and why the enum is the real find |
| [triattention.md](docs/triattention.md) | a KV *eviction* method: four integration defects fixed, compaction corrupting the cache, and the calibration work that outlived it |
| [exllamav3-arch.md](docs/exllamav3-arch.md) | where exllamav3 branches by GPU architecture |
| [feasibility-2026-08-03.md](docs/feasibility-2026-08-03.md) | the original research report (frozen) |
| [bench/README.md](bench/README.md) | the dependency-bump gate: what it captures and why |

## Quick start

    pip install --no-deps --no-build-isolation -e deps/exllamav3   # builds the CUDA ext
    # if CUDA_HOME's toolkit does not match torch's, see
    #   docs/tensor-parallel.md "Finishing this"
    pip install --no-deps -e .

    vllm serve turboderp/Llama-3.2-1B-Instruct-exl3 --revision 3.0bpw

## Layout

    vllm_exl3_plugin/format.py     on-disk format arithmetic (no torch)
    vllm_exl3_plugin/ops.py        wrappers over exllamav3_ext
    vllm_exl3_plugin/tp.py         tensor-parallel slicing rules
    vllm_exl3_plugin/env.py        environment-variable knobs
    vllm_exl3_plugin/log.py        logging setup
    vllm_exl3_plugin/quantization/ EXL3Config, linear, lm_head, embedding,
                                   fused_moe
    deps/exllamav3                 submodule, reference checkout
    docs/                          subject notes; see "Notes" above
    tests/                         runnable without a GPU
    bench/                         dependency-bump gate; baselines in expected/

## Tests

    python -m unittest discover -s tests

66 tests. The format tests need neither torch, vLLM, nor a GPU; the kernel
oracles and the end-to-end generations skip themselves without CUDA and
exllamav3.

The oracle tests import the `exllamav3` package (not just its extension), which
needs a few small extras the plugin itself does not:

    pip install --no-deps kbnf formatron frozendict general-sam

Separately, `bench/run.py check` gates a dependency bump on what the engine
actually serves — token ids, per-position logprobs and resident weight bytes,
against baselines committed in `bench/expected/`. Run it before and after moving
the vLLM or exllamav3 pin. See [bench/README.md](bench/README.md).

## Gotcha: killing stray GPU processes

Use `tools/reap`, not `pkill`. With no arguments it reaps whatever currently holds GPU
memory, found via `nvidia-smi --query-compute-apps` so process names never enter into it;
`-n` dry-runs, and `-d` sends `SIGABRT` first for a faulthandler stack dump, which is the
intended workflow when a MoE run hangs. It refuses to signal any ancestor of itself.

The two obvious `pkill` forms both fail, and both fail *silently*:

- `pkill -f VLLM::EngineCore` matches the full command line — including that of the shell
  running the `pkill` itself, which contains the pattern as an argument. It kills its own
  caller, and anything chained after it never runs.
- `pkill -x VLLM::EngineCore` matches against `comm`, which the kernel truncates to 15
  characters, so the pattern never matches at all. The orphan survives, keeps holding GPU
  memory, and makes the *next* run fail memory profiling — which looks like a code
  regression rather than leftover state.

If reaching for `pkill` regardless, the safe forms are a bracketed pattern
(`pkill -f '[v]llm_gen'`) or `ps` plus explicit PID filtering.

## Gotcha: BOS tokens

Some checkpoints (gemma-4 among them) put `<bos>` in `chat_template.jinja` and
do *not* add it in the tokenizer, even with `add_special_tokens=True`. Gemma
produces garbage without it. Drive models through their chat template
(`llm.chat`, or the server's chat endpoint) rather than raw completion prompts.

## Requirements

- CUDA, compute capability 8.0+ (Ampere). No ROCm — exllamav3 has none. The
  submodule points at [our fork](https://github.com/yeasah/exllamav3), which
  carries the sm_90+ barrier fix directly; MoE will hang on Hopper/Blackwell
  against unpatched upstream exllamav3.
- float16 or bfloat16 activations. The kernels are fp16; `exl3_mm` casts at the
  kernel boundary, so bf16 models keep a bf16 residual stream.
- TP=1, 2, 4 and 8 have run on hardware and reproduce TP=1 token for token.
  TP=3, 5, 6 and 7 are implemented and proven arithmetically but unexercised,
  and warn at startup. Not every checkpoint admits every degree — run
  `tools/tp_preflight.py` first; see
  [docs/tensor-parallel.md](docs/tensor-parallel.md).

## Patches

`tools/checkpoint_survey.py` screens a checkpoint before you spend the bandwidth: it
sorts every stored tensor into what this plugin serves, what it never loads, and what it
does not recognize — the last being a reliable predictor of real work. `--remote` answers
the same question from Hub metadata, without downloading. It screens *storage*, not
architecture support.

`tools/host_survey.py` screens a *host* before you spend the rental: stdlib-only and
single-file so it runs on a bare box, it reports GPU/driver/ECC split into fields that
can change model output and fields that only change throughput, and `--compare` diffs
two boxes while classifying each difference. Exit 1 refuses a box (uncorrected ECC,
mismatched GPUs, or an output-relevant difference from the baseline).

`tools/transcript_sweep.py` mines this project's own conversation transcripts for
findings that were established in a session and never written down — the failure
mode of the exploratory work that produces most of them. `extract` pools every
session file, drops tool traffic and dedupes by content so a forked or resumed
session contributes each exchange once; `signature` surfaces the assistant
passages that read like a finding; `check` asks whether a candidate is already in
`docs/`, `TODO.md`, the memories or the field notes, which is the step that keeps
a sweep from producing duplicates; `mark` records where it stopped so the next one
knows where to start (`docs/data/sweeps.json`). Read `user-*.md` first — the human
turns are about a fifth of the volume and carry most of the signal.

The plugin needs a patched vLLM, vendored as the `deps/vllm` submodule: our fork
on branch `appliance/v0.28.0`, which is the **v0.28.0** pin plus the commits
below. It reproduces the tree the baselines in `bench/expected/` were captured
from. [patches.md](patches.md) is the index — what each commit does, and how to
offer one upstream.

    git submodule update --init deps/vllm
    VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION=<v0.28.0 wheel> \
    pip install --no-deps --no-build-isolation -e deps/vllm

Use the precompiled path: the branch is pure Python, so a source build is half
an hour for nothing — and its default job count OOMs on consumer hardware.
[patches.md](patches.md) has the exact command and why.

These changes used to live as `.patch` files applied by hand to a checkout
outside the project; the submodule replaces that.

| commit | why | still needed because |
|---|---|---|
| [`vllm-fused-param-capability-check`](patches.md) | lets a parameter declare that it splits fused checkpoint tensors itself; Qwen3.5 will not load without it | `handles_fused_shards` has 0 occurrences upstream |
| [`vllm-replicated-linear-weight-loader-v2`](patches.md) | `ReplicatedLinear` is the one `LinearBase` subclass with no `weight_loader_v2` branch; needed to serve multimodal models through vLLM's Transformers backend | still no `weight_loader_v2` branch upstream |
| [`vllm-embed-quant-config`](patches.md) | 86 of 131 model files never pass `quant_config` to their `VocabParallelEmbedding`, so no quantized embedding can be served on those architectures — silently dense for a tied model, a load failure for a block-quantized one. Defaults it from the config being built under, in one place rather than 86 | `VocabParallelEmbedding` still calls `get_quant_method` only when `quant_config` is passed explicitly. Still worth filing |
| [`vllm-transformers-backend-logit-softcap`](patches.md) | the Transformers backend reads only `logit_scale`, never MuseGlimmer's `output_multiplier`, and `LogitsProcessor` applies its scale *after* the cap where the model needs it before | upstream landed `soft_cap=final_logit_softcapping` at 0.28 but neither of the other two. This is the halved remainder: the alias, and the fold-into-cap identity `tanh(z/(T/m))·(T/m)·m == T·tanh(z·m/T)` |

Three further commits on the branch are TurboQuant-specific and are covered in
[patches.md](patches.md) and [docs/turboquant-kv.md](docs/turboquant-kv.md): the
sliding-window page-size fixes, the `boundary:N` lever, and the
`_continuation_prefill` copy that drops a full-context temporary.

### Retired at the 0.28 bump


Kept here as a record, since "why did this stop being needed" is the question a
future bump asks:

- **`vllm-gemma4-transformers-5.15-per-layer.patch`** — upstream landed generic
  per-layer arch config. `Gemma4Config.verify_and_update_config` now reads
  `model_config.model_arch_config` and indexes `arch_config[i].head_size`.
- **`vllm-transformers-backend-embedding-postprocess.patch`** — retired by a
  *better* mechanism than ours. Upstream no longer substitutes the input
  embedding; it **rebases its class**
  (`type(cls.__name__, (cls, _VocabParallelEmbeddingBase), {})`), so a model's
  own `forward` — MuseGlimmer's `embed_norm` — survives by construction rather
  than by our detection heuristic. The quantized-embedding seam survives too:
  `replace_embedding_class` passes `quant_config` into
  `VocabParallelEmbedding.__init__`, so our methods still attach.

Both retirements were confirmed by the gate, not just by reading: the
Muse-Glimmer entry that exercises both paths captures at `0.000e+00` against its
pre-bump baseline, so upstream's mechanism and ours agree exactly.

## Environment variables

| variable | default | effect |
|---|---|---|
| `EXL3_RECONSTRUCT_THRESHOLD` | 144 | rows above which to decode to dense fp16 and use cuBLAS; 0 always uses the fused kernel |
| `EXL3_DEQUANTIZE` | 0 | Phase 0 behaviour: dequantize at load. Correctness oracle, no memory saving |
| `EXL3_DENSE_EMBED` | 0 | keep a tied model's embedding dense instead of serving it from the quantized `lm_head` (see [docs/embeddings.md](docs/embeddings.md)) |
| `EXL3_EMBED_BLOCK_CHUNK` | 256 | distinct 128-row blocks decoded per pass in the quantized-embedding gather; lower bounds peak memory |
| `VLLM_DISABLE_COMPILE_CACHE` | 0 | set to 1 while editing this plugin — vLLM's compile cache cannot see plugin code |
