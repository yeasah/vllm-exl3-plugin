# vllm-exl3-plugin

Serve [EXL3](https://github.com/turboderp-org/exllamav3) quantized models with
[vLLM](https://github.com/vllm-project/vllm).

EXL3 is exllamav3's trellis-coded quantization format: plain safetensors and a plain HF
`config.json`, with fused CUDA kernels that multiply directly from the quantized weights.
This plugin registers EXL3 as a vLLM quantization backend (`vllm.general_plugins`), so an
EXL3 checkpoint serves through vLLM's normal engine: the OpenAI-compatible server, continuous
batching, prefix caching, CUDA graphs, tensor parallelism and quantized KV caches. Weights
stay quantized in VRAM, and exllamav3's kernels do the work; the plugin builds no kernels of
its own.

**Part 1, using it:** [Requirements](#requirements) · [Install](#install) ·
[Differences from plain vLLM](#how-it-differs-from-plain-vllm) ·
[vLLM changes beyond EXL3](#vllm-changes-beyond-exl3) ·
[Environment variables](#environment-variables)

**Part 2, details:** [Tools](#tools) · [Notes](#notes) · [Layout](#layout) ·
[Tests](#tests) · [Development gotchas](#development-gotchas) · [The vLLM fork](#the-vllm-fork)

---

# Part 1: Using it

## Requirements

- An NVIDIA GPU of compute capability 8.0+ (Ampere or newer) and CUDA. exllamav3 also builds
  for ROCm now, but this plugin has only been run on CUDA, and the install below is CUDA-only.
  The exllamav3 submodule is [our fork](https://github.com/yeasah/exllamav3), which carries the
  sm_90+ barrier fix: MoE hangs on Hopper and Blackwell against unpatched upstream.
- float16 or bfloat16 activations. The kernels are fp16 and cast at the boundary, so bf16
  models keep a bf16 residual stream.
- Python 3.10+.
- For more than one GPU on PyTorch up to 2.14.1, a rebuilt `libc10_cuda.so`
  (`tools/torch-c10cuda-fix.sh`): exllamav3 turns on expandable segments, and with them stock
  torch can unmap memory another GPU is still writing. See [docs/upstream.md](docs/upstream.md)
  "PyTorch".

## Install

In a fresh Python virtual environment of its own. This installs **forks** of vLLM and exllamav3, so
do not share the environment with anything that expects stock vLLM or exllamav3:

    git clone --recurse-submodules https://github.com/yeasah/vllm-exl3-plugin
    cd vllm-exl3-plugin
    ./install.sh

    vllm serve turboderp/Llama-3.2-1B-Instruct-exl3 --revision 3.0bpw

`./install.sh --check` reports what is installed and whether vLLM is the patched build.

Optional:

- `./install.sh --extras` adds flash-linear-attention and causal-conv1d, the latter built against
  your torch. vLLM's own models do not need them (vLLM carries its own kernels); transformers
  does, so they matter for the Transformers backend on linear-attention models and for the
  quantization and eval tooling, which otherwise fall back to slow reference code. To add them
  to an install that already works, without rebuilding the rest:
  `pip install flash-linear-attention && pip install --no-build-isolation causal-conv1d`.
- `vllm download-kernels` fetches FlashInfer's precompiled kernels (about 1.5 GB) and silences
  the startup warning about them; without them, the few FlashInfer ops used compile on first use.
- More than one GPU: see the PyTorch fix under [Requirements](#requirements).

The plugin needs a patched vLLM, vendored as the `deps/vllm` submodule (our fork's
`appliance/v0.32.0` branch: the v0.32.0 release plus pure-Python commits). `install.sh` does
the following, which can also be run by hand, starting from an environment with vLLM 0.32.0's
dependencies (`pip install vllm==0.32.0`, which the fork then replaces):

    # 1. exllamav3: builds the CUDA extension. If CUDA_HOME's toolkit does not
    #    match torch's, see docs/tensor-parallel.md "Finishing this"
    pip install --no-deps --no-build-isolation -e deps/exllamav3

    # 2. the patched vLLM, using the released wheel's binaries (the commits are pure
    #    Python, so building from source is half an hour for nothing)
    pip install 'cmake>=3.26.1' ninja 'packaging>=24.2' 'setuptools>=77.0.3,<81.0.0' \
        'setuptools-scm>=8.0' 'setuptools-rust>=1.9.0' wheel jinja2
    git -C deps/vllm fetch --no-tags https://github.com/vllm-project/vllm.git tag v0.32.0
    VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_LOCATION=https://files.pythonhosted.org/packages/a5/0d/0b15fda9a56ac485ebb607e466f0e3386c8c115bafeced6514298bd325b5/vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl \
    pip install --no-deps --no-build-isolation -e deps/vllm

    # 3. the plugin
    pip install --no-deps -e .

`python -c "import vllm; print(vllm.__version__)"` should then name the base tag, the patch
count, our commit and `.precompiled` (for example `0.32.1.dev16+g1f3b03290`). The tag fetch has
to come before the vLLM install, and [patches.md](patches.md) explains why, along with what
each commit does.

## How it differs from plain vLLM

It works like vLLM, with EXL3 checkpoints. The quantization method is read from the
checkpoint, so `--quantization` is not needed. The main differences:

- **Bit rates are branches.** EXL3 repositories usually publish one revision per bit rate,
  with `main` holding only the model card, so pass `--revision` (`--revision 4.00bpw`). The
  model card names its branches.
- **Embeddings are quantized by default.** A dense token embedding is encoded on load into a
  block-scaled 4-bit format (about 4.5 bpw, at or near the model's own noise floor), and cached
  under `~/.cache/vllm-exl3-plugin/blockq-embeddings/` so it happens once per checkpoint; the
  cache evicts least-recently-used entries past 16 GB. A checkpoint that already stores a
  quantized embedding (exllamav3's trellis table from `convert.py --embed_bits`, or `bq_*`) is
  served as stored. A tied model is served from its quantized `lm_head`. `EXL3_BLOCKQ_ON_LOAD=0`
  serves a dense embedding dense. See [docs/embeddings.md](docs/embeddings.md).
- **Tensor parallelism admits only some degrees per checkpoint.** A shard has to be a whole
  number of 128-wide Hadamard blocks, so which TP degrees a checkpoint admits depends on its
  shapes, and a degree it does not admit is refused at load: no Qwen3 checkpoint (151,936-token
  vocabulary) runs at TP=2, for example. `tools/tp_preflight.py <repo>` reports the admitted degrees before you
  download. TP=1, 2, 4 and 8 are validated on hardware and reproduce TP=1 token for token;
  3, 5, 6 and 7 are implemented but unexercised, and warn. See
  [docs/tensor-parallel.md](docs/tensor-parallel.md).
- **On GPUs with peer access, TP can crash at CUDA-graph capture.** exllamav3's expandable
  segments and vLLM's custom all-reduce do not mix (workstation cards, NVLink pairs, most
  cloud multi-GPU). Pass `--disable-custom-all-reduce`, or set `EXL3_EXPANDABLE_SEGMENTS=0`.
- **Vision towers are dense; offload them.** Multimodal EXL3 checkpoints keep their vision
  tower unquantized. Offloading it costs almost nothing, since it runs once per image:
  `--cpu-offload-gb N --cpu-offload-params visual`. `--language-model-only` drops it. An
  image-heavy request on a small card may still need a pixel cap through
  `--mm-processor-kwargs`. See [docs/media-encoders.md](docs/media-encoders.md).
- **`--cpu-offload-gb` offloads EXL3 weights, with limits.** Prefill is slow, since offloaded
  weights cross PCIe for every token; it cannot offload more than fits in VRAM, because loading
  peaks at the full checkpoint; and MoE experts are offloaded naively. An offload built around
  an MoE expert cache, which addresses all three, is in development
  ([docs/cpu-offload.md](docs/cpu-offload.md)).
- **MoE works,** through vLLM's `FusedMoE` with exllamav3's `exl3_mgemm`, CUDA graphs included.
  See [docs/moe.md](docs/moe.md).
- **Drive models through their chat template.** Some checkpoints (gemma-4 among them) put
  `<bos>` only in the template, and produce garbage from raw completion prompts.

## vLLM changes beyond EXL3

The fork also changes vLLM behaviour that is not specific to EXL3. Most are automatic.
Each commit is described in [patches.md](patches.md).

- **TurboQuant KV with sliding-window models.** A quantized KV cache can coexist with
  sliding-window layers (gemma-4, Laguna, Muse-Glimmer), which stock v0.32.0 rejects.
  Automatic with `--kv-cache-dtype turboquant_4bit_nc` (or another TurboQuant type).
- **`boundary:N` in `--kv-cache-dtype-skip-layers`.** TurboQuant keeps a boundary of native
  layers at each end of the stack; `boundary:N` sets how many, which the stock flag cannot, and
  layer indices still add to it. On Qwen3-4B the whole quality effect is layer 0:
  `--kv-cache-dtype-skip-layers 0 boundary:0` matches the stock protection at 18% fewer KV bytes.
  See [docs/turboquant-kv.md](docs/turboquant-kv.md).
- **TurboQuant prefill no longer grows with context.** Continuation prefill dequantizes the
  cached context in slabs instead of whole, so its VRAM tracks the chunk rather than the
  context. Automatic.
- **Attention backends declare the workspace the profiler cannot see** (TurboQuant, FlashInfer),
  so KV sizing accounts for it and a high `gpu_memory_utilization` holds instead of failing at
  first inference. Automatic. See [docs/memory-accounting.md](docs/memory-accounting.md).
- **Vision encoders use less peak memory**: the attention preamble stops copying q/k/v it
  already has, and the MLP runs in chunks. Automatic.
- **More models load**, through the Transformers backend (`--model-impl transformers`) and for
  quantized embeddings: fused-shard handling (Qwen3.5), a v2 weight loader for
  `ReplicatedLinear`, a non-standard logit multiplier (Muse-Glimmer), and a default
  `quant_config` for embeddings. See [docs/transformers-backend.md](docs/transformers-backend.md).

## Environment variables

Defaults suit serving; most of these exist for measurement or to isolate a problem.

| variable | default | effect |
|---|---|---|
| `EXL3_BLOCKQ_ON_LOAD` | 1 | encode an untied model's dense embedding to blockq while it loads; 0 serves it dense |
| `EXL3_BLOCKQ_CACHE` | 1 | cache that encoding, keyed by the checkpoint's resolved commit; 0 re-encodes every start |
| `EXL3_BLOCKQ_CACHE_DIR` | `$XDG_CACHE_HOME/vllm-exl3-plugin/blockq-embeddings` | where the cache lives; safe to delete at any time |
| `EXL3_BLOCKQ_CACHE_MAX_GB` | 16 | the cache's ceiling: each write evicts least-recently-used entries down to it; 0 never evicts |
| `EXL3_BLOCKQ_ENCODE_CHUNK` | 16384 | rows per on-load encode pass; lower bounds the fp32 temporary |
| `EXL3_DENSE_EMBED` | 0 | keep a tied model's embedding dense instead of serving it from the quantized `lm_head` |
| `EXL3_EXPANDABLE_SEGMENTS` | 1 | exllamav3's allocator choice; 0 keeps PyTorch's classic allocator (see TP above) |
| `EXL3_RECONSTRUCT_THRESHOLD` | 144 | rows above which a linear decodes to fp16 and uses cuBLAS; 0 always uses the fused kernel |
| `EXL3_RECONSTRUCT_TILE_MB` | 32 | tile size for that decode; 0 decodes the weight whole |
| `EXL3_SCRATCH_POOL` | 1 | reuse the decode path's scratch buffers; 0 allocates per call |
| `EXL3_EMBED_BLOCK_CHUNK` | 256 | 128-row blocks decoded per pass in the tied-embedding gather; lower bounds peak memory |
| `EXL3_EMBED_STATIC_MAX` | 512 | largest batch the capture-safe embedding gather handles; must cover vLLM's CUDA-graph sizes |
| `EXL3_DEQUANTIZE` | 0 | dequantize every weight at load: a correctness oracle with no memory saving |
| `VLLM_DISABLE_COMPILE_CACHE` | 0 | set to 1 while editing the plugin: vLLM's compile cache cannot see plugin code |

---

# Part 2: Details

## Tools

For serving:

- `tools/tp_preflight.py` reports which TP degrees a checkpoint admits, before you download it.
- `tools/checkpoint_survey.py` sorts every stored tensor of a checkpoint into what this plugin
  serves, what it never loads, and what it does not recognize (the last predicts real work);
  `--remote` answers from Hub metadata without downloading. It screens storage, not
  architecture support.
- `tools/quantize_embedding.py <checkpoint> <output>` rewrites a checkpoint to store a quantized
  embedding (`bq_*`, or exllamav3's trellis with `--trellis K`); `--sidecar` adds `bq_*` beside
  the dense one. Rarely needed now that embeddings are encoded on load and the quantization
  pipeline writes trellis tables; useful to ship a smaller checkpoint for an older plugin or
  for exllamav3.

For development:

- `tools/host_survey.py` screens a host before you spend the rental: stdlib-only and
  single-file, it reports GPU/driver/ECC split into fields that can change model output and
  fields that only change throughput, and `--compare` diffs two boxes. Exit 1 refuses a box
  (uncorrected ECC, mismatched GPUs, or an output-relevant difference from the baseline).
- `tools/compose_checkpoint.py` builds one EXL3 checkpoint from the tensors of others: start
  from a base and replace whole *modules* with another checkpoint's copies, by category
  (`head`, `vision`, `mtp`, `experts`, `shared-experts`, `body`) or glob. It exists for
  allocation questions decided per category rather than per tensor, and for repairing one
  tensor family without redoing a conversion. Modules move whole, because a trellis paired
  with another conversion's `suh` is noise that still loads; it refuses to write on a hard
  mismatch, and hardlinks untouched shards. `--restore` puts the **dense original** back (how
  a bf16 vision tower is grafted into a checkpoint that quantized one), and `--verify`
  dequantizes a sample to prove the dense tensor is the one the trellis was made from.
- `tools/reap` kills leftover GPU processes; see [Development gotchas](#development-gotchas).
- `tools/transcript_sweep.py` mines this project's conversation transcripts for findings that
  were established in a session and never written down. `extract` pools and dedupes the
  session files, `signature` surfaces passages that read like a finding, `check` asks whether
  one is already recorded, and `mark` records where a sweep stopped (`docs/data/sweeps.json`).
  Read `user-*.md` first: the human turns are about a fifth of the volume and most of the signal.
- `tools/torch-c10cuda-fix.sh` rebuilds torch's `libc10_cuda.so` with the multi-GPU allocator
  fix; `--check` reports whether the installed one is patched.

Making EXL3 quantizations (calibration, per-tensor recipes, evaluation, model cards) is the
job of `quantization/quant.py`; its method and measurements are in
[docs/calibration.md](docs/calibration.md), [docs/qbench.md](docs/qbench.md) and
[docs/recall.md](docs/recall.md).

## Notes

`docs/` holds one note per subject: the durable record of what was measured, what was ruled
out, and why each design went the way it did. Open tasks live in [TODO.md](TODO.md), the only
document that carries sequencing; lower-priority and speculative work is in
[FUTURE.md](FUTURE.md). [docs/exllamav3-arch.md](docs/exllamav3-arch.md) indexes where
exllamav3 behaves differently by GPU architecture, the first thing to consult when something
inexplicable turns up on hardware other than Blackwell.

| note | subject |
|---|---|
| [format-and-loading.md](docs/format-and-loading.md) | on-disk format, driving vLLM's loader, quantized `lm_head` |
| [cpu-offload.md](docs/cpu-offload.md) | why vLLM's offloaders skip EXL3, what a PCIe round trip per token actually costs, why prefill pays it per token too, why *which* experts you offload cannot matter, and why llama.cpp's `-ot` recipes mean something else |
| [kernels.md](docs/kernels.md) | fused kernels, reconstruct threshold, CUDA graphs, bf16, benchmarks |
| [tensor-parallel.md](docs/tensor-parallel.md) | Hadamard-block-128 sharding, what each TP degree admits, hardware results |
| [moe.md](docs/moe.md) | `exl3_mgemm` behind `FusedMoE`, the Laguna scale factor, the sm_90+ barrier hang |
| [embeddings.md](docs/embeddings.md) | quantized embeddings, block-scaled vs. trellis, storage format, depth selection, serving under torch.compile |
| [blockq-format.md](docs/blockq-format.md) | the block-scaled embedding format itself: layout, decode/encode reference, invariants, how to produce and consume it |
| [qbench.md](docs/qbench.md) | quality measurement across formats on the served path |
| [calibration.md](docs/calibration.md) | what a conversion's calibration protects and what it leaves uncovered: why Qwen3.8's quants looked like a lottery, and a conversational calibration mix that cuts excess KLD 12-18% on independent chat evals (real user prompts included) at no cost on raw text |
| [capability-suite.md](docs/capability-suite.md) | whether a served configuration still *does the work*: what a paired comparison has to hold fixed, three metrics worth tracking (correctness, cost, indecision), why an agentic harness is actively harmful on a Q&A benchmark, and 4-bit KV measured at no detectable cost on either axis |
| [recall.md](docs/recall.md) | whether a quant keeps what the unquantized model knows: kept, confident errors, recall by confidence band, per-model slices, and why conversions differ more on facts than on KL |
| [yaqa.md](docs/yaqa.md) | YAQA-quality rounding in the quantizer: what it buys, what it costs, why the converter's forward-only stream is the obstacle |
| [transformers-backend.md](docs/transformers-backend.md) | serving architectures vLLM has no implementation for |
| [media-encoders.md](docs/media-encoders.md) | how many bytes a vision/audio tower is (up to 12.7% of a low-bpw exl3 checkpoint), why evicting one is nearly free, what vLLM 0.29.0 changed, and why the weights are not the cost that gates vision |
| [turboquant-kv.md](docs/turboquant-kv.md) | TurboQuant's KV page geometry: the mispriced page that stopped a quantized cache coexisting with sliding-window layers, and what first/last-N boundary protection costs |
| [memory-accounting.md](docs/memory-accounting.md) | where the VRAM goes: the backend workspaces vLLM's profiler cannot see, why `gpu_memory_utilization` stops being safe above ~0.85, and which consumer has to be engineered away rather than budgeted for |
| [kvarn.md](docs/kvarn.md) | a low-bit KV cache ported, measured and shelved: +17% context for 3.5x prefill, and why the enum is the real find |
| [triattention.md](docs/triattention.md) | a KV *eviction* method: four integration defects fixed, compaction corrupting the cache, and the calibration work that outlived it |
| [kv-pager.md](docs/kv-pager.md) | KV residency as a block-table rewrite: the measurements and the instruments that took them, and **closed 2026-09-08**, three exits that put KV eviction and offload off the table categorically |
| [exllamav3-arch.md](docs/exllamav3-arch.md) | where exllamav3 branches by GPU architecture |
| [pre-ampere.md](docs/pre-ampere.md) | Volta and Pascal below the sm_80 floor: the CUDA 13 toolchain is the wall rather than the kernels, Pascal ruled out for good, Volta parked |
| [upstream.md](docs/upstream.md) | the outward-facing queue: patches we carry, what is worth offering and in what shape, what to check before filing, and what a bump costs the TQ patches |
| [feasibility-2026-08-03.md](docs/feasibility-2026-08-03.md) | the original research report (frozen) |
| [capability-suite/README.md](capability-suite/README.md) | post-process analysis for hand-run capability evals |
| [bench/README.md](bench/README.md) | the dependency-bump gate: what it captures and why |

## Layout

    install.sh                     installs exllamav3, the patched vLLM and the plugin; --check
    vllm_exl3_plugin/format.py     on-disk format arithmetic (no torch)
    vllm_exl3_plugin/ops.py        wrappers over exllamav3_ext
    vllm_exl3_plugin/tp.py         tensor-parallel slicing rules
    vllm_exl3_plugin/env.py        environment-variable knobs
    vllm_exl3_plugin/log.py        logging setup
    vllm_exl3_plugin/blockq*.py    the block-scaled embedding format and its on-load cache
    vllm_exl3_plugin/trellis_embed.py  exllamav3's trellis embedding, decoded in plain torch
    vllm_exl3_plugin/quantization/ EXL3Config, linear, lm_head, embedding, fused_moe
    quantization/                  the quantization pipeline (quant.py) and its templates
    deps/exllamav3                 submodule: our exllamav3 fork
    deps/vllm                      submodule: our vLLM fork (see patches.md)
    docs/                          subject notes; see "Notes" above
    tests/                         runnable without a GPU
    bench/                         dependency-bump gate; baselines in expected/

## Tests

    python -m unittest discover -s tests

The format tests need neither torch, vLLM nor a GPU; the kernel oracles and the end-to-end
generations skip themselves without CUDA and exllamav3. The oracle tests import the
`exllamav3` package (not just its extension), which needs a few extras the plugin itself does
not:

    pip install --no-deps kbnf formatron frozendict general-sam

Separately, `bench/run.py check` gates a dependency bump on what the engine actually serves:
token ids, per-position logprobs and resident weight bytes, against baselines committed in
`bench/expected/`. Run it before and after moving the vLLM or exllamav3 pin. See
[bench/README.md](bench/README.md).

## Development gotchas

### Killing stray GPU processes

Use `tools/reap`, not `pkill`. With no arguments it reaps whatever currently holds GPU memory,
found via `nvidia-smi --query-compute-apps` so process names never enter into it; `-n`
dry-runs, and `-d` sends `SIGABRT` first for a faulthandler stack dump, which is the intended
workflow when a MoE run hangs. It refuses to signal any ancestor of itself. It reaps **every**
GPU, so do not use it while a job you mean to keep is running on another card: kill by PID or
by process group instead.

The two obvious `pkill` forms both fail, and both fail *silently*:

- `pkill -f VLLM::EngineCore` matches the full command line, including that of the shell
  running the `pkill`, which contains the pattern as an argument. It kills its own caller, and
  anything chained after it never runs.
- `pkill -x VLLM::EngineCore` matches against `comm`, which the kernel truncates to 15
  characters, so the pattern never matches at all. The orphan survives, keeps holding GPU
  memory, and makes the *next* run fail memory profiling, which looks like a code regression
  rather than leftover state.

If reaching for `pkill` regardless, the safe forms are a bracketed pattern
(`pkill -f '[v]llm_gen'`) or `ps` plus explicit PID filtering.

### BOS tokens

Some checkpoints (gemma-4 among them) put `<bos>` in `chat_template.jinja` and do *not* add it
in the tokenizer, even with `add_special_tokens=True`. Gemma produces garbage without it.
Drive models through their chat template (`llm.chat`, or the server's chat endpoint) rather
than raw completion prompts.

## The vLLM fork

The patched vLLM is the `deps/vllm` submodule: our fork's `appliance/v0.32.0` branch, the
v0.32.0 pin plus the commits listed in [patches.md](patches.md), which reproduces the tree the
baselines in `bench/expected/` were captured from. [patches.md](patches.md) is the index: what
each commit does, how to install it (Part 1 has the short form), how to offer one upstream,
and which earlier patches retired at which bump and why. These changes used to live as
`.patch` files applied by hand to a checkout outside the project; the submodule replaced that.
