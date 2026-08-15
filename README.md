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
kernels do the multiply, under torch.compile and CUDA graphs. On Llama-3.2-1B
@3bpw that is 2.35 GiB -> 0.86 GiB with decode 22% *faster* than the dense path;
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
Qwen3.5-35B-A3B (10.63 GiB, needs the `patches/` change to load) and Laguna-XS-2.1
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

**Quantized embeddings** are served for tied models: the fp16 `embed_tokens` is
never loaded, and lookups come from the checkpoint's existing quantized `lm_head`.
Qwen3-0.6B goes 508 -> 323 MiB resident; gemma-4-12B gains 1.15 GiB of KV headroom,
for ~3% decode cost. Untied models still load a dense embedding — they have no
quantized one to reuse. See [docs/embeddings.md](docs/embeddings.md).

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
| [embeddings.md](docs/embeddings.md) | quantized embeddings, per-row vs. trellis, depth selection |
| [qbench.md](docs/qbench.md) | quality measurement across formats on the served path |
| [exllamav3-arch.md](docs/exllamav3-arch.md) | where exllamav3 branches by GPU architecture |
| [feasibility-2026-08-03.md](docs/feasibility-2026-08-03.md) | the original research report (frozen) |

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

## Tests

    python -m unittest discover -s tests

64 tests. The format tests need neither torch, vLLM, nor a GPU; the kernel
oracles and the end-to-end generations skip themselves without CUDA and
exllamav3.

The oracle tests import the `exllamav3` package (not just its extension), which
needs a few small extras the plugin itself does not:

    pip install --no-deps kbnf formatron frozendict general-sam

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

`patches/` holds changes to *vLLM*, applied to a source checkout:

| patch | why |
|---|---|
| `vllm-fused-param-capability-check.patch` | lets a parameter declare that it splits fused checkpoint tensors itself; Qwen3.5 will not load without it |
| `vllm-gemma4-transformers-5.15-per-layer.patch` | gemma-4 on transformers >= 5.15, which moves `head_dim`/`num_key_value_heads` into per-layer configs |

## Environment variables

| variable | default | effect |
|---|---|---|
| `EXL3_RECONSTRUCT_THRESHOLD` | 144 | rows above which to decode to dense fp16 and use cuBLAS; 0 always uses the fused kernel |
| `EXL3_DEQUANTIZE` | 0 | Phase 0 behaviour: dequantize at load. Correctness oracle, no memory saving |
| `EXL3_DENSE_EMBED` | 0 | keep a tied model's embedding dense instead of serving it from the quantized `lm_head` (see [docs/embeddings.md](docs/embeddings.md)) |
| `EXL3_EMBED_BLOCK_CHUNK` | 256 | distinct 128-row blocks decoded per pass in the quantized-embedding gather; lower bounds peak memory |
| `VLLM_DISABLE_COMPILE_CACHE` | 0 | set to 1 while editing this plugin — vLLM's compile cache cannot see plugin code |
