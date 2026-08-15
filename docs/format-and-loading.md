# Format and loading

*Originally "Phase 0 — registration spike". The on-disk EXL3 format, how vLLM's
loader is driven to fill it, and the quantized `lm_head`.*

Goal, unchanged from the feasibility report: **prove the registration path and
confirm an EXL3 checkpoint loads and generates correct tokens through vLLM's
forward pass**, single GPU, TP=1, no CUDA graphs, no fused kernels.

**Status: done and verified on hardware.** An EXL3 checkpoint loads through
vLLM and generates coherent tokens, at both a uniform and a mixed bit width, and
the dequantization is bit-for-bit identical to exllamav3's own, including
models with a quantized `lm_head`. 30 tests passed at the time.
See "Verified" below for what was actually run and "Remaining gaps" for what
Phase 0 deliberately does not cover.

## What Phase 0 does at runtime

`process_weights_after_loading` fully dequantizes each EXL3 tensor into a dense
fp16 matrix, and `apply` is a plain `F.linear`. That throws away the entire
memory saving, on purpose: it separates *"does vLLM's loader fill EXL3's tensor
layout correctly"* from *"do the fused kernels behave under vLLM's serving
loop"*, and only the first question belongs to Phase 0. The only exllamav3 entry
point used is `ext.reconstruct`; the Hadamard is done in torch.

The dequantization is a transcription of `LinearEXL3.get_weight_tensor`, in the
same order and the same precision, so it should agree with exllamav3 bit for
bit. That equality is the cheapest correctness oracle available and is the first
thing to check on real hardware.

## Ground truth established

Read out of exllamav3 v1.3.0 (`0b9745c5`), vLLM `main` @ `4a3447d2` (2026-08-03,
re-verified unchanged on `edbc4969a`), `vllm-gguf-plugin` @ `main`, and the
safetensors headers of real EXL3 repos on the Hub.

### On-disk layout, per quantized linear

    <key>.trellis   int16   [in_features // 16, out_features // 16, 16 * K]
    <key>.suh       fp16    [in_features]
    <key>.svh       fp16    [out_features]
    <key>.bias      fp16    [out_features]      optional
    <key>.mcg       int32   []  (0-dim)        codebook selector, optional
    <key>.mul1      int32   []  (0-dim)        codebook selector, optional
    <key>.su/.sv    int16   packed signs        legacy, pre-v0.0.2

Verified against `turboderp/Llama-3.2-1B-Instruct-exl3` @ 3.0bpw: `q_proj`
2048x2048 → trellis `[128, 128, 48]`; `k_proj` 2048x512 → `[128, 32, 48]`;
`down_proj` 8192x2048 → `[512, 128, 48]`; `lm_head` 2048x128256 → `[128, 8016,
96]` (head_bits 6). K is recoverable from the shape alone: 256 weights per 16x16
tile at K bits pack into 16*K int16.

`mcg`/`mul1` are read by the kernels as *booleans* — `reconstruct(w, trellis, K,
bool mcg, bool mul1)` — the multiplier constants are compiled in. Their presence
is implied by the `codebook` field in `quantization_config`, which is what the
plugin keys off, because vLLM treats any registered-but-unloaded parameter as a
fatal error (`default_loader.py:466`).

### Two findings that change the feasibility report

**1. The tensor-parallel question is answered, and exllamav3 answers it.**

The feasibility report called TP-vs-Hadamard "the single largest open technical
question". It is already solved upstream: exllamav3 ships its own tensor
parallelism (`LinearEXL3.tp_import_split`), and its splitting rule is explicit.

- Output split: `suh` replicated whole, `svh` sliced at `first`, trellis sliced
  on dim 1 at `first // 16`, bias sliced.
- Input split: `suh` sliced, `svh` replicated whole, trellis sliced on dim 0 at
  `first // 16`, bias taken only on rank 0.
- The planner declares `TPAllocation(channel_width = 128, channels_to_split =
  out_features // 128)`.

So the rule is: **storage is tile-granular (16), but correctness is
Hadamard-block-granular (128)**. A split on a multiple of 128 is exact, because
the transform is block-diagonal and no block straddles the boundary. A split
that is not is silently wrong, not an error.

This is not automatically satisfied by the TP degrees vLLM uses. Llama-3.2-1B
has 8 KV heads of dim 64, so `k_proj`/`v_proj` produce 512 channels: fine at
TP=2 and TP=4, **broken at TP=8** (64 channels per rank). `format.check_tp_split`
encodes the rule and the test suite pins that specific case. Phase 2 is
therefore much smaller than budgeted, but it needs a real guard rather than an
assumption.

**2. Fused QKV cannot use `exl3_mgemm`, at any phase.**

The report suggested reusing the pointer-table MoE kernel for merged
projections. That does not work, for a reason that is structural rather than
incidental: **EXL3 assigns bit widths per tensor, and mixed-bpw checkpoints use
different widths inside one layer.** `Llama-3.2-1B-Instruct-exl3` @ 3.5bpw has
q=4, k=5, v=5 bits in every single layer. Different K means a different trellis
last dimension, so there is no concatenated representation to build — and
`MultiLinear`, the wrapper over `exl3_mgemm`, asserts `all(l.inner.K == self.K)`
along with equal in/out features.

A merged linear must therefore be N separate weights, N launches, and a concat —
the same shape as what the GGUF plugin does. The plugin is built that way from
the start.

Two smaller consequences of the same fact: `suh` also differs per projection
(the sign vector is shared across a Hessian group, but `regularize()` then folds
in per-input-channel RMS of *that* weight plus a per-tensor global scale), so
even equal-K projections cannot share one input transform; and a single
concatenated `suh` is meaningless.

### Other constraints found

- **fp16 only.** exllamav3's kernels are fp16 throughout. Most EXL3 repos
  inherit `torch_dtype: bfloat16` from their base model, so vLLM defaults to
  bfloat16 and `vllm/config/vllm.py:728` hard-errors. Serving needs an explicit
  `--dtype float16`. Loud failure, documented.
  **Superseded — see [kernels.md](kernels.md) "bfloat16 activations".** `exl3_mm`
  casts at the kernel boundary, so `get_supported_act_dtypes` now returns both and
  the explicit `--dtype float16` is no longer needed.
- **Dimensions are padded to 128 on disk.** `Linear(pad_to = 128)` rounds both
  dimensions up before quantizing, so stored dims can exceed the model's real
  dims (gpt-oss: 2880 → 2944). Padded output columns hold quantization noise and
  padded input rows only ever multiply zeros, so trimming both is exact —
  `process_weights_after_loading` does that trim.
- **Quantized `lm_head` needs its own method class.** EXL3 quantizes it at
  `head_bits`; vLLM's `ParallelLMHead` is a `VocabParallelEmbedding`, not a
  `LinearBase`. Supported — see "The lm_head problem" below.
- **No loader or config-parser patching is needed.** `quant_method: "exl3"` in
  `config.json` is picked up by vLLM's normal detection
  (`vllm/config/model.py:1128`), and `register_quantization_config` appends to
  both `QUANTIZATION_METHODS` and `current_platform.supported_quantization`.
  This is the whole set of GGUF monkeypatches that EXL3 does not need.
- **`override_quantization_method` now takes `hf_config`.** vLLM main calls it
  with that keyword on every registered method. The GGUF plugin's signature
  still omits it, which would `TypeError` — a live example of the
  "everything below the registration API moves fast" warning. This plugin does
  not override the method at all, so it is unaffected.

## The checkpoint is not a complete description of itself

Worth stating plainly, because it shapes how much any consumer can trust: **an EXL3
checkpoint can carry numerical conventions recorded nowhere in the checkpoint** —
not `config.json`, not `quantization_config.json`, not the tensor metadata. Some
live only as literals in exllamav3's per-architecture Python
(`exllamav3/architecture/*.py`), and a consumer reading only the files gets a
silently wrong model rather than an error.

The worked example is Laguna's `interm_div = 128.0`, where the divisor is baked
into the stored weights while the compensating multiply exists only in
`architecture/laguna.py` — so the checkpoint's own stated
`moe_routed_scaling_factor` is correct for the *original* weights and wrong by
exactly 128 for the ones on disk. Full account in [moe.md](moe.md).

Two consequences that generalize past that one model:

- **A per-layer oracle cannot catch this class of bug.** The layer is exact *given
  its inputs*; a scale applied outside it survives every check that recomputes the
  layer from dequantized weights. When a new model produces degenerate output but
  its layers verify exact, suspect a scale convention before suspecting kernels.
- **Recover such constants by measuring the weights, not by hardcoding per
  architecture**, and raise rather than guess — `format.infer_interm_divisor` is
  the pattern: it measures the ratio, snaps to a power of two, and refuses anything
  that is neither ~1 nor near one. The wrong constant here is a silent factor-of-N
  error in every routed expert, not a crash.

The useful control for detecting one is a tensor the same converter treated
*differently* in the same layer — Laguna's shared expert (`interm_div = 1.0`)
against its routed experts.

## Layout

    vllm_exl3_plugin/
      __init__.py          register()
      plugin.py            idempotent entry point (vllm.general_plugins)
      format.py            on-disk format arithmetic — no torch, unit-tested
      ops.py               lazy wrappers over exllamav3_ext
      quantization/
        config.py          EXL3Config
        linear.py          EXL3LinearMethod, weight_loader_v2-native
        lm_head.py         EXL3LMHeadMethod for quantized output projections
    tests/
      test_format.py       no GPU, torch or vLLM needed
      test_kernels.py      oracles against exllamav3 itself
      test_codebooks.py    mcg/mul1 + head detection
      test_e2e.py          full vLLM generations
      remote_tensors.py    ranged reads of one layer from a remote checkpoint

(`tests/test_tp.py` joined them in [tensor-parallel.md](tensor-parallel.md).
Counts are deliberately not recorded here — see README for the current total.)

`format.py` deliberately has no torch dependency so the shape rules — where a
format misunderstanding turns into silently wrong output — stay testable
anywhere.

`ops.py` depends on the compiled extension (`exllamav3_ext`, a *top-level*
module) and not on the `exllamav3` Python package. Importing the package runs an
`__init__.py` that drags in the model, tokenizer, cache and generator stack plus
formatron, kbnf, marisa_trie and flash-linear-attention — dependency-resolution
risk against vLLM's own pins, for code a worker never uses. Note the extension
requires `import torch` first in the process (it links `libc10.so`); `ops.py`
imports torch at module scope, so `ext()` is always safe.

## Verified

Environment: RTX 5070 Ti (sm_120, 16 GiB), torch 2.13.0+cu130, CUDA 13.3,
editable vLLM `edbc4969a`, exllamav3 v1.3.0 built from the submodule.

1. **The vLLM API surface still matches.** Re-checked on `edbc4969a`:
   `register_quantization_config`, the four `BasevLLMParameter.load_*` methods,
   `register_weight_loader_v2_supported_method`, `maybe_update_config` and
   `get_hf_file_to_dict` all have the signatures the plugin codes against.
2. **Registration works through the entry point.** `load_general_plugins()` →
   `get_quantization_config("exl3")` resolves to `EXL3Config`, and `"exl3"`
   lands in `QUANTIZATION_METHODS`.
3. **The dequantization is bit-exact.** `ops.dense_weight` equals
   `LinearEXL3.get_weight_tensor()` with `rtol=0, atol=0` on `q_proj`, `k_proj`
   and `down_proj` of a real checkpoint, and agrees with the actual fused
   `exl3_gemm` forward path within fp16 tolerance. The locally constructed
   Sylvester-Hadamard matches `exllamav3.util.hadamard.get_hadamard` exactly at
   orders 16/32/64/128.
4. **The load path works.** `EXL3Parameter`'s opt-out of vLLM's preallocated
   parameter protocol drives cleanly through `QKVParallelLinear`,
   `MergedColumnParallelLinear` and `RowParallelLinear`; picking the device up
   from `torch.empty(0).device` in `create_weights` lands correctly.
5. **End to end.** `turboderp/Llama-3.2-1B-Instruct-exl3` generates coherent
   text at 3.0bpw *and* at 3.5bpw. The 3.5bpw case is the one that matters: it
   has q=4, k=5, v=5 bits inside a single merged QKV linear, so it exercises the
   per-shard design directly. Model loading reports 2.35 GiB — i.e. the full
   fp16 model, exactly as the dequantize-at-load strategy predicts.
6. **The "no quantization_config.json" fallback works.** The 0.0.0-era test repo
   has no such file; `maybe_update_config` swallows the miss and falls back to
   treating every linear as quantized, which is correct for it.

Reproduce with `python -m unittest discover -s tests` (30 tests at the time, ~45s
on the above machine once the checkpoints are cached).

## Remaining gaps

Phase 0 scope, deliberately:

- Dequantize-at-load means no memory saving at all. That is Phase 1's job.
- TP > 1 raises `NotImplementedError`. The rule is known and encoded
  (`format.check_tp_split`) but no sharded path exists yet.
- MoE is untouched (Phase 3).

Not yet exercised, and worth knowing before they bite:

- **Bias.** Llama has none on these projections, so padded-output bias handling
  is still theoretical.
- **`get_min_capability` / `get_supported_act_dtypes` rejection messages.**
- **A `mul1` checkpoint end to end.** `mcg` is now covered by MiniCPM5-1B; the
  only `mul1` repos are 12B+, which Phase 0's dequantize-at-load cannot fit.
  The decode math is verified either way.

## Codebooks, and testing against models that do not fit

EXL3's newer procedural codebooks (`mcg`, then `mul1`) only appear in repos from
~v0.0.12 on, which are large. Under Phase 0's dequantize-at-load, gemma-4-12B at
3bpw is ~5 GB on disk but ~24 GB resident — untestable on a 16 GB card.

Validating the *format* does not need the model, though. `tests/remote_tensors.py`
reads one module's tensors straight out of a remote safetensors shard using HTTP
range requests, which is a few MB regardless of repo size. On that basis:

- `ops.dense_weight` is bit-exact against exllamav3 under **`mcg`**
  (`MiniCPM5-1B-exl3@3.00bpw`) and **`mul1`**
  (`gemma-4-12B-it-exl3@3.00bpw_mul1`), `rtol=0, atol=0`.
- A guard test confirms decoding the same trellis with and without `mcg`
  produces *different* weights, so the above is not vacuously passing. The flags
  are booleans selecting a compiled-in multiplier; passing the wrong one is not
  an error, just silently wrong numbers.
- `mcg`/`mul1` are stored as **0-dim** int32 scalars, not `[1]` tensors.
- The `tensor_storage` path and the quantized-`lm_head` rejection both fire
  correctly, and the no-`quantization_config.json` fallback still works.

This technique generalizes: any format question about any repo, however large,
can be answered without downloading it.

## The lm_head problem (solved)

Every EXL3 repo from ~v0.0.12 onward sets `head_bits`, i.e. quantizes `lm_head`.
That is invisible on tied-embedding models, because vLLM skips `lm_head.*`
entirely — which is exactly why the original Phase 0 target worked. But of the
small models available, every one with a modern codebook is untied:

| candidate | size | codebook | head |
|---|---|---|---|
| `Llama-3.2-1B-Instruct-exl3` | 1B | none (3inst) | tied — skipped |
| `Qwen3-0.6B-exl3` | 0.6B | none (3inst) | tied — skipped |
| `MiniCPM5-1B-exl3` | 1B | mcg | untied, quantized at 6bpw |
| `nanochat-d34-exl3` | 0.5B | mcg | untied (arch not in vLLM anyway) |
| `gemma-4-12B-it-exl3` | 12B | mul1 | untied |

So quantized `lm_head` was not a Phase 1+ nicety: without it there was **no
checkpoint at all** that could exercise the new codebooks end to end, and no way
to serve essentially any model above ~3B (large models rarely tie embeddings).

`quantization/lm_head.py` implements it. Two things differ from the linear case:

- **The weight loader.** `VocabParallelEmbedding` passes its own v1-style
  `weight_loader(param, loaded_weight)`, which assumes a preallocated tensor it
  can `narrow` along a vocab dimension. An EXL3 trellis is tile-granular with
  its own padding, so that cannot work; we substitute a loader that stores,
  the same way the GGUF plugin substitutes `_gguf_embedding_weight_loader`.
- **Two independent vocab paddings.** vLLM pads the vocabulary to a multiple of
  64; exllamav3 padded the output dimension to a multiple of 128 before
  quantizing. Both can exceed `org_vocab_size` and they need not agree — 128 is
  a multiple of 64, so EXL3's padding is always the wider of the two, but the
  two round differently whenever `org_vocab_size` falls between. EXL3's padded
  rows hold quantization *noise* rather than zeros, so they are trimmed and the
  result is zero-padded to whatever vLLM allocated, matching the convention
  `VocabParallelEmbedding.weight_loader` uses for unquantized heads.

Deciding *whether* a head is quantized has to be exactly right, because being
wrong either way is fatal rather than degraded: registering parameters the
checkpoint never fills makes `default_loader` reject the model, and not
registering them leaves `lm_head.trellis` unclaimed. `head_is_quantized()`
resolves it as: tied ⇒ no; else `tensor_storage` if present; else the presence
of `head_bits`. Note that vLLM *constructs* a `ParallelLMHead` even for tied
models and only ties it afterwards, so "tied" genuinely has to be checked rather
than inferred from the layer.

Verified end to end on `MiniCPM5-1B-exl3@3.00bpw` — untied 6-bit head, `mcg`
codebook, `tensor_storage` path — with the tied-head case still passing.

## Phase 1 preview

Replace `apply` with a `direct_register_custom_op` wrapper over `exl3_gemm` /
`exl3_gemv`, keeping the quantized tensors resident. The batch-dependent
dispatch (four paths, including a full dequant above 144 rows) has to live
*inside* the opaque op so `torch.compile` and CUDA-graph capture see a stable
signature, and every op needs a `register_fake`. The open questions there are
the cooperative-launch lock buffer and the on-disk autotune cache under
multi-process workers.


## CPU offload: why vLLM's UVA offloader skips EXL3 entirely

*Traced from TODO, where it had accumulated as an investigation log. The work
itself is open — TODO `cpu-offload`.*

`vllm serve --cpu-offload-gb` silently offloads nothing for EXL3. The log reports
`Total CPU offloaded parameters: 0.01` on an EXL3 checkpoint where the same model
as AWQ offloads 3.63 GiB under otherwise identical settings.

There are two independent causes, and the second is the harder one.

**Construction-time eligibility.** The UVA offloader
(`vllm/model_executor/offloader/uva.py`) decides per-module eligibility at
construction time, before checkpoint weights load, by peeking at
`next(module.parameters()).device`. Our `EXL3Parameter` placeholders
(`create_weights`, `linear.py:184`) are `Parameter(data=None)` — a default empty
*CPU* tensor — so every EXL3-quantized layer reads as "already on CPU" and gets
skipped outright, before any real weight exists. The 0.01 GiB that does get
offloaded is just the ordinary dense params living directly on decoder-layer
submodules outside `EXL3LinearMethod` (RMSNorm weights and the like).

**Parameter replacement.** Even if that check passed,
`process_weights_after_loading` (`linear.py:272`) replaces the placeholders with
brand-new `Parameter` objects (`exl3_trellis_N` / `exl3_suh_N` / `exl3_svh_N`, or
`exl3_weight` on the dequantize path) built fresh on-device. The offloader never
sees these — it wrapped modules once, at construction, before the replacement
happened. AWQ does not hit this because it preallocates its weight tensor
on-device at construction and mutates it in place the whole way through, so the
offloader's construction-time view stays attached to the same tensor that is still
serving inference later.

**Considered and rejected: match AWQ's pattern.** Preallocating the correct final
shape at construction needs the per-tensor bit width `K` known upfront, since
trellis shape depends on it. Checkpoints since ~v0.0.2 carry a `tensor_storage` map
in `quantization_config.json` recording bit width per module (already fetched in
`config.py`'s `_load_tensor_storage`, just not for this purpose) — but it is known
to be incomplete on real checkpoints (`Muse-Glimmer-30B-exl3` omits 303 quantized
modules), does not exist pre-v0.0.2 at all, and improving it going forward would
not retroactively fix what is already published. More to the point, the project is
moving toward repaired-only checkpoints anyway (see [embeddings.md](embeddings.md)), so
betting on
upstream checkpoint metadata quality is not where the value is.

**Chosen direction.** Do not make EXL3 loading conform to "preallocate and mutate
in place". `process_weights_after_loading` already holds the finished
trellis/suh/svh tensors, at real final shape, at exactly the moment it builds them.
Reach into vLLM's offloader singleton there (`get_offloader()`,
`vllm/model_executor/offloader/base.py:111`), and if it is a `UVAOffloader` with
remaining `cpu_offload_max_bytes` budget, do what `uva.py`'s `_maybe_offload_to_cpu`
does: pin the tensor, wrap it with `get_accelerator_view_from_cpu_tensor`, and
register that instead of a plain on-device `Parameter` — updating the offloader's
own byte counter so later layers' budget accounting stays correct.
Bits-agnostic, checkpoint-vintage-agnostic, no vLLM changes, no plugin storage-path
refactor. Covers only the UVA backend, not `PrefetchOffloader` (a different
per-layer streaming design), which is acceptable since UVA is close to universal on
the CUDA-only stack this project already requires.

**Known cost, going in with eyes open.** This reaches into a vLLM internal —
`get_offloader()` / `UVAOffloader` are not public API — which is one more surface
that can silently break on a vLLM version bump, the same standing risk already
taken on with `exl3_mgemm`'s call sites across the exllamav3 fork transition. Worth
it here: it is the only way to make CPU offload work for EXL3 at all.

## transformers 5.15: heterogeneous configs

transformers 5.15 formalised per-layer configuration. Attributes that differ
across layers now live in `config.per_layer_config[i]`, and reading one off the
top-level config raises `AmbiguousGlobalPerLayerAttributeError`.

Two properties of that make it more disruptive than it first looks:

- **It subclasses `RuntimeError`, not `AttributeError`.** So the defensive
  idiom `getattr(config, name, default)` does not catch it, `hasattr` does not
  return False, and `config_a == config_b` raises while comparing. Every
  defensive read of a per-layer attribute propagates.
- **The old top-level aliases can disappear.** gemma-4's `global_head_dim` is
  folded into the per-layer entries and is simply absent afterwards, so
  `getattr(config, "global_head_dim", config.head_dim)` silently returns the
  *sliding* dimension for full-attention layers -- 256 where the checkpoint has
  512 -- and fails much later as `Attempted to load weight (512) into parameter
  (256)`.

The documented escape hatch (`allow_global_per_layer_attribute_access = True`)
is **not** a fix for that second point: it restores the global read, but the
global value is the wrong one. Verified -- it gets past config resolution and
then fails at weight loading.

`patches/vllm-gemma4-transformers-5.15-per-layer.patch` reads the per-layer
configs at the five sites gemma-4 reaches, and takes the max where vLLM wants a
single number for buffer sizing (matching `get_num_experts_from_block_configs`,
which already did this for NemotronH). For gemma-4-12B the per-layer max
reproduces the pre-5.15 values exactly: `head_dim` max(256, 512) = 512, and
`num_key_value_heads` max(1, 8) = 8 = its old top-level value.

Worth expecting again: Laguna already has per-layer head counts, and Muse
Glimmer has per-layer rope theta and layer types. Heterogeneous configs are
becoming normal, and this class of break will recur.
