# Phase 0 — registration spike

Goal, unchanged from the feasibility report: **prove the registration path and
confirm an EXL3 checkpoint loads and generates correct tokens through vLLM's
forward pass**, single GPU, TP=1, no CUDA graphs, no fused kernels.

Status: the plugin is written and the format arithmetic is under test. Nothing
has been executed against a GPU — this environment has no CUDA device, no torch
and no vLLM installed. Everything below marked "unverified" is exactly that.

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

Read out of exllamav3 v1.3.0 (`0b9745c5`), vLLM `main` @ `4a3447d2` (2026-08-03),
`vllm-gguf-plugin` @ `main`, and the safetensors headers of real EXL3 repos on
the Hub.

### On-disk layout, per quantized linear

    <key>.trellis   int16   [in_features // 16, out_features // 16, 16 * K]
    <key>.suh       fp16    [in_features]
    <key>.svh       fp16    [out_features]
    <key>.bias      fp16    [out_features]      optional
    <key>.mcg       uint32  [1]                 codebook selector, optional
    <key>.mul1      uint32  [1]                 codebook selector, optional
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
- **Dimensions are padded to 128 on disk.** `Linear(pad_to = 128)` rounds both
  dimensions up before quantizing, so stored dims can exceed the model's real
  dims (gpt-oss: 2880 → 2944). Padded output columns hold quantization noise and
  padded input rows only ever multiply zeros, so trimming both is exact —
  `process_weights_after_loading` does that trim.
- **Quantized `lm_head` is not supported yet.** EXL3 quantizes it at
  `head_bits`; vLLM's `ParallelLMHead` is a `VocabParallelEmbedding`, not a
  `LinearBase`, so it needs a different method class. Tied-embedding models are
  unaffected — vLLM skips `lm_head.*` outright (`llama.py:538`) — which is why
  the Phase 0 target is a tied model.
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

## Layout

    vllm_exl3_plugin/
      __init__.py          register()
      plugin.py            idempotent entry point (vllm.general_plugins)
      format.py            on-disk format arithmetic — no torch, unit-tested
      ops.py               lazy wrappers over exllamav3_ext
      quantization/
        config.py          EXL3Config
        linear.py          EXL3LinearMethod, weight_loader_v2-native
    tests/
      test_format.py       14 tests, runnable with no GPU/torch/vLLM

`format.py` deliberately has no torch dependency so the shape rules — where a
format misunderstanding turns into silently wrong output — are testable now
rather than after hardware arrives.

## Not yet done / unverified

Nothing here has run. In rough order of risk:

1. **The whole load path.** `EXL3Parameter` opts out of vLLM's preallocated
   parameter protocol (it has to: trellis shape depends on a bit width not known
   until the tensor is seen) by implementing the four `load_*` methods as pure
   stores. Whether vLLM's `AutoWeightsLoader` drives that cleanly for
   `QKVParallelLinear` and `MergedColumnParallelLinear` is unverified.
2. **Device placement.** `create_weights` picks the device up from
   `torch.empty(0).device`, relying on vLLM building the model inside a device
   context. Plausible, unverified.
3. **The dequantization math.** Should equal `LinearEXL3.get_weight_tensor()`
   exactly. Untested — this is the first thing to run.
4. **`quantization_config.json` fetch.** `maybe_update_config` pulls it via
   `get_hf_file_to_dict`; failure is swallowed and falls back to "everything is
   quantized". Untested against both a repo that has it and one that does not.
5. Bias handling on padded output dimensions.
6. Whether `get_min_capability` / `get_supported_act_dtypes` rejections produce
   the intended messages.

## First runs, once there is a GPU

1. `python -c "import vllm_exl3_plugin; vllm_exl3_plugin.register()"` — then
   check `get_quantization_config("exl3")` resolves.
2. Dequantization oracle: load one tensor from a real checkpoint with
   exllamav3's own `LinearEXL3`, call `get_weight_tensor()`, and compare against
   `ops.dense_weight` on the same tensors. Expect exact equality.
3. `vllm serve turboderp/Llama-3.2-1B-Instruct-exl3 --revision 3.0bpw
   --dtype float16 --enforce-eager` and generate. Tied embeddings, all dims
   multiples of 128, uniform K=3, no `mcg`/`mul1`, no bias — the easiest
   possible target.
4. Then the same at 3.5bpw, which exercises mixed K inside a merged linear.

## Phase 1 preview

Replace `apply` with a `direct_register_custom_op` wrapper over `exl3_gemm` /
`exl3_gemv`, keeping the quantized tensors resident. The batch-dependent
dispatch (four paths, including a full dequant above 144 rows) has to live
*inside* the opaque op so `torch.compile` and CUDA-graph capture see a stable
signature, and every op needs a `register_fake`. The open questions there are
the cooperative-launch lock buffer and the on-disk autotune cache under
multi-process workers.
