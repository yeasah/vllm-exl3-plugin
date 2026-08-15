# Serving models vLLM does not implement

*EXL3 through vLLM's Transformers backend (`--model-impl transformers`), which
builds a model from the transformers modelling definition when vLLM has no native
implementation of the architecture. Tracked in TODO as `transformers-backend`.*

**Status: it works.** An EXL3 checkpoint served through the Transformers backend is
token-for-token identical to the same checkpoint served through vLLM's native
implementation. Text-only models need no changes at all; multimodal models need one
small vLLM patch. That moves the plugin's model coverage from "what vLLM implements
natively" to approximately "what transformers implements".

## Why it was in doubt

The plugin hooks vLLM's own layer construction. `get_quant_method()` dispatches on
vLLM module prefixes, and `EXL3Config.apply_vllm_mapper` translates checkpoint names
into vLLM's naming for a given architecture — both of which assume vLLM built the
model. The Transformers backend does not run transformers' modules; it takes the
modelling *definition* and substitutes vLLM layers into it, so neither the prefixes
nor the mapper's assumptions were guaranteed to hold.

The expected failure mode was the one this project keeps meeting: modules silently
classified as unquantized, then a dense fp16 allocation that OOMs pointing nowhere
near the cause.

## What actually happens

Instrumenting `get_quant_method` on `turboderp/Muse-Glimmer-30B-exl3` @2.00bpw
(`MuseGlimmerForConditionalGeneration`, which vLLM has no implementation for, and
which transformers 5.15 does support), 622 calls:

| result | count | |
|---|---|---|
| `EXL3LinearMethod` | 567 | every quantized linear, language model **and vision tower** |
| `None` | 53 | all `Attention` modules — the KV-cache-quant question, not linears |
| `UnquantizedLinearMethod` | 1 | `model.vision_tower.patch_embedder.patch_embedding`, correctly |
| `EXL3LMHeadMethod` | 1 | the quantized head |

Prefixes resolve correctly (`model.vision_tower.layers.0.attn.qkv_proj` and so on),
so the mapper and the index-based `is_quantized` both survive. Worth noting this
model is the case that forced the safetensors-index-as-ground-truth decision — its
`tensor_storage` omits all 303 vision-tower modules — and the vision tower is
separately quantized at `vision_bits: 4`. Both held up.

## The one blocker: `ReplicatedLinear` has no `weight_loader_v2`

`ColumnParallelLinear` (`linear.py:512`) and `RowParallelLinear` (`:1704`) both pick
between `weight_loader_v2` and `weight_loader` based on
`WEIGHT_LOADER_V2_SUPPORTED`. `ReplicatedLinear` passes its v1 loader
unconditionally and defines no v2 method at all.

So `EXL3Parameter` placeholders — `Parameter(data=None)`, i.e. size `[0]` — reach
the generic loader, which asserts an exact size match:

    AssertionError: Tried to load weights of size torch.Size([1])
                    to a parameter of size torch.Size([0])

The `[1]` is an `mcg`/`mul1` 0-dim codebook scalar, reshaped by `linear.py:388`.

**This is reachable only through the Transformers backend, and only for models with
unsharded submodules.** In-tree vLLM models use the sharded linear classes for
quantized layers; the Transformers backend uses `ReplicatedLinear` for everything it
does not shard — 154 layers in Muse-Glimmer, essentially the whole vision tower.
Measured directly: MiniCPM5-1B and Llama-3.2-1B both load and generate correctly
through the Transformers backend with **no** patch, because their decoder layers are
all column/row/QKV/merged-column.

`patches/vllm-replicated-linear-weight-loader-v2.patch` adds the missing branch and
a `weight_loader_v2` that does a whole-tensor load, since a replicated layer
partitions nothing. `BasevLLMParameter.load_row_parallel_weight` is exactly that
no-narrowing load, and is what `PerTensorScaleParameter` already relies on for the
same reason. The patch reads as a consistency fix rather than an EXL3 special case,
so it is a plausible upstream contribution.

Known limit, to check before proposing it upstream: a parameter class that overrides
`load_row_parallel_weight` to *narrow* (`RowvLLMParameter` and its subclasses) would
compute a shard offset from `tp_rank` against an unpartitioned tensor. Nothing
reaches `ReplicatedLinear` that way today, because the path did not exist before.

## Correctness: token-for-token against the native path

The useful experiment is not a big model — it is the same checkpoint under both
implementations, which separates "does the backend serve EXL3 correctly" from any
question about a particular checkpoint.

`turboderp/MiniCPM5-1B-exl3` @3.00bpw — untied, EXL3-quantized `lm_head` at 6 bits,
`mcg` codebook — greedy, through the chat template, 24 tokens:

| | token ids | text |
|---|---|---|
| `model_impl="vllm"` | `[8, 220, 608, 5390, 14293, 374, ...]` | *"…The capital of France is Paris"* |
| `model_impl="transformers"` | identical | identical |

Per-step top-1 logprobs differ by at most ~0.035, on soft-confidence tokens only,
never changing an argmax — the same tile-shape-dependent fp16 accumulation noise
already characterized for tensor parallelism in
[tensor-parallel.md](tensor-parallel.md), and expected here because the two backends
fuse linears differently and so accumulate in a different order.

`turboderp/Llama-3.2-1B-Instruct-exl3` @3.0bpw (tied) also generates correctly
through the backend: *"The capital of France is Paris."*, every logprob >= -0.002,
stopping on EOS.

## Open: Muse-Glimmer's output quality

Loading is not the same as working. `Muse-Glimmer-30B-exl3` @2.00bpw loads (3.37 GiB
KV cache, 67,856 tokens, ~21 tok/s eager on a 16 GiB card) but does not produce
usable text:

- raw completion prompt → garbage, which is the documented BOS trap (see README) and
  expected;
- through the chat template → a single token `200008` = `<|eot|>`, at logprob
  **-0.0025**, on every prompt tried.

That is *confident*, not degenerate — a degenerate uniform distribution over this
200k vocabulary would be about -12.2 — so it is not the all-zero-logits failure seen
on Laguna. Candidates, unranked: the chat template needs something not being given
(the `<|eom|>`/`<|eot|>` layout at 200007/200008 suggests a channel-style template
where the assistant must open a channel), genuine damage at 2.00bpw, which is the
most aggressive rate and where the body sits while the vision tower is at 4 bits, or
something in this path that only shows up on this model.

None of this is attributable to the backend, which the MiniCPM result isolates.
