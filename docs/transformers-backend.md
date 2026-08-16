# Serving models vLLM does not implement

*EXL3 through vLLM's Transformers backend (`--model-impl transformers`), which
builds a model from the transformers modelling definition when vLLM has no native
implementation of the architecture. Tracked in TODO as `transformers-backend`.*

**Status: it works.** An EXL3 checkpoint served through the Transformers backend is
token-for-token identical to the same checkpoint served through vLLM's native
implementation, and — on Muse-Glimmer, which vLLM has no native implementation of —
to native exllamav3. Text-only models on plain architectures need no changes at all;
multimodal models, and models whose architecture puts arithmetic in places the
backend does not look, need vLLM patches. That moves the plugin's model coverage
from "what vLLM implements natively" to approximately "what transformers
implements".

**The recurring hazard is not the plugin.** Every defect found here is the backend
dropping something the transformers modelling code does *outside* the plain
module graph, and doing it silently — see [What the backend drops](#what-the-backend-drops).

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

Per-step top-1 logprobs over those 24 generated steps differ by at most ~0.035, on
soft-confidence tokens only, never changing an argmax — the same tile-shape-dependent
fp16 accumulation noise already characterized for tensor parallelism in
[tensor-parallel.md](tensor-parallel.md), and expected here because the two backends
fuse linears differently and so accumulate in a different order.

**Refined once `bench/` gated both paths**, which scores 75 teacher-forced prompt
positions rather than 24 generated steps: max |Δlogprob| **0.190**, max KL 0.016, and
**one** argmax disagreement out of 75. The greedy continuations remain identical, so
the token-for-token claim above stands as written — but the per-position divergence is
about 5x the generated-step figure, and it is not quite true that an argmax never
moves. Both entries are in the fast tier (`minicpm5-1B ... mcg` and `... via
transformers backend`), each gated against its own baseline; the pair is what keeps
this claim honest as vLLM changes underneath it.

`turboderp/Llama-3.2-1B-Instruct-exl3` @3.0bpw (tied) also generates correctly
through the backend: *"The capital of France is Paris."*, every logprob >= -0.002,
stopping on EOS.

## What the backend drops

Muse-Glimmer was the case that surfaced this. `Muse-Glimmer-30B-exl3` @2.00bpw
loaded (3.37 GiB KV cache, 67,856 tokens, ~21 tok/s eager on a 16 GiB card) and
emitted confident nonsense — an immediate `<|eom|>`/`<|eot|>` at a logprob near
zero. The checkpoint was fine and so was the plugin: **native exllamav3 on the same
token ids generates correctly at 2.00bpw**, opening the template's reasoning channel
and reasoning coherently:

    ' to=self<|message|>What is the capital of France?\n\nWe should answer.
     Probably Paris. Simple.\n\nWe can respond with Paris...'

Two independent defects, both in vLLM's Transformers backend, both silent, and
neither EXL3-specific — an unquantized Muse-Glimmer served with
`--model-impl transformers` has both.

### 1. The dropped embedding norm — the actual failure

`base.py` substitutes the model's input embedding wholesale with a
`VocabParallelEmbedding`. It recognizes exactly one form of custom behaviour, a
scalar `embed_scale` attribute, and discards everything else. Muse-Glimmer's
embedding is `MuseGlimmerTextNormedEmbedding`:

```python
def forward(self, input_ids):
    return self.embed_norm(super().forward(input_ids))
```

an **unweighted** RMSNorm over the looked-up rows. No `embed_scale`, so it is
dropped — and being unweighted, no weight goes unloaded, so nothing warns. The
residual stream starts at the wrong scale from token zero.

Confirmed from both directions, which is what makes it the whole cause rather than
a contributor. Deleting the same norm from native exllamav3 (which models it as
`model.language_model.embed_tokens.embed_norm`) reproduces vLLM's output **token for
token**:

| | greedy ids |
|---|---|
| vLLM, unpatched | `[200007, 198, 200007, 191099, 845, 845, 200008]` |
| native, `embed_norm` ablated | `[200007, 198, 200007, 191099, 845, 845, 200008]` |

both decoding to `'<|eom|>\n<|eom|>ிட��<|eot|>'`. Fixed by
`patches/vllm-transformers-backend-embedding-postprocess.patch`.

### 2. The dropped logit transform — invisible behind the first

transformers applies Muse-Glimmer's logit transform in
`MuseGlimmerForConditionalGeneration.forward`, one level *above* `self.model(...)`,
which is all the backend runs:

    logits = logits * output_multiplier                # 0.19611613513818404
    logits = T * tanh(logits / T)                      # final_logit_softcapping = 20

The backend's own `compute_logits` reads an attribute named `logit_scale` (this
model spells it `output_multiplier`, so it defaults to 1.0) and never passes
`soft_cap`. Neither transform survives.

Both are monotonic, so **greedy decoding is unaffected** — which is why this hid
behind the first defect instead of causing a visible failure. What it corrupts is
every reported logprob and every temperature-dependent sampling decision: the
effective temperature is ~5.1x too low. This also invalidates the reasoning that
originally made this look like a template problem — the observed `-0.0025` was an
artifact of the missing transform, not the model's real confidence, which is nearer
`-0.27`.

`LogitsProcessor` applies its `scale` *after* the cap, so the two cannot simply both
be passed. But it can express the intended order exactly, by folding the multiplier
into the cap:

    tanh(z / (T/m)) * (T/m) * m  ==  T * tanh(z * m / T)

so `soft_cap = T/m` with `scale = m`, reducing to the plain cap at `m == 1`. Fixed by
`patches/vllm-transformers-backend-logit-softcap.patch`, with no change to
`LogitsProcessor` itself.

### Who else these hit

Neither defect involves the quantization path, so the population is "whatever goes
through the Transformers backend" — which means unquantized checkpoints too. Surveyed
against the installed transformers (5.15.0):

**The dropped embedding norm: Muse-Glimmer only, today.** Of the ~48 `nn.Embedding`
subclasses in transformers, almost all are the `*ScaledWordEmbedding` family (Bart,
Whisper, M2M100, MBart, XGLM, MiniCPM3, every Gemma) — already covered by the
backend's `embed_scale` branch, and invisible to the new detection because
`embed_scale` is an `nn.Buffer`, not a submodule. `MuseGlimmerTextNormedEmbedding` is
the only one applying a *module* to the lookup. So the bug class is general but the
current membership is one architecture — and vLLM has no native implementation of it,
so **`meta-models/Muse-Glimmer-30B` in plain bf16 is broken on vLLM in exactly the way
our EXL3 copy was.** Its parent architecture, `kimi_k25`, has neither the normed
embedding nor the multiplier, and is natively implemented besides.

`IdeficsDecoupledEmbedding` is the one near-miss: one submodule and an overridden
`forward`, but the submodule is a second `nn.Embedding` for a split vocabulary, not a
postprocess. Hence the `isinstance(..., nn.Embedding)` guard.

**The dropped logit transform: a real list.** Architectures whose config carries
`final_logit_softcapping` and which vLLM does *not* implement natively — so the
backend is the only way to serve them, and it drops the cap silently:
`vaultgemma`, `nanochat`, `t5gemma`, `t5gemma2`, `muse_glimmer`. Additionally, anyone
passing `--model-impl transformers` explicitly on gemma-2/3/4 loses it too. Worth
noting for this project: gemma-4-12B-it carries `final_logit_softcapping: 30.0`, and
is safe only because vLLM implements `Gemma4UnifiedForConditionalGeneration`
natively.

**Known remaining gap.** The Granite family (`granite`, `granitemoe`,
`granitemoeshared`, `granitemoehybrid`, `granitemoe_swa`, `granite4_vision`) scales
logits under a third spelling, `logits_scaling`, and *divides* rather than multiplies.
The patch does not cover it — no Granite checkpoint was on hand to verify against, and
guessing at an inverse is how the original bug got written. Cohere's `logit_scale`
needs nothing, being the name the backend already reads.

### Result

With both patches, vLLM matches native exllamav3 on Muse-Glimmer @2.00bpw: 40 greedy
tokens identical, and step-0 top-15 logprobs agreeing to ~0.03 nats — the same fp16
accumulation noise floor characterized above and in
[tensor-parallel.md](tensor-parallel.md).

Regression-checked on MiniCPM5-1B @3.00bpw, which has a plain `nn.Embedding` and no
softcapping: native impl and backend remain identical to each other, to the last
digit of the top-0 logprob (`-2.6226e-06`), and neither new branch fires.

### Ruled out along the way

Worth recording, because each looked plausible and cost nothing to eliminate:

- **The chat template.** It really is channel-style — the generation prompt ends at a
  bare `<|start|>assistant` with no `<|message|>`, so the model must emit its own
  recipient. But it was being applied correctly all along; native produces
  ` to=self<|message|>` from exactly these ids.
- **2.00bpw damage.** Native generates coherently at that rate.
- **The quantized vision tower** (`vision_bits: 4`, a first for this plugin). Not
  reached on a text-only prompt.
- **`qk_scale_factor: 3.87`.** Applied inside the attention module, on `query_states`
  after `qk_norm`, so it survives the backend's attention-interface swap.
- **`MuseGlimmerTextCenteredRMSNorm` → vLLM's `GemmaRMSNorm`.** The substitution is
  correct despite the name: it does not center, it is `x·rsqrt(mean(x²)+eps)·(1+w)`,
  which is exactly `GemmaRMSNorm`.
- **Per-layer RoPE and sliding-window layout** (3× sliding at theta 500000, every 4th
  full at theta 0 = NoPE). transformers handles both inside the module graph the
  backend keeps; the 63-token probe is far under the 2048 window besides.

### The general shape

Both defects are the same failure mode, and it is the one to expect next: the
backend runs the model's *base* module graph, so anything transformers does **above**
it (`ForConditionalGeneration.forward`) or **inside a layer it substitutes**
(the embedding, the norms, the linears) is at risk of being dropped. It is dropped
silently whenever the missing arithmetic carries no weights, because weight-loading
is the only thing that checks.
