# Phase 3 — mixture of experts

Goal from the feasibility report: **MoE via `exl3_moe`/`exl3_mgemm` adapted to
vLLM's `FusedMoE` interface.**

**Status: working on `gemma-4-26B-A4B-it-exl3`** — a 26B MoE in **9.46 GiB**,
answering correctly. `Laguna-XS-2.1-exl3` loads and runs but still generates
degenerate output; see "The Laguna holdout".

## `exl3_mgemm` is usable here, unlike merged QKV

Phase 1 established that a merged QKV *cannot* use exllamav3's pointer-table
kernel: EXL3 assigns bit widths per tensor and real checkpoints mix them inside
one layer (q=4, k=5, v=5), while `MultiLinear` asserts equal K.

Experts do not have that problem. Scanning every expert tensor in Laguna-XS
(121,212 tensors) and gemma-4-26B (47,652) found **a single bit width
throughout**, with no mixed-K group anywhere. `process_weights_after_loading`
re-checks per layer rather than trusting it, because a mixed-K checkpoint would
otherwise decode with the wrong K and produce plausible-looking nonsense.

The kernel is built for exactly this shape of problem: one input broadcast to N
experts, int64 pointer tables, expert-range filtering, and a **fused weighted
reduction** that scales each expert's result by its routing weight and sums into
`C[0]`. vLLM's `apply(layer, x, topk_weights, topk_ids, ...)` maps onto it
almost directly.

## Design

**Parameters are collectors, not one per tensor.** Laguna has 256 experts x 3
projections x 39 layers; a parameter per stored tensor would mean ~120,000
`nn.Parameter` objects. Each layer instead registers eight collectors (`w13_*`
and `w2_*` per EXL3 sub-tensor) that accumulate per-expert tensors as vLLM's
loader delivers them. The `w13_`/`w2_` naming is not ours to choose: vLLM's
expert mapping rewrites `experts.{id}.gate_proj.` to `experts.w13_` and keeps
whatever suffix follows.

**Pointer tables address per-expert tensors directly**, rather than a stacked
`[num_experts, ...]` copy. The kernel dereferences each address independently,
so stacking buys nothing and costs a second full copy of every expert weight
while both exist — on a 256-expert model that is enough to exhaust a 16 GB card
mid-load. exllamav3's own `MultiLinear` builds its tables the same way.

## Four bugs this needed

1. **`is_quantized` could not see MoE layers at all.** The module is named
   `...mlp.experts` while the checkpoint only has `...experts.0.gate_proj`, so an
   exact lookup missed and vLLM silently fell back to **dense fp16 experts** —
   an immediate OOM whose only clue was `Using TRITON Unquantized MoE backend`.
   Fixed by checking quantized *ancestors*, precomputed once.
2. **vLLM reads any 3D checkpoint tensor as packed multi-expert weights**
   (`is_fused = loaded_weight.dim() == 3`), but an EXL3 trellis is natively 3D.
   There is no hook — the rank test is the whole decision — so `plugin.py` lifts
   trellis tensors to 4D on the way in and the loader drops the axis again. This
   is the plugin's first monkeypatch, and it is a *format* collision rather than
   the container-format problem GGUF needs patches for.
3. **`exl3_mgemm` refuses expert-range filtering together with the multi-token
   weighted reduction.** `min_index` must be `-1` without expert parallelism;
   passing `0` looks harmless and raises at the first forward.
4. **Gemma 4 splices `.moe.` into expert paths in its own `_weight_iterator`**
   (`.experts.{id}.` becomes `.moe.experts.{id}.`), which `apply_vllm_mapper`
   never sees because it is not part of `hf_to_vllm_mapper`. So the module sits
   at `...layers.N.moe.experts` while `tensor_storage` says
   `...layers.N.experts.0.gate_proj`. `is_quantized` now also tries the prefix
   with any one interior component removed, accepting only a match against a
   known quantized ancestor.

## The activation is not always silu

gemma-4 uses `gelu_tanh`. The activation now comes from `layer.activation`
rather than being assumed, and an unknown one raises rather than silently
substituting.

Worth recording a near-miss: vLLM's `activation.py` describes gated activations
as *"gate x activation(up)"*, which reads backwards from what the kernels do.
`SiluAndMul`'s docstring is explicit — `silu(x[:d]) * x[d:]`, i.e. the
activation applies to the **gate** half — and the kernels agree. We had it
right, but the earlier per-layer MoE oracle could not have caught it either way,
because the reference it compared against made the same assumption. An oracle
only tests what it does not share with the thing under test.

## The Laguna holdout

`Laguna-XS-2.1-exl3` @2.00bpw loads (8.54 GiB) and runs, but emits only
whitespace or dashes. Ruled out by measurement:

- **Not the MoE layer.** Hooking a live `RoutedExperts` and recomputing it from
  dequantized per-expert weights gives `max_abs_err = 0.0000` over routing
  weights and the fused reduction.
- **Not configuration.** 256 experts, silu, `experts_per_tok=8`, no fused shared
  expert, stored intermediate 512, K=2 — all as the checkpoint says.
- **Not prompting.** Its tokenizer *does* add BOS (unlike gemma-4-12B), the chat
  template renders sensibly, and raw-completion and chat prompts degenerate
  alike.
- **Not shared experts.** vLLM applies those itself, before `forward_modular`,
  and returns them separately; ignoring them in `apply()` is correct.

What is left is everything around the MoE block: attention, the dense layer 0,
the quantized untied `lm_head`, or the sigmoid routing. Next step is the
technique that resolved gemma-4-12B — feed the same token ids to vLLM and to
exllamav3 and compare hidden states layer by layer to find the first divergence.
Since the MoE layer is provably exact, the informative outcome is *which* layer
type diverges.
