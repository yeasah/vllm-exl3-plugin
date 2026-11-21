# vLLM patches: what is vital, what is optimization

*Policy adopted at the v0.34 bump (2026-11-21).*

The plugin runs on our vLLM fork (`deps/vllm`, branch `appliance/v<version>`), and that stays the recommended setup:
the fork is the version the plugin has been tested against, and it carries the optimizations below. But the plugin
should also run on a **stock vLLM of a compatible version**, for anyone who does not want to trust or use the fork.
So the patches split into two kinds, by what happens without them.

**Vital: the plugin cannot load or serve correctly without it.** That includes model-specific fixes, and silently
wrong output counts the same as a failed load. Vital patches live in the plugin, as monkeypatches
(`vllm_exl3_plugin/stock_compat.py`, and the graph default in `plugin.py`), never in the fork. Each one is scoped to
EXL3 -- an EXL3 parameter, or a model built under an EXL3 config -- so stock vLLM's behaviour is untouched for
everything else, and each is harmless where vLLM already does the right thing. That makes the patch independent of
whether the fork, stock vLLM, or a future upstream with the fix is underneath. The generic forms these replace are in
the git history of the fork's branches, if vLLM ever becomes a place to send them.

**Optimization: everything else.** VRAM work above all, non-essential fixes, behaviour improvements. These live in
the fork. A stock vLLM serves EXL3 without them, worse.

## Vital, as of v0.34

| patch | what fails without it | where |
|---|---|---|
| an embedding built without `quant_config` takes the EXL3 config it is built under | a quantized embedding is unreachable on most model files (they never pass it): a load error for blockq, a silently dense embedding beside a quantized head on a tied model | `stock_compat._embedding_quant_config_default` (fork `be1219d32`) |
| a fused QKV / gate_up tensor reaches an EXL3 parameter whole, with its shard id | the merged and QKV loaders hand a fused tensor whole only to vLLM's own two parameter types; Qwen3.5 EXL3 checkpoints do not load | `stock_compat._fused_shard_loading` (fork `00896553c`) |
| an EXL3 parameter in a `ReplicatedLinear` loads whole through its own row loader | `ReplicatedLinear` has no v2 loader; quantized models through the Transformers backend do not load | `stock_compat._replicated_linear_loading` (fork `bacb74d98`) |
| full decode graphs by default for EXL3 models with GDN layers that vLLM captures breakable | breakable capture fails for EXL3 + GDN (docs/upstream.md); v0.34 makes Qwen3.5 and family breakable by default | `plugin._patch_gdn_graph_default` (was Qwen4Exp only) |

Retired at v0.34: the Muse Glimmer vendored vision config (`1f3b03290`; upstream deleted the config, #60823). Pending:
the Transformers backend's logit multiplier (`40fb01056`), Muse Glimmer's alone -- vital while Muse is served through
the backend (every logprob and temperature-sampled choice is off by the multiplier), moot if the plugin serves the
native `muse_glimmer.py` v0.34 added, which applies `output_multiplier` before the soft cap itself.

## Optimization, carried on `appliance/v0.34.0`

- TurboQuant: `boundary:N` in `--kv-cache-dtype-skip-layers`; dtype conversion inside the continuation-prefill copy;
  slabbed continuation prefill (its invariant test, the slab dequant, the slabbing).
- The workspace floor (a discarded profiling phase no longer sets it; V2 runner only now).
- Vision towers: chunked MLP over tokens; no q/k/v copy in Qwen2.5-VL's attention preamble (GLM-4.1V's is upstream's
  own rework now).
- Checkpoint download progress in `weight_utils`.

Re-test before carrying (each was a fix for a v0.32 failure; v0.34 rewrote what it patched):

- TurboQuant with sliding-window layers (`045b18123`): the primary page priced through the first layer's backend.
  v0.34 still prices it through `backend_cls` but added its own packing logic beside it, and moved `unify` to
  `kv_cache_planning.py`.
- Backend workspace declarations (`82853c68e`, `abab1caed`): v0.34 added `reserve_workspace()` and TurboQuant reserves
  during profiling; and the extensible KV cache, on by default, sizes after warmup, when builders' workspace exists.
  Likely moot by default, perhaps not with extensible KV off.

## Checking it

A stock vLLM of the fork's version, in its own venv: the plugin's tests and the bench gate's fast tier. Tests that use
a fork-only optimization (TurboQuant's slabbing, `boundary:N`) skip on stock.
