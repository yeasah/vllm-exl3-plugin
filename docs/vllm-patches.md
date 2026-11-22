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
| a Transformers-backend model's `output_multiplier` folded into the soft cap (MuseGlimmer) | every logprob, temperature-sampled choice and, through bf16 rounding near the cap's saturation, greedy output wrong: 28 of 106 argmaxes flipped on the bench's Muse backend entry; turboderp's Muse checkpoints quantize the vision adapter, which the native implementation builds as `nn.Linear`, so the backend is their only path | `stock_compat._transformers_logit_multiplier` (fork `40fb01056`) |
| padding rows' expert id -1 becomes expert 0 at weight 0 at the MoE op's entry (`ops._skip_padding_rows`) | v0.34 routers mark CUDA-graph, DP and dummy padding rows with expert id -1 (`VLLM_MOE_SKIP_PADDING`, on by default): the grouped path's bincount raised (Qwen3.5-35B-A3B at TP=2 could not start), the per-row kernels read the pointer tables out of bounds, and the expert cache counted them as the last expert | `vllm_exl3_plugin/ops.py`, not a patch to vLLM: the op adapts to the new contract |
| refuse to serve when expandable segments are on, torch lacks the #196258 fix and the process holds torch memory on two or more devices; checked after load and after warmup | PyTorch #196258: under expandable segments a freed page is unmapped on every device but only the current one is synchronized, so a kernel queued on another GPU writes into memory that is gone or remapped -- Xid 31, or silent corruption. vLLM's workers are one device each (measured at TP=2), so this checks an invariant; it is vital because nothing else stands between a broken one and corrupt output | `vllm_exl3_plugin/allocator_safety.py`, not a patch: a check in the plugin's load and warmup hooks |

Retired at v0.34: the Muse Glimmer vendored vision config (`1f3b03290`; upstream deleted the config, #60823).

Retired 2026-11-22: full decode graphs by default for EXL3 models with GDN layers (`plugin._patch_gdn_graph_default`).
Breakable capture failed for them because of the plugin's own stream guard, not vLLM (docs/upstream.md, "CUDA graphs
under breakable capture"); with that fixed, vLLM's breakable default stands.

## Optimization, carried on `appliance/v0.34.0`

- TurboQuant: `boundary:N` in `--kv-cache-dtype-skip-layers`; dtype conversion inside the continuation-prefill copy;
  slabbed continuation prefill (its invariant test, the slab dequant, the slabbing).
- The workspace floor (a discarded profiling phase no longer sets it; V2 runner only now).
- Vision towers: chunked MLP over tokens; no q/k/v copy in Qwen2.5-VL's attention preamble (GLM-4.1V's is upstream's
  own rework now).
- Checkpoint download progress in `weight_utils`.

- TurboQuant with a full-attention skip layer: `unify` scales a padded spec's pad with its block (`4d3b2c78`). What is
  left of `045b18123`; v0.34 fixed its load-bearing half (the primary page priced through the first layer's backend)
  itself, and `unify` now pads a non-divisible attention page directly.
- The extensible KV cache's margin floor scales with the device on CUDA (`78104318`): 2% of its memory between 256 MiB
  and upstream's 1 GiB, the non-torch share at most half of it; `VLLM_KV_CACHE_MARGIN_FLOOR_MB` overrides. On a 16 GiB
  card the flat 1 GiB went unused (Qwen3.8-27B 3.00bpw TurboQuant, a 260K-token prompt grew torch 0.16 GiB, inside the
  measured reserve; nothing outside torch grew).
- The extensible KV cache's margin goes on the allocated peak only; torch's reservation growth is kept at 1.1 x its
  measurement plus the non-torch share (`6b2557b4`). Qwen3.6-35B-A3B at 256K on 16 GiB: 3.69 GiB kept became 2.48.
  The measurements are in [memory-accounting.md](memory-accounting.md), "The extensible KV cache's margin".
- Expandable segments on by default where torch carries the PyTorch #196258 fix (the stamp `tools/torch-c10cuda-fix.sh`
  leaves) and there is no tensor parallelism (`f6f5443b`); an explicit setting stands. TP stays off because custom
  all-reduce cannot IPC-export expandable segments ([tensor-parallel.md](tensor-parallel.md)). Qwen3.6 at 256K then keeps 1.51 GiB.

Not carried: backend workspace declarations (`82853c68e`, `abab1caed`). With the extensible KV cache on, the v0.34
default, the workspace is measured; they would matter only with it off, where the 27B TurboQuant config sizes 0.13 GiB
past what 0.32 did and runs out mid-prefill. Revisit if extensible-off ever needs to be a supported path.

## Checking it

A stock vLLM of the fork's version, in its own venv: the plugin's tests and the bench gate's fast tier. Tests that use
a fork-only optimization (TurboQuant's slabbing, `boundary:N`) skip on stock.
