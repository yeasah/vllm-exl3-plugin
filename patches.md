# vLLM patches

The plugin needs a patched vLLM. Those patches used to live here as `.patch`
files applied by hand to a checkout somewhere outside the project; they now live
as commits on [`appliance/v0.34.0`](https://github.com/yeasah/vllm/tree/appliance/v0.34.0)
in our fork, vendored as the `deps/vllm` submodule. This file is the index: what
each commit does and why, so the set can be read without checking out the fork.

Since the 0.34 bump the fork carries **optimizations only**: what the plugin needs in order to load and
serve correctly lives in the plugin as monkeypatches, so a stock vLLM of the same version serves EXL3 too
([docs/vllm-patches.md](docs/vllm-patches.md) has the policy and the list).

The branch is based on the **v0.34.0** tag, which is the pin the plugin, the
`bench/` baselines and every serving measurement in `docs/` are built against.
It is not based on upstream `main` and is not rebased continuously — see
*Offering these upstream* below.


## Installing it

Do **not** let it build the CUDA extensions. Every commit on this branch is pure
Python -- none touches `csrc/`, `cmake/` or `setup.py` -- so the binaries from
the released v0.34.0 wheel are correct, and `VLLM_USE_PRECOMPILED=1` fetches
them instead of spending half an hour compiling:

    git submodule update --init deps/vllm
    git -C deps/vllm fetch --no-tags https://github.com/vllm-project/vllm.git tag v0.34.0   # BEFORE installing -- see below
    VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_LOCATION=https://files.pythonhosted.org/packages/0b/2c/55f3b76ceadbfc6d515f16b1c17e2e69136c8e54d47be2a94b9f7326f9bb/vllm-0.34.0-cp38-abi3-manylinux_2_28_x86_64.whl \
    pip install --no-deps --no-build-isolation -e deps/vllm

From v0.32 the build imports `setuptools_rust` even when precompiled (the Rust frontend and extensions also come from the
wheel, so no toolchain is needed), and `--no-build-isolation` means the build requirements must already be installed:

    pip install 'cmake>=3.26.1' ninja 'packaging>=24.2' 'setuptools>=77.0.3,<81.0.0' 'setuptools-scm>=8.0' 'setuptools-rust>=1.9.0' wheel jinja2

A venv carried over from v0.29 also holds `flashinfer-cubin` 0.6.18, which FlashInfer 0.7 refuses to import beside; uninstall it.

The wheel location has to be given explicitly. Left to itself the precompiled
path resolves a wheel *by commit* -- from `VLLM_PRECOMPILED_WHEEL_COMMIT`, else
by inferring a base commit in `main`. Neither works here: our HEAD has never
been seen by `wheels.vllm.ai`, and the v0.34.0 tag is cut on a release branch
rather than on `main`, so there is no nightly wheel for it either. The released
PyPI wheel is the stable source. Confirm afterwards that the version string
names the base tag, the patch count, our commit and `.precompiled`:

    vllm-0.34.1.dev12+g68c13ab5f

**The tag fetch has to happen before the install, and this file used to have it
after.** A submodule is cloned with no tags at all (the refspec is
`+refs/heads/*` only), and vLLM versions itself with `setuptools-scm`, which
runs `git describe` *at install time* and writes the answer into
`vllm/_version.py` and the dist-info name, where it is then frozen. With no tag
reachable, scm falls back to `0.1.dev<commits-since-root>`, and the install
claims **`0.1.dev20058+g1a736c073.precompiled`** — which is what this file used
to tell you to expect. Nothing gates on it, but `vllm.__version__` is recorded
in every `bench/` env block and in any bug report, and `0.1` names neither the
base nor the patches. Fetching first costs nothing and produces the string
above: `0.34.1` is scm's guess at the *next* release, `dev12` is our twelve
commits, `g68c13ab5f` is which ones.

The same fetch is what makes `git describe` legible in the submodule, so
`bench/` provenance names the base instead of reading
`src.vllm.describe: 1a736c073`:

    v0.34.0-12-g68c13ab5fe    # base tag and patch count, both legible

For a build where fetching tags is not wanted, `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM`
overrides the whole computation — but it goes stale on the next commit, so it is
an escape hatch and not the arrangement to standardize on.

**If you do need a source build** (a change under `csrc/`, or a mismatched
torch), cap the job count:

    MAX_JOBS=8 pip install --no-deps --no-build-isolation -e deps/vllm

The default comes from the CPU count, which on consumer hardware counts
hyperthreads and e-cores and so overshoots *RAM* rather than cores -- this box
declares 24 threads against 23 GiB, and nvcc wants about a gigabyte per
translation unit, so the default `-j 24` OOMs partway through.

## The commits

Newest last; the branch applies them in this order on top of `v0.34.0`.

| commit | what it fixes |
|---|---|
| [`22cb6f413`](https://github.com/yeasah/vllm/commit/22cb6f413) | **`boundary:N` is unreachable.** TurboQuant already computes a boundary of `n` native layers at each end of the stack, but `n` cannot be set, so the configurations on the memory/quality frontier cannot be expressed. Exposes it as a keyword in `--kv-cache-dtype-skip-layers`, which already carries a keyword vocabulary. Also fixes the parser rejecting non-integer entries. |
| [`12968cfa5`](https://github.com/yeasah/vllm/commit/12968cfa5) | **`_continuation_prefill` materializes a full-context temporary.** `k_full[:n] = k.to(qdtype)` converts out-of-place where `copy_` would convert inside the copy. Measured at the real shapes: 230.0 MiB → 0.0, bit-identical. One of the four buffers that made up 914 MiB of a 930 MiB prefill peak. |
| [`83dfe5613`](https://github.com/yeasah/vllm/commit/83dfe5613) | **Tests only: the invariant that lets continuation prefill be chunked.** Every query attends to all of the cached prefix, so the prefix is unmasked and can be cut anywhere while only the current chunk is causal. That depends on flash-attn aligning a shorter query to the *end* of the key sequence, which nothing in the tree states, so it is checked against fp32 SDPA under the explicit mask. |
| [`f04a0ce6d`](https://github.com/yeasah/vllm/commit/f04a0ce6d) | **`_tq_full_dequant_kv` can only start at the beginning of the context.** Everything derives from `pos = program_id(0)`, so the destination has to be as large as the context. Adds `POS_OFFSET`, applied to the cache index and not the output one, plus the config knob the chunked path reads. Defaults leave both call sites unchanged. |
| [`fcdead389`](https://github.com/yeasah/vllm/commit/fcdead389) | **Continuation prefill's VRAM scales with context instead of with the chunk.** Five allocations sized by `cached_len` or `max_model_len`: 588 MiB standing plus 774 MiB of growth per 100K prefill, against a 2.65 GiB KV cache. Slabs the cached prefix and merges by log-sum-exp, so the buffers are the slab and the partials are the chunk. 1580 MiB back at the peak for ~1% of prefill throughput; greedy output identical at 100K. Both paths stay — the monolithic one is all that works without flash-attention, and `tq_prefill_workspace_mib=0` selects it, which is what makes the two comparable inside one build. |
| [`76f501baf`](https://github.com/yeasah/vllm/commit/76f501baf) | **A checkpoint download says nothing until it finishes.** Logs when one starts, how it is going, and when a process waits on another's download of the same files. Behaviour only. |
| [`870312214`](https://github.com/yeasah/vllm/commit/870312214) | **The vision MLP's two full-width buffers become the peak once attention stops hiding them** — 538 MiB each on Qwen3.8 at full resolution. Chunks the pointwise MLP over tokens against a byte budget (`VLLM_MM_ENCODER_MLP_CHUNK_MB`, default 256), so one setting holds across towers whose intermediate widths differ 3x. Qwen3.8 1.653 → 1.165 GiB (−32.6% overall), GLM-4.1V 1.854 → 1.438 (−22.5%); on a real GLM serve, KV 5.89 → 6.31 GiB and context 154,480 → 165,456 tokens. **Not bit-exact in general**: cuBLAS picks its kernel from M, so a smaller chunk can reduce over K in a different order — exact at the default budget on both towers, ~1 bf16 ULP out at small ones. |
| [`cb61d4381`](https://github.com/yeasah/vllm/commit/cb61d4381) | **The vision attention preamble copies q/k/v it already has.** A ViT's peak is the attention preamble, not the MLP: `einops` yields a strided view when the rearrange moves the q/k selector to dim 0, and the code forces it contiguous for a triton rotary kernel that takes `x.stride(...)` explicitly and never needed it. That kernel also supports `inplace`, which `ApplyRotaryEmb.forward_cuda` never passes, so the rotary allocates a second full-size output. Both are exactly bit-identical — they move lifetimes, not GEMM shapes. Qwen3.8 1.728 → 1.653 GiB; GLM-4.1V barely moves, because its `torch.cat([q, k])` re-makes the copy, but it unmasks the MLP below. Qwen2.5-VL half only at v0.34: upstream reworked GLM-4.1V's preamble itself (q, k contiguous, v kept a view for FlashInfer's `cu_seqlens`). |
| [`ce8e60c8a`](https://github.com/yeasah/vllm/commit/ce8e60c8a) | **A discarded profiling phase sets the workspace floor for the whole process.** The CUDA-graph memory profiler builds a full set of attention metadata builders, measures capture, and throws them away — but `_ensure_workspace_size` only grows, and the phase runs before auto-fit revises `max_model_len` down. TurboQuant's reserve is therefore taken at the *declared* context: 1024 MB where the fitted length needs 588. Brackets the phase and shrinks back to the sizes held on entry, so `profile_run`'s mark — the one `lock_workspace` expects to survive — is kept. Measured 440 MiB on a 16 GiB card, token-for-token identical. V2 runner half only at v0.34: the V1 runner is gone (#63373). |
| [`4d3b2c781`](https://github.com/yeasah/vllm/commit/4d3b2c781) | **A padded KV spec's pad goes stale when `unify` rescales its block.** What is left of `045b18123`: v0.34 fixed that commit's load-bearing bug (the primary page priced through the first layer's backend) and pads a non-divisible attention page itself, but a padded spec whose page divides the maximum is still rescaled by block size alone, so TurboQuant with a full-attention skip layer (gemma-3-1b, layer 5 held native) asserts. Scales the pad with the block. |
| [`781043181`](https://github.com/yeasah/vllm/commit/781043181) | **The extensible KV cache's margin floor is a flat 1 GiB** -- 6% of a 16 GiB card, 12% of an 8 GiB one, and measured unused on one (a 260K-token prompt grew torch 0.16 GiB, inside the measured reserve). On CUDA the floor is now 2% of device memory between 256 MiB and the old 1 GiB (datacenter parts unchanged), the non-torch share half of it; `VLLM_KV_CACHE_MARGIN_FLOOR_MB` overrides. Qwen3.8-27B TurboQuant, 512-token chunks: KV 3.89 → 4.56 GiB. |
| [`68c13ab5f`](https://github.com/yeasah/vllm/commit/68c13ab5f) | **Prompt logits are computed 1024 rows at a time whatever the vocabulary** -- 485 MiB of bf16 logits at Qwen3.5's 248k -- and the step that sizes an extensible KV cache requests prompt logprobs whenever they are allowed, so that chunk fed every user's transient reserve. Chunks to 64 MiB of logits, the row size learned once from a discarded one-row probe so a prompt that fits one chunk is computed in one call as before. At `vllm serve`'s 2048-token batches: reserve 0.96 → 0.48 GiB, KV +17%, the same as disabling logprobs; a 2000-token prompt-logprob request 2.90 → 2.96 s. |

## Offering these upstream

Each commit is self-contained and touches only its own concern, so producing a
PR is `git cherry-pick <sha>` onto a branch off whatever `main` is at the time —
which is the only base that would be valid anyway. We deliberately do **not**
keep parallel PR branches rebased onto a moving `main`: that is continuous work
against an event that, on the evidence in [docs/upstream.md](docs/upstream.md),
has not been arriving. That note tracks which of these are worth offering, in
what shape, and what to check first.

## Other branches in the fork

The fork also holds work that is not part of the appliance stack. None of these
is pinned by the submodule; check one out in a scratch clone to run it.

- **`tq-sliding-window`** — referenced by an open upstream PR. Kept because that
  reference has to stay valid, not because we depend on it.
- **`experiment/kvarn`** — the KVarN port: PR 46812 carried onto the current
  backend contract, the `layer_name` propagation from `Attention` to the impl
  that made it actually produce correct output, a decode-path bisect knob, and
  the appliance patches needed to load an EXL3 checkpoint at all. Based on
  upstream `main` (`v0.28.1rc0-235`), *not* on v0.28.0: the port was written
  against the post-`TQ*Spec` contract, and v0.28.0 is cut on a release branch
  whose merge-base with `main` is far older (`v0.26.1rc0-844`), so there is no
  cheap rebase — and rebasing would invalidate the measurements anyway.
  Shelved; [docs/kvarn.md](docs/kvarn.md) has the verdict.
- **`reference/kvarn-pr-46812`** — upstream PR 46812's own diff rebased onto
  v0.28.0, original authorship intact. Not our code. Kept because the PR is
  decaying upstream and the rebase was the expensive part.

## Retired

Kept as history because the reasons are still instructive; both are described
where they are referenced.

- At the 0.34 bump, under the vital/optimization split: `be1219d32` (embedding `quant_config` default),
  `00896553c` (fused-shard loading), `bacb74d98` (`ReplicatedLinear` v2 loading) and `40fb01056` (the
  Transformers backend's logit multiplier) **moved into the plugin** as EXL3-scoped monkeypatches
  (`vllm_exl3_plugin/stock_compat.py`); with the old commits re-applied, Qwen3.5-35B-A3B at TP=2 captured
  bit-identical, so the move is faithful. `1f3b03290` (Muse Glimmer's vendored config) retired: upstream deleted
  the config (#60823). `045b18123` reduced to `4d3b2c781`. `82853c68e` and `abab1caed` (backend workspace
  declarations) not carried: the extensible KV cache, on by default, measures what they declared; they would
  matter only with it off (docs/vllm-patches.md).

- `vllm-gemma4-transformers-5.15-per-layer.patch` — upstream landed generic per-layer arch
  config; retired at v0.28.0. `Gemma4Config.verify_and_update_config` now reads
  `model_config.model_arch_config` and indexes `arch_config[i].head_size`.
- `vllm-transformers-backend-embedding-postprocess.patch` — retired at v0.28.0 by a *better*
  mechanism than ours. Upstream no longer substitutes the input embedding; it **rebases its
  class** (`type(cls.__name__, (cls, _VocabParallelEmbeddingBase), {})`), so a model's own
  `forward` (MuseGlimmer's `embed_norm`) survives by construction rather than by our detection
  heuristic. The quantized-embedding seam survives too: `replace_embedding_class` passes
  `quant_config` into `VocabParallelEmbedding.__init__`, so our methods still attach. Both 0.28
  retirements were confirmed by the gate: the Muse-Glimmer entry exercising both paths captured
  at `0.000e+00` against its pre-bump baseline.
- At the 0.29 bump, **nothing** retired: all seven commits then carried were still needed, and
  two needed conflict resolution (see *What the 0.29 bump actually cost* in
  [docs/upstream.md](docs/upstream.md)).
- `exllamav3-sm90-barrier.patch` — folded into our exllamav3 fork's history when
  we started tracking it. See [docs/exllamav3-arch.md](docs/exllamav3-arch.md).
- *chat-template revision* — never a `.patch`, only ever an uncommitted edit in
  a local v0.27.0 checkout, which is why it is recorded here now that it is
  gone. `_try_get_processor_chat_template` did not pass a revision, so the
  processor lookup fell back to `main`: wrong for any repo served off a
  non-default branch, and it leaves a ref to an unfetched commit in the hub
  cache. v0.28.0 fixes it and goes further, threading both `revision` and
  `code_revision` and keying the cache on them. Retired at v0.28.0.
