# vLLM patches

The plugin needs a patched vLLM. Those patches used to live here as `.patch`
files applied by hand to a checkout somewhere outside the project; they now live
as commits on [`appliance/v0.29.0`](https://github.com/yeasah/vllm/tree/appliance/v0.29.0)
in our fork, vendored as the `deps/vllm` submodule. This file is the index: what
each commit does and why, so the set can be read without checking out the fork.

The branch is based on the **v0.29.0** tag, which is the pin the plugin, the
`bench/` baselines and every serving measurement in `docs/` are built against.
It is not based on upstream `main` and is not rebased continuously — see
*Offering these upstream* below.


## Installing it

Do **not** let it build the CUDA extensions. Every commit on this branch is pure
Python -- none touches `csrc/`, `cmake/` or `setup.py` -- so the binaries from
the released v0.29.0 wheel are correct, and `VLLM_USE_PRECOMPILED=1` fetches
them instead of spending half an hour compiling:

    git submodule update --init deps/vllm
    git -C deps/vllm fetch --tags origin        # BEFORE installing -- see below
    VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_LOCATION=https://files.pythonhosted.org/packages/ca/09/7f79450e21bd1c2a0897ab946a544816a4f7e04c54f0ca49849b14b12d6a/vllm-0.29.0-cp38-abi3-manylinux_2_28_x86_64.whl \
    pip install --no-deps --no-build-isolation -e deps/vllm

The wheel location has to be given explicitly. Left to itself the precompiled
path resolves a wheel *by commit* -- from `VLLM_PRECOMPILED_WHEEL_COMMIT`, else
by inferring a base commit in `main`. Neither works here: our HEAD has never
been seen by `wheels.vllm.ai`, and the v0.29.0 tag is cut on a release branch
rather than on `main`, so there is no nightly wheel for it either. The released
PyPI wheel is the stable source. Confirm afterwards that the version string
names the base tag, the patch count, our commit and `.precompiled`:

    vllm-0.29.1.dev7+g1a736c073.precompiled

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
above: `0.29.1` is scm's guess at the *next* release, `dev7` is our seven
commits, `g1a736c073` is which ones.

The same fetch is what makes `git describe` legible in the submodule, so
`bench/` provenance names the base instead of reading
`src.vllm.describe: 1a736c073`:

    v0.29.0-7-g1a736c073e    # base tag and patch count, both legible

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

Newest last; the branch applies them in this order on top of `v0.29.0`.

| commit | what it fixes |
|---|---|
| [`fe1cdc942`](https://github.com/yeasah/vllm/commit/fe1cdc942) | **`VocabParallelEmbedding` never receives a `quant_config`.** 86 of 131 model files omit it, so no quantized embedding can be served on those architectures — silently dense for a tied model, a load failure for a block-quantized one. Defaults it from `get_current_vllm_config()` in one place rather than touching 86 call sites. |
| [`31b57b066`](https://github.com/yeasah/vllm/commit/31b57b066) | **A parameter cannot declare that it splits fused checkpoint shards itself.** Adds a `handles_fused_shards` capability, checked before the generic fused-shard path. Qwen3.5 checkpoints do not load without it. |
| [`587400bd0`](https://github.com/yeasah/vllm/commit/587400bd0) | **`ReplicatedLinear` has no `weight_loader_v2` branch** — the only `LinearBase` subclass without one, which any quantized model reaching it through the Transformers backend needs. |
| [`93b36225a`](https://github.com/yeasah/vllm/commit/93b36225a) | **The Transformers backend reads only `logit_scale`**, never a model's own spelling (MuseGlimmer's `output_multiplier`), and applies the scale *after* the soft cap where such a model needs it before. Folds the multiplier into the cap via an identity that reduces to today's behaviour at 1. |
| [`81566217e`](https://github.com/yeasah/vllm/commit/81566217e) | **A quantized KV cache could not coexist with sliding-window layers.** The quantized primary was priced through the *first* attention layer's backend, which with skip layers is usually a native one, so the page-size alignment arithmetic could not be satisfied. Policy-free: no default moves. |
| [`2778e4a20`](https://github.com/yeasah/vllm/commit/2778e4a20) | **`boundary:N` is unreachable.** TurboQuant already computes a boundary of `n` native layers at each end of the stack, but `n` cannot be set, so the configurations on the memory/quality frontier cannot be expressed. Exposes it as a keyword in `--kv-cache-dtype-skip-layers`, which already carries a keyword vocabulary. Also fixes the parser rejecting non-integer entries. |
| [`1a736c073`](https://github.com/yeasah/vllm/commit/1a736c073) | **`_continuation_prefill` materializes a full-context temporary.** `k_full[:n] = k.to(qdtype)` converts out-of-place where `copy_` would convert inside the copy. Measured at the real shapes: 230.0 MiB → 0.0, bit-identical. One of the four buffers that made up 914 MiB of a 930 MiB prefill peak. |
| [`881c7345b`](https://github.com/yeasah/vllm/commit/881c7345b) | **A discarded profiling phase sets the workspace floor for the whole process.** The CUDA-graph memory profiler builds a full set of attention metadata builders, measures capture, and throws them away — but `_ensure_workspace_size` only grows, and the phase runs before auto-fit revises `max_model_len` down. TurboQuant's reserve is therefore taken at the *declared* context: 1024 MB where the fitted length needs 588. Brackets the phase and shrinks back to the sizes held on entry, so `profile_run`'s mark — the one `lock_workspace` expects to survive — is kept. Measured 440 MiB on a 16 GiB card, token-for-token identical. |
| [`6f4a1654b`](https://github.com/yeasah/vllm/commit/6f4a1654b) | **Tests only: the invariant that lets continuation prefill be chunked.** Every query attends to all of the cached prefix, so the prefix is unmasked and can be cut anywhere while only the current chunk is causal. That depends on flash-attn aligning a shorter query to the *end* of the key sequence, which nothing in the tree states, so it is checked against fp32 SDPA under the explicit mask. |
| [`0c54f54b2`](https://github.com/yeasah/vllm/commit/0c54f54b2) | **`_tq_full_dequant_kv` can only start at the beginning of the context.** Everything derives from `pos = program_id(0)`, so the destination has to be as large as the context. Adds `POS_OFFSET`, applied to the cache index and not the output one, plus the config knob the chunked path reads. Defaults leave both call sites unchanged. |
| [`740dd8b6a`](https://github.com/yeasah/vllm/commit/740dd8b6a) | **Continuation prefill's VRAM scales with context instead of with the chunk.** Five allocations sized by `cached_len` or `max_model_len`: 588 MiB standing plus 774 MiB of growth per 100K prefill, against a 2.65 GiB KV cache. Slabs the cached prefix and merges by log-sum-exp, so the buffers are the slab and the partials are the chunk. 1580 MiB back at the peak for ~1% of prefill throughput; greedy output identical at 100K. Both paths stay — the monolithic one is all that works without flash-attention, and `tq_prefill_workspace_mib=0` selects it, which is what makes the two comparable inside one build. |
| [`6ac849972`](https://github.com/yeasah/vllm/commit/6ac849972) | **A backend's workspace reservation is invisible to the KV budget.** The profiling window closes before any metadata builder exists, so what a builder reserves is spent from whatever `gpu_memory_utilization` left unclaimed — which is why the knob could not be set to the card's maximum: at 0.98 TurboQuant sized a KV cache that fit, then OOMed taking 96 MiB behind the budget's back. Adds `AttentionBackend.get_reserved_workspace_bytes`, default zero, subtracted before auto-fit. TurboQuant prices the same reservation sets the builder hands to the workspace manager, so the declaration cannot drift from the allocation. Declares 96.00 MiB where `VLLM_DEBUG_WORKSPACE` shows 96.00 MB taken; 0.98 goes from OOM-at-startup to full 262144 context and an 80,793-token prompt served. |

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

- `vllm-gemma4-transformers-5.15-per-layer.patch` — upstream landed a generic
  equivalent; retired at v0.28.0. See [README.md](README.md).
- `exllamav3-sm90-barrier.patch` — folded into our exllamav3 fork's history when
  we started tracking it. See [docs/exllamav3-arch.md](docs/exllamav3-arch.md).
- *chat-template revision* — never a `.patch`, only ever an uncommitted edit in
  a local v0.27.0 checkout, which is why it is recorded here now that it is
  gone. `_try_get_processor_chat_template` did not pass a revision, so the
  processor lookup fell back to `main`: wrong for any repo served off a
  non-default branch, and it leaves a ref to an unfetched commit in the hub
  cache. v0.28.0 fixes it and goes further, threading both `revision` and
  `code_revision` and keying the cache on them. Retired at v0.28.0.
