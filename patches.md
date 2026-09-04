# vLLM patches

The plugin needs a patched vLLM. Those patches used to live here as `.patch`
files applied by hand to a checkout somewhere outside the project; they now live
as commits on [`appliance/v0.28.0`](https://github.com/yeasah/vllm/tree/appliance/v0.28.0)
in our fork, vendored as the `deps/vllm` submodule. This file is the index: what
each commit does and why, so the set can be read without checking out the fork.

The branch is based on the **v0.28.0** tag, which is the pin the plugin, the
`bench/` baselines and every serving measurement in `docs/` are built against.
It is not based on upstream `main` and is not rebased continuously — see
*Offering these upstream* below.


## Installing it

Do **not** let it build the CUDA extensions. Every commit on this branch is pure
Python -- none touches `csrc/`, `cmake/` or `setup.py` -- so the binaries from
the released v0.28.0 wheel are correct, and `VLLM_USE_PRECOMPILED=1` fetches
them instead of spending half an hour compiling:

    git submodule update --init deps/vllm
    VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_LOCATION=https://files.pythonhosted.org/packages/87/d7/97f6ecc2ae883e601e08d7cef87cb54ececeefcfe6b5e12d5d92f8d06d6b/vllm-0.28.0-cp38-abi3-manylinux_2_28_x86_64.whl \
    pip install --no-deps --no-build-isolation -e deps/vllm

The wheel location has to be given explicitly. Left to itself the precompiled
path resolves a wheel *by commit* -- from `VLLM_PRECOMPILED_WHEEL_COMMIT`, else
by inferring a base commit in `main`. Neither works here: our HEAD has never
been seen by `wheels.vllm.ai`, and the v0.28.0 tag is cut on a release branch
rather than on `main`, so there is no nightly wheel for it either. The released
PyPI wheel is the stable source. Confirm afterwards that the version string
carries our commit and `.precompiled`:

    vllm-0.1.dev20058+g1f1617e26.precompiled

Then fetch the tags into the submodule once. A submodule is cloned without
them, so `git describe` in `deps/vllm` returns a bare hash and `bench/`
provenance reads `src.vllm.describe: 1f1617e26` instead of naming the base:

    git -C deps/vllm fetch --tags origin

    v0.28.0-7-g1f1617e260    # base tag and patch count, both legible

**If you do need a source build** (a change under `csrc/`, or a mismatched
torch), cap the job count:

    MAX_JOBS=8 pip install --no-deps --no-build-isolation -e deps/vllm

The default comes from the CPU count, which on consumer hardware counts
hyperthreads and e-cores and so overshoots *RAM* rather than cores -- this box
declares 24 threads against 23 GiB, and nvcc wants about a gigabyte per
translation unit, so the default `-j 24` OOMs partway through.

## The commits

Newest last; the branch applies them in this order on top of `v0.28.0`.

| commit | what it fixes |
|---|---|
| [`c69d01ed8`](https://github.com/yeasah/vllm/commit/c69d01ed8) | **`VocabParallelEmbedding` never receives a `quant_config`.** 86 of 131 model files omit it, so no quantized embedding can be served on those architectures — silently dense for a tied model, a load failure for a block-quantized one. Defaults it from `get_current_vllm_config()` in one place rather than touching 86 call sites. |
| [`8694dbfff`](https://github.com/yeasah/vllm/commit/8694dbfff) | **A parameter cannot declare that it splits fused checkpoint shards itself.** Adds a `handles_fused_shards` capability, checked before the generic fused-shard path. Qwen3.5 checkpoints do not load without it. |
| [`d22146319`](https://github.com/yeasah/vllm/commit/d22146319) | **`ReplicatedLinear` has no `weight_loader_v2` branch** — the only `LinearBase` subclass without one, which any quantized model reaching it through the Transformers backend needs. |
| [`344802e1b`](https://github.com/yeasah/vllm/commit/344802e1b) | **The Transformers backend reads only `logit_scale`**, never a model's own spelling (MuseGlimmer's `output_multiplier`), and applies the scale *after* the soft cap where such a model needs it before. Folds the multiplier into the cap via an identity that reduces to today's behaviour at 1. |
| [`86e3b1eef`](https://github.com/yeasah/vllm/commit/86e3b1eef) | **A quantized KV cache could not coexist with sliding-window layers.** The quantized primary was priced through the *first* attention layer's backend, which with skip layers is usually a native one, so the page-size alignment arithmetic could not be satisfied. Policy-free: no default moves. |
| [`ce1685699`](https://github.com/yeasah/vllm/commit/ce1685699) | **`boundary:N` is unreachable.** TurboQuant already computes a boundary of `n` native layers at each end of the stack, but `n` cannot be set, so the configurations on the memory/quality frontier cannot be expressed. Exposes it as a keyword in `--kv-cache-dtype-skip-layers`, which already carries a keyword vocabulary. Also fixes the parser rejecting non-integer entries. |
| [`1f1617e26`](https://github.com/yeasah/vllm/commit/1f1617e26) | **`_continuation_prefill` materializes a full-context temporary.** `k_full[:n] = k.to(qdtype)` converts out-of-place where `copy_` would convert inside the copy. Measured at the real shapes: 230.0 MiB → 0.0, bit-identical. One of the four buffers that made up 914 MiB of a 930 MiB prefill peak. |

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
