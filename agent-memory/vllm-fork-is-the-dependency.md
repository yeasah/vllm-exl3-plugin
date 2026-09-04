---
name: vllm-fork-is-the-dependency
description: "The patched vLLM is now the deps/vllm submodule, not a loose checkout plus patches/ -- what that changed, and which vLLM trees on this box are still live"
metadata:
  node_type: memory
  type: project
---

As of 2026-09-04 the plugin's vLLM is the **`deps/vllm` submodule**, pinned to
`appliance/v0.28.0` in `github.com/yeasah/vllm` (v0.28.0 plus 7 commits). The
`patches/` directory is gone; `patches.md` is the index. `deps/exllamav3` was
already a submodule, so both dependencies now live inside the project.

**Why:** before this, the plugin depended on a checkout somewhere outside the
repo with patches applied as unstaged working-tree edits. That drift was real,
not hypothetical -- the `_continuation_prefill` `copy_` fix existed *only* as an
uncommitted edit in a working tree, in no `.patch` file and no git history. The
migration is verified: `bench/run.py check --tier fast` passes 9/9 at
`KL max 0.000e+00`, even though the binaries changed from a local CUDA 13.3
source build to the released wheel's.

**How to apply:**

- Installing needs `VLLM_USE_PRECOMPILED=1` *and* an explicit
  `VLLM_PRECOMPILED_WHEEL_LOCATION`; automatic resolution cannot work because
  our HEAD is unknown to `wheels.vllm.ai` and the v0.28.0 tag is cut on a
  release branch. A source build needs `MAX_JOBS=8` -- the default follows CPU
  count, which on consumer hardware overshoots *RAM*. Both are in `patches.md`.
- `bench/` provenance now reads `src.vllm.diff_sha: None` and
  `dirty_files: 0`. A non-`None` `diff_sha` on a dependency is now an
  **anomaly** -- someone edited a submodule without committing -- not the
  normal state it used to be.
- The blessed manifest still records the pre-migration
  `vllm 0.28.1.dev0+g2cf0a6915.d20260825.cu133`, so `bench/run.py env` reports
  a `vllm` line plus ~34 packages from the lm_eval/TriAttention work. Expected
  and benign; a `check` still passes. Do not chase it, and re-bless only when
  something else warrants it.
- The pre-migration trees were deleted on 2026-09-04, reclaiming 27G
  (`~/.venv-vllm-fork`, `~/.venv-vllm-stock`, `~/git/vllm-v0.28.0`,
  `~/git/vllm-kvarn`) once their branches were confirmed on the fork by
  `ls-remote`. The KVarN port and the rebased upstream PR now live only in the
  fork, as `experiment/kvarn` and `reference/kvarn-pr-46812`. What remains:
  `~/git/vllm-fork` (the fork's working checkout, has both remotes),
  `~/git/vllm` (v0.27.0, superseded -- the bench baselines were re-blessed at
  v0.28.0, so its old claim to being "the pin" is dead), plus
  `~/git/vllm-stock`, `~/git/vllm-triattention`, `~/git/vllm-gguf-plugin`.
  The live venv is `~/.venv`.
- v0.28.0's compiled extensions use stable-ABI names --
  `vllm._C_stable_libtorch`, `vllm._moe_C_stable_libtorch`. `vllm._C` and
  `vllm._moe_C` no longer exist, so a probe for them reports a false breakage.

See [[check-upstream-before-patching-vllm]] and [[bench-suite-purpose]].
