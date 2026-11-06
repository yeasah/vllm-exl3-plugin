---
name: exl3-fork-transition-and-mgemm-break
description: "vllm-exl3-plugin now tracks a fork of exllamav3 (yeasah/exllamav3); the upstream bump that came with forking broke exl3_mgemm's call sites, now fixed and revalidated"
metadata: 
  node_type: memory
  type: project
  originSessionId: d041f792-5c0a-4189-a19a-b1e0850e1167
  modified: 2026-08-09T01:04:04.849Z
---

`deps/exllamav3` now tracks `https://github.com/yeasah/exllamav3` instead of
upstream `turboderp-org/exllamav3` (forked 2026-08-08, so the two carried fixes
— sm_90+ MGEMM barrier, [[exl3-embed-head-tax-vs-gguf]]'s embedding
quantization work when it happens — can live as real commits instead of
`patches/*.patch` files applied at build time). The repoint's base commit also
pulled in ~60 unrelated upstream commits (DeepseekV4/DSA additions, autotune
changes, "MGEMM: Support per-matrix-N dim") between the old and new pins.

That last one broke the plugin immediately: `exl3_mgemm` gained two new
trailing optional params (`size_n_list`, `c_ptrs`), and since its pybind
binding declares no `py::arg` defaults, Python must pass all 18 positions
where 16 used to work — every MoE forward call raised
`TypeError: exl3_mgemm(): incompatible function arguments`. Fixed in
`vllm_exl3_plugin/ops.py` (commit 4dee8a2) by passing `None` for both at all
three call sites (`exl3_gemm.cu` only consults them when `size_n_list` is
given, so `None` reproduces the old behavior exactly). `exl3_gemm` (the dense,
non-MoE path) was untouched by the upstream bump.

**Why:** confirms the "back to zero validation" risk flagged when the bump was
first noticed was real and immediate, not theoretical — a vendored dependency
bump silently broke a call site with no code change on our side. Revalidated
properly, not just by arg-count arithmetic: a live generation through the real
kernel (Laguna-XS-2.1, 256 experts) produced a correct answer, and separately
the user ran real `vllm serve` chat completions across all six reference
models (gemma-4-12B, gemma-4-31B, gemma-4-26B-A4B MoE, qwen3.6-27B,
qwen3.5-35B-A3B MoE, laguna-XS-2.1 MoE), CUDA graphs included. Only known gap:
qwen3.5-35B asserts under CUDA graphs (`Shape: 2 out of considered ranges:
[(1, 1)]`) — confirmed pre-existing, unrelated to this bump or fix.

**How to apply:** after any future exllamav3 fork/upstream bump, don't trust
"the test suite passes" alone without checking *which* tests ran. When this note
was written no unit test touched `exl3_mgemm`/`_exl3_moe_mm`/`RoutedExperts` at
all, so a break in exactly this path sailed through green.

**That gap has since been closed** (verified 2026-08-15): `tests/test_tp.py`
(`TestMoEShardedMathMatches`) calls `ops.exl3_moe_mm` against a real
Laguna-XS-2.1 checkpoint. But it is `@requires_gpu`, so it *skips silently* on a
CPU-only box — "OK (skipped=N)" on a machine without CUDA still tells you nothing
about this path. Confirm the MoE tests actually ran, and for a pin bump also
repeat the manual real-model validation matrix (this note plus
[[verify-across-execution-modes]]) rather than relying on unit coverage alone.
