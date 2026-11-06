---
name: exl3-checkpoints-carry-unrecorded-scales
description: "EXL3 checkpoints can bake in scale factors that only exllamav3's architecture files know about, so config.json alone is not enough to consume one correctly"
metadata: 
  node_type: memory
  type: project
  originSessionId: 94844e7e-fafd-47b5-b2cf-e671747f8da1
  modified: 2026-08-06T15:51:51.607Z
---

An EXL3 checkpoint can carry numerical conventions that are recorded **nowhere
in the checkpoint** — not `config.json`, not `quantization_config.json` — and
live only as literals in exllamav3's per-architecture Python
(`exllamav3/architecture/*.py`). Found 2026-08-06 with Laguna-XS-2.1, where
routed `up_proj` weights are pre-divided by `interm_div = 128.0` and exllamav3
compensates by multiplying `moe_routed_scaling_factor` by the same constant.
`Linear.load_exl3` ignores its own `weight_scale`, so the divisor is baked into
the stored weights while the compensation is not.

**Why:** any non-exllamav3 consumer reading the checkpoint's stated
`moe_routed_scaling_factor` gets a silently wrong model — correct-looking
magnitudes everywhere except a constant factor on one branch. A per-layer
oracle cannot catch it, because the layer is exact *given its inputs*.

**How to apply:** when a new EXL3 model produces degenerate output but its
layers verify exact, suspect a scale convention before suspecting kernels.
Compare magnitudes against a control that the same converter treated
differently (Laguna's shared expert vs its routed experts). Recover such
constants by measuring the weights rather than hardcoding per architecture, and
raise rather than guess — see `format.infer_interm_divisor`.

**Now promoted to the repo** (2026-08-15): `docs/format-and-loading.md`, section
"The checkpoint is not a complete description of itself", states the general rule
and the detection method; `docs/moe.md` has the Laguna worked example. Point
future readers there rather than re-deriving. Related:
[[project-doc-conventions]].
