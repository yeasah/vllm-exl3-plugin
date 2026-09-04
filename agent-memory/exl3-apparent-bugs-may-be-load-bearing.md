---
name: exl3-apparent-bugs-may-be-load-bearing
description: Before "fixing" something EXL3 gets mathematically wrong, measure whether the wrongness is doing work; its heuristics often earn their keep through side effects nobody wrote down
metadata:
  type: project
---

EXL3 heuristics frequently derive their value from a *side effect* that is nowhere
documented, so a change that makes the maths correct can delete the benefit. Established
2026-08-31: `apply_out_scales` divides the weight by its per-output-channel RMS, leaving
a non-identity output metric that `ldlq()` ignores. Restoring that metric — the
mathematically correct fix, free of gradients or extra passes — costs up to +57% KL,
because dropping it is precisely what makes the rounding minimize *relative* rather than
absolute per-channel error. A sweep over the correction strength puts the optimum exactly
at the shipped behaviour. Full write-up in `docs/yaqa.md`, "The ignored `H_O` is
load-bearing".

**Why:** the converter's defaults encode tuning nobody recorded the rationale for
(`--out_scales` defaults to `always`, which makes the `auto` skew heuristic beside it
dead code). Reading the source tells you what it computes, not what the computation is
buying. Same family as [[exl3-checkpoints-carry-unrecorded-scales]] and
[[check-the-artifact-not-the-prose]], one level up: not an unrecorded convention but an
unrecorded *reason*.

**How to apply:** when an EXL3 behaviour looks like an oversight, budget a measurement
before a patch, and include an arm that removes the surrounding heuristic entirely — that
arm is what reveals whether the "bug" was carrying it. Cheap to do: a one-layer quantize
plus whole-model KL, with a bit-identical duplicate arm as the noise floor
(`tools/yaqa/outscales.py` is the worked example). Also check what the *harness* holds
fixed — `probe.py` had been disabling `apply_out_scales`, so a year of YAQA numbers were
scored against a baseline the converter never ships.
