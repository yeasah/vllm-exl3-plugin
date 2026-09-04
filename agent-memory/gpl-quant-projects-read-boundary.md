---
name: gpl-quant-projects-read-boundary
description: "GLQ and other GPL-3.0 QuIP#/QTIP-lineage projects: read their claims, never their source; this plugin is not copyleft"
metadata:
  type: project
---

Recorded 2026-09-01. **GLQ** (`github.com/cnygaard/glq`) is a GPL-3.0 vLLM quantization
plugin from the same QuIP#/QTIP lineage as EXL3: E8-lattice or QTIP-trellis body weights
(`E8RHTLinear`), with an INT8/KIVI KV cache — *not* an E8 KV cache, whatever the Slack
advertising implied.

**The rule:** published claims, READMEs, model cards and benchmark numbers are facts and
are fair game. The implementation is expression. Do not open GLQ's source, do not quote
it, do not paraphrase it into this repo — and say so if asked to. The same applies to any
other copyleft project in this space.

**What the GPL does *not* restrict is running it.** Copyleft binds distribution of
derived work, not execution or measurement, so `pip install glq` in an isolated venv and
benchmarking its 44 published checkpoints against ours is clean, and is the most valuable
legal use of it. The trap is incidental source exposure — a traceback prints source
lines, and pip leaves the source somewhere greppable. Black box, out of the project tree,
don't debug it.

**Why:** ideas and measurements are not copyrightable; source is. This plugin is not
copyleft and a single derived kernel would be an unwinnable argument. The cheap way to
keep the boundary defensible is to not have looked, and that is only cheap *before* the
fact.

**How to apply:** when hunting for reference implementations of E8/lattice KV or trellis
kernels, check the license before the code. Permissive alternatives in the same space
exist (NexusQuant, Apache-2.0 but withdrawn by its author; Higman-sims-quant, MIT). The
landscape entry with provenance lives in the field notes — see [[ecosystem-field-notes]].
Related: [[exllamav3-threads-scope]].
