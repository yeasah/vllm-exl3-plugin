---
name: exl3-embed-head-tax-vs-gguf
description: "EXL3 loses to GGUF on embed+head bytes by 3-4x, and fixing it would put the project ahead of every GPU-serving quantization format; the measurements now live in docs/embeddings.md"
metadata:
  node_type: memory
  type: project
  originSessionId: d041f792-5c0a-4189-a19a-b1e0850e1167
  modified: 2026-08-15T14:10:35.623Z
---

The strategic finding that motivated the whole quantized-embedding line of work:
EXL3 spends 3-4x more on embed+head than GGUF (3.3-4x on Llama-3.2-1B; still
14.4% of the whole package on gemma-4-26B-A4B-it at 2.10bpw), because it never
quantizes the embedding and never exploits a tie. On real total file size that is
enough for bartowski's IQ4_XS to be *smaller than every EXL3 checkpoint tested*
and beat two of three on KLD — the opposite of what embed/head-excluded plots
show. AWQ, GPTQ, AutoRound and DASHQ all ship fp16 embeddings too, so **GGUF is
the only format that gets this right, and fixing it puts the project ahead of the
entire GPU-quantization ecosystem rather than at parity with one competitor.**

**All measurements, tables and derivations now live in `docs/embeddings.md`**
("Against other formats" for the GGUF/ecosystem comparison, and the rest of the
note for the depth study). Do not re-derive them here; that file supersedes this
one for anything measured.

**Why:** this is the "why does any of this matter" framing that is easy to lose
once deep in per-row-vs-trellis detail. The work is not tidying — it closes the
one axis where EXL3 currently loses outright, and the win is largest exactly at
low bpw, i.e. for the VRAM-constrained users this project targets
([[project-vision-local-inference-appliance]]).

**Three beliefs from the 2026-08-08 investigation were later falsified** — noting
only that they existed, since re-reading the reasoning is what would mislead:
per-row extraction from a trellis *is* cheap and exact (so this is a real VRAM
win, not disk-only); the trellis is *not* generally better than per-row RTN for an
embedding (it is 89x worse on gemma-4-12B, while being 63x better for a head); and
there is **no universal embedding bit-depth constant** — sensitivity spans 35x
across models, so any heuristic fitted to one family is actively harmful on
another. Detail in `docs/embeddings.md`.

**Standing merge policy for the exllamav3 fork, set 2026-09-10:** upstream's qbench
excludes `token_embd.weight` from its size accounting; our fork counts it. **Never
adopt that exclusion on a merge** — it is the disagreement this whole line of work
rests on, since a size number that omits the embedding cannot show the tax. It
conflicts on every merge that touches `eval/qbench/engines.py` (it did at v1.4.9),
and taking upstream's side is *silent*: every plot and table keeps rendering while
the axis quietly stops meaning what the surrounding documents say, and comparability
with every figure already in `docs/embeddings.md` and `docs/qbench.md` is lost. The
reasoning is recorded in that engines.py docstring and in docs/qbench.md's "Scope"
section. Upstream's orthogonal exclusions (NextN/MTP blocks) are fine to take.

**How to apply:** when weighing embedding/head work against other tasks, price it
against this competitive gap rather than against internal tidiness. The fork-side
pipeline fix is worth attempting to upstream, with low expectation of it landing.
YAQA sits in the same layer of the stack but is a *different kind* of win —
embed/head work recovers already-wasted bytes to reach parity, YAQA is a fidelity
improvement beyond parity. Related:
[[qbench-size-claims-vs-appliance-capacity-planning]],
[[project-doc-conventions]].
