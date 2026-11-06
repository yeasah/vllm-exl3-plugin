---
name: check-the-artifact-not-the-prose
description: Error messages, comments and READMEs state intent, not behavior; verify against the actual file before reasoning from them
metadata:
  type: feedback
---

Docstrings, error-message text, comments and READMEs describe what someone
*meant*, and drift from what the code does. Do not treat them as predicates.
Check the artifact — the checkpoint, the file listing, the actual branch.

Corrected twice in one session (2026-08-26):

- Claimed exllamav3's `sc_*` tools were deleted at v1.4.3 because the README's
  path convention (`util/sc_*.py`) globbed empty. They live in the repo root.
- Claimed `tools/quantize_embedding.py` refused tied checkpoints, because its
  `SystemExit` says "a tied checkpoint has nothing for this tool to do". That is
  prose in a *not-found* branch, there was no tie check at all, and the claim it
  makes is false for EXL3 — tied checkpoints do carry a dense embedding. The
  user had made tied blockq checkpoints and knew it firsthand.

**Why:** both times the artifact was one command away, and both times the wrong
conclusion propagated into committed docs before being caught — the second
reversed a correct finding the user had supplied.

**How to apply:** when about to assert what code does from something written
*about* it, run the check first — `ls` the path, open the checkpoint, read the
branch. Especially when the prose conveniently confirms the current hypothesis.
Related: [[prove-the-guard-fires]], [[verify-token-ids-not-rendered-text]].
