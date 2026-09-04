---
name: prove-the-guard-fires
description: Never ship a check, guard or alarm without reintroducing the failure and watching it trigger
metadata:
  type: feedback
---

When adding a sanity check, assertion, accounting guard or regression test, put the
original failure back and demonstrate the guard catching it, then say so in the commit.
A guard nobody has seen fail is a comment.

**Why:** it is cheap (usually a couple of minutes — monkeypatch the table, drop a tensor
from the counted set) and it repeatedly finds that the guard is weaker than intended. The
qbench storage guard looked fine as a ratio-against-threshold until it was aimed at a
model with a vision tower, where the slack could hide a dropped embedding entirely; that
test is what forced the redesign to per-tensor classification.

**How to apply:** demonstrate two cases where practical — the failure it was built for,
and a harder variant where something legitimately large is missing for unrelated reasons.
Pairs with [[verify-across-execution-modes]] and the project's habit of computing a figure
two independent ways and requiring agreement.

