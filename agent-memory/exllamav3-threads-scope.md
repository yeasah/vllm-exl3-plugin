---
name: exllamav3-threads-scope
description: Pull an exllamav3 thread only when it changes what we ship or how we measure — this is not the fix-exllamav3 show
metadata:
  type: feedback
---

exllamav3's decisions are often thinly founded, so almost any thread pulled
there will find something. That is a trap: follow enough of them and the project
becomes an exllamav3 maintenance effort instead of a vLLM plugin.

**The test before pulling:** does the answer change what *we* ship, or how *we*
measure? If yes, pull it. If it is just exllamav3 being poorly founded and the
fix would live upstream, note it and move on.

Scored against 2026-08-26's threads:
- Tied+blockq corruption — **in scope**, it was our code and a wrong-output bug.
- Layer-curve / marginal convention — **in scope**, our docs feed our decisions.
- mcg vs mul1 codebook — **at the edge**. It produced a real result (+4.3% decode
  for ~2-3% ppl, a tail-shape trade) and does inform which codebook to use for
  new quants, but the stakes were small and it ran long.

**Why:** the user flagged "poorly-supported-decision-turtles all the way down"
and does not want that show. Also a real asymmetry worth reusing: a speed gain
is bounded and fully characterized, while a redistributed divergence is small in
magnitude but unbounded in consequence — you cannot say it won't matter for some
particular input. Those do not trade off cleanly, so decide once and stop.

**How to apply:** when an exllamav3 oddity appears mid-task, say what it is and
what it would cost to chase, then let the user choose rather than chasing by
default. Related: [[iteration-cost-tolerance]], [[ecosystem-field-notes]].
