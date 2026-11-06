---
name: openwebui-not-a-clean-instrument
description: Two silent corruptions in openwebui that invalidate model observations made through it — a mislabeled penalty default and a concurrent-tagging race
metadata:
  type: project
---

Observations of model behavior made through openwebui carry two defects that
corrupt them silently. Both found 2026-08-18.

**`frequency_penalty`'s displayed default is wrong, and the wrong value is
severe.** Clicking into the field shows `1.1` — the default for
*repetition_penalty*, a different parameter on a different scale. Real
`frequency_penalty` default is `0.0`, range `-2.0..2.0`, additive and scaled by
occurrence count. At `1.1` it is over half maximum strength and reliably
destroys long reasoning traces partway through, because the penalty accumulates
on exactly the tokens a trace legitimately repeats. Repeatable, not sampling
noise. Any session where that field was opened was running with it applied.

**Tagging two agents in one message races them.** They do not serialize. Each
may render against the other's *partially streamed* field and read it as a
finished (often empty) message, then reason about the silence. Which one gets
starved varies per turn, so the handicap cannot be corrected for afterward.
Tag one agent per message if either needs the full conversation.

**Why:** both failures produce fluent, plausible output, so nothing in the
transcript flags them. The `~/circus.json` thread has ~6 exchanges built
entirely on phantom blank replies.

**How to apply:** treat anything measured through this UI as anecdote, not data
— [[qbench-size-claims-vs-appliance-capacity-planning]] and the served-path
work want the API directly. Nothing in `bench/` or the plugin sets penalties;
only openwebui-mediated observations are suspect.
