---
name: ecosystem-field-notes
description: "~/notes/vllm-ecosystem-field-notes.md holds facts about the stack rather than the project; the routing rule, the provenance requirement, and when a repeat becomes a seam"
metadata:
  type: project
---

Established 2026-08-23, after scanning the project's whole transcript history for
knowledge that had been established and never written anywhere.

**The file is `~/notes/vllm-ecosystem-field-notes.md`, deliberately outside the
repo.** It holds facts about vLLM, huggingface_hub, transformers, exllamav3,
hardware and measurement method that are *not about this project* and outlive it.
Part 1 is recurring seams, Part 2 is facts by subject, Part 3 is falsified claims.

**Routing.** Three cases, and the third is the one that needs a rule:

- *About our code or our decisions* → `docs/` subject note. Never the field notes;
  letting project facts in is how the file stops being useful.
- *About a system we do not own, with no project bearing* → field notes. This is
  the self-describing case: "we learned this and there is nowhere to put it".
- *Both* → **the project note owns the incident, the field notes own the
  pattern.** They answer different questions — `docs/` answers "why does our code
  do this", the field notes answer "what should I check next time" — so the
  duplication is intended, not redundancy to be resolved.

**The test for whether a general form exists:** *would this have happened to
someone who never touched EXL3?* If yes, extract the general statement and file
it. The offload seam, the Transformers-backend rule and the `quant_config`
plumbing gap all pass; the Laguna divisor does not, but the *class* it belongs to
("conventions living in code, not the checkpoint") does.

**Provenance is not optional.** Every entry carries how it was established and
when. An unsourced negative fact is indistinguishable from the forum mythology
these exist to correct, and a dated entry with its method attached can be
re-checked in minutes where a bare assertion has to be rediscovered. Falsified
claims stay in Part 3 rather than being deleted, because the wrong version is what
gets remembered.

**Two instances make a seam.** A single incident is a fact; the second instance of
the same underlying joint is the signal to promote both into Part 1 with a *tell*
— the observable that identifies it next time ("something reports success with an
implausibly small number", "greedy is fine and logprobs are not"). Seam analysis
is periodic rather than per-incident, because no single write-up can see a pattern
across months; a bump or a "this feels familiar" moment is the natural trigger.

**Why:** the interaction space of this stack is unbounded and cannot be mapped,
but the load-bearing edges are few and they recur — six seams accounted for ~28
incidents across five months. Documentation of any kind describes units, not
interactions, and generated documentation structurally cannot state *scope*, which
is a negative fact and exists nowhere in the source to be extracted. So the
negative facts have to be accumulated by hitting them, and the only thing that
makes that affordable is never hitting one twice.

**How to apply:** when writing up work, ask the routing question before reaching
for `docs/`. If the answer is "both", write the incident where it belongs and the
pattern in the field notes, and cross-reference by subject rather than by section
number. Related: [[project-doc-conventions]], [[check-upstream-before-patching-vllm]],
[[prove-the-guard-fires]].
