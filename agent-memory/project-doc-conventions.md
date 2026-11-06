---
name: project-doc-conventions
description: "vllm-exl3-plugin docs are subject-named notes in docs/, TODO.md holds only open tasks and is referenced by stable slug, never by item number"
metadata:
  node_type: memory
  type: project
  originSessionId: 68cea33f-5688-456b-8dc6-9c580ae1af97
  modified: 2026-08-15T14:10:54.562Z
---

Established 2026-08-15, restructuring documentation that had drifted.

**There are no `PHASEn.md` files any more.** They were renamed to subject notes in
`docs/` (`git mv`, so history follows): `format-and-loading.md`, `kernels.md`,
`tensor-parallel.md`, `moe.md`, `embeddings.md`, plus new `qbench.md`,
`exllamav3-arch.md`, and `feasibility-2026-08-03.md` (frozen, with a header
listing what it got wrong). Phase language survives *inside* notes where it is
historically accurate; it is no longer the addressing scheme. Older memories and
commit messages still say "PHASE2.md" — translate rather than going looking.

**TODO.md holds open tasks only**, governed by a policy block at the top of the
file. The operative test: *if it would still be true and worth reading after the
item is closed, it does not belong there.* Measurements, ruled-out hypotheses, bug
post-mortems and chronology go straight to the matching `docs/` note when written,
never to TODO first for later migration. Each item is ~a screen: outcome, what it
unblocks, current best-candidate approach and why, pointer to the note. Closed
items are deleted and get one line in a capped "Recently closed" section.

**Cross-reference TODO items by stable slug, never by position** — `TODO:
repair-tool`, not `TODO #2`. Do not carry a copy of the slug list here; it rots.
Read it from the file: `grep '^## `' TODO.md`. (As of 2026-08-23 there were 15,
including four added after this memory was written — `bench-suite`,
`transformers-backend`, `multimodal`, `gemma4-e2b`, `capability-suite` — which is
exactly the rot being avoided.)

**Why:** the numbering *was* the problem, twice over. The feasibility report's
"Phase 4" meant packaging while `PHASE4.md` meant quantized embeddings, which is
what forced the awkward switch to "Phase A/B" lettering for a fresh sequence. And
positional TODO references were already embedded in code and notes, so any
eviction would silently repoint them at the wrong item. Subject names never go
stale; plan positions go stale on contact.

**How to apply:** when writing up work, ask which subject note owns it and put it
there — adding to TODO.md is for tasks, not findings. When adding a doc, name it
for its subject and add a row to README's Notes table. The user values the
historical record highly (design decisions, prior results, what was ruled out), so
evicting from TODO means *moving*, never deleting — the frustration being solved
was not having too much history but being unable to find current state inside it.
Related: [[user-wants-to-follow-debugging]],
[[ecosystem-field-notes]] for facts that belong outside the repo entirely.
