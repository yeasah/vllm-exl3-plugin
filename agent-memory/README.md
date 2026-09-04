# agent-memory

Curated notes an AI assistant keeps about this project, one fact per file.
They are the distillate of working sessions — the things worth carrying between
them — and they are tracked here for three reasons: they are project context
like any other, they benefit from review in diffs, and they were previously
protected by nothing but VM backup rotation.

The live copies live in the assistant's own memory directory, which is keyed to
this repository's absolute path; the files there are symlinks back into this
directory, so editing either edits both.

## What is here, and what is not

Only memories that describe **the project** — its domain, conventions,
measurement practice and history. Anything specific to one operator or one
machine is deliberately kept out: their infrastructure, their working
preferences, credentials and paths. Those exist, they are simply not the kind
of thing that would help someone else working on this code.

A consequence: a few `[[wiki-links]]` here point at memories that are not
published, so they will not resolve in this directory. That is expected, and
the link text names the subject well enough to be read as prose. Links to
`[[bench-suite-purpose]]` resolve to nothing anywhere yet — that one marks a
memory worth writing rather than one being withheld.

## Format

YAML frontmatter (`name`, `description`, `metadata.type`) and a short body.
`description` is what a session sees when deciding whether a memory is relevant,
so it carries the hook. `feedback` and `project` entries follow the body with
**Why:** and **How to apply:** — the reasoning and the action, kept apart
because the reasoning is what survives when the specifics change.

These are working notes, not documentation. Where they overlap with `docs/`,
`docs/` is authoritative: it is written to be read, these are written to be
recalled.
