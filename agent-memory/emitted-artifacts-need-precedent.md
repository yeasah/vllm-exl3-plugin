---
name: emitted-artifacts-need-precedent
description: "\"It loads\" is a property of one consumer's loader, not of the format -- before emitting a durable artifact in a shape you invented, check what publishers actually emit"
metadata:
  node_type: memory
  type: feedback
---

When this project *produces* an artifact others will read -- a checkpoint
directory, a file layout, a metadata schema -- passing our own loader is not
evidence the shape is correct. It is evidence that one consumer tolerates it.

**The incident, 2026-09-04.** `tools/quantize_embedding.py --sidecar` first
wrote `bq_*` into an extra shard beside the source files and added an index only
where one already existed. From a single-file checkpoint that yields a directory
with two `.safetensors` and no index. It loaded, was measured, and was tested --
because vLLM globs `*.safetensors` and only consults an index when one is
present. The user asked for precedent before accepting it as a way to produce
checkpoints in general. There was none: across the 69 published snapshots in the
local HF cache, 32 were single-file with no index, 14 sharded with an index, 2
single-file *with* an index, and **zero** had multiple files and no index.

**Why it matters more for output than input.** A checkpoint outlives the reason
it was made and the loader that validated it. Reading a weird artifact is a
problem you find immediately; emitting one is a problem someone else finds
later, and by then there are twelve of them on disk.

**The check is cheap, which is the point.** The HF cache is a corpus of what
publishers actually do -- one shell loop over `models--*/snapshots/*/` counting
`.safetensors` against `*.index.json` answered it in seconds, and flatly
contradicted a design already implemented, tested and committed. The same survey
also produced the *fix*: `openbmb/MiniCPM5-1B` ships one
`model-00000-of-00001.safetensors` plus an index, which made "always use the
sharded convention" an attested shape rather than a guess.

**How to apply:** before emitting a layout or schema, survey what already exists
locally and prefer the attested form even when it costs something -- here a
rename, which was free because the files were hardlinked. When no precedent
exists, that is the finding, not an obstacle to route around. And write the test
against the convention (exact names, index present, `weight_map` matching what
is on disk) rather than against "the file exists", because the second passes for
the wrong layout too. Related: [[check-the-artifact-not-the-prose]],
[[prove-the-guard-fires]].
