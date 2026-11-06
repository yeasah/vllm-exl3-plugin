---
name: qbench-size-claims-vs-appliance-capacity-planning
description: "qbench's \"size\" axis should mean total weight bytes only; KV-cache/serving-capacity accounting belongs to the appliance, not the eval harness"
metadata: 
  node_type: memory
  type: project
  originSessionId: d041f792-5c0a-4189-a19a-b1e0850e1167
  modified: 2026-08-08T18:25:11.034Z
---

`eval/qbench`'s bpw/vram accounting (`engines.py`) currently excludes the embedding table
entirely (structurally, for the exl3 backend — `Embedding` isn't a `Linear` so the accounting
walk never sees it) and excludes it by name for the HF/GGUF backends too, undercounting real
checkpoint size. When these plots get used on model cards next to download links, the audience
reads the axis as "how big is this file" regardless of caption scoping, so that undercount is a
real defect worth fixing (sum every tensor's stored bytes, not just ≥2D non-embed/head/router
weights) for any plot presented as a size/VRAM comparison.

Deliberately out of scope for that fix: KV cache, activation memory, batching/offload tradeoffs.
Decided in conversation that these should NOT be folded into qbench's size number, because (a)
they're independent of the model/quantization being compared (kv quant choice, batch size,
offload strategy), and (b) the audience comparing community quantizations already treats KV cache
as a separate, well-tooled budgeting step (dedicated calculators exist) — mixing it in would make
the comparison *less* legible to exactly the users who know how to handle that axis themselves.

**Why:** this creates a clean scope boundary that matters again once appliance work resumes: the
packaged vLLM+EXL3+KV-cache product (see [[project-vision-local-inference-appliance]]) is exactly
the place where full "will this fit and run" capacity planning (weights + KV cache at a target
context length + batching) belongs, because a non-expert appliance user is precisely the audience
who does *not* bring their own calculator. Don't blur the two: qbench answers "how big is this
weight file," the appliance answers "will this configuration run on this hardware."

**How to apply:** when extending qbench's size/vram output, stop at total real stored bytes across
the checkpoint. When building the appliance's capacity-planning/sizing logic, that's the place to
integrate KV cache and batching math — a separate component, not a qbench flag.
