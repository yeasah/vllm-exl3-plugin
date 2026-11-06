---
name: project-vision-local-inference-appliance
description: "A separate downstream appliance project will consume vllm-exl3-plugin; it broadens which use cases this project weighs, but does not narrow this project to its audience"
metadata: 
  node_type: memory
  type: project
  originSessionId: 94844e7e-fafd-47b5-b2cf-e671747f8da1
  modified: 2026-08-08T19:16:23.610Z
---

There is a larger goal (stated 2026-08-05): package vLLM + EXL3 + KV-cache
optimizations (turboquant, triattention) into a Docker image with a frontend
orchestration/management layer, so people who are not "arms-deep in the
ecosystem" can run good local inference.

**That is a separate project which will be a *user* of this one** (clarified
2026-08-15), not this project's own end goal. vllm-exl3-plugin is a general
quantization backend; the appliance is downstream of it. Keep the boundary — this
distinction is deliberately *not* documented at the vllm-exl3-plugin level,
because the plugin's decisions should read as generally applicable rather than as
serving one consumer.

`triattention` is the next component the user wants to look at, once this plugin
reaches a successful conclusion. It already exists at `/home/ypell/git/triattention`
alongside `vllm`, `vllm-omni` and this repo.

**Why it still matters here:** it is a **broadening of consideration, not a
constraint**. Knowing a memory-constrained, modest-hardware consumer exists widens
the set of plausible use cases this project weighs — it does not license tuning
the plugin to that audience, or treating other use cases as secondary. The
ultimate decisions should stay applicable to plausible use cases generally.

**Target audience hypothesis** (stated 2026-08-08, held loosely — may turn out
false, in which case the work still stands on its own for other audiences):
small teams, limited hardware and limited utilization, running efficiently-sized
sparse (MoE) models in the ~5-20 t/s generation band — a personal, per-user
pivot point that's too slow to feel good for interactive chat but plenty fast
for long-horizon asynchronous/agentic use. For that audience, concurrency is a
poor trade in the interactive case (not enough headroom to make batching worth
it) but effectively inevitable and valuable for long-running async/agentic
workloads, since those already spend most of their wall-clock time on
tool-call/compile/browse latency rather than generation — which is the actual
argument for vLLM specifically (continuous batching) over a simpler
single-stream local engine. For that same audience, priorities are ~100% memory
efficiency over peak throughput, which can lead to counterintuitive calls (e.g.
disabling CUDA graphs may be correct despite the speed cost, since graph
capture buys decode latency by spending the memory this audience doesn't have
to spare). EXL3 (weight quantization) is the first lever; KV cache is expected
to be an even bigger one for this specific shape (sparse MoE compresses total
weight bytes without touching attention-state cost, so KV cache's share of the
memory budget grows *because* EXL3 worked) — not started yet, deliberately
sequenced after EXL3. At minimum, even if the full appliance vision doesn't
materialize, this is expected to produce a low-resource-compatible vLLM useful
to the user personally.

**How to apply:** favour decisions that survive packaging — pinned/reproducible
builds, sane defaults over required flags, clear startup errors over silent
degradation. Treat anything that forces a user to pass an obscure env var or
match a bit-rate branch by hand as a defect worth fixing rather than documenting.
Packaging and CI against vLLM main is the real finish line for the plugin, and it
is what makes it consumable downstream.

Use the audience hypothesis to **widen the option set, not to pick the winner**:
it is why an unglamorous memory-efficiency option (e.g. trading CUDA graphs' speed
for their capture memory) deserves to be on the table and measured at all, rather
than dismissed as obviously wrong. It is not grounds for making that the default,
and there is no standing "memory beats throughput" rule for this project. Where a
decision genuinely only serves the appliance's audience, it belongs in the
appliance, not here. Related: [[project-doc-conventions]].

**The field notes are seed material for it** (decided 2026-08-23).
`~/notes/vllm-ecosystem-field-notes.md` is intended for eventual publication as part
of, or as the genesis of, that project — the negative facts about this stack are
precisely what an appliance's users would otherwise have to rediscover. Re-verification
against current versions is deferred to that point rather than done continuously; until
then the as-of convention holds and the file stays private. See
[[ecosystem-field-notes]].
