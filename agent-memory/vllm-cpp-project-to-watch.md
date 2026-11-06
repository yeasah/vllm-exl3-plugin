---
name: vllm-cpp-project-to-watch
description: mudler/vllm.cpp — C++ port of vLLM worth watching; its token-for-token gating and its null perf result are both relevant to us
metadata: 
  node_type: memory
  type: reference
  originSessionId: 0014c7cb-d253-41bd-9304-bdba01e6a0e4
  modified: 2026-08-17T14:27:11.137Z
---

<https://github.com/mudler/vllm.cpp> — a C++20 port of vLLM's serving core
(continuous batching, paged KV, prefix caching, V1 scheduler), pre-release but
substantial (~2900 commits, 37 architectures). Also heavily AI-assisted
(`.claude/skills`, `.agents/`, Claude Agent SDK), which makes it an interesting
comparison for how this project works.

**Not a target.** Its goals — no Python dependency, 66 MiB binary vs 9.1 GiB
install — are ones the user explicitly does not care about, and bifurcating for
them would be wrong.

**Why it is worth remembering anyway, on 2026-08-17:**

- **It gates correctness by token-for-token identity against vLLM**, across all 37
  architectures. Independent convergence on `bench/`'s core design, from a project
  whose whole purpose is to *replace* the reference — so it had every incentive to
  pick a looser bar.
- **Its headline perf result is a null one, and that is the useful part.** The C++
  port matches Python vLLM throughput at every concurrency level. So vLLM's Python
  orchestration is not costing throughput at scale, which means our plugin's
  Python-side dispatch is not where optimization wins live either. The ~3% decode
  cost on the tied-embedding path is real GPU work (block decode plus two
  Hadamards), not overhead — see [[exl3-embed-head-tax-vs-gguf]].
- **It benchmarks throughput and binary size, not VRAM.** That is the axis this
  project competes on, and a C++ port does not move it: weights are weights. Host
  RAM and startup latency could be genuinely better, which is mildly relevant to
  [[project-vision-local-inference-appliance]], but second-order next to the ~2 GiB
  in an unquantized embedding.
- **Its quantization support is NVFP4, FP8 and GGUF k-quants** (Q3_K..Q6_K) —
  another from-scratch implementation reaching for GGUF rather than inventing a
  format, which is a small data point for the `gguf-embeddings` question.
