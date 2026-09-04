---
name: reap-stray-gpu-processes
description: "Use tools/reap for orphaned GPU processes in vllm-exl3-plugin; pkill -x never matches an EngineCore, and a survivor makes the next run's memory profiling look like a code regression"
metadata:
  node_type: memory
  type: feedback
---

In vllm-exl3-plugin, clear orphaned GPU processes with **`tools/reap`**. No args
reaps whatever holds GPU memory, found via `nvidia-smi --query-compute-apps`, so
process names do not matter; `-n` dry-runs; `-d` sends SIGABRT first for a
faulthandler stack dump, which is the intended workflow for the MoE hangs. It
walks the ancestor chain and refuses to signal anything in it.

`pkill -x VLLM::EngineCore` is the trap it exists to replace: `-x` matches
against `comm`, which the kernel truncates to 15 characters, so the pattern
never matches. Nothing is killed and no error is printed.

**Why:** the failure is silent and mimics an unrelated bug. A surviving
EngineCore keeps holding VRAM, so the *next* run fails memory profiling or
allocates a smaller KV cache — which reads as a code regression in whatever was
changed since. The cost is never the kill itself, it is misattributing the
downstream symptom to the wrong cause.

**How to apply:** reach for `tools/reap` first. If using `pkill` anyway, the
safe forms are a bracketed pattern (`pkill -f '[v]llm_gen'`) or `ps` plus
explicit PID filtering — and see [[pkill-self-kill-wrapper]] for why the
unbracketed form is hazardous from a tool call. Related:
[[verify-across-execution-modes]].
