---
name: verify-across-execution-modes
description: A fix verified only under enforce_eager says nothing about the CUDA-graph path; vary execution mode and engine config before claiming a bug is fixed
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 94844e7e-fafd-47b5-b2cf-e671747f8da1
  modified: 2026-08-07T18:02:27.307Z
---

Do not claim a vLLM-side bug is fixed after testing one execution mode. On
2026-08-06 a MoE hang was declared fixed on the strength of 7/7 clean runs — all
of them `enforce_eager=True`, short prompt, `max_model_len=2048`. The graph path
still hung, and so did other models at other context lengths. The real cause was
elsewhere entirely (an sm_90+ barrier in exllamav3), and the "fix" was a
coincidence of one model at one configuration.

**Why:** for this project the axes that flip outcomes are `{graphs, eager}`,
`max_model_len` / `max_num_seqs` (they set `max_num_batched_tokens`, hence kernel
grid geometry), and model. Prompt content matters far less than it appears to. A
single-cell result generalizes to nothing, and CUDA graphs are where the real
throughput lives — 1.9x on dense, 4.9x on MoE — so an eager-only verdict also
misprices the fix.

**How to apply:** run a small matrix, not a repeat count, before saying "fixed".
One run per cell first for breadth, repeats only where a cell fails. Record
per-cell `token_ids[0]` and tok/s alongside pass/fail — a hang and a silently
degenerate first token are both invisible in decoded text. See
[[reap-stray-gpu-processes]] for cleaning up between cells.
