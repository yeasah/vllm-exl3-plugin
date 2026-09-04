---
name: verify-token-ids-not-rendered-text
description: Never declare a model correct from decoded text alone; check token ids and first-step logprobs
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 94844e7e-fafd-47b5-b2cf-e671747f8da1
  modified: 2026-08-06T20:02:48.402Z
---

Do not call a model "working" on the strength of decoded output text. Check the
raw `token_ids` and the per-step top-1 logprob. On 2026-08-06 Laguna-XS-2.1 was
reported as working twice, on text that read `"The capital of France is
**Paris**."` while its actual output began with token id 0 — a special `UNK`
that vLLM hides during detokenization, produced by an all-NaN prefill hidden
state. The leading `0` was printed in the very first run and read past.

**Why:** detokenization deliberately suppresses special tokens, so a broken
token is invisible in text. Greedy decoding hides it further — argmax of a
degenerate tensor still returns *something*, and only sampling at temperature >
0 exposes it. Two independent layers of concealment meant the bug survived
several "verified" runs.

**How to apply:** for any correctness claim, report `token_ids[0]` and the
step-1 top-1 logprob. A top-1 of exactly `-ln(vocab_size)` means uniform logits,
i.e. a constant/zero logits row. Also sample at temperature > 0, not only
greedy. When a hidden state is suspect, hook `compute_logits` and print
`nonfinite` counts. Related: [[exl3-checkpoints-carry-unrecorded-scales]],
[[user-wants-to-follow-debugging]].
