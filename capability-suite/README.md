# capability-suite

Post-process analysis for capability runs performed by hand — `evalscope` today,
possibly others later. The rationale, the metrics and what the instruments do to the
measurement live in [docs/capability-suite.md](../docs/capability-suite.md); this
directory is the code.

Scope today is **analysis only**: the runs are launched manually and these tools read
what the harness persisted. Managing the runs themselves is a plausible direction, not
a current one.

## `restarts.py`

Counts reconsideration markers in a run's reasoning text, per item, alongside token
count and correctness. The third metric beside correctness and cost: a long trajectory
can be long because the problem is long (legitimate enumeration) or because the model
will not commit, and marker *density* separates those where token count cannot.

    import sys; sys.path.insert(0, "capability-suite")
    from restarts import load
    d = load("~/evalscope/outputs/<run-dir>")
    # {index: {tokens, restarts, density, correct}}

Imported as written, 2026-09-18. Known rough edges, in the order they matter:

- The marker regex is hand-written and unvalidated. Phrasing is model-specific, which
  biases cross-model comparison; within-model comparisons, which is what it is for, are
  unaffected.
- `load()` assumes `evalscope`'s layout and the `gpqa_diamond_default.jsonl` filename.
- Reasoning text is read from `messages[].content[].reasoning`, which is where
  `evalscope` puts it — *not* `choices[].message`, which is null on these runs.
- No CLI. It is a library function called from a scratch script.

Measured on three 198-item GPQA-Diamond runs: raw restart count is a marginally better
failure predictor than token count (AUC 0.746-0.790 against 0.738-0.757) while density
is worse as a predictor and better as a per-item diagnostic. Neither is yet tested on
the question that matters, which is which metric moves most *between arms*.
