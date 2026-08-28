# YAQA probe

Answers one question before any converter work: does YAQA-quality rounding
(arXiv:2505.22988) actually lower the end-to-end KL on EXL3's quantizer? Findings and
interpretation live in [../../docs/yaqa.md](../../docs/yaqa.md); this file says what
the pieces are.

| file | what it does |
|---|---|
| `rounding.py` | the YAQA wavefront, calling EXL3's own `quantize_tiles()` |
| `test_guard.py` | three correctness guards; run this first, it needs no model |
| `probe.py` | quantize one layer, measure whole-model KL against the original |
| `hessian_fit.py` | how well a sketch approximates the true Hessian, no quantizer involved |

```bash
python3 tools/yaqa/test_guard.py
python3 tools/yaqa/probe.py --model <hf-dir> --layers 14 --projs mlp.down_proj --bits 3
python3 tools/yaqa/hessian_fit.py --model <hf-dir> --layers 14 --projs mlp.down_proj
```

## Why the guards matter

A null result is only worth reporting if the thing being tested works, and every
component here can fail silently — a wrong sweep order, a mistransformed `H_O`, or a
noise floor larger than the effect all look like "YAQA does not help".

1. **`H_O = I` reproduces EXL3's `ldlq()` bit-for-bit.** Identical encoded indices,
   `max |Δrecon| = 0`. The wavefront is a legal reordering of the existing row sweep.
2. **Synthetic low-rank `H_O` cuts the loss YAQA targets by ~83%** while *increasing*
   LDLQ's objective and doubling `‖Δ‖²`. An algorithm that improved every metric at
   once would mean the harness was measuring nothing.
3. **The transformed Hessians are the right ones for the transformed weight**, to
   ~1e-7. Rounding happens in EXL3's incoherence-processed space, so `H_O` needs the
   same `(sv, had_n)` treatment `finalize_capture_H()` gives `H_I`. Getting this wrong
   does not crash; it silently optimizes a scrambled objective.

`probe.py` additionally carries a `ldlq-dup` arm — a bit-identical copy of the `ldlq`
weight — through the KL measurement as a live noise floor. It must read `+0.0%`. With
TF32 left on or SDPA instead of eager attention it does not, and few-percent effects
become unreadable.

## Notes

- Sketch B needs no custom autograd at this scale. Appendix A.9's einsums are just
  `H_I += GᵀG`, `H_O += GGᵀ` for the per-*sequence* weight gradient of the
  Monte-Carlo-sampled cross entropy — the real Fisher, not the empirical one, so the
  label is sampled from the model's own output and is not the next token.
- Context length is capped by the LM head, not the model: a 152k vocabulary at 2048
  tokens is a 1.24 GiB fp32 logit tensor, which OOMs a 16 GiB card during the backward.
  `mc_sample()` chunks the softmax to keep the peak down; beyond that, lower `--ctx`.
- EXL3's bundled calibration corpus is ~1.42M tokens, i.e. 694 unique rows at 2048
  tokens. The paper uses 2K–64K sequences, so matching its budget needs a real corpus.
