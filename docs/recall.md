# Recall

Whether a quantization keeps what the unquantized model knows, measured with closed-book questions scored
token by token. The working record is below; what reaches the model card is facts kept (sum of min(P, Q)
over sum of P), confident errors, and kept by band of the reference's confidence, from a per-model slice
that `quant.py recall-slice` chooses and refuses when thin. `docs/calibration.md` and `docs/yaqa.md` carry
the experiments that used it.

## Moved from TODO.md (2026-11-05)

*The working record of these items as they stood when TODO.md was cut back to open tasks, kept verbatim. Open work on them, if any, is in TODO.md or FUTURE.md under the same slug or the one named.*

### `recall-slice` — A hallucination detector: does a quant keep what the reference knows?

**Outcome:** decide whether recall joins the pipeline's eval and the card, and whether YAQA's cost on it
can be removed. `eval/recall_trace.py` (PopQA, MIT: closed-book questions with subject page views) gives
qbench per-row answer log-prob and argmax hit under teacher forcing; analysis takes each question's
reference-preferred alias, conditioned on the reference producing it (`quant_work/_recall/analyze.py`).
About 5 minutes per arm on a 0.6B. The all-questions variant is tail noise: stay conditioned on known.

**Known (Qwen3.8-27B, 1830 known of 7678, floor 99.2% retained):** retention 67.6 / 88.1 / 92.3 / 94.0 /
97.0 / 98.5% at 2 / 3 / 3.5 / 4 / 5 / 6 bpw, with a slump in the middle popularity quintiles that the
baselines share; 2 bpw is a cliff (66.2% uncalibrated too), which backs retiring it. Recipe and calibration
cost no recall: at 4 bpw one-sided holds 95.7% against the baseline's 95.1%, while halving KL. **YAQA does
cost it:** the 4 bpw blend retains 94.0%, dlp -0.092 against -0.034 ([-0.075, -0.041] paired), for only -2.6%
wild and -7% swe KL. The damage is confident same-category substitution (a director for a director), at
2 bpw pulling toward famous names. A few "losses" are valid variants or engine near-ties; retention slightly
overstates real loss, and the floor absorbs that.

**The question set decides the sign.** PopQA alone misleads, so `eval/recall_wikidata.py` adds 12,006
Wikidata facts across 12 domains (`recall_trace.py --facts`). On MiniCPM5-2B the own-voice recipe is -4.0
points on PopQA and +3.0 on the facts, and a recipe measured with PopQA rows gains +4.4 on PopQA but +0.9 on
the facts: it protects PopQA, not recall in general. Report numeric answers (years, chromosomes) as their
own slice and keep them out of the headline: on Ornith-9B, `gene_chromosome` alone moves 36.9 to 85.8 with
the recipe. History has no non-numeric relation yet.

**Known (Ornith-1.5-9B facts, text answers, 1706 known):** calibration +10.7 points at 3 bpw but +1.1 at 4
(not significant). Recipe +2.8 and +1.2. YAQA adds retention alone but not on top of a recipe. Calibration
is the strong lever only at low rates.

**K-quants are the target for recall, not KL.** A same-size K-quant gives an achievable retention and a
warning light. Compare served text bytes: the 27B EXL3 checkpoint carries 0.91 GB of vision tower that a
GGUF keeps in `mmproj`, and its 2.55 GB BF16 embedding serves at about 0.72 GB under blockq. The bit target
covers the body alone (4.00 is 4.01 bpw over the layers; head at 6), so this is a size-axis correction, not
an allocation one. GGUFs stay off the KLD comparison plot, where they aren't competitive, and go in any
recall comparison. UD files are per-tensor mixes, not plain K-quants.

**Before publishing the 27B or Ornith-9B,** at one bitrate/calibration/recipe config first, on recall, wild
and swe:
- YAQA blends: measure the impact of different mixes. Unblended (no YAQA) is the current 27B pick.
- Recall content in calibration: measure it the same way.
- Make recall metrics as well behaved as they can be, and decide whether the card can present them clearly
  without overstating what they are.
- A UD 3.0 GGUF arm in the recall comparison only.

Then a policy question: a retention floor for published rungs.
