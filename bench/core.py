"""Capture and comparison numerics, shared by `bench/` and `tools/tp_compare.py`.

Both tools ask the same question in different framings -- "do these two engine
configurations compute the same thing?" -- so the numerics live here once.
`tp_compare` compares two live captures against each other (is TP=2 within the
eager-vs-graphs noise floor?); `bench` compares a live capture against a
baseline committed to the repo (did a dependency bump change what we serve?).

The measurement is teacher-forced, and that choice is load-bearing. Sampled-token
comparison conflates numerical error with model confidence, and once two runs
diverge they are on *different contexts*, so everything after the first
difference compares different prompts rather than different arithmetic. Feeding
both configurations a fixed token sequence and reading `prompt_logprobs` scores
every position against an identical context.

What a capture records, and why each part earns its place:

- **prompt token ids**, so a comparison can refuse to run when the two sides were
  not asked the same question. Detokenized text hides special tokens; ids do not.
- **top-k logprobs at every prompt position**, the sensitive channel. A monotonic
  error -- a missing logit scale, a dropped soft cap -- leaves every argmax
  untouched and is *completely invisible* in generated tokens. vLLM's
  Transformers backend dropped MuseGlimmer's `output_multiplier` and
  `final_logit_softcapping` while producing 40 of 40 greedy tokens identical to
  the correct model; only the logprobs moved. See docs/transformers-backend.md.
- **greedy continuation ids**, the coarse channel, and the human-readable one. A
  dropped embedding norm in the same backend changed 7 of 7 tokens.
- **reported weight bytes**, which no logit comparison can reach. The tied
  embedding path serves a model's embedding from its quantized `lm_head` and
  never loads the fp16 `embed_tokens`; if a vLLM change breaks `tie_weights` or
  the tied-skip mapper, the model still produces correct logits and silently
  costs a GiB. That is the project's whole thesis, so it is gated.
"""

from __future__ import annotations

import math

#: Deliberately spans the confidence range, because that is the axis that makes
#: token-based comparison lie. The factual prompt is near-deterministic; the
#: open-ended one leaves the model genuinely uncertain, and is long enough to
#: contribute most of the scored positions.
PROMPTS = [
    "What is the capital of France?",
    "I am comparing three approaches to quantizing large language models: "
    "trellis coding, group-wise integer quantization, and low-rank adapters "
    "applied post-training. For each one, explain the core idea, where the "
    "error comes from, and which hardware makes it fast.",
]


def prompt_ids(tok, text: str) -> list[int]:
    """Chat-templated token ids for one prompt.

    `apply_chat_template` returns a BatchEncoding for some tokenizers and a plain
    list for others, and iterating a BatchEncoding yields its *keys* -- so the
    shape has to be normalized rather than assumed.
    """
    ids = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(i) for i in ids]


def capture_prompts(llm, tok, k: int = 20, new_tokens: int = 24) -> list[dict]:
    """Score every prompt position and greedily continue, for each prompt.

    One `generate` call per prompt does both: `prompt_logprobs` scores the fixed
    context, `max_tokens` produces the continuation. Greedy (temperature 0) so
    the continuation is a property of the model rather than of a seed.
    """
    from vllm import SamplingParams

    out = []
    for text in PROMPTS:
        ids = prompt_ids(tok, text)
        params = SamplingParams(
            temperature=0.0, max_tokens=new_tokens, prompt_logprobs=k, logprobs=k
        )
        result = llm.generate({"prompt_token_ids": ids}, params)[0]

        steps = []
        for pos in result.prompt_logprobs or []:
            if pos is None:  # first position has no predecessor to score
                steps.append(None)
                continue
            steps.append({str(t): round(lp.logprob, 6) for t, lp in pos.items()})

        gen = result.outputs[0]
        out.append(
            {
                "ids": ids,
                "steps": steps,
                "generated_ids": [int(t) for t in gen.token_ids],
                "generated_text": tok.decode(list(gen.token_ids)),
            }
        )
    return out


def kl(p: dict, q: dict) -> float:
    """KL(P||Q) over P's support, renormalized.

    Both sides are truncated to top-k, so Q may not cover all of P. Restricting
    to the shared support and renormalizing keeps this finite; it understates
    divergence when the top-k sets disagree, which is itself reported separately.
    """
    shared = [t for t in p if t in q]
    if not shared:
        return float("nan")
    zp = math.log(sum(math.exp(p[t]) for t in shared))
    zq = math.log(sum(math.exp(q[t]) for t in shared))
    total = 0.0
    for t in shared:
        lp, lq = p[t] - zp, q[t] - zq
        total += math.exp(lp) * (lp - lq)
    return total


def compare_prompt(pa: dict, pb: dict) -> dict:
    """Per-position divergence between two captures of the same prompt.

    Returns `{"comparable": False}` when the two sides were not asked the same
    question, rather than reporting a divergence that means nothing.
    """
    if pa["ids"] != pb["ids"]:
        return {"comparable": False}

    kls, dtop, disagree, n = [], [], 0, 0
    for sa, sb in zip(pa["steps"], pb["steps"]):
        if not sa or not sb:
            continue
        n += 1
        kls.append(kl(sa, sb))
        ta = max(sa, key=sa.get)
        tb = max(sb, key=sb.get)
        if ta != tb:
            disagree += 1
        if ta in sb:
            dtop.append(abs(sa[ta] - sb[ta]))

    finite = [v for v in kls if v == v]
    return {
        "comparable": True,
        "positions": n,
        "argmax_disagreements": disagree,
        "kl_max": max(finite) if finite else 0.0,
        "kl_mean": sum(finite) / len(finite) if finite else 0.0,
        "dlogprob_max": max(dtop) if dtop else 0.0,
        "dlogprob_mean": sum(dtop) / len(dtop) if dtop else 0.0,
        "generated_match": pa.get("generated_ids") == pb.get("generated_ids"),
    }
