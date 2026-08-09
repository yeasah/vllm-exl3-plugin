#!/usr/bin/env python3
"""Compare two engine configurations by distribution, not by sampled tokens.

Sampled-token comparison is a bad instrument for this. It conflates numerical
error with model confidence -- Laguna-XS at 2bpw diverges at token 16 of 48 on an
open-ended prompt (top-1 logprob -1.2) and not at all on a factual one (-0.0001),
same checkpoint -- and worse, once two runs diverge they are on *different
contexts*, so everything after the first difference compares different prompts
rather than different arithmetic.

So teacher-force instead: feed both configurations the same fixed token sequence
and read `prompt_logprobs`, which scores every position against an identical
context. Then the comparison is per-position and the model's confidence drops out.

The number that matters is not any single divergence but its size relative to a
**noise floor**. Capture eager vs CUDA graphs at TP=1 -- same weights, same
sharding, no collectives -- and whatever differs there is the irreducible
kernel-ordering noise for that checkpoint. TP=2 is then judged against it:

    capture  tp1 eager      -> a.json      \\  noise floor
    capture  tp1 graphs     -> b.json      /
    capture  tp2 eager      -> c.json
    compare  a.json b.json                 # floor
    compare  a.json c.json                 # must be comparable to the floor

Usage:
    tools/tp_compare.py capture <model> <out.json> [--tp N] [--eager] [--k 20]
    tools/tp_compare.py compare <a.json> <b.json>
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

#: Deliberately spans the confidence range, because that is the axis that makes
#: token-based comparison lie. The factual prompt is near-deterministic; the
#: open-ended one leaves the model genuinely uncertain.
PROMPTS = [
    "What is the capital of France?",
    "I am comparing three approaches to quantizing large language models: "
    "trellis coding, group-wise integer quantization, and low-rank adapters "
    "applied post-training. For each one, explain the core idea, where the "
    "error comes from, and which hardware makes it fast.",
]


def resolve(target: str) -> str:
    """Snapshot that actually holds weights (see tp_preflight.resolve)."""
    if os.path.isdir(target):
        return target
    hits = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--" + target.replace("/", "--")
        + "/snapshots/*"))
    hits += glob.glob(os.path.expanduser(
        "$HF_HOME/hub/models--" + target.replace("/", "--") + "/snapshots/*"))
    hits += glob.glob(os.path.join(
        os.environ.get("HF_HOME", "/nonexistent"), "hub",
        "models--" + target.replace("/", "--"), "snapshots", "*"))
    if not hits:
        raise SystemExit(f"no local snapshot for {target!r}")
    n, best = max((len(glob.glob(os.path.join(h, "*.safetensors"))), h)
                  for h in hits)
    if not n:
        raise SystemExit(f"no snapshot of {target!r} contains safetensors")
    return best


def capture(args) -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    from vllm import LLM, SamplingParams

    path = resolve(args.model)
    llm = LLM(model=path, max_model_len=4096, max_num_seqs=2,
              tensor_parallel_size=args.tp, gpu_memory_utilization=0.85,
              enforce_eager=args.eager)
    tok = llm.get_tokenizer()

    out = {"model": args.model, "tp": args.tp,
           "mode": "eager" if args.eager else "graphs", "k": args.k,
           "prompts": []}
    for text in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                      tokenize=True, add_generation_prompt=True)
        # Returns a BatchEncoding for some tokenizers and a plain list for
        # others; iterating a BatchEncoding yields its *keys*.
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = [int(i) for i in ids]
        # max_tokens=1 because we only want the prompt scored; prompt_logprobs
        # gives one distribution per prompt position, all at fixed context.
        r = llm.generate({"prompt_token_ids": ids},
                         SamplingParams(temperature=0.0, max_tokens=1,
                                        prompt_logprobs=args.k))[0]
        steps = []
        for pos in r.prompt_logprobs or []:
            if pos is None:          # first position has no predecessor
                steps.append(None)
                continue
            steps.append({str(t): round(lp.logprob, 6) for t, lp in pos.items()})
        out["prompts"].append({"ids": list(ids), "steps": steps})

    with open(args.out, "w") as f:
        json.dump(out, f)
    n = sum(len([s for s in p["steps"] if s]) for p in out["prompts"])
    print(f"wrote {args.out}: tp={args.tp} {out['mode']}, {n} scored positions")


def kl(p: dict, q: dict) -> float:
    """KL(P||Q) over P's support, renormalized.

    Both sides are truncated to top-k, so Q may not cover all of P. Restricting
    to the shared support and renormalizing keeps this finite; it understates
    divergence when the top-k sets disagree, which is itself reported.
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


def compare(args) -> None:
    a = json.load(open(args.a))
    b = json.load(open(args.b))
    print(f"A: tp{a['tp']} {a['mode']}   B: tp{b['tp']} {b['mode']}\n")
    for i, (pa, pb) in enumerate(zip(a["prompts"], b["prompts"])):
        if pa["ids"] != pb["ids"]:
            print(f"  prompt {i}: token ids differ, not comparable")
            continue
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
        print(f"  prompt {i}: {n} positions")
        print(f"    argmax disagreements : {disagree}/{n}")
        print(f"    KL(A||B)  max / mean : {max(finite):.3e} / "
              f"{sum(finite)/len(finite):.3e}")
        print(f"    |dlogprob| of A top-1: max {max(dtop):.3e} / "
              f"mean {sum(dtop)/len(dtop):.3e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("model"); c.add_argument("out")
    c.add_argument("--tp", type=int, default=1)
    c.add_argument("--eager", action="store_true")
    c.add_argument("--k", type=int, default=20)
    c.set_defaults(func=capture)
    d = sub.add_parser("compare")
    d.add_argument("a"); d.add_argument("b")
    d.set_defaults(func=compare)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
