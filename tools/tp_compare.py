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
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The prompts and the divergence numerics are shared with bench/, which asks the
# same question against a committed baseline rather than against a second live
# capture. One implementation, because two copies of a KL that drift apart would
# undermine both tools at once.
from bench.core import PROMPTS, compare_prompt, kl, prompt_ids  # noqa: E402,F401


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
        ids = prompt_ids(tok, text)
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


def compare(args) -> None:
    a = json.load(open(args.a))
    b = json.load(open(args.b))
    print(f"A: tp{a['tp']} {a['mode']}   B: tp{b['tp']} {b['mode']}\n")
    for i, (pa, pb) in enumerate(zip(a["prompts"], b["prompts"])):
        m = compare_prompt(pa, pb)
        if not m["comparable"]:
            print(f"  prompt {i}: token ids differ, not comparable")
            continue
        print(f"  prompt {i}: {m['positions']} positions")
        print(f"    argmax disagreements : {m['argmax_disagreements']}"
              f"/{m['positions']}")
        print(f"    KL(A||B)  max / mean : {m['kl_max']:.3e} / "
              f"{m['kl_mean']:.3e}")
        print(f"    |dlogprob| of A top-1: max {m['dlogprob_max']:.3e} / "
              f"mean {m['dlogprob_mean']:.3e}")


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
