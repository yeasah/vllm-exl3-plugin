#!/usr/bin/env python
"""Needle-in-a-haystack retrieval under a TurboQuant KV cache.

The companion to `gsm8k_kv.py`, and the reason it exists: many-shot GSM8K's long
context is 175 near-identical exemplars, which is both highly redundant and
minimally retrieval-stressing, so a null result there does not transfer. This
task is the opposite -- one fact, one position, no redundancy, and answering
requires attending to a single distant token span.

    run:     niah_kv.py run MODEL KV BOUNDARY N OUT.json CTX_TOKENS
    report:  gsm8k_kv.py report OUT.json ...        (same JSON schema)

Every config sees an identical trial list -- same haystack, same needle depths,
same values, fixed seed -- so per-item results pair exactly and the McNemar test
in `gsm8k_kv.py` applies unchanged. Depth is drawn uniformly per trial rather
than gridded, which gives smooth coverage of the depth axis for the profile.

Single-needle retrieval saturates: on Qwen3-4B every configuration scored 93-100%
at 2K/8K/32K with no ordering by KV quality, so it resolves nothing. NEEDLES>1
switches to the multi-needle retrieve-all form -- K facts at random depths, list
them all -- which is harder in the way that matters (recall of *every* distant
span, not one) and scores per needle, so N trials yield N*K paired outcomes.

Env: UTIL (gpu_memory_utilization), SKIP_SLIDING, NEEDLES (default 1).
MML is set from CTX_TOKENS.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsm8k_kv import build_llm, override_boundary  # noqa: E402

NEEDLE = "The special magic {key} number is: {value}."
QUESTION = (
    "\n\nWhat is the special magic {key} number mentioned in the text above? "
    "Answer with the number only."
)
QUESTION_ALL = (
    "\n\nThe text above contains {k} statements of the form \"The special magic "
    "<city> number is: <number>\". List every one of those numbers, separated by "
    "commas.\nAnswer:"
)
QUESTION_SAME = (
    "\n\nThe text above states \"The special magic {key} number is: <number>\" "
    "{k} times, each with a different number. List all {k} of those numbers, "
    "separated by commas.\nAnswer:"
)
KEYS = ["Denver", "Osaka", "Lisbon", "Nairobi", "Quito", "Bergen", "Perth",
        "Halifax", "Cordoba", "Tromso", "Dunedin", "Salvador"]


def run(model, kv, boundary, n, outp, ctx):
    override_boundary(boundary)
    os.environ.setdefault("MML", str(ctx + 512))

    from datasets import load_dataset
    from vllm import SamplingParams

    llm, skips = build_llm(model, kv)
    tok = llm.get_tokenizer()

    # One haystack, reused across trials, so only needle depth and value vary.
    # wikitext-103 rather than openwebtext: the cached copy of the latter is
    # script-based and modern `datasets` refuses to load it.
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    hay, i = "", 0
    while len(hay) < (ctx + 2000) * 4:      # ~4 chars/token, then trim exactly
        hay += ds[i]["text"]
        i += 1
    hay_ids = tok.encode(hay)[:ctx]

    k = int(os.environ.get("NEEDLES", 1))
    # SAME_KEY makes every needle share one key, so the k spans are
    # indistinguishable except by content and position. That attacks
    # discrimination rather than recall -- distinct keys are trivially separable
    # and saturate (200/200 at k=8), which is the failure this replaces.
    same = bool(int(os.environ.get("SAME_KEY", 0)))
    rng = random.Random(20260830)
    prompts, golds, depths = [], [], []
    for _ in range(n):
        one = rng.choice(KEYS)
        keys = [one] * k if same else rng.sample(KEYS, k)
        picks = sorted(
            (rng.random(), rng.randrange(100000, 999999), key) for key in keys
        )
        text, prev = "", 0
        for depth, value, key in picks:
            cut = int(depth * len(hay_ids))
            text += tok.decode(hay_ids[prev:cut]) + "\n"
            text += NEEDLE.format(key=key, value=value) + "\n"
            prev = cut
        text += tok.decode(hay_ids[prev:])
        prompts.append(
            text + (QUESTION.format(key=picks[0][2]) if k == 1
                    else QUESTION_SAME.format(k=k, key=picks[0][2]) if same
                    else QUESTION_ALL.format(k=k))
        )
        golds.append([str(v) for _, v, _ in picks])
        depths.append([d for d, _, _ in picks])
    print(f"PROMPT CONTEXT: {len(tok.encode(prompts[0]))} tokens, "
          f"{n} trials x {k} needles", flush=True)

    outs = llm.generate(
        prompts,
        SamplingParams(
            max_tokens=24 if k == 1 else 16 * k + 64,
            temperature=0,
            stop=["\n\n", "\nThe special magic"],
        ),
    )
    # One binary outcome per needle, so the paired unit is (trial, needle).
    items = [int(g in o.outputs[0].text) for o, gs in zip(outs, golds) for g in gs]
    if os.environ.get("DUMP"):
        for o, gs in list(zip(outs, golds))[: int(os.environ["DUMP"])]:
            print("GOLD:", ",".join(gs), flush=True)
            print("OUT :", repr(o.outputs[0].text[:400]), flush=True)
            print("FINISH:", o.outputs[0].finish_reason, flush=True)
    depths = [d for ds in depths for d in ds]

    res = dict(
        model=model, kv=kv, boundary=boundary, n=len(items), correct=sum(items),
        acc=sum(items) / len(items), skip_layers=skips, items=items,
        task="niah", ctx=ctx, depths=depths, needles=k, trials=n,
        same_key=same,
    )
    json.dump(res, open(outp, "w"), indent=1)
    print("RESULT", json.dumps({k: v for k, v in res.items()
                                if k not in ("items", "depths")}), flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    _, _, model, kv, boundary, n, outp, ctx = sys.argv[:8]
    run(model, kv, boundary, int(n), outp, int(ctx))
