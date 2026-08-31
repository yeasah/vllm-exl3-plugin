#!/usr/bin/env python
"""GSM8K under a TurboQuant KV cache, with the boundary skip layers controllable.

Answers "what does `get_boundary_skip_layers` actually buy?" by scoring the same
1319 GSM8K problems under KV-cache configurations that differ in exactly one way.

    run:     gsm8k_kv.py run MODEL KV BOUNDARY N OUT.json [fewshot|chat]
    report:  gsm8k_kv.py report OUT.json [OUT.json ...]

KV is `auto` or a `turboquant_*` preset. BOUNDARY is:

    on          stock behaviour -- first/last 2 layers keep a native cache
    off         no boundary skips at all
    layers:0,1  an explicit set, for isolating which of them matter

`off` and `layers:` work by replacing `TurboQuantConfig.get_boundary_skip_layers`.
That is legitimate rather than a hack: `EngineArgs.create_engine_config` calls it
in *this* process while building the config, before any engine subprocess exists,
so the override lands on the real code path and the engine's own
`kv_cache_dtype_skip_layers` is printed back to prove which layers were skipped.
There is no CLI for this -- that is the point of the experiment.

Sliding-window models need their sliding layers held native (TurboQuant cannot
serve a sliding window); pass them in SKIP_SLIDING as a comma-separated list.

Env: MML (max_model_len), UTIL (gpu_memory_utilization), MAXTOK, SKIP_SLIDING,
SHOTS (few-shot count -- the context-length knob for long-context runs).

Decoding is greedy and, with enforce_eager, bit-reproducible: the same config run
twice scored 1220/1319 both times with zero per-item disagreement. So per-item
flips between two configurations are attributable to the KV cache and nothing
else, which is what makes the paired test in `report` meaningful.
"""

import glob
import json
import math
import os
import re
import sys

# Native bf16 K+V bytes per head per token is 4 * head_dim; a TurboQuant slot is
# head_dim + 6 at 4 bits (see TurboQuantConfig.slot_size_aligned).
TQ_SLOT = {"4bit_nc": 134, "k3v4_nc": 118, "3bit_nc": 102}  # head_dim 128


def override_boundary(boundary):
    """Force TurboQuant's boundary skip layers before the engine config is built.

    ``EngineArgs.create_engine_config`` calls ``get_boundary_skip_layers`` in this
    process, so replacing it here lands on the real code path. Must be called
    before constructing the LLM. "on" leaves stock behaviour alone.
    """
    if boundary == "on":
        return
    from vllm.model_executor.layers.quantization.turboquant.config import (
        TurboQuantConfig,
    )

    forced = [] if boundary == "off" else boundary.split(":", 1)[1].split(",")
    forced = [x for x in forced if x]
    TurboQuantConfig.get_boundary_skip_layers = staticmethod(
        lambda mc, n=2, _l=forced: list(_l)
    )


def build_llm(model, kv):
    """Engine with the requested KV dtype; prints back the effective skip list."""
    from vllm import LLM

    kwargs = dict(
        model=model,
        max_model_len=int(os.environ.get("MML", 2048)),
        gpu_memory_utilization=float(os.environ.get("UTIL", 0.60)),
        enforce_eager=True,
        trust_remote_code=True,
    )
    if os.environ.get("SKIP_SLIDING"):
        kwargs["kv_cache_dtype_skip_layers"] = os.environ["SKIP_SLIDING"].split(",")
    if kv != "auto":
        kwargs["kv_cache_dtype"] = kv
    llm = LLM(**kwargs)
    skips = llm.llm_engine.vllm_config.cache_config.kv_cache_dtype_skip_layers
    print("EFFECTIVE SKIP LAYERS:", skips, flush=True)
    return llm, list(skips)


def run(model, kv, boundary, n, outp, mode):
    override_boundary(boundary)

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    # Shot count is the context-length knob: task, metric and scoring stay fixed
    # while the prompt grows, so long-context runs stay comparable to the 5-shot
    # ones. Every problem shares the prefix, so prefix caching pays the long
    # prefill once. ~192 tokens/shot on GSM8K: 42 shots ~ 8K, 175 ~ 32K.
    n_shots = int(os.environ.get("SHOTS", 5))
    shots = "".join(
        f"Question: {train[i]['question']}\nAnswer: {train[i]['answer']}\n\n"
        for i in range(n_shots)
    )
    gold = [r["answer"].split("####")[-1].strip().replace(",", "") for r in test]

    llm, skips = build_llm(model, kv)
    shot_tokens = len(llm.get_tokenizer().encode(shots)) if n_shots else 0
    print(f"PROMPT CONTEXT: {n_shots} shots = {shot_tokens} tokens", flush=True)

    if mode == "chat":
        # Few-shot completion is the lm-eval `gsm8k` shape and the right default;
        # chat mode exists for instruct models with no usable base behaviour.
        sp = SamplingParams(max_tokens=int(os.environ.get("MAXTOK", 512)), temperature=0)
        outs = llm.chat(
            [
                [
                    {
                        "role": "user",
                        "content": r["question"] + "\n\nSolve step by step, then give "
                        "the final numeric answer on its own line after ####.",
                    }
                ]
                for r in test
            ],
            sp,
        )
    else:
        sp = SamplingParams(
            max_tokens=320, temperature=0, stop=["Question:", "\n\n\n"]
        )
        outs = llm.generate(
            [shots + f"Question: {r['question']}\nAnswer:" for r in test], sp
        )

    num = re.compile(r"-?\d[\d,]*\.?\d*")
    items = []
    for o, g in zip(outs, gold):
        text = o.outputs[0].text
        tail = text.split("####")[-1] if "####" in text else text
        found = num.findall(tail.replace(",", ""))
        pred = found[-1].rstrip(".") if found else None
        try:
            ok = pred is not None and abs(float(pred) - float(g)) < 1e-4
        except ValueError:
            ok = False
        items.append(int(ok))

    res = dict(
        model=model, kv=kv, boundary=boundary, n=n, correct=sum(items),
        acc=sum(items) / n, skip_layers=list(skips), items=items,
        n_shots=n_shots, shot_tokens=shot_tokens, mode=mode,
    )
    json.dump(res, open(outp, "w"), indent=1)
    print("RESULT", json.dumps({k: v for k, v in res.items() if k != "items"}), flush=True)
    sys.stdout.flush()
    os._exit(0)  # vLLM can hang on interpreter shutdown; the result is on disk


def mcnemar(a, b):
    """Two-sided exact McNemar over per-item correctness. Paired: same problems,
    deterministic decoding, so discordance is the KV change and nothing else."""
    lost = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    gained = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n = lost + gained
    if not n:
        return lost, gained, 1.0
    p = 2 * sum(math.comb(n, k) for k in range(min(lost, gained) + 1)) / 2**n
    return lost, gained, min(1.0, p)


def report(paths):
    rs = [json.load(open(p)) for p in paths]
    print(f"{'preset':10} {'native layers':16} {'n':>5} {'correct':>8} {'acc':>8}")
    for r in sorted(rs, key=lambda r: (r["kv"], len(r["skip_layers"]))):
        st = "{" + ",".join(sorted(r["skip_layers"], key=int)) + "}"
        print(f"{r['kv'].replace('turboquant_',''):10} {st:16} "
              f"{r['n']:5} {r['correct']:8} {r['acc']:7.2%}")
    print(f"\n{'A':28} {'B':28} {'dacc':>7} {'lost':>5} {'gain':>5} {'p':>9}")
    for i, a in enumerate(rs):
        for b in rs[i + 1:]:
            if a["n"] != b["n"]:
                continue
            lost, gained, p = mcnemar(a["items"], b["items"])
            na = f"{a['kv'].replace('turboquant_','')}/{a['boundary']}"
            nb = f"{b['kv'].replace('turboquant_','')}/{b['boundary']}"
            print(f"{na:28} {nb:28} {b['acc']-a['acc']:+7.2%} "
                  f"{lost:5} {gained:5} {p:9.2e}")


if __name__ == "__main__":
    if sys.argv[1] == "report":
        paths = [q for p in sys.argv[2:] for q in (glob.glob(p) or [p])]
        report(paths)
    else:
        _, _, model, kv, boundary, n, outp = sys.argv[:7]
        run(model, kv, boundary, int(n), outp,
            sys.argv[7] if len(sys.argv) > 7 else "fewshot")
