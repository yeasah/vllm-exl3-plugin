#!/usr/bin/env python3
"""Does a paged request still answer the question?

Everything measured so far says the machinery is correct. None of it says the
output is any good, and a bad number on its own cannot say *why* -- a pager
that retrieves nothing at a tight budget could be a broken mechanism or an
honest policy failure, and those want completely different work next.

So four arms, chosen to make a number attributable:

    off       no pager                     the reference answer
    full      pager, evicting nothing      must be bit-identical to off, or
                                           every other number here is void
    recency   sinks + the newest blocks    the shippable baseline, and the one
                                           that structurally cannot fetch: its
                                           window only slides forward, so a
                                           needle behind it is gone
    oracle    recency + the needle's block a ceiling, told the answer. If this
                                           retrieves where recency does not,
                                           the mechanism is sound and the gap
                                           is policy

A magic number is planted at a known token offset, so the block holding it is
known and the oracle can be told about it. The needle sits at an exact block
boundary: one straddling two blocks would survive the loss of either, and the
arms would stop differing in exactly one thing.

    tools/kv_pager_quality.py MODEL [--ctx N] [--budget N] [--needle-block N]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blocktable_evict import answered, needle_prompt  # noqa: E402
from blocktable_permute import capture, compare, haystack  # noqa: E402
from kv_pager import WorkerPager, state as pager_state  # noqa: E402
from kv_pager.policy import Oracle, OracleLate  # noqa: E402
from kv_pager_run import patch_spec  # noqa: E402

VALUE = 918273
ARMS = ("off", "full", "recency", "oracle", "oracle_late")


def one_arm(args):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = haystack(tok, args.ctx)
    prompt_ids, first, last = needle_prompt(tok, ids, args.block_size,
                                            args.needle_block, VALUE)
    if first != last:
        raise SystemExit(f"needle spans blocks {first}..{last}; widen it")

    pager_state.reset()
    if args.arm != "off":
        budget = 0 if args.arm == "full" else args.budget
        policy = "full" if args.arm == "full" else args.arm
        Oracle.must_keep = {first}
        OracleLate.must_keep = {first}
        # Evicted through prefill, wanted back once decoding starts.
        # A few steps into decoding, so the block is evicted first and asked
        # back afterwards. Paging is decode-only, so `after = prompt_len` would
        # keep it from the very first decision and never restore anything.
        OracleLate.after = len(prompt_ids) + 2
        patch_spec(budget, args.sink, policy)

    # Chunked prefill on purpose. With the whole prompt in one chunk the
    # scheduler makes exactly one residency decision before decoding starts, so
    # a block can never be evicted *and* asked back before the answer is
    # generated -- and the answer to a needle question arrives in the first
    # token or two. Chunking gives the round trip somewhere to happen, and it
    # is what a long prompt does anyway.
    llm = LLM(model=args.model, max_model_len=len(prompt_ids) + args.tokens + 64,
              gpu_memory_utilization=args.util, enforce_eager=True,
              enable_prefix_caching=False, max_num_seqs=1,
              max_num_batched_tokens=args.max_batched,
              trust_remote_code=True)
    pager = None
    if args.arm != "off":
        pager = WorkerPager(host_slots=args.host_slots,
                            scheduler=llm.llm_engine.engine_core.engine_core.scheduler)
        pager.install()

    out = llm.generate([{"prompt_token_ids": prompt_ids}],
                       SamplingParams(temperature=0.0, max_tokens=args.tokens,
                                      logprobs=20))[0]
    result = {"req": capture(out), "text": out.outputs[0].text,
              "needle_block": first, "prompt_len": len(prompt_ids)}
    if pager is not None:
        result["pager"] = pager.summary()
    with open(args.out, "w") as f:
        json.dump(result, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=12)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--sink", type=int, default=2)
    ap.add_argument("--needle-block", type=int, default=40)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--host-slots", type=int, default=1024)
    ap.add_argument("--util", type=float, default=0.55)
    ap.add_argument("--max-batched", type=int, default=512)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.arm:
        return one_arm(args) or 0

    arms = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in ARMS:
            path = os.path.join(tmp, f"{name}.json")
            print(f"=== {name} ===", flush=True)
            cmd = [sys.executable, "-u", __file__, args.model,
                   "--ctx", str(args.ctx), "--tokens", str(args.tokens),
                   "--budget", str(args.budget), "--sink", str(args.sink),
                   "--needle-block", str(args.needle_block),
                   "--block-size", str(args.block_size),
                   "--host-slots", str(args.host_slots),
                   "--util", str(args.util),
                   "--max-batched", str(args.max_batched),
                   "--arm", name, "--out", path]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(proc.stdout[-1500:], proc.stderr[-2500:])
                raise SystemExit(f"arm {name} failed")
            with open(path) as f:
                arms[name] = json.load(f)

    ref = arms["off"]
    print(f"\n{args.model}  ctx={args.ctx}  budget={args.budget} of "
          f"{ref['prompt_len'] // args.block_size} blocks  "
          f"needle in block {ref['needle_block']}")
    for name in ARMS:
        a = arms[name]
        hit = answered(a["text"], VALUE)
        m = compare(ref["req"], a["req"])
        exact = m["ids_match"] and m["kl_max"] == 0.0 and m["dlogprob_max"] == 0.0
        p = a.get("pager", {})
        g = p.get("guard", {})
        tag = ("reference" if name == "off" else
               "bit-identical" if exact else
               "tokens identical" if m["ids_match"] else
               f"diverges at step {m['first_divergence']}")
        print(f"  {name:8s} needle {'FOUND' if hit else 'lost '}  {tag:24s}"
              f"  out {p.get('copied_out', 0):4d} in {p.get('copied_in', 0):4d}"
              f"  guard {g.get('violations', 0)}")
        print(f"           {a['text'].strip()[:64]!r}")

    ok_full = (arms["full"]["req"]["ids"] == ref["req"]["ids"])
    if not ok_full:
        print("\n  VOID: the control arm is not identical to the reference")
        return 1
    hits = {n: answered(arms[n]["text"], VALUE) for n in ARMS}
    if not hits["off"]:
        print("\n  inconclusive: the model cannot retrieve this needle even "
              "with the whole context, so no arm below says anything")
    elif hits["oracle"] and not hits["recency"]:
        print("\n  the mechanism can deliver at this budget and recency cannot "
              "-- the gap is policy, which is the phase to work on next")
        late = arms.get("oracle_late", {}).get("pager", {})
        moved, unbacked = late.get("copied_in", 0), late.get("missing_host_copy", 1)
        if moved and not unbacked:
            print(f"  the restore path ran ({moved} block(s)) and every "
                  f"restored block was backed by a host copy. It lands a few "
                  f"decode steps in, and a needle answer arrives in the first "
                  f"token or two, so this arm cannot show the round trip "
                  f"*changing* an answer -- that proof is kv_roundtrip.py, "
                  f"where a destroyed and restored block gives bit-identical "
                  f"output through the model")
        elif not moved:
            print("  note: oracle_late restored nothing, so the restore path "
                  "went unexercised here")
        else:
            print(f"  [!] {unbacked} restore(s) had no host copy behind them: "
                  f"the model read whatever the pool had left in that block")
    elif hits["oracle"] and hits["recency"]:
        print("\n  this budget is not tight enough to separate the policies")
    elif not hits["oracle"]:
        print("\n  even told which block holds the answer, a budget this tight "
              "loses it -- the budget, not the policy, is the binding "
              "constraint here")
    return 0


if __name__ == "__main__":
    sys.exit(main())
