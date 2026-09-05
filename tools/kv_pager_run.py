#!/usr/bin/env python3
"""Run a real engine with the pager wired end to end.

Three arms, and the middle one is the one that makes the others readable:

    off     no pager at all -- the reference
    full    the entire pager wired up, evicting nothing. Manager publishing,
            worker view, guard, all of it, with a policy that keeps everything
            resident. **Must be bit-identical to `off`.** Any deviation is a
            mechanism bug, and it is the arm that keeps a bug from being read
            later as an accuracy result
    paged   a real budget with the stress policy, which churns residency on
            purpose so that blocks are evicted, copied out, and asked back --
            because recency never fetches and would leave the restore path,
            the entire difference between paging and eviction, unexercised

What `paged` is checked on here is the plumbing rather than the quality: every
restore had a host copy behind it, the guard saw no violation, and the blocks
held stayed bounded. Whether the output is any *good* at a given budget is the
capability measurement, and it needs the control arm above to be meaningful.

Each arm runs in its own process. Not fastidiousness: an engine does not fully
release its KV pool when the object goes away, so a second arm built in the
same process fails on free memory -- and if it did not, the arms would be
sharing an allocator whose state the first one shaped.

    tools/kv_pager_run.py MODEL [--ctx N] [--budget N] [--tokens N]
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blocktable_permute import capture, compare, haystack  # noqa: E402
from kv_pager import WorkerPager, state as pager_state  # noqa: E402
from kv_pager.manager import build_manager_class, make_spec_class, register  # noqa: E402


def patch_spec(budget, sink, policy):
    """Make every full-attention layer ask for a paged spec instead.

    `Attention.get_kv_cache_spec` is where a layer's spec is chosen, so this is
    the one place a plugin has to reach; the spec and manager themselves go in
    through the registry with vLLM unmodified.
    """
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    paged_cls = make_spec_class()
    register(paged_cls, build_manager_class())
    original = Attention.get_kv_cache_spec

    def hooked(self, vllm_config):
        spec = original(self, vllm_config)
        if type(spec) is not FullAttentionSpec:
            return spec               # sliding window and friends are not ours
        common = {f.name: getattr(spec, f.name) for f in fields(spec)}
        return paged_cls(**common, budget_blocks=budget, sink_blocks=sink,
                         policy_name=policy)

    Attention.get_kv_cache_spec = hooked
    return original


def build(model, args, budget, policy):
    from vllm import LLM

    pager_state.reset()
    restore = None
    if budget is not None:
        restore = patch_spec(budget, args.sink, policy)
    llm = LLM(model=model, max_model_len=args.ctx + args.tokens + 64,
              gpu_memory_utilization=args.util, enforce_eager=True,
              enable_prefix_caching=False, max_num_seqs=1,
              trust_remote_code=True)
    return llm, restore


def run_arm(model, args, budget, policy, prompts, params):
    llm, _ = build(model, args, budget, policy)
    pager = None
    if budget is not None:
        scheduler = llm.llm_engine.engine_core.engine_core.scheduler
        pager = WorkerPager(host_slots=args.host_slots, scheduler=scheduler)
        pager.install()
    outs = llm.generate(prompts, params)
    result = {"reqs": [capture(o) for o in outs]}
    if pager is not None:
        result["pager"] = pager.summary()
    return result


ARMS = {"off": (None, None), "full": (0, "full"), "paged": (None, "stress")}


def one_arm(args):
    """Run a single arm in this process and write it out."""
    import json

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    budget, policy = ARMS[args.arm]
    if args.arm == "paged":
        budget, policy = args.budget, args.policy
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = haystack(tok, args.ctx)
    prompts = [{"prompt_token_ids": ids[: args.ctx - 7]}]
    params = SamplingParams(temperature=0.0, max_tokens=args.tokens,
                            logprobs=20, ignore_eos=True)
    result = run_arm(args.model, args, budget, policy, prompts, params)
    with open(args.out, "w") as f:
        json.dump(result, f)


def main():
    import json
    import subprocess
    import tempfile

    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--sink", type=int, default=2)
    ap.add_argument("--host-slots", type=int, default=512)
    ap.add_argument("--util", type=float, default=0.55)
    ap.add_argument("--policy", default="stress")
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.arm:
        return one_arm(args) or 0

    arms = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("off", "full", "paged"):
            path = os.path.join(tmp, f"{name}.json")
            print(f"\n=== arm {name} ===", flush=True)
            cmd = [sys.executable, "-u", __file__, args.model,
                   "--ctx", str(args.ctx), "--tokens", str(args.tokens),
                   "--budget", str(args.budget), "--sink", str(args.sink),
                   "--host-slots", str(args.host_slots),
                   "--util", str(args.util), "--policy", args.policy,
                   "--arm", name, "--out", path]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(proc.stdout[-2000:])
                print(proc.stderr[-3000:])
                raise SystemExit(f"arm {name} failed ({proc.returncode})")
            with open(path) as f:
                arms[name] = json.load(f)

    print(f"\n{args.model}  ctx={args.ctx}  budget={args.budget} blocks")
    ref = arms["off"]["reqs"][0]
    ok = True
    for name in ("full", "paged"):
        arm = arms[name]
        m = compare(ref, arm["reqs"][0])
        exact = m["ids_match"] and m["kl_max"] == 0.0 and m["dlogprob_max"] == 0.0
        tag = ("bit-identical to off" if exact else
               "tokens identical" if m["ids_match"] else
               f"diverges at step {m['first_divergence']}")
        p = arm.get("pager", {})
        g = p.get("guard", {})
        print(f"  {name:6s} {tag}")
        print(f"         copied out {p.get('copied_out', 0)}, in "
              f"{p.get('copied_in', 0)}, restores with no host copy "
              f"{p.get('missing_host_copy', 0)}, clock mismatches "
              f"{p.get('clock_mismatch', 0)}")
        print(f"         guard: {g.get('violations', 0)} violation(s) over "
              f"{g.get('steps_checked', 0)} step(s), active "
              f"{g.get('active_checks')}")
        if name == "full" and not exact:
            ok = False
            print("         [!] the control arm is not bit-identical: the "
                  "wiring changes the result before any block is dropped")
        if name == "paged":
            if p.get("copied_out", 0) == 0 or p.get("copied_in", 0) == 0:
                ok = False
                print("         [!] nothing moved; the transport is untested")
            if p.get("missing_host_copy", 0):
                ok = False
                print("         [!] a block was restored with nothing behind "
                      "it -- the model read whatever the pool left there")
        if g.get("violations"):
            ok = False
            for line in g.get("first", []):
                print(f"         [!] {line}")
    print(f"\n  {'plumbing holds' if ok else 'PLUMBING IS BROKEN'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
