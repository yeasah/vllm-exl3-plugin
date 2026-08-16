#!/usr/bin/env python3
"""The gate: does this build still serve what the committed baseline serves?

    bench/run.py list                 what the matrix covers, and why
    bench/run.py check [--tier fast]  compare a fresh capture to the baseline
    bench/run.py bless [--tier fast]  record the current build as the baseline
    bench/run.py capture <entry> OUT  one entry, for hand inspection

Run `check` before and after a vLLM or exllamav3 bump. `bless` only after
reading a `check` failure and deciding the change is intended -- blessing is how
a real regression becomes the new normal, so it is deliberately a separate verb.

## Thresholds

Exact equality is the wrong gate. Benign changes upstream -- kernel selection,
fusion, accumulation order -- move logprobs slightly without changing what the
model does, and a gate that fires on those gets ignored, which is worse than no
gate. The defaults below sit in the gap between the two populations we have
actually measured:

- benign cross-implementation noise on this project is ~0.02-0.03 nats on top-1
  logprobs (native vLLM vs Transformers backend on MiniCPM5-1B and
  Muse-Glimmer; eager vs CUDA graphs at TP=1 on Laguna-XS)
- a real defect is orders of magnitude larger. The dropped MuseGlimmer logit
  transform moved top-1 logprobs by ~15 nats while changing no token at all.

Two floors were measured on this build rather than guessed, and they bracket the
budget:

- **Same build, re-run: exactly 0.0** on both metrics across all 388 scored
  positions of the fast tier. Teacher-forced decoding at fixed context is
  deterministic here, so a `check` that changes nothing reports nothing.
- **Same weights, different kernels: ~0.157 nats / 0.013 KL.** That is
  Qwen3-0.6B eager vs CUDA graphs with the embedding path taken out of the
  picture entirely (`EXL3_DENSE_EMBED=1`), which is the closest available proxy
  for what a benign upstream change does -- same arithmetic, different kernel
  selection and accumulation order.

So `dlogprob_max` at 0.25 sits above the kernel-drift floor and ~60x below the
one real defect we have numbers for, which moved logprobs by ~15 nats.

`argmax_disagreements` and the greedy continuation stay **exact**, and that is a
deliberate choice rather than an oversight: a kernel change large enough to flip
an argmax at fixed context is one a human should look at. It will occasionally
fire on something benign. Firing on something benign and making you read it is
the intended cost; the alternative is a gate that quietly absorbs the next
`embed_norm`.

`weight_bytes` is exact. It is vLLM's own "Model loading took N GiB", and it
does not drift for benign reasons -- if it moves, either the checkpoint changed
or a path like tied-embedding serving stopped working.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXPECTED = os.path.join(HERE, "expected")

sys.path.insert(0, ROOT)
from bench import core, suite  # noqa: E402

DEFAULT_TOLERANCE = {
    "dlogprob_max": 0.25,
    "kl_max": 5e-2,
    "argmax_disagreements": 0,
}

#: vLLM reports this once per load, from model_runner. Two decimal places in
#: GiB is ~10 MiB of resolution, which is far finer than the ~1 GiB regression
#: this exists to catch.
WEIGHT_RE = re.compile(r"Model loading took ([\d.]+) GiB")


def run_entry(entry: suite.Entry, out_path: str, timeout: int) -> dict:
    """Capture one entry in its own process, returning the measurement."""
    cmd = [sys.executable, os.path.join(HERE, "capture.py"), entry.name,
           "--out", out_path]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    print(f"  -- {entry.label}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, env=env, timeout=timeout,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    if proc.returncode != 0 or "BENCH_CAPTURE_OK" not in proc.stdout:
        # The whole log, not a tail. vLLM reports an engine-core failure in the
        # parent as "See root cause above", where "above" is the child's output
        # hundreds of lines earlier -- a tail truncates exactly the line needed.
        log_path = out_path + ".log"
        with open(log_path, "w") as f:
            f.write(proc.stdout)
        cause = [ln for ln in proc.stdout.splitlines()
                 if "ERROR" in ln and "core.py" in ln]
        detail = "\n".join(cause[-12:]) if cause else \
            "\n".join(proc.stdout.strip().splitlines()[-15:])
        raise SystemExit(f"capture failed for {entry.name!r} "
                         f"(exit {proc.returncode}); full log at {log_path}\n"
                         f"{detail}")

    data = json.load(open(out_path))
    found = WEIGHT_RE.findall(proc.stdout)
    # Under TP there is one line per worker; they are shards of one model, so
    # the total is what corresponds to the single-GPU number.
    data["weight_gib"] = round(sum(float(v) for v in found), 2) if found else None
    if data["weight_gib"] is None:
        print("     ! no weight line found; weight gate inactive for this entry")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=1)
    return data


def check_entry(entry: suite.Entry, fresh: dict, base: dict) -> list[str]:
    """Threshold the fresh capture against the baseline. Returns failures."""
    tol = {**DEFAULT_TOLERANCE, **entry.tolerance}
    failures = []

    if fresh.get("weight_gib") is not None and base.get("weight_gib") is not None:
        if fresh["weight_gib"] != base["weight_gib"]:
            failures.append(
                f"weight bytes {base['weight_gib']} -> {fresh['weight_gib']} GiB")

    for i, (pf, pb) in enumerate(zip(fresh["prompts"], base["prompts"])):
        m = core.compare_prompt(pb, pf)
        if not m["comparable"]:
            failures.append(f"prompt {i}: token ids differ from baseline "
                            f"(tokenizer or template changed)")
            continue
        if m["argmax_disagreements"] > tol["argmax_disagreements"]:
            failures.append(
                f"prompt {i}: {m['argmax_disagreements']}/{m['positions']} "
                f"argmax disagreements")
        if not m["generated_match"]:
            failures.append(f"prompt {i}: greedy continuation changed")
        if m["dlogprob_max"] > tol["dlogprob_max"]:
            failures.append(
                f"prompt {i}: |dlogprob| max {m['dlogprob_max']:.3e} "
                f"> {tol['dlogprob_max']:.3e}")
        if m["kl_max"] > tol["kl_max"]:
            failures.append(
                f"prompt {i}: KL max {m['kl_max']:.3e} > {tol['kl_max']:.3e}")
        print(f"     prompt {i}: {m['positions']} pos, "
              f"argmax {m['argmax_disagreements']}, "
              f"|dlogprob| max {m['dlogprob_max']:.3e}, "
              f"KL max {m['kl_max']:.3e}, "
              f"greedy {'ok' if m['generated_match'] else 'CHANGED'}")
    return failures


def cmd_list(args) -> int:
    for e in suite.by_tier(args.tier):
        print(f"{e.name}  [{e.tier}]")
        print(f"    {e.model}@{e.revision}  impl={e.model_impl} "
              f"{'eager' if e.enforce_eager else 'graphs'} tp={e.tensor_parallel_size}")
        print(f"    {e.exercises}")
    return 0


def cmd_capture(args) -> int:
    run_entry(suite.by_name(args.entry), args.out, args.timeout)
    print(f"wrote {args.out}")
    return 0


def cmd_bless(args) -> int:
    os.makedirs(EXPECTED, exist_ok=True)
    blessed = 0
    for e in suite.by_tier(args.tier):
        if e.known_broken:
            print(f"  -- {e.label}\n     ! known broken, not blessed: "
                  f"{e.known_broken.splitlines()[0]}")
            continue
        run_entry(e, os.path.join(EXPECTED, f"{e.name}.json"), args.timeout)
        blessed += 1
    print(f"\nblessed {blessed} entries into {EXPECTED}")
    print("review the diff before committing -- blessing a real regression "
          "is how a gate stops working")
    return 0


def cmd_check(args) -> int:
    import tempfile

    failed = {}
    known = []
    entries = suite.by_tier(args.tier)
    with tempfile.TemporaryDirectory() as tmp:
        for e in entries:
            if e.known_broken:
                # Still run it: the cheapest way to learn a known defect is
                # fixed is for its entry to stop failing.
                try:
                    run_entry(e, os.path.join(tmp, f"{e.name}.json"), args.timeout)
                except SystemExit:
                    known.append(e.name)
                    print(f"     known broken, as expected")
                    continue
                failed[e.name] = ["known_broken entry now captures cleanly -- "
                                  "clear known_broken and bless it"]
                continue
            baseline = os.path.join(EXPECTED, f"{e.name}.json")
            if not os.path.exists(baseline):
                print(f"  -- {e.label}\n     ! no baseline; run bless")
                failed[e.name] = ["no baseline recorded"]
                continue
            fresh = run_entry(e, os.path.join(tmp, f"{e.name}.json"), args.timeout)
            problems = check_entry(e, fresh, json.load(open(baseline)))
            if problems:
                failed[e.name] = problems

    print()
    if known:
        print(f"known broken, not gated: {', '.join(known)}")
    if not failed:
        print(f"PASS: {len(entries) - len(known)} entries match baseline")
        return 0
    print(f"FAIL: {len(failed)}/{len(entries)} entries diverged from baseline")
    for name, problems in failed.items():
        print(f"  {name}")
        for p in problems:
            print(f"    - {p}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timeout", type=int, default=1800,
                    help="per-entry seconds; a hung EngineCore is contained here")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list"); p.add_argument("--tier", default="all")
    p.set_defaults(func=cmd_list)
    p = sub.add_parser("check"); p.add_argument("--tier", default="fast")
    p.set_defaults(func=cmd_check)
    p = sub.add_parser("bless"); p.add_argument("--tier", default="fast")
    p.set_defaults(func=cmd_bless)
    p = sub.add_parser("capture"); p.add_argument("entry"); p.add_argument("out")
    p.set_defaults(func=cmd_capture)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
