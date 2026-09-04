#!/usr/bin/env python3
"""The gate: does this build still serve what the committed baseline serves?

    bench/run.py list                 what the matrix covers, and why
    bench/run.py check [--tier fast]  compare a fresh capture to the baseline
    bench/run.py bless [--tier fast]  record the current build as the baseline
    bench/run.py capture <entry> OUT  one entry, for hand inspection
    bench/run.py verify               do the baselines agree on what built them
    bench/run.py perf-check  --platform TAG   throughput vs this machine
    bench/run.py perf-bless  --platform TAG   record this machine's throughput

`check` reads *what* is served and is blind to how fast; `perf-check` is the
other half, and a bump wants both.

The two are stored differently on purpose. A correctness baseline is a fact about
this codebase and its dependencies, so it lives flat in `expected/` and should
hold anywhere. A perf baseline is that *plus a machine*, so it lives under
`expected/perf/<platform>/` and is never compared across platforms -- see
`platform_tag` below for why identification is the operator's job.

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
import shutil
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

#: Which attention backend vLLM *chose*. Nothing in an entry names it: it is
#: derived from head dim, dtype, KV dtype, sliding window and what is installed,
#: so it can change silently under a bump or an environment difference and take
#: the meaning of the entry with it. `qwen3.8-27B ... MTP fp8` is the worked
#: example -- it served once, then stopped, because selection landed on
#: FlashInfer and FlashInfer was unreachable. Recorded, and compared.
BACKEND_RE = re.compile(
    r"Using (?:AttentionBackendEnum\.)?(\w+)(?: attention)? backend"
)

#: What was left for the KV cache after weights and activations. Reported rather
#: than gated: it is a graded VRAM-efficiency signal where `weight_gib` is exact,
#: and it is the number that moves when something upstream quietly inflates.
KVMEM_RE = re.compile(r"Available KV cache memory: ([-\d.]+) GiB")


#: Standard CUDA toolkit locations, searched only when `nvcc` is not already on
#: PATH. FlashInfer needs either the `flashinfer_cubin` package or `nvcc` to JIT,
#: and vLLM selects FlashInfer for some (model, KV dtype) combinations -- so
#: without this an entry's fate depends on the operator's shell rather than on
#: the build. The `qwen3.8-27B ... MTP fp8` entry is the case that found it.
_CUDA_BIN_CANDIDATES = ("/usr/local/cuda/bin", "/opt/cuda/bin")


def _env_with_nvcc() -> dict:
    """Child environment, with `nvcc` made discoverable if it is installed."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if shutil.which("nvcc") is not None:
        return env
    for cand in _CUDA_BIN_CANDIDATES:
        if os.path.exists(os.path.join(cand, "nvcc")):
            env["PATH"] = cand + os.pathsep + env.get("PATH", "")
            return env
    return env


def run_entry(entry: suite.Entry, out_path: str, timeout: int) -> dict:
    """Capture one entry in its own process, returning the measurement."""
    cmd = [sys.executable, os.path.join(HERE, "capture.py"), entry.name,
           "--out", out_path]
    env = _env_with_nvcc()
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
    backends = sorted(set(BACKEND_RE.findall(proc.stdout)))
    data["attn_backend"] = backends[0] if len(backends) == 1 else (backends or None)
    kvmem = KVMEM_RE.findall(proc.stdout)
    data["kv_cache_gib"] = round(float(kvmem[-1]), 2) if kvmem else None
    with open(out_path, "w") as f:
        json.dump(data, f, indent=1)
    return data


def check_entry(entry: suite.Entry, fresh: dict, base: dict) -> list[str]:
    """Threshold the fresh capture against the baseline. Returns failures."""
    tol = {**DEFAULT_TOLERANCE, **entry.tolerance}
    failures = []

    # Reported, never failed on. Different hardware is a legitimate thing to run
    # a correctness check on -- that is rather the point of a portable baseline
    # -- but it changes how a failure should be read, so it is surfaced first.
    core.report_environment_diff(base.get("environment", {}),
                                 fresh.get("environment", {}))

    # A fixture entry serves a checkpoint this run derived rather than one the
    # Hub pinned, so "the checkpoint changed" is a live possibility that it is
    # not for other entries -- and it looks exactly like a model regression if
    # not named. Reported before the logprob comparison for that reason; it is
    # the thing to read first when a fixture entry fails.
    fb, ff = base.get("fixture"), fresh.get("fixture")
    if (fb or ff) and fb != ff:
        if fb and ff and fb.get("digest") != ff.get("digest"):
            failures.append(
                f"fixture {ff.get('kind')} content changed: "
                f"{fb.get('digest')} -> {ff.get('digest')} (the derived "
                f"checkpoint differs, so every difference below follows from "
                f"it; re-bless if the producer change was intended)")
        else:
            failures.append(f"fixture record changed: {fb} -> {ff}")

    # The chosen backend is not something an entry asks for -- it is derived.
    # A change means this entry is measuring a different code path than the
    # baseline did, which invalidates the comparison rather than failing it on
    # numbers, so it is gated rather than reported.
    if base.get("attn_backend") and fresh.get("attn_backend") != base.get("attn_backend"):
        failures.append(
            f"attention backend {base['attn_backend']} -> {fresh['attn_backend']} "
            f"(selection is derived from head dim, dtypes, sliding window and "
            f"what is installed; this entry is no longer testing the same path)")
    # Reported, never failed on: a graded VRAM signal, noisy by nature.
    if base.get("kv_cache_gib") is not None and fresh.get("kv_cache_gib") is not None:
        delta = fresh["kv_cache_gib"] - base["kv_cache_gib"]
        if abs(delta) >= 0.05:
            print(f"     kv cache headroom {base['kv_cache_gib']:.2f} -> "
                  f"{fresh['kv_cache_gib']:.2f} GiB ({delta:+.2f})")

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


#: One-sided, and generous. Throughput is not deterministic the way logprobs
#: are, but on the dev card it is far steadier than expected: repeated runs
#: inside one process spread ~1%, and medians *across* processes spread ~0.5%
#: (decode 2750.9 / 2738.9 / 2741.2 tok/s on three fresh runs). So -10% is about
#: 20x the observed noise while still catching anything worth the name.
#:
#: Only regressions fail. A large speedup with correct logits is good news, and
#: work being silently skipped is what the correctness gate is for.
PERF_REGRESSION_PCT = 10.0

PERF_EXPECTED = os.path.join(HERE, "expected", "perf")

#: The full package list behind `environment()`'s `pkg.digest`, written by
#: `bless`. A digest says something moved; this says what.
MANIFEST = os.path.join(HERE, "expected", "environment.txt")


def platform_tag(args) -> str:
    """The operator's name for the machine, and it is deliberately mandatory.

    A correctness baseline is a fact about this codebase and its dependencies. A
    perf baseline is that *plus a machine*, and the machine cannot be identified
    from inside it -- firmware, thermals, host contention and the hypervisor are
    all invisible and all move throughput. Trying to fingerprint it automatically
    would produce a key that looks authoritative and is not.

    So identification is the operator's job, which also makes it the operator's
    choice how coarse to be: `rtx5070ti-dev` if one box, `vast-8x3090-a` if
    several rentals need telling apart. There is no default on purpose. A
    baseline silently filed under "default" and later compared against a
    different machine is worse than having no baseline, because it reports a
    regression that is really a change of computer.
    """
    tag = args.platform or os.environ.get("BENCH_PLATFORM")
    if not tag:
        env = core.environment()
        guess = (env.get("gpu") or "unknown").lower().replace(" ", "-")
        raise SystemExit(
            "perf baselines are per-machine, so this needs a platform tag.\n"
            f"  --platform <tag>   or   BENCH_PLATFORM=<tag>\n"
            f"  this box looks like: {guess}  (gpu_count={env.get('gpu_count')}, "
            f"driver={env.get('driver')})\n"
            "The tag is yours to choose; it only has to mean the same machine "
            "next time.")
    return tag.strip().replace("/", "-").replace(" ", "-")


def run_perf_entry(entry: suite.Entry, out_path: str, timeout: int,
                   reps: int, tag: str) -> dict:
    cmd = [sys.executable, os.path.join(HERE, "perf.py"), entry.name,
           "--out", out_path, "--reps", str(reps)]
    # Forwarded explicitly: the tag may have come from --platform, which the
    # child has no other way to see.
    env = dict(_env_with_nvcc(), BENCH_PLATFORM=tag)
    print(f"  -- {entry.label}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, env=env, timeout=timeout,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    if proc.returncode != 0 or "BENCH_PERF_OK" not in proc.stdout:
        log_path = out_path + ".log"
        with open(log_path, "w") as f:
            f.write(proc.stdout)
        cause = [ln for ln in proc.stdout.splitlines()
                 if "ERROR" in ln and "core.py" in ln]
        detail = "\n".join(cause[-12:]) if cause else \
            "\n".join(proc.stdout.strip().splitlines()[-15:])
        raise SystemExit(f"perf failed for {entry.name!r} "
                         f"(exit {proc.returncode}); full log at {log_path}\n"
                         f"{detail}")
    data = json.load(open(out_path))
    print(f"     decode {data['decode']:.0f} tok/s (+-{data['decode_spread_pct']}%)"
          f"   prefill {data['prefill']:.0f} tok/s "
          f"(+-{data['prefill_spread_pct']}%)")
    return data


def cmd_perf_bless(args) -> int:
    warn_if_plugin_dirty()
    tag = platform_tag(args)
    target = os.path.join(PERF_EXPECTED, tag)
    os.makedirs(target, exist_ok=True)
    entries = suite.perf_by_tier(args.tier)
    for e in entries:
        run_perf_entry(e, os.path.join(target, f"{e.name}.json"),
                       args.timeout, args.reps, tag)
    print(f"\nblessed {len(entries)} perf entries into {target}")
    print(f"these numbers describe platform {tag!r} and nothing else; "
          "another machine needs its own bless")
    return 0


def cmd_perf_check(args) -> int:
    import tempfile

    tag = platform_tag(args)
    target = os.path.join(PERF_EXPECTED, tag)
    if not os.path.isdir(target):
        known = sorted(os.listdir(PERF_EXPECTED)) if os.path.isdir(PERF_EXPECTED) else []
        print(f"no perf baselines for platform {tag!r}.")
        print(f"  known platforms: {', '.join(known) or '(none)'}")
        print("  run perf-bless on this machine first -- comparing against "
              "another machine's numbers would measure the hardware, not the build.")
        return 1

    failed = {}
    entries = suite.perf_by_tier(args.tier)
    with tempfile.TemporaryDirectory() as tmp:
        for e in entries:
            baseline_path = os.path.join(target, f"{e.name}.json")
            if not os.path.exists(baseline_path):
                print(f"  -- {e.label}\n     ! no baseline for {tag!r}; "
                      "run perf-bless")
                failed[e.name] = [f"no baseline recorded for platform {tag!r}"]
                continue
            fresh = run_perf_entry(e, os.path.join(tmp, f"{e.name}.json"),
                                   args.timeout, args.reps, tag)
            base = json.load(open(baseline_path))
            problems = []
            # The tag says these are the same machine. If the machine disagrees,
            # say so -- a mislabelled baseline turns a hardware change into a
            # phantom regression, which is the failure this scoping prevents.
            core.report_environment_diff(base.get("environment", {}),
                                         fresh.get("environment", {}))
            # As for correctness entries: a fixture entry serves a checkpoint
            # this run derived, so "the checkpoint changed" has to be separable
            # from "throughput regressed".
            fb, ff = base.get("fixture"), fresh.get("fixture")
            if (fb or ff) and fb != ff:
                problems.append(f"fixture record changed: {fb} -> {ff}")
            for metric in ("decode", "prefill"):
                delta = (fresh[metric] - base[metric]) / base[metric] * 100
                verdict = "ok"
                if delta < -PERF_REGRESSION_PCT:
                    verdict = "REGRESSION"
                    problems.append(
                        f"{metric} {base[metric]:.0f} -> {fresh[metric]:.0f} "
                        f"tok/s ({delta:+.1f}%)")
                print(f"     {metric:8} {delta:+6.1f}% vs baseline  {verdict}")
            if problems:
                failed[e.name] = problems

    print()
    if not failed:
        print(f"PASS: {len(entries)} perf entries within "
              f"{PERF_REGRESSION_PCT:.0f}% of baseline")
        return 0
    print(f"FAIL: {len(failed)}/{len(entries)} perf entries regressed")
    for name, problems in failed.items():
        print(f"  {name}")
        for p in problems:
            print(f"    - {p}")
    return 1


def environment_drift() -> tuple[bool, list[str]]:
    """Compare the live environment against the blessed manifest.

    Returns (differs, lines). Used by `env` to report and by `check` to decide
    whether to refuse -- the same comparison either way, so the two can never
    disagree about what "the environment moved" means.
    """
    if not os.path.exists(MANIFEST):
        return False, [f"no blessed manifest at {MANIFEST}; run bless"]
    with open(MANIFEST) as f:
        blessed = [ln.strip() for ln in f if ln.strip()]
    live = core.package_manifest()

    def as_map(rows):
        out = {}
        for r in rows:
            name, _, ver = r.partition("==")
            out[name.lower()] = ver
        return out

    b, lv = as_map(blessed), as_map(live)
    lines = []
    for name in sorted(set(b) | set(lv)):
        ov, nv = b.get(name), lv.get(name)
        if ov == nv:
            continue
        if ov is None:
            lines.append(f"  + {name} {nv}  (added since bless)")
        elif nv is None:
            lines.append(f"  - {name} {ov}  (removed since bless)")
        else:
            lines.append(f"  ~ {name} {ov} -> {nv}")
    return bool(lines), lines


def cmd_env(args) -> int:
    """What has moved in the environment since the baselines were blessed."""
    differs, lines = environment_drift()
    if not os.path.exists(MANIFEST):
        print(lines[0])
        return 1
    if not differs:
        print(f"environment matches the blessed manifest "
              f"({len(core.package_manifest())} packages)")
        return 0
    print(f"environment differs from {MANIFEST}:\n")
    for ln in lines:
        print(ln)
    print(f"\n{len(lines)} package(s) differ. This is reported, not fatal: a "
          f"check will still run and will\nsay so. Use `check --strict-env` to "
          f"refuse instead, or re-bless to accept the\ncurrent environment as "
          f"the reference.")
    return 1


def cmd_verify(args) -> int:
    """Do all baselines in a set agree about what produced them?

    A baseline set is meant to be one snapshot of one build. Nothing enforces
    that: `bless` writes entries one at a time over the better part of an hour,
    and anything that changes underneath it silently splits the set.

    This exists because the per-entry warning could not catch the real case. The
    suite dirtied its *own* tree -- baselines live inside the repo whose
    provenance is recorded, so the second entry saw the first one's output --
    and every entry disagreed with every other about the state it came from. No
    amount of operator discipline would have prevented that, and only comparing
    the finished set across entries revealed it.
    """
    import glob
    from collections import defaultdict

    fields = ("src.plugin", "src.vllm", "src.exllamav3")
    ok = True

    def check(paths, label) -> bool:
        if not paths:
            print(f"{label}: no baselines")
            return True
        groups, missing = defaultdict(list), []
        for p in sorted(paths):
            env = (json.load(open(p)).get("environment") or {})
            if not env:
                missing.append(os.path.basename(p))
                continue
            key = tuple(json.dumps(env.get(f), sort_keys=True) for f in fields)
            groups[key].append(os.path.basename(p))
        good = not missing and len(groups) == 1
        extra = f" + {len(missing)} with none" if missing else ""
        print(f"\n{label}: {len(paths)} baselines, {len(groups)} distinct "
              f"provenance(s){extra}  -> {'OK' if good else 'MIXED'}")
        for name in missing:
            print(f"   !! no provenance recorded: {name}")
        for key, names in groups.items():
            env = {f: json.loads(v) for f, v in zip(fields, key)}
            print("   " + "  ".join(
                f"{f.removeprefix('src.')}={(env[f] or {}).get('describe')}"
                f"/dirty={(env[f] or {}).get('dirty_files')}" for f in fields))
            if len(groups) > 1:
                for n in names:
                    print(f"      - {n}")
        return good

    ok &= check(glob.glob(os.path.join(EXPECTED, "*.json")), "correctness")
    for d in sorted(glob.glob(os.path.join(PERF_EXPECTED, "*"))):
        if os.path.isdir(d):
            ok &= check(glob.glob(os.path.join(d, "*.json")),
                        f"perf [{os.path.basename(d)}]")

    print("\n" + ("ALL CONSISTENT" if ok else
                  "INCONSISTENT -- this set is a mixture of builds, so a check "
                  "against it compares against no single thing. Re-bless."))
    return 0 if ok else 1


def cmd_list(args) -> int:
    for e in suite.by_tier(args.tier):
        print(f"{e.name}  [{e.tier}]")
        derived = f" fixture={e.fixture}" if e.fixture else ""
        if e.kv_cache_dtype:
            derived += f" kv={e.kv_cache_dtype}"
        if e.speculative_config:
            derived += f" spec={e.speculative_config.get('method')}"
        if e.language_model_only:
            derived += " lm-only"
        print(f"    {e.model}@{e.revision}  impl={e.model_impl} "
              f"{'eager' if e.enforce_eager else 'graphs'} "
              f"tp={e.tensor_parallel_size}{derived}")
        print(f"    {e.exercises}")
    return 0


def cmd_capture(args) -> int:
    run_entry(suite.by_name(args.entry), args.out, args.timeout)
    print(f"wrote {args.out}")
    return 0


def warn_if_plugin_dirty() -> None:
    """A baseline blessed from a dirty plugin tree records a state that never recurs.

    A dirty vLLM used to be normal here, back when `patches/` lived in its
    working tree; now that it is a pinned submodule, `source_provenance` records
    that separately and a dirty one is an anomaly rather than the baseline. This
    warning is about the plugin's own tree, which is different: `diff_sha` then
    names a working state nobody can return to, and a later `check` cannot tell
    whether it is comparing against committed code or against a half-finished
    edit. Worse, editing during a long `bless` gives *different* entries
    different provenance, which is how this warning came to exist.
    """
    prov = core.source_provenance(ROOT, core._PROVENANCE_EXCLUDE) or {}
    if prov.get("dirty_files"):
        print(f"  ! plugin tree is dirty ({prov['dirty_files']} files, "
              f"diff_sha {prov.get('diff_sha')}).")
        print("    Baselines will record a working state that cannot be "
              "recovered later; commit first if these are meant to last.")
        print("    Do not edit the tree while this runs -- entries blessed "
              "before and after would disagree about what produced them.")


def cmd_bless(args) -> int:
    warn_if_plugin_dirty()
    os.makedirs(EXPECTED, exist_ok=True)
    blessed = 0
    for e in suite.by_tier(args.tier):
        if e.known_broken:
            print(f"  -- {e.label}\n     ! known broken, not blessed: "
                  f"{e.known_broken.splitlines()[0]}")
            continue
        run_entry(e, os.path.join(EXPECTED, f"{e.name}.json"), args.timeout)
        blessed += 1
    # The manifest behind `environment()`'s pkg.digest: a digest says something
    # moved, this says what. One per bless, since a bless is one snapshot.
    try:
        with open(MANIFEST, "w") as f:
            f.write("\n".join(core.package_manifest()) + "\n")
    except Exception as exc:  # pragma: no cover
        print(f"     ! could not write environment manifest: {exc}")

    print(f"\nblessed {blessed} entries into {EXPECTED}")
    print("review the diff before committing -- blessing a real regression "
          "is how a gate stops working")
    return 0


def cmd_check(args) -> int:
    import tempfile

    # Checked once, before anything loads: the environment is the same for every
    # entry, and discovering it after a 15-minute tier is a waste.
    differs, lines = environment_drift()
    if differs:
        print(f"environment differs from the blessed manifest "
              f"({len(lines)} package(s)):")
        for ln in lines[:8]:
            print(ln)
        if len(lines) > 8:
            print(f"  ... {len(lines) - 8} more; run `bench/run.py env` for all")
        if getattr(args, "strict_env", False):
            print("\nrefusing: --strict-env is set. Restore the environment, or "
                  "re-bless to accept\nthis one as the reference.")
            return 1
        print("  (reported, not fatal -- pass --strict-env to refuse)\n")

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
    p = sub.add_parser("verify")
    p.set_defaults(func=cmd_verify)
    p = sub.add_parser("env")
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("check"); p.add_argument("--tier", default="fast")
    p.add_argument("--strict-env", action="store_true",
                   help="refuse to run if the environment differs from the "
                        "blessed manifest")
    p.set_defaults(func=cmd_check)
    p = sub.add_parser("bless"); p.add_argument("--tier", default="fast")
    p.set_defaults(func=cmd_bless)
    p = sub.add_parser("capture"); p.add_argument("entry"); p.add_argument("out")
    p.set_defaults(func=cmd_capture)
    for name, fn in (("perf-check", cmd_perf_check), ("perf-bless", cmd_perf_bless)):
        p = sub.add_parser(name)
        p.add_argument("--tier", default="fast")
        p.add_argument("--reps", type=int, default=5)
        p.add_argument("--platform", default=None,
                       help="operator's name for this machine; perf baselines "
                            "are per-machine. Or set BENCH_PLATFORM.")
        p.set_defaults(func=fn)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
