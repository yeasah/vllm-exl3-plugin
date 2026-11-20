#!/usr/bin/env python3
"""Probe a model before its ladder: draw noise, recipe gain and YAQA gain at one mid rate, side by side.

Builds, at one rate (default 4.00):
  - N draws of the configuration the ladder would use (sign-draw offsets 0..N-1, EXL3_SEED_IDX_OFFSET),
  - a uniform-bits rung at offset 0 (the recipe's control), only when `measure` has run: with no measurement the draws
    are uniform already (most MoEs), and this rung would be draw 0 again, byte for byte on the same host,
  - with --arm TAG=ARGS (repeatable), a rung converted with `quantize ARGS` at offset 0, named <rate>-probe-TAG, against
    the configuration it departs from: the uniform rung when ARGS has --no-recipe and there is one, draw 0 otherwise.
    Its size is reported beside its gain, since an arm may spend bits outside the rate (`--arm hq2="--hq 2"`),
  - with --yaqa, a rung converted against `yaqa --probe`'s factor at offset 0 (its control is draw 0),
  - one draw at --low-bits (default 2.50, head 5 below 4 as the ladder has it), the ladder's configuration otherwise,
then scores them with one restricted qbench and reports each slice's excess KLD, the draws' spread, and each gain
measured against that spread. A gain smaller than what a redraw does is labelled as such: on Qwen3.6-35B-A3B's swe a
redraw moved a 4.00 by 10-17%, mostly draw x trace interaction (docs/calibration.md, "A draw's swe rank does not
transfer") -- noise an arm's gain has to clear either way.

The low draw is evidence for a sub-3 rung, not a verdict: the ladder starts at 3.00 unless one is chosen. It is reported
against the draws -- each slice's excess KLD as a multiple of theirs, recall kept and confident errors beside theirs --
so the gradient from the probe rate down is in view. One draw: recall varies by a few points between draws at a rate,
so a result near the line is the reviewer's call.

Mechanical by design: it only drives `quant.py quantize`, `yaqa --probe` and `qbench`, and every step is skipped when
its output exists, so a rerun picks up where one stopped. With --yaqa the factor is collected first, on every GPU;
then the quantizes run as one wave, the draws on the first --draws GPUs and the uniform and YAQA rungs on the rest. Run from quantization/:

    python3 probe.py -m openbmb/MiniCPM5-2B --draws 5 --yaqa
"""
import argparse, json, os, re, shlex, subprocess, sys, types

HERE = os.path.dirname(os.path.abspath(__file__))
QUANT = os.path.join(HERE, "quant.py")
LOG = os.devnull            # each quant.py step keeps its own logs
sys.path.insert(0, HERE)
import quant                # its child handling: own sessions, and Ctrl-C passed on to them


def run(cmd, env=None):
    print("=== " + " ".join(cmd), flush=True)
    rc = quant.run_logged(cmd, LOG, {**(env or {}), "PYTHONUNBUFFERED": "1"})   # own session: Ctrl-C is ours to pass on
    if rc:
        raise SystemExit(f"=== failed ({rc}): {' '.join(cmd)}")


def run_lanes(model, lanes):
    """Each lane's steps in turn, the lanes side by side; a step whose output exists is skipped when it is reached.
    One lane failing lets the others finish (their outputs stand for a rerun), then fails the probe."""
    import threading
    failed = []
    def lane(steps):
        try:
            for done, cmd, env in steps:
                if not done():
                    run(["python3", QUANT, "-m", model, *cmd], env=env)
        except SystemExit as e:
            failed.append(str(e))
    if len(lanes) > 1:          # side by side, every quant.py job reports in start/finish lines, its output in its log
        lanes = [[(done, cmd, {**(env or {}), "QUANT_LOG_ONLY": "1"}) for done, cmd, env in steps] for steps in lanes]
    threads = [threading.Thread(target=lane, args=(steps,)) for steps in lanes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failed:
        raise SystemExit("\n".join(failed))


def job_dir(model):
    return os.path.join(HERE, "work", model.split("/")[-1] + "-exl3")


def rung(rate):
    return f"{rate:.2f}bpw"


def main(a):
    W = job_dir(a.model)
    base = rung(a.bits)
    draws = [f"{base}-probe-d{i}" for i in range(a.draws)]
    arms = list(draws)
    # quantize's own test for a recipe: a "self" category measured
    recipe = "self" in quant.measured_categories(types.SimpleNamespace(measure=os.path.join(W, "_measure")))
    uniform = f"{base}-probe-uniform" if recipe else None
    if uniform:
        arms.append(uniform)
    else:
        print("=== no measurement: the draws are uniform bits, so there is no recipe to control for ===")
    extra = {f"{base}-probe-{tag}": argv for tag, argv in a.arm}
    arms += list(extra)
    yaqa = f"{base}-probe-yaqa"
    if a.yaqa:
        arms.append(yaqa)
    low_spec = f"{a.low_bits}:5" if a.low_bits and a.low_bits < 4 else (str(a.low_bits) if a.low_bits else None)
    low = quant.bit_specs(low_spec, "probe-low")[0][0] if low_spec else None
    if low:
        arms.append(low)
    common = ["-b", str(a.bits)] + (["--no-yaqa"])
    built = lambda name: lambda: os.path.isfile(os.path.join(W, name, "quantization_config.json"))
    devs = a.device.split(",")
    if a.yaqa:
        # the factor first, alone: its pass wants every GPU it can get (a box is sized for it, so there is nothing to
        # share it with), and it is the YAQA rung's only dependency -- after it, every quantize can go in one wave
        factored = lambda: os.path.isfile(os.path.join(W, "_hess", "probe", "DONE.json"))
        yaqa_dev = a.yaqa_devices or a.device
        trace_args = shlex.split(a.trace_args)
        if not {"-d", "--device"} & set(trace_args):
            trace_args += ["-d", yaqa_dev]
        # the probe rows through the eval generator, then their factor
        run_lanes(a.model, [[(factored, ["yaqa-traces", "--probe", *trace_args], None),
                             (factored, ["yaqa", "--probe", "-d", yaqa_dev] +
                              (["--layers-per-run", str(a.layers_per_run)] if a.layers_per_run else []) +
                              (["--reserve-vram", a.reserve_vram] if a.reserve_vram else []), None)]])
    # then the quantizes, one GPU each: the draws on the first --draws GPUs, the uniform and YAQA rungs on the GPUs
    # past them (a dependency split, not a scheduler: quantize still deals the draws within its share); with no
    # spare GPU they run in turn on all of them
    spare = devs[a.draws:]
    spare += list(reversed(devs)) * (a.quant_parallel - 1)
    draw_dev = ",".join(devs[:a.draws]) if spare else a.device
    uniform_dev = spare[0] if spare else a.device
    yaqa_rung_dev = spare[1] if len(spare) > 1 else uniform_dev
    # draws: the ladder's own configuration (recipe when measured), no YAQA, one seed offset each -- one quantize
    # call, so quant.py deals them to the GPUs like a ladder (finished draws are skipped there)
    draw_lane = [(lambda: all(built(name)() for name in draws),
                  ["quantize", *common, "--suffix", "probe", "--draws", str(a.draws), "-d", draw_dev], None)]
    uniform_lane = [(built(uniform), ["quantize", *common, "--no-recipe", "--suffix", "probe-uniform", "-d", uniform_dev],
                     {"EXL3_SEED_IDX_OFFSET": "0"})] if uniform else []
    # the extra arms after the uniform rung, each one quantize with its own arguments
    uniform_lane += [(built(name), ["quantize", *common, *argv, "--suffix", "probe-" + name.split("-probe-", 1)[1], "-d", uniform_dev],
                      {"EXL3_SEED_IDX_OFFSET": "0"}) for name, argv in extra.items()]
    # the low draw after the uniform rung: the ladder's configuration at that rate, seed offset 0 like the draws' first
    uniform_lane += [(built(low), ["quantize", "-b", low_spec, "--no-yaqa", "--suffix", "probe-low", "-d", uniform_dev],
                      {"EXL3_SEED_IDX_OFFSET": "0"})] if low else []
    yaqa_lane = [(built(yaqa), ["quantize", "-b", str(a.bits), "--yaqa-source", "probe", "--suffix", "probe-yaqa",
                                "-d", yaqa_rung_dev], {"EXL3_SEED_IDX_OFFSET": "0"})] if a.yaqa else []
    if not spare:
        lanes = [draw_lane + uniform_lane + yaqa_lane]
    elif yaqa_rung_dev == uniform_dev:
        lanes = [draw_lane, uniform_lane + yaqa_lane]
    else:
        lanes = [draw_lane, uniform_lane, yaqa_lane]
    run_lanes(a.model, [l for l in lanes if l])
    # recall too, when the model has a slice: YAQA has cost recall that KL read as a win (Qwen3.8-27B)
    slices = a.slices.split(",")
    if os.path.isfile(os.path.join(W, "_traces", "eval_recall.json")) and "recall" not in slices:
        slices.append("recall")
    run(["python3", QUANT, "-m", a.model, "qbench", "-d", a.device, "--slices", ",".join(slices), "--revisions", ",".join(arms),
         "--allow-converter-mismatch"])
    report(W, slices, draws, uniform, yaqa if a.yaqa else None, low,
           {name: (uniform if uniform and "--no-recipe" in argv else draws[0]) for name, argv in extra.items()})


def report(W, slices, draws, uniform, yaqa, low=None, extra=None):
    import math
    out = {"slices": {}}
    print(f"\n=== probe report: {os.path.basename(W)}, {len(draws)} draws ===")
    print(f"{'slice':8} {'draw excess (mean, spread)':>34}  {'recipe: uniform/recipe d0':>27}  {'yaqa: yaqa/d0':>16}")
    for sl in slices:
        rows = {r["label"]: r for r in json.load(open(os.path.join(W, "main", f"qb_{sl}.json")))}
        if sl == "recall":
            kept = {l: rows[l]["recall_kept"] for l in draws + [uniform] + ([yaqa] if yaqa else []) if "recall_kept" in rows.get(l, {})}
            d = [kept[l] for l in draws if l in kept]
            if not d:
                continue
            mean = sum(d) / len(d)
            sd = (sum((x - mean) ** 2 for x in d) / (len(d) - 1)) ** 0.5 if len(d) > 2 else float("nan")
            d0 = kept.get(draws[0])
            def pts(l):
                if l not in kept or d0 is None:
                    return ""
                diff = 100 * (kept[l] - d0)
                return f"{diff:+.1f} pts" + (" (within draw noise)" if not math.isnan(sd) and abs(diff) < 200 * sd else "")
            print(f"{'recall':8} kept {mean:.3f} ({100 * (max(d) - min(d)):.1f} pts range, {100 * sd:.1f} pts sd)"
                  f"  uniform {pts(uniform):>24}  yaqa {pts(yaqa) if yaqa else '':>24}")
            out["slices"]["recall"] = {"draws": d, "mean": mean, "sd": sd, "uniform": kept.get(uniform), "yaqa": kept.get(yaqa) if yaqa else None}
            continue
        fl = rows["Noise floor"]["kld"]
        ex = {l: rows[l]["kld"] - fl for l in draws + [uniform] + ([yaqa] if yaqa else []) if l in rows}
        d = [ex[l] for l in draws if l in ex]
        if not d:
            continue
        mean = sum(d) / len(d)
        spread = (max(d) - min(d)) / mean if len(d) > 1 else float("nan")
        sd = (sum((x - mean) ** 2 for x in d) / (len(d) - 1)) ** 0.5 / mean if len(d) > 2 else float("nan")
        d0 = ex.get(draws[0])
        rec = d0 / ex[uniform] if uniform in ex and d0 else None            # below 1: the recipe helps
        yq = ex[yaqa] / d0 if yaqa and yaqa in ex and d0 else None          # below 1: YAQA helps
        def vs_noise(g):
            if g is None or math.isnan(sd) or sd == 0:
                return ""
            return " (within draw noise)" if abs(1 - g) < 2 * sd else ""
        print(f"{sl:8} {mean:9.5f} ({100 * spread:4.1f}% range, {100 * sd if not math.isnan(sd) else float('nan'):4.1f}% sd)"
              f"  {'' if rec is None else f'{1 / rec:.3f}':>14}{vs_noise(rec)}"
              f"  {'' if yq is None else f'{yq:.3f}':>10}{vs_noise(yq)}")
        out["slices"][sl] = {"draws": d, "mean": mean, "range": spread, "sd": sd,
                             "uniform_over_recipe": None if rec is None else 1 / rec, "yaqa_over_control": yq}
    if extra:
        out["arms"] = {name: report_arm(W, slices, control, name, out["slices"]) for name, control in extra.items()}
    if low:
        out["low"] = report_low(W, slices, draws, low)
    with open(os.path.join(W, "main", "probe.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> {os.path.join(W, 'main', 'probe.json')}")


def report_arm(W, slices, control, hq, noise):
    """An --arm against its control, its size beside its gain: an arm that spends bytes outside the rate's budget (-hq)
    has a gain only beyond what those bytes buy along the ladder, which one rate cannot show; a budget-matched control
    can (docs/moe.md). Draw noise from the main report's spread."""
    out = {"control": control}
    sized = {r["label"]: r for r in json.load(open(os.path.join(W, "main", f"qb_{slices[0]}.json")))}
    if "vram_gb" in sized.get(hq, {}) and "vram_gb" in sized.get(control, {}):
        out["size"] = sized[hq]["vram_gb"] / sized[control]["vram_gb"]
    size = f", {100 * (out['size'] - 1):+.1f}% the size" if "size" in out else ""
    print(f"\n=== {hq} against {control}{size} ===")
    for sl in slices:
        rows = {r["label"]: r for r in json.load(open(os.path.join(W, "main", f"qb_{sl}.json")))}
        if hq not in rows or control not in rows:
            continue
        if sl == "recall":
            if "recall_kept" in rows[hq] and "recall_kept" in rows[control]:
                diff = 100 * (rows[hq]["recall_kept"] - rows[control]["recall_kept"])
                sd = (noise.get("recall") or {}).get("sd", float("nan"))
                within = " (within draw noise)" if sd == sd and abs(diff) < 200 * sd else ""
                print(f"{'recall':8} kept {diff:+.1f} pts{within}")
                out["recall_pts"] = diff
            continue
        fl = rows["Noise floor"]["kld"]
        ratio = (rows[hq]["kld"] - fl) / (rows[control]["kld"] - fl)
        sd = (noise.get(sl) or {}).get("sd", float("nan"))
        within = " (within draw noise)" if sd == sd and abs(1 - ratio) < 2 * sd else ""
        print(f"{sl:8} excess {ratio:.3f}x the control's{within}")
        out[sl] = ratio
    return out


def report_low(W, slices, draws, low):
    """The low draw beside the probe-rate draws, for the reviewer: no verdict, since one draw near the line is a call."""
    out = {"revision": low}
    print(f"\n=== {low}, one draw, against the {len(draws)} {draws[0].split('-')[0]} draws ===")
    for sl in slices:
        rows = {r["label"]: r for r in json.load(open(os.path.join(W, "main", f"qb_{sl}.json")))}
        if low not in rows:
            continue
        if sl == "recall":
            pick = lambda key: [rows[l][key] for l in draws if key in rows.get(l, {})]
            kept, errs = pick("recall_kept"), pick("recall_confident_errors")
            if not kept:
                continue
            r = rows[low]
            print(f"{'recall':8} kept {100 * r['recall_kept']:.1f}% against {100 * sum(kept) / len(kept):.1f}% "
                  f"({100 * (r['recall_kept'] - sum(kept) / len(kept)):+.1f} pts); confident errors "
                  f"{100 * r['recall_confident_errors']:.2f}% against {100 * sum(errs) / max(len(errs), 1):.2f}%")
            out["recall"] = {"kept": r["recall_kept"], "draws_kept": sum(kept) / len(kept),
                             "confident_errors": r["recall_confident_errors"],
                             "draws_confident_errors": sum(errs) / max(len(errs), 1)}
            continue
        fl = rows["Noise floor"]["kld"]
        d = [rows[l]["kld"] - fl for l in draws if l in rows]
        if not d:
            continue
        ex, mean = rows[low]["kld"] - fl, sum(d) / len(d)
        print(f"{sl:8} excess {ex:.5f}, {ex / mean:.2f}x the draws' {mean:.5f}")
        out[sl] = {"excess": ex, "over_draws": ex / mean}
    return out


#: quantize arguments the probe sets itself, and the tags its own arms use
_OWNED = {"-b", "--bits", "--suffix", "-d", "--device", "--draws"}
_RESERVED = re.compile(r"^(d\d+|uniform|yaqa|low)$")


def parse_arm(spec):
    """TAG=ARGS -> (tag, [args])"""
    tag, sep, rest = spec.partition("=")
    if not sep or not re.fullmatch(r"[A-Za-z0-9_.]+", tag) or _RESERVED.match(tag):
        raise argparse.ArgumentTypeError(f"{spec!r}: TAG=ARGS, TAG letters, digits, '_' or '.', not one of the probe's own")
    argv = shlex.split(rest)
    owned = sorted({x.split("=", 1)[0] for x in argv} & _OWNED)
    if owned:
        raise argparse.ArgumentTypeError(f"{spec!r}: {', '.join(owned)} is the probe's to set")
    return tag, argv


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("-m", "--model", required=True)
    p.add_argument("-b", "--bits", type=float, default=4.0)
    p.add_argument("--draws", type=int, default=5)
    p.add_argument("--low-bits", type=float, default=2.5,
                   help="one draw at this rate too, as evidence for a sub-3 rung (0: none); reported, not decided")
    p.add_argument("--yaqa", action="store_true", help="also the YAQA arm (yaqa --probe's factor)")
    p.add_argument("--arm", action="append", default=[], type=parse_arm, metavar="TAG=ARGS",
                   help="also a rung converted with `quantize ARGS` (one shell-quoted string), named <rate>-probe-TAG and "
                        "reported with its size against draw 0, or the uniform rung if ARGS has --no-recipe; repeatable. "
                        "E.g. --arm hq2='--hq 2'")
    p.add_argument("--layers-per-run", type=int, default=None, help="yaqa: decoder layers per factor pass")
    p.add_argument("--trace-args", default="",
                   help="yaqa-traces: its generator options, one string, e.g. "
                        "'--generator Qwen/Qwen3.6-35B-A3B-FP8 -d 0,1 --vllm-args {\"tensor_parallel_size\": 2}'")
    p.add_argument("--reserve-vram", default=None, help="yaqa: GB kept free of weights per device, with several (-d 0,1)")
    p.add_argument("--slices", default="wild,swe,self")
    p.add_argument("-d", "--device", default="0",
                   help="GPUs for the probe: the YAQA factor pass and qbench use all of them; the quantizes go one GPU each, "
                        "the draws on the first --draws and the uniform (then the low draw) and YAQA rungs on the next two "
                        "(in turn on one spare, or after the draws with none)")
    p.add_argument("--quant-parallel", type=int, default=1,
                   help="Maximum number of quantize runs to allow per GPU")
    p.add_argument("--yaqa-devices", default=None,
                   help="GPUs for the YAQA rows and factor pass instead of all of -d")
    a = p.parse_args()
    if not a.model or "/" not in a.model:
        raise SystemExit("-m takes the base model's repo id, as quant.py does")
    os.chdir(HERE)          # quant.py's paths (work/, ../deps/exllamav3) are relative to it
    try:
        main(a)
    except KeyboardInterrupt:
        quant.stop_children()
        raise SystemExit(130)
