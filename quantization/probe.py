#!/usr/bin/env python3
"""Probe a model before its ladder: draw noise, and each component's gain on its own, at one mid rate.

Builds, at one rate (default 4.00), each component alone against the plain calibrated rung:
  - N draws with calibration only -- uniform bits, no YAQA (sign-draw offsets 0..N-1, EXL3_SEED_IDX_OFFSET): the
    draw noise every arm's gain is measured against, and draw 0 is every arm's control,
  - a recipe rung at offset 0, when `measure` has run (quantize's own test, a "self" category measured),
  - with --yaqa, a rung converted against `yaqa --probe`'s factor at offset 0, uniform bits,
  - with --arm TAG=ARGS (repeatable), a rung converted with `quantize ARGS` at offset 0, uniform bits unless ARGS
    says --recipe, named <rate>-probe-TAG (`--arm hq2='--hq 2'`),
  - one draw at --low-bits (default 2.50, head 5 below 4 as the ladder has it), configured like the draws,
then scores them with one restricted qbench and reports the draws' spread and each arm's excess KLD over draw 0's,
below 1 better, marked "~" where a redraw could do as much: on Qwen3.6-35B-A3B's swe a redraw moved a 4.00 by 10-17%,
mostly draw x trace interaction (docs/calibration.md, "A draw's swe rank does not transfer"). Each arm's size is beside
its gain, since an arm may spend bits outside the rate (-hq), which one rate cannot price: a budget-matched control
can (docs/moe.md). Components are tested alone, not composed: the ladder may combine them, the probe does not.

The low draw is evidence for a sub-3 rung, not a verdict: the ladder starts at 3.00 unless one is chosen. It is reported
against the draws -- each slice's excess KLD as a multiple of theirs, recall kept and confident errors beside theirs --
so the gradient from the probe rate down is in view. One draw: recall varies by a few points between draws at a rate,
so a result near the line is the reviewer's call.

Mechanical by design: it only drives `quant.py quantize`, `yaqa --probe` and `qbench`, and every step is skipped when
its output exists, so a rerun picks up where one stopped. With --yaqa the factor is collected first, on every GPU;
then the quantizes are scheduled over -d (each GPU listed --quant-parallel times). Run from quantization/:

    python3 probe.py -m openbmb/MiniCPM5-2B --draws 5 --yaqa
"""
import argparse, json, os, re, shlex, subprocess, sys, types, queue

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


def run_jobs(devs, model, jobs):
    """Schedule the list of jobs across the listed devices.
    One job failing lets the others finish (their outputs stand for a rerun), then fails the probe."""
    import threading
    failed = []
    def job_thread(job, job_devs, job_id, q):
        try:
            (done, _, cmd, env) = job
            if not done():
                run(["python3", QUANT, "-m", model, *cmd, "-d", ",".join(job_devs)], env=env)
        except SystemExit as e:
            failed.append(str(e))
        finally:
            q.put(job_id)

    if len(jobs) > 1:          # side by side, every quant.py job reports in start/finish lines, its output in its log
        jobs = [(done, job_devs, cmd, {**(env or {}), "QUANT_LOG_ONLY": "1"}) for done, job_devs, cmd, env in jobs]
    available_devs = devs
    running_jobs = {}
    thread_q = queue.Queue()
    job_id = 0
    while running_jobs or jobs:
        # schedule as many jobs as fit in available devices, satisfying device requests as much as possible.
        # fully optimizing this in general is a 2D packing exercise with one dimension (time) not currently predictable
        # and correlated with another dimension (number of devices) which is flexible. we're not going there,
        # heuristics could hurt as much as they help, and the normal case is one big bunch (the draws) followed by
        # a whole bunch of singletons, so the only packing that matters comes first, and it already gets the most
        # devices it can get. so: just go through the list in order
        while available_devs and jobs:
            next_job = jobs[0]
            num_assigned_devs = min(next_job[1], len(available_devs))
            if (num_assigned_devs == 0):
                raise SystemExit("Invalid job defined with zero devices")
            assigned_devs = available_devs[:num_assigned_devs]
            available_devs = available_devs[num_assigned_devs:]
            thread = threading.Thread(target=job_thread, args=(next_job,assigned_devs,job_id,thread_q))
            thread.start()
            jobs = jobs[1:]
            running_jobs[job_id] = (thread, assigned_devs)
            job_id += 1

        # wait for a job to finish, and return the devices to the pool
        finished_id = thread_q.get()
        (thread, assigned_devs) = running_jobs[finished_id]
        thread.join()
        available_devs += assigned_devs
        del running_jobs[finished_id]

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
    recipe = None
    if "self" in quant.measured_categories(types.SimpleNamespace(measure=os.path.join(W, "_measure"))):
        recipe = f"{base}-probe-recipe"
        arms.append(recipe)
    else:
        print("=== no measurement: no recipe arm ===")
    extra = {f"{base}-probe-{tag}": argv for tag, argv in a.arm}
    arms += list(extra)
    yaqa = f"{base}-probe-yaqa"
    if a.yaqa:
        arms.append(yaqa)
    low_spec = f"{a.low_bits}:5" if a.low_bits and a.low_bits < 4 else (str(a.low_bits) if a.low_bits else None)
    low = quant.bit_specs(low_spec, "probe-low")[0][0] if low_spec else None
    if low:
        arms.append(low)
    # calibration alone: what every arm departs from by one component (an --arm may say --recipe; the last one wins)
    plain = ["--no-yaqa", "--no-recipe"]
    common = ["-b", str(a.bits)] + plain
    built = lambda name: lambda: os.path.isfile(os.path.join(W, name, "quantization_config.json"))
    devlist = a.device.split(",")
    if a.yaqa:
        # the factor first, alone: its pass wants every GPU it can get (a box is sized for it, so there is nothing to
        # share it with), and it is the YAQA rung's only dependency -- after it, every quantize can go in one wave
        factored = lambda: os.path.isfile(os.path.join(W, "_hess", "probe", "DONE.json"))
        trace_args = shlex.split(a.trace_args)
        # the probe rows through the eval generator, then their factor
        run_jobs(devlist, a.model, [(factored, len(devlist), ["yaqa-traces", "--probe", *trace_args], None)])
        run_jobs(devlist, a.model, [(factored, len(devlist), ["yaqa", "--probe"] +
                                    (["--layers-per-run", str(a.layers_per_run)] if a.layers_per_run else []) +
                                    (["--reserve-vram", a.reserve_vram] if a.reserve_vram else []), None)])
    # draws: calibration only, one seed offset each -- one quantize call, so quant.py deals them to the GPUs like a
    # ladder (finished draws are skipped there); every other arm is one component on top of draw 0's configuration
    zero = {"EXL3_SEED_IDX_OFFSET": "0"}
    jobs = [(lambda: all(built(name)() for name in draws), a.draws,
             ["quantize", *common, "--suffix", "probe", "--draws", str(a.draws)], None)]
    if recipe:
        jobs.append((built(recipe), 1, ["quantize", "-b", str(a.bits), "--no-yaqa", "--recipe", "--suffix", "probe-recipe"],
                     zero))
    jobs += [(built(name), 1, ["quantize", *common, *argv, "--suffix", "probe-" + name.split("-probe-", 1)[1]], zero)
             for name, argv in extra.items()]
    if low:
        jobs.append((built(low), 1, ["quantize", "-b", low_spec, *plain, "--suffix", "probe-low"], zero))
    if a.yaqa:
        jobs.append((built(yaqa), 1, ["quantize", "-b", str(a.bits), "--no-recipe", "--yaqa-source", "probe",
                                      "--suffix", "probe-yaqa"], zero))
    quant_devs = devlist * (a.quant_parallel)
    run_jobs(quant_devs, a.model, jobs)
    # recall too, when the model has a slice: YAQA has cost recall that KL read as a win (Qwen3.8-27B)
    slices = a.slices.split(",")
    if os.path.isfile(os.path.join(W, "_traces", "eval_recall.json")) and "recall" not in slices:
        slices.append("recall")
    run(["python3", QUANT, "-m", a.model, "qbench", "-d", a.device, "--slices", ",".join(slices), "--revisions", ",".join(arms),
         "--allow-converter-mismatch"])
    report(W, slices, draws, [x for x in [recipe, yaqa if a.yaqa else None, *extra] if x], low)


def report(W, slices, draws, arms, low=None):
    """The draws' spread, then every arm against draw 0 in one table: excess KLD as arm / draw 0 (below 1 better), recall
    as points kept over draw 0's, "~" where a redraw could do as much (2 sd), and each arm's size over draw 0's."""
    out = {"slices": {}, "arms": {name: {} for name in arms}}
    d0 = draws[0]
    short = lambda name: name.split("-probe-", 1)[1]
    loaded = {sl: {r["label"]: r for r in json.load(open(os.path.join(W, "main", f"qb_{sl}.json")))} for sl in slices}
    print(f"\n=== probe report: {os.path.basename(W)}, {len(draws)} draws; arms over draw 0, below 1 better, "
          f"~ within draw noise ===")
    print(f"{'slice':8} {'draws':>10} {'range':>7} {'sd':>6}  " + "".join(f"{short(n):>10}" for n in arms))
    for sl in slices:
        rows = loaded[sl]
        if sl == "recall":
            kept = {l: rows[l]["recall_kept"] for l in draws + arms if "recall_kept" in rows.get(l, {})}
            d = [kept[l] for l in draws if l in kept]
            if not d or d0 not in kept:
                continue
            mean = sum(d) / len(d)
            sd = (sum((x - mean) ** 2 for x in d) / (len(d) - 1)) ** 0.5 if len(d) > 2 else float("nan")
            cells = ""
            for n in arms:
                if n not in kept:
                    cells += f"{'':>10}"
                    continue
                diff = 100 * (kept[n] - kept[d0])
                tilde = "~" if sd == sd and abs(diff) < 200 * sd else " "
                cells += f"{diff:+8.1f}{tilde} "
                out["arms"][n]["recall_pts"] = diff
            print(f"{'recall':8} {100 * mean:9.1f}% {100 * (max(d) - min(d)):5.1f}pt {100 * sd:4.1f}pt  {cells}")
            out["slices"]["recall"] = {"draws": d, "mean": mean, "sd": sd}
            continue
        fl = rows["Noise floor"]["kld"]
        ex = {l: rows[l]["kld"] - fl for l in draws + arms if l in rows}
        d = [ex[l] for l in draws if l in ex]
        if not d or d0 not in ex:
            continue
        mean = sum(d) / len(d)
        spread = (max(d) - min(d)) / mean if len(d) > 1 else float("nan")
        sd = (sum((x - mean) ** 2 for x in d) / (len(d) - 1)) ** 0.5 / mean if len(d) > 2 else float("nan")
        cells = ""
        for n in arms:
            if n not in ex:
                cells += f"{'':>10}"
                continue
            ratio = ex[n] / ex[d0]
            tilde = "~" if sd == sd and abs(1 - ratio) < 2 * sd else " "
            cells += f"{ratio:8.3f}{tilde} "
            out["arms"][n][sl] = ratio
        print(f"{sl:8} {mean:10.5f} {100 * spread:6.1f}% {100 * sd:5.1f}%  {cells}")
        out["slices"][sl] = {"draws": d, "mean": mean, "range": spread, "sd": sd}
    sized = loaded[slices[0]]
    cells = ""
    for n in arms:
        if "vram_gb" in sized.get(n, {}) and "vram_gb" in sized.get(d0, {}):
            out["arms"][n]["size"] = sized[n]["vram_gb"] / sized[d0]["vram_gb"]
            cells += f"{100 * (out['arms'][n]['size'] - 1):+8.1f}% "
        else:
            cells += f"{'':>10}"
    if arms:
        print(f"{'size':8} {'':>10} {'':>7} {'':>6}  {cells}")
    if low:
        out["low"] = report_low(W, slices, draws, low)
    with open(os.path.join(W, "main", "probe.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> {os.path.join(W, 'main', 'probe.json')}")


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
_RESERVED = re.compile(r"^(d\d+|recipe|yaqa|low)$")


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
                   help="also a rung converted with `quantize ARGS` (one shell-quoted string) on top of the draws' "
                        "configuration (uniform bits unless ARGS says --recipe, no YAQA), named <rate>-probe-TAG and "
                        "reported with its size against draw 0; repeatable. E.g. --arm hq2='--hq 2'")
    p.add_argument("--layers-per-run", type=int, default=None, help="yaqa: decoder layers per factor pass")
    p.add_argument("--trace-args", default="",
                   help="yaqa-traces: its generator options, one string, e.g. "
                        "'--generator Qwen/Qwen3.6-35B-A3B-FP8 -d 0,1 --vllm-args {\"tensor_parallel_size\": 2}'")
    p.add_argument("--reserve-vram", default=None, help="yaqa: GB kept free of weights per device, with several (-d 0,1)")
    p.add_argument("--slices", default="wild,swe,self")
    p.add_argument("-d", "--device", default="0",
                   help="GPUs for the probe: the YAQA factor pass and qbench use all of them; the quantizes are "
                        "scheduled over them, the draws first, each other arm on one")
    p.add_argument("--quant-parallel", type=int, default=1,
                   help="Maximum number of quantize runs to allow per GPU")
    a = p.parse_args()
    if not a.model or "/" not in a.model:
        raise SystemExit("-m takes the base model's repo id, as quant.py does")
    os.chdir(HERE)          # quant.py's paths (work/, ../deps/exllamav3) are relative to it
    quant.interrupt_on_sigterm()
    try:
        main(a)
    except KeyboardInterrupt:
        quant.stop_children()
        raise SystemExit(130)
