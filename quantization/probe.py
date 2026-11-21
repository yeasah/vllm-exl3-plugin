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
import argparse, json, math, os, re, shlex, subprocess, sys, types, queue

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
    # calibration alone: what every arm departs from by one component (an --arm may say --recipe; the last one wins)
    plain = ["--no-yaqa", "--no-recipe"]
    common = ["-b", str(a.bits)] + plain
    # the arms, each (tag, rate spec, quantize arguments): one component over the draws' configuration
    table = []
    # quantize's own test for a recipe: a "self" category measured
    if "self" in quant.measured_categories(types.SimpleNamespace(measure=os.path.join(W, "_measure"))):
        table.append(("recipe", str(a.bits), ["--no-yaqa", "--recipe"]))
    else:
        print("=== no measurement: no recipe arm ===")
    if a.yaqa:
        table.append(("yaqa", str(a.bits), ["--no-recipe", "--yaqa-source", "probe"]))
    table += [(tag, str(a.bits), [*plain, *argv]) for tag, argv in a.arm]
    low_spec = f"{a.low_bits}:5" if a.low_bits and a.low_bits < 4 else (str(a.low_bits) if a.low_bits else None)
    if low_spec:
        table.append(("low", low_spec, plain))
    counts = dict(a.arm_draws)
    unknown = sorted(set(counts) - {t for t, _, _ in table})
    if unknown:
        raise SystemExit(f"--arm-draws {', '.join(unknown)}: no such arm (have {', '.join(t for t, _, _ in table)})")
    names = {tag: arm_names(spec, tag, counts.get(tag, 1)) for tag, spec, _ in table}
    for tag, spec, _ in table:
        adopt_singleton(W, quant.bit_specs(spec, f"probe-{tag}")[0][0], names[tag][0])
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
    # ladder (finished draws are skipped there); every arm the same way, its draws from offset 0 up
    jobs = [draw_job(built, draws, common, "probe")]
    jobs += [draw_job(built, names[tag], ["-b", spec, *argv], f"probe-{tag}") for tag, spec, argv in table]
    quant_devs = devlist * (a.quant_parallel)
    run_jobs(quant_devs, a.model, jobs)
    # recall too, when the model has a slice: YAQA has cost recall that KL read as a win (Qwen3.8-27B)
    slices = a.slices.split(",")
    if os.path.isfile(os.path.join(W, "_traces", "eval_recall.json")) and "recall" not in slices:
        slices.append("recall")
    qbench_devs = devlist * (a.qbench_parallel)
    revisions = draws + [n for tag, _, _ in table for n in names[tag]]
    run(["python3", QUANT, "-m", a.model, "qbench", "-d", ",".join(qbench_devs), "--slices", ",".join(slices),
         "--revisions", ",".join(revisions), "--allow-converter-mismatch"])
    report(W, slices, draws, {tag: names[tag] for tag, _, _ in table if tag != "low"}, names.get("low"))


def arm_names(spec, tag, n):
    """An arm's rungs: <rate>-probe-<tag>-d0 .. -d{n-1}, a singleton too, so more draws later keep the first."""
    rev = quant.bit_specs(spec, f"probe-{tag}")[0][0]
    return [f"{rev}-d{k}" for k in range(n)]


def draw_job(built, names, args, suffix):
    """The quantize job for a set of draws at seed offsets 0 .. n-1: quantize --draws names them -d0.., and one draw is
    a singleton named -d0 at offset 0, the same rung --draws would build first."""
    done = lambda: all(built(n)() for n in names)
    if len(names) == 1:
        return (done, 1, ["quantize", *args, "--suffix", f"{suffix}-d0"], {"EXL3_SEED_IDX_OFFSET": "0"})
    return (done, len(names), ["quantize", *args, "--suffix", suffix, "--draws", str(len(names))], None)


def adopt_singleton(W, old, new):
    """An arm built before arms were named -d0 (<rate>-probe-<tag>) becomes its -d0: the rung, its scored rows, logs and
    recipe files renamed, so nothing is rebuilt or rescored."""
    if not os.path.isdir(os.path.join(W, old)) or os.path.exists(os.path.join(W, new)):
        return
    import glob
    for path in glob.glob(os.path.join(W, "main", "qb_*.json")) + glob.glob(os.path.join(W, "main", "_qbench_parts", "qb_*.json")):
        with open(path) as f:
            rows = json.load(f)
        if any(r.get("label") == old for r in rows):
            for r in rows:
                if r.get("label") == old:
                    r["label"] = new
            with open(path + ".tmp", "w") as f:
                json.dump(rows, f, indent=2)
            os.replace(path + ".tmp", path)
    for src, dst in quant._promote_files(types.SimpleNamespace(dir=W), old, new):
        os.rename(src, dst)
    os.rename(os.path.join(W, old), os.path.join(W, new))
    print(f"=== {old} is now {new} ===")


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def report(W, slices, draws, arms, low=None):
    """The draws' spread, then every arm against the draws' mean in one table: excess KLD as arm / mean (below 1 better),
    recall as points kept over the mean's, "~" where a redraw could do as much, and each arm's size over draw 0's. An arm
    of several draws (--arm-draws) counts as their mean, and its "~" is the difference of two means, 2 sd * sqrt(1/n +
    1/N) with sd the main draws' spread. Against the mean, not draw 0: an arm shares draw 0's seed, but nothing shows a
    seed's luck carries over to another configuration (draw rank does not even transfer across traces), and on the
    35B's swe draw 0 sat 7.6% above the mean, which went into every ratio. probe.json keeps each draw's ratio."""
    out = {"slices": {}, "arms": {tag: {"draws": len(ns), "revisions": ns} for tag, ns in arms.items()}}
    d0 = draws[0]
    loaded = {sl: {r["label"]: r for r in json.load(open(os.path.join(W, "main", f"qb_{sl}.json")))} for sl in slices}
    col = lambda tag: tag + (f"x{len(arms[tag])}" if len(arms[tag]) > 1 else "")
    print(f"\n=== probe report: {os.path.basename(W)}, {len(draws)} draws; arms over the draws' mean, below 1 better, "
          f"~ within draw noise ===")
    print(f"{'slice':8} {'draws':>10} {'range':>7} {'sd':>6}  " + "".join(f"{col(t):>10}" for t in arms))
    widen = lambda n: (1 / n + 1 / len(draws)) ** 0.5
    excess = {}
    for sl in slices:
        rows = loaded[sl]
        if sl == "recall":
            kept = lambda ns: [rows[l]["recall_kept"] for l in ns if "recall_kept" in rows.get(l, {})]
            d = kept(draws)
            if not d:
                continue
            mean = _mean(d)
            sd = (sum((x - mean) ** 2 for x in d) / (len(d) - 1)) ** 0.5 if len(d) > 2 else float("nan")
            cells = ""
            for tag, ns in arms.items():
                k = kept(ns)
                if not k:
                    cells += f"{'':>10}"
                    continue
                diff = 100 * (_mean(k) - mean)
                tilde = "~" if sd == sd and abs(diff) < 200 * sd * widen(len(k)) else " "
                cells += f"{diff:+8.1f}{tilde} "
                out["arms"][tag]["recall_pts"] = diff
                out["arms"][tag]["recall_pts_each"] = [100 * (x - mean) for x in k]
            print(f"{'recall':8} {100 * mean:9.1f}% {100 * (max(d) - min(d)):5.1f}pt {100 * sd:4.1f}pt  {cells}")
            out["slices"]["recall"] = {"draws": d, "mean": mean, "sd": sd}
            continue
        fl = rows["Noise floor"]["kld"]
        ex = lambda ns: [rows[l]["kld"] - fl for l in ns if l in rows]
        d = ex(draws)
        if not d:
            continue
        mean = _mean(d)
        spread = (max(d) - min(d)) / mean if len(d) > 1 else float("nan")
        sd = (sum((x - mean) ** 2 for x in d) / (len(d) - 1)) ** 0.5 / mean if len(d) > 2 else float("nan")
        excess[sl] = {"draws": mean, **{tag: _mean(ex(ns)) for tag, ns in arms.items() if ex(ns)},
                      **({"low": _mean(ex(low))} if low and ex(low) else {})}
        cells = ""
        for tag, ns in arms.items():
            e = ex(ns)
            if not e:
                cells += f"{'':>10}"
                continue
            ratio = _mean(e) / mean
            tilde = "~" if sd == sd and abs(1 - ratio) < 2 * sd * widen(len(e)) else " "
            cells += f"{ratio:8.3f}{tilde} "
            out["arms"][tag][sl] = ratio
            out["arms"][tag].setdefault("each", {})[sl] = [x / mean for x in e]
        print(f"{sl:8} {mean:10.5f} {100 * spread:6.1f}% {100 * sd:5.1f}%  {cells}")
        out["slices"][sl] = {"draws": d, "mean": mean, "range": spread, "sd": sd}
    sized = loaded[slices[0]]
    gb = lambda ns: _mean([sized[n]["vram_gb"] for n in ns if "vram_gb" in sized.get(n, {})])
    sizes = {"draws": gb([d0]), **{tag: gb(ns) for tag, ns in arms.items()}, **({"low": gb(low)} if low else {})}
    cells = ""
    for tag in arms:
        if sizes.get(tag) and sizes["draws"]:
            out["arms"][tag]["size"] = sizes[tag] / sizes["draws"]
            cells += f"{100 * (out['arms'][tag]['size'] - 1):+8.1f}% "
        else:
            cells += f"{'':>10}"
    if arms:
        print(f"{'size':8} {'':>10} {'':>7} {'':>6}  {cells}")
    noise = {sl: v["sd"] for sl, v in out["slices"].items() if sl != "recall"}
    size_matched(out, excess, sizes, d0, {tag: len(ns) for tag, ns in arms.items()}, low, col, noise, len(draws))
    if low:
        out["low"] = report_low(W, slices, draws, low)
    with open(os.path.join(W, "main", "probe.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> {os.path.join(W, 'main', 'probe.json')}")


#: The rule of thumb above the knee: excess KLD halves per half bit
_PER_HALF_BIT = 2.0

def size_matched(out, excess, sizes, d0, counts, low, col, noise=None, n_draws=1):
    """Each arm against the draws' mean projected to the arm's size, for arms that are not the same size: the size
    difference turned into bits, and the bits into excess KLD over a bracket. Bits from the low draw's size (the same
    configuration at another rate, so bytes per bpw with every fixed part counted), or without one from size / rate.
    Not from the low draw's KLD for the slope alone: it is there because it may be at or past the knee, so the bracket
    runs from the rule's 2x per half bit to the low draw's own rate per half bit, capped at 2 (a chord, so it leans
    high). "~" when the bracket's pessimistic end is within draw noise. A projection, not a measurement; the
    budget-matched rung is the measurement (docs/moe.md)."""
    moved = [t for t in counts if sizes.get(t) and sizes.get("draws") and abs(sizes[t] / sizes["draws"] - 1) >= 0.005]
    if not moved:
        return
    rate = float(re.match(r"([\d.]+)bpw", d0).group(1))
    low_rate = float(re.match(r"([\d.]+)bpw", low[0]).group(1)) if low else None
    if low and sizes.get("low") and low_rate != rate:
        per_bit, how = (sizes["draws"] - sizes["low"]) / (rate - low_rate), f"the {low_rate:.2f} draw's size"
    else:
        per_bit, how = sizes["draws"] / rate, "size / rate"
    bits = {t: (sizes[t] - sizes["draws"]) / per_bit for t in moved}
    print(f"\n=== size-matched (projected): arm / the draws' mean at the arm's size, bits from {how} ("
          + ", ".join(f"{col(t)} {b:+.3f} bpw" for t, b in bits.items()) + "); from the low draw's own rate per half "
          f"bit (capped at {_PER_HALF_BIT:g}; a chord, so it leans high) to the rule's {_PER_HALF_BIT:g}x ===")
    print(f"{'slice':8} " + "".join(f"{col(t):>14} " for t in moved))
    for sl, ex in excess.items():
        own = None
        if ex.get("low") and low_rate is not None and rate > low_rate:
            own = min((ex["low"] / ex["draws"]) ** (1 / (2 * (rate - low_rate))), _PER_HALF_BIT)
        rates = sorted({_PER_HALF_BIT} | ({own} if own else set()))
        cells = ""
        for t in moved:
            if ex.get(t):
                adj = sorted(ex[t] / ex["draws"] * r ** (2 * bits[t]) for r in rates)
                sd = (noise or {}).get(sl, float("nan")) * (1 / counts[t] + 1 / n_draws) ** 0.5
                near = min(adj, key=lambda v: abs(1 - v))
                tilde = "~" if sd == sd and abs(1 - near) < 2 * sd else " "
                cells += (f"{adj[0]:6.3f}-{adj[-1]:.3f}{tilde}" if adj[-1] - adj[0] >= 0.005 else f"{adj[0]:13.3f}{tilde}")
                out["arms"][t].setdefault("size_matched", {})[sl] = [adj[0], adj[-1]]
            else:
                cells += f"{'':>14}"
        print(f"{sl:8} {cells}")
    for t in moved:
        out["arms"][t]["bits_over_draws"] = bits[t]


def report_low(W, slices, draws, low):
    """The low draw(s) beside the probe-rate draws, for the reviewer: no verdict, since one draw near the line is a call."""
    out = {"revisions": low}
    tag = low[0] + (f" (mean of {len(low)})" if len(low) > 1 else ", one draw")
    print(f"\n=== {tag}, against the {len(draws)} {draws[0].split('-')[0]} draws ===")
    rate = float(re.match(r"([\d.]+)bpw", draws[0]).group(1)) - float(re.match(r"([\d.]+)bpw", low[0]).group(1))
    for sl in slices:
        rows = {r["label"]: r for r in json.load(open(os.path.join(W, "main", f"qb_{sl}.json")))}
        if not any(l in rows for l in low):
            continue
        if sl == "recall":
            pick = lambda ns, key: [rows[l][key] for l in ns if key in rows.get(l, {})]
            kept, errs = pick(draws, "recall_kept"), pick(draws, "recall_confident_errors")
            lk, le = pick(low, "recall_kept"), pick(low, "recall_confident_errors")
            if not kept or not lk:
                continue
            print(f"{'recall':8} kept {100 * _mean(lk):.1f}% against {100 * _mean(kept):.1f}% "
                  f"({100 * (_mean(lk) - _mean(kept)):+.1f} pts); confident errors "
                  f"{100 * _mean(le):.2f}% against {100 * _mean(errs):.2f}%")
            out["recall"] = {"kept": _mean(lk), "draws_kept": _mean(kept),
                             "confident_errors": _mean(le), "draws_confident_errors": _mean(errs)}
            continue
        fl = rows["Noise floor"]["kld"]
        d = [rows[l]["kld"] - fl for l in draws if l in rows]
        lo = [rows[l]["kld"] - fl for l in low if l in rows]
        if not d or not lo:
            continue
        ex, mean = _mean(lo), _mean(d)
        # in the rule of thumb's units: well over 2x per half bit is a knee between the two rates
        per_half = (ex / mean) ** (1 / (2 * rate)) if rate > 0 else float("nan")
        print(f"{sl:8} excess {ex:.5f}, {ex / mean:.2f}x the draws' {mean:.5f} ({per_half:.2f}x per half bit)")
        out[sl] = {"excess": ex, "over_draws": ex / mean, "per_half_bit": per_half}
    return out


#: quantize arguments the probe sets itself, and the tags its own arms use
_OWNED = {"-b", "--bits", "--suffix", "-d", "--device", "--draws"}
_RESERVED = re.compile(r"^(d\d+|recipe|yaqa|low)$|-d\d+$")


def parse_arm_draws(spec):
    """TAG=N -> (tag, n)"""
    tag, sep, n = spec.partition("=")
    if not sep or not n.isdigit() or int(n) < 1:
        raise argparse.ArgumentTypeError(f"{spec!r}: TAG=N, N a whole number of draws")
    return tag, int(n)


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
    p.add_argument("--arm-draws", action="append", default=[], type=parse_arm_draws, metavar="TAG=N",
                   help="draws of an arm (an --arm tag, or recipe, yaqa, low), at seed offsets 0..N-1, reported as their "
                        "mean (default 1); raising it later builds only the new draws. Repeatable")
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
    p.add_argument("--qbench-parallel", type=int, default=1,
                   help="Maximum number of qbench runs to allow per GPU")
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
