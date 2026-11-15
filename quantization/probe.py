#!/usr/bin/env python3
"""Probe a model before its ladder: draw noise, recipe gain and YAQA gain at one mid rate, side by side.

Builds, at one rate (default 4.00):
  - N draws of the configuration the ladder would use (sign-draw offsets 0..N-1, EXL3_SEED_IDX_OFFSET),
  - a uniform-bits rung at offset 0 (the recipe's control),
  - with --yaqa, a rung converted against `yaqa --probe`'s factor at offset 0 (its control is draw 0),
then scores them with one restricted qbench and reports each slice's excess KLD, the draws' spread, and each gain
measured against that spread. A gain smaller than what a redraw does is labelled as such: on Qwen3.6-35B-A3B's swe a
redraw moved a 4.00 by 10-17%, as much as the most extreme calibration tried (TODO.md multi-draw, yaqa-probe).

Mechanical by design: it only drives `quant.py quantize`, `yaqa --probe` and `qbench`, and every step is skipped when
its output exists, so a rerun picks up where one stopped. Run from quantization/:

    python3 probe.py -m openbmb/MiniCPM5-2B --draws 5 --yaqa
"""
import argparse, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
QUANT = os.path.join(HERE, "quant.py")


def run(cmd, env=None):
    print("=== " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=HERE, env={**os.environ, **(env or {})})
    if rc:
        raise SystemExit(f"=== failed ({rc}): {' '.join(cmd)}")


def job_dir(model):
    return os.path.join(HERE, "work", model.split("/")[-1] + "-exl3")


def rung(rate):
    return f"{rate:.2f}bpw"


def main(a):
    W = job_dir(a.model)
    base = rung(a.bits)
    draws = [f"{base}-probe-d{i}" for i in range(a.draws)]
    arms = list(draws)
    uniform = f"{base}-probe-uniform"
    arms.append(uniform)
    yaqa = f"{base}-probe-yaqa"
    if a.yaqa:
        arms.append(yaqa)
    dev = ["-d", a.device]
    common = ["-b", str(a.bits)] + (["--no-yaqa"])
    # draws: the ladder's own configuration (recipe when measured), no YAQA, one seed offset each -- one quantize
    # call, so quant.py deals them to the GPUs like a ladder (finished draws are skipped there)
    if not all(os.path.isfile(os.path.join(W, name, "quantization_config.json")) for name in draws):
        run(["python3", QUANT, "-m", a.model, "quantize", *common, "--suffix", "probe", "--draws", str(a.draws), *dev])
    if not os.path.isfile(os.path.join(W, uniform, "quantization_config.json")):
        run(["python3", QUANT, "-m", a.model, "quantize", *common, "--no-recipe", "--suffix", "probe-uniform", *dev],
            env={"EXL3_SEED_IDX_OFFSET": "0"})
    if a.yaqa:
        if not os.path.isfile(os.path.join(W, "_hess", "probe", "DONE.json")):
            run(["python3", QUANT, "-m", a.model, "yaqa", "--probe", *dev] +
                (["--layers-per-run", str(a.layers_per_run)] if a.layers_per_run else []) +
                (["--reserve-vram", a.reserve_vram] if a.reserve_vram else []))
        if not os.path.isfile(os.path.join(W, yaqa, "quantization_config.json")):
            run(["python3", QUANT, "-m", a.model, "quantize", "-b", str(a.bits), "--yaqa-source", "probe", "--suffix", "probe-yaqa", *dev],
                env={"EXL3_SEED_IDX_OFFSET": "0"})
    # recall too, when the model has a slice: YAQA has cost recall that KL read as a win (Qwen3.8-27B)
    slices = a.slices.split(",")
    if os.path.isfile(os.path.join(W, "_traces", "eval_recall.json")) and "recall" not in slices:
        slices.append("recall")
    run(["python3", QUANT, "-m", a.model, "qbench", *dev, "--slices", ",".join(slices), "--revisions", ",".join(arms),
         "--allow-converter-mismatch"])
    report(W, slices, draws, uniform, yaqa if a.yaqa else None)


def report(W, slices, draws, uniform, yaqa):
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
    with open(os.path.join(W, "main", "probe.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> {os.path.join(W, 'main', 'probe.json')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("-m", "--model", required=True)
    p.add_argument("-b", "--bits", type=float, default=4.0)
    p.add_argument("--draws", type=int, default=5)
    p.add_argument("--yaqa", action="store_true", help="also the YAQA arm (yaqa --probe's factor)")
    p.add_argument("--layers-per-run", type=int, default=None, help="yaqa: decoder layers per factor pass")
    p.add_argument("--reserve-vram", default=None, help="yaqa: GB kept free of weights per device, with several (-d 0,1)")
    p.add_argument("--slices", default="wild,swe,self")
    p.add_argument("-d", "--device", default="0", help="GPUs for every step: quantize deals the draws to them, yaqa splits the model "
                                                    "across them, qbench splits the arms")
    a = p.parse_args()
    if not a.model or "/" not in a.model:
        raise SystemExit("-m takes the base model's repo id, as quant.py does")
    main(a)
