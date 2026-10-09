#!/usr/bin/env python3

import argparse
import json
import sys
import os
import re
import glob as globmod
import subprocess
import shutil
import datetime
import yaml
from jinja2 import Environment, FileSystemLoader
from huggingface_hub import ModelCard, ModelCardData, HfApi, hf_hub_download, snapshot_download

class Job:
    """Every path and repo id the subcommands work from, derived once.

    Each do_* used to rebuild these from args, and collect_metadata and its helpers
    built work paths a third time with "work" hardcoded -- so -w/--workdir applied to
    some of them and silently not to others. Nothing below this class computes a path.
    """
    def __init__(self, args):
        self.base_repo = args.model
        self.name = f"{args.model.split('/')[1]}-exl3"
        self.repo = f"{args.org}/{self.name}"
        self.dir = os.path.join(args.workdir, self.name)
        self.main = os.path.join(self.dir, "main")
        # Per-model trace artifacts (calibration generator, calibration trace, eval traces).
        # Not a revision: a leading underscore keeps it out of revisions(), so never uploaded.
        self.traces = os.path.join(self.dir, "_traces")
        self.cal_data = os.path.join(self.traces, "cal.safetensors")
        # Uncalibrated sweep: card section 2's common baseline. Kept, not uploaded.
        self.baseline = os.path.join(self.dir, "_baseline")

    def revdir(self, rev):
        return os.path.join(self.dir, rev)

    def revisions(self):
        if not os.path.isdir(self.dir):
            return []
        return sorted(r for r in os.listdir(self.dir)
                      if r not in ('main', 'logs') and not r.startswith('_')
                      and os.path.isdir(self.revdir(r)))

    def baseline_revisions(self):
        if not os.path.isdir(self.baseline):
            return []
        return sorted(r for r in os.listdir(self.baseline)
                      if os.path.isfile(os.path.join(self.baseline, r, "quantization_config.json")))

    def eval_trace(self, slice_):
        return os.path.join(self.traces, f"eval_{slice_}.json")

    def log(self, name):
        return os.path.join(self.dir, "logs", f"{name}.log")

    def is_complete(self, rev):
        # A revision is finished when convert.py wrote its config; main is finished
        # when the aggregate card has been rendered. README.md alone cannot say this:
        # convert.py copies the base model's README into every revision, so it is
        # present long before anything has been generated.
        marker = "README.md" if rev == 'main' else "quantization_config.json"
        return os.path.isfile(os.path.join(self.revdir(rev), marker))


# The conversational calibration mix (docs/calibration.md): generated once per model by an
# uncalibrated 8bpw -- as good a generator as a calibrated quant, and it brings no calibration
# of its own -- with thinking on at medium effort and 1024 new tokens, the setup measured.
CAL_TEMPLATE_VARS = '{"enable_thinking": true, "reasoning_effort": "medium"}'
CAL_MAX_NEW_TOKENS = 1024
# Calibration mix shares. No agent share: real coding-agent sessions (mini-swe-agent on
# swe-rebench-v2) at 0.20 bought ~5% on swe for 3% on wild on Ornith-9B, 2026-10-05
# (docs/calibration.md); ctx_trace.py keeps the slice for a work-pattern source
CAL_SHARES = {"raw": 0.25, "ctx": 0.35, "loop": 0.25, "self": 0.10, "random": 0.05}
# English document kinds for the ctx and loop slices, drawn equally. diff: CommitPackFT commits
# as `git log -p` shows them; 21% less error on held-out diffs, nothing else moved (Ornith-9B,
# 2026-10-05, docs/calibration.md "Diffs as a calibration document kind")
CAL_DOC_KINDS = "web,wiki,technical,code,diff"
# The card's composite is the independent tier only (TODO.md card-composite): real users'
# first prompts (WildChat) and real coding-agent sessions (Open-SWE-Traces), as
# (conversations, weight). Eval traces come from bf16 (or an FP8 release): a calibration scores
# worse on another quant's sampled text, and no pipeline quant may write the text it is judged
# on. swe is 200 sessions: at 40, three rows carried half the excess and the draw spread was
# mostly which rows were drawn (docs/calibration.md); qbench streams its 2M+ context tokens in
# groups that fit the card
EVAL_SLICES = {"wild": (60, 0.80), "swe": (200, 0.20)}
EVAL_SEED = 1
# Diagnostic slices, constructed by ctx_trace.py with scaffolding disjoint from calibration's:
# never in the composite, only in the card's worst-slice check against a reference ladder
DIAG_SLICES = {"ctx_user": "documents in a user turn", "ctx_tool": "documents as tool results",
               "ctx_ml": "multilingual documents", "loop": "tool-use loops", "self": "the model's own voice"}
DIAG_N = 30
HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_SELF_PROMPTS = os.path.join(HERE, "eval_self_prompts.json")       # held out of calibration
CODE_DOCS = os.path.join(HERE, "..", "deps", "vllm", "vllm", "**", "*.py")   # code documents for ctx slices


def eval_generator_named(job):
    """The repo that wrote the eval traces' answers, when it is not the base model itself (e.g. an
    FP8 release generated through vLLM); None for the base model or no trace."""
    path = job.eval_trace(next(iter(EVAL_SLICES)))
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        model = json.load(f).get("model", "")
    m = re.search(r"models--([^/]+?)--([^/]+)/snapshots/", model)
    repo = f"{m.group(1)}/{m.group(2)}" if m else model
    return None if repo == job.base_repo else repo

def eval_generator(job, args):
    """(model path, backend args) for the eval traces. Never an EXL3 checkpoint: eval text from a
    quant scores one calibration against another quant's sampled text (docs/calibration.md)."""
    src = args.eval_generator or job.base_repo
    path = src if os.path.isdir(src) else snapshot_download(repo_id=src)
    qc = json.load(open(os.path.join(path, "config.json"))).get("quantization_config") or {}
    if qc.get("quant_method") == "exl3" or os.path.isfile(os.path.join(path, "quantization_config.json")):
        raise SystemExit(f"--eval-generator {src} is an EXL3 checkpoint; eval traces must come from the base model or an FP8/bf16 release")
    backend = ["--backend", "vllm", "--vllm", args.vllm_args] if args.eval_backend == "vllm" else []
    return path, backend

def collect_metadata(job, qbench):
    """qbench: {slice: results list}. Excess KLD is each quant's KLD minus the slice's noise
    floor (bf16 rounding noise), so slices with different floors add up meaningfully."""
    data = { 'this_model': job.repo,
             'base_model': job.base_repo }

    base_sizes = tensor_survey_remote(job.base_repo, 'main')
    data['base_disk_bytes'] = base_sizes['total']
    data['multimodal'] = base_sizes['encoder'] > 0

    data['eval_generator'] = eval_generator_named(job)

    floor = {sl: parse_qbench(res, 'Noise floor')['kld'] for sl, res in qbench.items()}
    ref_ppl = {sl: next(r['ppl'] for r in res if r.get('group') == 'reference')
               for sl, res in qbench.items()}
    composite = [sl for sl in EVAL_SLICES if sl in qbench]
    data['composite'] = len(composite) == len(EVAL_SLICES)
    data['base_raw_ppl'] = ref_ppl.get('raw')

    revs = []
    for rev in job.revisions():
        sizes = tensor_survey_local(job.revdir(rev))
        quant = quant_config(job, rev)
        if not quant:
            continue
        qb = {sl: parse_qbench(res, rev) for sl, res in qbench.items()}
        if any(v is None for v in qb.values()):
            continue
        excess = {sl: qb[sl]['kld'] - floor[sl] for sl in qb}
        r = { 'name': rev,
              'bits': quant['bits'],
              'head_bits': quant.get('head_bits'),
              'calibration': calibration_of(job, rev),
              'disk_bytes': sizes['total'],
              'embed_bytes': sizes['embed'],
              'encoder_bytes': sizes['encoder'],
              'blockq_bytes': int(sizes['embed'] / 16 * 4.5),
              'excess': excess,
              'raw_excess': excess.get('raw'),
              'raw_ppl': qb['raw']['ppl'] if 'raw' in qb else None }
        if data['composite']:
            r['composite'] = sum(EVAL_SLICES[sl][1] * excess[sl] for sl in composite)
            r['dppl'] = 100 * sum(EVAL_SLICES[sl][1] * (qb[sl]['ppl'] / ref_ppl[sl] - 1)
                                  for sl in composite)
        revs.append(r)
    data['revisions'] = sorted(revs, key=lambda x: x['bits'])
    data['reference'] = reference_section(job, qbench, floor, data) if data['composite'] else None
    return data

def ref_label(repo, branch):
    return f"ref:{repo}@{branch}"

def baseline_label(rev):
    return f"baseline:{rev}"

def ladders_of(branches):
    """Group a reference repo's branches into ladders by name: the part before the bit rate
    ('SC_4.00bpw_H5' -> 'SC'; '4.0bpw', '2.75bpw_H5' -> ''). Branches ending in _V<n> are
    vision-quantized twins of another rung (turboderp's SC_4.00bpw_H5_V6), skipped by name: the
    publisher's naming implies the text weights are the same as the twin's."""
    out = {}
    for b in sorted(branches):
        if b == "main" or re.search(r"_V\d+$", b):
            continue
        out.setdefault(re.sub(r"_?\d+(\.\d+)?bpw.*$", "", b), []).append(b)
    return out

def load_reference_info(job):
    """main/reference.json: the reference repos qbench scored, by ladder, and the uncalibrated
    baseline sweep. Accepts the earlier single-repo format."""
    path = os.path.join(job.main, "reference.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        info = json.load(f)
    if "repo" in info:   # 2026-10-03 format: one repo, labels ref/<branch>
        info = {"references": [{"repo": info["repo"], "ladders": {"": info["revisions"]}}], "baseline": {}}
    return info

def reference_section(job, qbench, floor, data):
    """Section 2 of the card. Plot: every ladder -- this card's and each reference's -- as excess
    KLD relative to the uncalibrated baseline at equal body size, so no ladder is drawn as a
    perfect line and all share one denominator (its draw noise is common-mode: order and gaps
    between ladders survive). Table: for each revision of ours, the most competitive rung at the
    same bit rate across all reference ladders, or, where none has that rate, the most
    competitive ladder interpolated at equal body size (asterisked). Raw text stays out, as on
    the rest of the card."""
    info = load_reference_info(job)
    if not info:
        return None
    import numpy as np
    slices = [sl for sl in qbench if sl != "raw"]
    # Sizes as on the rest of the card (minus input embeddings), the total for the size column,
    # and the body (minus the output head too) for anything interpolated: head bits can change
    # between rungs (turboderp's 2.75bpw_H5 -> 3.0bpw), and those bytes buy little
    def survey(path):
        sz = tensor_survey_local(path)
        with open(os.path.join(path, "quantization_config.json")) as f:
            q = json.load(f)
        return { 'size': (sz['total'] - sz['embed']) / 1024**3, 'total': sz['total'] / 1024**3,
                 'body': (sz['total'] - sz['embed'] - sz['head']) / 1024**3,
                 'bits': q.get('bits'), 'head_bits': q.get('head_bits') }
    def arm(label, path, need):
        rows = {sl: parse_qbench(qbench[sl], label) for sl in need}
        if any(r is None for r in rows.values()):
            return None
        ex = {sl: rows[sl]['kld'] - floor[sl] for sl in rows}
        return { **survey(path), 'label': label, 'excess': ex,
                 'composite': sum(w * ex[sl] for sl, (_, w) in EVAL_SLICES.items()) }
    def interp(ladder, x):
        pts = sorted((r['body'], r['composite']) for r in ladder)
        if len(pts) < 2 or not pts[0][0] * 0.98 <= x <= pts[-1][0] * 1.02:
            return None
        return float(np.exp(np.interp(x, [p[0] for p in pts], np.log([p[1] for p in pts]))))

    ours = {r['name']: arm(r['name'], job.revdir(r['name']), slices) for r in data['revisions']}
    ours = {k: v for k, v in ours.items() if v}
    if not ours:
        return None
    lo = min(v['body'] for v in ours.values()) * 0.9; hi = max(v['body'] for v in ours.values()) * 1.1
    ladders = []   # (display name, [rungs])
    for ref in info["references"]:
        named = [n for n in ref["ladders"]]
        for name, branches in ref["ladders"].items():
            rungs = [arm(ref_label(ref["repo"], b), p, slices) for b, p in branches.items()]
            for r, b in zip(rungs, branches):
                if r:
                    r['branch'] = b
            # rungs far outside this card's size range (SC's 1.4-1.8 bpw against a 2-6 bpw set) add nothing
            rungs = [r for r in rungs if r and lo <= r['body'] <= hi]
            if rungs:
                disp = ref["repo"] if len(named) == 1 else f"{ref['repo']} ({name or 'plain'})"
                ladders.append((disp, rungs))
    if not ladders:
        return None
    base = [a for a in (arm(baseline_label(rev), p, list(EVAL_SLICES)) for rev, p in info.get("baseline", {}).items()) if a]

    def describe_slice(sl):
        return DESCRIBE.get(sl) or DIAG_SLICES.get(sl, sl)
    rows = []
    for r in data['revisions']:
        o = ours.get(r['name'])
        if not o:
            continue
        row = { 'name': r['name'], 'bits': r['bits'], 'size': o['size'], 'vs': None, 'interpolated': False }
        # same nominal bit rate, preferring the same head bits within a ladder; best across ladders
        same = []
        for disp, rungs in ladders:
            m = [x for x in rungs if x['bits'] is not None and abs(x['bits'] - r['bits']) < 0.01]
            if m:
                pick = min(m, key=lambda x: (x['head_bits'] != o['head_bits'], x['composite']))
                same.append((pick['composite'], disp, pick))
        if same:
            _, disp, best = min(same, key=lambda t: t[0])
            row['vs'] = o['composite'] / best['composite']
            rel = {sl: o['excess'][sl] / best['excess'][sl] for sl in slices}
            worst = max(rel, key=rel.get)
            d = o['total'] - best['total']
            row.update({ 'against': f"{disp}: {best['branch']}",
                         'size_delta': f"{d:+.2f} GiB" if abs(d) >= 0.1 else f"{d * 1024:+.0f} MiB",
                         'size_delta_pct': 100 * (o['total'] / best['total'] - 1),
                         'worst': rel[worst], 'worst_desc': describe_slice(worst) })
        else:
            cands = [(v, disp) for disp, rungs in ladders if (v := interp(rungs, o['body'])) is not None]
            if cands:
                v, disp = min(cands)
                row.update({ 'vs': o['composite'] / v, 'interpolated': True, 'against': disp })
        rows.append(row)

    traces = []
    if len(base) >= 2:
        for disp, rungs in [("this card", list(ours.values()))] + ladders:
            pts = [(x['size'], x['composite'] / b) for x in sorted(rungs, key=lambda x: x['body'])
                   if (b := interp(base, x['body'])) is not None]
            if pts:
                traces.append({ 'name': disp, 'points': pts, 'ours': disp == "this card" })
    return { 'rows': rows, 'traces': traces, 'repos': [r["repo"] for r in info["references"]],
             'baseline': len(base) >= 2, 'slices': slices, 'slice_descs': [describe_slice(sl) for sl in slices] }

def calibration_of(job, rev):
    path = os.path.join(job.revdir(rev), "calibration.json")
    if not os.path.isfile(path):
        return "unrecorded"
    with open(path) as f:
        return json.load(f).get("calibration", "unrecorded")

def tensor_category(name):
    m = f".{name}."
    if re.search(r"\.(vision|visual|vision_tower|vision_adapter|"
                 r"vision_projection|mm_projector|multi_modal|audio_tower)\.", m):
        return "encoder"
    if re.search(r"\.embed_tokens\.", m):
        return "embed"
    if re.search(r"\.lm_head\.", m) and not name.startswith("mtp."):
        return "head"
    return None

def quant_config(job, revision):
    conf_file = "quantization_config.json"
    conf_path = os.path.join(job.revdir(revision), conf_file)
    if os.path.isfile(conf_path):
        with open(conf_path, "r") as f:
            return json.load(f)
    return None

def tensor_survey(tensors):
    out = { 'total': 0,
            'embed': 0,
            'encoder': 0,
            'head': 0 }
    for name, size in tensors:
        out['total'] += size
        cat = tensor_category(name)
        if cat:
            out[cat] += size
    return out

def tensor_survey_local(path):
    def tensors():
        for shard in sorted(globmod.glob(os.path.join(path, "*.safetensors"))):
            with open(shard, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                header = json.loads(f.read(n))
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                a, b = meta["data_offsets"]
                yield name, b - a
    return tensor_survey(tensors())

def tensor_survey_remote(repo, revision):
    def tensors():
        meta = HfApi().get_safetensors_metadata(repo, revision=revision)
        for f in meta.files_metadata.values():
            for name, tensor in f.tensors.items():
                a, b = tensor.data_offsets
                yield name, b - a
    return tensor_survey(tensors())
    
def has_chat_template(repo):
    """Does the base repo ship a chat template (standalone file or inline)?

    Base models often do not (meta-llama/Llama-3.2-3B), and qbench's `render` mode
    raises on them. Raw text is the right instrument for those anyway: it is the
    distribution they were trained on.
    """
    # From the local snapshot, which every stage downloads anyway: no Hub API call, so it
    # works under HF_HUB_OFFLINE=1 and on a host whose route to the Hub is flaky
    path = snapshot_download(repo_id=repo)
    if any(os.path.isfile(os.path.join(path, f)) for f in ("chat_template.jinja", "chat_template.json")):
        return True
    tc = os.path.join(path, "tokenizer_config.json")
    if os.path.isfile(tc):
        with open(tc) as f:
            return bool(json.load(f).get("chat_template"))
    return False

def parse_qbench(qbench, label):
    for res in qbench:
        if res['label'] == label:
            return res
    return None

def do_upload(job, args):
    api = HfApi()
    api.create_repo(repo_id=job.repo,
                    private=args.private,
                    repo_type='model',
                    exist_ok=True)

    for rev in ['main'] + job.revisions():
        if not job.is_complete(rev):
            continue
        print(f"=== UPLOADING {job.revdir(rev)} ===")
        api.create_branch(repo_id=job.repo,
                          branch=rev,
                          exist_ok=True)
        api.upload_folder(folder_path=job.revdir(rev),
                          repo_id=job.repo,
                          revision=rev,
                          # qbench project files hold local paths; their results are published
                          ignore_patterns=["qbench-*.yaml"])

def plot_quality_vs_size(job, meta, path):
    """Section 1 of the card: this card's own checkpoints, quality against size, no comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2 = "#1f1f1f", "#ffffff", "#aaaaaa"
    AXES, GRID, GRID2, LINE = "#555555", "#555555", "#444444", "#2a78d6"
    key = 'composite' if meta['composite'] else 'raw_excess'
    revs = [r for r in meta['revisions'] if r.get(key)]
    xs = [(r['disk_bytes'] - r['embed_bytes']) / 1024**3 for r in revs]
    ys = [r[key] for r in revs]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.grid(True, which="minor", color=GRID2, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXES)
    ax.tick_params(colors=INK2)
    ax.plot(xs, ys, color=LINE, lw=2, zorder=2)
    ax.scatter(xs, ys, s=70, color=LINE, edgecolor=SURF, linewidth=1.5, zorder=3)
    for r, x, y in zip(revs, xs, ys):
        ax.annotate(f"{r['bits']:.2f} bpw", (x, y), xytext=(8, 4), textcoords="offset points",
                    fontsize=8, color=INK)
        ax.annotate(f"{y:0.4f}", (x, y), xytext=(8, -4), textcoords="offset points",
                    fontsize=6, color=INK)
    ax.set_yscale("log")
    ax.margins(x=0.1)
    ax.set_xlabel("weight size (GiB, excluding input embeddings)", color=INK2)
    ax.set_ylabel("excess KLD vs bf16 (log, lower is better)", color=INK2)
    fig.text(0.1, 0.97, f"{job.name}: quality vs size", fontsize=11.5, color=INK, va="top")
    sub = ("composite: " + " + ".join(f"{w:.2f} {DESCRIBE[sl]}" for sl, (_, w) in EVAL_SLICES.items())
           if meta['composite'] else "raw web text (this model has no chat template)")
    fig.text(0.1, 0.915, sub, fontsize=7.5, color=INK2, va="top")
    fig.subplots_adjust(top=0.86, left=0.13, right=0.97, bottom=0.13)
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)

DESCRIBE = {"wild": "real user prompts (WildChat)", "swe": "real agent sessions (Open-SWE-Traces)"}

def plot_vs_reference(job, ref, path):
    """Section 2 of the card: each ladder relative to the uncalibrated baseline at equal size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, AXES, GRID = "#1f1f1f", "#ffffff", "#aaaaaa", "#555555", "#555555"
    OURS, BASE = "#eb6834", "#888888"
    OTHERS = ["#2a78d6", "#3fb27f", "#b07fd6", "#d6b12a"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXES)
    ax.tick_params(colors=INK2)
    ax.axhline(1.0, color=BASE, lw=1.5, ls=(0, (2, 2)), zorder=1, label="uncalibrated (baseline)")
    ys_all = [1.0]
    others = iter(OTHERS)
    for t in ref['traces']:
        xs, ys = zip(*t['points']); ys_all += ys
        color = OURS if t['ours'] else next(others)
        ax.plot(xs, ys, color=color, lw=2, zorder=3 if t['ours'] else 2,
                marker="D" if t['ours'] else "o", markersize=6, markeredgecolor=SURF, label=t['name'])
        if t['ours']:
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.2f}", (x, y), xytext=(0, -13), textcoords="offset points",
                            fontsize=7, color=INK, ha="center")
    lo, hi = min(ys_all), max(ys_all)
    pad = max(0.03, (hi - lo) * 0.12)
    ax.set_ylim(lo - pad, hi + pad)
    ax.margins(x=0.06)
    ax.set_xlabel("weight size (GiB, excluding input embeddings)", color=INK2)
    ax.set_ylabel("excess KLD relative to\nuncalibrated at equal size", color=INK2)
    leg = ax.legend(frameon=False, fontsize=7.5, loc="best")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.text(0.13, 0.97, "Against other quants of this model", fontsize=11, color=INK, va="top")
    fig.subplots_adjust(top=0.89, left=0.15, right=0.97, bottom=0.14)
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)

# Published in main/traces, so a set can be reproduced: (file in _traces, published name)
PUBLISHED_TRACES = [("cal.safetensors", "calibration.safetensors"),
                    ("cal.manifest.json", "calibration.manifest.json"),
                    ("cal_cal.json", "calibration.json"),
                    ("sampling.json", "sampling.json")] + \
                   [(f"eval_{sl}.json", f"eval_{sl}.json") for sl in EVAL_SLICES]

DATA_NOTICE = """# Data in this directory

These files make the quantizations in this repository reproducible. They contain token ids
(the base model's tokenizer) of text drawn from the sources below, plus answers generated by
the model itself. Redistributed under each source's terms; the share-alike terms of the
Wikipedia text apply to the files that contain it, not to the model weights.

| file | what it is | sources |
|---|---|---|
| `calibration.safetensors` | the packed calibration rows passed to exllamav3's `convert.py --cal_data` | as `calibration.json`, plus exllamav3's bundled calibration corpus |
| `calibration.json` | the conversational calibration trace before packing (documents, commits as diffs, tool loops, the model's own answers) | exllamav3's bundled corpus (C4, Wikipedia, code, technical text); Wikipedia 20231101 in 13 languages ([wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia), CC BY-SA 3.0 / GFDL); commits from [bigcode/commitpackft](https://huggingface.co/datasets/bigcode/commitpackft) (MIT; only samples whose repository license is permissive, each under that license) |
| `calibration.manifest.json` | slice shares and composition of the packed rows | - |
| `sampling.json` | how every trace here was sampled: modes and the rules picking one per conversation; each trace row records its mode | the model publisher's recommendations (source inside) |
| `eval_wild.json` | the card's real-user-prompt eval: first user turns, answered by the unquantized model | [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) (ODC-BY) |
| `eval_swe.json` | the card's agent-session eval: sessions cut at a turn the unquantized model rewrites | [nvidia/Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) (CC BY 4.0) |

Built with `quantization/quant.py traces` in [vllm-exl3-plugin](https://github.com/yeasah/vllm-exl3-plugin).
"""

def publish_traces(job):
    """Copy the trace artifacts into main/traces under stable names; False if there are none."""
    have = [(src, dst) for src, dst in PUBLISHED_TRACES if os.path.isfile(os.path.join(job.traces, src))]
    if not have:
        return False
    out = os.path.join(job.main, "traces")
    os.makedirs(out, exist_ok=True)
    for src, dst in have:
        shutil.copyfile(os.path.join(job.traces, src), os.path.join(out, dst))
    with open(os.path.join(out, "DATA_NOTICE.md"), "w") as f:
        f.write(DATA_NOTICE)
    return True

def do_card(job, args):
    os.makedirs(job.main, exist_ok=True)

    qbench = {}
    for sl in list(EVAL_SLICES) + list(DIAG_SLICES) + ["raw"]:
        path = os.path.join(job.main, f"qb_{sl}.json")
        if os.path.isfile(path):
            with open(path) as f:
                qbench[sl] = json.load(f)
    if 'raw' not in qbench:
        print("=== no qbench results: run qbench first ===")
        return False
    meta = collect_metadata(job, qbench)
    plot_quality_vs_size(job, meta, os.path.join(job.main, "quality_vs_size.png"))
    meta['traces_published'] = publish_traces(job)
    if meta['reference'] and meta['reference']['traces']:
        plot_vs_reference(job, meta['reference'], os.path.join(job.main, "vs_reference.png"))

    # get base metadata and add content
    base_card = ModelCard.load(job.base_repo)
    base_metadata = base_card.data.to_dict()
    base_metadata['base_model'] = job.base_repo
    base_metadata['base_model_relation'] = 'quantized'
    if 'tags' not in base_metadata:
        base_metadata['tags'] = []
    base_metadata['tags'].append('exl3')
    card_data = ModelCardData(**base_metadata)
    card = ModelCard.from_template(card_data, template_path=args.template,
                                   show_ppl=args.ppl, eval_slices=EVAL_SLICES, **meta)
    card.save(os.path.join(job.main, "README.md"))

def logit_cache_dir(job, args):
    """Each model owns its logit cache by default (<model>-exl3/_logit_cache, never uploaded), so
    pruning what one model's bench series did not use cannot touch another model's entries."""
    return os.path.abspath(args.logit_cache or os.path.join(job.dir, "_logit_cache"))

def qbench_project(job, args, slice_, arms):
    """One qbench project per slice: a bf16-generated eval trace, or raw web text. arms: [(label, path)]"""
    project = { "title": f"{job.name}: {slice_}",
                "logit_cache": { "dir": logit_cache_dir(job, args),
                                 "max_size_gb": args.logit_cache_size },
                "models": [ { "label": "HF BF16", "group": "reference", "engine": "transformers",
                              "repo": job.base_repo, "options": { "streaming": True } } ] +
                          [ { "label": label, "group": "EXL3", "engine": "exllamav3",
                              "source": os.path.abspath(path) } for label, path in arms ],
                "output": { "results": os.path.abspath(os.path.join(job.main, f"qb_{slice_}.json")) } }
    if slice_ == "raw":
        # A guard, not a target: raw web text framed as nothing matches no real use
        project["test_data"] = { "source": "openwebtext10k", "rows": 50, "length": 2048, "stride": 2048 }
        project["tokenizer"] = { "repo": job.base_repo, "template": False }
    else:
        project["test_trace"] = os.path.abspath(job.eval_trace(slice_))
    return project

def do_qbench(job, args):
    os.makedirs(job.main, exist_ok=True)
    revisions = [rev for rev in job.revisions() if job.is_complete(rev)]
    if not revisions:
        print("=== no revisions found ===")
        return False
    slices = [sl for sl in list(EVAL_SLICES) + list(DIAG_SLICES) if os.path.isfile(job.eval_trace(sl))] + ["raw"]
    missing = [sl for sl in EVAL_SLICES if sl not in slices]
    if missing:
        print(f"=== no eval traces for {', '.join(missing)}: the card falls back to raw text "
              f"(run `traces` first for the composite) ===")
    arms = [(rev, job.revdir(rev)) for rev in revisions]
    # Existing sets of this model's quants (each grouped into ladders by branch name), and the
    # uncalibrated baseline sweep from `baseline`: section 2 of the card
    references = []
    for repo in args.reference or []:
        branches = [b.name for b in HfApi().list_repo_refs(repo).branches]
        ladders = {name: {b: snapshot_download(repo, revision=b) for b in bs}
                   for name, bs in ladders_of(branches).items()}
        references.append({"repo": repo, "ladders": ladders})
        arms += [(ref_label(repo, b), p) for bs in ladders.values() for b, p in bs.items()]
    baseline = {rev: os.path.join(job.baseline, rev) for rev in job.baseline_revisions()}
    if references or baseline:
        with open(os.path.join(job.main, "reference.json"), "w") as f:
            json.dump({"references": references, "baseline": baseline}, f, indent=1)
    base_arms = [(baseline_label(rev), p) for rev, p in baseline.items()]
    print(f"=== running bench on {' '.join(label for label, _ in arms + base_arms)}; slices: {' '.join(slices)} ===")
    script = os.path.join(args.exllamav3dir, "eval", "qbench.py")
    series_start = datetime.datetime.now().replace(microsecond=0)
    for sl in slices:
        qbench_file = os.path.join(job.main, f"qbench-{sl}.yaml")
        with open(qbench_file, "w") as f:
            # the baseline only enters through the composite
            yaml.safe_dump(qbench_project(job, args, sl, arms + (base_arms if sl in EVAL_SLICES else [])),
                           f, sort_keys=False)
        rc = run_logged([ "python3", script, qbench_file, "-d", str(args.device) ], job.log(f"qbench-{sl}"))
        if rc:
            print(f"=== qbench on {sl} FAILED (exit {rc}) ===")
            return False
    # The series succeeded: reference logits it did not use (a superseded eval trace, a dropped
    # reference) can only be dead weight. Results and KL vectors are never pruned.
    if args.prune:
        print(f"=== pruning reference logits unused since {series_start.isoformat()} ===")
        run_logged([ "python3", os.path.join(args.exllamav3dir, "eval", "qbench_prune.py"),
                     logit_cache_dir(job, args), "--unused-since", series_start.isoformat() ], job.log("qbench-prune"))
    return True

def run_logged(cmd, log_path, env=None):
    """Run cmd, echoing its output to the terminal and appending it to log_path.

    convert.py's `!!` warnings (non-finite rows, Cholesky retries, fallbacks) and its
    per-layer error lines are the only record of how a conversion went; the checkpoint
    keeps none of it. Raw bytes are copied, so the converter's progress bars still render.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    sys.stdout.flush()  # our own === lines must reach the terminal before the child's output
    with open(log_path, "ab") as log:
        log.write(f"\n=== {' '.join(cmd)}\n".encode())
        log.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env=None if env is None else {**os.environ, **env})
        while chunk := os.read(proc.stdout.fileno(), 65536):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log.write(chunk)
            log.flush()
        return proc.wait()

def run_until(output, cmd, log_path, tries, env=None, cleanup=None):
    """Run cmd until it has produced output, at most `tries` times."""
    for attempt in range(1, tries + 1):
        if os.path.exists(output):
            return True
        if attempt > 1:
            print(f"=== ATTEMPT {attempt} of {tries} for {output} ===")
            if cleanup:
                cleanup()
        run_logged(cmd, log_path, env)
    return os.path.exists(output)

SAMPLING_DIR = os.path.join(HERE, "sampling")

def resolve_sampling_profile(job, args, path):
    """The sampling profile every trace is generated with, written to _traces/sampling.json:
    an explicit --sampling-profile; else a checked-in quantization/sampling/<org>__<model>.json
    (per-mode recommendations, which live only in model-card prose); else one mode from the
    base model's generation_config.json. With none of those, refuse rather than fall back to
    exllamav3's default, which is no model's recommendation -- unless asked for explicitly."""
    out = os.path.join(job.traces, "sampling.json")
    if args.sampling_profile == "exllamav3-default":
        return None
    src = args.sampling_profile or os.path.join(SAMPLING_DIR, job.base_repo.replace("/", "__") + ".json")
    if os.path.isfile(src):
        shutil.copyfile(src, out)          # contents only: some volumes refuse chmod/utime
        return out
    if args.sampling_profile:
        raise FileNotFoundError(src)
    gc = os.path.join(path, "generation_config.json")
    keys = ("temperature", "top_k", "top_p", "min_p", "repetition_penalty")
    if os.path.isfile(gc):
        with open(gc) as f:
            g = json.load(f)
        mode = {k: g[k] for k in keys if k in g}
        if mode:
            with open(out, "w") as f:
                json.dump({"source": f"{job.base_repo} generation_config.json", "modes": {"default": mode},
                           "rules": [{"mode": "default"}]}, f, indent=1)
            return out
    raise SystemExit(f"=== no sampling profile for {job.base_repo}: no {src}, and the base model ships no "
                     f"generation_config.json with sampling values. Write a profile from its model card, "
                     f"or pass --sampling-profile exllamav3-default to use exllamav3's default ===")

def do_traces(job, args):
    """Per-model trace artifacts: the calibration generator (an uncalibrated 8bpw), the
    conversational calibration trace it writes, and the bf16-generated eval traces the card's
    composite is scored on. Each step is skipped when its output already exists."""
    if not has_chat_template(job.base_repo):
        print(f"=== {job.base_repo} has no chat template: conversational traces do not apply; "
              f"quantize with --calibration default ===")
        return False
    path = snapshot_download(repo_id=job.base_repo)
    os.makedirs(job.traces, exist_ok=True)
    one_gpu = {"CUDA_VISIBLE_DEVICES": str(args.device)}
    profile = resolve_sampling_profile(job, args, path)
    sampling = ["--sampling_profile", profile] if profile else []
    print(f"=== sampling: {profile or 'exllamav3 default (explicitly requested)'} ===")
    tries = args.retries + 1

    gen = os.path.join(job.traces, "gen-8bpw-uncal")
    gen_work = os.path.join(job.traces, "_work-gen")
    print(f"=== calibration generator: uncalibrated 8bpw -> {gen} ===")
    if not run_until(os.path.join(gen, "quantization_config.json"),
                     [ "python3", os.path.join(args.exllamav3dir, "convert.py"),
                       "-b", "8", "-hb", "8", "-vb", "16", "--uncalibrated",
                       "-i", path, "-w", gen_work, "-o", gen, "-d", "0" ],
                     job.log("traces-gen"), tries, env=one_gpu,
                     cleanup=lambda: shutil.rmtree(gen_work, ignore_errors=True)):   # uncalibrated jobs cannot resume
        print("=== calibration generator FAILED ==="); return False
    shutil.rmtree(gen_work, ignore_errors=True)

    ctx_trace = os.path.join(args.exllamav3dir, "ctx_trace.py")

    # bf16 may need every visible GPU; a layer split on stock torch <= 2.14.1 is exposed to
    # pytorch#196258 (docs/upstream.md "PyTorch"), which the local library rebuild fixes
    check = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "torch-c10cuda-fix.sh")
    if subprocess.call([check, "--check"]):
        print("!! torch is not patched for pytorch#196258; a multi-GPU bf16 run may fault (fork free_mem() covers load)")
    eval_path, eval_backend = eval_generator(job, args)
    eval_desc = f"{args.eval_generator or 'bf16'}" + (" via vLLM" if eval_backend else "")
    for sl, (n, _) in EVAL_SLICES.items():
        print(f"=== eval trace ({sl}, {n} conversations, {eval_desc}) -> {job.eval_trace(sl)} ===")
        if not run_until(job.eval_trace(sl),
                         [ "python3", "-u", ctx_trace, "-m", eval_path, "-cs", "65536",
                           "-o", os.path.join(job.traces, "eval"), "--docs", "eval",
                           "--slices", sl, "-n", str(n), "--seed", str(EVAL_SEED) ] + sampling + eval_backend,
                         job.log(f"traces-eval-{sl}"), tries,
                         env=None if args.eval_devices is None else {"CUDA_VISIBLE_DEVICES": args.eval_devices}):
            print(f"=== eval trace {sl} FAILED ==="); return False

    print(f"=== diagnostic eval traces ({', '.join(DIAG_SLICES)}, {DIAG_N} each, {eval_desc}) ===")
    if not run_until(job.eval_trace(list(DIAG_SLICES)[-1]),
                     [ "python3", "-u", ctx_trace, "-m", eval_path, "-cs", "65536",
                       "-o", os.path.join(job.traces, "eval"), "--docs", "eval",
                       "--slices", ",".join(DIAG_SLICES), "-n", str(DIAG_N), "--seed", str(EVAL_SEED),
                       "--code_glob", CODE_DOCS, "--self_from", EVAL_SELF_PROMPTS ] + sampling + eval_backend,
                     job.log("traces-eval-diag"), tries,
                     env=None if args.eval_devices is None else {"CUDA_VISIBLE_DEVICES": args.eval_devices}):
        print("=== diagnostic eval traces FAILED ==="); return False

    # Calibration last: its agent slice must keep out every repository the eval swe slice uses
    print(f"=== calibration trace -> {job.cal_data} ===")
    if not run_until(job.cal_data,
                     [ "python3", "-u", ctx_trace, "-m", gen, "-cs", "65536",
                       "-o", os.path.join(job.traces, "cal"), "--docs", "cal", "--cal_out", job.cal_data,
                       "--shares", json.dumps(CAL_SHARES), "--doc_kinds", CAL_DOC_KINDS,
                       "-tv", CAL_TEMPLATE_VARS, "--max_new_tokens", str(CAL_MAX_NEW_TOKENS),
                       "--exclude_self", EVAL_SELF_PROMPTS, "--exclude_swe_from", job.eval_trace("swe") ] + sampling,
                     job.log("traces-cal"), tries, env=one_gpu):
        print("=== calibration trace FAILED ==="); return False
    decontaminate_self(job)
    return True

def decontaminate_self(job):
    """Drop own-voice eval rows whose prompt the calibration trace also used. Calibration excludes
    the held-out prompts now, but a trace built before that (Ornith-1.5-9B, 2026-10-02) did not."""
    cal_trace = os.path.join(job.traces, "cal_cal.json")
    if not (os.path.isfile(cal_trace) and os.path.isfile(job.eval_trace("self"))):
        return
    with open(cal_trace) as f:
        used = {r.get("conversation") for r in json.load(f)["rows"] if r.get("slice") == "self"}
    with open(job.eval_trace("self")) as f:
        trace = json.load(f)
    overlap = sorted({r["conversation"] for r in trace["rows"]} & used)
    if not overlap:
        return
    trace["rows"] = [r for r in trace["rows"] if r["conversation"] not in used]
    trace["meta"]["dropped_calibration_overlap"] = overlap
    trace["meta"]["rows"] = len(trace["rows"])
    with open(job.eval_trace("self"), "w") as f:
        json.dump(trace, f)
    print(f"!! own-voice eval: dropped prompts {overlap}, which the calibration trace also used")

def do_baseline(job, args):
    """The uncalibrated sweep that card section 2 measures every ladder against: the same bit
    rates as `quantize`, converted with --uncalibrated (no calibration of any kind, so it favors
    no ladder), into <model>-exl3/_baseline. Uncalibrated jobs cannot resume; a failed one restarts."""
    path = snapshot_download(repo_id=job.base_repo)
    for b in args.bits.split(','):
        tb = b.split(':')
        bits, headbits = tb[0], (tb[1] if len(tb) > 1 else "6")
        rev = f"{float(bits):0.2f}bpw" + (f"-H{int(headbits)}" if len(tb) > 1 else "")
        out = os.path.join(job.baseline, rev); work = os.path.join(job.baseline, f"_work-{rev}")
        print(f"=== BASELINE {rev} (uncalibrated) ===")
        if not run_until(os.path.join(out, "quantization_config.json"),
                         [ "python3", os.path.join(args.exllamav3dir, "convert.py"), "--uncalibrated",
                           "-hq", "-b", bits, "-hb", headbits, "-vb", "16",
                           "-i", path, "-w", work, "-o", out, "-d", str(args.device) ],
                         job.log(f"baseline-{rev}"), args.retries + 1,
                         cleanup=lambda: shutil.rmtree(work, ignore_errors=True)):
            print(f"=== BASELINE {rev} FAILED ==="); return False
        shutil.rmtree(work, ignore_errors=True)
    return True

def do_quantize(job, args):
    path = snapshot_download(repo_id=job.base_repo)
    if args.calibration == "mix" and not os.path.isfile(job.cal_data):
        print(f"=== no calibration trace at {job.cal_data}: run `traces` first "
              f"(or --calibration default for the bundled corpus) ===")
        return False
    if args.calibration == "mix":
        with open(os.path.splitext(job.cal_data)[0] + ".manifest.json") as f:
            manifest = json.load(f)
        calibration = { "calibration": "conversational mix",
                        "generator": "uncalibrated 8bpw of the base model",
                        **{k: manifest[k] for k in ("shares", "doc_kinds", "ml_frac", "cal_rows", "cal_cols", "composition")
                           if k in manifest} }
    else:
        calibration = { "calibration": "exllamav3 default corpus" }

    for b in args.bits.split(','):
        tb = b.split(':')
        bits = tb[0]
        headbits = "6"
        rev = f"{float(bits):0.2f}bpw"
        if len(tb) > 1:
            headbits = tb[1]
            rev += f"-H{int(headbits)}"

        revdir=job.revdir(rev)
        workdir=os.path.join(revdir, "_work")
        donefile=os.path.join(revdir, "quantization_config.json")
        script=os.path.join(args.exllamav3dir, "convert.py")
        # Outside revdir on purpose: `upload` publishes each revision directory whole.
        log_path=job.log(rev)

        # A failed attempt resumes from its checkpoint rather than ending the sweep. The
        # usual failure is a host OOM at the head, the job's memory peak, and resuming is
        # safe: a conversion OOM-killed there and resumed came out byte-identical to an
        # uninterrupted one (single GPU, 2026-09-21).
        attempt = 0
        while not os.path.isfile(donefile) and attempt <= args.retries:
            if attempt:
                print(f"=== ATTEMPT {attempt + 1} of {args.retries + 1} FOR {revdir} ===")
            attempt += 1
            if os.path.isfile(os.path.join(workdir, "args.json")):
                print(f"=== RESUMING QUANTIZTION OF {revdir} ===")
                run_logged([ "python3", script,
                             "-w", workdir,
                             "-r",
                             "-d", str(args.device) ], log_path)
            else:
                print(f"=== QUANTIZING {revdir} ({calibration['calibration']}) ===")
                run_logged([ "python3", script,
                             "-hq",
                             "-b", bits,
                             "-hb", headbits,
                             "-vb", "16",
                             "-i", path,
                             "-w", workdir,
                             "-o", revdir,
                             "-d", str(args.device) ] +
                           ([ "--cal_data", job.cal_data ] if args.calibration == "mix" else []), log_path)

        if not os.path.isfile(donefile):
            print(f"=== QUANTIZATION OF {revdir} FAILED after {attempt} attempt(s) ===")
            return False
        else:
            print(f"=== QUANTIZATION OF {revdir} COMPLETE ===")
            if os.path.isdir(workdir):
                shutil.rmtree(workdir)
            # Nothing in an EXL3 checkpoint records its calibration (docs/calibration.md)
            with open(os.path.join(revdir, "calibration.json"), "w") as f:
                json.dump(calibration, f, indent=1)
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--exllamav3dir', default='../deps/exllamav3')
    parser.add_argument('-w', '--workdir', default='work')
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-o', '--org', default='yeasah')
    subparsers = parser.add_subparsers()

    cmd_traces = subparsers.add_parser('traces')
    cmd_traces.add_argument('-d', '--device', default=0,
                            help='GPU for the calibration generator and trace')
    cmd_traces.add_argument('--eval-devices', default=None,
                            help='CUDA_VISIBLE_DEVICES for the bf16 eval traces (default: all visible)')
    cmd_traces.add_argument('--retries', type=int, default=2)
    cmd_traces.add_argument('--eval-generator', default=None,
                            help='repo or path that writes the eval traces\' answers (default: the base model). '
                                 'For models too large to generate from locally in bf16, the FP8 release with '
                                 '--eval-backend vllm. Never an EXL3 checkpoint')
    cmd_traces.add_argument('--eval-backend', default='exllamav3', choices=['exllamav3', 'vllm'],
                            help='vllm: one engine per ctx_trace process, with --vllm-args')
    cmd_traces.add_argument('--vllm-args', default='{}',
                            help='vLLM engine arguments, JSON. Qwen3.8-27B-FP8 on 2x16 GB: '
                                 '\'{"max_num_batched_tokens": 1024, "enforce_eager": true, "gpu_memory_utilization": 0.968}\'')
    cmd_traces.add_argument('--sampling-profile', default=None,
                            help='JSON of sampling modes and per-conversation rules (default: '
                                 'quantization/sampling/<org>__<model>.json, else the base model\'s '
                                 'generation_config.json; "exllamav3-default" for exllamav3\'s own)')
    cmd_traces.set_defaults(func=do_traces)

    cmd_quantize = subparsers.add_parser('quantize')
    cmd_quantize.add_argument('-b', '--bits', default='2:5,3:5,4,5,6')
    cmd_quantize.add_argument('-d', '--device', default=0)
    cmd_quantize.add_argument('--calibration', default='mix', choices=['mix', 'default'],
                              help='mix: the conversational trace from `traces`; default: the bundled corpus')
    cmd_quantize.add_argument('--retries', type=int, default=2,
                              help='resume a failed revision this many times before stopping')
    cmd_quantize.set_defaults(func=do_quantize)

    cmd_baseline = subparsers.add_parser('baseline')
    cmd_baseline.add_argument('-b', '--bits', default='2:5,3:5,4,5,6')
    cmd_baseline.add_argument('-d', '--device', default=0)
    cmd_baseline.add_argument('--retries', type=int, default=2)
    cmd_baseline.set_defaults(func=do_baseline)

    cmd_qbench = subparsers.add_parser('qbench')
    cmd_qbench.add_argument('--logit_cache', default=None,
                            help='logit cache dir (default: <model>-exl3/_logit_cache, owned by this model)')
    # Reference logits are full-vocabulary: ~35 GB for a 60-conversation WildChat trace on a
    # 248k vocabulary, ~150 GB for a full set of slices. Pruning after each series keeps only
    # what the current series used, so the cap can be generous.
    cmd_qbench.add_argument('--logit_cache_size', default=200, type=int, help='cap in GB')
    cmd_qbench.add_argument('--prune', default=True, action=argparse.BooleanOptionalAction,
                            help='after a successful series, evict reference logits it did not use')
    cmd_qbench.add_argument('-d', '--device', default=0)
    cmd_qbench.add_argument('--reference', action='append',
                            help='HF repo of an existing set of this model\'s quants to compare against '
                                 '(card section 2); repeatable')
    cmd_qbench.set_defaults(func=do_qbench)

    cmd_card = subparsers.add_parser('card')
    cmd_card.add_argument('--template', default='templates/model_card.jinja')
    # Raw-text perplexity is meaningless for some models (gpt-oss: ~3600 even for
    # the unquantized reference, on every engine); KLD stays valid, so drop only PPL.
    cmd_card.add_argument('--ppl', default=True,
                          action=argparse.BooleanOptionalAction)
    cmd_card.set_defaults(func=do_card)

    cmd_upload = subparsers.add_parser('upload')
    cmd_upload.add_argument('--private', default=True,
                            action=argparse.BooleanOptionalAction)
    cmd_upload.set_defaults(func=do_upload)

    args = parser.parse_args()
    ok = args.func(Job(args), args)
    sys.exit(0 if ok is not False else 1)

if __name__ == "__main__":
    main()
