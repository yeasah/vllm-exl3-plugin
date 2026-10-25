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
import time
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
        # Uncalibrated sweep: card section 2's common baseline. Not in the model repo; `upload
        # --uncalibrated` puts it (and the generator) in a separate repo, `fetch --uncalibrated`
        # brings them back
        self.baseline = os.path.join(self.dir, "_baseline")
        # The calibration generator, an uncalibrated 8bpw: built by `baseline` (or by `traces` if
        # missing), used by `traces`
        self.generator = os.path.join(self.traces, "gen-8bpw-uncal")
        self.uncal_repo = f"{self.repo}-uncalibrated"
        # Per-tensor sensitivity measurements from `measure`: two ends of the recipe's measurement
        # mix, blended at quantize time (--ownvoice-share), so a new share needs no remeasuring
        self.measure = os.path.join(self.dir, "_measure")

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
# Per-model composite weights, quantization/composite/<org>__<model>.json: a model its publisher
# positions for one use (Qwen3.8-27B: coding and agents) is weighted for that use. Decide them
# from the publisher's description, before seeing which ladder a weighting favors.
COMPOSITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "composite")

def composite_weights(job):
    """(weights {slice: w}, summary or None): the model's file, else EVAL_SLICES' defaults."""
    path = os.path.join(COMPOSITE_DIR, job.base_repo.replace("/", "__") + ".json")
    if not os.path.isfile(path):
        return {sl: w for sl, (_, w) in EVAL_SLICES.items()}, None
    with open(path) as f:
        spec = json.load(f)
    w = spec["weights"]
    assert set(w) == set(EVAL_SLICES) and abs(sum(w.values()) - 1) < 1e-6, f"{path}: weights must cover {list(EVAL_SLICES)} and sum to 1"
    return w, spec.get("summary")
EVAL_SEED = 1
# Diagnostic slices, constructed by ctx_trace.py with scaffolding disjoint from calibration's:
# never in the composite, only in the card's worst-slice check against a reference ladder
DIAG_SLICES = {"ctx_user": "documents in a user turn", "ctx_tool": "documents as tool results",
               "ctx_ml": "multilingual documents", "loop": "tool-use loops", "self": "the model's own voice"}
DIAG_N = 30
# What each eval slice is and what it is for: TECHNICAL.md's slice section
SLICE_DOCS = {
    "wild": ("Real user prompts", "First user turns from WildChat-1M: real people's requests, in many languages, about half "
             "with pasted material. The unquantized model answers; the quant is scored on those answers.",
             "Everyday conversational use."),
    "swe": ("Real agent sessions", "Coding-agent sessions from Open-SWE-Traces: the task, the tool schemas, another agent's "
            "recorded actions and the real tool output, cut at an assistant turn that the unquantized model writes "
            "itself. Repositories are disjoint from any calibration source.",
            "The model's own turns deep inside long agentic contexts."),
    "ctx_user": ("Documents in a user turn", "Held-out web text, Wikipedia and source files pasted into a user turn "
                 "with a question about them. Constructed.", "Reading external text a user supplies."),
    "ctx_tool": ("Documents as tool results", "The same kinds of held-out documents returned by a tool call. "
                 "Constructed, with tool names and phrasing calibration never uses.", "Reading tool output."),
    "ctx_ml": ("Multilingual documents", "Wikipedia articles in 13 languages, in a user turn or as tool results. "
               "Constructed.", "Reading non-English text."),
    "loop": ("Tool-use loops", "Two or three tool calls in a row, each returning a document, then the model's answer. "
             "Constructed.", "Multi-step tool use."),
    "self": ("The model's own voice", "Conversational prompts held out of calibration, answered by the model.",
             "The model's own conversational register: where calibrating on the model's own chat is strongest."),
    "raw": ("Raw web text", "Plain web text (openwebtext), no chat template, scored at every position.",
            "A guard, not a target: no real use looks like this, so it bounds what a conversational "
            "calibration costs on text it does not cover."),
}

RAW_SPEC = { "source": "openwebtext10k", "rows": 50, "length": 2048, "stride": 2048 }   # qbench's raw slice

def trace_stats(job, sl):
    """(rows, context tokens, scored tokens) of an eval slice, or None: from its trace, else
    main/slices.json (recorded whenever the trace was present, published, fetched), else for raw
    its fixed spec (passages scored at every position)."""
    path = job.eval_trace(sl)
    if os.path.isfile(path):
        with open(path) as f:
            meta = json.load(f).get("meta", {})
        return meta.get("rows"), meta.get("input_tokens"), meta.get("output_tokens")
    rec = os.path.join(job.main, "slices.json")
    if os.path.isfile(rec):
        with open(rec) as f:
            got = json.load(f).get(sl)
        if got:
            return tuple(got)
    if sl == "raw":
        n = RAW_SPEC["rows"] * RAW_SPEC["length"]
        return RAW_SPEC["rows"], n, n
    return None

def record_slice_stats(job):
    """main/slices.json: every slice's trace size, kept when a trace is later absent (a fetched
    job has only what was published)."""
    rec = os.path.join(job.main, "slices.json")
    stats = {}
    if os.path.isfile(rec):
        with open(rec) as f:
            stats = json.load(f)
    for sl in list(EVAL_SLICES) + list(DIAG_SLICES) + ["raw"]:
        if (t := trace_stats(job, sl)) is not None:
            stats[sl] = list(t)
    os.makedirs(job.main, exist_ok=True)
    with open(rec, "w") as f:
        json.dump(stats, f, indent=1)
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

def cal_generator(job, args):
    """(backend args) for the calibration trace. Its generator is the uncalibrated 8bpw EXL3, which
    vLLM serves through vllm-exl3-plugin; its engine arguments are its own (--cal-vllm-args), not
    the eval generator's. EXL3 does not split every geometry over TP>1 (Qwen3-0.6B at TP=2 fails):
    '{"tensor_parallel_size": 1}' runs it on one GPU."""
    return ["--backend", "vllm", "--vllm", args.cal_vllm_args] if args.cal_backend == "vllm" else []

def collect_metadata(job, qbench):
    """qbench: {slice: results list}. Excess KLD is each quant's KLD minus the slice's noise
    floor (bf16 rounding noise), so slices with different floors add up meaningfully."""
    data = { 'this_model': job.repo,
             'base_model': job.base_repo }

    base_sizes = tensor_survey_remote(job.base_repo, 'main')
    data['base_disk_bytes'] = base_sizes['total']
    data['multimodal'] = base_sizes['encoder'] > 0

    data['eval_generator'] = eval_generator_named(job)
    data['generation_config'] = published_generation_config(job)[1]

    floor = {sl: parse_qbench(res, 'Noise floor')['kld'] for sl, res in qbench.items()}
    data['floor'] = floor
    ref_ppl = {sl: next(r['ppl'] for r in res if r.get('group') == 'reference')
               for sl, res in qbench.items()}
    composite = [sl for sl in EVAL_SLICES if sl in qbench]
    data['composite'] = len(composite) == len(EVAL_SLICES)
    W, data['composite_note'] = composite_weights(job)
    data['agentic'] = W.get('swe', 0) > W.get('wild', 0)        # the weights put agent work first
    data['eval_slices'] = {sl: (n, W[sl]) for sl, (n, _) in EVAL_SLICES.items()}
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
              # The MTP draft: off by default in both engines (exllamav3 loads it only as a separate
              # draft model), and an external draft model is often the better choice where supported
              'mtp_bytes': sizes.get('mtp', 0),
              'blockq_bytes': int(sizes['embed'] / 16 * 4.5),
              'excess': excess,
              'raw_excess': excess.get('raw'),
              'raw_ppl': qb['raw']['ppl'] if 'raw' in qb else None }
        if data['composite']:
            r['composite'] = sum(W[sl] * excess[sl] for sl in composite)
            r['dppl'] = 100 * sum(W[sl] * (qb[sl]['ppl'] / ref_ppl[sl] - 1)
                                  for sl in composite)
        revs.append(r)
    data['revisions'] = sorted(revs, key=lambda x: x['bits'])
    data['reference'] = reference_section(job, qbench, floor, data) if data['composite'] else None
    data['thresholds'] = []
    for t in load_thresholds(job):
        qb = {sl: parse_qbench(res, threshold_label(t["repo"])) for sl, res in qbench.items()}
        if any(v is None for v in qb.values()):
            print(f"=== threshold {t['repo']} is not scored on every slice: left off the card ===")
            continue
        excess = {sl: qb[sl]['kld'] - floor[sl] for sl in qb}
        sz = tensor_survey_remote(t['repo'], t['revision'])
        data['thresholds'].append({ 'repo': t['repo'], 'name': t['repo'].split("/")[-1], 'excess': excess,
                                    # as the size plot's axis: without input embeddings and MTP
                                    'size': (sz['total'] - sz['embed'] - sz.get('mtp', 0)) / 1024**3,
                                    'composite': sum(W[sl] * excess[sl] for sl in composite) if data['composite'] else None,
                                    'raw_excess': excess.get('raw') })
    return data

def ref_label(repo, branch):
    return f"ref:{repo}@{branch}"

def threshold_label(repo):
    return f"thr:{repo}"

def load_thresholds(job):
    """main/thresholds.json: checkpoints qbench scored to draw as lines, [{repo, revision, engine}]"""
    path = os.path.join(job.main, "thresholds.json")
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return json.load(f)

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

# Which calibration trace a reference ladder was built from, by its publisher's convention (the
# configs do not record one): turboderp's SC_ ladders use the cal_trace.json on main; a set made
# by this pipeline records "conversational mix" per revision and publishes traces/calibration.json
def calibration_trace_of(repo, ladder, branches):
    try:
        if ladder == "SC":
            return hf_hub_download(repo, "cal_trace.json")
        with open(hf_hub_download(repo, "calibration.json", revision=next(iter(branches)))) as f:
            if json.load(f).get("calibration") == "conversational mix":
                return hf_hub_download(repo, "traces/calibration.json")
    except Exception:
        pass
    return None

def eval_prompt_snippets(job, sl, tok):
    """Per eval row, a few short windows of its first user message: what a calibration trace
    containing the same prompt would also contain. From the eval trace; for the self slice
    without one (not published before 2026-10-10), from the held-out list, whose text is
    sc_trace.py's CONVERSATIONS."""
    def windows(text):
        # 50-character windows at a quarter, half and three quarters of the prompt, clamped to it:
        # unclamped, a short prompt's first window started below zero and came out empty, and an
        # empty window is "found" in any text (5 false wild hits against turboderp's SC trace)
        text = text.strip()
        if len(text) < 60:
            return [text] if len(text) >= 20 else []
        out = []
        for f in (0.25, 0.5, 0.75):
            a = min(max(0, int(len(text) * f) - 25), len(text) - 50)
            out.append(text[a:a + 50])
        return [w for w in out if len(w) >= 20]
    path = job.eval_trace(sl)
    if os.path.isfile(path):
        with open(path) as f:
            rows = json.load(f)["rows"]
        out = []
        for r in rows:
            m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", tok.decode(r["input_ids"]), re.S)
            out.append(windows(m.group(1)) if m else [])
        return out if any(out) else None
    if sl == "self":
        sys.path.insert(0, os.path.join(HERE, "..", "deps", "exllamav3"))
        from sc_trace import CONVERSATIONS
        with open(EVAL_SELF_PROMPTS) as f:
            return [windows(CONVERSATIONS[r["conversation"]][0]) for r in json.load(f)["rows"]]
    return None

def contamination_map(job, info, slices):
    """{repo: {ladder: {slice: fraction of eval rows whose prompt is in that ladder's calibration}}},
    for ladders with a known calibration trace; cached in main/contamination.json."""
    cache = os.path.join(job.main, "contamination.json")
    if os.path.isfile(cache):
        with open(cache) as f:
            return json.load(f)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(job.base_repo)
    snippets = {sl: sn for sl in slices if (sn := eval_prompt_snippets(job, sl, tok))}
    out = {}
    for ref in info.get("references", []):
        if ref.get("type") == "gguf":           # imatrix data: not checked yet
            continue
        for ladder, branches in ref["ladders"].items():
            trace = calibration_trace_of(ref["repo"], ladder, branches)
            if not trace:
                continue
            with open(trace) as f:
                text = "\n".join(tok.decode(r.get("input_ids", []) + r.get("response_ids", [])) for r in json.load(f)["rows"])
            out.setdefault(ref["repo"], {})[ladder] = {
                sl: sum(any(w in text for w in ws) for ws in sn if ws) / max(1, sum(1 for ws in sn if ws))
                for sl, sn in snippets.items() }
            hit = {sl: round(v, 2) for sl, v in out[ref["repo"]][ladder].items() if v >= CONTAMINATED}
            print(f"=== calibration of {ref['repo']} ({ladder or 'plain'}) contains eval prompts: {hit or 'none'} ===")
    with open(cache, "w") as f:
        json.dump(out, f, indent=1)
    return out

CONTAMINATED = 0.05     # share of a slice's prompts found in a competitor's calibration

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
    recorded = info.get("sizes", {})
    def survey(path, label=None):
        sz = recorded.get(label) or quant_size(path)
        # The comparison axes count language-model weights only: GGUF keeps the vision encoder in
        # a separate mmproj file and drops the MTP draft, so EXL3's (0.85 GiB + 0.12 GiB on an
        # Ornith-9B checkpoint) would otherwise read as size that buys nothing
        other = sz.get('mtp', 0) + sz.get('encoder', 0)
        return { 'size': (sz['total'] - sz['embed'] - other) / 1024**3, 'total': sz['total'] / 1024**3,
                 'body': (sz['total'] - sz['embed'] - sz['head'] - other) / 1024**3,
                 'bits': sz.get('bits'), 'head_bits': sz.get('head_bits'), 'gguf': path.endswith(".gguf") }
    def arm(label, path, need):
        # The composite's slices are required (every comparison is built on them); any other slice
        # counts where scored -- a reference scored on the eval slices but never on raw text still
        # belongs in the comparison, and dropping it silently emptied the card's section 2
        rows = {sl: parse_qbench(qbench[sl], label) for sl in need}
        if any(rows.get(sl) is None for sl in EVAL_SLICES if sl in need):
            return None
        rows = {sl: r for sl, r in rows.items() if r is not None}
        ex = {sl: rows[sl]['kld'] - floor[sl] for sl in rows}
        return { **survey(path, label), 'label': label, 'excess': ex,
                 'composite': sum(w * ex[sl] for sl, w in composite_weights(job)[0].items()) }
    def interp(ladder, x, sl=None, reach=1.02):
        """A ladder's composite (or one slice's excess) at body size x, log-interpolated. Within
        `reach` of either end it extends the end segment's log-linear slope instead. Log, not linear,
        though a rate between rungs is a per-tensor K mix: on the 27B, the 3.5 bpw rung sits 4% under
        the log value of its 3 and 4 bpw neighbours for both the uncalibrated baseline and the recipe
        ladder (21-22% under the linear mean); only turboderp's published half rungs sit near linear
        (2.5: 9%, 3.5: 3.6% under it), with the same K map as the baseline's."""
        val = (lambda r: r['composite']) if sl is None else (lambda r: r['excess'][sl])
        pts = sorted((r['body'], val(r)) for r in ladder)
        if len(pts) < 2 or not pts[0][0] / reach <= x <= pts[-1][0] * reach or min(p[1] for p in pts) <= 0:
            return None
        xs, ly = [p[0] for p in pts], np.log([p[1] for p in pts])
        if xs[0] <= x <= xs[-1]:
            return float(np.exp(np.interp(x, xs, ly)))
        i = 0 if x < xs[0] else len(xs) - 2
        return float(np.exp(ly[i] + (x - xs[i]) * (ly[i + 1] - ly[i]) / (xs[i + 1] - xs[i])))
    W = composite_weights(job)[0]
    def geo(a, b, skip=()):
        """Weighted geometric mean of per-slice excess ratios a/b over the composite's slices
        ({slice: excess} each): every slice counts at its stated weight, whatever its absolute
        scale (a weighted sum of excesses lets the slice with the larger excess dominate). A
        slice where both sides sit under 2x its noise floor is left out and the rest
        reweighted; if all are, all are used. Returns (ratio, slices left out, all near floor)."""
        keep = [sl for sl in W if max(a[sl], b[sl]) >= 2 * floor[sl] and sl not in skip]
        dropped = [sl for sl in W if sl not in keep and sl not in skip]
        use = keep or list(W)
        tw = sum(W[sl] for sl in use)
        r = float(np.exp(sum(W[sl] / tw * np.log(max(a[sl], 1e-12) / max(b[sl], 1e-12)) for sl in use)))
        return r, (dropped if keep else []), not keep

    every = list(qbench)                    # raw included: TECHNICAL.md shows every slice
    ours = {r['name']: arm(r['name'], job.revdir(r['name']), every) for r in data['revisions']}
    ours = {k: v for k, v in ours.items() if v}
    if not ours:
        return None
    lo = min(v['body'] for v in ours.values()) * 0.9; hi = max(v['body'] for v in ours.values()) * 1.1
    contam = contamination_map(job, info, every)
    ladders = []   # (display name, [rungs])
    for ref in info["references"]:
        named = [n for n in ref["ladders"]]
        for name, branches in ref["ladders"].items():
            rungs = [arm(ref_label(ref["repo"], b), p, every) for b, p in branches.items()]
            for r, b in zip(rungs, branches):
                if r:
                    r['branch'] = b
                    r['contaminated'] = sorted(sl for sl, v in contam.get(ref["repo"], {}).get(name, {}).items()
                                               if v >= CONTAMINATED)
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
    # Ratios are geo(): per-slice ratios, geometrically weighted. A slice where both quants sit
    # within ~2x the unquantized model's own rounding noise compares differences too small to
    # matter (Qwen3.8-27B 6 bpw: swe excess at a third of the floor) and is left out, the row
    # saying so; a row where every slice is is marked as noise, not dropped
    # A slice any compared set was calibrated on is out of the worst-slice column for every row:
    # dropping it only against that set would leave the column measuring different things per row
    tainted = sorted({sl for _, rungs in ladders for x in rungs for sl in x.get('contaminated', ())})
    tainted_by = {sl: sorted({disp for disp, rungs in ladders for x in rungs if sl in x.get('contaminated', ())})
                  for sl in tainted}
    rows = []
    for r in data['revisions']:
        o = ours.get(r['name'])
        if not o:
            continue
        row = { 'name': r['name'], 'bits': r['bits'], 'size': o['size'], 'vs': None, 'interpolated': False,
                'near_floor': False, 'dropped': [] }
        # same nominal bit rate, preferring the same head bits within a ladder; the strongest
        # competitor across ladders is the one our ratio against is highest
        same = []
        for disp, rungs in ladders:
            m = [x for x in rungs if x['bits'] is not None and abs(x['bits'] - r['bits']) < 0.01]
            if m:
                pick = min(m, key=lambda x: (x['head_bits'] != o['head_bits'],
                                             -geo(o['excess'], x['excess'], x.get('contaminated', ()))[0]))
                same.append((geo(o['excess'], pick['excess'], pick.get('contaminated', ())), disp, pick))
        if same:
            (ratio, dropped, allnear), disp, best = max(same, key=lambda t: t[0][0])
            row.update({ 'vs': ratio, 'near_floor': allnear, 'dropped': [describe_slice(sl) for sl in dropped] })
            rel = {sl: o['excess'][sl] / best['excess'][sl] for sl in slices
                   if sl not in tainted and sl in o['excess'] and sl in best['excess']}
            worst = max(rel, key=rel.get)
            d = o['total'] - best['total']
            row.update({ 'against': f"{disp}: {best['branch']}",
                         'size_delta': f"{d:+.2f} GiB" if abs(d) >= 0.1 else f"{d * 1024:+.0f} MiB",
                         'size_delta_pct': 100 * (o['total'] / best['total'] - 1),
                         'worst': rel[worst], 'worst_desc': describe_slice(worst) })
        else:
            cands = []
            for disp, rungs in ladders:
                v = {sl: interp(rungs, o['body'], sl) for sl in W}
                if all(x is not None for x in v.values()):
                    cands.append((geo(o['excess'], v), disp, rungs))
            if cands:
                (ratio, dropped, allnear), disp, rungs = max(cands, key=lambda t: t[0][0])
                row.update({ 'vs': ratio, 'interpolated': True, 'against': disp, 'near_floor': allnear,
                             'dropped': [describe_slice(sl) for sl in dropped] })
                # the worst slice against the same equal-size interpolation (a GGUF ladder is
                # always compared this way, so without it the column would never cover one)
                rel = {sl: o['excess'][sl] / v for sl in slices if sl not in tainted
                       if (v := interp(rungs, o['body'], sl))}
                if rel:
                    worst = max(rel, key=rel.get)
                    row.update({ 'worst': rel[worst], 'worst_desc': describe_slice(worst) })
        rows.append(row)

    traces = []
    if len(base) >= 2:
        for disp, rungs in [("this card", list(ours.values()))] + ladders:
            # The baseline is extrapolated up to the card's own 10% size window, so a rung just past
            # its ends (Q6_K's ~6.6 real bpw against a 6 bpw top) still plots; drawn hollow
            bl, bh = min(b['body'] for b in base), max(b['body'] for b in base)
            ours_trace = disp == "this card"
            pts = []
            for x in sorted(rungs, key=lambda x: x['body']):
                b = {sl: interp(base, x['body'], sl, reach=1.1) for sl in W}
                if all(v is not None for v in b.values()):
                    # label: our bit rate, or the other set's branch / quant type. Head bits stay out of
                    # it, as they do for ours: a branch name carries a non-default head where its
                    # publisher put one, and TECHNICAL.md states what the body axis credits a big head
                    lab = f"{x['bits']:.2f}" if ours_trace else re.sub(r"(?i)bpw", "", str(x.get('branch') or x['label']))
                    # extrapolated only past the baseline's ends by more than rounding (a recipe rung's
                    # body can come out a hair under its uniform namesake's)
                    pts.append((x['body'], geo(x['excess'], b)[0], not bl * 0.995 <= x['body'] <= bh * 1.005, lab))
            if pts:
                traces.append({ 'name': disp, 'points': pts, 'ours': disp == "this card" })
    return { 'rows': rows, 'traces': traces, 'repos': [r["repo"] for r in info["references"]],
             'baseline': len(base) >= 2, 'slices': slices, 'slice_descs': [describe_slice(sl) for sl in slices],
             'ladders': ladders, 'ours': ours, 'base': base,
             'tainted': [{'slice': describe_slice(sl), 'by': tainted_by[sl]} for sl in tainted],
             'parity': [{'repo': ref['repo'], 'excess': {describe_slice(sl): row['kld'] - floor[sl] for sl in every
                                                         if (row := parse_qbench(qbench[sl], ref_label(ref['repo'], 'bf16')))},
                         'floor': {describe_slice(sl): floor[sl] for sl in every}}
                        for ref in info.get('references', []) if ref.get('parity')] }

def slice_comparisons(job, qbench, floor, data, ref):
    """TECHNICAL.md's per-slice section: for each slice, every ladder's excess by size (the plot)
    and ours against each ladder's rung at the same bit rate (the table), cells where both sit
    under 2x the slice's noise floor marked. Without references, ours (and the baseline) only."""
    if ref:
        ours, ladders, base = ref['ours'], ref['ladders'], ref['base']
    else:
        ours = {}
        for r in data['revisions']:
            ex = {sl: (row['kld'] - floor[sl]) for sl in qbench if (row := parse_qbench(qbench[sl], r['name']))}
            if ex:
                ours[r['name']] = {'label': r['name'], 'excess': ex,
                                   'size': (r['disk_bytes'] - r['embed_bytes'] - r['encoder_bytes'] - r['mtp_bytes']) / 1024**3,
                                   'bits': r['bits']}
        ladders, base = [], []
    out = []
    for sl in qbench:
        # Compared sets are drawn on the body axis their numbers are computed on (a rung with a
        # bigger head sits further right on total size than the size its ratios assume)
        xk = 'body' if ladders else 'size'
        series = [("this card", [(o[xk], o['excess'][sl]) for o in ours.values() if sl in o['excess']], True)]
        series += [(disp, [(x[xk], x['excess'][sl]) for x in rungs if sl in x['excess']], False) for disp, rungs in ladders]
        if base and all(sl in b['excess'] for b in base):
            series.append(("uncalibrated EXL3 (baseline)", [(b[xk], b['excess'][sl]) for b in base], False))
        table = []
        for name, o in ours.items():
            if sl not in o['excess']:
                continue
            cells = []
            for disp, rungs in ladders:
                m = [x for x in rungs if x.get('bits') is not None and o.get('bits') is not None
                     and abs(x['bits'] - o['bits']) < 0.01 and sl in x['excess']]
                if not m:
                    # No rung at this rate (a GGUF ladder never has one): log-interpolated at equal body size
                    pts = sorted((x['body'], x['excess'][sl]) for x in rungs if sl in x['excess'] and 'body' in x)
                    if len(pts) >= 2 and pts[0][0] * 0.98 <= o.get('body', 0) <= pts[-1][0] * 1.02 and min(p[1] for p in pts) > 0:
                        import numpy as np
                        v = float(np.exp(np.interp(o['body'], [p[0] for p in pts], np.log([p[1] for p in pts]))))
                        cells.append({ 'ratio': o['excess'][sl] / v, 'near': max(o['excess'][sl], v) < 2 * floor[sl],
                                       'interpolated': True, 'contaminated': False })
                    else:
                        cells.append(None)
                    continue
                x = min(m, key=lambda x: (x['head_bits'] != o.get('head_bits'), x['excess'][sl]))
                cells.append({ 'ratio': o['excess'][sl] / x['excess'][sl] if x['excess'][sl] > 0 else None,
                               'near': max(o['excess'][sl], x['excess'][sl]) < 2 * floor[sl], 'branch': x.get('branch'),
                               'contaminated': sl in x.get('contaminated', ()) })
            table.append({ 'bits': o['bits'], 'excess': o['excess'][sl], 'cells': cells })
        table.sort(key=lambda t: t['bits'])
        n = trace_stats(job, sl)
        title, what, why = SLICE_DOCS.get(sl, (sl, "", ""))
        tainted_by = sorted({disp for disp, rungs in ladders for x in rungs if sl in x.get('contaminated', ())})
        anchor = re.sub(r"\s+", "-", re.sub(r"[^\w\s-]", "", title.lower()).strip())
        out.append({ 'slice': sl, 'title': title, 'anchor': anchor, 'what': what, 'why': why, 'floor': floor[sl], 'tainted_by': tainted_by,
                     'weight': composite_weights(job)[0].get(sl), 'stats': n, 'series': series, 'table': table,
                     'ladders': [disp for disp, _ in ladders], 'plot': f"plots/slice_{sl}.png",
                     'thresholds': [(t['name'], t['excess'].get(sl)) for t in data.get('thresholds', [])],
                     'body_axis': bool(ladders) })
    return out

def plot_slice(sc, path):
    """One slice: every ladder's excess KLD by size, log scale, the region under 2x the slice's
    noise floor shaded (differences there are within the unquantized model's own rounding noise)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, AXES, GRID = "#1f1f1f", "#ffffff", "#aaaaaa", "#555555", "#555555"
    OURS, BASE = "#eb6834", "#888888"
    OTHERS = iter(["#2a78d6", "#3fb27f", "#b07fd6", "#d6b12a"])
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXES)
    ax.tick_params(colors=INK2, which="both")
    ys_all = []
    for name, pts, ours in sc['series']:
        pts = sorted(p for p in pts if p[1] > 0)
        if not pts:
            continue
        xs, ys = zip(*pts); ys_all += ys
        base = name.startswith("uncalibrated")
        color = OURS if ours else BASE if base else next(OTHERS)
        ax.plot(xs, ys, color=color, lw=2, ls=(0, (2, 2)) if base else "-", zorder=3 if ours else 2,
                marker="D" if ours else None if base else "o", markersize=5, markeredgecolor=SURF, label=name)
    ax.set_yscale("log")
    thr = [(n, v) for n, v in sc.get('thresholds', []) if v and v > 0]
    for name, v in thr:
        ax.axhline(v, color=INK2, lw=1.0, ls=(0, (6, 3)), zorder=1)
        ax.text(0.01, v, f"{name} {v:.5f}", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                fontsize=7, color=INK2)
    if ys_all:
        lo = min(min(ys_all), sc['floor'], *[v for _, v in thr]) / 1.5
        ax.set_ylim(lo, max([*ys_all, *[v for _, v in thr]]) * 1.5)
        ax.axhspan(lo, 2 * sc['floor'], color="#ffffff", alpha=0.07, lw=0, zorder=0)
        ax.axhline(2 * sc['floor'], color=INK2, lw=0.8, ls=":", zorder=1)
        ax.text(0.99, 2 * sc['floor'], "2x noise floor", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=7, color=INK2)
    ax.margins(x=0.06)
    ax.set_xlabel("transformer body (GiB; no output head, input embeddings, encoder or MTP)" if sc.get('body_axis') else
                  "language model weights (GiB; no input embeddings, encoder or MTP)", color=INK2)
    ax.set_ylabel("excess KLD (log)", color=INK2)
    leg = ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.text(0.13, 0.97, sc['title'], fontsize=11, color=INK, va="top")
    fig.subplots_adjust(top=0.89, left=0.13, right=0.97, bottom=0.14)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)

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
    # The MTP draft block (1.7% of an Ornith-9B 4 bpw checkpoint): a model feature, not quantized
    # weight a format comparison should be sized by -- GGUF conversions drop it
    if name.startswith("mtp.") or ".mtp." in m:
        return "mtp"
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
            'head': 0,
            'mtp': 0 }
    for name, size in tensors:
        out['total'] += size
        cat = tensor_category(name)
        if cat:
            out[cat] += size
    return out

# A weightless stand-in for a published quant (`fetch`): {"repo", "revision"}; sizes are read
# from the Hub's safetensors headers instead
PUBLISHED_MARKER = "published.json"
FETCHED_REFERENCE = "_fetched_reference.json"     # in a fetched job dir: reference.json as published

def tensor_survey_local(path):
    marker = os.path.join(path, PUBLISHED_MARKER)
    if not globmod.glob(os.path.join(path, "*.safetensors")) and os.path.isfile(marker):
        with open(marker) as f:
            m = json.load(f)
        return tensor_survey_remote(m["repo"], m["revision"])
    # a hub snapshot resolved for its configs only: the same survey from the Hub's headers
    hf = HF_SNAPSHOT.search(path.rstrip("/"))
    if hf and not hf.group(4) and not globmod.glob(os.path.join(path, "*.safetensors")):
        return tensor_survey_remote(f"{hf.group(1)}/{hf.group(2)}", hf.group(3))
    # The shards the index lists, as the Hub's metadata counts them: a branch inherits main's
    # files, and turboderp's carry cal_trace.safetensors (4 MiB of calibration token ids), which a
    # bare glob counted as weights
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            shards = sorted(os.path.join(path, s) for s in set(json.load(f)["weight_map"].values()))
    else:
        shards = sorted(globmod.glob(os.path.join(path, "*.safetensors")))
    def tensors():
        for shard in shards:
            with open(shard, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                header = json.loads(f.read(n))
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                a, b = meta["data_offsets"]
                yield name, b - a
    return tensor_survey(tensors())

def gguf_tensor_survey(path):
    """tensor_survey for a GGUF: token_embd is the input embedding, output.weight the head,
    NextN blocks the draft (mtp); bits per weight from element counts, for the size axis."""
    from gguf import GGUFReader
    out = {'total': 0, 'embed': 0, 'encoder': 0, 'head': 0, 'mtp': 0}
    body_bytes = body_n = head_bytes = head_n = 0
    for t in GGUFReader(path).tensors:
        out['total'] += t.n_bytes
        if t.name == "token_embd.weight":
            out['embed'] += t.n_bytes
        elif t.name == "output.weight":
            out['head'] += t.n_bytes; head_bytes += t.n_bytes; head_n += t.n_elements
        elif ".nextn." in t.name:
            out['mtp'] += t.n_bytes
        elif len(t.shape) >= 2:
            body_bytes += t.n_bytes; body_n += t.n_elements
    out['bits'] = round(8 * body_bytes / body_n, 2) if body_n else None
    out['head_bits'] = round(8 * head_bytes / head_n, 2) if head_n else None
    return out

def quant_size(path):
    """Sizes for a quant, in bytes: an EXL3 checkpoint dir (or its published stand-in) or a GGUF
    file, by category (embed, head, encoder, mtp) of the total."""
    if path.endswith(".gguf"):
        sz = gguf_tensor_survey(path)
        bits, head_bits = sz.pop('bits'), sz.pop('head_bits')
    else:
        sz = tensor_survey_local(path)
        with open(os.path.join(path, "quantization_config.json")) as f:
            q = json.load(f)
        bits, head_bits = q.get('bits'), q.get('head_bits')
    return { **sz, 'bits': bits, 'head_bits': head_bits }

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

UNCAL_README = """---
base_model: {base}
tags:
- exl3
- not-for-inference
---
# {name}: uncalibrated conversions (not for inference)

Working artifacts for [{repo}](https://huggingface.co/{repo}), not quants to run. Use that repo.

- `baseline/<rev>/`: uncalibrated EXL3 conversions of [{base}](https://huggingface.co/{base}) at
  the published bit rates, the common baseline every set on the model card is measured against.
  No calibration of any kind, so they favor no calibration, and are worse than every published quant.
- `generator/`: an uncalibrated 8 bpw conversion that generates the calibration trace's answers.

Each folder's `converter.json` records the exllamav3 converter that made it (git tree hashes);
the pipeline refuses to compare against a conversion from a different converter. Made and used by
`quantization/quant.py` in [vllm-exl3-plugin](https://github.com/yeasah/vllm-exl3-plugin)
(`upload --uncalibrated`, `fetch --uncalibrated`).
"""

def upload_uncalibrated(job, args):
    """The uncalibrated conversions (baseline ladder, calibration generator) to <repo>-uncalibrated:
    a separate repo, so its storage can be retired whole and it carries its own not-for-inference
    card; subfolders rather than branches, which further discourages loading them."""
    if os.path.isfile(os.path.join(job.dir, "FETCHED.json")):
        # its _baseline holds weightless size stand-ins, not conversions
        print("=== card-only job dir: no uncalibrated conversions to upload ==="); return False
    items = [(os.path.join(job.baseline, rev), f"baseline/{rev}") for rev in job.baseline_revisions()]
    if os.path.isfile(os.path.join(job.generator, "quantization_config.json")):
        items.append((job.generator, "generator"))
    if not items:
        print("=== no uncalibrated conversions to upload ==="); return True
    api = HfApi()
    api.create_repo(repo_id=job.uncal_repo, private=args.private, repo_type='model', exist_ok=True)
    readme = os.path.join(job.dir, "_uncal_README.md")
    with open(readme, "w") as f:
        f.write(UNCAL_README.format(name=job.name, repo=job.repo, base=job.base_repo))
    api.upload_file(path_or_fileobj=readme, path_in_repo="README.md", repo_id=job.uncal_repo,
                    commit_message=args.message or f"{job.name}: uncalibrated conversions, README")
    for local, remote in items:
        if not os.path.isfile(os.path.join(local, CONVERTER_FILE)):
            # built before converters were recorded: published as unknown, which a fetch refuses by default
            with open(os.path.join(local, CONVERTER_FILE), "w") as f:
                json.dump({"tree": None, "commit": None, "dirty": None,
                           "note": "made before the converter was recorded"}, f, indent=1)
        print(f"=== UPLOADING {local} -> {job.uncal_repo}/{remote} ===")
        api.upload_folder(folder_path=local, repo_id=job.uncal_repo, path_in_repo=remote,
                          commit_message=args.message or f"{job.name}: uncalibrated {remote}")
    return True

# Published JSON must not carry this machine's paths: they mean nothing elsewhere, and a re-card
# from a fetched stub once published the stubs' paths in place of the real sources. On upload,
# every absolute path becomes something resolvable anywhere.
# What a reference or threshold needs on disk to be keyed and sized; weights come only on a miss
CONFIGS_ONLY = ["*.json", "*.jinja", "*.txt", "*.model"]

HF_SNAPSHOT = re.compile(r"/models--([^/]+?)--([^/]+)/snapshots/([0-9a-f]{7,})(?:/(.+))?$")

def portable(path, job):
    """An absolute path as an identifier that means the same thing on any machine."""
    marker = f"/{job.name}/"
    local = os.path.join(job.dir, path.split(marker, 1)[1]) if marker in path else path   # another machine's job dir
    m = os.path.join(local, PUBLISHED_MARKER)
    if os.path.isfile(m):                       # a fetched stand-in: the published quant it stands for
        with open(m) as f:
            pub = json.load(f)
        if not pub.get("size_stand_in"):        # a baseline's stand-in only lends its size: not its identity
            return f"hf:{pub['repo']}@{pub['revision']}"
    hf = HF_SNAPSHOT.search(path.rstrip("/"))
    if hf:
        org, name, sha, rest = hf.groups()
        return f"hf:{org}/{name}@{sha}" + (f"/{rest}" if rest else "")
    if marker in path or os.path.abspath(local).startswith(os.path.abspath(job.dir) + os.sep):
        return os.path.relpath(local, job.dir)
    return f"local:{os.path.basename(path.rstrip('/'))}"

def portable_paths(obj, job, skip=("rows",)):
    """Every absolute-path string in a JSON value made portable; `skip` keys (trace rows: prompt
    text that could look like a path) are left untouched."""
    if isinstance(obj, dict):
        return {k: (v if k in skip else portable_paths(v, job, skip)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [portable_paths(v, job, skip) for v in obj]
    if isinstance(obj, str) and obj.startswith("/") and "\n" not in obj and len(obj) < 4096:
        return portable(obj, job)
    return obj

def stage_main(job, fetched):
    """main/ as published: a copy with portable JSON. A fetched job dir publishes the
    reference.json it fetched, not the one fetch rewrote to point at its local stand-ins."""
    stage = os.path.join(job.dir, "_upload_main")
    shutil.rmtree(stage, ignore_errors=True)
    # not published: qbench project files (local paths), part files, exploration rows kept for the
    # record, and screening results on sampled slices (qb_<slice>.s<N>.json)
    shutil.copytree(job.main, stage, ignore=shutil.ignore_patterns("qbench-*.yaml", "_qbench_parts", "_exploration_*",
                                                                   "qb_*.s[0-9]*.json"))
    original = os.path.join(job.dir, FETCHED_REFERENCE)
    if fetched and os.path.isfile(original):
        shutil.copyfile(original, os.path.join(stage, "reference.json"))
    # results rows for references the card does not compare against (dropped from reference.json,
    # e.g. a superseded release of this repo) stay local; the published results match the card
    ref_file = os.path.join(stage, "reference.json")
    if os.path.isfile(ref_file):
        with open(ref_file) as f:
            repos = {r["repo"] for r in json.load(f).get("references", [])}
        for qf in globmod.glob(os.path.join(stage, "qb_*.json")):
            with open(qf) as f:
                rows = json.load(f)
            keep = [r for r in rows if not r["label"].startswith("ref:") or r["label"][4:].split("@")[0] in repos]
            if len(keep) != len(rows):
                with open(qf, "w") as f:
                    json.dump(keep, f, indent=2)
    for f in [os.path.join(stage, "reference.json")] + globmod.glob(os.path.join(stage, "traces", "*.json")):
        if os.path.isfile(f):
            with open(f) as fh:
                d = json.load(fh)
            with open(f, "w") as fh:
                json.dump(portable_paths(d, job), fh, indent=None if os.path.getsize(f) > 1 << 20 else 1)
    return stage

def do_upload(job, args):
    if args.uncalibrated:
        if not upload_uncalibrated(job, args):
            return False
    if not args.quants:
        return True
    api = HfApi()
    api.create_repo(repo_id=job.repo,
                    private=args.private,
                    repo_type='model',
                    exist_ok=True)

    fetched = os.path.isfile(os.path.join(job.dir, "FETCHED.json"))
    gen_config, gen_record = published_generation_config(job)
    if gen_record and gen_record["action"] != "unchanged":
        print(f"=== generation_config.json {gen_record['action']} in each revision: the base model card's "
              f"{gen_record['mode']} sampling {gen_record['values']} ===")
    if fetched:
        print("=== fetched job dir: uploading main only (its revisions are weightless stand-ins) ===")
    for rev in ['main'] + ([] if fetched else job.revisions()):
        if not job.is_complete(rev):
            continue
        folder = stage_main(job, fetched) if rev == "main" else job.revdir(rev)
        if rev != "main" and gen_config is not None:
            with open(os.path.join(folder, "generation_config.json"), "w") as f:
                json.dump(gen_config, f, indent=2)
        msg = args.message or (f"{job.name}: model card, results and traces" if rev == "main"
                               else f"{job.name} {rev}")
        print(f"=== UPLOADING {job.revdir(rev)}: {msg} ===")
        api.create_branch(repo_id=job.repo,
                          branch=rev,
                          exist_ok=True)
        api.upload_folder(folder_path=folder,
                          repo_id=job.repo,
                          revision=rev,
                          commit_message=msg,
                          # the branch becomes exactly this folder: files an earlier upload had that this
                          # one does not are removed (the earlier revision stays in the branch history)
                          delete_patterns="*",
                          # qbench project files hold local paths; their results are published
                          ignore_patterns=["qbench-*.yaml", "_qbench_parts/*"])
        if rev == "main":
            shutil.rmtree(folder, ignore_errors=True)

def draw_thresholds(ax, lines, ys, color):
    """Horizontal lines at known checkpoints' excess ([(name, value)]), labelled at the left edge
    (the line tends to cross the ladder between 4 and 5 bpw, where the right side has rung labels),
    with the y range widened to keep them in view. Call before any explicit ylim."""
    lines = [(n, v) for n, v in lines if v and v > 0]
    for name, v in lines:
        ax.axhline(v, color=color, lw=1.0, ls=(0, (6, 3)), zorder=1)
        ax.text(0.01, v, f"{name} {v:.4f}", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                fontsize=7, color=color)
    if lines and ys:
        lo, hi = min(list(ys) + [v for _, v in lines]), max(list(ys) + [v for _, v in lines])
        ax.set_ylim(lo / 1.5, hi * 1.5)

def plot_quality_vs_size(job, meta, path):
    """Section 1 of the card: this card's own checkpoints, quality against size, no comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2 = "#1f1f1f", "#ffffff", "#aaaaaa"
    AXES, GRID, GRID2, LINE = "#555555", "#555555", "#444444", "#2a78d6"
    key = 'composite' if meta['composite'] else 'raw_excess'
    revs = [r for r in meta['revisions'] if r.get(key)]
    xs = [(r['disk_bytes'] - r['embed_bytes'] - r['mtp_bytes']) / 1024**3 for r in revs]
    ys = [r[key] for r in revs]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.grid(True, which="minor", color=GRID2, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXES)
    ax.tick_params(colors=INK2, which="both")
    ax.plot(xs, ys, color=LINE, lw=2, zorder=2)
    ax.scatter(xs, ys, s=70, color=LINE, edgecolor=SURF, linewidth=1.5, zorder=3)
    for r, x, y in zip(revs, xs, ys):
        ax.annotate(f"{r['bits']:.2f} bpw", (x, y), xytext=(8, 4), textcoords="offset points",
                    fontsize=8, color=INK)
        ax.annotate(f"{y:0.4f}", (x, y), xytext=(8, -4), textcoords="offset points",
                    fontsize=6, color=INK)
    ax.set_yscale("log")
    ax.margins(x=0.1)
    draw_thresholds(ax, [(f"{t['name']} ({t['size']:.1f} GiB)", t[key]) for t in meta.get('thresholds', []) if t.get(key)], ys, INK2)
    ax.set_xlabel("weight size (GiB, excluding input embeddings and MTP)", color=INK2)
    ax.set_ylabel("excess KLD vs bf16 (log, lower is better)", color=INK2)
    fig.text(0.1, 0.97, f"{job.name}: quality vs size", fontsize=11.5, color=INK, va="top")
    sub = ("composite: " + " + ".join(f"{w:.2f} {DESCRIBE[sl]}" for sl, w in composite_weights(job)[0].items())
           if meta['composite'] else "raw web text (this model has no chat template)")
    fig.text(0.1, 0.915, sub, fontsize=7.5, color=INK2, va="top")
    fig.subplots_adjust(top=0.86, left=0.13, right=0.97, bottom=0.13)
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)

DESCRIBE = {"wild": "real user prompts (WildChat)", "swe": "real agent sessions (Open-SWE-Traces)"}

def plot_vs_reference(job, ref, path):
    """Section 2 of the card: each ladder relative to the uncalibrated baseline at equal body size,
    every point labelled (our bit rate and ratio; another set's branch or quant type)."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, AXES, GRID = "#1f1f1f", "#ffffff", "#aaaaaa", "#555555", "#555555"
    OURS, BASE = "#eb6834", "#888888"
    OTHERS = ["#2a78d6", "#3fb27f", "#b07fd6", "#d6b12a"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXES)
    ax.tick_params(colors=INK2, which="both")
    ax.axhline(1.0, color=BASE, lw=1.5, ls=(0, (2, 2)), zorder=1, label="uncalibrated EXL3 (baseline)")
    ys_all = [1.0]
    any_ext = False
    labels = []     # (x, y, text, color, ours)
    others = iter(OTHERS)
    for ti, t in enumerate(ref['traces']):
        xs, ys, ext, labs = zip(*t['points']); ys_all += ys
        color = OURS if t['ours'] else next(others)
        mk = "D" if t['ours'] else "o"
        ax.plot(xs, ys, color=color, lw=2, zorder=3 if t['ours'] else 2, label=t['name'])
        ax.scatter([x for x, e in zip(xs, ext) if not e], [y for y, e in zip(ys, ext) if not e],
                   marker=mk, s=36, color=color, edgecolor=SURF, zorder=4)
        ax.scatter([x for x, e in zip(xs, ext) if e], [y for y, e in zip(ys, ext) if e],
                   marker=mk, s=36, facecolor=SURF, edgecolor=color, linewidth=1.5, zorder=4)
        any_ext = any_ext or any(ext)
        labels += [(x, y, f"{l}: {y:.2f}" if t['ours'] else l, INK if t['ours'] else color, t['ours'], ti)
                   for x, y, l in zip(xs, ys, labs)]
    lo, hi = min(ys_all), max(ys_all)
    pad = max(0.03, (hi - lo) * 0.14)
    ax.set_ylim(lo - pad, hi + pad)
    ax.margins(x=0.08)
    ax.set_xlabel("transformer body (GiB; no output head, input embeddings, encoder or MTP)", color=INK2)
    ax.set_ylabel("excess KLD relative to\nuncalibrated EXL3 at equal size", color=INK2)
    if any_ext:
        ax.scatter([], [], marker="o", s=36, facecolor=SURF, edgecolor=INK2, linewidth=1.5,
                   label="hollow: past the baseline's sizes, extrapolated")
    leg = ax.legend(frameon=False, fontsize=7.5, loc="best")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.text(0.13, 0.97, "Against other quants of this model", fontsize=11, color=INK, va="top")
    fig.subplots_adjust(top=0.89, left=0.15, right=0.97, bottom=0.13)

    # Greedy label placement, ours first. Each label goes on the side of its point away from the
    # other traces there (so a trace's labels sit consistently on its outer side), at the first
    # offset that overlaps no marker, line, placed label or the legend, stays in the axes, and sits
    # nearer its own point than any other trace's (a label must not read as a neighbour's)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    box = ax.get_window_extent(rend)
    taken = [leg.get_window_extent(rend)]
    disp = [(ax.transData.transform((x, y)), ti) for x, y, *_, ti in labels]
    for (px, py), _ in disp:
        taken.append(matplotlib.transforms.Bbox([[px - 5, py - 5], [px + 5, py + 5]]))
    lines = []
    for t in ref['traces']:         # the lines too, sampled along each segment
        d = ax.transData.transform([(p[0], p[1]) for p in t['points']])
        lines.append(d)
        for (x0, y0), (x1, y1) in zip(d[:-1], d[1:]):
            for f in np.linspace(0, 1, 12):
                px, py = x0 + f * (x1 - x0), y0 + f * (y1 - y0)
                taken.append(matplotlib.transforms.Bbox([[px - 1.5, py - 1.5], [px + 1.5, py + 1.5]]))
    ABOVE = [(0, 7, "center", "bottom"), (6, 6, "left", "bottom"), (-6, 6, "right", "bottom"), (0, 16, "center", "bottom")]
    BELOW = [(0, -7, "center", "top"), (6, -6, "left", "top"), (-6, -6, "right", "top"), (0, -16, "center", "top")]
    SIDES = [(7, 0, "left", "center"), (-7, 0, "right", "center")]
    def side(px, py, ti):
        """The side away from the other traces at this x: below a point every other trace runs above
        (and vice versa); in between, whichever side has fewer traces within reach."""
        up = down = near_up = near_down = 0
        for tj, d in enumerate(lines):
            if tj == ti or len(d) < 1 or not d[0][0] <= px <= d[-1][0]:
                continue
            oy = float(np.interp(px, d[:, 0], d[:, 1]))
            up, down = up + (oy > py), down + (oy < py)
            if abs(oy - py) < 60:
                near_up, near_down = near_up + (oy > py), near_down + (oy < py)
        if up and not down:
            return "below"
        if down and not up:
            return "above"
        return "below" if near_up > near_down else "above"
    for x, y, text, color, ours, ti in sorted(labels, key=lambda l: not l[4]):
        px, py = ax.transData.transform((x, y))
        prefer = BELOW if side(px, py, ti) == "below" else ABOVE
        offsets = prefer + SIDES + (ABOVE if prefer is BELOW else BELOW)
        best = None
        for dx, dy, ha, va in offsets:
            a = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=6.5 if ours else 6, color=color, zorder=5)
            bb = a.get_window_extent(rend).expanded(1.05, 1.1)
            cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
            own = np.hypot(cx - px, cy - py)
            # points coinciding with this one (within a few pixels) share its spot: being near them is fine
            nearest_other = min((np.hypot(cx - qx, cy - qy) for (qx, qy), tj in disp
                                 if tj != ti and np.hypot(qx - px, qy - py) > 6), default=1e9)
            inside = box.x0 <= bb.x0 and bb.x1 <= box.x1 and box.y0 <= bb.y0 and bb.y1 <= box.y1
            if inside and own < nearest_other and not any(bb.overlaps(t) for t in taken):
                best = a
                break
            a.remove()
        if best is None:    # nothing clear: the preferred side, overlapping
            dx, dy, ha, va = prefer[0]
            best = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                               fontsize=6.5 if ours else 6, color=color, zorder=5)
        taken.append(best.get_window_extent(rend))
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)

# Published in main/traces, so a set can be reproduced: (file in _traces, published name)
PUBLISHED_TRACES = [("cal.safetensors", "calibration.safetensors"),
                    ("cal.manifest.json", "calibration.manifest.json"),
                    ("cal_cal.json", "calibration.json"),
                    ("sampling.json", "sampling.json")] + \
                   [(f"eval_{sl}.json", f"eval_{sl}.json") for sl in list(EVAL_SLICES) + list(DIAG_SLICES)]

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
| `eval_ctx_user.json`, `eval_ctx_tool.json`, `eval_ctx_ml.json`, `eval_loop.json` | the diagnostic evals: held-out documents in a user turn, as tool results, multilingual, and in tool-use loops, answered by the unquantized model | [stas/openwebtext-10k](https://huggingface.co/datasets/stas/openwebtext-10k) (OpenWebText, CC0), [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) (CC BY-SA 3.0), source files of [vLLM](https://github.com/vllm-project/vllm) (Apache 2.0), Wikipedia 20231101 (CC BY-SA 3.0 / GFDL) |
| `eval_self.json` | the diagnostic own-voice eval: prompts held out of calibration, answered by the unquantized model | the prompt list of exllamav3's `sc_trace.py` (MIT) |

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
    record_slice_stats(job)
    meta = collect_metadata(job, qbench)
    plot_quality_vs_size(job, meta, os.path.join(job.main, "quality_vs_size.png"))
    meta['traces_published'] = publish_traces(job)
    note = os.path.join(job.dir, "card_note.md")      # per-job text under the card's intro, e.g. a superseded release
    meta['card_note'] = open(note).read().strip() if os.path.isfile(note) else None
    meta['recipe'] = any(os.path.isfile(os.path.join(job.revdir(r['name']), "recipe.yaml")) for r in meta['revisions'])
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
                                   show_ppl=args.ppl, **meta)
    card.save(os.path.join(job.main, "README.md"))

    # The companion: what each eval slice is, every slice compared, calibration, reproducing
    slices = slice_comparisons(job, qbench, meta['floor'], meta, meta['reference'])
    for sc in slices:
        plot_slice(sc, os.path.join(job.main, sc['plot']))
    calib = {}
    for r in meta['revisions']:
        path = os.path.join(job.revdir(r['name']), "calibration.json")
        if os.path.isfile(path):
            with open(path) as f:
                calib = json.load(f)
            break
    weights, note = composite_weights(job)
    spec_path = os.path.join(COMPOSITE_DIR, job.base_repo.replace("/", "__") + ".json")
    weight_spec = json.load(open(spec_path)) if os.path.isfile(spec_path) else None
    env = Environment(loader=FileSystemLoader(os.path.dirname(os.path.abspath(args.template))))
    with open(os.path.join(job.main, "TECHNICAL.md"), "w") as f:
        f.write(env.get_template("technical.jinja").render(
            this_model=job.repo, base_model=job.base_repo, slices=slices, weights=weights, weight_note=note,
            weight_spec=weight_spec, calibration=calib, reference=meta['reference'], **{k: meta[k] for k in
            ('traces_published', 'eval_generator', 'multimodal', 'thresholds') if k in meta}))

def fetch_uncalibrated(job, args):
    """Populate a working job dir with the uncalibrated conversions published in <repo>-uncalibrated
    (baseline rungs, calibration generator), so `baseline` (and `traces`) skip what is already
    made. Only what is missing locally is downloaded; nothing is marked FETCHED."""
    if os.path.isfile(os.path.join(job.dir, "FETCHED.json")):
        print("=== this job dir is card-only (fetched without --uncalibrated): fetch uncalibrated "
              "conversions into a working job dir instead ==="); return False
    api = HfApi()
    files = api.list_repo_files(job.uncal_repo)
    want = {}
    for f in files:
        parts = f.split("/")
        if parts[0] == "baseline" and len(parts) > 2:
            want.setdefault(f"baseline/{parts[1]}", os.path.join(job.baseline, parts[1]))
        elif parts[0] == "generator" and len(parts) > 1:
            want.setdefault("generator", job.generator)
    have = {r: d for r, d in want.items() if os.path.isfile(os.path.join(d, "quantization_config.json"))}
    for r in have:
        print(f"=== {r}: present locally, kept ===")
    todo = {r: d for r, d in want.items() if r not in have}
    if todo:
        stage = os.path.join(job.dir, "_uncal_fetch")
        snapshot_download(job.uncal_repo, local_dir=stage, allow_patterns=[f"{r}/*" for r in todo])
        for r, d in todo.items():
            os.makedirs(os.path.dirname(d), exist_ok=True)
            os.rename(os.path.join(stage, r), d)        # same filesystem: no second copy of the weights
            print(f"=== {r} -> {d} ===")
        shutil.rmtree(stage, ignore_errors=True)
    for p, why in converter_mismatches(list(want.values()), args).items():
        print(f"!! {p}: {why}; `baseline` and `qbench` will refuse it without --allow-converter-mismatch")
    print(f"=== fetched {len(todo)} uncalibrated conversion(s) from {job.uncal_repo} ===")
    return True

def do_fetch(job, args):
    """A card-only job directory from the published repo, no weights: main's results,
    reference.json and traces, each revision's configs, and weightless stand-ins (published.json)
    for every quant the card reads sizes from -- ours, each reference rung, and the uncalibrated
    baseline, which is not published: an uncalibrated conversion at the same bits and head bits is
    byte-identical in size to a calibrated one (Ornith-1.5-9B 4 bpw: 7311656070 bytes uncalibrated,
    default and mix alike), so our same-named revision stands in. The job is marked FETCHED, and
    every command but card and fetch refuses it. With --uncalibrated, instead: see fetch_uncalibrated."""
    if args.uncalibrated:
        return fetch_uncalibrated(job, args)
    api = HfApi()
    revs = [b.name for b in api.list_repo_refs(job.repo).branches if b.name != "main"]
    os.makedirs(job.main, exist_ok=True)
    os.makedirs(job.traces, exist_ok=True)
    snap = snapshot_download(job.repo, revision="main", allow_patterns=["qb_*.json", "reference.json", "slices.json", "traces/*"])
    for f in globmod.glob(os.path.join(snap, "qb_*.json")) + globmod.glob(os.path.join(snap, "slices.json")):
        shutil.copyfile(f, os.path.join(job.main, os.path.basename(f)))
    published = {dst: src for src, dst in PUBLISHED_TRACES}
    for f in globmod.glob(os.path.join(snap, "traces", "*")):
        name = os.path.basename(f)
        if name in published:
            shutil.copyfile(f, os.path.join(job.traces, published[name]))
    def stand_in(path, repo, revision, **extra):
        os.makedirs(path, exist_ok=True)
        for name in ("quantization_config.json", "calibration.json"):
            try:
                shutil.copyfile(hf_hub_download(repo, name, revision=revision), os.path.join(path, name))
            except Exception:
                if name == "quantization_config.json":
                    raise
        with open(os.path.join(path, PUBLISHED_MARKER), "w") as f:
            json.dump({"repo": repo, "revision": revision, **extra}, f, indent=1)
    for rev in revs:
        stand_in(job.revdir(rev), job.repo, rev)
    ref_path = os.path.join(snap, "reference.json")
    if os.path.isfile(ref_path):
        shutil.copyfile(ref_path, os.path.join(job.dir, FETCHED_REFERENCE))
        with open(ref_path) as f:
            info = json.load(f)
        for ref in info.get("references", []):
            if ref.get("type") == "gguf":       # sizes are recorded in reference.json; no stand-in needed
                continue
            for ladder in ref["ladders"].values():
                for b in ladder:
                    ladder[b] = os.path.abspath(os.path.join(job.dir, "_published", f"{ref['repo'].replace('/', '__')}@{b}"))
                    stand_in(ladder[b], ref["repo"], b)
        base = {}
        for rev in info.get("baseline", {}):
            if rev in revs:
                base[rev] = os.path.abspath(os.path.join(job.baseline, rev))
                stand_in(base[rev], job.repo, rev, size_stand_in="our same-named revision: an uncalibrated "
                         "conversion at the same bits and head bits is byte-identical in size")
            else:
                print(f"=== baseline {rev} has no published revision of the same name to stand in for its size: left out ===")
        info["baseline"] = base
        with open(os.path.join(job.main, "reference.json"), "w") as f:
            json.dump(info, f, indent=1)
    with open(os.path.join(job.dir, "FETCHED.json"), "w") as f:
        json.dump({"repo": job.repo, "revisions": revs, "fetched": stamp()}, f, indent=1)
    print(f"=== fetched {job.repo}: {len(revs)} revisions, card-only (no weights) ===")
    return True

def logit_cache_dir(job, args):
    """Each model owns its logit cache by default (<model>-exl3/_logit_cache, never uploaded), so
    pruning what one model's bench series did not use cannot touch another model's entries."""
    return os.path.abspath(args.logit_cache or os.path.join(job.dir, "_logit_cache"))

def qbench_project(job, args, slice_, arms, results=None):
    """One qbench project per slice: a bf16-generated eval trace, or raw web text. arms: [(label, path)]"""
    project = { "title": f"{job.name}: {slice_}",
                "logit_cache": { "dir": logit_cache_dir(job, args),
                                 "max_size_gb": args.logit_cache_size },
                "models": [ { "label": "HF BF16", "group": "reference", "engine": "transformers",
                              "repo": job.base_repo, "options": { "streaming": True } } ] +
                          [ { "label": label, "group": "threshold", "engine": "transformers", "source": os.path.abspath(path),
                              "options": { "streaming": True } } if label.startswith("thr:") else
                            { "label": label, "group": "GGUF", "engine": "llamacpp", "source": os.path.abspath(path),
                                "options": { "n_gpu_layers": 999 } } if path.endswith(".gguf") else
                            { "label": label, "group": "EXL3", "engine": "exllamav3",
                              "source": os.path.abspath(path) } for label, path in arms ],
                "output": { "results": os.path.abspath(results or os.path.join(job.main, f"qb_{slice_}.json")) } }
    if slice_ == "raw":
        # A guard, not a target: raw web text framed as nothing matches no real use
        project["test_data"] = dict(RAW_SPEC)
        project["tokenizer"] = { "repo": job.base_repo, "template": False }
    else:
        project["test_trace"] = os.path.abspath(job.eval_trace(slice_))
    return project

def qbench_parallel(job, args, sl, sl_arms, slots, script, out=None):
    """One slice over several GPUs: the bf16 reference and noise floor first, alone (two scorers
    missing the cache together would each compute it), then the quants split across the GPUs
    reading it from the shared cache, then their results merged into qb_<slice>.json in arm
    order. The reference and floor come from the reference pass; every part recomputes neither."""
    part_dir = os.path.join(job.main, "_qbench_parts")
    os.makedirs(part_dir, exist_ok=True)
    def part(name, part_arms, slot):
        yml = os.path.join(part_dir, f"qbench-{sl}-{name}.yaml")
        res = os.path.join(part_dir, f"qb_{sl}-{name}.json")
        with open(yml, "w") as f:
            yaml.safe_dump(qbench_project(job, args, sl, part_arms, results=res), f, sort_keys=False)
        rc = run_logged([ "python3", script, yml, "-d", "0" ], job.log(f"qbench-{sl}-{name}"), gpu_env(slot), echo=False)
        return res if rc == 0 and os.path.isfile(res) else None
    print(f"=== qbench {sl}: reference and noise floor on gpu {slots[0]} ===", flush=True)
    ref = part("ref", [], slots[0])
    if ref is None:
        report_failures(["reference"], f"qbench {sl}", {"reference": job.log(f"qbench-{sl}-ref")})
        return False
    k = min(len(slots), len(sl_arms))
    groups = [sl_arms[i::k] for i in range(k)]                # round-robin: neighbouring bitrates spread
    results = {}
    def task(i):
        def run(slot, echo):
            results[i] = part(f"p{i}", groups[i], slot)
            return results[i] is not None
        return f"{sl} part {i} ({len(groups[i])} quants)", run
    tasks = [task(i) for i in range(k)]
    failed = run_jobs(tasks, slots[:k], "qbench")
    if not report_failures(failed, "qbench", {name: job.log(f"qbench-{sl}-p{i}") for i, (name, _) in enumerate(tasks)}):
        return False
    with open(ref) as f:
        merged = [r for r in json.load(f) if r.get("group") in ("reference", "noise_floor")]
    by_label = {}
    for i in range(k):
        with open(results[i]) as f:
            by_label.update({r["label"]: r for r in json.load(f) if r.get("group") not in ("reference", "noise_floor")})
    missing = [label for label, _ in sl_arms if label not in by_label]
    if missing:
        print(f"=== qbench {sl}: no results for {', '.join(missing)} ===")
        return False
    merged += [by_label[label] for label, _ in sl_arms]
    with open(out or os.path.join(job.main, f"qb_{sl}.json"), "w") as f:
        json.dump(merged, f, indent=2)
    print(f"=== qbench {sl}: {len(sl_arms)} quants over {k} GPUs, merged ===", flush=True)
    return True

def qbench_parity(job, args, sl, parity_arms, slots, script):
    """A GGUF set's bf16 file, scored like a quant: what llama.cpp's numerics alone differ from the
    bf16 reference by. It may not fit one card (16.7 GB for a 9B), so llama.cpp gets every listed
    GPU in one process (its own layer split, not torch's). Optional: a failure is reported and the
    slice's results stand without it."""
    part_dir = os.path.join(job.main, "_qbench_parts")
    os.makedirs(part_dir, exist_ok=True)
    yml = os.path.join(part_dir, f"qbench-{sl}-parity.yaml")
    res = os.path.join(part_dir, f"qb_{sl}-parity.json")
    with open(yml, "w") as f:
        yaml.safe_dump(qbench_project(job, args, sl, parity_arms, results=res), f, sort_keys=False)
    print(f"=== qbench {sl}: GGUF bf16 parity on gpu {','.join(slots)} ===", flush=True)
    rc = run_logged([ "python3", script, yml, "-d", "0" ], job.log(f"qbench-{sl}-parity"),
                    {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": ",".join(slots)}, echo=False)
    if rc or not os.path.isfile(res):
        print(f"=== qbench {sl}: GGUF parity FAILED (exit {rc}); results stand without it ===")
        return False
    out = os.path.join(job.main, f"qb_{sl}.json")
    with open(out) as f:
        merged = json.load(f)
    with open(res) as f:
        add = {r["label"]: r for r in json.load(f) if r.get("group") not in ("reference", "noise_floor")}
    merged = [r for r in merged if r["label"] not in add] + list(add.values())
    with open(out, "w") as f:
        json.dump(merged, f, indent=2)
    return True

def merge_results(path, new_path):
    """qb_<slice>.json after a subset run: the new rows replace same-labelled ones, the rest stay
    (earlier variants, references and the baseline need not be rescored to keep their rows)."""
    with open(new_path) as f:
        new = json.load(f)
    old = []
    if os.path.isfile(path):
        with open(path) as f:
            old = json.load(f)
    labels = {r["label"] for r in new}
    keep_ref = [r for r in old if r.get("group") in ("reference", "noise_floor")]
    merged = (keep_ref or [r for r in new if r.get("group") in ("reference", "noise_floor")]) + \
             [r for r in old if r.get("group") not in ("reference", "noise_floor") and r["label"] not in labels] + \
             [r for r in new if r.get("group") not in ("reference", "noise_floor")]
    with open(path, "w") as f:
        json.dump(merged, f, indent=2)

SAMPLE_SEED = 0

def sample_spec(spec):
    """'swe=50,wild=20' -> {'swe': 50, 'wild': 20}"""
    return {k.strip(): int(v) for k, v in (x.split("=") for x in (spec or "").split(",") if x.strip())}

def sampled_trace(job, sl, n):
    """A fixed subset of a slice's eval trace, for screening (the same rows for every variant):
    eval_<slice>.s<n>.json, written once. Its own results (qb_<slice>.s<n>.json) and its own
    reference logits (the cache is keyed by trace), so it never mixes with the full slice."""
    key = f"{sl}.s{n}"
    out = job.eval_trace(key)
    if not os.path.isfile(out):
        import random
        with open(job.eval_trace(sl)) as f:
            t = json.load(f)
        if n >= len(t["rows"]):
            raise SystemExit(f"=== --sample {sl}={n}: the trace has only {len(t['rows'])} rows ===")
        idx = sorted(random.Random(SAMPLE_SEED).sample(range(len(t["rows"])), n))
        t["rows"] = [t["rows"][i] for i in idx]
        t["meta"] = {**t.get("meta", {}), "rows": n, "sample": {"of": os.path.basename(job.eval_trace(sl)),
                                                                 "seed": SAMPLE_SEED, "indices": idx}}
        with open(out, "w") as f:
            json.dump(t, f)
        print(f"=== {sl}: sampled {n} of its rows -> {os.path.basename(out)} ===")
    return key

def do_qbench(job, args):
    os.makedirs(job.main, exist_ok=True)
    revisions = [rev for rev in job.revisions() if job.is_complete(rev)]
    # Subsets for fast exploration: --revisions scores only those (merged into the existing results,
    # no references or baseline), --slices only those slices. Decisions still want a full run
    subset = bool(args.revisions or args.slices)
    base_want = []
    if args.revisions:
        want = [] if args.revisions == "none" else args.revisions.split(",")
        # baseline rungs by label (baseline:<rev>): a sampled or new slice needs its normalizer scored
        base_want = [w.split(":", 1)[1] for w in want if w.startswith("baseline:")]
        want = [w for w in want if not w.startswith("baseline:")]
        unknown = [r for r in want if r not in revisions] + [b for b in base_want if b not in job.baseline_revisions()]
        if unknown:
            print(f"=== no complete revision {', '.join(unknown)} ==="); return False
        revisions = [r for r in revisions if r in want]
    if not revisions and not (args.revisions and (args.reference or args.reference_gguf or args.threshold or base_want)):
        print("=== no revisions found ===")
        return False
    slices = [sl for sl in list(EVAL_SLICES) + list(DIAG_SLICES) if os.path.isfile(job.eval_trace(sl))] + ["raw"]
    if args.slices:
        alias = {"scored": list(EVAL_SLICES), "diag": list(DIAG_SLICES)}
        want = [x for w in args.slices.split(",") for x in alias.get(w, [w])]
        bad = [w for w in want if w not in slices]
        if bad:
            print(f"=== no eval trace for slice {', '.join(bad)} (have: {', '.join(slices)}) ==="); return False
        slices = [sl for sl in slices if sl in want]
    # --sample: screen on a fixed subset of a slice's rows (results and reference logits kept apart)
    samples = sample_spec(args.sample)
    bad = [sl for sl in samples if sl not in slices or sl == "raw"]
    if bad:
        print(f"=== --sample: {', '.join(bad)} not among the slices scored ==="); return False
    slices = [sampled_trace(job, sl, samples[sl]) if sl in samples else sl for sl in slices]
    missing = [sl for sl in EVAL_SLICES if not os.path.isfile(job.eval_trace(sl))]    # absent traces, not --slices
    if missing:
        print(f"=== no eval traces for {', '.join(missing)}: the card falls back to raw text "
              f"(run `traces` first for the composite) ===")
    arms = [(rev, job.revdir(rev)) for rev in revisions]
    # Existing sets of this model's quants (each grouped into ladders by branch name), and the
    # uncalibrated baseline sweep from `baseline`: section 2 of the card
    references = []
    for ref in args.reference or []:
        (repo, *patterns) = ref.split(":")
        branches = [b.name for b in HfApi().list_repo_refs(repo).branches]
        # Rungs below --reference-min-bits are left out (turboderp's SC ladder runs down to 1.4)
        # Rungs above --reference-max-bits are left out (default 8.9 catches up to a hypothetical 8bpw + HQ)
        keep = lambda b: (args.reference_max_bits >= float(re.search(r"(\d+(?:\.\d+)?)bpw", b).group(1)) >= args.reference_min_bits 
                          and (len(patterns) == 0 or sum(int(re.search(pattern, b) is not None) for pattern in patterns) > 0))
        # configs only: qbench fetches a rung's weights itself if it has to run it (a cache miss);
        # sizes come from the Hub's safetensors headers. A fully cached series downloads no weights
        ladders = {name: {b: snapshot_download(repo, revision=b, allow_patterns=CONFIGS_ONLY) for b in bs if keep(b)}
                   for name, bs in ladders_of(branches).items()}
        ladders = {name: bs for name, bs in ladders.items() if bs}
        references.append({"repo": repo, "ladders": ladders})
        arms += [(ref_label(repo, b), p) for bs in ladders.values() for b, p in bs.items()]
    # GGUF sets: one ladder of the chosen quant types, scored by llama.cpp; with --gguf-parity, the
    # set's bf16 GGUF too, whose excess is llama.cpp's own numerics against the bf16 reference
    parity_arms = []
    for repo in args.reference_gguf or []:
        files = [f for f in HfApi().list_repo_files(repo) if f.endswith(".gguf") and not f.startswith("mmproj")]
        def pick(q):
            # The quant type is the whole suffix: "IQ3_XXS" must not take unsloth's "UD-IQ3_XXS"
            # (a different recipe, then labelled as the plain type and possibly scored twice)
            m = [f for f in files if re.search(rf"(?<![A-Za-z0-9])(?<!UD-){re.escape(q)}\.gguf$", f, re.I)]
            if not m:
                print(f"=== {repo} has no {q} GGUF: left out ===")
            elif len(m) > 1:
                raise SystemExit(f"=== {repo}: {q} matches {m}; name the file's own quant suffix ===")
            return m[0] if m else None
        rungs = {q: hf_hub_download(repo, f) for q in args.gguf_quants.split(",") if (f := pick(q))}
        ref = {"repo": repo, "type": "gguf", "ladders": {"GGUF": rungs} if rungs else {}}
        if args.gguf_parity and (f := pick("bf16") or pick("f16")):
            ref["parity"] = hf_hub_download(repo, f)
            parity_arms.append((ref_label(repo, "bf16"), ref["parity"]))
        references.append(ref)
        arms += [(ref_label(repo, q), p) for q, p in rungs.items()]
    # Thresholds: a line to read the ladder against where no other set of quants exists (the
    # publisher's FP8), scored by the reference's engine, so its excess is the checkpoint's weights
    # alone; kept out of reference.json, which is what turns on the card's comparison section
    thresholds = []
    for t in args.threshold or []:
        repo, _, rev = t.partition("@")
        path = snapshot_download(repo, revision=rev or None, allow_patterns=CONFIGS_ONLY)
        thresholds.append({"repo": repo, "revision": os.path.basename(path.rstrip("/")), "engine": "transformers"})
        arms.append((threshold_label(repo), path))
    if thresholds:
        tfile = os.path.join(job.main, "thresholds.json")
        prev = load_thresholds(job) if subset else []
        mine = {t["repo"] for t in thresholds}
        with open(tfile, "w") as f:
            json.dump([t for t in prev if t["repo"] not in mine] + thresholds, f, indent=1)
    baseline = {rev: os.path.join(job.baseline, rev) for rev in job.baseline_revisions()
                if not args.revisions or rev in base_want}
    if refuse_converter_mismatch(list(baseline.values()), args, "baseline"):
        return False
    if references or baseline:
        # Every compared quant's size, recorded where the scores are: a re-card from the published
        # repo then needs neither the weights nor a header the Hub cannot parse (GGUF)
        sizes = {label: quant_size(p) for label, p in arms if label.startswith("ref:")}
        sizes.update({baseline_label(rev): quant_size(p) for rev, p in baseline.items()})
        info = {"references": references, "baseline": baseline, "sizes": sizes}
        prev = load_reference_info(job) if args.revisions else None
        if prev:
            # a subset run adds or replaces its own references and keeps every other entry
            mine = {r["repo"] for r in references}
            info = {"references": [r for r in prev.get("references", []) if r["repo"] not in mine] + references,
                    "baseline": prev.get("baseline", {}), "sizes": {**prev.get("sizes", {}), **sizes}}
        with open(os.path.join(job.main, "reference.json"), "w") as f:
            json.dump(info, f, indent=1)
    base_arms = [(baseline_label(rev), p) for rev, p in baseline.items()]
    print(f"=== running bench on {' '.join(label for label, _ in arms + base_arms)}; slices: {' '.join(slices)} ===")
    script = os.path.join(args.exllamav3dir, "eval", "qbench.py")
    series_start = datetime.datetime.now().replace(microsecond=0)
    # One scorer per GPU: a second on the same card would size its memory budgets from the same
    # free VRAM as the first
    slots = list(dict.fromkeys(device_slots(args.device)))
    for sl in slices:
        sl_arms = arms + (base_arms if sl.split(".")[0] in EVAL_SLICES else [])   # the baseline only enters through the composite
        qbench_file = os.path.join(job.main, f"qbench-{sl}.yaml")
        final = os.path.join(job.main, f"qb_{sl}.json")
        out = os.path.join(job.main, f"_qb_{sl}.subset.json") if args.revisions else final
        with open(qbench_file, "w") as f:                        # the whole slice, as run (or as merged)
            yaml.safe_dump(qbench_project(job, args, sl, sl_arms, results=None if out == final else out), f, sort_keys=False)
        if len(slots) == 1 or len(sl_arms) < 2:
            rc = run_logged([ "python3", script, qbench_file, "-d", "0" ], job.log(f"qbench-{sl}"), gpu_env(slots[0]))
            if rc:
                print(f"=== qbench on {sl} FAILED (exit {rc}) ===")
                return False
        elif not qbench_parallel(job, args, sl, sl_arms, slots, script, out=out):
            return False
        if out != final:
            merge_results(final, out); os.remove(out)
            print(f"=== qbench {sl}: {', '.join(l for l, _ in sl_arms)} merged into {os.path.basename(final)} ===")
    if parity_arms:
        for sl in slices:
            qbench_parity(job, args, sl, parity_arms, slots, script)
    # The series succeeded: reference logits it did not use (a superseded eval trace, a dropped
    # reference) can only be dead weight. Results and KL vectors are never pruned.
    record_slice_stats(job)
    # A subset series used only some slices' reference logits: pruning would evict the others
    if args.prune and not subset:
        print(f"=== pruning reference logits unused since {series_start.isoformat()} ===")
        run_logged([ "python3", os.path.join(args.exllamav3dir, "eval", "qbench_prune.py"),
                     logit_cache_dir(job, args), "--unused-since", series_start.isoformat() ], job.log("qbench-prune"))
    return True

def do_compare(job, args):
    """A table of scored revisions from main/qb_<slice>.json, for exploring variants: per slice,
    excess KLD as a share of the uncalibrated baseline at the same rate (log-interpolated
    between baseline rungs, absolute excess outside them); the model's composite (weighted geometric mean over its composite
    slices); with --base, every number as a ratio to that revision; the recipe's own-voice share."""
    import math
    import numpy as np
    qb = {}
    samples = sample_spec(args.sample)
    for sl in list(EVAL_SLICES) + list(DIAG_SLICES):
        p = os.path.join(job.main, f"qb_{sl}.s{samples[sl]}.json" if sl in samples else f"qb_{sl}.json")
        if os.path.isfile(p):
            with open(p) as f:
                qb[sl] = json.load(f)
        elif sl in samples:
            print(f"=== no {os.path.basename(p)}: score it with qbench --sample {sl}={samples[sl]} ==="); return False
    if args.slices:
        alias = {"scored": list(EVAL_SLICES), "diag": list(DIAG_SLICES)}
        want = [x for w in args.slices.split(",") for x in alias.get(w, [w])]
        qb = {sl: v for sl, v in qb.items() if sl in want}
    if not qb:
        print("=== no qbench results ==="); return False
    scored = sorted({r["label"] for v in qb.values() for r in v
                     if r.get("group") not in ("reference", "noise_floor") and not r["label"].startswith("baseline:")})
    revs = args.revisions.split(",") if args.revisions else scored
    if args.no_references:
        revs = [r for r in revs if not r.startswith("ref:")]
    base_revs = job.baseline_revisions()
    unmatched, interpolated = set(), set()
    def baseline_of(rev):        # the uncalibrated rung at the same rate: the longest baseline name it starts with
        if rev.startswith("ref:"):  # a reference rung: by its nominal bit rate (heads may differ)
            hit = re.search(r"(\d+(?:\.\d+)?)bpw", rev.split("@", 1)[-1])
            m = [b for b in base_revs if hit and abs(float(re.match(r"[\d.]+", b).group()) - float(hit.group(1))) < 0.01]
            return min(m, key=len) if m else None
        m = [b for b in base_revs if rev == b or rev.startswith(b + "-")]
        return max(m, key=len) if m else None
    def value(rev, sl):
        rows = {r["label"]: r for r in qb[sl]}
        fl = rows.get("Noise floor", {}).get("kld")
        r = rows.get(rev)
        if r is None or fl is None:
            return None
        b = rows.get(baseline_label(baseline_of(rev))) if baseline_of(rev) else None
        if b is None:               # no rung at its rate: log-interpolated at equal body bits, as on the card
            pts = sorted((x["bpw_layer"], x["kld"] - fl) for lab, x in rows.items()
                         if lab.startswith("baseline:") and x.get("bpw_layer"))
            at = r.get("bpw_layer")
            if at and len(pts) >= 2 and pts[0][0] <= at <= pts[-1][0] and min(p[1] for p in pts) > 0:
                interpolated.add(rev)
                return (r["kld"] - fl) / math.exp(np.interp(at, [p[0] for p in pts], np.log([p[1] for p in pts])))
        if b is None and base_revs and not args.base:
            unmatched.add(rev)          # outside the baseline ladder: absolute excess, marked, not a share
        return (r["kld"] - fl) / (b["kld"] - fl) if b else r["kld"] - fl
    def share(rev):
        if rev.startswith("ref:"):
            return "ref"
        if rev.startswith("thr:"):
            return "thr"
        try:
            with open(os.path.join(job.revdir(rev), "calibration.json")) as f:
                m = json.load(f).get("recipe", {}).get("measurement", "")
        except (FileNotFoundError, json.JSONDecodeError):
            return "-"
        hit = re.search(r"own-voice share ([0-9.]+)", m)
        return hit.group(1) if hit else ("custom" if m else "uniform")
    if not args.revisions:          # by rate, then references, uniform, then by share
        def key(rev):
            sh = share(rev)
            hit = re.search(r"(\d+(?:\.\d+)?)bpw", rev.split("@", 1)[-1])
            return (float(hit.group(1)) if hit else 99.0, baseline_of(rev) or rev, -2.0 if sh == "ref" else -1.0 if sh == "uniform" else
                    float(sh) if sh[0].isdigit() else 2.0, rev)
        revs = sorted(revs, key=key)
    W = {sl: w for sl, w in composite_weights(job)[0].items() if sl in qb}
    sls = [sl for sl in list(EVAL_SLICES) + list(DIAG_SLICES) if sl in qb]
    table = {rev: {sl: value(rev, sl) for sl in sls} for rev in revs}
    base = table.get(args.base) if args.base else None
    if args.base and base is None:
        print(f"=== --base {args.base}: not scored ==="); return False
    def comp(v):
        if any(v.get(sl) is None or v[sl] <= 0 for sl in W) or not W:
            return None
        tw = sum(W.values())
        return math.exp(sum(w / tw * math.log(v[sl]) for sl, w in W.items()))
    rel = lambda v: {sl: (v[sl] / base[sl] if v[sl] is not None and base and base.get(sl) else v[sl]) for sl in sls}
    shown = {rev: rel(v) if base else v for rev, v in table.items()}
    unit = f"ratio to {args.base}" if base else ("share of uncalibrated at the same rate" if base_revs else "absolute excess KLD")
    print(f"{unit}; composite: " + " + ".join(f"{w:.2f} {sl}" for sl, w in W.items()) + " (geometric)")
    revs = [r for r in revs if any(table[r][sl] is not None for sl in sls)]
    w0 = max(len(r) for r in revs + ["revision"])
    print(f"{'revision':{w0}s} {'share':>7s} " + " ".join(f"{(sl + '.s' + str(samples[sl]) if sl in samples else sl):>8s}" for sl in sls)
          + f" {'composite':>9s}")
    for rev in revs:
        v = shown[rev]
        c = comp(v)
        mark = "^" if rev in unmatched else "*" if rev in interpolated else " "
        cells = " ".join(f"{v[sl]:7.3f}{mark}" if v[sl] is not None else f"{'-':>8s}" for sl in sls)
        print(f"{rev:{w0}s} {share(rev):>7s} {cells} {c:9.3f}" if c is not None else f"{rev:{w0}s} {share(rev):>7s} {cells} {'-':>9s}")
    if interpolated:
        print("* no uncalibrated baseline at this rate: a share of the baseline log-interpolated at equal body bits")
    if unmatched:
        print("^ no uncalibrated baseline at this rate: absolute excess KLD, not comparable with the shares above")
    return True

def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")

class Stamped:
    """stdout with a timestamp at the start of each of quant.py's own lines. Children's output is
    copied through .buffer (raw bytes, progress bars intact) and is not touched."""
    def __init__(self, stream):
        self._s, self._bol = stream, True
    def write(self, text):
        out = []
        for part in text.splitlines(keepends=True):
            if self._bol and part.strip():
                out.append(stamp() + " ")
            out.append(part)
            self._bol = part.endswith("\n")
        return self._s.write("".join(out))
    def __getattr__(self, name):
        return getattr(self._s, name)

def run_logged(cmd, log_path, env=None, echo=True):
    """Run cmd, echoing its output to the terminal and appending it to log_path.

    convert.py's `!!` warnings (non-finite rows, Cholesky retries, fallbacks) and its
    per-layer error lines are the only record of how a conversion went; the checkpoint
    keeps none of it. Raw bytes are copied, so the converter's progress bars still render.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    sys.stdout.flush()  # our own === lines must reach the terminal before the child's output
    t0 = time.time()
    with open(log_path, "ab") as log:
        log.write(f"\n=== {stamp()} {' '.join(cmd)}\n".encode())
        log.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env=None if env is None else {**os.environ, **env})
        while chunk := os.read(proc.stdout.fileno(), 65536):
            if echo:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            log.write(chunk)
            log.flush()
        rc = proc.wait()
        dt = time.time() - t0
        took = f"{dt / 60:.1f} min" if dt >= 60 else f"{dt:.0f} s"
        log.write(f"\n=== {stamp()} exit {rc} after {took}\n".encode())
        return rc

def run_until(output, cmd, log_path, tries, env=None, cleanup=None, echo=True):
    """Run cmd until it has produced output, at most `tries` times."""
    for attempt in range(1, tries + 1):
        if os.path.exists(output):
            return True
        if attempt > 1:
            print(f"=== ATTEMPT {attempt} of {tries} for {output} ===")
            if cleanup:
                cleanup()
        run_logged(cmd, log_path, env, echo)
    return os.path.exists(output)

def device_slots(spec):
    """-d for quantize/baseline/qbench: '0', or '0,1,2,3' for one job per GPU at a time; a device
    listed twice ('0,0,1,1') runs two jobs on it (conversions only: they leave the GPU idle
    through their CPU-bound stretches, and fit twice on a large card)."""
    return [d.strip() for d in str(spec).split(",") if d.strip()]

def gpu_env(slot):
    """Each job sees only its own GPU, as device 0 -- one device per process, so nothing here ever
    runs a multi-GPU context (pytorch#196258 aside, there is nothing to share). PCI order, so
    the indices are nvidia-smi's."""
    return {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": slot}

def run_jobs(tasks, slots, what):
    """tasks: [(name, fn(slot, echo) -> ok)]. One slot: in order, the child's output echoed as
    before. More: the tasks are dealt to GPUs up front in proportion to each GPU's slots (a device
    listed twice runs two at a time), and each GPU works through its own share at its own
    concurrency. Not a shared queue of free slots: with one, the last tasks go to whichever slots
    free first, so six rungs on '0,1,0,1' could end with two on one card and the other idle (a
    gemma-4-12b ladder, 2026-10-23); dealt up front, each card runs two then one. No stealing
    either, which would bring the same tail back. Children's output goes to their logs only (four
    interleaved progress bars read as noise), with a line per start and finish. Every task runs,
    whatever fails; returns the names that failed."""
    if len(slots) == 1:
        return [name for name, fn in tasks if not fn(slots[0], True)]
    import threading, time
    from collections import Counter
    width = Counter(slots)                       # concurrency per GPU
    gpus = list(dict.fromkeys(slots))
    share = {g: [] for g in gpus}
    for task in tasks:                           # deal: the GPU with the fewest tasks per slot so far
        g = min(gpus, key=lambda g: (len(share[g]) / width[g], gpus.index(g)))
        share[g].append(task)
    failed, lock = [], threading.Lock()
    def say(line):                       # whole lines: print() writes text and newline separately
        with lock:
            sys.stdout.write(line + "\n"); sys.stdout.flush()
    def run(slot, task):
        name, fn = task
        t0 = time.time()
        say(f"=== [gpu {slot}] {what} {name} started ===")
        ok = False
        try:
            ok = fn(slot, False)
        except Exception as e:
            say(f"=== [gpu {slot}] {what} {name}: {type(e).__name__}: {e} ===")
        say(f"=== [gpu {slot}] {what} {name} {'done' if ok else 'FAILED'} after {(time.time() - t0) / 60:.0f} min ===")
        if not ok:
            with lock:
                failed.append(name)
    def lane(g, items):                          # one of this GPU's slots: take its next task in turn
        while True:
            with lock:
                if not items:
                    return
                task = items.pop(0)
            run(g, task)
    threads = [threading.Thread(target=lane, args=(g, share[g])) for g in gpus for _ in range(width[g])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return failed

def report_failures(failed, what, logs):
    """logs: name -> log path; prints each failure's log tail (parallel jobs did not echo)."""
    for name in failed:
        print(f"=== {what} {name} FAILED; end of {logs[name]}: ===")
        try:
            with open(logs[name], "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 4000))
                tail = f.read().decode(errors="replace").replace("\r", "\n")
            print("\n".join(l for l in tail.split("\n") if l.strip())[-2000:])
        except OSError:
            pass
    return not failed

SAMPLING_DIR = os.path.join(HERE, "sampling")

#: Sampling keys a published generation_config.json carries (all of them read by vLLM's server defaults)
GC_SAMPLING_KEYS = ("temperature", "top_k", "top_p", "min_p", "presence_penalty", "frequency_penalty",
                    "repetition_penalty")

def published_generation_config(job):
    """(generation_config.json to publish with each revision, record for the card), or (None, None).
    From the checked-in sampling profile's catch-all mode (its rule without a slice): the base model card's
    stated recommendation for general use, which several models give only in prose -- shipping no
    generation_config.json, so servers fall back to temperature 1.0 untruncated, or one holding the
    benchmark-reproduction settings instead. Only the sampling keys are set; the base model's other keys
    (token ids, ...) are kept. The card states what was added or changed and from where."""
    prof = os.path.join(SAMPLING_DIR, job.base_repo.replace("/", "__") + ".json")
    if not os.path.isfile(prof):
        return None, None
    with open(prof) as f:
        p = json.load(f)
    mode_name = next((r["mode"] for r in p.get("rules", []) if set(r) == {"mode"}), None)
    if mode_name is None:
        return None, None
    mode = {k: v for k, v in p["modes"][mode_name].items() if k in GC_SAMPLING_KEYS}
    base = None
    if HfApi().file_exists(job.base_repo, "generation_config.json"):
        with open(hf_hub_download(job.base_repo, "generation_config.json")) as f:
            base = json.load(f)
    if base is None:
        action, prior, gc = "added", {}, {**mode, "do_sample": True}
    else:
        changed = [k for k in mode if base.get(k) != mode[k]]
        if not changed:
            return base, {"action": "unchanged", "mode": mode_name, "source": p["source"], "values": mode, "prior": {}}
        action, prior, gc = "modified", {k: base.get(k) for k in changed}, {**base, **mode, "do_sample": True}
    return gc, {"action": action, "mode": mode_name, "source": p["source"], "values": mode, "prior": prior}

def recipe_floor_args(meas):
    """sc_optimize --floor_fit for a MoE measurement (grouped expert targets carry "members"): routing
    flips put a KL floor under every perturbation, which on Qwen3.6-35B-A3B was ~70% of the top level's
    KL, nearly equal for every tensor, so unfitted sensitivities come out flat. Dense models keep the
    plain fit; their floor is ~0.1% and the fit would only reshuffle on two-point noise."""
    with open(meas) as f:
        moe = any("members" in r for r in json.load(f)["results"])
    return ["-ff"] if moe else []

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

def trace_setup(job, args):
    """(base path, sampling args, tries) shared by both trace commands, or None to stop."""
    if not has_chat_template(job.base_repo):
        print(f"=== {job.base_repo} has no chat template: conversational traces do not apply; "
              f"quantize with --calibration default ===")
        return None
    path = snapshot_download(repo_id=job.base_repo)
    os.makedirs(job.traces, exist_ok=True)
    profile = resolve_sampling_profile(job, args, path)
    print(f"=== sampling: {profile or 'exllamav3 default (explicitly requested)'} ===")
    return path, (["--sampling_profile", profile] if profile else []), args.retries + 1

def eval_env(args):
    return None if args.eval_devices is None else {"CUDA_VISIBLE_DEVICES": args.eval_devices}

def do_eval_traces(job, args):
    """The bf16-generated eval traces the card's composite is scored on. Holds the reference model
    (bf16, or its FP8 release via vLLM) and nothing this job builds, so it can run on its own on a
    box sized for that model. Each step is skipped when its output exists."""
    setup = trace_setup(job, args)
    if setup is None:
        return False
    path, sampling, tries = setup
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
                         job.log(f"traces-eval-{sl}"), tries, env=eval_env(args)):
            print(f"=== eval trace {sl} FAILED ==="); return False

    print(f"=== diagnostic eval traces ({', '.join(DIAG_SLICES)}, {DIAG_N} each, {eval_desc}) ===")
    if not run_until(job.eval_trace(list(DIAG_SLICES)[-1]),
                     [ "python3", "-u", ctx_trace, "-m", eval_path, "-cs", "65536",
                       "-o", os.path.join(job.traces, "eval"), "--docs", "eval",
                       "--slices", ",".join(DIAG_SLICES), "-n", str(DIAG_N), "--seed", str(EVAL_SEED),
                       "--code_glob", CODE_DOCS, "--self_from", EVAL_SELF_PROMPTS ] + sampling + eval_backend,
                     job.log("traces-eval-diag"), tries, env=eval_env(args)):
        print("=== diagnostic eval traces FAILED ==="); return False
    return True

# Which generator wrote the calibration trace, as `quantize` records it per revision
CAL_GENERATOR_FILE = "cal_generator.json"

def cal_generator_named(job):
    """How the calibration trace's generator is described in a revision's calibration.json. Traces
    from before `cal-traces --generator` recorded nothing and were all the uncalibrated 8bpw."""
    f = os.path.join(job.traces, CAL_GENERATOR_FILE)
    if os.path.isfile(f):
        with open(f) as fh:
            return json.load(fh)["generator"]
    return "uncalibrated 8bpw of the base model"

def do_cal_traces(job, args):
    """The conversational calibration trace. --generator uncalibrated writes it with the uncalibrated
    8bpw EXL3 (built here if `baseline` has not), the stand-in for a bf16 that cannot run where the
    ladder is built; --generator bf16 writes it with the eval traces' own generator and engine, where
    that model fits, which needs no quant of this job at all. Needs the eval traces: its agent slice
    keeps out every repository the eval swe slice uses."""
    setup = trace_setup(job, args)
    if setup is None:
        return False
    path, sampling, tries = setup
    if not os.path.isfile(job.eval_trace("swe")):
        print(f"=== no eval swe trace at {job.eval_trace('swe')}: run `eval-traces` first (the calibration "
              f"keeps out its repositories) ==="); return False
    ctx_trace = os.path.join(args.exllamav3dir, "ctx_trace.py")
    if args.generator == "bf16":
        gen, cal_backend = eval_generator(job, args)
        env = eval_env(args)
        named = f"{args.eval_generator} (the eval traces' generator)" if args.eval_generator else "the base model (bf16)"
    else:
        # -d may list several GPUs for the calibration trace (a large generator autosplits over them);
        # the generator's conversion is single-GPU and sees only the first
        env = gpu_env(",".join(device_slots(args.device)))
        gen = job.generator
        if not os.path.isfile(os.path.join(gen, "quantization_config.json")):
            # Normally made by `baseline` alongside the uncalibrated ladder (the same kind of job, so it
            # spreads over GPUs with them); built here when traces are wanted without baselines
            print("=== calibration generator missing (normally made by `baseline`): building it ===")
            if not build_generator(job, args, path, gpu_env(device_slots(args.device)[0])):
                print("=== calibration generator FAILED ==="); return False
        cal_backend = cal_generator(job, args)
        named = "uncalibrated 8bpw of the base model"
    print(f"=== calibration trace ({named}) -> {job.cal_data} ===")
    if not run_until(job.cal_data,
                     [ "python3", "-u", ctx_trace, "-m", gen, "-cs", "65536",
                       "-o", os.path.join(job.traces, "cal"), "--docs", "cal", "--cal_out", job.cal_data,
                       "--shares", json.dumps(CAL_SHARES), "--doc_kinds", CAL_DOC_KINDS,
                       "-tv", CAL_TEMPLATE_VARS, "--max_new_tokens", str(CAL_MAX_NEW_TOKENS),
                       "--exclude_self", EVAL_SELF_PROMPTS, "--exclude_swe_from", job.eval_trace("swe") ] + sampling + cal_backend,
                     job.log("traces-cal"), tries, env=env):
        print("=== calibration trace FAILED ==="); return False
    with open(os.path.join(job.traces, CAL_GENERATOR_FILE), "w") as f:
        json.dump({"generator": named}, f, indent=1)
    decontaminate_self(job)
    return True

def do_traces(job, args):
    """Both trace sets: `eval-traces`, then `cal-traces`."""
    return do_eval_traces(job, args) and do_cal_traces(job, args)

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

# Uncalibrated conversions are stable artifacts for a model *and converter*: a baseline from an
# older converter is a different converter's draw, and every card ratio against it would mix the
# converter change into the calibration comparison. Each records the converter that made it: git's
# tree hashes of exllamav3/ and convert.py (not the commit, which also moves for eval-only changes)
CONVERTER_FILE = "converter.json"

def converter_id(xdir):
    """The converter in exllamav3dir: {'tree', 'commit', 'dirty'}; tree None when not knowable."""
    def git(*a):
        return subprocess.check_output(["git", "-C", xdir, *a], text=True, stderr=subprocess.DEVNULL).strip()
    try:
        trees = git("rev-parse", "HEAD:exllamav3", "HEAD:convert.py").split()
        dirty = bool(git("status", "--porcelain", "--", "exllamav3", "convert.py"))
        return {"tree": None if dirty else ":".join(trees), "commit": git("rev-parse", "HEAD"), "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"tree": None, "commit": None, "dirty": None}

def record_converter(out, args):
    with open(os.path.join(out, CONVERTER_FILE), "w") as f:
        json.dump({**converter_id(args.exllamav3dir), "recorded": stamp()}, f, indent=1)

def converter_mismatches(paths, args):
    """{path: why} for uncalibrated conversions not provably from the current converter."""
    cur = converter_id(args.exllamav3dir)
    out = {}
    for p in paths:
        try:
            with open(os.path.join(p, CONVERTER_FILE)) as f:
                rec = json.load(f)
        except FileNotFoundError:
            rec = {"tree": None}
        if cur["tree"] is None:
            out[p] = "the current converter has uncommitted changes" if cur["dirty"] else "the current converter is not a git checkout"
        elif rec.get("tree") is None:
            out[p] = "made by an unrecorded or modified converter"
        elif rec["tree"] != cur["tree"]:
            how = "adopted under" if rec.get("adopted") else "made by"
            out[p] = f"{how} converter {(rec.get('commit') or '?')[:10]}, current {cur['commit'][:10]}"
    return out

def adopt_converter(paths, args):
    """Record the current converter on conversions that carry no record, after a human has checked
    that the converter has not changed since they were made (e.g. the fork's history under
    exllamav3/ and convert.py). One recorded decision instead of an override passed on every run,
    which would teach the flag and dilute the refusal. Never overwrites a record that names
    another converter: that one is a true record of a different converter, and must be rebuilt."""
    cur = converter_id(args.exllamav3dir)
    if cur["tree"] is None:
        print("=== cannot adopt: the current converter has uncommitted changes or is not a git checkout ==="); return False
    for p in paths:
        try:
            with open(os.path.join(p, CONVERTER_FILE)) as f:
                rec = json.load(f)
        except FileNotFoundError:
            rec = {"tree": None}
        if rec.get("tree") is not None:
            print(f"=== {p}: already recorded ({'current' if rec['tree'] == cur['tree'] else 'a different converter'}); not adopted ===")
            continue
        with open(os.path.join(p, CONVERTER_FILE), "w") as f:
            json.dump({**cur, "adopted": True, "recorded": stamp(),
                       "note": "adopted: made before converters were recorded, judged by a person to be from this "
                               "converter; not verified by a build"}, f, indent=1)
        print(f"=== {p}: adopted under converter {cur['commit'][:10]} ===")
    return True

def refuse_converter_mismatch(paths, args, what):
    bad = converter_mismatches(paths, args)
    if not bad:
        return False
    for p, why in bad.items():
        print(f"!! {what} {p}: {why}")
    if args.allow_converter_mismatch:
        print(f"!! --allow-converter-mismatch: using them anyway")
        return False
    print(f"=== {what}: not from the current converter, so comparisons against them would include the "
          f"converter change; rebuild them (delete the directories) or pass --allow-converter-mismatch ===")
    return True

def build_generator(job, args, path, env, echo=True):
    """The calibration generator: an uncalibrated 8bpw conversion of the base model."""
    work = os.path.join(job.traces, "_work-gen")
    os.makedirs(job.traces, exist_ok=True)
    existed = os.path.isfile(os.path.join(job.generator, "quantization_config.json"))
    if echo:
        print(f"=== calibration generator: uncalibrated 8bpw -> {job.generator} ===")
    ok = run_until(os.path.join(job.generator, "quantization_config.json"),
                   [ "python3", os.path.join(args.exllamav3dir, "convert.py"),
                     "-b", "8", "-hb", "8", "-vb", "16", "--uncalibrated",
                     "-i", path, "-w", work, "-o", job.generator, "-d", "0" ],
                   job.log("traces-gen"), args.retries + 1, env=env,
                   cleanup=lambda: shutil.rmtree(work, ignore_errors=True), echo=echo)   # uncalibrated jobs cannot resume
    shutil.rmtree(work, ignore_errors=True)
    if ok and not existed:      # only what this run built: never stamp an older conversion as current
        record_converter(job.generator, args)
    return ok

def bit_specs(spec, suffix=None):
    """'2:5,3:5,4' -> [(rev, bits, headbits)]; a suffix is appended to every revision name"""
    out = []
    for b in spec.split(','):
        tb = b.split(':')
        bits, headbits = tb[0], (tb[1] if len(tb) > 1 else "6")
        out.append((f"{float(bits):0.2f}bpw" + (f"-H{int(headbits)}" if len(tb) > 1 else "") + (f"-{suffix}" if suffix else ""),
                    bits, headbits))
    return out

def do_baseline(job, args):
    """The uncalibrated sweep that card section 2 measures every ladder against: the same bit
    rates as `quantize`, converted with --uncalibrated (no calibration of any kind, so it favors
    no ladder), into <model>-exl3/_baseline; and the calibration generator `traces` uses, an
    uncalibrated 8bpw (for models with a chat template). One task pool, spread over -d. A rung or
    generator already present (made earlier, or brought by `fetch --uncalibrated`) is kept if the
    current converter made it. Uncalibrated jobs cannot resume; a failed one restarts."""
    path = snapshot_download(repo_id=job.base_repo)
    specs = bit_specs(args.bits)
    present = [os.path.join(job.baseline, rev) for rev, _, _ in specs
               if os.path.isfile(os.path.join(job.baseline, rev, "quantization_config.json"))]
    if args.adopt_converter:
        gen = [job.generator] if os.path.isfile(os.path.join(job.generator, "quantization_config.json")) else []
        if not adopt_converter(present + gen, args):
            return False
    if refuse_converter_mismatch(present, args, "baseline"):
        return False
    gen_wanted = args.generator and has_chat_template(job.base_repo)
    if gen_wanted and os.path.isfile(os.path.join(job.generator, "quantization_config.json")):
        for p, why in converter_mismatches([job.generator], args).items():
            # traces only; no comparison runs against it, so a note rather than a refusal
            print(f"!! calibration generator {p}: {why} (kept: it writes calibration text, not a comparison)")
    def task(rev, bits, headbits):
        def run(slot, echo):
            out = os.path.join(job.baseline, rev); work = os.path.join(job.baseline, f"_work-{rev}")
            existed = os.path.isfile(os.path.join(out, "quantization_config.json"))
            if echo:
                print(f"=== BASELINE {rev} (uncalibrated) ===")
            ok = run_until(os.path.join(out, "quantization_config.json"),
                           [ "python3", os.path.join(args.exllamav3dir, "convert.py"), "--uncalibrated",
                             "-hq", "-b", bits, "-hb", headbits, "-vb", "16", "--out_scales", OUT_SCALES,
                             *([] if HALF_STEPS else ["--no_half_steps"]),
                             "-i", path, "-w", work, "-o", out, "-d", "0" ],
                           job.log(f"baseline-{rev}"), args.retries + 1, env=gpu_env(slot),
                           cleanup=lambda: shutil.rmtree(work, ignore_errors=True), echo=echo)
            if ok:
                shutil.rmtree(work, ignore_errors=True)
                if not existed:     # only what this run built: never stamp an older rung as current
                    record_converter(out, args)
            return ok
        return rev, run
    def gen_task():
        def run(slot, echo):
            return build_generator(job, args, path, gpu_env(slot), echo)
        return "generator", run
    tasks = ([gen_task()] if gen_wanted else []) + [task(*sp) for sp in specs]   # biggest first
    failed = run_jobs(tasks, device_slots(args.device), "BASELINE")
    return report_failures(failed, "BASELINE", {**{rev: job.log(f"baseline-{rev}") for rev, _, _ in specs},
                                                "generator": job.log("traces-gen")})

# A per-tensor bit recipe (sc_measure -> sc_optimize) beats uniform bits at equal size by 5-16%,
# and what it is measured on decides where: our mix's own-voice rows protect chat, the whole mix
# protects agent work, and on a hybrid 9B the two trade hard (docs/qbench.md "The SC recipe clears
# the bar on the conversational eval"). Two ends are measured once; quantize blends them.
# Measured per content category, at the finest grain the calibration trace labels: own voice,
# documents by kind in a user turn and as tool results, tool loops, raw web text, random tokens.
# sc_measure scores every row independently and perturbs by (tensor, level) seed with Hessians from
# fixed rows, so per-category runs are an exact partition of one big measurement: any grouping or
# weighting is arithmetic afterwards, and a run can be chunked to bound memory.
# Default: own voice only, all a recipe at the default share uses. On Qwen3-0.6B, Ornith-1.5-9B and
# Qwen3.8-27B an own-voice recipe was best or near-best everywhere; mixes helped only the 9B's agent
# slices and cost the 27B badly (docs/qbench.md). Other categories (--rows) are for exploring
MEASURE_ROWS = {"self": 20, "ctx_user:*": 0, "ctx_tool:*": 0, "loop": 0, "raw": 0, "random": 0}
MEASURE_CHUNK = 20         # rows per sc_measure run: its boundary cache grows with rows
MEASURE_H_ROWS = 64        # Hessian-capture rows, the same in every run: the calibration's first
OWNVOICE_SHARE = 1.0       # unconditional: see MEASURE_ROWS
# exllamav3 1.6.4 changed four defaults that move every dense recipe or conversion. Three stay pinned to
# the pre-1.6.4 behaviour until each is measured as its own arm (TODO exl3-1.6.4-defaults); half steps
# were measured and adopted (HALF_STEPS, RECIPE_K_STEP):
MEASURE_GROUPS = "experts"   # sc_measure -g: a MoE layer's routed experts are one target (a per-expert
                             # measurement is impractical); dense tensors stay individually measured, so
                             # their keys and seeds match every earlier measurement chunk (upstream: all)
RECIPE_TIES = "k_proj+v_proj,gate_proj+up_proj"   # sc_optimize -tie (upstream adds q_proj and GDN qkv+z)
OUT_SCALES = "always"        # convert.py --out_scales (upstream: auto, off for some gate/up projections)
HALF_STEPS = True            # convert.py's 1.5/2.5/3.5 steps below 4 bpw, as upstream: -b 3.5 is a native 3.5,
                             # so the half-rung baselines are what a plain conversion now gives; False passes
                             # --no_half_steps, the pre-1.6.4 whole-bit mix. Integer rungs unchanged (-b 3 -hq
                             # on Qwen3-0.6B is byte-identical either way)
RECIPE_K_STEP = 0.5          # sc_optimize -ks: half-bit steps below 4 bpw (whole above). At equal size, against
                             # the integer recipe: Qwen3-0.6B 3.5 bpw -6..-8% wild, -10% swe (two seeds), 2.5 bpw
                             # -8% / -12%; Ornith-1.5-9B 3.5 bpw -6% / -4% (docs/calibration.md)

def row_labels(job, ids, k=32):
    """Each calibration row's content category, by matching its opening tokens against the
    calibration trace's conversations (raw web text and random tokens are not in the trace)."""
    with open(os.path.join(job.traces, "cal_cal.json")) as f:
        rows = json.load(f)["rows"]
    idx = {}
    for r in rows:
        seq = r["input_ids"] + r["response_ids"]
        for i in range(len(seq) - k + 1):
            idx.setdefault(hash(tuple(seq[i:i + k])), set()).add(r.get("slice") or "?")
    out = []
    for row in ids:
        hit = idx.get(hash(tuple(row[:k].tolist())))
        out.append("/".join(sorted(hit)) if hit else ("random" if row.float().std() > 0.25 * row.max() else "raw"))
    return out

def exl3_calibration_data(args):
    """exllamav3's conversion/calibration_data.py, loaded by path: the row packer the calibration file was
    made with, without importing the exllamav3 package (which loads its CUDA extension)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "exl3_calibration_data", os.path.join(args.exllamav3dir, "exllamav3", "conversion", "calibration_data.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def trace_bos_id(path, rows):
    """The BOS id to open every packed row with, or None: the tokenizer's BOS when the chat template
    starts every trace conversation with it (gemma), none when it emits none (Qwen)."""
    from transformers import AutoTokenizer
    bos = AutoTokenizer.from_pretrained(path).bos_token_id
    return bos if bos is not None and rows and all(r["input_ids"][:1] == [bos] for r in rows) else None

def measure_categories(job, cal, args, path):
    """{category: [rows of token ids]}: trace conversations packed per category as the calibration
    packs a slice (concatenated, cut to full width, each row opened with BOS on a model that uses one),
    kind-pure; raw and random rows from the calibration file past its Hessian rows."""
    cols = cal.shape[1]
    with open(os.path.join(job.traces, "cal_cal.json")) as f:
        rows = json.load(f)["rows"]
    pack_rows = exl3_calibration_data(args).pack_rows
    bos_id = trace_bos_id(path, rows)
    streams = {}
    for r in rows:
        sl = r.get("slice")
        cat = f"{sl}:{r.get('kind')}" if sl in ("ctx_user", "ctx_tool") else sl
        streams.setdefault(cat, []).extend(r["input_ids"] + r["response_ids"])
    out = {c: pack_rows(st, cols, bos_id) for c, st in streams.items()}
    rest = cal[MEASURE_H_ROWS:]
    for i, lab in enumerate(row_labels(job, rest)):
        if lab in ("raw", "random"):
            out.setdefault(lab, []).append(rest[i].tolist())
    return out

def measure_counts(cats, spec):
    """Rows wanted per category: the defaults, overridden by --rows 'pattern=N,...' (fnmatch)."""
    import fnmatch
    rules = list(MEASURE_ROWS.items()) + [(k.strip(), int(v)) for k, v in
                                          (x.split("=") for x in (spec or "").split(",") if x.strip())]
    want = {}
    for c in cats:
        for pat, n in rules:                    # later rules win
            if fnmatch.fnmatch(c, pat):
                want[c] = n
    return want

def category_dir(job):
    return os.path.join(job.measure, "categories")

def category_chunks(job, cat):
    """[(start, end, path)] of this category's measured row ranges, in order."""
    d, out = category_dir(job), []
    for f in os.listdir(d) if os.path.isdir(d) else []:
        m = re.fullmatch(re.escape(cat.replace(":", "__")) + r"\.(\d+)-(\d+)\.json", f)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), os.path.join(d, f)))
    return sorted(out)

# The trellis's weight-space error at K=2, a property of the converter rather than of the model:
# 0.2617-0.2628 for every body tensor of a 9B and a 27B, probed on real rungs (2026-10-22, converter
# 1.4.9); anchors written from it match the 27B's probe within 1% (head 1.2%). Re-probe after a
# converter change before trusting it (`measure --rfn-anchor probe`).
RFN_K2 = 0.2617

def rfn_at_k2(probe, k, out):
    """The probe's body anchors restated at K=2, the rate sc_measure's noise levels (-rs 1.0,0.5)
    and every recipe so far were calibrated at. The trellis error is a converter constant: one
    probed rung is ~0.262 rfn at K=2 for every body tensor of a 9B and a 27B alike, and halves per
    bit (1.973-1.984x from K=2 to K=3 on the 27B), so the lowest rung of any ladder anchors the
    same levels. The head keeps its own K and rfn: it is quantized at its own rate on every rung."""
    with open(probe) as f:
        data = json.load(f)
    for r in data["results"]:
        if r["K"] == k and "lm_head" not in r["key"]:
            r["rfn"], r["K"] = r["rfn"] * 2 ** (k - 2), 2
    data["anchored_from_k"] = k
    with open(out, "w") as f:
        json.dump(data, f, indent=2)

def do_measure(job, args):
    """Per-tensor sensitivity of the unquantized model per content category (see MEASURE_ROWS;
    --rows raises or lowers a category's count, e.g. 'ctx_*:code=10'). Reused by every later
    quantize; raising a count measures only the rows not yet measured. Needs the calibration trace.
    The rfn anchors are the converter's constant by default (no quant of this job needed); --rfn-anchor
    probe measures them on the lowest integer-rate `baseline` rung instead."""
    import torch
    from safetensors.torch import load_file, save_file
    if not os.path.isfile(job.cal_data):
        print("=== no calibration (run traces first) ==="); return False
    path = snapshot_download(repo_id=job.base_repo)
    os.makedirs(category_dir(job), exist_ok=True)
    cal = load_file(job.cal_data)["input_ids"]
    hess = cal[:MEASURE_H_ROWS]
    cats = measure_categories(job, cal, args, path)
    want = measure_counts(cats, args.rows)
    rfn = os.path.join(job.measure, "rfn.json")
    # The anchor rung: the lowest integer rate (a fractional rate mixes K per tensor)
    whole = sorted((float(m.group(1)), r) for r in job.baseline_revisions()
                   if (m := re.match(r"(\d+)\.00bpw", r)))
    gpus = device_slots(args.device)
    if not os.path.isfile(rfn) and args.rfn_anchor == "constant":
        rc = run_logged([ "python3", os.path.join(args.exllamav3dir, "sc_rfn_probe.py"), "-mr", path,
                          "--constant", str(RFN_K2), "-o", rfn ], job.log("measure-rfn"))
        if rc or not os.path.isfile(rfn):
            print(f"=== rfn anchors FAILED (log {job.log('measure-rfn')}) ==="); return False
    if not os.path.isfile(rfn):
        if whole:
            probe = os.path.join(job.measure, "rfn-probe.json")
            if not os.path.isfile(probe):
                rc = run_logged([ "python3", os.path.join(args.exllamav3dir, "sc_rfn_probe.py"), "-mq",
                                  os.path.join(job.baseline, whole[0][1]), "-mr", path, "-d", "0", "-o", probe ],
                                job.log("measure-rfn"), gpu_env(gpus[0]))
                if rc or not os.path.isfile(probe):
                    print(f"=== rfn probe FAILED (log {job.log('measure-rfn')}) ==="); return False
            rfn_at_k2(probe, int(whole[0][0]), rfn)
        else:
            print("!! no integer-rate baseline rung for rfn anchors: measuring at sc_measure's global noise levels")
    plan = []
    for cat in sorted(cats):
        n = min(want.get(cat, 0), len(cats[cat]))
        if want.get(cat, 0) > len(cats[cat]):
            print(f"!! {cat}: {want[cat]} rows wanted, the trace holds {len(cats[cat])}")
        done = category_chunks(job, cat)
        start = done[-1][1] if done else 0
        for a in range(start, n, MEASURE_CHUNK):
            plan.append((cat, a, min(a + MEASURE_CHUNK, n)))
    print(f"=== measuring {len(plan)} run(s): " + ", ".join(f"{c}[{a}:{b}]" for c, a, b in plan) + " ===" if plan
          else "=== every category measured to its row count ===")
    for cat, a, b in plan:
        name = f"{cat.replace(':', '__')}.{a}-{b}"
        rows_file = os.path.join(category_dir(job), name + ".safetensors")
        save_file({"input_ids": torch.cat([torch.tensor(cats[cat][a:b], dtype=torch.long), hess], 0).contiguous()}, rows_file)
        out = os.path.join(category_dir(job), name + ".json")
        dev = ["-gs", args.gpu_split] if args.gpu_split else ["-d", "0"]
        rc = run_logged([ "python3", os.path.join(args.exllamav3dir, "sc_measure.py"), "-m", path, *dev,
                          "-r", str(b - a), "-hr", str(MEASURE_H_ROWS), "--shaped", "-rk", "-g", MEASURE_GROUPS,
                          "-tr", rows_file, "-o", out + ".part" ] +
                        (["-bc"] if args.boundary_cpu else []) +
                        (["-rr", rfn, "-rs", "1.0,0.5"] if os.path.isfile(rfn) else []),
                        job.log(f"measure-{name}"), gpu_env(",".join(gpus)) if args.gpu_split else gpu_env(gpus[0]))
        os.remove(rows_file)
        if rc or not os.path.isfile(out + ".part"):
            print(f"=== measurement {name} FAILED (log {job.log(f'measure-{name}')}) ==="); return False
        os.replace(out + ".part", out)
    return True

def category_klds(job, cat, n=None):
    """(template measurement, {tensor: [[row KLs] per level]}) over a category's first n measured rows."""
    chunks = category_chunks(job, cat)
    tmpl, per = None, {}
    for _, _, p in chunks:
        with open(p) as f:
            d = json.load(f)
        tmpl = tmpl or d
        for r in d["results"]:
            lv = per.setdefault(r["key"], [[] for _ in r["levels"]])
            for i, l in enumerate(r["levels"]):
                lv[i].extend(l["kld_rows"])
    if n is not None:
        per = {k: [l[:n] for l in lv] for k, lv in per.items()}
    return tmpl, per

def measured_categories(job):
    d = category_dir(job)
    return sorted({re.sub(r"\.\d+-\d+\.json$", "", f).replace("__", ":") for f in os.listdir(d)
                   if re.search(r"\.\d+-\d+\.json$", f)}) if os.path.isdir(d) else []

def mix_weights(job, cats, cat_shares):
    """The calibration's own proportions over the measured categories (rows per slice from its
    manifest; a document slice split over its kinds by their tokens in the trace): weighting
    category means by these, un-normalized, is measuring the whole calibration."""
    import fnmatch
    rules = [(k.strip(), float(v)) for k, v in (x.split("=") for x in cat_shares.split(",") if x.strip())] if cat_shares else []
    with open(os.path.join(job.traces, "cal.manifest.json")) as f:
        comp = {k: v["rows"] for k, v in json.load(f)["composition"].items()}
    with open(os.path.join(job.traces, "cal_cal.json")) as f:
        rows = json.load(f)["rows"]
    tok = {}
    for r in rows:
        if r.get("slice") in ("ctx_user", "ctx_tool"):
            k = f"{r['slice']}:{r.get('kind')}"
            tok[k] = tok.get(k, 0) + len(r["input_ids"]) + len(r["response_ids"])
    ctx_tok = sum(v for k, v in tok.items() if k in cats)
    w = {}
    for c in cats:
        if c in tok:
            w[c] = comp.get("ctx", 0) * tok[c] / ctx_tok
        else:
            w[c] = comp.get(c, 0)
        factor = 1.0
        for pat, n in rules:                    # later rules win, as in measure --rows: '*=0,ctx_user:code=1'
            if fnmatch.fnmatch(c, pat):
                factor = n
        w[c] *= factor
    return w

def blend_measurement(job, groups, out):
    """A measurement for sc_optimize from category measurements: each group ({category: weight},
    weights over category mean KLs, un-normalized) sums to a sensitivity per tensor and level; each
    group is normalized to equal total KL (absolute KL differs ~10x by content, and allocation
    depends only on relative sensitivity), then the groups are blended at their shares."""
    cache, tmpl = {}, None
    def mean_kl(cat):
        if cat not in cache:
            t, per = category_klds(job, cat)
            cache[cat] = {k: [sum(l) / len(l) for l in lv] for k, lv in per.items()}
            nonlocal tmpl
            tmpl = tmpl or t
        return cache[cat]
    acc = {}
    for share, members in groups:
        tw = sum(members.values())
        if tw <= 0:
            raise SystemExit(f"=== recipe blend: every category in a group with share {share} has weight 0 "
                             f"(check --category-shares) ===")
        g = {}
        for cat, w in members.items():
            print(f"including {round(w/tw*share*100, 2)}% {cat}")
            for k, lv in mean_kl(cat).items():
                cur = g.setdefault(k, [0.0] * len(lv))
                for i, v in enumerate(lv):
                    cur[i] += w / tw * v
        total = sum(v for lv in g.values() for v in lv)
        for k, lv in g.items():
            cur = acc.setdefault(k, [0.0] * len(lv))
            for i, v in enumerate(lv):
                cur[i] += share * v / total
    for r in tmpl["results"]:
        for i, l in enumerate(r["levels"]):
            l["kld"] = acc[r["key"]][i]
            l.pop("kld_rows", None)
    tmpl["trace"] = "blend: " + "; ".join(f"{s} x {sorted(m)}" for s, m in groups)
    with open(out, "w") as f:
        json.dump(tmpl, f)
    return out

def ownvoice_groups(job, share, cat_shares):
    """The own-voice dial: own voice at `share`, the whole calibration at the rest (as far as it was
    measured; at share 1 the rest is not consulted, so an own-voice-only measurement suffices)."""
    if share >= 1.0:
        return [(1.0, {"self": 1.0})]
    cats = measured_categories(job)
    if cats == ["self"]:
        raise SystemExit("=== --ownvoice-share below 1 needs other categories measured (measure --rows) ===")
    return [(share, {"self": 1.0}), (1 - share, mix_weights(job, cats, cat_shares))]

def publish_recipe(job, src, dst, what):
    """recipe.yaml for a revision: sc_optimize's header names this machine's model path and the
    measurement file by its working name; the published copy names the base model by Hub revision
    and describes the measurement instead."""
    import yaml
    with open(src) as f:
        r = yaml.safe_load(f)
    model = HF_SNAPSHOT.search(str(r.get("model", "")).rstrip("/"))
    r["model"] = f"hf:{model.group(1)}/{model.group(2)}@{model.group(3)}" if model else job.base_repo
    r["measurement"] = what
    with open(dst, "w") as f:
        f.write("# Per-tensor bit recipe (sc_optimize), the allocation this revision was converted with\n")
        yaml.safe_dump(r, f, sort_keys=False)

def recipe_description(share, measurement=None):
    if measurement:
        return os.path.basename(measurement)
    if share >= 1.0:
        return "own-voice sensitivity (sc_measure on the calibration's own-voice rows)"
    return f"blend of own-voice and calibration-mix sensitivities, own-voice share {share}"

def plan_recipes(job, args, recipe_source, share, rfn):
    """quantize --plan: solve each revision's recipe as the real run would (seconds), convert
    nothing, and compare it with every recipe already solved here. Near the solver's decision
    boundaries many knob settings give the same allocation, hence the same quant: an exact match
    means the conversion (about an hour at 27B) would duplicate an existing variant."""
    import yaml
    if not recipe_source:
        print("=== --plan: these revisions would be uniform bits; nothing to solve ==="); return True
    rdir = os.path.join(job.dir, "_recipes"); pdir = os.path.join(rdir, "_plan")
    os.makedirs(pdir, exist_ok=True)
    existing = {}
    for f in sorted(globmod.glob(os.path.join(rdir, "*.yaml"))):
        with open(f) as fh:
            existing[os.path.basename(f)[:-5]] = yaml.safe_load(fh)["tensors"]
    for rev, bits, headbits in bit_specs(args.bits, args.suffix):
        meas = args.measurement or blend_measurement(job, ownvoice_groups(job, share, args.category_shares),
                                                     os.path.join(pdir, f"{rev}.measurement.json"))
        out = os.path.join(pdir, f"{rev}.yaml")
        rc = run_logged([ "python3", os.path.join(args.exllamav3dir, "sc_optimize.py"), "-m", meas, "-b", bits, "-hb", headbits,
                          "-al", str(args.alpha), "-mink", str(args.min_k), "-tie", RECIPE_TIES, "-ks", str(RECIPE_K_STEP), *(["-all", str(args.alpha_low)] if args.alpha_low else []), *recipe_floor_args(meas), "-o", out ] + ([ "-rr", rfn ] if rfn else []),
                        job.log(f"plan-{rev}"), None, False)
        if rc or not os.path.isfile(out):
            print(f"=== {rev}: solve FAILED (log {job.log(f'plan-{rev}')}) ==="); return False
        with open(out) as fh:
            t = yaml.safe_load(fh)["tensors"]
        k = float(bits)   # the target rate, which a half-step recipe can hit exactly (3.5)
        diffs = sorted((sum(t[x] != e.get(x) for x in t), name) for name, e in existing.items() if set(e) == set(t))
        same = [n for d, n in diffs if d == 0]
        hist = dict(sorted({b: list(t.values()).count(b) for b in set(t.values())}.items()))
        print(f"=== {rev}: {sum(b != k for b in t.values())} of {len(t)} tensors off uniform {k}; bits {hist} ===")
        if same:
            print(f"    SAME RECIPE as {', '.join(same)}: converting would duplicate it")
        else:
            print("    nearest: " + ", ".join(f"{n} ({d} tensors differ)" for d, n in diffs[:4]) if diffs else "    nothing to compare with")
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
                        "generator": cal_generator_named(job),
                        **{k: manifest[k] for k in ("shares", "doc_kinds", "ml_frac", "cal_rows", "cal_cols", "composition")
                           if k in manifest} }
    else:
        calibration = { "calibration": "exllamav3 default corpus" }
    script = os.path.join(args.exllamav3dir, "convert.py")
    # Where each revision's bits come from: an explicit --measurement; else, if `measure` has run,
    # a blend at --ownvoice-share (default by the model's use: agent-first or not); else uniform
    measured = "self" in measured_categories(job)
    recipe_source = args.recipe and (args.measurement or measured)
    share = args.ownvoice_share if args.ownvoice_share is not None else OWNVOICE_SHARE
    rfn = args.rfn or (os.path.join(job.measure, "rfn.json") if os.path.isfile(os.path.join(job.measure, "rfn.json")) else None)
    if recipe_source:
        print(f"=== per-tensor recipe from " + (args.measurement if args.measurement else
              f"the measurements, own-voice share {share}" + (" (default)" if args.ownvoice_share is None else "")) + " ===")
    elif args.recipe and not measured:
        print("=== uniform bits: no measurements (run `measure` for a per-tensor recipe) ===")
    if args.plan:
        return plan_recipes(job, args, recipe_source, share, rfn)

    def task(rev, bits, headbits):
        def run(slot, echo):
            revdir = job.revdir(rev)
            workdir = os.path.join(revdir, "_work")
            donefile = os.path.join(revdir, "quantization_config.json")
            # Outside revdir on purpose: `upload` publishes each revision directory whole.
            log_path = job.log(rev)
            say = print if echo else (lambda *a, **k: None)
            # With --measurement, the bits come from a per-tensor recipe solved for this rate
            # (sc_optimize), passed to convert.py in place of -b/-hb
            recipe = os.path.join(job.dir, "_recipes", f"{rev}.yaml") if recipe_source else None
            if recipe and not os.path.isfile(donefile) and not os.path.isfile(os.path.join(workdir, "args.json")):
                os.makedirs(os.path.dirname(recipe), exist_ok=True)
                # blended here, per revision, so a share is a quantize-time choice (and could vary by rung)
                meas = args.measurement or blend_measurement(job, ownvoice_groups(job, share, args.category_shares),
                                                             os.path.join(job.dir, "_recipes", f"{rev}.measurement.json"))
                rc = run_logged([ "python3", os.path.join(args.exllamav3dir, "sc_optimize.py"),
                                  "-m", meas, "-b", bits, "-hb", headbits,
                                  "-al", str(args.alpha), "-mink", str(args.min_k), "-tie", RECIPE_TIES, "-ks", str(RECIPE_K_STEP), *(["-all", str(args.alpha_low)] if args.alpha_low else []), *recipe_floor_args(meas), "-o", recipe ] +
                                ([ "-rr", rfn ] if rfn else []),
                                job.log(f"{rev}-recipe"), None, echo)
                if rc or not os.path.isfile(recipe):
                    print(f"=== RECIPE FOR {rev} FAILED (sc_optimize exit {rc}, log {job.log(f'{rev}-recipe')}); not converting ===")
                    return False
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
                    say(f"=== RESUMING QUANTIZTION OF {revdir} ===")
                    run_logged([ "python3", script, "-w", workdir, "-r", "-d", "0" ], log_path, gpu_env(slot), echo)
                else:
                    say(f"=== QUANTIZING {revdir} ({calibration['calibration']}) ===")
                    run_logged([ "python3", script,
                                 "-vb", "16", "--out_scales", OUT_SCALES,
                                 "-i", path,
                                 "-w", workdir,
                                 "-o", revdir,
                                 "-d", "0" ] +
                               ([ "-rcp", recipe ] if recipe else [ "-hq", "-b", bits, "-hb", headbits ]
                                + ([] if HALF_STEPS else ["--no_half_steps"])) +
                               ([ "--cal_data", job.cal_data ] if args.calibration == "mix" else []),
                               log_path, gpu_env(slot), echo)
            if not os.path.isfile(donefile):
                say(f"=== QUANTIZATION OF {revdir} FAILED after {attempt} attempt(s) ===")
                return False
            say(f"=== QUANTIZATION OF {revdir} COMPLETE ===")
            if os.path.isdir(workdir):
                shutil.rmtree(workdir)
            # Nothing in an EXL3 checkpoint records its calibration (docs/calibration.md), nor the
            # solver inputs behind a recipe's allocation: the recipe is published beside it
            rec = dict(calibration)
            if recipe:
                publish_recipe(job, recipe, os.path.join(revdir, "recipe.yaml"), recipe_description(share, args.measurement))
                rec["recipe"] = { "measurement": recipe_description(share, args.measurement),
                                  "rfn": rfn and os.path.basename(rfn), "alpha": args.alpha, "min_k": args.min_k,
                                  **({"alpha_low": args.alpha_low} if args.alpha_low else {}) }
            with open(os.path.join(revdir, "calibration.json"), "w") as f:
                json.dump(rec, f, indent=1)
            return True
        return rev, run
    specs = bit_specs(args.bits, args.suffix)
    failed = run_jobs([task(*sp) for sp in specs], device_slots(args.device), "QUANTIZE")
    return report_failures(failed, "QUANTIZE", {rev: job.log(rev) for rev, _, _ in specs})

def main():
    parser = argparse.ArgumentParser(description="EXL3 quantization pipeline. Usual order: baseline, traces, "
                                     "measure, quantize, qbench, card, upload")
    parser.add_argument('-x', '--exllamav3dir', default='../deps/exllamav3')
    parser.add_argument('-w', '--workdir', default='work')
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-o', '--org', default='yeasah')
    subparsers = parser.add_subparsers()

    cmd_baseline = subparsers.add_parser('baseline', help="uncalibrated conversions: the card's baseline ladder and "
                                         "the calibration generator (single-GPU jobs, spread over -d)")
    cmd_baseline.add_argument('-b', '--bits', default='3:5,3.5:5,4,5,6')
    cmd_baseline.add_argument('-d', '--device', default='0',
                              help='GPUs, e.g. 0 or 0,1,2,3: one conversion per GPU at a time; list a GPU twice for two on it (nvidia-smi numbering)')
    cmd_baseline.add_argument('--retries', type=int, default=2)
    cmd_baseline.add_argument('--generator', default=True, action=argparse.BooleanOptionalAction,
                              help='also the uncalibrated 8bpw calibration generator `traces` uses (models with a chat template)')
    cmd_baseline.add_argument('--allow-converter-mismatch', action='store_true',
                              help='reuse rungs a different (or unrecorded) converter made, this run only')
    cmd_baseline.add_argument('--adopt-converter', action='store_true',
                              help='record the current converter on present rungs (and the generator) that carry no '
                                   'record, once you have checked the converter has not changed since they were made; '
                                   'never overrides a record of a different converter')
    cmd_baseline.set_defaults(func=do_baseline)


    def trace_args(p):
        p.add_argument('-d', '--device', default='0',
                       help='GPUs for the calibration generator (uncalibrated): its conversion uses the first, the '
                            'calibration trace all of them (a large generator autosplits), e.g. 0,1')
        p.add_argument('--eval-devices', default=None,
                       help='CUDA_VISIBLE_DEVICES for the bf16 eval traces, and for a bf16 calibration trace (default: all visible)')
        p.add_argument('--retries', type=int, default=2)
        p.add_argument('--eval-generator', default=None,
                       help='repo or path that writes the eval traces\' answers (default: the base model). '
                            'For models too large to generate from locally in bf16, the FP8 release with '
                            '--eval-backend vllm. Never an EXL3 checkpoint')
        # vLLM: substantially faster for the eval traces on TP>1 and never measurably slower
        p.add_argument('--eval-backend', default='vllm', choices=['exllamav3', 'vllm'],
                       help='engine for the eval traces (default vllm: one engine per ctx_trace process, with --vllm-args)')
        # bf16 where it fits: on Qwen3-0.6B (2026-10-23, docs/calibration.md) it ties the uncalibrated
        # generator on wild and beats it on swe by 6-8% at 3-4 bpw, against a 3% reseed spread
        p.add_argument('--generator', default='bf16', choices=['bf16', 'uncalibrated'],
                       help="what writes the calibration trace: the eval traces' own generator and engine (default; "
                            "needs that model to fit), or the uncalibrated 8bpw EXL3 stand-in (built if missing) "
                            "where it does not")
        p.add_argument('--cal-backend', default='exllamav3', choices=['exllamav3', 'vllm'],
                       help='engine for an uncalibrated-generator calibration trace; vllm uses --cal-vllm-args')
        p.add_argument('--cal-vllm-args', default='{}',
                       help='vLLM engine arguments for the uncalibrated generator, JSON; e.g. \'{"tensor_parallel_size": 1}\' '
                            'where EXL3 cannot split the model over the GPUs')
        p.add_argument('--vllm-args', default='{}',
                       help='vLLM engine arguments for the eval generator (and a bf16 calibration trace), JSON. '
                            'Qwen3.8-27B-FP8 on 2x16 GB: \'{"max_num_batched_tokens": 1024, "enforce_eager": true, '
                            '"gpu_memory_utilization": 0.968}\'')
        p.add_argument('--sampling-profile', default=None,
                       help='JSON of sampling modes and per-conversation rules (default: '
                            'quantization/sampling/<org>__<model>.json, else the base model\'s '
                            'generation_config.json; "exllamav3-default" for exllamav3\'s own)')
    for name, fn, helptext in (
            ("traces", do_traces, "eval traces, then the calibration trace (`eval-traces` and `cal-traces`)"),
            ("eval-traces", do_eval_traces, "the bf16-generated eval traces; needs only the reference model, so it "
                                            "can run alone on a box sized for it"),
            ("cal-traces", do_cal_traces, "the calibration trace (after eval-traces), from the uncalibrated 8bpw "
                                          "generator or, with --generator bf16, from the eval generator")):
        p = subparsers.add_parser(name, help=helptext)
        trace_args(p)
        p.set_defaults(func=fn)

    cmd_measure = subparsers.add_parser('measure', help="per-tensor sensitivity on the calibration's own-voice rows and "
                                         "on the whole calibration, for quantize's recipes (reused across shares)")
    cmd_measure.add_argument('-d', '--device', default='0', help='GPU for the measurement (the first listed; all with --gpu-split)')
    cmd_measure.add_argument('--rows', default=None,
                             help="rows per category over the defaults (own voice 20, the rest 0), 'pattern=N,...' (fnmatch): "
                                  "e.g. 'self=40,ctx_*=10' to explore mixes. "
                                  "Categories: self, ctx_user:<kind>, ctx_tool:<kind>, loop, raw, random")
    cmd_measure.add_argument('--gpu-split', default=None,
                             help='GB per listed GPU to split the unquantized model across: load caps, filled in order '
                                  'and lifted once loaded, so a card keeps (its VRAM - what it was given) for the '
                                  'measurement. Cap every card but the last at VRAM minus its working set and ~1 GB of '
                                  'CUDA context; the last takes the remainder, and with it the head and the logits, the '
                                  'largest working set. The caps only need to add up to the model. '
                                  'E.g. 9,10 for a 9B on 2x16 GB; 41,41 for a 35B-A3B (67 GB bf16, ~6 GB working set) on '
                                  '2x48 GB')
    cmd_measure.add_argument('--rfn-anchor', default='constant', choices=['constant', 'probe'],
                             help="noise anchors: the converter's constant (default; needs nothing but the model), or "
                                  "probed on the lowest integer-rate baseline rung (re-check the constant after a "
                                  "converter change)")
    cmd_measure.add_argument('--boundary-cpu', default=True, action=argparse.BooleanOptionalAction,
                             help="keep sc_measure's per-module input cache in host memory (needed past ~10 rows of a 9B on 16 GB)")
    cmd_measure.set_defaults(func=do_measure)

    cmd_quantize = subparsers.add_parser('quantize', help="the calibrated ladder")
    cmd_quantize.add_argument('-b', '--bits', default='3:5,3.5:5,4,5,6')
    cmd_quantize.add_argument('-d', '--device', default='0',
                              help='GPUs, e.g. 0 or 0,1,2,3: one conversion per GPU at a time; list a GPU twice for two on it (nvidia-smi numbering)')
    cmd_quantize.add_argument('--calibration', default='mix', choices=['mix', 'default'],
                              help='mix: the conversational trace from `traces`; default: the bundled corpus')
    cmd_quantize.add_argument('--retries', type=int, default=2,
                              help='resume a failed revision this many times before stopping')
    cmd_quantize.add_argument('--plan', action='store_true',
                              help='solve the recipes and compare them with every recipe already solved (identical = '
                                   'the conversion would duplicate a variant); convert nothing')
    cmd_quantize.add_argument('--suffix', default=None,
                              help='appended to every revision name (e.g. 4.00bpw-rcp), to keep variants apart')
    cmd_quantize.add_argument('--recipe', default=True, action=argparse.BooleanOptionalAction,
                              help='per-tensor recipe from `measure`\'s measurements when present (--no-recipe: uniform bits)')
    cmd_quantize.add_argument('--ownvoice-share', type=float, default=None,
                              help='own-voice share of the recipe measurement, 0..1 (default 1: own voice only; below 1 '
                                   'needs other categories measured, measure --rows)')
    cmd_quantize.add_argument('--category-shares', default=None,
                             help="scale recipe shares per category (0..1), 'pattern=N,...' (fnmatch): e.g. 'ctx_*:code=0.5,self=0'. "
                                  "Categories: self, ctx_user:<kind>, ctx_tool:<kind>, loop, raw, random "
                                  "Kinds: code, diff, ml, technical, web, wiki")
    cmd_quantize.add_argument('--measurement', default=None,
                              help='sc_measure.py sensitivity JSON: each revision\'s bits come from a recipe sc_optimize '
                                   'solves for its rate (published beside it as recipe.yaml) instead of uniform -b/-hb')
    cmd_quantize.add_argument('--rfn', default=None,
                              help='per-tensor error anchors (sc_rfn_probe.py JSON) for sc_optimize, as used for the measurement')
    cmd_quantize.add_argument('--alpha', type=float, default=2.0,
                              help='sc_optimize KLD-vs-noise exponent (its docs recommend 2.0 over the fitted value)')
    cmd_quantize.add_argument('--min-k', type=int, default=2, help='sc_optimize minimum per-tensor bits')
    cmd_quantize.add_argument('--alpha-low', type=float, default=None,
                              help='sc_optimize exponent for demotions below the 2-bit rfn anchor (with --min-k 1; '
                                   'upstream measured 2.35 on Qwen3.8-27B for 2->1 bits)')
    cmd_quantize.set_defaults(func=do_quantize)

    cmd_qbench = subparsers.add_parser('qbench', help="score every set on the eval slices against the bf16 reference")
    cmd_qbench.add_argument('--allow-converter-mismatch', action='store_true',
                            help='score baseline rungs a different (or unrecorded) converter made')
    cmd_qbench.add_argument('--slices', default=None,
                            help="only these slices, e.g. wild,swe ('scored': the composite's; 'diag': the diagnostic ones). "
                                 "For exploration: no pruning, and the card wants every slice")
    cmd_qbench.add_argument('--sample', default=None,
                            help="screen on a fixed subset of a slice's rows, 'slice=N,...' (e.g. swe=50): its own trace "
                                 "(eval_<slice>.s<N>.json), results (qb_<slice>.s<N>.json) and reference logits")
    cmd_qbench.add_argument('--revisions', default=None,
                            help="only these revisions (e.g. a new --suffix variant; 'none' for references only), merged "
                                 "into the existing results; the baseline is not rescored, references only if given")
    cmd_qbench.add_argument('--logit_cache', default=None,
                            help='logit cache dir (default: <model>-exl3/_logit_cache, owned by this model)')
    # Reference logits are full-vocabulary: ~35 GB for a 60-conversation WildChat trace on a
    # 248k vocabulary, ~150 GB for a full set of slices. Pruning after each series keeps only
    # what the current series used, so the cap can be generous.
    cmd_qbench.add_argument('--logit_cache_size', default=200, type=int, help='cap in GB')
    cmd_qbench.add_argument('--prune', default=True, action=argparse.BooleanOptionalAction,
                            help='after a successful series, evict reference logits it did not use')
    cmd_qbench.add_argument('-d', '--device', default='0',
                            help='GPUs, e.g. 0 or 0,1,2,3: one scorer per GPU at a time, the quants of each slice split across them (nvidia-smi numbering)')
    cmd_qbench.add_argument('--reference-gguf', action='append',
                            help='a GGUF repo to compare against (repeatable): the --gguf-quants files, scored by llama.cpp')
    # The best GGUF per byte at each size: below ~3.5 bpw the lattice-codebook I-quants (EXL3's
    # nearest relatives; K-quants are scalar), IQ4_XS's non-linear grid at 4, scalar K-quants above.
    # No Q8_0: ~8.5 real bpw is outside any card's comparison window, so it would be scored and dropped
    cmd_qbench.add_argument('--gguf-quants', default='IQ3_XXS,IQ3_M,IQ4_XS,Q5_K_M,Q6_K',
                            help='quant types to take from each --reference-gguf repo (a type the repo lacks is left out)')
    # Off by default: it checks the bench's relationship to llama.cpp, not this model's quants
    # (within the noise floor on every slice, Qwen3-0.6B 2026-10-10), and a bf16 GGUF is the most
    # expensive file in the set. Rerun when llama.cpp or the bench changes
    cmd_qbench.add_argument('--gguf-parity', default=False, action=argparse.BooleanOptionalAction,
                            help="also score the repo's bf16 GGUF: llama.cpp's own numerics against the bf16 reference")
    cmd_qbench.add_argument('--reference-min-bits', type=float, default=0.0,
                            help='skip reference rungs below this bit rate (e.g. 2.0)')
    cmd_qbench.add_argument('--reference-max-bits', type=float, default=8.9,
                            help='skip reference rungs above this bit rate (e.g. 6.0)')
    cmd_qbench.add_argument('--threshold', action='append',
                            help="a checkpoint everybody knows, e.g. the publisher's FP8 (REPO[@REVISION], repeatable): "
                                 "scored on every slice by the bf16 reference's own engine and drawn as a line on the "
                                 "card's size plot and TECHNICAL.md's slice plots, not compared in section 2")
    cmd_qbench.add_argument('--reference', action='append',
                            help='HF repo of an existing set of this model\'s quants to compare against '
                                 '(card section 2); repeatable')
    cmd_qbench.set_defaults(func=do_qbench)

    cmd_compare = subparsers.add_parser('compare', help="table of scored revisions: per slice and composite, against the "
                                         "uncalibrated baseline or a --base revision (for exploring variants)")
    cmd_compare.add_argument('--revisions', default=None, help='revisions to show (default: every scored one)')
    cmd_compare.add_argument('--base', default=None, help='show every number as a ratio to this revision')
    cmd_compare.add_argument('--slices', default=None, help="only these slices ('scored', 'diag', or names)")
    cmd_compare.add_argument('--no-references', action='store_true', help='leave out reference rows (ref:<repo>@<branch>)')
    cmd_compare.add_argument('--sample', default=None, help="read a slice's sampled results instead, 'slice=N' (as qbench --sample)")
    cmd_compare.set_defaults(func=do_compare)

    cmd_card = subparsers.add_parser('card', help="the model card and TECHNICAL.md from the qbench results")
    cmd_card.add_argument('--template', default='templates/model_card.jinja')
    # Raw-text perplexity is meaningless for some models (gpt-oss: ~3600 even for
    # the unquantized reference, on every engine); KLD stays valid, so drop only PPL.
    cmd_card.add_argument('--ppl', default=True,
                          action=argparse.BooleanOptionalAction)
    cmd_card.set_defaults(func=do_card)

    cmd_fetch = subparsers.add_parser('fetch', help="card-only job dir from the published repo (no weights); "
                                     "with --uncalibrated, the published uncalibrated conversions into a working job dir")
    cmd_fetch.add_argument('--uncalibrated', action='store_true',
                           help='baseline rungs and the generator from <repo>-uncalibrated, so `baseline` can skip them')
    cmd_fetch.set_defaults(func=do_fetch)

    cmd_upload = subparsers.add_parser('upload', help="the ladder and main to <org>/<model>-exl3")
    cmd_upload.add_argument('--private', default=True,
                            action=argparse.BooleanOptionalAction)
    cmd_upload.add_argument('--uncalibrated', action='store_true',
                            help='also the baseline rungs and the generator, to <org>/<model>-exl3-uncalibrated '
                                 '(a separate, not-for-inference repo)')
    cmd_upload.add_argument('-M', '--message', default=None,
                            help='commit message for every upload (default: names the model and revision)')
    cmd_upload.add_argument('--quants', default=True, action=argparse.BooleanOptionalAction,
                            help='the model repo itself (--no-quants with --uncalibrated: only the uncalibrated repo)')
    cmd_upload.set_defaults(func=do_upload)

    args = parser.parse_args()
    fetched = os.path.join(args.workdir, f"{args.model.split('/')[1]}-exl3", "FETCHED.json")
    if os.path.isfile(fetched) and args.func not in (do_card, do_fetch, do_upload):
        sys.exit(f"{os.path.dirname(fetched)} was made by `fetch` from the published repo and holds no weights: "
                 "only card, upload (main only) and fetch run there")
    sys.stdout = Stamped(sys.stdout)       # after parsing: --help and usage errors stay unstamped
    ok = args.func(Job(args), args)
    sys.exit(0 if ok is not False else 1)

if __name__ == "__main__":
    main()
