#!/usr/bin/env python3

import argparse
import json
import sys
import os
import re
import glob as globmod
import subprocess
import shutil
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

    def revdir(self, rev):
        return os.path.join(self.dir, rev)

    def revisions(self):
        if not os.path.isdir(self.dir):
            return []
        return sorted(r for r in os.listdir(self.dir)
                      if r not in ('main', 'logs') and not r.startswith('_')
                      and os.path.isdir(self.revdir(r)))

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
# The card's composite is the independent tier only (TODO.md card-composite): real users'
# first prompts (WildChat) and real coding-agent sessions (Open-SWE-Traces), as
# (conversations, weight). Eval traces come from bf16: a calibration scores worse on another
# quant's sampled text, and no pipeline quant may write the text it is judged on.
EVAL_SLICES = {"wild": (60, 0.80), "swe": (40, 0.20)}
EVAL_SEED = 1


def collect_metadata(job, qbench):
    """qbench: {slice: results list}. Excess KLD is each quant's KLD minus the slice's noise
    floor (bf16 rounding noise), so slices with different floors add up meaningfully."""
    data = { 'this_model': job.repo,
             'base_model': job.base_repo }

    base_sizes = tensor_survey_remote(job.base_repo, 'main')
    data['base_disk_bytes'] = base_sizes['total']
    data['multimodal'] = base_sizes['encoder'] > 0

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
    return data

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
            'encoder': 0 }
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
                          revision=rev)

def plot_quality_vs_size(job, meta, path):
    """Section 1 of the card: this card's own checkpoints, quality against size, no comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, GRID, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6"
    key = 'composite' if meta['composite'] else 'raw_excess'
    revs = [r for r in meta['revisions'] if r.get(key)]
    xs = [(r['disk_bytes'] - r['embed_bytes']) / 1024**3 for r in revs]
    ys = [r[key] for r in revs]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.grid(True, which="minor", color="#f0efea", lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors=INK2)
    ax.plot(xs, ys, color=LINE, lw=2, zorder=2)
    ax.scatter(xs, ys, s=70, color=LINE, edgecolor=SURF, linewidth=1.5, zorder=3)
    for r, x, y in zip(revs, xs, ys):
        ax.annotate(f"{r['bits']:.2f} bpw", (x, y), xytext=(8, 4), textcoords="offset points",
                    fontsize=8, color=INK)
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

def do_card(job, args):
    os.makedirs(job.main, exist_ok=True)

    qbench = {}
    for sl in list(EVAL_SLICES) + ["raw"]:
        path = os.path.join(job.main, f"qb_{sl}.json")
        if os.path.isfile(path):
            with open(path) as f:
                qbench[sl] = json.load(f)
    if 'raw' not in qbench:
        print("=== no qbench results: run qbench first ===")
        return False
    meta = collect_metadata(job, qbench)
    plot_quality_vs_size(job, meta, os.path.join(job.main, "quality_vs_size.png"))

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

def qbench_project(job, args, slice_, revisions):
    """One qbench project per slice: a bf16-generated eval trace, or raw web text."""
    project = { "title": f"{job.name}: {slice_}",
                # relative to main/, as when the project file was rendered there
                "logit_cache": { "dir": os.path.abspath(os.path.join(job.main, args.logit_cache)),
                                 "max_size_gb": args.logit_cache_size },
                "models": [ { "label": "HF BF16", "group": "reference", "engine": "transformers",
                              "repo": job.base_repo, "options": { "streaming": True } } ] +
                          [ { "label": rev, "group": "EXL3", "engine": "exllamav3",
                              "source": os.path.abspath(job.revdir(rev)) } for rev in revisions ],
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
    slices = [sl for sl in EVAL_SLICES if os.path.isfile(job.eval_trace(sl))] + ["raw"]
    missing = [sl for sl in EVAL_SLICES if sl not in slices]
    if missing:
        print(f"=== no eval traces for {', '.join(missing)}: the card falls back to raw text "
              f"(run `traces` first for the composite) ===")
    print(f"=== running bench on revisions: {' '.join(revisions)}; slices: {' '.join(slices)} ===")
    script = os.path.join(args.exllamav3dir, "eval", "qbench.py")
    for sl in slices:
        qbench_file = os.path.join(job.main, f"qbench-{sl}.yaml")
        with open(qbench_file, "w") as f:
            yaml.safe_dump(qbench_project(job, args, sl, revisions), f, sort_keys=False)
        rc = run_logged([ "python3", script, qbench_file, "-d", str(args.device) ], job.log(f"qbench-{sl}"))
        if rc:
            print(f"=== qbench on {sl} FAILED (exit {rc}) ===")
            return False
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
    print(f"=== calibration trace -> {job.cal_data} ===")
    if not run_until(job.cal_data,
                     [ "python3", "-u", ctx_trace, "-m", gen, "-cs", "65536",
                       "-o", os.path.join(job.traces, "cal"), "--docs", "cal", "--cal_out", job.cal_data,
                       "-tv", CAL_TEMPLATE_VARS, "--max_new_tokens", str(CAL_MAX_NEW_TOKENS) ],
                     job.log("traces-cal"), tries, env=one_gpu):
        print("=== calibration trace FAILED ==="); return False

    # bf16 may need every visible GPU; a layer split on stock torch <= 2.14.1 is exposed to
    # pytorch#196258 (docs/upstream.md "PyTorch"), which the local library rebuild fixes
    check = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "torch-c10cuda-fix.sh")
    if subprocess.call([check, "--check"]):
        print("!! torch is not patched for pytorch#196258; a multi-GPU bf16 run may fault (fork free_mem() covers load)")
    for sl, (n, _) in EVAL_SLICES.items():
        print(f"=== eval trace ({sl}, {n} conversations, bf16) -> {job.eval_trace(sl)} ===")
        if not run_until(job.eval_trace(sl),
                         [ "python3", "-u", ctx_trace, "-m", path, "-cs", "65536",
                           "-o", os.path.join(job.traces, "eval"), "--docs", "eval",
                           "--slices", sl, "-n", str(n), "--seed", str(EVAL_SEED) ],
                         job.log(f"traces-eval-{sl}"), tries,
                         env=None if args.eval_devices is None else {"CUDA_VISIBLE_DEVICES": args.eval_devices}):
            print(f"=== eval trace {sl} FAILED ==="); return False
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
                        **{k: manifest[k] for k in ("shares", "ml_frac", "cal_rows", "cal_cols", "composition")
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
    cmd_traces.set_defaults(func=do_traces)

    cmd_quantize = subparsers.add_parser('quantize')
    cmd_quantize.add_argument('-b', '--bits', default='2:5,3:5,4,5,6')
    cmd_quantize.add_argument('-d', '--device', default=0)
    cmd_quantize.add_argument('--calibration', default='mix', choices=['mix', 'default'],
                              help='mix: the conversational trace from `traces`; default: the bundled corpus')
    cmd_quantize.add_argument('--retries', type=int, default=2,
                              help='resume a failed revision this many times before stopping')
    cmd_quantize.set_defaults(func=do_quantize)

    cmd_qbench = subparsers.add_parser('qbench')
    cmd_qbench.add_argument('--logit_cache', default='../../_logit_cache')
    cmd_qbench.add_argument('--logit_cache_size', default=25)
    cmd_qbench.add_argument('-d', '--device', default=0)
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
