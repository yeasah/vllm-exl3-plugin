#!/usr/bin/env python3

import argparse
import json
import sys
import os
import re
import glob as globmod
import subprocess
import shutil
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

    def revdir(self, rev):
        return os.path.join(self.dir, rev)

    def revisions(self):
        if not os.path.isdir(self.dir):
            return []
        return sorted(r for r in os.listdir(self.dir)
                      if r != 'main' and os.path.isdir(self.revdir(r)))

    def is_complete(self, rev):
        # A revision is finished when convert.py wrote its config; main is finished
        # when the aggregate card has been rendered. README.md alone cannot say this:
        # convert.py copies the base model's README into every revision, so it is
        # present long before anything has been generated.
        marker = "README.md" if rev == 'main' else "quantization_config.json"
        return os.path.isfile(os.path.join(self.revdir(rev), marker))


def collect_metadata(job, qbench):
    data = { 'this_model': job.repo,
             'base_model': job.base_repo }

    base_sizes = tensor_survey_remote(job.base_repo, 'main')
    data['base_disk_bytes'] = base_sizes['total']
    data['multimodal'] = base_sizes['encoder'] > 0
        
    qb_base = parse_qbench(qbench, 'Noise floor')
    data['base_kld'] = qb_base['kld']
    data['base_ppl'] = qb_base['ppl']

    revs = []

    for rev in job.revisions():
        sizes = tensor_survey_local(job.revdir(rev))
        quant = quant_config(job, rev)
        if not quant:
            continue
        bits = quant['bits']
        qb_rev = parse_qbench(qbench, rev)
        if qb_rev:
            revs.append({ 'name': rev,
                          'disk_bytes': sizes['total'],
                          'embed_bytes': sizes['embed'],
                          'encoder_bytes': sizes['encoder'],
                          'blockq_bytes': int(sizes['embed'] / 16 * 4.5),
                          'kld': qb_rev['kld'],
                          'ppl': qb_rev['ppl'],
                          'bits': bits })
    data['revisions'] = sorted(revs, key=lambda x: x['bits'])
    return data

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

def do_card(job, args):
    os.makedirs(job.main, exist_ok=True)

    with open(os.path.join(job.main, "qb_results.json"), "r") as f:
        qbench = json.load(f)
    meta = collect_metadata(job, qbench)

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

def do_qbench(job, args):
    os.makedirs(job.main, exist_ok=True)

    params = { "this_model": job.name,
               "base_model": job.base_repo,
               "out_dir": job.main,
               "logit_cache_dir": args.logit_cache,
               "logit_cache_size": args.logit_cache_size,
               "revisions": {rev: os.path.join("..", rev)
                             for rev in job.revisions() if job.is_complete(rev)} }

    if len(params['revisions']):
        print(f"=== running bench on revisions: {' '.join(params['revisions'])} ===")
        jinja = Environment(loader = FileSystemLoader(os.path.dirname(args.template)))
        qbench = jinja.get_template(os.path.basename(args.template))
        qbench_file = os.path.join(job.main, "qbench.yaml")
        with open(qbench_file, "w") as f:
            f.write(qbench.render(params))

        script=os.path.join(args.exllamav3dir, "eval", "qbench.py")
        subprocess.call([ "python3", script,
                          qbench_file,
                          "-d", str(args.device) ])
    else:
        print("=== no revisions found ===")

def do_quantize(job, args):
    path = snapshot_download(repo_id=job.base_repo)
    
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

        if not os.path.isfile(donefile):
            if os.path.isfile(os.path.join(workdir, "args.json")):
                print(f"=== RESUMING QUANTIZTION OF {revdir} ===")
                subprocess.call([ "python3", script,
                                  "-w", workdir,
                                  "-r",
                                  "-d", str(args.device) ])
            else:
                print(f"=== QUANTIZING {revdir} ===")
                subprocess.call([ "python3", script,
                                  "-hq",
                                  "-b", bits,
                                  "-hb", headbits,
                                  "-vb", "16",
                                  "-i", path,
                                  "-w", workdir,
                                  "-o", revdir,
                                  "-d", str(args.device) ])

        if not os.path.isfile(donefile):
            print(f"=== QUANTIZATION OF {revdir} FAILED ===")
            return False
        else:
            print(f"=== QUANTIZATION OF {revdir} COMPLETE ===")
            if os.path.isdir(workdir):
                shutil.rmtree(workdir)
    return True
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--exllamav3dir', default='../deps/exllamav3')
    parser.add_argument('-w', '--workdir', default='work')
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-o', '--org', default='yeasah')
    subparsers = parser.add_subparsers()
    
    cmd_quantize = subparsers.add_parser('quantize')
    cmd_quantize.add_argument('-b', '--bits', default='2:5,3:5,4,5,6')
    cmd_quantize.add_argument('-d', '--device', default=0)
    cmd_quantize.set_defaults(func=do_quantize)

    cmd_qbench = subparsers.add_parser('qbench')
    cmd_qbench.add_argument('--template', default='templates/qbench.jinja')
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
    args.func(Job(args), args)

if __name__ == "__main__":
    main()
