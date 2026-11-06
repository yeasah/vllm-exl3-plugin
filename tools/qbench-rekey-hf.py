#!/usr/bin/env python3
"""One-off (2026-10-22, exllamav3 062304c): re-key qbench cache entries for sources in a Hugging Face hub cache from (path, stamp)
to (repo, commit, file), and the raw slice's data key from the tokenizer's resolved path to its
repo and commit. Old model keys are taken from the manifest (key -> source as noted when scored),
so the old files need not exist. Raw data keys are mapped only where recomputing the old key from a
candidate tokenizer path reproduces the recorded one exactly.
Usage: tools/qbench-rekey-hf.py [--apply] <cache qbench dir>..."""
import sys, os, json, glob
import yaml
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deps", "exllamav3", "eval"))
from qbench.data import sha_key, source_identity, raw_data_key, HF_SNAPSHOT
from qbench.measure import BF16_ROUNDING_EPS

def new_model_key(e, noise):
    raw = e.get("options", {}) or {}
    return sha_key({"v": 1, "engine": e["engine"], **source_identity(e["source"]),
                    "options": {k: v for k, v in raw.items() if k != "streaming"},
                    "noise": BF16_ROUNDING_EPS if noise else 0,
                    **({"quantize_keying": 2} if "quantize" in raw else {})})

def find_project(root, name):
    job = os.path.dirname(os.path.dirname(os.path.abspath(root)))
    for p in [os.path.join(job, "main", name), os.path.join(job, name)] + glob.glob(os.path.join(job, "*", name)):
        if os.path.isfile(p):
            return p
    return None

apply = "--apply" in sys.argv
for root in [a for a in sys.argv[1:] if a != "--apply"]:
    man = json.load(open(os.path.join(root, "manifest.json")))
    remap, why = {}, {}
    # newest first: where two old keys (two cache locations) reach one new key, the newest wins
    for k, e in sorted(man.items(), key=lambda kv: kv[1].get("seen", ""), reverse=True):
        if not e.get("source") or not HF_SNAPSHOT.search(os.path.normpath(e["source"])):
            continue
        parts = k.split("_")
        role = e.get("role")
        # the model key: the reference's for its logits and self-score (<data>_<ref>_self_m5)
        mk = parts[1] if role in ("reference logits", "reference self-score") else parts[2]
        assert len(mk) == 16, (k, role)
        nk = new_model_key(e, role == "noise floor")
        if mk != nk and mk not in remap:
            remap[mk] = nk; why[mk] = e.get("label")
    # raw data keys
    hf_sources = {os.path.normpath(e["source"]) for e in man.values()
                  if e.get("source") and HF_SNAPSHOT.search(os.path.normpath(e["source"]))}
    raw_old = {k.split("_")[0] for k, e in man.items() if os.path.basename(e.get("project", "")).startswith("qbench-raw")}
    for proj in sorted({e["project"] for e in man.values() if os.path.basename(e.get("project", "")).startswith("qbench-raw")}):
        pf = find_project(root, proj)
        if not pf:
            continue
        p = yaml.safe_load(open(pf))
        if "test_data" not in p:
            continue
        tok = p["tokenizer"]
        for src in hf_sources:
            ident = source_identity(src)
            if ident["hf"] != tok.get("repo") or ident["file"]:
                continue
            old = sha_key({"v": 1, "test_data": p["test_data"], "tokenizer": {**tok, "source": src}})
            if old in raw_old and old not in remap:
                new = raw_data_key({"test_data": p["test_data"], "tokenizer": {**tok, "source": src}})
                remap[old] = new; why[old] = f"raw data ({proj}, verified)"
    unverified = raw_old - set(remap)
    files = set(os.listdir(root))
    moves, clash = [], []
    # files under the newest old keys move first, so a superseded duplicate never takes the slot
    rank = {o: i for i, o in enumerate(remap)}
    def order(f):
        return sorted(rank[x] for x in f.partition(".")[0].split("_") if x in rank) or [len(rank)]
    for f in sorted(files, key=lambda f: (order(f), f)):
        stem, dot, ext = f.partition(".")
        p = stem.split("_")
        q = [remap.get(x, x) if len(x) == 16 else x for x in p]
        if q != p:
            g = "_".join(q) + (dot + ext if dot else "")
            (clash if g in files or g in {m[1] for m in moves} else moves).append((f, g))
    print(f"{root}: {len(remap)} key(s) remapped, {len(moves)} file(s) to move, {len(clash)} superseded duplicate(s) left as they are"
          + (f"; raw data keys not verified (left alone): {sorted(unverified)}" if unverified else ""))
    for o, n in remap.items():
        print(f"   {o} -> {n}  {why[o]}")
    if apply:
        for f, g in moves:
            os.rename(os.path.join(root, f), os.path.join(root, g))
        new = {}
        for k, e in sorted(man.items(), key=lambda kv: kv[1].get("seen", "")):
            new["_".join(remap.get(x, x) if len(x) == 16 else x for x in k.split("_"))] = e
        tmp = os.path.join(root, "manifest.json.tmp")
        json.dump(new, open(tmp, "w"), indent=1)
        os.replace(tmp, os.path.join(root, "manifest.json"))
