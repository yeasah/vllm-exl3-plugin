"""What a quant says where recall scored it as losing a fact: forgetting, or the same answer in another form?

For each arm, the N known questions it kept least of (per question min(P, P_ref) / P_ref, as qbench's recall_kept),
the quant's greedy answer to the scored prompt, classified mechanically against the scored answer and, for PopQA, the
question's other aliases. No judge: the "different" bucket is printed for reading, and the reading is the result
(docs/recall.md, "Does recall score answer form?"). --leak then bounds what form slips move the card's kept.

    python3 tools/recall_scan.py -w quantization/work/<job> --arms 3.50bpw-H5,4.00bpw -n 100 -o scan.json --show_different
"""
import argparse, json, math, os, re, sys, unicodedata
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deps", "exllamav3"))


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(the|a|an|of)\b", " ", s)
    return " ".join(s.split())


HEDGE = re.compile(r"\b(i don.?t know|not sure|unknown|cannot|can.?t|no information|unclear|i.?m not)\b", re.I)


def classify(gen, answer, aliases):
    g = gen.strip().split("\n")[0].strip().rstrip(".")
    if not g or HEDGE.search(g):
        return "hedge"
    if g.startswith(answer) or g == answer:
        return "same"           # the scored string after all: a near-tie or the engine
    ng, forms = norm(g), [norm(a) for a in [answer] + aliases if norm(a)]
    if any(ng == f for f in forms):
        return "form"           # equal after case, articles and punctuation, or another alias
    if any(f and (f in ng or (len(ng) >= 3 and ng in f)) for f in forms):
        return "partial"        # contains it, or is a shorter piece of it ("Leonardo" for "Leonardo da Vinci")
    return "different"


def main(a):
    import torch
    from exllamav3 import model_init, Generator, Job, GreedySampler
    trace = json.load(open(os.path.join(a.workdir, "_traces", "eval_recall.json")))
    pool = json.load(open(os.path.join(a.workdir, "_traces", "eval_recall_pool.json")))
    aliases = defaultdict(list)
    for r in pool["rows"]:
        aliases[r["qid"]].append(r["answer"])
    qb = {x["label"]: x for x in json.load(open(os.path.join(a.workdir, "main", "qb_recall.json")))}
    out = {}
    for arm in a.arms.split(","):
        rows = qb[arm]["rows"]
        known = [i for i, x in enumerate(rows) if x and x.get("ref_hit")]
        kept = {i: min(math.exp(rows[i]["lp"]), math.exp(rows[i]["ref_lp"])) / math.exp(rows[i]["ref_lp"]) for i in known}
        worst = sorted(known, key=kept.get)[:a.n]
        # the arm's checkpoint, in the job dir
        p = argparse.ArgumentParser(); model_init.add_args(p, cache=True)
        args = p.parse_args(["-m", os.path.join(a.workdir, arm), "-cs", "16384"])
        model, config, cache, tokenizer = model_init.init(args, quiet=True, progress=False)
        gen = Generator(model=model, cache=cache, tokenizer=tokenizer)
        jobs = []
        for i in worst:
            t = trace["rows"][i]
            jobs.append(Job(input_ids=torch.tensor([t["input_ids"]]), max_new_tokens=24, sampler=GreedySampler(),
                            stop_conditions=config.eos_token_id_list, identifier=i))
        gen.enqueue(jobs)
        while gen.num_remaining_jobs():
            gen.iterate()
        res = []
        for job in jobs:
            i, t = job.identifier, trace["rows"][job.identifier]
            others = [x for x in aliases[t["qid"]] if x != t["answer"]]
            text = job.full_completion
            res.append({"row": i, "kind": t["kind"], "answer": t["answer"], "gen": text, "kept": kept[i],
                        "ref_p": math.exp(rows[i]["ref_lp"]), "class": classify(text, t["answer"], others)})
        model.unload(); del gen, cache, model; torch.cuda.empty_cache()
        c = Counter(r["class"] for r in res)
        print(f"== {os.path.basename(a.workdir.rstrip('/'))} {arm}: {len(known)} known, worst {len(res)} by kept "
              f"(kept {min(r['kept'] for r in res):.3f}-{max(r['kept'] for r in res):.3f})")
        print("   " + "  ".join(f"{k} {c[k]}" for k in ("same", "form", "partial", "different", "hedge")))
        for r in res:
            if r["class"] in ("form", "partial") or (r["class"] == "different" and a.show_different):
                print(f"   [{r['class']:9}] {r['kind'][:18]:18} want {r['answer'][:40]!r:42} got {r['gen'].strip()[:50]!r}")
        out[arm] = res
    json.dump(out, open(a.output, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--workdir", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--show_different", action="store_true")
    main(ap.parse_args())
