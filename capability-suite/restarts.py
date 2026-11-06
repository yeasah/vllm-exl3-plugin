import json, glob, re, sys, statistics as st

MARKERS = re.compile(r"""(
   actually,?\s+wait | wait,?\s+(?:but|no|actually|let) | let\s+me\s+reconsider
 | let\s+me\s+re-?(?:check|do|examine|think|compute|calculate|read|visit)
 | on\s+second\s+thought | hold\s+on | scratch\s+that | i\s+made\s+(?:an\s+)?(?:error|mistake)
 | but\s+actually | hmm,?\s+(?:wait|but|actually) | that'?s\s+(?:not\s+right|wrong)
 | let'?s\s+re-?(?:check|do|examine|think)
)""", re.I | re.X)

def reasoning(r):
    for m in r.get("messages", []):
        if m.get("role") != "assistant": continue
        c = m.get("content")
        if isinstance(c, list):
            return "".join(b.get("reasoning") or "" for b in c if isinstance(b, dict))
        if isinstance(c, str): return c
    return ""

def load(run):
    pf = glob.glob(f"{run}/predictions/*/gpqa_diamond_default.jsonl")[0]
    rf = glob.glob(f"{run}/reviews/*/gpqa_diamond_default.jsonl")[0]
    rev = {json.loads(l)["index"]: json.loads(l)["sample_score"]["score"]["value"]["accuracy"]
           for l in open(rf)}
    out = {}
    for line in open(pf):
        r = json.loads(line)
        t = reasoning(r); tok = r["model_output"]["usage"]["output_tokens"]
        n = len(MARKERS.findall(t))
        out[r["index"]] = dict(tokens=tok, restarts=n,
                               density=n / max(1, tok / 1000), correct=rev.get(r["index"]))
    return out
