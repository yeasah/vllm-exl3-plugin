"""Client: decode tok/s and prefill via a running server, max_tokens=1 for prefill."""
import json, sys, time, random, urllib.request
base = "http://127.0.0.1:8123/v1"; model = "m"
def post(p, n, **kw):
    b = json.dumps({"model": model, "prompt": p, "max_tokens": n, "temperature": 0.7, "seed": 3, **kw}).encode()
    r = urllib.request.Request(base + "/completions", b, {"Content-Type": "application/json"})
    t = time.time(); o = json.load(urllib.request.urlopen(r, timeout=1800)); return time.time() - t, o
Ps = ["<start_of_turn>user\nExplain how a skip list works, with code in Python.<end_of_turn>\n<start_of_turn>model\n",
      "<start_of_turn>user\nWrite a detailed account of the Peloponnesian War.<end_of_turn>\n<start_of_turn>model\n",
      "<start_of_turn>user\nDescribe how to design a Kalman filter for a drone.<end_of_turn>\n<start_of_turn>model\n"]
post(Ps[0], 8)
for rep in range(2):
    tt = gen = 0
    for p in Ps:
        t, o = post(p, 200, ignore_eos=True); tt += t; gen += o["usage"]["completion_tokens"]
    print(f"decode: {gen/tt:.1f} t/s ({gen} tok)", flush=True)
r = random.Random(5); w = "the of and to in a is that for it as was with be by on not he his but at are this have from or had".split()
for rep in range(2):
    p = " ".join(r.choice(w) + str(r.randrange(99999)) for _ in range(400))
    t, o = post(p, 1); n = o["usage"]["prompt_tokens"]
    print(f"prefill: {n} tok in {t:.2f}s = {n/t:.0f} t/s", flush=True)
