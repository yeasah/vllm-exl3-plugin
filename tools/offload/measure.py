"""Client: decode tok/s and prefill via a running server, max_tokens=1 for prefill.
usage: measure.py [OUT]   OUT gets greedy completions and prompt logprobs, to diff across placements."""
import json, sys, time, random, urllib.request
base = "http://127.0.0.1:8123/v1"; model = "m"
def post(p, n, path="/completions", **kw):
    body = {"messages": [{"role": "user", "content": p}]} if path == "/chat/completions" else {"prompt": p}
    b = json.dumps({"model": model, "max_tokens": n, "temperature": 0.7, "seed": 3, **body, **kw}).encode()
    r = urllib.request.Request(base + path, b, {"Content-Type": "application/json"})
    t = time.time(); o = json.load(urllib.request.urlopen(r, timeout=1800)); return time.time() - t, o
Ps = ["Explain how a skip list works, with code in Python.",
      "Write a detailed account of the Peloponnesian War.",
      "Describe how to design a Kalman filter for a drone."]
r = random.Random(5); w = "the of and to in a is that for it as was with be by on not he his but at are this have from or had".split()
Ls = [" ".join(r.choice(w) + str(r.randrange(99999)) for _ in range(400)) for _ in range(2)]
post(Ps[0], 8)
for rep in range(2):
    tt = gen = 0
    for p in Ps:
        t, o = post(p, 200, "/chat/completions", ignore_eos=True); tt += t; gen += o["usage"]["completion_tokens"]
    print(f"decode: {gen/tt:.1f} t/s ({gen} tok)", flush=True)
for p in Ls:
    t, o = post(p, 1); n = o["usage"]["prompt_tokens"]
    print(f"prefill: {n} tok in {t:.2f}s = {n/t:.0f} t/s", flush=True)
if len(sys.argv) > 1:
    # placement changes when weights arrive, never which, so these must match resident exactly.
    # Chat goes through the server's own template; the long prompt's logprobs cover prefill,
    # which short prompts never reach on the staged path. Last, because prompt logprobs over a
    # large vocab allocate past vLLM's memory profile and can leave the server unable to serve.
    chat = [post(p, 160, "/chat/completions", temperature=0)[1]["choices"][0]["message"]["content"] for p in Ps]
    lp = post(Ls[0], 1, temperature=0, prompt_logprobs=1)[1]["choices"][0]["prompt_logprobs"]
    json.dump({"chat": chat, "prompt_logprobs": lp}, open(sys.argv[1], "w"))
