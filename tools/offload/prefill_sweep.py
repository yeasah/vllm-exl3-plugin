"""Client: time-to-first-token across prompt lengths, against a server on :8123.
usage: prefill_sweep.py [TOKENS ...]   default 128 512 1024 2048 3072

Run against resident and offloaded servers with the same --max-num-batched-tokens: staging
adds a cost per prefill chunk, so the difference should step with chunks, not grow with tokens."""
import json, sys, time, random, urllib.request
base = "http://127.0.0.1:8123/v1"
def post(p):
    b = json.dumps({"model": "m", "prompt": p, "max_tokens": 1, "temperature": 0}).encode()
    r = urllib.request.Request(base + "/completions", b, {"Content-Type": "application/json"})
    t = time.time(); o = json.load(urllib.request.urlopen(r, timeout=1800)); return time.time() - t, o
w = "the of and to in a is that for it as was with be by on not he his but at are this have from or had".split()
post("warm up")
for target in [int(a) for a in sys.argv[1:]] or [128, 512, 1024, 2048, 3072]:
    r = random.Random(target)
    # ~5.9 tokens per word with this vocabulary; trimmed by the reported count, not exact
    p = " ".join(r.choice(w) + str(r.randrange(99999)) for _ in range(max(1, target * 10 // 59)))
    ts = []
    for _ in range(3):
        t, o = post(p); ts.append(t)
    n = o["usage"]["prompt_tokens"]
    print(f"ttft: {n} tok {1e3 * min(ts):.0f} ms", flush=True)
