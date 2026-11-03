"""Client: decode tok/s on natural long answers, one request at a time, against a server on :8123.
usage: decode_bench.py [MAX_TOKENS]   default 900

Unlike measure.py's 200-token ignore_eos runs, every request stops on its own and most run long, so
a cache sees whole requests: cold start at each new topic, then steady state."""
import json, sys, time, urllib.request
Ps = ["Write a long, detailed tutorial on implementing a B-tree in C, with code.",
      "Explain the causes and course of the Thirty Years' War in depth.",
      "Derive the Kalman filter equations step by step and discuss tuning.",
      "Write a short story about a lighthouse keeper who finds a message in a bottle.",
      "Compare TCP congestion control algorithms: Reno, CUBIC and BBR, in detail.",
      "Explain how photosynthesis works at the molecular level.",
      "Write a detailed guide to sourdough baking, from starter to loaf.",
      "Explain the proof of the fundamental theorem of algebra and its history."]
n = int(sys.argv[1]) if len(sys.argv) > 1 else 900
tt = gen = 0
for p in Ps:
    b = json.dumps({"model": "m", "messages": [{"role": "user", "content": p}], "max_tokens": n,
                    "temperature": 0.7, "seed": 3}).encode()
    r = urllib.request.Request("http://127.0.0.1:8123/v1/chat/completions", b, {"Content-Type": "application/json"})
    t = time.time(); o = json.load(urllib.request.urlopen(r, timeout=1800)); t = time.time() - t
    k = o["usage"]["completion_tokens"]; tt += t; gen += k
    print(f"{k:5d} tok {k / t:6.1f} t/s", flush=True)
print(f"decode: {gen / tt:.1f} t/s ({gen} tok)", flush=True)
