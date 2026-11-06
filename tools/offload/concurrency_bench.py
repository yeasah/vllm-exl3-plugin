"""Client: N concurrent streams of unrelated requests against a server on :8123, for a fixed time.
usage: concurrency_bench.py N [SECONDS] [MAX_TOKENS]   defaults 180 s, 600 tokens

Each stream starts at a staggered offset and sends prompts from its own rotation of a shared pool
back to back, each stopping on its own, so arrivals and their prefill chunks land in the middle of
other streams' decodes. Reports aggregate decode t/s and the median per-request rate."""
import json, statistics, sys, threading, time, urllib.request
POOL = ["Write a long, detailed tutorial on implementing a B-tree in C, with code.",
        "Explain the causes and course of the Thirty Years' War in depth.",
        "Derive the Kalman filter equations step by step and discuss tuning.",
        "Write a short story about a lighthouse keeper who finds a message in a bottle.",
        "Compare TCP congestion control algorithms: Reno, CUBIC and BBR, in detail.",
        "Explain how photosynthesis works at the molecular level.",
        "Write a detailed guide to sourdough baking, from starter to loaf.",
        "Explain the proof of the fundamental theorem of algebra and its history.",
        "Describe the architecture of a modern out-of-order CPU core.",
        "Write a persuasive essay for and against nuclear power.",
        "Explain how the immune system distinguishes self from non-self.",
        "Write a Rust implementation of a lock-free queue and explain it.",
        "Summarize the plot and themes of Moby-Dick in detail.",
        "Explain general relativity's prediction of gravitational waves.",
        "Plan a two-week trip through Japan with daily itineraries.",
        "Explain how SQL query planners choose join orders."]
n = int(sys.argv[1]); dur = float(sys.argv[2]) if len(sys.argv) > 2 else 180
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 600
end = time.time() + dur; lock = threading.Lock(); done = []  # (tokens, seconds) per finished request

def stream(i):
    time.sleep(i * 2.0)  # stagger arrivals
    k = i
    while time.time() < end:
        p = POOL[k % len(POOL)]; k += n
        b = json.dumps({"model": "m", "messages": [{"role": "user", "content": p}], "max_tokens": max_tokens,
                        "temperature": 0.7, "seed": k}).encode()
        r = urllib.request.Request("http://127.0.0.1:8123/v1/chat/completions", b, {"Content-Type": "application/json"})
        t = time.time(); o = json.load(urllib.request.urlopen(r, timeout=1800)); t = time.time() - t
        with lock: done.append((o["usage"]["completion_tokens"], t))

t0 = time.time()
ts = [threading.Thread(target=stream, args=(i,)) for i in range(n)]
for t in ts: t.start()
for t in ts: t.join()
wall = time.time() - t0
tok = sum(k for k, _ in done)
print(f"concurrency {n}: {len(done)} requests, {tok} tok in {wall:.0f} s = {tok / wall:.1f} t/s aggregate, "
      f"median per request {statistics.median(k / t for k, t in done):.1f} t/s", flush=True)
