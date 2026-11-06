"""Client: replay the chat requests a vLLM request log recorded (--enable-log-requests at DEBUG,
"Request <id> JSON body: {...}" lines), in order, each after the previous one finishes, against a
server on :8123. Streams each response through to the end, as the original client did.
usage: replay_requests.py LOG [FIRST] [COUNT]"""
import json, re, sys, time, urllib.request

log, first = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
count = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
bodies = [json.loads(m[1]) for line in open(log, errors="replace")
          if (m := re.search(r"Request \S+ JSON body: (.*)$", line))][first:first + count]
print(f"replaying {len(bodies)} requests from {log}", flush=True)
for i, b in enumerate(bodies, first):
    req = urllib.request.Request("http://127.0.0.1:8123/v1/chat/completions", json.dumps(b).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time(); chunks = 0
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            chunks += line.startswith(b"data: {")
    print(f"request {i}: {len(b['messages'])} messages, {chunks} chunks in {time.time() - t:.1f} s", flush=True)
print("REPLAY DONE", flush=True)
