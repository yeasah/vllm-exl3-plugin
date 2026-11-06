"""Record every (layer, token) -> topk_ids in a served-style workload.

usage: routing_trace.py MODEL REVISION OUT
"""
import os, sys, json, time, random
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import numpy as np, torch
model, rev, out = sys.argv[1:4]
from vllm_exl3_plugin.quantization import fused_moe
LOG = []; LAYERS = {}; TAG = ["none"]
orig = fused_moe.EXL3MoEMethod.apply
def apply(self, layer, x, topk_weights, topk_ids, *a, **k):
    i = LAYERS.setdefault(id(layer), len(LAYERS))
    LOG.append((TAG[0], i, topk_ids.detach().cpu().numpy().astype(np.int16).copy(),
                topk_weights.detach().float().cpu().numpy().astype(np.float32).copy()))
    return orig(self, layer, x, topk_weights, topk_ids, *a, **k)
fused_moe.EXL3MoEMethod.apply = apply
from vllm import LLM, SamplingParams
kw = dict(max_model_len=6000, max_num_seqs=1, enforce_eager=True, gpu_memory_utilization=0.9,
          max_num_batched_tokens=2048, language_model_only=True, enable_prefix_caching=False)
llm = LLM(model, revision=rev, **kw)
tok = llm.get_tokenizer()
import exllamav3
D = os.path.dirname(exllamav3.__file__) + "/conversion/standard_cal_data/"
def chunks(name, n, size):
    t = open(D + name, encoding="utf8").read(); r = random.Random(7)
    return [t[p:p + size] for p in (r.randrange(0, max(1, len(t) - size)) for _ in range(n))]
Q = {
 "chat": ["Explain how a bloom filter works and when you would not use one.",
  "Write a short story about a lighthouse keeper who finds a door in the sea.",
  "What caused the fall of the Western Roman Empire? Give a balanced answer.",
  "Plan a four-day trip to Kyoto for someone who hates crowds.",
  "Why is the sky blue but sunsets red? Explain with the physics.",
  "Draft a polite email declining a job offer after accepting another.",
  "Compare the French and Haitian revolutions in terms of their aims.",
  "Give me a recipe for lentil soup and explain why each step matters."],
 "code": ["Write a Python LRU cache without functools, with tests.",
  "Implement a lock-free single-producer single-consumer ring buffer in C++.",
  "Write a SQL query to find the second highest salary per department, and explain it.",
  "In Rust, write a tokenizer for arithmetic expressions and a Pratt parser for it.",
  "Refactor a JavaScript callback chain into async/await and explain the error handling.",
  "Write a bash script that rotates logs and keeps seven compressed copies.",
  "Implement Dijkstra's algorithm in Go with a heap.",
  "Write a Python decorator that retries with exponential backoff and jitter."],
 "math": ["Prove that the square root of 2 is irrational, then that there are infinitely many primes.",
  "A train leaves at 3pm at 60 mph; another at 4pm at 80 mph. When does the second catch up? Show steps.",
  "Compute the eigenvalues of [[2,1],[1,2]] and explain what they mean.",
  "Solve x^2 - 5x + 6 = 0 by three methods.",
  "Derive the formula for the sum of a geometric series."]}
def cut(s, n=4800): return tok.decode(tok.encode(s, add_special_tokens=False)[:n])
def chat(s): return tok.apply_chat_template([{"role": "user", "content": s}], tokenize=False, add_generation_prompt=True)
jobs = [(k, chat(q), 300) for k, qs in Q.items() for q in qs]
Q2 = {"lang": ["Explique en français comment fonctionne un moteur à combustion interne, en détail.",
  "Schreibe auf Deutsch einen ausführlichen Text über die Geschichte der Hanse.",
  "请用中文详细解释光合作用的过程以及它为什么重要。",
  "日本語で、江戸時代の社会構造について詳しく説明してください。",
  "Explica en español las causas de la Guerra Civil Española con detalle.",
  "Подробно объясни по-русски, как работает ядерный реактор."],
 "chem": ["Explain the mechanism of an SN2 reaction and what controls its rate.",
  "Describe how CRISPR-Cas9 edits a genome, step by step, including off-target effects.",
  "What are the tradeoffs among statute of limitations rules in contract law versus tort law?",
  "Explain how a differential pair amplifier works, with the small-signal analysis."]}
jobs += [(k, chat(q), 300) for k, qs in Q2.items() for q in qs]
jobs += [("long_tech", chat("Explain the main points of this passage:\n\n" + cut(c)), 160) for c in chunks("technical.utf8", 2, 12000)]
jobs += [("long_ml", chat("Translate the gist of this into English:\n\n" + cut(c)), 160) for c in chunks("multilingual.utf8", 2, 12000)]
jobs += [("long_wiki", chat("Summarize the following and list its key claims:\n\n" + cut(c)), 160) for c in chunks("wiki.utf8", 3, 14000)]
jobs += [("long_code", chat("Review this code and describe what it does:\n\n" + cut(c)), 160) for c in chunks("code.utf8", 3, 14000)]
random.Random(1).shuffle(jobs)
sp = lambda n: SamplingParams(temperature=0.7, top_p=0.95, max_tokens=n, seed=11)
res = []; t0 = time.time()
for n, (kind, p, m) in enumerate(jobs):
    TAG[0] = f"{n}:{kind}"
    t = time.time(); o = llm.generate([p], sp(m), use_tqdm=False)[0]
    res.append((n, kind, len(o.prompt_token_ids), len(o.outputs[0].token_ids), time.time() - t, o.outputs[0].finish_reason))
    print(res[-1], flush=True)
tags = [r[0] for r in LOG]; ids = [r[1] for r in LOG]
np.savez_compressed(out, tags=np.array(tags), layers=np.array(ids),
    topk=np.array([r[2] for r in LOG], dtype=object), w=np.array([r[3] for r in LOG], dtype=object))
json.dump(res, open(out + ".json", "w")); print("done", time.time() - t0, len(LOG))
