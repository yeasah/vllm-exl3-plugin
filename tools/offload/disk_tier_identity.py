"""Disk tier identity: the same prompts' logprobs with some experts read from the checkpoint and with all of them in RAM.

usage: disk_tier_identity.py MODEL [--share 0.4] [--slots 0.2] [--chunk 2048] [--out DIR]

Two arms, one engine each in its own process: `ram` (every offloaded expert in host rows) and `disk`
(EXL3_DISK_TIER=SHARE). Both fix the slots (EXL3_EXPERT_SLOTS) and run eager with the grouped MoE path at every batch
size (EXL3_MOE_GROUPED_MIN=1), which a disk-tier layer takes anyway; so the kernels, the slot contents and the
summation order match, and only where an expert's bytes come from differs. Anything short of bit-identical prompt
logprobs and greedy tokens is a staging bug. The disk arm runs with EXL3_CACHE_VERIFY=1. Prompts are real text
of several lengths, the longest several chunks, so short tails and multi-chunk prefill are both covered.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--share", default="0.4", help="EXL3_DISK_TIER for the disk arm")
ap.add_argument("--slots", default="0.2", help="EXL3_EXPERT_SLOTS for both arms")
ap.add_argument("--chunk", type=int, default=2048, help="max_num_batched_tokens")
ap.add_argument("--offload-gb", type=float, default=64)
ap.add_argument("--decode", type=int, default=16, help="greedy tokens per prompt")
ap.add_argument("--out", default=None)
ap.add_argument("--disk-graphs", action="store_true",
                help="the disk arm under breakable piecewise graphs, the MoE op an eager break (decode path A)")
ap.add_argument("--graphs-both", action="store_true", help="both arms under --disk-graphs' capture, the fair comparison")
ap.add_argument("--arm", default=None, help=argparse.SUPPRESS)
a = ap.parse_args()


def prompts(tok):
    here = os.path.dirname(os.path.abspath(__file__))
    text = []
    for name in sorted(os.listdir(os.path.join(here, "..", "..", "docs"))):
        if name.endswith(".md"):
            with open(os.path.join(here, "..", "..", "docs", name)) as f:
                text.append(f.read())
    ids = tok("\n\n".join(text))["input_ids"]
    out, at = [], 0
    for n in (7, 45, 300, 2049, 2 * a.chunk + 333):   # a tail under the old grouped threshold; a chunk + 1; multi-chunk
        out.append(ids[at:at + n])
        at += n
    return out


def run_arm(arm):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # the plugin's stats live in this process then
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from vllm_exl3_plugin import cpu_offload

    graphs = os.environ.get("EXL3_MOE_EAGER_BREAK") == "1" or os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH") == "1"
    llm = LLM(a.model, enforce_eager=not graphs,
              **({"compilation_config": {"cudagraph_mode": "PIECEWISE"}} if graphs else {}), max_num_batched_tokens=a.chunk, max_model_len=8192,
              cpu_offload_gb=a.offload_gb, gpu_memory_utilization=0.85, seed=0, max_num_seqs=8,
              limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})   # text only: no encoder budget per chunk
    ps = prompts(llm.get_tokenizer())
    sp = SamplingParams(temperature=0, max_tokens=a.decode, prompt_logprobs=1, logprobs=1)
    t = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ps], sp, use_tqdm=False)
    took = time.time() - t
    res = []
    for p, o in zip(ps, outs):
        plp = [None if d is None else d[p[i]].logprob for i, d in enumerate(o.prompt_logprobs)]
        c = o.outputs[0]
        res.append({"n": len(p), "prompt_logprobs": plp, "tokens": list(c.token_ids),
                    "logprobs": [d[t].logprob for d, t in zip(c.logprobs, c.token_ids)]})
    stats = {k: cpu_offload._STATS.get(k) for k in ("disk_experts", "disk_reads", "disk_admitted", "expert_bytes")}
    store = cpu_offload._DISK.get("store")
    if store is not None:
        stats.update(store_reads=store.reads, store_gib=store.bytes / 2**30)
    with open(os.path.join(a.out, f"{arm}.json"), "w") as f:
        json.dump({"results": res, "seconds": took, "stats": stats}, f)


if a.arm:
    run_arm(a.arm)
    sys.exit(0)

a.out = a.out or os.path.join(os.environ.get("TMPDIR", "/tmp"), "disk_tier_identity")
os.makedirs(a.out, exist_ok=True)
base = dict(os.environ, EXL3_EXPERT_SLOTS=a.slots, EXL3_MOE_GROUPED_MIN="1", EXL3_DISK_TIER="off")
arms = {"ram": base, "disk": dict(base, EXL3_DISK_TIER=a.share, EXL3_CACHE_VERIFY="1")}
if a.disk_graphs or a.graphs_both:
    # graph replay pads decode batches to capture sizes, so against an eager arm its numbers differ by that alone
    for arm in (("ram", "disk") if a.graphs_both else ("disk",)):
        arms[arm].update(VLLM_USE_BREAKABLE_CUDAGRAPH="1", EXL3_MOE_EAGER_BREAK="1")
for arm, env in arms.items():
    cmd = [sys.executable, os.path.abspath(__file__), a.model, "--arm", arm, "--out", a.out, "--chunk", str(a.chunk),
           "--offload-gb", str(a.offload_gb), "--decode", str(a.decode)]
    with open(os.path.join(a.out, f"{arm}.log"), "w") as log:
        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc:
        sys.exit(f"{arm} arm failed ({rc}); see {a.out}/{arm}.log")

r = {arm: json.load(open(os.path.join(a.out, f"{arm}.json"))) for arm in arms}
bad = 0
for i, (x, y) in enumerate(zip(r["ram"]["results"], r["disk"]["results"])):
    plp = sum(1 for u, v in zip(x["prompt_logprobs"], y["prompt_logprobs"]) if u != v)
    tok = x["tokens"] != y["tokens"]
    lp = sum(1 for u, v in zip(x["logprobs"], y["logprobs"]) if u != v)
    bad += plp + tok + lp
    print(f"prompt {i} ({x['n']:5d} tokens): prompt logprobs differ at {plp}, greedy tokens "
          f"{'DIFFER' if tok else 'match'}, decode logprobs differ at {lp}")
for arm in arms:
    print(f"{arm:5s}: {r[arm]['seconds']:.2f} s generate; {r[arm]['stats']}")
print("IDENTICAL" if not bad else f"NOT IDENTICAL ({bad} differences)")
sys.exit(1 if bad else 0)
