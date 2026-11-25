"""How far serving's transients reach past what the extensible KV cache's sizing step measured.

usage: margin_scan.py MODEL [--revision R] [--engine-args JSON] [--phases a,b,...] [--out FILE]

The extensible KV cache keeps free the larger of the sizing step's allocated transient and torch's reservation
growth, plus a margin (`kv_cache_keep_bytes` in the fork) -- upstream measured up to 45% beyond the allocated peak on GB200
under adversarial workloads it did not name. This runs the engine in-process at a lowered gpu_memory_utilization
(0.75 by default), so torch's cap leaves room and demand shows instead of being clipped, then drives hostile
phases and reports, per phase, torch's reserved and allocated peaks and non-torch growth over the post-startup
baseline, against the transient the sizing step kept. The margin's job is the excess over that transient.
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from bench.margin import hostile_phases  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--revision", default=None)
ap.add_argument("--engine-args", default="{}")
ap.add_argument("--phases", default="long_prefill,concurrent_sampling,prompt_logprobs,n4,json,prefix_hits,images")
ap.add_argument("--max-model-len", type=int, default=65536)
ap.add_argument("--util", type=float, default=0.75)
ap.add_argument("--out", default=None)
a = ap.parse_args()


def main():
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import logging

    import torch
    from vllm import LLM

    captured = []
    from vllm.v1.worker import extensible_kv_cache as ext

    orig_info = ext.logger.info

    def info(msg, *args, **kwargs):   # vLLM's loggers do not propagate to a handler added here
        if isinstance(msg, str) and msg.startswith("Memory after warmup"):
            captured.append(msg % args)
        return orig_info(msg, *args, **kwargs)

    ext.logger.info = info
    kw = dict(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=a.util,
              enable_extensible_kv_cache=True, max_num_seqs=8)
    if a.revision:
        kw["revision"] = a.revision
    kw.update(json.loads(a.engine_args))
    llm = LLM(**kw)
    sizing = {}
    if captured:
        m = re.search(r"([\d.]+) GiB kept for transient activations \(([\d.]+) GiB allocated at peak, ([\d.]+) GiB of torch "
                      r"reservation growth\)", captured[-1])
        if m:
            sizing = dict(kept=float(m.group(1)), alloc_peak=float(m.group(2)), growth=float(m.group(3)))
    tok = llm.get_tokenizer()
    docs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs")
    text = "\n\n".join(open(os.path.join(docs, n)).read() for n in sorted(os.listdir(docs)) if n.endswith(".md"))
    ids = tok(text)["input_ids"]
    ids = (ids * (a.max_model_len // len(ids) + 2))
    mml = llm.llm_engine.vllm_config.model_config.max_model_len
    gib = 1 << 30

    def baseline():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return dict(reserved=torch.cuda.memory_reserved(), allocated=torch.cuda.memory_allocated(),
                    nontorch=(total - free) - torch.cuda.memory_reserved())

    def run(name, fn):
        b = baseline()
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        err = None
        try:
            fn()
        except Exception as e:   # an OOM is a result, not a crash of the scan
            err = f"{type(e).__name__}: {str(e)[:160]}"
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        r = dict(phase=name, seconds=round(time.time() - t, 1),
                 reserved_growth=(torch.cuda.max_memory_reserved() - b["reserved"]) / gib,
                 allocated_growth=(torch.cuda.max_memory_allocated() - b["allocated"]) / gib,
                 nontorch_growth=((total - free) - torch.cuda.memory_reserved() - b["nontorch"]) / gib, error=err)
        print("PHASE", json.dumps(r), flush=True)
        return r

    long_n = min(mml - 300, a.max_model_len - 300)
    phases = hostile_phases(llm, ids, long_n)
    start = baseline()
    torch.cuda.reset_peak_memory_stats()
    results = []
    for p in a.phases.split(","):
        if p in phases:
            peak_before = torch.cuda.max_memory_reserved()
            results.append(run(p, phases[p]))
            torch.cuda.reset_peak_memory_stats()
            results[-1]["_peak_reserved"] = max(peak_before, 0)
    # cumulative: torch's reservation never shrinks between phases unless emptied, so the scan's worst is the
    # highest reservation reached over the post-startup baseline, plus non-torch growth
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    reserved_now = torch.cuda.memory_reserved()
    worst = (max([reserved_now] + [r.pop("_peak_reserved") + 0 for r in results]) - start["reserved"]) / gib
    worst = max(worst, max(r["reserved_growth"] for r in results)) + max(
        ((total - free) - reserved_now - start["nontorch"]) / gib, 0)
    summary = dict(model=a.model, revision=a.revision, engine_args=json.loads(a.engine_args), sizing=sizing,
                   phases=results, worst_growth=worst,
                   excess_over_transient=(worst - sizing["kept"]) if sizing else None,
                   excess_fraction=((worst - sizing["kept"]) / sizing["kept"]) if sizing.get("kept") else None)
    print("SUMMARY", json.dumps(summary), flush=True)
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
