"""Hostile serving against the extensible KV cache's keep, for one entry. Runs as a subprocess; `run.py` drives it.

The extensible KV cache sizes itself after warmup and keeps free a transient plus a margin
(`kv_cache_keep_bytes` in the fork). That margin was measured, not derived (docs/memory-accounting.md), so it is a
safety property an upstream change can erode without any entry noticing: a sampler, attention backend or allocator
change that allocates more past the sizing step's peak passes every correctness and throughput check right up to
the request that OOMs a server.

So this serves the way the margin binds -- `gpu_memory_utilization` unset, the device deciding -- with the cap the
engine set on torch's allocator in force, and drives the phases that reach furthest past that peak. It reads the
cap exactly (by wrapping `set_torch_memory_limit`) rather than recomputing the formula, so it follows whatever the
formula becomes. Per phase it records torch's allocated and reserved peaks, the allocator's retries (the cap
starting to bite: it emptied its cache to fit) and OOMs, and non-torch growth against the share kept for it.

Allocation is what the gate judges. Reservation past allocation is fragmentation, which an allocator at its cap
sheds by retrying, so it costs throughput before it costs a request; that is why retries are reported and the
allocated peak is the margin's real consumer.
"""

from __future__ import annotations

import argparse
import json
import os
import time

GIB = 1 << 30


def hostile_phases(llm, ids: list[int], long_n: int) -> dict:
    """The workloads that reach furthest past the sizing step's peak, measured in docs/memory-accounting.md.
    Shared with tools/memory/margin_scan.py so the scan and the gate drive the same thing."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import StructuredOutputsParams

    P = lambda n, at=0: TokensPrompt(prompt_token_ids=ids[at:at + n])  # noqa: E731
    phases = {
        "long_prefill": lambda: llm.generate([P(long_n)], SamplingParams(max_tokens=16), use_tqdm=False),
        "concurrent_sampling": lambda: llm.generate(
            [P(4000, 977 * i) for i in range(8)],
            SamplingParams(max_tokens=256, temperature=0.9, top_k=50, top_p=0.9, min_p=0.05, presence_penalty=0.5,
                           frequency_penalty=0.5, repetition_penalty=1.1, logprobs=20, ignore_eos=True), use_tqdm=False),
        "prompt_logprobs": lambda: llm.generate([P(min(8000, long_n))], SamplingParams(max_tokens=1, prompt_logprobs=20),
                                                use_tqdm=False),
        "n4": lambda: llm.generate([P(2000, 31)], SamplingParams(n=4, max_tokens=256, temperature=1.0, logprobs=5,
                                                                 ignore_eos=True), use_tqdm=False),
        "json": lambda: llm.generate([P(1000, 77 * i) for i in range(4)], SamplingParams(
            max_tokens=200, structured_outputs=StructuredOutputsParams(json={"type": "object", "properties": {
                "name": {"type": "string"}, "items": {"type": "array", "items": {"type": "integer"}}},
                "required": ["name", "items"]})), use_tqdm=False),
        "prefix_hits": lambda: llm.generate(
            [TokensPrompt(prompt_token_ids=ids[:min(16000, long_n)] + ids[50000 + 100 * i:50000 + 100 * i + 200])
             for i in range(8)], SamplingParams(max_tokens=64), use_tqdm=False),
    }

    def images():
        from PIL import Image
        imgs = [Image.new("RGB", (1024, 1024), (i * 30 % 255, 90, 200)) for i in range(4)]
        msgs = [[{"role": "user", "content": [{"type": "image_pil", "image_pil": im},
                                              {"type": "text", "text": "Describe."}]}] for im in imgs]
        llm.chat(msgs, SamplingParams(max_tokens=32), use_tqdm=False)

    phases["images"] = images
    return phases


def filler_ids(tok, max_model_len: int) -> list[int]:
    """Real prose (the project's docs), repeated past the longest prompt a phase asks for."""
    docs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
    text = "\n\n".join(open(os.path.join(docs, n)).read() for n in sorted(os.listdir(docs)) if n.endswith(".md"))
    ids = tok(text)["input_ids"]
    return ids * (max(max_model_len, 51000) // len(ids) + 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("entry")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bench import core, suite

    entry = suite.margin_by_name(args.entry)
    # In-process, so this process's torch is the worker's and its allocator statistics are the ones that matter.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    for key, value in entry.child_env().items():
        os.environ[key] = value

    import torch
    from vllm import LLM
    from vllm.v1.worker import extensible_kv_cache as ext

    caps: list[int] = []
    orig_set = ext.ExtensibleKVCache.set_torch_memory_limit

    def set_limit(self, num_bytes):
        if num_bytes is not None:
            caps.append(int(num_bytes))
        return orig_set(self, num_bytes)

    ext.ExtensibleKVCache.set_torch_memory_limit = set_limit
    # For watching the gate fail: scale the keep the engine computes, as an upstream change eroding it would.
    scale = float(os.environ.get("BENCH_MARGIN_KEEP_SCALE", "1"))
    if scale != 1:
        orig_keep = ext.kv_cache_keep_bytes
        ext.kv_cache_keep_bytes = lambda *a, **k: int(orig_keep(*a, **k) * scale)
        print(f"BENCH_MARGIN_NOTE keep scaled by {scale}", flush=True)

    llm = LLM(
        model=entry.model,
        revision=entry.revision,
        model_impl=entry.model_impl,
        enforce_eager=entry.enforce_eager,
        max_model_len=entry.max_model_len,
        # Unset on purpose: an explicit budget that leaves room keeps only the transient, and the margin never binds.
        enable_extensible_kv_cache=True,
        language_model_only=entry.language_model_only,
        **({"kv_cache_dtype": entry.kv_cache_dtype} if entry.kv_cache_dtype else {}),
        **entry.engine_args,
    )
    if not caps:
        raise SystemExit("BENCH_MARGIN_FAIL the extensible KV cache set no cap on torch's allocator; "
                         "the margin is not in force, so there is nothing to measure")
    cap = caps[-1]
    mml = llm.llm_engine.vllm_config.model_config.max_model_len
    ids = filler_ids(llm.get_tokenizer(), mml)

    torch.cuda.synchronize()
    stats = torch.cuda.memory_stats()
    free0, total = torch.cuda.mem_get_info()
    start = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                 nontorch=(total - free0) - torch.cuda.memory_reserved())
    nontorch_share = ext.kv_cache_non_torch_reserve_bytes(torch.device("cuda", torch.cuda.current_device()))

    phases = hostile_phases(llm, ids, min(mml, entry.max_model_len) - 300)
    rows = []
    for name in entry.margin_phases:
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_stats()
        t0, err = time.time(), None
        try:
            phases[name]()
        except Exception as e:  # an OOM is the result being looked for, not a crash of the run
            err = f"{type(e).__name__}: {str(e)[:200]}"
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
        free, _ = torch.cuda.mem_get_info()
        rows.append(dict(
            phase=name, seconds=round(time.time() - t0, 1), error=err,
            allocated_peak=torch.cuda.max_memory_allocated(),
            allocated_after=torch.cuda.memory_allocated(),
            reserved_peak=torch.cuda.max_memory_reserved(),
            alloc_retries=after.get("num_alloc_retries", 0) - before.get("num_alloc_retries", 0),
            ooms=after.get("num_ooms", 0) - before.get("num_ooms", 0),
            nontorch_growth=(total - free) - torch.cuda.memory_reserved() - start["nontorch"],
        ))
        r = rows[-1]
        print(f"BENCH_MARGIN_PHASE {name}: allocated {(r['allocated_peak'] - start['allocated']) / GIB:.2f} of "
              f"{(cap - start['allocated']) / GIB:.2f} GiB room, retries {r['alloc_retries']}, "
              f"ooms {r['ooms']}{', ' + err if err else ''}", flush=True)

    result = dict(entry=entry.name, label=entry.label, cap=cap, start=start, total=total,
                  nontorch_share=nontorch_share, phases=rows, environment=core.environment())
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print("BENCH_MARGIN_OK", flush=True)
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception as exc:  # pragma: no cover - teardown is best-effort
        print(f"BENCH_MARGIN_WARN shutdown failed: {exc}")


if __name__ == "__main__":
    main()
