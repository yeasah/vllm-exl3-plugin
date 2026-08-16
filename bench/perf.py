"""Throughput measurement for one entry. Runs as a subprocess; `run.py` drives it.

The correctness gate reads *what* is served and is blind to *how fast*, so a bump
that silently cost 30% would pass every entry in it. This closes that.

The workload deliberately reproduces the shape in docs/kernels.md -- decode is 8
concurrent sequences x 128 tokens, prefill is 4 x ~2.2k tokens, fp16, prefix
caching off -- so the recorded numbers there and the numbers here are the same
measurement, and the table in that note stays a live reference rather than a
historical one. On the dev card it reproduces to ~0.1% on decode.

Two things differ deliberately from the correctness entries:

- **CUDA graphs are on.** Correctness entries mostly force eager because that is
  the cleaner comparison; nobody serves that way, and perf measured eager would
  gate a configuration nobody runs.
- **`ignore_eos`**, so decode produces exactly the token count asked for. Without
  it throughput would vary with where the model chose to stop, which is a
  property of the weights rather than of the kernels.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

#: docs/kernels.md's shape. Changing either invalidates every perf baseline and
#: breaks comparability with that note, so treat them as fixed.
DECODE_SEQS, DECODE_TOKENS = 8, 128
PREFILL_SEQS, PREFILL_TOKENS = 4, 2200


def _build_prompts(tok):
    # Distinct texts so nothing can dedupe or hit a cache, and long enough that
    # the prefill shape is reached after truncation.
    filler = (
        "Quantization trades numerical precision for memory bandwidth. "
        "The trellis code spends bits unevenly across a block. "
    )
    prefill = [
        {"prompt_token_ids": tok.encode(f"Document {i}. " + filler * 120)[:PREFILL_TOKENS]}
        for i in range(PREFILL_SEQS)
    ]
    decode = [
        {"prompt_token_ids": tok.encode(f"Sequence {i}. Continue this text:")}
        for i in range(DECODE_SEQS)
    ]
    return prefill, decode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("entry")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bench import core, suite

    entry = suite.perf_by_name(args.entry)

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    for key, value in entry.env.items():
        os.environ[key] = str(value)

    from vllm import LLM, SamplingParams

    # The workload defines its own context requirement, so derive it rather than
    # trusting each entry to remember: an entry left at the matrix default of
    # 2048 rejects the 2200-token prefill prompts, and does so only *after*
    # loading the model, which on a 35B is four minutes to reach a typo.
    max_model_len = max(entry.max_model_len, PREFILL_TOKENS + 64)
    if max_model_len != entry.max_model_len:
        print(f"BENCH_PERF_NOTE raising max_model_len "
              f"{entry.max_model_len} -> {max_model_len} for the prefill shape")

    llm = LLM(
        model=entry.model,
        revision=entry.revision,
        model_impl=entry.model_impl,
        enforce_eager=entry.enforce_eager,
        tensor_parallel_size=entry.tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=entry.gpu_memory_utilization,
        dtype="float16",
        enable_prefix_caching=False,
        max_num_seqs=max(16, DECODE_SEQS),
    )
    tok = llm.get_tokenizer()
    prefill_prompts, decode_prompts = _build_prompts(tok)

    decode_sp = SamplingParams(
        temperature=0.0, max_tokens=DECODE_TOKENS, ignore_eos=True
    )
    prefill_sp = SamplingParams(temperature=0.0, max_tokens=1)

    def once() -> tuple[float, float]:
        t0 = time.perf_counter()
        outs = llm.generate(decode_prompts, decode_sp, use_tqdm=False)
        elapsed = time.perf_counter() - t0
        decode = sum(len(o.outputs[0].token_ids) for o in outs) / elapsed

        t0 = time.perf_counter()
        outs = llm.generate(prefill_prompts, prefill_sp, use_tqdm=False)
        elapsed = time.perf_counter() - t0
        prefill = sum(len(o.prompt_token_ids) for o in outs) / elapsed
        return decode, prefill

    once()  # discard: autotune, allocator growth and clocks all settle here
    rows = [once() for _ in range(args.reps)]

    result = {
        "entry": entry.name,
        "label": entry.label,
        "reps": args.reps,
        # Not the identity of the platform -- the operator's tag is that, and it
        # is the directory this lands in. This is the cross-check on the tag.
        "platform": os.environ.get("BENCH_PLATFORM"),
        "environment": core.environment(),
    }
    for name, vals in (
        ("decode", [r[0] for r in rows]),
        ("prefill", [r[1] for r in rows]),
    ):
        # Median rather than mean: a single scheduling hiccup should not move the
        # baseline, and the distribution is tight enough that it barely differs.
        result[name] = round(statistics.median(vals), 1)
        result[f"{name}_spread_pct"] = round(
            (max(vals) - min(vals)) / statistics.median(vals) * 100, 2
        )
        result[f"{name}_samples"] = [round(v, 1) for v in vals]

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)

    print(
        f"BENCH_PERF_OK decode={result['decode']} tok/s "
        f"(+-{result['decode_spread_pct']}%) "
        f"prefill={result['prefill']} tok/s (+-{result['prefill_spread_pct']}%)"
    )

    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception as exc:  # pragma: no cover - teardown is best-effort
        print(f"BENCH_PERF_WARN shutdown failed: {exc}")


if __name__ == "__main__":
    main()
