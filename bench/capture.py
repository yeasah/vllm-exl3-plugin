"""Capture one entry. Runs as a subprocess; `bench/run.py` is the entry point.

Standalone so each engine gets a clean process. It writes the measurement to
`--out` and leaves the weight-bytes number to its parent, which reads it off
vLLM's own log line rather than reaching into engine internals -- the model
lives in the EngineCore subprocess under v1, where there is nothing to reach
into from here.
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("entry")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--new-tokens", type=int, default=24)
    args = ap.parse_args()

    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bench import core, suite

    entry = suite.by_name(args.entry)

    # Set before vLLM starts its engine core.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # The autotune/compile caches are keyed on things a version bump changes; a
    # gate that reads a stale cache is measuring the cache, not the build.
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    for key, value in entry.env.items():
        os.environ[key] = str(value)

    from vllm import LLM

    llm = LLM(
        model=entry.model,
        revision=entry.revision,
        model_impl=entry.model_impl,
        enforce_eager=entry.enforce_eager,
        tensor_parallel_size=entry.tensor_parallel_size,
        max_model_len=entry.max_model_len,
        gpu_memory_utilization=entry.gpu_memory_utilization,
    )
    tok = llm.get_tokenizer()
    prompts = core.capture_prompts(llm, tok, k=args.k, new_tokens=args.new_tokens)

    with open(args.out, "w") as f:
        json.dump(
            {
                "entry": entry.name,
                "label": entry.label,
                "model": entry.model,
                "revision": entry.revision,
                "model_impl": entry.model_impl,
                "enforce_eager": entry.enforce_eager,
                "tensor_parallel_size": entry.tensor_parallel_size,
                "k": args.k,
                # Correctness baselines are meant to be portable -- they are a
                # fact about this codebase, not about a machine -- so unlike perf
                # they are not filed per-platform. But "meant to be" is not
                # "are": fp16 accumulation depends on tile shapes, which depend
                # on the GPU, so a check run on different hardware can move
                # logprobs and occasionally an argmax. Recording this lets
                # `check` say so instead of reporting a phantom regression.
                "platform": os.environ.get("BENCH_PLATFORM"),
                "environment": core.environment(),
                "prompts": prompts,
            },
            f,
            indent=1,
        )

    n = sum(len([s for s in p["steps"] if s]) for p in prompts)
    print(f"BENCH_CAPTURE_OK positions={n}")

    # Without this the still-running engine core outlives the caller's reference
    # and holds the GPU, which matters when the next entry is about to load.
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception as exc:  # pragma: no cover - teardown is best-effort
        print(f"BENCH_CAPTURE_WARN shutdown failed: {exc}")


if __name__ == "__main__":
    main()
