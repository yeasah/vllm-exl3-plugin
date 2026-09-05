#!/usr/bin/env python3
"""Prove the residency guard fires, one deliberate fault at a time.

A guard nobody has watched fail is a comment. This runs a real engine and
injects, per arm, exactly one of the mistakes the allocator will be able to
make, then checks that the guard reports *that* check and that a clean run
reports nothing:

    clean       nothing injected                  -> no violations
    stray       a resident entry points at a block the request does not own
                (what a freed-and-reallocated block looks like from the
                worker's side, which is the failure with nothing to announce
                it)                               -> ownership
    tailswap    the partial tail is moved out of last place, so this step's
                own key is written into a block the kernel no longer treats as
                the tail                          -> write_target
    shortview   a block is dropped from the row without shortening seq_len, so
                the kernel reads one entry past the resident set
                                                  -> length
    overlap     one request's view is given a block belonging to another
                                                  -> exclusivity (and
                                                     ownership, which is the
                                                     same event seen twice)

The faults are injected in the same place a pager would impose residency --
after `prepare_attn`, on the gathered view -- so the guard is being tested
against the surface it will actually watch, not a mock of it.

    tools/kv_guard_selftest.py MODEL [--ctx N] [--reqs 2]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blocktable_permute import haystack  # noqa: E402
from kv_pager import ResidencyGuard  # noqa: E402

FAULTS = {
    "clean": set(),
    "stray": {"ownership"},
    "tailswap": {"write_target"},
    "shortview": {"length"},
    "overlap": {"exclusivity"},
}


class Harness:
    """Injects one fault per arm and runs the guard on the result."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.fault = "clean"
        self.guard = None
        self.runner = None

    def install(self):
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner

        if getattr(GPUModelRunner.prepare_attn, "_guard_hooked", False):
            return
        original = GPUModelRunner.prepare_attn

        def hooked(runner, input_batch, *args, **kwargs):
            self.runner = runner
            block_tables, slot_mappings = original(runner, input_batch,
                                                   *args, **kwargs)
            self.apply(runner, input_batch, block_tables[0],
                       slot_mappings[0] if slot_mappings is not None else None)
            return block_tables, slot_mappings

        hooked._guard_hooked = True
        GPUModelRunner.prepare_attn = hooked

    def apply(self, runner, batch, table, slot_mapping):
        block_size = runner.block_tables.kernel_block_sizes[0]
        seq_lens = batch.seq_lens.clone()
        intended = {}
        rows = []
        for b in range(batch.num_reqs):
            if int(batch.num_scheduled_tokens[b]) != 1:
                continue
            computed = int(batch.num_computed_tokens_np[b])
            n_full = computed // block_size
            if n_full < 3:
                continue
            rows.append((b, batch.req_ids[b], computed, n_full))
        if not rows:
            return
        # A correct view of the whole request: every full block, tail last,
        # seq_len covering exactly those. Faults are departures from this.
        for b, req_id, computed, n_full in rows:
            intended[req_id] = n_full + 1

        for b, req_id, computed, n_full in rows:
            row = table[b]
            if self.fault == "stray":
                # A block id near the top of the pool, distinct per request so
                # this arm trips ownership alone -- an injected fault that also
                # trips a second check tests two things and isolates neither.
                row[1] = int(runner.kv_caches[0].shape[0]) - 3 - b
            elif self.fault == "tailswap":
                row[[1, n_full]] = row[[n_full, 1]]
            elif self.fault == "shortview":
                # Drop one block, leave seq_len claiming it is still there.
                keep = [i for i in range(n_full) if i != 2] + [n_full]
                row[:len(keep)] = row[keep]
                intended[req_id] = n_full           # policy meant one fewer
            elif self.fault == "overlap" and b > 0:
                row[1] = int(table[0][1])

        self.guard.check_step(batch, table, seq_lens, slot_mapping,
                              block_size, intended)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--reqs", type=int, default=2)
    ap.add_argument("--util", type=float, default=0.60)
    args = ap.parse_args()

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, max_model_len=args.ctx + args.tokens + 64,
              gpu_memory_utilization=args.util, enforce_eager=True,
              enable_prefix_caching=False, max_num_seqs=args.reqs,
              trust_remote_code=True)
    scheduler = llm.llm_engine.engine_core.engine_core.scheduler

    harness = Harness(scheduler)
    harness.install()

    tok = llm.get_tokenizer()
    ids = haystack(tok, args.ctx)
    prompts = [{"prompt_token_ids": ids[: args.ctx - 7 - 37 * r]}
               for r in range(args.reqs)]
    params = SamplingParams(temperature=0.0, max_tokens=args.tokens,
                            ignore_eos=True)

    print()
    failures = 0
    for fault, expected in FAULTS.items():
        harness.fault = fault
        harness.guard = ResidencyGuard(scheduler)
        llm.generate(prompts, params)
        s = harness.guard.summary()
        fired = set(s["by_check"])
        ok = (fired >= expected) and (expected or not fired)
        failures += not ok
        print(f"  {fault:10s} {s['steps_checked']:4d} steps checked, "
              f"{s['violations']:4d} violations {sorted(fired) or '(none)'}"
              f"   {'ok' if ok else 'UNGUARDED'}")
        if not ok:
            print(f"      expected {sorted(expected) or '(none)'}")
        for line in s["first"][:1]:
            print(f"      e.g. {line}")

    print(f"\n  {'every injected fault was caught' if not failures else
             f'{failures} arm(s) went undetected'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
