#!/usr/bin/env python3
"""Can a block be freed and restored on a running request?

The last unanswered mechanical question before a pager is worth building.
`blocktable_evict.py` showed the engine will attend to a subset of a request's
blocks; it never gave one back. Both tools so far impose a *view* and leave the
bytes where they are, so nothing they measured depends on the KV surviving a
round trip through host memory and landing somewhere else.

That round trip is the pager's actual data path, and each leg can fail
differently: the read can miss part of the block, the physical block can be
reused by another request while it is away, the write-back can land in the
wrong place, and the view can point at the old address. So this does the whole
cycle inside a single decode step, where it has an exact reference:

    save     the block's KV for every layer is copied out to pinned host memory
    destroy  the GPU block is filled with noise -- standing in for the block
             being freed and handed to another request, which is the thing that
             makes a pager save memory and the thing most likely to corrupt it
    restore  the host copy is written into a *different* physical block
    repoint  the residency view names the new block instead of the old one

If all four legs work, the model reads exactly the bytes it would have read
anyway, from a new address, and the step's output is **bit-identical** to a run
that never touched anything. That is a real reference, unlike eviction, which
changes what the model computes and so has nothing to compare against.

`nocopy` is the arm that makes a pass mean something: identical in every respect
except that the restore is skipped, so the view points at a block holding noise.
It must diverge. If it does not, the destroy never happened and `relocate` was
passing on bytes that were never in danger.

Physical relocation is deliberate rather than incidental. A pager that brings a
block back into whichever slot is free is the useful kind; one that must restore
to the same address would be a much weaker mechanism, and yesterday's result
that block *order* carries no meaning is what makes the stronger version
plausible. This checks it directly.

    tools/kv_roundtrip.py run MODEL OUT.json [--block N] [--ctx N]
    tools/kv_roundtrip.py report OUT.json ...

Single KV-cache-group models only: with a hybrid stack the layers do not share
a block space and "the block" stops being well defined.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blocktable_permute import capture, compare, engine_info, haystack  # noqa: E402

ARMS = ("control", "relocate", "nocopy", "control2")


def scribble(t):
    """Overwrite a KV block with garbage, whatever the cache dtype is.

    An fp8 cache is stored as bytes, so `normal_` does not apply; and random
    bytes reinterpreted as e4m3 include the NaN and Inf exponent patterns,
    which would propagate through attention and make the negative control fail
    for a reason unrelated to the bytes being wrong. Staying inside the finite
    range keeps "the model read the wrong numbers" distinct from "the model
    read a NaN".
    """
    if t.is_floating_point():
        t.normal_()
    else:
        t.random_(1, 127)


class RoundTrip:
    """Save, destroy, restore and repoint one block, every decode step.

    Hooks `prepare_attn` after the fact, like `blocktable_evict.py` and for the
    same reason: the slot mapping for this step's own key was already computed
    from the untouched row, so the new key still lands correctly and only the
    kernel's view is rewritten. The block being relocated is never the tail, so
    nothing this does can collide with that write.
    """

    def __init__(self, block, scratch_from_end=1):
        self.block = block                  # row index to relocate
        self.scratch_from_end = scratch_from_end
        self.mode = None
        self.runner = None
        self.saved = None                   # host copy, per layer
        self.reset("control")

    def reset(self, mode):
        self.mode = mode
        self.rows = 0
        self.done = 0
        self.saved = None
        self.checks = {"destroyed": None, "restored_exact": None,
                       "source_still_destroyed": None,
                       "scratch_block": None, "source_block": None}

    def install(self):
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner

        if getattr(GPUModelRunner.prepare_attn, "_roundtrip_hooked", False):
            return
        original = GPUModelRunner.prepare_attn

        def hooked(runner, input_batch, *args, **kwargs):
            self.runner = runner
            out = original(runner, input_batch, *args, **kwargs)
            self.apply(runner, input_batch, out[0])
            return out

        hooked._roundtrip_hooked = True
        GPUModelRunner.prepare_attn = hooked

    def caches(self, runner):
        """The KV tensors this block lives in, one per layer."""
        return runner.kv_caches

    def apply(self, runner, batch, block_tables):
        if self.mode in ("control", "control2"):
            return
        for b in range(batch.num_reqs):
            if int(batch.num_scheduled_tokens[b]) != 1:
                continue                    # decode steps only
            computed = int(batch.num_computed_tokens_np[b])
            block_size = runner.block_tables.kernel_block_sizes[0]
            n_full = computed // block_size
            if self.block >= n_full:
                continue                    # not a full block yet
            self.rows += 1
            self._cycle(runner, block_tables[0][b])

    def _cycle(self, runner, row):
        caches = self.caches(runner)
        src = int(row[self.block].item())
        scratch = caches[0].shape[0] - self.scratch_from_end
        assert scratch != src, "scratch block collides with the source"

        # Save and destroy happen once. After that the block *is* evicted:
        # its only copy lives on the host, and the GPU original is gone the way
        # a freed block is gone. Re-saving each step would quietly launder the
        # destruction -- the second save would read back what the first restore
        # wrote, and the test would pass without the host copy mattering.
        if self.saved is None:
            self.saved = [c[src].to("cpu", non_blocking=False).clone()
                          for c in caches]
            for c in caches:
                scribble(c[src])
            self.checks["destroyed"] = bool(
                not torch.equal(caches[0][src].cpu(), self.saved[0])
            )
            self.checks["source_block"] = src
            self.checks["scratch_block"] = scratch
        saved = self.saved

        if self.mode == "relocate":
            for c, s in zip(caches, saved):
                c[scratch].copy_(s)
            if self.checks["restored_exact"] is None:
                self.checks["restored_exact"] = all(
                    torch.equal(c[scratch].cpu(), s) for c, s in zip(caches, saved)
                )
                # The source must still be wreckage at this point, or the
                # restore is reading from a block that was never freed.
                self.checks["source_still_destroyed"] = bool(
                    not torch.equal(caches[0][src].cpu(), saved[0])
                )
        elif self.mode == "nocopy":
            # Same repoint, no restore: the view names a block holding noise.
            for c in caches:
                scribble(c[scratch])
            self.checks["restored_exact"] = False

        # repoint: the view names the new address, in the old row position.
        row[self.block] = scratch
        self.done += 1


def run(args):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=args.ctx + args.tokens + 64,
        gpu_memory_utilization=args.util,
        enforce_eager=True,
        enable_prefix_caching=False,
        max_num_seqs=1,
        trust_remote_code=True,
        **({"kv_cache_dtype": args.kv} if args.kv != "auto" else {}),
    )
    tok = llm.get_tokenizer()
    ids = haystack(tok, args.ctx)
    prompts = [{"prompt_token_ids": ids[: args.ctx - 7]}]
    params = SamplingParams(temperature=0.0, max_tokens=args.tokens,
                            logprobs=args.k, ignore_eos=True)

    trip = RoundTrip(args.block)
    trip.install()

    out = {"model": args.model, "kv": args.kv, "ctx": args.ctx,
           "tokens": args.tokens,
           "block": args.block, "arms": {}}
    for arm in ARMS:
        trip.reset(arm)
        results = llm.generate(prompts, params)
        out["arms"][arm] = {"hook": {"rows": trip.rows, "done": trip.done,
                                     **trip.checks},
                            "reqs": [capture(r) for r in results]}
        h = out["arms"][arm]["hook"]
        print(f"ARM {arm}: relocated {h['done']}/{h['rows']} decode steps"
              + (f", block {h['source_block']} -> {h['scratch_block']}"
                 if h["source_block"] is not None else ""), flush=True)

    groups = len(trip.runner.kv_cache_config.kv_cache_groups)
    if groups != 1:
        print(f"WARNING: {groups} KV cache groups; results are not meaningful")
    out["engine"] = engine_info(trip.runner)
    out["engine"]["kv_groups"] = groups
    out["engine"]["layers"] = len(trip.runner.kv_caches)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}", flush=True)
    report_one(out)


def identical(m):
    return m["ids_match"] and m["kl_max"] == 0.0 and m["dlogprob_max"] == 0.0


def report_one(out):
    e = out.get("engine", {})
    print(f"\n{out['model']}  kv={out.get('kv', 'auto')}  ctx={out['ctx']}  "
          f"relocating row block "
          f"{out['block']} every decode step")
    if e:
        print(f"  backend {'+'.join(e['backends'])}  {e.get('layers')} layers, "
              f"{e.get('kv_groups')} KV cache group(s)")
    ref = out["arms"]["control"]
    for arm in ARMS[1:]:
        got = out["arms"][arm]
        h = got["hook"]
        m = compare(ref["reqs"][0], got["reqs"][0])
        tag = ("bit-identical" if identical(m) else
               "tokens identical" if m["ids_match"] else
               f"tokens diverge at step {m['first_divergence']}")
        print(f"  {arm}: {h['done']}/{h['rows']} steps relocated -> {tag}"
              f", |dlogprob| max {m['dlogprob_max']:.2e}, KL max "
              f"{m['kl_max']:.2e}")
        if h["source_block"] is not None:
            print(f"      destroy changed the source block: "
                  f"{'yes' if h['destroyed'] else 'NO'}; restore byte-exact: "
                  f"{h['restored_exact']}; source still wreckage at restore: "
                  f"{h.get('source_still_destroyed')}")
        gone = m.get("top_token_gone", 0)
        if gone:
            print(f"      reference top token outside top-k at {gone} step(s)"
                  f" -- the KL above is over the steps where it was not")
    verdict(out)


def verdict(out):
    arms = out["arms"]
    ref = arms["control"]["reqs"][0]
    reloc = compare(ref, arms["relocate"]["reqs"][0])
    nocopy = compare(ref, arms["nocopy"]["reqs"][0])
    ctl2 = compare(ref, arms["control2"]["reqs"][0])
    h = arms["relocate"]["hook"]

    if not h["done"]:
        print("  VERDICT: inconclusive -- no step was relocated")
        return
    if not identical(ctl2):
        print("  VERDICT: inconclusive -- the engine is not reproducible")
        return
    if not h["destroyed"]:
        print("  VERDICT: inconclusive -- overwriting the source block did not "
              "change it, so nothing was ever at risk")
        return
    if identical(nocopy):
        print("  VERDICT: inconclusive -- the arm that skips the restore is "
              "also bit-identical, so the model is not reading the relocated "
              "block at all")
        return
    if identical(reloc):
        print(f"  VERDICT: a block can be freed and restored on a running "
              f"request -- {h['done']} round trips through host memory into a "
              f"different physical block, output bit-identical, while the same "
              f"relocation without the copy diverges at step "
              f"{nocopy['first_divergence']}")
    else:
        print(f"  VERDICT: the round trip is LOSSY -- output diverges at step "
              f"{reloc['first_divergence']} even though the restore reported "
              f"byte-exact={h['restored_exact']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("model")
    r.add_argument("out")
    r.add_argument("--ctx", type=int, default=2048)
    r.add_argument("--tokens", type=int, default=32)
    r.add_argument("--block", type=int, default=40)
    r.add_argument("--k", type=int, default=20)
    r.add_argument("--util", type=float, default=0.60)
    r.add_argument("--kv", default="auto")
    r.set_defaults(func=run)
    p = sub.add_parser("report")
    p.add_argument("out", nargs="+")
    p.set_defaults(func=lambda a: [report_one(json.load(open(f))) for f in a.out])
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
