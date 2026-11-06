#!/usr/bin/env python3
"""Can the engine be told a request has fewer blocks than positions?

`blocktable_permute.py` settled that block *index* carries no positional
meaning, so residency can be expressed by rewriting a block table. It did not
settle the step a pager actually needs, which is harder: dropping blocks
shortens the row, so the token at position `p` stops living at row index
`p // block_size` -- and that identity is what the slot-mapping kernel is built
on. If the engine cannot be handed a request with fewer blocks than positions,
the whole approach is derailed, so this measures it before anything is built.

**What is imposed, and what is not.** The hook rewrites only the *gathered*
block table -- the per-step copy the kernel reads -- and the `seq_lens` the
attention metadata is built from. The persistent rows, the scheduler and the
allocator are untouched, so nothing is freed and no memory is saved. That
separation is deliberate: this asks whether the *kernel* can be told, which is
the question that could derail the plan. Reclaiming the blocks afterwards is
ordinary work vLLM already does on preemption.

The rewrite runs *after* the real `prepare_attn`, which means the slot mapping
for this step's own key was already computed from the untouched row and is
still correct. The evicted view keeps the partial tail block last, so the new
key lands where it always would and no slot surgery is needed anywhere.

Eviction has no free reference: dropping blocks changes what the model
computes, so there is nothing to compare the output against. Four checks stand
in for one, and three of them do have exact references:

    viewfull   every block resident, presented through the eviction path.
               Must be bit-identical to control -- it is the off-by-one
               detector for the residency arithmetic, since a tail count or a
               seq_len that is wrong by one shows up here and nowhere else.
    evict      a real budget: sink blocks + the most recent, middle dropped.
               Unreferenced by construction; this is what runs.
    poison     the same residency, with the row entries past the resident
               prefix overwritten by a duplicate of a resident block. Must be
               bit-identical to `evict`, which is what proves the kernel stops
               at `ceil(seq_len / block_size)` and the trailing entries are
               genuinely not read.
    control2   nothing, again: the engine is still reproducible afterwards.

`--needle` swaps the prompt for a retrieval one and asks the question the
numbers cannot: does the model attend to *exactly* the resident set? A magic
number is planted at a known token offset, so the block holding it is known,
and two arms differ only in whether that one block is in the budget. If
retrieval survives dropping the needle's block, the eviction was not real.

    tools/blocktable_evict.py run MODEL OUT.json [--budget N] [--sink N]
                                  [--needle] [--ctx N] ...

Env and engine setup are `blocktable_permute.py`'s; see docs/kv-pager.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blocktable_permute import (  # noqa: E402
    capture,
    compare,
    engine_info,
    haystack,
)

NUMERIC_ARMS = ("control", "viewfull", "evict", "poison", "control2")
# Two halves of what `evict` does at once, for pinning down which half a
# failure belongs to. `seqonly` is a degenerate policy (it keeps the oldest
# blocks and writes the new key outside the attended range) but it is
# mechanically a request with fewer blocks than positions, which is the point.
DIAGNOSTIC_ARMS = ("control", "seqonly", "rowonly")
NEEDLE_ARMS = ("control", "keep_needle", "drop_needle")

NEEDLE = " The special magic Denver number is: {value}. "
QUESTION = ("\n\nWhat is the special magic Denver number mentioned in the text "
            "above? Answer with the number only.\nAnswer:")


class Evictor:
    """Imposes a residency view on each decode step, after `prepare_attn`.

    Runs after the real one rather than before it, which is what keeps the KV
    write correct for free: `compute_slot_mappings` has already turned this
    step's position into a slot using the untouched row, so the new key goes
    where it always would. Only the gathered block table and `seq_lens` are
    rewritten, and both are per-step views -- the persistent rows the scheduler
    appends to never change.
    """

    def __init__(self, budget, sink, needle_block=None):
        self.budget = budget          # resident full blocks, excluding the tail
        self.sink = sink              # how many of them are taken from the front
        self.needle_block = needle_block
        self.runner = None
        self.mode = None
        self.plan = {}
        self.reset("control")

    def reset(self, mode):
        self.mode = mode
        self.rows = 0
        self.touched = 0
        self.min_resident = None      # blocks presented, at the tightest step
        self.max_positions = 0        # positions those blocks stood in for
        self.dropped = 0

    def install(self):
        """Capture the runner and the step's decode rows; wrap the builders.

        Two hooks, because the two things a residency view needs are decided in
        different places. `prepare_attn` is where the batch is legible -- which
        rows are decoding, and how many tokens each has cached. The rewrite
        itself has to happen later, inside the attention metadata builder,
        for a reason that cost a hang to learn: `input_batch.seq_lens` is
        overloaded. It is both "how many keys attention reads" and "how far
        through the prompt this request is" -- `sampler.py` classifies a row
        with `seq_len < prefill_len` as still prefilling and emits no token for
        it -- so lowering it in place makes the request stop advancing and the
        engine spins forever. A pager needs those two numbers to differ, so the
        view must be handed to the kernel without the rest of the engine
        seeing it.
        """
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner

        if getattr(GPUModelRunner.prepare_attn, "_evict_hooked", False):
            return
        original = GPUModelRunner.prepare_attn

        def hooked(runner, input_batch, *args, **kwargs):
            self.runner = runner
            out = original(runner, input_batch, *args, **kwargs)
            self.plan = {
                b: int(input_batch.num_computed_tokens_np[b])
                for b in range(input_batch.num_reqs)
                if int(input_batch.num_scheduled_tokens[b]) == 1
            }
            self.rows += len(self.plan)
            self.wrap_builders(runner)
            return out

        hooked._evict_hooked = True
        GPUModelRunner.prepare_attn = hooked

    def wrap_builders(self, runner):
        """Interpose on every attention metadata builder, once."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        for groups in runner.attn_groups:
            for group in groups:
                full = isinstance(group.kv_cache_spec, FullAttentionSpec)
                for i, builder in enumerate(group.metadata_builders):
                    if getattr(builder.build, "_evict_hooked", False):
                        continue
                    orig = builder.build

                    # Callers pass `common_attn_metadata` by keyword, so the
                    # wrapper cannot take it positionally.
                    def wrapper(*a, _orig=orig, _full=full,
                                _spec=group.kv_cache_spec, **kw):
                        key = "common_attn_metadata"
                        if key in kw:
                            kw[key] = self.view(kw[key], _full, _spec)
                        elif a:
                            a = (self.view(a[0], _full, _spec),) + a[1:]
                        return _orig(*a, **kw)

                    wrapper._evict_hooked = True
                    builder.build = wrapper

    def resident(self, n_full):
        """Row indices of the full blocks to keep, in ascending order.

        Sink plus most-recent is the recency shape every eviction policy starts
        from. `keep_needle` and `drop_needle` spend one slot of the *same*
        budget differently, so the two arms are equal in everything except
        which single block survives.
        """
        if self.mode in ("control", "control2", "viewfull", "seqonly"):
            return list(range(n_full))
        budget = min(self.budget, n_full)
        sink = min(self.sink, budget)
        keep = list(range(sink))
        if self.mode == "keep_needle" and self.needle_block is not None:
            if self.needle_block < n_full and self.needle_block not in keep:
                keep.append(self.needle_block)
        recent = budget - len(keep)
        keep += [i for i in range(n_full - recent, n_full) if i not in keep]
        return sorted(set(keep))

    def view(self, common, full, spec):
        """Return metadata describing the resident set, or `common` unchanged.

        Everything is a copy: the shared `seq_lens` and the gathered block
        table are what the sampler and the next step's bookkeeping read, and
        they must keep saying what is actually cached. `slot_mapping` is passed
        through untouched and is still correct -- it was computed from the true
        row before this, so this step's own key lands in the tail block, which
        the view always keeps and always keeps last.
        """
        if self.mode in ("control", "control2") or not full or not self.plan:
            return common
        block_size = spec.block_size
        seq = common.seq_lens.clone()
        table = common.block_table_tensor.clone()
        for b, computed in self.plan.items():
            if b >= table.shape[0]:
                continue
            self._row(table[b], seq, b, computed, block_size)
        return common.replace(seq_lens=seq, block_table_tensor=table)

    def _row(self, row, seq, b, computed, block_size):
        n_full = computed // block_size
        # The tail block holds this step's own key, so it is always resident
        # and always last. Its valid count is `seq_len` minus the full blocks
        # in front of it, which is the arithmetic `viewfull` exists to check.
        tail_count = computed % block_size + 1
        keep = self.resident(n_full)
        old = row.clone()
        k = len(keep)
        if self.mode == "seqonly":
            # Row untouched, only the length claim changes: the diagnostic
            # half that isolates seq_lens from the row rewrite.
            k = min(self.budget, n_full)
            seq[b] = k * block_size
            self.touched += 1
            self._note(k, computed)
            return
        if k:
            row[:k] = old[torch.as_tensor(keep, device=row.device)]
        row[k] = old[n_full]
        seq[b] = computed + 1 if self.mode == "rowonly" else k * block_size + tail_count
        if self.mode == "poison" and k:
            # Anything the kernel is not supposed to look at. A duplicate of a
            # resident block rather than a random id: if it were read it would
            # double-count real keys, rather than produce a NaN some kernel
            # might handle specially.
            row[k + 1:] = old[keep[0]]
        self.touched += 1
        self._note(k + 1, computed)

    def _note(self, blocks, computed):
        self.dropped = max(self.dropped, computed // 16 - blocks)
        self.min_resident = (blocks if self.min_resident is None
                             else min(self.min_resident, blocks))
        self.max_positions = max(self.max_positions, computed + 1)


def needle_prompt(tok, ids, block_size, depth_blocks, value):
    """Splice a magic number into the haystack at a known block boundary.

    Planting it at an exact multiple of the block size is what makes the two
    needle arms differ in one block and nothing else -- a needle straddling a
    boundary would live in two blocks and could survive the loss of either.
    """
    text = NEEDLE.format(value=value)
    ndl = tok.encode(text, add_special_tokens=False)
    at = depth_blocks * block_size
    spliced = ids[:at] + ndl + ids[at + len(ndl):]
    q = tok.encode(QUESTION, add_special_tokens=False)
    return spliced + q, at // block_size, (at + len(ndl) - 1) // block_size


def run(args):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=args.model,
        max_model_len=args.ctx + args.tokens + 64,
        gpu_memory_utilization=args.util,
        enforce_eager=True,
        enable_prefix_caching=False,
        max_num_seqs=max(args.reqs, 1),
        trust_remote_code=True,
    )
    if args.kv != "auto":
        kwargs["kv_cache_dtype"] = args.kv
    llm = LLM(**kwargs)
    tok = llm.get_tokenizer()
    ids = haystack(tok, args.ctx)

    value = 918273
    needle_block = None
    if args.needle:
        block_size = args.probe_block_size
        prompt_ids, first, last = needle_prompt(
            tok, ids, block_size, args.needle_block, value)
        if first != last:
            raise SystemExit(f"needle spans blocks {first}..{last}; widen it")
        needle_block = first
        prompts = [{"prompt_token_ids": prompt_ids}]
        arms = NEEDLE_ARMS
    else:
        prompts = [{"prompt_token_ids": ids[: args.ctx - 7 - 37 * r]}
                   for r in range(args.reqs)]
        arms = DIAGNOSTIC_ARMS if args.diagnose else NUMERIC_ARMS

    evictor = Evictor(args.budget, args.sink, needle_block)
    evictor.install()

    params = SamplingParams(temperature=0.0, max_tokens=args.tokens,
                            logprobs=args.k, ignore_eos=not args.needle)
    out = {"model": args.model, "kv": args.kv, "ctx": args.ctx,
           "tokens": args.tokens, "reqs": len(prompts), "budget": args.budget,
           "sink": args.sink, "needle": args.needle,
           "needle_block": needle_block, "value": value, "arms": {}}
    for arm in arms:
        evictor.reset(arm)
        results = llm.generate(prompts, params)
        out["arms"][arm] = {
            "hook": {"rows": evictor.rows, "touched": evictor.touched,
                     "min_resident": evictor.min_resident,
                     "max_positions": evictor.max_positions,
                     "dropped": evictor.dropped},
            "reqs": [capture(r) for r in results],
            "text": [r.outputs[0].text for r in results],
        }
        h = out["arms"][arm]["hook"]
        print(f"ARM {arm}: {h['touched']}/{h['rows']} decode rows, tightest "
              f"view {h['min_resident']} blocks for {h['max_positions']} "
              f"positions", flush=True)

    out["engine"] = engine_info(evictor.runner)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}", flush=True)
    report_one(out)


def identical(m):
    """Bit-identical, not merely close: these arms have exact references."""
    return m["ids_match"] and m["kl_max"] == 0.0 and m["dlogprob_max"] == 0.0


def report_one(out):
    e = out.get("engine", {})
    print(f"\n{out['model']}  kv={out['kv']}  ctx={out['ctx']}  "
          f"budget {out['budget']} blocks (sink {out['sink']})"
          + ("  [needle]" if out["needle"] else ""))
    if e:
        print(f"  backend {'+'.join(e['backends'])}  kernel block "
              f"{e['kernel_block_sizes']}")
    ref = out["arms"]["control"]
    for arm, got in out["arms"].items():
        if arm == "control":
            continue
        h = got["hook"]
        print(f"  {arm}: tightest view {h['min_resident']} blocks standing in "
              f"for {h['max_positions']} positions, {h['dropped']} dropped")
        for i, (a, b) in enumerate(zip(ref["reqs"], got["reqs"])):
            m = compare(a, b)
            tag = ("bit-identical" if identical(m) else
                   "tokens identical" if m["ids_match"] else
                   f"tokens diverge at step {m['first_divergence']}")
            gone = m.get("top_token_gone", 0)
            print(f"      req{i}: {tag}, |dlogprob| max {m['dlogprob_max']:.2e}"
                  f", KL max {m['kl_max']:.2e}"
                  + (f", reference top token outside top-k at {gone} step(s)"
                     if gone else ""))
    verdict(out)


def answered(text, value):
    return str(value) in re.sub(r"[,\s]", "", text)


def verdict(out):
    arms = out["arms"]
    if out["needle"]:
        ctrl = answered(arms["control"]["text"][0], out["value"])
        keep = answered(arms["keep_needle"]["text"][0], out["value"])
        drop = answered(arms["drop_needle"]["text"][0], out["value"])
        print(f"  needle in block {out['needle_block']}, budget "
              f"{out['budget']}: full context {'found' if ctrl else 'MISSED'}"
              f" | needle resident {'found' if keep else 'MISSED'}"
              f" | needle evicted {'found' if drop else 'lost'}")
        for arm in NEEDLE_ARMS:
            print(f"      {arm}: {arms[arm]['text'][0].strip()[:70]!r}")
        if not ctrl:
            print("  VERDICT: inconclusive -- the model cannot retrieve this "
                  "needle with the whole context resident, so the eviction "
                  "arms have nothing to say")
        elif keep and not drop:
            print("  VERDICT: eviction is real and exact -- the answer "
                  "survives when its block is in the budget and is lost when "
                  "the same budget spends that slot elsewhere")
        elif keep and drop:
            print("  VERDICT: eviction is NOT reaching the kernel -- the "
                  "answer survived with its block dropped, so the model is "
                  "still attending to blocks the view excluded")
        else:
            print("  VERDICT: inconclusive -- retrieval failed even with the "
                  "needle resident, so the budget is too tight to separate "
                  "eviction from ordinary context loss")
        return

    ref = arms["control"]["reqs"]

    def cmp(name, against=None):
        base = arms[against]["reqs"] if against else ref
        return [compare(a, b) for a, b in zip(base, arms[name]["reqs"])]

    if "viewfull" not in arms:
        for name in arms:
            if name == "control":
                continue
            print(f"  {name}: ran, "
                  + ("bit-identical" if all(identical(m) for m in cmp(name))
                     else "output changed"))
        print("  VERDICT: diagnostic run -- no claim, only which half runs")
        return
    view, ctl2 = cmp("viewfull"), cmp("control2")
    poison = cmp("poison", against="evict")
    if not arms["evict"]["hook"]["touched"]:
        print("  VERDICT: inconclusive -- the hook never imposed a view")
        return
    if not all(identical(m) for m in ctl2):
        print("  VERDICT: inconclusive -- the engine is not reproducible")
        return
    if not all(identical(m) for m in view):
        m = next(m for m in view if not identical(m))
        print(f"  VERDICT: the residency path is WRONG -- presenting every "
              f"block through it already changes the output (|dlogprob| "
              f"{m['dlogprob_max']:.2e}), so its arithmetic is off before any "
              f"block is dropped")
        return
    if not all(identical(m) for m in poison):
        print("  VERDICT: the engine reads PAST the resident prefix -- "
              "poisoning the row entries beyond it changed the output, so "
              "seq_lens is not the authority and a shortened row is unsafe")
        return
    h = arms["evict"]["hook"]
    print(f"  VERDICT: the engine can be told a request has fewer blocks than "
          f"positions -- {h['min_resident']} blocks stood in for "
          f"{h['max_positions']} positions, the residency path is bit-exact "
          f"when it drops nothing, and nothing past the resident prefix is "
          f"read. What the model then computes has no reference here; "
          f"--needle is the check for that.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("model")
    r.add_argument("out")
    r.add_argument("--kv", default="auto")
    r.add_argument("--ctx", type=int, default=2048)
    r.add_argument("--tokens", type=int, default=32)
    r.add_argument("--reqs", type=int, default=2)
    r.add_argument("--k", type=int, default=20)
    r.add_argument("--util", type=float, default=0.60)
    r.add_argument("--budget", type=int, default=16)
    r.add_argument("--sink", type=int, default=2)
    r.add_argument("--needle", action="store_true")
    r.add_argument("--diagnose", action="store_true")
    r.add_argument("--needle-block", type=int, default=40)
    r.add_argument("--probe-block-size", type=int, default=16)
    r.set_defaults(func=run)
    p = sub.add_parser("report")
    p.add_argument("out", nargs="+")
    p.set_defaults(func=lambda a: [report_one(json.load(open(f))) for f in a.out])
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
