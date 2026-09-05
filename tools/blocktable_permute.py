#!/usr/bin/env python3
"""Does block *order* matter? The gate on score-driven KV residency.

A pager that manages residency by rewriting `req_to_blocks` is only possible if
a request's block table can be reordered without changing what the model
computes. The argument says it can: a decode step's attention output is
`sum_i softmax(q.k_i) v_i` over the resident set, softmax is permutation
invariant, and each cached K already carries the RoPE rotation for its own
position -- so nothing downstream of the cache should be able to tell block 7
from block 3. If that holds, residency is a block-table rewrite on stock FA,
with no mask, no `null_block` and no FA4. If it does not, something reads
position from slot index and the whole plan changes shape.

This measures it instead of arguing it. Between decode steps the hook permutes
the *full* blocks of every running request's block table row, leaving the
partial tail block in place; the tail has to stay last because a paged kernel
derives its valid-token count from `seq_len` minus the full blocks before it,
and because the next token's slot mapping is computed from that same row.

Five arms, in this order, on one engine, and four of them are controls:

    control    no permutation -- the reference
    identity   the same GPU index copy with the identity permutation -- the
               rewrite *mechanism* is inert, or nothing else here means anything
    permute    the test: full blocks reshuffled before every decode step
    tailswap   negative control: the partial tail swapped with a full block
    control2   no permutation again -- the engine is reproducible run to run,
               and tailswap's corruption stayed inside its own request

`tailswap` is what makes a `permute` pass mean anything. A test that only ever
reports "no change" cannot distinguish a permutation-invariant kernel from a
hook that never fired, so one arm deliberately breaks the invariant the other
respects: it moves the block holding positions the kernel is about to read
*and* write. If tailswap does not diverge, the instrument is broken and the
permute result should be discarded.

The verdict is read at layer 0 of the first rewritten decode step (`AttnProbe`),
because that is the only place the arms are comparable input-for-input.
Generated tokens are the wrong instrument: an execution-mode change alone has
been measured flipping 9 argmax decisions in 91 positions on this repo's own
checkpoints, and permuting blocks changes FlashAttention's accumulation order,
so bitwise-identical output was never the prediction. The trajectory is still
reported -- truncated at the first token disagreement, since past it the arms
are decoding different contexts (the lesson `tp_compare.py` records) -- but as
a consequence, not as the claim.

Only full-attention KV cache groups are permuted. A sliding-window layer's mask
is built from key position, which a paged kernel reconstructs from the block's
index in the row -- exactly the "reads position from slot index" failure this
test looks for, but by design rather than by accident. Those groups are counted
and skipped, and a model made only of them cannot answer the question here.
`--all-groups` permutes them anyway, which turns that reasoning into a
measurement: on gemma-4-E2B it moves layer 0 by 5.61 on a scale of 11.4.

    tools/blocktable_permute.py run MODEL OUT.json [--kv fp8] [--graphs]
                                    [--ctx N] [--tokens N] [--all-groups]
    tools/blocktable_permute.py report OUT.json [OUT.json ...]

Env: HF_HUB_OFFLINE etc. as usual; the engine is forced in-process
(VLLM_ENABLE_V1_MULTIPROCESSING=0) because the hook has to run in the same
interpreter as the model runner. Prefix caching is disabled so that a permuted
block can never be handed to another request.

Results and what they settle: docs/kv-pager.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.core import kl  # noqa: E402

ARMS = ("control", "identity", "permute", "tailswap", "control2")


def _depth(name):
    """Sort key: the layer index inside a module path like `layers.7.self_attn`."""
    nums = [int(p) for p in name.split(".") if p.isdigit()]
    return (nums[0] if nums else 1 << 30, name)


class AttnProbe:
    """Captures every attention layer's output at one chosen decode step.

    The point of a direct probe: at the *first* step the permutation acts on,
    both arms enter layer 0 with bit-identical inputs -- the prompt was
    prefilled without any rewrite, so the cache contents, the sampled token and
    therefore Q, K and V all match to the bit, and the new key lands in the
    same physical slot either way (the tail block is never moved). The only
    difference in the universe at that instant is the order of the block ids.
    So layer 0's output here is the permutation's effect, measured rather than
    inferred from what it does to logits sixteen layers later.

    Deeper layers are reported too, but they are no longer clean: they see an
    input that already carries the perturbation, so they show how it grows, not
    where it came from.

    Forward hooks do not run inside a replayed CUDA graph, so this is an
    eager-mode instrument and reports nothing under `--graphs`.
    """

    def __init__(self):
        self.active = False
        self.arm = None
        self.data = {}
        self.installed = False

    def install(self, model):
        if self.installed:
            return
        try:
            from vllm.model_executor.layers.attention import Attention
        except ImportError:      # pre-rename tree
            from vllm.attention.layer import Attention

        layers = [(n, m) for n, m in model.named_modules() if isinstance(m, Attention)]
        for name, module in layers:
            module.register_forward_hook(
                lambda mod, args, out, name=name: self._record(name, out)
            )
        self.installed = bool(layers)

    def _record(self, name, out):
        if not self.active or not isinstance(out, torch.Tensor):
            return
        self.data.setdefault(self.arm, {})[name] = out.detach().clone()

    def compare(self, ref_arm, arm, permuted=()):
        """Per-layer distance in ulps -- adjacent representable values apart.

        Absolute error means nothing without the scale it sits on, and a
        relative error means nothing near zero. The unit that answers the
        actual question is the *representation's own step*: reordering the
        terms of a sum can move the fp32 accumulator by a few parts in 1e7, so
        after rounding to the output dtype it can change an element by one
        representable step and no more. An element two or more steps away was
        not produced by adding the same numbers in a different order.

        Distance is read off the bit patterns rather than estimated from a
        norm, so it is exact and scale-free: IEEE floats of the same sign are
        ordered by their integer encodings, and the map below extends that
        across zero.
        """
        a, b = self.data.get(ref_arm, {}), self.data.get(arm, {})
        out = []
        # Depth order, not registration order: "the first layer to see the
        # permutation" is the whole claim, so it must really be the first.
        for name in sorted(a, key=_depth):
            if name not in b:
                continue
            x, y = a[name], b[name]
            d = _ulp_distance(x, y)
            xf, yf = x.float(), y.float()
            diff = (xf - yf).abs()
            scale = xf.abs().max().item()
            # Elements near zero are the difference of signed terms that nearly
            # cancelled, so their low bits carry no information and their ulp
            # distance can be enormous while the absolute difference is
            # nothing. The claim is about elements that carry magnitude, so the
            # headline number is restricted to those; the unrestricted one is
            # reported beside it rather than hidden.
            sig = xf.abs() >= 0.01 * scale
            sig_d = d[sig]
            out.append({
                "layer": name,
                # A layer in a group the hook skips is a check, not a result:
                # it must come back bit-identical, or the skip did not hold.
                "permuted": name in permuted,
                "elements": int(d.numel()),
                "scale": scale,
                "max_abs": diff.max().item(),
                "ulps_max": int(d.max()),
                "over_1_ulp": int((d > 1).sum()),
                "sig_elements": int(sig.sum()),
                "sig_ulps_max": int(sig_d.max()) if sig_d.numel() else 0,
                "sig_over_1_ulp": int((sig_d > 1).sum()),
                # One representable step at the tensor's own scale: the
                # coarsest thing the output dtype can express there, and so the
                # size a rounding difference is allowed to reach.
                "step_at_scale": torch.finfo(x.dtype).eps * scale,
                "exact": bool((d == 0).all()),
            })
        return out


_ULP_INT = {
    torch.bfloat16: (torch.int16, 1 << 15),
    torch.float16: (torch.int16, 1 << 15),
    torch.float32: (torch.int32, 1 << 31),
}


def _ulp_distance(x, y):
    """Elementwise count of representable values between two tensors.

    Zero means the same bits. One means adjacent -- the smallest difference the
    dtype can express, which is what a reassociated sum is allowed to produce.
    """
    if x.dtype != y.dtype:
        raise ValueError(f"dtype mismatch: {x.dtype} vs {y.dtype}")
    if x.dtype not in _ULP_INT:
        # Nothing sensible to say about an exotic dtype; fall back to equality.
        return (x != y).to(torch.int64)
    itype, half = _ULP_INT[x.dtype]

    def order(t):
        # Sign-magnitude to a monotone integer, so subtraction counts steps
        # across zero as well as within one sign.
        i = t.contiguous().view(itype).to(torch.int64)
        return torch.where(i < 0, -half - i, i)

    return (order(x) - order(y)).abs()


class Permuter:
    """Rewrites block table rows between steps, in the persistent batch.

    Hooks `GPUModelRunner.prepare_attn`, which is the one point where the row
    is settled and not yet read: `execute_model` has already flushed the
    scheduler's staged appends into the persistent GPU block table, and the two
    consumers -- the gather that hands the kernel its block table, and the
    triton kernel that turns positions into KV slots -- both run inside it,
    from the same rows. Rewriting here therefore moves the attention read and
    the KV write together, and nothing else has to be kept in sync: appends are
    staged at `num_blocks[req]` regardless of order, so the permutation
    survives the request growing a block.

    The rows live on the GPU (they are `StagedWriteTensor`s), so the rewrite is
    an index copy on device rather than a numpy shuffle -- the older
    `gpu_model_runner.py` staged block tables on the host and is not the runner
    this engine uses.
    """

    def __init__(self, probe=None, all_groups=False):
        self.all_groups = all_groups
        self.mode = None
        self.runner = None
        self.rng = None
        self.probe = probe
        self.reset("control", 0)

    def reset(self, mode, seed):
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.rows = 0          # decoding request x cache group pairs seen
        self.touched = 0       # rows it actually rewrote
        self.blocks_max = 0    # widest row it rewrote, in kernel blocks
        self.skipped_groups = 0
        self.checked = False   # multiset preservation, verified once per arm
        self.decode_steps = 0  # forwards in which some request was decoding

    def install(self):
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner

        if getattr(GPUModelRunner.prepare_attn, "_permute_hooked", False):
            return
        original = GPUModelRunner.prepare_attn

        def hooked(runner, input_batch, *args, **kwargs):
            self.runner = runner
            self.apply(runner, input_batch)
            return original(runner, input_batch, *args, **kwargs)

        hooked._permute_hooked = True
        GPUModelRunner.prepare_attn = hooked

    def apply(self, runner, batch):
        decoding = any(
            int(batch.num_scheduled_tokens[b]) == 1 for b in range(batch.num_reqs)
        )
        if decoding:
            self.decode_steps += 1
        if self.probe is not None:
            # Armed for exactly the first decode forward of every arm -- the
            # one place the arms are still comparable input-for-input.
            self.probe.install(runner.model)
            self.probe.arm = self.mode
            self.probe.active = decoding and self.decode_steps == 1
        if self.mode in ("control", "control2"):
            return
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        tables = runner.block_tables
        specs = [g.kv_cache_spec for g in runner.kv_cache_config.kv_cache_groups]
        for b in range(batch.num_reqs):
            if int(batch.num_scheduled_tokens[b]) != 1:
                continue        # prefill or chunk: the row is still filling
            req_idx = int(batch.idx_mapping_np[b])
            computed = int(batch.num_computed_tokens_np[b])
            for g, table in enumerate(tables.block_tables):
                # One row per cache group per decoding request: a hybrid model
                # has several block tables for the same request, and only some
                # of them are the hook's business.
                self.rows += 1
                if (not self.all_groups and g < len(specs)
                        and not isinstance(specs[g], FullAttentionSpec)):
                    # Sliding window and friends: the kernel rebuilds key
                    # position from the block's index in the row, so order is
                    # load-bearing there by construction. Counted, not touched.
                    self.skipped_groups += 1
                    continue
                block_size = tables.kernel_block_sizes[g]
                self._rewrite(
                    table.gpu[req_idx],
                    computed // block_size,
                    bool(computed % block_size),
                )

    def _rewrite(self, row, n_full, has_tail):
        """Permute one row whose leading `n_full` blocks are completely full.

        Full is the whole point: every one of those blocks holds block_size
        real keys, so moving it changes only which physical block sits at which
        index. The partial tail, if there is one, holds `computed % block_size`
        keys at a count the kernel infers from `seq_len` and the blocks before
        it, and it is where this step's own key lands -- so it stays put.
        """
        if self.mode == "identity":
            # Same GPU index copy, same rows, permutation = the identity. The
            # control for the rewrite *mechanism*: if this is not bit-exact,
            # nothing measured through the hook can be blamed on ordering.
            if n_full < 2:
                return
            idx = torch.arange(n_full, device=row.device)
            row[:n_full] = row[:n_full][idx]
        elif self.mode == "permute":
            if n_full < 2:
                return
            perm = torch.as_tensor(
                self.rng.permutation(n_full), device=row.device, dtype=torch.long
            )
            before = row[:n_full].clone() if not self.checked else None
            row[:n_full] = row[:n_full][perm]
            if before is not None:
                assert torch.equal(*(torch.sort(x).values
                                     for x in (before, row[:n_full]))), \
                    "the rewrite lost or duplicated a block"
                self.checked = True
        elif self.mode == "tailswap":
            # Only while a partial tail exists: with none, index n_full is an
            # empty block, and moving it is just another permutation of the
            # full blocks -- i.e. the arm being controlled for.
            if n_full < 1 or not has_tail:
                return
            idx = torch.tensor(
                [int(self.rng.integers(n_full)), n_full],
                device=row.device, dtype=torch.long,
            )
            row[idx] = row[idx.flip(0)]
            self.checked = True
        else:
            raise ValueError(self.mode)
        self.touched += 1
        self.blocks_max = max(self.blocks_max, n_full)


def permuted_layers(runner, all_groups=False):
    """Names of the attention layers whose block table the hook may rewrite."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    return {
        name
        for group in runner.kv_cache_config.kv_cache_groups
        if all_groups or isinstance(group.kv_cache_spec, FullAttentionSpec)
        for name in group.layer_names
    }


def engine_info(runner):
    """What the arms actually ran on -- the result is only about this."""
    tables = runner.block_tables
    return {
        "block_sizes": list(tables.block_sizes),
        "kernel_block_sizes": list(tables.kernel_block_sizes),
        "backends": sorted({
            g.backend.get_name() if hasattr(g.backend, "get_name")
            else g.backend.__name__
            for groups in runner.attn_groups for g in groups
        }),
        "groups": [type(g.kv_cache_spec).__name__
                   for g in runner.kv_cache_config.kv_cache_groups],
    }


def haystack(tok, ctx):
    """`ctx` tokens of wikitext-103, the same source `niah_kv.py` draws on."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    text, i = "", 0
    while len(text) < (ctx + 2000) * 4:
        text += ds[i]["text"]
        i += 1
    return tok.encode(text)[:ctx]


def run(args):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    from vllm import LLM, SamplingParams

    probe = None if args.graphs or args.no_probe else AttnProbe()
    permuter = Permuter(probe, args.all_groups)
    permuter.install()

    kwargs = dict(
        model=args.model,
        max_model_len=args.ctx + args.tokens + 64,
        gpu_memory_utilization=args.util,
        enforce_eager=not args.graphs,
        enable_prefix_caching=False,   # permuted blocks must never be reused
        max_num_seqs=max(args.reqs, 1),
        trust_remote_code=True,
    )
    if args.block_size:
        kwargs["block_size"] = args.block_size
    if args.kv != "auto":
        kwargs["kv_cache_dtype"] = args.kv
    llm = LLM(**kwargs)
    tok = llm.get_tokenizer()

    ids = haystack(tok, args.ctx)
    # Distinct lengths, none of them a multiple of any plausible block size:
    # a request whose prompt ends exactly on a block boundary has no partial
    # tail, which is precisely the case the negative control cannot act on, and
    # an arm that silently skips a request is not a control. The lengths also
    # differ between requests so that a row/index mix-up in the hook shows up
    # as a mismatch rather than as a coincidence.
    prompts = [{"prompt_token_ids": ids[: args.ctx - 7 - 37 * r]}
               for r in range(args.reqs)]
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.tokens,
        logprobs=args.k,
        ignore_eos=True,       # every arm takes exactly the same number of steps
    )

    out = {
        "model": args.model,
        "kv": args.kv,
        "mode": "graphs" if args.graphs else "eager",
        "ctx": args.ctx,
        "tokens": args.tokens,
        "reqs": args.reqs,
        "k": args.k,
        "all_groups": args.all_groups,
        "arms": {},
    }
    for arm in ARMS:
        permuter.reset(arm, args.seed)
        results = llm.generate(prompts, params)
        out["arms"][arm] = {
            "hook": {
                "rows": permuter.rows,
                "touched": permuter.touched,
                "blocks_max": int(permuter.blocks_max),
                "skipped_groups": permuter.skipped_groups,
            },
            "reqs": [capture(r) for r in results],
        }
        h = out["arms"][arm]["hook"]
        print(f"ARM {arm}: rewrote {h['touched']}/{h['rows']} decode rows, "
              f"widest {h['blocks_max']} full blocks", flush=True)

    out["engine"] = engine_info(permuter.runner)
    if probe is not None and probe.data:
        permuted = permuted_layers(permuter.runner, args.all_groups)
        out["probe"] = {
            arm: probe.compare("control", arm, permuted) for arm in ARMS[1:]
        }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}", flush=True)
    report_one(out)


def capture(result):
    gen = result.outputs[0]
    steps = []
    for pos in gen.logprobs or []:
        steps.append({str(t): round(lp.logprob, 6) for t, lp in pos.items()})
    return {
        "prompt_len": len(result.prompt_token_ids),
        "ids": [int(t) for t in gen.token_ids],
        "steps": steps,
    }


def compare(ref, arm):
    """Per-decode-step divergence, truncated at the first token disagreement.

    Past that point the two arms are continuing different token sequences, so
    their distributions are not answering the same question and averaging them
    in would understate or overstate divergence arbitrarily.

    `clean` is the one step that needs no floor to interpret. Generated token 0
    comes out of the prefill forward, which no arm touches -- so at step 1 the
    two arms have bit-identical weights, cache and input token, and whatever
    separates their distributions is the rewrite and nothing else, with no
    compounding yet. It is the probe's measurement taken at the logits instead
    of at layer 0, which makes it the only one available under CUDA graphs.
    """
    n = min(len(ref["ids"]), len(arm["ids"]))
    first = next((i for i in range(n) if ref["ids"][i] != arm["ids"][i]), None)
    upto = n if first is None else first + 1
    kls, dtop, gone = [], [], 0
    for i in range(upto):
        a, b = ref["steps"][i], arm["steps"][i]
        if not a or not b:
            continue
        kls.append(kl(a, b))
        t = max(a, key=a.get)
        if t in b:
            dtop.append(abs(a[t] - b[t]))
        else:
            # The reference's own choice is not even in the other arm's top-k.
            # That is the largest divergence this metric can see, and it is
            # invisible in `dlogprob_max`, which averages over what remains.
            gone += 1
    finite = [v for v in kls if v == v]
    return {
        "steps": upto,
        "first_divergence": first,
        "ids_match": first is None and len(ref["ids"]) == len(arm["ids"]),
        "kl_max": max(finite) if finite else 0.0,
        # No comparable entry means the reference's top token fell out of the
        # other arm's top-k entirely, which is a large divergence -- reporting
        # it as 0.0 would read as agreement.
        "dlogprob_max": max(dtop) if dtop else (0.0 if not kls else float("nan")),
        "dlogprob_mean": (sum(dtop) / len(dtop) if dtop
                          else (0.0 if not kls else float("nan"))),
        # Step 0 is the prefill's own output: it must be identical in every
        # arm, and if it is not, the run's premise is broken rather than its
        # result interesting.
        "top_token_gone": gone,
        "prefill_step_clean": len(kls) > 0 and kls[0] == 0.0,
        "clean_kl": kls[1] if len(kls) > 1 else float("nan"),
        "clean_dlogprob": dtop[1] if len(dtop) > 1 else float("nan"),
    }


def report_one(out):
    e = out.get("engine", {})
    print(f"\n{out['model']}  kv={out['kv']}  {out['mode']}  "
          f"ctx={out['ctx']} x{out['reqs']} reqs, {out['tokens']} decode steps")
    if e:
        print(f"  backend {'+'.join(e['backends'])}  kernel block "
              f"{e['kernel_block_sizes']} (alloc {e['block_sizes']})  "
              f"groups {'+'.join(e['groups'])}")
    ref = out["arms"]["control"]
    for arm in ARMS[1:]:
        got = out["arms"][arm]
        h = got["hook"]
        print(f"  {arm}: rewrote {h['touched']}/{h['rows']} decode rows"
              + (f", skipped {h['skipped_groups']} non-full-attention"
                 if h["skipped_groups"] else ""))
        for i, (a, b) in enumerate(zip(ref["reqs"], got["reqs"])):
            m = compare(a, b)
            tag = ("tokens identical" if m["ids_match"]
                   else f"tokens DIVERGE at step {m['first_divergence']}")
            print(f"      req{i} ({a['prompt_len']} tok): {tag}, over "
                  f"{m['steps']} comparable steps |dlogprob| max "
                  f"{m['dlogprob_max']:.2e} mean {m['dlogprob_mean']:.2e}, "
                  f"KL max {m['kl_max']:.2e}")
            print(f"           first rewritten step alone: KL "
                  f"{m['clean_kl']:.2e}, |dlogprob| {m['clean_dlogprob']:.2e}"
                  + ("" if m["prefill_step_clean"]
                     else "   [!] the untouched prefill step already differs"))
        layers = out.get("probe", {}).get(arm)
        if layers:
            touched = [l for l in layers if l.get("permuted", True)]
            if not touched:
                continue
            first = touched[0]
            # A skipped layer is only evidence about the skip if it runs
            # *before* the first layer that was permuted. Deeper ones read a
            # hidden state that already carries the perturbation, so their
            # changing says nothing about their own block table.
            cut = _depth(first["layer"])
            upstream = [l for l in layers
                        if not l.get("permuted", True) and _depth(l["layer"]) < cut]
            if upstream:
                dirty = [l["layer"] for l in upstream if not l["exact"]]
                print(f"      probe: {len(upstream)} skipped layers run before "
                      f"the first permuted one, " +
                      ("all bit-identical" if not dirty else
                       f"but {len(dirty)} changed: {dirty[0]} [!] -- the "
                       f"group skip did not hold"))
            exact = sum(1 for l in layers if l["exact"])
            shape = ("bit-identical" if first["exact"] else
                     f"{first['sig_ulps_max']} ulp max over "
                     f"{first['sig_elements']} elements with magnitude "
                     f"({first['sig_over_1_ulp']} past 1 ulp), "
                     f"|d| <= {first['max_abs']:.2e} on scale "
                     f"{first['scale']:.2e}")
            print(f"      probe, first permuted step: {first['layer']} "
                  f"{shape}; {exact}/{len(layers)} layers bit-identical")
    verdict(out)


def verdict(out):
    ref = out["arms"]["control"]

    def arm(name):
        return [compare(a, b)
                for a, b in zip(ref["reqs"], out["arms"][name]["reqs"])]

    perm, tail = arm("permute"), arm("tailswap")
    ctl2, ident = arm("control2"), arm("identity")

    # Per request, never pooled: the arms act on each request separately, and
    # a maximum over requests can end up comparing one arm's request 0 against
    # another arm's request 1.
    for i in range(len(perm)):
        print(f"  first rewritten step, req{i}: permute KL "
              f"{perm[i]['clean_kl']:.2e} | tailswap KL "
              f"{tail[i]['clean_kl']:.2e} | identity KL "
              f"{ident[i]['clean_kl']:.2e} | rerun KL {ctl2[i]['clean_kl']:.2e}")

    def clean(ms):
        vals = [m["clean_kl"] for m in ms if m["clean_kl"] == m["clean_kl"]]
        return max(vals) if vals else float("nan")

    if not out["arms"]["permute"]["hook"]["touched"]:
        print("  VERDICT: inconclusive -- the hook never rewrote a row")
        return
    if not all(m["ids_match"] for m in ctl2):
        print("  VERDICT: inconclusive -- the engine is not reproducible run "
              "to run, so nothing here is attributable to the rewrite")
        return
    if not all(m["ids_match"] for m in ident):
        print("  VERDICT: inconclusive -- the identity rewrite already changed "
              "the output, so the mechanism perturbs and the arms mean nothing")
        return
    if all(m["ids_match"] for m in tail) and not clean(tail) > 0:
        print("  VERDICT: inconclusive -- the negative control did not change "
              "the output, so this run cannot detect a wrong block table")
        return

    # Token identity is deliberately not the bar. This repo has measured an
    # execution-mode change alone flipping 9 argmax decisions in 91 positions,
    # so a run demanding bit-identical generations would reject configurations
    # that are correct. Both tests below are relative: one to what the output
    # dtype can represent, one to what a single genuinely wrong block does in
    # this very run.
    def touched(name):
        got = [l for l in out.get("probe", {}).get(name, [])
               if l.get("permuted", True)]
        return got

    layers, tail_layers = touched("permute"), touched("tailswap")
    if layers and tail_layers:
        p, t = layers[0], tail_layers[0]
        within = p["max_abs"] <= p["step_at_scale"]
        quiet = p["sig_over_1_ulp"] * 3 <= t["sig_over_1_ulp"]
        detail = (f"at {p['layer']}: |d| <= {p['max_abs']:.2e} against a "
                  f"representable step of {p['step_at_scale']:.2e}, "
                  f"{p['sig_over_1_ulp']}/{p['sig_elements']} elements moved "
                  f"past one step against the negative control's "
                  f"{t['sig_over_1_ulp']}")
        if p["exact"]:
            print("  VERDICT: block order does not matter -- the first "
                  "attention to see a permuted table returned a bit-identical "
                  "result")
        elif within and quiet:
            print(f"  VERDICT: block order does not matter beyond rounding -- "
                  f"{detail}")
        else:
            print(f"  VERDICT: block order MATTERS -- {detail}")
        return

    # No probe: CUDA graph replays run no Python hooks, so the finest
    # measurement left is the first rewritten step's logits.
    pairs = [(a["clean_kl"], b["clean_kl"]) for a, b in zip(perm, tail)
             if b["clean_kl"] == b["clean_kl"] and b["clean_kl"] > 0]
    cp = max((a for a, _ in pairs), default=float("nan"))
    ct = min((b for _, b in pairs), default=0.0)
    if not pairs:
        print("  VERDICT (no probe): inconclusive -- the negative control left "
              "the first rewritten step untouched, so there is nothing to "
              "judge the permutation against")
    elif all(a * 3 <= b for a, b in pairs):
        print(f"  VERDICT (no probe): consistent with block order not "
              f"mattering -- at the first rewritten step the permutation moves "
              f"the distribution by KL {cp:.2e} against {ct:.2e} for one "
              f"genuinely wrong block")
    else:
        print(f"  VERDICT (no probe): inconclusive -- at the logits the "
              f"permutation ({cp:.2e}) and one genuinely wrong block "
              f"({ct:.2e}) overlap in size, so this measurement cannot "
              f"separate them. Sixteen layers of amplification is too coarse "
              f"an instrument for the question; the eager-mode probe reads "
              f"the attention that actually saw the rewrite.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("model")
    r.add_argument("out")
    r.add_argument("--kv", default="auto")
    r.add_argument("--graphs", action="store_true")
    r.add_argument("--ctx", type=int, default=2048)
    r.add_argument("--tokens", type=int, default=64)
    r.add_argument("--reqs", type=int, default=2)
    r.add_argument("--k", type=int, default=20)
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--util", type=float, default=0.60)
    r.add_argument("--block-size", type=int, default=0)
    r.add_argument("--no-probe", action="store_true")
    # Sliding-window and other non-full-attention groups are skipped by
    # default because their kernels reconstruct key position from the block's
    # index in the row. This turns the reasoning into a measurement by
    # permuting them anyway.
    r.add_argument("--all-groups", action="store_true")
    r.set_defaults(func=run)
    p = sub.add_parser("report")
    p.add_argument("out", nargs="+")
    p.set_defaults(func=lambda a: [report_one(json.load(open(f))) for f in a.out])
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
