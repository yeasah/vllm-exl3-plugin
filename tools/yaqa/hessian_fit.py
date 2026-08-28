#!/usr/bin/env python3
"""How well does Sketch B actually approximate the true Hessian, on our models?

Theorem 3.4 says YAQA's end-to-end error is bounded through the cosine similarity
between its Kronecker sketch H_O (x) H_I and the true Hessian H. That is the whole
mechanism, and it is measurable directly -- no rounding, no quantizer, no converter.

H is the Fisher, H = E_b[vec(G_b) vec(G_b)^T] over per-sequence gradients G_b of the
Monte-Carlo-sampled cross entropy at the model output. So for any Kronecker sketch,

    <H, H_O (x) H_I>  =  E_b[ tr(G_b^T H_O G_b H_I) ]

and for any weight perturbation D the *true* second-order end-to-end error is

    vec(D)^T H vec(D)  =  E_b[ <D, G_b>_F^2 ]

Both are exact expectations over held-out sequences, computed from cached gradients.
LDLQ is the sketch (H_O = I, H_I = E[x^T x]), so the two sit in the same frame and
the comparison is apples to apples. The paper's "normalized cosine similarity" drops
the ||H|| factor, which is common to both and expensive; this does the same.

If the sketch's alignment is barely better than LDLQ's on our models, the 30% does
not transfer and no amount of converter engineering will recover it.
"""

import argparse, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "deps", "exllamav3"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe import build_rows, collect, mc_sample


def align_streaming(model, linear, rows, device, vocab, seed, sketches):
    """Alignment of each candidate sketch to the true Hessian, in one pass.

    <H, H_O (x) H_I> = E_b[tr(G_b^T H_O G_b H_I)], and tr(G^T H_O G H_I) is
    sum((H_O @ G) * (G @ H_I)) -- two matmuls. Accumulating per sequence instead of
    stacking the gradients matters: one G for an 8B down_proj is 235 MB, so holding 48
    of them would cost 11.3 GB on top of the model.

    `sketches` maps a label to (H_I, H_O). Returns label -> normalized alignment.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    linear.weight.requires_grad_(True)
    # HF no-ops gradient checkpointing outside train() mode. No dropout in these
    # architectures, so this changes memory and nothing else.
    if getattr(model, "is_gradient_checkpointing", False):
        model.train()
        model.config.use_cache = False
    g = torch.Generator(device = device).manual_seed(seed)
    acc = {k: 0.0 for k in sketches}
    for i in range(rows.shape[0]):
        logits = model(rows[i : i + 1].to(device)).logits.float().reshape(-1, vocab)
        y = mc_sample(logits, g)
        F.cross_entropy(logits, y, reduction = "sum").backward()
        G = linear.weight.grad.detach().float()
        for k, (H_I, H_O) in sketches.items():
            acc[k] += ((H_O @ G) * (G @ H_I)).sum().item()
        linear.weight.grad = None
        del logits, G
    linear.weight.requires_grad_(False)
    model.eval()
    n = rows.shape[0]
    return {k: acc[k] / n / (sketches[k][1].norm().item() * sketches[k][0].norm().item())
            for k in sketches}


def collect_snapshots(model, linear, cal, device, vocab, marks):
    """Sketch B accumulated incrementally, snapshotted at each sequence count in `marks`."""
    k, m = linear.in_features, linear.out_features
    for p in model.parameters():
        p.requires_grad_(False)
    linear.weight.requires_grad_(True)
    # HF no-ops gradient checkpointing outside train() mode. No dropout in these
    # architectures, so this changes memory and nothing else.
    if getattr(model, "is_gradient_checkpointing", False):
        model.train()
        model.config.use_cache = False
    H_I = torch.zeros((k, k), dtype = torch.float, device = device)
    H_O = torch.zeros((m, m), dtype = torch.float, device = device)
    g = torch.Generator(device = device).manual_seed(1234)
    out = {}
    for i in range(max(marks)):
        r = i % cal.shape[0]
        logits = model(cal[r : r + 1].to(device)).logits.float().reshape(-1, vocab)
        y = mc_sample(logits, g)
        F.cross_entropy(logits, y, reduction = "sum").backward()
        Gw = linear.weight.grad.detach().float()
        H_I.addmm_(Gw.T, Gw)
        H_O.addmm_(Gw, Gw.T)
        linear.weight.grad = None
        del logits, Gw
        if (i + 1) in marks:
            out[i + 1] = (H_I / (i + 1), H_O / (i + 1))
    linear.weight.requires_grad_(False)
    model.eval()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--layers", type = int, nargs = "+", default = [14])
    ap.add_argument("--projs", nargs = "+", default = ["mlp.down_proj"])
    ap.add_argument("--cal-seqs", type = int, default = 128)
    ap.add_argument("--sketch-seqs", type = int, nargs = "+", default = [64, 256, 1024])
    ap.add_argument("--holdout", type = int, default = 64)
    ap.add_argument("--ctx", type = int, default = 1024)
    ap.add_argument("--attn", default = "eager",
                    help = "eager keeps the KL measurement's noise floor at exactly zero; "
                           "sdpa is cheaper but only matters without --grad-checkpoint")
    ap.add_argument("--grad-checkpoint", action = "store_true",
                    help = "recompute activations in the backward pass. Turns the "
                           "activation term from O(layers) into O(1), which is what makes "
                           "an 8B fit on a 24 GiB card")
    ap.add_argument("--device-map", default = None,
                    help = "'auto' to shard the model across GPUs; autograd crosses "
                           "device boundaries transparently, so the test does not need "
                           "to fit on one card")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False

    tok = AutoTokenizer.from_pretrained(args.model)
    mk = dict(dtype = torch.bfloat16, attn_implementation = args.attn)
    if args.device_map:
        mk["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **mk)
    if not args.device_map:
        model = model.to(device)
    model.eval()
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs = {"use_reentrant": False})
    vocab = model.config.vocab_size

    cal, e = build_rows(tok, args.cal_seqs, args.holdout, args.ctx, ["in-domain"]); ho = e["in-domain"]
    print(f" -- calibration {tuple(cal.shape)}, holdout {tuple(ho.shape)} (disjoint), "
          f"sketch {args.sketch_seqs} seqs")

    marks = sorted(args.sketch_seqs)
    print(f"\n  alignment <H, H_O (x) H_I> / (||H_O|| ||H_I||), higher is better; "
          f"held-out {args.holdout} seqs")
    print(f"\n{'layer.proj':24s}{'LDLQ':>12s}" +
          "".join(f"{f'B@{c}':>12s}" for c in marks) + f"{'best/LDLQ':>11s}")
    for layer in args.layers:
        for proj in args.projs:
            linear = model.get_submodule(f"model.layers.{layer}.{proj}")
            m = linear.out_features
            H_act, _, _ = collect(model, linear, cal, device, 1, vocab)
            snaps = collect_snapshots(model, linear, cal, device, vocab, marks)
            sk = {"ldlq": (H_act, torch.eye(m, device = device))}
            for c in marks:
                sk[c] = snaps[c]
            a = align_streaming(model, linear, ho, device, vocab, 99, sk)
            cs = [a[c] for c in marks]
            print(f"{f'L{layer}.{proj}':24s}{a['ldlq']:12.4e}" +
                  "".join(f"{c:12.4e}" for c in cs) +
                  f"{max(cs) / a['ldlq']:10.2f}x", flush = True)
            del snaps, sk
            torch.cuda.empty_cache()
            print(f"    peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                  flush = True)
            torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
