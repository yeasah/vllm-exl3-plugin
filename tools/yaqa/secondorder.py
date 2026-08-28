#!/usr/bin/env python3
"""Does the second-order model still predict the KL at the layer where YAQA fails?

L1.mlp.down_proj is worse than LDLQ at every bitrate, on two models, at four data
budgets, and the damage is entirely output-side: at that layer Sketch B's `H_I` alone is
the *best* arm on the board. Since H_I and H_O come from the same gradients in the same
loop, "bad data" does not explain it.

The remaining suspect is YAQA's premise rather than its inputs. YAQA minimizes

    vec(D)^T H vec(D),   H = the Fisher = E_b[vec(G_b) vec(G_b)^T]

which is the *second-order* approximation of the end-to-end KL. That approximation has a
radius, and L1.mlp.down_proj has by far the most anisotropic H_O in the model (50% of its
trace in 10% of its eigenvalues). If a 2-bit perturbation lands outside that radius, then
minimizing the quadratic stops being the same thing as minimizing the KL -- and an
algorithm that minimizes it harder does worse.

That is directly checkable, because both halves are computable exactly:

  predicted KL(a)  =  1/2 * a^2 * E_b[<D, G_b>_F^2] / T   (exact, held-out sequences)
  actual KL(a)     =  measured, splicing W - a*D into the model

The 1/T matters and is easy to get wrong: `l` is the cross entropy *summed* over a
sequence, so E_b[<D, G_b>^2] is the second-order KL summed over that sequence's T
tokens, while the measured KL is a per-token mean. Without it the prediction is out by
three orders of magnitude.

Sweeping the scale `a` traces the quadratic out to the perturbation size the quantizer
actually produces. The signature of the hypothesis is a ratio actual/predicted that stays
near 1 for LDLQ and blows up for YAQA at the anomalous layer, while a healthy layer keeps
both near 1. The alternative -- that YAQA simply *raises* the true second-order error at
L1, i.e. the sketch is misaligned rather than the quadratic being invalid -- shows up as
a higher Q for YAQA, and implies a completely different fix.

Run a healthy layer alongside the anomalous one; the comparison is the measurement.

Read the ratio at a=1 and treat small-a rows as context only. KL at a=0 is exactly zero
(checked), but by a=0.125 every arm shows the same ~6e-4 regardless of its own D --
arm-independent, so it is not the perturbation. It is the heavy tail of a per-token KL
over few eval sequences: a handful of tokens sit near a decision boundary and any
perturbation flips them. It is a fixed offset, negligible against a=1 and dominant below
a=0.25, so only the a=1 column supports a conclusion.
"""

import argparse, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "deps", "exllamav3"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe import build_rows, layer_hessians, round_arms, mc_sample


def quad_terms(model, linear, rows, device, vocab, seed, deltas):
    """E_b[<D, G_b>^2] per arm -- exactly vec(D)^T H vec(D) for the Fisher H.

    Streams the held-out gradients: one G for an 8192x2048 layer is 64 MiB, and only the
    scalar contractions need to survive the loop.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    linear.weight.requires_grad_(True)
    if getattr(model, "is_gradient_checkpointing", False):
        model.train()
        model.config.use_cache = False
    g = torch.Generator(device = device).manual_seed(seed)
    acc = {k: 0.0 for k in deltas}
    for i in range(rows.shape[0]):
        logits = model(rows[i : i + 1].to(device)).logits.float().reshape(-1, vocab)
        y = mc_sample(logits, g)
        F.cross_entropy(logits, y, reduction = "sum").backward()
        G = linear.weight.grad.detach().float()
        for k, D in deltas.items():
            acc[k] += (D * G).sum().item() ** 2
        linear.weight.grad = None
        del logits, G
    linear.weight.requires_grad_(False)
    model.eval()
    # /T: l is summed over the sequence, the measured KL is a per-token mean.
    T = rows.shape[1]
    return {k: v / rows.shape[0] / T for k, v in acc.items()}


@torch.inference_mode()
def kl_at_scale(model, linear, deltas, alphas, eval_rows, device, vocab):
    """Measured KL(original || W - a*D) for each arm and each scale."""
    orig = linear.weight.data.clone()
    tot = {(k, a): 0.0 for k in deltas for a in alphas}
    n = 0
    for i in range(eval_rows.shape[0]):
        ids = eval_rows[i : i + 1].to(device)
        linear.weight.data.copy_(orig)
        ref_lp = F.log_softmax(model(ids).logits.float().reshape(-1, vocab), dim = -1)
        ref_p = ref_lp.exp()
        for k, D in deltas.items():
            for a in alphas:
                linear.weight.data.copy_((orig.float() - a * D).to(orig.dtype))
                lp = F.log_softmax(model(ids).logits.float().reshape(-1, vocab), dim = -1)
                tot[(k, a)] += (ref_p * (ref_lp - lp)).sum().item()
                del lp
        n += ref_p.shape[0]
        del ref_p, ref_lp
    linear.weight.data.copy_(orig)
    return {k: v / n for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--layers", type = int, nargs = "+", default = [1, 7])
    ap.add_argument("--proj", default = "mlp.down_proj")
    ap.add_argument("--bits", type = int, default = 2)
    ap.add_argument("--cal-seqs", type = int, default = 600)
    ap.add_argument("--sketch-seqs", type = int, default = 2048)
    ap.add_argument("--holdout", type = int, default = 48)
    ap.add_argument("--eval-seqs", type = int, default = 16)
    ap.add_argument("--ctx", type = int, default = 2048)
    ap.add_argument("--seed", type = int, default = 7)
    ap.add_argument("--alphas", type = float, nargs = "+", default = [0.25, 0.5, 1.0])
    ap.add_argument("--grad-checkpoint", action = "store_true")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype = torch.bfloat16, attn_implementation = "eager").to(device).eval()
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs = {"use_reentrant": False})
    vocab = model.config.vocab_size

    cal, e = build_rows(tok, args.cal_seqs, args.holdout + args.eval_seqs, args.ctx, ["in-domain"]); rest = e["in-domain"]
    ho, ev = rest[: args.holdout], rest[args.holdout :]
    print(f" -- cal {tuple(cal.shape)}, holdout {tuple(ho.shape)}, eval {tuple(ev.shape)}, "
          f"K={args.bits}, sketch {args.sketch_seqs}")

    for layer in args.layers:
        linear = model.get_submodule(f"model.layers.{layer}.{args.proj}")
        tag = f"L{layer}.{args.proj}"
        print(f"\n == {tag}", flush = True)
        H_act, H_I_B, H_O_B = layer_hessians(model, linear, cal, device, args.sketch_seqs, vocab)

        recon, names = round_arms(linear, H_act, H_I_B, H_O_B, args.bits,
                                  [0.001], args.seed, device, False)
        W = linear.weight.data.float()
        deltas = {k: (W - v.float()) for k, v in recon.items()}
        del recon

        Q = quad_terms(model, linear, ho, device, vocab, 99, deltas)
        kls = kl_at_scale(model, linear, deltas, args.alphas, ev, device, vocab)

        print(f"    {'arm':14s}{'Q/token':>12s}" +
              "".join(f"{f'a={a:g} pred':>12s}{f'a={a:g} act':>12s}{'ratio':>8s}"
                      for a in args.alphas), flush = True)
        for k in names:
            row = f"    {k:14s}{Q[k]:12.4e}"
            for a in args.alphas:
                pred = 0.5 * a * a * Q[k]
                act = kls[(k, a)]
                row += f"{pred:12.4e}{act:12.4e}{act / max(pred, 1e-30):8.2f}"
            print(row, flush = True)
        del deltas, Q
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
