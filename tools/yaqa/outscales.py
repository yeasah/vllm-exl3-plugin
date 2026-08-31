#!/usr/bin/env python3
"""What is EXL3's own ignored `H_O` worth?

`regularize()` with `apply_out_scales` on -- the converter's *default*, not a corner
case -- folds a per-output-channel scale into `sv` and divides the weight by it. The
layer's true output metric is the identity, so by the congruence in
`test_guard.py:test_transform_consistency` the metric in quantization space becomes

    H_O' = B D_sv B          (B = blockwise Hadamard, orthogonal)

which is not the identity once `sv` carries those scales. `ldlq()` rounds as though it
were. This script measures what that costs, by rounding the same layer both ways and
scoring the whole model's KL against the unquantized original.

Unlike `probe.py` this needs **no gradients, no Fisher sketch and no second pass**: the
Hessian is already sitting in `sv`. The only data-dependent quantity is the activation
Hessian EXL3 already collects, so a run is minutes rather than the ~18 min/layer the
YAQA sketch costs, and the bundled calibration corpus is sufficient.

Arms:

  ldlq        out-scales ON,  H_O = I       -- what EXL3 ships today
  ldlq+ho     out-scales ON,  H_O = B D_sv B -- the free fix
  ldlq-noos   out-scales OFF, H_O = I        -- exact, but gives up out-scales entirely
  ldlq-dup    bit-identical copy of `ldlq`   -- live noise floor, must read +0.0%

`ldlq-noos` is a different `regularize()` call, so it is not a rounding-only contrast
the way `ldlq+ho` is. It is here because "turn the heuristic off" is the other way to
make the metric exact, and a fix is only worth building if it beats that.
"""

import argparse, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "deps", "exllamav3"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import exllamav3.modules.quant.exl3_lib.quantize as qz
from exllamav3.modules.quant.exl3_lib.quantize import (
    regularize, preapply_had_l, preapply_had_r, had_k, had_n, block_rms,
    blockwise_preapply_had_l_, blockwise_preapply_had_r_,
)
from rounding import yaqa_round
from probe import build_rows, kl_vs_ref, pooled, boot_ci, prep_hessian


def collect_act(model, linear, cal, device):
    """Activation Hessian E[xᵀx] -- the only data-dependent quantity here."""
    k = linear.in_features
    H = torch.zeros((k, k), dtype = torch.float, device = device)
    n_tok = 0

    def hook(mod, inp, out):
        nonlocal n_tok
        x = inp[0].detach().reshape(-1, k).float()
        H.addmm_(x.T, x)
        n_tok += x.shape[0]

    h = linear.register_forward_hook(hook)
    with torch.inference_mode():
        for i in range(cal.shape[0]):
            model(cal[i : i + 1].to(device))
    h.remove()
    return H / n_tok


def out_scale_hessian(sv_r, sigma, device, alpha = 1.0):
    """`H_O' = B (D_sv + damp) B`, damped in the eigenbasis.

    `B` is orthogonal, so `H_O'`'s eigenvalues are exactly `sv²` and the damping has to
    be added to *that* diagonal, before the transform. Handing `prep_hessian` an identity
    and letting it damp would scale the whole matrix by `(1 + sigma)` instead, which is
    not damping at all -- it changes nothing about the conditioning.
    """
    n = sv_r.numel()
    # alpha interpolates the output metric: 1 is the true one, 0 is the identity EXL3
    # rounds with today, negative over-corrects past it. At alpha = 0 this is exactly I,
    # so `block_ldl` returns L_O = 0 and the arm must reproduce `ldlq` bit-for-bit --
    # a built-in check that the knob is wired to the thing it claims to be.
    d = sv_r.reshape(-1).square().pow(alpha)
    H = torch.diag(d)
    ones = torch.ones(n, device = device)
    return prep_hessian(H, ones, had_n, {"sigma_reg": sigma}, sigma = sigma)


def transformed(H, sign, blk, sigma):
    """`prep_hessian` without the block-LDL -- the transformed Hessian itself, for
    scoring the objectives rather than for rounding."""
    H = H.clone()
    H.diagonal().add_(sigma * H.diagonal().mean())
    s = sign.reshape(-1, 1)
    H *= s.T
    blockwise_preapply_had_r_(H, blk)
    H *= s
    blockwise_preapply_had_l_(H, blk)
    return H


def round_arms(linear, H_act, K, sigma_o, alphas, seed, device, arms_wanted):
    """Round one layer every way. `ldlq` and `ldlq+ho` share one `regularize()` call,
    so the only difference between those two is the rounding."""
    k, m = linear.in_features, linear.out_features
    torch.manual_seed(seed)
    W = linear.weight.data.T.contiguous().float()          # (k, n) = (in, out)
    su0 = (torch.randn(k, device = device).sign() + 1e-5).sign().float().unsqueeze(1)
    sv0 = (torch.randn(m, device = device).sign() + 1e-5).sign().float().unsqueeze(0)
    H_diag = H_act.diagonal().clone()

    recon, info, obj = {}, {}, {}
    for os_on in (True, False):
        if os_on and not ({"ldlq", "ho"} & set(arms_wanted)): continue
        if not os_on and "ldlq-noos" not in arms_wanted: continue
        qa = {"K": K, "devices": [0], "buf_size_k": 128,
              "sigma_reg": 0.025, "apply_out_scales": os_on}
        L_act = prep_hessian(H_act, su0, had_k, qa)
        H_act_q = transformed(H_act, su0, had_k, qa["sigma_reg"])
        weight_r = W.clone()
        _, weight_r, g_scale, su_r, sv_r = regularize(
            weight_r, su0.clone(), sv0.clone(), qa, False, H_diag, None, q_fallback = False)

        def finish(q):
            q = preapply_had_l(q, had_k) * su_r
            q = preapply_had_r(q, had_n) * sv_r
            return q.T.contiguous().to(torch.bfloat16)

        if os_on:
            a = sv_r.abs()
            ev = a.reshape(-1).square()
            info["sv_std"] = (a.std() / a.mean()).item()
            info["sv_maxmin"] = (a.max() / a.min().clamp(min = 1e-30)).item()
            info["eff_rank"] = 100 * (ev.sum() ** 2 / ev.square().sum()).item() / m
            todo = [("ldlq", None)]
            for a in alphas:
                todo.append((f"ho/a{a:g}",
                             out_scale_hessian(sv_r, sigma_o[0], device, a)))
        else:
            todo = [("ldlq-noos", None)]

        # the *matrix*, not the LDL factor `out_scale_hessian` returns
        H_O_full = (transformed(torch.diag(sv_r.reshape(-1).square()),
                                torch.ones(m, device = device), had_n, 0.0)
                    if os_on else None)
        for name, L_O in todo:
            if name.split("/")[0] not in [a.split("/")[0] for a in arms_wanted]: continue
            st = {}
            wq, _ = yaqa_round(weight_r.clone(), L_act, L_O, qa, stats = st)
            recon[name] = finish(wq)
            info[name] = (st.get("fed_sq", 0) / max(st.get("fed_n", 1), 1)) ** 0.5
            if os_on:
                # Which objective did each arm actually move? `obj_true` is the loss the
                # layer really incurs, `obj_ldlq` is the one `ldlq()` believes it is
                # minimizing. If `ldlq+ho` does not lower `obj_true`, the fix is broken
                # and no KL number from it means anything.
                D = weight_r - wq
                HI = H_act_q
                obj[name] = (
                    torch.einsum("kn,nm,jm,kj->", D, H_O_full, D, HI).item(),
                    torch.einsum("kn,jn,kj->", D, D, HI).item(),
                    D.square().sum().item())
    return recon, info, obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--layers", type = int, nargs = "+", default = [8])
    ap.add_argument("--projs", nargs = "+", default = ["self_attn.q_proj"])
    ap.add_argument("--bits", type = int, nargs = "+", default = [3])
    ap.add_argument("--cal-seqs", type = int, default = 128)
    ap.add_argument("--eval-seqs", type = int, default = 32)
    ap.add_argument("--ctx", type = int, default = 2048)
    ap.add_argument("--cal-file", default = None)
    ap.add_argument("--eval-source", nargs = "+", default = ["in-domain"],
                    choices = ["in-domain", "code", "literary"])
    ap.add_argument("--arms", nargs = "+", default = ["ldlq", "ho", "ldlq-noos"])
    ap.add_argument("--alpha", type = float, nargs = "+", default = [1.0],
                    help = "output-metric exponent: H_O = B D_sv^(2a) B. 1 is the true "
                           "metric, 0 is the identity EXL3 rounds with today")
    ap.add_argument("--sigma-o", type = float, nargs = "+", default = [0.001])
    ap.add_argument("--attn", default = "eager")
    ap.add_argument("--seed", type = int, default = 7)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype = torch.bfloat16, attn_implementation = args.attn).to(device)
    model.eval()
    vocab = model.config.vocab_size

    cal, evs = build_rows(tok, args.cal_seqs, args.eval_seqs, args.ctx,
                          args.eval_source, args.cal_file)
    print(f" -- {args.model}")
    print(f" -- calibration {tuple(cal.shape)}, eval "
          + ", ".join(f"{k}{tuple(v.shape)}" for k, v in evs.items())
          + f", K={args.bits}, sigma_o={args.sigma_o}, alpha={args.alpha}")

    results = {}
    for layer in args.layers:
        for proj in args.projs:
            tag = f"L{layer}.{proj}"
            linear = model.get_submodule(f"model.layers.{layer}.{proj}")
            print(f"\n == {tag}: in={linear.in_features} out={linear.out_features}", flush = True)
            t0 = time.time()
            H_act = collect_act(model, linear, cal, device)
            # The converter's own heuristic, for reference -- it is bypassed by the
            # --out_scales default of "always", but says whether "auto" would agree.
            d = H_act.diagonal().sqrt().sort(descending = True).values
            skew = (d[: d.numel() // 50].sum() / d.sum()).item()
            print(f"    input skew {skew:.4f} -> auto would say "
                  f"{'ON' if skew < 0.15 else 'OFF'}   ({time.time() - t0:.0f}s)", flush = True)

            results[tag] = {}
            for K in args.bits:
                recon, info, obj = round_arms(linear, H_act, K, args.sigma_o,
                                              args.alpha, args.seed, device, args.arms)
                if "sv_std" in info:
                    print(f"    sv std/mean {info['sv_std']:.3f}  max/min "
                          f"{info['sv_maxmin']:.1f}  H_O eff.rank {info['eff_rank']:.1f}%")
                if obj:
                    b = obj["ldlq"]
                    print(f"    {'arm':14s}{'tr(D H_O Dt H_I)':>20s}{'tr(D I Dt H_I)':>20s}{'|D|^2':>16s}")
                    for nm, v in obj.items():
                        print(f"    {nm:14s}" + "".join(
                            f"{x:12.5e} {100 * (x / b[i] - 1):+6.1f}%" for i, x in enumerate(v)))
                # Per-output-channel error profile. `apply_out_scales` divides the
                # weight by its per-channel RMS, so rounding with H_O = I in that space
                # minimizes *relative* per-channel error; restoring H_O reverts toward
                # uniform *absolute* error. slope = d log(rel err) / d log(ocs), so 0
                # means uniform relative error and -1 means uniform absolute error.
                Worig = linear.weight.data.float()
                ocs_l = block_rms(Worig.T.contiguous(), dim = 0, keepdim = True).reshape(-1)
                ocs_l = (ocs_l / ocs_l.mean()).clamp(min = 1e-12).log()
                ocs_l -= ocs_l.mean()
                for nm in list(recon):
                    d = (recon[nm].float() - Worig)
                    rel = (d.square().mean(1).sqrt() /
                           Worig.square().mean(1).sqrt().clamp(min = 1e-12))
                    y = rel.clamp(min = 1e-30).log(); y = y - y.mean()
                    slope = (ocs_l @ y / ocs_l.square().sum()).item()
                    print(f"    profile {nm:14s} slope {slope:+.3f}   "
                          f"rel-err spread(std of log) {y.std().item():.3f}")
                recon["ldlq-dup"] = recon["ldlq"].clone()
                results[tag][K] = {}
                for src, ev in evs.items():
                    per_seq, counts = kl_vs_ref(model, linear, recon, ev, device, vocab)
                    kls = pooled(per_seq, counts)
                    base = kls["ldlq"]
                    for name in recon:
                        lo, hi = boot_ci(per_seq, counts, name)
                        fed = f"  fedRMS {info[name]:.3f}" if isinstance(info.get(name), float) else ""
                        print(f"    K={K} {src:10s} {name:14s} KL {kls[name]:.6f}  "
                              f"{100 * (kls[name] / base - 1):+7.1f}%  "
                              f"[{100 * lo:+6.1f}, {100 * hi:+6.1f}]{fed}", flush = True)
                    results[tag][K][src] = kls
                del recon
                torch.cuda.empty_cache()

    for K in args.bits:
        for src in args.eval_source:
            names = list(next(iter(results.values()))[K][src].keys())
            print(f"\n K={K}  eval={src}\n{'':26s}" + "".join(f"{n:>14s}" for n in names))
            for tag, per_k in results.items():
                kls = per_k[K][src]
                base = kls["ldlq"]
                print(f"{tag:26s}" + "".join(
                    f"{100 * (kls[n] / base - 1):+13.1f}%" for n in names))


if __name__ == "__main__":
    main()
