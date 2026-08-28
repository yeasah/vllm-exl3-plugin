#!/usr/bin/env python3
"""Does YAQA rounding actually lower the end-to-end KL on our quantizer?

The paper's headline table is Llama + QTIP. This asks the same question on our own
instrument, at the smallest scale that can answer it: quantize *one* linear layer of
a small model, leave everything else bf16, and measure the KL of the whole model
against the original. That is exactly the quantity YAQA claims to minimize, it is
fully deterministic given fixed sign vectors, and it needs no converter changes.

Arms (all share one `regularize()` call, so the *only* difference is the rounding):

  ldlq       H_I = E[xᵀx],  H_O = I     -- what EXL3 ships today
  yaqa-o     H_I = E[xᵀx],  H_O = B     -- adds output feedback only
  yaqa       H_I = B,       H_O = B     -- full YAQA, Hessian Sketch B
  ldlq-b     H_I = B,       H_O = I     -- Sketch B's input factor without the wavefront

`ldlq-b` and `yaqa-o` decompose the result: they say whether any win comes from the
output-side feedback, from the better input Hessian, or only from both together.

Sketch B (paper 3.2.2, Appendix A.9) is one round of power iteration on the true
Hessian from an identity start, which collapses to: for each calibration *sequence*,
take the weight gradient G of the Monte-Carlo-sampled cross entropy at the model's
own output, then accumulate H_I += GG and H_O += GG. No custom autograd is
needed at this scale -- it is just `.grad`, once per sequence.
"""

import argparse, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "deps", "exllamav3"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import exllamav3.modules.quant.exl3_lib.quantize as qz
from exllamav3.modules.quant.exl3_lib.quantize import (
    block_ldl, regularize, preapply_had_l, preapply_had_r,
    blockwise_preapply_had_l_, blockwise_preapply_had_r_, had_k, had_n,
)
from rounding import yaqa_round

CAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                       "deps", "exllamav3", "exllamav3", "conversion", "standard_cal_data")


def mc_sample(logits, g, chunk = 256):
    """Draw y_t ~ softmax(logits_t) without materializing a full fp32 softmax.

    The 152k-wide vocabulary makes that tensor the peak allocation of the whole probe
    (1.24 GiB at 2048 tokens), and it exists only to draw one sample per token.
    """
    out = []
    with torch.no_grad():
        for i in range(0, logits.shape[0], chunk):
            p = F.softmax(logits[i : i + chunk].float(), dim = -1)
            out.append(torch.multinomial(p, 1, generator = g).squeeze(-1))
            del p
    return torch.cat(out)


CAL_FILES = ["wiki.utf8", "c4.utf8", "technical.utf8", "code.utf8",
             "multilingual.utf8", "tiny.utf8"]
EVAL_TEXTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                          "deps", "exllamav3", "eval", "eval_texts")


def _tok_files(tokenizer, paths):
    text = ""
    for f in paths:
        with open(f, encoding = "utf8", errors = "replace") as fh:
            text += fh.read() + "\n"
    return tokenizer(text, return_tensors = "pt").input_ids[0]


def build_rows(tokenizer, n_cal, n_eval, ctx, sources, cal_file = None):
    """One calibration set, and evaluation rows from each requested distribution.

    `sources` matters more than it looks. EXL3-SC measured 1.30x *better* than plain on
    its own calibration distribution and 1.17x *worse* on neutral text -- a 1.52x swing in
    standing from the evaluation set alone (docs/qbench.md). YAQA estimates a Fisher from
    calibration data, a far higher-variance statistic than LDLQ's E[x^T x], so it has
    strictly more capacity to overfit that distribution. Scoring it only on a held-out
    slice of its own corpus would reproduce exactly the mistake that flattered SC.

      in-domain : disjoint slice of the same mixture -- same distribution
      code      : code.utf8, held out of calibration entirely
      literary  : eval_texts/ (Austen, Doyle, Dick) -- a separate corpus

    Calibration is built once and is identical for every source, including the file
    exclusions any of them imply. Rebuilding it per source would mean the arms were
    calibrated differently, and the comparison across distributions would be worthless.

    `cal_file` replaces the bundled mixture with an external corpus. That is the only way
    to reach the paper's data budget: the bundled text is 942K tokens once `code.utf8` is
    held out, so a 2048-sequence sketch recycles 460 rows 4.5x, which is 4.5x *below* the
    smallest configuration the paper ever reports (A.11, 2K sequences of 2K tokens, all
    unique). Fresh Monte-Carlo labels on a repeated row reduce label-sampling variance and
    do nothing for data-sampling variance.
    """
    if cal_file:
        cal_ids = _tok_files(tokenizer, [cal_file])
    else:
        files = list(CAL_FILES)
        if "code" in sources:
            files = [f for f in files if f != "code.utf8"]
        cal_ids = _tok_files(tokenizer, [os.path.join(CAL_DIR, f) for f in files])

    need = n_cal * ctx + (n_eval * ctx if "in-domain" in sources else 0)
    assert cal_ids.numel() >= need, \
        f"calibration corpus has {cal_ids.numel()} tokens, need {need}"
    cal = cal_ids[: n_cal * ctx].view(n_cal, ctx)

    evs = {}
    for src in sources:
        if src == "in-domain":
            ids = cal_ids[n_cal * ctx : (n_cal + n_eval) * ctx]
        elif src == "code":
            ids = _tok_files(tokenizer, [os.path.join(CAL_DIR, "code.utf8")])
        elif src == "literary":
            ids = _tok_files(tokenizer, sorted(
                os.path.join(EVAL_TEXTS, f) for f in os.listdir(EVAL_TEXTS)
                if f.endswith(".txt")))
        else:
            raise ValueError(src)
        have = ids.numel() // ctx
        assert have >= 1, f"eval source '{src}' has only {ids.numel()} tokens"
        n = min(have, n_eval)
        if n < n_eval:
            print(f" !! eval source '{src}' yields {n} rows, not {n_eval}")
        evs[src] = ids[: n * ctx].view(n, ctx)
    return cal, evs


def collect(model, linear, cal, device, sketch_seqs, vocab):
    """Activation Hessian E[xx], and Sketch B's H_I / H_O."""
    k = linear.in_features
    m = linear.out_features
    H_act = torch.zeros((k, k), dtype = torch.float, device = device)
    n_tok = 0

    def hook(mod, inp, out):
        nonlocal n_tok
        x = inp[0].detach().reshape(-1, k).float()
        H_act.addmm_(x.T, x)
        n_tok += x.shape[0]

    h = linear.register_forward_hook(hook)
    with torch.inference_mode():
        for i in range(cal.shape[0]):
            model(cal[i : i + 1].to(device))
    h.remove()
    H_act /= n_tok

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
    t0 = time.time()
    for i in range(sketch_seqs):
        r = i % cal.shape[0]
        ids = cal[r : r + 1].to(device)
        logits = model(ids).logits.float().reshape(-1, vocab)
        # Real Fisher: cross entropy against a Monte-Carlo sample from the model's own
        # output distribution, not the next-token label (that is the *empirical* Fisher,
        # which the paper is explicit is a different matrix).
        y = mc_sample(logits, g)
        loss = F.cross_entropy(logits, y, reduction = "sum")
        loss.backward()
        G = linear.weight.grad.detach().float()
        H_I.addmm_(G.T, G)
        H_O.addmm_(G, G.T)
        linear.weight.grad = None
        del logits, loss, G
        if (i + 1) % 64 == 0:
            print(f"    sketch B {i + 1}/{sketch_seqs}  ({time.time() - t0:.0f}s)", flush = True)
    linear.weight.requires_grad_(False)
    model.eval()
    H_I /= sketch_seqs
    H_O /= sketch_seqs
    return H_act, H_I, H_O


def incoherence(H):
    """mu = max|Q_ij| * sqrt(n) for H = Q L Q^T. Theorem 3.4's advantage carries
    mu_I^2 mu_O^2, so this is the quantity EXL3's block-diagonal Hadamard controls
    less well than a full-dimension RHT would."""
    n = H.shape[0]
    _, Q = torch.linalg.eigh(H.double())
    return (Q.abs().max() * (n ** 0.5)).item()


def prep_hessian(H, sign, blk, qa, sigma = None, report = None):
    """Damp and incoherence-process a Hessian, then block-LDL it.

    Mirrors `finalize_capture_H()` exactly, including its convention for the order of
    sign flips and blockwise Hadamard. Applied to H_O with (sv, had_n) it is the
    output-side analogue: the loss in quantization space is tr(D_q H_O' D_q H_I')
    with H_O' = H_n D_sv H_O D_sv H_n.
    """
    H = H.clone()
    if sigma is None:
        sigma = qa.get("sigma_reg", 0.025)
    H.diagonal().add_(sigma * H.diagonal().mean())
    s = sign.reshape(-1, 1)
    H *= s.T
    blockwise_preapply_had_r_(H, blk)
    H *= s
    blockwise_preapply_had_l_(H, blk)
    if report:
        print(f"    mu({report}) = {incoherence(H):.2f}   (n = {H.shape[0]})", flush = True)
    L, _ = block_ldl(H, 16, qa, False)
    dr = torch.arange(H.shape[0])
    L[dr, dr] = 0
    return L


@torch.inference_mode()
def kl_vs_ref(model, linear, weights, eval_rows, device, vocab):
    """Mean per-token KL(original || arm) with only this layer's weight swapped."""
    orig = linear.weight.data.clone()
    per_seq = {name: [] for name in weights}
    counts = []
    for i in range(eval_rows.shape[0]):
        ids = eval_rows[i : i + 1].to(device)
        linear.weight.data.copy_(orig)
        ref_lp = F.log_softmax(model(ids).logits.float().reshape(-1, vocab), dim = -1)
        ref_p = ref_lp.exp()
        for name, w in weights.items():
            linear.weight.data.copy_(w)
            lp = F.log_softmax(model(ids).logits.float().reshape(-1, vocab), dim = -1)
            per_seq[name].append((ref_p * (ref_lp - lp)).sum().item())
            del lp
        counts.append(ref_p.shape[0])
        del ref_p, ref_lp
    linear.weight.data.copy_(orig)
    return per_seq, counts


def pooled(per_seq, counts):
    n = sum(counts)
    return {k: sum(v) / n for k, v in per_seq.items()}


def boot_ci(per_seq, counts, name, base = "ldlq", iters = 4000, seed = 0):
    """Bootstrap CI for (arm/base - 1), resampling *sequences*.

    Per-token KL is heavy-tailed and the pooled mean moved 4x between two adjacent eval
    slices, so a point estimate with no interval is not a measurement. Resampling the same
    sequences costs nothing extra -- the forward passes already happened.
    """
    import random
    r = random.Random(seed)
    a, b, c = per_seq[name], per_seq[base], counts
    n = len(c)
    out = []
    for _ in range(iters):
        idx = [r.randrange(n) for _ in range(n)]
        sa = sum(a[i] for i in idx)
        sb = sum(b[i] for i in idx)
        if sb > 0:
            out.append(sa / sb - 1.0)
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def round_arms(linear, H_act, H_I_B, H_O_B, K, sigma_o, seed, device, full_had_out):
    """Round one layer every way, sharing a single transform.

    Returns (recon, arms) where `recon` maps arm name -> dequantized weight in the
    model's own (out, in) bf16 layout, ready to splice. Every arm shares one
    `regularize()` call and one pair of sign vectors, so the only difference between
    them is the rounding.

    Factored out so `probe.py` and `secondorder.py` round identically: if they drifted,
    the perturbation one script analyses would not be the one the other measured.
    """
    k, m = linear.in_features, linear.out_features
    qa = {"K": K, "devices": [0], "buf_size_k": 128,
          "sigma_reg": 0.025, "apply_out_scales": False}
    hn = m if full_had_out else had_n
    qz.had_n = hn

    torch.manual_seed(seed)
    W = linear.weight.data.T.contiguous().float()   # (k, n) = (in, out), EXL3 layout
    su = (torch.randn(k, device = device).sign() + 1e-5).sign().float().unsqueeze(1)
    sv = (torch.randn(m, device = device).sign() + 1e-5).sign().float().unsqueeze(0)
    H_diag = H_act.diagonal().clone()

    L_act = prep_hessian(H_act, su, had_k, qa)
    L_I_B = prep_hessian(H_I_B, su, had_k, qa)
    L_O = {sg: prep_hessian(H_O_B, sv, hn, qa, sigma = sg) for sg in sigma_o}

    weight_r = W.clone()
    _, weight_r, g_scale, su_r, sv_r = regularize(
        weight_r, su.clone(), sv.clone(), qa, False, H_diag, None, q_fallback = False)

    arms = {"ldlq": (L_act, None), "ldlq-b": (L_I_B, None)}
    for sg in sigma_o:
        arms[f"yaqa-o/{sg:g}"] = (L_act, L_O[sg])
        arms[f"yaqa/{sg:g}"] = (L_I_B, L_O[sg])

    def finish(q):
        q = preapply_had_l(q, had_k) * su_r
        q = preapply_had_r(q, hn) * sv_r
        return q.T.contiguous().to(torch.bfloat16)

    recon = {}
    for name, (LI, LO) in list(arms.items()):
        wq, _ = yaqa_round(weight_r.clone(), LI, LO, qa)
        recon[name] = finish(wq)
    return recon, list(arms)


def layer_hessians(model, linear, cal, device, sketch_seqs, vocab, quiet = False):
    """Activation Hessian and Sketch B, with a note on H_O's spectral concentration."""
    m = linear.out_features
    H_act, H_I_B, H_O_B = collect(model, linear, cal, device, sketch_seqs, vocab)
    if not quiet:
        evo = torch.linalg.eigvalsh(H_O_B.double()).flip(0).clamp(min = 1e-30)
        cf = evo.cumsum(0) / evo.sum()
        r50 = int((cf < 0.50).sum().item()) + 1
        r90 = int((cf < 0.90).sum().item()) + 1
        print(f"    H_O {m}x{m}: 50%/90% of trace in top {r50} / {r90} eigs "
              f"({100 * r50 / m:.0f}% / {100 * r90 / m:.0f}%)", flush = True)
    return H_act, H_I_B, H_O_B


def run_one(model, linear, cal, evs, args, sigma_o, device, vocab, tag):
    """One layer: collect Hessians once, then round and score at every bitrate.

    Collecting Sketch B is ~18 minutes per layer at the paper's budget and does not
    depend on the bitrate, while rounding and the KL pass are minutes. So K is swept
    inside one collection rather than re-paying it per bitrate.
    """
    k, m = linear.in_features, linear.out_features
    print(f"\n == {tag}: in={k} out={m}", flush = True)
    H_act, H_I_B, H_O_B = layer_hessians(model, linear, cal, device, args.sketch_seqs, vocab)

    out = {}
    for K in args.bits:
        recon, names = round_arms(linear, H_act, H_I_B, H_O_B, K, sigma_o,
                                  args.seed, device, args.full_had_out)
        recon["ldlq-dup"] = recon["ldlq"].clone()
        # The sketch does not depend on the evaluation set, so every distribution is
        # scored from one collection -- which is also the only way the in-domain and
        # neutral numbers are strictly comparable.
        out[K] = {}
        for src, ev in evs.items():
            per_seq, counts = kl_vs_ref(model, linear, recon, ev, device, vocab)
            kls = pooled(per_seq, counts)
            base = kls["ldlq"]
            for name in names + ["ldlq-dup"]:
                lo, hi = boot_ci(per_seq, counts, name)
                print(f"    K={K} {src:10s} {name:14s} KL {kls[name]:.6f}  "
                      f"{100 * (kls[name] / base - 1):+7.1f}%  "
                      f"[{100 * lo:+6.1f}, {100 * hi:+6.1f}]", flush = True)
            out[K][src] = kls
        del recon
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--layers", type = int, nargs = "+", default = [14])
    ap.add_argument("--projs", nargs = "+", default = ["mlp.down_proj"])
    ap.add_argument("--bits", type = int, nargs = "+", default = [3])
    ap.add_argument("--cal-seqs", type = int, default = 128)
    ap.add_argument("--sketch-seqs", type = int, default = 512)
    ap.add_argument("--eval-seqs", type = int, default = 32)
    ap.add_argument("--ctx", type = int, default = 1024)
    ap.add_argument("--cal-file", default = None,
                    help = "external calibration corpus; needed to reach the paper's "
                           "unique-token budget (the bundled mix cannot)")
    ap.add_argument("--eval-source", nargs = "+", default = ["in-domain"],
                    choices = ["in-domain", "code", "literary"],
                    help = "distribution to score on; see get_rows()")
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
    ap.add_argument("--seed", type = int, default = 7)
    ap.add_argument("--full-had-out", action = "store_true",
                    help = "full-width Hadamard on the output side instead of EXL3's "
                           "block-128; breaks TP sharding, isolates the incoherence effect")
    ap.add_argument("--rescale", action = "store_true",
                    help = "add a +rs arm that re-matches the codebook scale to the "
                           "distribution YAQA actually feeds the quantizer")
    ap.add_argument("--sigma-o", type = float, nargs = "+", default = [0.001],
                    help = "diagonal damping for H_O; EXL3 uses 0.025 for H_I, the paper ~1e-4")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

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

    cal, evs = build_rows(tok, args.cal_seqs, args.eval_seqs, args.ctx,
                          args.eval_source, args.cal_file)
    print(f" -- {args.model}")
    print(f" -- calibration {tuple(cal.shape)}, eval "
          + ", ".join(f"{k}{tuple(v.shape)}" for k, v in evs.items()) + ", "
          f"sketch {args.sketch_seqs} seqs, K={args.bits}, eval={args.eval_source}")
    reuse = args.sketch_seqs / cal.shape[0]
    print(f" -- unique calibration tokens: {cal.numel():,}"
          + (f"  (each row seen {reuse:.1f}x)" if reuse > 1.01 else "  (no reuse)")
          + (f"  corpus: {os.path.basename(args.cal_file)}" if args.cal_file else ""))

    results = {}
    for layer in args.layers:
        for proj in args.projs:
            tag = f"L{layer}.{proj}"
            linear = model.get_submodule(f"model.layers.{layer}.{proj}")
            results[tag] = run_one(model, linear, cal, evs, args,
                                   args.sigma_o, device, vocab, tag)

    for K in args.bits:
        for src in args.eval_source:
            names = list(next(iter(results.values()))[K][src].keys())
            print(f"\n K={K}  eval={src}\n{'':24s}" + "".join(f"{n:>15s}" for n in names))
            for tag, per_k in results.items():
                kls = per_k[K][src]
                base = kls["ldlq"]
                print(f"{tag:24s}" + "".join(
                    f"{100 * (kls[n] / base - 1):+14.1f}%" for n in names))


if __name__ == "__main__":
    main()
