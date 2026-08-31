#!/usr/bin/env python3
"""Does YAQA's measured `H_O` agree with `apply_out_scales`?

`outscales.py` finds that restoring the output metric EXL3 drops -- `H_O' = B D_sv B`
-- makes KL *worse*, and that KL tracks the per-output-channel error profile: uniform
*relative* error (what out-scales plus `H_O = I` produces) beats uniform *absolute*
error. That is an empirical claim about downstream sensitivity, and Sketch B measures
that sensitivity directly, so it can be checked rather than assumed.

The full quantization-space metric under YAQA *and* out-scales is

    B D_sv H_O_yaqa D_sv B

so the `D_sv` factor is only harmful if `H_O_yaqa` roughly cancels it, i.e. if measured
output sensitivity falls like `ocs`. This fits `log diag(H_O_yaqa)` against `log ocs`
and reports the slope. A slope near -2 means YAQA independently endorses what
`apply_out_scales` already does, and the composed metric is close to the identity EXL3
happens to round with. A slope near 0 would mean the two corrections are independent and
`outscales.py`'s result needs a different explanation.
"""

import argparse, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "deps", "exllamav3"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from exllamav3.modules.quant.exl3_lib.quantize import block_rms
from probe import build_rows, collect


def fit(x, y):
    """Least-squares slope and Pearson r of y on x, both already logged."""
    x = x - x.mean(); y = y - y.mean()
    slope = (x @ y / x.square().sum()).item()
    r = (x @ y / (x.norm() * y.norm())).item()
    return slope, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--layers", type = int, nargs = "+", default = [8])
    ap.add_argument("--projs", nargs = "+", default = ["self_attn.q_proj"])
    ap.add_argument("--cal-seqs", type = int, default = 64)
    ap.add_argument("--sketch-seqs", type = int, default = 128)
    ap.add_argument("--ctx", type = int, default = 1024)
    ap.add_argument("--attn", default = "eager")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype = torch.bfloat16, attn_implementation = args.attn).to(device)
    model.eval()
    vocab = model.config.vocab_size
    cal, _ = build_rows(tok, args.cal_seqs, 0, args.ctx, [])

    print(f"{'tensor':26s}{'slope':>9s}{'r':>7s}{'ocs sd':>9s}{'compo sd':>10s}")
    for layer in args.layers:
        for proj in args.projs:
            linear = model.get_submodule(f"model.layers.{layer}.{proj}")
            _, _, H_O = collect(model, linear, cal, device, args.sketch_seqs, vocab)

            W = linear.weight.data.T.contiguous().float()
            ocs = block_rms(W, dim = 0, keepdim = True).reshape(-1)
            ocs = ocs / ocs.mean()
            d = H_O.diagonal().clamp(min = 1e-30)
            slope, r = fit(ocs.log(), d.log())

            # The composed metric's own diagonal: diag(H_O_yaqa) * ocs. If the slope is
            # -2 this is flat, which is the identity EXL3 already rounds with.
            comp = (d * ocs.square())
            comp = comp / comp.mean()
            print(f"L{layer}.{proj:20s}{slope:+9.2f}{r:+7.2f}"
                  f"{ocs.log().std().item():9.3f}{comp.log().std().item():10.3f}",
                  flush = True)


if __name__ == "__main__":
    main()
