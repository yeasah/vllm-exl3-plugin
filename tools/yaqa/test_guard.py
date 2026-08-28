#!/usr/bin/env python3
"""Guards for the YAQA rounding kernel.

1. With H_O = I the wavefront must reproduce EXL3's own `ldlq()`. The wavefront is
   a legal reordering of it -- the input-feedback term for tile (i, j) reads only
   column-block j at rows below, and every such tile is on an earlier wavefront --
   so the encoded indices must match exactly. Summation order differs, so the
   reconstructions are compared with a tolerance and the *indices* with none.

2. On synthetic data with a known H_O ⊗ H_I, YAQA must reduce the proxy loss
   tr(Δ H_O Δᵀ H_I) relative to LDLQ. If it does not, the implementation is wrong
   regardless of what any end-to-end number says.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "deps", "exllamav3"))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from exllamav3.modules.quant.exl3_lib.quantize import block_ldl, ldlq
from rounding import yaqa_round


def make_L(H, quant_args):
    L, _ = block_ldl(H.clone(), 16, quant_args, False)
    dr = torch.arange(H.shape[0])
    L[dr, dr] = 0
    return L


def random_psd(n, rank, device, seed):
    g = torch.Generator(device = device).manual_seed(seed)
    A = torch.randn(rank, n, generator = g, device = device, dtype = torch.float)
    H = A.T @ A / rank
    H.diagonal().add_(0.01 * H.diagonal().mean())
    return H


def proxy_loss(D, H_I, H_O):
    # exl3 layout: D is (k, n) = (in, out); loss = tr(D H_O Dᵀ H_I)
    return torch.einsum("kn,nm,jm,kj->", D, H_O, D, H_I).item()


def main():
    dev = torch.device("cuda:0")
    K = 3
    qa = {"K": K, "devices": [0], "buf_size_k": 128}
    k, n = 512, 512
    torch.manual_seed(0)

    W = torch.randn(k, n, device = dev, dtype = torch.float)
    H_I = random_psd(k, 4 * k, dev, 1)
    H_O = random_psd(n, n // 4, dev, 2)          # low rank, as the paper reports for real H_O

    L_I = make_L(H_I, qa)
    L_O = make_L(H_O, qa)

    # --- Guard 1: H_O = I reproduces ldlq ---------------------------------------
    wq_ref, enc_ref = ldlq(W.clone(), L_I.clone(), qa)
    wq_wf, enc_wf = yaqa_round(W.clone(), L_I, None, qa)
    idx_match = torch.equal(enc_ref.to(dev), enc_wf)
    max_dev = (wq_ref.to(dev) - wq_wf).abs().max().item()
    print(f"guard 1  encoded indices identical: {idx_match}   max |Δrecon|: {max_dev:.3e}")
    if not idx_match:
        diff = (enc_ref.to(dev) != enc_wf).sum().item()
        print(f"         !! {diff} / {enc_ref.numel()} index elements differ")

    # --- Guard 2: YAQA lowers the proxy loss it claims to minimize ---------------
    wq_ldlq, _ = yaqa_round(W.clone(), L_I, None, qa)
    wq_yaqa, _ = yaqa_round(W.clone(), L_I, L_O, qa)
    l_ldlq = proxy_loss(W - wq_ldlq, H_I, H_O)
    l_yaqa = proxy_loss(W - wq_yaqa, H_I, H_O)
    # also the loss LDLQ itself targets, which YAQA is allowed to do worse on
    i_ldlq = proxy_loss(W - wq_ldlq, H_I, torch.eye(n, device = dev))
    i_yaqa = proxy_loss(W - wq_yaqa, H_I, torch.eye(n, device = dev))
    print(f"guard 2  tr(Δ H_O Δᵀ H_I):  LDLQ {l_ldlq:.5e}   YAQA {l_yaqa:.5e}   "
          f"({100 * (l_yaqa / l_ldlq - 1):+.1f}%)")
    print(f"         tr(Δ  I  Δᵀ H_I):  LDLQ {i_ldlq:.5e}   YAQA {i_yaqa:.5e}   "
          f"({100 * (i_yaqa / i_ldlq - 1):+.1f}%)")
    print(f"         plain ‖Δ‖²:        LDLQ {(W - wq_ldlq).square().sum().item():.5e}   "
          f"YAQA {(W - wq_yaqa).square().sum().item():.5e}")


def _main():
    main()
    test_transform_consistency()


def test_transform_consistency():
    """Is the transformed H_O the right Hessian for the transformed weight?

    Rounding happens in EXL3's incoherence-processed space, so the Hessians handed to
    the wavefront must be the ones for *that* space. The reconstruction path is

        W_orig = (had_l(W_q) * su) -> had_r -> * sv

    so for the loss tr(D_orig H_O D_orig^T H_I) to equal tr(D_q H_O' D_q^T H_I') the
    transformed Hessians must be H_I' = T(H_I; su, had_k) and H_O' = T(H_O; sv, had_n)
    with T the same sign-flip-then-blockwise-Hadamard used by finalize_capture_H().

    Nothing else in the harness tests this. If it is wrong, YAQA optimizes a scrambled
    objective, which would look like a result that is usually neutral and sometimes
    much worse -- indistinguishable from "the algorithm does not transfer".
    """
    import sys as _s
    _s.path.insert(0, __import__("os").path.dirname(__file__))
    from exllamav3.modules.quant.exl3_lib.quantize import (
        preapply_had_l, preapply_had_r, blockwise_preapply_had_l_,
        blockwise_preapply_had_r_, had_k, had_n,
    )
    dev = torch.device("cuda:0")
    k, n = 256, 384
    torch.manual_seed(3)

    su = (torch.randn(k, device = dev).sign() + 1e-5).sign().float().unsqueeze(1)
    sv = (torch.randn(n, device = dev).sign() + 1e-5).sign().float().unsqueeze(0)
    H_I = random_psd(k, 2 * k, dev, 11)
    H_O = random_psd(n, n // 3, dev, 12)
    D_q = torch.randn(k, n, device = dev, dtype = torch.float)

    # quantization space -> original space, exactly as quantize_exl3() reconstructs
    D_o = preapply_had_l(D_q.clone(), had_k) * su
    D_o = preapply_had_r(D_o, had_n) * sv

    def T(H, sign, blk):
        H = H.clone()
        s = sign.reshape(-1, 1)
        H *= s.T
        blockwise_preapply_had_r_(H, blk)
        H *= s
        blockwise_preapply_had_l_(H, blk)
        return H

    lhs = torch.einsum("kn,nm,jm,kj->", D_o, H_O, D_o, H_I).item()
    rhs = torch.einsum("kn,nm,jm,kj->", D_q, T(H_O, sv, had_n), D_q, T(H_I, su, had_k)).item()
    rel = abs(rhs - lhs) / abs(lhs)
    print(f"guard 3  tr(D H_O Dᵀ H_I): original {lhs:.6e}  transformed {rhs:.6e}  "
          f"rel err {rel:.2e}  {'OK' if rel < 1e-4 else '!! MISMATCH'}")
    return rel < 1e-4


if __name__ == "__main__":
    _main()
