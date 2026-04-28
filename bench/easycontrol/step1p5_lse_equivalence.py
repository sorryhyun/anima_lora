#!/usr/bin/env python
"""EasyControl Phase 1.5 — LSE-decomposed extended self-attn equivalence.

Question
--------
Phase 1's patched ``self_attn.forward`` adds a per-block scalar additive bias
``b_cond`` on the cond positions. None of the alt attention backends accept a
per-key additive bias on a key subset, so Phase 1 forces ``torch SDPA`` with an
explicit ``attn_mask``. With a non-``None`` mask, torch SDPA falls off the
flash and mem-efficient backends and dispatches to the **math kernel**, which
materializes the full ``[B, n_h, S_t, S_t + S_c]`` attention matrix —
~1 GB / block at bf16 (often 2 GB at fp32 internal accum). That peak repeats
per block. ``--gradient_checkpointing`` cannot save it (it lives inside one
op). On real hardware this OOMs.

Phase 1.5's fix: decompose the extended-key softmax into two ordinary
memory-efficient flash-attention calls plus an LSE arithmetic combine. Per
block::

    out_t, lse_t = flash_attn(Q, K_t, V_t, return_lse=True)
    out_c, lse_c = flash_attn(Q, K_c, V_c, return_lse=True)
    lse_c_adj    = lse_c + b_cond
    joint_lse    = logaddexp(lse_t, lse_c_adj)
    α            = exp(lse_t - joint_lse)        # ∈ [0,1]
    β            = exp(lse_c_adj - joint_lse)    # = 1 - α
    out          = α · out_t + β · out_c

Forward equivalence falls out of standard softmax algebra. Backward
equivalence is more subtle: FA2's stock ``FlashAttnFunc.backward`` discards
the gradient through ``softmax_lse``, so a "two FA + Python combine" using
``flash_attn_func(..., return_attn_probs=True)`` drops the *path-2* gradient
through α/β and gets q/k_t/k_c gradients wrong (error scales as α·β).

``_ExtendedSelfAttnLSEFunc`` works around this by calling
``_wrapped_flash_attn_forward / _backward`` directly and feeding ``joint_lse``
(target tile) and ``joint_lse - b`` (cond tile) into the per-tile FA backward
— the joint-softmax derivative then decomposes correctly across tiles.

This bench verifies::

  1. Forward output matches the masked-SDPA reference within fp32 ulp.
  2. Backward gradients (dq, dk_t, dv_t, dk_c, dv_c, db_cond) match the
     masked-SDPA reference within fp32 ulp.

A naive "two FA + Python combine" path is included for contrast — its
forward matches but its dq/dk gradients drift by α·β·‖out_t − out_c‖.

Usage
-----
    uv run python bench/easycontrol/step1p5_lse_equivalence.py
    uv run python bench/easycontrol/step1p5_lse_equivalence.py --b_cond 0.0
    uv run python bench/easycontrol/step1p5_lse_equivalence.py --dtype bf16
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from library.log import setup_logging  # noqa: E402
from networks import attention_dispatch as anima_attention  # noqa: E402
from networks.methods.easycontrol import _ExtendedSelfAttnLSEFunc  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- reference paths


def masked_extended_sdpa(q, k_t, v_t, k_c, v_c, b_cond, softmax_scale):
    """Reference masked-SDPA path (Phase 1's math-kernel-dispatched path).

    q, k_*, v_* in BLHD; b_cond is a 0-d tensor (or scalar). Returns BLHD.
    """
    B, S_t, H, D = q.shape
    S_c = k_c.shape[1]
    k_ext = torch.cat([k_t, k_c], dim=1)
    v_ext = torch.cat([v_t, v_c], dim=1)
    q_s = q.transpose(1, 2)  # [B, H, S_t, D]
    k_s = k_ext.transpose(1, 2)
    v_s = v_ext.transpose(1, 2)
    b = b_cond.to(q_s.dtype)
    target_zeros = torch.zeros(S_t, device=q.device, dtype=q_s.dtype)
    cond_b = b.expand(S_c)
    attn_bias = torch.cat([target_zeros, cond_b], dim=0)
    attn_mask = attn_bias.view(1, 1, 1, S_t + S_c)
    out = F.scaled_dot_product_attention(
        q_s, k_s, v_s, attn_mask=attn_mask, scale=softmax_scale
    )
    return out.transpose(1, 2).contiguous()  # [B, S_t, H, D]


def naive_lse_combine(q, k_t, v_t, k_c, v_c, b_cond, softmax_scale):
    """The naive 'two FA + Python combine via flash_attn_func' path.

    Forward is bit-identical to ``_ExtendedSelfAttnLSEFunc``. Backward DROPS
    the path-2 gradient through α/β because FA2's ``FlashAttnFunc.backward``
    discards the upstream gradient on softmax_lse. Included as a contrast —
    its dq/dk gradients should diverge from the reference once b_cond grows
    past ~-10.
    """
    out_t, lse_t = anima_attention.attention_with_lse(
        q,
        k_t,
        v_t,
        attn_mode="flash",
        softmax_scale=softmax_scale,
    )
    out_c, lse_c = anima_attention.attention_with_lse(
        q,
        k_c,
        v_c,
        attn_mode="flash",
        softmax_scale=softmax_scale,
    )
    lse_c_adj = lse_c + b_cond.to(lse_c.dtype)
    joint_lse = torch.logaddexp(lse_t, lse_c_adj)
    alpha = (lse_t - joint_lse).exp()
    beta = (lse_c_adj - joint_lse).exp()
    alpha_bd = alpha.transpose(1, 2).unsqueeze(-1).to(out_t.dtype)
    beta_bd = beta.transpose(1, 2).unsqueeze(-1).to(out_c.dtype)
    return alpha_bd * out_t + beta_bd * out_c


def custom_lse_func(q, k_t, v_t, k_c, v_c, b_cond, softmax_scale):
    """Phase 1.5 path — custom autograd Function with correct backward."""
    return _ExtendedSelfAttnLSEFunc.apply(
        q,
        k_t,
        v_t,
        k_c,
        v_c,
        b_cond,
        softmax_scale,
    )


# ----------------------------------------------------------------- helpers


def make_inputs(B, H, S_t, S_c, D, dtype, device, gen, requires_grad=True):
    def randn_norm(shape):
        x = torch.randn(*shape, generator=gen, device=device, dtype=torch.float32)
        # RMSNorm-like unit magnitude per head per position (post q_norm/k_norm).
        x = x / x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        return x.to(dtype)

    q = randn_norm((B, S_t, H, D))
    k_t = randn_norm((B, S_t, H, D))
    v_t = (
        torch.randn(B, S_t, H, D, generator=gen, device=device, dtype=torch.float32)
        * 0.5
    ).to(dtype)
    k_c = randn_norm((B, S_c, H, D))
    v_c = (
        torch.randn(B, S_c, H, D, generator=gen, device=device, dtype=torch.float32)
        * 0.5
    ).to(dtype)
    if requires_grad:
        for t in (q, k_t, v_t, k_c, v_c):
            t.requires_grad_(True)
    return q, k_t, v_t, k_c, v_c


def rel_l2(a, b):
    a = a.float()
    b = b.float()
    n = (a - b).norm()
    d = b.norm().clamp_min(1e-12)
    return (n / d).item()


def abs_max(a, b):
    return (a.float() - b.float()).abs().max().item()


# ----------------------------------------------------------------- the sweep


def compare_one(name, ref_out, ref_grads, test_out, test_grads):
    rows = []
    rows.append((f"{name}/out", rel_l2(test_out, ref_out), abs_max(test_out, ref_out)))
    grad_names = ["dq", "dk_t", "dv_t", "dk_c", "dv_c", "db_cond"]
    for gname, ref_g, test_g in zip(grad_names, ref_grads, test_grads):
        if ref_g is None and test_g is None:
            continue
        rows.append((f"{name}/{gname}", rel_l2(test_g, ref_g), abs_max(test_g, ref_g)))
    return rows


def run_one(B, H, S_t, S_c, D, dtype, device, gen, b_cond_value, seed):
    softmax_scale = D**-0.5

    # Reference: masked SDPA.
    q, k_t, v_t, k_c, v_c = make_inputs(B, H, S_t, S_c, D, dtype, device, gen)
    b_cond = torch.tensor(
        b_cond_value, device=device, dtype=torch.float32, requires_grad=True
    )
    ref_out = masked_extended_sdpa(q, k_t, v_t, k_c, v_c, b_cond, softmax_scale)
    # Use a deterministic upstream gradient so all paths see the same dout.
    g_gen = torch.Generator(device=device).manual_seed(seed + 9999)
    dout = (
        torch.randn(*ref_out.shape, generator=g_gen, device=device, dtype=ref_out.dtype)
        * 0.1
    )
    ref_out.backward(dout)
    ref_grads = (
        q.grad.detach().clone(),
        k_t.grad.detach().clone(),
        v_t.grad.detach().clone(),
        k_c.grad.detach().clone(),
        v_c.grad.detach().clone(),
        b_cond.grad.detach().clone(),
    )

    # Custom LSE Function (the Phase 1.5 path).
    gen_c = torch.Generator(device=device).manual_seed(seed)
    qc, k_tc, v_tc, k_cc, v_cc = make_inputs(B, H, S_t, S_c, D, dtype, device, gen_c)
    b_cond_c = torch.tensor(
        b_cond_value, device=device, dtype=torch.float32, requires_grad=True
    )
    custom_out = custom_lse_func(qc, k_tc, v_tc, k_cc, v_cc, b_cond_c, softmax_scale)
    custom_out.backward(dout)
    custom_grads = (
        qc.grad.detach().clone(),
        k_tc.grad.detach().clone(),
        v_tc.grad.detach().clone(),
        k_cc.grad.detach().clone(),
        v_cc.grad.detach().clone(),
        b_cond_c.grad.detach().clone(),
    )

    # Naive LSE combine (forward only — backward will be wrong by design).
    gen_n = torch.Generator(device=device).manual_seed(seed)
    qn, k_tn, v_tn, k_cn, v_cn = make_inputs(B, H, S_t, S_c, D, dtype, device, gen_n)
    b_cond_n = torch.tensor(
        b_cond_value, device=device, dtype=torch.float32, requires_grad=True
    )
    naive_out = naive_lse_combine(qn, k_tn, v_tn, k_cn, v_cn, b_cond_n, softmax_scale)
    naive_out.backward(dout)
    naive_grads = (
        qn.grad.detach().clone(),
        k_tn.grad.detach().clone(),
        v_tn.grad.detach().clone(),
        k_cn.grad.detach().clone(),
        v_cn.grad.detach().clone(),
        b_cond_n.grad.detach().clone(),
    )

    rows = []
    rows.extend(compare_one("custom_lse", ref_out, ref_grads, custom_out, custom_grads))
    rows.extend(compare_one("naive_lse", ref_out, ref_grads, naive_out, naive_grads))
    return rows


def format_table(all_rows):
    header = f"{'comparison':30s} | {'rel_l2':>12s} | {'abs_max':>12s}"
    sep = "-" * len(header)
    lines = [header, sep]
    for name, rel, amax in all_rows:
        lines.append(f"{name:30s} | {rel:12.3e} | {amax:12.3e}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--s_t", type=int, default=4096)
    p.add_argument("--s_c", type=int, default=4096)
    p.add_argument("--n_heads", type=int, default=16)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--n_trials", type=int, default=3)
    p.add_argument(
        "--b_cond",
        type=float,
        default=None,
        help="b_cond value to test. Defaults to a sweep over [-10.0, -3.0, 0.0, 2.0].",
    )
    p.add_argument(
        "--dtype",
        choices=["fp32", "bf16", "fp16"],
        default="bf16",
        help="bf16 by default to match training-time precision.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--strict_threshold",
        type=float,
        default=5e-3,
        help="rel_l2 above which the path is considered to disagree with reference.",
    )
    p.add_argument(
        "--out_json",
        default="bench/easycontrol/results/step1p5_lse_equivalence.json",
    )
    args = p.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.error("CUDA requested but not available")
        sys.exit(1)
    if anima_attention._wrapped_flash_attn_forward is None:
        logger.error("flash-attn not installed — Phase 1.5 path requires FA2")
        sys.exit(1)

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        args.dtype
    ]
    device = torch.device(args.device)

    if args.b_cond is None:
        b_conds = [-10.0, -3.0, 0.0, 2.0]
    else:
        b_conds = [args.b_cond]

    logger.info(
        f"step-1.5 LSE equivalence | S_t={args.s_t} S_c={args.s_c} "
        f"H={args.n_heads} D={args.head_dim} dtype={args.dtype} "
        f"b_cond_sweep={b_conds} device={device}"
    )

    summary = {"config": vars(args), "results": []}
    for b_cond_value in b_conds:
        all_rows = []
        for trial in range(args.n_trials):
            gen = torch.Generator(device=device).manual_seed(args.seed + trial)
            rows = run_one(
                args.batch,
                args.n_heads,
                args.s_t,
                args.s_c,
                args.head_dim,
                dtype,
                device,
                gen,
                b_cond_value,
                args.seed + trial,
            )
            all_rows.append(rows)
        # Aggregate: max over trials per comparison name.
        agg = {}
        for trial_rows in all_rows:
            for name, rel, amax in trial_rows:
                prev = agg.get(name, (0.0, 0.0))
                agg[name] = (max(prev[0], rel), max(prev[1], amax))
        # Preserve order from last trial for printing.
        ordered = [(name, *agg[name]) for name, _, _ in all_rows[-1]]

        print()
        print(f"=== b_cond = {b_cond_value} ===")
        print(format_table(ordered))

        verdict = []
        for name, rel, amax in ordered:
            tier = "PASS" if rel < args.strict_threshold else "FAIL"
            verdict.append(
                {"name": name, "rel_l2_max": rel, "abs_max": amax, "tier": tier}
            )
        summary["results"].append({"b_cond": b_cond_value, "rows": verdict})
        print()
        print("Verdict:")
        for v in verdict:
            print(
                f"  {v['name']:30s}  rel_l2_max = {v['rel_l2_max']:.3e}  →  {v['tier']}"
            )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"saved summary to {out_path}")


if __name__ == "__main__":
    main()
