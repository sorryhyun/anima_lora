#!/usr/bin/env python
"""EasyControl proposal — Risk #1 verification: step-0 baseline equivalence.

Question
--------
The proposal extends each ``Block.self_attn`` with condition-side keys/values:

    K = [K_target ; K_cond]   V = [V_target ; V_cond]

and claims that "zero-init V_c is sufficient" for the step-0 forward to match
the no-condition forward. This script tests that claim numerically and ranks
several initialization strategies by how well they hold the equivalence.

Math (worth restating because the answer falls right out of it)
---------------------------------------------------------------
Let ``L_t = QK_t^T / sqrt(d)`` and ``L_c = QK_c^T / sqrt(d)``.

    out_baseline = softmax(L_t) @ V_t
    out_extended = softmax([L_t ; L_c]) @ [V_t ; V_c]
                 = α · out_baseline  +  sm_c @ V_c
        where α = Z_t / (Z_t + Z_c),
              Z_t = sum(exp(L_t), keys),  Z_c = sum(exp(L_c), keys).

Conclusion from the math alone:
* ``V_c = 0`` makes the **second** term zero, but the **first** term is still
  rescaled by ``α``. Equivalence holds *only if* ``α = 1``, which requires
  ``Z_c = 0``, i.e. the cond logits are -∞ (mask) or extremely negative.
* Zero-init K_c (so ``L_c ≈ 0``) does NOT give Z_c = 0; it gives
  ``Z_c = S_c · 1 = S_c``. Then ``α = Z_t / (Z_t + S_c)``, which is < 1 unless
  Z_t >> S_c.

This script confirms that prediction empirically and measures how bad the
rescale is for plausible Anima-scale shapes.

Init strategies tested
----------------------
1. v_zero_only:       V_c = 0,  K_c ~ N(0, σ_init)        (proposal v0)
2. kv_zero:           V_c = 0,  K_c = 0  (cond logits = 0, exp = 1)
3. cond_input_zero:   cond_tokens = 0   (V_c = K_c = 0 by linearity)
4. logit_bias_-10:    cond logits get a learnable bias init to -10 (mass leak ≈ 0)
5. logit_bias_-30:    cond logits bias init to -30 (effectively masked)
6. masked:            cond logits set to -inf (true mask, exact equivalence)
7. v_zero_alpha_compensated:
       V_c = 0 AND we rescale ``out_extended /= α`` post-hoc to undo the leak.
       Demonstrates that the rescale really is the only failure mode.

For each strategy we report ``rel_l2 = ‖out_ext - out_base‖ / ‖out_base‖``
across many random trials, plus the distribution of ``α`` per query position.

Usage
-----
    uv run python bench/active/easycontrol/step0_equivalence.py
    uv run python bench/active/easycontrol/step0_equivalence.py --s_t 4096 --s_c 4096
    uv run python bench/active/easycontrol/step0_equivalence.py --dtype bf16
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- attention math

def sdpa_with_optional_bias(q, k, v, attn_bias=None):
    """Reference SDPA with an optional per-key bias (broadcast over Q rows / heads).

    q, k, v: [B, H, S_q or S_k, D]
    attn_bias: [S_k] or None — added to logits before softmax.
    """
    d = q.shape[-1]
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(d)
    if attn_bias is not None:
        logits = logits + attn_bias
    attn = logits.softmax(dim=-1)
    return attn @ v, attn


def baseline_attention(q, k_t, v_t):
    out, _ = sdpa_with_optional_bias(q, k_t, v_t)
    return out


def extended_attention(q, k_t, v_t, k_c, v_c, cond_logit_bias=None, masked=False):
    """Run the proposed extended self-attention.

    cond_logit_bias: scalar or [S_c] tensor added to the **cond** logits only.
                     Models a learnable additive bias on cond keys.
    masked: if True, cond logits are set to -inf (hard mask, baseline check).
    """
    k_ext = torch.cat([k_t, k_c], dim=-2)
    v_ext = torch.cat([v_t, v_c], dim=-2)
    s_t = k_t.shape[-2]
    s_c = k_c.shape[-2]
    bias = None
    if masked:
        bias = torch.zeros(s_t + s_c, device=q.device, dtype=q.dtype)
        bias[s_t:] = float("-inf")
    elif cond_logit_bias is not None:
        bias = torch.zeros(s_t + s_c, device=q.device, dtype=q.dtype)
        if torch.is_tensor(cond_logit_bias):
            bias[s_t:] = cond_logit_bias.to(q.dtype)
        else:
            bias[s_t:] = float(cond_logit_bias)
    out, attn = sdpa_with_optional_bias(q, k_ext, v_ext, attn_bias=bias)
    return out, attn


def alpha_from_attention(attn, s_t):
    """Recover per-row α = sum of softmax mass on target positions, [B, H, S_q]."""
    return attn[..., :s_t].sum(dim=-1)


# ----------------------------------------------------------------- init helpers

def make_target_qkv(B, H, S_t, D, dtype, device, gen):
    """Realistic target Q/K/V at training init.

    Q, K are post-RMSNorm so unit L2 magnitude per head: norm to unit length.
    V is post-Identity (Anima v_norm = nn.Identity), Gaussian.
    """
    q = torch.randn(B, H, S_t, D, generator=gen, device=device, dtype=dtype)
    k = torch.randn(B, H, S_t, D, generator=gen, device=device, dtype=dtype)
    v = torch.randn(B, H, S_t, D, generator=gen, device=device, dtype=dtype) * 0.5
    # RMSNorm-like unit-magnitude per head per position.
    q = q / q.float().pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt().to(dtype)
    k = k / k.float().pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt().to(dtype)
    return q, k, v


def make_cond_kv(strategy, B, H, S_c, D, q_dtype, device, gen, sigma_init=1.0):
    """Build (k_c, v_c, cond_logit_bias, masked) for a given init strategy.

    Returns tensors with the same dtype as q_dtype.
    """
    cond_logit_bias = None
    masked = False

    if strategy == "v_zero_only":
        # K_c at standard init (unit-norm post k_norm), V_c = 0.
        k_c = torch.randn(B, H, S_c, D, generator=gen, device=device, dtype=q_dtype)
        k_c = k_c / k_c.float().pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt().to(q_dtype)
        v_c = torch.zeros(B, H, S_c, D, device=device, dtype=q_dtype)

    elif strategy == "kv_zero":
        # Both K_c and V_c = 0. (Models zero-init on the cond projection weights.)
        k_c = torch.zeros(B, H, S_c, D, device=device, dtype=q_dtype)
        v_c = torch.zeros(B, H, S_c, D, device=device, dtype=q_dtype)

    elif strategy == "cond_input_zero":
        # cond_tokens = 0 → linear(0) = 0 → both K_c and V_c are 0. Equivalent to
        # kv_zero numerically; included as a separate row because in the actual
        # proposal "zero-init the condition embedder" is a real design lever.
        k_c = torch.zeros(B, H, S_c, D, device=device, dtype=q_dtype)
        v_c = torch.zeros(B, H, S_c, D, device=device, dtype=q_dtype)

    elif strategy == "logit_bias_-10":
        # Random K_c and V_c, but a -10 additive bias on cond logits.
        k_c = torch.randn(B, H, S_c, D, generator=gen, device=device, dtype=q_dtype)
        k_c = k_c / k_c.float().pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt().to(q_dtype)
        v_c = torch.randn(B, H, S_c, D, generator=gen, device=device, dtype=q_dtype) * 0.5
        cond_logit_bias = -10.0

    elif strategy == "logit_bias_-30":
        k_c = torch.randn(B, H, S_c, D, generator=gen, device=device, dtype=q_dtype)
        k_c = k_c / k_c.float().pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt().to(q_dtype)
        v_c = torch.randn(B, H, S_c, D, generator=gen, device=device, dtype=q_dtype) * 0.5
        cond_logit_bias = -30.0

    elif strategy == "masked":
        # Hard mask — cond positions cannot receive softmax mass.
        k_c = torch.randn(B, H, S_c, D, generator=gen, device=device, dtype=q_dtype)
        v_c = torch.randn(B, H, S_c, D, generator=gen, device=device, dtype=q_dtype)
        masked = True

    elif strategy == "v_zero_alpha_compensated":
        # V_c = 0 but we rescale by 1/α post-hoc. Diagnostic: confirms the only
        # failure mode for v_zero_only is the leaked-mass rescale.
        k_c = torch.randn(B, H, S_c, D, generator=gen, device=device, dtype=q_dtype)
        k_c = k_c / k_c.float().pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt().to(q_dtype)
        v_c = torch.zeros(B, H, S_c, D, device=device, dtype=q_dtype)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return k_c, v_c, cond_logit_bias, masked


# ----------------------------------------------------------------- the sweep

STRATEGIES = [
    "v_zero_only",
    "kv_zero",
    "cond_input_zero",
    "logit_bias_-10",
    "logit_bias_-30",
    "masked",
    "v_zero_alpha_compensated",
]


def run_one(strategy, q, k_t, v_t, S_c, D, gen, device):
    B, H, S_t, _ = q.shape
    k_c, v_c, bias, masked = make_cond_kv(
        strategy, B, H, S_c, D, q.dtype, device, gen
    )

    out_base = baseline_attention(q, k_t, v_t)
    out_ext, attn = extended_attention(
        q, k_t, v_t, k_c, v_c, cond_logit_bias=bias, masked=masked
    )
    alpha = alpha_from_attention(attn, S_t)

    if strategy == "v_zero_alpha_compensated":
        # Divide rowwise by α to undo the leaked-mass rescale.
        out_ext = out_ext / alpha.unsqueeze(-1).clamp_min(1e-12)

    diff = (out_ext - out_base).float()
    base_norm = out_base.float().norm()
    diff_norm = diff.norm()
    rel_l2 = (diff_norm / base_norm.clamp_min(1e-12)).item()
    abs_max = diff.abs().max().item()

    return {
        "strategy": strategy,
        "rel_l2": rel_l2,
        "abs_max": abs_max,
        "alpha_min": alpha.min().item(),
        "alpha_mean": alpha.mean().item(),
        "alpha_max": alpha.max().item(),
    }


def aggregate(rows_per_trial):
    """Average across trials per strategy."""
    by = {}
    for trial in rows_per_trial:
        for r in trial:
            by.setdefault(r["strategy"], []).append(r)
    out = []
    for strategy, rs in by.items():
        out.append(
            {
                "strategy": strategy,
                "rel_l2_mean": sum(r["rel_l2"] for r in rs) / len(rs),
                "rel_l2_max": max(r["rel_l2"] for r in rs),
                "abs_max_mean": sum(r["abs_max"] for r in rs) / len(rs),
                "alpha_mean": sum(r["alpha_mean"] for r in rs) / len(rs),
                "alpha_min_overall": min(r["alpha_min"] for r in rs),
                "alpha_max_overall": max(r["alpha_max"] for r in rs),
                "n_trials": len(rs),
            }
        )
    # Order to match STRATEGIES.
    order = {s: i for i, s in enumerate(STRATEGIES)}
    out.sort(key=lambda r: order[r["strategy"]])
    return out


def format_table(rows):
    header = (
        f"{'strategy':30s} | {'rel_l2 mean':>12s} | {'rel_l2 max':>12s} | "
        f"{'α mean':>8s} | {'α min':>8s} | {'α max':>8s}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"{r['strategy']:30s} | {r['rel_l2_mean']:12.3e} | {r['rel_l2_max']:12.3e} | "
            f"{r['alpha_mean']:8.4f} | {r['alpha_min_overall']:8.4f} | {r['alpha_max_overall']:8.4f}"
        )
    return "\n".join(lines)


def verdict(rows, threshold_strict=1e-3, threshold_acceptable=1e-2):
    """Decide pass/fail per strategy with two thresholds.

    'strict' = within bf16 numerical noise of the baseline (truly equivalent).
    'acceptable' = perturbation small enough that step-0 generation looks
                   identical to the human eye but isn't bit-exact.
    """
    out = []
    for r in rows:
        rel = r["rel_l2_max"]
        if rel < threshold_strict:
            tier = "EXACT (within numerical noise)"
        elif rel < threshold_acceptable:
            tier = "ACCEPTABLE (small perturbation)"
        else:
            tier = "FAIL (visible perturbation)"
        out.append((r["strategy"], rel, tier))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--s_t", type=int, default=4096, help="target token count")
    p.add_argument("--s_c", type=int, default=4096, help="cond token count")
    p.add_argument("--n_heads", type=int, default=16)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--n_trials", type=int, default=8)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32",
                   help="fp32 for math clarity; bf16 to match train-time precision.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_json", default="bench/active/easycontrol/results/step0_equivalence.json")
    args = p.parse_args()

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = torch.device(args.device)

    logger.info(
        f"step-0 equivalence sweep | S_t={args.s_t} S_c={args.s_c} "
        f"H={args.n_heads} D={args.head_dim} dtype={args.dtype} device={device}"
    )

    rows_per_trial = []
    for trial in range(args.n_trials):
        gen = torch.Generator(device=device).manual_seed(args.seed + trial)
        q, k_t, v_t = make_target_qkv(
            args.batch, args.n_heads, args.s_t, args.head_dim, dtype, device, gen
        )
        trial_rows = []
        for s in STRATEGIES:
            with torch.no_grad():
                trial_rows.append(
                    run_one(s, q, k_t, v_t, args.s_c, args.head_dim, gen, device)
                )
        rows_per_trial.append(trial_rows)

    rows = aggregate(rows_per_trial)
    table = format_table(rows)
    print()
    print(table)
    print()

    print("Verdict (rel_l2_max thresholded):")
    for strategy, rel, tier in verdict(rows):
        print(f"  {strategy:30s}  rel_l2_max = {rel:.3e}   →  {tier}")

    summary = {
        "config": {
            "s_t": args.s_t,
            "s_c": args.s_c,
            "n_heads": args.n_heads,
            "head_dim": args.head_dim,
            "batch": args.batch,
            "n_trials": args.n_trials,
            "dtype": args.dtype,
            "device": str(device),
        },
        "rows": rows,
        "verdict": [
            {"strategy": s, "rel_l2_max": rel, "tier": tier}
            for s, rel, tier in verdict(rows)
        ],
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"saved summary to {out_path}")


if __name__ == "__main__":
    main()
