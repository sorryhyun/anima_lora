#!/usr/bin/env python3
"""Bilinear drift share of trained ``weight_svd`` LoRAs.

A ``weight_svd`` LoRA starts at ``A0 = V_r^T / sqrt(3)`` (W0's top-r right
singular vectors) and ``B = 0``, so a trained adapter splits exactly as
``ΔW = s·B·A0 + s·B·D`` with ``D = A − A0``: the update along the initial input
basis, and the part from the basis rotating (the ``ΔB·ΔA`` drift term of arXiv
2606.12883, accumulated). This reports ``||B·D|| / ||B·A0||`` per module, its
median, and ``||D|| / ||A0||``.

``A0`` is rebuilt with the trainer's own ``_top_right_singular_vectors`` on the
DiT weight the runtime adapted: self-attn q/k/v and cross-attn k/v are fused at
runtime, so their shared ``down`` was seeded from the concatenated weight and
their ``up`` blocks concatenate back. Eigenvector signs are aligned per row.
The saved ``down`` has channel scaling baked out, so ``D`` is in raw input
coordinates. Only for ``down_init = "weight_svd"``, ``svd_slice = 0``.

    make daemon-run ARGS="--stall-timeout 600 bench/anima_abclip/drift_share.py \\
        output/ckpt/nomask_cast_s42.safetensors output/ckpt/alpha512_cast_s42.safetensors"
"""

from __future__ import annotations

import argparse
import math
import statistics as st
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from bench._common import make_run_dir, write_result
from library.env import resolve_under_home
from networks.lora_modules.lora import _top_right_singular_vectors

DIT = "models/diffusion_models/anima-base-v1.0.safetensors"
# (LoRA fragments sharing one down, DiT weights concatenated on dim 0)
GROUPS = [
    (
        ["self_attn_q_proj", "self_attn_k_proj", "self_attn_v_proj"],
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    ),
    (
        ["cross_attn_k_proj", "cross_attn_v_proj"],
        ["cross_attn.k_proj", "cross_attn.v_proj"],
    ),
    (["self_attn_output_proj"], ["self_attn.output_proj"]),
    (["cross_attn_q_proj"], ["cross_attn.q_proj"]),
    (["cross_attn_output_proj"], ["cross_attn.output_proj"]),
    (["mlp_layer1"], ["mlp.layer1"]),
    (["mlp_layer2"], ["mlp.layer2"]),
    (["adaln_modulation_self_attn_2"], ["adaln_modulation_self_attn.2"]),
    (["adaln_modulation_cross_attn_2"], ["adaln_modulation_cross_attn.2"]),
    (["adaln_modulation_mlp_2"], ["adaln_modulation_mlp.2"]),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("loras", nargs="+", type=Path)
    ap.add_argument("--dit", default=DIT)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    loras = {p.stem: load_file(str(p)) for p in args.loras}
    first = next(iter(loras.values()))
    nblocks = 1 + max(
        int(k.split("_")[3]) for k in first if k.startswith("lora_unet_blocks_")
    )
    rows: dict[str, list[dict]] = {name: [] for name in loras}
    worst_cos = 1.0
    with safe_open(str(resolve_under_home(args.dit)), "pt") as f:
        for b in range(nblocks):
            for lfr, dfr in GROUPS:
                pre = [f"lora_unet_blocks_{b}_{x}" for x in lfr]
                if f"{pre[0]}.lora_down.weight" not in first:
                    continue
                W = torch.cat(
                    [
                        f.get_tensor(f"net.blocks.{b}.{d}.weight").to(dev).float()
                        for d in dfr
                    ]
                )
                r = first[f"{pre[0]}.lora_down.weight"].shape[0]
                A0 = (_top_right_singular_vectors(W, r).T / math.sqrt(3)).cpu()
                for name, sd in loras.items():
                    A = sd[f"{pre[0]}.lora_down.weight"].float()
                    A0s = A0 * torch.sign((A * A0).sum(1, keepdim=True))
                    cos = torch.nn.functional.cosine_similarity(A, A0s, dim=1)
                    worst_cos = min(worst_cos, float(cos.min()))
                    U = torch.cat([sd[f"{x}.lora_up.weight"].float() for x in pre])
                    D = A - A0s
                    rows[name].append(
                        {
                            "type": lfr[0],
                            "drift": float((U @ D).norm() / (U @ A0s).norm()),
                            "dA": float(D.norm() / A0s.norm()),
                            "cos": float(cos.median()),
                        }
                    )

    metrics = {}
    for name, rs in rows.items():
        by: dict[str, list[float]] = {}
        for x in rs:
            by.setdefault(x["type"], []).append(x["drift"])
        metrics[name] = {
            "drift_median": round(st.median(x["drift"] for x in rs), 4),
            "dA_median": round(st.median(x["dA"] for x in rs), 4),
            "row_cos_median": round(st.median(x["cos"] for x in rs), 4),
            "drift_by_type": {k: round(st.median(v), 4) for k, v in by.items()},
        }
    metrics["worst_row_cos"] = round(worst_cos, 4)
    run_dir = make_run_dir("anima_abclip", label=args.label or "drift_share")
    write_result(
        run_dir,
        script=__file__,
        args={"loras": [str(p) for p in args.loras], "dit": args.dit},
        metrics=metrics,
    )
    for name, m in metrics.items():
        if isinstance(m, dict):
            print(
                f"{name:24s} drift {m['drift_median']:.3f}  dA {m['dA_median']:.3f}  "
                f"cos {m['row_cos_median']:.4f}"
            )
    print(f"worst row cos {worst_cos:.4f} -> {run_dir}")


if __name__ == "__main__":
    main()
