"""E25b post-close descriptive: the trained native offset W.phi(0).

Registered in the Stage-2 descriptives row ("||W.phi(0)|| per rescond
checkpoint -- the size of the trained native offset") but never emitted by
e25b_ship_read.py; first measured 2026-08-12 post-close. Decomposes each
trained res-cond projection into the common-mode component (W.phi(0), the
offset EVERY step including native receives) vs the resolution-differential
component (W.phi(s) - W.phi(0), the part that actually distinguishes routes).

Result (e25b_native_offset.json): the lever trained ~90% common-mode --
||W.phi(0)|| 0.13-0.19 vs differential 0.009-0.02, cos(phi0, phis) 0.975-0.999
across all 8 carrier checkpoints. The conditioning was used as a global bias
on the adaln trunk, shifting the native operating point; the LoRA co-adapted
around it (the ship read's ANTI-direction dW and below-yardstick renders).
This measurement motivates E25e (re-centered delta W.(phi(s) - phi(0))).

CPU-only; reads output/ckpt/e25b2_*rescond*.safetensors.
"""

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from library.anima.models import sigma_lowres_res_cond_delta

ROOT = Path(__file__).resolve().parents[5]
CKPT = ROOT / "output" / "ckpt"
OUT = Path(__file__).resolve().parent / "e25b_native_offset.json"

S896 = math.log2(896 / 1024)
S768 = math.log2(768 / 1024)

CARRIERS = [
    f"e25b2_{corpus}_{arm}_{seed}"
    for corpus in ("hews", "channel")
    for arm in ("rescond", "rescond768")
    for seed in ("s1001", "s1002", "s1003")
]


def main() -> None:
    ts = torch.zeros(1, 1)
    cos = torch.nn.functional.cosine_similarity
    rows = {}
    for name in CARRIERS:
        path = CKPT / f"{name}.safetensors"
        if not path.exists():
            continue
        with safe_open(str(path), framework="pt") as f:
            proj = f.get_tensor("sigma_lowres_res_cond_proj").float()
        d = {
            s: sigma_lowres_res_cond_delta(proj, s, ts).flatten()
            for s in (0.0, S896, S768)
        }
        rows[name] = {
            "norm_phi0": d[0.0].norm().item(),
            "norm_phi896": d[S896].norm().item(),
            "norm_phi768": d[S768].norm().item(),
            "norm_diff_896_minus_0": (d[S896] - d[0.0]).norm().item(),
            "norm_diff_768_minus_0": (d[S768] - d[0.0]).norm().item(),
            "cos_phi0_phi896": cos(d[0.0], d[S896], dim=0).item(),
            "cos_phi0_phi768": cos(d[0.0], d[S768], dim=0).item(),
        }
    common = [r["norm_phi0"] for r in rows.values()]
    diff = [r["norm_diff_896_minus_0"] for r in rows.values()]
    summary = {
        "n_carriers": len(rows),
        "norm_phi0_range": [min(common), max(common)],
        "norm_diff_896_range": [min(diff), max(diff)],
        "reading": (
            "common-mode dominated: the trained lever is ~90% a "
            "resolution-independent W.phi(0) bias applied to every step "
            "(native included); the resolution-differential component is "
            "~6-13% of the delta norm. Motivates E25e (re-centered)."
        ),
    }
    OUT.write_text(json.dumps({"summary": summary, "per_ckpt": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    for name, r in rows.items():
        print(
            f"{name:38s} |Wphi0| {r['norm_phi0']:.4f}  "
            f"|d896-d0| {r['norm_diff_896_minus_0']:.4f}  "
            f"cos {r['cos_phi0_phi896']:.4f}"
        )


if __name__ == "__main__":
    main()
