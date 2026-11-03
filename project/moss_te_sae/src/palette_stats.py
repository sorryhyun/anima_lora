"""Global colour statistics of lora_ab renders, per (prompt, seed), against no LoRA.

For each arm folder (``output/tests/ab_<lora>_50st_fp32s``) and the no-LoRA renders
(``x0.0`` in ``--none``), CIELAB means over the whole image: L*, chroma C*, a*, b*.
``recovered`` is the arm's shift from no LoRA projected on the target arm's shift
(``--target``), as a fraction: 1 = the target's palette, 0 = no LoRA's. Compositions
differ between arms, so only the global statistics are compared.

    .venv/bin/python project/moss_te_sae/src/palette_stats.py \
        --none output/tests/ab_caststation_rows_trigger_reg4_50st_fp32s \
        --target output/tests/ab_caststation_full_int8_50st_fp32s \
        --arms output/tests/ab_caststation_rows_trigger_int8_50st_fp32s,output/tests/ab_caststation_rows_trigger_reg4_50st_fp32s
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from library.env import resolve_under_home


def lab(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
        -1,
    )


def stats(path: Path) -> np.ndarray:
    """``[L*, C*, a*, b*]`` means."""
    x = lab(path).reshape(-1, 3)
    return np.array(
        [x[:, 0].mean(), np.hypot(x[:, 1], x[:, 2]).mean(), x[:, 1].mean(), x[:, 2].mean()]
    )


def renders(folder: Path, multiplier: str) -> dict[tuple[int, int], Path]:
    pat = re.compile(rf"base_p(\d+)_s(\d+)_x{re.escape(multiplier)}_er_sde_.*\.png")
    out = {}
    for p in folder.glob("base_p*.png"):
        if m := pat.fullmatch(p.name):
            out[int(m[1]), int(m[2])] = p
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--none", required=True, help="folder holding the x0.0 renders")
    ap.add_argument("--target", required=True, help="folder of the palette target arm")
    ap.add_argument("--arms", required=True, help="comma-separated arm folders")
    args = ap.parse_args()
    none = renders(resolve_under_home(args.none), "0.0")
    target = renders(resolve_under_home(args.target), "1.0")
    folders = [resolve_under_home(a) for a in args.arms.split(",")]
    arms = {f.name: renders(f, "1.0") for f in folders}
    keys = sorted(none)
    names = ["L", "C", "a", "b"]
    rows = []
    for key in keys:
        s0, st = stats(none[key]), stats(target[key])
        shift = st - s0
        row = {"prompt": key[0], "seed": key[1], "none": s0.round(2).tolist(),
               "target": st.round(2).tolist()}
        for name, arm in arms.items():
            sa = stats(arm[key])
            d = sa - s0
            row[name] = {
                "stats": sa.round(2).tolist(),
                "recovered": round(float(d @ shift / (shift @ shift)), 3),
                "residual": round(float(np.linalg.norm(d - shift)), 2),
            }
        rows.append(row)
    print("p s   none " + " ".join(f"{n:>6}" for n in names) + "   target shift")
    for r in rows:
        shift = np.array(r["target"]) - np.array(r["none"])
        line = f"{r['prompt']} {r['seed']}  " + " ".join(f"{v:6.1f}" for v in r["none"])
        line += "   " + " ".join(f"{v:+5.1f}" for v in shift)
        for name in arms:
            line += f" | {name[-30:]}: rec {r[name]['recovered']:+.2f}"
        print(line)
    for name in arms:
        rec = [r[name]["recovered"] for r in rows]
        print(f"{name}: recovered mean {np.mean(rec):+.2f}, per image {rec}")
    out = folders[-1] / "palette_stats.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
