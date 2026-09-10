#!/usr/bin/env python3
"""Rebuild per-(prompt,seed) 2-D arm grids from a finished run's per-arm PNGs.

The original run wrote a single stacked contact sheet that ballooned to ~100 MB.
This reassembles the already-saved ``p{pi}_s{sj}_{arm}.png`` files into one
downscaled ``grid_p{pi}_s{sj}.png`` per (prompt, seed) — no re-inference. Labels
(sat / contrast / Δ-drift) come from the run's result.json rows.

    uv run python _archive/bench/mist/make_grids.py _archive/bench/mist/results/<run-dir> [--rm-sheet]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # _archive/bench/mist/ -> repo root
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image  # noqa: E402

from _archive.bench.mist.render_compare import _grid  # noqa: E402


def _safe(lab: str) -> str:
    return lab.replace(" ", "_").replace(".", "p").replace("/", "-")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--grid_cols", type=int, default=4)
    ap.add_argument("--cell_px", type=int, default=512)
    ap.add_argument("--rm-sheet", action="store_true",
                    help="Delete the oversized contact_sheet.png after rebuilding.")
    args = ap.parse_args()

    res = json.loads((args.run_dir / "result.json").read_text())
    m = res["metrics"]
    arms = m["arms"]
    prompts = m.get("prompts", [])
    # (prompt_idx, seed, arm) -> row, for labels.
    by = {(r["prompt_idx"], r["seed"], r["arm"]): r for r in m["rows"]}

    pis = sorted({r["prompt_idx"] for r in m["rows"]})
    sjs = sorted({r["seed"] for r in m["rows"]})
    made = 0
    for pi in pis:
        for sj in sjs:
            imgs, labels = [], []
            for lab in arms:
                p = args.run_dir / f"p{pi}_s{sj}_{_safe(lab)}.png"
                if not p.exists():
                    print(f"  missing {p.name}, skipping arm")
                    continue
                imgs.append(Image.open(p).convert("RGB"))
                row = by.get((pi, sj, lab), {})
                sat = row.get("saturation", 0.0)
                con = row.get("rms_contrast", 0.0)
                drift = row.get("drift", 0.0)
                labels.append(
                    f"{lab}  sat{sat:.3f} c{con:.3f}"
                    + (f"  Δ={drift:.3f}" if lab != "baseline" else "")
                )
            if not imgs:
                continue
            cap = prompts[pi] if pi < len(prompts) else ""
            out = args.run_dir / f"grid_p{pi}_s{sj}.png"
            _grid(imgs, labels, f"p{pi} s{sj}: {cap}",
                  ncols=args.grid_cols, cell=args.cell_px).save(out)
            made += 1

    sheet = args.run_dir / "contact_sheet.png"
    if args.rm_sheet and sheet.exists():
        sheet.unlink()
        print(f"removed {sheet.name}")
    print(f"wrote {made} grids to {args.run_dir}")


if __name__ == "__main__":
    main()
