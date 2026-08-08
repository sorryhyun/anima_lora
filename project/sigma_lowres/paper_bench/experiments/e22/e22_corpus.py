#!/usr/bin/env python
"""sigma_lowres E22 corpus prep — realized resize-factor table + stratified
16-image probe list (CPU, header reads only).

Pre-registered in README.md: for every image of the standard 40-image probe
corpus (``e13/e1b_probe_list.json``), factor = source-pixel area /
resized-pixel area (``image_dataset/`` original vs ``post_image_dataset/
resized/``). Rank-based deterministic selection, seed 42 (rank on factor,
ties broken by (artist, stem)): 8 **near-native** = smallest factors
(target boundary ≤ ~1.3) + 8 **heavy** = largest factors (target ≥ ~2).
If a stratum cannot fill at the target boundary the extreme ranks are taken
and the realized boundary recorded — the estimand does not move.

Outputs (this dir, committed):
 * ``resize_factors.json``  — the full 40-image factor table + realized
   stratum boundaries
 * ``e22_probe_list.json``  — the 16-image ``--probe_list`` for 22.1, tags
   ``stratum`` + ``resize_factor`` copied verbatim into per-image rows
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
STANDARD_LIST = (
    REPO_ROOT / "project/sigma_lowres/paper_bench/experiments/e13/e1b_probe_list.json"
)
SOURCE_ROOT = REPO_ROOT / "image_dataset"
RESIZED_ROOT = REPO_ROOT / "post_image_dataset" / "resized"
SEED = 42  # recorded for provenance; selection is rank-based (no RNG draw)
N_PER_STRATUM = 8
TARGET_NEAR = 1.3  # near-native target boundary (≤)
TARGET_HEAVY = 2.0  # heavy target boundary (≥)

IMG_EXTS = (".webp", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".avif")


def find_source(artist: str, stem: str) -> Path:
    # artist dirs may be symlinks (nested layout) — resolve through them
    d = SOURCE_ROOT / artist
    for ext in IMG_EXTS:
        p = d / f"{stem}{ext}"
        if p.exists():
            return p
    hits = [p for p in d.glob(f"{stem}.*") if p.suffix.lower() != ".txt"]
    if len(hits) == 1:
        return hits[0]
    raise SystemExit(f"source image not found or ambiguous: {artist}/{stem} → {hits}")


def main() -> None:
    entries = json.loads(STANDARD_LIST.read_text())
    if isinstance(entries, dict):
        entries = entries["images"]

    table = []
    for e in entries:
        artist, stem = e["artist"], e["stem"]
        src = find_source(artist, stem)
        dst = RESIZED_ROOT / artist / f"{stem}.png"
        if not dst.exists():
            raise SystemExit(f"resized image missing: {dst}")
        with Image.open(src) as im:
            sw, sh = im.size
        with Image.open(dst) as im:
            rw, rh = im.size
        table.append(
            {
                "artist": artist,
                "stem": stem,
                "source": f"{sw}x{sh}",
                "resized": f"{rw}x{rh}",
                "factor": round((sw * sh) / (rw * rh), 4),
            }
        )

    # rank on factor, deterministic tie-break on (artist, stem)
    ranked = sorted(table, key=lambda r: (r["factor"], r["artist"], r["stem"]))
    near = ranked[:N_PER_STRATUM]
    heavy = ranked[-N_PER_STRATUM:]
    boundaries = {
        "near_native_target": TARGET_NEAR,
        "near_native_realized_max": near[-1]["factor"],
        "near_native_fills_target": near[-1]["factor"] <= TARGET_NEAR,
        "heavy_target": TARGET_HEAVY,
        "heavy_realized_min": heavy[0]["factor"],
        "heavy_fills_target": heavy[0]["factor"] >= TARGET_HEAVY,
    }

    (HERE / "resize_factors.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "standard_list": str(STANDARD_LIST.relative_to(REPO_ROOT)),
                "n_per_stratum": N_PER_STRATUM,
                "boundaries": boundaries,
                "table": ranked,
            },
            indent=1,
        )
    )

    probe = [
        {
            "artist": r["artist"],
            "stem": r["stem"],
            "stratum": stratum,
            "resize_factor": r["factor"],
        }
        for stratum, rows in (("near_native", near), ("heavy", heavy))
        for r in rows
    ]
    (HERE / "e22_probe_list.json").write_text(json.dumps({"images": probe}, indent=1))

    print(f"factors: {ranked[0]['factor']} .. {ranked[-1]['factor']}")
    for k, v in boundaries.items():
        print(f"  {k}: {v}")
    for p in probe:
        print(
            f"  {p['stratum']:<12} {p['resize_factor']:>8}  {p['artist']}/{p['stem']}"
        )
    print(f"→ {HERE / 'resize_factors.json'}\n→ {HERE / 'e22_probe_list.json'}")


if __name__ == "__main__":
    main()
