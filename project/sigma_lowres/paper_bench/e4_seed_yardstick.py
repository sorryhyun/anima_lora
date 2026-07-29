#!/usr/bin/env python3
"""E4 seed-noise yardstick — is the σ-demote render footprint inside the seed lottery?

Defensibility test for "sigma896 renders differ visibly from native": compares
the within-seed arm effect against the recipe's own cross-seed noise, on the
SAME frozen (prompt, gen_seed) grid.

    within-seed arm pairs   cos(native_sK ~ sigma896_sK)      — the intervention
    cross-seed same-arm     cos(native_sK ~ native_sL)        — the yardstick
                            cos(sigma896_sK ~ sigma896_sL)

Verdict per artist: if mean cos(native~sigma896 | same seed) ≥ mean
cos(native~native | cross seed), the demote footprint is smaller than the seed
lottery — sample-level differences are recipe noise, not degradation evidence.

Reads the saved eval renders (`<eval_dir>/renders/<artist_slug>/<arm>/<stem>.png`,
cfg 1.0 / 20 steps passes), PE-Core-pools each (seed, arm) set once, and writes
`yardstick.json` + `yardstick.md`.

Usage (GPU — daemon)::

    make daemon-run ARGS="project/sigma_lowres/paper_bench/e4_seed_yardstick.py \
        --prompts .../e4_prompts_sfw.json \
        --evals s1001=<dir> s1002=<dir> s1003=<dir> --out <dir>"
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from library.runtime.device import clean_memory_on_device  # noqa: E402
from library.training.cmmd import pool_and_normalize  # noqa: E402
from library.vision.encoder import (  # noqa: E402
    encode_pe_from_imageminus1to1,
    load_pe_encoder,
)

ARMS = ("native", "sigma896", "unsafe768")


def slug(artist: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", artist).strip("_")


def _pool_dir(bundle, d: Path, stems: list[str]) -> torch.Tensor:
    pooled = []
    for s in stems:
        with Image.open(d / f"{s}.png") as im:
            t = torch.from_numpy(np.asarray(im.convert("RGB"))).float()
        img = (t / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
        feats = encode_pe_from_imageminus1to1(bundle, img.to(bundle.device))
        pooled.append(pool_and_normalize(feats[0]).to("cpu"))
    return torch.stack(pooled)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", required=True, help="e4_prompts_sfw.json")
    ap.add_argument(
        "--evals",
        nargs="+",
        required=True,
        metavar="sSEED=DIR",
        help="cfg-1.0/20-step eval run dirs keyed by train seed",
    )
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    evals = dict(kv.split("=", 1) for kv in args.evals)
    prompts = json.loads(Path(args.prompts).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_pe_encoder(device)

    report: dict = {"evals": evals, "arms": list(args.arms), "cells": {}}
    for artist, cell in prompts["cells"].items():
        stems = [p["stem"] for p in cell["prompts"]]
        aslug = slug(artist)
        pools: dict[tuple[str, str], torch.Tensor] = {}
        for seed, d in evals.items():
            for arm in args.arms:
                rdir = Path(d) / "renders" / aslug / arm
                if not all((rdir / f"{s}.png").exists() for s in stems):
                    print(f"skip {aslug} {seed}/{arm}: renders incomplete", flush=True)
                    continue
                pools[(seed, arm)] = _pool_dir(bundle, rdir, stems)

        def mean_cos(a: tuple[str, str], b: tuple[str, str]) -> float | None:
            if a not in pools or b not in pools:
                return None
            return float((pools[a] * pools[b]).sum(dim=1).mean())

        seeds = sorted({s for s, _ in pools})
        within = {
            f"{a}~{b}|{s}": mean_cos((s, a), (s, b))
            for s in seeds
            for a, b in itertools.combinations(args.arms, 2)
        }
        cross = {
            f"{arm}|{sa}~{sb}": mean_cos((sa, arm), (sb, arm))
            for arm in args.arms
            for sa, sb in itertools.combinations(seeds, 2)
        }
        within = {k: v for k, v in within.items() if v is not None}
        cross = {k: v for k, v in cross.items() if v is not None}

        eff = [v for k, v in within.items() if k.startswith("native~sigma896")]
        yard = [v for k, v in cross.items() if k.startswith("native|")]
        verdict = None
        if eff and yard:
            verdict = {
                "arm_effect_mean_cos": sum(eff) / len(eff),
                "seed_yardstick_mean_cos": sum(yard) / len(yard),
                "inside_seed_lottery": sum(eff) / len(eff) >= sum(yard) / len(yard),
            }
        report["cells"][artist] = {
            "n_prompts": len(stems),
            "within_seed_arm_pairs": within,
            "cross_seed_same_arm": cross,
            "verdict": verdict,
        }
        print(f"[{aslug}] verdict: {verdict}", flush=True)

    del bundle
    clean_memory_on_device(device)

    (out / "yardstick.json").write_text(json.dumps(report, indent=1))
    lines = ["# E4 seed-noise yardstick", ""]
    for artist, c in report["cells"].items():
        lines += [f"## {artist}", ""]
        lines += ["| within-seed arm pair | mean cos |", "|---|---|"]
        lines += [
            f"| {k} | {v:.4f} |" for k, v in sorted(c["within_seed_arm_pairs"].items())
        ]
        lines += ["", "| cross-seed same arm | mean cos |", "|---|---|"]
        lines += [
            f"| {k} | {v:.4f} |" for k, v in sorted(c["cross_seed_same_arm"].items())
        ]
        v = c["verdict"]
        if v:
            lines += [
                "",
                f"**Verdict**: arm effect {v['arm_effect_mean_cos']:.4f} vs seed "
                f"yardstick {v['seed_yardstick_mean_cos']:.4f} → "
                + (
                    "**inside the seed lottery** (defensible)"
                    if v["inside_seed_lottery"]
                    else "**outside the seed lottery** (real footprint — quality "
                    "case must rest on distribution metrics + eyeball)"
                ),
                "",
            ]
    (out / "yardstick.md").write_text("\n".join(lines))
    print(f"wrote {out / 'yardstick.md'}", flush=True)


if __name__ == "__main__":
    main()
