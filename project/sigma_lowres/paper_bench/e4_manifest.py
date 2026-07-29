"""E4 frozen train/prompt manifest generator (run ONCE, before training).

E4 (`required_experiments.md`) is the end-to-end A/B: native baseline vs
σ-conditional demotion (measured-safe 1024→896 @ σ>0.5, + yarnsig) vs an
unsafe-route negative control (1024→768 unconditional), identical data
order/noise/init, 3 seeds — scored by realized wall-clock, examples/s,
peak mem, CMMD, val loss, matched renders, and (new here) paired PE-Core
cosine between arms at matched (prompt, seed).

This script freezes, per artist, BEFORE any training:

- the train/holdout stem split (seeded permutation; the holdout side is
  both the CMMD real-reference pool and the prompt source);
- the verification prompts: holdout captions with per-item generation
  seeds (GEN_SEED_BASE + i, the generalize.py convention) and the
  holdout image's own free-fit bucket as the render size;
- the three arms as argv fragments + the three train seeds;
- argv-ready fnmatch alternation path_patterns (train side), verified
  against the real resized tree via the production filter.

Artists: `hews` (primary, 121 qualifying 1024-tier stems) and
`channel_(caststation)` (secondary, 30 — thin holdout, wide CMMD floor;
kept as confirmation/eyeball arm). Chosen 2026-07-29 to avoid the
well-known sincos corpus.

Arm C (`unsafe768`) additionally requires (a) the 1024:768 demote-sibling
emit and (b) a small trainer patch making the train-side route
selectable (`SIGMA_DEMOTE_ROUTE` in `library/datasets/buckets.py` is a
1024:896 constant today) — both recorded in the manifest as prereqs.

Usage:
    python project/sigma_lowres/paper_bench/e4_manifest.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from library.datasets.path_filter import filter_paths_by_glob  # noqa: E402

SEED = 20260729
ARTISTS = ["hews", "channel_(caststation)"]  # primary first
TRAIN_FRAC = 0.5  # maximize the holdout reference pool (paired design:
TRAIN_CAP = 64  # both arms share membership, so absolute train size
HOLDOUT_FLOOR = 3  # matters less than reference-pool width)
N_PROMPTS = 24  # per artist, capped by holdout size
GEN_SEED_BASE = 20_000  # generalize.py convention (disjoint from eyeball 0..2)
TRAIN_SEEDS = [1001, 1002, 1003]
BAND_1024 = (4032, 4200)  # frozen 1024-tier free-fit token band

LORA = REPO / "post_image_dataset" / "lora"
RESIZED = REPO / "post_image_dataset" / "resized"
IMAGES = REPO / "image_dataset"
NPZ_RE = re.compile(r"^(.*)_(\d+)x(\d+)_anima\.npz$")


def esc(s: str) -> str:
    """fnmatch-escape [ and ] (the only specials legal in our names)."""
    return s.replace("[", "[[]").replace("]", "[]]")


def qualifying_stems(artist: str) -> dict[str, tuple[int, int]]:
    """stem -> (W, H) for stems with a 1024-tier npz + TE cache + caption."""
    d = LORA / artist
    te = {f.name[: -len("_anima_te.safetensors")] for f in d.glob("*_anima_te.safetensors")}
    caps = {p.stem for p in (IMAGES / artist).rglob("*.txt")}
    out: dict[str, tuple[int, int]] = {}
    for f in d.glob("*_anima.npz"):
        m = NPZ_RE.match(f.name)
        if not m:
            continue
        stem, w, h = m.group(1), int(m.group(2)), int(m.group(3))
        tokens = (w // 16) * (h // 16)
        if BAND_1024[0] <= tokens <= BAND_1024[1] and stem in te and stem in caps:
            out[stem] = (w, h)
    return out


def caption_for(artist: str, stem: str) -> str:
    hits = sorted((IMAGES / artist).rglob(f"{stem}.txt"))
    assert hits, f"caption vanished for {artist}/{stem}"
    return hits[0].read_text(encoding="utf-8").strip()


def main() -> None:
    rng = np.random.default_rng(SEED)
    manifest: dict[str, dict] = {}
    patterns: dict[str, str] = {}
    all_pngs = sorted(str(p) for p in RESIZED.glob("*/*.png"))

    for artist in ARTISTS:
        sizes = qualifying_stems(artist)
        stems = sorted(sizes)
        n = len(stems)
        train_n = min(TRAIN_CAP, round(TRAIN_FRAC * n), n - HOLDOUT_FLOOR)
        order = rng.permutation(n)
        train = sorted(stems[i] for i in order[:train_n])
        holdout = sorted(set(stems) - set(train))

        p_order = rng.permutation(len(holdout))
        picked = [holdout[i] for i in p_order[: min(N_PROMPTS, len(holdout))]]
        prompts = [
            dict(
                stem=s,
                caption=caption_for(artist, s),
                width=sizes[s][0],
                height=sizes[s][1],
                gen_seed=GEN_SEED_BASE + i,
            )
            for i, s in enumerate(picked)
        ]

        pattern = "|".join(f"{esc(artist)}/{esc(s)}.*" for s in train)
        mask = filter_paths_by_glob(all_pngs, str(RESIZED), pattern)
        matched = {
            str(Path(p).relative_to(RESIZED)).rsplit(".", 1)[0]
            for p, m in zip(all_pngs, mask)
            if m
        }
        expected = {f"{artist}/{s}" for s in train}
        assert matched == expected, (
            f"{artist}: pattern matched {len(matched)} != manifest "
            f"{len(expected)}; diff {matched ^ expected}"
        )
        patterns[artist] = pattern

        manifest[artist] = dict(
            n_qualifying=n,
            train=train,
            holdout=holdout,
            sizes={s: list(sizes[s]) for s in stems},
            prompts=prompts,
        )
        print(
            f"{artist}: qualifying={n} train={len(train)} holdout={len(holdout)} "
            f"prompts={len(prompts)}"
        )

    arms = dict(
        native=dict(extra_argv=[], demote_routes=[]),
        sigma896=dict(
            extra_argv=["--sigma_lowres", "--sigma_lowres_yarnsig"],
            demote_routes=["1024:896"],
            note="threshold default 0.5 (measured-safe gate); bare yarnsig "
            "= probe operating point 1,4,0.35,2",
        ),
        unsafe768=dict(
            extra_argv=["--sigma_lowres", "--sigma_lowres_threshold", "0.0"],
            demote_routes=["1024:768"],
            note="negative control: unconditional demotion on the "
            "no-certifiable-window route",
            requires=[
                "1024:768 demote-sibling emit (preprocess --sigma_demote 1024:768)",
                "trainer patch: SIGMA_DEMOTE_ROUTE (library/datasets/buckets.py) "
                "made selectable per run (currently 1024:896 constant)",
            ],
        ),
    )

    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M")
    out = HERE / "runs" / f"{stamp}-e4-manifest"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e4_manifest.json").write_text(
        json.dumps(
            dict(
                seed=SEED,
                rule=(
                    f"per artist: qualifying = 1024-tier npz + TE cache + caption; "
                    f"seeded permutation split, train = min({TRAIN_CAP}, "
                    f"round({TRAIN_FRAC}*n), n-{HOLDOUT_FLOOR}); prompts = first "
                    f"{N_PROMPTS} of a seeded holdout permutation, gen_seed = "
                    f"{GEN_SEED_BASE}+i, render size = the holdout image's own "
                    f"free-fit bucket"
                ),
                artists=ARTISTS,
                train_seeds=TRAIN_SEEDS,
                common_argv=["--deterministic", "--paired_step_rng"],
                arms=arms,
                eval_protocol=dict(
                    infer_steps=20,
                    cfg=1.0,
                    reference_pool="all holdout stems (real images)",
                    member_pool="all train stems (real images)",
                    metrics=[
                        "cmmd_holdout (PE-Core pooled MMD^2 vs holdout pool)",
                        "cmmd_member (memorization pull)",
                        "real-vs-real noise floor",
                        "paired PE-Core cosine across arms at matched (prompt, gen_seed)",
                        "realized wall-clock / examples/s / peak mem / val loss",
                    ],
                    convention="bench/reft/eval_cmmd.py + bench/memorization/generalize.py",
                ),
                cells=manifest,
            ),
            indent=1,
        )
    )
    for artist, pattern in patterns.items():
        slug = re.sub(r"[^A-Za-z0-9]+", "_", artist).strip("_")
        (out / f"path_pattern_{slug}_train.txt").write_text(pattern + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
