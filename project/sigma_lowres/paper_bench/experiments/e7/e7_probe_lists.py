"""E7 probe-list builder — frozen probe cells for the 2x2 style factorial.

Consumes ONLY the frozen split manifest
(``runs/20260729-1158-e7-manifest/e7_manifest.json``: train/holdout/
reserved stems + per-image redundancy, seed 20260729) and emits one
probe-list JSON per adapter for ``run_sigma_probe.py --probe_list``:

  cells per adapter, N=12 each (cell tags are adapter-relative):
    S1   trained-on            (cluster's train manifest)
    S2   held-out, trained artists — redundancy-MATCHED to S1
    S3s  never-seen artists, same style cluster   (reserved set R_c)
    S3x  never-seen artists, opposite cluster     (reserved set R_c')

  R_flat and R_dirty are single frozen sets probed by BOTH adapters
  (S3s for the own-cluster adapter, S3x for the other), so the
  adapter x probe-style interaction reads paired per-stem.

Selection (deterministic, no RNG beyond the manifest's frozen content):
  S1: train stems sorted by redundancy, 12 evenly spaced picks with an
      outward-walk per-artist cap of 2 (select_probe_set's rule).
  S2: for each S1 stem in order, the unused holdout stem (trained
      artists only) with nearest redundancy, per-artist cap 2.
  R:  reserved stems sorted by redundancy, 12 evenly spaced picks with
      per-artist cap 3 (4 reserved artists per cluster).

Every selected stem is validated against the live cache (complete
1024-tier record) so the GPU runs cannot die on a stale stem.

Usage:
    python project/sigma_lowres/paper_bench/experiments/e7/e7_probe_lists.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER_BENCH = HERE.parents[1]  # project/sigma_lowres/paper_bench
SIGMA = HERE.parents[2]  # project/sigma_lowres
RUNS = PAPER_BENCH / "runs"
REPO_ROOT = HERE.parents[4]
sys.path.insert(0, str(REPO_ROOT))

from project.sigma_lowres.bench.tier_routing.redundancy import (  # noqa: E402
    score_corpus,
)

MANIFEST = RUNS / "20260729-1158-e7-manifest" / "e7_manifest.json"
N_CELL = 12
TIER = 1024


def spaced_picks(
    pool: list[tuple[str, str, float]], n: int, max_per_artist: int
) -> list[tuple[str, str, float]]:
    """Evenly spaced picks over the redundancy-sorted pool with an
    outward-walk per-artist cap (mirrors select_probe_set)."""
    pool = sorted(pool, key=lambda t: (t[2], t[0], t[1]))
    if len(pool) <= n:
        return pool
    picks: list[tuple[str, str, float]] = []
    per_artist: dict[str, int] = {}
    taken: set[int] = set()
    targets = [round(i * (len(pool) - 1) / (n - 1)) for i in range(n)]
    for t in targets:
        for off in range(len(pool)):
            hit = None
            for idx in (t + off, t - off):
                if idx < 0 or idx >= len(pool) or idx in taken:
                    continue
                a = pool[idx][0]
                if per_artist.get(a, 0) >= max_per_artist:
                    continue
                hit = idx
                break
            if hit is not None:
                picks.append(pool[hit])
                taken.add(hit)
                per_artist[pool[hit][0]] = per_artist.get(pool[hit][0], 0) + 1
                break
    return picks


def matched_picks(
    anchors: list[tuple[str, str, float]],
    pool: list[tuple[str, str, float]],
    max_per_artist: int,
) -> list[tuple[str, str, float]]:
    """Greedy nearest-redundancy match, one pool pick per anchor,
    without replacement, per-artist capped."""
    avail = sorted(pool, key=lambda t: (t[2], t[0], t[1]))
    per_artist: dict[str, int] = {}
    out: list[tuple[str, str, float]] = []
    for _, _, r in anchors:
        best = None
        for i, (a, s, rr) in enumerate(avail):
            if per_artist.get(a, 0) >= max_per_artist:
                continue
            d = abs(rr - r)
            if best is None or d < best[0]:
                best = (d, i)
        d, i = best
        a, s, rr = avail.pop(i)
        per_artist[a] = per_artist.get(a, 0) + 1
        out.append((a, s, rr))
    return out


def main() -> None:
    m = json.loads(MANIFEST.read_text())
    clusters = m["clusters"]

    def entries(dd: dict[str, list[str]], red: dict[str, float]):
        return [(a, s, red[f"{a}/{s}"]) for a, ss in dd.items() for s in ss]

    cells: dict[str, dict[str, list[tuple[str, str, float]]]] = {}
    for c, d in clusters.items():
        red = d["redundancy"]
        s1 = spaced_picks(entries(d["train"], red), N_CELL, max_per_artist=2)
        s2 = matched_picks(
            s1, entries(d["holdout"], red), max_per_artist=2
        )
        r = spaced_picks(entries(d["reserved"], red), N_CELL, max_per_artist=3)
        cells[c] = {"S1": s1, "S2": s2, "R": r}

    # validate against the live cache: complete 1024-tier records only
    all_artists = sorted(
        {a for c in cells.values() for cell in c.values() for a, _, _ in cell}
    )
    records = {
        (r.artist, r.stem): r
        for r in score_corpus(artists=all_artists)
    }
    problems = []
    for c, cc in cells.items():
        for cell, picks in cc.items():
            for a, s, _ in picks:
                rec = records.get((a, s))
                if rec is None:
                    problems.append(f"{c}/{cell}: {a}/{s} not in cache")
                elif rec.tier != TIER or not rec.complete():
                    problems.append(
                        f"{c}/{cell}: {a}/{s} tier={rec.tier} "
                        f"complete={rec.complete()}"
                    )
    if problems:
        raise SystemExit("probe-list validation FAILED:\n" + "\n".join(problems))

    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M")
    out_dir = RUNS / f"{stamp}-e7-probe-lists"
    out_dir.mkdir(parents=True, exist_ok=True)

    other = {"flat": "dirty", "dirty": "flat"}
    for c in clusters:
        images = []

        def add(picks, cell, membership, style):
            for a, s, r in picks:
                images.append(
                    dict(
                        artist=a,
                        stem=s,
                        cell=cell,
                        membership=membership,
                        style=style,
                        manifest_redundancy=round(r, 4),
                    )
                )

        add(cells[c]["S1"], "S1", "train", c)
        add(cells[c]["S2"], "S2", "holdout", c)
        add(cells[c]["R"], "S3s", "reserved", c)
        add(cells[other[c]]["R"], "S3x", "reserved", other[c])
        payload = dict(
            adapter=f"output/ckpt/anima_soup_e7_{c}.safetensors",
            manifest=str(MANIFEST.relative_to(REPO_ROOT)),
            seed=m["seed"],
            rule=__doc__.split("Selection", 1)[1].split("Usage")[0].strip(),
            images=images,
        )
        path = out_dir / f"probe_list_{c}.json"
        path.write_text(json.dumps(payload, indent=1))
        reds = [i["manifest_redundancy"] for i in images]
        print(f"{c}: {len(images)} images -> {path}")
        for cell in ("S1", "S2", "S3s", "S3x"):
            sub = [i for i in images if i["cell"] == cell]
            rs = [i["manifest_redundancy"] for i in sub]
            arts = len({i["artist"] for i in sub})
            print(
                f"  {cell}: n={len(sub)} artists={arts} "
                f"redundancy {min(rs):.3f}..{max(rs):.3f} "
                f"median {sorted(rs)[len(rs) // 2]:.3f}"
            )
        print(f"  total redundancy {min(reds):.3f}..{max(reds):.3f}")
    # S1/S2 match quality
    for c in clusters:
        d = [abs(a[2] - b[2]) for a, b in zip(cells[c]["S1"], cells[c]["S2"])]
        print(f"{c}: S1-S2 |redundancy delta| mean {sum(d) / len(d):.4f} "
              f"max {max(d):.4f}")
    print("out:", out_dir)


if __name__ == "__main__":
    main()
