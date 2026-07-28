"""Phase 0 — aligned-pair census for the instruction-edit arm (easycontrol_request.md).

The aligned-pair arm's blocker is a number nobody recorded for sanitize: how
many near-twin pairs the corpus actually yields per discriminator slice. This
census answers it BEFORE any training:

    1. one shared gather → same-size prune → PE-Spatial embed → Stage-A/B
       twin match over the whole crawl corpus (sanitize's gates: sim_min 0.85,
       match_frac_min 0.3, cell_match_min 0.9),
    2. every accepted twin is then scored against ALL four slices at once
       (text/bubbles, clothing/state, expression, object) with **per-tag
       pivot** semantics — a slice tag counts when it appears in exactly one
       member, i.e. exactly when it lands in the delta caption. The stricter
       sanitize `tag_any` semantics (other member carries NONE of the set) is
       recorded alongside for comparison,
    3. delta captions are generated per the subject_edit contract (additions
       in target caption order + `-`-prefixed removals; direction doubling =
       2 examples per twin) and their length distribution reported vs the
       `max_delta` band,
    4. member saturation histogram (the identity_saturation_min 0.4 context:
       sanitize's mined targets skewed 94% pale),
    5. top-20 spot-check sheets per slice (side-by-side JPEG + index.html
       with pivots + delta caption) for the eyeball pass.

Verdict line: total usable pairs post-doubling (union over slices) vs the
~600 floor — below it, stop and reassess (widen slices vs teacher synthesis).

Usage (GPU work → daemon command job, not a bare background shell):
    uv run python project/directedit_ec/bench/run_pair_census.py
    uv run python project/directedit_ec/bench/run_pair_census.py \
        --limit-artists 20 --spotcheck-n 5          # cheap smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image  # noqa: E402

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from easycontrol_adapters.tools.near_twins.engine import (  # noqa: E402
    _mean_saturation,
    keep_size_cohabiting,
)
from easycontrol_adapters.tools.near_twins.__main__ import expand_path  # noqa: E402
from library.vision.pe_features import (  # noqa: E402
    Member,
    embed_members,
    gather_members,
    normalize_tag,
)
from library.vision.pe_matching import match_grids  # noqa: E402

RESULTS_ROOT = REPO_ROOT / "project" / "directedit_ec" / "bench" / "results"

# Discriminator slices (easycontrol_request.md §Data pairs, priority order).
# text_bubbles is sanitize's [staging] tag_any list verbatim; the other three
# were drawn from the actual crawl-corpus tag frequencies (2026-07-26, 16,053
# captions) — every tag below occurs, most in the hundreds-to-thousands.
SLICES: dict[str, list[str]] = {
    "text_bubbles": [
        "speech bubble",
        "thought bubble",
        "blank speech bubble",
        "spoken text",
        "english text",
        "japanese text",
        "sound effects",
        "text focus",
        "zzz",
        "company name",
        "character name",
    ],
    "clothing_state": [
        "alternate costume",
        "clothes lift",
        "shirt lift",
        "skirt lift",
        "undressing",
        "dressing",
        "open jacket",
        "open shirt",
        "open fly",
        "unbuttoned",
        "off shoulder",
        "nude",
        "topless",
        "bottomless",
        "underwear only",
        "no bra",
        "no panties",
        "naked shirt",
        "partially undressed",
        "clothes down",
        "barefoot",
    ],
    "expression": [
        "smile",
        "grin",
        "smirk",
        "frown",
        "pout",
        "open mouth",
        "closed mouth",
        "parted lips",
        "closed eyes",
        "one eye closed",
        "wink",
        "half-closed eyes",
        "blush",
        "tears",
        "crying",
        "angry",
        "embarrassed",
        "surprised",
        "nervous",
        "expressionless",
        "tongue out",
        "clenched teeth",
        "sweatdrop",
        "heart-shaped pupils",
        "ahegao",
        "drooling",
    ],
    "object": [
        "holding",
        "hat",
        "glasses",
        "bag",
        "umbrella",
        "weapon",
        "sword",
        "gun",
        "cup",
        "phone",
        "cellphone",
        "food",
        "flower",
        "book",
        "headphones",
        "mask",
        "scarf",
        "gloves",
        "earrings",
        "necklace",
        "hairband",
        "hair ribbon",
        "hair bow",
        "crown",
        "halo",
        "cigarette",
        "microphone",
        "stuffed toy",
        "teddy bear",
        "drink",
        "bottle",
    ],
}




def _default_image_dirs() -> str:
    import os

    root = os.environ.get("CAPTION_CORPUS_DIR")
    base = Path(root) if root else Path.home() / "gelcrawl"
    return str(base / "retrieved")


def read_ordered_tags(txt_path: Path) -> list[str]:
    """Caption sidecar → ordered raw tag list (display form, order preserved)."""
    if not txt_path.is_file():
        return []
    raw = txt_path.read_text(encoding="utf-8", errors="ignore")
    return [t.strip() for t in raw.split(",") if t.strip()]


def delta_caption(src: list[str], dst: list[str]) -> list[str]:
    """subject_edit contract: additions in the target's caption order, then
    `-`-prefixed removals in the source's order; matching is normalized."""
    src_n = {normalize_tag(t) for t in src}
    dst_n = {normalize_tag(t) for t in dst}
    additions = [t for t in dst if normalize_tag(t) not in src_n]
    removals = [t for t in src if normalize_tag(t) not in dst_n]
    return additions + [f"-{t}" for t in removals]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase-0 aligned-pair census (all slices, one pass)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--image-dirs",
        default=_default_image_dirs(),
        help="comma-separated <dir>/<artist>/<id>.<ext> trees",
    )
    p.add_argument("--artists", default=None, help="comma-separated allowlist")
    p.add_argument(
        "--limit-artists",
        type=int,
        default=0,
        help="cap artist count for a smoke run (0=all)",
    )
    # Twin gates — sanitize [staging] values, the spec's alignment definition.
    p.add_argument("--sim-min", type=float, default=0.85)
    p.add_argument("--grid", type=int, default=7)
    p.add_argument("--cell-match-min", type=float, default=0.9)
    p.add_argument("--ratio", type=float, default=0.8)
    p.add_argument("--match-frac-min", type=float, default=0.3)
    p.add_argument(
        "--geom-check",
        action="store_true",
        help="RANSAC translation pass (sanitize ran without it)",
    )
    p.add_argument(
        "--max-extra-diff",
        type=int,
        default=25,
        help="cap on differing tags outside the slice's list",
    )
    p.add_argument(
        "--max-delta",
        type=int,
        default=24,
        help="report-only instruction-length cap (subject_edit's)",
    )
    p.add_argument("--spotcheck-n", type=int, default=20)
    p.add_argument("--spotcheck-max-per-artist", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--label", default="pair-census")
    return p.parse_args(argv)


def census_pair(
    rec: dict, tags_a: list[str], tags_b: list[str], max_extra: int
) -> dict:
    """Attach slice memberships + delta captions to an accepted twin record."""
    norm_a = {normalize_tag(t) for t in tags_a}
    norm_b = {normalize_tag(t) for t in tags_b}
    sym = norm_a ^ norm_b
    slices: dict[str, dict] = {}
    for name, tag_list in SLICES.items():
        slice_set = {normalize_tag(t) for t in tag_list}
        pivots = sorted(sym & slice_set)
        if not pivots:
            continue
        extra = len(sym - slice_set)
        strict_any = bool(slice_set & norm_a) != bool(slice_set & norm_b)
        if extra <= max_extra:
            slices[name] = {"pivots": pivots, "extra": extra, "strict_any": strict_any}
    rec["slices"] = slices
    rec["sym_delta"] = len(sym)
    rec["delta_a_to_b"] = ", ".join(delta_caption(tags_a, tags_b))
    rec["delta_b_to_a"] = ", ".join(delta_caption(tags_b, tags_a))
    return rec


def spotcheck_export(
    run_dir: Path, pairs: list[dict], members: dict[str, Member], args
) -> list[str]:
    """Per slice: top-N cleanest pairs as side-by-side JPEG + an index.html."""
    out_paths: list[str] = []
    base = run_dir / "spotcheck"
    for name in SLICES:
        ranked = sorted(
            (p for p in pairs if name in p["slices"]),
            key=lambda p: (p["slices"][name]["extra"], -p["cosine"]),
        )
        picked, per_artist = [], Counter()
        for p in ranked:
            if per_artist[p["artist"]] >= args.spotcheck_max_per_artist:
                continue
            per_artist[p["artist"]] += 1
            picked.append(p)
            if len(picked) >= args.spotcheck_n:
                break
        if not picked:
            continue
        sdir = base / name
        sdir.mkdir(parents=True, exist_ok=True)
        rows = []
        for rank, p in enumerate(picked):
            ma, mb = members[p["a"]], members[p["b"]]
            try:
                ims = []
                for m in (ma, mb):
                    with Image.open(m.image_path) as im:
                        im = im.convert("RGB")
                        im.thumbnail((10_000, 640))
                        ims.append(im.copy())
                sheet = Image.new(
                    "RGB",
                    (
                        ims[0].width + ims[1].width + 8,
                        max(ims[0].height, ims[1].height),
                    ),
                    (32, 32, 32),
                )
                sheet.paste(ims[0], (0, 0))
                sheet.paste(ims[1], (ims[0].width + 8, 0))
                jpg = f"{rank:02d}_{p['artist']}_{p['a']}-{p['b']}.jpg"
                sheet.save(sdir / jpg, quality=88)
            except Exception as e:  # noqa: BLE001 — one bad decode ≠ dead census
                print(
                    f"  [warn] spotcheck decode failed {p['a']}: {e}", file=sys.stderr
                )
                continue
            info = p["slices"][name]
            rows.append(
                f"<div><h3>#{rank} {p['artist']} {p['a']} | {p['b']} "
                f"(cos {p['cosine']:.3f}, match {p['match_frac']:.2f}, "
                f"extra {info['extra']})</h3>"
                f"<p>pivots: <b>{', '.join(info['pivots'])}</b><br>"
                f"A→B: {p['delta_a_to_b'] or '(empty)'}<br>"
                f"B→A: {p['delta_b_to_a'] or '(empty)'}</p>"
                f'<img src="{jpg}" style="max-width:100%"></div>'
            )
        (sdir / "index.html").write_text(
            "<meta charset=utf-8><body style='background:#222;color:#ddd;"
            "font-family:sans-serif'>"
            f"<h1>{name} — {len(picked)} of "
            f"{sum(1 for p in pairs if name in p['slices'])} pairs</h1>"
            + "\n".join(rows),
            encoding="utf-8",
        )
        out_paths.append(f"spotcheck/{name}/index.html")
    return out_paths


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    args = parse_args()
    start_heartbeat()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    image_dirs = [Path(expand_path(d)) for d in args.image_dirs.split(",") if d.strip()]
    artists_filter = (
        {a.strip() for a in args.artists.split(",")} if args.artists else None
    )
    print(f"Gathering members from {image_dirs} …", flush=True)
    by_artist = gather_members(image_dirs, artists_filter)
    if args.limit_artists:
        by_artist = dict(sorted(by_artist.items())[: args.limit_artists])
    n_total = sum(len(v) for v in by_artist.values())

    # Same-size prune only (no tag pivot — every slice is judged post-match).
    pairable = {
        a: kept
        for a, m in by_artist.items()
        if len(kept := keep_size_cohabiting(m)) >= 2
    }
    n_pairable = sum(len(v) for v in pairable.values())
    print(
        f"{len(by_artist)} artists, {n_total} images; same-size gate → "
        f"{n_pairable} embeddable in {len(pairable)} artists",
        flush=True,
    )

    from library.vision import load_pe_encoder

    print(f"Loading PE-Spatial-B16-512 on {device} …", flush=True)
    bundle = load_pe_encoder(torch.device(device), name="pe_spatial")

    members_by_stem: dict[str, Member] = {}
    pairs: list[dict] = []
    n_stage_a = n_twins = 0
    for k, (artist, members) in enumerate(sorted(pairable.items())):
        feats = embed_members(bundle, members, args.batch_size, args.num_workers)
        stems = [m.stem for m in members if m.stem in feats]
        idx = {m.stem: m for m in members}
        tag_cache = {s: read_ordered_tags(idx[s].txt_path) for s in stems}
        import numpy as np

        cls = np.stack([feats[s].cls for s in stems]) if stems else None
        found = 0
        for i in range(len(stems)):
            for j in range(i + 1, len(stems)):
                si, sj = stems[i], stems[j]
                if idx[si].wh != idx[sj].wh:
                    continue
                cos = float(cls[i] @ cls[j])
                if cos < args.sim_min:
                    continue
                n_stage_a += 1
                match = match_grids(
                    feats[si],
                    feats[sj],
                    args.grid,
                    args.cell_match_min,
                    args.ratio,
                    args.geom_check,
                )
                if match.match_frac < args.match_frac_min:
                    continue
                n_twins += 1
                if not tag_cache[si] or not tag_cache[sj]:
                    continue  # twin without both captions — uncountable
                rec = census_pair(
                    {
                        "artist": artist,
                        "a": si,
                        "b": sj,
                        "cosine": cos,
                        "match_frac": round(match.match_frac, 4),
                    },
                    tag_cache[si],
                    tag_cache[sj],
                    args.max_extra_diff,
                )
                if rec["slices"]:
                    pairs.append(rec)
                    members_by_stem[si] = idx[si]
                    members_by_stem[sj] = idx[sj]
                found += 1
        if found:
            print(
                f"  [{k + 1}/{len(pairable)}] {artist}: {found} twin(s), "
                f"{sum(1 for p in pairs if p['artist'] == artist)} sliced",
                flush=True,
            )

    # ---- aggregation --------------------------------------------------------
    slice_counts = {
        name: {
            "pairs": sum(1 for p in pairs if name in p["slices"]),
            "pairs_strict_any": sum(
                1 for p in pairs if p["slices"].get(name, {}).get("strict_any")
            ),
            "artists": len({p["artist"] for p in pairs if name in p["slices"]}),
            "top_pivots": Counter(
                t
                for p in pairs
                if name in p["slices"]
                for t in p["slices"][name]["pivots"]
            ).most_common(15),
        }
        for name in SLICES
    }
    usable = len(pairs)  # union over slices, deduped by construction
    post_double = 2 * usable

    delta_lens = sorted(
        len([t for t in p["delta_a_to_b"].split(",") if t.strip()]) for p in pairs
    )
    len_stats = {}
    if delta_lens:
        len_stats = {
            "median": delta_lens[len(delta_lens) // 2],
            "p90": delta_lens[int(len(delta_lens) * 0.9)],
            "max": delta_lens[-1],
            "frac_within_max_delta": round(
                sum(1 for n in delta_lens if n <= args.max_delta) / len(delta_lens), 4
            ),
            "hist": Counter(min(d // 5 * 5, 40) for d in delta_lens).most_common(),
        }

    print("Computing member saturation histogram …", flush=True)
    sats = [_mean_saturation(m.image_path) for m in members_by_stem.values()]
    sat_hist = Counter(round(min(s, 0.999) * 10) / 10 for s in sats)
    sat_metrics = {
        "n_members": len(sats),
        "frac_below_0.4": round(sum(1 for s in sats if s < 0.4) / len(sats), 4)
        if sats
        else None,
        "hist_0.1_bins": sorted(sat_hist.items()),
    }

    run_dir = make_run_dir("directedit_ec", label=args.label, root=RESULTS_ROOT)
    (run_dir / "pairs_manifest.json").write_text(
        json.dumps(
            {"slices": {k: v for k, v in SLICES.items()}, "pairs": pairs},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    artifacts = ["pairs_manifest.json"]
    artifacts += spotcheck_export(run_dir, pairs, members_by_stem, args)

    verdict = "PASS" if post_double >= 600 else "FAIL"
    metrics = {
        "n_images": n_total,
        "n_embeddable": n_pairable,
        "n_stage_a_pairs": n_stage_a,
        "n_twins": n_twins,
        "n_usable_pairs": usable,
        "n_post_doubling": post_double,
        "volume_floor": 600,
        "verdict": verdict,
        "per_slice": slice_counts,
        "delta_length": len_stats,
        "saturation": sat_metrics,
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        artifacts=artifacts,
        label=args.label,
    )
    print(json.dumps(metrics, indent=1, default=str))
    print(
        f"\n[census] {usable} usable twins → {post_double} post-doubling "
        f"(floor 600) → {verdict}   → {run_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
