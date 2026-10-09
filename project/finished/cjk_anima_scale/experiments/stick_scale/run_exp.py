#!/usr/bin/env python
"""stick_scale — a trained arm's shared direction rescaled, the per-row residual kept (2026-10-03)

The trained rows are a long shared mean per family plus a per-row residual
of near-even length. ``shared_dir`` showed a trained update's mean is "more
glyphs in the region" (dup up) and Stage B that the donors' mean moves
composition with the repeats; ``delta_scale`` shrank the whole Δ (reads fell,
the line lengthened). Here only the mean moves: every trained row of a
family becomes ``row + (s − 1) · m_family``; every other row stays. No
training.

``--rows`` picks the rows:

- ``seed`` (``st0``): ``seed_retrain_0930``, the rows its chain trained
  (moved against the 0921 seed) — kana 167 (|m| 147, residuals 206) and
  kanji 1 185 (|m| 143, residuals 196), the two means at cos 0.78; s ∈
  0.75 / 0.5 / 0.25 / 0, paired against the seed's routed floor (s = 1,
  cached).
- ``anchor``: ``reseed_anchor_cold_kana_anchor``, its 166 cold kana rows
  (|m| 137, residuals 213) over the 0921 seed — rendered as trained: a row
  the 0930 chain added and the arm lacks sits at its pack row (Δ 0), as the
  arm's own eval has it. s ∈ 0.9 / 1.0 / 1.1 on the kana-only keys (the
  arm trained no kanji), paired against s = 1.0, which renders here (no
  ``sent`` cache for the arm).

``shared_dir``'s ``Rows`` (``sigma_split``'s Splitter with the ext rows
swapped per encode), one row set at every σ, the ``sent`` ruler's seed-0
keys, ``sigma_split``'s reads, placement and sheets (EN ref | the 0930
floor | arms).

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      make daemon-run ARGS="project/cjk_anima_scale/experiments/stick_scale/run_exp.py \\
      --label st0"                               # --rows seed
      … --label sta0 --rows anchor
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, SEED_ROWS, SEED_ROWS_0921, bootstrap, load_experiment  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

NAME = "stick_scale"
SOURCES = {
    # rows → (trained.pt, scales, reference arm, kana-only keys, families' row counts)
    "seed": (
        SEED_ROWS,
        (0.75, 0.5, 0.25, 0.0),
        "floor",
        False,
        {"kana": 167, "kanji": 1185},
    ),
    "anchor": (
        OUT / "experiments" / "reseed_anchor_cold_kana_anchor" / "trained.pt",
        (0.9, 1.0, 1.1),
        "s100",
        True,
        {"kana": 166},
    ),
}


def _tag(s: float) -> str:
    return f"s{int(round(s * 100)):03d}"


def _is_kana(t: str) -> bool:
    return all(0x3040 <= ord(c) < 0x3100 for c in t)


def rescaled(src: Path, scales: tuple, fam_n: dict) -> tuple[dict, dict]:
    """Absolute rows in the seed's ext layout: ``seed`` (the Splitter's own,
    for the plumbing check) and one set per ``s``."""
    import torch
    import unicodedata as ud

    from library.anima.vocab_pack import load_vocab_pack
    from library.env import default_checkpoints
    from library.inference.text import ensure_text_strategies
    from probe.merge_tables import row_texts

    SD = load_experiment("shared_dir")
    seed, sd = SD.abs_rows(SEED_ROWS)
    ids = [int(e) for e in sd["ext_ids"]]
    pos = {e: i for i, e in enumerate(ids)}
    arm, ad = SD.abs_rows(src)
    rows = torch.zeros_like(seed)  # a row the arm lacks: its pack row
    for i, e in enumerate(ad["ext_ids"]):
        rows[pos[int(e)]] = arm[i]
    old, od = SD.abs_rows(SEED_ROWS_0921)
    o = {int(e): old[i] for i, e in enumerate(od["ext_ids"])}
    have = {int(e) for e in ad["ext_ids"]}
    moved = [
        pos[e]
        for e in have
        if e not in o or (rows[pos[e]] - o[e]).norm() > SD.MOVED_REL * o[e].norm()
    ]
    pack_path = os.environ["ANIMA_VOCAB_PACK"]
    tok, _ = ensure_text_strategies(
        default_checkpoints().text_encoder, vocab_pack=pack_path
    )
    text = row_texts(tok, load_vocab_pack(pack_path), [ids[i] for i in moved])
    fam: dict = {k: [] for k in fam_n}
    for i in moved:
        t = text.get(ids[i], "")
        if len(t) == 1 and 0x3040 <= ord(t) < 0x3100:
            fam["kana"].append(i)
        elif (
            len(t) == 1 and ud.name(t, "").startswith("CJK UNIFIED") and "kanji" in fam
        ):
            fam["kanji"].append(i)
    got = {k: len(v) for k, v in fam.items()}
    assert got == fam_n, got
    means = {k: rows[v].mean(0) for k, v in fam.items()}
    sets = {"seed": seed}
    for s in scales:
        x = rows.clone()
        for k, v in fam.items():
            x[v] += (s - 1) * means[k]
        sets[_tag(s)] = x
    stats = {
        k: {
            "rows": len(v),
            "row_norm": round(float(rows[v].norm(dim=1).mean()), 2),
            "mean_norm": round(float(means[k].norm()), 2),
            "residual_norm": round(float((rows[v] - means[k]).norm(dim=1).mean()), 2),
            "row_norm_at": {
                _tag(s): round(float(sets[_tag(s)][v].norm(dim=1).mean()), 2)
                for s in scales
            },
        }
        for k, v in fam.items()
    }
    if "kanji" in means:
        stats["mean_cos_kana_kanji"] = round(
            float(
                torch.nn.functional.cosine_similarity(
                    means["kana"], means["kanji"], dim=0
                )
            ),
            4,
        )
    return sets, stats


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--label", required=True)
    p.add_argument("--rows", choices=list(SOURCES), default="seed")
    p.add_argument(
        "--switch", type=float, default=0.8
    )  # Rows.render's; one set both sides
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    src, scales, ref, kana_only, fam_n = SOURCES[args.rows]
    sets, stats = rescaled(src, scales, fam_n)
    print(f"{src}: {json.dumps(stats)}", flush=True)
    SD = load_experiment("shared_dir")
    SS = load_experiment("sigma_split")
    items = [
        it
        for it in SS.floor_items()
        if it["seed"] == 0 and (not kana_only or _is_kana(it["text"]))
    ]
    arms = [_tag(s) for s in scales]
    print(f"{len(arms)} arms × {len(items)} renders, vs {ref}: {arms}", flush=True)
    if args.dry_run:
        return
    root = OUT / "experiments" / f"{NAME}_{src.parent.name}"
    rw = SD.Rows(SS, sets)
    chk = root / "check.png"  # plumbing: the seed rows vs the floor cache of record
    chk.unlink(missing_ok=True)
    rw.render(chk, items[0], "seed", "seed", args.switch)
    import numpy as np
    from PIL import Image

    check_d = float(
        np.abs(
            np.asarray(Image.open(chk), np.float32)
            - np.asarray(Image.open(items[0]["floor_file"]), np.float32)
        ).mean()
    )
    print(f"check: mean |Δpx| {check_d:.3f}", flush=True)
    t0 = time.time()
    manifests = {}
    for a in arms:
        manifests[a] = []
        for n, it in enumerate(items):
            fn = (
                root
                / a
                / "img"
                / f"{a}_p{it['pi']:02d}_{it['text']}_{it['clause']}_s{it['seed']}.png"
            )
            rw.render(fn, it, a, a, args.switch)
            manifests[a].append(
                {
                    k: it[k]
                    for k in ("seed", "pi", "prompt", "text", "clause", "caption")
                }
                | {"file": str(fn), "cond": a}
            )
            if n % 32 == 0:
                print(
                    f"  {a} {n}/{len(items)} · {(time.time() - t0) / 60:.1f} min",
                    flush=True,
                )
    device = rw.sp.device
    rw.sp.free()
    for a in arms:
        SS.read_arm(manifests[a], root / a, device)

    from cjk_scale import reads as R

    chars = sorted({it["text"] for it in items})
    keys = {(it["text"], it["clause"], it["pi"], it["seed"]) for it in items}
    hits = {
        "floor": {
            k: v
            for k, v in R.hits(SS.FLOOR_READS, chars, SS.CLAUSE).items()
            if k in keys
        }
    }
    recs = {
        "floor": [
            m
            for m in json.loads(SS.FLOOR_READS.read_text("utf-8"))
            if (m["text"], m["clause"], m["pi"], m["seed"]) in keys
        ]
    }
    for a in arms:
        f = root / a / "native_reads.json"
        recs[a] = json.loads(f.read_text("utf-8"))
        hits[a] = R.hits(f, chars, SS.CLAUSE)
    metrics: dict = {
        "rows": args.rows,
        "src": str(src),
        "split": stats,
        "keys": len(items),
        "reference": ref,
        "check_mean_abs_px": check_d,
    }
    for name in ("floor", *arms):
        metrics[name] = {"tally": R.tally(hits[name])["total"]["words"]}
        if name != ref:
            metrics[name][f"vs_{ref}"] = R.paired(hits[name], hits[ref])
        print(f"===== {name} {metrics[name]}", flush=True)
    place = {}
    for name, ms in recs.items():
        ps = [SS.placement(m) for m in ms]
        place[name] = {
            k: round(st.mean(q[k] for q in ps), 4)
            for k in ("box", "box_h", "flat_white")
        } | {
            k: round(st.mean(m[k] for m in ms if m.get(k) is not None), 4)
            for k in ("en_cos", "en_cos_out", "box_iou")
        }
        print(f"  placement {name:<6} {place[name]}", flush=True)
    metrics["placement"] = place
    SS.sheets(items, arms, recs, root / "sheets")
    run_dir = make_run_dir(
        NAME, label=args.label, root=LINE / "experiments" / NAME / "results"
    )
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(root)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
