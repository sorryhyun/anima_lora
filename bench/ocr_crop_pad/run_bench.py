#!/usr/bin/env python3
"""OCR reader crop margin: the shipped 0.12 × long side against capped margins.

    make daemon-run ARGS="--queue --stall-timeout 900 bench/ocr_crop_pad/run_bench.py"

``anime_tools.ocr.reader.pad_box`` grows a detector box by ``0.12 × max(w, h)``
on all four sides. On a long thin box (a vertical column, a horizontal line)
that margin is wider than the box's short side and pulls neighbouring lines
into the crop. Arms (every one is the shipped margin, capped):

* ``now``          — ``0.12 · long``
* ``cap12`` / ``cap24`` — ``min(0.12 · long, N px)``
* ``short30``      — ``min(0.12 · long, 0.3 · short)``
* ``short30_cap12`` — both caps

Sets:

* ``sfx`` — the sincos hand-labelled SFX gate (597 rows): exact / sim against
  ``text_hand``. Checks that a smaller margin does not clip strokes.
* ``speech`` — the sincos speech rows; ``text_hand`` is the PP-OCRv6 read of
  the box, so this is agreement, not accuracy. A neighbour in the crop lowers it.
* ``neigh`` — doujin sidecar lines (``post_image_dataset/ocr/*``) whose
  ``now`` margin takes in ≥15 % of another kept line's box. No labels: the
  metric is how often the read contains that neighbour's text (longest common
  run ≥ 3 and ≥ half the neighbour), plus guard drops and read length.

Crops are cut as the shipped reader cuts them (axis-aligned, clipped to the
page); identical crops across arms are read once.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import pandas as pd

from bench._common import make_run_dir, write_result
from library.env import resolve_under_home

LINE = resolve_under_home("project/ocr_reader")
sys.path.insert(0, str(LINE / "ocr"))
import textnorm  # noqa: E402

PAD = 0.12
ARMS = {
    "now": lambda w, h: PAD * max(w, h),
    "cap12": lambda w, h: min(PAD * max(w, h), 12),
    "cap24": lambda w, h: min(PAD * max(w, h), 24),
    "short30": lambda w, h: min(PAD * max(w, h), 0.3 * min(w, h)),
    "short30_cap12": lambda w, h: min(PAD * max(w, h), 0.3 * min(w, h), 12),
}


def padded(box, W, H, grow_fn):
    x0, y0, x1, y1 = (int(v) for v in box)
    g = round(grow_fn(x1 - x0, y1 - y0))
    return (max(0, x0 - g), max(0, y0 - g), min(W, x1 + g), min(H, y1 + g))


def nospace(s: str) -> str:
    return "".join(s.split())


def lcs(a: str, b: str) -> int:
    return (
        SequenceMatcher(None, a, b, autojunk=False)
        .find_longest_match(0, len(a), 0, len(b))
        .size
    )


def sincos_rows():
    df = pd.read_csv(
        LINE / "assets/sfx_labels_sincos_597.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    df = df[(df.status != "unchecked") & (df.text_hand.str.strip() != "")]
    pages = LINE / "eval_data/sincos/pages"
    out = []
    for r in df.itertuples():
        kind = {"sfx": "sfx", "speech": "speech"}.get(r.kind_hand)
        if kind:
            out.append(
                dict(
                    set=kind,
                    key=f"sincos/{r.row}",
                    page=str(pages / f"{r.stem}.png"),
                    box=json.loads(r.box),
                    text=r.text_hand,
                )
            )
    return out


def neighbour_rows(ocr_root: Path, resized_root: Path, max_rows: int):
    from anime_tools.captions.ocr_sidecar import read_ocr

    out = []
    for p in sorted(ocr_root.rglob("*.ocr.txt")):
        rel = p.relative_to(ocr_root)
        stem = p.name[: -len(".ocr.txt")]
        page = next(
            (
                q
                for q in (resized_root / rel.parent).glob(f"{stem}.*")
                if q.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
            ),
            None,
        )
        if page is None:
            continue
        L = [ln for ln in read_ocr(p) if ln.text.strip()]
        for i, ln in enumerate(L):
            x0, y0, x1, y1 = ln.box
            g = round(PAD * max(x1 - x0, y1 - y0))
            P = (x0 - g, y0 - g, x1 + g, y1 + g)
            neigh = []
            for j, m in enumerate(L):
                if j == i:
                    continue
                a0, b0, a1, b1 = m.box
                ix = max(0, min(P[2], a1) - max(P[0], a0))
                iy = max(0, min(P[3], b1) - max(P[1], b0))
                ox = max(0, min(x1, a1) - max(x0, a0))
                oy = max(0, min(y1, b1) - max(y0, b0))
                if ix * iy - ox * oy > 0.15 * (a1 - a0) * (b1 - b0):
                    neigh.append(m.text)
            if neigh:
                out.append(
                    dict(
                        set="neigh",
                        key=f"{rel.parent}/{stem}/{ln.seq}",
                        page=str(page),
                        box=list(ln.box),
                        text="",
                        neigh=neigh,
                    )
                )
    return out[:max_rows] if max_rows else out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_neigh", type=int, default=0)
    args = ap.parse_args()

    from anime_tools.ocr.reader import OcrReader

    rows = sincos_rows() + neighbour_rows(
        resolve_under_home("post_image_dataset/ocr"),
        resolve_under_home("post_image_dataset/resized"),
        args.max_neigh,
    )
    print({s: sum(r["set"] == s for r in rows) for s in ("sfx", "speech", "neigh")})

    # one crop per distinct (page, padded box)
    imgs: dict[str, object] = {}
    uniq: dict[tuple, int] = {}
    crops = []
    for r in rows:
        img = imgs.get(r["page"])
        if img is None:
            img = imgs[r["page"]] = cv2.imread(r["page"])
        H, W = img.shape[:2]
        r["crop"] = {}
        for arm, fn in ARMS.items():
            b = padded(r["box"], W, H, fn)
            k = (r["page"], b)
            if k not in uniq:
                uniq[k] = len(crops)
                crops.append(img[b[1] : b[3], b[0] : b[2]])
            r["crop"][arm] = uniq[k]
    imgs.clear()
    print(f"{len(rows)} rows × {len(ARMS)} arms → {len(crops)} distinct crops")

    reader = OcrReader.load(device="cuda", batch_size=args.batch_size)
    t0 = time.time()
    reads = reader.read_scored(crops)
    wall = time.time() - t0
    print(f"read {len(crops)} crops in {wall:.0f} s")

    recs = []
    for r in rows:
        for arm in ARMS:
            res = reads[r["crop"][arm]]
            pred = res[0] if res else ""
            rec = dict(
                set=r["set"], key=r["key"], arm=arm, pred=pred, dropped=res is None
            )
            if r["set"] == "neigh":
                t = nospace(pred)
                rec["leak"] = any(
                    len(n := nospace(m)) >= 2
                    and t != n
                    and (k := lcs(t, n)) >= 3
                    and k >= 0.5 * len(n)
                    for m in r["neigh"]
                )
                rec["len"] = len(t)
            else:
                rec["exact"] = textnorm.exact_key(pred) == textnorm.exact_key(r["text"])
                rec["exact_noheart"] = textnorm.exact_key(pred).replace(
                    "♡", ""
                ) == textnorm.exact_key(r["text"]).replace("♡", "")
                rec["sim"] = SequenceMatcher(
                    None, nospace(pred), nospace(r["text"]), autojunk=False
                ).ratio()
            recs.append(rec)
    df = pd.DataFrame(recs)

    metrics: dict[str, dict] = {}
    lines = [
        "| arm | sfx exact | sfx exact ♡-blind | sfx sim | speech exact | speech sim | neigh leak | neigh dropped | neigh len |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        s = df[(df.arm == arm) & (df.set == "sfx")]
        sp = df[(df.arm == arm) & (df.set == "speech")]
        ng = df[(df.arm == arm) & (df.set == "neigh")]
        m = dict(
            sfx_n=len(s),
            sfx_exact=int(s.exact.sum()),
            sfx_exact_noheart=int(s.exact_noheart.sum()),
            sfx_sim=float(s.sim.mean()),
            speech_n=len(sp),
            speech_exact=int(sp.exact.sum()),
            speech_sim=float(sp.sim.mean()),
            neigh_n=len(ng),
            neigh_leak=int(ng.leak.sum()),
            neigh_dropped=int(ng.dropped.sum()),
            neigh_len=float(ng.len.mean()),
        )
        metrics[arm] = m
        lines.append(
            f"| {arm} | {m['sfx_exact']}/{m['sfx_n']} | {m['sfx_exact_noheart']} | {m['sfx_sim']:.3f} | "
            f"{m['speech_exact']}/{m['speech_n']} | {m['speech_sim']:.3f} | "
            f"{m['neigh_leak']}/{m['neigh_n']} | {m['neigh_dropped']} | {m['neigh_len']:.1f} |"
        )
    table = "\n".join(lines)
    print(table)

    run_dir = make_run_dir("ocr_crop_pad", args.label)
    df.to_json(run_dir / "reads.jsonl", orient="records", lines=True, force_ascii=False)
    (run_dir / "table.md").write_text(table + "\n")
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["reads.jsonl", "table.md"],
        extra=dict(distinct_crops=len(crops), read_wall_s=wall),
    )
    print(f"→ {run_dir}")


if __name__ == "__main__":
    main()
