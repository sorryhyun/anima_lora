#!/usr/bin/env python3
"""plan_render S0c — panel samples → a trainer dataset tree.

Reads the S0b outputs (``output/render/<name>/samples_<ed>.jsonl`` +
``bubbles_<ed>.jsonl`` + ``boxes_<ed>.jsonl`` + ``reads_<ed>.jsonl``), crops
each panel out of the corpus page, resizes it onto the **768 tier** free-fit
band (``anime_tools.buckets.freefit_bucket`` — native aspect, patch-grid
token count inside the band; never 512, § 512-is-not-a-text-resolution), and
writes::

    <out>/<ed>/resized/<artist>/<work>/<work>_<page>_p<k>.png   the target
    <out>/<ed>/resized/<artist>/<work>/<work>_<page>_p<k>.txt   the caption
    <out>/<ed>/resized/boxes.jsonl                              holes + intact boxes, resized coords
    <out>/<ed>/heldout/...                                      the same for the held-out artists

Caption = the fixed minimal bag (``manga, speech bubble, japanese text`` /
``english text``) + one text clause carrying the holed bubbles' lines in
reading order (plan decision 4). JA uses ``anime_tools``' text clause
grammar verbatim; EN composes the same shape with an ``English text reads
as`` header locally (``TEXT_PREFIXES`` is JA-hardcoded on the pinned tag).

Gate (plan decision 3): a sample whose smallest holed box has a glyph height
under ``--min_glyph`` px after the resize is dropped — the proxy is the box's
thickness (height for rows, width for columns) over the read's line count,
because the detector gives blocks, not lines. Reported per edition.

Split (decision 8): ``--heldout`` artist dirs, the same across editions,
drawn by a seeded pick from the middle third of artists by sample count so
neither a giant nor a four-page artist is the test set. Artist names go on
disk only (gitignored tree) — never into a doc.

CPU only. Run from the repo root::

    python project/cjk_aware_anima_dit/render/cut.py --corpus <root> --editions en ja
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import corpus_boxes as cb  # noqa: E402

REPO = cb.REPO
BAG = {
    "en": ("manga", "speech bubble", "english text"),
    "ja": ("manga", "speech bubble", "japanese text"),
    "ko": ("manga", "speech bubble", "korean text"),
}
HEADER = {"en": "English text reads as ", "ko": "Korean text reads as "}


def caption_for(ed: str, lines: list[str]) -> str:
    from anime_tools.captions import position_clauses as pc

    if ed == "ja":
        return pc.compose_caption(BAG[ed], [pc.text_clause(lines)])
    # EN / KO: the JA grammar's shape with the language in the header. Built
    # by hand because PositionClause only renders a text clause for a prefix in
    # TEXT_PREFIXES (JA on the pinned tag) — same quoting rule, same layout.
    tags = ", ".join(f'"{ln.replace(chr(34), "”")}"' for ln in lines if ln)
    return f"{', '.join(BAG[ed])}. {HEADER[ed]}{tags}."


def join_lines(ed: str, lines: list[str]) -> str:
    """One bubble's member-box reads as one caption line — EN with a space,
    JA / KO with nothing — with adjacent duplicates (the reader emitting a
    line twice) collapsed."""
    out: list[str] = []
    for ln in lines:
        if ln and (not out or out[-1] != ln):
            out.append(ln)
    return (" " if ed == "en" else "").join(out)


def pick_heldout(counts: dict[str, int], n: int, seed: int) -> list[str]:
    ranked = sorted(counts, key=lambda a: counts[a])
    third = len(ranked) // 3
    middle = ranked[third : len(ranked) - third] or ranked
    return sorted(random.Random(seed).sample(middle, min(n, len(middle))))


def _scale_box(b, x0: int, y0: int, s: float, ox: int, oy: int, W: int, H: int):
    return [
        max(0, min(W, int(round((b[0] - x0) * s)) - ox)),
        max(0, min(H, int(round((b[1] - y0) * s)) - oy)),
        max(0, min(W, int(round((b[2] - x0) * s)) - ox)),
        max(0, min(H, int(round((b[3] - y0) * s)) - oy)),
    ]


def run(a) -> None:
    import cv2
    from anime_tools.buckets import freefit_band_for_edge, freefit_bucket

    root = cb.corpus_root(a.corpus)
    band = freefit_band_for_edge(a.edge)
    src = cb.out_dir(a.name)
    # held-out artists: the same across editions, from the combined counts
    counts: collections.Counter = collections.Counter()
    samples = {}
    for ed in a.editions:
        samples[ed] = cb.load_jsonl(src / f"samples_{ed}.jsonl", key=None)
        counts.update(s["rel"].split("/")[1] for s in samples[ed])
    heldout = set(pick_heldout(counts, a.heldout, a.seed))
    print(
        f"held-out: {len(heldout)} of {len(counts)} artist dirs "
        f"({sum(counts[h] for h in heldout)} of {sum(counts.values())} samples)",
        flush=True,
    )

    for ed in a.editions:
        bubbles = {
            (b["rel"], b["k"]): b
            for b in cb.load_jsonl(src / f"bubbles_{ed}.jsonl", key=None)
        }
        boxes = cb.load_jsonl(src / f"boxes_{ed}.jsonl")
        reads = cb.load_jsonl(src / f"reads_{ed}.jsonl")
        out = Path(a.out) / ed
        writers = {}
        for split in ("resized", "heldout"):
            (out / split).mkdir(parents=True, exist_ok=True)
            writers[split] = (out / split / "boxes.jsonl").open("w", encoding="utf-8")
        n = collections.Counter()
        glyphs: list[float] = []
        t0 = time.time()
        page_cache: tuple[str, object] | None = None
        for s in sorted(samples[ed], key=lambda s: (s["rel"], s["k"])):
            artist, work = s["rel"].split("/")[1:3]
            page = Path(s["rel"]).stem
            split = "heldout" if artist in heldout else "resized"
            stem = f"{work}_{page}_p{s['k']}"
            dst = out / split / artist / work / f"{stem}.png"
            if dst.exists() and not a.overwrite:
                n["skip"] += 1
                continue
            if page_cache is None or page_cache[0] != s["rel"]:
                page_cache = (s["rel"], cb.read_bgr(root / s["rel"]))
            bgr = page_cache[1]
            if bgr is None:
                n["unreadable"] += 1
                continue
            x0, y0, x1, y1 = s["panel"]
            crop = bgr[y0:y1, x0:x1]
            h, w = crop.shape[:2]
            if min(h, w) < 16:
                n["degenerate"] += 1
                continue
            W, H = freefit_bucket(w, h, band)
            sc = max(W / w, H / h)  # cover: exact when the ratio clamp did not fire
            rw, rh = max(W, int(round(w * sc))), max(H, int(round(h * sc)))
            resized = cv2.resize(
                crop,
                (rw, rh),
                interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC,
            )
            ox, oy = (rw - W) // 2, (rh - H) // 2
            if max(rw - W, rh - H) >= 16:  # beyond free-fit's sub-patch residual
                n["ratio_clamped"] += 1
            resized = resized[oy : oy + H, ox : ox + W]

            # holed bubbles: lines in reading order, member boxes scaled, glyph gate
            holes, bubs, lines, gmin = [], [], [], 1e9
            page_boxes = boxes[s["rel"]]["boxes"]
            page_texts = reads[s["rel"]]["texts"]
            for k in s["bubbles"]:
                b = bubbles[(s["rel"], k)]
                line = join_lines(ed, b["lines"])
                if not line:
                    continue
                member = []
                for bx in b["boxes"]:
                    j = page_boxes.index(bx)
                    n_lines = max(
                        1, len([t for t in page_texts[j].splitlines() if t.strip()])
                    )
                    thick = (bx[2] - bx[0]) if b["column"] else (bx[3] - bx[1])
                    gmin = min(gmin, thick * sc / n_lines)
                    member.append(_scale_box(bx, x0, y0, sc, ox, oy, W, H))
                holes += member
                bubs.append(
                    {
                        "k": k,
                        "box": _scale_box(b["box"], x0, y0, sc, ox, oy, W, H),
                        "boxes": member,
                        "line": line,
                        "column": b["column"],
                    }
                )
                lines.append(line)
            if not lines:
                n["no_lines"] += 1
                continue
            glyphs.append(gmin)
            if gmin < a.min_glyph:
                n["drop_glyph"] += 1
                continue
            intact = [
                {
                    "k": it["k"],
                    "box": _scale_box(it["box"], x0, y0, sc, ox, oy, W, H),
                    "reason": it["reason"],
                }
                for it in s.get("intact", [])
            ]
            caption = caption_for(ed, lines)
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), resized)
            dst.with_suffix(".txt").write_text(caption, encoding="utf-8")
            cb.append_jsonl(
                writers[split],
                {
                    "rel": f"{artist}/{work}/{stem}.png",
                    "stem": stem,
                    "size": [W, H],
                    "holes": holes,
                    "bubbles": bubs,
                    "intact": intact,
                    "caption": caption,
                    "src": s["rel"],
                    "panel": s["panel"],
                    "scale": round(sc, 4),
                    "glyph_min": round(gmin, 1),
                },
            )
            n[split] += 1
            if a.limit and n["resized"] + n["heldout"] >= a.limit:
                break
        for fh in writers.values():
            fh.close()
        glyphs.sort()
        q = [glyphs[int(len(glyphs) * p)] for p in (0.1, 0.5, 0.9)] if glyphs else []
        print(
            f"[{ed}] cut: {n['resized']} train + {n['heldout']} held-out written, "
            f"{n['drop_glyph']} dropped under {a.min_glyph} px glyphs "
            f"(kept {100 * (n['resized'] + n['heldout']) / max(1, len(samples[ed])):.1f} % "
            f"of {len(samples[ed])}; glyph p10/p50/p90 {[round(x) for x in q]}), "
            f"{n['ratio_clamped']} ratio-clamped, {n['skip']} skipped, "
            f"{n['no_lines'] + n['unreadable'] + n['degenerate']} unusable, "
            f"{time.time() - t0:.0f}s → {out}",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--corpus", default=None, help="corpus root (or ANIMA_RENDER_CORPUS)"
    )
    ap.add_argument("--name", default="s0b", help="S0b output dir under output/render/")
    ap.add_argument("--editions", nargs="+", default=["en", "ja"], choices=cb.EDITIONS)
    ap.add_argument("--out", default="post_image_dataset/render")
    ap.add_argument("--edge", type=int, default=768, help="free-fit tier edge")
    ap.add_argument("--min_glyph", type=float, default=20.0, help="px after resize")
    ap.add_argument("--heldout", type=int, default=2, help="held-out artist dirs")
    ap.add_argument("--limit", type=int, default=0, help="samples per edition (QA)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
