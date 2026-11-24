#!/usr/bin/env python3
"""Page mode: a page-level reader over the whole sincos pages, scored as recall
against the hand labels.

    ANIMA_MANGA109S_ROOT=… make daemon-run ARGS="project/ocr_reader/ocr/eval_page.py --reader ovis"

A page reader (OvisOCR2) gives one Markdown document per page with no text
boxes, so a label row cannot be paired with a span of the output. A row counts
as **hit** when its ``exact_key`` is a substring of the ``exact_key`` of the
page's output, with ``<img>`` blocks dropped as the card does. Position is
ignored, so this is an upper bound on what the page read would yield, and is
not comparable to the crop-mode ``exact`` of ``eval_sfx.py``. A short key
(1–2 chars, e.g. ``っ``) can hit by accident inside any other line, so the
table splits rows at key length 3.

``img_hit`` counts the rows that are hit only once the dropped ``<img>`` blocks
are kept (alt text / captions in them), i.e. SFX the parser filed as a picture.

Writes ``reports/eval/ocr_eval_page_<name>.md`` + ``output/ocr/eval/page_<name>.jsonl``
(one row per page, raw output included).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_manga109 as ev  # noqa: E402
import eval_sfx  # noqa: E402
import textnorm  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--reader", choices=["ovis", "ovis3"], default="ovis")
    ap.add_argument("--ckpt")
    ap.add_argument("--name")
    ap.add_argument("--labels", type=Path, default=eval_sfx.LABELS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, help="first N pages (smoke)")
    a = ap.parse_args()
    name = a.name or a.reader

    df = eval_sfx.load_labels(["sfx", "speech"], False, a.labels)
    stems = sorted(df.stem.unique())
    if a.limit:
        stems = stems[: a.limit]
        df = df[df.stem.isin(stems)]
    pages = []
    for s in stems:
        p = eval_sfx.FROZEN / "pages" / f"{s}.png"
        img = cv2.imread(str(p if p.is_file() else eval_sfx.PAGES / f"{s}.png"))
        assert img is not None, s
        pages.append(img)

    reader = ev.READERS[a.reader](a.ckpt, a.device)
    reader.max_tokens = a.max_new_tokens
    reader.filter_imgtags = False
    t0 = time.time()
    raw = []
    bar = tqdm(total=len(pages), unit="page", mininterval=0, file=sys.stdout)
    for s in range(0, len(pages), a.bs):
        chunk = pages[s : s + a.bs]
        raw += reader.read(chunk, [None] * len(chunk), a.bs)
        bar.update(len(chunk))
        bar.set_postfix(out_chars=len(raw[-1]))
    bar.close()
    wall = time.time() - t0

    kept = {s: textnorm.exact_key(reader.drop_imgtags(r)) for s, r in zip(stems, raw)}
    full = {s: textnorm.exact_key(r) for s, r in zip(stems, raw)}
    df = df.copy()
    df["key"] = df.text.map(textnorm.exact_key)
    df["hit"] = [k in kept[s] for k, s in zip(df.key, df.stem)]
    df["img_hit"] = [
        (k in full[s]) and not h for k, s, h in zip(df.key, df.stem, df.hit)
    ]
    df["long"] = df.key.str.len() >= 3

    ev.OUT.mkdir(parents=True, exist_ok=True)
    with (ev.OUT / f"page_{name}.jsonl").open("w", encoding="utf-8") as f:
        for s, r in zip(stems, raw):
            rows = df[df.stem == s]
            f.write(
                json.dumps(
                    dict(
                        stem=s,
                        raw=r,
                        n_img=r.count('<img src="images/bbox_'),
                        hits=rows[rows.hit].row.tolist(),
                        misses=rows[~rows.hit].row.tolist(),
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )

    n_img = sum(r.count('<img src="images/bbox_') for r in raw)
    md = [
        f"# page mode — {name} (sincos, {len(stems)} pages, {wall / 60:.1f} min)\n",
        f"Labels `{a.labels.name}`. hit = label `exact_key` ⊂ page `exact_key` "
        "(position-blind recall; not comparable to crop-mode exact).\n",
        f"`<img>` blocks emitted: {n_img} over {len(stems)} pages.\n",
        "| kind | rows | hit | hit, key ≥ 3 | hit only with `<img>` kept |",
        "|---|---|---|---|---|",
    ]
    for kind in ("sfx", "speech"):
        k = df[df.kind == kind]
        lg = k[k.long]
        md.append(
            f"| {kind} | {len(k)} | {k.hit.sum()} ({100 * k.hit.mean():.1f} %) | "
            f"{lg.hit.sum()} / {len(lg)} ({100 * lg.hit.mean():.1f} %) | "
            f"{k.img_hit.sum()} |"
        )
    md.append("\nSample (first 5 pages, raw):\n")
    for s, r in list(zip(stems, raw))[:5]:
        md.append(f"- `{s}`: `{r[:300]!r}`")
    out = "\n".join(md) + "\n"
    if not a.limit:
        (ev.REPORTS / f"ocr_eval_page_{name}.md").write_text(out, encoding="utf-8")
    print(out, flush=True)


if __name__ == "__main__":
    main()
