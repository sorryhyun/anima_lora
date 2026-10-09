#!/usr/bin/env python
"""ko_reader_cal — can ``vl`` (stock PaddleOCR-VL 1.6) read a lone Hangul
syllable at all? ``proposal_jamo.md`` § 3 prerequisite 2: a KO read is
``vl`` alone, so before a miss on a render counts, each syllable is read on
font-drawn glyphs through the ruler's crop path (``Readers.read_image``: the
detector's boxes padded 12 % and the whole image; a hit = any box reading
exactly the syllable under ``text.norm``).

The syllables: ``assets/jamo_sets.json``'s J128 and H. Each is drawn alone
(``render_grid`` 1 × 1, 512², the builder's lone renderer) in every face that
covers Hangul (one Noto Serif CJK weight, the six KO faces, LXGW WenKai), at
``PX`` font px, in a bubble and on a flat canvas. The detector and ``vl`` run
batched over the crops; ``sfx`` (the JA reader) is not loaded.

    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/ko_reader_cal.py"

GPU, minutes. → ``results/<ts>-ko-reader-cal/``: ``reads.jsonl`` (every
image), ``result.json`` (hit rates per syllable / face / px / frame, the
misses by jamo position), ``sheet_misses.png``.
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))
from reseed import bootstrap  # noqa: E402

bootstrap()

PX = (
    32,
    64,
    128,
)  # font px on a 512² canvas: the training sizes (16–52) and a render's
SIZE = (512, 512)
FACES = (
    "NotoSerifCJK-Regular.ttc",
    "NanumGothic-Regular.ttf",
    "DoHyeon-Regular.ttf",
    "Jua-Regular.ttf",
    "BlackHanSans-Regular.ttf",
    "NanumPenScript-Regular.ttf",
    "NanumMyeongjo-Regular.ttf",
    "LXGWWenKai-Regular.ttf",
)
BATCH = 32
SEED = 0


def faces() -> dict:
    from common.render.flat import FONT_DIR, find_fonts
    from reseed.pools import KOZH_FONTS

    have = find_fonts() + [str(p) for p in (FONT_DIR / KOZH_FONTS).glob("*.[ot]tf")]
    out = {Path(f).name: f for f in have if Path(f).name in FACES}
    assert set(out) == set(FACES), f"missing faces {set(FACES) - set(out)}"
    return out


def draw(syls: str) -> list:
    """``(meta, PIL image)`` per syllable × face × px × frame."""
    from data.grid import render_grid

    rng = random.Random(SEED)
    out = []
    for s in syls:
        for name, path in faces().items():
            for px in PX:
                for bubble in (True, False):
                    im, boxes = render_grid(
                        [s], 1, 1, SIZE, [path], rng, bubble, (px / SIZE[0],) * 2
                    )
                    meta = {"text": s, "face": name, "px": px, "bubble": bubble}
                    out.append((meta, im))
    return out


def crops(det, bgr) -> list:
    """``Readers.read_image``'s crops: every detector box padded 12 %, then
    the whole image."""
    import numpy as np

    H, W = bgr.shape[:2]
    out = []
    for b in det.detect(bgr):
        x0, y0, x1, y1 = (int(v) for v in b)
        pw, ph = int(0.12 * (x1 - x0)), int(0.12 * (y1 - y0))
        c = bgr[max(0, y0 - ph) : min(H, y1 + ph), max(0, x0 - pw) : min(W, x1 + pw)]
        out.append(np.ascontiguousarray(c))
    out.append(bgr)
    return out


def jamo_diff(read: str, text: str) -> str | None:
    """Which positions a one-syllable misread got wrong (``"cho+jung"`` …);
    ``None`` unless ``read`` is one syllable."""
    from reseed.jamo import decompose, is_syllable

    if len(read) != 1 or not is_syllable(read):
        return None
    a, b = decompose(read), decompose(text)
    return "+".join(n for n, x, y in zip(("cho", "jung", "jong"), a, b) if x != y)


def main() -> None:
    import numpy as np
    import torch
    from anime_tools.ocr.animetext import AnimeTextDetector

    from bench._common import make_run_dir, write_result
    from common.readers import StockVl16, contact_sheet
    from common.text import norm
    from reseed import HOME as RH

    sets = json.loads((RH / "assets" / "jamo_sets.json").read_text(encoding="utf-8"))
    syls = "".join(dict.fromkeys(sets["J128"] + sets["H"]))
    t0 = time.time()
    items = draw(syls)
    print(f"drew {len(items)} images ({len(syls)} syllables)", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det = AnimeTextDetector.load(device=device)
    vl = StockVl16(device)
    flat = []  # (item index, box index, crop)
    for k, (_m, im) in enumerate(items):
        bgr = np.ascontiguousarray(np.array(im.convert("RGB"))[:, :, ::-1])
        for j, c in enumerate(crops(det, bgr)):
            flat.append((k, j, c))
    reads: dict = defaultdict(list)
    for i in range(0, len(flat), BATCH):
        part = flat[i : i + BATCH]
        for (k, _j, _c), (txt, _n) in zip(part, vl.read([c for _, _, c in part])):
            reads[k].append(txt)
        if i // BATCH % 20 == 0:
            print(
                f"  vl {i + len(part)} / {len(flat)} crops "
                f"({(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )
    recs = []
    for k, (m, _im) in enumerate(items):
        rs = reads[k]
        hit = any(norm(r) == norm(m["text"]) for r in rs)
        best = rs[-1] if rs else ""  # the whole image's read
        recs.append(
            {
                **m,
                "reads": rs,
                "hit": hit,
                "jamo": None if hit else jamo_diff(norm(best), m["text"]),
            }
        )

    def rate(key):
        g: dict = defaultdict(lambda: [0, 0])
        for r in recs:
            g[key(r)][0] += r["hit"]
            g[key(r)][1] += 1
        return {str(k): round(h / n, 3) for k, (h, n) in sorted(g.items())}

    per_syl = rate(lambda r: r["text"])
    n_per = len(recs) // len(syls)
    weak = {s: v for s, v in per_syl.items() if v < 0.75}
    metrics = {
        "images": len(recs),
        "hit": round(sum(r["hit"] for r in recs) / len(recs), 3),
        "per_face": rate(lambda r: r["face"]),
        "per_px": rate(lambda r: r["px"]),
        "per_frame": rate(lambda r: "bubble" if r["bubble"] else "flat"),
        "per_set": {
            n: round(
                sum(r["hit"] for r in recs if r["text"] in sets[n])
                / sum(r["text"] in sets[n] for r in recs),
                3,
            )
            for n in ("J64", "J96", "J128", "H")
        },
        "reads_per_syllable": n_per,
        "weak_syllables": weak,  # hit < 0.75 of their reads
        "miss_jamo": dict(
            Counter(r["jamo"] for r in recs if not r["hit"]).most_common()
        ),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    out = make_run_dir("cjk_anima_reseed", label="ko-reader-cal", root=RH / "results")
    with (out / "reads.jsonl").open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "per_syllable.json").write_text(
        json.dumps(per_syl, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    miss = [
        (
            items[k][1],
            [
                f"{r['text']} {r['face'][:10]}",
                f"{r['px']} {'b' if r['bubble'] else 'f'}",
                (r["reads"] or [""])[-1][:20],
            ],
        )
        for k, r in enumerate(recs)
        if not r["hit"]
    ]
    arts = ["reads.jsonl", "per_syllable.json"]
    if miss:
        contact_sheet(miss[:160], out / "sheet_misses.png", thumb=160, cols=10)
        arts.append("sheet_misses.png")
    write_result(
        out,
        script=__file__,
        args={"px": PX, "faces": FACES, "seed": SEED},
        label="ko-reader-cal",
        metrics=metrics,
        artifacts=arts,
    )
    print(
        json.dumps(
            {k: v for k, v in metrics.items() if k != "weak_syllables"},
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(f"weak ({len(weak)}): {weak}", flush=True)
    print(f"→ {out}", flush=True)


if __name__ == "__main__":
    main()
