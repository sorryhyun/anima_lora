#!/usr/bin/env python3
"""Text detection on hitocrawl JA ↔ KO edition twins: the same balloon,
lettered in Japanese and in Korean, through one detector.

    make daemon-run ARGS="project/finished/ocr_reader/ocr/eval_hito_det.py --reader animetext"
    .venv/bin/python project/finished/ocr_reader/ocr/eval_hito_det.py --reader ppocr_manga --device cpu
    .venv/bin/python project/finished/ocr_reader/ocr/eval_hito_det.py --rescore animetext ppocr_manga

Ground truth is ``hito/build_crops.py``'s diff regions, kept only when both
sides have ink (``box_ja`` and ``box_twin_in_ja``) in works whose pages are\nstill on disk: JA scored on the JA page
at ``box_ja``, KO on the original twin page at ``cut_twin`` with the crop pad
taken back off. The regions are what the editions changed, not every text on
the page (an untranslated SFX is in neither), so only recall is scored —
``hit`` (a predicted box centre inside the GT box), ``found`` (union cover
≥ 0.5) and ``hull`` (cover of the bbox of the preds whose centre is inside,
clipped to GT; neutral to box grain). Pred boxes per page is reported as a
count, not a precision.

Writes ``output/ocr/eval/hito_det_<name>.jsonl`` and
``reports/ocr_eval_hito_det_<name>.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_manga109 as ev  # noqa: E402
import eval_page_det as pdet  # noqa: E402

HITO = Path("/media/sorryhyun/new/dataset/hito_ocr/ko")
CRAWL = Path.home() / "hitocrawl" / "retrieved"
PAD = 0.12  # build_crops.py --pad default
SIZES = (12, 24, 48, 96)


def unpad_twin(r, scale: float, W: int, H: int) -> list[int]:
    """``cut_twin`` minus the pad ``padded()`` added on the JA side."""
    b = r["box_twin_in_ja"]
    p = max(4, int(PAD * max(b[2] - b[0], b[3] - b[1])))
    pj = [min(p, b[0]), min(p, b[1]), min(p, W - b[2]), min(p, H - b[3])]
    c = r["cut_twin"]
    return [
        int(c[0] + pj[0] / scale),
        int(c[1] + pj[1] / scale),
        int(c[2] - pj[2] / scale),
        int(c[3] - pj[3] / scale),
    ]


def gt_pages() -> list[dict]:
    """One row per JA page with its twin and both GT box lists."""
    import cv2

    out = []
    for wd in sorted(p for p in HITO.iterdir() if (p / "regions.jsonl").exists()):
        scales = {
            pg["ja"]: pg["scale"]
            for pg in json.loads((wd / "pages.json").read_text())
            if pg.get("status") == "ok"
        }
        by_page: dict[str, list[dict]] = {}
        for ln in (wd / "regions.jsonl").open(encoding="utf-8"):
            r = json.loads(ln)
            if r.get("box_ja") and r.get("cut_twin"):
                by_page.setdefault(r["ja_page"], []).append(r)
        for ja_page, rs in sorted(by_page.items()):
            work = CRAWL / rs[0]["work"]
            if not (work / "ja").is_dir():  # gallery pulled (FLAGGED.md)
                break
            ja_path = work / "ja" / ja_page
            H, W = cv2.imread(str(ja_path), cv2.IMREAD_REDUCED_GRAYSCALE_8).shape
            H, W = H * 8, W * 8  # only feeds the pad clip at the border
            s = scales[ja_page]
            out.append(
                dict(
                    work=rs[0]["work"],
                    ja=str(ja_path),
                    ko=str(work / "ko" / rs[0]["twin_page"]),
                    gt_ja=[r["box_ja"] for r in rs],
                    gt_ko=[unpad_twin(r, s, W, H) for r in rs],
                )
            )
    return out


def detector(a):
    if a.reader == "ppocr_manga":
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        path = a.ckpt or hf_hub_download(pdet.PPOCR_REPO, a.det_file)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        sess._in_dtype = (
            np.float16 if "float16" in sess.get_inputs()[0].type else np.float32
        )
        return lambda imgs: [pdet.ppocr_detect(sess, im) for im in imgs]
    from anime_tools.ocr.animetext import AnimeTextDetector

    det = AnimeTextDetector.load(device=a.device)
    return lambda imgs: [
        [[int(v) for v in bx] for bx in bxs] for bxs in det.detect_batch(imgs)
    ]


def run(pages, a) -> list[dict]:
    import cv2

    detect = detector(a)
    rows = []
    for s in range(0, len(pages), 16):
        chunk = pages[s : s + 16]
        for lang in ("ja", "ko"):
            imgs = [cv2.imread(pg[lang]) for pg in chunk]
            for pg, img, boxes in zip(chunk, imgs, detect(imgs)):
                h, w = img.shape[:2]
                pg[f"wh_{lang}"] = [w, h]
                pg[f"pred_{lang}"] = boxes
        rows += chunk
        print(f"{len(rows)} / {len(pages)} pages", flush=True)
    return rows


def score(rows) -> dict[str, list[dict]]:
    out = {"ja": [], "ko": []}
    for r in rows:
        for lang in ("ja", "ko"):
            w, h = r[f"wh_{lang}"]
            pb = r[f"pred_{lang}"]
            pm = pdet._mask(pb, h, w)
            for x0, y0, x1, y1 in r[f"gt_{lang}"]:
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                area = max(1, (x1 - x0) * (y1 - y0))
                ins = [
                    b
                    for b in pb
                    if x0 <= (b[0] + b[2]) / 2 <= x1 and y0 <= (b[1] + b[3]) / 2 <= y1
                ]
                hull = 0.0
                if ins:
                    hx0 = max(x0, min(b[0] for b in ins))
                    hy0 = max(y0, min(b[1] for b in ins))
                    hx1 = min(x1, max(b[2] for b in ins))
                    hy1 = min(y1, max(b[3] for b in ins))
                    hull = max(0, hx1 - hx0) * max(0, hy1 - hy0) / area
                out[lang].append(
                    dict(
                        hit=bool(ins),
                        found=bool(pm[y0:y1, x0:x1].sum() / area >= pdet.COVER),
                        hull=hull,
                        short=min(x1 - x0, y1 - y0) * 960 / max(w, h),
                    )
                )
    return out


def table(name: str, rows, wall: float | None) -> str:
    sc = score(rows)
    n = len(rows)
    md = [
        f"### `{name}` — {n} page pairs, {len({r['work'] for r in rows})} works"
        + (f", {wall / 60:.1f} min" if wall is not None else ""),
        "",
        "pred boxes / page: "
        + ", ".join(
            f"{lang} {sum(len(r[f'pred_{lang}']) for r in rows) / max(1, n):.1f}"
            for lang in ("ja", "ko")
        ),
        "",
        "| lang | GT short side (px @ 960) | GT | hit | found (cov ≥ 0.5) | hull ≥ 0.5 | mean hull |",
        "|---|---|---|---|---|---|---|",
    ]
    edges = (0, *SIZES, None)
    for lang in ("ja", "ko"):
        for lo, hi in [(None, None), *zip(edges[:-1], edges[1:])]:
            k = [
                x
                for x in sc[lang]
                if lo is None or (x["short"] >= lo and (hi is None or x["short"] < hi))
            ]
            if not k:
                continue
            band = "all" if lo is None else f"{lo}–{hi}" if hi else f"≥ {lo}"
            md.append(
                f"| {lang} | {band} | {len(k)} | "
                f"{100 * np.mean([x['hit'] for x in k]):.1f} % | "
                f"{100 * np.mean([x['found'] for x in k]):.1f} % | "
                f"{100 * np.mean([x['hull'] >= 0.5 for x in k]):.1f} % | "
                f"{np.mean([x['hull'] for x in k]):.3f} |"
            )
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--reader", choices=["animetext", "ppocr_manga"])
    ap.add_argument("--det_file", default="det/manga_det_v0.2.onnx")
    ap.add_argument("--ckpt")
    ap.add_argument("--name")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rescore", nargs="+", metavar="NAME")
    a = ap.parse_args()

    if a.rescore:
        out = ["# page detection — hitocrawl JA ↔ KO twins\n"]
        for name in a.rescore:
            path = ev.OUT / f"hito_det_{name}.jsonl"
            out.append(table(name, [json.loads(ln) for ln in path.open()], None))
        print("\n".join(out))
        return
    if not a.reader:
        ap.error("--reader is required unless --rescore")
    name = a.name or a.reader

    t0 = time.time()
    rows = run(gt_pages(), a)
    wall = time.time() - t0
    ev.OUT.mkdir(parents=True, exist_ok=True)
    with (ev.OUT / f"hito_det_{name}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    md = table(name, rows, wall)
    (ev.REPORTS / f"ocr_eval_hito_det_{name}.md").write_text(md, encoding="utf-8")
    print(md, flush=True)


if __name__ == "__main__":
    main()
