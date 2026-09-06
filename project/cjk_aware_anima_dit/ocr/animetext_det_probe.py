#!/usr/bin/env python3
"""Can ``deepghs/AnimeText_yolo`` (YOLO12, single class ``text_block``) replace
PP-OCRv6's DB detector — or the whole 3-layer detection stack (PP DB → VL
Spotting → MIT mask components) — in front of the SFX reader?

    # GPU (daemon): boxes for every (model, imgsz) config, then the SFX reader on every box
    make daemon-run ARGS="--stall-timeout 0 project/cjk_aware_anima_dit/ocr/animetext_det_probe.py --stage det read"
    # CPU: coverage vs the current stack + hand labels, floor, manga-ocr best-match, sheets
    python project/cjk_aware_anima_dit/ocr/animetext_det_probe.py --stage eval

Detection targets ("is a known line covered by a YOLO box?" — IoU ≥ 0.3 or
either-way containment ≥ 0.5, the ``reread.py`` thresholds):

* PP-OCRv6 v3 records (``ocr_records_sincos_ppocr_v3.jsonl``) — "speech recall
  ≥ PP DB" is the O5 gate; kind from the hand labels where the box is labelled.
* the hybrid_vl records (448 lines: PP + Spotting + mask-component reads) per
  kind and per source engine;
* the hand-labelled rows (``sfx_labels_sincos.tsv``, kind_hand);
* MIT mask components (min side 32) on the 133 masked pages; the
  masked-but-no-box floor.

Precision proxy: YOLO boxes overlapping nothing known, split by whether the
page has any known text; contact sheets under ``output/tests/ocr_animetext/``
(green = YOLO, red = PP v3, blue = mask component, magenta = hybrid record).

With reads: YOLO boxes → SfxReader (guard + ≥ 2 chars + has_script) become
records → floor_count, best-match sim to manga-ocr on the 40 A/B pages, and the
hand-SFX exact / sim through the best-IoU YOLO box.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manga109 as m109  # noqa: E402

REPO = m109.REPO
ASSETS = m109.ASSETS
OUT = REPO / "output/tests/ocr_animetext"
MODELS = REPO / "models/animetext_yolo"
CONFIGS = [("yolo12l", 640), ("yolo12l", 1024), ("yolo12l", 0), ("yolo12x", 1024)]
"""(model, imgsz); imgsz 0 = native page size padded to a multiple of 32."""

Box = tuple[int, int, int, int]


def _pkg():
    try:
        from anime_tools.ocr import reread, sfx
    except ImportError:
        sys.path.insert(0, str(REPO.parent / "anime_tools"))
        for k in [k for k in sys.modules if k.startswith("anime_tools")]:
            del sys.modules[k]
        from anime_tools.ocr import reread, sfx
    return reread, sfx


def cfg_name(model: str, imgsz: int) -> str:
    return f"{model}_{imgsz or 'native'}"


# --------------------------------------------------------------------------- det


def letterbox(img, imgsz: int):
    """Top-left letterbox to ``imgsz`` (or the native size padded to /32 when 0);
    returns the canvas and the scale applied to the image."""
    import cv2

    h0, w0 = img.shape[:2]
    if imgsz:
        r = min(imgsz / h0, imgsz / w0)
        nw, nh = round(w0 * r), round(h0 * r)
        cw, ch = imgsz, imgsz
    else:
        r, nw, nh = 1.0, w0, h0
        cw, ch = (w0 + 31) // 32 * 32, (h0 + 31) // 32 * 32
    canvas = np.full((ch, cw, 3), 114, np.uint8)
    canvas[:nh, :nw] = (
        cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA) if r != 1.0 else img
    )
    return canvas, r


def decode(out, r: float, w0: int, h0: int, conf: float, nms: float):
    """``(5, N)`` YOLO head → ``[(x0, y0, x1, y1, score)]`` in image coords."""
    import cv2

    pred = out.T  # (N, 5): cx cy w h score
    keep = pred[:, 4] >= conf
    if not keep.any():
        return []
    p = pred[keep]
    xywh = np.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2, p[:, 2], p[:, 3]], 1)
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), p[:, 4].tolist(), conf, nms)
    boxes = []
    for i in np.array(idx).flatten():
        x, y, w, h = xywh[i] / r
        x0, y0 = max(int(x), 0), max(int(y), 0)
        x1, y1 = min(int(x + w), w0), min(int(y + h), h0)
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1, y1, float(p[i, 4])))
    return boxes


def run_det(opts, pages: list[Path]) -> None:
    import cv2
    import onnxruntime as ort

    OUT.mkdir(parents=True, exist_ok=True)
    if opts.device.startswith("cuda") and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()  # the nvidia-* wheels' cuDNN/CUDA onto the loader path
    # Bounded arena: the default BFC arena + EXHAUSTIVE cuDNN search grows past
    # 15 GB over the native config's per-page shapes and never shrinks — the SFX
    # reader needs the card afterwards.
    cuda_opts = {
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "HEURISTIC",
        "gpu_mem_limit": 4 * 1024**3,
    }
    providers = (
        [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
        if opts.device.startswith("cuda")
        else ["CPUExecutionProvider"]
    )
    for model, imgsz in opts.configs:
        path = OUT / f"boxes_{cfg_name(model, imgsz)}.jsonl"
        if path.exists() and not opts.overwrite:
            print(f"{path.name}: exists, skip", flush=True)
            continue
        sess = ort.InferenceSession(
            str(MODELS / f"{model}_animetext/model.onnx"), providers=providers
        )
        inp = sess.get_inputs()[0].name
        t0, n = time.time(), 0
        with path.open("w", encoding="utf-8") as f:
            for p in pages:
                img = cv2.imread(str(p))
                h0, w0 = img.shape[:2]
                canvas, r = letterbox(img, imgsz)
                x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
                out = sess.run(None, {inp: np.ascontiguousarray(x)})[0][0]
                boxes = decode(out, r, w0, h0, opts.min_conf, opts.nms)
                n += len(boxes)
                f.write(
                    json.dumps(
                        {
                            "stem": p.stem,
                            "size": [w0, h0],
                            "boxes": [list(b[:4]) for b in boxes],
                            "scores": [round(b[4], 4) for b in boxes],
                        }
                    )
                    + "\n"
                )
        print(
            f"{cfg_name(model, imgsz)}: {len(pages)} pages, {n} boxes ≥ {opts.min_conf}, "
            f"{time.time() - t0:.1f}s",
            flush=True,
        )
        del sess
        import gc

        gc.collect()


# --------------------------------------------------------------------------- read


def load_boxes(model: str, imgsz: int, conf: float) -> dict[str, list[Box]]:
    """Boxes at or above ``conf``, per stem."""
    path = OUT / f"boxes_{cfg_name(model, imgsz)}.jsonl"
    by: dict[str, list[Box]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        by[r["stem"]] = [
            tuple(b) for b, s in zip(r["boxes"], r["scores"], strict=True) if s >= conf
        ]
    return by


def run_read(opts, pages: list[Path]) -> None:
    """The SFX reader over every box of every config (a box read once — the
    configs overlap heavily, keyed on ``(stem, box)``)."""
    import cv2

    _, sfx = _pkg()
    raw_path = OUT / "reads.jsonl"
    done: dict[str, str | None] = {}
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[f"{r['stem']}:{r['box']}"] = r["text"]
    todo: dict[str, list[Box]] = defaultdict(list)
    for model, imgsz in opts.configs:
        for stem, boxes in load_boxes(model, imgsz, opts.min_conf).items():
            for b in boxes:
                if f"{stem}:{list(b)}" not in done and b not in todo[stem]:
                    todo[stem].append(b)
    n = sum(len(v) for v in todo.values())
    print(
        f"read stage: {n} boxes on {len(todo)} pages ({len(done)} cached)", flush=True
    )
    if not n:
        return
    t0 = time.time()
    reader = sfx.SfxReader.load(
        device=opts.device,
        base_dir=REPO / "models/paddleocr_vl_1.6",
        batch_size=opts.bs,
    )
    print(f"reader loaded {time.time() - t0:.1f}s", flush=True)
    t0, k = time.time(), 0
    with raw_path.open("a", encoding="utf-8") as f:
        for p in pages:
            boxes = todo.get(p.stem)
            if not boxes:
                continue
            bgr = cv2.imread(str(p))
            reads = reader.read_boxes(bgr, boxes)
            for b, t in zip(boxes, reads, strict=True):
                f.write(
                    json.dumps(
                        {"stem": p.stem, "box": list(b), "text": t}, ensure_ascii=False
                    )
                    + "\n"
                )
            f.flush()
            k += len(boxes)
            if k % 200 < len(boxes):
                print(
                    f"  {k}/{n} boxes {k / (time.time() - t0):.1f} crops/s", flush=True
                )
    print(f"read stage done: {k} boxes {time.time() - t0:.1f}s", flush=True)


# --------------------------------------------------------------------------- eval


def _load_jsonl(path: Path) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by[r["stem"]].append(r)
    return dict(by)


def _contain(inner: Box, outer: Box) -> float:
    """Share of ``inner``'s area inside ``outer``."""
    ix = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return ix * iy / max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))


def denest(boxes: list[Box], policy: str, th: float = 0.85) -> list[Box]:
    """YOLO emits a balloon block *and* its columns. ``outer`` keeps the block
    (drops a box ≥ th inside a larger one); ``inner`` keeps the columns (drops
    a box that holds ≥ 2 others at ≥ th); ``raw`` keeps both."""
    if policy == "raw" or len(boxes) < 2:
        return boxes
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])  # noqa: E731
    keep = []
    for b in boxes:
        others = [o for o in boxes if o is not b]
        if policy == "outer":
            if any(area(o) > area(b) and _contain(b, o) >= th for o in others):
                continue
        elif policy == "inner":
            if sum(area(o) < area(b) and _contain(o, b) >= th for o in others) >= 2:
                continue
        keep.append(b)
    return keep


def _covered(box: Box, boxes: list[Box], reread) -> bool:
    return reread.covered(tuple(box), boxes)


def _best_iou(box: Box, boxes: list[Box], reread) -> tuple[float, Box | None]:
    best, arg = 0.0, None
    for b in boxes:
        iou, _ = reread.overlap(tuple(box), b)
        if iou > best:
            best, arg = iou, b
    return best, arg


def load_hand(path: Path) -> list[dict]:
    import pandas as pd

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return [
        {
            "stem": r.stem,
            "box": tuple(json.loads(r.box)),
            "kind": r.kind_hand or r.kind_rec,
            "text": r.text_hand,
        }
        for _, r in df.iterrows()
        if r.stem
    ]


def eval_config(
    model: str,
    imgsz: int,
    conf: float,
    pages: list[Path],
    *,
    pp: dict[str, list[dict]],
    hyb: dict[str, list[dict]],
    hand: list[dict],
    comps: dict[str, list[Box]],
    masked: set[str],
    reads: dict[str, str | None],
    ref: dict[str, list[dict]],
    ab_stems: list[str],
    rec,
    reread,
    sfx,
    min_chars: int,
    nest: str = "raw",
) -> dict:
    yolo = {s: denest(b, nest) for s, b in load_boxes(model, imgsz, conf).items()}
    hand_kind = {(h["stem"], h["box"]): h["kind"] for h in hand}
    from ocr_sfx_rule import line_kind  # noqa: F401  (kept import-light below)

    def kind_of(stem: str, r: dict) -> str:
        return (
            hand_kind.get((stem, tuple(r["box"])))
            or r.get("kind")
            or line_kind(r["text"])
        )

    cov: dict[str, Counter] = defaultdict(Counter)
    # 1. PP-OCRv6 v3 records per kind
    for stem, recs in pp.items():
        yb = yolo.get(stem, [])
        for r in recs:
            k = kind_of(stem, r)
            cov["pp_v3"][f"{k}:n"] += 1
            cov["pp_v3"][f"{k}:hit"] += _covered(r["box"], yb, reread)
    # 2. hybrid_vl records per kind and per engine family
    for stem, recs in hyb.items():
        yb = yolo.get(stem, [])
        for r in recs:
            hit = _covered(r["box"], yb, reread)
            k = r.get("kind", "?")
            cov["hybrid"][f"{k}:n"] += 1
            cov["hybrid"][f"{k}:hit"] += hit
            eng = r.get("engine", "")
            fam = (
                "ppocr"
                if eng.startswith("ppocr")
                else "spotting"
                if eng.startswith("vl16_spotting")
                else "mask_comp"
            )
            cov["hybrid_engine"][f"{fam}:n"] += 1
            cov["hybrid_engine"][f"{fam}:hit"] += hit
    # 3. hand rows
    for h in hand:
        yb = yolo.get(h["stem"], [])
        cov["hand"][f"{h['kind']}:n"] += 1
        cov["hand"][f"{h['kind']}:hit"] += _covered(h["box"], yb, reread)
    # 4. mask components + floor
    for stem, cs in comps.items():
        yb = yolo.get(stem, [])
        for c in cs:
            cov["mask_comp"]["n"] += 1
            cov["mask_comp"]["hit"] += _covered(c, yb, reread)
    floor_pages = sorted(s for s in masked if not yolo.get(s))
    # 5. precision proxy
    known_pages = {s for s, v in hyb.items() if v} | {s for s, v in comps.items() if v}
    unk_known, unk_empty, n_boxes = 0, 0, 0
    unk_examples: list[tuple[str, Box]] = []
    for stem, yb in yolo.items():
        n_boxes += len(yb)
        known = [tuple(r["box"]) for r in hyb.get(stem, [])] + comps.get(stem, [])
        for b in yb:
            if not _covered(b, known, reread):
                if stem in known_pages:
                    unk_known += 1
                else:
                    unk_empty += 1
                unk_examples.append((stem, b))
    pages_any = sum(1 for s in yolo if yolo[s])

    def valid(t: str | None) -> bool:
        return t is not None and len(rec.norm(t)) >= min_chars and reread.has_script(t)

    unk_valid = (
        sum(1 for s, b in unk_examples if valid(reads.get(f"{s}:{list(b)}")))
        if reads
        else None
    )
    out = {
        "config": cfg_name(model, imgsz),
        "nest": nest,
        "conf": conf,
        "boxes": n_boxes,
        "pages_with_box": pages_any,
        "floor": len(floor_pages),
        "floor_pages": floor_pages,
        "cov": {k: dict(v) for k, v in cov.items()},
        "uncovered_boxes_on_known_pages": unk_known,
        "boxes_on_pages_with_nothing_known": unk_empty,
        "uncovered_examples": unk_examples[:400],
        "uncovered_valid_reads": unk_valid,
    }
    # 6. reads → records
    if reads:
        records: dict[str, list[dict]] = {}
        n_rej = n_short = 0
        for stem, yb in yolo.items():
            rs = []
            for b in yb:
                t = reads.get(f"{stem}:{list(b)}")
                if t is None:
                    n_rej += 1
                    continue
                if not valid(t):
                    n_short += 1
                    continue
                rs.append({"stem": stem, "box": list(b), "text": t})
            if rs:
                records[stem] = rs
        masks_dir = REPO / "post_image_dataset/masks/sincos"
        _, floor_r, miss_r = rec.floor_count(masks_dir, records)
        m_sim, hi, n_ref = rec.best_match_sim(ref, records, ab_stems)
        # hand SFX rows through the best-IoU box
        ex = sims = n_h = 0
        for h in hand:
            if h["kind"] != "sfx":
                continue
            n_h += 1
            iou, b = _best_iou(h["box"], yolo.get(h["stem"], []), reread)
            t = (
                reads.get(f"{h['stem']}:{list(b)}")
                if b is not None and iou > 0
                else None
            )
            if t is None:
                continue
            s = rec.sim(h["text"], t)
            sims += s
            ex += rec.norm(h["text"]) == rec.norm(t)
        out["reads"] = {
            "records": sum(len(v) for v in records.values()),
            "pages_with_line": len(records),
            "guard_rejected": n_rej,
            "under_floor": n_short,
            "floor": floor_r,
            "floor_pages": miss_r,
            "best_match_sim": round(m_sim, 3),
            "best_match_ge_0.9": hi,
            "n_ref": n_ref,
            "hand_sfx_exact": ex,
            "hand_sfx_sim_sum": round(sims, 2),
            "hand_sfx_n": n_h,
        }
        (OUT / f"records_{cfg_name(model, imgsz)}_{nest}_c{conf}.jsonl").write_text(
            "".join(
                json.dumps(r, ensure_ascii=False) + "\n"
                for v in records.values()
                for r in v
            ),
            encoding="utf-8",
        )
    return out


def sheets(opts, pages: list[Path], results: list[dict], pp, hyb, comps) -> None:
    """Per-config PNG sheets: the floor pages, the pages with uncovered YOLO
    boxes on nothing-known pages (precision), and 12 random known pages."""
    import cv2

    for res in results:
        model, imgsz = res["config"].rsplit("_", 1)
        imgsz = 0 if imgsz == "native" else int(imgsz)
        yolo = load_boxes(model, imgsz, res["conf"])
        d = OUT / "sheets" / res["config"]
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        empty = sorted(
            {s for s, b in res["uncovered_examples"] if s not in hyb and s not in comps}
        )
        known = [p.stem for p in pages if p.stem in hyb]
        picks = (
            [("floor", s) for s in res["floor_pages"][:12]]
            + [("unknown", s) for s in empty[:12]]
            + [
                ("known", s)
                for s in rng.choice(known, min(12, len(known)), replace=False)
            ]
        )
        for tag, stem in picks:
            img = cv2.imread(
                str(REPO / "post_image_dataset/resized/sincos" / f"{stem}.png")
            )
            if img is None:
                continue
            for r in hyb.get(stem, []):
                x0, y0, x1, y1 = r["box"]
                cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 255), 2)
            for r in pp.get(stem, []):
                x0, y0, x1, y1 = r["box"]
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
            for x0, y0, x1, y1 in comps.get(stem, []):
                cv2.rectangle(img, (x0, y0), (x1, y1), (255, 128, 0), 2)
            for x0, y0, x1, y1 in yolo.get(stem, []):
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 0), 3)
            cv2.imwrite(str(d / f"{tag}_{stem}.png"), img)


def run_eval(opts, pages: list[Path]) -> None:
    reread, sfx = _pkg()
    rec = m109.pilot_records()
    base = REPO / "post_image_dataset/cjk_unmask"
    pp = _load_jsonl(base / "ocr_records_sincos_ppocr_v3.jsonl")
    hyb = _load_jsonl(base / "ocr_records_sincos_hybrid_vl.jsonl")
    hand = load_hand(ASSETS / "sfx_labels_sincos.tsv")
    ref = _load_jsonl(base / "ocr_records_sincos.jsonl")
    ab_path = REPO / "output/tests/vl16_ab/ab.jsonl"
    ab_stems = [
        json.loads(r)["stem"] for r in ab_path.read_text(encoding="utf-8").splitlines()
    ]

    import cv2

    masks_dir = REPO / "post_image_dataset/masks/sincos"
    comps: dict[str, list[Box]] = {}
    masked: set[str] = set()
    for mp in sorted(masks_dir.glob("*_mask.png")):
        stem = mp.name[: -len("_mask.png")]
        masked.add(stem)
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        comps[stem] = reread.mask_components(m, min_side=opts.comp_min_side)

    reads: dict[str, str | None] = {}
    raw_path = OUT / "reads.jsonl"
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            reads[f"{r['stem']}:{r['box']}"] = r["text"]

    # the v1 kind rule for unlabelled PP records
    ocr_sfx = m109._load_by_path(
        "ocr_sfx_rule", REPO / "project/cjk_aware_anima/datasets/ocr_sfx.py"
    )
    sys.modules["ocr_sfx_rule"] = ocr_sfx

    results = []
    for model, imgsz in opts.configs:
        if not (OUT / f"boxes_{cfg_name(model, imgsz)}.jsonl").exists():
            continue
        for conf in opts.confs:
            for nest in opts.nest:
                results.append(
                    eval_config(
                        model,
                        imgsz,
                        conf,
                        pages,
                        pp=pp,
                        hyb=hyb,
                        hand=hand,
                        comps=comps,
                        masked=masked,
                        reads=reads,
                        ref=ref,
                        ab_stems=ab_stems,
                        rec=rec,
                        reread=reread,
                        sfx=sfx,
                        min_chars=opts.min_chars,
                        nest=nest,
                    )
                )
    (OUT / "eval.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))

    # ---- report
    def pct(c: Counter | dict, k: str) -> str:
        n, h = c.get(f"{k}:n", 0), c.get(f"{k}:hit", 0)
        return f"{h}/{n} ({100 * h / n:.0f}%)" if n else "—"

    n_pp_pages = sum(1 for v in pp.values() if v)
    n_hyb_pages = sum(1 for v in hyb.values() if v)
    _, floor_pp, _ = rec.floor_count(masks_dir, pp)
    _, floor_hyb, _ = rec.floor_count(masks_dir, hyb)
    ref_pp = rec.best_match_sim(ref, pp, ab_stems)
    ref_hyb = rec.best_match_sim(ref, hyb, ab_stems)
    md = [
        "# AnimeText_yolo as the detector — sincos probe",
        "",
        f"{len(pages)} pages, {len(masked)} masked, {sum(len(v) for v in comps.values())} "
        f"mask components (min side {opts.comp_min_side}). Covered = IoU ≥ 0.3 or "
        "containment ≥ 0.5 (`reread.py`). Baselines: PP-OCRv6 v3 "
        f"{sum(len(v) for v in pp.values())} lines on {n_pp_pages} pages, floor {floor_pp}, "
        f"best-match {ref_pp[0]:.3f} ({ref_pp[1]} ≥ 0.9); hybrid_vl "
        f"{sum(len(v) for v in hyb.values())} lines on {n_hyb_pages} pages, floor {floor_hyb}, "
        f"best-match {ref_hyb[0]:.3f} ({ref_hyb[1]} ≥ 0.9).",
        "",
        "## Box-level coverage of what the current stack finds",
        "",
        "| config | nest | conf | boxes | pages w/ box | PP v3 speech | PP v3 sfx | PP v3 all | "
        "hybrid speech | hybrid sfx | hybrid chrome | hyb ppocr | hyb spotting | hyb mask-comp | "
        "hand speech | hand sfx | mask comps | floor (masked, no box) | uncovered on known pages | boxes on empty pages | uncovered w/ valid read |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        c = r["cov"]
        ppc = c.get("pp_v3", {})
        all_n = sum(v for k, v in ppc.items() if k.endswith(":n"))
        all_h = sum(v for k, v in ppc.items() if k.endswith(":hit"))
        md.append(
            f"| {r['config']} | {r['nest']} | {r['conf']} | {r['boxes']} | {r['pages_with_box']} | "
            f"{pct(ppc, 'speech')} | {pct(ppc, 'sfx')} | {all_h}/{all_n} ({100 * all_h / max(all_n, 1):.0f}%) | "
            f"{pct(c.get('hybrid', {}), 'speech')} | {pct(c.get('hybrid', {}), 'sfx')} | "
            f"{pct(c.get('hybrid', {}), 'chrome')} | {pct(c.get('hybrid_engine', {}), 'ppocr')} | "
            f"{pct(c.get('hybrid_engine', {}), 'spotting')} | {pct(c.get('hybrid_engine', {}), 'mask_comp')} | "
            f"{pct(c.get('hand', {}), 'speech')} | {pct(c.get('hand', {}), 'sfx')} | "
            f"{c.get('mask_comp', {}).get('hit', 0)}/{c.get('mask_comp', {}).get('n', 0)} | "
            f"**{r['floor']}** | {r['uncovered_boxes_on_known_pages']} | {r['boxes_on_pages_with_nothing_known']} | "
            f"{r['uncovered_valid_reads'] if r['uncovered_valid_reads'] is not None else '—'} |"
        )
    if any("reads" in r for r in results):
        md += [
            "",
            "## YOLO boxes → SFX reader → records",
            "",
            "| config | nest | conf | records | pages w/ line | guard rej | under floor | floor | "
            "best-match sim (n ref) | ≥ 0.9 | hand-SFX exact | hand-SFX mean sim |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in results:
            if "reads" not in r:
                continue
            q = r["reads"]
            md.append(
                f"| {r['config']} | {r['nest']} | {r['conf']} | {q['records']} | {q['pages_with_line']} | "
                f"{q['guard_rejected']} | {q['under_floor']} | **{q['floor']}** | "
                f"{q['best_match_sim']} ({q['n_ref']}) | {q['best_match_ge_0.9']} | "
                f"{q['hand_sfx_exact']}/{q['hand_sfx_n']} | "
                f"{q['hand_sfx_sim_sum'] / max(q['hand_sfx_n'], 1):.3f} |"
            )
    md += [
        "",
        "Sheets: `output/tests/ocr_animetext/sheets/<config>/` "
        "(green YOLO, red PP v3, blue mask component, magenta hybrid record).",
    ]
    report = OUT / "report.md"
    report.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    if not opts.no_sheets:
        sheets(
            opts,
            pages,
            [
                r
                for r in results
                if r["conf"] == opts.confs[0] and r["nest"] == opts.nest[0]
            ],
            pp,
            hyb,
            comps,
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--stage",
        nargs="+",
        choices=["det", "read", "eval"],
        default=["det", "read", "eval"],
    )
    ap.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="model:imgsz, e.g. yolo12l:1024 yolo12x:0",
    )
    ap.add_argument(
        "--min_conf", type=float, default=0.1, help="det stage keeps boxes ≥ this"
    )
    ap.add_argument(
        "--confs", nargs="*", type=float, default=[0.426, 0.25], help="eval thresholds"
    )
    ap.add_argument("--nms", type=float, default=0.5)
    ap.add_argument("--comp_min_side", type=int, default=32)
    ap.add_argument("--min_chars", type=int, default=2)
    ap.add_argument(
        "--nest",
        nargs="*",
        choices=["raw", "outer", "inner"],
        default=["raw", "outer", "inner"],
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no_sheets", action="store_true")
    opts = ap.parse_args()
    opts.configs = (
        [(c.split(":")[0], int(c.split(":")[1])) for c in opts.configs]
        if opts.configs
        else CONFIGS
    )
    pages = sorted((REPO / "post_image_dataset/resized/sincos").glob("*.png"))
    if opts.limit:
        pages = pages[: opts.limit]
    print(
        f"{len(pages)} pages; configs {[cfg_name(*c) for c in opts.configs]}",
        flush=True,
    )
    if "det" in opts.stage:
        run_det(opts, pages)
    if "read" in opts.stage:
        run_read(opts, pages)
    if "eval" in opts.stage:
        run_eval(opts, pages)


if __name__ == "__main__":
    main()
