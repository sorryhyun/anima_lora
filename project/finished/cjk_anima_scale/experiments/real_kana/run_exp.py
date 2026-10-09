#!/usr/bin/env python
"""real_kana — the kana rows polished on real, kanji-free OCR images (2026-09-28)

``retrain_kana``'s rows trained on synthetic renders only (scene bubbles,
grid cells, windows). This asks whether a short warm pass on real images —
the training set's own pages, read by the OCR stage — moves them further:
the target renders (はい / こんにちは) and the doubling ``retrain_kana`` left
(こんんにちちぱ).

Data: every image under ``post_image_dataset/ocr/`` whose OCR lines hold no
kanji (hand-lettered SFX mostly: median 3 glyphs), with its training
caption verbatim (``post_image_dataset/resized/**/<stem>.txt``; its text
clause carries the lines). A line enters the loss only if the caption quotes
it. Each image is centre-cropped to the nearest line shape (``SHAPES``, the
shapes ``retrain_kana`` batched) and resized; an image whose crop cuts a
kept box is dropped. The loss box is the union of the kept lines' boxes
(``layout = "grid"``, the trainer's ``GRID_BOX`` union); the in-box share
counts every kept glyph. σ: ``windows.window`` of the item's kind (a
multi-glyph line routed = ``multi``) at its median line px; no window →
dropped.

Train: the line's trainer as is (μ 0, lr 1e-3 cosine, warmup 0.1), **warm**
from ``retrain_kana``'s merged rows (the line starts singles cold; this is
the exception the experiment measures), routed, ``--steps`` per row over the
174 kana rows. Every other row a caption touches rides frozen at
``retrain_kana``'s value.

Read (routed), each paired with ``retrain_kana``'s cached renders of the
same grid:
  target  the ``target`` ruler (7 captions × 2 seeds) vs
          ``retrain_kana/target/``
  words   ``retrain_kana``'s 13 read words, en, 4 prompts × 2 seeds
          (``retrain_read``'s grid) vs ``retrain_kana/native_r4_en/``
  singles ``retrain_read``'s 14 singles, swap, same grid, vs
          ``retrain_kana/native_r4_swap/``

``--native``: each image at its own training size (the 1024-tier bucket it
was cached at: 2 990–4 200 tokens), no crop, batch 1, the blocks compiled
dynamic-seq over the images' token range — the text as the page drew it,
at its own scale (the 512 arm halves every glyph and never reaches the
0.7–0.9 band). Data ``run0928_real_kana_native``, arm ``real_kana_native``.

``--full_sigma``: no band law — every item's band is [0, 1], which
``fm_training_batch``'s affine map leaves as the LoRA trainer's own draw
(``configs/base.toml``: sigmoid, shift 1.0), and no image is dropped for a
missing window (the band law has no multi row past 64 px: the large SFX).
Suffix ``_sig`` on the data dir and the arm.

``--plain_mse``: no in-box weight (the trainer's ``BOX_SHARE`` = 0: plain MSE
over the canvas, the LoRA trainer's loss). Train-only — the arm gains
``_mse``, the data dir is ``--native`` / ``--full_sigma``'s.

Legs:
  data   (CPU) the data dir ``OUT/run0928_real_kana[_native]/data``
  train  (GPU) ``OUT/experiments/real_kana[_<steps>]``
  read   (GPU) the three reads above

``--dry_run`` prints what the data leg would keep and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"  # every leg is routed (docstring)
os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, REPO, bootstrap, load_experiment, pin_old_seed  # noqa: E402

bootstrap()
pin_old_seed()  # the retrain reads against the old seed's floor of record
from bench._common import make_run_dir, write_result  # noqa: E402
from cjk_scale import reads as scoring  # noqa: E402

RR = load_experiment("retrain_read")

EXP = OUT / "experiments"
NAME = "real_kana"
DATA = "run0928_real_kana"
BASE = "retrain_kana"  # the rows this pass warms from and is read against
CONTEXT = OUT / BASE / "trained.pt"
OCR = REPO / "post_image_dataset" / "ocr"
RESIZED = REPO / "post_image_dataset" / "resized"
SHAPES = (
    (384, 640),
    (416, 624),
    (448, 448),
    (448, 512),
    (448, 576),
    (448, 640),
    (512, 448),
    (512, 512),
    (576, 448),
    (624, 416),
    (640, 384),
    (640, 448),
)  # W × H, retrain_kana's batching shapes
STEPS = 6  # steps / row: 174 rows → 1 044 steps ≈ 18 epochs of ≈ 57 batches
NATIVE_STEPS = 12  # --native, batch 1: 2 088 steps ≈ 9 epochs of 234 images
HAN = re.compile(r"[㐀-鿿豈-﫿]")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs", nargs="+", default=["data"], choices=["data", "train", "read"]
    )
    p.add_argument(
        "--steps", type=int, help="steps / row (default STEPS / NATIVE_STEPS)"
    )
    p.add_argument("--native", action="store_true", help="native size, batch 1")
    p.add_argument("--full_sigma", action="store_true", help="LoRA σ draw, no band law")
    p.add_argument("--plain_mse", action="store_true", help="no in-box weight")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def rc_of(name: str):
    from cjk_scale.config import load_run

    base = load_run(BASE)
    from cjk_scale.config import RunConfig

    return RunConfig(
        name=name,
        path=Path(__file__),
        vocabs=base.vocabs,
        read=base.read,
        context=BASE,
    )


# ----------------------------------------------------------------------------
# data


def ocr_lines(f: Path) -> list[tuple[list[int], str]]:
    """``(box, text)`` per OCR line (``idx  x0,y0,x1,y1  det  read  text``)."""
    out = []
    for ln in f.read_text(encoding="utf-8").splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        cols = ln.split("\t")
        out.append(([int(v) for v in cols[1].split(",")], cols[-1]))
    return out


def caption_lines(caption: str) -> list[str]:
    from anime_tools.captions.position_clauses import TEXT_PREFIXES, parse_caption

    out = []
    for cl in parse_caption(caption).clauses:
        if cl.prefix in TEXT_PREFIXES or any(
            cl.prefix.startswith(p) for p in TEXT_PREFIXES
        ):
            out += [t[1:-1] for t in cl.tags if len(t) >= 2 and t[0] == t[-1] == '"']
    return out


def nearest_shape(w: int, h: int) -> tuple[int, int]:
    return min(SHAPES, key=lambda s: abs(math.log((s[0] / s[1]) / (w / h))))


def crop_box(w: int, h: int, shape) -> tuple[int, int, int, int]:
    """The centre crop of ``w × h`` to ``shape``'s aspect."""
    a = shape[0] / shape[1]
    if w / h > a:
        cw = round(h * a)
        x0 = (w - cw) // 2
        return x0, 0, x0 + cw, h
    ch = round(w / a)
    y0 = (h - ch) // 2
    return 0, y0, w, y0 + ch


def collect(
    native: bool = False, full_sigma: bool = False
) -> tuple[list[dict], Counter]:
    """The kept images, before any write: per image its source, shape, crop
    and the kept lines' boxes in the target shape (``native``: the image's
    own size, no crop)."""
    from PIL import Image

    from cjk_scale.windows import glyph_count, window

    kept, why = [], Counter()
    for f in sorted(OCR.rglob("*.ocr.txt")):
        rel = f.relative_to(OCR).as_posix()[: -len(".ocr.txt")]
        lines = ocr_lines(f)
        if not lines:
            why["no_lines"] += 1
            continue
        if any(HAN.search(t) for _b, t in lines):
            why["kanji"] += 1
            continue
        imgs = [
            p for p in RESIZED.glob(f"{rel}.*") if p.suffix in (".png", ".jpg", ".webp")
        ]
        cap_f = RESIZED / f"{rel}.txt"
        if not imgs or not cap_f.exists():
            why["no_image_or_caption"] += 1
            continue
        caption = cap_f.read_text(encoding="utf-8").strip()
        quoted = caption_lines(caption)
        if any(HAN.search(t) for t in quoted):
            why["kanji_in_caption"] += 1
            continue
        use = [(b, t) for b, t in lines if t in quoted and re.search(r"[ぁ-ヿ]", t)]
        if not use:
            why["no_quoted_kana_line"] += 1
            continue
        w, h = Image.open(imgs[0]).size
        shape = (w, h) if native else nearest_shape(w, h)
        cx0, cy0, cx1, cy1 = (0, 0, w, h) if native else crop_box(w, h, shape)
        if any(b[0] < cx0 or b[1] < cy0 or b[2] > cx1 or b[3] > cy1 for b, _t in use):
            why["crop_cuts_a_box"] += 1
            continue
        sx, sy = shape[0] / (cx1 - cx0), shape[1] / (cy1 - cy0)
        boxes = [
            [
                int((b[0] - cx0) * sx),
                int((b[1] - cy0) * sy),
                math.ceil((b[2] - cx0) * sx),
                math.ceil((b[3] - cy0) * sy),
            ]
            for b, _t in use
        ]
        texts = [t for _b, t in use]
        pxs = [
            math.sqrt(max(1, (b[2] - b[0]) * (b[3] - b[1])) / max(1, glyph_count(t)))
            for b, t in zip(boxes, texts)
        ]
        px = statistics.median(pxs)
        kind = "multi" if any(glyph_count(t) > 1 for t in texts) else "single"
        win = (
            SimpleNamespace(lo=0.0, hi=1.0) if full_sigma else window(kind, px, "scene")
        )
        if win is None:
            why["no_window"] += 1
            continue
        why["kept"] += 1
        kept.append(
            {
                "rel": rel,
                "src_file": str(imgs[0]),
                "size": [w, h],
                "crop": [cx0, cy0, cx1, cy1],
                "shape": list(shape),
                "caption": caption,
                "texts": texts,
                "boxes": boxes,
                "px": round(px, 1),
                "kind": kind,
                "band": [win.lo, win.hi],
            }
        )
    return kept, why


def build(
    kept: list[dict], vocabs: list[str], dname: str, native: bool = False
) -> Path:
    from PIL import Image

    from cjk_scale.windows import glyph_count

    d = OUT / dname / "data"
    if d.exists():
        shutil.rmtree(d)
    (d / "img").mkdir(parents=True)
    glyphs = set(vocabs)
    with (d / "train.jsonl").open("w", encoding="utf-8") as out:
        for k, it in enumerate(kept):
            dst = d / "img" / f"real_{k:04d}.png"
            im = Image.open(it["src_file"]).convert("RGB")
            if not native:
                im = im.crop(tuple(it["crop"])).resize(
                    tuple(it["shape"]), Image.LANCZOS
                )
            im.save(dst)
            bx = it["boxes"]
            union = [
                min(b[0] for b in bx),
                min(b[1] for b in bx),
                max(b[2] for b in bx),
                max(b[3] for b in bx),
            ]
            text = " ".join(it["texts"])
            rec = {
                "file": str(dst),
                "text": text,
                "caption": it["caption"],
                "src": "scene",
                "kind": "real_ocr",
                "recipe": "real_ocr",
                "group": "b{:02d}{:02d}".format(*(round(10 * x) for x in it["band"])),
                "layout": "grid",  # loss box = the kept lines' union (GRID_BOX)
                "units": sorted({c for c in text if c in glyphs}),
                "shape": it["shape"],
                "px": it["px"],
                "law_kind": it["kind"],
                "window": it["band"],
                "band": it["band"],
                "box": union,
                "boxes": bx,
                "glyphs": glyph_count(text),
                "rel": it["rel"],
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    base = OUT / BASE / "data"
    shutil.copy2(base / "eval.json", d / "eval.json")
    shutil.copy2(base / "vocabs.json", d / "vocabs.json")
    (d / "build.json").write_text(
        json.dumps(
            {
                "run": dname,
                "source": "post_image_dataset/ocr, kanji-free images (real_kana)",
                "glyph_route": True,  # train.py routes the captions per glyph
                "n_items": len(kept),
                "shapes": dict(Counter("x".join(map(str, it["shape"])) for it in kept)),
                "bands": dict(Counter(str(it["band"]) for it in kept)),
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return d


def suffix(args) -> str:
    return ("_native" if args.native else "") + ("_sig" if args.full_sigma else "")


# ----------------------------------------------------------------------------
# train


def patch_native(kept: list[dict]) -> None:
    """Batch 1, and the blocks compiled dynamic-seq over the images' token
    range instead of one static graph per token count (43 of them)."""
    from cjk_scale import train as tr
    from library.runtime import harness

    tr.BATCH = 1
    toks = [(w // 16) * (h // 16) for w, h in (it["shape"] for it in kept)]
    orig = harness.compile_blocks_for_training

    def dynamic(unet, network, *, backend, n_token_families=None, **kw):
        return orig(
            unet,
            network,
            backend=backend,
            dynamic_seq=True,
            seq_range=(min(toks), max(toks)),
            **kw,
        )

    harness.compile_blocks_for_training = dynamic


def patch_split_mask() -> None:
    """``BoxSplit`` logs in / out of ``rec['box']``; here the loss box is the
    lines' union (``boxes``), so the log reads the same mask as the loss."""
    import train.stage as ts

    from cjk_scale.loss import box_mask

    ts._box_mask = lambda shape, recs, device: box_mask(shape, recs, device, True)


# ----------------------------------------------------------------------------
# reads


def read_target(rc, arm: Path, data: Path) -> dict:
    from cjk_scale.eval import TRAINED_ARM, ruler_args
    from stages import run as run_stage

    dst = arm / "target" / "native_reads.json"
    if not dst.exists():
        a = ruler_args(rc, TRAINED_ARM, "target")
        a.arm_path, a.data_path = str(arm), str(data)
        run_stage("target", a)
    return scoring.hits(dst, ["はい", "こんにちは"], "verbatim")


def main():
    args = parse_args()
    default = NATIVE_STEPS if args.native else STEPS
    args.steps = args.steps or default
    name = NAME + suffix(args) + ("_mse" if args.plain_mse else "")
    name = name if args.steps == default else f"{name}_{args.steps}"
    rc = rc_of(name)
    vocabs = json.loads((OUT / BASE / "data" / "vocabs.json").read_text("utf-8"))
    kept, why = collect(args.native, args.full_sigma)
    shapes = Counter("x".join(map(str, it["shape"])) for it in kept)
    bands = Counter(str(it["band"]) for it in kept)
    glyphs = Counter(
        c for it in kept for t in it["texts"] for c in t if c in set(vocabs)
    )
    print(f"images: {dict(why)}", flush=True)
    print(f"shapes: {dict(sorted(shapes.items()))}", flush=True)
    print(
        f"bands: {dict(bands)}; px median {statistics.median(it['px'] for it in kept):.1f}",
        flush=True,
    )
    print(
        f"kana rows drawn: {len(glyphs)} / {len(vocabs)}; top {glyphs.most_common(12)}",
        flush=True,
    )
    metrics: dict = {
        "images": dict(why),
        "shapes": dict(shapes),
        "bands": dict(bands),
        "rows_drawn": len(glyphs),
        "glyph_counts": dict(glyphs),
        "steps_per_row": args.steps,
        "native": args.native,
        "full_sigma": args.full_sigma,
        "plain_mse": args.plain_mse,
        "context": str(CONTEXT),
    }
    if args.dry_run:
        for it in kept[:8]:
            print(it["rel"], it["shape"], it["band"], it["px"], it["texts"], flush=True)
        return
    run_dir = make_run_dir(
        "real_kana",
        label=args.label,
        root=LINE / "experiments" / "real_kana" / "results",
    )
    dname = DATA + suffix(args)
    data = OUT / dname / "data"
    if "data" in args.legs:
        build(kept, vocabs, dname, args.native)
        print(f"data: {len(kept)} items → {data}", flush=True)
    if "train" in args.legs:
        from cjk_scale.train import train

        assert CONTEXT.exists(), CONTEXT
        patch_split_mask()
        if args.native:
            patch_native(kept)
        if args.plain_mse:
            from cjk_scale import train as tr

            tr.BOX_SHARE = 0.0  # bs 0 → plain MSE (train.py's `bs` line)
        train(
            rc,
            data=data,
            out=EXP / name,
            cold=False,
            steps_per_row=args.steps,
            context=CONTEXT,
        )
    if "read" in args.legs:
        arm, base = EXP / name, OUT / BASE
        words, singles = list(rc.read), list(RR.HIRA + RR.KATA)
        out = metrics.setdefault("reads", {})
        h = {
            "target": (
                read_target(rc, arm, data),
                scoring.hits(
                    base / "target" / "native_reads.json",
                    ["はい", "こんにちは"],
                    "verbatim",
                ),
            ),
            "words": (
                RR.grid_hits(RR.ensure(rc, arm, words, "en"), words, "en"),
                RR.grid_hits(base / "native_r4_en" / "native_reads.json", words, "en"),
            ),
            "singles": (
                RR.grid_hits(RR.ensure(rc, arm, singles, "swap"), singles, "swap"),
                RR.grid_hits(
                    base / "native_r4_swap" / "native_reads.json", singles, "swap"
                ),
            ),
        }
        for grp, (mine, ref) in h.items():
            print(f"{name} · {grp}:", flush=True)
            out.setdefault(name, {})[grp] = scoring.tally(mine)
            print(f"{BASE} · {grp}:", flush=True)
            out.setdefault(BASE, {})[grp] = scoring.tally(ref)
            pr = scoring.paired(mine, ref)
            out[name][f"{grp}_paired_vs_{BASE}"] = pr
            print(f"  {grp} {name} vs {BASE} {pr}", flush=True)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(OUT / dname), str(EXP / name)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
