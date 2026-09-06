#!/usr/bin/env python3
"""O5 (``plan_ocr.md``): text-kind segmentation — speech / SFX / background per pixel.

    ANIMA_MANGA109S_ROOT=… make daemon-run ARGS="--stall-timeout 0 \\
        project/cjk_aware_anima_dit/ocr/kind_seg.py train --run kind_r34 --epochs 6"
    … kind_seg.py eval-val    --run kind_r34      # val books: box-level kind + SFX recall
    … kind_seg.py eval-sincos --run kind_r34      # the 338 hand labels + the 133 masked pages
    … kind_seg.py export      --run kind_r34      # ONNX, dynamic H×W (the anime_tools session layer)

**Supervision** (decision 5): Manga109-s ``<text>`` boxes = ``speech`` (1),
COO onomatopoeia polygons = ``sfx`` (2), everything else background (0);
where a polygon overlaps a text box the tighter COO polygon wins. Labels are
rasterised from the XML on the fly (no label files on disk). The official
COO book split ∩ Manga109-s (74 / 7 / 6) is the only split — the same file
the reader used, so no test book leaks. Ship build only here (no AnimeText).

**Model**: one ``segmentation_models_pytorch`` U-Net (``--encoder``, default
``resnet34`` ImageNet init), 3 classes, weighted CE + soft dice, trained on
random ``--crop`` windows at native spread resolution (1654×1170; COO min
side p10 is 25 px, so no downscale) with scale / brightness / JPEG / tint
jitter. bf16 autocast. Output ``output/ocr/<run>/`` (``best.pt`` = highest
val SFX box recall, ``history.jsonl``, ``summary.md``).

**Eval** is box-level, because the consumers are boxes: for each GT line the
predicted kind = the majority class of the pixels inside its box that are
not background (``bg`` if fewer than ``--min_text_frac`` of the box is
text); **kind accuracy** over the boxes and **SFX recall** (a GT SFX box
whose predicted kind is ``sfx``). On sincos the GT boxes are the hand
labels (``assets/sfx_labels_sincos.tsv``, kind_hand), scored beside the v1
text rule + chrome list on the same rows (B1's baseline), and the 133
masked pages get an SFX component count (predicted ``sfx`` ∩ MIT text
pixels, components ≥ 32 px) beside the O4 reader's SFX count on them.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manga109 as m109  # noqa: E402

REPO = m109.REPO
OUT_ROOT = REPO / "output/ocr"
CLASSES = ("bg", "speech", "sfx")
BG, SPEECH, SFX = 0, 1, 2
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- labels


def page_index(books: list[str]) -> dict[tuple[str, int], list[m109.Line]]:
    """Every line of every page of ``books`` (speech then sfx)."""
    by: dict[tuple[str, int], list[m109.Line]] = defaultdict(list)
    for b in books:
        for ln in m109.iter_text(b):
            by[(b, ln.page)].append(ln)
        for ln in m109.iter_coo(b):
            by[(b, ln.page)].append(ln)
    return dict(by)


def rasterise(lines: list[m109.Line], h: int, w: int) -> np.ndarray:
    """Label map: speech boxes first, then COO polygons on top (they win)."""
    lab = np.zeros((h, w), dtype=np.uint8)
    for kind_id in (SPEECH, SFX):
        kind = CLASSES[kind_id]
        for ln in lines:
            if ln.kind != kind:
                continue
            pts = np.asarray(ln.poly, dtype=np.float32).reshape(-1, 2)
            if len(pts) < 3:
                continue
            cv2.fillPoly(lab, [np.round(pts).astype(np.int32)], kind_id)
    return lab


def poly_box(poly) -> tuple[int, int, int, int]:
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    return (
        int(pts[:, 0].min()),
        int(pts[:, 1].min()),
        int(pts[:, 0].max()) + 1,
        int(pts[:, 1].max()) + 1,
    )


# --------------------------------------------------------------------------- data


class PageCrops:
    """Random crops of Manga109-s spreads with on-the-fly labels."""

    def __init__(self, books: list[str], crop: int, seed: int, augment: bool):
        self.index = page_index(books)
        self.keys = sorted(self.index)
        self.crop = crop
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, i: int):
        book, page = self.keys[i]
        img = cv2.imread(str(m109.page_path(book, page)))
        lab = rasterise(self.index[(book, page)], *img.shape[:2])
        rng = random.Random(self.rng.random() + i)
        if self.augment:
            s = rng.uniform(0.7, 1.25)
            if abs(s - 1) > 0.02:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                lab = cv2.resize(lab, img.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        H, W = lab.shape
        c = self.crop
        # bias the window toward text: half the time centre on a random labelled pixel
        if rng.random() < 0.5 and lab.any():
            ys, xs = np.nonzero(lab)
            k = rng.randrange(len(ys))
            y0 = int(np.clip(ys[k] - rng.randrange(c), 0, max(0, H - c)))
            x0 = int(np.clip(xs[k] - rng.randrange(c), 0, max(0, W - c)))
        else:
            y0 = rng.randrange(max(1, H - c + 1))
            x0 = rng.randrange(max(1, W - c + 1))
        img = img[y0 : y0 + c, x0 : x0 + c]
        lab = lab[y0 : y0 + c, x0 : x0 + c]
        if img.shape[0] < c or img.shape[1] < c:
            pad = ((0, c - img.shape[0]), (0, c - img.shape[1]))
            img = np.pad(img, (*pad, (0, 0)), mode="edge")
            lab = np.pad(lab, pad, mode="constant")
        if self.augment:
            img = photometric(img, rng)
        return to_tensor(img), lab.astype(np.int64)


def photometric(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Brightness / contrast, an occasional invert (white-on-dark lettering),
    a colour tint (doujin pages are not grey), JPEG."""
    out = img.astype(np.float32)
    a = rng.uniform(0.75, 1.25)
    b = rng.uniform(-30, 30)
    out = out * a + b
    if rng.random() < 0.1:
        out = 255 - out
    if rng.random() < 0.5:
        tint = np.array([rng.uniform(0.8, 1.2) for _ in range(3)], dtype=np.float32)
        out = out * tint
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        q = rng.randint(40, 95)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


def to_tensor(bgr: np.ndarray) -> np.ndarray:
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1).copy()


# --------------------------------------------------------------------------- model


def build_model(encoder: str, weights: str | None = "imagenet"):
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder_name=encoder, encoder_weights=weights, classes=3)


def load_run(run: str, device: str):
    import torch

    d = OUT_ROOT / run
    args = json.loads((d / "args.json").read_text())
    model = build_model(args["encoder"], None)
    model.load_state_dict(torch.load(d / "best.pt", map_location="cpu"))
    return model.to(device).eval(), args


def _pad32(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    H, W = img.shape[:2]
    ph, pw = (-H) % 32, (-W) % 32
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge")
    return img, (H, W)


def predict_page(model, bgr: np.ndarray, device: str, tile: int = 1024) -> np.ndarray:
    """Per-pixel class map for one page (tiled with overlap when larger than
    ``tile`` on both sides; the spread fits in one pass at 1654×1170)."""
    import torch

    padded, (H, W) = _pad32(bgr)
    x = torch.from_numpy(to_tensor(padded))[None].to(device)
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        logits = model(x)
    prob = torch.softmax(logits.float(), 1)[0].cpu().numpy()
    return prob[:, :H, :W]


def box_kind(prob: np.ndarray, box, min_text_frac: float) -> str:
    x0, y0, x1, y1 = (int(v) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(prob.shape[2], x1), min(prob.shape[1], y1)
    if x1 <= x0 or y1 <= y0:
        return "bg"
    cls = prob[:, y0:y1, x0:x1].argmax(0)
    text = cls != BG
    if text.mean() < min_text_frac:
        return "bg"
    return CLASSES[SFX] if (cls[text] == SFX).mean() >= 0.5 else CLASSES[SPEECH]


# --------------------------------------------------------------------------- train


def cmd_train(a):
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    out = OUT_ROOT / a.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    split = m109.load_split()
    tr = PageCrops(split["train"], a.crop, a.seed, augment=True)
    if a.smoke:
        tr.keys = tr.keys[: a.bs * 10]
    print(
        f"train pages {len(tr)} crop {a.crop} bs {a.bs} epochs {a.epochs}", flush=True
    )
    model = build_model(a.encoder).to("cuda")
    w = torch.tensor(a.class_weights, dtype=torch.float32, device="cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = (len(tr) // a.bs) * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=max(1, steps), pct_start=0.1
    )
    hist = out / "history.jsonl"
    hist.write_text("")
    best = (-1.0, 0)
    step = 0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        dl = DataLoader(
            tr, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True
        )
        losses = []
        for x, y in dl:
            x, y = x.to("cuda"), y.to("cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
            logits = logits.float()
            ce = torch.nn.functional.cross_entropy(logits, y, weight=w)
            p = torch.softmax(logits, 1)
            oh = torch.nn.functional.one_hot(y, 3).permute(0, 3, 1, 2).float()
            inter = (p * oh).sum((0, 2, 3))
            dice = 1 - (2 * inter + 1) / (p.sum((0, 2, 3)) + oh.sum((0, 2, 3)) + 1)
            loss = ce + dice[1:].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step < steps:
                sched.step()
            step += 1
            losses.append(loss.item())
            if step % 50 == 0:
                print(
                    f"ep {ep} step {step}/{steps} loss {np.mean(losses[-50:]):.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s "
                    f"vram {torch.cuda.max_memory_allocated() / 2**30:.1f}G",
                    flush=True,
                )
        m = eval_books(model, split["val"], "cuda", a.min_text_frac, limit=a.val_limit)
        m.update(tag=f"ep{ep}", step=step, train_loss=float(np.mean(losses)))
        with hist.open("a") as f:
            f.write(json.dumps(m) + "\n")
        print(f"[val ep{ep}] {json.dumps(m)}", flush=True)
        if m["sfx_recall"] > best[0]:
            best = (m["sfx_recall"], ep)
            torch.save(model.state_dict(), out / "best.pt")
    rows = [json.loads(ln) for ln in hist.read_text().splitlines()]
    md = [
        f"# {a.run} — kind segmentation U-Net ({a.encoder}, crop {a.crop}, bs {a.bs}, lr {a.lr}, "
        f"epochs {a.epochs}, train pages {len(tr)})\n",
        "| tag | box kind acc | speech recall | sfx recall | sfx precision | boxes | train loss |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['tag']} | {100 * r['kind_acc']:.1f} % | {100 * r['speech_recall']:.1f} % | "
            f"{100 * r['sfx_recall']:.1f} % | {100 * r['sfx_precision']:.1f} % | {r['n_boxes']} | {r['train_loss']:.4f} |"
        )
    md.append(f"\nbest = ep{best[1]} (val SFX recall {100 * best[0]:.1f} %)")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


# --------------------------------------------------------------------------- eval


def eval_books(model, books, device, min_text_frac, limit: int | None = None) -> dict:
    """Box-level metrics over every page of ``books``."""
    idx = page_index(books)
    keys = sorted(idx)
    if limit:
        keys = keys[:limit]
    conf: Counter = Counter()
    for book, page in keys:
        lines = idx[(book, page)]
        if not lines:
            continue
        bgr = cv2.imread(str(m109.page_path(book, page)))
        prob = predict_page(model, bgr, device)
        for ln in lines:
            conf[(ln.kind, box_kind(prob, poly_box(ln.poly), min_text_frac))] += 1
    return metrics(conf)


def metrics(conf: Counter) -> dict:
    n = sum(conf.values())
    correct = conf[("speech", "speech")] + conf[("sfx", "sfx")]
    gt_sfx = sum(v for (g, _), v in conf.items() if g == "sfx")
    gt_sp = sum(v for (g, _), v in conf.items() if g == "speech")
    pr_sfx = sum(v for (_, p), v in conf.items() if p == "sfx")
    return {
        "n_boxes": n,
        "kind_acc": correct / max(1, n),
        "speech_recall": conf[("speech", "speech")] / max(1, gt_sp),
        "sfx_recall": conf[("sfx", "sfx")] / max(1, gt_sfx),
        "sfx_precision": conf[("sfx", "sfx")] / max(1, pr_sfx),
        "confusion": {f"{g}->{p}": v for (g, p), v in sorted(conf.items())},
    }


def cmd_eval_val(a):
    model, args = load_run(a.run, a.device)
    split = m109.load_split()
    m = eval_books(model, split[a.split], a.device, a.min_text_frac)
    md = [
        f"# kind_seg `{a.run}` on Manga109-s `{a.split}` (box-level)\n",
        f"boxes {m['n_boxes']}: kind acc **{100 * m['kind_acc']:.1f} %**, speech recall "
        f"{100 * m['speech_recall']:.1f} %, **SFX recall {100 * m['sfx_recall']:.1f} %**, "
        f"SFX precision {100 * m['sfx_precision']:.1f} %\n",
        "| gt → pred | n |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in m["confusion"].items()],
    ]
    (m109.LINE / "reports" / f"kind_seg_{a.run}_{a.split}.md").write_text(
        "\n".join(md) + "\n"
    )
    print("\n".join(md))


def cmd_eval_sincos(a):
    """The hand labels (kind accuracy vs the v1 rule) + the masked pages."""
    import pandas as pd

    rec = m109.pilot_records()
    model, args = load_run(a.run, a.device)
    labels = pd.read_csv(
        m109.ASSETS / "sfx_labels_sincos.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    labels["box"] = labels.box.map(json.loads)
    pages_dir = REPO / "post_image_dataset/resized/sincos"
    masks_dir = REPO / "post_image_dataset/masks/sincos"
    conf_seg: Counter = Counter()
    conf_rule: Counter = Counter()
    rows = []
    for stem, g in labels.groupby("stem"):
        bgr = cv2.imread(str(pages_dir / f"{stem}.png"))
        prob = predict_page(model, bgr, a.device)
        for _, r in g.iterrows():
            gt = r.kind_hand
            pk = box_kind(prob, r.box, a.min_text_frac)
            rk = rec.record_kind(r.text_rec)
            conf_seg[(gt, pk)] += 1
            conf_rule[(gt, rk)] += 1
            rows.append(dict(stem=stem, gt=gt, seg=pk, rule=rk, text=r.text_hand))
    seg, rule = metrics(conf_seg), metrics(conf_rule)

    # masked pages: SFX components of the prediction inside the MIT text pixels
    sfx_records = defaultdict(int)
    reread = REPO / "post_image_dataset/cjk_unmask/ocr_records_sincos_hybrid_vl.jsonl"
    if reread.exists():
        for line in reread.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rr = json.loads(line)
                if rr.get("kind") == "sfx":
                    sfx_records[rr["stem"]] += 1
    comp_rows = []
    for mp in sorted(masks_dir.glob("*_mask.png")):
        stem = mp.name[: -len("_mask.png")]
        bgr = cv2.imread(str(pages_dir / f"{stem}.png"))
        mask = cv2.imread(str(mp), 0)
        if mask.shape != bgr.shape[:2]:
            mask = cv2.resize(mask, bgr.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        prob = predict_page(model, bgr, a.device)
        cls = prob.argmax(0)
        sfx_px = ((cls == SFX) & (mask == 0)).astype(np.uint8)
        k = max(3, int(round(0.025 * mask.shape[1])) | 1)
        sfx_px = cv2.morphologyEx(
            sfx_px,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
        )
        n, _, stats, _ = cv2.connectedComponentsWithStats(sfx_px, connectivity=8)
        comps = sum(1 for i in range(1, n) if min(stats[i][2], stats[i][3]) >= 32)
        comp_rows.append((stem, comps, sfx_records.get(stem, 0)))
    n_pages_seg = sum(1 for _, c, _ in comp_rows if c)
    n_pages_rec = sum(1 for _, _, r in comp_rows if r)
    both = sum(1 for _, c, r in comp_rows if c and r)

    md = [
        f"# kind_seg `{a.run}` on sincos (hand labels + masked pages)\n",
        "## The 338 hand-labelled boxes — kind by the segmenter vs the v1 text rule\n",
        "| | kind acc | speech recall | SFX recall | SFX precision |",
        "|---|---|---|---|---|",
        f"| segmenter | **{100 * seg['kind_acc']:.1f} %** | {100 * seg['speech_recall']:.1f} % | "
        f"**{100 * seg['sfx_recall']:.1f} %** | {100 * seg['sfx_precision']:.1f} % |",
        f"| v1 text rule + chrome list | {100 * rule['kind_acc']:.1f} % | {100 * rule['speech_recall']:.1f} % | "
        f"{100 * rule['sfx_recall']:.1f} % | {100 * rule['sfx_precision']:.1f} % |",
        "",
        "(chrome rows count as wrong for both unless predicted chrome — the segmenter has no chrome class, "
        "so its ceiling on kind acc is the non-chrome share.)",
        "",
        "| gt → pred (segmenter) | n |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in seg["confusion"].items()],
        "",
        f"## The {len(comp_rows)} masked pages — predicted-SFX components inside the MIT text pixels\n",
        f"pages with ≥ 1 SFX component: **{n_pages_seg}** (segmenter) vs {n_pages_rec} pages with a "
        f"kind=sfx record after the O4 re-read; both: {both}. Components total "
        f"{sum(c for _, c, _ in comp_rows)} vs SFX records {sum(r for _, _, r in comp_rows)}.",
        "",
        "| stem | SFX comps (seg) | SFX records (O4) |",
        "|---|---|---|",
        *[f"| {s} | {c} | {r} |" for s, c, r in comp_rows if c or r],
    ]
    rep = m109.LINE / "reports" / f"kind_seg_{a.run}_sincos.md"
    rep.write_text("\n".join(md) + "\n")
    pd.DataFrame(rows).to_json(
        OUT_ROOT / a.run / "sincos_boxes.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    print("\n".join(md[:22]))
    print(f"\nwrote {rep}")


def cmd_export(a):
    import torch

    model, args = load_run(a.run, "cpu")
    x = torch.zeros(1, 3, 512, 512)
    path = OUT_ROOT / a.run / "kind_seg.onnx"
    torch.onnx.export(
        model,
        x,
        str(path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "n", 2: "h", 3: "w"},
            "logits": {0: "n", 2: "h", 3: "w"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"wrote {path} ({path.stat().st_size / 2**20:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--run", required=True)
    t.add_argument("--encoder", default="resnet34")
    t.add_argument("--crop", type=int, default=768)
    t.add_argument("--bs", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--class_weights", type=float, nargs=3, default=[1.0, 3.0, 8.0])
    t.add_argument("--min_text_frac", type=float, default=0.05)
    t.add_argument("--val_limit", type=int, default=400, help="val pages per epoch")
    t.add_argument("--workers", type=int, default=6)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--smoke", action="store_true")
    for name in ("eval-val", "eval-sincos", "export"):
        e = sub.add_parser(name)
        e.add_argument("--run", required=True)
        e.add_argument("--device", default="cuda")
        e.add_argument("--min_text_frac", type=float, default=0.05)
        if name == "eval-val":
            e.add_argument("--split", default="val", choices=m109.SPLITS)
    a = ap.parse_args()
    {
        "train": cmd_train,
        "eval-val": cmd_eval_val,
        "eval-sincos": cmd_eval_sincos,
        "export": cmd_export,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
