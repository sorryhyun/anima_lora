#!/usr/bin/env python3
"""Probe B — automatic positional clauses: SAM3 girl instances → crops → tagger.

For each input image, detect "girl" instances with SAM3 (per-instance boxes +
scores), order them left→right by box center, tag each padded crop with the
Anima Tagger, and compose a positional caption suffix in the dataset's
hand-written convention (``On the left, <tags>. On the middle, …``).

By default runs on every resized image whose caption already carries
hand-written positional clauses (the ground-truth set) plus the
1girl+multiple-views showcase, and scores the pipeline against those clauses:
instance-count agreement, per-position character-name agreement, and
per-position hair-color agreement. The gate is the number of DETECTED
instances, not the girls-count tag — so multiple-views images of a single
character are handled by the same machinery.

Run through the daemon:
    make daemon-run ARGS="_archive/bench/position_captions/probe_autocaption.py --label autocaption"
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# Monkey-patch numpy for sam3 compatibility (upstream pins numpy<2 and uses np.bool)
if not hasattr(np, "bool"):
    np.bool = np.bool_

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from bench._common import make_run_dir, write_result  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402

CLAUSE_RE = re.compile(
    r"[Oo]n the (left|middle|center|right)[,\s]+"
    r"(.*?)(?=\.?\s*[Oo]n the (?:left|middle|center|right)|\.?\s*$)",
    re.S,
)

HAIR_COLORS = {
    "black hair",
    "brown hair",
    "blonde hair",
    "blue hair",
    "pink hair",
    "grey hair",
    "white hair",
    "purple hair",
    "red hair",
    "green hair",
    "orange hair",
    "aqua hair",
    "light brown hair",
    "silver hair",
    "light blue hair",
}

# Tag families worth surfacing in a positional clause (probe heuristic; the
# real pipeline would draw this from the taxonomy).
PERSON_SUBSTRINGS = ("hair", "eyes", "twintails", "ponytail", "braid", "ahoge")
OUTFIT_SUBSTRINGS = (
    "maid",
    "bunny",
    "swimsuit",
    "leotard",
    "bikini",
    "uniform",
    "dress",
    "jacket",
    "shirt",
    "skirt",
    "kimono",
    "hoodie",
    "sweater",
)
MAX_CLAUSE_TAGS = 8

# 1girl + multiple views showcase — no ground-truth clauses, proposal only.
DEFAULT_EXTRA = ["post_image_dataset/resized/channel_(caststation)/8090164.png"]


def position_names(n: int) -> list[str]:
    if n == 1:
        return ["center"]
    if n == 2:
        return ["left", "right"]
    if n == 3:
        return ["left", "middle", "right"]
    ordinals = ["second", "third", "fourth", "fifth", "sixth", "seventh"]
    names = ["leftmost"]
    for i in range(1, n - 1):
        names.append(
            f"{ordinals[i - 1]} from left"
            if i - 1 < len(ordinals)
            else f"{i + 1} from left"
        )
    names.append("rightmost")
    return names


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def dedupe_sorted(
    boxes: list[list[float]], scores: list[float], thr: float
) -> list[int]:
    """Greedy IoU suppression (keep higher score), then left→right by center x."""
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    keep: list[int] = []
    for i in order:
        if all(iou(boxes[i], boxes[j]) < thr for j in keep):
            keep.append(i)
    return sorted(keep, key=lambda i: (boxes[i][0] + boxes[i][2]) / 2)


def discover_gt(resized_root: Path) -> list[Path]:
    out = []
    for txt in sorted(resized_root.rglob("*.txt")):
        if txt.name.endswith(".variants.txt"):
            continue
        if "on the left" not in txt.read_text(encoding="utf-8").lower():
            continue
        img = txt.with_suffix(".png")
        if img.exists():
            out.append(img)
    return out


def parse_gt(caption: str) -> list[dict]:
    clauses = []
    for m in CLAUSE_RE.finditer(caption):
        pos = m.group(1).lower().replace("center", "middle")
        tags = [t.strip(" .").lower() for t in m.group(2).split(",") if t.strip(" .")]
        clauses.append({"pos": pos, "tags": tags})
    return clauses


def clause_tags(pred: dict, characters: set[str]) -> list[str]:
    kept = pred["kept"]
    groups = pred.get("groups") or {}
    tags = [t for t in kept if t in characters]
    for g in ("hair_color", "eye_color"):
        winner = groups.get(g)
        if winner and winner not in tags:
            tags.append(winner)
    for t in sorted(kept, key=lambda t: -kept[t]):
        if t in tags:
            continue
        if any(s in t for s in PERSON_SUBSTRINGS) or any(
            s in t for s in OUTFIT_SUBSTRINGS
        ):
            tags.append(t)
        if len(tags) >= MAX_CLAUSE_TAGS:
            break
    return tags


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="Explicit image paths (repo-relative or absolute). "
        "Default: ground-truth clause set + multiple-views showcase.",
    )
    p.add_argument("--resized-root", default="post_image_dataset/resized")
    p.add_argument("--checkpoint", default="models/sam3/sam3.pt")
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--iou-threshold", type=float, default=0.65)
    p.add_argument(
        "--pad", type=float, default=0.06, help="bbox padding, fraction of box size"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--tagger_dir",
        default=None,
        help="AnimaTagger checkpoint dir (default: DEFAULT_TAGGER_DIR; pass "
        "models/captioners/anima-tagger-dbv4 for the external backend).",
    )
    p.add_argument("--label", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    label = args.label or "autocaption"
    run_dir = make_run_dir(
        "position_captions",
        label=label,
        root="_archive/bench/position_captions/results",
    )
    crops_dir = run_dir / "crops"
    crops_dir.mkdir(exist_ok=True)

    if args.images:
        image_paths = [
            Path(s) if Path(s).is_absolute() else REPO_ROOT / s for s in args.images
        ]
    else:
        image_paths = discover_gt(REPO_ROOT / args.resized_root)
        for s in DEFAULT_EXTRA:
            p = REPO_ROOT / s
            if p.exists() and p not in image_paths:
                image_paths.append(p)
    if not image_paths:
        raise SystemExit("no input images found")

    # ---- pass 1: SAM3 girl-instance detection ------------------------------
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("Loading SAM3...", flush=True)
    model = build_sam3_image_model(
        device=args.device,
        eval_mode=True,
        checkpoint_path=str(REPO_ROOT / args.checkpoint),
        load_from_HF=False,
    )
    processor = Sam3Processor(model)

    detections: dict[str, list[dict]] = {}
    crops: dict[tuple[str, int], Image.Image] = {}
    for i, path in enumerate(image_paths, 1):
        img = Image.open(path).convert("RGB")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = processor.set_image(img)
            out = processor.set_text_prompt(prompt="girl", state=state)
        boxes, scores = [], []
        for b, s in zip(out["boxes"], out["scores"]):
            s = float(s)
            if s < args.score_threshold:
                continue
            b = b.tolist() if torch.is_tensor(b) else list(b)
            boxes.append([float(x) for x in b])
            scores.append(s)
        keep = dedupe_sorted(boxes, scores, args.iou_threshold)
        w, h = img.size
        insts = []
        for k, idx in enumerate(keep):
            x1, y1, x2, y2 = boxes[idx]
            px, py = (x2 - x1) * args.pad, (y2 - y1) * args.pad
            box = (
                max(0, int(x1 - px)),
                max(0, int(y1 - py)),
                min(w, int(x2 + px)),
                min(h, int(y2 + py)),
            )
            crops[(str(path), k)] = img.crop(box)
            insts.append({"box": list(box), "score": round(scores[idx], 3)})
        detections[str(path)] = insts
        print(
            f"[{i}/{len(image_paths)}] {path.name}: {len(insts)} girl instance(s)",
            flush=True,
        )

    del processor, model
    clean_memory_on_device(device)

    # ---- pass 2: tag crops, compose clauses, score against ground truth ----
    from anime_tools.tagger.tagger import (  # noqa: E402
        DEFAULT_TAGGER_DIR,
        AnimaTagger,
        ensure_tagger_checkpoint,
    )
    from library.env import resolve_under_home  # noqa: E402

    ckpt_dir = ensure_tagger_checkpoint(
        resolve_under_home(args.tagger_dir or DEFAULT_TAGGER_DIR)
    )
    tagger = AnimaTagger(ckpt_dir, device=args.device)
    with open(Path(ckpt_dir) / "vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    characters = {t["name"] for t in vocab["tags"] if t["category"] == "character"}

    rows = []
    char_hits = char_total = hair_hits = hair_total = 0
    for path in image_paths:
        key = str(path)
        insts = detections[key]
        names = position_names(len(insts))
        preds = []
        for k, inst in enumerate(insts):
            pred = tagger.predict(crops[(key, k)])
            crop_file = f"{path.stem}_{k}_{names[k].replace(' ', '-')}.png"
            crops[(key, k)].save(crops_dir / crop_file)
            kept = {t: round(float(p), 3) for t, p in pred["kept"].items()}
            # The hair_color group is softmax_when_solo — it stays silent on
            # contaminated / multi-person crops, so fall back to the highest
            # kept hair-color tag.
            winner = (pred.get("groups") or {}).get("hair_color")
            if winner is None:
                kept_hair = sorted(set(kept) & HAIR_COLORS, key=lambda t: -kept[t])
                winner = kept_hair[0] if kept_hair else None
            preds.append(
                {
                    **inst,
                    "pos": names[k],
                    "clause_tags": clause_tags(pred, characters),
                    "hair_color": winner,
                    "kept": kept,
                    "crop": f"crops/{crop_file}",
                }
            )
        proposed = " ".join(
            f"On the {p['pos']}, {', '.join(p['clause_tags'])}." for p in preds
        )

        txt = path.with_suffix(".txt")
        gt = parse_gt(txt.read_text(encoding="utf-8")) if txt.exists() else []
        row = {
            "image": str(path.relative_to(REPO_ROOT)),
            "n_pred": len(insts),
            "n_gt": len(gt),
            "count_match": (len(gt) == len(insts)) if gt else None,
            "proposed": proposed,
            "gt_clauses": gt,
            "instances": preds,
        }
        if gt and len(gt) == len(insts):
            for p, g in zip(preds, gt):
                gset = set(g["tags"])
                gchars = gset & characters
                if gchars:
                    char_total += 1
                    if gchars & set(p["kept"]):
                        char_hits += 1
                ghair = gset & HAIR_COLORS
                if ghair:
                    hair_total += 1
                    if p["hair_color"] in ghair:
                        hair_hits += 1
        rows.append(row)
        print(f"{path.name}: pred={len(insts)} gt={len(gt)}", flush=True)
        print(f"  proposed: {proposed}", flush=True)

    scored = [r for r in rows if r["count_match"] is not None]
    (run_dir / "per_image.json").write_text(json.dumps(rows, indent=2))
    metrics = {
        "n_images": len(rows),
        "n_gt_images": len(scored),
        "count_accuracy": round(sum(r["count_match"] for r in scored) / len(scored), 4)
        if scored
        else None,
        "char_position_accuracy": round(char_hits / char_total, 4)
        if char_total
        else None,
        "char_positions_scored": char_total,
        "hair_position_accuracy": round(hair_hits / hair_total, 4)
        if hair_total
        else None,
        "hair_positions_scored": hair_total,
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=label,
        artifacts=["per_image.json", "crops"],
        device=str(device),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
