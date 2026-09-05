"""A/B: PaddleOCR-VL-1.6 (transformers, native in >=5.0) vs the PP-OCRv6 sidecars on the sincos pages.

Three VL readings per page: ``Spotting:`` on the whole page (x2 upscale, boxes + text),
``OCR:`` on the whole page, and ``OCR:`` on each PP-OCRv6 detector quad from the sidecar
(same boxes, so it isolates the recognizer). Writes ab.jsonl + ab.md to --out.

Gotcha: the shipped generation_config.json has ``use_cache: false`` -- generate() then
re-runs the vision tower every decode token (8 tok/s). Pass ``use_cache=True`` (85 tok/s,
byte-identical text). Run via ``make daemon-run ARGS="--stall-timeout 0 <this> --n 40"``.
Report: project/cjk_aware_anima_dit/reports/0905_paddleocr_vl16_vs_ppocrv6.md
"""

import argparse
import json
import sys
import time
from pathlib import Path
import torch
from PIL import Image

ROOT = Path("/home/sorryhyun/anima/anima_lora")
sys.path.insert(0, str(ROOT))
from anime_tools.captions.ocr_sidecar import read_ocr  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=30)
p.add_argument("--out", default=str(ROOT / "output/tests/vl16_ab"))
p.add_argument("--model", default=str(ROOT / "models/paddleocr_vl_1.6"))
p.add_argument("--source", choices=["resized", "orig"], default="resized")
args = p.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

from transformers import AutoProcessor, AutoModelForImageTextToText  # noqa: E402

t0 = time.time()
model = (
    AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    .to("cuda")
    .eval()
)
proc = AutoProcessor.from_pretrained(args.model)
print(
    f"model loaded {time.time() - t0:.1f}s  vram {torch.cuda.memory_allocated() / 2**30:.2f} GB",
    flush=True,
)


def run(image, prompt, max_pixels, max_new=768):
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = proc.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={
            "size": {
                "shortest_edge": proc.image_processor.size["shortest_edge"],
                "longest_edge": max_pixels,
            }
        },
    ).to(model.device)
    with torch.inference_mode():
        o = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False, use_cache=True
        )
    return proc.decode(o[0][inputs["input_ids"].shape[-1] : -1]), int(
        inputs["input_ids"].shape[-1]
    )


sidecars = sorted(
    (ROOT / "post_image_dataset/cjk_unmask/ocr/sincos").glob("*.ocr.txt")
)[: args.n]
manga = {}
for line in (
    (ROOT / "post_image_dataset/cjk_unmask/ocr_records_sincos.jsonl")
    .read_text()
    .splitlines()
):
    r = json.loads(line)
    manga.setdefault(r["stem"], []).append(r["text"])

rows = []
md = ["# PaddleOCR-VL-1.6 vs PP-OCRv6 — sincos pages\n"]
for sc in sidecars:
    stem = sc.name.split(".")[0]
    img_path = ROOT / f"post_image_dataset/resized/sincos/{stem}.png"
    if args.source == "orig":
        cand = list((ROOT / "image_dataset/sincos").glob(f"{stem}.*"))
        img_path = cand[0] if cand else img_path
    if not img_path.is_file():
        continue
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    pp = read_ocr(sc)
    rec = {
        "stem": stem,
        "size": [W, H],
        "ppocr": [(ln.text, ln.score, ln.box) for ln in pp],
        "manga_ocr": manga.get(stem, []),
    }

    # spotting on the whole page (upscale x2 when <1500 as the card does)
    sp_im = (
        im.resize((W * 2, H * 2), Image.Resampling.LANCZOS) if max(W, H) < 1500 else im
    )
    t = time.time()
    text, ntok = run(sp_im, "Spotting:", 2048 * 28 * 28, 1024)
    rec["spotting"] = text
    rec["spotting_s"] = round(time.time() - t, 2)
    rec["spotting_tokens"] = ntok
    t = time.time()
    text, ntok = run(im, "OCR:", 1280 * 28 * 28, 1024)
    rec["page_ocr"] = text
    rec["page_ocr_s"] = round(time.time() - t, 2)
    rec["page_tokens"] = ntok

    # same-box recognition: the PP-OCRv6 quads, read by the VLM
    crops = []
    t = time.time()
    for ln in pp:
        x0, y0, x1, y1 = ln.box
        pad = 4
        c = im.crop(
            (max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad))
        )
        ct, _ = run(c, "OCR:", 1280 * 28 * 28, 256)
        crops.append(ct.strip())
    rec["crop_ocr"] = crops
    rec["crop_ocr_s"] = round(time.time() - t, 2)
    rows.append(rec)
    (out / "ab.jsonl").open("a", encoding="utf-8").write(
        json.dumps(rec, ensure_ascii=False) + "\n"
    )

    md.append(f"## {stem}  ({W}x{H})\n")
    md.append(f"**manga-ocr (ref):** {' ⏐ '.join(rec['manga_ocr']) or '—'}\n")
    md.append("| # | PP-OCRv6 | score | VL-1.6 same crop |\n|---|---|---|---|")
    for i, (ln, ct) in enumerate(zip(pp, crops), 1):
        md.append(f"| {i} | {ln.text} | {ln.score:.2f} | {ct} |")
    md.append(
        f"\n**VL-1.6 Spotting (page, {rec['spotting_s']}s):**\n```\n{rec['spotting'][:1500]}\n```"
    )
    md.append(
        f"**VL-1.6 OCR (page, {rec['page_ocr_s']}s):**\n```\n{rec['page_ocr'][:1500]}\n```\n"
    )
    (out / "ab.md").write_text("\n".join(md), encoding="utf-8")
    print(
        f"{stem}: pp={len(pp)} spot={rec['spotting_s']}s page={rec['page_ocr_s']}s crops={rec['crop_ocr_s']}s "
        f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f}GB",
        flush=True,
    )
print("done", len(rows))
