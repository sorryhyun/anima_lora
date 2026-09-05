"""Prompt-context variants + batched decoding for PaddleOCR-VL-1.6 on the sincos crops/pages."""

import json
import re
import sys
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
import torch
from PIL import Image

ROOT = Path("/home/sorryhyun/anima/anima_lora")
sys.path.insert(0, str(ROOT))
from transformers import AutoProcessor, AutoModelForImageTextToText  # noqa: E402

OUT = ROOT / "output/tests/vl16_prompt"
OUT.mkdir(parents=True, exist_ok=True)
mp = ROOT / "models/paddleocr_vl_1.6"
model = (
    AutoModelForImageTextToText.from_pretrained(
        mp, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    .to("cuda")
    .eval()
)
proc = AutoProcessor.from_pretrained(mp)
MIN = proc.image_processor.size["shortest_edge"]


def norm(s):
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"[\s。、．，,.・…‥「」『』!！?？~～〜❤♥♡♪☆★()（）\-ー—–|｜]", "", s)


def sim(a, b):
    a, b = norm(a), norm(b)
    return SequenceMatcher(None, a, b).ratio() if a or b else 1.0


def runaway(t):
    return bool(re.search(r"(.)\1{9,}|(..+?)\2{5,}", t))


def build(images, prompt, system, max_pixels):
    texts = []
    for _ in images:
        msgs = (
            [{"role": "system", "content": [{"type": "text", "text": system}]}]
            if system
            else []
        ) + [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        texts.append(
            proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        )
    return proc(
        text=texts,
        images=images,
        padding=True,
        padding_side="left",
        return_tensors="pt",
        images_kwargs={"size": {"shortest_edge": MIN, "longest_edge": max_pixels}},
    ).to("cuda")


def gen(images, prompt, system=None, max_pixels=1280 * 28 * 28, max_new=256, **kw):
    inputs = build(images, prompt, system, max_pixels)
    n = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        o = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False, use_cache=True, **kw
        )
    outs = []
    for row in o:
        ids = row[n:].tolist()
        ids = [
            i
            for i in ids
            if i not in (proc.tokenizer.eos_token_id, proc.tokenizer.pad_token_id)
        ]
        outs.append(proc.tokenizer.decode(ids).strip())
    return outs


# ---------------- data ----------------
rows = [
    json.loads(ln)
    for ln in (ROOT / "output/tests/vl16_ab/ab.jsonl").open(encoding="utf-8")
]
crops, meta = [], []
for r in rows:
    im = Image.open(
        ROOT / f"post_image_dataset/resized/sincos/{r['stem']}.png"
    ).convert("RGB")
    W, H = im.size
    for t, s, b in r["ppocr"]:
        x0, y0, x1, y1 = b
        crops.append(
            im.crop((max(0, x0 - 4), max(0, y0 - 4), min(W, x1 + 4), min(H, y1 + 4)))
        )
        meta.append({"stem": r["stem"], "pp": t, "score": s, "ref": r["manga_ocr"]})
print("crops", len(crops), flush=True)

SYS = "This is a Japanese manga page. The text is Japanese, written vertically, read top to bottom, with columns ordered right to left."
CROP_PROMPTS = {
    "P0 OCR:": ("OCR:", None),
    "P1 OCR:+hint": (
        "OCR: Japanese manga dialogue, vertical text, read top to bottom.",
        None,
    ),
    "P2 sys+OCR:": ("OCR:", SYS),
    "P3 ja-OCR:": ("Japanese manga OCR:", None),
}
report = {}
# ---------------- A. prompt variants on crops (batch 1) ----------------
base = None
for name, (prompt, system) in CROP_PROMPTS.items():
    torch.cuda.synchronize()
    t = time.time()
    outs = [gen([c], prompt, system)[0] for c in crops]
    torch.cuda.synchronize()
    dt = time.time() - t
    if base is None:
        base = outs
    vs_pp = sum(sim(o, m["pp"]) for o, m in zip(outs, meta)) / len(outs)
    refs = [
        (o, max((sim(o, x) for x in m["ref"]), default=None))
        for o, m in zip(outs, meta)
    ]
    refs = [x for _, x in refs if x is not None]
    same = sum(o == b for o, b in zip(outs, base))
    ra = sum(runaway(o) for o in outs)
    empty = sum(not o for o in outs)
    report[name] = dict(
        s_per_crop=round(dt / len(crops), 3),
        sim_vs_pp=round(vs_pp, 3),
        sim_vs_ref=round(sum(refs) / len(refs), 3),
        identical_to_P0=same,
        runaway=ra,
        empty=empty,
        outs=outs,
    )
    print(name, {k: v for k, v in report[name].items() if k != "outs"}, flush=True)
json.dump(
    {k: v for k, v in report.items()},
    (OUT / "crop_prompts.json").open("w", encoding="utf-8"),
    ensure_ascii=False,
    indent=1,
)

# ---------------- B. spotting prompt variants on 12 pages ----------------
SPOT_PROMPTS = {
    "S0 Spotting:": ("Spotting:", None),
    "S1 Spotting:+hint": (
        "Spotting: Japanese manga page, vertical text, columns read right to left, top to bottom.",
        None,
    ),
    "S2 sys+Spotting:": ("Spotting:", SYS),
}
pages = [r for r in rows if len(r["ppocr"]) >= 3][:12]
spot = {}
for name, (prompt, system) in SPOT_PROMPTS.items():
    res = []
    t = time.time()
    for r in pages:
        im = Image.open(
            ROOT / f"post_image_dataset/resized/sincos/{r['stem']}.png"
        ).convert("RGB")
        im2 = im.resize((im.width * 2, im.height * 2), Image.Resampling.LANCZOS)
        out = gen([im2], prompt, system, max_pixels=2048 * 28 * 28, max_new=1024)[0]
        lines = [ln for ln in out.splitlines() if ln.strip()]
        xs = [
            int(m.group(1))
            for ln in lines
            for m in [re.search(r"<\|LOC_(\d+)\|>", ln)]
            if m
        ]
        rtl = sum(1 for a, b in zip(xs, xs[1:]) if b <= a)
        ltr = sum(1 for a, b in zip(xs, xs[1:]) if b > a)
        texts = [re.sub(r"<\|LOC_\d+\|>", "", ln).strip() for ln in lines]
        best = [max((sim(x, tt) for tt in texts), default=0) for x in r["manga_ocr"]]
        res.append(
            dict(
                stem=r["stem"],
                n=len(lines),
                rtl=rtl,
                ltr=ltr,
                runaway=runaway(out),
                ref_sim=(sum(best) / len(best)) if best else None,
                texts=texts,
            )
        )
    dt = time.time() - t
    agg = dict(
        s_per_page=round(dt / len(pages), 2),
        lines=sum(x["n"] for x in res),
        rtl_pairs=sum(x["rtl"] for x in res),
        ltr_pairs=sum(x["ltr"] for x in res),
        runaway=sum(x["runaway"] for x in res),
        ref_sim=round(
            sum(x["ref_sim"] for x in res if x["ref_sim"] is not None)
            / max(1, sum(x["ref_sim"] is not None for x in res)),
            3,
        ),
    )
    spot[name] = dict(agg=agg, pages=res)
    print(name, agg, flush=True)
json.dump(
    spot,
    (OUT / "spot_prompts.json").open("w", encoding="utf-8"),
    ensure_ascii=False,
    indent=1,
)

# ---------------- C. batching on crops ----------------
order = sorted(range(len(crops)), key=lambda i: crops[i].size[0] * crops[i].size[1])
ref1 = [None] * len(crops)
print("batching (crops sorted by area, left-padded):", flush=True)
for bs in (1, 4, 8, 16, 32):
    outs = [None] * len(crops)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for k in range(0, len(order), bs):
        idx = order[k : k + bs]
        for i, o in zip(idx, gen([crops[i] for i in idx], "OCR:")):
            outs[i] = o
    torch.cuda.synchronize()
    dt = time.time() - t
    if bs == 1:
        ref1 = outs
    same = sum(a == b for a, b in zip(outs, ref1))
    close = sum(sim(a, b) >= 0.9 for a, b in zip(outs, ref1))
    print(
        f"  bs={bs:2d}: {dt:.1f}s  {len(crops) / dt:.1f} crops/s  identical_to_bs1={same}/{len(crops)} sim>=0.9={close}  peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB",
        flush=True,
    )
    if bs > 1 and same < len(crops):
        diffs = [(ref1[i], outs[i]) for i in range(len(crops)) if ref1[i] != outs[i]][
            :6
        ]
        for a, b in diffs:
            print("     bs1:", repr(a[:40]), "| bsN:", repr(b[:40]))
# batched spotting, pages of different sizes
print("batched spotting on 8 pages:", flush=True)
imgs = [
    Image.open(ROOT / f"post_image_dataset/resized/sincos/{r['stem']}.png").convert(
        "RGB"
    )
    for r in pages[:8]
]
imgs = [
    im.resize((im.width * 2, im.height * 2), Image.Resampling.LANCZOS) for im in imgs
]
for bs in (1, 4, 8):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    outs = []
    for k in range(0, len(imgs), bs):
        outs += gen(
            imgs[k : k + bs], "Spotting:", max_pixels=2048 * 28 * 28, max_new=1024
        )
    torch.cuda.synchronize()
    dt = time.time() - t
    if bs == 1:
        refs = outs
    print(
        f"  bs={bs}: {dt:.1f}s  {len(imgs) / dt:.2f} pages/s identical={sum(a == b for a, b in zip(outs, refs))}/8  peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB",
        flush=True,
    )
