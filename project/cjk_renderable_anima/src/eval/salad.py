"""Stage ``salad`` — Probe 0: does the base model's text salad hold real units?

Base model, EN prompts asking for manga speech bubbles / signs; detector + two
readers over the output. Agreement between an independent stock reader and
the manga-tuned one is the unit test — the tuned reader alone hallucinates.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from common.models import generate_to, load_generator, load_vae
from common.paths import OUT
from common.readers import Readers, contact_sheet, load_bgr
from common.text import CJK_RE, cer

SALAD_PROMPTS = [
    "manga page, monochrome, two girls talking, speech bubbles with japanese text, screentone",
    "manga panel, close-up of a boy shouting, large speech bubble, japanese text, sound effects",
    "manga, 1girl, surprised, speech bubble, japanese text, comic sound effect text, monochrome",
    "shoujo manga panel, 1girl smiling, speech bubble with japanese text, flowers, screentone",
    "manga, 1boy, 1girl, speech bubbles, japanese text, classroom, monochrome",
    "anime screenshot, 1girl, city street at night, japanese shop signs, neon signs with japanese text",
    "1girl, holding a sign with japanese text, simple background, anime style",
    "anime style, storefront, japanese signboard text, poster with japanese text, daytime",
    "4koma manga, 2girls, speech bubbles, japanese text, simple background",
    "manga cover, title text in japanese, 1girl, colorful, bold lettering",
    "manga, korean text in speech bubbles, 1girl, monochrome, webtoon style",
    "poster, chinese text, 1girl, anime style, calligraphy",
]


def stage_salad(a):
    import torch

    out = OUT / "salad"
    (out / "img").mkdir(parents=True, exist_ok=True)
    args, gen, device, shared = load_generator(
        a.salad_size, a.steps, a.cfg, out / "img"
    )
    vae = load_vae(device)
    manifest = []
    t0 = time.time()
    for pi, prompt in enumerate(SALAD_PROMPTS):
        for seed in range(a.seeds):
            fn = out / "img" / f"p{pi:02d}_s{seed}.png"
            generate_to(fn, args, gen, shared, vae, device, prompt, seed)
            manifest.append({"file": str(fn), "prompt": prompt, "seed": seed})
    print(f"salad: {len(manifest)} images in {time.time() - t0:.0f}s", flush=True)
    del vae, shared
    torch.cuda.empty_cache()
    _read_salad(a, out, manifest)


def _read_salad(a, out: Path, manifest):
    from PIL import Image

    rd = Readers(a.device)
    rows_sheet, recs = [], []
    n_box = n_cjk_sfx = n_cjk_vl = n_agree = 0
    chars: dict[str, int] = {}
    for m in manifest:
        reads = rd.read_image(load_bgr(Path(m["file"])), whole=False)
        m["reads"] = reads
        recs.append(m)
        im = Image.open(m["file"]).convert("RGB")
        for r in reads:
            n_box += 1
            s = r["sfx"] or ""
            v = r["vl"] or ""
            cs = bool(CJK_RE.search(s))
            cv = bool(CJK_RE.search(v))
            n_cjk_sfx += cs
            n_cjk_vl += cv
            agree = cs and cv and cer(s, v) <= 0.5
            n_agree += agree
            for ch in s:
                if CJK_RE.match(ch):
                    chars[ch] = chars.get(ch, 0) + 1
            x0, y0, x1, y1 = r["box"]
            rows_sheet.append(
                (
                    im.crop((x0, y0, x1, y1)),
                    [
                        f"sfx: {s} ({(r['sfx_conf'] or 0):.2f})",
                        f"vl16: {v}",
                        f"agree={int(agree)}",
                    ],
                )
            )
    (out / "reads.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1))
    for k in range(0, len(rows_sheet), 40):
        contact_sheet(
            rows_sheet[k : k + 40], out / f"sheet_{k // 40:02d}.png", thumb=200, cols=5
        )
    top = sorted(chars.items(), key=lambda x: -x[1])[:40]
    rep = [
        "# Probe 0 — base-model salad legibility",
        "",
        f"images {len(manifest)} · detector boxes {n_box} ({n_box / max(1, len(manifest)):.1f}/img)",
        f"boxes where the SFX reader emits ≥1 CJK char: {n_cjk_sfx} ({n_cjk_sfx / max(1, n_box):.0%})",
        f"boxes where stock VL16 emits ≥1 CJK char: {n_cjk_vl} ({n_cjk_vl / max(1, n_box):.0%})",
        f"boxes where both agree (CER ≤ 0.5 between readers): {n_agree} ({n_agree / max(1, n_box):.0%})",
        "",
        "top chars (SFX reader): " + " ".join(f"{c}:{n}" for c, n in top),
        "",
        "Reading: agreement between an independent stock reader and the manga-tuned one on the",
        "same crop is the unit test — a manga-tuned reader alone will hallucinate kana from salad.",
        "Sheets: sheet_*.png (crop / sfx read / vl16 read / agree).",
    ]
    (out / "report.md").write_text("\n".join(rep))
    print("\n".join(rep), flush=True)
