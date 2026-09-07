#!/usr/bin/env python3
"""A4-landscape contact sheet: how the OCR sidecars merge into captions at export.
One image per page; art is raster with the boxes drawn, everything else is real
selectable text — every line with its det / score, the export clauses (det floor,
speech / SFX split, SFX deduped) and the no-floor contrast. Then
``google-chrome --headless=new --print-to-pdf=<pdf> --no-pdf-header-footer <html>``.

    .venv/bin/python project/cjk_aware_anima_dit/probes/ocr_merge_sheet.py
"""

from __future__ import annotations
import base64
import html
import io
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from anime_tools.captions.ocr_sfx import split_lines
from anime_tools.captions.ocr_sidecar import (
    DEFAULT_MIN_DET,
    read_ocr,
    usable_lines,
    with_ocr_clause,
)
from anime_tools.captions.position_clauses import parse_caption

REPO = Path("/home/sorryhyun/anima/anima_lora")

RESIZED, OCRDIR = REPO / "post_image_dataset/resized", REPO / "post_image_dataset/ocr"
OUT = REPO / "output/tests/ocr_merge_sheet/ocr_merge_sheet.html"
TTC = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
N, SEED, PX = 100, 0, 1500
LOW_DET, LOW_SCORE = (
    DEFAULT_MIN_DET,
    0.9,
)  # what the sheet flags in orange; LOW_DET = the export floor
f_num = ImageFont.truetype(TTC, 24)

rows = []
for sc in sorted(OCRDIR.rglob("*.ocr.txt")):
    rel = sc.relative_to(OCRDIR)
    stem = rel.name[: -len(".ocr.txt")]
    img = RESIZED / rel.parent / f"{stem}.png"
    if img.is_file():
        rows.append((rel.parent / stem, img, sc))
total = len(rows)
random.Random(SEED).shuffle(rows)
rows = rows[:N]


def clause_str(c):
    return c.prefix + ", ".join(c.tags) + "."


def e(t):
    return html.escape(t)


pages, nsfx = [], 0
for i, (key, imgp, sc) in enumerate(rows, 1):
    lines = read_ocr(sc)
    texts = [ln.text for ln in lines]
    speech, sfx = split_lines(texts)
    nsfx += bool(sfx)
    sfxset = set(sfx)
    used = usable_lines(lines)
    ndrop = len(lines) - len(used)
    im = Image.open(imgp).convert("RGB")
    s = min(PX / im.width, PX / im.height, 1.0)
    art = im.resize(
        (max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS
    )
    d = ImageDraw.Draw(art)
    for n, ln in enumerate(lines, 1):
        b = [int(v * s) for v in ln.box]
        col = (200, 0, 0) if ln.text in sfxset else (20, 20, 20)
        if ln.det < LOW_DET or ln.score < LOW_SCORE:
            col = (224, 112, 0)
        d.rectangle(b, outline=col, width=max(2, art.width // 420))
        d.text((b[0] + 3, max(0, b[1] - 28)), str(n), font=f_num, fill=col)
    buf = io.BytesIO()
    art.save(buf, "JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()

    cf = RESIZED / key.parent / f"{key.name}.txt"
    cap = cf.read_text(encoding="utf-8").strip() if cf.is_file() else ""
    p = parse_caption(cap)
    bag = ", ".join(p.flat_tags)
    pos = " ".join(clause_str(x) for x in p.position_clauses)
    exp = " ".join(
        clause_str(x) for x in parse_caption(with_ocr_clause(cap, lines)).text_clauses
    )
    spl = " ".join(
        clause_str(x)
        for x in parse_caption(with_ocr_clause(cap, lines, min_det=0)).text_clauses
    )

    def conf(ln):
        low = ln.det < LOW_DET or ln.score < LOW_SCORE
        return f'<span class="{"low" if low else "conf"}">d{ln.det:.2f} s{ln.score:.2f}</span>'

    ocr_html = " <span class=sep>|</span> ".join(
        f'<span class="{"sfx" if ln.text in sfxset else "sp"}">{n}:{e(ln.text)}</span> {conf(ln)}'
        for n, ln in enumerate(lines, 1)
    )
    nlow_d = sum(ln.det < LOW_DET for ln in lines)
    nlow_s = sum(ln.score < LOW_SCORE for ln in lines)
    pages.append(f"""<section>
<div class=art><img src="data:image/jpeg;base64,{b64}"></div>
<div class=txt>
<h1>{e(str(key))} <span class=meta>{len(lines)} lines · {len(sfx)} SFX by rule · det&lt;{LOW_DET}: {nlow_d} · score&lt;{LOW_SCORE}: {nlow_s} · [{i}/{len(rows)}]</span></h1>
<h2>raw OCR lines, reading order — <code>d</code> detector box confidence · <code>s</code> reader token confidence</h2><p class=grey>{ocr_html}</p>
<h2>caption before OCR{" (flat bag + position clauses)" if pos else " (flat bag)"}</h2>
<p class=grey>{e(bag)}{(" " + e(pos)) if pos else ""}</p>
<h2>EXPORT appends this — <code>with_ocr_clause()</code>: det ≥ {LOW_DET} ({ndrop} line{"s" if ndrop != 1 else ""} below the floor left out), speech / SFX split, SFX one per sound</h2>
<p class=exp>{e(exp) if exp else "(nothing — every line is under the floor)"}</p>
<h2>without the floor (<code>min_det=0</code>) it would append</h2>
<p class=split>{e(spl)}</p>
</div>
<footer>red = SFX by rule · black = speech · orange = det&lt;{LOW_DET} (export floor) or score&lt;{LOW_SCORE} &nbsp;·&nbsp; min_chars 2 · nested lines dropped &nbsp;·&nbsp; sample {len(rows)} of {total} sidecars, seed {SEED}</footer>
</section>""")

CSS = """
@page { size: A4 landscape; margin: 8mm; }
* { box-sizing: border-box; }
body { margin:0; font-family:"Noto Sans CJK JP","Noto Sans CJK KR",sans-serif;
       -webkit-print-color-adjust:exact; print-color-adjust:exact; background:#eee; }
section { width:281mm; height:194mm; background:#fff; page-break-after:always;
          display:grid; grid-template-columns:158mm 1fr; gap:5mm;
          padding:0; position:relative; margin:0 auto 6mm; overflow:hidden; }
@media print { body{background:#fff} section{margin:0} }
.art { display:flex; align-items:flex-start; justify-content:center; overflow:hidden; }
.art img { max-width:100%; max-height:186mm; object-fit:contain; }
.txt { padding-right:2mm; overflow:hidden; }
h1 { font-size:11.5px; margin:0 0 5px; font-weight:600; }
h1 .meta { font-weight:400; color:#666; font-size:10px; }
h2 { font-size:9.5px; margin:7px 0 2px; font-weight:600; color:#0058c0; }
p  { font-size:10px; line-height:1.45; margin:0 0 0 8px; word-break:break-word; }
code { font-family:ui-monospace,monospace; font-size:9.5px; background:#eef3fb; padding:0 2px; }
.grey{color:#666} .exp{color:#111} .split{color:#c40000}
.sfx{color:#c40000} .sp{color:#333} .sep{color:#bbb}
.conf{color:#999; font-size:8.5px; font-family:ui-monospace,monospace}
.low{color:#e07000; font-size:8.5px; font-family:ui-monospace,monospace; font-weight:600}
footer { position:absolute; left:0; bottom:2mm; width:100%; text-align:center;
         font-size:8px; color:#999; }
"""
OUT.write_text(
    f"<!doctype html><meta charset=utf-8><title>Anima OCR → caption merge</title>"
    f"<style>{CSS}</style>{''.join(pages)}",
    encoding="utf-8",
)
print(
    f"{len(rows)} pages, {nsfx} with >=1 rule-SFX line -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)"
)
