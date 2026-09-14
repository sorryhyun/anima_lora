#!/usr/bin/env python
"""wake_probe — can an *address* alone wake JA glyph rendering in the frozen DiT?

Hypothesis (2026-09-13): Anima saw JA/KO/ZH glyph pixels in pretraining but the
T5 side collapsed every CJK caption to ``<unk>``, so the DiT learned the glyph
*texture* under a presence tag and never a character-level address. If discrete
glyph units exist in the weights, a trained ext row (the address) should steer a
frozen DiT to draw a specific kana. Nothing in the line has run a pixel loss on
the rows with the DiT frozen — JA-BODY had a target-stream LoRA and a whole-crop
inpaint loss; JA-SHIP had no trainable weight on the row → pixel path.

Stages (each one daemon job; ``--stage all`` chains them):

  salad   Probe 0 — base model, EN prompts asking for manga speech bubbles / signs;
          detector + two readers over the output: does the salad contain real,
          reader-agreed kana/kanji units?
  data    build the glyph set: font renders (Noto Sans/Serif CJK weights) of 1–3
          kana strings + kana-only corpus bubble crops; eval prompt sets.
          ``--balanced N`` renders N distinct strings per shared layout (W2a).
  train   frozen DiT, frozen Qwen; trainable = a delta on the ext rows the
          training captions touch (arm ``rows``), that + a LoRA on every
          Linear of ``llm_adapter.blocks`` (arm ``rows_adapter``), or a glyph
          encoder g(render of the piece) → row delta shared across every row
          (arm ``encoder``, W2d; ``--held_out N`` for the generalisation
          test). Plain rectified-flow loss on the glyph crops.
  classify  same-noise N-way diffusion classifier over the trained single kana
          (delta on vs off): which σ carries identity, do the rows discriminate.
  native  scene prompts (the blind-pairs set) + a kana clause, delta 0/1, read:
          does the address survive an ordinary prompt outside the template?
  eval    T2I the eval set with the delta scaled 0 (floor) and 1 (trained),
          same seeds; read; CER vs the floor. An EN string set is the pipeline
          control (no training needed; the base reads Latin).

    make daemon-run ARGS="--label wake-salad --stall-timeout 0 \
        project/cjk_aware_anima_dit/probes/wake_probe.py --stage salad"
    make daemon-run ARGS="--label wake-train-rows --stall-timeout 0 \
        project/cjk_aware_anima_dit/probes/wake_probe.py --stage data train eval --arm rows"
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import sys
import time
import unicodedata
from glob import glob
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LINE = REPO / "project" / "cjk_aware_anima_dit"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(LINE / "ocr"))

OUT = REPO / "output" / "wake_probe"
CORPUS_TRAIN = REPO / "post_image_dataset" / "render" / "ja" / "resized"
CORPUS_HELD = REPO / "post_image_dataset" / "render" / "ja" / "heldout"

HIRA = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
KATA = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
KANA = HIRA + KATA
KANA_RE = re.compile(r"^[ぁ-ゟァ-ヿー〜っ・…！？!?]+$")
CJK_RE = re.compile(r"[぀-ヿ぀-ゟ㐀-䶿一-鿿]")

TPL_BUBBLE = 'manga, speech bubble, japanese text. Japanese text reads as "{}".'
# report / sheet order; ``word`` / ``word_held`` / ``line`` are the 2026-09-14
# word-address groups (``--words``)
EVAL_GROUPS = (
    "single",
    "single_held",
    "word",
    "word_held",
    "line",
    "combo",
    "corpus",
    "en",
)
# a word piece: kana / kanji / long vowel only (no ・ … punctuation pieces)
WORD_RE = re.compile(r"^[ぁ-ゟァ-ヺー一-鿿]+$")
TPL_PLAIN = (
    'japanese text, white background, simple background. Japanese text reads as "{}".'
)
TPL_EN = 'manga, speech bubble, english text. English text reads as "{}".'

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

EN_WORDS = [
    "HELLO",
    "STOP",
    "YES",
    "NO WAY",
    "WAIT",
    "SORRY",
    "WHAT",
    "OK",
    "RUN",
    "HELP",
    "GO",
    "HEY",
]


# ----------------------------------------------------------------------------
# text utils


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    return "".join(
        ch
        for ch in s
        if not ch.isspace() and ch not in "「」『』、。,.!?！？…・〜~\"'()（）"
    )


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp: str, ref: str) -> float:
    r = norm(ref)
    if not r:
        return 1.0
    return min(1.0, lev(norm(hyp), r) / len(r))


# ----------------------------------------------------------------------------
# model plumbing


def _ck():
    from library.env import default_checkpoints

    return default_checkpoints()


def _gen_args(size: int, steps: int, cfg: int | float, save: Path):
    from anima_lora.inference import GenerationRequest

    ck = _ck()
    req = GenerationRequest(
        prompt="",
        image_size=(size, size),
        infer_steps=steps,
        guidance_scale=cfg,
        seed=0,
        dit=ck.dit,
        vae=ck.vae,
        text_encoder=ck.text_encoder,
        attn_mode="flash",
        save_path=str(save),
    )
    return req.to_args()


def _load_vae(device):
    import torch

    from library.models import qwen_vae

    vae = qwen_vae.load_vae(
        _ck().vae, device="cpu", disable_mmap=True, disable_cache=True, vae_2d=True
    )
    return vae.to(device, dtype=torch.bfloat16).eval()


def _decode(vae, latent, device):
    import torch

    with torch.no_grad():
        px = vae.decode_to_pixels(latent.to(device, dtype=vae.dtype))
    if px.ndim == 5:
        px = px.squeeze(2)
    from library.inference.output import pixels_to_pil

    return pixels_to_pil(px[0].float().cpu())


class ExtDelta:
    """Trainable delta on the ext rows of ``llm_adapter.embed`` (mirrors
    EasyControlNetwork._hook_ext_rows: prepended pre-hook sees raw ids, forward
    hook runs after the pack's and adds the delta on top of the pack rows)."""

    def __init__(self, anima, ext_ids, dim, device, row_scale: float):
        import torch

        from library.anima.ext_vocab import T5_TABLE_SIZE

        self.T = T5_TABLE_SIZE
        self.ext_ids = sorted(int(i) for i in ext_ids)
        self.index = {e: i for i, e in enumerate(self.ext_ids)}
        self.raw = torch.nn.Parameter(
            torch.zeros(len(self.ext_ids), dim, device=device)
        )
        self.row_scale = row_scale  # mean pack-row norm: raw is in row-norm units
        self.scale = 1.0
        self.state: dict = {}
        embed = anima.llm_adapter.embed
        lut = torch.full((max(self.ext_ids) + 2,), -1, dtype=torch.long)
        for e, i in self.index.items():
            lut[e] = i
        self.lut = lut.to(device)

        def pre(module, args):
            self.state.pop("mask", None)
            if args and torch.is_tensor(args[0]):
                mask = args[0] >= self.T
                if bool(mask.any()):
                    self.state["mask"] = mask
                    self.state["ext"] = args[0][mask] - self.T

        def post(module, args, output):
            mask = self.state.pop("mask", None)
            if mask is None or self.scale == 0.0:
                self.state.pop("ext", None)
                return None
            ext = self.state.pop("ext")
            ext = torch.clamp(ext, max=self.lut.numel() - 1)
            loc = self.lut[ext]
            known = loc >= 0
            d = torch.zeros(
                ext.numel(),
                self.raw.shape[1],
                device=output.device,
                dtype=self.raw.dtype,
            )
            d[known] = self.raw[loc[known]] * self.row_scale
            out = output.clone()
            out[mask] = out[mask] + (d * self.scale).to(out.dtype)
            return out

        self.handles = [
            embed.register_forward_pre_hook(pre, prepend=True),
            embed.register_forward_hook(post),
        ]

    def state_dict(self):
        return {
            "ext_ids": self.ext_ids,
            "raw": self.raw.detach().cpu(),
            "row_scale": self.row_scale,
        }

    def load(self, sd):
        assert sd["ext_ids"] == self.ext_ids
        self.raw.data.copy_(sd["raw"].to(self.raw.device))
        self.row_scale = sd["row_scale"]


class AdapterLoRA:
    """Rank-r LoRA on every Linear of ``llm_adapter.blocks`` (monkeypatched
    forward, B zero-init). ``scale`` 0 restores the stock adapter."""

    def __init__(self, anima, rank: int, device):
        import torch

        self.params = torch.nn.ParameterList()
        self.scale = 1.0
        self.patched = []
        for name, m in anima.llm_adapter.blocks.named_modules():
            if not isinstance(m, torch.nn.Linear):
                continue
            a = torch.nn.Parameter(
                torch.randn(rank, m.in_features, device=device)
                / math.sqrt(m.in_features)
            )
            b = torch.nn.Parameter(torch.zeros(m.out_features, rank, device=device))
            self.params.append(a)
            self.params.append(b)
            orig = m.forward
            alpha = 1.0 / rank

            def fwd(x, orig=orig, a=a, b=b):
                y = orig(x)
                if self.scale == 0.0:
                    return y
                h = (x.to(a.dtype) @ a.t()) @ b.t()
                return y + (h * (alpha * self.scale)).to(y.dtype)

            m.forward = fwd
            self.patched.append(name)

    def state_dict(self):
        return {
            "params": [p.detach().cpu() for p in self.params],
            "names": self.patched,
        }

    def load(self, sd):
        for p, q in zip(self.params, sd["params"]):
            p.data.copy_(q.to(p.device))


# ----------------------------------------------------------------------------
# readers


class Readers:
    def __init__(self, device: str):
        from anime_tools.ocr.animetext import AnimeTextDetector
        from anime_tools.ocr.sfx import SfxReader

        self.det = AnimeTextDetector.load(device=device)
        self.sfx = SfxReader.load(device=device, batch_size=16)
        import pseudo_label as pl

        self.vl = pl.SWEEPERS["stock"](None, device)

    def read_image(self, bgr, *, whole: bool = True):
        """Return list of dict(box, sfx, sfx_conf, vl) — one per detector box,
        plus the whole image as a pseudo-box when ``whole``."""
        import numpy as np

        H, W = bgr.shape[:2]
        boxes = [tuple(int(v) for v in b) for b in self.det.detect(bgr)]
        crops = []
        for b in boxes:
            x0, y0, x1, y1 = b
            pw, ph = int(0.12 * (x1 - x0)), int(0.12 * (y1 - y0))
            c = bgr[
                max(0, y0 - ph) : min(H, y1 + ph), max(0, x0 - pw) : min(W, x1 + pw)
            ]
            crops.append(np.ascontiguousarray(c))
        if whole:
            boxes.append((0, 0, W, H))
            crops.append(bgr)
        if not crops:
            return []
        sfx = self.sfx.read_scored(crops)
        vl = self.vl.read(crops)
        out = []
        for b, s, v in zip(boxes, sfx, vl):
            out.append(
                {
                    "box": list(b),
                    "whole": b == (0, 0, W, H),
                    "sfx": None if s is None else s[0],
                    "sfx_conf": None if s is None else float(s[1]),
                    "vl": v[0],
                }
            )
        return out


def _bgr(path: Path):
    import numpy as np
    from PIL import Image

    return np.ascontiguousarray(np.array(Image.open(path).convert("RGB"))[:, :, ::-1])


def _label_font(size=22):
    from PIL import ImageFont

    for p in glob("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc") + glob(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
    ):
        return ImageFont.truetype(p, size, index=0)
    return ImageFont.load_default()


def _sheet(rows, path: Path, thumb=256, cols=4):
    """rows: list of (PIL image, [lines of text]). Grid contact sheet."""
    from PIL import Image, ImageDraw

    font = _label_font(18)
    cell_h = thumb + 22 * 4
    n = len(rows)
    r = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * (thumb + 8), r * cell_h + 8), "white")
    d = ImageDraw.Draw(sheet)
    for i, (im, lines) in enumerate(rows):
        x = (i % cols) * (thumb + 8) + 4
        y = (i // cols) * cell_h + 4
        t = im.copy()
        t.thumbnail((thumb, thumb))
        sheet.paste(t, (x, y))
        for j, ln in enumerate(lines[:4]):
            d.text((x, y + thumb + 2 + 22 * j), ln[:26], fill="black", font=font)
    sheet.save(path)


# ----------------------------------------------------------------------------
# stage: salad


def stage_salad(a):
    import torch

    from library.inference.generation import generate, get_generation_settings
    from library.inference.models import load_dit_model, load_shared_models

    out = OUT / "salad"
    (out / "img").mkdir(parents=True, exist_ok=True)
    args = _gen_args(a.salad_size, a.steps, a.cfg, out / "img")
    gen = get_generation_settings(args)
    device = gen.device
    shared = load_shared_models(args)
    shared["conds_cache"] = {}
    anima = load_dit_model(args, device, torch.bfloat16)
    shared["model"] = anima
    vae = _load_vae(device)
    manifest = []
    t0 = time.time()
    for pi, prompt in enumerate(SALAD_PROMPTS):
        for seed in range(a.seeds):
            fn = out / "img" / f"p{pi:02d}_s{seed}.png"
            if not fn.exists():
                a2 = copy.deepcopy(args)
                a2.prompt = prompt
                a2.seed = seed
                lat = generate(a2, gen, shared)
                _decode(vae, lat, device).save(fn)
            manifest.append({"file": str(fn), "prompt": prompt, "seed": seed})
    print(f"salad: {len(manifest)} images in {time.time() - t0:.0f}s", flush=True)
    del anima, vae, shared
    torch.cuda.empty_cache()
    _read_salad(a, out, manifest)


def _read_salad(a, out: Path, manifest):
    from PIL import Image

    rd = Readers(a.device)
    rows_sheet, recs = [], []
    n_box = n_cjk_sfx = n_cjk_vl = n_agree = 0
    chars: dict[str, int] = {}
    for m in manifest:
        bgr = _bgr(Path(m["file"]))
        reads = rd.read_image(bgr, whole=False)
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
    if rows_sheet:
        for k in range(0, len(rows_sheet), 40):
            _sheet(
                rows_sheet[k : k + 40],
                out / f"sheet_{k // 40:02d}.png",
                thumb=200,
                cols=5,
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


# ----------------------------------------------------------------------------
# stage: data


def _fonts():
    paths = sorted(glob("/usr/share/fonts/opentype/noto/Noto*CJK*.ttc"))
    paths += glob("/usr/share/fonts/truetype/droid/DroidSansFallback*.ttf")
    return paths


JITTER_BG_LIGHT = [
    "white",
    "white",
    (235, 235, 235),
    (245, 240, 230),
    (220, 225, 235),
    (250, 235, 200),
    (200, 215, 235),
]
JITTER_BG_DARK = [(30, 30, 30), (20, 25, 40), (60, 40, 50), (45, 45, 45), (10, 10, 10)]
JITTER_INK_DARK = [
    "black",
    "black",
    (30, 30, 30),
    (140, 30, 30),
    (30, 40, 140),
    (20, 100, 50),
    (150, 60, 120),
]
JITTER_INK_LIGHT = [
    "white",
    "white",
    (240, 235, 200),
    (250, 220, 60),
    (120, 220, 240),
    (255, 150, 160),
]


def _sample_layout(n: int, rng: random.Random, size=512, mode: str = "v1") -> dict:
    """Every random choice of one render for an ``n``-char string, drawn in the
    pre-W2a order so unbalanced data dirs rebuild bit-identically.

    ``mode="jitter"`` (data lever, 2026-09-14) draws the v1 fields and then
    overrides what v1 held constant: glyph position (anywhere on the canvas /
    inside the bubble), size down to 60 px, ink colour + optional outline,
    dark backgrounds, a bubble of random size and place. Every row then has
    the same layout statistics and the only thing a row can explain is the
    glyph — the constant "big black glyph centred on a light canvas" was the
    shared gradient direction that drove the encoder's table to rank 1."""
    lay = {"bubble": rng.random() < 0.6}
    lay["bg"] = rng.choice(
        ["white", "white", (235, 235, 235), (245, 240, 230), (220, 225, 235)]
    )
    if lay["bubble"]:
        # light screentone-ish dots + a white ellipse
        lay["dots"] = [(rng.randrange(size), rng.randrange(size)) for _ in range(900)]
        lay["pad"] = rng.randint(30, 70)
        lay["outline"] = rng.randint(2, 5)
    lay["vertical"] = rng.random() < 0.65 if n > 1 else rng.random() < 0.3
    lay["fs"] = (
        rng.randint(110, 200) if n == 1 else rng.randint(int(320 / n), int(400 / n))
    )
    lay["color"] = rng.choice(["black", "black", (30, 30, 30), (60, 40, 40)])
    lay["rot"] = rng.uniform(-6, 6) if rng.random() < 0.3 else None
    if mode == "jitter":
        dark = rng.random() < 0.35
        lay["bg"] = rng.choice(JITTER_BG_DARK if dark else JITTER_BG_LIGHT)
        # bubble fill stays white, so ink inside a bubble is always dark
        ink_light = dark and not lay["bubble"]
        lay["color"] = rng.choice(JITTER_INK_LIGHT if ink_light else JITTER_INK_DARK)
        if rng.random() < 0.25:
            lay["stroke"] = rng.randint(2, 6)
            lay["stroke_fill"] = "black" if ink_light else "white"
        lay["fs"] = (
            rng.randint(60, 200) if n == 1 else rng.randint(int(180 / n), int(400 / n))
        )
        # normalised anchors; _render_string maps them into the feasible range
        # once it knows the text extent (layout dicts stay font-free)
        lay["pos"] = (rng.random(), rng.random())
        if lay["bubble"]:
            lay["box"] = (
                rng.random(),
                rng.random(),
                rng.uniform(0.45, 1.0),
                rng.uniform(0.45, 1.0),
            )
    return lay


def _render_string(
    text: str,
    font_path: str,
    rng: random.Random,
    size=512,
    layout=None,
    mode: str = "v1",
):
    """``layout`` (from ``_sample_layout``) pins canvas/bubble/size/position so
    several strings render in the same layout; ``None`` draws a fresh one."""
    from PIL import Image, ImageDraw, ImageFont

    n = len(text)
    lay = layout if layout is not None else _sample_layout(n, rng, size, mode)
    bubble, bg, fs, color = lay["bubble"], lay["bg"], lay["fs"], lay["color"]
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)

    def extent(fs):
        font = ImageFont.truetype(font_path, fs, index=0)
        if lay["vertical"]:
            tw = max(d.textlength(ch, font=font) for ch in text)
            th = n * fs * 1.05
        else:
            tw = d.textlength(text, font=font)
            th = fs
        return font, tw, th

    # text block extent (w, h) and its centre; v1 = canvas centre, jitter =
    # anchored inside the bubble's inscribed rectangle / the canvas
    font, tw, th = extent(fs)
    if bubble and "pos" not in lay:
        # v1 fit (2026-09-14, word data): a 3-char vertical string at 400/n px
        # ran past the ellipse (可愛い). Shrink only when the block does not fit
        # the inscribed rectangle of the centred ellipse — singles never do,
        # so the pre-word data dirs' single renders are unchanged.
        half = (size / 2 - lay["pad"]) / 2**0.5 - 6
        k = min(1.0, half / max(tw / 2, 1e-6), half / max(th / 2, 1e-6))
        if k < 1.0:
            fs = max(12, int(fs * k))
            font, tw, th = extent(fs)
    cx, cy = size / 2, size / 2
    ebox = (
        (lay["pad"], lay["pad"], size - lay["pad"], size - lay["pad"])
        if bubble
        else None
    )
    if "pos" in lay:
        m = 8
        if bubble:
            u, v, sw, sh = lay["box"]
            # half-axes: big enough that the inscribed rectangle holds the text
            amin = (tw / 2 + m) * 2**0.5
            bmin = (th / 2 + m) * 2**0.5
            amax = bmax = size / 2 - m
            ea = min(amax, max(amin, sw * amax))
            eb = min(bmax, max(bmin, sh * bmax))
            ecx = ea + m + u * max(0.0, size - 2 * (ea + m))
            ecy = eb + m + v * max(0.0, size - 2 * (eb + m))
            ebox = (ecx - ea, ecy - eb, ecx + ea, ecy + eb)
            rx = max(0.0, ea / 2**0.5 - tw / 2 - m / 2)
            ry = max(0.0, eb / 2**0.5 - th / 2 - m / 2)
            cx = ecx + (2 * lay["pos"][0] - 1) * rx
            cy = ecy + (2 * lay["pos"][1] - 1) * ry
        else:
            cx = tw / 2 + m + lay["pos"][0] * max(0.0, size - tw - 2 * m)
            cy = th / 2 + m + lay["pos"][1] * max(0.0, size - th - 2 * m)
    if bubble:
        for x, y in lay["dots"]:
            d.ellipse((x, y, x + 2, y + 2), fill=(150, 150, 150))
        d.ellipse(ebox, fill="white", outline="black", width=lay["outline"])
    stroke = {}
    if lay.get("stroke"):
        stroke = {"stroke_width": lay["stroke"], "stroke_fill": lay["stroke_fill"]}
    if lay["vertical"]:
        y = cy - th / 2
        for ch in text:
            w = d.textlength(ch, font=font)
            d.text((cx - w / 2, y), ch, fill=color, font=font, **stroke)
            y += fs * 1.05
    else:
        d.text(
            (cx - tw / 2, cy - fs / 2 - fs * 0.1), text, fill=color, font=font, **stroke
        )
    if lay["rot"] is not None:
        im = im.rotate(lay["rot"], fillcolor=bg, resample=Image.BICUBIC)
    return im, bubble


def _corpus_lines(boxes_jsonl: Path, max_len: int):
    out = []
    for ln in boxes_jsonl.read_text().splitlines():
        r = json.loads(ln)
        for b in r["bubbles"]:
            t = b["line"]
            if (
                KANA_RE.match(t)
                and 1 <= len(t) <= max_len
                and any(c in KANA for c in t)
            ):
                out.append((t, r["rel"], b["box"]))
    return out


def _crop_bubble(img_path: Path, box, size=512):
    from PIL import Image

    im = Image.open(img_path).convert("RGB")
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    side = int(max(w, h) * 1.35) + 24
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    L = int(max(0, min(cx - side / 2, im.width - side)))
    T = int(max(0, min(cy - side / 2, im.height - side)))
    crop = im.crop((L, T, min(im.width, L + side), min(im.height, T + side)))
    canvas = Image.new("RGB", (side, side), (240, 240, 240))
    canvas.paste(crop, (0, 0))
    return canvas.resize((size, size), Image.LANCZOS)


def _data_dir(a):
    return OUT / ("data" + (f"_{a.data_tag}" if a.data_tag else ""))


def _qwen_pieces():
    """CPU-only: (Qwen3 tokenizer, qwen id → pack ext row). The pack's ext
    rows are Qwen pieces, many of them whole words (ありがとう / 行く / 明日 are
    one piece → one row), so a "word address" is an existing row."""
    from library.anima.vocab_pack import VocabPack, resolve_pack_prefix
    from library.anima.weights import load_qwen3_tokenizer

    ck = _ck()
    tok = load_qwen3_tokenizer(ck.text_encoder)
    pack = VocabPack.load(resolve_pack_prefix(ck.vocab_pack))
    q = {int(k): int(v) for k, v in pack.mapping["qwen"].items()}
    return tok, q


def _pieces(tok, q, text: str):
    """text → [(piece text, ext row or None)] on the Qwen side."""
    out = []
    for i in tok.encode(text, add_special_tokens=False):
        out.append((tok.decode([i]), q.get(int(i))))
    return out


def _word_inventory(tok, q, n: int, min_len: int = 2):
    """The ``n`` most frequent multi-char single-piece words in the training
    corpus bubbles (piece frequency over every line, not the length-capped
    subset) plus their counts."""
    from collections import Counter

    cnt: Counter = Counter()
    for ln in (CORPUS_TRAIN / "boxes.jsonl").read_text().splitlines():
        r = json.loads(ln)
        for b in r["bubbles"]:
            for p, row in _pieces(tok, q, b["line"]):
                if row is not None and len(p) >= min_len and WORD_RE.match(p):
                    cnt[p] += 1
    return cnt.most_common(n)


def _arm_dir(a):
    return OUT / (
        a.arm
        + (f"_{a.data_tag}" if a.data_tag else "")
        + (f"_{a.arm_tag}" if a.arm_tag else "")
    )


def stage_data(a):
    rng = random.Random(0)
    out = _data_dir(a)
    kana = list(a.only_chars) if a.only_chars else list(KANA)
    (out / "img").mkdir(parents=True, exist_ok=True)
    fonts = _fonts()
    print(f"fonts: {len(fonts)}", flush=True)

    # eval strings first so the training pool can exclude the combos
    if a.only_chars:
        singles_eval = kana[:18]
    else:
        singles_eval = rng.sample(list(HIRA), 12) + rng.sample(list(KATA), 6)
    combos_eval = set()
    n_possible = len(kana) ** 2 + len(kana) ** 3
    n_eval_combos = min(
        18, n_possible // 2
    )  # a tiny alphabet cannot fill 18 (smoke runs)
    while len(combos_eval) < n_eval_combos:
        k = rng.choice([2, 3])
        combos_eval.add("".join(rng.choice(kana) for _ in range(k)))
    held = _corpus_lines(CORPUS_HELD / "boxes.jsonl", 4)
    rng.shuffle(held)
    corpus_eval = []
    seen = set()
    if a.only_chars:
        held = [ln for ln in held if all(c in kana for c in ln[0] if c in KANA)]
    for t, rel, box in held:
        if t not in seen and len(corpus_eval) < 10:
            seen.add(t)
            corpus_eval.append(t)

    # 2026-09-14 word addresses: the inventory gains the corpus's most frequent
    # single-piece words (each its own pack row), K of them held out; corpus
    # lines are kept only when every piece is a trained row (kana single or
    # word) so ``line`` evaluates addresses *in sequence*, not coverage
    words: list = []
    words_held: list = []
    piece_ok = None
    if a.words:
        tok, qmap = _qwen_pieces()
        freq = _word_inventory(tok, qmap, a.words, a.word_min_len)
        words = [w for w, _ in freq]
        wrng = random.Random(a.seed + 13)
        words_held = (
            sorted(wrng.sample(words, a.held_out_words)) if a.held_out_words else []
        )
        words_train = [w for w in words if w not in words_held]
        kana_rows = {
            p for c in kana for p, row in _pieces(tok, qmap, c) if row is not None
        }
        trained_pieces = kana_rows | set(words_train)

        def piece_ok(text: str) -> bool:  # noqa: F811
            ps = _pieces(tok, qmap, text)
            return all(row is not None and p in trained_pieces for p, row in ps)

        (out / "words.json").write_text(
            json.dumps(
                {"freq": freq, "held": words_held, "kana_pieces": sorted(kana_rows)},
                ensure_ascii=False,
                indent=1,
            )
        )
        print(
            f"words: {len(words)} (held {len(words_held)}: {' '.join(words_held)}); "
            f"top {' '.join(f'{w}:{c}' for w, c in freq[:20])}",
            flush=True,
        )
        # eval: trained words, held-out words, covered held-out corpus lines
        words_eval = wrng.sample(words_train, min(a.n_word_eval, len(words_train)))
        held_lines = _corpus_lines(CORPUS_HELD / "boxes.jsonl", a.line_max_len)
        wrng.shuffle(held_lines)
        lines_eval, seenl = [], set()
        for t, _rel, _box in held_lines:
            ps = _pieces(tok, qmap, t)
            if t in seenl or not (2 <= len(ps) <= 3) or not piece_ok(t):
                continue
            seenl.add(t)
            lines_eval.append(t)
            if len(lines_eval) >= a.n_line_eval:
                break
        print(
            f"words eval: {len(words_eval)} trained, {len(words_held)} held, "
            f"{len(lines_eval)} covered 2-3 piece lines",
            flush=True,
        )

    n_single = a.n_single
    n_target = min(a.n_combo, 50 * (n_possible - n_eval_combos))
    recs = []
    if a.balanced:
        # W2a: groups of `g` distinct strings rendered in ONE layout (font, canvas,
        # bubble, glyph size/position) — layout cancels inside the batch, identity
        # is the only gradient. Singles: each round partitions the shuffled
        # inventory (every kana exactly n_single times when g | len(kana)).
        g = a.balanced
        assert len(kana) >= g, f"--balanced {g} needs ≥ {g} chars"
        groups = []
        for _ in range(n_single):
            perm = kana[:]
            rng.shuffle(perm)
            for j in range(0, len(perm), g):
                grp = perm[j : j + g]
                if len(grp) < g:
                    grp += rng.sample([c for c in kana if c not in grp], g - len(grp))
                groups.append(grp)
        n_combo = 0
        while n_combo < n_target:
            k = rng.choice([2, 3])
            grp: list = []
            for _ in range(1000):
                s = "".join(rng.choice(kana) for _ in range(k))
                if s not in combos_eval and s not in grp:
                    grp.append(s)
                    if len(grp) == g:
                        break
            if len(grp) < g:  # tiny alphabet (smoke runs)
                break
            groups.append(grp)
            n_combo += g
        for lid, grp in enumerate(groups):
            font = rng.choice(fonts)
            lay = _sample_layout(len(grp[0]), rng, mode=a.layout)
            for s in grp:
                im, bubble = _render_string(s, font, rng, layout=lay)
                fn = out / "img" / f"font_{len(recs):05d}.png"
                im.save(fn)
                recs.append(
                    {
                        "file": str(fn),
                        "text": s,
                        "caption": (TPL_BUBBLE if bubble else TPL_PLAIN).format(s),
                        "src": "font",
                        "layout_id": lid,
                    }
                )
    else:
        items = []
        # font renders: every kana ×N + random combos (+ every trained word ×N)
        for ch in kana:
            for _ in range(n_single):
                items.append(("font", ch))
        for w in words:
            if w in words_held:
                continue
            for _ in range(n_single):
                items.append(("font", w))
        n_combo = 0
        while n_combo < n_target:
            k = rng.choice([2, 3])
            s = "".join(rng.choice(kana) for _ in range(k))
            if s in combos_eval:
                continue
            items.append(("font", s))
            n_combo += 1
        for i, (kind, s) in enumerate(items):
            im, bubble = _render_string(s, rng.choice(fonts), rng, mode=a.layout)
            fn = out / "img" / f"font_{i:05d}.png"
            im.save(fn)
            recs.append(
                {
                    "file": str(fn),
                    "text": s,
                    "caption": (TPL_BUBBLE if bubble else TPL_PLAIN).format(s),
                    "src": "font",
                }
            )
    # corpus crops
    lines = _corpus_lines(
        CORPUS_TRAIN / "boxes.jsonl", a.line_max_len if a.words else 6
    )
    if a.only_chars:
        lines = [ln for ln in lines if all(c in kana for c in ln[0] if c in KANA)]
    if piece_ok is not None:
        n0 = len(lines)
        lines = [ln for ln in lines if piece_ok(ln[0])]
        print(
            f"corpus: {len(lines)}/{n0} lines fully covered by trained rows", flush=True
        )
    rng.shuffle(lines)
    lines = lines[: a.n_corpus]
    corpus_lid = len(recs)  # past every font layout id
    for j, (t, rel, box) in enumerate(lines):
        try:
            im = _crop_bubble(CORPUS_TRAIN / rel, box)
        except Exception as e:  # noqa: BLE001
            print("skip", rel, e)
            continue
        fn = out / "img" / f"corpus_{j:05d}.png"
        im.save(fn)
        recs.append(
            {
                "file": str(fn),
                "text": t,
                "caption": TPL_BUBBLE.format(t),
                "src": "corpus",
            }
        )
    if a.balanced:
        # corpus crops have no shared layout: plain shuffled groups of g (the
        # tail that does not fill a group is dropped) so every batch is one group
        corpus = [r for r in recs if r["src"] == "corpus"]
        corpus = corpus[: len(corpus) - len(corpus) % a.balanced]
        for j, r in enumerate(corpus):
            r["layout_id"] = corpus_lid + j // a.balanced
        recs = [r for r in recs if r["src"] == "font"] + corpus
    (out / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs)
    )
    ev = (
        [
            {"group": "single", "text": s, "caption": TPL_BUBBLE.format(s)}
            for s in singles_eval
        ]
        + [
            {"group": "combo", "text": s, "caption": TPL_BUBBLE.format(s)}
            for s in sorted(combos_eval)
        ]
        + [
            {"group": "corpus", "text": s, "caption": TPL_BUBBLE.format(s)}
            for s in corpus_eval
        ]
        + [{"group": "en", "text": s, "caption": TPL_EN.format(s)} for s in EN_WORDS]
    )
    if a.words:
        ev += (
            [
                {"group": "word", "text": s, "caption": TPL_BUBBLE.format(s)}
                for s in words_eval
            ]
            + [
                {"group": "word_held", "text": s, "caption": TPL_BUBBLE.format(s)}
                for s in words_held
            ]
            + [
                {"group": "line", "text": s, "caption": TPL_BUBBLE.format(s)}
                for s in lines_eval
            ]
        )
    (out / "eval.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1))
    from collections import Counter

    c = Counter(r["src"] for r in recs)
    print(
        f"data: {len(recs)} train items {dict(c)}; eval {len(ev)} prompts", flush=True
    )
    if a.balanced:  # whole groups side by side: layout must match within a row
        by_lid: dict = {}
        for r in recs:
            by_lid.setdefault(r["layout_id"], []).append(r)
        sheet_recs = [
            r
            for lid in rng.sample(sorted(by_lid), min(10, len(by_lid)))
            for r in by_lid[lid]
        ]
    else:
        sheet_recs = rng.sample(recs, min(40, len(recs)))
    _sheet(
        [
            (
                __import__("PIL.Image", fromlist=["Image"]).open(r["file"]),
                [r["text"], r["src"]],
            )
            for r in sheet_recs
        ],
        out / "sheet_train.png",
        thumb=160,
        cols=a.balanced * 2 if a.balanced else 8,
    )


# ----------------------------------------------------------------------------
# W2d: amortized glyph encoder (arm ``encoder``)


def _row_texts(tok, pack, rows):
    """ext row → the piece text it stands for (Qwen piece, char row or symbol
    row); rows the pack cannot name are left out (zero delta)."""
    inv_q = {int(v): int(k) for k, v in pack.mapping["qwen"].items()}
    inv_c = {int(v): k for k, v in pack.mapping.get("char", {}).items()}
    inv_s = {int(v): k for k, v in pack.mapping.get("sym_char", {}).items()}
    qtok = tok.qwen3_tokenizer
    out = {}
    for r in rows:
        r = int(r)
        if r in inv_q:
            t = qtok.decode([inv_q[r]]).strip()
        elif r in inv_c:
            t = inv_c[r]
        elif r in inv_s:
            t = inv_s[r]
        else:
            continue
        if t:
            out[r] = t
    return out


def _glyph_bank(texts, fonts, size: int):
    """uint8 (rows, fonts, size, size) grayscale renders, ink 0 on 255: the
    encoder's input, one render per font so the font is drawn per step."""
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont

    bank = np.full((len(texts), len(fonts), size, size), 255, dtype=np.uint8)
    for fi, fp in enumerate(fonts):
        cache: dict = {}
        for ri, t in enumerate(texts):
            n = max(1, len(t))
            fs = int(size * 0.78 / n) if n > 1 else int(size * 0.78)
            font = cache.get(fs)
            if font is None:
                font = cache[fs] = ImageFont.truetype(fp, fs, index=0)
            im = Image.new("L", (size, size), 255)
            d = ImageDraw.Draw(im)
            left, top, right, bottom = d.textbbox((0, 0), t, font=font)
            d.text(
                ((size - (right - left)) / 2 - left, (size - (bottom - top)) / 2 - top),
                t,
                fill=0,
                font=font,
            )
            bank[ri, fi] = np.array(im)
    return torch.from_numpy(bank)


def _glyph_batch(
    bank, device, rng: random.Random, shift: int = 6, fonts=None, font_mean=False
):
    """One render per row: a random font (or ``fonts`` per row), or with
    ``font_mean`` the mean render over every font (a font-free glyph
    descriptor — attempt 7 showed the output tracking font 8× more than
    glyph), plus one random shift for the batch, as ink in [0, 1]."""
    import torch

    R, F = bank.shape[:2]
    if font_mean:
        x = bank.to(device).float().mean(dim=1).div_(255.0)
    else:
        f = (
            torch.tensor(fonts)
            if fonts is not None
            else torch.tensor([rng.randrange(F) for _ in range(R)])
        )
        x = bank[torch.arange(R), f].to(device).float().div_(255.0)
    x = 1.0 - x  # ink 1, paper 0
    if shift:
        dx, dy = rng.randint(-shift, shift), rng.randint(-shift, shift)
        x = torch.roll(x, shifts=(dy, dx), dims=(1, 2))
    return x.unsqueeze(1)


class GlyphEncoder:
    """``g(glyph render) → Δ_row`` in row-norm units. Small CNN + MLP, last
    layer zero-init (step 0 = pack rows), output split into an identity part
    and one layout vector:

        Δ_r = (d_r − mean_rows d) + c

    Three attempts taught the shape. (1) With a zero-init last layer every
    one of the ``hidden`` weights feeding an output coordinate steps by
    ``lr`` in the same direction, so the output moves ``hidden × lr`` per
    step → ``out_scale`` 1/64. (2) Any component every row shares gets the
    *summed* gradient of every ext token in the batch — a direction
    consistent enough that Adam marches at full lr forever (a shared bias
    reached 36× row norm, every row identical). (3) A per-row output-norm
    cap does not help: at the cap the output is ``d / ‖d‖``, the internal
    ``d`` keeps growing along the common direction and the per-glyph part is
    divided by it (spread 0.000 for 750 steps).

    So the common mode is *projected out* of the CNN's output — centring
    across the full row table every step removes the common-mode gradient
    from the shared weights, leaving the per-glyph part with the same
    inconsistent gradients free rows had (they saturated at ~1× on every
    rows arm) — and the layout mode ("big glyph on a blank canvas", the
    direction every rows arm converged to) lives in one free vector ``c`` at
    the rows lr, bounded on the **parameter** after each optimizer step
    (``clamp_common``), never on the output."""

    def __new__(
        cls,
        dim: int,
        width: int = 32,
        out_scale: float = 1.0 / 64,
        common_cap: float = 0.75,
        pool: str = "spatial",
        glyph_size: int = 96,
        head_init: str = "zero",
    ):
        import torch
        from torch import nn

        ch = [1, width, width * 2, width * 4, width * 8]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1), nn.GELU()]
        grid = (glyph_size + 15) // 16
        feat = ch[-1]

        class _Enc(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(*layers)
                self.pool = pool
                # ``mean``: global mean pool — channel statistics only, which
                # carry ink mass / stroke weight (font) and not arrangement
                # (glyph); ``spatial``: flatten the final grid and project,
                # so the arrangement survives
                self.proj = (
                    nn.Linear(feat * grid * grid, feat) if pool == "spatial" else None
                )
                self.norm = nn.LayerNorm(feat)
                self.head = nn.Sequential(
                    nn.Linear(feat, 512), nn.GELU(), nn.Linear(512, dim)
                )
                # rank lever (2026-09-14): attempt 10's zero-init last layer
                # grew as one outer product (PR 2.8, table PR 1.0) — with
                # Adam a consistent gradient direction on shared weights
                # marches while the per-glyph tail random-walks. ``random``
                # keeps the default full-rank init; stage_train rescales it
                # to --init_spread row norms on the reference render
                if head_init == "zero":
                    nn.init.zeros_(self.head[-1].weight)
                nn.init.zeros_(self.head[-1].bias)
                self.common = nn.Parameter(torch.zeros(dim))
                self.out_scale = out_scale
                self.common_cap = common_cap

            def features(self, x):
                """pooled conv features, pre-LayerNorm (the collapse diagnostic
                reads their spread across rows)"""
                h = self.conv(x)
                if self.pool == "spatial":
                    return self.proj(h.flatten(1))
                return h.mean(dim=(2, 3))

            def identity(self, x):
                """centred per-glyph part only (mean over the rows passed in —
                call with the full table)"""
                h = self.norm(self.features(x))
                d = self.head(h) * self.out_scale
                return d - d.mean(dim=0, keepdim=True)

            def forward(self, x):
                return self.identity(x) + self.common

            @torch.no_grad()
            def clamp_common(self):
                n = self.common.norm()
                if n > self.common_cap:
                    self.common.mul_(self.common_cap / n)

            def enc_params(self):
                return [p for n, p in self.named_parameters() if n != "common"]

        return _Enc()


# ----------------------------------------------------------------------------
# stage: train


def _encode_captions(captions, device):
    """Unique captions → dict caption -> (prompt_embeds, attn_mask, t5_ids, t5_mask) on CPU."""
    import torch

    from library.inference.models import load_text_encoder
    from library.inference.text import ensure_text_strategies

    tok, enc = ensure_text_strategies(_ck().text_encoder, vocab_pack=None)
    te = load_text_encoder(
        text_encoder=_ck().text_encoder, dtype=torch.bfloat16, device=device
    ).eval()
    uniq = sorted(set(captions))
    cache = {}
    with torch.no_grad():
        for i in range(0, len(uniq), 16):
            chunk = uniq[i : i + 16]
            tokens = tok.tokenize(chunk)
            pe, am, t5, t5m = enc.encode_tokens(tok, [te], tokens)
            for j, c in enumerate(chunk):
                cache[c] = (
                    pe[j].to(torch.bfloat16).cpu(),
                    am[j].cpu(),
                    t5[j].long().cpu(),
                    t5m[j].cpu(),
                )
    te.to("cpu")
    del te
    torch.cuda.empty_cache()
    return cache


def _ext_ids_of(cache):
    from library.anima.ext_vocab import T5_TABLE_SIZE

    ids = set()
    for _, (_, _, t5, _) in cache.items():
        ids.update(int(v) - T5_TABLE_SIZE for v in t5.tolist() if v >= T5_TABLE_SIZE)
    return ids


def stage_train(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model
    from library.runtime.noise import fm_training_batch

    torch.manual_seed(a.seed)
    data = _data_dir(a)
    arm_dir = _arm_dir(a)
    arm_dir.mkdir(parents=True, exist_ok=True)
    recs_all = [
        json.loads(ln) for ln in (data / "train.jsonl").read_text().splitlines()
    ]
    ev = json.loads((data / "eval.json").read_text())
    args = _gen_args(a.train_size, a.steps, a.cfg, arm_dir)
    device = get_generation_settings(args).device

    # W2d held-out split: N single chars never appear in a training item (a
    # single, a combo or a corpus line containing them); the eval set gains
    # every held-out char as group ``single_held``.
    held: list = []
    keep = list(range(len(recs_all)))
    if a.held_out or a.held_out_chars:
        assert a.arm == "encoder", "--held_out is the encoder arm's generalisation test"
        inv = sorted(
            {r["text"] for r in recs_all if r["src"] == "font" and len(r["text"]) == 1}
        )
        if a.held_out_chars:
            # Run 2: an IDS-structured split — composites whose atoms stay
            # trained — is chosen by hand, not drawn
            held = list(a.held_out_chars)
            missing = [c for c in held if c not in inv]
            assert not missing, f"--held_out_chars not in the inventory: {missing}"
        else:
            held = random.Random(a.seed + 7).sample(inv, a.held_out)
        hs = set(held)
        keep = [i for i, r in enumerate(recs_all) if not (hs & set(r["text"]))]
        for e in ev:
            if e["group"] == "single" and e["text"] in hs:
                e["group"] = "single_held"
        present = {e["text"] for e in ev if e["group"] == "single_held"}
        ev += [
            {"group": "single_held", "text": c, "caption": TPL_BUBBLE.format(c)}
            for c in held
            if c not in present
        ]
        (arm_dir / "eval.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1))
        (arm_dir / "held_out.json").write_text(json.dumps(held, ensure_ascii=False))
        print(
            f"held-out {len(held)} chars {''.join(held)}; train items {len(keep)}/{len(recs_all)}",
            flush=True,
        )
    recs = [recs_all[i] for i in keep]

    # 1. text (Qwen side + pack-routed T5 ids), pre-adapter
    t0 = time.time()
    cache = _encode_captions([r["caption"] for r in recs], device)
    train_ext = _ext_ids_of(cache)
    ev_cache = _encode_captions([e["caption"] for e in ev], device)
    ev_ext = {
        e["text"]: sorted(_ext_ids_of({e["caption"]: ev_cache[e["caption"]]}))
        for e in ev
    }
    cov = {
        t: (len([x for x in ids if x in train_ext]), len(ids))
        for t, ids in ev_ext.items()
    }
    print(
        f"text: {len(cache)} captions, {len(train_ext)} ext rows touched, {time.time() - t0:.0f}s",
        flush=True,
    )
    (arm_dir / "eval_coverage.json").write_text(
        json.dumps(cov, ensure_ascii=False, indent=1)
    )

    # 2. latents
    t0 = time.time()
    lat_file = data / f"latents_{a.train_size}.pt"
    if lat_file.exists():
        lat = torch.load(lat_file)
    else:
        vae = _load_vae(device)
        lat = []
        with torch.no_grad():
            for i in range(0, len(recs_all), 8):
                px = np.stack(
                    [
                        np.array(
                            Image.open(r["file"])
                            .convert("RGB")
                            .resize((a.train_size, a.train_size))
                        )
                        for r in recs_all[i : i + 8]
                    ]
                )
                px = (
                    torch.from_numpy(px)
                    .permute(0, 3, 1, 2)
                    .float()
                    .div(127.5)
                    .sub(1.0)
                    .to(device)
                )  # IMAGE_TRANSFORMS range
                lat.append(vae.encode_pixels_to_latents(px).float().cpu())
        lat = torch.cat(lat)
        torch.save(lat, lat_file)
        del vae
        torch.cuda.empty_cache()
    if len(keep) != len(recs_all):
        lat = lat[keep]
    print(f"latents: {tuple(lat.shape)} in {time.time() - t0:.0f}s", flush=True)

    # 3. frozen DiT + trainables
    anima = load_dit_model(args, device, torch.bfloat16)
    anima.requires_grad_(False)
    from library.anima.vocab_pack import attached_pack_rows

    assert attached_pack_rows(anima), "no vocab pack attached to the DiT"
    # mean row norm of the rows we touch (the pack table lives on the hook closure; read it via the strategy)
    from library.anima.vocab_pack import strategy_pack
    from library.inference.text import ensure_text_strategies

    tok, _ = ensure_text_strategies(_ck().text_encoder, vocab_pack=None)
    pack = strategy_pack(tok)
    rows = pack.table[sorted(train_ext)].float()
    row_scale = float(rows.norm(dim=1).mean())
    print(
        f"pack rows: mean norm {row_scale:.3f} (std {rows.norm(dim=1).std():.3f}), dim {rows.shape[1]}",
        flush=True,
    )
    enc = None
    bank = None
    free = None
    free_mask = None
    if a.arm == "encoder":
        rows_all = sorted(set(train_ext) | {i for ids in ev_ext.values() for i in ids})
        row_text = _row_texts(tok, pack, rows_all)
        rows_all = [r for r in rows_all if r in row_text]
        bank = _glyph_bank([row_text[r] for r in rows_all], _fonts(), a.glyph_size)
        delta = ExtDelta(anima, rows_all, rows.shape[1], device, row_scale)
        enc = GlyphEncoder(
            rows.shape[1],
            out_scale=a.out_scale,
            common_cap=a.common_cap,
            pool=a.enc_pool,
            glyph_size=a.glyph_size,
            head_init=a.head_init,
        ).to(device)
        font_mean = a.font_mode == "mean"
        if a.head_init == "random":
            with torch.no_grad():
                xref = _glyph_batch(
                    bank,
                    device,
                    random.Random(0),
                    shift=0,
                    fonts=[0] * bank.shape[0],
                    font_mean=font_mean,
                )
                spread0 = float(enc.identity(xref).norm(dim=1).mean())
                enc.head[-1].weight.mul_(a.init_spread / max(spread0, 1e-8))
                spread1 = float(enc.identity(xref).norm(dim=1).mean())
            print(
                f"encoder head random init: spread {spread0:.4f} → {spread1:.3f} row norms",
                flush=True,
            )
        params = [
            {"params": enc.enc_params(), "lr": a.lr_enc},
            {"params": [enc.common], "lr": a.lr_common},
        ]
        is_train_row = torch.tensor([r in train_ext for r in delta.ext_ids])
        if a.init_encoder:
            src = torch.load(a.init_encoder, map_location="cpu", weights_only=False)
            enc.load_state_dict({k: v.to(device) for k, v in src["encoder"].items()})
            print(
                f"encoder warm start: {a.init_encoder} (arm {src.get('arm')}, "
                f"common norm {float(enc.common.norm()):.3f})",
                flush=True,
            )
        if a.free_residual > 0:
            # Run 1d hybrid: row_i = g(glyph_i) + f_i on trained rows only
            # (mask zeroes held-out rows in the forward, so they get neither
            # a residual nor a gradient); μ · mean_i ‖f_i‖² over trained rows
            free = torch.nn.Parameter(
                torch.zeros(len(delta.ext_ids), rows.shape[1], device=device)
            )
            free_mask = is_train_row.float().unsqueeze(1).to(device)
            n_free = float(is_train_row.sum())
            params.append({"params": [free], "lr": a.lr_free})
            n_warm = 0
            if a.init_free:
                # warm start the per-row residual by ext id (rows the source
                # never had stay at zero — new words start from g alone)
                src_f = torch.load(a.init_free, map_location="cpu", weights_only=False)
                src_idx = {int(e): i for i, e in enumerate(src_f["delta"]["ext_ids"])}
                with torch.no_grad():
                    for i, e in enumerate(delta.ext_ids):
                        j = src_idx.get(int(e))
                        if j is not None:
                            free[i] = src_f["free"][j].to(device)
                            n_warm += 1
            print(
                f"free residual: {int(n_free)} trained rows, μ {a.free_residual:g}, lr {a.lr_free:g}"
                + (
                    f", {n_warm} rows warm-started from {a.init_free}"
                    if a.init_free
                    else ""
                ),
                flush=True,
            )
        print(
            f"encoder: {sum(p.numel() for p in enc.parameters()) / 1e6:.2f}M params, "
            f"{len(rows_all)} rows ({int(is_train_row.sum())} in training captions), "
            f"glyph bank {tuple(bank.shape)}, out_scale {a.out_scale:.4g}, "
            f"common lr {a.lr_common:g} cap {a.common_cap:g}, pool {a.enc_pool}, "
            f"font {a.font_mode}",
            flush=True,
        )
    else:
        delta = ExtDelta(anima, train_ext, rows.shape[1], device, row_scale)
        params = [{"params": [delta.raw], "lr": a.lr_rows}]
    lora = None
    if a.arm == "rows_adapter":
        lora = AdapterLoRA(anima, a.adapter_rank, device)
        params.append({"params": list(lora.params), "lr": a.lr_adapter})
        print(
            f"adapter LoRA r{a.adapter_rank} on {len(lora.patched)} Linears", flush=True
        )
    opt = torch.optim.AdamW(params, weight_decay=0.0, betas=(0.9, 0.99))
    sched = None
    if a.lr_decay == "cosine":
        # the identity has no parameter-side bound (attempt 8/9: linear norm
        # growth, 1.6× at 2000 steps); cosine to 0 stops it late
        import math

        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda st: 0.5 * (1 + math.cos(math.pi * min(st / a.train_steps, 1.0)))
        )
    anima.train()
    if a.grad_ckpt:
        anima.enable_gradient_checkpointing(unsloth_offload=False)
    if a.compile:
        # Block compile is the repo's first OOM remedy (bit-exact, cuts
        # activation memory); one train_size → one token family. Compile after
        # the trainables are attached (the ExtDelta hooks sit on the adapter's
        # embed, the adapter LoRA patches Linears the blocks never see).
        from library.runtime.harness import compile_blocks_for_training

        compile_blocks_for_training(
            anima,
            None,
            backend="inductor",
            n_token_families=1,
            activation_memory_budget=a.activation_memory_budget,
            partitioner_aggressive_recomputation=bool(a.aggressive_recompute),
            grad_ckpt=bool(a.grad_ckpt),
        )

    # 4. loop — balanced data (W2a) draws one layout group per batch
    groups = None
    if "layout_id" in recs[0]:
        by_lid: dict = {}
        for i, r in enumerate(recs):
            by_lid.setdefault(r["layout_id"], []).append(i)
        groups = list(by_lid.values())
        sizes = {len(x) for x in groups}
        assert sizes == {a.batch}, (
            f"balanced data has group sizes {sizes}; pass --batch to match"
        )
        print(f"batching: {len(groups)} layout groups of {a.batch}", flush=True)
    unit = 1 if groups else a.batch
    n = len(groups) if groups else len(recs)
    order = list(range(n))
    random.Random(a.seed).shuffle(order)
    ptr = 0
    log = []
    aug_rng = random.Random(a.seed + 11)
    killed = ""
    t0 = time.time()
    for step in range(1, a.train_steps + 1):
        if ptr + unit > n:
            random.Random(a.seed + step).shuffle(order)
            ptr = 0
        sel = order[ptr : ptr + unit]
        ptr += unit
        idx = groups[sel[0]] if groups else sel
        latents = lat[idx].to(device)
        noise = torch.randn_like(latents)
        noisy, ts, target = fm_training_batch(
            latents,
            noise,
            dtype=torch.bfloat16,
            device=device,
            t_min=a.t_min,
            t_max=a.t_max,
        )
        pe = torch.stack([cache[recs[i]["caption"]][0] for i in idx]).to(device)
        am = torch.stack([cache[recs[i]["caption"]][1] for i in idx]).to(device)
        t5 = torch.stack([cache[recs[i]["caption"]][2] for i in idx]).to(device)
        t5m = torch.stack([cache[recs[i]["caption"]][3] for i in idx]).to(device)
        pm = torch.zeros(
            len(idx),
            1,
            latents.shape[-2],
            latents.shape[-1],
            dtype=torch.bfloat16,
            device=device,
        )
        if enc is not None:
            delta.raw = enc(_glyph_batch(bank, device, aug_rng, font_mean=font_mean))
            if free is not None:
                delta.raw = delta.raw + free * free_mask
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = anima(
                noisy.unsqueeze(2),
                ts,
                pe,
                padding_mask=pm,
                target_input_ids=t5,
                target_attention_mask=t5m,
                source_attention_mask=am,
            )
        pred = pred.squeeze(2)
        loss_fm = F.mse_loss(pred.float(), target.float())
        loss = loss_fm
        decor_val = None
        if enc is not None and a.decor > 0:
            # plan_wake Run 1b amended: the FM gradient's sign-consistent
            # direction marches the table to rank 1 under Adam (attempt 10
            # PR 1.0, rinit 23 → 1.24). Penalise pairwise cos² of the centred
            # *trained* rows (189×189 per step, no DiT forward): ≈ 1 at PR 1,
            # → 0 as rows spread to the free-rows geometry (pairwise cos 0.04).
            # Held-out rows are excluded from both the centring and the pairs
            # so they get no direct gradient.
            tr = delta.raw[is_train_row.to(delta.raw.device)].float()
            cen_tr = tr - tr.mean(0, keepdim=True)
            cn_tr = F.normalize(cen_tr, dim=1)
            sim_tr = cn_tr @ cn_tr.T
            n_tr = sim_tr.shape[0]
            decor_val = ((sim_tr**2).sum() - (sim_tr.diagonal() ** 2).sum()) / (
                n_tr * (n_tr - 1)
            )
            loss = loss + a.decor * decor_val
        free_pen = None
        if free is not None:
            free_pen = ((free * free_mask) ** 2).sum() / n_free
            loss = loss + a.free_residual * free_pen
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        if enc is not None:
            enc.clamp_common()  # projected descent on c: no creep
        if step % 25 == 0 or step == 1:
            dn = (delta.raw.detach() * row_scale).norm(dim=1)
            if enc is not None:
                dn_held = dn[~is_train_row.to(dn.device)]
                dn = dn[is_train_row.to(dn.device)]
            rec = {
                "step": step,
                "loss": loss_fm.item(),
                "delta_norm_mean": float(dn.mean()),
                "delta_norm_max": float(dn.max()),
                "rel": float(dn.mean() / row_scale),
                "it_s": step / (time.time() - t0),
            }
            if enc is not None:
                # identity lives in the spread between rows, not the common
                # mode. The training draw's spread swings 0.03–0.26 between
                # logs with the font/shift drawn (attempt 4), so the kill rule
                # reads a fixed reference render: font 0, no shift.
                raw = delta.raw.detach()
                rec["rel_spread"] = float(
                    (raw - raw.mean(0, keepdim=True)).norm(dim=1).mean()
                )
                with torch.no_grad():
                    xref = _glyph_batch(
                        bank,
                        device,
                        aug_rng,
                        shift=0,
                        fonts=[0] * bank.shape[0],
                        font_mean=font_mean,
                    )
                    ref = enc.identity(xref)
                    feat = enc.features(xref)
                rec["rel_spread_ref"] = float(ref.norm(dim=1).mean())
                # feature spread across rows relative to the common feature:
                # ~0 here with ~0 spread = dead features, not a weight kick
                rec["feat_spread"] = float(
                    (feat - feat.mean(0, keepdim=True)).norm(dim=1).mean()
                    / feat.mean(0).norm().clamp_min(1e-6)
                )
                rec["rel_common"] = float(enc.common.detach().norm())
                rec["rel_max"] = float(raw.norm(dim=1).max())
                # table rank: attempt 10's centred table had participation
                # ratio 1.0 (one axis, cos 0.85 with c) and nearest-neighbour
                # cos ≥ 0.84 for every kana — the instrument the data lever
                # and the rank lever are judged on
                cen = (raw - raw.mean(0, keepdim=True)).float()
                sv = torch.linalg.svdvals(cen)
                pw = sv**2 / (sv**2).sum().clamp_min(1e-12)
                rec["table_pr"] = float(1.0 / (pw**2).sum().clamp_min(1e-12))
                cn = F.normalize(cen, dim=1)
                sim = cn @ cn.T
                sim.fill_diagonal_(-1.0)
                rec["nn_cos"] = float(sim.max(dim=1).values.mean())
                if decor_val is not None:
                    rec["decor"] = float(decor_val.detach())
                    rec["loss_total"] = float(loss.detach())
                if free is not None:
                    # the hybrid's instrument: how much of the trained rows'
                    # identity the free residual carries vs the shared g
                    # (rel_spread_ref). ≫ 1 = g learned nothing (lookup)
                    fn = (free.detach() * free_mask).norm(dim=1)
                    rec["free_norm"] = float(fn.sum() / n_free)
                    rec["free_max"] = float(fn.max())
                    rec["free_ratio"] = float(
                        rec["free_norm"] / max(rec["rel_spread_ref"], 1e-6)
                    )
                    rec["loss_total"] = float(loss.detach())
                    svg = torch.linalg.svdvals(ref.float())
                    pwg = svg**2 / (svg**2).sum().clamp_min(1e-12)
                    rec["g_pr"] = float(1.0 / (pwg**2).sum().clamp_min(1e-12))
                if dn_held.numel():
                    rec["rel_held"] = float(dn_held.mean() / row_scale)
            if lora is not None:
                rec["lora_b_norm"] = float(
                    sum(p.norm() ** 2 for p in list(lora.params)[1::2]) ** 0.5
                )
            log.append(rec)
            print(json.dumps(rec), flush=True)
            if enc is not None:
                # plan_wake W2d kill rules: identity not moving, or a row
                # walking off-manifold — stop before spending the eval
                if (
                    a.kill_spread_step
                    and step >= a.kill_spread_step
                    and rec["rel_spread_ref"] < a.kill_spread
                ):
                    killed = (
                        f"KILL: rel_spread_ref {rec['rel_spread_ref']:.4f} < {a.kill_spread} "
                        f"at step {step} (>= {a.kill_spread_step})"
                    )
                if a.kill_max_row and rec["rel_max"] > a.kill_max_row:
                    killed = f"KILL: max row norm {rec['rel_max']:.3f} > {a.kill_max_row}× at step {step}"
                if killed:
                    # save the table anyway — a killed run is still renderable
                    # (attempt 8 was killed at 2.5× with nothing to eval)
                    print(killed, flush=True)
                    break
    if enc is not None:
        # the shipped table: the encoder's mean over fonts, no shift — the
        # ExtDelta format so eval / native / classify run unchanged
        with torch.no_grad():
            enc.eval()
            if font_mean:
                delta.raw = enc(
                    _glyph_batch(bank, device, aug_rng, shift=0, font_mean=True)
                )
            else:
                delta.raw = torch.stack(
                    [
                        enc(
                            _glyph_batch(
                                bank,
                                device,
                                aug_rng,
                                shift=0,
                                fonts=[f] * bank.shape[0],
                            )
                        )
                        for f in range(bank.shape[1])
                    ]
                ).mean(0)
            if free is not None:
                delta.raw = delta.raw + free * free_mask
    sd = {"delta": delta.state_dict(), "arm": a.arm, "args": vars(a)}
    if enc is not None:
        sd["encoder"] = {k: v.cpu() for k, v in enc.state_dict().items()}
        sd["held_out"] = held
        sd["row_text"] = {int(r): row_text[r] for r in delta.ext_ids}
        if free is not None:
            sd["free"] = (free.detach() * free_mask).cpu()
    if lora is not None:
        sd["lora"] = lora.state_dict()
        sd["adapter_rank"] = a.adapter_rank
    sd["killed"] = killed
    torch.save(sd, arm_dir / "trained.pt")
    (arm_dir / "train_log.json").write_text(json.dumps(log, indent=1))
    if killed:
        raise SystemExit(killed)
    print(
        f"train: {a.train_steps} steps in {(time.time() - t0) / 60:.1f} min → {arm_dir / 'trained.pt'}",
        flush=True,
    )
    del anima
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
# stage: classify (diffusion-classifier diagnostic, before a W2c loss)


def stage_classify(a):
    """Same-noise N-way diffusion classifier over the single kana.

    Fresh held-out single renders (unseen fonts/layouts); each is noised once
    per σ and scored under every kana's caption in its own template — identical
    inputs, only the address differs. Summed FM error per candidate, right =
    argmin. Run with the arm's delta and with it off (pack rows) as the control.
    Answers: which σ band carries identity (where a CE term belongs), the error
    gap/spread that sets its temperature, and whether the rows already
    discriminate (then renders fail in sampling, not in the rows).
    """
    import numpy as np
    import torch
    from PIL import Image

    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model

    data = _data_dir(a)
    arm_dir = _arm_dir(a)
    recs = [json.loads(ln) for ln in (data / "train.jsonl").read_text().splitlines()]
    singles = {r["text"] for r in recs if r["src"] == "font" and len(r["text"]) == 1}
    kana = [c for c in KANA if c in singles]
    tpls = {"bubble": TPL_BUBBLE, "plain": TPL_PLAIN}
    sigmas = [float(x) for x in a.cls_t.split(",")]
    out = arm_dir / (f"classify_{a.eval_tag}" if a.eval_tag else "classify")
    (out / "img").mkdir(parents=True, exist_ok=True)
    args = _gen_args(a.train_size, a.steps, a.cfg, out)
    device = get_generation_settings(args).device

    # 1. held-out renders: own rng stream, so fonts/layouts are not the train set's
    rng = random.Random(10_000 + a.seed)
    fonts = _fonts()
    items = []
    for ki, ch in enumerate(kana):
        for j in range(a.cls_per_kana):
            im, bubble = _render_string(ch, rng.choice(fonts), rng, size=a.train_size)
            fn = out / "img" / f"{ki:02d}_{j}.png"
            im.save(fn)
            items.append(
                {"file": str(fn), "text": ch, "tpl": "bubble" if bubble else "plain"}
            )

    # 2. text + latents + one noise draw per (item, σ), shared by both conds
    cache = _encode_captions([tpls[t].format(c) for t in tpls for c in kana], device)
    vae = _load_vae(device)
    with torch.no_grad():
        px = np.stack([np.array(Image.open(it["file"]).convert("RGB")) for it in items])
        px = torch.from_numpy(px).permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
        lat = torch.cat(
            [
                vae.encode_pixels_to_latents(px[i : i + 8].to(device)).float().cpu()
                for i in range(0, len(px), 8)
            ]
        )
    del vae
    torch.cuda.empty_cache()
    g = torch.Generator().manual_seed(a.seed)
    noise = torch.randn((len(items), len(sigmas), *lat.shape[1:]), generator=g)

    # 3. frozen DiT + the arm's delta
    sd = torch.load(arm_dir / "trained.pt")
    assert "lora" not in sd, "classify covers rows-only arms"
    anima = load_dit_model(args, device, torch.bfloat16)
    anima.requires_grad_(False)
    anima.eval()
    delta = ExtDelta(
        anima,
        sd["delta"]["ext_ids"],
        sd["delta"]["raw"].shape[1],
        device,
        sd["delta"]["row_scale"],
    )
    delta.load(sd["delta"])

    # 4. err[cond, item, σ, kana] = FM error summed over the latent
    K = len(kana)
    conds = ("trained", "floor")
    err = torch.zeros(len(conds), len(items), len(sigmas), K)
    h, w = lat.shape[-2:]
    t0 = time.time()
    n_fwd = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ci, cond in enumerate(conds):
            delta.scale = 1.0 if cond == "trained" else 0.0
            for ii, it in enumerate(items):
                caps = [tpls[it["tpl"]].format(c) for c in kana]
                x = lat[ii : ii + 1].to(device)
                for si, sig in enumerate(sigmas):
                    nz = noise[ii, si : si + 1].to(device)
                    target = nz - x
                    noisy = ((1.0 - sig) * x + sig * nz).to(torch.bfloat16)
                    for k0 in range(0, K, a.cls_batch):
                        cs = caps[k0 : k0 + a.cls_batch]
                        b = len(cs)
                        pred = anima(
                            noisy.repeat(b, 1, 1, 1).unsqueeze(2),
                            torch.full((b,), sig, device=device),
                            torch.stack([cache[c][0] for c in cs]).to(device),
                            padding_mask=torch.zeros(
                                b, 1, h, w, dtype=torch.bfloat16, device=device
                            ),
                            target_input_ids=torch.stack([cache[c][2] for c in cs]).to(
                                device
                            ),
                            target_attention_mask=torch.stack(
                                [cache[c][3] for c in cs]
                            ).to(device),
                            source_attention_mask=torch.stack(
                                [cache[c][1] for c in cs]
                            ).to(device),
                        ).squeeze(2)
                        err[ci, ii, si, k0 : k0 + b] = (
                            ((pred.float() - target) ** 2).sum(dim=(1, 2, 3)).cpu()
                        )
                        n_fwd += b
                if ii % 8 == 7:
                    print(
                        f"classify {cond}: item {ii + 1}/{len(items)}, "
                        f"{n_fwd / (time.time() - t0):.1f} fwd/s",
                        flush=True,
                    )
    del anima
    torch.cuda.empty_cache()
    labels = torch.tensor([kana.index(it["text"]) for it in items])
    torch.save(
        {
            "err": err,
            "labels": labels,
            "kana": kana,
            "sigmas": sigmas,
            "conds": conds,
            "items": items,
        },
        out / "classify.pt",
    )
    _report_classify(out, err, labels, kana, sigmas, conds)


def _report_classify(out: Path, err, labels, kana, sigmas, conds):
    import torch

    K = len(kana)
    N = len(labels)
    ar = torch.arange(N)

    def stats(e):  # e: N×K summed errors
        right = e[ar, labels]
        rank = (e < right[:, None]).sum(1).float()  # 0 = right kana wins
        wrong_mean = (e.sum(1) - right) / (K - 1)
        gap = (wrong_mean - right).mean()
        spread = e.std(1).mean()
        return rank, gap, spread, right.mean()

    L = [
        "# classify — same-noise diffusion classifier",
        "",
        f"{N} held-out single renders, {K}-way (chance {1 / K:.3f}); summed FM "
        "error per latent, right kana = argmin. Σσ = errors summed over the grid.",
        "",
        "| cond | σ | top-1 | mean rank | err right | gap (wrong − right) | spread | gap/spread |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ci, cond in enumerate(conds):
        for si, sig in list(enumerate(sigmas)) + [(None, None)]:
            e = err[ci].sum(1) if si is None else err[ci, :, si]
            rank, gap, spread, right = stats(e)
            L.append(
                f"| {cond} | {'Σσ' if sig is None else f'{sig:.2f}'} | "
                f"{(rank == 0).float().mean():.3f} | {rank.mean() + 1:.1f} | "
                f"{right:.0f} | {gap:.1f} | {spread:.1f} | {gap / spread:.3f} |"
            )
    L += ["", "## Per kana (Σσ): right/total, most-picked wrong kana", ""]
    for ci, cond in enumerate(conds):
        pred = err[ci].sum(1).argmin(1)
        parts = []
        for ki, ch in enumerate(kana):
            m = labels == ki
            hit = int((pred[m] == ki).sum())
            wrong = [kana[int(q)] for q in pred[m] if int(q) != ki]
            parts.append(
                f"{ch} {hit}/{int(m.sum())}" + (f"→{''.join(wrong)}" if wrong else "")
            )
        picks = torch.bincount(pred, minlength=K)
        top = sorted(range(K), key=lambda k: -int(picks[k]))[:5]
        L.append(f"- **{cond}**: " + ", ".join(parts))
        L.append(
            "  - most-picked overall: "
            + ", ".join(f"{kana[k]} ×{int(picks[k])}" for k in top)
        )
    (out / "classify.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


# ----------------------------------------------------------------------------
# stage: eval


def stage_eval(a):
    import torch

    from library.inference.generation import generate, get_generation_settings
    from library.inference.models import load_dit_model, load_shared_models

    data = _data_dir(a)
    arm_dir = _arm_dir(a)
    ev_file = arm_dir / "eval.json"  # encoder arms: held-out singles added
    if not ev_file.exists():
        ev_file = data / "eval.json"
    ev = json.loads(ev_file.read_text())
    if a.eval_groups:
        keep = set(a.eval_groups.split(","))
        ev = [e for e in ev if e["group"] in keep]
    if a.eval_limit:
        seen: dict = {}
        ev = [
            e
            for e in ev
            if seen.setdefault(e["group"], []).append(1)
            or len(seen[e["group"]]) <= a.eval_limit
        ]
    sd = torch.load(arm_dir / "trained.pt")
    eval_dir = arm_dir / f"eval_{a.eval_tag}" if a.eval_tag else arm_dir
    eval_dir.mkdir(parents=True, exist_ok=True)
    args = _gen_args(a.eval_size, a.steps, a.cfg, eval_dir / "img")
    gen = get_generation_settings(args)
    device = gen.device
    shared = load_shared_models(args)
    shared["conds_cache"] = {}
    anima = load_dit_model(args, device, torch.bfloat16)
    anima.eval()
    shared["model"] = anima
    delta = ExtDelta(
        anima,
        sd["delta"]["ext_ids"],
        sd["delta"]["raw"].shape[1],
        device,
        sd["delta"]["row_scale"],
    )
    delta.load(sd["delta"])
    lora = None
    if "lora" in sd:
        lora = AdapterLoRA(anima, sd["adapter_rank"], device)
        lora.load(sd["lora"])
    vae = _load_vae(device)
    (eval_dir / "img").mkdir(parents=True, exist_ok=True)
    manifest = []
    t0 = time.time()
    conds = ("trained",) if a.no_floor else ("floor", "trained")
    for cond in conds:
        s = 0.0 if cond == "floor" else 1.0
        delta.scale = s
        if lora is not None:
            lora.scale = s
        shared["conds_cache"].clear()
        for ei, e in enumerate(ev):
            for seed in range(a.seeds):
                fn = eval_dir / "img" / f"{cond}_{e['group']}_{ei:03d}_s{seed}.png"
                if not fn.exists():
                    a2 = copy.deepcopy(args)
                    a2.prompt = e["caption"]
                    a2.seed = seed
                    with torch.no_grad():
                        lat = generate(a2, gen, shared)
                    _decode(vae, lat, device).save(fn)
                manifest.append({"file": str(fn), "cond": cond, "seed": seed, **e})
    print(
        f"eval gen: {len(manifest)} images in {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )
    del anima, vae, shared
    torch.cuda.empty_cache()
    _read_eval(a, eval_dir, manifest, arm_dir)


def _read_eval(a, arm_dir: Path, manifest, train_dir: Path | None = None):
    """``arm_dir`` is where reads/report/sheets land; ``train_dir`` (default the
    same) holds ``eval_coverage.json`` from the train stage."""
    train_dir = train_dir or arm_dir
    from collections import defaultdict

    from PIL import Image

    rd = Readers(a.device)
    for m in manifest:
        reads = rd.read_image(_bgr(Path(m["file"])), whole=True)
        m["reads"] = reads
        best_s = min([cer(r["sfx"] or "", m["text"]) for r in reads] or [1.0])
        best_v = min([cer(r["vl"] or "", m["text"]) for r in reads] or [1.0])
        m["cer_sfx"], m["cer_vl"] = best_s, best_v
        m["exact"] = any(norm(r["sfx"] or "") == norm(m["text"]) for r in reads)
    (arm_dir / "eval_reads.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1)
    )
    agg = defaultdict(list)
    for m in manifest:
        agg[(m["group"], m["cond"])].append(m)
    lines = [
        f"# wake_probe — arm `{a.arm}` eval",
        "",
        "| group | cond | n | CER sfx | CER vl16 | exact (sfx) |",
        "|---|---|---|---|---|---|",
    ]
    for g in EVAL_GROUPS:
        for c in ("floor", "trained"):
            ms = agg.get((g, c), [])
            if not ms:
                continue
            lines.append(
                f"| {g} | {c} | {len(ms)} | {sum(m['cer_sfx'] for m in ms) / len(ms):.3f} | "
                f"{sum(m['cer_vl'] for m in ms) / len(ms):.3f} | {sum(m['exact'] for m in ms)}/{len(ms)} |"
            )
    cov = (
        json.loads((train_dir / "eval_coverage.json").read_text())
        if (train_dir / "eval_coverage.json").exists()
        else {}
    )
    if cov:
        lines += [
            "",
            "eval ext-row coverage (rows seen in training / rows in the string):",
        ]
        for g in EVAL_GROUPS:
            if g == "en":
                continue
            xs = [
                cov[m["text"]]
                for m in manifest
                if m["group"] == g
                and m["cond"] == "trained"
                and m["seed"] == 0
                and m["text"] in cov
            ]
            if xs:
                lines.append(f"- {g}: {sum(x[0] for x in xs)}/{sum(x[1] for x in xs)}")
    lines += [
        "",
        "Sheets: sheet_<group>.png — floor row then trained row per string, seed 0; label = ref / sfx read / vl16 read.",
    ]
    (arm_dir / "report.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    for g in EVAL_GROUPS:
        rows = []
        for m in [m for m in manifest if m["group"] == g and m["seed"] == 0]:
            r0 = m["reads"][-1] if m["reads"] else {"sfx": "", "vl": ""}
            rows.append(
                (
                    Image.open(m["file"]).convert("RGB"),
                    [
                        f"{m['cond']}: {m['text']}",
                        f"sfx {r0['sfx'] or ''}",
                        f"vl {r0['vl'] or ''}",
                    ],
                )
            )
        if rows:
            _sheet(rows, arm_dir / f"sheet_{g}.png", thumb=192, cols=6)


# ----------------------------------------------------------------------------
# stage: native (does the address survive a scene prompt?)

NATIVE_PROMPTS = (
    REPO / "project" / "cjk_aware_anima" / "assets" / "unmask_eval_prompts.txt"
)
NATIVE_CLAUSES = {
    # the trained clause shape, hung off a scene prompt instead of the template
    "en": '{p}, japanese text. Japanese text reads as "{k}".',
    # the user's phrasing: a Japanese-language clause (its own words route to
    # untrained pack rows; only the kana row carries the delta)
    "ja": "{p}. ひらがなの「{k}」という文字がある。",
}


def stage_native(a):
    """Render the blind-pairs scene prompts with a kana clause appended, delta
    off (floor) and on (trained), same seeds; read; sheet per (prompt, kana).
    The eval set asks for the glyph on a bare canvas; this asks for it inside
    an ordinary scene — the product condition W3 needs."""
    import torch

    from library.inference.generation import generate, get_generation_settings
    from library.inference.models import load_dit_model, load_shared_models

    arm_dir = _arm_dir(a)
    sd = torch.load(arm_dir / "trained.pt")
    assert "lora" not in sd, "native covers rows-only arms"
    out = arm_dir / (f"native_{a.eval_tag}" if a.eval_tag else "native")
    (out / "img").mkdir(parents=True, exist_ok=True)
    prompts = [
        ln.strip()
        for ln in Path(a.native_prompts).read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    if a.native_limit:
        prompts = prompts[: a.native_limit]
    chars = [c for c in a.native_chars.split(",") if c]
    clauses = [c for c in a.native_clauses.split(",") if c]
    items = [
        {
            "pi": pi,
            "prompt": p,
            "text": k,
            "clause": cl,
            "caption": NATIVE_CLAUSES[cl].format(p=p, k=k),
        }
        for pi, p in enumerate(prompts)
        for k in chars
        for cl in clauses
    ]
    args = _gen_args(a.eval_size, a.steps, a.cfg, out / "img")
    gen = get_generation_settings(args)
    device = gen.device
    shared = load_shared_models(args)
    shared["conds_cache"] = {}
    anima = load_dit_model(args, device, torch.bfloat16)
    anima.eval()
    shared["model"] = anima
    delta = ExtDelta(
        anima,
        sd["delta"]["ext_ids"],
        sd["delta"]["raw"].shape[1],
        device,
        sd["delta"]["row_scale"],
    )
    delta.load(sd["delta"])
    trained = set(delta.ext_ids)
    # which ext rows each caption touches, and how many of them carry a delta:
    # a JA clause whose tokenizer merges 「あ」 into one piece misses the row
    cache = _encode_captions([it["caption"] for it in items], device)
    for it in items:
        ids = sorted(_ext_ids_of({it["caption"]: cache[it["caption"]]}))
        it["ext_rows"] = len(ids)
        it["trained_rows"] = len([x for x in ids if x in trained])
    for cl in clauses:
        xs = [it for it in items if it["clause"] == cl]
        print(
            f"clause {cl}: ext rows/caption {sum(x['ext_rows'] for x in xs) / len(xs):.1f}, "
            f"trained rows/caption {sum(x['trained_rows'] for x in xs) / len(xs):.2f}",
            flush=True,
        )
    del cache
    vae = _load_vae(device)
    manifest = []
    t0 = time.time()
    conds = ("trained",) if a.no_floor else ("floor", "trained")
    for cond in conds:
        delta.scale = 0.0 if cond == "floor" else 1.0
        shared["conds_cache"].clear()
        for it in items:
            for seed in range(a.seeds):
                fn = (
                    out
                    / "img"
                    / f"{cond}_p{it['pi']:02d}_{it['text']}_{it['clause']}_s{seed}.png"
                )
                if not fn.exists():
                    a2 = copy.deepcopy(args)
                    a2.prompt = it["caption"]
                    a2.seed = seed
                    with torch.no_grad():
                        lat = generate(a2, gen, shared)
                    _decode(vae, lat, device).save(fn)
                manifest.append({"file": str(fn), "cond": cond, "seed": seed, **it})
    print(
        f"native gen: {len(manifest)} images in {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )
    del anima, vae, shared
    torch.cuda.empty_cache()
    _read_native(a, out, manifest, chars, clauses, conds)


def _read_native(a, out: Path, manifest, chars, clauses, conds):
    from collections import defaultdict

    from PIL import Image

    rd = Readers(a.device)
    for m in manifest:
        reads = rd.read_image(_bgr(Path(m["file"])), whole=True)
        m["reads"] = reads
        m["cer_sfx"] = min([cer(r["sfx"] or "", m["text"]) for r in reads] or [1.0])
        m["cer_vl"] = min([cer(r["vl"] or "", m["text"]) for r in reads] or [1.0])
        m["hit_sfx"] = any(norm(r["sfx"] or "") == norm(m["text"]) for r in reads)
        m["hit_vl"] = any(norm(r["vl"] or "") == norm(m["text"]) for r in reads)
        m["exact"] = m["hit_sfx"] and m["hit_vl"]
        m["any_cjk"] = any(CJK_RE.search(r["sfx"] or "") for r in reads)
    (out / "native_reads.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1)
    )
    agg = defaultdict(list)
    for m in manifest:
        agg[(m["clause"], m["cond"])].append(m)
    lines = [
        f"# wake_probe — arm `{a.arm}` native (scene prompts + kana clause)",
        "",
        f"prompts: `{a.native_prompts}`; chars {' '.join(chars)}; {a.seeds} seed(s); {a.eval_size}²",
        "",
        "| clause | cond | n | CER sfx | CER vl16 | hit sfx | hit vl | both | any CJK read |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cl in clauses:
        for c in conds:
            ms = agg.get((cl, c), [])
            if not ms:
                continue
            lines.append(
                f"| {cl} | {c} | {len(ms)} | {sum(m['cer_sfx'] for m in ms) / len(ms):.3f} | "
                f"{sum(m['cer_vl'] for m in ms) / len(ms):.3f} | {sum(m['hit_sfx'] for m in ms)} | "
                f"{sum(m['hit_vl'] for m in ms)} | {sum(m['exact'] for m in ms)} | "
                f"{sum(m['any_cjk'] for m in ms)} |"
            )
    lines += ["", "per kana (trained, both readers):", ""]
    for k in chars:
        for cl in clauses:
            ms = [
                m
                for m in manifest
                if m["text"] == k and m["clause"] == cl and m["cond"] == "trained"
            ]
            if ms:
                lines.append(f"- {k} / {cl}: {sum(m['exact'] for m in ms)}/{len(ms)}")
    lines += ["", "per prompt (trained, both readers, all kana/clauses):", ""]
    by_p = defaultdict(list)
    for m in manifest:
        if m["cond"] == "trained":
            by_p[m["pi"]].append(m)
    for pi in sorted(by_p):
        ms = by_p[pi]
        lines.append(
            f"- p{pi:02d} `{ms[0]['prompt']}`: {sum(m['exact'] for m in ms)}/{len(ms)}"
        )
    lines += [
        "",
        "Sheets: sheet_<kana>_<clause>.png — one row per prompt: "
        + ", ".join(f"{c} s{s}" for s in range(a.seeds) for c in conds)
        + "; label = prompt idx / sfx read / vl16 read.",
    ]
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    for k in chars:
        for cl in clauses:
            rows = []
            for pi in sorted(by_p):
                for seed in range(a.seeds):
                    for c in conds:
                        ms = [
                            m
                            for m in manifest
                            if m["pi"] == pi
                            and m["text"] == k
                            and m["clause"] == cl
                            and m["seed"] == seed
                            and m["cond"] == c
                        ]
                        if not ms:
                            continue
                        m = ms[0]
                        r0 = m["reads"][-1] if m["reads"] else {"sfx": "", "vl": ""}
                        rows.append(
                            (
                                Image.open(m["file"]).convert("RGB"),
                                [
                                    f"p{pi:02d} {c} s{seed}: {k}",
                                    f"sfx {r0['sfx'] or ''}",
                                    f"vl {r0['vl'] or ''}",
                                ],
                            )
                        )
            if rows:
                _sheet(
                    rows,
                    out / f"sheet_{k}_{cl}.png",
                    thumb=192,
                    cols=len(conds) * a.seeds,
                )


# ----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--stage",
        nargs="+",
        default=["all"],
        choices=["all", "salad", "data", "train", "eval", "classify", "native"],
    )
    p.add_argument("--arm", default="rows", choices=["rows", "rows_adapter", "encoder"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=28, help="inference steps")
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument(
        "--seeds", type=int, default=2, help="seeds per prompt (salad: 3 recommended)"
    )
    p.add_argument("--salad_size", type=int, default=768)
    p.add_argument("--train_size", type=int, default=512)
    p.add_argument("--eval_size", type=int, default=512)
    p.add_argument("--n_single", type=int, default=6, help="font renders per kana")
    p.add_argument("--n_combo", type=int, default=700)
    p.add_argument("--n_corpus", type=int, default=600)
    p.add_argument("--train_steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument(
        "--lr_rows", type=float, default=3e-3, help="in units of the mean pack-row norm"
    )
    p.add_argument("--lr_adapter", type=float, default=1e-4)
    p.add_argument(
        "--lr_enc", type=float, default=3e-4, help="encoder arm: AdamW lr on the CNN"
    )
    p.add_argument(
        "--lr_common",
        type=float,
        default=1e-3,
        help="encoder arm: lr on the shared layout vector c (row-norm units; the rows lr)",
    )
    p.add_argument(
        "--common_cap",
        type=float,
        default=0.75,
        help="encoder arm: ‖c‖ bound in row norms, applied to the parameter after each step",
    )
    p.add_argument(
        "--out_scale",
        type=float,
        default=1.0 / 64,
        help="encoder arm: scale on the zero-init head (raise one notch to 1/16 if spread stays flat)",
    )
    p.add_argument(
        "--kill_spread",
        type=float,
        default=0.05,
        help="encoder arm: abort if rel_spread_ref (fixed font, no shift) is below this once --kill_spread_step is reached",
    )
    p.add_argument(
        "--kill_spread_step",
        type=int,
        default=600,
        help="encoder arm: 0 disables the spread rule (--kill_max_row 0 disables the max-row rule)",
    )
    p.add_argument(
        "--kill_max_row",
        type=float,
        default=2.0,
        help="encoder arm: abort if any row's delta passes this many row norms",
    )
    p.add_argument(
        "--glyph_size", type=int, default=96, help="encoder arm: glyph render side"
    )
    p.add_argument(
        "--head_init",
        default="zero",
        choices=["zero", "random"],
        help="encoder arm: last head layer zero (attempts 1–10) or random full-rank, "
        "rescaled so the step-0 identity spread is --init_spread row norms",
    )
    p.add_argument(
        "--init_spread",
        type=float,
        default=1.0,
        help="encoder arm: --head_init random target spread on the reference render",
    )
    p.add_argument(
        "--decor",
        type=float,
        default=0.0,
        help="encoder arm: λ on mean_{i≠j} cos²(r_i, r_j) over the centred trained "
        "rows of the encoder table (0 = off; plan_wake Run 1b amended: ≈ 0.02 "
        "against an FM loss of ≈ 0.04)",
    )
    p.add_argument(
        "--free_residual",
        type=float,
        default=0.0,
        help="encoder arm: μ on mean_i ‖f_i‖² for a per-row free residual on the "
        "trained rows (row = g(glyph) + f_i; held-out rows get g only). 0 = off. "
        "plan_wake Run 1d: the semi-amortised hybrid — f carries the identity "
        "magnitude the shared head cannot, the L2 pushes what g can explain into g",
    )
    p.add_argument(
        "--lr_free",
        type=float,
        default=1e-3,
        help="encoder arm: lr of the free residual, row-norm units (W1 rows: 1e-3; 3e-3 walks off-manifold)",
    )
    p.add_argument(
        "--init_encoder",
        default="",
        help="encoder arm: warm-start the encoder (conv/proj/head + common) from another arm's trained.pt",
    )
    p.add_argument(
        "--lr_decay",
        default="none",
        choices=["none", "cosine"],
        help="train: lr schedule over --train_steps (cosine to 0; all param groups)",
    )
    p.add_argument(
        "--enc_pool",
        default="spatial",
        choices=["spatial", "mean"],
        help="encoder arm: feature pooling — spatial keeps the arrangement (glyph), "
        "mean keeps channel statistics only (attempts 4–7 tracked font, not glyph)",
    )
    p.add_argument(
        "--font_mode",
        default="mean",
        choices=["mean", "random"],
        help="encoder arm: input render — mean over every font (font-free) or one random font per row per step",
    )
    p.add_argument(
        "--held_out",
        type=int,
        default=0,
        help="encoder arm: N single chars removed from every training item and "
        "evaluated as group single_held (the generalisation test)",
    )
    p.add_argument(
        "--held_out_chars",
        default="",
        help="encoder arm: explicit held-out chars instead of --held_out's draw "
        "(Run 2: IDS composites whose atoms are trained, e.g. 明休男岩加相困森)",
    )
    p.add_argument("--adapter_rank", type=int, default=16)
    p.add_argument("--grad_ckpt", type=int, default=1)
    p.add_argument(
        "--compile",
        type=int,
        default=1,
        help="train: per-block torch.compile of the frozen DiT (the OOM remedy of record)",
    )
    p.add_argument("--activation_memory_budget", type=float, default=0.99)
    p.add_argument(
        "--aggressive_recompute",
        type=int,
        default=1,
        help="compile: partitioner aggressive recomputation (−VRAM, +~12 % s/it); 0 when memory allows",
    )
    p.add_argument(
        "--t_min",
        type=float,
        default=None,
        help="restrict FM timesteps (W2 σ-restriction lever; None = full range)",
    )
    p.add_argument("--t_max", type=float, default=None)
    p.add_argument(
        "--eval_groups",
        default="",
        help="eval: comma list of groups to render (single,combo,corpus,en); default all",
    )
    p.add_argument(
        "--eval_limit",
        type=int,
        default=0,
        help="eval: first N prompts per group (0 = all)",
    )
    p.add_argument(
        "--eval_tag",
        default="",
        help="eval: write img/reads/report/sheets under <arm>/eval_<tag>/ (e.g. a second eval_size)",
    )
    p.add_argument(
        "--no_floor",
        action="store_true",
        help="eval: skip the delta-scale-0 floor renders (identical across arms on the same eval set)",
    )
    p.add_argument(
        "--cls_t",
        default="0.1,0.2,0.35,0.5,0.65,0.8,0.95",
        help="classify: σ grid (DiT-scale) to score the candidates at",
    )
    p.add_argument(
        "--cls_per_kana",
        type=int,
        default=2,
        help="classify: held-out renders per kana",
    )
    p.add_argument(
        "--cls_batch",
        type=int,
        default=24,
        help="classify: candidate captions per DiT forward",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--native_prompts",
        default=str(NATIVE_PROMPTS),
        help="native: scene prompt file (one per line; default the blind-pairs set)",
    )
    p.add_argument(
        "--native_chars",
        default="あ,か,す",
        help="native: comma list of kana to hang off every scene prompt",
    )
    p.add_argument(
        "--native_clauses",
        default=",".join(NATIVE_CLAUSES),
        help="native: clause shapes to append (" + ", ".join(NATIVE_CLAUSES) + ")",
    )
    p.add_argument(
        "--native_limit", type=int, default=0, help="native: first N prompts (0 = all)"
    )
    p.add_argument(
        "--data_tag",
        default="",
        help="suffix for output/wake_probe/data_<tag> and <arm>_<tag>",
    )
    p.add_argument(
        "--arm_tag",
        default="",
        help="suffix for the arm dir only (<arm>_<data_tag>_<arm_tag>): a second "
        "train recipe on the same data without overwriting the first",
    )
    p.add_argument(
        "--balanced",
        type=int,
        default=0,
        help="data: groups of N distinct strings sharing one layout (font, canvas, "
        "bubble, glyph size/position); train then batches one group per step "
        "(W2a; needs --batch N). 0 = shuffled items",
    )
    p.add_argument(
        "--layout",
        default="v1",
        choices=["v1", "jitter"],
        help="data: v1 = pre-W2a renders (big centred dark glyph on a light canvas, "
        "bit-identical rebuilds); jitter = random position / size / ink colour / "
        "outline / dark backgrounds / bubble box (the 2026-09-14 data lever)",
    )
    p.add_argument(
        "--only_chars",
        default="",
        help="restrict the kana inventory (textual-inversion regime: few chars, many exposures)",
    )
    p.add_argument(
        "--words",
        type=int,
        default=0,
        help="data: add the N most frequent single-Qwen-piece words of the training "
        "corpus to the inventory (each is an existing pack row = one address); "
        "corpus lines are then kept only when every piece is a trained row",
    )
    p.add_argument("--word_min_len", type=int, default=2)
    p.add_argument(
        "--held_out_words",
        type=int,
        default=0,
        help="data: K of the words removed from every training item, eval group word_held",
    )
    p.add_argument(
        "--n_word_eval",
        type=int,
        default=16,
        help="data: trained words in eval group word",
    )
    p.add_argument(
        "--n_line_eval",
        type=int,
        default=16,
        help="data: held-out corpus lines of 2-3 pieces, every piece trained (eval group line)",
    )
    p.add_argument(
        "--line_max_len",
        type=int,
        default=8,
        help="data: corpus line length cap in word mode",
    )
    p.add_argument(
        "--init_free",
        default="",
        help="train: warm-start the free residual by ext id from another arm's trained.pt",
    )
    a = p.parse_args()
    stages = ["salad", "data", "train", "eval"] if "all" in a.stage else a.stage
    OUT.mkdir(parents=True, exist_ok=True)
    for s in stages:
        print(f"===== stage {s} ({a.arm})", flush=True)
        {
            "salad": stage_salad,
            "data": stage_data,
            "train": stage_train,
            "eval": stage_eval,
            "classify": stage_classify,
            "native": stage_native,
        }[s](a)


if __name__ == "__main__":
    main()
