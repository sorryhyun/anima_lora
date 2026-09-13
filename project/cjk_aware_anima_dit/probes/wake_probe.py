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
          training captions touch (arm ``rows``) or that + a LoRA on every
          Linear of ``llm_adapter.blocks`` (arm ``rows_adapter``). Plain
          rectified-flow loss on the glyph crops.
  classify  same-noise N-way diffusion classifier over the trained single kana
          (delta on vs off): which σ carries identity, do the rows discriminate.
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


def _sample_layout(n: int, rng: random.Random, size=512) -> dict:
    """Every random choice of one render for an ``n``-char string, drawn in the
    pre-W2a order so unbalanced data dirs rebuild bit-identically."""
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
    return lay


def _render_string(
    text: str, font_path: str, rng: random.Random, size=512, layout=None
):
    """``layout`` (from ``_sample_layout``) pins canvas/bubble/size/position so
    several strings render in the same layout; ``None`` draws a fresh one."""
    from PIL import Image, ImageDraw, ImageFont

    n = len(text)
    lay = layout if layout is not None else _sample_layout(n, rng, size)
    bubble, bg, fs, color = lay["bubble"], lay["bg"], lay["fs"], lay["color"]
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)
    if bubble:
        for x, y in lay["dots"]:
            d.ellipse((x, y, x + 2, y + 2), fill=(150, 150, 150))
        pad = lay["pad"]
        d.ellipse(
            (pad, pad, size - pad, size - pad),
            fill="white",
            outline="black",
            width=lay["outline"],
        )
    font = ImageFont.truetype(font_path, fs, index=0)
    if lay["vertical"]:
        total = n * fs * 1.05
        y = (size - total) / 2
        for ch in text:
            w = d.textlength(ch, font=font)
            d.text(((size - w) / 2, y), ch, fill=color, font=font)
            y += fs * 1.05
    else:
        w = d.textlength(text, font=font)
        d.text(
            ((size - w) / 2, (size - fs) / 2 - fs * 0.1), text, fill=color, font=font
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
            lay = _sample_layout(len(grp[0]), rng)
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
        # font renders: every kana ×N + random combos
        for ch in kana:
            for _ in range(n_single):
                items.append(("font", ch))
        n_combo = 0
        while n_combo < n_target:
            k = rng.choice([2, 3])
            s = "".join(rng.choice(kana) for _ in range(k))
            if s in combos_eval:
                continue
            items.append(("font", s))
            n_combo += 1
        for i, (kind, s) in enumerate(items):
            im, bubble = _render_string(s, rng.choice(fonts), rng)
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
    lines = _corpus_lines(CORPUS_TRAIN / "boxes.jsonl", 6)
    if a.only_chars:
        lines = [ln for ln in lines if all(c in kana for c in ln[0] if c in KANA)]
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
    recs = [json.loads(ln) for ln in (data / "train.jsonl").read_text().splitlines()]
    ev = json.loads((data / "eval.json").read_text())
    args = _gen_args(a.train_size, a.steps, a.cfg, arm_dir)
    device = get_generation_settings(args).device

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
            for i in range(0, len(recs), 8):
                px = np.stack(
                    [
                        np.array(
                            Image.open(r["file"])
                            .convert("RGB")
                            .resize((a.train_size, a.train_size))
                        )
                        for r in recs[i : i + 8]
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
        loss = F.mse_loss(pred.float(), target.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == 1:
            dn = (delta.raw.detach() * row_scale).norm(dim=1)
            rec = {
                "step": step,
                "loss": float(loss),
                "delta_norm_mean": float(dn.mean()),
                "delta_norm_max": float(dn.max()),
                "rel": float(dn.mean() / row_scale),
                "it_s": step / (time.time() - t0),
            }
            if lora is not None:
                rec["lora_b_norm"] = float(
                    sum(p.norm() ** 2 for p in list(lora.params)[1::2]) ** 0.5
                )
            log.append(rec)
            print(json.dumps(rec), flush=True)
    sd = {"delta": delta.state_dict(), "arm": a.arm, "args": vars(a)}
    if lora is not None:
        sd["lora"] = lora.state_dict()
        sd["adapter_rank"] = a.adapter_rank
    torch.save(sd, arm_dir / "trained.pt")
    (arm_dir / "train_log.json").write_text(json.dumps(log, indent=1))
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
    ev = json.loads((data / "eval.json").read_text())
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
    for g in ("single", "combo", "corpus", "en"):
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
        for g in ("single", "combo", "corpus"):
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
    for g in ("single", "combo", "corpus", "en"):
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


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--stage",
        nargs="+",
        default=["all"],
        choices=["all", "salad", "data", "train", "eval", "classify"],
    )
    p.add_argument("--arm", default="rows", choices=["rows", "rows_adapter"])
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
        "--only_chars",
        default="",
        help="restrict the kana inventory (textual-inversion regime: few chars, many exposures)",
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
        }[s](a)


if __name__ == "__main__":
    main()
