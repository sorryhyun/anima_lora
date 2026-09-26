"""Stage ``cf_sense`` — Gate 0 of ``idea.md``: can the caption move the frozen
DiT's x0-estimate off a rendered glyph B toward a glyph A, and at which σ?

Two renders per pair on one layout (A and B differ under the glyphs only);
B's latent is noised once per σ; the DiT runs under caption A and caption B
on that same input. Read in the glyph box, with ``d = x0_A − x0_B``:

  pos(c) = ⟨x̂0(c) − x0_B, d⟩ / ‖d‖²     where x̂0(c) = x_σ − σ · v_θ(x_σ, c)

so pos = 0 is "the estimate sits on B", 1 is "on A"; ``move`` =
pos(c_A) − pos(c_B) is the caption's leverage — the sensitivity factor the CF
gradient needs (``idea.md`` § 2). ``--cf_lang en`` runs nonsense Latin words
with no ext id in any caption (the ceiling: text the model reads natively);
``ja`` runs the arm's trained rows, delta on (``trained``) and off (``floor``).
Pairs are ``id`` (two units of one length) and, for 2-unit strings, ``order``
(the permutation), so the read separates identity from order leverage.

Size, layout and font are controlled per run (``plan_band.md`` Stage A):
``--cf_glyph_px`` a list of glyph px cycled over the items (one pair at every
px with ``--cf_per_pair`` = its length), ``--cf_layout`` ``mixed`` (the Gate 0
draw: a bubble on 60 % of pairs, the bubble caption on all) / ``flat`` /
``bubble`` / ``grid`` (the pair in the centre cell of a 3 × 3 flat grid, EN
only), ``--cf_font`` a pinned font, ``--cf_text`` ``letter`` / ``word`` /
``string2``. Every item records ``px`` / ``font`` / ``bubble`` / ``glyphs`` /
``ink_a`` / ``ink_b`` so the report and ``probe/cf_rebin.py`` bin by them.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import torch
from common.hooks import ExtDelta
from common.models import (
    dit_forward,
    encode_captions,
    encode_images,
    gen_args,
    load_trained,
    load_vae,
)
from common.paths import arm_dir
from common.prompts import TPL_BUBBLE, TPL_EN, TPL_PLAIN, TPL_PLAIN_EN, grid_caption
from common.render.flat import (
    JITTER_BG_LIGHT,
    find_fonts,
    pick_font,
    render_string,
    sample_layout,
)
from common.render.ink import glyph_count, glyph_features, ink_pixels
from common.shapes import parse_shape, wh
from common.text import KANA

# --cf_layout → share of pairs drawn inside the flat renderer's ellipse
_BUBBLE_FRAC = {"mixed": 0.6, "flat": 0.0, "bubble": 1.0}
_LETTERS = "ABDEFGHKLMNPRSTUVWXYZ"  # no I / J / O / Q / C: a stroke, not a glyph
# the ink ruler of plan_kanji C.0 (`--cf_units kanji` terciles)
_INK_FONT = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"


def _en_word_pairs(rng: random.Random, n_pairs: int) -> list[str]:
    """Nonsense two-word Latin strings "ZORP KAV" whose word swap keeps the T5
    piece multiset (words are space-separated, so pieces never cross a word);
    every word must be ≥ 2 pieces so the contrast is piece order, not one id."""
    from library.anima.weights import load_t5_tokenizer

    t5 = load_t5_tokenizer(None)
    cons, vow = "BDFGKLMNPRSTVZ", "AEIOU"

    def word():
        n = rng.randint(2, 3)
        w = "".join(rng.choice(cons) + rng.choice(vow) for _ in range(n))
        if rng.random() < 0.6:
            w += rng.choice(cons)
        return w

    out, seen = [], set()
    while len(out) < n_pairs:
        a, b = word(), word()
        if a == b or (a, b) in seen:
            continue
        if len(t5.tokenize(a)) < 2 or len(t5.tokenize(b)) < 2:
            continue
        if sorted(t5.tokenize(f"{a} {b}")) != sorted(t5.tokenize(f"{b} {a}")):
            continue
        seen.add((a, b))
        out.append(f"{a} {b}")
    return out


def _glyph_ink(ch: str, px: int = 48, font: str = _INK_FONT) -> float:
    """Ink of one glyph drawn alone, in latent cells² (the plan_kanji C.0
    measure: bbox-tight, Noto Serif CJK Regular, 48 px)."""
    from PIL import Image, ImageDraw, ImageFont

    im = Image.new("L", (px * 3, px * 3), 255)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(font, px, index=0)
    d.text((px, px), ch, font=f, fill=0)
    return ink_pixels(im, d.textbbox((px, px), ch, font=f)) / 64.0


def _ja_strata(row_text: dict, spec: str) -> list[list[str]]:
    """``--cf_units`` → strata of trained single rows (plan_kanji Stage C).
    ``kana`` = one stratum, the trained kana (the Gate 0 draw, bit-identical);
    ``kanji`` = the trained single kanji in three ink terciles; ``chars:g0/g1``
    = explicit groups, every char a trained single row."""
    singles = {t for t in row_text.values() if len(t) == 1}
    if spec == "kana":
        return [sorted(t for t in singles if t in KANA)]
    if spec == "kanji":
        kanji = sorted(t for t in singles if "一" <= t <= "鿿")
        assert len(kanji) >= 6, "cf_sense --cf_units kanji needs ≥ 6 trained kanji rows"
        ranked = sorted(kanji, key=lambda c: (_glyph_ink(c), c))
        k = len(ranked) // 3
        return [ranked[:k], ranked[k : 2 * k], ranked[2 * k :]]
    assert spec.startswith("chars:"), (
        f"--cf_units {spec!r}: kana | kanji | chars:g0/g1/…"
    )
    groups = [list(g) for g in spec[len("chars:") :].split("/") if g]
    missing = [c for g in groups for c in g if c not in singles]
    assert not missing, f"--cf_units chars: not trained single rows: {''.join(missing)}"
    for k, g in enumerate(groups):
        ink = sorted(_glyph_ink(c) for c in g)
        print(
            f"cf_sense stratum s{k}: {''.join(g)} (ink/glyph cells² at 48 px "
            f"median {ink[len(ink) // 2]:.1f}, {ink[0]:.1f}–{ink[-1]:.1f})",
            flush=True,
        )
    return groups


def _ja_pairs(
    row_text: dict, rng: random.Random, n: int, piece: bool, units_spec: str = "kana"
) -> list[dict]:
    """``single``: ``id`` pairs of trained single rows, ``order`` pairs of two
    singles vs their reversal, drawn within one stratum of ``units_spec``
    (``kana`` = the trained kana, one stratum; ``kanji`` = ink terciles;
    ``chars:g0/g1`` = explicit) and stamped with it. ``piece``: the same over
    the table's kana-only multi-glyph rows (``id`` = two rows of one glyph
    count, ``order`` = two rows concatenated vs swapped — the caption
    tokenises the concatenation as it would in a sentence)."""
    if piece:
        units = sorted(
            t for t in row_text.values() if len(t) >= 2 and all(c in KANA for c in t)
        )
        assert len(units) >= 4, "cf_sense piece needs ≥ 4 kana-only multi-glyph rows"
        by_len: dict[int, list[str]] = {}
        for t in units:
            by_len.setdefault(len(t), []).append(t)
        lens = [k for k, v in by_len.items() if len(v) >= 2]
        strata = [units]
    else:
        strata = _ja_strata(row_text, units_spec)
        for s in strata:
            assert len(s) >= 2, "cf_sense ja needs ≥ 2 trained single rows per stratum"
    out = []
    for i in range(n):
        if piece:
            a, b = rng.sample(by_len[rng.choice(lens)], 2)
            out.append({"kind": "id", "a": a, "b": b})
        else:
            k = i % len(strata)
            a, b = rng.sample(strata[k], 2)
            out.append({"kind": "id", "a": a, "b": b, "stratum": k})
    for i in range(n):
        k = i % len(strata)
        a, b = rng.sample(strata[k], 2)
        out.append({"kind": "order", "a": a + b, "b": b + a, "stratum": k})
    return out


def _en_pairs(rng: random.Random, n: int, piece: bool) -> list[dict]:
    """``single``: ``id`` = two nonsense words of one length, ``order`` = the
    word swap. ``piece``: ``id`` = two two-word strings of matched lengths,
    ``order`` = a three-word string vs its first two words swapped."""
    words = _en_word_pairs(rng, 2 * n if piece else n)
    out = []
    if piece:
        for s, t in zip(words[::2], words[1::2], strict=True):
            a1, a2 = s.split(" ")
            b1, b2 = t.split(" ")
            b1, b2 = (b1 * 3)[: len(a1)], (b2 * 3)[: len(a2)]
            out.append({"kind": "id", "a": f"{a1} {a2}", "b": f"{b1} {b2}"})
            out.append(
                {"kind": "order", "a": f"{a1} {a2} {b1}", "b": f"{a2} {a1} {b1}"}
            )
        return out
    for s in words:
        a, b = s.split(" ")
        if len(a) != len(b):
            b = (b * 3)[: len(a)]
        out.append({"kind": "id", "a": a, "b": b})
        out.append({"kind": "order", "a": s, "b": f"{b} {a}"})
    return out


def _letter_pairs(rng: random.Random, n: int) -> list[dict]:
    """``--cf_text letter``: ``id`` pairs of two capital letters — the
    one-glyph end of the size curve (a kana row's EN twin). No ``order``
    pairs: a single glyph has no order."""
    out, seen = [], set()
    while len(out) < n:
        a, b = rng.sample(_LETTERS, 2)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append({"kind": "id", "a": a, "b": b})
    return out


def _filler(rng: random.Random, like: str) -> str:
    """A nonsense string of ``like``'s shape (a letter, or words of the same
    lengths) for the grid cells around the pair — drawn once per pair, so A
    and B differ in the centre cell only."""
    cons, vow = "BDFGKLMNPRSTVZ", "AEIOU"
    if len(like) == 1:
        return rng.choice(_LETTERS)
    words = []
    for w in like.split(" "):
        s = "".join(rng.choice(cons) + rng.choice(vow) for _ in range(-(-len(w) // 2)))
        words.append(s[: len(w)])
    return " ".join(words)


def _diff_boxes(im_a, im_b):
    """Pixel bbox where A and B differ, and the same box in latent cells
    (VAE 8×) padded by one cell."""
    from PIL import ImageChops

    x0, y0, x1, y1 = ImageChops.difference(im_a, im_b).getbbox() or (0, 0, *im_a.size)
    W, H = im_a.size
    cells = (
        max(0, x0 // 8 - 1),
        max(0, y0 // 8 - 1),
        min(W // 8, -(-x1 // 8) + 1),
        min(H // 8, -(-y1 // 8) + 1),
    )
    return (x0, y0, x1, y1), cells


def _item(p, fa, fb, im_a, im_b, cap_a, cap_b, **fields) -> dict:
    pbox, box = _diff_boxes(im_a, im_b)
    return {
        **p,
        "file_a": str(fa),
        "file_b": str(fb),
        "box": box,
        "cap_a": cap_a,
        "cap_b": cap_b,
        "glyphs": glyph_count(p["a"]),
        "ink_a": ink_pixels(im_a, pbox),
        "ink_b": ink_pixels(im_b, pbox),
        **fields,
    }


def _render_pair(
    p: dict,
    fonts,
    rng: random.Random,
    size,
    out: Path,
    i: int,
    en: bool,
    layout: str = "mixed",
    px: int | None = None,
    font: str | None = None,
):
    """A and B on one layout / font; the box is where the two renders differ,
    in latent cells (VAE 8×) padded by one cell.

    The layout and font draws are consumed whatever ``layout`` / ``px`` /
    ``font`` say, so the pair sequence of a run is the Gate 0 sequence for
    the same seed and ``probe/cf_rebin.py`` can replay it."""
    if layout == "grid":
        return _render_pair_grid(p, fonts, rng, size, out, i, en, px, font)
    lay = sample_layout(len(p["a"]), rng, size, bubble_frac=_BUBBLE_FRAC[layout])
    if en:
        lay["vertical"] = False
    if px is not None:
        lay["fs"] = int(px)
    drawn = pick_font(p["a"] + p["b"], fonts, rng)
    font_path = font or drawn
    im_a, bubble = render_string(
        p["a"], font_path, rng, size=size, layout=lay, fit_text=p["b"]
    )
    im_b, _ = render_string(
        p["b"], font_path, rng, size=size, layout=lay, fit_text=p["a"]
    )
    fa, fb = out / "img" / f"{i:03d}_a.png", out / "img" / f"{i:03d}_b.png"
    im_a.save(fa)
    im_b.save(fb)
    if en:
        # ``mixed`` keeps the Gate 0 caption (the bubble template on every
        # pair); the controlled layouts caption what was drawn
        tpl = TPL_EN if (layout == "mixed" or bubble) else TPL_PLAIN_EN
    else:
        tpl = TPL_BUBBLE if bubble else TPL_PLAIN
    # shape descriptors of the pair's glyphs drawn alone (plan_kanji: the
    # straightness read); `straight` = the pair mean, binned by the report
    ga = glyph_features(p["a"], int(lay["fs"]), font_path)
    gb = glyph_features(p["b"], int(lay["fs"]), font_path)
    return _item(
        p,
        fa,
        fb,
        im_a,
        im_b,
        tpl.format(p["a"]),
        tpl.format(p["b"]),
        layout=layout,
        bubble=bool(bubble),
        px=int(lay["fs"]),
        font=Path(font_path).stem,
        straight_a=ga["straight"],
        straight_b=gb["straight"],
        axis_a=ga["axis"],
        axis_b=gb["axis"],
        fill_a=ga["fill"],
        fill_b=gb["fill"],
        straight=(ga["straight"] + gb["straight"]) / 2,
    )


def _render_pair_grid(p, fonts, rng, size, out, i, en, px, font):
    """The pair in the centre cell of a 3 × 3 flat grid (the data stage's
    ``--grid 3x3`` frame and caption: one position clause per cell), the
    eight other cells holding fillers of the pair's shape, identical in A
    and B. ``px`` is the asked glyph px; a cell that cannot hold it shrinks
    (the ``render_grid`` rule) and the item records the px drawn."""
    from PIL import Image, ImageDraw, ImageFont

    assert en, "cf_sense --cf_layout grid is the EN ceiling probe"
    W, H = wh(size)
    cols = rows = 3
    cw, ch = W / cols, H / rows
    font_path = font or pick_font(p["a"] + p["b"], fonts, rng)
    fill = [_filler(rng, p["a"]) for _ in range(cols * rows - 1)]
    bg = rng.choice(JITTER_BG_LIGHT)
    # the data stage's --grid_fill 0.5–0.8 band when no px is asked
    fs0 = int(px) if px is not None else int(rng.uniform(0.5, 0.8) * min(cw, ch))
    room = 0.9 * min(cw, ch)
    d0 = ImageDraw.Draw(Image.new("RGB", (W, H), bg))

    def fit(texts, fs):
        while True:
            f = ImageFont.truetype(font_path, fs, index=0)
            bxs = [d0.textbbox((0, 0), t, font=f) for t in texts]
            tw = max(b[2] - b[0] for b in bxs)
            th = max(b[3] - b[1] for b in bxs)
            if (tw <= room and th <= room) or fs <= 12:
                return fs
            fs = max(12, int(fs * min(room / tw, room / th) * 0.98))

    fs_cells = [fit([t], fs0) for t in fill]
    fs_mid = fit([p["a"], p["b"]], fs0)  # the max extent, as fit_text does

    def draw(mid):
        im = Image.new("RGB", (W, H), bg)
        d = ImageDraw.Draw(im)
        units = fill[:4] + [mid] + fill[4:]
        for k, u in enumerate(units):
            r, c = divmod(k, cols)
            fs = fs_mid if k == 4 else fs_cells[k if k < 4 else k - 1]
            f = ImageFont.truetype(font_path, fs, index=0)
            bx = d.textbbox((0, 0), u, font=f)
            tw, th = bx[2] - bx[0], bx[3] - bx[1]
            d.text(
                (c * cw + cw / 2 - tw / 2 - bx[0], r * ch + ch / 2 - th / 2 - bx[1]),
                u,
                fill="black",
                font=f,
            )
        return im, units

    im_a, units_a = draw(p["a"])
    im_b, units_b = draw(p["b"])
    fa, fb = out / "img" / f"{i:03d}_a.png", out / "img" / f"{i:03d}_b.png"
    im_a.save(fa)
    im_b.save(fb)
    return _item(
        p,
        fa,
        fb,
        im_a,
        im_b,
        grid_caption("flat", cols, rows, units_a, lang="english"),
        grid_caption("flat", cols, rows, units_b, lang="english"),
        layout="grid",
        bubble=False,
        px=int(fs_mid),
        font=Path(font_path).stem,
    )


def _box_slice(box):
    x0, y0, x1, y1 = box
    return slice(y0, y1), slice(x0, x1)


def stage_cf_sense(a):
    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model

    en = a.cf_lang == "en"
    piece = a.cf_rows == "piece"
    layout = a.cf_layout
    text = a.cf_text if a.cf_text != "auto" else ("string2" if piece else "word")
    assert en or a.cf_text == "auto", (
        "--cf_text shapes the EN pairs; ja pairs are the rows"
    )
    assert layout != "grid" or en, "cf_sense --cf_layout grid is the EN ceiling probe"
    px_list = [int(x) for x in a.cf_glyph_px.split(",") if x.strip()]
    # plan_canvas D.0 / D.1: a WxH canvas; --train_size² otherwise
    size = parse_shape(a.eval_shape) if a.eval_shape else a.train_size
    font = a.cf_font or None
    name = (
        f"cf_sense_{a.cf_lang}"
        + ("_piece" if piece else "")
        + (f"_{layout}" if layout != "mixed" else "")
        + (f"_{text}" if a.cf_text != "auto" else "")
        + (f"_{Path(font).stem}" if font else "")
        + ("" if en or piece or a.cf_units == "kana" else f"_{a.cf_units[:5]}")
        + (f"_{a.eval_tag}" if a.eval_tag else "")
    )
    out = arm_dir(a) / name
    (out / "img").mkdir(parents=True, exist_ok=True)
    rng = random.Random(30_000 + a.seed)
    if en:
        if text == "letter":
            pairs = _letter_pairs(rng, a.cf_pairs)
        else:
            pairs = _en_pairs(rng, a.cf_pairs, text == "string2")
        conds = ("base",)
    else:
        sd = load_trained(arm_dir(a))
        assert "lora" not in sd, "cf_sense covers rows-only arms"
        if "row_text" in sd:
            row_text = {int(r): t for r, t in sd["row_text"].items()}
        else:  # newer arms save ext ids only; decode them through the pack
            from probe.merge_tables import row_text_map

            row_text = row_text_map([int(e) for e in sd["delta"]["ext_ids"]])
        pairs = _ja_pairs(row_text, rng, a.cf_pairs, piece, a.cf_units)
        conds = ("trained", "floor")
    fonts = find_fonts()
    items = []
    for i, p in enumerate(pairs):
        for j in range(a.cf_per_pair):
            # px cycles over the items, so --cf_per_pair = len(px_list) gives
            # every pair every px
            px = px_list[len(items) % len(px_list)] if px_list else None
            it = _render_pair(
                p, fonts, rng, size, out, len(items), en, layout, px, font
            )
            if "stratum" in p:  # plan_kanji: the stratum × px cell
                it["cell"] = f"s{p['stratum']}@{it['px']}"
            items.append(it)
    print(
        f"cf_sense {a.cf_lang} {a.cf_rows} {layout} {text}: {len(items)} pairs "
        f"({sum(it['kind'] == 'id' for it in items)} id, "
        f"{sum(it['kind'] == 'order' for it in items)} order); "
        f"px {sorted({it['px'] for it in items})}, fonts {sorted({it['font'] for it in items})}",
        flush=True,
    )

    args = gen_args(size, a.steps, a.cfg, out)
    device = get_generation_settings(args).device
    captions = sorted({c for it in items for c in (it["cap_a"], it["cap_b"])})
    cache = encode_captions(captions, device)
    vae = load_vae(device)
    lat = encode_images(
        vae, [f for it in items for f in (it["file_a"], it["file_b"])], device
    )
    del vae
    torch.cuda.empty_cache()
    sigmas = [float(x) for x in a.cf_t.split(",")]
    g = torch.Generator().manual_seed(a.seed)
    noise = torch.randn((len(items), len(sigmas), *lat.shape[1:]), generator=g)
    anima = load_dit_model(args, device, torch.bfloat16)
    anima.requires_grad_(False)
    anima.eval()
    delta = None if en else ExtDelta.from_state(anima, sd["delta"], device)

    # pos[cond, item, σ, (cap_A, cap_B)], leak[cond, item, σ] = out-of-box /
    # in-box per-cell energy of x̂0(c_A) − x̂0(c_B)
    pos = torch.zeros(len(conds), len(items), len(sigmas), 2)
    leak = torch.zeros(len(conds), len(items), len(sigmas))
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ci, cond in enumerate(conds):
            if delta is not None:
                delta.scale = 1.0 if cond == "trained" else 0.0
            for ii, it in enumerate(items):
                x_a = lat[2 * ii : 2 * ii + 1].to(device)
                x_b = lat[2 * ii + 1 : 2 * ii + 2].to(device)
                ys, xs = _box_slice(it["box"])
                d = (x_a - x_b)[..., ys, xs]
                dd = (d * d).sum().clamp(min=1e-8)
                for si, sig in enumerate(sigmas):
                    nz = noise[ii, si : si + 1].to(device)
                    noisy = ((1.0 - sig) * x_b + sig * nz).to(torch.bfloat16)
                    caps = [it["cap_a"], it["cap_b"]]
                    v = dit_forward(
                        anima,
                        noisy.repeat(2, 1, 1, 1),
                        torch.full((2,), sig, device=device),
                        cache,
                        caps,
                        device,
                    ).float()
                    x0_hat = noisy.float() - sig * v  # (2, C, H, W)
                    rel = (x0_hat - x_b)[..., ys, xs]
                    pos[ci, ii, si] = ((rel * d).sum(dim=(1, 2, 3)) / dd).cpu()
                    diff2 = ((x0_hat[0] - x0_hat[1]) ** 2).mean(dim=0)  # (H, W)
                    m = torch.zeros_like(diff2)
                    m[ys, xs] = 1.0
                    e_in = (diff2 * m).sum() / m.sum().clamp(min=1)
                    e_out = (diff2 * (1 - m)).sum() / (1 - m).sum().clamp(min=1)
                    leak[ci, ii, si] = float(e_out / e_in.clamp(min=1e-8))
                if ii % 8 == 7:
                    n_fwd = (ci * len(items) + ii + 1) * len(sigmas) * 2
                    print(
                        f"cf_sense {cond}: pair {ii + 1}/{len(items)}, "
                        f"{n_fwd / (time.time() - t0):.1f} fwd/s",
                        flush=True,
                    )
    del anima
    torch.cuda.empty_cache()
    meta = {
        "lang": a.cf_lang,
        "rows": a.cf_rows,
        "layout": layout,
        "text": text,
        "font": font,
        "glyph_px": px_list,
        "seed": a.seed,
        "cf_pairs": a.cf_pairs,
        "cf_per_pair": a.cf_per_pair,
        "size": list(wh(size)),
    }
    torch.save(
        {
            "pos": pos,
            "leak": leak,
            "sigmas": sigmas,
            "conds": conds,
            "items": items,
            "meta": meta,
        },
        out / "cf_sense.pt",
    )
    _report(out, pos, leak, sigmas, conds, items, a.cf_lang)


def _report(out: Path, pos, leak, sigmas, conds, items, lang):
    kinds = sorted({it["kind"] for it in items})
    L = [
        f"# cf_sense — caption leverage on a B-rendered input ({lang})",
        "",
        f"{len(items)} pairs; B's latent noised once per σ, the DiT run under caption A "
        "and caption B on the same input. pos(c) = ⟨x̂0(c) − x0_B, x0_A − x0_B⟩ / ‖x0_A − x0_B‖² "
        "in the glyph box (0 = on B, 1 = on A); move = pos(c_A) − pos(c_B) is the caption's "
        "leverage; leak = out-of-box / in-box per-cell energy of x̂0(c_A) − x̂0(c_B). "
        "Kill (idea.md Gate 0): EN move < ≈ 0.1 below σ 0.7.",
        "",
    ]
    for kind in kinds:
        sel = torch.tensor([it["kind"] == kind for it in items])
        L += [
            f"## {kind} ({int(sel.sum())} pairs)",
            "",
            "| cond | σ | move mean ± sd | move median | move > 0.25 | pos(c_A) | pos(c_B) | leak out/in |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for ci, cond in enumerate(conds):
            for si, sig in enumerate(sigmas):
                p = pos[ci, sel, si]
                mv = p[:, 0] - p[:, 1]
                L.append(
                    f"| {cond} | {sig:.2f} | {mv.mean():+.3f} ± {mv.std():.3f} | "
                    f"{mv.median():+.3f} | {(mv > 0.25).float().mean():.2f} | "
                    f"{p[:, 0].mean():+.3f} | {p[:, 1].mean():+.3f} | "
                    f"{leak[ci, sel, si].mean():.3f} |"
                )
        L.append("")
    # straightness terciles over the run's items (plan_kanji: complexity
    # that is not ink) — `sbin` 0 = curviest third, 2 = straightest
    st = sorted(it["straight"] for it in items if it.get("straight") is not None)
    if len(st) >= 6:
        q1, q2 = st[len(st) // 3], st[2 * len(st) // 3]
        for it in items:
            if it.get("straight") is not None:
                it["sbin"] = (
                    0 if it["straight"] < q1 else (1 if it["straight"] < q2 else 2)
                )
    for key, title in (
        ("px", "glyph px"),
        ("font", "font"),
        ("stratum", "stratum"),
        ("cell", "stratum @ px"),
        ("sbin", "straightness tercile"),
    ):
        vals = sorted({it.get(key) for it in items if it.get(key) is not None})
        if len(vals) < 2:
            continue
        L += [
            f"## By {title} (first cond, mean move; peak = argmax σ)",
            "",
            f"| kind | {title} | n | ink/glyph cells² | "
            + " | ".join(f"{s:.2f}" for s in sigmas)
            + " | peak |",
            "|---|---|---|---|" + "---|" * len(sigmas) + "---|",
        ]
        for kind in kinds:
            for v in vals:
                sel = torch.tensor(
                    [it["kind"] == kind and it.get(key) == v for it in items]
                )
                if not sel.any():
                    continue
                mv = (pos[0, sel, :, 0] - pos[0, sel, :, 1]).mean(dim=0)
                ink = torch.tensor(
                    [it["ink_a"] / max(it["glyphs"], 1) / 64 for it in items]
                )[sel]
                L.append(
                    f"| {kind} | {v} | {int(sel.sum())} | {ink.median():.1f} | "
                    + " | ".join(f"{float(x):+.3f}" for x in mv)
                    + f" | {sigmas[int(mv.argmax())]:.2f} |"
                )
        L.append("")
    L += [
        "## Per pair, move by σ (first cond)",
        "",
        "| kind | A | B | " + " | ".join(f"{s:.2f}" for s in sigmas) + " |",
        "|---|---|---|" + "---|" * len(sigmas),
    ]
    for ii, it in enumerate(items):
        mv = pos[0, ii, :, 0] - pos[0, ii, :, 1]
        L.append(
            f"| {it['kind']} | {it['a']} | {it['b']} | "
            + " | ".join(f"{float(x):+.2f}" for x in mv)
            + " |"
        )
    (out / "cf_sense.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[: 8 + 2 * (len(conds) * len(sigmas) + 5)]), flush=True)
