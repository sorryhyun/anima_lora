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
from common.prompts import TPL_BUBBLE, TPL_EN, TPL_PLAIN
from common.render.flat import find_fonts, pick_font, render_string, sample_layout
from common.text import KANA
from eval.classify import _en_word_pairs


def _ja_pairs(row_text: dict, rng: random.Random, n: int, piece: bool) -> list[dict]:
    """``single``: ``id`` pairs of trained single kana, ``order`` pairs of two
    singles vs their reversal. ``piece``: the same over the table's kana-only
    multi-glyph rows (``id`` = two rows of one glyph count, ``order`` = two
    rows concatenated vs swapped — the caption tokenises the concatenation
    as it would in a sentence)."""
    if piece:
        units = sorted(
            t for t in row_text.values() if len(t) >= 2 and all(c in KANA for c in t)
        )
        assert len(units) >= 4, "cf_sense piece needs ≥ 4 kana-only multi-glyph rows"
        by_len: dict[int, list[str]] = {}
        for t in units:
            by_len.setdefault(len(t), []).append(t)
        lens = [k for k, v in by_len.items() if len(v) >= 2]
    else:
        units = sorted(t for t in row_text.values() if len(t) == 1 and t in KANA)
        assert len(units) >= 2, "cf_sense ja needs ≥ 2 trained single kana rows"
    out = []
    for _ in range(n):
        if piece:
            a, b = rng.sample(by_len[rng.choice(lens)], 2)
        else:
            a, b = rng.sample(units, 2)
        out.append({"kind": "id", "a": a, "b": b})
    for _ in range(n):
        a, b = rng.sample(units, 2)
        out.append({"kind": "order", "a": a + b, "b": b + a})
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
            out.append({"kind": "order", "a": f"{a1} {a2} {b1}", "b": f"{a2} {a1} {b1}"})
        return out
    for s in words:
        a, b = s.split(" ")
        if len(a) != len(b):
            b = (b * 3)[: len(a)]
        out.append({"kind": "id", "a": a, "b": b})
        out.append({"kind": "order", "a": s, "b": f"{b} {a}"})
    return out


def _render_pair(p: dict, fonts, rng: random.Random, size, out: Path, i: int, en: bool):
    """A and B on one layout / font; the box is where the two renders differ,
    in latent cells (VAE 8×) padded by one cell."""
    from PIL import ImageChops

    lay = sample_layout(len(p["a"]), rng, size)
    if en:
        lay["vertical"] = False
    font = pick_font(p["a"] + p["b"], fonts, rng)
    im_a, bubble = render_string(
        p["a"], font, rng, size=size, layout=lay, fit_text=p["b"]
    )
    im_b, _ = render_string(p["b"], font, rng, size=size, layout=lay, fit_text=p["a"])
    fa, fb = out / "img" / f"{i:03d}_a.png", out / "img" / f"{i:03d}_b.png"
    im_a.save(fa)
    im_b.save(fb)
    x0, y0, x1, y1 = ImageChops.difference(im_a, im_b).getbbox() or (0, 0, *im_a.size)
    W, H = im_a.size
    box = (
        max(0, x0 // 8 - 1),
        max(0, y0 // 8 - 1),
        min(W // 8, -(-x1 // 8) + 1),
        min(H // 8, -(-y1 // 8) + 1),
    )
    tpl = TPL_EN if en else (TPL_BUBBLE if bubble else TPL_PLAIN)
    return {
        **p,
        "file_a": str(fa),
        "file_b": str(fb),
        "box": box,
        "cap_a": tpl.format(p["a"]),
        "cap_b": tpl.format(p["b"]),
    }


def _box_slice(box):
    x0, y0, x1, y1 = box
    return slice(y0, y1), slice(x0, x1)


def stage_cf_sense(a):
    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model

    en = a.cf_lang == "en"
    piece = a.cf_rows == "piece"
    name = (
        f"cf_sense_{a.cf_lang}"
        + ("_piece" if piece else "")
        + (f"_{a.eval_tag}" if a.eval_tag else "")
    )
    out = arm_dir(a) / name
    (out / "img").mkdir(parents=True, exist_ok=True)
    rng = random.Random(30_000 + a.seed)
    if en:
        pairs = _en_pairs(rng, a.cf_pairs, piece)
        conds = ("base",)
    else:
        sd = load_trained(arm_dir(a))
        assert "lora" not in sd, "cf_sense covers rows-only arms"
        if "row_text" in sd:
            row_text = {int(r): t for r, t in sd["row_text"].items()}
        else:  # newer arms save ext ids only; decode them through the pack
            from probe.merge_tables import row_text_map

            row_text = row_text_map([int(e) for e in sd["delta"]["ext_ids"]])
        pairs = _ja_pairs(row_text, rng, a.cf_pairs, piece)
        conds = ("trained", "floor")
    fonts = find_fonts()
    items = []
    for i, p in enumerate(pairs):
        for j in range(a.cf_per_pair):
            items.append(_render_pair(p, fonts, rng, a.train_size, out, len(items), en))
    print(
        f"cf_sense {a.cf_lang} {a.cf_rows}: {len(items)} pairs "
        f"({sum(it['kind'] == 'id' for it in items)} id, "
        f"{sum(it['kind'] == 'order' for it in items)} order)",
        flush=True,
    )

    args = gen_args(a.train_size, a.steps, a.cfg, out)
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
    torch.save(
        {"pos": pos, "leak": leak, "sigmas": sigmas, "conds": conds, "items": items},
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
