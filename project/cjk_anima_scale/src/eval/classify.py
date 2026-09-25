"""Stages ``classify`` / ``classify_str`` — same-noise diffusion classifiers.

A held-out render is noised once per σ and scored under competing captions
(identical inputs, only the address differs); summed FM error per candidate,
right = argmin. Run with the arm's delta and with it off (pack rows) as the
control.
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
from common.paths import arm_dir, data_dir
from common.prompts import TPL_BUBBLE, TPL_EN, TPL_PLAIN
from common.render.flat import find_fonts, render_string, sample_layout
from common.text import KANA
from data.inventory import clean_kana_strings, qwen_pieces

TEMPLATES = {"bubble": TPL_BUBBLE, "plain": TPL_PLAIN}
CONDS = ("trained", "floor")


def _setup(a, out: Path, items, captions):
    """Text cache, latents of the rendered items, one noise draw per (item, σ)
    shared by both conds, and the frozen DiT with the arm's delta hooked."""
    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model

    sd = load_trained(arm_dir(a))
    assert "lora" not in sd, "classify covers rows-only arms"
    args = gen_args(a.train_size, a.steps, a.cfg, out)
    device = get_generation_settings(args).device
    cache = encode_captions(captions, device)
    vae = load_vae(device)
    lat = encode_images(vae, [it["file"] for it in items], device)
    del vae
    torch.cuda.empty_cache()
    sigmas = [float(x) for x in a.cls_t.split(",")]
    g = torch.Generator().manual_seed(a.seed)
    noise = torch.randn((len(items), len(sigmas), *lat.shape[1:]), generator=g)
    anima = load_dit_model(args, device, torch.bfloat16)
    anima.requires_grad_(False)
    anima.eval()
    delta = ExtDelta.from_state(anima, sd["delta"], device)
    return device, cache, lat, sigmas, noise, anima, delta


def _errors(anima, x, nz, sig, captions, cache, device):
    """Summed FM error of one noised latent under each caption (one forward)."""
    target = nz - x
    noisy = ((1.0 - sig) * x + sig * nz).to(torch.bfloat16)
    b = len(captions)
    pred = dit_forward(
        anima,
        noisy.repeat(b, 1, 1, 1),
        torch.full((b,), sig, device=device),
        cache,
        captions,
        device,
    )
    return ((pred.float() - target) ** 2).sum(dim=(1, 2, 3)).cpu()


def _progress(stage, cond, ii, n, n_fwd, t0):
    if ii % 8 == 7:
        print(
            f"{stage} {cond}: item {ii + 1}/{n}, {n_fwd / (time.time() - t0):.1f} fwd/s",
            flush=True,
        )


# ----------------------------------------------------------------------------
# classify: N-way over single kana


def stage_classify(a):
    """Same-noise N-way diffusion classifier over the single kana.

    Fresh held-out single renders (unseen fonts/layouts); each is noised once
    per σ and scored under every kana's caption in its own template.
    Answers: which σ band carries identity (where a CE term belongs), the error
    gap/spread that sets its temperature, and whether the rows already
    discriminate (then renders fail in sampling, not in the rows).
    """
    import json

    recs = [
        json.loads(ln) for ln in (data_dir(a) / "train.jsonl").read_text().splitlines()
    ]
    singles = {r["text"] for r in recs if r["src"] == "font" and len(r["text"]) == 1}
    kana = [c for c in KANA if c in singles]
    out = arm_dir(a) / (f"classify_{a.eval_tag}" if a.eval_tag else "classify")
    (out / "img").mkdir(parents=True, exist_ok=True)

    # held-out renders: own rng stream, so fonts/layouts are not the train set's
    rng = random.Random(10_000 + a.seed)
    fonts = find_fonts()
    items = []
    for ki, ch in enumerate(kana):
        for j in range(a.cls_per_kana):
            im, bubble = render_string(ch, rng.choice(fonts), rng, size=a.train_size)
            fn = out / "img" / f"{ki:02d}_{j}.png"
            im.save(fn)
            items.append(
                {"file": str(fn), "text": ch, "tpl": "bubble" if bubble else "plain"}
            )
    captions = [TEMPLATES[t].format(c) for t in TEMPLATES for c in kana]
    device, cache, lat, sigmas, noise, anima, delta = _setup(a, out, items, captions)

    # err[cond, item, σ, kana] = FM error summed over the latent
    K = len(kana)
    err = torch.zeros(len(CONDS), len(items), len(sigmas), K)
    t0 = time.time()
    n_fwd = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ci, cond in enumerate(CONDS):
            delta.scale = 1.0 if cond == "trained" else 0.0
            for ii, it in enumerate(items):
                caps = [TEMPLATES[it["tpl"]].format(c) for c in kana]
                x = lat[ii : ii + 1].to(device)
                for si, sig in enumerate(sigmas):
                    nz = noise[ii, si : si + 1].to(device)
                    for k0 in range(0, K, a.cls_batch):
                        cs = caps[k0 : k0 + a.cls_batch]
                        err[ci, ii, si, k0 : k0 + len(cs)] = _errors(
                            anima, x, nz, sig, cs, cache, device
                        )
                        n_fwd += len(cs)
                _progress("classify", cond, ii, len(items), n_fwd, t0)
    del anima
    torch.cuda.empty_cache()
    labels = torch.tensor([kana.index(it["text"]) for it in items])
    torch.save(
        {
            "err": err,
            "labels": labels,
            "kana": kana,
            "sigmas": sigmas,
            "conds": CONDS,
            "items": items,
        },
        out / "classify.pt",
    )
    _report_classify(out, err, labels, kana, sigmas, CONDS)


def _report_classify(out: Path, err, labels, kana, sigmas, conds):
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
# classify_str: where in σ are order / count / identity decided for strings?


def _string_pairs(row_text: dict, rng: random.Random, n_pairs: int, n_triples: int):
    """Kana strings whose every permutation tokenizes to the same single-kana
    rows (so "なに" vs "にな" is a pure order contrast, never a word row).
    Returns (pairs, triples, kana) — strings, not captions."""
    tok, q = qwen_pieces()
    text_row = {t: r for r, t in row_text.items()}
    kana = sorted(t for t in text_row if len(t) == 1 and t in KANA)
    pairs = clean_kana_strings(tok, q, kana, rng, n_pairs, 2, rows=text_row)
    triples = clean_kana_strings(
        tok, q, kana, rng, n_triples, 3, excl=pairs, rows=text_row, skip_reversed=False
    )
    return pairs, triples, kana


def _string_candidates(s: str, kana: list, rng: random.Random) -> dict:
    """Named candidate strings for one rendered string ``s``."""
    others = [k for k in kana if k not in s]
    c = rng.choice(others)
    cands = {"S": s, "rev": s[::-1]}
    if len(s) == 3:
        cands["rot"] = s[1:] + s[0]
        cands["first2"] = s[:2]
    cands["first"] = s[0]
    cands["last"] = s[-1]
    cands["sub0"] = c + s[1:]
    cands["sub1"] = s[0] + c + s[2:]
    return cands


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


def _en_candidates(s: str, rng: random.Random, pool: list[str]) -> dict:
    a, b = s.split(" ")
    c = rng.choice([w for p in pool for w in p.split(" ") if w not in (a, b)])
    return {
        "S": s,
        "rev": f"{b} {a}",
        "first": a,
        "last": b,
        "sub0": f"{c} {b}",
        "sub1": f"{a} {c}",
    }


def stage_classify_str(a):
    """Same-noise diffusion classifier over *strings* of trained kana rows.

    Each held-out 2- or 3-kana render is noised once per σ and scored under
    its own caption (S) and named alternatives: reversed / rotated (order),
    single pieces and prefixes (count), one piece substituted (identity).
    Reports, per σ and cond, how often S beats each alternative and the
    gap/spread — i.e. at which σ the DiT reads order, how many units, and
    which units, for the arm's rows. Floor (delta off) is the control.
    """
    en = a.cls_lang == "en"
    name = "classify_str_en" if en else "classify_str"
    out = arm_dir(a) / (f"{name}_{a.eval_tag}" if a.eval_tag else name)
    (out / "img").mkdir(parents=True, exist_ok=True)
    tpls = {"bubble": TPL_EN, "plain": TPL_EN} if en else TEMPLATES

    rng = random.Random(20_000 + a.seed)
    if en:
        # base-model order control: no ext id in any caption, delta inert
        pairs = _en_word_pairs(rng, a.cls_pairs)
        triples, kana = [], []
    else:
        row_text = {int(r): t for r, t in load_trained(arm_dir(a))["row_text"].items()}
        pairs, triples, kana = _string_pairs(row_text, rng, a.cls_pairs, a.cls_triples)
    print(
        f"classify_str: {len(pairs)} pairs {pairs}\n{len(triples)} triples {triples}",
        flush=True,
    )
    fonts = find_fonts()
    items = []
    for si, s in enumerate(pairs + triples):
        cands = (
            _en_candidates(s, rng, pairs) if en else _string_candidates(s, kana, rng)
        )
        for j in range(a.cls_per_kana):
            lay = sample_layout(len(s), rng, a.train_size)
            if en:
                lay["vertical"] = False
            im, bubble = render_string(
                s, rng.choice(fonts), rng, size=a.train_size, layout=lay
            )
            fn = out / "img" / f"{si:02d}_{j}.png"
            im.save(fn)
            items.append(
                {
                    "file": str(fn),
                    "text": s,
                    "tpl": "bubble" if bubble else "plain",
                    "cands": cands,
                }
            )
    cand_names = sorted({k for it in items for k in it["cands"]})
    captions = [tpls[it["tpl"]].format(c) for it in items for c in it["cands"].values()]
    device, cache, lat, sigmas, noise, anima, delta = _setup(a, out, items, captions)

    C = len(cand_names)
    err = torch.full((len(CONDS), len(items), len(sigmas), C), float("nan"))
    t0 = time.time()
    n_fwd = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ci, cond in enumerate(CONDS):
            delta.scale = 1.0 if cond == "trained" else 0.0
            for ii, it in enumerate(items):
                names = [n for n in cand_names if n in it["cands"]]
                cs = [tpls[it["tpl"]].format(it["cands"][n]) for n in names]
                x = lat[ii : ii + 1].to(device)
                for si, sig in enumerate(sigmas):
                    nz = noise[ii, si : si + 1].to(device)
                    e = _errors(anima, x, nz, sig, cs, cache, device)
                    for n, v in zip(names, e):
                        err[ci, ii, si, cand_names.index(n)] = v
                    n_fwd += len(cs)
                _progress("classify_str", cond, ii, len(items), n_fwd, t0)
    del anima
    torch.cuda.empty_cache()
    torch.save(
        {
            "err": err,
            "cand_names": cand_names,
            "sigmas": sigmas,
            "conds": CONDS,
            "items": items,
        },
        out / "classify_str.pt",
    )
    _report_classify_str(out, err, cand_names, sigmas, CONDS, items)


def _report_classify_str(out: Path, err, cand_names, sigmas, conds, items):
    S = cand_names.index("S")
    contrasts = {
        "order (S < rev)": ["rev"],
        "order3 (S < rot)": ["rot"],
        "count (S < first,last)": ["first", "last"],
        "count3 (S < first2)": ["first2"],
        "identity (S < sub0,sub1)": ["sub0", "sub1"],
    }
    n2 = sum(len(it["text"]) == 2 for it in items)
    n3 = len(items) - n2
    L = [
        "# classify_str — same-noise diffusion classifier over kana strings",
        "",
        f"{n2} 2-kana + {n3} 3-kana held-out renders; summed FM error per latent "
        "under the true caption S vs named alternatives, same noise. "
        "win = S has the lower error against every listed alternative; "
        "gap = mean(alt − S) / mean spread over all candidates.",
        "",
    ]
    for name, alts in contrasts.items():
        ai = [cand_names.index(x) for x in alts if x in cand_names]
        if not ai:
            continue
        L += [
            f"## {name}",
            "",
            "| cond | " + " | ".join(f"σ {s:.2f}" for s in sigmas) + " |",
            "|---|" + "---|" * len(sigmas),
        ]
        for ci, cond in enumerate(conds):
            cells = []
            for si in range(len(sigmas)):
                e = err[ci, :, si, :]
                valid = ~torch.isnan(e[:, ai]).any(1)
                if int(valid.sum()) == 0:
                    cells.append("–")
                    continue
                es = e[valid]
                win = (es[:, ai] > es[:, S : S + 1]).all(1).float().mean()
                gap = (es[:, ai].mean(1) - es[:, S]).mean()
                spread = torch.nanmean(
                    torch.tensor([float(torch.std(r[~torch.isnan(r)])) for r in es])
                )
                cells.append(f"{win:.2f} ({gap / spread:+.2f})")
            L.append(f"| {cond} | " + " | ".join(cells) + " |")
        L.append("")
    L += [
        "## Per-item argmin (trained), σ columns",
        "",
        "| text | tpl | " + " | ".join(f"{s:.2f}" for s in sigmas) + " |",
        "|---|---|" + "---|" * len(sigmas),
    ]
    for ii, it in enumerate(items):
        row = []
        for si in range(len(sigmas)):
            e = err[0, ii, si]
            k = int(torch.argmin(torch.nan_to_num(e, nan=float("inf"))))
            row.append(cand_names[k])
        L.append(f"| {it['text']} | {it['tpl']} | " + " | ".join(row) + " |")
    (out / "classify_str.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
