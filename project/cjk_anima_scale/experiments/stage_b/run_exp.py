#!/usr/bin/env python
"""stage_b — proposal.md Stage B: a diverse singles donor, held-out kana (2026-09-26)

Stage A (transplant_piece) moved a piece direction out of sample onto held-out
piece rows. This is the singles version, where the line mode (and doubling)
lives. 36 donor kana are trained on in-line real words (spell_b's
``scene_spelled``: unspaced on the image, spaced in the caption, so every
glyph trains its single row), plus a **count tier** (proposal § 3.4 a:
one donor glyph alone at line px in a bubble it fills 0.2–0.4 of, captioned
alone). Ten kana no donor word contains are held out; their shared-Δ
transplant is read on five spelled words made only of them.

- **Donors** (``DONOR``): the 36 most frequent hiragana, after the held-out
  ten, in the pure-hiragana lines of the manga109s dialogue pool — with の
  swapped for the 37th (ふ): の and を are the two kana whose space-prefixed
  form is another Qwen token (``" の"`` → row 1296, not の's 31), so they
  cannot be spelled.
- **Donor words**: those lines (edge punctuation stripped) of 2–6 glyphs,
  every glyph a donor and none repeated (a repeated glyph in training would
  itself teach doubling), and no 3-glyph substring of ``HELD_IN``. That leaves
  622 words, and every donor is in at least 8 of them. A draw picks a donor
  glyph uniformly, then one of its words, so exposure is balanced per row.
- **Held out** (``HELD``): ひ ま わ り さ く ら み ど も; words ``HELD_WORDS``.
- ``HELD_IN`` = こんにちは, all donor glyphs, trigram-held-out from the donor
  words: the donor's own composition read. Its keys (spelled + the five
  singles, en) and あ う が と (en) are already in the floor cache.
- Count tier: glyph px 28–40 font (the single 24–40 px row, 0.5–0.7; a
  single under 24 px has no window, so the b0305 line px get no count
  items), in b0507 at weight 0.3.

Legs:
  data      (CPU) the donor data dir
  train     (GPU) the donor rows, the fixed trainer (90 steps / row)
  read      (GPU) the donor on its own rows: ``HELD_IN`` spelled + the donor
            singles the floor holds (en), vs the floor cache
  floor     (GPU) the one floor render (proposal § 2): the held-out spelled
            words + the held-out singles, en + swap, into ``native_spell/``
  build     (CPU) ``u_S`` (the donors' mean tangential Δ), its split-half cos,
            the step (mean projection); arms = the seed rows of ``HELD`` +
            α · step · ``u_S`` (``--alphas``) and a random ⟂ control at max α
  transplant (GPU) the arms on the held-out keys, vs the floor

``--dry_run`` prints the word list, the table, the encodings and the arms.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from collections import Counter
from pathlib import Path

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, SEED_ROWS, bootstrap, floor_dir  # noqa: E402

bootstrap()
EXP = OUT / "experiments"  # the row arms (a trained.pt each)

from bench._common import make_run_dir, write_result  # noqa: E402

NAME = "run0926_stage_b"
DONOR = "いなんかあうしたこだそれでよはおとふすてにるやえつがねきめろけちせばほご"
HELD = "ひまわりさくらみども"
HELD_WORDS = ("ひまわり", "さくら", "みどり", "くもり", "まくら")
HELD_IN = "こんにちは"
DONOR_READ = (HELD_IN, "こ", "ん", "に", "ち", "は", "あ", "う", "が", "と")
WORD_LEN = (2, 6)
COUNT_WEIGHT = 0.3  # of b0507's items
TAG = "spell"  # the native_spell/ cache the reads share

_WORDS: list = []  # the donor words (set in main before the build forks)
_BY_GLYPH: dict = {}


def spell(s: str) -> str:
    return " ".join(s)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs",
        nargs="+",
        default=["data"],
        choices=["data", "train", "read", "floor", "build", "transplant"],
    )
    p.add_argument("--alphas", type=float, nargs="+", default=[1.0])
    p.add_argument("--no_random", action="store_true", help="skip the control arm")
    p.add_argument("--workers", type=int, help="data: render processes")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def arm_rc():
    from cjk_scale.config import RunConfig

    return RunConfig(
        name=NAME, path=Path(__file__), vocabs=(f"chars:{DONOR}",), read=()
    )


# ----------------------------------------------------------------------------
# words


def donor_words() -> list[str]:
    from cjk_scale.config import phrase_file

    assert not set(DONOR) & set(HELD), set(DONOR) & set(HELD)
    assert len(set(DONOR)) == len(DONOR) == 36
    ds = set(DONOR)
    tri = {HELD_IN[i : i + 3] for i in range(len(HELD_IN) - 2)}
    out = set()
    for ln in Path(phrase_file()).read_text(encoding="utf-8").splitlines():
        t = ln.split("\t")[0]
        t = re.sub(r"^[「『\s]+", "", t)
        t = re.sub(r"[！？!?、。…・〜～「」『』\s]+$", "", t)
        if not (WORD_LEN[0] <= len(t) <= WORD_LEN[1]):
            continue
        if set(t) <= ds and len(set(t)) == len(t) and not any(x in t for x in tri):
            out.add(t)
    return sorted(out)


def set_words(words: list[str]) -> dict:
    _WORDS[:] = words
    _BY_GLYPH.clear()
    for g in DONOR:
        _BY_GLYPH[g] = [w for w in words if g in w]
    cov = {g: len(v) for g, v in _BY_GLYPH.items()}
    assert min(cov.values()) > 0, cov
    assert not any(set(w) & set(HELD) for w in words)
    return cov


# ----------------------------------------------------------------------------
# recipes


def scene_spelled(pools, rng, p: dict):
    """A donor word in one bubble, unspaced on the image, spaced in the
    caption (spell_b's recipe); the word is drawn glyph-first."""
    from cjk_scale.recipes import _draw_scene, _target

    word = rng.choice(_BY_GLYPH[rng.choice(DONOR)])
    f = p.get("fill", [0.7, 1.0])
    lo, hi = f if isinstance(f, list) else (f, f)
    fill = rng.uniform(float(lo), float(hi))
    item = _draw_scene(
        pools,
        rng,
        word,
        min_glyph=int(p.get("min_glyph", 28)),
        fill=fill,
        max_lines=1,
        target_px=_target(rng, p),
        fill_max=fill,
        fill_min=float(p.get("fill_min", 0)),
    )
    if item is None:
        return None
    quoted = f'"{word}"'
    assert item.caption.count(quoted) == 1, item.caption
    item.caption = item.caption.replace(quoted, f'"{spell(word)}"')
    return item


def scene_single_small(pools, rng, p: dict):
    """The count tier: one donor glyph at line px, in a bubble whose one-glyph
    fit it fills ``fill`` (0.2–0.4) of, captioned alone."""
    from cjk_scale.recipes import _draw_scene, _fit_px, _target

    glyph = rng.choice(pools.singles)
    target = _target(rng, p)
    lo, hi = (float(x) for x in p["fill"])
    ok = {
        j
        for j in pools.single_idx
        if lo <= target / max(_fit_px(pools.scenes[j]["region"], 1, True), 1e-6) <= hi
    }
    if not ok:
        return None
    return _draw_scene(
        dataclasses.replace(pools, single_idx=ok),
        rng,
        glyph,
        min_glyph=int(p.get("min_glyph", 12)),
        fill=hi,
        max_lines=1,
        singles_only=True,
        target_px=target,
        fill_max=hi,
        fill_min=lo,
    )


def table() -> tuple:
    """spell_b's two piece-band ``scene_piece`` tiers drawing the donor words,
    brought in by the single kind; b0507 adds the count tier."""
    from cjk_scale.builder import TABLE, Group, Tier

    tiers = {
        g.name: next(t for t in g.tiers if t.recipe == "scene_piece")
        for g in TABLE
        if g.kind == "piece"
    }
    count = Tier(
        "scene_single_small",
        COUNT_WEIGHT,
        {"glyph_px": [28, 40], "fill": [0.2, 0.4], "min_glyph": 12},
    )
    return (
        Group(
            "b0507",
            "single",
            (0.5, 0.7),
            0.5,
            (
                Tier("scene_spelled", 1 - COUNT_WEIGHT, tiers["b0507"].params),
                count,
            ),
        ),
        Group(
            "b0305",
            "single",
            (0.3, 0.5),
            0.5,
            (Tier("scene_spelled", 1.0, tiers["b0305"].params),),
        ),
    )


# ----------------------------------------------------------------------------
# encodings + rows


def encoder():
    import os

    from transformers import AutoTokenizer

    from library.anima import ext_vocab
    from library.anima.ext_vocab import T5_TABLE_SIZE, HybridT5Encoder
    from library.anima.vocab_pack import resolve_pack_prefix
    from library.env import resolve_under_home

    t5 = AutoTokenizer.from_pretrained(
        resolve_under_home("library/anima/configs/t5_old")
    )
    qw = AutoTokenizer.from_pretrained(
        resolve_under_home("library/anima/configs/qwen3_06b")
    )
    _, mapping = ext_vocab.load_ext_assets(
        resolve_pack_prefix(os.environ["ANIMA_VOCAB_PACK"])
    )
    enc = HybridT5Encoder.from_mapping(t5, qw, mapping)

    def ext(text):
        ids, mask = enc.encode(f'Japanese text reads as "{text}".', 512)
        return [
            i - T5_TABLE_SIZE for i, m in zip(ids, mask) if m and i >= T5_TABLE_SIZE
        ]

    return ext


def check_spelling(ext, words) -> dict:
    """Every glyph is one single ext id with a seed row; every spelled word
    (donor, held-out, ``HELD_IN``) encodes to its glyphs' ids and nothing else."""
    single = {c: ext(c) for c in DONOR + HELD}
    assert all(len(v) == 1 for v in single.values()), {
        c: v for c, v in single.items() if len(v) != 1
    }
    ids = {c: v[0] for c, v in single.items()}
    for w in (*words, *HELD_WORDS, HELD_IN):
        got = ext(spell(w))
        assert got == [ids[c] for c in w], (w, got)
    return ids


def rows(path: Path) -> tuple[dict, dict]:
    """``{ext id: effective row}`` (raw × row_scale) and the file itself."""
    import torch

    sd = torch.load(path, map_location="cpu", weights_only=False)
    d = sd["delta"]
    s = float(d["row_scale"])
    return {int(e): d["raw"][i].float() * s for i, e in enumerate(d["ext_ids"])}, sd


def tangential(run_rows: dict, seed: dict, ids: list[int]):
    """Δ vs the seed for ``ids``, each with its component along its own seed
    row removed (transplant_line § 1)."""
    import torch
    import torch.nn.functional as F

    D = torch.stack([run_rows[e] - seed[e] for e in ids])
    S = F.normalize(torch.stack([seed[e] for e in ids]), dim=1)
    return D - (D * S).sum(1, keepdim=True) * S


def direction(ids: dict) -> dict:
    import torch
    import torch.nn.functional as F

    seed, _ = rows(SEED_ROWS)
    donor, _ = rows(OUT / NAME / "trained.pt")
    d_ids = [ids[c] for c in DONOR]
    T = tangential(donor, seed, d_ids)
    u = F.normalize(T.mean(0), dim=0)
    P = T @ u
    g = torch.Generator().manual_seed(0)
    r = torch.randn(u.shape[0], generator=g)
    r = F.normalize(r - (r @ u) * u, dim=0)
    rel = [
        float((donor[e] - seed[e]).norm() / seed[e].norm()) for e in d_ids
    ]  # the donor rows' drift
    # the donors' pairwise cos (off-diagonal mean), as spell_b § 5 read it
    Tn = F.normalize(T, dim=1)
    C = Tn @ Tn.T
    n = len(d_ids)
    return {
        "u": u,
        "r": r,
        "step": float(P.mean()),
        "proj_sd": round(float(P.std()), 3),
        "proj": {c: round(float(x), 2) for c, x in zip(DONOR, P)},
        "energy_frac": round(float((P**2).sum() / (T**2).sum()), 4),
        "pairwise_cos": round(float((C.sum() - n) / (n * (n - 1))), 4),
        "split_half_cos": round(
            float(
                F.normalize(T[0::2].mean(0), dim=0)
                @ F.normalize(T[1::2].mean(0), dim=0)
            ),
            4,
        ),
        "rel_drift": {c: round(x, 3) for c, x in zip(DONOR, rel)},
        "held_seed_norm": {c: round(float(seed[ids[c]].norm()), 2) for c in HELD},
    }


def arm_names(label: str, alphas, random: bool) -> dict:
    out = {f"tb_{label}_u{a:g}": ("u", a) for a in alphas}
    if random:
        out[f"tb_{label}_rand{max(alphas):g}"] = ("r", max(alphas))
    return out


def build_arm(name: str, vec, step: float, tgt_ids: list[int]) -> dict:
    """The seed's trained.pt with ``step · vec`` added to the held-out rows."""
    import torch

    sd = torch.load(SEED_ROWS, map_location="cpu", weights_only=False)
    d = sd["delta"]
    s = float(d["row_scale"])
    idx = {int(e): i for i, e in enumerate(d["ext_ids"])}
    raw = d["raw"].clone()
    moved = {}
    for e in tgt_ids:
        i = idx[e]
        before = raw[i].float() * s
        after = before + step * vec
        raw[i] = (after / s).to(raw.dtype)
        moved[e] = {
            "rel": round(float(step * vec.norm() / before.norm()), 3),
            "norm_ratio": round(float(after.norm() / before.norm()), 3),
        }
    out = {
        **sd,
        "delta": {**d, "raw": raw},
        "arm": "rows",
        "seed_merged": str(SEED_ROWS),
        "transplant": {"step": step, "rows": tgt_ids, "moved": moved, "src": NAME},
    }
    dst = EXP / name
    dst.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst / "trained.pt")
    return moved


# ----------------------------------------------------------------------------
# reads


def held_keys() -> list[str]:
    return [*(spell(w) for w in HELD_WORDS), *HELD]


def donor_keys() -> list[str]:
    return [spell(k) if len(k) > 1 else k for k in DONOR_READ]


def native_read(arm_path: Path, data_path: Path, chars, clauses: str) -> Path:
    from cjk_scale.config import RunConfig
    from cjk_scale.eval import TRAINED_ARM, probe_args
    from stages import run as run_stage

    reads = arm_path / f"native_{TAG}" / "native_reads.json"
    cl = clauses.split(",")
    if reads.exists():
        held = {(m["text"], m["clause"]) for m in json.loads(reads.read_text("utf-8"))}
        if all((k, c) in held for k in chars for c in cl):
            print(f"  (read from disk: {reads})", flush=True)
            return reads
    rc = RunConfig(name=NAME, path=Path(__file__), vocabs=(), read=())
    a = probe_args(
        rc,
        TRAINED_ARM,
        ["native"],
        [
            "--eval_tag",
            TAG,
            "--native_chars",
            ",".join(chars),
            "--native_clauses",
            clauses,
        ],
    )
    a.arm_path, a.data_path = str(arm_path), str(data_path)
    run_stage("native", a)
    return reads


def check_floor(chars, clauses: str) -> Path:
    floor = floor_dir() / f"native_{TAG}" / "native_reads.json"
    held = {(m["text"], m["clause"]) for m in json.loads(floor.read_text("utf-8"))}
    miss = [(k, c) for k in chars for c in clauses.split(",") if (k, c) not in held]
    assert not miss, f"floor cache lacks {miss} — run the floor leg"
    return floor


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def hits(path: Path, chars, clauses: str) -> dict:
    """``{(text, clause, pi, seed): {official, contained, le1, le2, repeat,
    dup, le1c}}`` per render — the paired unit. ``dup``: a word's read holds a
    doubled glyph (ひまわりり; no target repeats a glyph); ``le1c``: ≤ 1 edit
    once doubled glyphs are collapsed, i.e. what in-word doubling costs."""
    from common.readers import norm

    cl = set(clauses.split(","))
    out = {}
    for m in json.loads(path.read_text("utf-8")):
        if m["text"] not in chars or m["clause"] not in cl:
            continue
        t = norm(m["text"])
        reads = [
            norm(r.get(x) or "") for r in m.get("reads", []) for x in ("sfx", "vl")
        ]
        best = min((_lev(r, t) for r in reads if r), default=len(t))
        best_c = min(
            (_lev(re.sub(r"(.)\1+", r"\1", r), t) for r in reads if r), default=len(t)
        )
        out[(m["text"], m["clause"], m["pi"], m["seed"])] = {
            "official": bool(m.get("hit_sfx")) and bool(m.get("hit_vl")),
            "contained": any(t in r for r in reads),
            "le1": len(t) > 2 and best <= 1,
            "le2": len(t) > 2 and best <= 2,
            "repeat": len(t) == 1 and any(r.count(t) >= 2 for r in reads),
            "kana": any(re.search(r"[ぁ-ヿ]", r) for r in reads),
            "dup": len(t) > 1 and any(re.search(r"(.)\1", r) for r in reads),
            "le1c": len(t) > 2 and best_c <= 1,
        }
    return out


METRICS = ("official", "contained", "le1", "le2", "repeat", "kana", "dup", "le1c")


def tally(h: dict) -> dict:
    per: dict = {}
    for (text, clause, _pi, _s), v in h.items():
        c = per.setdefault(f"{text}|{clause}", dict.fromkeys(("n", *METRICS), 0))
        c["n"] += 1
        for k in METRICS:
            c[k] += v[k]
    words = {k: v for k, v in per.items() if len(k.split("|")[0].replace(" ", "")) > 1}
    singles = {k: v for k, v in per.items() if k not in words}
    tot = {
        name: {k: sum(v[k] for v in grp.values()) for k in ("n", *METRICS)}
        for name, grp in (("words", words), ("singles", singles))
    }
    for k, c in sorted(per.items()):
        print(
            f"  {k:<16} off {c['official']:>2}  cont {c['contained']:>2}  "
            f"≤1 {c['le1']:>2}  ≤2 {c['le2']:>2}  rep {c['repeat']:>2}  "
            f"dup {c['dup']:>2}  ≤1c {c['le1c']:>2} / {c['n']}",
            flush=True,
        )
    for name, c in tot.items():
        print(
            f"  {name.upper():<16} off {c['official']:>3}  cont {c['contained']:>3}  "
            f"≤1 {c['le1']:>3}  ≤2 {c['le2']:>3}  rep {c['repeat']:>3}  "
            f"kana {c['kana']:>3}  dup {c['dup']:>3}  ≤1c {c['le1c']:>3} / {c['n']}",
            flush=True,
        )
    return {"per_key": per, "total": tot}


def paired(a: dict, b: dict) -> dict:
    """McNemar (exact two-sided binomial) of ``a`` vs ``b`` on their shared
    renders: ``{metric: [gained, lost, p]}``."""
    from math import comb

    keys = sorted(set(a) & set(b))
    out = {}
    for k in METRICS:
        g = sum(a[x][k] and not b[x][k] for x in keys)
        lo = sum(b[x][k] and not a[x][k] for x in keys)
        n = g + lo
        p = (
            min(1.0, 2 * sum(comb(n, i) for i in range(min(g, lo) + 1)) / 2**n)
            if n
            else 1.0
        )
        out[k] = [g, lo, float(f"{p:.2g}")]
    return {"n": len(keys), **out}


# ----------------------------------------------------------------------------


def main():
    args = parse_args()
    from cjk_scale import recipes

    recipes.RECIPES["scene_spelled"] = scene_spelled
    recipes.RECIPES["scene_single_small"] = scene_single_small
    rc = arm_rc()
    words = donor_words()
    cov = set_words(words)
    print(
        f"donor words {len(words)} (by length {dict(sorted(Counter(map(len, words)).items()))}); "
        f"per-glyph words min {min(cov.values())} ({min(cov, key=cov.get)}), "
        f"median {sorted(cov.values())[len(cov) // 2]}",
        flush=True,
    )
    tb = table()
    for g in tb:
        print(
            f"{g.name} σ {g.band}: {[(t.recipe, t.weight, t.params) for t in g.tiers]}",
            flush=True,
        )
    ext = encoder()
    ids = check_spelling(ext, words)
    print(
        f"encodings ok; held ids {json.dumps({c: ids[c] for c in HELD}, ensure_ascii=False)}",
        flush=True,
    )
    arms = arm_names(args.label, args.alphas, not args.no_random)
    print(f"arms: {arms}", flush=True)
    metrics: dict = {
        "donor": DONOR,
        "held": HELD,
        "held_words": list(HELD_WORDS),
        "held_in": HELD_IN,
        "n_words": len(words),
        "per_glyph_words": cov,
    }
    if args.dry_run:
        print(" ".join(words[:80]), flush=True)
        return
    run_dir = make_run_dir(
        "stage_b", label=args.label, root=LINE / "experiments" / "stage_b" / "results"
    )
    (OUT / NAME).mkdir(parents=True, exist_ok=True)
    (OUT / NAME / "donor_words.json").write_text(
        json.dumps(words, ensure_ascii=False, indent=0), encoding="utf-8"
    )
    if "data" in args.legs:
        from cjk_scale.builder import build

        build(rc, workers=args.workers, table=tb)
    if "train" in args.legs:
        from cjk_scale.train import train

        train(rc)
    if "read" in args.legs:
        chars = donor_keys()
        floor = check_floor(chars, "en")
        print("floor (the seed dir's native_spell/ cache), donor keys:", flush=True)
        fh = hits(floor, chars, "en")
        metrics["donor_floor"] = tally(fh)
        print(f"{NAME}, donor keys:", flush=True)
        th = hits(
            native_read(OUT / NAME, OUT / NAME / "data", chars, "en"), chars, "en"
        )
        metrics["donor_trained"] = tally(th)
        metrics["donor_paired"] = paired(th, fh)
        print(f"  paired vs floor {metrics['donor_paired']}", flush=True)
    if "floor" in args.legs:
        from cjk_scale.eval import ensure_native_floor

        n = ensure_native_floor(rc, f"native_{TAG}", held_keys(), "en,swap")
        print(f"floor: {n} keys rendered into native_{TAG}/", flush=True)
        metrics["floor_rendered"] = n
    if "build" in args.legs:
        dr = direction(ids)
        info = {k: v for k, v in dr.items() if k not in ("u", "r")}
        print(json.dumps(info, ensure_ascii=False), flush=True)
        metrics["direction"] = info
        metrics["arms"] = {}
        tgt = [ids[c] for c in HELD]
        for name, (which, a) in arms.items():
            moved = build_arm(name, dr[which], a * dr["step"], tgt)
            print(f"built {name}: {json.dumps(moved)}", flush=True)
            metrics["arms"][name] = {"vec": which, "alpha": a, "moved": moved}
    if "transplant" in args.legs:
        chars = held_keys()
        floor = check_floor(chars, "en,swap")
        print("floor (the seed dir's native_spell/ cache), held-out keys:", flush=True)
        fh = hits(floor, chars, "en,swap")
        metrics["floor"] = tally(fh)
        reads = metrics.setdefault("reads", {})
        arm_h = {}
        for name in arms:
            print(f"{name}:", flush=True)
            arm_h[name] = hits(
                native_read(EXP / name, EXP / name / "data", chars, "en,swap"),
                chars,
                "en,swap",
            )
            reads[name] = tally(arm_h[name])
            reads[name]["paired_vs_floor"] = paired(arm_h[name], fh)
            print(f"  paired vs floor {reads[name]['paired_vs_floor']}", flush=True)
        us = [n for n, (w, _a) in arms.items() if w == "u"]
        rs = [n for n, (w, _a) in arms.items() if w == "r"]
        for r in rs:
            for u in us:
                reads[u][f"paired_vs_{r}"] = paired(arm_h[u], arm_h[r])
                print(f"  {u} vs {r} {reads[u][f'paired_vs_{r}']}", flush=True)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(OUT / NAME), *(str(EXP / n) for n in arms)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
