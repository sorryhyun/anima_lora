#!/usr/bin/env python
"""kozh_geometry — where a ``lang`` run's KO / ZH rows sit against its seed's
kana and kanji balls (``_archive/task_report.md`` § 2's reads, for an arm).

Effective units (``raw × row_scale``, ``_offsets``). The seed's
families by glyph: kana (hiragana / katakana / ー) and kanji (CJK ideographs)
among the seed's rows. A family's stick = its rows' mean, its spikes = rows
less the stick, its ball's top-``K`` = the spikes' first ``K`` PCs. Per new
group (the run's ``lang``):

- row norm against the seed families'; the run's other rows held (max |Δ|);
- the group's own stick (its 8 rows' mean): length, cos to either seed stick;
- each row's cos to either stick and its stick component (row · ŝ / |s|),
  beside the seed rows' own;
- the group's spikes (rows less the group's stick): the share of their energy
  in either ball's top-K, beside 8 held-out rows of the same family (the ball
  refitted without them, ``DRAWS`` draws) and the random K / dim;
- each new row's nearest seed rows (cos of rows, cos of spikes).

    .venv/bin/python project/cjk_anima_reseed/probes/kozh_geometry.py kozh16

CPU, seconds. Prints; ``--out`` writes the numbers as json.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the reseed project
from reseed import REPO, bootstrap  # noqa: E402

bootstrap()

import numpy as np  # noqa: E402

K = 40
DRAWS = 200
PACK_JSON = "models/vocab_packs/anima_cjk_vocab_pack_punct.json"


def _offsets(path: str) -> dict:
    """ext id → offset (effective units) of a trained file's rows."""
    import torch

    p = Path(path) if Path(path).is_absolute() else REPO / path
    d = torch.load(p, map_location="cpu", weights_only=False)
    if "delta" in d:
        d = d["delta"]
    rs = float(d["row_scale"])
    return {int(e): (r.float() * rs).numpy() for e, r in zip(d["ext_ids"], d["raw"])}


def _cos(a, b) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _chars() -> dict:
    """ext id → char (the pack's char map, then its single-char qwen pieces)."""
    from transformers import AutoTokenizer

    j = json.loads((REPO / PACK_JSON).read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(str(REPO / "library/anima/configs/qwen3_06b"))
    c2r = dict(j["char"])
    for q, r in j["qwen"].items():
        s = tok.decode([int(q)])
        if len(s) == 1 and s not in c2r:
            c2r[s] = r
    out: dict = {}
    for c, r in c2r.items():
        out.setdefault(int(r), c)
    return out


def _family(c: str) -> str | None:
    if len(c) != 1:
        return None
    if "ぁ" <= c <= "ゖ" or "ァ" <= c <= "ヺ" or c == "ー":
        return "kana"
    if unicodedata.name(c, "").startswith("CJK UNIFIED"):
        return "kanji"
    return None


def _top(B: np.ndarray, k: int = K) -> np.ndarray:
    return np.linalg.svd(B, full_matrices=False)[2][:k]


def _in_top(X: np.ndarray, V: np.ndarray) -> float:
    return float(((X @ V.T) ** 2).sum() / (X**2).sum())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--out", default="")
    a = p.parse_args()
    from reseed.config import load

    run = load(a.run)
    run.use_pack()
    assert run.lang, f"{run.name}: no lang"
    arm = _offsets(str(run.dir / "trained.pt"))
    seed = _offsets(str(run.seed_rows()))
    ch = _chars()
    from reseed.pools import ext_encoder

    ext = ext_encoder()
    new = {}
    for g, lang in run.lang.items():
        (e,) = ext(True, g)
        assert e not in seed, f"{g} ({e}) has a seed row"
        new.setdefault(lang, []).append((g, e))
    held = max(float(np.abs(arm[e] - seed[e]).max()) for e in seed if e in arm)
    fam = {"kana": [], "kanji": []}
    for e in sorted(seed):
        f = _family(ch.get(e, ""))
        if f and np.linalg.norm(seed[e]) > 0:
            fam[f].append(e)
    R = {f: np.stack([seed[e] for e in es]) for f, es in fam.items()}
    stick = {f: X.mean(0) for f, X in R.items()}
    ball = {f: R[f] - stick[f] for f in R}
    top = {f: _top(ball[f]) for f in R}
    dim = R["kana"].shape[1]
    res: dict = {
        "run": run.name,
        "seed_rows": str(run.seed_rows()),
        "others_held_max_abs": held,
        "families": {f: len(es) for f, es in fam.items()},
        "random_top": K / dim,
    }
    print(
        f"{run.name} on {run.seed_rows().parent.name}: kana {len(fam['kana'])}, kanji "
        f"{len(fam['kanji'])} rows; every other row held (max |Δ| {held:.2e})"
    )
    print(
        f"sticks: |kana| {np.linalg.norm(stick['kana']):.1f}  |kanji| "
        f"{np.linalg.norm(stick['kanji']):.1f}  cos {_cos(stick['kana'], stick['kanji']):.3f}"
        f"  ({np.degrees(np.arccos(_cos(stick['kana'], stick['kanji']))):.0f}°)"
    )
    rng = np.random.default_rng(0)
    seed_reads = {}
    for f in R:
        n = np.linalg.norm(R[f], axis=1)
        cs = {g: R[f] @ stick[g] / (n * np.linalg.norm(stick[g])) for g in R}
        comp = {g: R[f] @ stick[g] / np.linalg.norm(stick[g]) ** 2 for g in R}
        sp = np.linalg.norm(ball[f], axis=1)
        # 8 held-out rows of the family: their spikes (their own mean out, as a
        # new group's) in the ball refitted without them, and in the other ball
        own, other = [], []
        g2 = "kanji" if f == "kana" else "kana"
        for _ in range(DRAWS):
            ix = rng.choice(len(R[f]), 8, replace=False)
            keep = np.setdiff1d(np.arange(len(R[f])), ix)
            Bk = R[f][keep] - R[f][keep].mean(0)
            X = R[f][ix] - R[f][ix].mean(0)
            own.append(_in_top(X, _top(Bk)))
            other.append(_in_top(X, top[g2]))
        seed_reads[f] = {
            "norm_median": float(np.median(n)),
            "spike_median": float(np.median(sp)),
            "cos_kana_stick": float(np.mean(cs["kana"])),
            "cos_kanji_stick": float(np.mean(cs["kanji"])),
            "comp_kana_stick": float(np.mean(comp["kana"])),
            "comp_kanji_stick": float(np.mean(comp["kanji"])),
            f"held8_in_{f}_top": float(np.mean(own)),
            f"held8_in_{g2}_top": float(np.mean(other)),
        }
        print(
            f"{f:7s} |row| {np.median(n):6.1f}  |spike| {np.median(sp):6.1f}  "
            f"cos→kana stick {np.mean(cs['kana']):.3f} kanji {np.mean(cs['kanji']):.3f}  "
            f"comp kana {np.mean(comp['kana']):.2f} kanji {np.mean(comp['kanji']):.2f}  "
            f"held-8 spikes in {f} top-{K} {np.mean(own):.3f}, in {g2} top-{K} "
            f"{np.mean(other):.3f}"
        )
    res["seed"] = seed_reads
    print(f"random top-{K} / {dim}: {K / dim:.3f}")
    allR = np.concatenate([R["kana"], R["kanji"]])
    allB = np.concatenate([ball["kana"], ball["kanji"]])
    all_c = [ch[e] for e in fam["kana"] + fam["kanji"]]
    res["groups"] = {}
    for lang, ge in new.items():
        X = np.stack([arm[e] for _, e in ge])
        gs = [g for g, _ in ge]
        n = np.linalg.norm(X, axis=1)
        s = X.mean(0)
        sp = X - s
        cs = {f: X @ stick[f] / (n * np.linalg.norm(stick[f])) for f in R}
        comp = {f: X @ stick[f] / np.linalg.norm(stick[f]) ** 2 for f in R}
        g_reads = {
            "glyphs": "".join(gs),
            "norm_median": float(np.median(n)),
            "spike_median": float(np.median(np.linalg.norm(sp, axis=1))),
            "stick_norm": float(np.linalg.norm(s)),
            "stick_cos_kana": _cos(s, stick["kana"]),
            "stick_cos_kanji": _cos(s, stick["kanji"]),
            "cos_kana_stick": float(cs["kana"].mean()),
            "cos_kanji_stick": float(cs["kanji"].mean()),
            "comp_kana_stick": float(comp["kana"].mean()),
            "comp_kanji_stick": float(comp["kanji"].mean()),
            "spikes_in_kana_top": _in_top(sp, top["kana"]),
            "spikes_in_kanji_top": _in_top(sp, top["kanji"]),
            "rows": {},
        }
        print(
            f"\n{lang} {''.join(gs)}: |row| {np.median(n):.1f}  |spike| "
            f"{g_reads['spike_median']:.1f}  own stick |s| {np.linalg.norm(s):.1f}, cos "
            f"kana {g_reads['stick_cos_kana']:.3f} kanji {g_reads['stick_cos_kanji']:.3f}\n"
            f"  rows cos→kana stick {cs['kana'].mean():.3f} kanji {cs['kanji'].mean():.3f}"
            f"  comp kana {comp['kana'].mean():.2f} kanji {comp['kanji'].mean():.2f}\n"
            f"  spikes in kana top-{K} {g_reads['spikes_in_kana_top']:.3f}, kanji top-{K} "
            f"{g_reads['spikes_in_kanji_top']:.3f}"
        )
        un = allR / np.linalg.norm(allR, axis=1, keepdims=True)
        ub = allB / np.linalg.norm(allB, axis=1, keepdims=True)
        for g, x, b in zip(gs, X, sp):
            cr = un @ (x / np.linalg.norm(x))
            cb = ub @ (b / np.linalg.norm(b))
            ir, ib = np.argsort(-cr)[:5], np.argsort(-cb)[:5]
            g_reads["rows"][g] = {
                "norm": float(np.linalg.norm(x)),
                "cos_kana_stick": float(cs["kana"][gs.index(g)]),
                "cos_kanji_stick": float(cs["kanji"][gs.index(g)]),
                "nearest_rows": [[all_c[i], float(cr[i])] for i in ir],
                "nearest_spikes": [[all_c[i], float(cb[i])] for i in ib],
            }
            print(
                f"  {g} |{np.linalg.norm(x):5.1f}|  rows "
                + " ".join(f"{all_c[i]} {cr[i]:.2f}" for i in ir)
                + "   spikes "
                + " ".join(f"{all_c[i]} {cb[i]:.2f}" for i in ib)
            )
        res["groups"][lang] = g_reads
    if len(new) == 2:
        (l1, e1), (l2, e2) = new.items()
        s1 = np.stack([arm[e] for _, e in e1]).mean(0)
        s2 = np.stack([arm[e] for _, e in e2]).mean(0)
        res["stick_cos_groups"] = _cos(s1, s2)
        print(f"\n{l1} stick ↔ {l2} stick cos {_cos(s1, s2):.3f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(f"→ {a.out}")


if __name__ == "__main__":
    main()
