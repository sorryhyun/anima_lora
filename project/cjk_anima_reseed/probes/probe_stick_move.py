#!/usr/bin/env python
"""probe_stick_move — the trained moves in stick / ball terms (2026-10-08)

Per family, for f0 − start, pres − f0 and pres − start (effective units,
start = `seed_fixed_1005_stick080`): the share of the move's energy in its
shared vector (Δstick: every row by the same vector, the ball translated),
|Δstick| / |stick|, cos(Δstick, stick), the angle the stick turns, its
length, and the per-row residual's rms against the ball's. Then the
families' Δstick against each other and pres's against f0's.

Then the ball's shape per row set: the spikes' length CV, their PR / k50 /
k90 / top-1 energy, the pairs' |cos| p95, the spikes' angle to the stick —
beside an isotropic ball of the same rows and lengths — and at f0's rows the
closest spike pairs and the top 3 PCs' extremes, by glyph. CPU, seconds;
reads the rows only.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the reseed project
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reseed import bootstrap  # noqa: E402

bootstrap()

import numpy as np  # noqa: E402
import probe_jl as J  # noqa: E402
from probe_geom import _families  # noqa: E402
from reseed import REPO  # noqa: E402


def _cos(a, b) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    rows = {
        "start": J._offsets(J.STICK080),
        "f0": J._offsets(J.F0),
        "pres": J._offsets(J.PRES),
    }
    ids = sorted(set.intersection(*(set(r) for r in rows.values())))
    fam = {f: [ids[k] for k in ks] for f, ks in _families(ids).items()}
    dstick = {}
    for f in ("kana", "kanji"):
        ks = [e for e in fam[f] if np.linalg.norm(rows["start"][e]) > 0]
        R = {n: np.stack([r[e] for e in ks]) for n, r in rows.items()}
        for a, b in (("f0", "start"), ("pres", "f0"), ("pres", "start")):
            X = R[a] - R[b]
            v = X.mean(0)
            dstick[f, a, b] = v
            s0, s1 = R[b].mean(0), R[a].mean(0)
            res = X - v
            ball = R[b] - s0
            ang = np.degrees(np.arccos(np.clip(_cos(s1, s0), -1, 1)))
            print(
                f"{f:5s} {a} − {b:5s} rows {len(X):4d}  "
                f"shared {len(X) * (v**2).sum() / (X**2).sum():.2f}  "
                f"|Δstick|/|stick| {np.linalg.norm(v) / np.linalg.norm(s0):.3f}  "
                f"cos(Δstick, stick) {_cos(v, s0):+.3f}  turn {ang:.2f}°  "
                f"|stick| {np.linalg.norm(s0):.1f}→{np.linalg.norm(s1):.1f}  "
                f"resid/ball {np.sqrt((res**2).sum(1).mean() / (ball**2).sum(1).mean()):.3f}"
            )
    for a, b in (("f0", "start"), ("pres", "f0")):
        print(
            f"cos(Δstick {a} − {b}: kana, kanji) "
            f"{_cos(dstick['kana', a, b], dstick['kanji', a, b]):+.3f}"
        )
    for f in ("kana", "kanji"):
        print(
            f"{f} cos(Δstick pres − f0, Δstick f0 − start) "
            f"{_cos(dstick[f, 'pres', 'f0'], dstick[f, 'f0', 'start']):+.3f}"
        )
    _shape(rows, fam)
    _glyphs(rows["f0"])


def _shape(rows: dict, fam: dict) -> None:
    rng = np.random.default_rng(0)

    def read(B, s):
        L = np.linalg.norm(B, axis=1)
        U = B / L[:, None]
        off = np.abs((U @ U.T)[~np.eye(len(B), dtype=bool)])
        ev = np.linalg.svd(B, compute_uv=False) ** 2
        cum = np.cumsum(ev) / ev.sum()
        ang = np.degrees(np.arccos(np.clip(U @ (s / np.linalg.norm(s)), -1, 1)))
        return (
            f"|spike| CV {L.std() / L.mean():.3f} "
            f"({L.min() / L.mean():.2f}–{L.max() / L.mean():.2f} of mean)  "
            f"PR {ev.sum() ** 2 / (ev**2).sum():.1f}  "
            f"k50 {np.searchsorted(cum, 0.5) + 1}  k90 {np.searchsorted(cum, 0.9) + 1}  "
            f"top1 {ev[0] / ev.sum():.3f}  |cos| p95 {np.quantile(off, 0.95):.3f}  "
            f"∠stick p5–p95 {np.quantile(ang, 0.05):.0f}–{np.quantile(ang, 0.95):.0f}°"
        )

    for f in ("kana", "kanji"):
        ks = [e for e in fam[f] if np.linalg.norm(rows["start"][e]) > 0]
        for name, r in rows.items():
            R = np.stack([r[e] for e in ks])
            print(f"{f:5s} {name:9s} n {len(ks):4d}  {read(R - R.mean(0), R.mean(0))}")
        B = R - R.mean(0)
        G = rng.standard_normal(B.shape)
        G -= G.mean(0)
        G *= (np.linalg.norm(B, axis=1) / np.linalg.norm(G, axis=1))[:, None]
        print(
            f"{f:5s} isotropic n {len(ks):4d}  {read(G, rng.standard_normal(B.shape[1]))}"
        )


def _glyphs(rows: dict) -> None:
    """The closest spike pairs and the top PCs' extremes, by glyph (as
    `probe_geom._families` maps them)."""
    import json

    from transformers import AutoTokenizer

    pk = REPO / "models/vocab_packs/anima_cjk_vocab_pack_preview51"
    j = json.loads(
        (pk.parent / "anima_cjk_vocab_pack_preview51.json").read_text(encoding="utf-8")
    )
    tr = json.loads(
        (pk / "anima_cjk_vocab_pack_preview51_trained.json").read_text(encoding="utf-8")
    )
    tok = AutoTokenizer.from_pretrained(str(REPO / "library/anima/configs/qwen3_06b"))
    c2r = dict(j["char"])
    for q, r in j["qwen"].items():
        s = tok.decode([int(q)])
        if len(s) == 1 and s not in c2r:
            c2r[s] = r
    for f, chars in (("kana", tr["hiragana"] + tr["katakana"]), ("kanji", tr["kanji"])):
        cs = [
            c
            for c in chars
            if c in c2r and c2r[c] in rows and np.linalg.norm(rows[c2r[c]]) > 0
        ]
        B = np.stack([rows[c2r[c]] for c in cs])
        B -= B.mean(0)
        U = B / np.linalg.norm(B, axis=1, keepdims=True)
        iu = np.triu_indices(len(cs), 1)
        C = (U @ U.T)[iu]
        top = np.argsort(-C)[:15]
        print(
            f"{f} closest spikes (f0): "
            + "  ".join(f"{cs[iu[0][k]]}{cs[iu[1][k]]} {C[k]:.2f}" for k in top)
        )
        _, sv, Vt = np.linalg.svd(B, full_matrices=False)
        for p in range(3):
            sc = B @ Vt[p]
            hi, lo = np.argsort(-sc)[:8], np.argsort(sc)[:8]
            print(
                f"  PC{p + 1} {sv[p] ** 2 / (sv**2).sum():.3f}  "
                f"+ {''.join(cs[i] for i in hi)}   − {''.join(cs[i] for i in lo)}"
            )


if __name__ == "__main__":
    main()
