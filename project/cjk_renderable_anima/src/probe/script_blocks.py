#!/usr/bin/env python
"""script_blocks — KR / ZH 12-row Δ0-recipe blocks against the Δ1 table
(2026-09-18, user's question: does Δ1's row geometry carry across scripts,
and does the one compositional script — hangul — compose?). CPU, seconds.

Three reads, all on ``trained.pt`` (``raw × row_scale``, absolute units):

1. shared direction: cos(m̂_block, m̂_Δ1) — same-loss JA blocks sit at
   0.67–0.72 (``table_geometry_2026_09_18.md``); a KR/ZH block at that level
   means m̂ is a text-render direction and Δ1's μ warm-starts any script.
2. hangul grid: rows 가거고구 / 나너노누 / 다더도두 (consonant × vowel);
   the vowel-difference vectors (가−거, 나−너, 다−더 …) and the consonant
   ones (가−나, 거−너 …) should be parallel if the row composes — cos among
   same-kind differences vs random-pair differences.
3. simplified hanzi ↔ Δ1 shinjitai: 气乐变见长时话 ↔ 気楽変見長時話 (same
   meaning, different glyph) against each Δ1 row's own shape neighbour and
   against random Δ1 rows; the control rows 门书车们这 have no Δ1 partner.

    .venv/bin/python project/cjk_renderable_anima/src/probe/script_blocks.py
"""

from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HOME = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(HOME / "project/cjk_renderable_anima/src"))
logging.disable(logging.CRITICAL)

WP = HOME / "output/wake_probe"
D1 = WP / "rows_synth_d1_d1_s53k"
KR = WP / "rows_synth_pair_kr_pairEN_rb62f_lr2e-3"
ZH = WP / "rows_synth_pair_zh_pairEN_rb62f_lr2e-3"
D0 = WP / "rows_synth_pair_d0_pairEN_rb62f_lr2e-3"  # the JA block, same recipe

GRID = [list("가거고구"), list("나너노누"), list("다더도두")]  # consonant rows × vowel cols
SIMP = dict(zip("气乐变见长时话", "気楽変見長時話"))
CTRL = list("门书车们这")


def load(arm_dir: Path):
    from bench.rows_manifold import load_arm, load_text_side
    from train.encoder import row_texts

    sd, raw, ext_ids = load_arm(arm_dir)
    _, pack, tok, _ = load_text_side()
    texts = row_texts(tok, pack, ext_ids)
    keep = [i for i, e in enumerate(ext_ids) if e in texts]
    rows = raw[keep] * float(sd["delta"]["row_scale"])
    return {texts[ext_ids[i]]: rows[k] for k, i in enumerate(keep)}


def mhat(t: dict[str, torch.Tensor]):
    m = torch.stack(list(t.values())).mean(0)
    return m / m.norm()


def cos(a, b) -> float:
    return float(F.cosine_similarity(a, b, dim=0))


def diff_parallelism(t, pairs, n_rand=2000, seed=0):
    """mean cos among the difference vectors of ``pairs`` vs random pairs."""
    d = [t[a] - t[b] for a, b in pairs]
    same = [cos(x, y) for x, y in itertools.combinations(d, 2)]
    g = torch.Generator().manual_seed(seed)
    names = list(t)
    rand = []
    for _ in range(n_rand):
        i = torch.randperm(len(names), generator=g)[:4].tolist()
        rand.append(cos(t[names[i[0]]] - t[names[i[1]]], t[names[i[2]]] - t[names[i[3]]]))
    rand = torch.tensor(rand)
    return sum(same) / len(same), float(rand.mean()), float(rand.quantile(0.95))


def main():
    d1 = load(D1)
    m1 = mhat(d1)
    blocks = {n: p for n, p in [("ja_d0", D0), ("kr", KR), ("zh", ZH)] if (p / "trained.pt").exists()}
    print("## 1. shared direction vs Δ1 (JA Δ0 block = same-loss reference)")
    tabs = {n: load(p) for n, p in blocks.items()}
    for n, t in tabs.items():
        mb = mhat(t)
        along = torch.stack([F.cosine_similarity(r, mb, dim=0) for r in t.values()]).mean()
        norms = torch.stack([r.norm() for r in t.values()])
        print(
            f"  {n:6s} rows {len(t):2d}  ‖row‖ {norms.mean():6.1f}  cos(row, m̂_block) {along:.3f}"
            f"  cos(m̂_block, m̂_Δ1) {cos(mb, m1):.3f}"
            f"  mean cos(row, m̂_Δ1) {torch.stack([F.cosine_similarity(r, m1, dim=0) for r in t.values()]).mean():.3f}"
        )
    for a, b in itertools.combinations(tabs, 2):
        print(f"  cos(m̂_{a}, m̂_{b}) {cos(mhat(tabs[a]), mhat(tabs[b])):.3f}")

    if "kr" in tabs:
        t = tabs["kr"]
        print("\n## 2. hangul grid — are the jamo differences parallel?")
        vow = [(r[j], r[k]) for r in GRID for j, k in itertools.combinations(range(4), 2)]
        con = [(GRID[a][c], GRID[b][c]) for c in range(4) for a, b in itertools.combinations(range(3), 2)]
        for name, pairs in [("vowel diffs (same consonant)", vow), ("consonant diffs (same vowel)", con)]:
            s, r, r95 = diff_parallelism(t, pairs)
            print(f"  {name:32s} n {len(pairs):2d}  mean cos {s:+.3f}   random {r:+.3f} (p95 {r95:+.3f})")
        # strict parallel test: same vowel change across consonants, e.g. 가−거 ∥ 나−너 ∥ 다−더
        strict = []
        for j, k in itertools.combinations(range(4), 2):
            d = [t[GRID[r][j]] - t[GRID[r][k]] for r in range(3)]
            strict += [cos(x, y) for x, y in itertools.combinations(d, 2)]
        print(f"  same vowel change across consonants (가−거 ∥ 나−너 …) n {len(strict)}  mean cos {sum(strict)/len(strict):+.3f}")
        C = torch.stack(list(t.values()))
        Cc = F.cosine_similarity(C[:, None], C[None], dim=-1)
        Cc.fill_diagonal_(float("nan"))
        names = list(t)
        print("  row-row cos: mean {:.3f} max {:.3f}; top-1 neighbours:".format(Cc.nan_to_num(-2)[Cc.nan_to_num(-2) > -2].mean(), Cc.nan_to_num(-2).max()))
        for i, n in enumerate(names):
            j = Cc.nan_to_num(-2)[i].argmax()
            print(f"    {n} → {names[j]} {Cc[i, j]:.2f}", end="")
        print()

    if "zh" in tabs:
        t = tabs["zh"]
        print("\n## 3. simplified hanzi ↔ Δ1 shinjitai (same meaning, different glyph)")
        D = torch.stack(list(d1.values()))
        dn = list(d1)
        g = torch.Generator().manual_seed(0)
        for s, j in SIMP.items():
            if j not in d1:
                print(f"  {s}↔{j}: {j} not in Δ1")
                continue
            c = cos(t[s], d1[j])
            allc = F.cosine_similarity(t[s][None], D, dim=-1)
            rank = int((allc > c).sum()) + 1
            top = allc.topk(3)
            print(
                f"  {s}↔{j}: cos {c:+.3f}  rank {rank}/{len(dn)} among Δ1 rows; "
                f"random Δ1 row {allc.mean():+.3f}; {s}'s top-3 in Δ1: "
                + " ".join(f"{dn[k]}{allc[k]:.2f}" for k in top.indices)
            )
        for s in CTRL:
            allc = F.cosine_similarity(t[s][None], D, dim=-1)
            top = allc.topk(3)
            print(f"  ctrl {s}: max cos to Δ1 {allc.max():+.3f}  top-3 " + " ".join(f"{dn[k]}{allc[k]:.2f}" for k in top.indices))


if __name__ == "__main__":
    main()
