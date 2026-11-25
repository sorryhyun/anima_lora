#!/usr/bin/env python
"""Row geometry — where a G1 run's trained rows sit in the table, before and after (CPU).

A row as the model sees it is its pack row plus the rows' delta (``seed_1008``: jp_v1's
training) plus what G1 added. Reads a run's save (``output/cjk_rl/g1/<run>/rows_<save>.pt``),
the pack (``--pack``, the one G1 renders on) and the checkpoint's own embedding rows
(``llm_adapter.embed``), and prints, lengths in row norms (the mean pack-row norm):

- sizes: the rows' norms, jp_v1's delta and G1's displacement a row, their cosine;
- a row against itself: norm before and after, the angle it turned, the radial share;
- a row against its neighbours: the nearest other row of the table, before and after,
  and the trained rows' mutual cosines;
- the displacements as a set: their singular values, the share of a displacement inside
  the top principal directions of the table's rows, beside jp_v1's delta and a drawn
  direction;
- the mean displacement against directions with a name: the rows' centroid, the
  checkpoint's rows' centroid, its ASCII-letter rows, jp_v1's mean delta, each class.

    .venv/bin/python project/cjk_rl/probes/row_geom.py --run g1long_1125_1335 --save s24
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
G1_OUT = REPO / "output" / "cjk_rl" / "g1"
PACK = "models/vocab_packs/anima_cjk_vocab_pack_punct"
DIT = "models/diffusion_models/anima_merged15.safetensors"
TOPK = (1, 8, 32, 128, 512)


def q(x) -> str:
    import torch

    v = torch.as_tensor(x, dtype=torch.float32)
    a, b, c = (float(v.quantile(p)) for p in (0.1, 0.5, 0.9))
    return f"{b:.3f} ({a:.3f} … {c:.3f})"


def main() -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    from library.anima.weights import load_t5_tokenizer

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--save", default="s24")
    ap.add_argument("--pack", default=PACK)
    ap.add_argument("--dit", default=DIT)
    a = ap.parse_args()
    F = torch.nn.functional
    cos = lambda x, y: float(F.cosine_similarity(x, y, dim=-1))  # noqa: E731

    sv = torch.load(
        G1_OUT / a.run / f"rows_{a.save}.pt", map_location="cpu", weights_only=False
    )
    ids, rs = [int(e) for e in sv["ext_ids"]], float(sv["row_scale"])
    pack = load_file(str(REPO / (a.pack + ".safetensors")))["ext_embed"].float()
    P = pack[ids] / rs
    init, raw = sv["init"].float() / rs, sv["raw"].float() / rs
    E0, E1 = P + init, P + raw
    tr = dict(sv["rows"])  # glyph -> index
    ti = torch.tensor(list(tr.values()))
    D = (raw - init)[ti]  # G1's displacement, (16, dim)
    J = init[ti]  # jp_v1's delta on the same rows
    had = init.norm(dim=1) > 0
    print(
        f"{a.run} {a.save}: table {tuple(E0.shape)}, row_scale {rs:.2f}; jp_v1 moved "
        f"{int(had.sum())} rows; trained here {len(tr)}"
    )

    print("\n-- sizes (row norms; median (10 % … 90 %))")
    print("pack row norm, the table:", q(P.norm(dim=1)))
    print("start row norm, the table:", q(E0.norm(dim=1)))
    print("jp_v1's delta, the rows it moved:", q(init[had].norm(dim=1)))
    print("jp_v1's delta, the trained rows:", q(J.norm(dim=1)))
    print("G1's displacement, the trained rows:", q(D.norm(dim=1)))
    print(
        "cosine of G1's displacement and jp_v1's delta, a row:",
        q(F.cosine_similarity(D, J)),
    )
    print("cosine of jp_v1's delta and the pack row:", q(F.cosine_similarity(J, P[ti])))

    print("\n-- a row against itself")
    n0, n1 = E0[ti].norm(dim=1), E1[ti].norm(dim=1)
    turn = torch.rad2deg(torch.acos(F.cosine_similarity(E0[ti], E1[ti]).clamp(-1, 1)))
    radial = (D * F.normalize(E0[ti], dim=1)).sum(1)
    print("norm before:", q(n0), "after:", q(n1), "change:", q(n1 - n0))
    print("angle turned, degrees:", q(turn))
    print("radial part of the displacement / its length:", q(radial / D.norm(dim=1)))
    pj = torch.rad2deg(torch.acos(F.cosine_similarity(P[ti], E0[ti]).clamp(-1, 1)))
    print("angle jp_v1 turned the same rows from the pack, degrees:", q(pj))

    print("\n-- a row against its neighbours")
    d00 = torch.cdist(E0, E0)
    d00.fill_diagonal_(float("inf"))
    print("nearest other row, the table:", q(d00.min(1).values))
    print("nearest other row, the trained rows before:", q(d00[ti].min(1).values))
    d10 = torch.cdist(E1[ti], E0)
    own = d10[torch.arange(len(ti)), ti]
    d10[torch.arange(len(ti)), ti] = float("inf")
    print("a trained row after, to its own start:", q(own))
    print("a trained row after, to the nearest other start row:", q(d10.min(1).values))
    print(
        "trained rows whose own start is still the nearest:",
        int((own < d10.min(1).values).sum()),
        "of",
        len(ti),
    )
    pd = lambda X: torch.pdist(X)  # noqa: E731
    c = E0.mean(0)
    pc = lambda X: torch.stack(  # noqa: E731
        [
            F.cosine_similarity(X[i] - c, X[j] - c, dim=0)
            for i, j in itertools.combinations(range(len(X)), 2)
        ]
    )
    print(
        "distance between two trained rows, before:",
        q(pd(E0[ti])),
        "after:",
        q(pd(E1[ti])),
    )
    print(
        "cosine of two trained rows about the table's centroid, before:",
        q(pc(E0[ti])),
        "after:",
        q(pc(E1[ti])),
    )
    print(
        "distance between two rows, the table:",
        q(
            pd(
                E0[
                    torch.randperm(len(E0), generator=torch.Generator().manual_seed(0))[
                        :600
                    ]
                ]
            )
        ),
    )

    print("\n-- the displacements as a set")
    s = torch.linalg.svdvals(D)
    e = s**2 / (s**2).sum()
    print(
        "energy by singular direction, the 16 displacements:",
        " ".join(f"{x:.3f}" for x in e[:6]),
        f"… participation {float(1 / (e**2).sum()):.1f} of {len(e)}",
    )
    sj = torch.linalg.svdvals(J)
    ej = sj**2 / (sj**2).sum()
    print(
        "the same, jp_v1's delta on these rows:",
        " ".join(f"{x:.3f}" for x in ej[:6]),
        f"… participation {float(1 / (ej**2).sum()):.1f}",
    )
    _, S, Vh = torch.linalg.svd(E0 - c, full_matrices=False)
    var = S**2 / (S**2).sum()
    g = torch.Generator().manual_seed(0)
    R = torch.randn(D.shape, generator=g)

    def inside(X, k):
        return float(((X @ Vh[:k].T) ** 2).sum() / (X**2).sum())

    print("share inside the table's top-k principal directions:")
    print("  k        " + " ".join(f"{k:>6}" for k in TOPK))
    print(
        "  the table's own variance "
        + " ".join(f"{float(var[:k].sum()):6.3f}" for k in TOPK)
    )
    for name, X in (
        ("G1's displacements", D),
        ("their mean", D.mean(0, keepdim=True)),
        ("jp_v1's delta, these rows", J),
        ("drawn directions", R),
    ):
        print(f"  {name:<26}" + " ".join(f"{inside(X, k):6.3f}" for k in TOPK))

    print("\n-- the mean displacement against named directions")
    m = D.mean(0)
    f = safe_open(str(REPO / a.dit), "pt")
    T5 = f.get_tensor("net.llm_adapter.embed.weight").float() / rs
    tok = load_t5_tokenizer()
    vocab = tok.get_vocab()
    alpha = [
        i
        for p, i in vocab.items()
        if p.lstrip("▁").isascii() and p.lstrip("▁").isalpha() and i < len(T5)
    ]
    print(
        f"checkpoint rows: {tuple(T5.shape)}, norm {q(T5.norm(dim=1))}; ASCII-letter pieces {len(alpha)}, norm {q(T5[alpha].norm(dim=1))}"
    )
    cT, cA = T5.mean(0), T5[alpha].mean(0)
    kind_t = ["hira" if k in sv["cfg"].get("hira", "") else "kanji" for k in tr]
    hi = torch.tensor([k == "hira" for k in kind_t])
    named = {
        "the trained rows' mean start row": E0[ti].mean(0),
        "the table's centroid": c,
        "the table's centroid from the trained rows' mean": c - E0[ti].mean(0),
        "the checkpoint rows' centroid from the trained rows' mean": cT
        - E0[ti].mean(0),
        "the ASCII-letter rows' centroid from the trained rows' mean": cA
        - E0[ti].mean(0),
        "jp_v1's mean delta, the trained rows": J.mean(0),
        "jp_v1's mean delta, every row it moved": init[had].mean(0),
        "the table's first principal direction": Vh[0],
    }
    print(
        f"mean displacement {float(m.norm()):.3f} long; hiragana {float(D[hi].mean(0).norm()):.3f}, kanji {float(D[~hi].mean(0).norm()):.3f}, their cosine {cos(D[hi].mean(0), D[~hi].mean(0)):+.3f}"
    )
    for name, v in named.items():
        print(
            f"  cosine with {name}: {cos(m, v):+.3f}   (that direction is {float(v.norm()):.3f} long)"
        )
    # each row's own displacement against its own direction to the letter rows
    to_a = cA[None] - E0[ti]
    print(
        "cosine of a row's displacement and its direction to the ASCII-letter centroid:",
        q(F.cosine_similarity(D, to_a)),
    )
    print(
        "distance to the ASCII-letter centroid, before:",
        q(to_a.norm(dim=1)),
        "after:",
        q((cA[None] - E1[ti]).norm(dim=1)),
    )
    print(
        "distance to the table's centroid, before:",
        q((E0[ti] - c).norm(dim=1)),
        "after:",
        q((E1[ti] - c).norm(dim=1)),
    )
    print(
        "per row: glyph, class, |jp_v1 delta|, |G1|, angle turned, cos(G1, jp_v1), cos(G1, mean of the others)"
    )
    for n, (k, i) in enumerate(tr.items()):
        others = torch.cat([D[:n], D[n + 1 :]]).mean(0)
        print(
            f"  {k} {kind_t[n]:<5} {float(J[n].norm()):.3f} {float(D[n].norm()):.3f} {float(turn[n]):5.1f} {cos(D[n], J[n]):+.3f} {cos(D[n], others):+.3f}"
        )


if __name__ == "__main__":
    main()
