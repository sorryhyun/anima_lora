#!/usr/bin/env python
"""wake_geometry — is there compositional structure in trained ext-row deltas?

Kanji have a ground-truth composition (IDS): 明 = 日 + 月. For a rows arm
trained on atoms and composites, ask whether Δ_composite lies nearer the sum of
its atoms' deltas than a random sum of the same size — with and without the
shared component (mean delta, the "one big glyph" layout mode) projected out.
CPU only; needs the arm's ``trained.pt`` and the tokenizer for the char → row map.

    .venv/bin/python project/cjk_renderable_anima/src/bench/wake_geometry.py \
        --arm_dir output/wake_probe/rows_k24_band \
        --pairs 明=日+月,林=木+木,森=木+木+木,休=人+木,好=女+子,男=田+力,岩=山+石,品=口+口+口,加=力+口,晶=日+日+日,相=木+目,困=口+木
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

TPL = 'manga, speech bubble, japanese text. Japanese text reads as "{}".'


def char_rows(chars):
    """char → ext row id (the T5-side id − table size) via the shipped pack."""
    from library.anima.ext_vocab import T5_TABLE_SIZE
    from library.env import default_checkpoints
    from library.inference.text import ensure_text_strategies

    tok, _ = ensure_text_strategies(default_checkpoints().text_encoder, vocab_pack=None)
    out = {}
    for c in chars:
        t5 = tok.tokenize(TPL.format(c))[2].flatten().tolist()
        ids = [int(v) - T5_TABLE_SIZE for v in t5 if v >= T5_TABLE_SIZE]
        assert len(ids) == 1, (c, ids)
        out[c] = ids[0]
    return out


def main():
    import torch

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm_dir", type=Path, required=True)
    p.add_argument("--pairs", required=True, help="comp=atom+atom,… (IDS pairs)")
    p.add_argument("--n_rand", type=int, default=2000)
    p.add_argument(
        "--table",
        default="raw",
        choices=["raw", "free", "g"],
        help="which table to read: raw = the shipped delta (g + f on the hybrid), "
        "free = the per-row residual f alone (Run 1d+), g = raw − free",
    )
    a = p.parse_args()
    sd = torch.load(a.arm_dir / "trained.pt", map_location="cpu")
    raw = sd["delta"]["raw"].float()  # rows × dim, in row-norm units
    if a.table != "raw":
        assert "free" in sd, f"{a.arm_dir} has no free residual (not a hybrid arm)"
        raw = sd["free"].float() if a.table == "free" else raw - sd["free"].float()
    ext_ids = sd["delta"]["ext_ids"]
    pairs = []
    for item in a.pairs.split(","):
        comp, atoms = item.split("=")
        pairs.append((comp, atoms.split("+")))
    chars = sorted({c for comp, ats in pairs for c in [comp, *ats]})
    rows = char_rows(chars)
    missing = [c for c in chars if rows[c] not in ext_ids]
    assert not missing, f"rows not in the arm: {missing}"
    D = {c: raw[ext_ids.index(rows[c])] for c in chars}
    atoms = sorted({c for _, ats in pairs for c in ats})
    comps = [c for c, _ in pairs]
    A = torch.stack([D[c] for c in atoms])
    C = torch.stack([D[c] for c in comps])
    allv = torch.cat([A, C])
    mean = allv.mean(0)
    cos = torch.nn.functional.cosine_similarity
    print(
        f"rows {len(chars)} (atoms {len(atoms)}, composites {len(comps)}), dim {raw.shape[1]}"
    )
    print(
        f"norms: atoms {A.norm(dim=1).mean():.2f} composites {C.norm(dim=1).mean():.2f} "
        f"(row-norm units); cos to mean: atoms {cos(A, mean[None]).mean():.2f} "
        f"composites {cos(C, mean[None]).mean():.2f}"
    )
    off = allv - allv.mean(0)
    pc = cos(off[:, None], off[None, :], dim=-1)
    n = len(allv)
    print(
        f"pairwise cos among deltas: raw {cos(allv[:, None], allv[None, :], dim=-1)[~torch.eye(n, dtype=bool)].mean():.3f}, "
        f"mean-removed {pc[~torch.eye(n, dtype=bool)].mean():.3f}"
    )
    rng = random.Random(0)
    print()
    print(
        "| comp | atoms | cos(Δc, ΣΔatoms) raw | random-sum pct | mean-removed | pct | nearest row (raw) |"
    )
    print("|---|---|---|---|---|---|---|")
    for comp, ats in pairs:
        k = len(ats)
        for label, V in (("raw", D), ("mr", {c: D[c] - mean for c in chars})):
            target = V[comp]
            s = sum(V[c] for c in ats)
            c_true = float(cos(target, s, dim=0))
            others = [c for c in chars if c != comp]
            rand = []
            for _ in range(a.n_rand):
                pick = [rng.choice(others) for _ in range(k)]
                rand.append(float(cos(target, sum(V[c] for c in pick), dim=0)))
            pct = sum(r < c_true for r in rand) / len(rand) * 100
            if label == "raw":
                raw_c, raw_p = c_true, pct
            else:
                mr_c, mr_p = c_true, pct
        sims = {c: float(cos(D[comp], D[c], dim=0)) for c in chars if c != comp}
        near = max(sims, key=sims.get)
        print(
            f"| {comp} | {'+'.join(ats)} | {raw_c:.3f} | {raw_p:.0f} | {mr_c:.3f} | {mr_p:.0f} | {near} {sims[near]:.2f} |"
        )
    print()
    print(
        "pct = percentile of the true cosine among sums of k random rows (50 = chance, ≥ 95 = structure)."
    )


if __name__ == "__main__":
    main()
