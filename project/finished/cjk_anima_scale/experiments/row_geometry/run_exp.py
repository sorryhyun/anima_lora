#!/usr/bin/env python
"""row_geometry — the retrain runs' rows in row space (2026-09-28, CPU)

Reads the trained singles of ``retrain_kana`` (174 cold kana rows) and the
C3 kanji arms (``c3_kanji`` 225 / row, ``c3_kanji_450`` 450 / row, 36 cold
kanji) against the seed rows, the raw pack rows and the T5 table. No model,
no render. Row = the trained delta ``raw × row_scale`` (cold: the pack row
is under it; effective = pack + delta). A run's rows are the ext rows that
differ from its ``context`` rows.

1. Pairwise cos / shared (mean) energy / PC1 / cos after centering.
2. The shared direction across runs, vs the seed's, ``u_S`` (the count
   twin's ``u_twin``, cos 0.956 to Stage B's ``u_S``), the pack and T5.
3. The centered structure: kana glyph pairs vs a permutation null.
4. Per-glyph: row geometry vs the C3 450 read (Spearman, 24 glyphs).
5. Participation ratio (effective rank), seed vs trained, same rows.
6. Rotation: Procrustes seed → trained (full space, in-sample vs shuffled
   rows) and inside top-k PC subspaces (fit on half, score the other half).

    run_exp.py --label rg1 [--dry_run]
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata as ud
from pathlib import Path

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, SEED_ROWS_0921, bootstrap, pin_old_seed  # noqa: E402

bootstrap()
pin_old_seed()  # the retrain reads against the old seed's floor of record
from bench._common import make_run_dir, write_result  # noqa: E402

PACK = "models/vocab_packs/anima_cjk_vocab_pack"
RUNS = {
    "kana": OUT / "retrain_kana" / "trained.pt",
    "kanji225": OUT / "experiments" / "c3_kanji" / "trained.pt",
    "kanji450": OUT / "experiments" / "c3_kanji_450" / "trained.pt",
}
P1 = {
    a: OUT / "experiments" / a / "trained.pt" for a in ("p1_mix", "p1_cold", "p1_lone")
}
U_TWIN = OUT / "run0926_count_twin" / "count_dir.pt"
C3_READ = LINE / "experiments/c3_kanji/results/20260928-1145-c3s450/result.json"
INK = LINE / "assets" / "glyph_ink.json"
PERM = 2000
REPS = 200


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def _rows(path):
    import torch

    d = torch.load(path, map_location="cpu", weights_only=False)["delta"]
    return {
        int(e): d["raw"][i].float() * d["row_scale"] for i, e in enumerate(d["ext_ids"])
    }


def _run_rows(path, tok, pack):
    """The run's trained rows (those differing from its context), keyed by glyph."""
    import torch
    from probe.merge_tables import row_texts

    sd = torch.load(path, map_location="cpu", weights_only=False)
    rows, ctx = _rows(path), _rows(sd["args"]["context"])
    changed = [
        e
        for e, v in rows.items()
        if e not in ctx or not torch.allclose(v, ctx[e], atol=1e-2)
    ]
    text = row_texts(tok, pack, changed)
    chars = list(
        dict.fromkeys("".join(v.split(":", 1)[1] for v in sd["args"]["vocabs"]))
    )
    t2id = {text[e]: e for e in changed if e in text}
    missing = [c for c in chars if c not in t2id]
    if missing:
        raise SystemExit(f"{path}: no trained row for {missing}")
    return chars, [t2id[c] for c in chars], rows


def _shared(X):
    import torch

    F = torch.nn.functional
    Xn = F.normalize(X, dim=1)
    off = ~torch.eye(len(X), dtype=bool)
    Xc = F.normalize(X - X.mean(0), dim=1)
    s = torch.linalg.svdvals(X)
    return {
        "norm": X.norm(dim=1).mean().item(),
        "pair_cos": (Xn @ Xn.T)[off].mean().item(),
        "mean_energy": (X.mean(0).norm() ** 2 / (X.norm(dim=1) ** 2).mean()).item(),
        "pc1": (s[0] ** 2 / (s**2).sum()).item(),
        "centered_pair_cos": (Xc @ Xc.T)[off].mean().item(),
    }


def _pr(X, center=True):
    import torch

    X = X.double()
    if center:
        X = X - X.mean(0)
    lam = torch.linalg.svdvals(X) ** 2
    return (lam.sum() ** 2 / (lam**2).sum()).item()


def _spearman(a, b):
    import torch

    def rk(x):
        return torch.argsort(torch.argsort(torch.tensor(x, dtype=torch.float))).float()

    a, b = rk(a), rk(b)
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm())).item()


def _unit(X):
    X = X.double()
    X = X - X.mean(0)
    return X / X.norm()


def _nuc(A, B):
    """‖Aᵀ B‖_* through thin QRs (rows ≪ dim)."""
    import torch

    _, Ta = torch.linalg.qr(A.T)  # Aᵀ = Qa Ta, Bᵀ = Qb Tb → Aᵀ B = Qa (Ta Tbᵀ) Qbᵀ
    _, Tb = torch.linalg.qr(B.T)
    return torch.linalg.svdvals(Ta @ Tb.T).sum().item()


def _rotation(A0, B0, g):
    """Procrustes A → B: full space in-sample (vs shuffled rows) and, with
    ≥ 100 rows, inside top-k subspaces fitted on half the rows."""
    import torch

    A, B = _unit(A0), _unit(B0)
    n = len(A)
    out = {
        "rows": n,
        "full_in_sample": 2 * _nuc(A, B) - 1,
        "full_shuffled": sum(
            2 * _nuc(A[torch.randperm(n, generator=g)], B) - 1 for _ in range(REPS)
        )
        / REPS,
    }
    if n < 100:
        return out
    A0, B0 = A0.double(), B0.double()
    for k in (8, 16, 32):
        acc = {"rot": 0.0, "shuffled": 0.0, "no_rot": 0.0, "ceiling": 0.0}
        for _ in range(REPS):
            p = torch.randperm(n, generator=g)
            tr, te = p[: n // 2], p[n // 2 :]
            ma, mb = A0[tr].mean(0), B0[tr].mean(0)
            Va = torch.linalg.svd(A0[tr] - ma, full_matrices=False)[2][:k].T
            Vb = torch.linalg.svd(B0[tr] - mb, full_matrices=False)[2][:k].T
            a_tr, b_tr = (A0[tr] - ma) @ Va, (B0[tr] - mb) @ Vb
            U, s, Wt = torch.linalg.svd(a_tr.T @ b_tr)
            R, sc = U @ Wt, s.sum() / a_tr.norm() ** 2
            bte = B0[te] - mb
            tot = (bte.norm() ** 2).item()

            def score(ain):
                pred = (((ain - ma) @ Va) @ R * sc) @ Vb.T
                return 1 - ((pred - bte).norm() ** 2).item() / tot

            d = A0[te] - ma
            c = (d * bte).sum() / d.norm() ** 2
            acc["rot"] += score(A0[te])
            acc["shuffled"] += score(A0[te][torch.randperm(len(te), generator=g)])
            acc["no_rot"] += 1 - ((c * d - bte).norm() ** 2).item() / tot
            acc["ceiling"] += ((bte @ Vb).norm() ** 2).item() / tot
        out[f"k{k}_held_out"] = {key: v / REPS for key, v in acc.items()}
    return out


def main():
    args = _args()
    if args.dry_run:
        for name, p in {**RUNS, **P1, "seed": SEED_ROWS_0921, "u_twin": U_TWIN}.items():
            print(f"{name:<9} {p} {'ok' if Path(p).exists() else 'MISSING'}")
        print(f"read {C3_READ} {'ok' if C3_READ.exists() else 'MISSING'}")
        return

    import torch
    from safetensors import safe_open

    from library.anima.vocab_pack import load_vocab_pack
    from library.env import default_checkpoints
    from library.inference.text import ensure_text_strategies

    F = torch.nn.functional
    g = torch.Generator().manual_seed(0)
    ck = default_checkpoints()
    tok, _ = ensure_text_strategies(ck.text_encoder, vocab_pack=PACK)
    pack = load_vocab_pack(PACK)
    table = pack.table.float()
    with safe_open(ck.dit, "pt") as f:
        t5 = f.get_tensor("net.llm_adapter.embed.weight").float()
    seed = _rows(SEED_ROWS_0921)
    u_twin = torch.load(U_TWIN, map_location="cpu", weights_only=False)[
        "u_twin"
    ].float()

    def cos(a, b):
        return F.cosine_similarity(a, b, dim=0).item()

    R, M = {}, {"shared": {}, "means": {}, "pr": {}, "rotation": {}}
    means = {}
    for name, path in RUNS.items():
        chars, ids, rows = _run_rows(path, tok, pack)
        has = torch.tensor([i in seed for i in ids])
        D = torch.stack([rows[i] for i in ids])
        Sd = torch.stack([seed.get(i, torch.zeros(D.shape[1])) for i in ids])
        P = table[ids]
        R[name] = dict(chars=chars, D=D, S=Sd, P=P, has=has)
        sh = {
            "rows": len(D),
            "seed_rows": int(has.sum()),
            "trained_delta": _shared(D),
            "seed_delta": _shared(Sd[has]),
            "pack": _shared(P),
            "trained_eff": _shared(P + D),
            "seed_eff": _shared((P + Sd)[has]),
            "cos_delta_own_pack": F.cosine_similarity(D, P).mean().item(),
            "cos_delta_own_seed": F.cosine_similarity(D[has], Sd[has]).mean().item(),
            "cos_delta_own_seed_centered": F.cosine_similarity(
                (D - D.mean(0))[has], (Sd - Sd[has].mean(0))[has]
            )
            .mean()
            .item(),
        }
        M["shared"][name] = sh
        means[name], means[name + "_seed"] = D.mean(0), Sd[has].mean(0)
        # Stage B's u_S convention: Δ vs seed, own seed row's component removed
        base = (P + Sd)[has]
        dv = (D - Sd)[has]
        bh = F.normalize(base, dim=1)
        dv = dv - (dv * bh).sum(1, keepdim=True) * bh
        md, mp = D.mean(0), P.mean(0)
        ph = F.normalize(mp, dim=0)
        M["means"][name] = {
            "vs_u_twin": cos(md, u_twin),
            "seed_mean_vs_u_twin": cos(means[name + "_seed"], u_twin),
            "delta_vs_seed_tangential_mean_vs_u_twin": cos(dv.mean(0), u_twin),
            "vs_own_pack_mean": cos(md, mp),
            "vs_whole_pack_mean": cos(md, table.mean(0)),
            "vs_t5_mean": cos(md, t5.mean(0)),
            "energy_along_pack_mean": ((md @ ph) ** 2 / md.norm() ** 2).item(),
            "eff_mean_vs_pack_mean": cos((P + D).mean(0), mp),
            "pack_mean_norm": mp.norm().item(),
            "eff_mean_norm": (P + D).mean(0).norm().item(),
            "added_vs_seed_mean": cos(
                md - means[name + "_seed"], means[name + "_seed"]
            ),
        }
        M["pr"][name] = {
            tag: {"centered": _pr(X), "uncentered": _pr(X, False)}
            for tag, X in [
                ("trained_delta", D[has]),
                ("seed_delta", Sd[has]),
                ("pack", P[has]),
                ("trained_eff", (P + D)[has]),
                ("seed_eff", (P + Sd)[has]),
                ("change", (D - Sd)[has]),
            ]
        }
        M["rotation"][name] = {
            "seed_delta_to_trained_delta": _rotation(Sd[has], D[has], g),
            "seed_eff_to_trained_eff": _rotation((P + Sd)[has], (P + D)[has], g),
            "pack_to_trained_eff": _rotation(P, P + D, g),
        }
    a, b = R["kanji225"]["D"], R["kanji450"]["D"]
    M["rotation"]["kanji225_to_450"] = _rotation(a, b, g)
    M["kanji_225_vs_450_same_row"] = {
        "cos": F.cosine_similarity(a, b).mean().item(),
        "centered_cos": F.cosine_similarity(a - a.mean(0), b - b.mean(0)).mean().item(),
        "norm_ratio": (b.norm(dim=1) / a.norm(dim=1)).mean().item(),
    }
    ks = list(means)
    M["mean_dir_cos"] = {
        f"{x}|{y}": cos(means[x], means[y])
        for i, x in enumerate(ks)
        for y in ks[i + 1 :]
    }
    M["pr"]["kana_subsets"] = {
        n: {
            tag: sum(
                _pr(X[torch.randperm(len(X), generator=g)[:n]]) for _ in range(REPS)
            )
            / REPS
            for tag, X in [
                ("trained_delta", R["kana"]["D"][R["kana"]["has"]]),
                ("seed_delta", R["kana"]["S"][R["kana"]["has"]]),
                ("pack", R["kana"]["P"][R["kana"]["has"]]),
            ]
        }
        for n in (20, 36)
    }

    # p1 arms on their 36 donors vs retrain_kana on the same rows
    K = R["kana"]
    kid = {c: i for i, c in enumerate(K["chars"])}
    _, kana_ids, _ = _run_rows(RUNS["kana"], tok, pack)
    id2k = {e: i for i, e in enumerate(kana_ids)}
    M["p1_same_rows"] = {}
    for arm, path in P1.items():
        _, ids, rows = _run_rows(path, tok, pack)
        X = torch.stack([rows[i] for i in ids])
        Y = K["D"][[id2k[i] for i in ids]]
        M["p1_same_rows"][arm] = {
            "rows": len(ids),
            "mean_energy": _shared(X)["mean_energy"],
            "retrain_kana_mean_energy": _shared(Y)["mean_energy"],
            "norm": X.norm(dim=1).mean().item(),
            "retrain_kana_norm": Y.norm(dim=1).mean().item(),
            "mean_dir_vs_retrain_kana": cos(X.mean(0), Y.mean(0)),
            "mean_dir_vs_kanji450": cos(X.mean(0), means["kanji450"]),
            "same_row_cos": F.cosine_similarity(X, Y).mean().item(),
        }

    # centered kana pair structure vs a permutation null
    ch = K["chars"]

    def centered_cos(X):
        Xc = F.normalize(X - X.mean(0), dim=1)
        return Xc @ Xc.T

    CD, CP = centered_cos(K["D"]), centered_cos(K["P"])
    hs = K["has"]
    s_chars = [c for c, h in zip(ch, hs.tolist()) if h]
    CS = centered_cos(K["S"][hs])
    sid = {c: i for i, c in enumerate(s_chars)}

    def base_of(c):
        d = ud.normalize("NFD", c)
        return d[0] if len(d) > 1 else None

    pairs = {
        "hira_kata": [(chr(c), chr(c + 0x60)) for c in range(0x3041, 0x3097)],
        "dakuten": [(base_of(c), c) for c in ch if base_of(c) and base_of(c) != c],
        "small_full": [
            (c, chr(ord(c) + 1)) for c in "ぁぃぅぇぉっゃゅょァィゥェォッャュョ"
        ],
    }
    M["kana_pairs"] = {"all_pairs": CD[~torch.eye(len(ch), dtype=bool)].mean().item()}
    for nm, prs in pairs.items():
        ij = [(kid[x], kid[y]) for x, y in prs if x in kid and y in kid]
        v = torch.tensor([CD[i, j].item() for i, j in ij])
        null = torch.stack(
            [
                CD[
                    torch.randint(len(ch), (len(ij),), generator=g),
                    torch.randint(len(ch), (len(ij),), generator=g),
                ].mean()
                for _ in range(PERM)
            ]
        )
        sv = [CS[sid[x], sid[y]].item() for x, y in prs if x in sid and y in sid]
        M["kana_pairs"][nm] = {
            "n": len(ij),
            "trained": v.mean().item(),
            "perm_p": (null >= v.mean()).float().mean().item(),
            "pack": sum(CP[i, j].item() for i, j in ij) / len(ij),
            "seed": sum(sv) / len(sv),
        }
    top = CD.clone().fill_diagonal_(-1)
    flat = torch.topk(top.flatten(), 30).indices.tolist()
    seen, tops = set(), []
    for t in flat:
        i, j = divmod(t, len(ch))
        if (j, i) not in seen:
            seen.add((i, j))
            tops.append([ch[i] + ch[j], round(CD[i, j].item(), 2)])
    M["kana_pairs"]["top"] = tops[:15]

    # per-glyph: row geometry vs the C3 450 read
    reads = json.loads(C3_READ.read_text())["metrics"]["reads"]
    ink = json.loads(INK.read_text())
    M["per_glyph"] = {}
    for name, key in (("kanji450", "c3_kanji_450"), ("kanji225", "c3_kanji")):
        r, pk = R[name], reads[key]["singles"]["per_key"]
        g_ = [c for c in r["chars"] if f"{c}|en" in pk]
        ii = [r["chars"].index(c) for c in g_]
        D = r["D"]
        mu = F.normalize(D.mean(0), dim=0)
        a_ = D @ mu
        feats = {
            "shared_share": (a_**2 / D.norm(dim=1) ** 2)[ii].tolist(),
            "norm": D.norm(dim=1)[ii].tolist(),
            "residual_norm": (D - a_[:, None] * mu).norm(dim=1)[ii].tolist(),
            "ink": [ink[c] for c in g_],
        }
        outs = {
            m: [pk[f"{c}|en"][m] for c in g_]
            for m in ("official", "contained", "repeat", "kana")
        }
        M["per_glyph"][name] = {
            "glyphs": len(g_),
            **{
                f: {m: _spearman(x, y) for m, y in outs.items()}
                for f, x in feats.items()
            },
        }

    run_dir = make_run_dir(
        "row_geometry", args.label, root=Path(__file__).parent / "results"
    )
    write_result(run_dir, script=__file__, args=args, metrics=M, label=args.label)
    print(json.dumps(M, ensure_ascii=False, indent=1))
    print(f"→ {run_dir}")


if __name__ == "__main__":
    main()
