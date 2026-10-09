#!/usr/bin/env python
"""stick_scale geometry — the cold kana rows in row space, the reads that led to the stick (2026-10-03, CPU)

The 166 kana rows every cold kana arm trains, on four row sets: the 0921
seed (``SEED_ROWS_0921``, the context they train over), ``retrain_kana`` (=
the 0930 seed's kana rows, identical), ``reseed_recap_cold_kana_hp`` and
``reseed_anchor_cold_kana_anchor``. Row = the delta ``raw × row_scale``
(effective = pack + delta); a run's rows = the ext rows that differ from the
0921 seed. Same-row reads use the 162 rows the 0921 seed has.

1. Same-row cos (raw / centered / mean direction) to the 0921 seed and arm
   to arm; ``row_geometry``'s shared-structure stats.
2. Per row: exposure (items, loss share = Σ 1 / glyphs of the item) and the
   items' mean σ band mid in each arm's data, against the row reads.
3. Against the T5 table (``llm_adapter.embed``): norm, nearest-T5 cos, the
   mean delta on T5's principal directions.
4. Stick (|mean|) and spikes (|row − mean|) per set; the 0921 seed's long
   spikes against the cold rows' cos to it.

    (CPU; run_exp.py is the render read)
    .venv/bin/python project/cjk_anima_scale/experiments/stick_scale/geometry.py --label geo
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, SEED_ROWS_0921, bootstrap, load_experiment  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

PACK = "models/vocab_packs/anima_cjk_vocab_pack"
ARMS = {  # name → (rows, the data dir it trained on)
    "retrain_kana": (OUT / "retrain_kana", OUT / "retrain_kana" / "data"),
    "recap_hp": (
        OUT / "experiments" / "reseed_recap_cold_kana_hp",
        OUT / "run1003_reseed_recap" / "data_recap_hp",
    ),
    "anchor": (
        OUT / "experiments" / "reseed_anchor_cold_kana_anchor",
        OUT / "run1003_reseed_anchor" / "data_recap_hp",
    ),
}
SEED_0930 = OUT / "seed_retrain_0930" / "trained.pt"
LONG_SPIKES = 15


def _rk(x):
    import torch

    return torch.argsort(torch.argsort(torch.as_tensor(x, dtype=torch.float))).float()


def spearman(a, b) -> float:
    a, b = _rk(a), _rk(b)
    a, b = a - a.mean(), b - b.mean()
    return round(float(a @ b / (a.norm() * b.norm())), 3)


def exposure(data: Path) -> dict:
    """glyph → (items, loss share, mean σ band mid)."""
    n, share, sig = collections.Counter(), collections.Counter(), collections.Counter()
    for line in open(data / "train.jsonl", encoding="utf-8"):
        r = json.loads(line)
        gl = [c for c in r["text"] if not c.isspace()]
        mid = sum(r["band"]) / 2
        for c in gl:
            n[c] += 1
            share[c] += 1 / len(gl)
            sig[c] += mid
    return {c: (n[c], share[c], sig[c] / n[c]) for c in n}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--label", required=True)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    paths = (
        {"seed0921": SEED_ROWS_0921, "seed0930": SEED_0930}
        | {k: v[0] / "trained.pt" for k, v in ARMS.items()}
        | {f"{k}_data": v[1] / "train.jsonl" for k, v in ARMS.items()}
    )
    if args.dry_run:
        for k, v in paths.items():
            print(f"{k:<14} {v} {'ok' if Path(v).exists() else 'MISSING'}")
        return

    import torch
    from safetensors import safe_open

    from library.anima.vocab_pack import load_vocab_pack
    from library.env import default_checkpoints
    from library.inference.text import ensure_text_strategies
    from probe.merge_tables import row_texts

    F = torch.nn.functional
    RG = load_experiment("row_geometry")
    ck = default_checkpoints()
    tok, _ = ensure_text_strategies(ck.text_encoder, vocab_pack=PACK)
    pack = load_vocab_pack(PACK)
    table = pack.table.float()
    with safe_open(ck.dit, "pt") as f:
        t5 = f.get_tensor("net.llm_adapter.embed.weight").float()
    seed = RG._rows(SEED_ROWS_0921)

    def trained(path: Path) -> dict:
        """glyph → ext id of the rows that differ from the 0921 seed."""
        rows = RG._rows(path)
        moved = [
            e
            for e, v in rows.items()
            if e not in seed or not torch.allclose(v, seed[e], atol=1e-2)
        ]
        text = row_texts(tok, pack, moved)
        return {text[e]: e for e in moved if e in text and len(text[e]) == 1}

    rows = {k: RG._rows(v[0] / "trained.pt") for k, v in ARMS.items()}
    c2id = {k: trained(v[0] / "trained.pt") for k, v in ARMS.items()}
    common = sorted(
        c
        for c in set.intersection(*map(set, c2id.values()))
        if 0x3040 <= ord(c) < 0x3100
    )
    ids = [c2id["recap_hp"][c] for c in common]
    assert all(c2id[k][c] == c2id["recap_hp"][c] for k in ARMS for c in common)
    has = torch.tensor([i in seed for i in ids])
    chars = [c for c, h in zip(common, has.tolist()) if h]
    P = table[ids][has]
    S = torch.stack([seed[i] for i, h in zip(ids, has.tolist()) if h])
    D = {
        k: torch.stack([rows[k][i] for i, h in zip(ids, has.tolist()) if h])
        for k in ARMS
    }
    s30 = RG._rows(SEED_0930)
    M: dict = {
        "rows": len(common),
        "with_0921_row": len(chars),
        "seed0930_kana_eq_retrain_kana": round(
            float(
                F.cosine_similarity(
                    torch.stack([s30[i] for i in ids]),
                    torch.stack([rows["retrain_kana"][i] for i in ids]),
                ).mean()
            ),
            6,
        ),
    }

    def ccos(X, Y):
        return float(F.cosine_similarity(X - X.mean(0), Y - Y.mean(0)).mean())

    # 1. same-row cos + shared structure
    M["shared"] = {
        k: RG._shared(X) | {"pr_centered": RG._pr(X), "pr_uncentered": RG._pr(X, False)}
        for k, X in {"seed0921": S, **D, "pack": P}.items()
    }
    M["vs_seed0921"] = {
        k: {
            "cos": float(F.cosine_similarity(X, S).mean()),
            "cos_centered": ccos(X, S),
            "mean_dir": float(F.cosine_similarity(X.mean(0), S.mean(0), dim=0)),
            "cos_own_pack": float(F.cosine_similarity(X, P).mean()),
            "norm_ratio": float((X.norm(dim=1) / S.norm(dim=1)).mean()),
        }
        for k, X in D.items()
    }
    names = list(D)
    M["arm_vs_arm"] = {
        f"{a}|{b}": {
            "cos": float(F.cosine_similarity(D[a], D[b]).mean()),
            "cos_centered": ccos(D[a], D[b]),
            "mean_dir": float(F.cosine_similarity(D[a].mean(0), D[b].mean(0), dim=0)),
        }
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }

    # 2. exposure / σ against the rows
    M["exposure"] = {}
    for k, (_, data) in ARMS.items():
        ex = exposure(data)
        cnt = [ex[c][0] for c in chars]
        sh = [ex[c][1] for c in chars]
        sg = [ex[c][2] for c in chars]
        X = D[k]
        cos_s = F.cosine_similarity(X, S).tolist()
        u = F.normalize(X.mean(0), dim=0)
        along = ((X @ u) ** 2 / (X**2).sum(1)).tolist()
        e = {
            "items": [min(cnt), max(cnt)],
            "share": [round(min(sh), 1), round(max(sh), 1)],
            "sigma_mid": [round(min(sg), 3), round(max(sg), 3)],
            "rho_share_cos_seed0921": spearman(sh, cos_s),
            "rho_sigma_cos_seed0921": spearman(sg, cos_s),
            "rho_share_norm": spearman(sh, X.norm(dim=1).tolist()),
            "rho_sigma_norm": spearman(sg, X.norm(dim=1).tolist()),
            "rho_share_along_mean": spearman(sh, along),
            "rho_sigma_along_mean": spearman(sg, along),
        }
        if k != "retrain_kana":
            ck_ = F.cosine_similarity(X, D["retrain_kana"]).tolist()
            e["rho_share_cos_retrain_kana"] = spearman(sh, ck_)
            e["rho_sigma_cos_retrain_kana"] = spearman(sg, ck_)
        M["exposure"][k] = e
    M["rho_cos_seed0921_recap_vs_retrain"] = spearman(
        F.cosine_similarity(D["recap_hp"], S).tolist(),
        F.cosine_similarity(D["retrain_kana"], S).tolist(),
    )

    # 3. against the T5 table
    g = torch.Generator().manual_seed(0)
    Tn = F.normalize(t5, dim=1)
    nn_i = torch.randperm(len(t5), generator=g)[:2000]
    c = Tn[nn_i] @ Tn.T
    c[torch.arange(2000), nn_i] = -1
    Tc = t5 - t5.mean(0)
    V = torch.linalg.svd(
        Tc[torch.randperm(len(Tc), generator=g)[:8000]], full_matrices=False
    )[2]
    M["t5"] = {
        "norm": [float(t5.norm(dim=1).mean()), float(t5.norm(dim=1).std())],
        "nearest_t5_cos_of_t5": float(c.max(1).values.median()),
        "random162_mean_energy": sum(
            RG._shared(t5[torch.randperm(len(t5), generator=g)[:162]])["mean_energy"]
            for _ in range(20)
        )
        / 20,
    }
    for k, X in {"pack": P, "seed0921": P + S, **{n: P + D[n] for n in D}}.items():
        M["t5"][f"{k}_eff"] = {
            "norm": [float(X.norm(dim=1).mean()), float(X.norm(dim=1).std())],
            "nearest_t5_cos": float(
                (F.normalize(X, dim=1) @ Tn.T).max(1).values.median()
            ),
        }
    for k in D:
        m = F.normalize(D[k].mean(0), dim=0)
        dc = D[k] - D[k].mean(0)
        M["t5"][f"{k}_mean_delta"] = {
            "energy_top10_pc": float(((V[:10] @ m) ** 2).sum()),
            "cos_t5_mean": float(F.cosine_similarity(m, t5.mean(0), dim=0)),
            "centered_energy_top50_pc": float(
                ((dc @ V[:50].T) ** 2).sum() / (dc**2).sum()
            ),
        }

    # 4. stick and spikes
    def burr(X):
        m = X.mean(0)
        sp = (X - m).norm(dim=1)
        return {
            "stick": float(m.norm()),
            "spike": [float(sp.mean()), float(sp.std())],
            "spike_cv": float(sp.std() / sp.mean()),
            "spike_minmax": [float(sp.min()), float(sp.max())],
        }

    M["burr_delta"] = {k: burr(X) for k, X in {"seed0921": S, **D}.items()}
    M["burr_eff"] = {k: burr(P + X) for k, X in {"seed0921": S, **D}.items()} | {
        "pack": burr(P),
        "t5_random162": burr(t5[torch.randperm(len(t5), generator=g)[:162]]),
    }
    ss = (S - S.mean(0)).norm(dim=1)
    top = torch.argsort(ss)[-LONG_SPIKES:]
    rest = torch.argsort(ss)[:-LONG_SPIKES]
    M["seed0921_long_spikes"] = {
        "glyphs": "".join(chars[i] for i in top.flip(0).tolist()),
        **{
            k: {
                "rho_spike_cos": spearman(
                    ss.tolist(), F.cosine_similarity(X, S).tolist()
                ),
                "cos_long": float(F.cosine_similarity(X[top], S[top]).mean()),
                "cos_rest": float(F.cosine_similarity(X[rest], S[rest]).mean()),
            }
            for k, X in D.items()
        },
    }
    print(json.dumps(M, ensure_ascii=False, indent=1), flush=True)
    run_dir = make_run_dir(
        "stick_scale",
        label=args.label,
        root=LINE / "experiments" / "stick_scale" / "results",
    )
    write_result(run_dir, script=__file__, args=args, label=args.label, metrics=M)
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
