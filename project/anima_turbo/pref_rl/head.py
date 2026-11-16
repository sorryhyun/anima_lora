"""Preference-head fits and reads (pref_rl/proposal.md § 4, P0 and P1).

    .venv/bin/python project/anima_turbo/pref_rl/head.py gelcrawl
    .venv/bin/python project/anima_turbo/pref_rl/head.py picks --session <name>

A head is linear on a standardised basis, fitted with a pairwise logistic loss between a
positive and a negative of the same group (artist on gelcrawl, round on picks), every group
weighted alike; λ is chosen on inner group folds. The read is the within-group AUROC on
held-out groups.

- ``gelcrawl`` (P0): held-out-artist folds per basis, then a fit on every artist saved as
  ``head_<basis>.npz`` for the picks' zero-shot read.
- ``picks`` (P1): per basis, the gelcrawl head zero-shot, a head fitted on the picks alone
  (leave one round out), and the gelcrawl head refitted on the picks (L2 toward its weights).
  Gate: a sign test over rounds, p < 0.05.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import labels  # noqa: E402
import store  # noqa: E402

LAMS = (0.01, 0.1, 1.0, 10.0, 100.0)
NEG_PER_POS = 8
INNER = 3


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sd = X.std(0)
    return X.mean(0), np.maximum(sd, 0.1 * np.median(sd) + 1e-6)


def pair_diffs(Z, y, g, rng) -> tuple[torch.Tensor, torch.Tensor]:
    """(positive − negative rows, weights): pairs inside a group, each group summing to 1 / groups."""
    pos, neg, wt = [], [], []
    for grp in np.unique(g):
        idx = np.flatnonzero(g == grp)
        p, n = idx[y[idx] == 1], idx[y[idx] == 0]
        if not len(p) or not len(n):
            continue
        k = min(NEG_PER_POS, len(n))
        for i in p:
            pos += [i] * k
            neg += rng.choice(n, k, replace=False).tolist()
        wt += [1.0 / (len(p) * k)] * (len(p) * k)
    D = torch.from_numpy(Z[pos] - Z[neg])
    w = torch.tensor(wt, dtype=torch.float32)
    if torch.cuda.is_available() and D.numel() > 2e7:
        D, w = D.cuda(), w.cuda()
    return D, w / w.sum()


def fit(D: torch.Tensor, wt: torch.Tensor, lam: float, w0=None) -> np.ndarray:
    w0 = (
        torch.zeros(D.shape[1], device=D.device)
        if w0 is None
        else torch.as_tensor(w0, dtype=torch.float32, device=D.device)
    )
    w = w0.clone().requires_grad_()
    opt = torch.optim.LBFGS([w], max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (wt * F.softplus(-(D @ w))).sum() + lam * (w - w0).pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().cpu().numpy()


def group_auroc(s, y, g) -> dict[str, float]:
    out = {}
    for grp in np.unique(g):
        m = g == grp
        if 0 < y[m].sum() < m.sum():
            out[str(grp)] = float(roc_auc_score(y[m], s[m]))
    return out


def contrast_groups(y, g) -> list[str]:
    return list(group_auroc(np.zeros(len(y)), y, g))


def fit_cv(Z, y, g, rng, w0=None) -> tuple[np.ndarray, float]:
    """λ by inner group folds (mean within-group AUROC), then a fit on all of it."""
    groups = np.array(contrast_groups(y, g))
    rng.shuffle(groups)
    score = {lam: [] for lam in LAMS}
    for part in np.array_split(groups, INNER):
        va = np.isin(g, part)
        D, wt = pair_diffs(Z[~va], y[~va], g[~va], rng)
        for lam in LAMS:
            w = fit(D, wt, lam, w0)
            score[lam] += group_auroc(Z[va] @ w, y[va], g[va]).values()
    lam = max(LAMS, key=lambda v: np.mean(score[v]))
    D, wt = pair_diffs(Z, y, g, rng)
    return fit(D, wt, lam, w0), lam


def summarize(aucs: dict[str, float]) -> dict:
    v = np.array(list(aucs.values()))
    wins, losses = int((v > 0.5).sum()), int((v < 0.5).sum())
    p = store._sign_test(wins, losses)
    return {
        "groups": len(v),
        "auroc": float(v.mean()) if len(v) else None,
        "median": float(np.median(v)) if len(v) else None,
        "wins": wins,
        "losses": losses,
        "p_sign": p,
        "pass": bool(p is not None and p < 0.05 and wins > losses),
    }


def _row(name: str, r: dict) -> str:
    if not r["groups"]:
        return f"  {name:<22} —"
    return (
        f"  {name:<22} AUROC {r['auroc']:.3f} (median {r['median']:.3f})  "
        f">0.5 {r['wins']}/{r['groups']}  sign p {r['p_sign']:.3g}"
        f"{'  PASS' if r['pass'] else ''}"
    )


# ── P0 ───────────────────────────────────────────────────────────────────


def artist_folds(g: np.ndarray, n: int) -> list[np.ndarray]:
    names, counts = np.unique(g, return_counts=True)
    folds: list[list[str]] = [[] for _ in range(n)]
    sizes = [0] * n
    for i in np.argsort(-counts):
        f = int(np.argmin(sizes))
        folds[f].append(names[i])
        sizes[f] += counts[i]
    return [np.isin(g, f) for f in folds]


def cmd_gelcrawl(args) -> None:
    items = labels.gelcrawl_items()
    out = {}
    for basis in args.basis or labels.bases_of(labels.GELCRAWL):
        X, y, g = labels.aligned(items, labels.load_feats(labels.GELCRAWL, basis))
        rng = np.random.default_rng(args.seed)
        aucs, lams = {}, []
        for va in artist_folds(g, args.folds):
            mu, sd = standardize(X[~va])
            Z = (X - mu) / sd
            w, lam = fit_cv(Z[~va], y[~va], g[~va], rng)
            aucs.update(group_auroc(Z[va] @ w, y[va], g[va]))
            lams.append(lam)
        mu, sd = standardize(X)
        w, lam = fit_cv((X - mu) / sd, y, g, rng)
        np.savez(
            labels.feat_dir(labels.GELCRAWL) / f"head_{basis}.npz",
            w=w, mu=mu, sd=sd, lam=lam,
        )  # fmt: skip
        out[basis] = {
            **summarize(aucs),
            "images": len(y),
            "selected": int(y.sum()),
            "fold_lams": lams,
            "lam": lam,
            "per_artist": aucs,
        }
        print(f"[{basis}] {len(y)} images, {int(y.sum())} selected, λ {lams} / {lam}")
        print(_row("held-out artists", out[basis]), flush=True)
    path = labels.feat_dir(labels.GELCRAWL).parent / "p0.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"-> {path}")


# ── P1 ───────────────────────────────────────────────────────────────────


def load_head(basis: str) -> dict | None:
    path = labels.feat_dir(labels.GELCRAWL) / f"head_{basis}.npz"
    return dict(np.load(path)) if path.exists() else None


def cmd_picks(args) -> None:
    session = store.Session(store.SESSIONS_ROOT / args.session)
    items = labels.session_items(session)
    out = {}
    for basis in args.basis or labels.bases_of(args.session):
        X, y, g = labels.aligned(items, labels.load_feats(args.session, basis))
        rounds = contrast_groups(y, g)
        print(
            f"[{basis}] {len(np.unique(g))} rounds ({len(rounds)} with a contrast, "
            f"P1 asks {store.P1_ROUNDS}), {len(y)} renders, {int(y.sum())} picked"
        )
        head = load_head(basis)
        Zg = (X - head["mu"]) / head["sd"] if head else None
        reads: dict[str, dict[str, float]] = {}
        if head:
            reads["zero-shot (gelcrawl)"] = group_auroc(Zg @ head["w"], y, g)
        if len(rounds) > INNER + 1:
            rng = np.random.default_rng(args.seed)
            alone, tuned = {}, {}
            for r in rounds:
                te = g == r
                mu, sd = standardize(X[~te])
                Z = (X - mu) / sd
                w, _ = fit_cv(Z[~te], y[~te], g[~te], rng)
                alone[r] = float(roc_auc_score(y[te], Z[te] @ w))
                if head:
                    w, _ = fit_cv(Zg[~te], y[~te], g[~te], rng, w0=head["w"])
                    tuned[r] = float(roc_auc_score(y[te], Zg[te] @ w))
            reads["picks alone"] = alone
            if head:
                reads["gelcrawl, then picks"] = tuned
        else:
            print(f"  fits need more than {INNER + 1} rounds with a contrast")
        out[basis] = {}
        for name, aucs in reads.items():
            out[basis][name] = {**summarize(aucs), "per_round": aucs}
            print(_row(name, out[basis][name]), flush=True)
    path = session.root / "p1.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"-> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("gelcrawl")
    a.add_argument("--folds", type=int, default=5)
    a.set_defaults(fn=cmd_gelcrawl)
    b = sub.add_parser("picks")
    b.add_argument("--session", required=True)
    b.set_defaults(fn=cmd_picks)
    for p in (a, b):
        p.add_argument("--basis", nargs="+")
        p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
