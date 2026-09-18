#!/usr/bin/env python
"""table_geometry — the free geometry reads on rows-arm tables (plan_synth4
R4.0, plan_synth3 K0a). No GPU: ``trained.pt`` + the shipped pack.

Per table (rows in absolute units, ``raw × row_scale``): the shared direction
m̂ (mean row) and its share of row energy, the spectrum's participation ratio,
the pairwise row-cos distribution, shape-neighbour pairs (dakuten family
が·ぎ·ぐ·げ·ご, hiragana ↔ katakana が/ガ) against unrelated pairs, and each
row's cos to the pack's pretrained row for the same piece. Across tables: cos
between shared directions (the K0 merge price) and, on the ext ids two tables
share, the mean per-row cos.

    .venv/bin/python project/cjk_renderable_anima/src/probe/table_geometry.py \
        --tables src53k=rows_synth_full_fm10k_full_s53k_qoff,d1=rows_synth_d1_d1_s53k

Prints markdown tables; ``--json <file>`` also dumps the numbers.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import OUT  # noqa: E402

DEFAULT_TABLES = ",".join(
    [
        "src53k=rows_synth_full_fm10k_full_s53k_qoff",
        "punct=rows_synth_punct_only_punct_only_s3k",
        "d1=rows_synth_d1_d1_s53k",
        "pair0_s3000=rows_synth_pair_d0_pair0_s3000",
        "pair0_s750=rows_synth_pair_d0_pair0_s750_lr2e-3",
        "pairEN_s1500=rows_synth_pair_d0_pairEN_s1500_lr2e-3",
        "pairEN_s750=rows_synth_pair_d0_pairEN_s750_lr2e-3",
        "rb62f=rows_synth_pair_d0_pairEN_rb62f_lr2e-3",
        "w8=rows_synth_pair_d0_pairEN_rb62f_lr2e-3_w8",
    ]
)

# shape neighbours: the dakuten family and the hiragana ↔ katakana pair
DAKUTEN = {
    "が": "か", "ぎ": "き", "ぐ": "く", "げ": "け", "ご": "こ",
    "ガ": "カ", "ギ": "キ", "グ": "ク", "ゲ": "ケ", "ゴ": "コ",
    "ざ": "さ", "じ": "し", "ず": "す", "ぜ": "せ", "ぞ": "そ",
    "だ": "た", "ぢ": "ち", "づ": "つ", "で": "て", "ど": "と",
    "ば": "は", "び": "ひ", "ぶ": "ふ", "べ": "へ", "ぼ": "ほ",
}  # fmt: skip
HIRA_ROW = ("がぎぐげご", "ざじずぜぞ", "だぢづでど", "ばびぶべぼ", "かきくけこ")
KATA_ROW = ("ガギグゲゴ", "ザジズゼゾ", "ダヂヅデド", "バビブベボ", "カキクケコ")


def load(name: str):
    path = OUT / name / "trained.pt" if not name.endswith(".pt") else Path(name)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    d = sd["delta"]
    ids = [int(e) for e in d["ext_ids"]]
    rs = float(d.get("row_scale", 1.0))
    rows = d["raw"].float() * rs
    if "c_flat" in sd:  # S0 tables: the flat switch rides on every flat item
        rows = rows + sd["c_flat"].float() * rs
    keep = rows.norm(dim=1) > 1e-6  # untouched rows carry no direction
    return ids, rows, keep, rs


def neighbour_pairs(texts: dict) -> tuple[list, list]:
    """(shape-neighbour id pairs, same-family same-script control pairs)."""
    by_text = {t: e for e, t in texts.items()}
    near, ctrl = [], []
    for row_h, row_k in zip(HIRA_ROW, KATA_ROW):
        for h, k in zip(row_h, row_k):
            if h in by_text and k in by_text:
                near.append((by_text[h], by_text[k]))  # が ↔ ガ
        for row in (row_h, row_k):
            present = [c for c in row if c in by_text]
            for a, b in itertools.combinations(present, 2):
                ctrl.append((by_text[a], by_text[b]))  # が ↔ ぎ: same row, dakuten shared
    for v, base in DAKUTEN.items():
        if v in by_text and base in by_text:
            near.append((by_text[v], by_text[base]))  # が ↔ か
    return near, ctrl


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def per_table(name, ids, rows, keep, rs, pack, texts):
    R = rows[keep]
    n = int(keep.sum())
    m = R.mean(0)
    mh = m / m.norm()
    energy = float((R.norm(dim=1) ** 2).sum())
    shared = float(((R @ mh) ** 2).sum()) / energy
    s = torch.linalg.svdvals(R)
    lam = s**2
    pr = float(lam.sum() ** 2 / (lam**2).sum())
    Rn = R / R.norm(dim=1, keepdim=True)
    C = Rn @ Rn.T
    iu = torch.triu_indices(n, n, offset=1)
    pc = C[iu[0], iu[1]]
    # residuals ⟂ m̂ — the per-row part after the shared direction
    Res = R - (R @ mh)[:, None] * mh[None]
    Resn = Res / Res.norm(dim=1, keepdim=True).clamp_min(1e-6)
    Cr = Resn @ Resn.T
    pcr = Cr[iu[0], iu[1]]
    kept_ids = [e for e, k in zip(ids, keep.tolist()) if k]
    idx = {e: i for i, e in enumerate(kept_ids)}
    near, ctrl = neighbour_pairs({e: texts[e] for e in kept_ids if e in texts})
    near_c = [float(C[idx[a], idx[b]]) for a, b in near]
    ctrl_c = [float(C[idx[a], idx[b]]) for a, b in ctrl]
    near_r = [float(Cr[idx[a], idx[b]]) for a, b in near]
    ctrl_r = [float(Cr[idx[a], idx[b]]) for a, b in ctrl]
    pk = pack.table[kept_ids].float()
    to_pack = torch.nn.functional.cosine_similarity(R, pk, dim=1)
    to_pack_res = torch.nn.functional.cosine_similarity(Res, pk, dim=1)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")  # noqa: E731
    return {
        "name": name,
        "rows": n,
        "row_scale": rs,
        "norm_mean": float(R.norm(dim=1).mean()),
        "shared_energy": shared,
        "participation_ratio": pr,
        "pair_cos_mean": float(pc.mean()),
        "pair_cos_p95": float(pc.quantile(0.95)),
        "resid_cos_mean": float(pcr.mean()),
        "resid_abs_cos_mean": float(pcr.abs().mean()),
        "resid_cos_p95": float(pcr.quantile(0.95)),
        "neighbour_n": len(near),
        "neighbour_cos": mean(near_c),
        "neighbour_resid_cos": mean(near_r),
        "control_n": len(ctrl),
        "control_cos": mean(ctrl_c),
        "control_resid_cos": mean(ctrl_r),
        "to_pack_cos": float(to_pack.mean()),
        "to_pack_resid_cos": float(to_pack_res.mean()),
        "_m": m,
        "_rows": {e: R[i] for e, i in idx.items()},
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tables", default=DEFAULT_TABLES, help="label=arm_dir, comma list")
    p.add_argument("--json", default="", help="dump the numbers here")
    a = p.parse_args()

    from train.encoder import row_texts

    from library.anima.vocab_pack import load_vocab_pack
    from library.env import default_checkpoints
    from library.inference.text import ensure_text_strategies

    ck = default_checkpoints()
    pack = load_vocab_pack(ck.vocab_pack)
    tok, _ = ensure_text_strategies(ck.text_encoder, vocab_pack=ck.vocab_pack)

    tabs = []
    for spec in a.tables.split(","):
        label, name = spec.split("=", 1)
        if not (OUT / name / "trained.pt").exists() and not name.endswith(".pt"):
            print(f"(skip {label}: {name} has no trained.pt)", flush=True)
            continue
        ids, rows, keep, rs = load(name)
        texts = row_texts(tok, pack, ids)
        tabs.append(per_table(label, ids, rows, keep, rs, pack, texts))

    print("\n## per table\n")
    print(
        "| table | rows | row_scale | ‖row‖ | shared m̂ energy | PR | pair cos mean / p95 | "
        "resid cos mean / |mean| / p95 | が↔ガ, が↔か (n) | same-row ctrl (n) | "
        "neighbour resid / ctrl resid | cos to pack row (full / resid) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in tabs:
        print(
            f"| {t['name']} | {t['rows']} | {t['row_scale']:.1f} | {t['norm_mean']:.1f} | "
            f"{t['shared_energy']:.3f} | {t['participation_ratio']:.1f} | "
            f"{t['pair_cos_mean']:.3f} / {t['pair_cos_p95']:.3f} | "
            f"{t['resid_cos_mean']:.3f} / {t['resid_abs_cos_mean']:.3f} / {t['resid_cos_p95']:.3f} | "
            f"{t['neighbour_cos']:.3f} ({t['neighbour_n']}) | {t['control_cos']:.3f} ({t['control_n']}) | "
            f"{t['neighbour_resid_cos']:.3f} / {t['control_resid_cos']:.3f} | "
            f"{t['to_pack_cos']:.3f} / {t['to_pack_resid_cos']:.3f} |"
        )

    print("\n## shared-direction cos across tables (upper: m̂·m̂; lower: mean per-row cos on shared ids (n))\n")
    names = [t["name"] for t in tabs]
    print("| | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    cross = {}
    for i, ti in enumerate(tabs):
        cells = []
        for j, tj in enumerate(tabs):
            if i == j:
                cells.append("—")
            elif i < j:
                c = cos(ti["_m"], tj["_m"])
                cross[f"{ti['name']}|{tj['name']}|mean_dir"] = c
                cells.append(f"{c:.3f}")
            else:
                shared = sorted(set(ti["_rows"]) & set(tj["_rows"]))
                if not shared:
                    cells.append("·")
                    continue
                cs = [cos(ti["_rows"][e], tj["_rows"][e]) for e in shared]
                c = sum(cs) / len(cs)
                cross[f"{tj['name']}|{ti['name']}|row_cos"] = c
                cross[f"{tj['name']}|{ti['name']}|row_n"] = len(shared)
                cells.append(f"{c:.3f} ({len(shared)})")
        print(f"| {ti['name']} | " + " | ".join(cells) + " |")

    if a.json:
        out = {
            "tables": [{k: v for k, v in t.items() if not k.startswith("_")} for t in tabs],
            "cross": cross,
        }
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
