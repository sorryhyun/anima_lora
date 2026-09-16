#!/usr/bin/env python
"""transplant_table — build no-training probe tables: the 53k table's shared
direction m̂ (the composite-trained render trigger) carrying a flat-only arm's
per-row identity residual.

Question (2026-09-16): a flat-only table (P0b) renders singles 36/36 but wipes
the scene; the composite-trained 53k table keeps the scene. The 53k table
splits into one shared direction m̂ (18 % of its energy, hit tracks it) plus
near-orthogonal residuals. If P0b's residual renders *inside a scene* when it
rides on the 53k m̂, new glyphs can be trained flat-only with m̂ pinned.

Writes one ``rows`` arm dir per cond (``trained.pt`` with ``delta`` only, plus
``transplant.json`` with the coefficients) under ``output/wake_probe/``:

    rows_transplant_m53k_p0bres          a_r · m̂_fam(53k) + f_r^{P0b ⊥ m̂_fam(P0b)}
    rows_transplant_m53k_p0bres_matched  same, residual rescaled to the 53k row's residual norm
    rows_transplant_m53k_only            a_r · m̂_fam(53k) alone (trigger without identity)
    rows_transplant_p0b_full             P0b's table as saved (flat-only baseline)

``a_r`` = the 53k row's own coefficient along its family direction when the
row is in the 53k table, else the family mean. Families: kana (basic hira +
kata) vs kanji/other (the two 53k family directions have cos 0.65).

    .venv/bin/python project/cjk_renderable_anima/src/probe/transplant_table.py
    # then per cond:
    make daemon-run ARGS="--stall-timeout 0 project/cjk_renderable_anima/src/wake_probe.py \
        --stage native --arm rows --data_tag transplant --arm_tag m53k_p0bres \
        --native_chars あ,か,す,日 --native_clauses en,swap --seeds 2 --no_floor --delta_parts full"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = REPO / "output" / "wake_probe"


def fam_of(text: str) -> str:
    from common.text import HIRA, KATA

    return "kana" if text in HIRA + KATA else "other"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--src",
        default="rows_synth_full_fm10k_full_s53k_qoff",
        help="arm whose m̂ is transplanted",
    )
    p.add_argument(
        "--donor",
        default="encoder_wdsek_w120_s24k_p0b",
        help="flat-only arm whose residuals ride on m̂",
    )
    p.add_argument("--tag", default="transplant")
    p.add_argument(
        "--conds",
        default="m53k_p0bres,m53k_p0bres_matched,m53k_only,p0b_full",
        help="which tables to write (names as in the docstring; 'p0b' reads as 'donor')",
    )
    a = p.parse_args()

    from train.encoder import row_texts

    from library.anima.vocab_pack import load_vocab_pack
    from library.env import default_checkpoints
    from library.inference.text import ensure_text_strategies

    S = torch.load(OUT / a.src / "trained.pt", map_location="cpu", weights_only=False)
    D = torch.load(OUT / a.donor / "trained.pt", map_location="cpu", weights_only=False)
    ck = default_checkpoints()
    pack = load_vocab_pack(ck.vocab_pack)
    tok, _ = ensure_text_strategies(ck.text_encoder, vocab_pack=ck.vocab_pack)

    s_raw = S["delta"]["raw"].float()
    s_ids = [int(i) for i in S["delta"]["ext_ids"]]
    d_raw = D["delta"]["raw"].float()
    d_ids = [int(i) for i in D["delta"]["ext_ids"]]
    s_text = row_texts(tok, pack, s_ids)
    d_text = (
        {int(k): v for k, v in D["row_text"].items()}
        if "row_text" in D
        else row_texts(tok, pack, d_ids)
    )
    assert abs(float(S["delta"]["row_scale"]) - float(D["delta"]["row_scale"])) < 2.0, (
        "row_scale units differ"
    )

    def fam_dirs(raw, ids, texts):
        out = {}
        named = [i for i, e in enumerate(ids) if e in texts]
        for fam in ("kana", "other"):
            ii = [i for i in named if fam_of(texts[ids[i]]) == fam]
            if (
                len(ii) < 3
            ):  # donor without this family: fall back to its all-row direction
                ii = named
            m = raw[ii].mean(0)
            out[fam] = (m / m.norm(), ii)
        return out

    s_dir = fam_dirs(s_raw, s_ids, s_text)
    d_dir = fam_dirs(d_raw, d_ids, d_text)
    s_coef = {fam: (s_raw[ii] @ mh) for fam, (mh, ii) in s_dir.items()}
    fam_mean_coef = {fam: float(c.mean()) for fam, c in s_coef.items()}
    fam_mean_resid = {
        fam: float(
            (s_raw[ii] - (s_raw[ii] @ mh)[:, None] * mh[None]).norm(dim=1).mean()
        )
        for fam, (mh, ii) in s_dir.items()
    }
    s_by_text = {s_text[e]: i for i, e in enumerate(s_ids) if e in s_text}

    n = len(d_ids)
    dim = d_raw.shape[1]
    # cond names: the first run (donor P0b) wrote m53k_p0bres / m53k_p0bres_matched /
    # m53k_only / p0b_full; later donors use the donor-neutral names below.
    alias = {
        "m53k_p0bres": "m53k_res",
        "m53k_p0bres_matched": "m53k_res_matched",
        "p0b_full": "donor_full",
    }
    want = [alias.get(c, c) for c in a.conds.split(",") if c]
    keep_old = any(c in alias for c in a.conds.split(","))
    tables = {
        k: torch.zeros(n, dim) for k in ("m53k_res", "m53k_res_matched", "m53k_only")
    }
    tables["donor_full"] = d_raw.clone()
    rec = []
    for i, e in enumerate(d_ids):
        t = d_text.get(e)
        if t is None:
            continue
        fam = fam_of(t)
        mh_s, _ = s_dir[fam]
        mh_d, _ = d_dir[fam]
        v = d_raw[i]
        resid = v - (v @ mh_d) * mh_d
        if t in s_by_text:
            sv = s_raw[s_by_text[t]]
            coef = float(sv @ mh_s)
            target = float((sv - coef * mh_s).norm())
            src = "row"
        else:
            coef, target, src = fam_mean_coef[fam], fam_mean_resid[fam], "family"
        tables["m53k_res"][i] = coef * mh_s + resid
        tables["m53k_res_matched"][i] = coef * mh_s + resid * (
            target / max(float(resid.norm()), 1e-6)
        )
        tables["m53k_only"][i] = coef * mh_s
        rec.append(
            {
                "text": t,
                "fam": fam,
                "coef": coef,
                "coef_src": src,
                "p0b_along_own_m": float(v @ mh_d),
                "p0b_resid_norm": float(resid.norm()),
                "matched_resid_norm": target,
            }
        )
    meta = {
        "src": a.src,
        "donor": a.donor,
        "cos_mhat_src_donor": {
            fam: float(s_dir[fam][0] @ d_dir[fam][0]) for fam in s_dir
        },
        "cos_src_kana_other": float(s_dir["kana"][0] @ s_dir["other"][0]),
        "fam_mean_coef_src": fam_mean_coef,
        "fam_mean_resid_src": fam_mean_resid,
        "rows": rec,
    }
    inv_alias = {v: k for k, v in alias.items()}
    for name, tab in tables.items():
        if name not in want:
            continue
        d = OUT / f"rows_{a.tag}_{inv_alias.get(name, name) if keep_old else name}"
        d.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "delta": {
                    "ext_ids": d_ids,
                    "raw": tab,
                    "row_scale": float(S["delta"]["row_scale"]),
                },
                "arm": "rows",
                "transplant": name,
                "killed": "",
            },
            d / "trained.pt",
        )
        json.dump(
            {"cond": name, **meta},
            open(d / "transplant.json", "w"),
            ensure_ascii=False,
            indent=1,
        )
        print(f"{d.name}: {n} rows, mean norm {tab.norm(dim=1).mean():.3f}")
    print(
        "cos(m̂ src, m̂ donor) per family:",
        meta["cos_mhat_src_donor"],
        "; src kana~other",
        round(meta["cos_src_kana_other"], 3),
    )
    for r in rec:
        if r["text"] in "あかす日":
            print(r)


if __name__ == "__main__":
    main()
