#!/usr/bin/env python3
"""Which teacher/voter pair survives the P1 filter — stock VL-1.6 or B′?

    ANIMA_ANIMETEXT_ROOT=… make daemon-run ARGS="\\
        project/cjk_aware_anima_dit/probes/ocr_voter_hayai_yield.py --n 1000"

plan2 § P1's filter keeps a crop when ``exact_key(teacher) == exact_key(voter)``
on the ♡-blind key, and the label takes the **teacher's** string. Shipped
choice: teacher B′, voter manga-ocr. Measured 2026-09-09: 47.9 % of a 100k draw
kept and **0 of the 130 hangul-dominant rows** — manga-ocr is JA-only, so it can
never vote on Korean.

This probe scores every pairing among four readers on one row set:

* ``bprime`` / ``mocr`` — the shipped pair, predictions reused from the 100k
  sweep on disk (so those columns cost nothing and match the measured numbers).
* ``stock``  — stock PaddleOCR-VL-1.6, the multilingual base B′ was tuned from.
  Read live.
* ``hayai``  — hayai v2.1.5, whose fine-tune set is modern scanlation **JA+KO**
  (``sorryhyun/paddleocr-vl-1.6-manga-lora`` discussion #1), the one KO-capable
  reader already wired into ``eval_manga109``. Read live.

Three arms: ``rand`` (``--n`` seeded rows — the honest overall yield), ``ko``
(the hangul-dominant rows) and ``zh`` (simplified-only markers, no kana, no
hangul — plan2 § K0). Both script arms are selected by **B′'s** output, so each
is biased toward crops B′ already calls that script; K1's stock sweep, not this
probe, is what sizes either pool.

The ``zh`` selector is a component test, not a character list: a char built on a
simplified-only radical (讠 钅 贝 见 页 鸟 马 门 车 鱼 纟 饣) or in a small set of
simplified forms with no Japanese counterpart. Shared shinjitai (国 来 当 体 数
点) is deliberately absent — it carries no signal. Read the arm's four-reader
dump before trusting the pairwise row: stock renders *Japanese* lines in
simplified glyphs (対酒ロケット四連装発射機 → 对潜口，卜四速装凳舟桨), so a
marker-selected arm carries JA rows whichever reader selects it.

Guards are applied **without** the truncation check, which needs token counts
the live readers here do not return; on the 100k sweep truncation was 0.5 % of
B′'s reads, so the four columns stay comparable. Writes
``output/ocr/pseudo/voter_matrix.{tsv,json}``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ocr"))
import eval_manga109 as ev  # noqa: E402
import manga109 as m109  # noqa: E402
import pseudo_label as pl  # noqa: E402
from animetext_crops import animetext_root  # noqa: E402

OUT = m109.REPO / "output/ocr/pseudo"
HANGUL = pl.HANGUL
KANA = pl.KANA
HANZI = re.compile(r"[一-鿿]")
# One grammar, shared with the filter that routes on it.
hangul_dominant = pl.hangul_dominant
simplified_zh = pl.simplified_zh

# Read live. ``None`` = the base weights: ``Vl16Reader`` falls through to
# ``models/paddleocr_vl_1.6`` when the ckpt holds no ``adapter_config.json``.
LIVE = {
    "stock": (ev.Vl16Reader, None),
    "hayai": (ev.HayaiReader, "JustANormalTinkerer/hayai-ocr-v2@v2.1.5"),
}
# Reused from the 100k sweep parquets.
CACHED = {"bprime": "vl16", "mocr": "manga_ocr"}
ALL = ["stock", "bprime", "hayai", "mocr"]


def arms(name: str, n: int, seed: int) -> dict[str, pd.DataFrame]:
    draw = pd.read_parquet(OUT / f"{name}_draw.parquet")
    df = draw
    for col, stem in CACHED.items():
        t = pd.read_parquet(OUT / f"{name}_{stem}.parquet")[["image_id", "k", "pred"]]
        df = df.merge(t.rename(columns={"pred": col}), on=["image_id", "k"])
    return {
        "rand": df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True),
        "ko": df[df.bprime.fillna("").map(hangul_dominant)].reset_index(drop=True),
        "zh": df[df.bprime.fillna("").map(simplified_zh)].reset_index(drop=True),
    }


def read_live(reader, df: pd.DataFrame, root: Path, bs: int) -> list[str]:
    crops, keep = [], []
    for i, r in enumerate(df.itertuples()):
        img = cv2.imread(str(root / r.path), cv2.IMREAD_COLOR)
        if img is not None:
            crops.append(img)
            keep.append(i)
    orients = ["vertical" if c.shape[0] > c.shape[1] else "horizontal" for c in crops]
    got = dict(zip(keep, reader.read(crops, orients, bs)))
    return [got.get(i, "") for i in range(len(df))]


def profile(strings: list[str]) -> dict[str, float]:
    n = max(len(strings), 1)
    return {
        "hangul_%": 100 * sum(bool(HANGUL.search(s)) for s in strings) / n,
        "kana_%": 100 * sum(bool(KANA.search(s)) for s in strings) / n,
        "han_%": 100 * sum(bool(HANZI.search(s)) for s in strings) / n,
        "simp_%": 100 * sum(simplified_zh(s) for s in strings) / n,
        "mean_len": sum(len(s) for s in strings) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="pl100k")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    root = animetext_root()
    data = arms(a.name, a.n, a.seed)
    print("arms: " + ", ".join(f"{k}={len(v)}" for k, v in data.items()), flush=True)

    for name, (cls, ckpt) in LIVE.items():
        print(f"\nloading {name} ({ckpt or 'base weights'})", flush=True)
        reader = cls(ckpt, a.device)
        for arm, df in data.items():
            df[name] = read_live(reader, df, root, a.bs)
            print(f"  {arm}: {len(df)} crops read", flush=True)
        del reader

    stats: dict[str, dict] = {}
    for arm, df in data.items():
        for r in ALL:
            df[f"ok_{r}"] = [pl.guard(str(p), 0) == "ok" for p in df[r].fillna("")]
        row: dict = {"n": len(df), "readers": {}, "pairs": {}}
        for r in ALL:
            row["readers"][r] = {
                "guard_ok_%": 100 * df[f"ok_{r}"].mean(),
                **profile(list(df[r].fillna("").astype(str))),
            }
        keys = {r: df[r].fillna("").astype(str).map(ev.exact_key) for r in ALL}
        for x, y in combinations(ALL, 2):
            both_ok = df[f"ok_{x}"] & df[f"ok_{y}"]
            agree = both_ok & (keys[x] == keys[y])
            row["pairs"][f"{x}+{y}"] = {
                "kept": int(agree.sum()),
                "kept_%": 100 * agree.mean(),
                "hangul_%": profile(list(df[x][agree]))["hangul_%"],
                "simp_%": profile(list(df[x][agree]))["simp_%"],
            }
            df[f"a_{x}_{y}"] = agree
        stats[arm] = row

    print("\n=== per reader (guard pass, then what it emits) ===")
    for arm, r in stats.items():
        print(f"  [{arm}] n={r['n']}")
        for name, v in r["readers"].items():
            print(
                f"    {name:7s} guard ok {v['guard_ok_%']:5.1f}%   "
                f"hangul {v['hangul_%']:5.1f}%  kana {v['kana_%']:5.1f}%  "
                f"han {v['han_%']:5.1f}%  simp {v['simp_%']:5.1f}%  "
                f"len {v['mean_len']:5.1f}"
            )

    print("\n=== pairwise agreement = filter yield (teacher+voter, order-free) ===")
    for arm, r in stats.items():
        print(f"  [{arm}] n={r['n']}")
        for pair, v in sorted(r["pairs"].items(), key=lambda kv: -kv[1]["kept"]):
            print(
                f"    {pair:16s} kept {v['kept']:5d}  {v['kept_%']:5.1f}%   "
                f"hangul {v['hangul_%']:5.1f}%  simp {v['simp_%']:5.1f}%"
            )

    for arm in ("ko", "zh"):
        print(f"\n=== {arm} arm, all four readers (first 25) ===")
        for r in data[arm].head(25).itertuples():
            print(f"  [{r.image_id}_{r.k}]")
            for name in ALL:
                print(f"      {name:7s} {getattr(r, name)!r}")

    both = pd.concat([d.assign(arm=k) for k, d in data.items()])
    cols = ["arm", "image_id", "k", "w", "h", "path", *ALL] + [
        c for c in both.columns if c.startswith(("ok_", "a_"))
    ]
    both[cols].to_csv(OUT / "voter_matrix.tsv", sep="\t", index=False)
    (OUT / "voter_matrix.json").write_text(
        json.dumps({"seed": a.seed, "arms": stats}, ensure_ascii=False, indent=1)
    )
    print(f"\nwrote {OUT / 'voter_matrix.tsv'} and voter_matrix.json", flush=True)


if __name__ == "__main__":
    main()
