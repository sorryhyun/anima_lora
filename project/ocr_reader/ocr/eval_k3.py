#!/usr/bin/env python3
"""The KO / ZH gate (roadmap P0 K3): any reader over the gelnote hand labels.

    make daemon-run ARGS="project/ocr_reader/ocr/eval_k3.py --reader vl16"                       # stock
    … --reader vl16 --ckpt output/ocr/vl16_b2_norm4/ep1 --name v3
    … --reader vl16 --ckpt output/ocr/<run>/ep1 --name <run>
    .venv/bin/python project/ocr_reader/ocr/eval_k3.py --compare stock v3 <run>   # CPU, paired

Labels: ``assets/k3_gelnote.tsv`` (``gate/k3_gelnote.py``), 50 KO + 50 ZH
crops, one per page, whole pages held out of every pseudo draw. ``text`` is the
hand label; rows with an empty ``text`` or ``status: skip`` are not scored.
``zh_script`` (simp / trad / neutral / mixed) splits the ZH table: a reader
that answers a traditional label in simplified scores a miss on exact.
Crops are the frozen ``eval_data/k3_gelnote/crops/`` (``gate/build_eval_data.py``).

Metrics, per language:

- **exact**: ``textnorm.exact_key`` (folds, whitespace-blind), the sincos key.
- **spaced**: ``textnorm.normalize_target`` (whitespace runs → one space). For
  KO this is exact *with* 띄어쓰기, which the line refuses to strip.
- **cer**: Levenshtein over the exact keys / label length.
- **kana leak**: the prediction carries kana and the label does not. This is
  the B′ failure (KO / ZH read as Japanese).
- sim / runaway as ``eval_manga109``.

The JA ``ー`` put-back (``eval_manga109.normalize_ja``) is not applied: it is a
PP-OCRv6 repair for kana text. n = 50 per language puts the binomial SE near
7 points; read single-run gaps under that as noise and use ``--compare``'s
paired counts across arms.

Writes ``reports/ocr_eval_k3_<name>.md`` + ``output/ocr/eval/k3_<name>.jsonl``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_manga109 as ev  # noqa: E402
import manga109 as m109  # noqa: E402
import textnorm  # noqa: E402

LABELS = m109.ASSETS / "k3_gelnote.tsv"
FROZEN = m109.REPO / "project/ocr_reader/eval_data/k3_gelnote"


def load_labels(path: Path = LABELS) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "status" not in df.columns:
        df["status"] = ""
    df = df[(df.text.str.strip() != "") & (df.status != "skip")]
    if "zh_script" not in df.columns:
        df["zh_script"] = ""
    return df.reset_index(drop=True)


def load_crops(df: pd.DataFrame) -> list:
    crops = [cv2.imread(str(FROZEN / p)) for p in df.path]
    missing = [p for p, c in zip(df.path, crops) if c is None]
    if missing:
        raise SystemExit(
            f"{len(missing)} crops unreadable under {FROZEN} (e.g. {missing[0]}); "
            "run gate/build_eval_data.py"
        )
    return crops


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def has_kana(s: str) -> bool:
    return any("ぁ" <= c <= "ヿ" and c not in "・ー" for c in s)


def score(df: pd.DataFrame, preds: list[str]) -> pd.DataFrame:
    rec = m109.pilot_records()
    rows = []
    for (_, r), p in zip(df.iterrows(), preds):
        pk, gk = textnorm.exact_key(p), textnorm.exact_key(r.text)
        rows.append(
            dict(
                pred=p,
                exact=pk == gk,
                spaced=textnorm.normalize_target(p)
                == textnorm.normalize_target(r.text),
                cer=levenshtein(pk, gk) / max(1, len(gk)),
                kana_leak=has_kana(p) and not has_kana(r.text),
                sim=rec.sim(p, r.text),
                runaway=rec.is_runaway(p),
            )
        )
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def summary(scored: pd.DataFrame, name: str, wall: float) -> str:
    md = [
        f"# OCR eval — `{name}` on K3 gelnote (`{LABELS.name}`)\n",
        f"Reader wall {wall:.0f} s for {len(scored)} crops.\n",
        "| lang | n | exact | spaced | cer | sim | kana leak | runaway |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for lang, g in scored.groupby("lang"):
        md.append(
            f"| {lang} | {len(g)} | {int(g.exact.sum())} ({100 * g.exact.mean():.0f} %) | "
            f"{int(g.spaced.sum())} | {g.cer.mean():.3f} | {g.sim.mean():.3f} | "
            f"{int(g.kana_leak.sum())} | {int(g.runaway.sum())} |"
        )
    zh = scored[(scored.lang == "zh") & (scored.zh_script != "")]
    if len(zh):
        md.append("\nZH by `zh_script` (simp / trad / neutral / mixed):")
        for s, g in zh.groupby("zh_script"):
            md.append(
                f"- {s}: exact {int(g.exact.sum())} / {len(g)}, cer {g.cer.mean():.3f}"
            )
    md.append(
        "\n## Every row\n\n| id | lang | gt | pred | exact | spaced | cer |\n|---|---|---|---|---|---|---|"
    )
    for _, r in scored.sort_values(["lang", "cer"], ascending=[True, False]).iterrows():
        pred = r.pred.replace("|", "\\|")[:40]
        md.append(
            f"| {r.id} | {r.lang} | {r.text} | {pred} | {'✓' if r.exact else ''} | "
            f"{'✓' if r.spaced else ''} | {r.cer:.2f} |"
        )
    return "\n".join(md) + "\n"


def compare(names: list[str]) -> str:
    """Per-language table over stored runs + paired wins / losses against the first."""
    runs = {
        n: pd.read_json(ev.OUT / f"k3_{n}.jsonl", lines=True).set_index("id")
        for n in names
    }
    ids = sorted(set.intersection(*(set(d.index) for d in runs.values())))
    base = runs[names[0]].loc[ids]
    md = [
        f"# K3 compare — {len(ids)} rows common to every run; paired vs `{names[0]}`\n",
        "| run | lang | exact | spaced | cer | kana leak | wins / losses | McNemar z |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n in names:
        d = runs[n].loc[ids]
        for lang in sorted(set(d.lang)):
            m = d.lang == lang
            g, b = d[m], base[m]
            w = int((g.exact & ~b.exact).sum())
            lo = int((~g.exact & b.exact).sum())
            z = (w - lo) / math.sqrt(w + lo) if w + lo else 0.0
            md.append(
                f"| {n} | {lang} | {int(g.exact.sum())} / {len(g)} | {int(g.spaced.sum())} | "
                f"{g.cer.mean():.3f} | {int(g.kana_leak.sum())} | "
                + ("— | — |" if n == names[0] else f"{w} / {lo} | {z:+.1f} |")
            )
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--reader", choices=sorted(ev.READERS))
    ap.add_argument("--ckpt")
    ap.add_argument("--name")
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument(
        "--compare", nargs="+", metavar="NAME", help="stored k3_<NAME>.jsonl runs (CPU)"
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=ev.MAX_NEW_TOKENS)
    a = ap.parse_args()
    if a.compare:
        print(compare(a.compare))
        return
    if not a.reader:
        ap.error("--reader is required unless --compare")
    name = a.name or (
        f"{a.reader}-{Path(a.ckpt).parent.name}-{Path(a.ckpt).name}"
        if a.ckpt
        else a.reader
    )
    df = load_labels(a.labels)
    if df.empty:
        raise SystemExit(f"no labelled rows in {a.labels}")
    crops = load_crops(df)
    reader = ev.READERS[a.reader](a.ckpt, a.device)
    reader.max_tokens = a.max_new_tokens
    t0 = time.time()
    preds = reader.read(crops, list(df.orient), a.bs)
    scored = score(df, preds)
    md = summary(scored, name, time.time() - t0)
    ev.OUT.mkdir(parents=True, exist_ok=True)
    scored.to_json(
        ev.OUT / f"k3_{name}.jsonl", orient="records", lines=True, force_ascii=False
    )
    ev.REPORTS.mkdir(exist_ok=True)
    (ev.REPORTS / f"ocr_eval_k3_{name}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
