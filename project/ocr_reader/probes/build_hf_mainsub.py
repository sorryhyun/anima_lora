#!/usr/bin/env python3
"""The main / sub training set (``crop_dataset.load_mainsub``) as a Hugging Face dataset folder
in the Hub layout: parquet shards with the crop bytes embedded, one config per seed.

* ``seed<N>/train-*.parquet``: ``load_mainsub('train', seed=N)``. The seed fixes the
  quarter of the unreviewed ruby subs and the negative draws; everything else is the same
  in every config.
* ``test/test-*.parquet``: ``load_mainsub('val')`` (the val books; no sampling), shared by
  every config.

Columns: ``image`` (the PNG crop as trained, before augmentation), ``prompt`` (``OCR:`` /
``OCR2:``), ``text`` (the normalized target; ``""`` = a negative), ``metric``, ``source``,
``book``, ``page``, ``crop_id``, ``orient``.

    ANIMA_MANGA109S_ROOT=... python project/ocr_reader/probes/build_hf_mainsub.py \\
        --out /media/sorryhyun/new/dataset/manga109s/derived/hf_mainsub [--seeds 0 1]

    datasets.load_dataset("<out>", "seed0")  # DatasetDict(train, test)

Manga109-s images may not be redistributed: the folder stays local.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from project.ocr_reader.core import crop_dataset as cd
from project.ocr_reader.core import manga109 as m109

SHARD_ROWS = 20_000
COLS = ["prompt", "text", "metric", "source", "book", "page", "crop_id", "orient"]


def features():
    from datasets import Features, Image, Value

    return Features(
        {
            "image": Image(),
            **{c: Value("int32" if c == "page" else "string") for c in COLS},
        }
    )


def write_split(df: pd.DataFrame, out: Path, split: str) -> int:
    d = m109.derived_root()
    schema = features().arrow_schema
    out.mkdir(parents=True, exist_ok=True)
    df = df.reset_index(drop=True)
    n_shards = max(1, -(-len(df) // SHARD_ROWS))
    for k in range(n_shards):
        part = df.iloc[k * SHARD_ROWS : (k + 1) * SHARD_ROWS]
        imgs = [
            {"bytes": (d / p).read_bytes(), "path": Path(p).name} for p in part.path
        ]
        cols = {
            "image": imgs,
            "prompt": part.task.tolist(),
            "text": part.target.tolist(),
            "metric": part.metric.tolist(),
            "source": part.source.tolist(),
            "book": part.book.astype(str).tolist(),
            "page": part.page.astype(int).tolist(),
            "crop_id": part.id.astype(str).tolist(),
            "orient": part.orient.astype(str).tolist(),
        }
        table = pa.Table.from_pydict(cols, schema=schema)
        pq.write_table(table, out / f"{split}-{k:05d}-of-{n_shards:05d}.parquet")
        print(f"[{out.name}] {split} shard {k + 1}/{n_shards}", flush=True)
    return len(df)


def counts(df: pd.DataFrame) -> str:
    g = df.groupby(["task", "metric"]).size()
    return ", ".join(f"{t} {m} {n:,}" for (t, m), n in g.items())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    a = ap.parse_args()
    lines = []
    test = cd.load_mainsub("val")
    write_split(test, a.out / "test", "test")
    lines.append(f"- `test` (val books): {len(test):,} rows — {counts(test)}")
    for s in a.seeds:
        tr = cd.load_mainsub("train", seed=s).sample(frac=1, random_state=s)
        write_split(tr, a.out / f"seed{s}", "train")
        lines.append(f"- `seed{s}` train: {len(tr):,} rows — {counts(tr)}")
    cfg = "\n".join(
        f"- config_name: seed{s}\n  data_files:\n"
        f"  - split: train\n    path: seed{s}/train-*\n"
        f"  - split: test\n    path: test/test-*"
        for s in a.seeds
    )
    card = f"""---
configs:
{cfg}
---

# Manga109 main / sub OCR crops (local build)

Built {datetime.date.today()} by `project/ocr_reader/probes/build_hf_mainsub.py`
from `crop_dataset.load_mainsub`; rules in `project/ocr_reader/README.md` § main / sub.
`OCR:` reads the crop's main text, `OCR2:` the rest (one item per line); `text == ""` is a
negative. Train rows are shuffled with the config's seed.

{chr(10).join(lines)}

Sources: Manga109-s (detector crops, train / val books) and KO / ZH pseudo rows
(gelnote, swaprelax_v5). Manga109-s images may not be redistributed: do not push.
"""
    (a.out / "README.md").write_text(card, encoding="utf-8")
    print(card)


if __name__ == "__main__":
    main()
