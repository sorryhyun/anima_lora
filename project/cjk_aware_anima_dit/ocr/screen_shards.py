#!/usr/bin/env python3
"""K1 streaming screen: shard parquet → crops in memory → hayai → write **hits only**.

    ANIMA_ANIMETEXT_ROOT=<root> make daemon-run ARGS="\\
        project/cjk_aware_anima_dit/ocr/screen_shards.py --name kscreen \\
        --shard 1 2 3 4 5 --bs 64"

Nothing is cut to disk up front. Each row group is decoded and cropped in
memory, hayai (``hayai_beam``, bf16) reads every crop, and only the rows whose
read matches a ``--script`` predicate (``hangul_dominant`` / ``simplified_zh``,
~0.5 % of the pool each way) are kept, the matched script in a ``script``
column: their crop goes to
``<root>/animetext_crops/screen/<split>/<image_id>_<k>.webp`` and their row to

* ``output/ocr/pseudo/<name>_draw.parquet``  — the draw schema, so the rest
  of the K1/K2 chain runs unchanged on it:
  ``pseudo_label.py --name <name> sweep --reader stock --rows_from hayai``
  then ``filter --teacher stock --voter hayai --script ko``;
* ``output/ocr/pseudo/<name>_hayai.parquet`` — the sweep-table schema.

Resume is per row group (``<name>_screen_state.json``); both parquets are
rewritten at every flush (hits are small). The next row group is decoded on a
thread while the GPU reads the current one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manga109 as m109  # noqa: E402
import pseudo_label as pl  # noqa: E402
from animetext_crops import ENCODE, animetext_root, crop_box  # noqa: E402

OUT = pl.OUT
DRAW_COLS = ["image_id", "k", "w", "h", "box", "path", "split", "script"]


def decode_group(parquet: Path, gi: int, pad: float, min_side: int):
    """All text-box crops of one row group, in memory."""
    t = pq.ParquetFile(parquet).read_row_group(gi).to_pylist()
    metas, crops = [], []
    for r in t:
        img = cv2.imdecode(
            np.frombuffer(r["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR
        )
        if img is None:
            continue
        obj = r["objects"]
        k = 0
        for box, cat in zip(obj["bbox"], obj["category"]):
            if cat != 0:
                continue
            crop, xyxy = crop_box(img, box, pad, min_side)
            if crop is None:
                continue
            metas.append(
                (int(r["image_id"]), k, int(crop.shape[1]), int(crop.shape[0]), xyxy)
            )
            crops.append(crop)
            k += 1
    return metas, crops


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--name", default="kscreen")
    ap.add_argument(
        "--parquet", nargs="*", default=[], help="shard parquet(s), in order"
    )
    ap.add_argument(
        "--shard",
        nargs="*",
        type=int,
        default=[],
        help="shard index(es) under <root>/hub/data/ — the root path may carry a "
        "space, which make ARGS would split",
    )
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--script",
        nargs="+",
        choices=sorted(pl.SCRIPT),
        default=["ko", "zh"],
        help="keep a crop whose read matches any of these; the first match is "
        "recorded in the draw's ``script`` column",
    )
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--pad", type=float, default=0.12)
    ap.add_argument("--min_side", type=int, default=8)
    ap.add_argument(
        "--flush", type=int, default=20, help="rewrite outputs every N row groups"
    )
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    root = animetext_root()
    parquets = [Path(p) for p in a.parquet] + [
        root / "hub" / "data" / f"{a.split}-{n:05d}-of-00006.parquet" for n in a.shard
    ]
    if not parquets:
        raise SystemExit("pass --parquet <file>… or --shard <n>…")
    crop_dir = root / "animetext_crops" / "screen" / a.split
    crop_dir.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    draw_path = OUT / f"{a.name}_draw.parquet"
    pred_path = OUT / f"{a.name}_hayai.parquet"
    state_path = OUT / f"{a.name}_screen_state.json"

    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    draw_rows = (
        pd.read_parquet(draw_path).to_dict("records") if draw_path.is_file() else []
    )
    pred_rows = (
        pd.read_parquet(pred_path).to_dict("records") if pred_path.is_file() else []
    )
    if draw_rows:
        print(f"resuming: {len(draw_rows)} hits on disk, state {state}", flush=True)

    reader = pl.SWEEPERS["hayai"](pl.CKPT["hayai"], a.device).r
    preds = [(sc, pl.SCRIPT[sc]) for sc in a.script]

    def which(pred: str) -> str | None:
        return next((sc for sc, f in preds if f(pred)), None)

    rec = m109.pilot_records()
    dec_pool = ThreadPoolExecutor(max_workers=1)  # next row group's crops
    pre_pool = ThreadPoolExecutor(max_workers=1)  # next batch's processor pass

    def flush():
        pd.DataFrame(draw_rows, columns=DRAW_COLS).to_parquet(draw_path, index=False)
        pd.DataFrame(pred_rows).to_parquet(pred_path, index=False)
        state_path.write_text(json.dumps(state))

    def batches():
        """(parquet, gi, n_groups, last_in_group, metas, crops) across every
        shard, with the next row group decoded on a thread."""
        for parquet in parquets:
            pf = pq.ParquetFile(parquet)
            n_groups = pf.metadata.num_row_groups
            start = int(state.get(parquet.name, 0))
            print(
                f"{parquet.name}: {pf.metadata.num_rows} rows, {n_groups} groups, "
                f"from {start}",
                flush=True,
            )
            if start >= n_groups:
                continue
            fut = dec_pool.submit(decode_group, parquet, start, a.pad, a.min_side)
            for gi in range(start, n_groups):
                metas, crops = fut.result()
                if gi + 1 < n_groups:
                    fut = dec_pool.submit(
                        decode_group, parquet, gi + 1, a.pad, a.min_side
                    )
                if not crops:
                    yield parquet, gi, n_groups, True, [], []
                    continue
                for s in range(0, len(crops), a.bs):
                    last = s + a.bs >= len(crops)
                    yield (
                        parquet,
                        gi,
                        n_groups,
                        last,
                        metas[s : s + a.bs],
                        crops[s : s + a.bs],
                    )

    t0, n_crops, n_hits, since_flush = time.time(), 0, 0, 0
    n_by: dict[str, int] = {}
    it = batches()
    nxt = next(it, None)
    pre = pre_pool.submit(reader.prepare, nxt[5]) if nxt and nxt[5] else None
    while nxt is not None:
        parquet, gi, n_groups, last, metas, crops = nxt
        inputs = pre.result() if pre is not None else None
        # The next batch's CPU pass runs while this one decodes on the GPU.
        nxt = next(it, None)
        pre = pre_pool.submit(reader.prepare, nxt[5]) if nxt and nxt[5] else None
        if inputs is not None:
            for (iid, k, w, h, xyxy), crop, pred in zip(
                metas, crops, reader.decode(inputs)
            ):
                sc = which(pred)
                if sc is None:
                    continue
                n_by[sc] = n_by.get(sc, 0) + 1
                fn = crop_dir / f"{iid}_{k}.webp"
                cv2.imwrite(str(fn), crop, ENCODE["webp"])
                draw_rows.append(
                    dict(
                        image_id=iid,
                        k=k,
                        w=w,
                        h=h,
                        box=xyxy,
                        path=str(fn.relative_to(root)),
                        split=a.split,
                        script=sc,
                    )
                )
                pred_rows.append(
                    dict(
                        image_id=iid,
                        k=k,
                        pred=pred,
                        n_tokens=0,
                        score=float("nan"),
                        runaway=bool(rec.is_runaway(pred)),
                    )
                )
                n_hits += 1
            n_crops += len(crops)
        if last:
            state[parquet.name] = gi + 1
            since_flush += 1
            if since_flush >= a.flush or gi + 1 == n_groups:
                flush()
                since_flush = 0
            if (gi + 1) % 5 == 0 or gi + 1 == n_groups:
                el = time.time() - t0
                print(
                    f"[screen] {parquet.name} group {gi + 1}/{n_groups}  {n_crops} crops  "
                    f"{n_crops / max(el, 1e-9):.1f} crops/s  hits {n_hits} "
                    f"({100 * n_hits / max(n_crops, 1):.3f}%) {n_by}",
                    flush=True,
                )
    flush()
    print(
        f"done: {n_crops} crops read, {n_hits} hits → {draw_path.name}, {pred_path.name}",
        flush=True,
    )


if __name__ == "__main__":
    main()
