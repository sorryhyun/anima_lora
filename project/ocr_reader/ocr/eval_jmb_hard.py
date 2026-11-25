#!/usr/bin/env python3
"""JMangaBench_Hard v1.0 (github.com/muscgab/JMangaBench_Hard) read by the shipped reader.

3 015 text-box crops (sound effects, stylised lettering, handwriting), one label each.
The crops are read by ``anime_tools.ocr.reader.OcrReader`` (``--version``: v7, the
default, or v6) under ``OCR:``, greedy, in batches of 16 sorted by area — the card's
code, which the leaderboard's v6 row was run with (there at most 80 new tokens; the
package's 200 here).

Arms (``--arms``), one prediction file each:

- ``given``: the crop as shipped, the raw decode. The leaderboard's protocol.
- ``guard``: the same reads through the package's :func:`guard` (a dropped read is an
  empty prediction). No second forward.
- ``rep``: the crop grown by v7's training margin (``tight_pad_box``'s rule on the crop's
  own size) with its edge pixels repeated, since the page around the box is not shipped.

Data (local, evaluation only, CC BY-NC-SA 4.0): ``eval_data/jmb_hard/`` holding the
Hub dataset's ``crops/`` and the repo's ``data/gt.jsonl`` and ``eval/``. Writes
``output/ocr/eval/jmb_hard/<name>/<arm>.jsonl`` (``{"key", "pred"}``, the benchmark's
format) and ``reads.jsonl`` (raw read and confidence a crop). Score with the
benchmark's own script:

    make daemon-run ARGS="--queue --stall-timeout 600 project/ocr_reader/ocr/eval_jmb_hard.py --name v7"
    .venv/bin/python -I project/ocr_reader/eval_data/jmb_hard/eval/score.py --pred output/ocr/eval/jmb_hard/v7/given.jsonl --name v7
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from project.ocr_reader.core.paths import EVAL_DATA, EVAL_OUT

ARMS = ("given", "guard", "rep")


def rep_pad(bgr):
    """The crop grown by v7's margin, edge pixels repeated."""
    import cv2
    from anime_tools.ocr.reader import (
        TIGHT_PAD_FRAC,
        TIGHT_PAD_PX,
        TIGHT_PAD_SHORT_FRAC,
    )

    h, w = bgr.shape[:2]
    p = round(
        min(TIGHT_PAD_FRAC * max(w, h), TIGHT_PAD_SHORT_FRAC * min(w, h), TIGHT_PAD_PX)
    )
    return cv2.copyMakeBorder(bgr, p, p, p, p, cv2.BORDER_REPLICATE) if p else bgr


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", type=Path, default=EVAL_DATA / "jmb_hard")
    ap.add_argument("--name", required=True, help="the output folder")
    ap.add_argument("--version", default="v7", choices=("v7", "v6"))
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--limit", type=int, help="read only this many crops (smoke)")
    a = ap.parse_args()

    import cv2
    import torch
    from anime_tools.ocr.reader import OcrReader, guard

    gt = [json.loads(x) for x in (a.data / "gt.jsonl").read_text("utf-8").splitlines()]
    gt = gt[: a.limit] if a.limit else gt
    crops = []
    for g in gt:
        im = cv2.imread(str(a.data / g["file"]), cv2.IMREAD_COLOR)
        assert im is not None, g["file"]
        crops.append(im)
    out = EVAL_OUT / "jmb_hard" / a.name
    out.mkdir(parents=True, exist_ok=True)
    rd = OcrReader.load(device="cuda", batch_size=a.bs, version=a.version)
    print(f"[jmb] {len(crops)} crops, reader {a.version}, arms {a.arms}", flush=True)

    reads = {g["key"]: {"key": g["key"]} for g in gt}
    for arm in ("given", "rep"):  # ``given`` is always read: ``guard`` takes its reads
        if arm == "rep" and "rep" not in a.arms:
            continue
        ims = crops if arm == "given" else [rep_pad(c) for c in crops]
        t0, raws = time.time(), []
        for s in range(0, len(ims), 512):  # a line of progress every 512 crops
            raws += rd.read_raw_scored(ims[s : s + 512])
            print(
                f"[jmb] {arm} {len(raws)}/{len(ims)} "
                f"{len(raws) / (time.time() - t0):.1f} crops/s",
                flush=True,
            )
        torch.cuda.synchronize()
        for g, (text, conf) in zip(gt, raws, strict=True):
            reads[g["key"]] |= {arm: text, f"{arm}_conf": round(conf, 4)}
        if arm in a.arms:
            with (out / f"{arm}.jsonl").open("w", encoding="utf-8") as fh:
                for g, (text, _) in zip(gt, raws, strict=True):
                    fh.write(
                        json.dumps({"key": g["key"], "pred": text}, ensure_ascii=False)
                        + "\n"
                    )
        if arm == "given" and "guard" in a.arms:
            dropped = 0
            with (out / "guard.jsonl").open("w", encoding="utf-8") as fh:
                for g, c, (text, _) in zip(gt, crops, raws, strict=True):
                    kept = guard(text, int(c.shape[1]), int(c.shape[0]))
                    dropped += kept is None
                    fh.write(
                        json.dumps(
                            {"key": g["key"], "pred": kept or ""}, ensure_ascii=False
                        )
                        + "\n"
                    )
            print(f"[jmb] guard dropped {dropped} reads", flush=True)
    with (out / "reads.jsonl").open("w", encoding="utf-8") as fh:
        for r in reads.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[jmb] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
