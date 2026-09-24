#!/usr/bin/env python3
"""Fetch the LoveHina manga-translation annotations and extract the JA side.

Source: https://github.com/PLippmann/multimodal-manga-translation (MIT), the
dataset released with Lippmann et al., "Context-Informed Machine Translation of
Manga using Multimodal Large Language Models" (COLING 2025, arXiv:2411.02589).
Professional Japanese->Polish translations of Love Hina vols 1 and 14.

We take the **Japanese side only**. The Polish target is useless to us (our
teacher needs an EN string), but the JA is exactly the register nothing else in
the corpus supplies: native, professionally written manga speech-bubble text
that no MT system produced. Its two jobs:

  1. a held-out **native-register eval set** -- the instrument for the
     register-mismatch risk, which MT-derived data cannot measure by
     construction (see the line's proposal, Risks);
  2. a quoted-text / short-utterance seed for the D6 templates.

Only the annotation JSON is fetched. The images live in Manga109-s, which
forbids redistribution and needs its own application -- irrelevant for the
text-only distillation, required only if Phase 4 (OCR + glyph rendering) ever
wants the pages.

    uv run python project/finished/cjk_aware_anima/datasets/lovehina.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/PLippmann/multimodal-manga-translation/main/LoveHina"
VOLUMES = ("LoveHina_vol01_fixed_order.json", "LoveHina_vol14_fixed_order.json")
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "assets" / "lovehina_ja.json"
CACHE_DIR = HERE / "assets" / ".lovehina"

KANA = re.compile(r"[぀-ヿ]")
KANJI = re.compile(r"[一-鿿]")
QUOTED = re.compile(r"[「『][^」』]*[」』]")


def fetch(name: str) -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / name
    if not cached.exists():
        print(f"  fetching {name}", file=sys.stderr)
        with urllib.request.urlopen(f"{RAW}/{name}", timeout=120) as r:
            cached.write_bytes(r.read())
    return json.loads(cached.read_text())


def extract(pages: list[dict], volume: str) -> list[dict]:
    """One record per speech-bubble line, keeping page + reading order + bbox.

    Reading order matters: consecutive lines on a page are a conversation, so
    they can be concatenated into multi-utterance samples later. bbox is kept
    only so Phase 4 can align to Manga109-s pages if it ever gets the images.
    """
    out = []
    for page in pages:
        idx = page.get("attrib", {}).get("index")
        for line in page.get("lines", []):
            ja = (line.get("text_jp") or "").strip()
            if not ja:
                continue
            a = line.get("attrib", {})
            out.append(
                {
                    "volume": volume,
                    "page": idx,
                    "read_order": line.get("read_order"),
                    "ja": ja,
                    "bbox": [
                        a.get("xmin"),
                        a.get("ymin"),
                        a.get("xmax"),
                        a.get("ymax"),
                    ],
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    records: list[dict] = []
    for name in VOLUMES:
        vol = name.split("_")[1]
        records.extend(extract(fetch(name), vol))

    text = "".join(r["ja"] for r in records)
    quoted = [r for r in records if QUOTED.search(r["ja"])]
    stats = {
        "source": "PLippmann/multimodal-manga-translation (MIT)",
        "lines": len(records),
        "total_chars": len(text),
        "mean_chars_per_line": round(len(text) / max(1, len(records)), 1),
        "unique_kanji": len(set(KANJI.findall(text))),
        "unique_kana": len(set(KANA.findall(text))),
        "lines_with_corner_quotes": len(quoted),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"stats": stats, "lines": records}, ensure_ascii=False, indent=1)
    )
    for k, v in stats.items():
        print(f"{k}: {v}", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
