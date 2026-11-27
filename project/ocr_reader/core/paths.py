"""Paths the OCR line reads and writes. Manga109-s roots are in :mod:`.manga109`
(``ANIMA_MANGA109S_ROOT``); the other external datasets take an ``ANIMA_*_ROOT`` override."""

from __future__ import annotations

import os
from pathlib import Path

from library.env import anima_home

REPO = anima_home()
LINE = REPO / "project/ocr_reader"
ASSETS = LINE / "assets"
EVAL_DATA = LINE / "eval_data"
EVAL_REPORTS = LINE / "reports/eval"
"""Generated eval tables (``ocr_eval_<name>.md``), local."""

OUTPUT = REPO / "output/ocr"
"""Runs (``output/ocr/<run>/``) and every script's working output below it."""
EVAL_OUT = OUTPUT / "eval"
"""Per-row eval reads (``<name>_<split>.jsonl``, ``sfx_<name>.jsonl`` …)."""

VL16_BASE = REPO / "models/paddleocr_vl_1.6"
OVIS3_BASE = REPO / "models/ovis_ocr3"  # ATH-MaaS/OvisOCR3 @ 4833d5d

LINE_ANNOTATIONS = ASSETS / "manga109s_line_annotations.json"
"""``bluolightning/manga109s-line-annotations`` (HF, MIT; geometry only): per Manga109-s
v2026 ``<text>``, ``line_lengths`` = characters per line in reading order, counted on
the XML text without its ``\\n`` (spaces included)."""


def _root(env: str, default: str) -> Path:
    return Path(os.environ.get(env) or default).expanduser()


GELNOTE = _root("ANIMA_GELNOTE_ROOT", "/media/sorryhyun/new/dataset/gelnote_crops")
SWAPRELAX = _root(
    "ANIMA_SWAPRELAX_ROOT", "/media/sorryhyun/new/dataset/hito_ocr/swap_relax_v5"
)
HITO = _root("ANIMA_HITO_ROOT", "/media/sorryhyun/new/dataset/hito_ocr/ko")
ANIMA_LOWCONF = _root(
    "ANIMA_LOWCONF_ROOT", "/media/sorryhyun/new/dataset/anima_ocr_lowconf"
)
"""The Anima training set's low-confidence v7 reads as a label set
(``labeling/anima_lowconf.py``): ``rows.jsonl`` and ``crops/``."""
