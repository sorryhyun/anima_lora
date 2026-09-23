"""ledger — ``runs/ledger.jsonl``: one line per submitted job (stage, tag,
steps, argv, job id, the pack the submit shell named)."""

from __future__ import annotations

import datetime as _dt
import json
import os

from .paths import LEDGER, RUNS


def append(**fields) -> dict:
    RUNS.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "vocab_pack": os.environ.get("ANIMA_VOCAB_PACK", ""),
        **fields,
    }
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def rows() -> list:
    if not LEDGER.exists():
        return []
    return [
        json.loads(ln)
        for ln in LEDGER.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
