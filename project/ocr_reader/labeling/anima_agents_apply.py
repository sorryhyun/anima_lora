#!/usr/bin/env python3
"""The agents' majority labels (``label_agent.md``, three agents per batch under
``output/ocr/anima_lowconf/agents/``) → ``assets/anima_mainsub.tsv`` with status ``agent``, for
the ``anima_lowconf`` rows that file does not hold yet. A row is written when at least two
agents gave the same main (whitespace-blind, ``review_page.vote``); a split row is left for
review. Rows the user reviewed (status ``ok`` / ``drop``) are never touched; the ``agent``
status keeps the agent labels apart from them.

    python project/ocr_reader/labeling/anima_agents_apply.py [--dry_run]
"""

from __future__ import annotations

import argparse

import pandas as pd

from project.ocr_reader.core.paths import ASSETS, OUTPUT
from project.ocr_reader.labeling.review_page import agent_votes, vote


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--agents", default=OUTPUT / "anima_lowconf/agents")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    path = ASSETS / "anima_mainsub.tsv"
    cols = ["source", "id", "status", "main", "sub"]
    old = (
        pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
        if path.exists()
        else pd.DataFrame(columns=cols)
    )
    have = set(old.id)
    esc = lambda t: t.replace("\n", "\\n")  # noqa: E731
    rows, split = [], 0
    for rid, vs in agent_votes(a.agents).items():
        if rid in have:
            continue
        main, sub, _ = vote(vs)
        if main is None:
            split += 1
            continue
        rows.append(
            dict(
                source="anima_lowconf",
                id=rid,
                status="agent",
                main=esc(main),
                sub=esc(sub),
            )
        )
    out = pd.concat([old, pd.DataFrame(rows, columns=cols)], ignore_index=True)
    print(
        f"[anima_agents] {len(rows)} agent rows written ({sum(r['main'] == '' for r in rows)} "
        f"empty main), {split} split rows left for review; file {len(old)} → {len(out)} rows"
    )
    if not a.dry_run:
        out.to_csv(path, sep="\t", index=False, quoting=3)


if __name__ == "__main__":
    main()
