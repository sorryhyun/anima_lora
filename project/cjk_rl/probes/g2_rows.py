#!/usr/bin/env python
"""G2 rows — a G1 run's read per row, and the rows sorted into the plan's groups (CPU).

Reads ``output/cjk_rl/g1/<run>/records.json`` (the read's pages, every arm) and
``log.jsonl`` (a line a row step). The arms are the run's own, in the order it read
them: ``start``, ``s00`` when the run began at a shared start, then each save; the last
is the trained arm, the one before it the earlier save.

Per row, on its ``n`` read pages an arm: ``solo`` under the tuned reader and under
hayai at every arm, ``on_page`` / ``extra`` / no-box pages / the picked crop's ce at the
start and at the trained arm, and the training log's ``exact`` count at each step.

Groups (``plan.md`` item 4; counts are of ``n`` pages, written here for n = 16 and
scaled when n differs):

- ``not_learned``: solo under 8 at the trained arm under both readers.
- ``past``: the tuned reader's solo at the trained arm under the earlier save's or the
  shared start's by 4 or more, or 4 or more pages where no box reads the glyph alone
  and a box reads it with other characters.
- ``reader``: the two readers' solo apart by 4 or more at the trained arm, or 4 or more
  pages where a box reads exactly a look-alike of the glyph under either reader
  (``LOOKALIKE``: small against full-size kana, the two scripts' shared shapes,
  katakana against kanji).

In a run with ``fold_small`` a small kana and its full-size glyph are one glyph: the
records' ``solo`` counts either read, the pair is not a look-alike, and a section lists
those rows' ``solo`` beside the count with the row's own glyph only.

For a row in any group: what its boxes read as at the trained arm, by reader, and a
labelled sheet, the start rows against the trained arm. ``--refs`` names earlier runs
whose records (``results/*-g1-<run>/records.json``) are printed for the glyphs they
share with this run.

Writes ``results/<ts>-g2_rows-<run>/`` (``rows.md``, ``result.json``, ``sheets/``).

    .venv/bin/python project/cjk_rl/probes/g2_rows.py --run <run> --refs g1shared_1125_1538 g1long_1125_1335
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

LINE = Path(__file__).resolve().parents[1]  # project/cjk_rl
REPO = LINE.parents[1]
G1_OUT = REPO / "output" / "cjk_rl" / "g1"

PAIRS = (
    # small against full-size
    "ぁあ ぃい ぅう ぇえ ぉお っつ ゃや ゅゆ ょよ "
    "ァア ィイ ゥウ ェエ ォオ ッツ ャヤ ュユ ョヨ ヵカ ヶケ "
    # the two scripts' shared shapes
    "へヘ べベ ぺペ りリ "
    # katakana against kanji
    "ニ二 ロ口 カ力 タ夕 エ工 ト卜 ハ八 ミ三 チ千"
).split()
LOOKALIKE: dict = defaultdict(set)
for _a, _b in PAIRS:
    LOOKALIKE[_a].add(_b)
    LOOKALIKE[_b].add(_a)


def records(run: str) -> list:
    f = G1_OUT / run / "records.json"
    if not f.exists():
        found = sorted((LINE / "results").glob(f"*-g1-{run}/records.json"))
        assert found, f"no records for {run}"
        f = found[-1]
    return json.loads(f.read_text("utf-8"))


def cell(xs: list) -> dict:
    ce = [r["ce"] for r in xs if r["ce"] is not None]
    return {
        "n": len(xs),
        "solo": sum(r["solo"] for r in xs),
        "hy_solo": sum(r.get("hy_solo", 0) for r in xs),
        "strict": sum(r.get("strict", r["solo"]) for r in xs),
        "hy_strict": sum(r.get("hy_strict", r.get("hy_solo", 0)) for r in xs),
        "on_page": sum(r["on_page"] for r in xs),
        "extra": round(sum(r["extra"] for r in xs) / len(xs), 2),
        "none": sum(r["case"] == "none" for r in xs),
        "ce": round(sum(ce) / len(ce), 2) if ce else None,
    }


def misreads(key: str, xs: list, field: str) -> tuple[Counter, int, int]:
    """Over a row's pages: the boxes that are not the glyph alone, the pages where a
    box holds the glyph with other characters and none reads it alone, and the pages
    where a box reads exactly a look-alike. The glyph is every read the run counted
    as it (a record's ``same``: a small kana and its full-size glyph in a run with
    ``fold_small``), and those are not its look-alikes."""
    seen, with_more, alike = Counter(), 0, 0
    for r in xs:
        ok = set(r.get("same") or [key])
        look = set().union(*(LOOKALIKE[k] for k in ok)) - ok
        bs = r.get(field) or []
        seen.update(b for b in bs if b not in ok)
        with_more += not ok & set(bs) and any(k in b for b in bs for k in ok)
        alike += any(b in look for b in bs)
    return seen, with_more, alike


def sheet(key: str, arms: list, at: dict, scenes: list, seeds: list, path: Path):
    from PIL import Image

    sys.path.insert(0, str(LINE / "probes"))
    import turbo_read  # noqa: F401  (puts the reseed line's packages on the path)
    from common.readers import contact_sheet

    rows = []
    for i in range(0, len(scenes), 2):
        for arm in arms:
            for si in scenes[i : i + 2]:
                for s in seeds:
                    r = at[(arm, key, si, s)]
                    rows.append(
                        (
                            Image.open(r["file"]).convert("RGB"),
                            [
                                f"{key} {arm} sc{si:02d} s{s}",
                                " / ".join(r["boxes"]),
                                "hy " + " / ".join(r.get("hy_boxes") or []),
                            ],
                        )
                    )
    contact_sheet(rows, path, thumb=256, cols=len(scenes[:2]) * len(seeds))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--refs", nargs="*", default=[])
    ap.add_argument("--no-sheets", action="store_true")
    a = ap.parse_args()

    recs = records(a.run)
    arms = list(dict.fromkeys(r["arm"] for r in recs))
    assert len(arms) >= 3, arms
    last, mid = arms[-1], arms[-2]
    shifted = "s00" if "s00" in arms else None
    keys = list(dict.fromkeys(r["key"] for r in recs))
    cls = {r["key"]: r["cls"] for r in recs}
    by = defaultdict(list)
    for r in recs:
        by[(r["arm"], r["key"])].append(r)
    exact = defaultdict(list)  # key → the training-side exact count at each step
    for ln in (G1_OUT / a.run / "log.jsonl").read_text("utf-8").splitlines():
        if ln:
            s = json.loads(ln)
            exact[s["key"]].append(s["exact"])

    rows, groups = {}, {"not_learned": [], "past": [], "reader": []}
    for k in keys:
        c = {arm: cell(by[(arm, k)]) for arm in arms}
        n = c[last]["n"]
        half, quarter = n / 2, n / 4
        seen, with_more, alike = misreads(k, by[(last, k)], "boxes")
        hy_seen, _, hy_alike = misreads(k, by[(last, k)], "hy_boxes")
        earlier = [c[mid]["solo"]] + ([c[shifted]["solo"]] if shifted else [])
        why = {
            "not_learned": c[last]["solo"] < half and c[last]["hy_solo"] < half,
            "past": max(earlier) - c[last]["solo"] >= quarter or with_more >= quarter,
            "reader": abs(c[last]["solo"] - c[last]["hy_solo"]) >= quarter
            or max(alike, hy_alike) >= quarter,
        }
        for g, hit in why.items():
            if hit:
                groups[g].append(k)
        rows[k] = {
            "cls": cls[k],
            "arms": c,
            "exact": exact[k],
            "with_more": with_more,
            "alike": alike,
            "hy_alike": hy_alike,
            "reads": seen.most_common(6),
            "hy_reads": hy_seen.most_common(6),
            "groups": [g for g, hit in why.items() if hit],
        }

    total = {
        arm: {
            c: {
                m: sum(rows[k]["arms"][arm][m] for k in keys if c in ("all", cls[k]))
                for m in ("n", "solo", "hy_solo", "on_page", "none")
            }
            for c in [*dict.fromkeys(cls.values()), "all"]
        }
        for arm in arms
    }

    def arrow(k: str, m: str) -> str:
        return " → ".join(str(rows[k]["arms"][arm][m]) for arm in arms)

    def line(k: str) -> str:
        v, s, t = rows[k], rows[k]["arms"][arms[0]], rows[k]["arms"][last]
        return (
            f"| {k} | {v['cls']} | {arrow(k, 'solo')} | {arrow(k, 'hy_solo')} | "
            f"{s['on_page']} → {t['on_page']} | {s['extra']} → {t['extra']} | "
            f"{s['none']} → {t['none']} | {s['ce']} → {t['ce']} | "
            f"{' '.join(map(str, v['exact']))} | {', '.join(v['groups'])} |"
        )

    head = (
        f"| glyph | class | solo {' → '.join(arms)} | hayai solo | on_page | extra | "
        "no box | ce | exact a step (training side) | groups |\n"
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    md = [f"# {a.run}: the read per row", ""]
    md += [
        "## Totals",
        "",
        "| arm | class | pages | solo | hayai solo | on_page | no box |",
    ]
    md += ["|---|---|---|---|---|---|---|"]
    for arm in arms:
        for c, v in total[arm].items():
            md.append(
                f"| {arm} | {c} | {v['n']} | {v['solo']} | {v['hy_solo']} | "
                f"{v['on_page']} | {v['none']} |"
            )
    for g, ks in groups.items():
        md += ["", f"## {g}: {len(ks)} of {len(keys)} rows", ""]
        if not ks:
            continue
        md += [head, *map(line, ks), ""]
        for k in ks:
            v = rows[k]
            md.append(
                f"- {k}: with other characters on {v['with_more']} pages; a look-alike "
                f"on {v['alike']} (hayai {v['hy_alike']}); boxes "
                f"{' '.join(f'{b}×{n}' for b, n in v['reads']) or '–'}; hayai "
                f"{' '.join(f'{b}×{n}' for b, n in v['hy_reads']) or '–'}"
            )
    folded = [k for k in keys if any("same" in r for r in by[(last, k)])]
    if folded:
        md += ["", "## Small and full-size kana, counted as one glyph", ""]
        md += [
            "`solo` in every table counts either read; here beside the count with the "
            "row's own glyph only.",
            "",
            f"| glyph | solo {' → '.join(arms)} | own glyph only | hayai solo | hayai, "
            "own glyph only |",
            "|---|---|---|---|---|",
        ]
        md += [
            f"| {k} | {arrow(k, 'solo')} | {arrow(k, 'strict')} | "
            f"{arrow(k, 'hy_solo')} | {arrow(k, 'hy_strict')} |"
            for k in folded
        ]
    md += ["", "## Every row", "", head, *map(line, keys)]

    for ref in a.refs:
        rr = [r for r in records(ref) if r["key"] in rows]
        if not rr:
            continue
        r_arms = list(dict.fromkeys(r["arm"] for r in rr))
        md += ["", f"## The glyphs shared with {ref}", ""]
        md += [
            f"| glyph | here, solo {' → '.join(arms)} | hayai | {ref}, solo {' → '.join(r_arms)} | hayai |"
        ]
        md += ["|---|---|---|---|---|"]
        tot = Counter()
        for k in dict.fromkeys(r["key"] for r in rr):
            cs = [
                cell([r for r in rr if r["key"] == k and r["arm"] == x]) for x in r_arms
            ]
            md.append(
                f"| {k} | {arrow(k, 'solo')} | {arrow(k, 'hy_solo')} | "
                f"{' → '.join(str(x['solo']) for x in cs)} | "
                f"{' → '.join(str(x['hy_solo']) for x in cs)} |"
            )
            for arm in arms:
                tot[("here", arm)] += rows[k]["arms"][arm]["solo"]
            for x, v in zip(r_arms, cs):
                tot[(ref, x)] += v["solo"]
        md.append(
            f"| all | {' → '.join(str(tot[('here', x)]) for x in arms)} | | "
            f"{' → '.join(str(tot[(ref, x)]) for x in r_arms)} | |"
        )

    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir("cjk_rl", label=f"g2_rows-{a.run}", root=LINE / "results")
    (run_dir / "rows.md").write_text("\n".join(md) + "\n", "utf-8")
    flagged = [k for k in keys if rows[k]["groups"]]
    if flagged and not a.no_sheets:
        at = {(r["arm"], r["key"], r["scene"], r["seed"]): r for r in recs}
        scenes = list(dict.fromkeys(r["scene"] for r in recs))
        seeds = sorted({r["seed"] for r in recs})
        (run_dir / "sheets").mkdir(exist_ok=True)
        for k in flagged:
            name = f"{rows[k]['cls']}_{k}.png"
            sheet(k, [arms[0], last], at, scenes, seeds, run_dir / "sheets" / name)
    write_result(
        run_dir,
        script="project/cjk_rl/probes/g2_rows.py",
        args=vars(a),
        label=a.run,
        metrics={"arms": arms, "total": total, "groups": groups, "rows": rows},
    )
    print("\n".join(md[: md.index("## Every row")]))
    print(f"→ {run_dir / 'rows.md'}  ({len(flagged)} sheets)")


if __name__ == "__main__":
    main()
