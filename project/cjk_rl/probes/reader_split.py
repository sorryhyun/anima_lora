#!/usr/bin/env python
"""reader_split — the crops the two readers split on, set aside (forward only; no
render, no update).

From the ``records.json`` of ``src/g1.py`` runs read with hayai's term: the pages where
one reader's ``solo`` holds and the other's does not, each page file once. Those pages
are read again from disk by ``reader_pair.run`` (AnimeText boxes, the tuned reader and
hayai v2.7 on every box's crop, hayai's confidence from a teacher-forced forward).

A box is kept when one reader reads the asked glyph and the other does not, on the
fresh reads; with ``fold_small`` a small kana and its full-size glyph are one glyph
(``g1.same``). Its crop (``common.readers.crop``, 12 % a side: what the readers saw) is
written as a PNG, and ``manifest.jsonl`` holds a line a crop: ``file``, ``run``,
``arm``, ``key`` (the asked glyph: what the caption asked for, not a checked label),
``accept``, ``case`` (``tuned_only`` | ``hayai_only``), both reads and confidences,
``near`` (the reader that missed read a look-alike, ``g1.near``), ``box`` and the page
file.

The run's settings are its config's header (``configs/<handle>_<mmdd>_<hhmm>.toml``).
Writes ``output/cjk_rl/reader_split/<config stem>/`` (``crops/``, ``manifest.jsonl``,
``reads.json``, ``select_<run>.json``) and ``results/<ts>-reader_split-<config stem>/``.

    make daemon-run ARGS="--queue --stall-timeout 600 project/cjk_rl/probes/reader_split.py --config project/cjk_rl/configs/<name>.toml"
    .venv/bin/python project/cjk_rl/probes/reader_split.py --config <cfg> --count            # CPU: the selection's counts
    .venv/bin/python project/cjk_rl/probes/reader_split.py --config <cfg> --export <dir>     # CPU: crops from a reads.json
"""

from __future__ import annotations

import argparse
import json
import tomllib
from collections import Counter
from pathlib import Path

import turbo_read as T

import g1  # noqa: E402  (src/, on the path through turbo_read)

ROOT = T.REPO / "output" / "cjk_rl" / "reader_split"


def select(run: str, seen: set) -> list:
    """``run``'s pages with one reader's ``solo`` and not the other's; a file in
    ``seen`` (an earlier run's, as a ``start_from`` arm is) is left out."""
    recs = json.loads((g1.ROOT / run / "records.json").read_text("utf-8"))
    out = []
    for r in recs:
        if "hy_solo" not in r or r["solo"] == r["hy_solo"] or r["file"] in seen:
            continue
        seen.add(r["file"])
        out.append({k: r[k] for k in ("arm", "key", "scene", "seed", "file")})
    return out


def export(pages: list, out: Path, fold: bool) -> dict:
    """Writes the split boxes' crops and ``manifest.jsonl``; returns the counts."""
    from common.readers import crop, load_bgr
    from common.text import norm
    from PIL import Image

    (out / "crops").mkdir(parents=True, exist_ok=True)
    n = Counter()
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as mf:
        for p in pages:
            ok = [norm(x) for x in g1.same(p["key"], fold)]
            nr = [norm(x) for x in g1.near(p["key"], fold)]
            bgr = None
            for b, r in enumerate(x for x in p["reads"] if not x.get("whole")):
                v, h = norm(r["sfx"] or ""), norm(r.get("hy") or "")
                if (v in ok) == (h in ok):
                    continue
                case = "tuned_only" if v in ok else "hayai_only"
                missed = h if v in ok else v
                if bgr is None:
                    bgr = load_bgr(Path(p["file"]))
                name = f"{p['set']}_{p['arm']}_{Path(p['file']).stem}_b{b}.png"
                Image.fromarray(crop(bgr, r["box"])[:, :, ::-1]).save(
                    out / "crops" / name
                )
                row = {
                    "file": f"crops/{name}",
                    "run": p["set"],
                    "arm": p["arm"],
                    "key": p["key"],
                    "accept": g1.same(p["key"], fold),
                    "case": case,
                    "tuned": r["sfx"],
                    "tuned_conf": r["sfx_conf"],
                    "hayai": r.get("hy"),
                    "hayai_conf": r.get("hy_conf"),
                    "near": missed in nr,
                    "box": r["box"],
                    "page": p["file"],
                }
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")
                n[case] += 1
                n[f"{case}_near"] += int(row["near"])
                n[f"{p['set']}/{case}"] += 1
            n["pages"] += 1
            n["pages_with_a_crop"] += int(bgr is not None)
    return dict(n)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True, help="its stem names the run")
    ap.add_argument("--count", action="store_true", help="CPU: the selection's counts")
    ap.add_argument("--export", help="CPU: crops from this directory's reads.json")
    ap.add_argument("--smoke", action="store_true", help="32 pages a run")
    a = ap.parse_args()

    cfg = tomllib.loads(Path(a.config).read_text("utf-8"))
    fold = bool(cfg.get("fold_small"))
    name = ("smoke_" if a.smoke else "") + Path(a.config).stem
    out = ROOT / name
    if a.export:
        pages = json.loads((Path(a.export) / "reads.json").read_text("utf-8"))
        print(export(pages, out, fold), f"→ {out}", flush=True)
        return
    seen: set = set()
    picked = {run: select(run, seen) for run in cfg["runs"]}
    for run, ps in picked.items():
        print(
            f"  {run}: {len(ps)} pages split, by arm {dict(Counter(p['arm'] for p in ps))}"
        )
    if a.count:
        return

    import reader_pair as RP
    from bench._common import make_run_dir, write_result

    out.mkdir(parents=True, exist_ok=True)
    sets = []
    for run, ps in picked.items():
        f = out / f"select_{run}.json"
        f.write_text(json.dumps(ps, ensure_ascii=False), "utf-8")
        sets.append({"name": run, "records": str(f)})
    info = RP.run(
        {"sets": sets, "hayai": cfg["hayai"], "revision": cfg["revision"]}, a.smoke, out
    )
    pages = json.loads((out / "reads.json").read_text("utf-8"))
    counts = export(pages, out, fold)
    print(f"  {counts}", flush=True)
    run_dir = make_run_dir(
        "cjk_rl", label=f"reader_split-{name}", root=T.LINE / "results"
    )
    write_result(
        run_dir,
        script="project/cjk_rl/probes/reader_split.py",
        args=vars(a) | {"cfg": cfg, "out": str(out)},
        label=name,
        metrics={
            "run": info,
            "selected": {k: len(v) for k, v in picked.items()},
            "crops": counts,
        },
    )
    print(f"→ {out / 'manifest.jsonl'}\n→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
