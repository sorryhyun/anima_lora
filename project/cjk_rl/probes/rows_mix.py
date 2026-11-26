#!/usr/bin/env python
"""Rows mix — one G1 save with some rows taken from other saves (CPU).

Reads the config's ``[base]`` save (``run``, ``save``) under ``output/cjk_rl/g1/`` and
replaces the rows of each ``[[take]]`` (``run``, ``save``, ``glyphs``) with that save's.
The result keeps the base's ``init`` (so a row's displacement counts from the base's
start rows) and is written as ``output/cjk_rl/g1/<config stem>/rows_mix.pt`` in the
format of ``src/g1.py``'s saves: ``[start_shift]`` reads it with ``save = "mix"`` and
``scripts/toolkits/bake_vocab_pack.py`` bakes it as any glyph-stage save. ``sources.json``
beside it names every row's run and save.

Prints the mean displacement of the mixed rows, in row norms, and its cosine with each
``refs`` save's (``[[refs]]``: ``run``, ``save``).

    .venv/bin/python project/cjk_rl/probes/rows_mix.py --config project/cjk_rl/configs/<handle>.toml
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
G1_OUT = REPO / "output" / "cjk_rl" / "g1"


def load(run: str, save: str) -> dict:
    import torch

    return torch.load(
        G1_OUT / run / f"rows_{save}.pt", map_location="cpu", weights_only=False
    )


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = tomllib.loads(args.config.read_text())
    handle = args.config.stem

    base = load(cfg["base"]["run"], cfg["base"]["save"])
    rows, raw = base["rows"], base["raw"].clone()
    steps = dict(base["steps"])
    src = {k: f"{cfg['base']['run']}/{cfg['base']['save']}" for k in rows}
    for t in cfg.get("take", []):
        sv = load(t["run"], t["save"])
        assert list(sv["ext_ids"]) == list(base["ext_ids"]), t["run"]
        for k in t["glyphs"]:
            assert sv["rows"][k] == rows[k], (k, sv["rows"][k], rows[k])
            assert src[k].startswith(cfg["base"]["run"]), f"{k} taken twice"
            raw[rows[k]] = sv["raw"][rows[k]]
            steps[k] = sv["steps"][k]
            src[k] = f"{t['run']}/{t['save']}"

    out = G1_OUT / handle
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ext_ids": base["ext_ids"],
            "rows": rows,
            "raw": raw,
            "init": base["init"],
            "row_scale": base["row_scale"],
            "steps": steps,
            "cfg": cfg | {"rows": (base.get("cfg") or {}).get("rows")},
        },
        out / "rows_mix.pt",
    )
    (out / "sources.json").write_text(json.dumps(src, ensure_ascii=False, indent=1))

    rs = float(base["row_scale"])
    init = base["init"].float()
    idx = list(rows.values())
    disp = raw.float()[idx] - init[idx]
    mean = disp.mean(0)
    taken = sum(len(t["glyphs"]) for t in cfg.get("take", []))
    print(f"{out / 'rows_mix.pt'}: {len(rows)} rows, {taken} taken")
    print(
        f"row displacement median {float(disp.norm(dim=1).median()) / rs:.3f}, "
        f"mean displacement {float(mean.norm()) / rs:.3f} row norms"
    )
    for r in cfg.get("refs", []):
        sv = load(r["run"], r["save"])
        i0 = sv["init"].float()
        j = list(sv["rows"].values())
        m = (sv["raw"].float()[j] - i0[j]).mean(0)
        c = float(torch.nn.functional.cosine_similarity(mean, m, dim=0))
        print(
            f"  against {r['run']}/{r['save']} ({len(j)} rows, "
            f"{float(m.norm()) / rs:.3f}): cosine {c:.3f}"
        )


if __name__ == "__main__":
    main()
