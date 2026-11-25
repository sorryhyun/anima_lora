#!/usr/bin/env python
"""Row line — a G1 run's saved rows moved along what they learned, read with no training.

Each arm is the start rows with the run's glyphs' rows displaced, read as G1 reads a
save (``g1.read``: the config's glyphs on its ``read_scenes`` and seeds in its ``frame``,
the tuned reader and hayai on the same boxes). An arm in ``[arms]`` names a save of the run
and

- ``scale`` (default 1): a row sits at its start plus ``scale`` times its own
  displacement to that save. ``scale = 2`` on the 8-step save is the 8 steps taken at
  twice their length with the gradients left where they were measured; ``scale = 1`` is
  the save read again.
- ``loo = true``: a row sits at its start plus the mean displacement of the run's
  other rows to that save (leave one out); its own training is not in it.
- ``rand = true``: the arm's vector is swapped for a drawn direction of its norm, a row
  its own draw (the null of the arm it is set on).
- ``saved = true``: a row sits where the save holds it, a shifted start included (the
  other keys measure from the save's own start, which a ``[start_shift]`` run moved).
- no ``save``: the start rows, read in this config's frame and scenes.

``en`` in the config is a list of EN words read beside the glyphs (class ``en``, the
null the next word). A word holds no trained row, so every arm reads it on the same rows.

The run's own reads (``results/*-g1-<run>/records.json``: the start rows and each save,
the same pages) are the references. Tables: ``pairs`` in the config, each a reference
or arm against an arm in ``g1.tabulate``'s table. Sheets: per pair and glyph, the two
as rows ``A`` / ``B`` in an order drawn per glyph (``sheets_<a>_<b>/key.json``).

The run's settings, question and gate are its config's header
(``configs/<handle>_<mmdd>_<hhmm>.toml``). Pages → ``output/cjk_rl/row_line/<stem>/``,
result → ``results/<ts>-row_line-<stem>/``.

    make daemon-run ARGS="--queue --stall-timeout 900 project/cjk_rl/probes/row_line.py --config project/cjk_rl/configs/<name>.toml"
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import glyph_g0 as G
import glyph_scenes as S
import turbo_read as T

import g1  # noqa: E402  (src/, on the path through turbo_read)
import gd  # noqa: E402

ROOT = T.REPO / "output" / "cjk_rl" / "row_line"
G1_OUT = T.REPO / "output" / "cjk_rl" / "g1"
LINE = Path(__file__).resolve().parents[1]


def tables(cfg: dict, gl: list, start) -> tuple:
    """``({arm: table}, {arm: {key: displacement in row norms}})``."""
    import torch

    saves, out, moved = {}, {}, {}
    for arm, spec in cfg["arms"].items():
        if "save" not in spec:
            out[arm], moved[arm] = start.clone(), {k: 0.0 for _, k, _ in gl}
            continue
        if spec["save"] not in saves:
            saves[spec["save"]] = torch.load(
                G1_OUT / cfg["run"] / f"rows_{spec['save']}.pt", weights_only=False
            )
        sv = saves[spec["save"]]
        init = sv["init"].float()
        row = {k: sv["rows"][k] for _, k, _ in gl if k in sv["rows"]}
        if spec.get("saved"):
            table = start.clone()
            for k, i in row.items():
                table[i] = sv["raw"][i].float()
            out[arm] = table
            moved[arm] = {
                k: round(float((table[i] - start[i]).norm()) / sv["row_scale"], 4)
                for k, i in row.items()
            }
            continue
        d = float((init - start).abs().max() / start.abs().max())
        assert init.shape == start.shape and d < 1e-2, (init.shape, d)
        disp = {k: sv["raw"][i].float() - init[i] for k, i in row.items()}
        table = start.clone()
        moved[arm] = {}
        for k, i in row.items():
            if spec.get("loo"):
                others = [  # the run's other rows, all of them, also under --smoke
                    sv["raw"][i2].float() - init[i2]
                    for j, i2 in sv["rows"].items()
                    if j != k
                ]
                v = torch.stack(others).mean(0)
            else:
                v = disp[k]
            if spec.get("rand"):  # a drawn direction at this vector's norm
                g = torch.Generator().manual_seed(cfg["shuffle_seed"] + int(i))
                n = torch.randn(v.shape, generator=g)
                v = n * (v.norm() / n.norm())
            v = spec.get("scale", 1.0) * v
            table[i] = start[i] + v
            moved[arm][k] = round(float(v.norm()) / sv["row_scale"], 4)
        out[arm] = table
    return out, moved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="its stem names the run")
    ap.add_argument("--smoke", action="store_true", help="two glyphs, one scene")
    a = ap.parse_args()
    cfg = tomllib.loads(Path(a.config).read_text("utf-8"))
    name = ("smoke_" if a.smoke else "") + Path(a.config).stem

    sc = S.scenes()
    scenes = list(cfg["read_scenes"])[: 1 if a.smoke else None]
    gl = g1.glyphs(cfg, a.smoke)
    en = list(cfg.get("en", []))[: 1 if a.smoke else None]
    gl += [("en", w, en[(n + 1) % len(en)]) for n, w in enumerate(en)]
    keys = {k for _, k, _ in gl}
    (ref_file,) = sorted((LINE / "results").glob(f"*-g1-{cfg['run']}/records.json"))
    ref = [
        r
        for r in json.loads(ref_file.read_text("utf-8"))
        if r["key"] in keys and r["scene"] in scenes and r["seed"] in cfg["read_seeds"]
    ]
    ref_arms = list(dict.fromkeys(r["arm"] for r in ref))
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)

    pol = gd.Policy([G.SHAPE], rows=cfg["rows"], grad_bs=1)
    pol.encode([G.framed(sc[i][cfg["frame"]], k) for _, k, _ in gl for i in scenes])
    rdr, hy = gd.Reader("cuda"), gd.Hayai("cuda")
    start = pol.raw.detach().float().cpu().clone()
    arms, moved = tables(cfg, gl, start)
    print(
        f"{name}: {len(gl)} glyphs × scenes {scenes} × seeds {cfg['read_seeds']}; arms "
        f"{list(arms)}; references {ref_arms} of {cfg['run']} ({len(ref)} pages)",
        flush=True,
    )
    for arm, m in moved.items():
        v = sorted(m.values())
        print(f"  {arm}: rows moved {v[0]} … {v[-1]} row norms", flush=True)
    recs = ref + g1.read(pol, rdr, cfg, gl, sc, scenes, arms, out, hy)

    every = ref_arms + list(arms)
    total = {}
    for arm in every:
        xs = [r for r in recs if r["arm"] == arm]
        total[arm] = {
            "n": len(xs),
            "solo": sum(r["solo"] for r in xs),
            "hy_solo": sum(r["hy_solo"] for r in xs),
            "on_page": sum(r["on_page"] for r in xs),
            "none": sum(r["case"] == "none" for r in xs),
            "null": sum(r["null"] for r in xs),
            "extra": round(sum(r["extra"] for r in xs) / len(xs), 2),
        }
        print(f"  total {arm:<6} {total[arm]}", flush=True)
    print("  solo of a glyph's pages, tuned (hayai): " + " | ".join(every), flush=True)
    per = {}
    for _, k, _ in gl:
        per[k] = {
            arm: [
                sum(r[m] for r in recs if r["arm"] == arm and r["key"] == k)
                for m in ("solo", "hy_solo")
            ]
            for arm in every
        }
        print(
            f"  {k} "
            + " | ".join(f"{per[k][x][0]:>2} ({per[k][x][1]:>2})" for x in every),
            flush=True,
        )

    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir("cjk_rl", label=f"row_line-{name}", root=LINE / "results")
    paired = {}
    for x, y in cfg["pairs"]:
        pair = [r for r in recs if r["arm"] in (x, y)]
        paired[f"{x}_{y}"] = g1.tabulate(pair, [x, y])
        for m in ("solo", "hy_solo", "extra", "none", "ce"):
            print(f"  {x} → {y} paired {m:<8} {paired[f'{x}_{y}']['paired'][m]}")
        g1.sheets(
            pair, [x, y], scenes, run_dir / f"sheets_{x}_{y}", cfg["shuffle_seed"]
        )
    (run_dir / "records.json").write_text(
        json.dumps([r for r in recs if r["arm"] in arms], ensure_ascii=False, indent=1),
        "utf-8",
    )
    write_result(
        run_dir,
        script="project/cjk_rl/probes/row_line.py",
        args=vars(a) | {"cfg": cfg, "model": T.MODEL, "out": str(out)},
        label=name,
        metrics={
            "moved": moved,
            "total": total,
            "per_glyph": per,
            "paired": paired,
            "references": str(ref_file),
        },
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
