#!/usr/bin/env python
"""Base sign — a G1 run's rows on the base model, in the sign frame.

G1 trains the rows on ``m15v11`` (4 steps, cfg 1) in the bubble frame. This probe
renders the same glyphs on another model of the ruler's table (``model``, e.g. ``base``:
the base DiT at the reads of record's sampler, VAE decode) with nothing but the rows on
it, each glyph in ``frame`` (``{t}`` is the glyph), at ``seeds``: once on the start
rows and once on a save of the run (``output/cjk_rl/g1/<run>/rows_<save>.pt``).

The read is G1's: the detector and the tuned reader 16 pages a call, hayai on the same
boxes, per page ``solo`` / ``on_page`` / ``extra``, the next glyph's ``solo`` (the
null), the reader's cross-entropy on the picked crop; the start rows against the save
in ``g1.tabulate``'s table. Sheets: per glyph the two arms as rows ``A`` / ``B`` in an
order drawn per glyph (``sheets/key.json`` holds it), and ``overview.png``, labelled, a
glyph a row.

The run's settings, question and gate are its config's header
(``configs/<handle>_<mmdd>_<hhmm>.toml``). Renders → ``output/cjk_rl/base_sign/<stem>/``,
result → ``results/<ts>-base_sign-<stem>/``.

    make daemon-run ARGS="--queue --stall-timeout 900 project/cjk_rl/probes/base_sign.py --config project/cjk_rl/configs/<name>.toml"
"""

from __future__ import annotations

import argparse
import json
import random
import time
import tomllib
from pathlib import Path

import glyph_g0 as G
import turbo_read as T

import g1  # noqa: E402  (src/, on the path through turbo_read)
import gd  # noqa: E402

ROOT = T.REPO / "output" / "cjk_rl" / "base_sign"
G1_OUT = T.REPO / "output" / "cjk_rl" / "g1"
THUMB = 512  # a sheet cell, px


def render(cfg: dict, gl: list, seeds: list, arms: dict, out: Path) -> dict:
    from common import models
    from eval.ruler.render import Renderer

    W, H = cfg["shape"]
    models.TOKEN_COUNTS[:] = [(H // 16) * (W // 16)]
    r = Renderer(cfg["model"])
    t0, n = time.time(), 0
    for arm, table in arms.items():
        r.set_arm(table)
        for _, key, _ in gl:
            for s in seeds:
                fn = out / arm / f"{key}_s{s}.png"
                n += not fn.exists()
                r.render(fn, cfg["frame"].replace("{t}", key), s, (W, H))
            print(f"  {arm} {key} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return {
        "renders": n,
        "s_per_render": round((time.time() - t0) / max(n, 1), 2),
        "steps": r.args.infer_steps,
        "cfg": r.args.guidance_scale,
        "dit": str(r.args.dit),
    }


def read(cfg: dict, gl: list, seeds: list, arms: list, out: Path) -> list:
    import numpy as np
    import torch
    from common.readers import crop, load_bgr
    from PIL import Image

    rdr, hy = gd.Reader("cuda"), gd.Hayai("cuda")
    recs = []
    for arm in arms:
        items = [
            (c, k, alt, s, out / arm / f"{k}_s{s}.png")
            for c, k, alt in gl
            for s in seeds
        ]
        for i in range(0, len(items), 16):
            chunk = items[i : i + 16]
            bgrs = [load_bgr(f) for *_, f in chunk]
            reads = rdr.rd.read_images(bgrs)
            own = [
                (b, x) for b, rs in enumerate(reads) for x in rs if not x.get("whole")
            ]
            crops = [
                Image.fromarray(
                    np.ascontiguousarray(crop(bgrs[b], x["box"])[:, :, ::-1])
                )
                for b, x in own
            ]
            for (_, x), text in zip(own, hy.read(crops)):
                x["hy"] = text
            for (cls, key, alt, s, f), bgr, rs in zip(chunk, bgrs, reads):
                H, W = bgr.shape[:2]
                case, x = gd.pick(key, rs, cfg["conf_min"])
                ce = None
                if x is not None:
                    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
                    p01 = torch.from_numpy(rgb).cuda().permute(2, 0, 1).float() / 255
                    with torch.no_grad():
                        box = gd.crop_box(x["box"], W, H)
                        ce = float(rdr.ce([p01], [bgr], [box], [key]))
                as_hy = [y | {"sfx": y.get("hy")} for y in rs]
                terms = G.page_terms(key, rs)
                recs.append(
                    {"arm": arm, "cls": cls, "key": key, "scene": 0, "seed": s}
                    | {"case": case, "ce": ce, "file": str(f)}
                    | {m: terms[m] for m in ("solo", "on_page", "extra", "r_seq")}
                    | {"null": G.page_terms(alt, rs)["solo"], "boxes": G.boxes(rs)}
                    | {
                        "hy_solo": G.page_terms(key, as_hy)["solo"],
                        "hy_boxes": G.boxes(as_hy),
                    }
                )
        print(f"  read {arm}", flush=True)
    return recs


def sheets(recs: list, arms: list, out: Path, seed: int) -> None:
    """Per glyph: row ``A`` then row ``B``, a seed a column."""
    from common.readers import contact_sheet
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    at = {(r["arm"], r["key"], r["seed"]): r for r in recs}
    seeds = sorted({r["seed"] for r in recs})
    key_of = {}
    for key in dict.fromkeys(r["key"] for r in recs):
        order = list(arms)
        rng.shuffle(order)
        key_of[key] = dict(zip("AB", order))
        rows = []
        for name, arm in zip("AB", order):
            for s in seeds:
                r = at[(arm, key, s)]
                rows.append(
                    (
                        Image.open(r["file"]).convert("RGB"),
                        [f"{key} {name} s{s}", " / ".join(r["boxes"])[:26]],
                    )
                )
        contact_sheet(rows, out / f"{key}.png", thumb=THUMB, cols=len(seeds))
    (out / "key.json").write_text(
        json.dumps(key_of, ensure_ascii=False, indent=1), "utf-8"
    )


def overview(recs: list, arms: list, path: Path) -> None:
    """One labelled sheet: a glyph a row, each arm's seeds side by side."""
    from common.readers import contact_sheet
    from PIL import Image

    at = {(r["arm"], r["key"], r["seed"]): r for r in recs}
    seeds = sorted({r["seed"] for r in recs})
    rows = [
        (
            Image.open(at[(arm, key, s)]["file"]).convert("RGB"),
            [f"{key} {arm} s{s}", " / ".join(at[(arm, key, s)]["boxes"])[:26]],
        )
        for key in dict.fromkeys(r["key"] for r in recs)
        for arm in arms
        for s in seeds
    ]
    contact_sheet(rows, path, thumb=THUMB, cols=len(arms) * len(seeds))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="its stem names the run")
    ap.add_argument("--smoke", action="store_true", help="two glyphs, two seeds")
    a = ap.parse_args()
    cfg = tomllib.loads(Path(a.config).read_text("utf-8"))
    name = ("smoke_" if a.smoke else "") + Path(a.config).stem

    import torch
    from eval.ruler.arms import tables

    gl = g1.glyphs(cfg, a.smoke)
    seeds = list(cfg["seeds"])[: 2 if a.smoke else None]
    save = cfg["save"]
    saved = torch.load(G1_OUT / cfg["run"] / f"rows_{save}.pt", weights_only=False)
    _, tabs = tables([cfg["rows"]])
    start = tabs[cfg["rows"]].float().cpu()
    # the save's own start rows are the ruler's table of `rows`: one unit, one id order
    d = float((saved["init"].float() - start).abs().max() / start.abs().max())
    assert saved["init"].shape == start.shape and d < 1e-2, (saved["init"].shape, d)
    moved = (saved["raw"].float() - saved["init"].float()).norm(dim=1) > 0
    print(
        f"{name}: {cfg['model']}, {len(gl)} glyphs × {len(seeds)} seeds × start / {save} "
        f"of {cfg['run']}; rows moved {int(moved.sum())}; start table off by {d:.1e}",
        flush=True,
    )
    arms = {"start": start, save: saved["raw"].float()}
    out = ROOT / name
    info = render(cfg, gl, seeds, arms, out)
    print(f"render: {info}", flush=True)
    recs = read(cfg, gl, seeds, list(arms), out)
    table = g1.tabulate(recs, list(arms))
    for k, v in table["per_glyph"].items():
        s, t = v["start"], v[save]
        print(
            f"  {k} solo {s['solo']:>2} → {t['solo']:>2} / {s['n']}  on_page "
            f"{s['on_page']:>2} → {t['on_page']:>2}  null {s['null']} → {t['null']}  "
            f"extra {s['extra']} → {t['extra']}  ce {s['ce']} → {t['ce']}  none "
            f"{s['none']} → {t['none']}  hayai solo {s['hy_solo']} → {t['hy_solo']}",
            flush=True,
        )
    for m, v in table["paired"].items():
        print(f"  paired {m:<8} {v}", flush=True)
    print(f"  total {table['total']}", flush=True)

    from bench._common import make_run_dir, write_result

    line = Path(__file__).resolve().parents[1]
    run_dir = make_run_dir("cjk_rl", label=f"base_sign-{name}", root=line / "results")
    sheets(recs, list(arms), run_dir / "sheets", cfg["shuffle_seed"])
    overview(recs, list(arms), run_dir / "overview.png")
    (run_dir / "records.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=1), "utf-8"
    )
    write_result(
        run_dir,
        script="project/cjk_rl/probes/base_sign.py",
        args=vars(a) | {"cfg": cfg, "out": str(out)},
        label=name,
        metrics={"render": info, "read": table},
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
