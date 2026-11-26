#!/usr/bin/env python
"""Template read — pool templates of several dialogue strings, their strings swapped for
short ones, read with no training.

``build`` (CPU) draws the templates and writes the captions; ``run`` renders and reads
them. A template is a pool item (``output/cjk_rl/pool/pool.json``, ``src/pool.py``: a
training caption whose own text was swapped out) of track ``dialogue`` with ``strings``
dialogue strings and no SFX clause, at one of the ``shapes`` commonest shapes. Nothing in
the caption changes but the quoted strings. Per template and kind (``kana``: the config's
``kana_file``; ``en``: its ``en`` words), a pair ``A``, ``B`` of the kind's list gives four
captions: the clause holding ``"A", "B"``, ``"B", "A"``, ``"A"`` alone and ``"B"`` alone.

A page is rendered at the template's shape on the config's rows (``[arm]``, as
``row_line.py`` names a save) and read by the tuned reader and hayai. Records have
``row_line.py``'s shape with ``arm`` the kind, ``scene`` the template's index and ``cls``
``multi`` (two strings) or ``word`` (one), so ``multi_read.py`` counts them with the
result as its own ``--alone``. Sheets: a file a template, a row a caption.

The run's settings, question and gate are its config's header. Captions →
``output/cjk_rl/pool/templ_<stem>.json`` (local: training captions), pages →
``output/cjk_rl/templ_read/<stem>/``, result → ``results/<ts>-templ_read-<stem>/``.

    .venv/bin/python project/cjk_rl/probes/templ_read.py build --config project/cjk_rl/configs/<name>.toml
    make daemon-run ARGS="--queue --stall-timeout 900 project/cjk_rl/probes/templ_read.py run --config project/cjk_rl/configs/<name>.toml"
"""

from __future__ import annotations

import argparse
import json
import random
import time
import tomllib
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LINE = Path(__file__).resolve().parents[1]
POOL = REPO / "output" / "cjk_rl" / "pool"
CLAUSE = "Japanese text reads as "
SEP = '", "'  # g1.SEP


def clause(strings: list) -> str:
    return CLAUSE + ", ".join(f'"{s}"' for s in strings) + "."


def build(cfg: dict, name: str) -> Path:
    items = json.loads((POOL / "pool.json").read_text("utf-8"))["items"]
    n = cfg["strings"]
    seen, tpl = set(), []
    for m in items:
        if (
            m["track"] == "dialogue"
            and len(m["texts"]) == n
            and not m["sfx"]
            and m["stem"] not in seen
            and m["caption"].endswith(clause(m["texts"]))
        ):
            seen.add(m["stem"])
            tpl.append(m)
    top = [s for s, _ in Counter(tuple(m["shape"]) for m in tpl).most_common()]
    top = top[: cfg["shapes"]]
    tpl = [m for m in tpl if tuple(m["shape"]) in top]
    rng = random.Random(cfg["shuffle_seed"])
    tpl = rng.sample(tpl, cfg["templates"])
    kana = json.loads((REPO / cfg["kana_file"]).read_text("utf-8"))
    kana = [p for p in kana if len(p) == n][: cfg["kana_pairs"]]
    en = [cfg["en"][i : i + n] for i in range(0, len(cfg["en"]), n)]
    out = []
    for t, m in enumerate(tpl):
        head = m["caption"][: -len(clause(m["texts"]))]
        for kind, pairs in (("kana", kana), ("en", en)):
            a, b = pairs[t % len(pairs)]
            for slot, strings in (
                ("AB", [a, b]),
                ("BA", [b, a]),
                ("A", [a]),
                ("B", [b]),
            ):
                out.append(
                    {
                        "t": t,
                        "stem": m["stem"],
                        "shape": m["shape"],
                        "was": [len(s) for s in m["texts"]],
                        "kind": kind,
                        "slot": slot,
                        "strings": strings,
                        "caption": head + clause(strings),
                    }
                )
    f = POOL / f"templ_{name}.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    print(
        f"{len(tpl)} templates of {len(seen)} with {n} strings and no SFX clause; "
        f"shapes {dict(Counter(tuple(m['shape']) for m in tpl))}; {len(out)} captions; "
        f"the templates' own string lengths {sorted(len(s) for m in tpl for s in m['texts'])}"
    )
    print(f"→ {f}")
    return f


def run(cfg: dict, name: str, smoke: bool) -> None:
    import glyph_g0 as G
    import multi_read as M
    import numpy as np
    import row_line as RL
    import torch
    import turbo_read as T
    from common.readers import contact_sheet, crop
    from PIL import Image

    import gd

    caps = json.loads(
        (POOL / f"templ_{Path(name).name.removeprefix('smoke_')}.json").read_text(
            "utf-8"
        )
    )
    if smoke:
        caps = [c for c in caps if c["t"] == 0]
    seeds = list(cfg["read_seeds"])
    out = T.REPO / "output" / "cjk_rl" / "templ_read" / name
    shapes = sorted({tuple(c["shape"]) for c in caps})
    pol = gd.Policy(shapes, rows=cfg["rows"], grad_bs=1)
    pol.encode([c["caption"] for c in caps])
    rdr, hy = gd.Reader("cuda"), gd.Hayai("cuda")
    start = pol.raw.detach().float().cpu().clone()
    gl = [
        ("multi" if len(c["strings"]) > 1 else "word", SEP.join(c["strings"]), None)
        for c in caps
        if c["kind"] == "kana"
    ]
    tabs, moved = RL.tables({"arms": {"rows": cfg["arm"]}} | cfg, gl, start)
    pol.raw.data.copy_(tabs["rows"])
    v = sorted(moved["rows"].values())
    print(
        f"{name}: {len(caps)} captions × seeds {seeds}, shapes {shapes}; rows moved "
        f"{v[0]} … {v[-1]} row norms",
        flush=True,
    )
    recs, t0 = [], time.time()
    for n, c in enumerate(caps):
        if n and n % 32 == 0:
            print(
                f"  caption {n} / {len(caps)} ({(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )
        _, px = pol.rollout(
            pol.noise(seeds, tuple(c["shape"])), pol.embed(c["caption"])
        )
        bgrs, reads = rdr.read(px)
        own = [(b, r) for b, rs in enumerate(reads) for r in rs if not r.get("whole")]
        crops = [
            Image.fromarray(np.ascontiguousarray(crop(bgrs[b], r["box"])[:, :, ::-1]))
            for b, r in own
        ]
        for (_, r), text in zip(own, hy.read(crops) if crops else []):
            r["hy"] = text
        for b, seed in enumerate(seeds):
            f = out / c["kind"] / f"t{c['t']:02d}_{c['slot']}_s{seed}.webp"
            f.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(bgrs[b][:, :, ::-1]).save(f, quality=95)
            recs.append(
                {
                    "arm": c["kind"],
                    "cls": "multi" if len(c["strings"]) > 1 else "word",
                    "key": SEP.join(c["strings"]),
                    "scene": c["t"],
                    "seed": seed,
                    "slot": c["slot"],
                    "stem": c["stem"],
                    "was": c["was"],
                    "file": str(f),
                    "boxes": G.boxes(reads[b]),
                    "hy_boxes": G.boxes([x | {"sfx": x.get("hy")} for x in reads[b]]),
                }
            )
    torch.cuda.synchronize()
    print(f"  read ({(time.time() - t0) / 60:.1f} min)", flush=True)

    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir("cjk_rl", label=f"templ_read-{name}", root=LINE / "results")
    (run_dir / "records.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=1), "utf-8"
    )
    per = {}  # a template: its pair pages, a kind
    for t in sorted({r["scene"] for r in recs}):
        per[t] = {"stem": next(r["stem"] for r in recs if r["scene"] == t)}
        for kind in ("kana", "en"):
            xs = [
                r
                for r in recs
                if r["scene"] == t and r["arm"] == kind and r["cls"] == "multi"
            ]
            ps = [M.page(r["key"].split(SEP), r["boxes"], False) for r in xs]
            per[t][kind] = {
                "pages": len(xs),
                "all": sum(p["all"] for p in ps),
                "own": sum(sum(p["own"]) for p in ps),
                "boxes": round(sum(p["boxes"] for p in ps) / len(ps), 2),
            }
        print(f"  template {t:02d} {per[t]}", flush=True)
    sheets = run_dir / "sheets"
    sheets.mkdir()
    for t in per:
        rows = [
            (
                Image.open(r["file"]).convert("RGB"),
                [
                    f"{r['arm']} {r['key'].replace(SEP, ' | ')} s{r['seed']}",
                    " / ".join(r["boxes"])[:30],
                ],
            )
            for r in recs
            if r["scene"] == t
        ]
        contact_sheet(rows, sheets / f"t{t:02d}.png", thumb=320, cols=4 * len(seeds))
    write_result(
        run_dir,
        script="project/cjk_rl/probes/templ_read.py",
        args={"cfg": cfg, "model": T.MODEL, "out": str(out), "smoke": smoke},
        label=name,
        metrics={"moved": moved, "per_template": per},
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=("build", "run"))
    ap.add_argument("--config", required=True, help="its stem names the run")
    ap.add_argument("--smoke", action="store_true", help="the first template")
    a = ap.parse_args()
    cfg = tomllib.loads(Path(a.config).read_text("utf-8"))
    name = Path(a.config).stem
    if a.cmd == "build":
        build(cfg, name)
    else:
        run(cfg, ("smoke_" if a.smoke else "") + name, a.smoke)


if __name__ == "__main__":
    main()
