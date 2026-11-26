#!/usr/bin/env python
"""A line break in the quote — does the page wrap where the quote breaks (forward only).

Pages: strings of the draw's ``val`` (``strings_file``), ``per_bin`` a length bin of
``bins``, on ``scenes`` × ``seeds``, rendered as ``s1.py`` renders them (the bubble
frame, the rows of ``[start]``, 4 Euler steps, the pixel head). A string of n glyphs is
asked three ways, the same scene and seeds each:

- ``none``: the quote holds the string;
- ``newline``: a line feed after glyph n // 2, what ``process_escape`` makes of a
  prompt's ``\\n``. The T5 side's ids are those of ``none``; the Qwen side holds the
  line feed's token;
- ``literal``: the two characters backslash and ``n`` there, as a dataset caption spells
  a break (``anime_tools.captions.position_clauses.LINE_BREAK``) and the TE cache
  tokenises it. Both sides hold tokens for it.

Read, per arm over the pages with a box (``gd.pick`` to the string without its break):
the read aligned to the string (``s1.align``), and

- ``at``: the read ends a line after the glyph before the asked break;
- ``near``: after that glyph or one of its neighbours;
- ``any``: the read has a break;
- ``hit``: positions hit over positions asked (pages with no box count as none hit);
- ``stray``: the read holds a backslash or a Latin n.

``at`` under ``none`` is the floor: how often the page wraps there unasked. The ± is the
standard error over strings of an arm's ``at`` minus ``none``'s.

Writes ``output/cjk_rl/quote_break/<config stem>/pages/<arm>/`` and
``results/<ts>-quote_break-<config stem>/`` (``records.json``, ``result.json``).

    make daemon-run ARGS="--queue --stall-timeout 900 project/cjk_rl/probes/quote_break.py --config project/cjk_rl/configs/<name>.toml"
"""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path

import glyph_g0 as G
import glyph_scenes as S
import turbo_read as T

import g1  # noqa: E402  (src/, on the path through turbo_read)
import gd  # noqa: E402
import s1  # noqa: E402

LINE = Path(__file__).resolve().parents[1]
OUT = T.REPO / "output" / "cjk_rl" / "quote_break"
BREAK = {"none": "", "newline": "\\n", "literal": "\\\\n"}  # before process_escape


def quote(au: list, arm: str) -> str:
    h = len(au) // 2
    return "".join(au[:h]) + BREAK[arm] + "".join(au[h:])


def tabulate(recs: list) -> dict:
    out = {}
    base = {(r["string"], r["scene"], r["seed"]): r for r in recs if r["arm"] == "none"}
    for arm in BREAK:
        xs = [r for r in recs if r["arm"] == arm]
        bx = [r for r in xs if r["case"] == "box"]
        by: dict = {}  # string -> at minus none's, a page
        for r in xs:
            b = base[(r["string"], r["scene"], r["seed"])]
            by.setdefault(r["string"], []).append(r["at"] - b["at"])
        d = [sum(v) / len(v) for v in by.values()]
        m = sum(d) / len(d)
        se = (sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)) ** 0.5
        out[arm] = {
            "pages": len(xs),
            "box": len(bx),
            "at": round(sum(r["at"] for r in bx) / max(len(bx), 1), 3),
            "near": round(sum(r["near"] for r in bx) / max(len(bx), 1), 3),
            "any": round(sum(r["lines"] > 1 for r in bx) / max(len(bx), 1), 3),
            "hit": round(sum(r["hit"] for r in xs) / sum(r["glyphs"] for r in xs), 3),
            "stray": sum(r["stray"] for r in bx),
            "at_minus_none": round(m, 3),
            "se": round(se, 3),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="its stem names the run")
    ap.add_argument("--smoke", action="store_true", help="a string a bin, one scene")
    a = ap.parse_args()
    import torch
    from PIL import Image

    cfg = tomllib.loads(Path(a.config).read_text("utf-8"))
    name = ("smoke_" if a.smoke else "") + Path(a.config).stem
    sc = S.scenes()
    scenes, seeds = list(cfg["scenes"]), list(cfg["seeds"])
    per_bin = cfg["per_bin"]
    if a.smoke:
        scenes, per_bin = scenes[:1], 1
    pool = json.loads((T.REPO / cfg["strings_file"]).read_text("utf-8"))["val"]
    strings = [s for b in cfg["bins"] for s in pool[b][:per_bin]]
    frame = cfg["frame"]
    jobs = [(s, si, arm) for s in strings for si in scenes for arm in BREAK]
    kana = [k for _, k, _ in g1.glyphs(cfg | {"fold_small": False}, False)]
    out = OUT / name
    for arm in BREAK:
        (out / "pages" / arm).mkdir(parents=True, exist_ok=True)

    def caption(s: str, si: int, arm: str) -> str:
        return G.framed(sc[si][frame], quote(s1.units(s), arm))

    pol = gd.Policy([G.SHAPE], rows=cfg["rows"], grad_bs=1)
    pol.encode([caption(*j) for j in jobs])
    rdr = gd.Reader("cuda")
    at, _ = s1.row_map(pol, sc[scenes[0]][frame], kana)
    start = pol.raw.detach().clone()
    if cfg.get("start"):
        start = g1.saved_rows(cfg["start"], {k: at[s1.key(k)] for k in kana}, start)
    pol.raw.data.copy_(start)
    print(
        f"{name}: {len(strings)} strings × {len(scenes)} scenes × {len(BREAK)} arms × "
        f"{len(seeds)} seeds",
        flush=True,
    )
    recs, t0 = [], time.time()
    for n, (s, si, arm) in enumerate(jobs):
        au = s1.units(s)
        ak = [s1.key(u) for u in au]
        asked = len(au) // 2 - 1  # the glyph the break follows
        e = pol.embed(caption(s, si, arm))
        x = pol.noise(seeds, G.SHAPE)
        with torch.no_grad():
            for i in range(pol.steps):
                v = pol.forward(i, x, e)
                if i < pol.steps - 1:
                    x = pol.euler(i, x, v)
            px = pol.decode(x)
        bgrs, reads = rdr.read(px)
        for b, seed in enumerate(seeds):
            case, r = gd.pick(s, reads[b], cfg["conf_min"])
            read = (r["sfx"] or "") if r is not None else ""
            ru, br = s1.read_units(read)
            pairs = s1.align(ak, [s1.key(u) for u in ru])
            after = {i for i, j in pairs if j in br}
            stem = f"{strings.index(s):02d}_sc{si:03d}_s{seed}"
            Image.fromarray(bgrs[b][:, :, ::-1]).save(
                out / "pages" / arm / f"{stem}.webp", quality=85
            )
            recs.append(
                {
                    "string": s,
                    "glyphs": len(au),
                    "scene": si,
                    "seed": seed,
                    "arm": arm,
                    "stem": stem,
                    "case": case,
                    "read": read,
                    "lines": len(br) + 1 if ru else 0,
                    "hit": len(pairs),
                    "at": int(asked in after),
                    "near": int(bool(after & {asked - 1, asked, asked + 1})),
                    "stray": int("\\" in read or "n" in read.lower() or "ｎ" in read),
                }
            )
        if (n + 1) % 24 == 0:
            print(
                f"  caption {n + 1} / {len(jobs)} ({(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )
    table = tabulate(recs)
    print("| arm | pages | box | at | near | any | hit | stray | at − none ± se |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, v in table.items():
        print(
            f"| {k} | {v['pages']} | {v['box']} | {v['at']} | {v['near']} | {v['any']} "
            f"| {v['hit']} | {v['stray']} | {v['at_minus_none']} ± {v['se']} |",
            flush=True,
        )

    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir("cjk_rl", label=f"quote_break-{name}", root=LINE / "results")
    (run_dir / "records.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=1), "utf-8"
    )
    write_result(
        run_dir,
        script="project/cjk_rl/probes/quote_break.py",
        args=vars(a) | {"cfg": cfg, "model": T.MODEL, "out": str(out)},
        label=name,
        metrics={"pages": len(recs), "table": table},
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
