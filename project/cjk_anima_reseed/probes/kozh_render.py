#!/usr/bin/env python
"""kozh_render — a ``lang`` run's rows drawn, against its seed: the ruler's
render path (``ruler.Renderer``: one ``ExtDelta``, routed, 28 steps, cfg 4,
seed 0) on the run's pack, at 512² (the training px), one seed.

Each row alone in a speech bubble, and a few words of the rows in a bubble,
on a held sign and plain — captioned in the row's language
(``pools.relang``). Two arms: the seed (``seed_1008``: the KO / ZH glyphs on
their untrained pack rows) and the run. Renders cached under
``output/cjk_anima_reseed/<run>/render/<arm>/``; a side-by-side sheet →
``results/<ts>-<run>-render/``. Read by eye (no KO / ZH reader).

    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/kozh_render.py kozh16"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))
sys.path.insert(0, str(HOME / "probes"))

# the run's words: none was trained as a window (no KO / ZH line) — a look
WORDS = {"kozh16": ("한양", "양가", "你们", "这个", "你说")}
SIZE = (512, 512)
SIGN = (
    '1girl, solo, holding sign, sign, {l} text. She is holding a sign that reads "{t}".'
)


def prompts(run, pools_lang: dict) -> list:
    """``(key, text, caption)``: every row alone in a bubble, every word in a
    bubble / on a sign / plain."""
    from common.prompts import TPL_BUBBLE, TPL_PLAIN
    from reseed.pools import relang

    def lang(t):
        return next(pools_lang[c] for c in t if c in pools_lang)

    out = [
        (f"bubble_{g}", g, relang(TPL_BUBBLE.format(g), lang(g))) for g in pools_lang
    ]
    for w in WORDS.get(run.name, ()):
        out += [
            (f"bubble_{w}", w, relang(TPL_BUBBLE.format(w), lang(w))),
            (f"sign_{w}", w, SIGN.format(l=lang(w), t=w)),
            (f"plain_{w}", w, relang(TPL_PLAIN.format(w), lang(w))),
        ]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run")
    a = p.parse_args()
    from reseed.config import PACKS, load

    run = load(a.run)
    assert run.lang and run.pack, f"{run.name}: a lang run on a base pack"
    os.environ["ANIMA_VOCAB_PACK"] = PACKS[run.pack]
    from reseed import HOME as RH
    from reseed import OUT, bootstrap

    bootstrap()
    os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
    import ruler as R

    R.PACK = run.pack
    seed = run.seed_rows().parent.name
    arms = [seed, run.name]
    known = R.arm_dirs()
    assert set(arms) <= set(known), f"ruler.PACK_ARMS lacks {set(arms) - set(known)}"
    its = prompts(run, run.lang)
    _, tabs = R.tables(arms + ["retrain_kana"])
    r = R.Renderer()
    info = {"check_rk": R.check(r, tabs["retrain_kana"])}
    t0 = time.time()
    files = {}
    for arm in arms:
        r.set_arm(tabs[arm])
        for n, (key, _t, cap) in enumerate(its):
            fn = OUT / run.name / "render" / arm / f"{key}_s{R.SEED_RENDER}.png"
            r.render(fn, cap, R.SEED_RENDER, SIZE)
            files[arm, key] = fn
            print(
                f"  {arm}: {n + 1} / {len(its)} {key} ({(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )
    from bench._common import make_run_dir, write_result
    from common.readers import contact_sheet
    from PIL import Image

    out = make_run_dir(
        "cjk_anima_reseed", label=f"{run.name}-render", root=RH / "results"
    )
    tiles = []
    for key, t, _cap in its:
        for arm in arms:
            tiles.append((Image.open(files[arm, key]), [f"{key} ", arm]))
    sheet = out / "sheet.png"  # seed | run, side by side, 4 prompts a row
    contact_sheet(tiles, sheet, thumb=256, cols=8)
    sheets = [str(sheet)]
    info["minutes"] = round((time.time() - t0) / 60, 1)
    write_result(
        out,
        script=__file__,
        args={"run": run.name},
        label=f"{run.name}-render",
        metrics={"renders": len(files), **info},
        artifacts=sheets,
    )
    print(f"→ {out}: {len(files)} renders, {info['minutes']} min", flush=True)


if __name__ == "__main__":
    main()
