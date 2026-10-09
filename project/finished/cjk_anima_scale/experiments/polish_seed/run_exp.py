#!/usr/bin/env python
"""polish_seed — ``polish_b1``'s polish on the new seed (2026-09-30)

``experiments/polish_b1`` exactly (its table, its ``scene_line`` tier, its
line pool rules), with the rows moved from ``retrain_kanji_b1`` (503 singles)
to ``seed_retrain_0930`` = ``retrain_kanji_b4``'s merged rows (the same file,
md5 ``af99aa93…``): 1 362 singles — ``retrain_kana`` + kanji b1–b4, warm,
every other row frozen at the seed. Only the constants move; the code is
``polish_b1/run_exp.py``, loaded as a module and patched here:

- **Rows / lines.** The trained singles are the five chain runs' ``vocabs.json``;
  a dialogue line may use any of them, so the line pool is the question this
  arm answers first (``--dry_run``: how many more lines than b1's 14 658).
  The held-out words are all five runs' ``read``.
- **The encode fold.** ``！ ？`` fold to the base T5 ``! ?`` at encode (the
  pack's ``fold``, plan_retrain § 2c; the seed trained folded), so a line
  carrying them has no ext row there — they join ``NATIVE`` (allowed, not
  a row) instead of dropping the line at the routed-encoding check.
- **Steps.** ``STEPS_PER_ROW`` 4 × 1 362 = 5 448 steps (user: 5 k; the
  trainer takes an integer per row), batch 4 → 21 792 items, one epoch.
- **μ** 0.1 by default here (``--mu``).

Legs as polish_b1's: ``data`` → ``OUT/run0930_polish_seed/data``;
``train`` → ``OUT/experiments/polish_seed_mu<μ>``. ``--color`` (any leg): the
scene pools without their monochrome / line-art canvases → ``run0930_polish_seed_color``
/ ``polish_seed_color_mu<μ>``.

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      .venv/bin/python project/cjk_anima_scale/experiments/polish_seed/run_exp.py \\
      --label s0930 --dry_run
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))

MOD = "polish_b1_exp"
_spec = importlib.util.spec_from_file_location(
    MOD, LINE / "experiments" / "polish_b1" / "run_exp.py"
)
P = importlib.util.module_from_spec(_spec)
sys.modules[MOD] = P  # the build workers unpickle ``_slot_worker`` by module name
_spec.loader.exec_module(P)

from bench._common import make_run_dir  # noqa: E402
from cjk_scale.paths import OUT, SEED_ROWS  # noqa: E402

CHAIN = (
    "retrain_kana",
    "retrain_kanji_b1",
    "retrain_kanji_b2",
    "retrain_kanji_b3",
    "retrain_kanji_b4",
)
MU = 0.1

# --color: every scene pool without its monochrome / line-art canvases (user,
# 2026-09-30) — a scene is kept iff ≥ COLOR_MIN of its pixels are coloured
# (HSV saturation > 0.15 and value > 0.15, on a 128² thumbnail); 0.06–0.10
# keep 1 096–1 032 of the 2 107 (by eye: < 0.06 is greyscale or line art with
# a speck of colour, 0.06–0.12 mixes pale-tinted sketches with flat colour on
# white). Its own data dir and arm.
COLOR = "--color" in sys.argv
if COLOR:
    sys.argv.remove("--color")
COLOR_MIN = 0.08
COLOR_CACHE = "scene_colorful.json"  # file → coloured share, under OUT/experiments/

P.NAME = "polish_seed_color" if COLOR else "polish_seed"
P.DATA = "run0930_polish_seed_color" if COLOR else "run0930_polish_seed"
P.BASE = CHAIN[-1]  # its trained.pt is the seed (same md5)
P.CONTEXT = SEED_ROWS
P.N_ROWS = 1362
P.STEPS_PER_ROW = 4
P.N_ITEMS = P.STEPS_PER_ROW * P.N_ROWS * P.BATCH
P.INIT_ANCHOR = (
    -1.0
)  # never equal to --mu: the arm is always named ``polish_seed_mu<μ>``
P.NATIVE = P.NATIVE | {"！", "？"}  # folded to base T5 at encode


def rc_of(name: str):
    from cjk_scale.config import RunConfig, load_run

    runs = [load_run(r) for r in CHAIN]
    specs = [v for r in runs for v in r.vocab_specs()]
    # one ``list:`` source: the stages' inventory reads only the first source of
    # a kind (``src/data/vocabs.py`` ``source``), so four ``list:@`` specs
    # would train b1's kanji alone
    files = [s[len("list:") :] for s in specs if s.startswith("list:")]
    specs = [s for s in specs if not s.startswith("list:")] + [
        "list:" + ",".join(files)
    ]
    return RunConfig(
        name=name,
        path=Path(__file__),
        vocabs=tuple(specs),
        read=tuple(dict.fromkeys(w for r in runs for w in r.read)),
        context=P.BASE,
    )


def trained_singles() -> set:
    out: set = set()
    for run in CHAIN:
        out |= set(json.loads((OUT / run / "data" / "vocabs.json").read_text("utf-8")))
    assert len(out) == P.N_ROWS and all(len(v) == 1 for v in out), len(out)
    return out


def _checked_build(build):
    def run(rc, *a, **k):
        out = build(rc, *a, **k)
        n = len(json.loads((Path(out) / "vocabs.json").read_text("utf-8")))
        assert n == P.N_ROWS, f"built {n} vocabs, not the seed's {P.N_ROWS} singles"
        return out

    return run


def colorful(file: str) -> float:
    import numpy as np
    from PIL import Image

    im = (
        np.asarray(Image.open(file).convert("RGB").resize((128, 128)), np.float32) / 255
    )
    mx, mn = im.max(-1), im.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-3)
    return float(((sat > 0.15) & (mx > 0.15)).mean())


def _color_scenes(load):
    def run(*a, **k):
        scenes = load(*a, **k)
        f = OUT / "experiments" / COLOR_CACHE
        cache = json.loads(f.read_text("utf-8")) if f.exists() else {}
        miss = [s["file"] for s in scenes if s["file"] not in cache]
        for file in miss:
            cache[file] = colorful(file)
        if miss:
            f.write_text(json.dumps(cache, indent=0), encoding="utf-8")
        kept = [s for s in scenes if cache[s["file"]] >= COLOR_MIN]
        by = lambda ss: dict(Counter(s["pool"] for s in ss))  # noqa: E731
        print(
            f"scenes --color (≥ {COLOR_MIN}): kept {by(kept)} of {by(scenes)}",
            flush=True,
        )
        return kept

    return run


from collections import Counter  # noqa: E402

import data.synth  # noqa: E402
from cjk_scale import builder  # noqa: E402

if COLOR:
    data.synth.load_scenes = _color_scenes(data.synth.load_scenes)
builder.build = _checked_build(builder.build)
P.rc_of = rc_of
P.trained_singles = trained_singles
P.make_run_dir = lambda _method, label, root: make_run_dir(
    "polish_seed", label=label, root=LINE / "experiments" / "polish_seed" / "results"
)

SENT_PROMPTS = (
    4  # retrain_read's grid: the first 4 scene prompts × 2 seeds = 8 / string
)
SHAPE_4K = "768x1344"  # experiments/target4k: the user's ComfyUI renders, 4 032 tokens
READS = {  # read name → (floor cache under <seed>/routed/, the arm's read file)
    "sent": "native_sent/native_reads.json",
    "target": "target/native_reads.json",
    "target_4k": "target_4k/native_reads.json",
}


def _official(m) -> bool:
    return bool(m.get("hit_sfx")) and bool(m.get("hit_vl"))


def _paired(fl: list, tr: list, key) -> dict:
    a = {key(m): _official(m) for m in fl}
    b = {key(m): _official(m) for m in tr}
    ks = a.keys() & b.keys()
    return {
        "n": len(ks),
        "floor": sum(a[k] for k in ks),
        "polish": sum(b[k] for k in ks),
        "only_polish": sum(b[k] and not a[k] for k in ks),
        "only_floor": sum(a[k] and not b[k] for k in ks),
    }


def read(label: str, mu: float) -> None:
    """The ``read`` leg: the polished rows on ``sent`` (the acceptance strings +
    the chain's ``read`` words, the ``en`` clause, 512², the first
    ``SENT_PROMPTS`` prompts), ``target`` (512²) and
    ``target_4k`` (``SHAPE_4K``), routed, each paired (same key × prompt ×
    seed) against the seed's routed floor cache. Renders land in the arm dir."""
    import dataclasses
    from collections import defaultdict
    from types import SimpleNamespace

    from bench._common import write_result
    from cjk_scale.eval import (
        ACCEPT_READ,
        TRAINED_ARM,
        _load_reads,
        _metrics,
        floor_arm_dir,
        probe_args,
        ruler_args,
    )
    from stages import run as run_stage

    arm = OUT / "experiments" / f"{P.NAME}_mu{mu:g}"
    assert (arm / "trained.pt").exists(), f"{arm}: not trained yet"
    rc = rc_of(P.NAME)
    rc = dataclasses.replace(rc, read=tuple(dict.fromkeys((*ACCEPT_READ, *rc.read))))
    floor = floor_arm_dir(dataclasses.replace(rc, name=P.BASE))  # <seed>/routed/
    run_dir = make_run_dir(
        "polish_seed",
        label=label,
        root=LINE / "experiments" / "polish_seed" / "results",
    )
    jobs = {
        "sent": ("native", ruler_args(rc, TRAINED_ARM, "sent")),  # native_limit below
        "target": ("target", probe_args(rc, TRAINED_ARM, ["target"])),
        "target_4k": (
            "target",
            probe_args(
                rc,
                TRAINED_ARM,
                ["target"],
                ["--eval_shape", SHAPE_4K, "--eval_tag", "4k"],
            ),
        ),
    }
    metrics: dict = {"arm": str(arm), "floor": str(floor), "shape_4k": SHAPE_4K}
    for name, (stage, a) in jobs.items():
        a.arm_path = str(arm)
        if name == "sent":
            a.native_limit = SENT_PROMPTS
        a.data_path = str(OUT / P.DATA / "data")
        run_stage(stage, a)
        fl, tr = _load_reads(floor / READS[name]), _load_reads(arm / READS[name])
        assert fl, f"no floor cache {floor / READS[name]}"
        key = lambda m: (m["text"], m.get("clause"), m["pi"], m["seed"])  # noqa: E731
        have = {key(m) for m in tr}
        fl = [m for m in fl if key(m) in have]  # the floor on the arm's grid only
        per_f, per_t = defaultdict(list), defaultdict(list)
        for m in fl:
            per_f[m["text"]].append(m)
        for m in tr:
            per_t[m["text"]].append(m)
        metrics[name] = {
            "polish": {t: _metrics("target", ms) for t, ms in per_t.items()},
            "floor": {t: _metrics("target", per_f[t]) for t in per_t},
            "paired": _paired(fl, tr, key),
            "paired_by_string": {t: _paired(per_f[t], per_t[t], key) for t in per_t},
        }
        print(f"===== {name}: paired official {metrics[name]['paired']}", flush=True)
        for t, pr in metrics[name]["paired_by_string"].items():
            print(
                f"  {t:<12} floor {pr['floor']:>2} → polish {pr['polish']:>2} / {pr['n']}",
                flush=True,
            )
    write_result(
        run_dir,
        script=__file__,
        args=SimpleNamespace(label=label, mu=mu, legs=["read"]),
        label=label,
        metrics=metrics,
        artifacts=[str(arm / f) for f in READS.values()],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    if "--read" in sys.argv:
        import argparse

        ap = argparse.ArgumentParser()
        ap.add_argument("--read", action="store_true")
        ap.add_argument("--label", required=True)
        ap.add_argument("--mu", type=float, default=MU)
        ra = ap.parse_args()
        read(ra.label, ra.mu)
        sys.exit()
    if not any(a == "--mu" or a.startswith("--mu=") for a in sys.argv[1:]):
        sys.argv += ["--mu", str(MU)]
    P.main()
