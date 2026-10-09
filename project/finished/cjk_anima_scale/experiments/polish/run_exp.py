#!/usr/bin/env python
"""polish — plan_polish.md step 1: a 4 k-token scene pool for the kana polish (2026-09-28)

The composite, not the model's own kana: the base draws a scene at the 1024
tier with an EN anchor in its bubble (every token pretrained, no ext row
touched, so the canvas is the DiT's own — low FM loss outside the text); the
line's data stage later erases the anchor and pastes the kana word, so the
only residual left for the rows is the text (user, 2026-09-28). The first
cut — ``retrain_kana``'s rows rendering the kana themselves, kept on an
exact read (28 renders under ``OUT/polish_yield/img``, stopped) — would have
trained on renders whose in-box loss is the model's own too.

This leg is the line's ``scenes`` stage as it ran for the 1 k pools
(``s1`` / ``s1w`` …: one closed bubble, the anchor read back, min box,
erase residual, specks), with three changes:

- **Frame** ``text_reads``: ``{bubble}`` tag, no ``english text`` tag,
  ``Text reads as "{a}".`` — added to ``scenes.stage.FRAMES`` in-process
  (``src/`` stays byte-faithful). ``synth.scene_caption`` keeps the frame
  and swaps only the quote, so the composite caption is
  ``<tags>. Text reads as "<kana>".``
- **Anchors**: short English sentences / several words (``ANCHORS``), not
  the 1 k pools' one-word ``hi`` / ``ok`` — the reader must read the whole
  anchor back (``norm`` drops case, spaces, punctuation).
- **Canvas**: the 1024 tier (``SHAPES``, 3 840 – 4 480 tokens), rendered at
  its own size (gen scale 1).

A kept scene that is white-background line art (``WHITE_MIN`` /
``SAT_MAX``) is rejected as ``white_lineart`` (user, 2026-09-28).

The judge, loosened after the first 200 (user, 2026-09-28; counts in
``result.json`` ``rejudge``): the anchor is erased, so its spelling does not
matter — a Latin read within a third of the anchor's letters
(``FUZZY``) counts as the anchor (``judge`` wrapped in-process: it sees the
fuzzed copy, the stored reads stay raw); the min box is 72 px (1 k's 56
doubled, 112, dropped 38 clean one-line bubbles; 56 at 4 k admits
57–70 px regions); an open bubble passes at seam ≥ 0.85 and art lost
≤ 0.12 (the stage's 0.9 / 0.02 rejected closed bubbles the flood missed).
Exact / 112 / 0.9 / 0.02: 32 / 200; this judge: 69 / 200.

Writes the pool ``OUT/scenes_<tag>/`` (``scenes.jsonl`` = kept) with the
stage's ``report.md`` (keep rate by reason), then a ``result.json`` here.
Re-running with a larger ``--n`` grows the pool (the stage keeps stored
rows). ``--rejudge`` (CPU) re-applies this judge to the stored reads.
``--dry_run`` prints the prompts without touching a model.

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      make daemon-run ARGS="--label polish-scenes --stall-timeout 0 \\
      project/cjk_anima_scale/experiments/polish/run_exp.py --label t4k"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, bootstrap  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

FRAME = "text_reads"
FRAME_DEF = (["{bubble}"], 'Text reads as "{a}".')
SHAPES = "1024,896x1152,1152x896,768x1280,1280x768,896x1280,1280x896"
MIN_BOX = 72  # 112 (1 k's 56 × 2) dropped clean one-line bubbles
OPEN_UNIFORM, OPEN_LOST = 0.85, 0.12  # the stage's 0.9 / 0.02
FUZZY = 1 / 3  # a Latin read within this share of the anchor's letters
# white-background line art (user, 2026-09-28: the kept set leaned to it —
# 18 / 69): near-white share ≥ WHITE_MIN and mean saturation < SAT_MAX on a
# 256-px thumbnail. Screentone manga pages sit under 0.8 white and stay.
WHITE_MIN, SAT_MAX = 0.8, 0.05
ANCHORS = (
    "wait for me",
    "see you tomorrow",
    "i am so hungry",
    "let's go home",
    "no way",
    "thank you so much",
    "what is this",
    "good morning",
    "leave me alone",
    "i knew it",
    "are you okay",
    "not again",
    "over here",
    "hurry up",
    "i'm sorry",
    "look at this",
    "that's mine",
    "just kidding",
    "me too",
    "stop it",
    "help me",
    "one more time",
    "welcome back",
    "good night",
    "watch out",
    "i'm home",
    "nice to meet you",
    "is that so",
    "come here",
    "don't move",
    "who are you",
    "it's fine",
    "take this",
    "so cute",
    "why me",
    "not bad",
    "trust me",
    "let me see",
    "give it back",
    "i did it",
)
GEN_STEPS, GEN_CFG, SEED = 28, 4.0, 0  # cjk_scale.eval's


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True, help="also the pool tag: scenes_<label>")
    p.add_argument("--n", type=int, default=200, help="prompts (the pool grows)")
    p.add_argument("--rejudge", action="store_true", help="CPU: this judge on the stored reads")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def fuzzy_judge() -> None:
    """Wrap ``scenes.judge.judge``: a Latin read within ``FUZZY`` of the
    anchor is passed as the anchor; the caller's reads are not touched."""
    import copy
    import re

    from common.text import lev, norm
    from scenes import judge as J

    raw = J.judge

    def judge(a, it, reads, bgr):
        anchor = norm(it["anchor"])
        reads = copy.deepcopy(reads)
        for r in reads:
            for x in ("vl", "sfx"):
                t = norm(r.get(x) or "")
                if (
                    t
                    and re.fullmatch(r"[a-z0-9']+", t)
                    and lev(t, anchor) <= max(1, int(len(anchor) * FUZZY))
                ):
                    r[x] = it["anchor"]
        reason = raw(a, it, reads, bgr)
        if reason == "pass":
            it["white"], it["sat"] = white_sat(bgr)
            if it["white"] >= WHITE_MIN and it["sat"] < SAT_MAX:
                return "white_lineart"
        return reason

    J.judge = judge


def white_sat(bgr) -> tuple[float, float]:
    """(near-white pixel share, mean max−min channel spread) on a 256-px
    thumbnail."""
    import numpy as np
    from PIL import Image

    im = Image.fromarray(bgr[:, :, ::-1])
    im.thumbnail((256, 256))
    x = np.asarray(im).astype(np.float32) / 255
    return round(float((x.min(2) > 0.9).mean()), 3), round(
        float((x.max(2) - x.min(2)).mean()), 3
    )


def stage_args(tag: str, n: int, rejudge: bool = False):
    from cli import build_parser
    from stages import STAGES

    argv = [
        "--stage",
        "scenes",
        "--scene_tag",
        tag,
        "--scene_n",
        str(n),
        "--scene_shapes",
        SHAPES,
        "--scene_frames",
        FRAME,
        "--scene_anchors",
        ",".join(ANCHORS),
        "--scene_min_box",
        str(MIN_BOX),
        "--scene_open_uniform",
        str(OPEN_UNIFORM),
        "--scene_open_lost",
        str(OPEN_LOST),
        "--scene_batch",
        "2",
        "--steps",
        str(GEN_STEPS),
        "--cfg",
        str(GEN_CFG),
        "--seed",
        str(SEED),
        *(["--scene_rejudge", "1"] if rejudge else []),
    ]
    return build_parser(STAGES).parse_args(argv)


def main():
    args = parse_args()
    assert all("," not in x for x in ANCHORS)
    from scenes import stage as scenes_stage

    scenes_stage.FRAMES[FRAME] = FRAME_DEF
    fuzzy_judge()
    a = stage_args(args.label, args.n, args.rejudge)
    if args.dry_run:
        items = scenes_stage.scene_items(a)
        shapes = Counter(tuple(it["shape"]) for it in items)
        print(
            f"scenes_{args.label}: {len(items)} prompts; shapes "
            + ", ".join(
                f"{w}x{h} ({(w // 16) * (h // 16)} tok) ×{c}"
                for (w, h), c in sorted(shapes.items())
            ),
            flush=True,
        )
        for it in items[:: max(1, len(items) // 10)]:
            print(f"  {it['i']:03d} {it['shape']} {it['prompt']}", flush=True)
        return
    from stages import run as run_stage

    run_stage("scenes", a)
    out = OUT / f"scenes_{args.label}"
    rows = [
        json.loads(ln)
        for ln in (out / "scenes_all.jsonl").read_text().splitlines()
        if ln
    ]
    reasons = Counter(r.get("reason") for r in rows)
    by_anchor: dict = {}
    for r in rows:
        c = by_anchor.setdefault(r["anchor"], Counter())
        c["n"] += 1
        c["pass"] += r.get("reason") == "pass"
    by_shape: dict = {}
    for r in rows:
        c = by_shape.setdefault("x".join(map(str, r["shape"])), Counter())
        c["n"] += 1
        c["pass"] += r.get("reason") == "pass"
    metrics = {
        "pool": out.name,
        "n": len(rows),
        "kept": reasons.get("pass", 0),
        "reasons": dict(reasons.most_common()),
        "by_anchor": {k: dict(v) for k, v in sorted(by_anchor.items())},
        "by_shape": {k: dict(v) for k, v in sorted(by_shape.items())},
        "frame": FRAME_DEF[1],
        "shapes": SHAPES,
        "min_box": MIN_BOX,
        "open_uniform": OPEN_UNIFORM,
        "open_lost": OPEN_LOST,
        "fuzzy": FUZZY,
        "rejudge": {
            "exact_112_0.9_0.02": 32,
            "fuzzy_72_0.9_0.02": 59,
            "fuzzy_72_0.85_0.12": 69,
            "fuzzy_56_0.85_0.12": 77,
            "fuzzy_72_0.85_0.12_no_white_lineart": 51,
        },
        "white_min": WHITE_MIN,
        "sat_max": SAT_MAX,
    }
    run_dir = make_run_dir(
        "polish", label=args.label, root=LINE / "experiments" / "polish" / "results"
    )
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(out)],
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=1), flush=True)
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
