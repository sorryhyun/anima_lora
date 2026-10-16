#!/usr/bin/env python
"""identity — the cold hiragana arms read on identity alone (motivation2.md § 7.3 /
§ 8, user 10-14: the sensitive ruler is too hard a read for a cold pass)

A small plain grid, 50 renders an arm:

- **words**: ``hira_cold_hp``'s 13 ``read`` words × p00 / p01 × seed 0;
- **singles**: ``SINGLES`` (24 of the 81 hiragana rows: the scale line's eight
  plus small kana, (han)dakuten, look-alike pairs, ゔ) × p00 × seed 0;

each as ``<prompt>. Text reads as "<key>".`` at 512², routed, on the ruler's
render path (``eval.ruler.render.Renderer``, 28 steps, cfg 4). The prompts are
the scale line's plain-read scenes (``retrain_kana/native_r4_plain``). The
floor is ``seed_retrain_0930``: the arms differ from it in the 81 hiragana
rows only, and its 81 rows are ``retrain_kana``'s.

Renders: ``output/cjk_anima_reseed/identity/<arm>/``; a render on disk is not
drawn again, reads cached per arm. Read → ``results/<ts>-identity-<label>/``:
per arm exact / contained / ≤ 1 / CER / dup on words and singles, paired on
exact against the floor, per-glyph single hits, sheets; and the EN-reference
ruler (``eval/enref.py``: ``en_cos`` / ``en_cos_out`` / ``box_iou`` against the
scale line's shared ``native_enref/512_28_4/`` — same prompt and seed,
``English text reads as "<word>"``, delta off; the refs ``retrain_kana``'s plain
read was scored on).

    make daemon-run ARGS="project/cjk_anima_reseed/probes/identity.py --arms seed_retrain_0930,hira_cold_hp,hira_cold_up --label hira_cold"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reseed import HOME, OUT, SCALE_OUT, bootstrap  # noqa: E402

bootstrap()
os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"  # the runs trained routed, read routed

FLOOR = "seed_retrain_0930"
SINGLES = "あうがくとひもりぬめねれわるろさちっゃぱぽぜゔを"
WORD_PROMPTS, SINGLE_PROMPTS, SEED = (0, 1), (0,), 0
ROOT = OUT / "identity"
SRC = SCALE_OUT / "retrain_kana" / "native_r4_plain" / "native_reads.json"
ENREF = SCALE_OUT / "native_enref" / "512_28_4"


def grid() -> list:
    """``[{key, kind, pi, prompt, caption, stem}]``."""
    from reseed.config import load

    prompts = {
        m["pi"]: m["prompt"] for m in json.loads(SRC.read_text(encoding="utf-8"))
    }
    words = load("hira_cold_hp").read
    out = []
    for kind, keys, pis in (
        ("word", words, WORD_PROMPTS),
        ("single", SINGLES, SINGLE_PROMPTS),
    ):
        for k in keys:
            for pi in pis:
                out.append(
                    {
                        "key": k,
                        "kind": kind,
                        "pi": pi,
                        "prompt": prompts[pi],
                        "caption": f'{prompts[pi]}. Text reads as "{k}".',
                        "stem": f"{k}_p{pi:02d}_s{SEED}",
                    }
                )
    return out


def render(names: list) -> None:
    import time

    from eval.ruler.arms import tables
    from eval.ruler.render import Renderer, check

    g = grid()
    _, tabs = tables(names + ["retrain_kana"])
    r = Renderer()
    check(r, tabs["retrain_kana"])
    t0 = time.time()
    for a in names:
        todo = [m for m in g if not (ROOT / a / f"{m['stem']}.png").exists()]
        r.set_arm(tabs[a])
        for n, m in enumerate(todo):
            r.render(ROOT / a / f"{m['stem']}.png", m["caption"], SEED, (512, 512))
            print(
                f"  {a}: {n + 1} / {len(todo)} {m['stem']} "
                f"({(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )


def read(names: list) -> dict:
    """``{arm: {stem: record}}``; the reader output cached in
    ``<arm>/reads.json``."""
    import torch

    from common.readers import Readers, load_bgr
    from eval.ruler.score import score_text

    g = grid()
    rd, recs = None, {}
    for a in names:
        f = ROOT / a / "reads.json"
        got = json.loads(f.read_text("utf-8")) if f.exists() else {}
        todo = [m for m in g if m["stem"] not in got]
        if todo:
            rd = rd or Readers("cuda")
            for m in todo:
                got[m["stem"]] = rd.read_image(
                    load_bgr(ROOT / a / f"{m['stem']}.png"), whole=True
                )
            f.write_text(json.dumps(got, ensure_ascii=False, indent=1), "utf-8")
        recs[a] = {
            m["stem"]: {
                **m,
                **score_text(m["key"], got[m["stem"]]),
                "reads": got[m["stem"]],
            }
            for m in g
        }
    enref(recs, rd or Readers("cuda"))
    del rd
    torch.cuda.empty_cache()
    return recs


def enref(recs: dict, rd) -> None:
    """``en_cos`` / ``en_cos_out`` / ``box_iou`` on every record, the EN box
    the largest box the readers find on the reference."""
    from common.readers import load_bgr
    from eval.enref import EnRef, enref_file, largest_box

    boxes = {}
    for pi in sorted({m["pi"] for r in recs.values() for m in r.values()}):
        ref = enref_file(ENREF, pi, SEED)
        assert ref.is_file(), f"no EN reference {ref}"
        boxes[ref.name] = largest_box(rd.read_image(load_bgr(ref), whole=True))
    er = EnRef("cuda", ENREF, boxes)
    for a, r in recs.items():
        for m in r.values():
            m["text"], m["seed"] = m["key"], SEED
            m["file"] = str(ROOT / a / f"{m['stem']}.png")
            m["en_cos"], m["en_cos_out"], m["box_iou"] = er.score(m)
            m.update(plain(m))


def plain(m: dict) -> dict:
    """The plain-type read (L3, 10-16; negswap's pixels): ``white`` / ``sat``
    over the render and inside the glyph box (the box that read the word, else
    the largest) — pale band / white box: box white up, box sat down."""
    import numpy as np
    from PIL import Image

    from eval.enref import glyph_box

    px = np.asarray(Image.open(m["file"]).convert("RGB"), np.float32) / 255.0

    def reads(p):
        hi, lo = p.max(-1), p.min(-1)
        return float(((lo > 0.85) & (hi - lo < 0.08)).mean()), float((hi - lo).mean())

    out = dict(zip(("white", "sat"), reads(px)))
    b = glyph_box(m)
    if b:
        x0, y0, x1, y1 = (int(v) for v in b)
        c = px[max(0, y0) : y1, max(0, x0) : x1]
        if c.size:
            out["box_white"], out["box_sat"] = reads(c)
    return out


def tally(rs: list) -> dict:
    n = len(rs)
    mean = lambda k: round(sum(float(r[k]) for r in rs) / max(1, n), 3)  # noqa: E731
    return {
        "n": n,
        "exact": sum(r["exact"] for r in rs),
        "contained": sum(r["contained"] for r in rs),
        "le1": sum(r["le1"] for r in rs),
        "cer": mean("cer"),
        "dup": sum(r["dup"] for r in rs),
        "en_cos": mean("en_cos"),
        "en_cos_out": mean("en_cos_out"),
        "box_iou": mean("box_iou"),
        **{
            k: round(
                sum(r[k] for r in rs if k in r) / max(1, sum(k in r for r in rs)), 3
            )
            for k in ("white", "sat", "box_white", "box_sat")
        },
    }


def paired(a: dict, b: dict, stems: list) -> dict:
    """Exact wins / losses of ``a`` over ``b`` on the same render slot, with the
    two-sided sign-test p."""
    from math import comb

    w = sum(a[s]["exact"] and not b[s]["exact"] for s in stems)
    lo = sum(b[s]["exact"] and not a[s]["exact"] for s in stems)
    n = w + lo
    p = (
        min(1.0, 2 * sum(comb(n, k) for k in range(min(w, lo) + 1)) / 2**n)
        if n
        else 1.0
    )
    return {"wins": w, "losses": lo, "p": round(p, 4)}


def sheets(recs: dict, out: Path) -> None:
    from PIL import Image

    from common.readers import contact_sheet

    names = list(recs)
    g = grid()
    out.mkdir(parents=True, exist_ok=True)
    for kind in ("word", "single"):
        ms = [m for m in g if m["kind"] == kind]
        for k in range(0, len(ms), 8):
            rows = []
            for m in ms[k : k + 8]:
                for a in names:
                    r = recs[a][m["stem"]]
                    mark = "✓" if r["exact"] else "≤1" if r["le1"] else ""
                    rows.append(
                        (
                            Image.open(ROOT / a / f"{m['stem']}.png").convert("RGB"),
                            [
                                f"{m['key']} p{m['pi']:02d}",
                                f"{a} {mark}",
                                f"{r['best'][:16]}",
                                f"cer {r['cer']:.2f}",
                            ],
                        )
                    )
            contact_sheet(
                rows, out / f"sheet_{kind}_{k // 8}.png", thumb=224, cols=len(names)
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", required=True, help="comma list (the floor first)")
    p.add_argument("--label", required=True)
    a = p.parse_args()
    names = list(dict.fromkeys([FLOOR, *a.arms.split(",")]))
    from eval.ruler.arms import arm_dirs

    unknown = set(names) - set(arm_dirs())
    assert not unknown, f"unknown arms {unknown}"
    render(names)
    recs = read(names)
    g = grid()
    res = {"arms": {}, "pairs": {}, "singles": {}}
    for arm in names:
        rs = list(recs[arm].values())
        res["arms"][arm] = {
            kind: tally([r for r in rs if r["kind"] == kind])
            for kind in ("word", "single")
        }
        print(f"  {arm:<22} {res['arms'][arm]}", flush=True)
        if arm != FLOOR:
            res["pairs"][arm] = {
                kind: paired(
                    recs[arm],
                    recs[FLOOR],
                    [m["stem"] for m in g if m["kind"] == kind],
                )
                for kind in ("word", "single")
            }
            print(f"    vs {FLOOR}: {res['pairs'][arm]}", flush=True)
    for m in g:
        if m["kind"] == "single":
            res["singles"][m["key"]] = {
                arm: recs[arm][m["stem"]]["best"] for arm in names
            }
    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir(
        "cjk_anima_reseed", label=f"identity-{a.label}", root=HOME / "results"
    )
    sheets(recs, run_dir / "sheets")
    (run_dir / "renders.json").write_text(
        json.dumps(
            {arm: list(r.values()) for arm, r in recs.items()},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    write_result(
        run_dir,
        script="project/cjk_anima_reseed/probes/identity.py",
        args=vars(a),
        metrics=res,
    )
    print(f"identity → {run_dir}", flush=True)


if __name__ == "__main__":
    main()
