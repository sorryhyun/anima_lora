#!/usr/bin/env python
"""canvas — plan_canvas.md (beside this script): does the base spell (and the band law hold) on a
~500-token canvas?

Cells (plan_canvas.md § 1), all training-free, on the seed rows
(``paths.floor_dir()`` / ``trained.pt`` — the rows every run starts from):

  d0   the canvas gate, per ``--canvas``: ``eval`` en group (the EN control,
       12 strings × 2 seeds), ``native`` あ / い × en / swap (8 prompts ×
       2 seeds), ``cf_sense --cf_lang ja`` (the identity-leverage peak σ;
       ``classify`` was pruned 2026-09-25, cf_sense ja is the identity read
       the line keeps). The 512² column is read from the reads the seed dir
       already holds (floor cache ``eval_reads.json`` en, ``native/``,
       ``cf_sense_ja/``) — nothing is rendered at 512².
  d1   the EN ceiling per px on the canvas: A.1 run 1's argv (flat, Noto
       Serif CJK Regular, letter + string2, px 24 … 128, σ 0.35 … 0.9),
       base only (no ext id in any caption). ``--d1_ref`` = A.1's arm dir
       for the 512² column (read-only).

Everything lands in ``output/cjk_anima_scale/canvas_<label>/`` — its
``trained.pt`` is a symlink to the seed's, its ``data/eval.json`` the en
entries of the floor cache (the same captions the 512² en reads used), each
canvas under ``*_c<WxH>/``. The seed dir is never written.

GPU: ``make daemon-run ARGS="project/cjk_anima_scale/experiments/canvas/run_exp.py --cell d0 --label d0 …"``
with ``ANIMA_VOCAB_PACK`` set. ``--dry_run`` prints the argv per stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, bootstrap, floor_dir  # noqa: E402

bootstrap()

from bench._common import make_run_dir, write_result  # noqa: E402
from cjk_scale.eval import (  # noqa: E402
    GEN_CFG,
    GEN_STEPS,
    NATIVE_CHARS,
    NATIVE_CLAUSES,
    SEED,
    SEEDS,
)

# A.1 run 1 (cf_band_a1_2026_09_23.md § 1) — the 512² column of d1
D1_FONT = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
D1_ARGV = [
    "--cf_lang", "en", "--cf_layout", "flat", "--cf_font", D1_FONT,
    "--cf_pairs", "16", "--cf_per_pair", "6",
    "--cf_glyph_px", "24,32,48,64,96,128",
    "--cf_t", "0.35,0.4,0.5,0.6,0.7,0.8,0.9",
]  # fmt: skip
D1_TEXTS = ("letter", "string2")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--cell", required=True, choices=["d0", "d1"])
    p.add_argument(
        "--canvas",
        default="256x512,512x256,256x256",
        help="comma list of WxH canvases (sides multiples of 16)",
    )
    p.add_argument(
        "--d1_ref",
        default="",
        help="d1: A.1's arm dir (holds cf_sense_en_flat_<text>_NotoSerifCJK-Regular_a1/)",
    )
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def tag(c: str) -> str:
    return f"c{c}"


def arm(label: str) -> Path:
    return OUT / f"canvas_{label}"


def prepare(label: str) -> Path:
    """The experiment's arm dir: the seed's rows by symlink, the en eval set
    from the floor cache (so the canvas en reads share the 512² captions)."""
    d = arm(label)
    (d / "data").mkdir(parents=True, exist_ok=True)
    tp = d / "trained.pt"
    if not tp.exists():
        tp.symlink_to(floor_dir() / "trained.pt")
    ev, seen = [], set()
    for m in ref_en_reads():
        if m["text"] not in seen:
            seen.add(m["text"])
            ev.append({k: m[k] for k in ("group", "text", "caption")})
    (d / "data" / "eval.json").write_text(
        json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return d


def stage_argv(label: str, stage: str, canvas: str, extra: list) -> list:
    d = arm(label)
    return [
        "--stage", stage, "--arm", "rows",
        "--data_path", str(d / "data"), "--arm_path", str(d),
        "--seeds", str(SEEDS), "--steps", str(GEN_STEPS), "--cfg", str(GEN_CFG),
        "--seed", str(SEED), "--no_floor",
        "--eval_shape", canvas, "--eval_tag", tag(canvas),
        *extra,
    ]  # fmt: skip


def plan(cell: str, label: str, canvases: list) -> list:
    """(stage, argv) per job, in order."""
    jobs = []
    for c in canvases:
        if cell == "d0":
            jobs.append(("eval", stage_argv(label, "eval", c, ["--eval_groups", "en"])))
            jobs.append(
                (
                    "native",
                    stage_argv(
                        label,
                        "native",
                        c,
                        [
                            "--native_chars",
                            NATIVE_CHARS,
                            "--native_clauses",
                            NATIVE_CLAUSES,
                        ],
                    ),
                )
            )
            jobs.append(
                ("cf_sense", stage_argv(label, "cf_sense", c, ["--cf_lang", "ja"]))
            )
        else:
            for t in D1_TEXTS:
                jobs.append(
                    (
                        "cf_sense",
                        stage_argv(label, "cf_sense", c, [*D1_ARGV, "--cf_text", t]),
                    )
                )
    return jobs


# ---------------------------------------------------------------------------
# reads


def _load(f: Path) -> list:
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def ref_en_reads() -> list:
    return [
        m
        for m in _load(floor_dir() / "eval_reads.json")
        if m["group"] == "en" and m.get("cond", "trained") == "trained"
    ]


def en_totals(ms: list) -> dict:
    return {"n": len(ms), "exact": sum(bool(m.get("exact")) for m in ms)}


def native_table(ms: list) -> dict:
    """text|clause → {n, official (sfx ∧ VL), loose (sfx ∨ VL)} — per glyph,
    never a total alone."""
    out: dict = {}
    for m in ms:
        if m.get("cond", "trained") != "trained":
            continue
        r = out.setdefault(
            f"{m['text']}|{m['clause']}", {"n": 0, "official": 0, "loose": 0}
        )
        r["n"] += 1
        r["official"] += bool(m.get("hit_sfx")) and bool(m.get("hit_vl"))
        r["loose"] += bool(m.get("hit_sfx")) or bool(m.get("hit_vl"))
    return dict(sorted(out.items()))


def cf_curves(pt: Path, key: str | None = None) -> dict:
    """cf_sense.pt → cond → kind [→ key value] → {σ: mean move, peak}."""
    import torch

    sd = torch.load(pt, map_location="cpu", weights_only=False)
    pos, sig, items = sd["pos"], sd["sigmas"], sd["items"]
    out: dict = {"pairs": [f"{it['a']}/{it['b']}" for it in items]}
    for ci, cond in enumerate(sd["conds"]):
        for kind in sorted({it["kind"] for it in items}):
            vals = sorted({it.get(key) for it in items}) if key else [None]
            for v in vals:
                sel = torch.tensor(
                    [
                        it["kind"] == kind and (key is None or it.get(key) == v)
                        for it in items
                    ]
                )
                if not sel.any():
                    continue
                mv = (pos[ci, sel, :, 0] - pos[ci, sel, :, 1]).mean(dim=0)
                node = out.setdefault(cond, {}).setdefault(kind, {})
                if key:
                    node = node.setdefault(str(v), {})
                node.update(
                    {
                        "n": int(sel.sum()),
                        "move": {
                            f"{s:.2f}": round(float(x), 4) for s, x in zip(sig, mv)
                        },
                        "peak": sig[int(mv.argmax())],
                    }
                )
    return out


def read_d0(label: str, canvases: list) -> dict:
    d, fl = arm(label), floor_dir()
    ref_cf = cf_curves(fl / "cf_sense_ja" / "cf_sense.pt")
    res = {
        "512": {
            "en": en_totals(ref_en_reads()),
            "native": native_table(_load(fl / "native" / "native_reads.json")),
            "cf_sense_ja": ref_cf,
        }
    }
    for c in canvases:
        cf = d / f"cf_sense_ja_{tag(c)}" / "cf_sense.pt"
        cur = cf_curves(cf) if cf.exists() else {}
        res[c] = {
            "en": en_totals(_load(d / f"eval_{tag(c)}" / "eval_reads.json")),
            "native": native_table(_load(d / f"native_{tag(c)}" / "native_reads.json")),
            "cf_sense_ja": cur,
            # same rng + same rows → the same pairs as the 512² read
            "cf_pairs_match": cur.get("pairs") == ref_cf["pairs"],
        }
    for c, r in res.items():
        tr = r["cf_sense_ja"].get("trained", {}).get("id", {})
        nat = " ".join(f"{k} {v['official']}/{v['n']}" for k, v in r["native"].items())
        print(
            f"{c:>8}: en {r['en']['exact']}/{r['en']['n']}  native {nat}  "
            f"cf id peak σ {tr.get('peak')}",
            flush=True,
        )
    return res


def read_d1(label: str, canvases: list, ref: str) -> dict:
    d = arm(label)
    res: dict = {}
    font = Path(D1_FONT).stem
    for t in D1_TEXTS:
        name = f"cf_sense_en_flat_{t}_{font}"
        if ref and (Path(ref) / f"{name}_a1" / "cf_sense.pt").exists():
            res.setdefault("512", {})[t] = cf_curves(
                Path(ref) / f"{name}_a1" / "cf_sense.pt", "px"
            )
        for c in canvases:
            pt = d / f"{name}_{tag(c)}" / "cf_sense.pt"
            if pt.exists():
                res.setdefault(c, {})[t] = cf_curves(pt, "px")
    for c, by_t in res.items():
        for t, cur in by_t.items():
            for kind, by_px in cur.get("base", {}).items():
                peaks = " ".join(f"{px}:{v['peak']}" for px, v in by_px.items())
                print(f"{c:>8} {t:<8} {kind:<6} peak σ by px  {peaks}", flush=True)
    return res


def main():
    args = parse_args()
    canvases = [c.strip() for c in args.canvas.split(",") if c.strip()]
    from common.shapes import parse_shape

    for c in canvases:
        W, H = parse_shape(c)
        assert W % 16 == 0 and H % 16 == 0, f"{c}: sides must be multiples of 16"
    jobs = plan(args.cell, args.label, canvases)
    for stage, argv in jobs:
        print(f"[{stage}] {' '.join(argv)}", flush=True)
    if args.dry_run:
        return
    run_dir = make_run_dir(
        "canvas", label=args.label, root=LINE / "experiments" / "canvas" / "results"
    )
    prepare(args.label)
    from cli import build_parser
    from stages import STAGES
    from stages import run as run_stage

    for stage, argv in jobs:
        print(f"===== {stage} {argv[argv.index('--eval_shape') + 1]}", flush=True)
        run_stage(stage, build_parser(STAGES).parse_args(argv))
    metrics = (
        read_d0(args.label, canvases)
        if args.cell == "d0"
        else read_d1(args.label, canvases, args.d1_ref)
    )
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(arm(args.label))],
        extra={"seed_rows": str(floor_dir()), "cell": args.cell},
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
