#!/usr/bin/env python
"""negswap — where the template sits: the identity grid rendered with the word's
swap as the negative prompt (fable review 10-16 of ``proposal_template.md``).

The rows sit in the cond branch only, so with the negative prompt the item's own
caption with its word swapped, CFG subtracts what the rows do whatever the word
(M: template, text-ness, canvas) and keeps what follows the word (S). If the pale
band goes and the glyph stays, the template lives in M; if the text region goes
with it, text-ness and template are one at the row level.

``hira_cold_hp``'s 13 ``read`` words × p00 / p01 × ``SEEDS`` on identity's render
path (``eval.ruler.render.Renderer``, 512², 28 steps, cfg 4, routed), two
conditions: ``plain`` (negative "", seed 0 read from ``identity/<arm>/`` where it
is on disk) and ``neg`` (negative = the caption with the word swapped, glyph by
glyph, for another of its script none of the word's own; one swap a word, the
same for every arm). The 4 katakana words sit on rows every arm shares
(``retrain_kana``'s katakana): their renders are the same in every arm.

Reads per render: the readers' exact / cer / dup; pixel ``white`` / ``ink`` /
``sat`` over the image and inside the read glyph box (the box that read the word,
else the largest; ``eval.enref.glyph_box``); ``en_cos_out`` against the EN
reference of the same prompt and seed. Renders → ``OUT/negswap/<arm>/<cond>/``;
read → ``results/<ts>-negswap-<label>/`` (tallies, paired neg vs plain per arm,
sheets EN | plain | neg per arm).

    make daemon-run ARGS="project/cjk_anima_reseed/probes/negswap.py --label floor_vs_cold"
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reseed import HOME, OUT, SCALE_OUT, bootstrap  # noqa: E402

bootstrap()
os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"  # the runs trained routed, read routed

ARMS = [
    "seed_retrain_0930",
    "hira_cold_hp",
    "hira_cold_sball_lr3",
    "hira_cold_sball_lr3_x2",
]
PROMPTS, SEEDS = (0, 1), (0, 1)
CONDS = ("plain", "neg")
ROOT = OUT / "negswap"
IDENTITY = OUT / "identity"
SRC = SCALE_OUT / "retrain_kana" / "native_r4_plain" / "native_reads.json"
ENREF = SCALE_OUT / "native_enref" / "512_28_4"
HIRA = "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろわをんゔ"
KATA = "ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロワヲンヴー"


def swap(word: str) -> str:
    al = HIRA if all(c in HIRA for c in word) else KATA
    rng = random.Random(f"negswap-{word}")
    pool = sorted(set(al) - set(word) - {"ー"})
    return "".join(rng.choice(pool) for _ in word)


def grid() -> list:
    from reseed.config import load

    prompts = {
        m["pi"]: m["prompt"] for m in json.loads(SRC.read_text(encoding="utf-8"))
    }
    out = []
    for k in load("hira_cold_hp").read:
        w = swap(k)
        for pi in PROMPTS:
            for s in SEEDS:
                out.append(
                    {
                        "key": k,
                        "swap": w,
                        "script": "hira" if all(c in HIRA for c in k) else "kata",
                        "pi": pi,
                        "seed": s,
                        "caption": f'{prompts[pi]}. Text reads as "{k}".',
                        "negative": f'{prompts[pi]}. Text reads as "{w}".',
                        "stem": f"{k}_p{pi:02d}_s{s}",
                    }
                )
    return out


def file_of(arm: str, cond: str, m: dict) -> Path:
    """A plain seed-0 render is identity's own (same path, caption and seed)."""
    if cond == "plain" and m["seed"] == 0:
        f = IDENTITY / arm / f"{m['stem']}.png"
        if f.exists():
            return f
    return ROOT / arm / cond / f"{m['stem']}.png"


def render(names: list) -> None:
    import copy

    from common.models import decode_image
    from eval.ruler.arms import tables
    from eval.ruler.render import Renderer, check
    from library.inference.generation import generate

    g = grid()
    _, tabs = tables(names + ["retrain_kana"])
    r = Renderer()
    check(r, tabs["retrain_kana"])
    t0 = time.time()
    for a in names:
        r.set_arm(tabs[a])
        todo = [(c, m) for c in CONDS for m in g if not file_of(a, c, m).exists()]
        for n, (c, m) in enumerate(todo):
            a2 = copy.deepcopy(r.args)
            a2.prompt, a2.seed, a2.image_size = m["caption"], m["seed"], (512, 512)
            a2.negative_prompt = m["negative"] if c == "neg" else ""
            with r.torch.no_grad():
                lat = generate(a2, r.gen, r.shared)
            fn = file_of(a, c, m)
            fn.parent.mkdir(parents=True, exist_ok=True)
            decode_image(r.vae, lat, r.device).save(fn)
            print(
                f"  {a} {c}: {n + 1} / {len(todo)} {m['stem']} "
                f"({(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )
    del r
    import torch

    torch.cuda.empty_cache()


def pixels(path: Path, box) -> dict:
    import numpy as np
    from PIL import Image

    px = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0

    def reads(p):
        hi, lo = p.max(-1), p.min(-1)
        return {
            "white": float(((lo > 0.85) & (hi - lo < 0.08)).mean()),
            "ink": float((hi < 0.25).mean()),
            "sat": float((hi - lo).mean()),
        }

    out = reads(px)
    if box:
        x0, y0, x1, y1 = (int(v) for v in box)
        c = px[max(0, y0) : y1, max(0, x0) : x1]
        if c.size:
            out.update({f"box_{k}": v for k, v in reads(c).items()})
    return out


def read(names: list) -> dict:
    import torch

    from common.readers import Readers, load_bgr
    from eval.enref import EnRef, enref_file, glyph_box, largest_box
    from eval.ruler.score import score_text

    g = grid()
    rd = None
    recs: dict = {}
    for a in names:
        for c in CONDS:
            f = ROOT / a / f"reads_{c}.json"
            got = json.loads(f.read_text("utf-8")) if f.exists() else {}
            todo = [m for m in g if m["stem"] not in got]
            if todo:
                rd = rd or Readers("cuda")
                for m in todo:
                    got[m["stem"]] = rd.read_image(
                        load_bgr(file_of(a, c, m)), whole=True
                    )
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(got, ensure_ascii=False, indent=1), "utf-8")
            recs[a, c] = {}
            for m in g:
                x = {
                    **m,
                    **score_text(m["key"], got[m["stem"]]),
                    "reads": got[m["stem"]],
                }
                x["text"], x["file"] = m["key"], str(file_of(a, c, m))
                x.update(pixels(Path(x["file"]), glyph_box(x)))
                recs[a, c][m["stem"]] = x
    rd = rd or Readers("cuda")
    boxes = {}
    for pi in PROMPTS:
        for s in SEEDS:
            ref = enref_file(ENREF, pi, s)
            assert ref.is_file(), f"no EN reference {ref}"
            boxes[ref.name] = largest_box(rd.read_image(load_bgr(ref), whole=True))
    er = EnRef("cuda", ENREF, boxes)
    for r in recs.values():
        for x in r.values():
            sc = er.score(x)
            x["en_cos"], x["en_cos_out"], x["box_iou"] = sc if sc else (None,) * 3
    del rd, er
    torch.cuda.empty_cache()
    return recs


KEYS = ("white", "ink", "sat", "box_white", "box_ink", "box_sat", "en_cos_out")


def tally(rs: list) -> dict:
    def mean(k):
        v = [float(r[k]) for r in rs if r.get(k) is not None]
        return round(sum(v) / len(v), 4) if v else None

    return {
        "n": len(rs),
        "exact": sum(r["exact"] for r in rs),
        "contained": sum(r["contained"] for r in rs),
        "cer": mean("cer"),
        "dup": sum(r["dup"] for r in rs),
        **{k: mean(k) for k in KEYS},
    }


def paired(a: dict, b: dict, stems: list) -> dict:
    """neg (a) against plain (b) on the same slot: exact wins / losses and, per
    pixel read, mean Δ with up / down."""
    from math import comb

    w = sum(a[s]["exact"] and not b[s]["exact"] for s in stems)
    lo = sum(b[s]["exact"] and not a[s]["exact"] for s in stems)
    n = w + lo
    p = (
        min(1.0, 2 * sum(comb(n, k) for k in range(min(w, lo) + 1)) / 2**n)
        if n
        else 1.0
    )
    out = {"exact_wins": w, "exact_losses": lo, "p": round(p, 4)}
    for k in KEYS + ("cer",):
        d = [
            float(a[s][k]) - float(b[s][k])
            for s in stems
            if a[s].get(k) is not None and b[s].get(k) is not None
        ]
        if d:
            out[k] = {
                "d": round(sum(d) / len(d), 4),
                "up": sum(x > 0 for x in d),
                "down": sum(x < 0 for x in d),
            }
    return out


def sheets(recs: dict, names: list, out: Path) -> None:
    from PIL import Image

    from common.readers import contact_sheet
    from eval.enref import enref_file

    out.mkdir(parents=True, exist_ok=True)
    g = grid()
    cols = 1 + 2 * len(names)
    for pi in PROMPTS:
        for s in SEEDS:
            rows = []
            for m in [m for m in g if m["pi"] == pi and m["seed"] == s]:
                rows.append(
                    (
                        Image.open(enref_file(ENREF, pi, s)).convert("RGB"),
                        [f"EN p{pi:02d} s{s}", m["key"], f"neg {m['swap']}", ""],
                    )
                )
                for a in names:
                    for c in CONDS:
                        x = recs[a, c][m["stem"]]
                        rows.append(
                            (
                                Image.open(x["file"]).convert("RGB"),
                                [
                                    f"{a.replace('hira_cold_', '')} {c}",
                                    f"{'✓' if x['exact'] else '✗'} cer {x['cer']:.2f}",
                                    f"bw {x.get('box_white', 0):.2f} bs {x.get('box_sat', 0):.2f}",
                                    f"w {x['white']:.2f} s {x['sat']:.2f}",
                                ],
                            )
                        )
            contact_sheet(rows, out / f"sheet_p{pi:02d}_s{s}.png", thumb=200, cols=cols)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--label", required=True)
    a = p.parse_args()
    names = a.arms.split(",")
    from eval.ruler.arms import arm_dirs

    unknown = set(names) - set(arm_dirs())
    assert not unknown, f"unknown arms {unknown}"
    render(names)
    recs = read(names)
    g = grid()
    res: dict = {"arms": {}, "pairs": {}}
    for arm in names:
        res["arms"][arm] = {}
        res["pairs"][arm] = {}
        for script in ("hira", "kata"):
            stems = [m["stem"] for m in g if m["script"] == script]
            for c in CONDS:
                res["arms"][arm][f"{script} {c}"] = tally(
                    [recs[arm, c][s] for s in stems]
                )
            res["pairs"][arm][script] = paired(
                recs[arm, "neg"], recs[arm, "plain"], stems
            )
        print(
            f"  {arm}: {json.dumps(res['arms'][arm], ensure_ascii=False)}", flush=True
        )
        print(
            f"    neg vs plain: {json.dumps(res['pairs'][arm], ensure_ascii=False)}",
            flush=True,
        )
    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir(
        "cjk_anima_reseed", label=f"negswap-{a.label}", root=HOME / "results"
    )
    sheets(recs, names, run_dir / "sheets")
    (run_dir / "renders.json").write_text(
        json.dumps(
            {f"{k[0]}|{k[1]}": list(v.values()) for k, v in recs.items()},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    write_result(
        run_dir,
        script="project/cjk_anima_reseed/probes/negswap.py",
        args=vars(a),
        metrics=res,
    )
    print(f"negswap → {run_dir}", flush=True)


if __name__ == "__main__":
    main()
