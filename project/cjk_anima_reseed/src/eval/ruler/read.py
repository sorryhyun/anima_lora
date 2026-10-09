"""The read: every render read (readers cached per arm) and scored
(``score``), each arm paired against the floor arms per bin and on the
unseen strings (``stats``) → ``results/<ts>-ruler-<label>/`` (+ sheets);
``sample`` draws random strings of a read into larger sheets."""

from __future__ import annotations

import json
from pathlib import Path

from eval.ruler import BINS, EN, FLOOR, RULER, VIEW, items, mode_dir, render_file
from eval.ruler.score import AtSim, flat_white, outside_mask, score_page, score_text
from eval.ruler.stats import paired, tally

UNSEEN = 0.15  # cov3 at or below: the string's trigrams the arms barely trained on


def read_renders(names: list) -> dict:
    """``{arm: {i: record}}``; the reads are cached in ``<arm>/reads.json``,
    the page scores recomputed (cheap) every time."""
    import torch
    import torch.nn.functional as F

    from common.readers import Readers, load_bgr
    from eval.enref import EnRef

    its = items()
    rd = None
    raw: dict = {}
    for a in [EN, *names]:
        f = mode_dir() / a / "reads.json"
        got = json.loads(f.read_text("utf-8")) if f.exists() else {}
        todo = [m for m in its if str(m["i"]) not in got]
        missing = [m["i"] for m in todo if not render_file(a, m["i"]).exists()]
        assert not missing, f"{a}: not rendered: {missing[:8]}…"
        if todo:
            rd = rd or Readers("cuda")
            for m in todo:
                got[str(m["i"])] = rd.read_image(
                    load_bgr(render_file(a, m["i"])), whole=True
                )
            f.write_text(
                json.dumps(got, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            print(f"read {a}: {len(todo)}", flush=True)
        raw[a] = got
    del rd
    torch.cuda.empty_cache()
    pe = EnRef("cuda", mode_dir() / EN, {})
    at = AtSim("cuda")
    recs: dict = {}
    for a in names:
        recs[a] = {}
        for m in its:
            i = m["i"]
            reads, en_reads = raw[a][str(i)], raw[EN][str(i)]
            f, ref = render_file(a, i), render_file(EN, i)
            fi, hw = pe.tokens(f)
            fr, _ = pe.tokens(ref)
            boxes = [r["box"] for r in reads + en_reads if not r.get("whole")]
            keep = outside_mask(pe, fi.shape[0], hw, boxes)
            if keep.sum() < 4:
                keep[:] = True
            recs[a][i] = {
                "i": i,
                "bin": m["bin"],
                "unseen": (m["cov3"] or 0) <= UNSEEN,
                "text": m["text"],
                "file": str(f),
                **score_text(m["text"], reads),
                **score_page(m["text"], reads, en_reads),
                **at(f, ref),
                "en_tok_out": float(
                    F.cosine_similarity(fi[keep], fr[keep], dim=1).mean()
                ),
                "fw_over_en": flat_white(str(f)) - flat_white(str(ref)),
                "reads": reads,
            }
    return recs


def sheets(recs: dict, out: Path, per: int = 8) -> None:
    """Per bin, ``per`` strings a sheet: a row per string, EN ref | the arms."""
    from PIL import Image

    from common.readers import contact_sheet

    out.mkdir(parents=True, exist_ok=True)
    names = list(recs)
    its = items()
    for b in BINS:
        xs = [m for m in its if m["bin"] == b]
        for k in range(0, len(xs), per):
            rows = []
            for m in xs[k : k + per]:
                i = m["i"]
                rows.append(
                    (
                        Image.open(render_file(EN, i)).convert("RGB"),
                        [f"r{i:02d} {m['text']}", f"EN {m['en']}"],
                    )
                )
                for a in names:
                    r = recs[a][i]
                    mark = "✓" if r["exact"] else "≤1" if r["le1"] else ""
                    rows.append(
                        (
                            Image.open(r["file"]).convert("RGB"),
                            [
                                f"{a} {mark}",
                                f"{r['best'][:22]}",
                                f"cer {r['cer']:.2f} tok {r['en_tok_out']:.3f}",
                                f"P {r['g_p']:.2f} R {r['g_r']:.2f} "
                                f"F1 {r['g_f1']:.2f} on {r['a_p']:.2f}",
                            ],
                        )
                    )
            contact_sheet(
                rows, out / f"sheet_{b}_{k // per}.png", thumb=256, cols=1 + len(names)
            )


def read(names: list, label: str, script: str) -> Path:
    from bench._common import make_run_dir, write_result

    from reseed import HOME

    recs = read_renders(names)
    t = tally(recs)
    for a, gs in t.items():
        for g, c in gs.items():
            print(f"  {a:<18} {g:<13} {c}", flush=True)
    pairs = {}
    for k, a in enumerate(names):
        # every arm against the floor and every arm named before it
        for b in dict.fromkeys([*FLOOR[1:], *names[:k]]):
            if a != b and b in recs:
                pairs[f"{a} vs {b}"] = pr = paired(recs[a], recs[b])
                print(f"  {a} vs {b}: {pr['all']}", flush=True)
    run_dir = make_run_dir(
        "cjk_anima_reseed", label=f"ruler-{VIEW.mode}-{label}", root=HOME / "results"
    )
    sheets(recs, run_dir / "sheets")
    (run_dir / "renders.json").write_text(
        json.dumps(
            {a: list(r.values()) for a, r in recs.items()}, ensure_ascii=False, indent=1
        ),
        encoding="utf-8",
    )
    write_result(
        run_dir,
        script=script,
        args={
            "arms": names,
            "label": label,
            "prompts": VIEW.mode,
            "unseen": UNSEEN,
            "pack": VIEW.pack,
            "marks_only": VIEW.marks_only,
        },
        label=label,
        metrics={"tally": t, "paired": pairs},
        artifacts=[str(RULER)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)
    return run_dir


def sample(
    run_dir: Path, n: int, seed: int, has: str = "", per: int = 4, thumb: int = 384
) -> Path:
    """``n`` random strings of a read (its ``renders.json``; ``has``: only
    those holding one of these chars), ``per`` a sheet: a row per string, EN
    ref | every arm read, larger than the read's sheets →
    ``<run_dir>/random_s<seed>[_<has>]/``."""
    import random

    from PIL import Image

    from common.readers import contact_sheet

    recs = json.loads((run_dir / "renders.json").read_text(encoding="utf-8"))
    names = list(recs)
    by = {a: {r["i"]: r for r in rs} for a, rs in recs.items()}
    its = {
        m["i"]: m
        for m in items()
        if all(m["i"] in b for b in by.values())
        and (not has or set(has) & set(m["text"]))
    }
    n = min(n, len(its))
    pick = sorted(random.Random(seed).sample(sorted(its), n))
    out = run_dir / (f"random_s{seed}" + (f"_{has}" if has else ""))
    out.mkdir(parents=True, exist_ok=True)
    for k in range(0, n, per):
        rows = []
        for i in pick[k : k + per]:
            m = its[i]
            rows.append(
                (
                    Image.open(render_file(EN, i)).convert("RGB"),
                    [f"r{i:02d} {m['bin']} {m['text']}", f"EN {m['en']}"],
                )
            )
            for a in names:
                r = by[a][i]
                mark = (
                    "✓"
                    if r["exact"]
                    else "≤1"
                    if r["le1"]
                    else "≤2"
                    if r["le2"]
                    else ""
                )
                rows.append(
                    (
                        Image.open(r["file"]).convert("RGB"),
                        [f"{a} {mark}", r["best"][:24], f"cer {r['cer']:.2f}"],
                    )
                )
        contact_sheet(
            rows, out / f"random_{k // per}.png", thumb=thumb, cols=1 + len(names)
        )
    print(f"{n} strings {pick} → {out}", flush=True)
    return out
