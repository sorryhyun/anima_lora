#!/usr/bin/env python
"""G1 — rows trained on the reader's loss through G-D's path, one row a glyph.

Policy and path: ``gd.Policy`` (``m15v11``, 4 Euler steps, pixel-head decode, rows
``seed_1008``) and ``gd.Reader``. A group is one glyph on one training scene
(``<scene, bubble frame>. Text reads as "<glyph>".``) at 4 seeds:

- steps 0 – 2 for the 4 renders in one forward each, no graph;
- per render, the last step once with a graph (it is the render's own last step); the
  page is read; a page with a box takes the reader's cross-entropy of the glyph on the
  picked crop (``gd.pick``), a page with none −log of the detector's top score (the
  detector's backward runs on those pages only); the loss is pulled to the glyph's row;
- the render's row gradient is clipped to ``clip`` and added to the row's sum.

With ``hayai_weight`` in the config (0 or absent: off), a page with a box also takes
hayai v2.7's cross-entropy of the glyph on the same crop (``gd.Hayai.ce``) at that
weight, and the read gains hayai's ``solo`` on the tuned reader's boxes (``hy_solo``).

With ``skip_lone_read`` in the config, a training page where the detector finds one box
is not read (``gd.Reader.read(skip_lone=True)``): the box is the pick whatever it
reads, and the loss is the same. The log counts such a page under ``lone`` instead of
``box`` / ``low``, and its ``exact`` is the loss forward's argmax being the target at
every token (``gd.Reader.ce``'s ``hit``), which leaves out a ``fold_small`` pair's
other glyph. The read on held-out scenes reads every box either way.

``reader_attn`` sets the tuned reader's attention (``gd.Reader``): ``flash_attention_2``
when absent, ``sdpa`` for the runs before 2026-11-25. hayai stays on its own SDPA in
fp32. ``reader_compile = true`` compiles the reader's layers for the loss forward and
backward (``gd.Reader.compile``; the read stays eager).

A row steps when ``accum`` of its renders are in: the mean gradient joins the row's
momentum (``beta``), and the row moves ``eta`` row norms against the momentum's
direction. A pass is every (glyph, training scene) once, shuffled; pass p draws seeds
``train_seed0 + 4p …``. With ``max_steps`` in the config a row that has taken that many
steps is passed over for the rest of the run, and a save is named by the steps taken.

Three keys, each off when absent, read a row's own renders before its step. A step's
renders were drawn at the row as it stood before that step, so their count reads that
row.

- ``near_score``: a render whose picked box reads a look-alike of the glyph
  (:func:`near`: the same kana with another voiced mark, and へ / ヘ and their voiced
  forms across the scripts) counts that much of an exact read. A step's ``score`` is
  its exact reads plus ``near_score`` times its look-alikes; the loss does not change.
  A ``lone`` page is not read and counts no look-alike. The read's tables gain
  ``near`` (pages with no solo and a box that reads a look-alike) and ``score``.
- ``taper``: the step's length is ``eta`` times ``min(1, 2 (1 − score / renders))``:
  full while half the renders or fewer read, none when all of them do.
- ``[back]`` (``drop``, ``window``, ``over``): a row's window is the summed score of
  its last ``window`` steps. The best window so far keeps the row its better step was
  drawn at. A window under the best by more than ``drop`` returns the row there with
  its momentum cleared, in place of that step (it counts as a step); the best and the
  window then start again from the returned row. With ``over`` (absent: 0) a window
  is a best only above that share of its renders, so at 0.5 a row returns only to
  where ``taper`` was shortening its step.

Glyphs come from the config's ``hira``, ``kata`` and ``kanji`` strings; a class may be
left out. With ``fold_small`` a small kana and its full-size glyph are one glyph to the
loss and the read (:func:`same`): a small kana's row takes the full-size glyph as its
target, either read counts as ``solo`` for either row, and a record of such a row also
holds ``strict``, the count with the glyph itself only.

The read (same process): every glyph on the read scenes at the read seeds, each page
read as training reads it; the start rows before training, then the rows after each
pass in ``save_passes`` (absent: the last pass), which are also written as
``rows_s<steps a row>.pt``. An arm is ``start`` or ``s<steps a row>``. Per glyph
``solo`` / ``on_page`` / ``extra`` (``glyph_g0.page_terms``), the next glyph's ``solo``
on the same pages (the null), the reader's cross-entropy on the picked crop, cases.
Tables: the start rows against each save. Sheets: per glyph the start rows and the
last save as rows ``A`` / ``B`` in an order drawn per glyph (``sheets/key.json`` holds
it).

With ``[start_shift]`` in the config (``run``, ``save``), every row trained here begins
at its start plus the mean displacement, to that save, of the rows that run trained, the
row's own left out (:func:`shared_start`). The ``start`` arm of the read stays the
unshifted rows, and a row's ``drift`` counts from the shifted start. The shifted rows
are read as arm ``s00`` only with ``read_shift = true`` (the runs before 11-26 read
them always).

``frame`` is one frame or a list of them (``["plain", "bubble"]``). With a list a
group's frame is drawn by its glyph, its scene and the pass, so a row meets each scene
in each frame in turn and each pass holds the frames in equal parts; the read runs
every page in each frame (a record holds ``frame``, a table ``by_frame``, and the
sheets are a folder a frame), and a log line counts its renders ``by_frame``. Scenes
are ids of ``glyph_scenes.scenes`` (both screens' selections).

``start_from = "<run>"`` takes the ``start`` arm from that run's ``records.json``
instead of rendering it, when it holds every page this run reads (glyph, scene, seed,
frame) and was read on the same rows, ``conf_min``, ``fold_small`` and readers; the null
is counted again for this run's glyphs. Otherwise the arm is read here.

``[read_refs]`` names other runs' saves (``<arm> = { run, save }``): the rows this run
trains, as that save holds them, are read on this run's pages as arm ``<arm>``, before
training.

The run's settings, question and gate are its config's header
(``configs/<handle>_<mmdd>_<hhmm>.toml``). Writes ``output/cjk_rl/g1/<config stem>/``
(``log.jsonl`` a row step, ``rows.pt`` and a ``rows_s<steps>.pt`` a save, the read's
pages, and ``records.json`` rewritten after each arm's read) and
``results/<ts>-g1-<config stem>/``.

    make daemon-run ARGS="--queue --stall-timeout 900 project/cjk_rl/src/g1.py --config project/cjk_rl/configs/<name>.toml"
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import tomllib
import unicodedata
from math import comb
from pathlib import Path

LINE = Path(__file__).resolve().parents[1]  # project/cjk_rl
sys.path.insert(0, str(LINE / "probes"))

import glyph_g0 as G  # noqa: E402
import glyph_scenes as S  # noqa: E402
import turbo_read as T  # noqa: E402

import gd  # noqa: E402

ROOT = T.REPO / "output" / "cjk_rl" / "g1"
CLASSES = ("hira", "kata", "kanji")  # a config may leave a class out
# a small kana and its full-size glyph: one glyph to a reader of a lone glyph
SMALL = dict(
    zip(
        "ぁぃぅぇぉっゃゅょァィゥェォッャュョヵヶ",
        "あいうえおつやゆよアイウエオツヤユヨカケ",
    )
)
FULL = {v: k for k, v in SMALL.items()}


def same(key: str, fold: bool) -> list:
    """The reads that count as ``key``, the loss's target first: with ``fold`` a small
    kana is its full-size glyph, and a full-size glyph read as its small one counts."""
    if fold and key in SMALL:
        return [SMALL[key], key]
    if fold and key in FULL:
        return [key, FULL[key]]
    return [key]


def shape(ch: str) -> str:
    """A kana without its voiced mark; へ and ヘ, one shape in both scripts, as へ."""
    base = unicodedata.normalize("NFD", ch)[0]
    return "へ" if base == "ヘ" else base


def near(key: str, fold: bool) -> list:
    """The kana that are look-alikes of ``key``: the same :func:`shape`, and not a
    read that counts as ``key`` (:func:`same`)."""
    ok = same(key, fold)
    shapes = {shape(k) for k in ok}
    kana = [chr(c) for c in (*range(0x3041, 0x3097), *range(0x30A1, 0x30FB))]
    return [c for c in kana if shape(c) in shapes and c not in ok]


def window_read(scores: list, best: float | None, back: dict, renders: int) -> tuple:
    """``(window, verdict)`` for a row's scores since its last return, ``renders`` a
    step: ``window`` is the sum of the last ``back["window"]`` (``None`` with fewer);
    ``verdict`` is ``"best"`` when it is the first or at least ``best`` and over
    ``back["over"]`` of the window's renders, ``"back"`` when it is under ``best`` by
    more than ``back["drop"]``, else ``None``."""
    n = back["window"]
    if len(scores) < n:
        return None, None
    w = sum(scores[-n:])
    if best is None or w >= best:
        return w, ("best" if w > back.get("over", 0.0) * n * renders else None)
    return w, ("back" if best - w > back["drop"] else None)


def terms(keys: list, reads: list) -> dict:
    """``glyph_g0.page_terms`` over the reads that count as the glyph: the best of
    each term."""
    ts = [G.page_terms(k, reads) for k in keys]
    return {
        "solo": max(t["solo"] for t in ts),
        "on_page": max(t["on_page"] for t in ts),
        "extra": min(t["extra"] for t in ts),
        "r_seq": max(t["r_seq"] for t in ts),
    }


def glyphs(cfg: dict, smoke: bool) -> list:
    """``[(cls, key, alt)]``; ``alt`` is the next glyph of the class that does not
    count as ``key`` (:func:`same`)."""
    out = []
    fold = bool(cfg.get("fold_small"))
    for cls in CLASSES:
        keys = list(cfg.get(cls, ""))
        for n, key in enumerate(keys):
            if smoke and n:
                continue
            later = keys[n + 1 :] + keys[:n]
            out.append((cls, key, next(k for k in later if k not in same(key, fold))))
    return out


def frames_of(cfg: dict) -> list:
    f = cfg["frame"]
    return [f] if isinstance(f, str) else list(f)


def start_records(cfg: dict, gl: list, scenes: list, hayai: bool) -> list | None:
    """The ``start`` arm of run ``cfg["start_from"]`` as this run's records, or ``None``
    with the reason printed when a page is missing or the read's settings differ."""
    src = cfg["start_from"]
    old = tomllib.loads((LINE / "configs" / f"{src}.toml").read_text("utf-8"))
    for k, default in (
        ("rows", None),
        ("conf_min", None),
        ("fold_small", False),
        ("reader_attn", "flash_attention_2"),
    ):
        if old.get(k, default) != cfg.get(k, default):
            print(
                f"  start_from {src}: `{k}` differs, the start rows are read",
                flush=True,
            )
            return None
    frames, many = frames_of(cfg), len(frames_of(cfg)) > 1
    old_frames = frames_of(old)
    at = {}
    for r in json.loads((ROOT / src / "records.json").read_text("utf-8")):
        if r["arm"] == "start":
            f = r.get("frame", old_frames[0])
            at[(r["key"], r["scene"], r["seed"], f)] = r
    out = []
    for f in frames:
        for si in scenes:
            for cls, key, alt in gl:
                for seed in cfg["read_seeds"]:
                    r = at.get((key, si, seed, f))
                    if r is None or (hayai and "hy_solo" not in r):
                        print(
                            f"  start_from {src}: no page for {key} scene {si} seed "
                            f"{seed} {f}, the start rows are read",
                            flush=True,
                        )
                        return None
                    r = {k: v for k, v in r.items() if k != "frame"}
                    r |= {"cls": cls, "null": int(_norm(alt) in r["boxes"])}
                    out.append(r | ({"frame": f} if many else {}))
    print(f"  start rows: {len(out)} pages of {src}'s read", flush=True)
    return out


def saved_rows(spec: dict, row: dict, start):
    """``start`` with each row of ``row`` as save ``spec["save"]`` of run
    ``spec["run"]`` holds it."""
    import torch

    sv = torch.load(
        ROOT / spec["run"] / f"rows_{spec['save']}.pt",
        map_location="cpu",
        weights_only=False,
    )
    table = start.clone()
    for k, i in row.items():
        assert sv["rows"][k] == i, (k, i, sv["rows"][k])
        table[i] = sv["raw"][i].to(start)
    return table


def shared_start(spec: dict, row: dict, start):
    """``start`` with each row of ``row`` (``{glyph: index}``) moved by the mean
    displacement, to save ``spec["save"]``, of the rows run ``spec["run"]`` trained,
    the glyph's own left out."""
    import torch

    sv = torch.load(
        ROOT / spec["run"] / f"rows_{spec['save']}.pt",
        map_location="cpu",
        weights_only=False,
    )
    init = sv["init"].float()
    disp = {k: sv["raw"][i].float() - init[i] for k, i in sv["rows"].items()}
    table = start.clone()
    for k, i in row.items():
        v = torch.stack([d for j, d in disp.items() if j != k]).mean(0)
        table[i] = start[i] + v.to(start)
    return table


def train(
    pol,
    rdr,
    cfg: dict,
    gl: list,
    sc: dict,
    scenes: list,
    out: Path,
    hy=None,
    on_save=None,
) -> dict:
    """``on_save(pass number, rows)`` runs after each pass in ``cfg["save_passes"]``,
    once the rows are on disk; its seconds are left out of the run's clock."""
    import torch

    W, H = G.SHAPE
    frames, accum = frames_of(cfg), cfg["accum"]
    skip_lone = bool(cfg.get("skip_lone_read"))
    dim = pol.raw.shape[1]
    init = pol.raw.detach().clone()
    cap = {
        (k, si, f): G.framed(sc[si][f], k)
        for _, k, _ in gl
        for si in scenes
        for f in frames
    }
    row = {k: pol.row_of(cap[(k, scenes[0], frames[0])]) for _, k, _ in gl}
    turn = {k: n for n, k in enumerate(row)} | {si: n for n, si in enumerate(scenes)}
    ok = {k: same(k, bool(cfg.get("fold_small"))) for k in row}
    w_near = float(cfg.get("near_score") or 0.0)
    nr = {
        k: [_norm(x) for x in near(k, bool(cfg.get("fold_small")))] if w_near else []
        for k in row
    }
    taper, back = bool(cfg.get("taper")), cfg.get("back")
    trail = {k: [] for k in row}  # (the row a step's renders were drawn at, score)
    best: dict = {}  # key -> (window score, row)
    returns = dict.fromkeys(row, 0)
    acc = {k: torch.zeros(dim, device=pol.dev) for k in row}
    mom = {k: torch.zeros(dim, device=pol.dev) for k in row}
    prev: dict = {}
    since = {k: [] for k in row}  # the renders since the row's last step
    steps = dict.fromkeys(row, 0)
    cap_steps = cfg.get("max_steps")  # a row at the cap is passed over
    rng = random.Random(cfg["shuffle_seed"])
    log = open(out / "log.jsonl", "a")
    t_run, n_groups, phase = time.time(), 0, {}

    def lap(name: str, t0: float) -> float:
        torch.cuda.synchronize()
        phase[name] = phase.get(name, 0.0) + time.time() - t0
        return time.time()

    def dump(path: Path, table) -> None:
        torch.save(
            {
                "ext_ids": pol.r.delta.ext_ids,
                "rows": {k: row[k] for k in row},
                "raw": table.cpu(),  # effective units, every ext row
                "init": init.cpu(),
                "row_scale": pol.row_scale,
                "steps": dict(steps),
                "cfg": cfg,
            },
            path,
        )

    per_row = cfg["passes"] * len(scenes)  # groups a row
    if cap_steps:
        per_row = min(per_row, cap_steps * accum // 4)
    total_groups = per_row * len(row)
    for p in range(cfg["passes"]):
        seeds = [cfg["train_seed0"] + 4 * p + k for k in range(4)]
        order = [(k, si) for k in row for si in scenes]
        rng.shuffle(order)
        for key, si in order:
            if cap_steps and steps[key] >= cap_steps:
                continue
            t = time.time()
            fr = frames[(turn[key] + turn[si] + p) % len(frames)]
            e_graph = pol.embed(cap[(key, si, fr)], grad=True)
            e = e_graph.detach()
            x3 = pol.head_steps(pol.noise(seeds, G.SHAPE), e)
            t = lap("head", t)
            for b in range(len(seeds)):
                px, ei = pol.last(x3[b : b + 1], e)
                t = lap("last_fwd", t)
                img = px.detach()
                bgrs, reads = rdr.read(img, skip_lone=skip_lone)
                t = lap("read", t)
                tgt = ok[key][0]
                case, r = gd.pick(tgt, reads[0], cfg["conf_min"])
                hit: list = []  # a lone page's exact, from the loss forward
                leaf = img.clone().requires_grad_(True)
                p01 = (gd.ste_clamp(leaf, -1.0, 1.0) + 1) / 2
                aux = None
                if r is None:
                    top = rdr.det_scores(p01).max(1).values
                    loss = top.clamp_min(1e-6).log().neg().sum()
                    total = loss
                else:
                    box = gd.crop_box(r["box"], W, H)
                    loss = rdr.ce(
                        [p01[0]],
                        bgrs,
                        [box],
                        [tgt],
                        hit=hit if case == "lone" else None,
                    ).sum()
                    total = loss
                    if hy is not None:
                        aux = hy.ce([p01[0]], [box], [tgt]).sum()
                        total = loss + cfg["hayai_weight"] * aux
                g_px = torch.autograd.grad(total, leaf)[0]
                t = lap("loss", t)
                g_e = torch.autograd.grad((px * g_px).sum(), ei)[0]
                g = pol.row_grads(e_graph, g_e)[0, row[key]]
                del px, ei
                t = lap("last_bwd", t)
                n = float(g.norm())
                acc[key] += g * min(1.0, cfg["clip"] / max(n, 1e-30))
                read_as = G.boxes([r]) if r is not None and r.get("sfx") else []
                since[key].append(
                    {
                        "case": case,
                        "frame": fr,
                        "loss": float(loss.detach()),  # the tuned reader's term
                        "hy": None if aux is None else float(aux.detach()),
                        "norm": n,
                        "exact": hit[0]
                        if case == "lone"
                        else read_as in [[_norm(x)] for x in ok[key]],
                        "near": read_as in [[x] for x in nr[key]],
                    }
                )
            del e_graph
            n_groups += 1
            if len(since[key]) >= accum:
                rs = since[key]
                gbar = acc[key] / len(rs)
                score = sum(x["exact"] + w_near * x["near"] for x in rs)
                window = verdict = None
                if back:
                    trail[key] = (
                        trail[key][-back["window"] + 1 :] if back["window"] > 1 else []
                    )
                    trail[key].append((pol.raw[row[key]].detach().clone(), score))
                    window, verdict = window_read(
                        [x for _, x in trail[key]],
                        best[key][0] if key in best else None,
                        back,
                        accum,
                    )
                cos = (
                    float(torch.nn.functional.cosine_similarity(gbar, prev[key], dim=0))
                    if key in prev
                    else None
                )
                if verdict == "back":
                    with torch.no_grad():
                        pol.raw[row[key]] = best[key][1]
                    mom[key].zero_()
                    del best[key]
                    trail[key] = []
                    prev.pop(key, None)
                    returns[key] += 1
                    length = None
                else:
                    if verdict == "best":  # of the window's rows, the better read
                        at = max(reversed(trail[key]), key=lambda x: x[1])[0]
                        best[key] = (window, at)
                    mom[key] = cfg["beta"] * mom[key] + gbar
                    d = mom[key] / mom[key].norm().clamp_min(1e-30)
                    length = min(1.0, 2.0 * (1.0 - score / len(rs))) if taper else 1.0
                    with torch.no_grad():
                        pol.raw[row[key]] -= length * cfg["eta"] * pol.row_scale * d
                    prev[key] = gbar.clone()
                steps[key] += 1
                crop = [x for x in rs if x["case"] != "none"]
                rec = {
                    "pass": p,
                    "key": key,
                    "step": steps[key],
                    "renders": len(rs),
                    "gbar": float(gbar.norm()),
                    "cos_prev": cos,
                    "drift": float((pol.raw[row[key]] - init[row[key]]).norm())
                    / pol.row_scale,
                    "cases": {
                        c: sum(x["case"] == c for x in rs)
                        for c in ("box", "low", "lone", "none")
                    },
                    "exact": sum(x["exact"] for x in rs),
                    "near": sum(x["near"] for x in rs),
                    "score": score,
                    "length": length,  # of eta; None: the row returned
                    "window": window,
                    "back": verdict == "back",
                    "ce": sum(x["loss"] for x in crop) / len(crop) if crop else None,
                    "ce_hy": (
                        sum(x["hy"] for x in crop) / len(crop)
                        if crop and hy is not None
                        else None
                    ),
                    "clipped": sum(x["norm"] > cfg["clip"] for x in rs),
                    **(
                        {
                            "by_frame": {
                                f: {
                                    "renders": sum(x["frame"] == f for x in rs),
                                    "exact": sum(
                                        x["exact"] for x in rs if x["frame"] == f
                                    ),
                                    "none": sum(
                                        x["case"] == "none"
                                        for x in rs
                                        if x["frame"] == f
                                    ),
                                }
                                for f in frames
                            }
                        }
                        if len(frames) > 1
                        else {}
                    ),
                    "groups": n_groups,
                    "min": round((time.time() - t_run) / 60, 2),
                }
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log.flush()
                acc[key].zero_()
                since[key] = []
            if n_groups % 16 == 0:
                print(
                    f"  pass {p} group {n_groups} / {total_groups} "
                    f"steps {sum(steps.values())} peak "
                    f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB "
                    f"({(time.time() - t_run) / 60:.1f} min)",
                    flush=True,
                )
        if p + 1 in cfg["save_passes"]:
            t_save = time.time()
            table = pol.raw.detach().clone()
            dump(out / f"rows_s{max(steps.values()):02d}.pt", table)
            if on_save is not None:
                on_save(p + 1, table)
            t_run += time.time() - t_save
    log.close()
    trained = pol.raw.detach().clone()
    dump(out / "rows.pt", trained)
    info = {
        "groups": n_groups,
        "minutes": round((time.time() - t_run) / 60, 2),
        "s_per_group": {k: round(v / n_groups, 3) for k, v in phase.items()},
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "steps": steps,
        "returns": sum(returns.values()),
        "rows_returned": sum(v > 0 for v in returns.values()),
        "drift": {
            k: round(float((trained[row[k]] - init[row[k]]).norm()) / pol.row_scale, 4)
            for k in row
        },
    }
    return {"init": init, "trained": trained, "info": info}


def _norm(s: str) -> str:
    from common.text import norm

    return norm(s)


def read(
    pol,
    rdr,
    cfg: dict,
    gl: list,
    sc: dict,
    scenes: list,
    arms: dict,
    out: Path,
    hy=None,
):
    """``[{arm, cls, key, scene, seed, case, solo, on_page, extra, null, ce, reads}]``
    and the pages on disk."""
    import numpy as np
    import torch
    from common.readers import crop
    from PIL import Image

    W, H = G.SHAPE
    seeds = list(cfg["read_seeds"])
    frames = frames_of(cfg)
    recs = []
    t0 = time.time()
    for arm, table in arms.items():
        pol.raw.data.copy_(table)
        for frame, si in [(f, si) for f in frames for si in scenes]:
            for n, (cls, key, alt) in enumerate(gl):
                if n and n % 32 == 0:
                    print(
                        f"  read {arm} scene {si} glyph {n} / {len(gl)} "
                        f"({(time.time() - t0) / 60:.1f} min)",
                        flush=True,
                    )
                caption = G.framed(sc[si][frame], key)
                ok = same(key, bool(cfg.get("fold_small")))
                _, px = pol.rollout(pol.noise(seeds, G.SHAPE), pol.embed(caption))
                bgrs, reads = rdr.read(px)
                if hy is not None:
                    own = [
                        (b, r)
                        for b, rs in enumerate(reads)
                        for r in rs
                        if not r.get("whole")
                    ]
                    crops = [
                        Image.fromarray(
                            np.ascontiguousarray(crop(bgrs[b], r["box"])[:, :, ::-1])
                        )
                        for b, r in own
                    ]
                    for (_, r), text in zip(own, hy.read(crops)):
                        r["hy"] = text
                p01 = (px.clamp(-1, 1) + 1) / 2
                for b, seed in enumerate(seeds):
                    case, r = gd.pick(ok[0], reads[b], cfg["conf_min"])
                    ce = None
                    if r is not None:
                        with torch.no_grad():
                            ce = float(
                                rdr.ce(
                                    [p01[b]],
                                    [bgrs[b]],
                                    [gd.crop_box(r["box"], W, H)],
                                    [ok[0]],
                                )
                            )
                    tag = f"_{frame}" if len(frames) > 1 else ""
                    f = out / "read" / arm / f"{key}_sc{si:02d}{tag}_s{seed}.webp"
                    f.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(bgrs[b][:, :, ::-1]).save(f, quality=95)
                    recs.append(
                        {"arm": arm, "cls": cls, "key": key, "scene": si, "seed": seed}
                        | ({"frame": frame} if len(frames) > 1 else {})
                        | {"case": case, "ce": ce, "file": str(f)}
                        | terms(ok, reads[b])
                        | {"null": G.page_terms(alt, reads[b])["solo"]}
                        | {"boxes": G.boxes(reads[b])}
                    )
                    if hy is not None:
                        as_hy = [x | {"sfx": x.get("hy")} for x in reads[b]]
                        recs[-1] |= {
                            "hy_solo": terms(ok, as_hy)["solo"],
                            "hy_boxes": G.boxes(as_hy),
                        }
                    if len(ok) > 1:  # the count with the glyph itself only
                        recs[-1] |= {
                            "same": ok,
                            "strict": G.page_terms(key, reads[b])["solo"],
                        }
                        if hy is not None:
                            recs[-1]["hy_strict"] = G.page_terms(key, as_hy)["solo"]
        print(f"  read {arm} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return recs


def sign_p(up: int, down: int) -> float:
    """Two-sided sign test."""
    n, k = up + down, min(up, down)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def tabulate(recs: list, arms: list, soft: tuple | None = None) -> dict:
    """``soft``: ``(near_score, {glyph: its look-alikes})``; a cell then holds ``near``
    (pages with no solo and a box that reads a look-alike) and ``score``."""
    a, b = arms
    keys = list(dict.fromkeys(r["key"] for r in recs))

    def cell(arm, key):
        xs = [r for r in recs if r["arm"] == arm and r["key"] == key]
        crop = [r["ce"] for r in xs if r["ce"] is not None]
        solo = sum(r["solo"] for r in xs)
        nr = (
            sum(
                not r["solo"] and any(x in soft[1][key] for x in r["boxes"]) for r in xs
            )
            if soft
            else 0
        )
        return (
            ({"near": nr, "score": solo + soft[0] * nr} if soft else {})
            | {
                "n": len(xs),
                "solo": sum(r["solo"] for r in xs),
                "on_page": sum(r["on_page"] for r in xs),
                "null": sum(r["null"] for r in xs),
                "extra": round(sum(r["extra"] for r in xs) / len(xs), 2),
                "r_seq": round(sum(r["r_seq"] for r in xs) / len(xs), 3),
                "none": sum(r["case"] == "none" for r in xs),
                "low": sum(r["case"] == "low" for r in xs),
                "ce": round(sum(crop) / len(crop), 3) if crop else None,
            }
            | ({"hy_solo": sum(r["hy_solo"] for r in xs)} if "hy_solo" in xs[0] else {})
        )

    per = {k: {arm: cell(arm, k) for arm in arms} for k in keys}
    out = {"per_glyph": per, "paired": {}, "total": {}}
    ms = ("solo", "on_page", "null", "extra", "r_seq", "none", "low", "ce")
    ms += ("score",) if soft else ()
    for m in ms + (("hy_solo",) if "hy_solo" in recs[0] else ()):
        pairs = [
            (per[k][a][m], per[k][b][m])
            for k in keys
            if per[k][a][m] is not None and per[k][b][m] is not None
        ]
        up = sum(y > x for x, y in pairs)
        down = sum(y < x for x, y in pairs)
        out["paired"][m] = {
            a: round(sum(x for x, _ in pairs) / len(pairs), 3),
            b: round(sum(y for _, y in pairs) / len(pairs), 3),
            "up": up,
            "down": down,
            "p": round(sign_p(up, down), 3),
            "glyphs": len(pairs),
        }
    for arm in arms:
        xs = [r for r in recs if r["arm"] == arm]
        out["total"][arm] = {
            "n": len(xs),
            "solo": sum(r["solo"] for r in xs),
            "on_page": sum(r["on_page"] for r in xs),
            "null": sum(r["null"] for r in xs),
            "none": sum(r["case"] == "none" for r in xs),
            **(
                {m: sum(per[k][arm][m] for k in keys) for m in ("near", "score")}
                if soft
                else {}
            ),
            **{
                f"solo_{c}": sum(r["solo"] for r in xs if r["cls"] == c)
                for c in CLASSES
            },
            **(
                {"hy_solo": sum(r["hy_solo"] for r in xs)} if "hy_solo" in xs[0] else {}
            ),
        }
    return out


def sheets(recs: list, arms: list, scenes: list, out: Path, seed: int) -> None:
    """Per glyph: rows ``A`` and ``B`` for each pair of scenes, 4 seeds a scene."""
    from PIL import Image

    from common.readers import contact_sheet

    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    key_of = {}
    at = {(r["arm"], r["key"], r["scene"], r["seed"]): r for r in recs}
    seeds = sorted({r["seed"] for r in recs})
    for key in dict.fromkeys(r["key"] for r in recs):
        order = list(arms)
        rng.shuffle(order)
        key_of[key] = dict(zip("AB", order))
        rows = []
        for i in range(0, len(scenes), 2):
            for name, arm in zip("AB", order):
                for si in scenes[i : i + 2]:
                    for s in seeds:
                        r = at[(arm, key, si, s)]
                        rows.append(
                            (
                                Image.open(r["file"]).convert("RGB"),
                                [
                                    f"{key} {name} sc{si:02d} s{s}",
                                    " / ".join(r["boxes"])[:26],
                                ],
                            )
                        )
        contact_sheet(rows, out / f"{key}.png", thumb=256, cols=2 * len(seeds))
    (out / "key.json").write_text(
        json.dumps(key_of, ensure_ascii=False, indent=1), "utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="its stem names the run")
    ap.add_argument("--smoke", action="store_true", help="two glyphs, two scenes")
    a = ap.parse_args()

    cfg = tomllib.loads(Path(a.config).read_text("utf-8"))
    name = ("smoke_" if a.smoke else "") + Path(a.config).stem
    sc = S.scenes()
    tr, rd_sc = list(cfg["train_scenes"]), list(cfg["read_scenes"])
    assert not {sc[i]["artist"] for i in tr} & {sc[i]["artist"] for i in rd_sc}, (
        "a read scene's folder is trained on"
    )
    if a.smoke:
        tr, rd_sc = tr[:2], rd_sc[:1]
        cfg = cfg | {"passes": 2, "accum": 4, "save_passes": [1, 2]}
        if cfg.get("max_steps"):
            cfg["max_steps"] = 3  # 2 steps a pass: the cap falls inside pass 2
    cfg = {"save_passes": [cfg["passes"]]} | cfg
    assert cfg["passes"] in cfg["save_passes"], "the last pass is a save"
    gl = glyphs(cfg, a.smoke)
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "log.jsonl").unlink(missing_ok=True)

    pol = gd.Policy([G.SHAPE], rows=cfg["rows"], grad_bs=1)
    frames = frames_of(cfg)
    pol.encode(
        [G.framed(sc[i][f], k) for _, k, _ in gl for i in tr + rd_sc for f in frames]
    )
    rdr = gd.Reader("cuda", attn=cfg.get("reader_attn", "flash_attention_2"))
    if cfg.get("reader_compile"):
        rdr.compile()
    hy = gd.Hayai("cuda") if cfg.get("hayai_weight") else None
    print(
        f"{name}: {len(gl)} glyphs × {len(tr)} training scenes × {cfg['passes']} "
        f"passes, frames {frames}; read on {rd_sc}; row_scale {pol.row_scale:.2f}",
        flush=True,
    )
    per_pass = len(tr) * 4 // cfg["accum"]  # steps a row a pass
    start = pol.raw.detach().clone()
    recs = None
    if cfg.get("start_from"):
        recs = start_records(cfg, gl, rd_sc, hy is not None)
    if recs is None:
        recs = read(pol, rdr, cfg, gl, sc, rd_sc, {"start": start}, out, hy)
    tables: dict = {}
    soft = None
    if cfg.get("near_score"):
        fold = bool(cfg.get("fold_small"))
        soft = (
            cfg["near_score"],
            {k: {_norm(x) for x in near(k, fold)} for _, k, _ in gl},
        )

    def keep() -> None:  # the records so far, beside the rows
        (out / "records.json").write_text(
            json.dumps(recs, ensure_ascii=False, indent=1), "utf-8"
        )

    keep()

    def read_arm(arm: str, rows) -> None:
        recs.extend(read(pol, rdr, cfg, gl, sc, rd_sc, {arm: rows}, out, hy))
        keep()
        pair = [r for r in recs if r["arm"] in ("start", arm)]
        tables[arm] = tabulate(pair, ["start", arm], soft)
        print(
            f"  {arm}: total {tables[arm]['total'][arm]}  paired solo "
            f"{tables[arm]['paired']['solo']}",
            flush=True,
        )
        if len(frames) > 1:
            tables[arm]["by_frame"] = {
                f: tabulate([r for r in pair if r["frame"] == f], ["start", arm], soft)
                for f in frames
            }
            for f, t in tables[arm]["by_frame"].items():
                print(f"    {f}: total {t['total']}", flush=True)

    def on_save(p: int, rows) -> None:
        taken = min(p * per_pass, cfg.get("max_steps") or p * per_pass)
        read_arm(f"s{taken:02d}", rows)

    row = {k: pol.row_of(G.framed(sc[tr[0]][frames[0]], k)) for _, k, _ in gl}
    for arm, spec in cfg.get("read_refs", {}).items():
        read_arm(arm, saved_rows(spec, row, start))
    pol.raw.data.copy_(start)
    if cfg.get("start_shift"):
        shifted = shared_start(cfg["start_shift"], row, start)
        moved = (shifted - start).norm(dim=1) / pol.row_scale
        print(
            f"  start_shift {cfg['start_shift']}: {int((moved > 0).sum())} rows moved "
            f"{float(moved[moved > 0].min()):.3f} … {float(moved.max()):.3f} row norms",
            flush=True,
        )
        if cfg.get("read_shift"):
            on_save(0, shifted)
        pol.raw.data.copy_(shifted)
    res = train(pol, rdr, cfg, gl, sc, tr, out, hy, on_save)
    print(f"train: {res['info']}", flush=True)
    pol.raw.data.copy_(res["trained"])
    last = list(tables)[-1]
    arms = ["start", last]
    table = tables[last]
    for k in table["per_glyph"]:
        by = [table["per_glyph"][k]["start"]] + [
            t["per_glyph"][k][arm] for arm, t in tables.items()
        ]
        print(
            f"  {k} solo by arm (start, {', '.join(tables)}) "
            + " → ".join(str(x["solo"]) for x in by)
            + ("  hayai " + " → ".join(str(x["hy_solo"]) for x in by) if hy else "")
            + ("  score " + " → ".join(str(x["score"]) for x in by) if soft else ""),
            flush=True,
        )
    for k, v in table["per_glyph"].items():
        s, t = v["start"], v[last]
        print(
            f"  {k} solo {s['solo']:>2} → {t['solo']:>2} / {s['n']}  on_page "
            f"{s['on_page']:>2} → {t['on_page']:>2}  null {s['null']} → {t['null']}  "
            f"extra {s['extra']} → {t['extra']}  ce {s['ce']} → {t['ce']}  none "
            f"{s['none']} → {t['none']}  drift {res['info']['drift'][k]}"
            + (f"  hayai solo {s['hy_solo']} → {t['hy_solo']}" if hy else ""),
            flush=True,
        )
    for m, v in table["paired"].items():
        print(f"  paired {m:<8} {v}", flush=True)
    print(f"  total {table['total']}", flush=True)

    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir("cjk_rl", label=f"g1-{name}", root=LINE / "results")
    pairs = [arms] + [[ref, last] for ref in cfg.get("read_refs", {})]
    for x, y in pairs:
        for f in frames:
            tag = ("" if x == "start" else f"_{x}") + (
                f"_{f}" if len(frames) > 1 else ""
            )
            sheets(
                [r for r in recs if r["arm"] in (x, y) and r.get("frame", f) == f],
                [x, y],
                rd_sc,
                run_dir / f"sheets{tag}",
                cfg["shuffle_seed"],
            )
    (run_dir / "records.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=1), "utf-8"
    )
    steps = [
        json.loads(x) for x in (out / "log.jsonl").read_text("utf-8").splitlines() if x
    ]
    cs = [s["cos_prev"] for s in steps if s["cos_prev"] is not None]
    write_result(
        run_dir,
        script="project/cjk_rl/src/g1.py",
        args=vars(a) | {"cfg": cfg, "model": T.MODEL, "out": str(out)},
        label=name,
        metrics={
            "train": res["info"]
            | {
                "row_steps": len(steps),
                # successive mean gradients of a row
                "cos_prev_mean": round(sum(cs) / len(cs), 4) if cs else None,
                "cos_prev_n": len(cs),
            },
            "read": tables,  # the start rows against each save
        },
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
