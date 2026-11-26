#!/usr/bin/env python
"""S1 — the kana rows trained on strings: a render teaches every row in its quote.

The sentence run of ``sent_rl.md``. Policy, path and readers are G1's (``gd.Policy``,
``gd.Reader``, ``gd.Hayai``; ``g1.py`` for the glyph stage's loop). What differs:

- A group is one string on one training scene at 4 seeds,
  ``<scene, bubble frame>. Text reads as "<string>".``. Group n takes string n of the
  draw and scene n mod the training scenes; a lap of the scenes draws new seeds
  (``train_seed0 + 4 lap …``). A string is used once. With ``train_seeds`` a group has
  that many seeds, its own (``train_seed0 + train_seeds n …``).
- The page's box is picked by its sequence score to the string (``gd.pick``). The tuned
  reader's target is the string with the line breaks of the page's own read of that box
  (:func:`score`): the read is aligned to the string, and a break the read holds after
  an aligned glyph goes into the target there, so where a render wraps costs nothing.
  hayai writes no breaks and takes the string as it is. A page with no box takes the
  detector's term.
- The backward covers the last ``back_steps`` steps (1 when absent): the image's
  cotangent goes through the last step to its input and to the embedding
  (``gd.Policy.last`` with ``wrt_x``), the input's cotangent through the step before
  (``gd.Policy.vjp``), and the steps' cotangents of the embedding are summed before
  the rows. One step's graph is alive at a time.
- The render's gradient goes to every trained row in the quote; a row's part is clipped
  to ``clip`` and added to the row's sum. A row steps when ``accum`` renders that hold
  it are in, whatever strings they were, as a row steps in G1 (momentum ``beta``, a
  step of ``eta`` row norms, ``max_steps``).
- With ``loss = "window"`` a page with a box takes, in place of the string's
  cross-entropy on the box, a glyph's own on its own window: for each glyph of the
  quote that has a row, the two readers' cross-entropy of that glyph alone (its tokens,
  the EOS left out) on the square its cross-attention points at (``loc.py``: blocks
  13 – 15 at σ 0.75 and 0.5, the peak inside the picked box, ``window`` glyph sides).
  ``window_rows = "sum"`` backs the windows' sum once and every row takes its part, as
  the string's loss does; ``"each"`` backs a row's windows alone, a backward a row, and
  the row takes that one. The box, the row's score and the detector's term are as
  without it. With ``skip_hit`` a position the page's aligned read hits has no window
  and no backward; a render that leaves a row none is in the row's score and not in
  its ``accum``, so a row steps on ``accum`` renders that backed it.
- A row's score for the guards (``near_score``, ``taper``, ``[back]``, as in G1): over
  the step's renders, the positions of the row's glyph that the aligned read hits, a
  look-alike (``g1.shape``) counting ``near_score``, over the positions asked, times
  ``accum``. A small kana is its own glyph.

Trained rows: the kana of the config's ``hira`` and ``kata``, and the marks that have a
row (``ー`` with ``―``, ``。``, ``、``, ``~``, a dot run, ``♥``, ``「``, ``」``). ``!`` and
``?`` are T5's own tokens. With ``[start]`` (``run``, ``save``) the kana rows begin where
that save holds them; the marks begin at the config's ``rows``.

Strings: ``strings_file`` (``probes/sent_strings.py``; local), ``train`` in drawn order
and ``val`` by length. The run trains on the first ``groups`` of ``train``. With
``[draw]`` a train string is left out of those when its glyphs of ``glyphs`` (keys)
run longer than ``run_max`` or are more than ``share_max`` of it (:func:`thinned`); the
read's strings are not thinned.

The read (the start rows, then each save in ``save_groups``), every page read as
``g1.read`` reads it, with the boxes' reads kept as written:

- ``val``: ``[read] val_per_bin`` strings a length bin (2 – 4, 5 – 8, 9 – 12) of the
  file's ``val`` (the reader's val split, in no train label);
- ``seen``: the last ``[read] seen`` strings of ``train`` (train labels this run does not
  train on);
- the kana alone, on ``[read] kana_scenes`` and ``kana_seeds``.

Per class and length: a box that is the string with its marks (glyph keys, :func:`key`)
and without them (``common.text.norm``, the line's count so far), under the tuned reader
and hayai; the positions hit and the glyphs drawn over the glyphs asked in the closest
box. Sheets: per ``val`` string the start rows and the last save as rows ``A`` / ``B`` in
a drawn order (``sheets/key.json``).

The run's settings, question and gate are its config's header. Writes
``output/cjk_rl/g1/<config stem>/`` (``log.jsonl`` a row step, ``groups.jsonl`` a group,
``rows_g<groups>.pt`` a save and ``rows.pt`` in G1's form, ``records.json``) and
``results/<ts>-s1-<config stem>/``.

    make daemon-run ARGS="--queue --stall-timeout 900 project/cjk_rl/src/s1.py --config project/cjk_rl/configs/<name>.toml"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import tomllib
import unicodedata
from pathlib import Path

LINE = Path(__file__).resolve().parents[1]  # project/cjk_rl
sys.path.insert(0, str(LINE / "probes"))

import glyph_g0 as G  # noqa: E402
import glyph_scenes as S  # noqa: E402
import turbo_read as T  # noqa: E402

import g1  # noqa: E402
import gd  # noqa: E402
import loc  # noqa: E402

DOT_RUN = re.compile(r"[.．・…‥]{2,}|…")
BINS = ((2, 4), (5, 8), (9, 12))
MARKS = ("ー", "。", "、", "~", "…", "♥", "「", "」")  # the marks with a row
FOLD = {"♥": "♡", "❤": "♡", "〜": "~", "―": "ー", "—": "ー", "─": "ー"}


def units(s: str) -> list:
    """The glyphs of ``s`` (no whitespace in it), a dot run one of them."""
    out, i = [], 0
    for m in DOT_RUN.finditer(s):
        out += list(s[i : m.start()]) + [m[0]]
        i = m.end()
    return out + list(s[i:])


def key(u: str) -> str:
    """A glyph as it is compared: a dot run ``…``, either heart ``♡``, the wave ``~``,
    a long dash as ``ー`` (the row it is drawn from), else NFKC."""
    if DOT_RUN.fullmatch(u):
        return "…"
    k = unicodedata.normalize("NFKC", u)
    k = k if len(k) == 1 else u
    return FOLD.get(k, k)


def keys(text: str) -> list:
    return [key(u) for u in units("".join(text.split()))]


def read_units(text: str) -> tuple:
    """A read → its glyphs and the indices after which a line of it ends."""
    us, br = [], set()
    for line in text.split("\n"):
        us += units("".join(line.split()))
        br.add(len(us) - 1)
    br -= {-1, len(us) - 1}
    return us, br


def align(a: list, b: list) -> list:
    """``(i, j)`` pairs of a longest common subsequence of ``a`` and ``b``."""
    n, m = len(a), len(b)
    L = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            L[i][j] = (
                L[i + 1][j + 1] + 1 if a[i] == b[j] else max(L[i + 1][j], L[i][j + 1])
            )
    out, i, j = [], 0, 0
    while i < n and j < m:
        if a[i] == b[j]:
            out.append((i, j))
            i, j = i + 1, j + 1
        elif L[i + 1][j] >= L[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def soft(k: str) -> str:
    """A glyph key with its look-alikes folded (``g1.shape``: a kana without its voiced
    mark, へ and ヘ one shape)."""
    return g1.shape(k) if len(k) == 1 and "ぁ" <= k <= "ヺ" else k


def score(au: list, read: str) -> tuple:
    """The string's glyphs ``au`` against a box's read → ``(hit, near, target)``: a bool
    a position on the alignment, a bool a position that only a look-alike holds, and
    the string with the read's line breaks."""
    ak = [key(u) for u in au]
    ru, br = read_units(read)
    rk = [key(u) for u in ru]
    pairs = align(ak, rk)
    hit = [False] * len(au)
    for i, _ in pairs:
        hit[i] = True
    near = [False] * len(au)
    for i, _ in align([soft(k) for k in ak], [soft(k) for k in rk]):
        near[i] = not hit[i]
    after = {i for i, j in pairs if j in br and i < len(au) - 1}
    target = "".join(u + ("\n" if i in after else "") for i, u in enumerate(au))
    return hit, near, target


def glyph_spans(au: list, target: str) -> list:
    """The character span of each glyph of ``au`` in ``target``, the string with or
    without line breaks (:func:`score`'s third)."""
    out, c = [], 0
    for u in au:
        assert target[c : c + len(u)] == u, (au, target)
        out.append((c, c + len(u)))
        c += len(u)
        c += target[c : c + 1] == "\n"
    return out


def own_ce(tok: tuple, spans: list, pos: list):
    """The part of a crop's cross-entropy that the glyphs at ``pos`` hold: the sum over
    the tokens that overlap one of their spans, over the count of all the crop's tokens
    (the scale of each token in ``ce``'s mean). ``tok``: a crop's ``(ce, spans)`` of
    ``gd.Reader.ce``. A token that covers two glyphs counts for both; a line break and
    the EOS for none. ``None`` where the tokens' spans are unknown."""
    each, off = tok
    if off is None:
        return None
    mine = [spans[p] for p in pos]
    keep = [j for j, (a, b) in enumerate(off) if any(a < y and b > x for x, y in mine)]
    return each[keep].sum() / len(each)


def thinned(strings: list, rule: dict) -> list:
    """The strings a ``[draw]`` rule keeps, in their order."""
    gl = {key(c) for c in rule["glyphs"]}
    out = []
    for s in strings:
        ks = keys(s)
        run = best = 0
        for k in ks:
            run = run + 1 if k in gl else 0
            best = max(best, run)
        share = sum(k in gl for k in ks) / len(ks)
        if best <= rule["run_max"] and share <= rule["share_max"]:
            out.append(s)
    return out


def bin_of(n: int) -> str | None:
    return next((f"{lo}-{hi}" for lo, hi in BINS if lo <= n <= hi), None)


def row_map(pol, scene: str, kana: list) -> tuple:
    """``({glyph key: index in raw}, {index: name})`` for the kana and the marks."""
    at, name = {}, {}
    (a,) = pol.rows_in(G.framed(scene, "あ"))
    for c in kana:
        (i,) = pol.rows_in(G.framed(scene, c))
        at[key(c)], name[i] = i, c
    for m in MARKS:
        (i,) = pol.rows_in(G.framed(scene, "あ" + m))[1:]
        assert i != a and i not in name, (m, i)
        at[key(m)], name[i] = i, m
    return at, name


def train(pol, rdr, cfg, strings, sc, scenes, out, hy, at, name, on_save) -> dict:
    """``on_save(groups, rows)`` runs after each group count in ``cfg["save_groups"]``,
    once the rows are on disk; its seconds are left out of the run's clock."""
    import torch

    W, H = G.SHAPE
    accum, nback = cfg["accum"], int(cfg.get("back_steps") or 1)
    w_near = float(cfg.get("near_score") or 0.0)
    taper, back = bool(cfg.get("taper")), cfg.get("back")
    cap_steps = cfg.get("max_steps")
    win = cfg.get("loss") == "window"
    each = win and cfg["window_rows"] == "each"
    assert not win or cfg["window_rows"] in ("sum", "each"), cfg["window_rows"]
    skip_hit = bool(cfg.get("skip_hit"))
    assert win or not skip_hit, "skip_hit is the window loss's"
    n_seeds = int(cfg.get("train_seeds") or 4)
    rec = loc.Recorder(pol.anima) if win else None
    dim = pol.raw.shape[1]
    init = pol.raw.detach().clone()
    acc = {i: torch.zeros(dim, device=pol.dev) for i in name}
    mom = {i: torch.zeros(dim, device=pol.dev) for i in name}
    since = {i: [] for i in name}  # the renders since the row's last step
    trail = {i: [] for i in name}  # (the row a step's renders were drawn at, score)
    best: dict = {}  # row -> (window score, row)
    prev: dict = {}
    steps = dict.fromkeys(name, 0)
    returns = dict.fromkeys(name, 0)
    log = open(out / "log.jsonl", "a")
    glog = open(out / "groups.jsonl", "a")
    t_run, phase, n_groups, n_backed = time.time(), {}, 0, 0

    def lap(k: str, t0: float) -> float:
        torch.cuda.synchronize()
        phase[k] = phase.get(k, 0.0) + time.time() - t0
        return time.time()

    def dump(path: Path, table) -> None:
        torch.save(
            {
                "ext_ids": pol.r.delta.ext_ids,
                "rows": {name[i]: i for i in name},
                "raw": table.cpu(),  # effective units, every ext row
                "init": init.cpu(),
                "row_scale": pol.row_scale,
                "steps": {name[i]: steps[i] for i in name},
                "cfg": cfg,
            },
            path,
        )

    def step(i: int, n: int) -> None:
        rs = since[i]
        gbar = acc[i] / sum(x["norm"] is not None for x in rs)
        pos = sum(x["pos"] for x in rs)
        raw = sum(x["hit"] + w_near * x["near"] for x in rs)
        sc16 = accum * raw / max(pos, 1)  # of ``accum`` renders, as G1's score
        window = verdict = None
        if back:
            trail[i] = trail[i][-back["window"] + 1 :] if back["window"] > 1 else []
            trail[i].append((pol.raw[i].detach().clone(), sc16))
            window, verdict = g1.window_read(
                [x for _, x in trail[i]], best[i][0] if i in best else None, back, accum
            )
        cos = (
            float(torch.nn.functional.cosine_similarity(gbar, prev[i], dim=0))
            if i in prev
            else None
        )
        if verdict == "back":
            with torch.no_grad():
                pol.raw[i] = best[i][1]
            mom[i].zero_()
            del best[i]
            trail[i] = []
            prev.pop(i, None)
            returns[i] += 1
            length = None
        else:
            if verdict == "best":  # of the window's rows, the better read
                best[i] = (window, max(reversed(trail[i]), key=lambda x: x[1])[0])
            mom[i] = cfg["beta"] * mom[i] + gbar
            d = mom[i] / mom[i].norm().clamp_min(1e-30)
            length = min(1.0, 2.0 * (1.0 - sc16 / accum)) if taper else 1.0
            with torch.no_grad():
                pol.raw[i] -= length * cfg["eta"] * pol.row_scale * d
            prev[i] = gbar.clone()
        steps[i] += 1
        log.write(
            json.dumps(
                {
                    "group": n,
                    "key": name[i],
                    "step": steps[i],
                    "renders": len(rs),
                    "backed": sum(x["norm"] is not None for x in rs),
                    "positions": pos,
                    "hit": sum(x["hit"] for x in rs),
                    "near": sum(x["near"] for x in rs),
                    "score": round(sc16, 3),
                    "length": length,  # of eta; None: the row returned
                    "window": window,
                    "back": verdict == "back",
                    "gbar": float(gbar.norm()),
                    "cos_prev": cos,
                    "drift": float((pol.raw[i] - init[i]).norm()) / pol.row_scale,
                    "none": sum(x["case"] == "none" for x in rs),
                    "clipped": sum((x["norm"] or 0.0) > cfg["clip"] for x in rs),
                    "min": round((time.time() - t_run) / 60, 2),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        log.flush()
        acc[i].zero_()
        since[i] = []

    for n, s in enumerate(strings):
        si = scenes[n % len(scenes)]
        first = n_seeds * n if "train_seeds" in cfg else 4 * (n // len(scenes))
        seeds = [cfg["train_seed0"] + first + k for k in range(n_seeds)]
        caption = G.framed(sc[si][cfg["frame"]], s)
        au = units(s)
        ak = [key(u) for u in au]
        here = sorted(set(pol.rows_in(caption)))
        assert set(here) == {at[k] for k in ak if k in at}, (s, here)
        live = [i for i in here if not (cap_steps and steps[i] >= cap_steps)]
        if not live:
            continue
        t = time.time()
        e_graph = pol.embed(caption, grad=True)
        e = e_graph.detach()
        if win:  # the quote tokens' maps at the last two steps
            cols, groups, upos = loc.token_cols(pol, caption, ak, at)
            cols = torch.tensor(cols, device=pol.dev)
            rec.cols = cols
        xs = pol.inputs(pol.noise(seeds, G.SHAPE), e, nback)
        m_prev = rec.sets()[:, 0] if win else None  # (4, tokens, Nq), σ 0.75
        t = lap("head", t)
        pages = []
        for b in range(len(seeds)):
            xi = None
            if win:
                rec.cols = cols
            if nback > 1:
                px, ei, xi = pol.last(xs[-1][b : b + 1], e, wrt_x=True)
            else:
                px, ei = pol.last(xs[-1][b : b + 1], e)
            if win:
                m = (m_prev[b] + rec.sets()[0, 0]) / 2
                m = torch.stack([m[g].mean(0) for g in groups])  # a glyph a map
                rec.cols = None
            t = lap("last_fwd", t)
            img = px.detach()
            bgrs, reads = rdr.read(img)
            t = lap("read", t)
            case, r = gd.pick(s, reads[0], cfg["conf_min"])
            leaf = img.clone().requires_grad_(True)
            p01 = (gd.ste_clamp(leaf, -1.0, 1.0) + 1) / 2
            aux = None
            hit, near, got = [False] * len(au), [False] * len(au), ""
            if r is None:
                top = rdr.det_scores(p01).max(1).values
                loss = top.clamp_min(1e-6).log().neg().sum()
                cots = [(live, torch.autograd.grad(loss, leaf)[0])]
            elif win:
                got = r["sfx"] or ""
                hit, near, _ = score(au, got)
                rb = [int(v) for v in r["box"]]
                side = loc.glyph_side(rb, len(au), cfg["window"])
                cs = loc.centres(m, rb, H // loc.TOK_PX, W // loc.TOK_PX)["raw"]
                g_by, loss, aux, n_win = {}, 0.0, 0.0, 0
                for c, p in zip(cs, upos):
                    i = at[ak[p]]
                    if i not in live or (skip_hit and hit[p]):
                        continue
                    # a graph a window, freed at once (a string's do not fit)
                    leaf_w = img.clone().requires_grad_(True)
                    w01 = (gd.ste_clamp(leaf_w, -1.0, 1.0) + 1) / 2
                    wbox = loc.window(c, side, W, H)
                    u, tk, htk = au[p], [], []
                    rdr.ce([w01[0]], bgrs, [wbox], [u], tokens=tk)
                    seen = own_ce(tk[0], [(0, len(u))], [0])
                    if seen is None:  # the tokens' spans unknown: the glyph and EOS
                        seen = tk[0][0].mean()
                    loss += float(seen.detach())
                    if hy is not None:
                        hy.ce([w01[0]], [wbox], [u], tokens=htk)
                        more = own_ce(htk[0], [(0, len(u))], [0])
                        aux += float(more.detach())
                        seen = seen + cfg["hayai_weight"] * more
                    g = torch.autograd.grad(seen, leaf_w)[0]
                    g_by[i] = g_by[i] + g if i in g_by else g
                    n_win += 1
                    del seen, tk, htk, w01, leaf_w
                loss = loss / n_win if n_win else None  # None: every position hit
                aux = aux / n_win if n_win and hy is not None else None
                if each:
                    cots = [([i], g) for i, g in g_by.items()]
                else:
                    cots = [(list(g_by), sum(g_by.values()))] if g_by else []
                del g_by
            else:
                got = r["sfx"] or ""
                hit, near, target = score(au, got)
                box = gd.crop_box(r["box"], W, H)
                loss = rdr.ce([p01[0]], bgrs, [box], [target]).sum()
                total = loss
                if hy is not None:
                    aux = hy.ce([p01[0]], [box], [s]).sum()
                    total = loss + cfg["hayai_weight"] * aux
                cots = [(live, torch.autograd.grad(total, leaf)[0])]
            t = lap("loss", t)
            row_g = {}  # row -> its gradient of this render
            for n_c, (mine, g_px) in enumerate(cots):
                if n_c:  # the last step again, with a graph of its own
                    g_e, g_x = pol.vjp(pol.steps - 1, xs[-1][b : b + 1], e, g_px)
                elif nback > 1:
                    g_e, g_x = torch.autograd.grad((px * g_px).sum(), (ei, xi))
                else:
                    g_e = torch.autograd.grad((px * g_px).sum(), ei)[0]
                if not n_c:
                    del px, ei, xi
                for k in range(2, nback + 1):  # the steps before the last, backwards
                    ge, g_x = pol.vjp(pol.steps - k, xs[-k][b : b + 1], e, g_x)
                    g_e = g_e + ge
                g_rows = pol.row_grads(e_graph, g_e)[0]
                for i in mine:
                    row_g[i] = g_rows[i].clone()
                del g_rows
            if not cots:
                del px, ei, xi
            del cots
            t = lap("bwd", t)
            norms = []
            for i in live:
                gn = None  # None: the render backed nothing of this row
                if i in row_g:
                    gn = float(row_g[i].norm())
                    norms.append(gn)
                    acc[i] += row_g[i] * min(1.0, cfg["clip"] / max(gn, 1e-30))
                pos = [p for p, k in enumerate(ak) if at.get(k) == i]
                since[i].append(
                    {
                        "case": case,
                        "pos": len(pos),
                        "hit": sum(hit[p] for p in pos),
                        "near": sum(near[p] for p in pos),
                        "norm": gn,
                    }
                )
            del row_g
            n_backed += len(norms)
            pages.append(
                {
                    "case": case,
                    # the tuned reader's term (a window's mean)
                    "loss": None if loss is None else float(loss),
                    "hy": None if aux is None else float(aux),
                    "hit": sum(hit),
                    "exact": keys(got) == ak,
                    "norm": max(norms, default=0.0),
                    "backed": len(norms),
                }
            )
        del e_graph
        n_groups += 1
        for i in live:
            if sum(x["norm"] is not None for x in since[i]) >= accum:
                step(i, n_groups)
        crop = [x for x in pages if x["case"] != "none" and x["loss"] is not None]
        glog.write(
            json.dumps(
                {
                    "group": n_groups,
                    "string": s,
                    "glyphs": len(au),
                    "rows": len(live),
                    "backed": sum(x["backed"] for x in pages),  # row-renders
                    "scene": si,
                    "cases": {
                        c: sum(x["case"] == c for x in pages)
                        for c in ("box", "low", "none")
                    },
                    "exact": sum(x["exact"] for x in pages),
                    "hit": round(
                        sum(x["hit"] for x in pages) / (len(pages) * len(au)), 3
                    ),
                    "ce": sum(x["loss"] for x in crop) / len(crop) if crop else None,
                    "ce_hy": (
                        sum(x["hy"] for x in crop) / len(crop)
                        if crop and hy is not None
                        else None
                    ),
                    "norm_max": round(max(x["norm"] for x in pages), 4),
                    "min": round((time.time() - t_run) / 60, 2),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        glog.flush()
        if n_groups % 16 == 0:
            print(
                f"  group {n_groups} / {len(strings)} row steps {sum(steps.values())} "
                f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB "
                f"({(time.time() - t_run) / 60:.1f} min)",
                flush=True,
            )
        if n_groups in cfg["save_groups"]:
            t_save = time.time()
            table = pol.raw.detach().clone()
            dump(out / f"rows_g{n_groups:04d}.pt", table)
            if on_save is not None:
                on_save(n_groups, table)
            t_run += time.time() - t_save
    log.close()
    glog.close()
    trained = pol.raw.detach().clone()
    dump(out / "rows.pt", trained)
    by = sorted(steps.values())
    info = {
        "groups": n_groups,
        "minutes": round((time.time() - t_run) / 60, 2),
        "s_per_group": {k: round(v / max(n_groups, 1), 3) for k, v in phase.items()},
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "row_renders": n_backed,
        "row_steps": sum(by),
        "rows_stepped": sum(v > 0 for v in by),
        "steps_max": by[-1],
        "steps": {name[i]: steps[i] for i in name},
        "returns": sum(returns.values()),
        "rows_returned": sum(v > 0 for v in returns.values()),
        "boundary": rdr.boundary,
        "drift": {
            name[i]: round(float((trained[i] - init[i]).norm()) / pol.row_scale, 4)
            for i in name
            if steps[i]
        },
    }
    return {"init": init, "trained": trained, "info": info}


def string_table(recs: list) -> dict:
    """Per class and length bin of the string records of one arm."""
    import word_read as W

    out = {}
    for cls in dict.fromkeys(r["cls"] for r in recs):
        for b in [f"{lo}-{hi}" for lo, hi in BINS] + ["all"]:
            xs = [
                r
                for r in recs
                if r["cls"] == cls and b in ("all", bin_of(len(units(r["key"]))))
            ]
            if not xs:
                continue
            asked = hits = drawn = 0
            for r in xs:
                ak = keys(r["key"])
                bs = [keys(x) for x in r["raw"]]
                if bs:
                    bk = max(bs, key=lambda x: len(align(ak, x)) / max(len(ak), len(x)))
                    hits += len(align(ak, bk))
                    drawn += len(bk)
                asked += len(ak)
            plain = [W.page(r["key"], r["boxes"], False) for r in xs]
            out[f"{cls}/{b}"] = {
                "pages": len(xs),
                "strings": len({r["key"] for r in xs}),
                "solo_marks": sum(keys(r["key"]) in map(keys, r["raw"]) for r in xs),
                "solo": sum(r["solo"] for r in xs),
                "hy_solo_marks": sum(
                    keys(r["key"]) in map(keys, r.get("hy_raw", [])) for r in xs
                ),
                "hy_solo": sum(r.get("hy_solo", 0) for r in xs),
                "pos_hit": round(hits / asked, 3),
                "pos_hit_plain": round(
                    sum(sum(p["hit"]) for p in plain)
                    / max(sum(len(p["hit"]) for p in plain), 1),
                    3,
                ),
                "drawn_over_asked": round(drawn / asked, 2),
                "none": sum(r["case"] == "none" for r in xs),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="its stem names the run")
    ap.add_argument("--smoke", action="store_true", help="24 renders and a small read")
    a = ap.parse_args()

    cfg = tomllib.loads(Path(a.config).read_text("utf-8"))
    name = ("smoke_" if a.smoke else "") + Path(a.config).stem
    sc = S.scenes()
    tr, rd_sc = list(cfg["train_scenes"]), list(cfg["read_scenes"])
    assert not {sc[i]["artist"] for i in tr} & {sc[i]["artist"] for i in rd_sc}, (
        "a read scene's folder is trained on"
    )
    rd = dict(cfg["read"])
    if a.smoke:
        tr, rd_sc = tr[:2], rd_sc[:1]
        g = 24 // int(cfg.get("train_seeds") or 4)  # 24 renders
        cfg = cfg | {"groups": g, "accum": 4, "save_groups": [g]}
        rd |= {"val_per_bin": 1, "seen": 1, "kana_scenes": rd["kana_scenes"][:1]}
    cfg = {"save_groups": [cfg["groups"]]} | cfg
    assert cfg["groups"] in cfg["save_groups"], "the last group is a save"
    frame = cfg["frame"]
    assert isinstance(frame, str), "one frame"
    pool = json.loads((T.REPO / cfg["strings_file"]).read_text("utf-8"))
    seen = pool["train"][len(pool["train"]) - rd["seen"] :]
    drawn = pool["train"][: len(pool["train"]) - rd["seen"]]
    if cfg.get("draw"):
        kept = thinned(drawn, cfg["draw"])
        print(f"[draw] leaves out {len(drawn) - len(kept)} of {len(drawn)} strings")
        drawn = kept
    strings = drawn[: cfg["groups"]]
    assert len(strings) == cfg["groups"], "the draw is shorter than the run"
    assert not set(seen) & set(strings), "a seen string is trained on"
    val = [s for b in pool["val"].values() for s in b[: rd["val_per_bin"]]]
    kana_gl = g1.glyphs(cfg | {"fold_small": False}, a.smoke)
    kana = [k for _, k, _ in g1.glyphs(cfg | {"fold_small": False}, False)]

    def gl_of(cls: str, xs: list) -> list:
        return [(cls, s, xs[(n + 1) % len(xs)]) for n, s in enumerate(xs)]

    sets = [  # (records, scenes, seeds)
        (gl_of("val", val), rd_sc, rd["seeds"]),
        (gl_of("seen", seen), rd_sc, rd["seeds"]),
        (kana_gl, [i for i in rd["kana_scenes"] if i in rd_sc], rd["kana_seeds"]),
    ]
    out = g1.ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    for f in ("log.jsonl", "groups.jsonl"):
        (out / f).unlink(missing_ok=True)

    pol = gd.Policy([G.SHAPE], rows=cfg["rows"], grad_bs=1)
    pol.encode(
        [G.framed(sc[tr[n % len(tr)]][frame], s) for n, s in enumerate(strings)]
        + [
            G.framed(sc[i][frame], k)
            for gl, ss, _ in sets
            for _, k, _ in gl
            for i in ss
        ]
    )
    rdr = gd.Reader("cuda", attn=cfg.get("reader_attn", "flash_attention_2"))
    if cfg.get("reader_compile"):
        rdr.compile()
    hy = gd.Hayai("cuda") if cfg.get("hayai_weight") else None
    at, row_name = row_map(pol, sc[tr[0]][frame], kana)
    start = pol.raw.detach().clone()
    if cfg.get("start"):
        start = g1.saved_rows(cfg["start"], {k: at[key(k)] for k in kana}, start)
    print(
        f"{name}: {len(strings)} groups on {len(tr)} training scenes, {len(row_name)} "
        f"rows, backward over {cfg.get('back_steps') or 1} steps; read {len(val)} val "
        f"and {len(seen)} seen strings on {rd_sc} × {rd['seeds']}, {len(kana_gl)} kana "
        f"on {sets[2][1]} × {rd['kana_seeds']}; row_scale {pol.row_scale:.2f}",
        flush=True,
    )
    rcfg = cfg | {"raw_reads": True, "fold_small": False}
    recs: list = []
    tables: dict = {}

    def read_arm(arm: str, rows) -> None:
        for gl, ss, seeds in sets:
            recs.extend(
                g1.read(
                    pol,
                    rdr,
                    rcfg | {"read_seeds": seeds},
                    gl,
                    sc,
                    ss,
                    {arm: rows},
                    out,
                    hy,
                )
            )
        (out / "records.json").write_text(
            json.dumps(recs, ensure_ascii=False, indent=1), "utf-8"
        )
        mine = [r for r in recs if r["arm"] == arm]
        kn = [r for r in mine if r["cls"] in g1.CLASSES]
        tables[arm] = {
            "strings": string_table([r for r in mine if r["cls"] in ("val", "seen")]),
            "kana": {
                "pages": len(kn),
                "solo": sum(r["solo"] for r in kn),
                "hy_solo": sum(r["hy_solo"] for r in kn) if hy else None,
                "none": sum(r["case"] == "none" for r in kn),
            },
        }
        for k, v in tables[arm]["strings"].items():
            print(f"  {arm} {k}: {v}", flush=True)
        print(f"  {arm} kana alone: {tables[arm]['kana']}", flush=True)

    read_arm("start", start)
    pol.raw.data.copy_(start)
    res = train(
        pol,
        rdr,
        cfg,
        strings,
        sc,
        tr,
        out,
        hy,
        at,
        row_name,
        lambda n, rows: read_arm(f"g{n:04d}", rows),
    )
    print(f"train: {res['info']}", flush=True)
    last = list(tables)[-1]

    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir("cjk_rl", label=f"s1-{name}", root=LINE / "results")
    g1.sheets(
        [r for r in recs if r["cls"] == "val" and r["arm"] in ("start", last)],
        ["start", last],
        rd_sc,
        run_dir / "sheets",
        cfg["shuffle_seed"],
    )
    (run_dir / "records.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=1), "utf-8"
    )
    kn = {}  # a kana: its solo by arm
    for r in recs:
        if r["cls"] in g1.CLASSES:
            kn.setdefault(r["key"], dict.fromkeys(tables, 0))[r["arm"]] += r["solo"]
    write_result(
        run_dir,
        script="project/cjk_rl/src/s1.py",
        args=vars(a) | {"cfg": cfg, "model": T.MODEL, "out": str(out)},
        label=name,
        metrics={"train": res["info"], "read": tables, "kana_solo": kn},
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
