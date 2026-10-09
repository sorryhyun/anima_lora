#!/usr/bin/env python
"""probe_split — kana_up's +0.1 split by where it acts and by what it shares (2026-10-04)

The question (user, 10-04, after ``_archive/reports/kana_up_2026_10_04.md``): the
upper edges + 0.1 bought text and cost the scene; dup and the "fill the blank
/ white canvas" look stay. Which side of σ carries each, and is it the rows'
shared direction or their per-row part? No training: ``kana_up`` and
``kana_mix`` differ only in the 166 kana rows (same seed rows, same data
recipe but the upper edges), so their rows are swapped at render.

- ``render`` (GPU): on the plain read's grid (``run.py read``: 13 words +
  14 singles × p00–p03 at seed 0 = 108 renders an arm, ``PLAIN_CLAUSE``), arms
  as (rows above ``--switch`` / rows below), the conditional switched as in
  ``cjk_anima_scale/experiments/sigma_split`` (``context_alt`` +
  ``tag_drop_sigma``; the negative pass untouched):

  - ``up_mix``  kana_up above, kana_mix below
  - ``mix_up``  kana_mix above, kana_up below
  - ``up_mean``    every one of kana_up's 166 rows = their mean, every σ:
    the shared "kana" direction alone, no identity left
  - ``up_nomean``  kana_up less that mean, every σ
  - ``up_rkstick`` kana_up less its mean plus retrain_kana's mean over the
    same 166 rows: kana_up's spikes on retrain_kana's stick, every σ
  - ``rk_self``    retrain_kana's rows through this path (the gs arms' control)
  - ``gs_self``    retrain_kana's rows with its 81 hiragana rows replaced by
    ``grid_small r0``'s (cold, flat grids only, σ 0.3–0.7), every σ
  - ``gs_rkstick`` ``gs_self`` with those 81 rows less their mean plus
    retrain_kana's mean over them: the low-band grid ball on retrain_kana's
    hiragana stick, every σ
  - ``rk_gsstick`` retrain_kana's 81 hiragana spikes on grid_small r0's stick
  - ``h0_self`` / ``h0_rkstick`` as ``gs_*`` with ``grid_lone recap_h0``'s rows
  - ``ball_rk``    the ``ball_rk`` run as trained: grid_small r0's data, its 81
    hiragana rows' ball trained cold on retrain_kana's stick (held)
  - ``ball_rk_bubble`` the same on the reseed table's four bubble tiers
  - ``ball_rkb_long`` ``ball_rk_bubble`` with its 81 hiragana rows less their
    mean scaled to retrain_kana's ball length (mean |row − mean|), the stick
    kept: the short ball (183 vs 209) at retrain_kana's length

  An arm whose ``manifest.json`` exists is not rendered again.

  (The mean of up − mix is 1.6 % of the difference's energy: the two cold runs
  share their mean direction at cos 0.985, so the split is of kana_up's own
  rows, the shared_dir split of ``cjk_anima_scale/hypothesis.md``.)

  The floors are the two runs' own renders (``…/<run>/native_r4_plain/``).
  A plumbing check renders one kana_up key through the split path first.
- ``read`` (GPU): readers + the EN-reference ruler (``sigma_split.read_arm``)
  and ``flat_white`` (``sigma_split.placement``: 16² patches, std < 6,
  mean > 225) on every new arm, ``flat_white`` added to the cached reads of
  every reseed run and the scale arms of record; tallies, pairs against both
  floors, sheets → ``results/<stamp>-<label>/``.
- ``traj`` (GPU): kana_up and kana_mix rows at every σ on the 13 words ×
  p00–p03 at seed 0, x̂0 decoded at ``TRAJ_SIGMAS`` and read: when the
  read's length passes the word's, and when a doubled glyph appears.

    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/probe_split.py --label s075"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import re
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the reseed project
from reseed import bootstrap  # noqa: E402

bootstrap()
os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"  # the runs trained routed, read routed

from reseed import HOME, OUT  # noqa: E402

NAME = "probe_split"
UP, MIX = "kana_up", "kana_mix"
PROMPTS, SEEDS = 4, 1  # the plain read's grid at seed 0 only (user, 10-04: not all 216)
KEYS = 27  # 13 words + 14 singles
CLAUSE = "plain"
# arm → (rows above the switch, rows below); a cross arm's name carries the switch
CROSS = {"up_mix": ("up", "mix"), "mix_up": ("mix", "up")}
# kana_up's rows split into their mean over the 166 rows (the shared "kana"
# direction, every row the same vector) and the per-row rest
SHARED = {
    "up_mean": ("up_mean", "up_mean"),
    "up_nomean": ("up_nomean", "up_nomean"),
    # kana_up's spikes on retrain_kana's stick (user, 10-04)
    "up_rkstick": ("up_rkstick", "up_rkstick"),
    # grid_small r0's hiragana rows on retrain_kana's: as trained, and their
    # spikes on retrain_kana's hiragana stick (user, 10-04)
    "rk_self": ("rk", "rk"),  # retrain_kana through this path: the gs arms' control
    "gs_self": ("gs_self", "gs_self"),
    "gs_rkstick": ("gs_rkstick", "gs_rkstick"),
    # retrain_kana's spikes on grid_small r0's stick; grid_lone recap_h0's
    # rows (σ 0 – half point, the shortest stick) as trained and on rk's stick
    # (user, 10-04)
    "rk_gsstick": ("rk_gsstick", "rk_gsstick"),
    "h0_self": ("h0_self", "h0_self"),
    "h0_rkstick": ("h0_rkstick", "h0_rkstick"),
    # grid_small r0's data, the ball trained on retrain_kana's stick (user, 10-04)
    "ball_rk": ("ball_rk", "ball_rk"),
    "ball_rk_bubble": ("ball_rk_bubble", "ball_rk_bubble"),
    # the bubble ball at retrain_kana's ball length, no training (user, 10-04)
    "ball_rkb_long": ("ball_rkb_long", "ball_rkb_long"),
}
RK = "retrain_kana"  # ``cjk_anima_scale``'s run: the stick swapped in
GS = "experiments/grid_small_cold_hira"  # ``cjk_anima_scale``'s grid_small r0
H0 = "experiments/grid_lone_cold_hira_recap_h0"  # its grid_lone recap_h0
# every arm with plain renders cached: flat_white only (no new render)
CACHED = {
    "kana_big": OUT / "kana_big",
    "kana_mix": OUT / MIX,
    "kana_up": OUT / UP,
}
TRAJ_SIGMAS = (0.95, 0.9, 0.85, 0.8, 0.7, 0.0)  # 0 = the final image
TRAJ_SEED = 0
SHEET_SEED = 0


def scale_arms() -> dict:
    from cjk_scale.paths import OUT as SCALE_OUT

    sys.path.insert(0, str(HOME))
    from run import READ_AGAINST

    return {Path(d).name: SCALE_OUT / d for d in READ_AGAINST}


def reads_of(arm_dir: Path) -> Path:
    return arm_dir / "native_r4_plain" / "native_reads.json"


def items() -> list[dict]:
    """The plain grid's 108 keys at seed 0, from kana_up's reads (caption, ext_rows)."""
    recs = json.loads(reads_of(OUT / UP).read_text("utf-8"))
    out = [
        {
            k: m[k]
            for k in ("pi", "prompt", "text", "clause", "caption", "seed", "ext_rows")
        }
        | {"floor_file": m["file"]}
        for m in recs
        if m["clause"] == CLAUSE and m["pi"] < PROMPTS and m["seed"] < SEEDS
    ]
    assert len(out) == KEYS * PROMPTS * SEEDS, f"{len(out)} keys"
    return out


def row_sets() -> tuple[dict, dict]:
    """``{up, mix, up_mean, up_nomean, up_rkstick}`` raw tables (row-norm units; a cold
    row starts at 0 = the pack row) + the energy shares."""
    import torch.nn.functional as F

    import torch

    from common.models import load_trained

    up, mix = (load_trained(OUT / r)["delta"] for r in (UP, MIX))
    assert up["ext_ids"] == mix["ext_ids"] and up["row_scale"] == mix["row_scale"]
    sets = {"up": up["raw"].float(), "mix": mix["raw"].float()}
    moved = (sets["up"] - sets["mix"]).norm(dim=1) > 0  # the 166 trained rows
    assert int(moved.sum()) == 166, f"{int(moved.sum())} rows moved, not 166"
    u, m = sets["up"][moved], sets["mix"][moved]
    mu = u.mean(0)

    def share(x):
        return float(len(x) * (x.mean(0) ** 2).sum() / (x**2).sum())

    info = {
        "rows": int(moved.sum()),
        "up_mean_energy": share(u),
        "mix_mean_energy": share(m),
        "cos_mean_up_mix": float(F.cosine_similarity(mu, m.mean(0), dim=0)),
        "mean_norm_up_mix": [float(mu.norm()), float(m.mean(0).norm())],
        "row_cos_up_mix": float(F.cosine_similarity(u, m, dim=1).mean()),
        "diff_mean_energy": share(u - m),
    }
    # retrain_kana's stick over the same 166 ext ids, in kana_up's row units
    from cjk_scale.paths import OUT as SCALE_OUT

    rk = load_trained(SCALE_OUT / RK)["delta"]
    pos = {int(e): i for i, e in enumerate(rk["ext_ids"])}
    ids = [int(up["ext_ids"][i]) for i in moved.nonzero().flatten().tolist()]
    rk_mu = rk["raw"][[pos[e] for e in ids]].float().mean(0) * (
        float(rk["row_scale"]) / float(up["row_scale"])
    )
    info["rk_stick_norm"] = float(rk_mu.norm())
    info["cos_stick_up_rk"] = float(F.cosine_similarity(mu, rk_mu, dim=0))
    for name, rows in (
        ("up_mean", mu.expand_as(u)),
        ("up_nomean", u - mu),
        ("up_rkstick", u - mu + rk_mu),
    ):
        t = sets["up"].clone()
        t[moved] = rows
        sets[name] = t
    # retrain_kana's whole table in kana_up's row units, and grid_small r0's
    # hiragana rows (those that differ from its context) on it
    assert [int(e) for e in rk["ext_ids"]] == [int(e) for e in up["ext_ids"]]
    rk_t = rk["raw"].float() * (float(rk["row_scale"]) / float(up["row_scale"]))
    sets["rk"] = rk_t
    upos = {int(e): i for i, e in enumerate(up["ext_ids"])}

    def cold_rows(run: str, ids: list | None = None):
        """``run``'s rows that differ from its context (``ids``: these only), in
        kana_up's row units."""
        sd = load_trained(SCALE_OUT / run)
        d = sd["delta"]
        ctx = torch.load(sd["args"]["context"], map_location="cpu", weights_only=False)
        ctx = ctx["delta"]
        cpos = {int(e): i for i, e in enumerate(ctx["ext_ids"])}
        eff = d["raw"].float() * float(d["row_scale"])
        c_eff = ctx["raw"].float() * float(ctx["row_scale"])
        pos = {int(e): i for i, e in enumerate(d["ext_ids"])}
        if ids is None:
            ids = [
                e
                for e, i in pos.items()
                if e not in cpos
                or not torch.allclose(eff[i], c_eff[cpos[e]], atol=1e-2)
            ]
        assert all(e in pos and e in upos for e in ids)
        return ids, eff[[pos[e] for e in ids]] / float(up["row_scale"])

    # grid_small r0's 81 hiragana rows: the hiragana every cold band arm trained
    hira, g = cold_rows(GS)
    assert len(hira) == 81, f"{len(hira)} gs rows"
    _, h0 = cold_rows(H0, hira)  # recap_h0 also trained ー: left at retrain_kana's
    rows_u = [upos[e] for e in hira]
    rkh = rk_t[rows_u]
    mu = {"gs": g.mean(0), "rk": rkh.mean(0), "h0": h0.mean(0)}
    info["hira_rows"] = len(hira)
    info["hira_stick_norm"] = {k: float(v.norm()) for k, v in mu.items()}
    info["hira_stick_cos"] = {
        f"{a}·{b}": float(F.cosine_similarity(mu[a], mu[b], dim=0))
        for a, b in (("gs", "rk"), ("h0", "rk"), ("gs", "h0"))
    }
    for name, rows in (
        ("gs_self", g),
        ("gs_rkstick", g - mu["gs"] + mu["rk"]),
        ("rk_gsstick", rkh - mu["rk"] + mu["gs"]),
        ("h0_self", h0),
        ("h0_rkstick", h0 - mu["h0"] + mu["rk"]),
    ):
        t = rk_t.clone()
        t[rows_u] = rows
        sets[name] = t
    # a ball run's rows: retrain_kana's merged table with the ball run's rows on it
    for name in ("ball_rk", "ball_rk_bubble"):
        f = OUT / name / "trained.pt"
        if not f.exists():
            continue
        b = load_trained(OUT / name)["delta"]
        assert [int(e) for e in b["ext_ids"]] == [int(e) for e in up["ext_ids"]]
        sets[name] = b["raw"].float() * (float(b["row_scale"]) / float(up["row_scale"]))
        bh = sets[name][rows_u]
        info[f"{name}_stick_cos_rk"] = float(
            F.cosine_similarity(bh.mean(0), mu["rk"], dim=0)
        )
        info[f"{name}_ball_cos_gs"] = float(
            F.cosine_similarity(bh - bh.mean(0), g - mu["gs"], dim=1).mean()
        )
    # the bubble ball scaled to retrain_kana's ball length, its stick kept
    if "ball_rk_bubble" in sets:
        bh = sets["ball_rk_bubble"][rows_u]
        bm = bh.mean(0)
        k = float((rkh - mu["rk"]).norm(dim=1).mean() / (bh - bm).norm(dim=1).mean())
        t = sets["ball_rk_bubble"].clone()
        t[rows_u] = bm + (bh - bm) * k
        sets["ball_rkb_long"] = t
        info["ball_rkb_long_scale"] = k
    print(f"row sets: {info}", flush=True)
    return sets, info


def make_splitter(sets: dict):
    from cjk_scale.paths import load_experiment

    SS = load_experiment("sigma_split")

    class RowSplitter(SS.Splitter):
        """``sigma_split.Splitter`` with a side = one of ``sets``' raw tables."""

        def __init__(self):
            super().__init__(OUT / MIX)
            self.sets = sets
            self.caches = {k: {} for k in sets}

        def encode(self, caption: str, side: str):
            from library.inference.text import prepare_text_inputs

            self.delta.scale = 1.0
            self.delta.raw.data.copy_(self.sets[side].to(self.delta.raw.device))
            self.shared["conds_cache"] = self.caches[side]
            return prepare_text_inputs(
                self._args(caption, 0), self.device, self.anima, self.shared
            )

    return SS, RowSplitter()


def arm_dir(arm: str) -> Path:
    return OUT / NAME / arm


def arms(switch: float) -> dict:
    return {f"{k}_s{switch:g}": v for k, v in CROSS.items()} | SHARED


def render(switch: float) -> float:
    import numpy as np
    from PIL import Image

    sets, info = row_sets()
    (OUT / NAME).mkdir(parents=True, exist_ok=True)
    (OUT / NAME / "row_sets.json").write_text(json.dumps(info, indent=1))
    SS, sp = make_splitter(sets)
    its = items()
    # plumbing: kana_up on both sides against its own cached render
    it = its[0]
    fn = OUT / NAME / "check.png"
    fn.unlink(missing_ok=True)
    sp.render(fn, it, "up", "up", switch)
    a = np.asarray(Image.open(fn), np.float32)
    b = np.asarray(Image.open(it["floor_file"]), np.float32)
    check = float(np.abs(a - b).mean())
    print(f"check {Path(it['floor_file']).name}: mean |Δpx| {check:.3f}", flush=True)
    assert check < 4.0, "the split path does not reproduce kana_up's render"
    # retrain_kana's table through the split path against its own render
    rk_reads = json.loads(reads_of(scale_arms()[RK]).read_text("utf-8"))
    rk_floor = next(
        m["file"]
        for m in rk_reads
        if (m["pi"], m["text"], m["clause"], m["seed"])
        == (it["pi"], it["text"], it["clause"], it["seed"])
    )
    fn = OUT / NAME / "check_rk.png"
    fn.unlink(missing_ok=True)
    sp.render(fn, it, "rk", "rk", switch)
    a = np.asarray(Image.open(fn), np.float32)
    b = np.asarray(Image.open(rk_floor), np.float32)
    check_rk = float(np.abs(a - b).mean())
    print(f"check rk {Path(rk_floor).name}: mean |Δpx| {check_rk:.3f}", flush=True)
    # 5.4 on p00 こんにちは (10-04): the same picture, outline and tint apart —
    # the gs arms pair against ``rk_self``, rendered here
    assert check_rk < 12.0, "the split path does not reproduce retrain_kana's render"
    t0 = time.time()
    for arm, (above, below) in arms(switch).items():
        if (arm_dir(arm) / "manifest.json").exists():
            continue
        manifest = []
        for n, it in enumerate(its):
            f = (
                arm_dir(arm)
                / "img"
                / f"{arm}_p{it['pi']:02d}_{it['text']}_{it['clause']}_s{it['seed']}.png"
            )
            sp.render(f, it, above, below, switch)
            manifest.append(
                {k: v for k, v in it.items() if k != "floor_file"}
                | {"file": str(f), "cond": arm}
            )
            if n % 24 == 23:
                print(
                    f"  {arm}: {n + 1} / {len(its)} ({(time.time() - t0) / 60:.1f} min)",
                    flush=True,
                )
        (arm_dir(arm) / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    sp.free()
    return check


def read(switch: float, check: float | None, label: str) -> None:
    import torch

    from bench._common import make_run_dir, write_result
    from cjk_scale import reads as R
    from cjk_scale.paths import load_experiment

    SS = load_experiment("sigma_split")
    device = torch.device("cuda")
    new = arms(switch)
    for arm in new:
        out = arm_dir(arm)
        if (out / "native_reads.json").exists():
            continue
        manifest = json.loads((out / "manifest.json").read_text("utf-8"))
        SS.read_arm(manifest, out, device)
        print(f"read {arm}", flush=True)
    paths = (
        {a: arm_dir(a) / "native_reads.json" for a in new}
        | {a: reads_of(d) for a, d in CACHED.items()}
        | {a: reads_of(d) for a, d in scale_arms().items()}
    )
    keys = {it["text"] for it in items()}
    words = sorted(k for k in keys if len(k) > 1)
    singles = sorted(k for k in keys if len(k) == 1)
    # EN refs' own flat_white, per (pi, seed): the scene's baseline
    from eval.enref import enref_file

    en_fw = {
        (pi, s): SS.placement({"file": str(enref_file(SS.ENREF, pi, s))})["flat_white"]
        for pi in range(PROMPTS)
        for s in range(SEEDS)
    }
    recs, hits, metrics = {}, {}, {"switch": switch, "check": check}
    metrics["row_sets"] = json.loads((OUT / NAME / "row_sets.json").read_text())
    metrics["en_flat_white"] = {f"p{k[0]}s{k[1]}": v for k, v in en_fw.items()}
    for a, p in paths.items():
        rs = [
            m
            for m in json.loads(p.read_text("utf-8"))
            if m["clause"] == CLAUSE and m["pi"] < PROMPTS and m["seed"] < SEEDS
        ]
        assert len(rs) == KEYS * PROMPTS * SEEDS, (a, len(rs))
        for m in rs:
            m["flat_white"] = SS.placement(m)["flat_white"]
        recs[a] = rs
        h = R.hits(p, keys, CLAUSE)
        hits[a] = {k: v for k, v in h.items() if k[2] < PROMPTS and k[3] < SEEDS}
        print(f"===== {a}", flush=True)
        t = R.tally(hits[a])["total"]
        metrics[a] = {"total": t, "scene": scene(rs, en_fw)}
        print(f"  scene {metrics[a]['scene']}", flush=True)
    metrics["paired"] = {}
    groups = (("words", words), ("singles", singles))
    for a in list(new) + [UP, MIX]:
        for b in (
            (UP, MIX)
            + (
                (RK, "rk_self")
                if (a[:3] in ("gs_", "rk_", "h0_") or a.startswith("ball_"))
                and a != "rk_self"
                else ()
            )
            + ((RK,) if a == "rk_self" else ())
        ):
            if a == b:
                continue
            for g, ks in groups:
                pr = R.paired(
                    *({k: v for k, v in hits[x].items() if k[0] in ks} for x in (a, b))
                )
                metrics["paired"][f"{g}: {a} vs {b}"] = pr
                print(f"  {g}: {a} vs {b} {pr}", flush=True)
    run_dir = make_run_dir("cjk_anima_reseed", label=label, root=HOME / "results")
    sheets(recs, list(new), run_dir / "sheets")
    write_result(
        run_dir,
        script=__file__,
        args={"switch": switch, "legs": "read", "label": label},
        label=label,
        metrics=metrics,
        artifacts=[str(OUT / NAME)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


def scene(rs: list, en_fw: dict) -> dict:
    """EN-ref cos out + flat_white, words / singles / singles p01; ``fw_up``:
    renders whose flat_white is ≥ 0.1 over their EN ref's."""

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None

    out = {}
    for g, sel in (
        ("words", lambda m: len(m["text"]) > 1),
        ("singles", lambda m: len(m["text"]) == 1),
        ("singles_p01", lambda m: len(m["text"]) == 1 and m["pi"] == 1),
    ):
        xs = [m for m in rs if sel(m)]
        out[g] = {
            "cos_out": mean([m.get("en_cos_out") for m in xs]),
            "flat_white": mean([m["flat_white"] for m in xs]),
            "fw_up": sum(
                m["flat_white"] - en_fw[(m["pi"], m["seed"])] >= 0.1 for m in xs
            ),
            "n": len(xs),
        }
    out["fw_up_by_prompt"] = {
        f"p{pi}": sum(
            m["flat_white"] - en_fw[(m["pi"], m["seed"])] >= 0.1
            for m in rs
            if m["pi"] == pi
        )
        for pi in range(PROMPTS)
    }
    return out


def sheets(recs: dict, new: list, out: Path) -> None:
    """One sheet per key at seed 0: rows = p00–p03, cols = EN ref | kana_mix |
    kana_up | the new arms."""
    from PIL import Image

    from cjk_scale.paths import load_experiment
    from common.readers import contact_sheet
    from eval.enref import enref_file

    SS = load_experiment("sigma_split")
    out.mkdir(parents=True, exist_ok=True)
    cols = [MIX, UP, *new]
    by = {a: {(m["text"], m["pi"], m["seed"]): m for m in recs[a]} for a in cols}
    for text in sorted({k[0] for k in by[UP]}):
        rows = []
        for pi in range(PROMPTS):
            ref = enref_file(SS.ENREF, pi, SHEET_SEED)
            rows.append((Image.open(ref).convert("RGB"), [f"EN ref p{pi:02d}"]))
            for a in cols:
                m = by[a][(text, pi, SHEET_SEED)]
                r0 = next((r for r in m.get("reads", []) if not r.get("whole")), {})
                rows.append(
                    (
                        Image.open(m["file"]).convert("RGB"),
                        [
                            f"{a}{' ✓' if m.get('exact') else ''}",
                            f"sfx {(r0.get('sfx') or '')[:14]}",
                            f"out {m.get('en_cos_out') or 0:.3f} fw {m['flat_white']:.2f}",
                        ],
                    )
                )
        contact_sheet(rows, out / f"sheet_{text}.png", thumb=224, cols=1 + len(cols))


def traj(label: str) -> None:
    """x̂0 per σ, kana_up / kana_mix rows at every σ, the 13 words × p00–p03."""
    import torch

    from bench._common import make_run_dir, write_result
    from common.readers import Readers, contact_sheet, read_scored
    from common.text import lev, norm
    from PIL import Image

    sets, _ = row_sets()
    SS, sp = make_splitter({k: sets[k] for k in ("up", "mix")})
    its = [it for it in items() if len(it["text"]) > 1 and it["seed"] == TRAJ_SEED]
    root = OUT / NAME / "traj"
    manifest = []
    for side in ("mix", "up"):
        for it in its:
            d = root / side / f"p{it['pi']:02d}_{it['text']}_s{TRAJ_SEED}"
            d.mkdir(parents=True, exist_ok=True)
            x0s: dict = {}
            sp.render(d / "sig0.00.png", it, side, side, 0.5, x0s)
            steps = sorted(x0s)
            for target in TRAJ_SIGMAS:
                if target == 0.0:
                    fn, s = d / "sig0.00.png", 0.0
                else:
                    i = min(steps, key=lambda j: abs(x0s[j][0] - target))
                    s = x0s[i][0]
                    fn = d / f"sig{s:.2f}.png"
                    sp.decode(x0s[i][1], fn)
                manifest.append(
                    {
                        "file": str(fn),
                        "side": side,
                        "pi": it["pi"],
                        "text": it["text"],
                        "sigma_target": target,
                        "sigma": s,
                    }
                )
        print(f"traj {side}: {len(its)} renders", flush=True)
    sp.free()
    rd = Readers(torch.device("cuda"))
    for m in manifest:
        reads = read_scored(rd, m)
        t = norm(m["text"])
        rs = [norm(r.get("sfx") or "") for r in reads if not r.get("whole")]
        rs = [r for r in rs if r]
        m["read"] = max(rs, key=len, default="")
        m["len_over"] = len(m["read"]) - len(t)
        m["dup"] = any(re.search(r"(.)\1", r) for r in rs)
        m["le1"] = min((lev(r, t) for r in rs), default=len(t)) <= 1
    del rd
    summary: dict = {}
    for side in ("mix", "up"):
        for target in TRAJ_SIGMAS:
            xs = [
                m for m in manifest if m["side"] == side and m["sigma_target"] == target
            ]
            summary[f"{side} σ{target:g}"] = {
                "n": len(xs),
                "any_read": sum(bool(m["read"]) for m in xs),
                "longer": sum(m["read"] != "" and m["len_over"] > 0 for m in xs),
                "dup": sum(m["dup"] for m in xs),
                "le1": sum(m["le1"] for m in xs),
            }
            print(f"  {side} σ{target:g} {summary[f'{side} σ{target:g}']}", flush=True)
    run_dir = make_run_dir(
        "cjk_anima_reseed", label=f"{label}-traj", root=HOME / "results"
    )
    out = run_dir / "sheets_traj"
    out.mkdir(parents=True, exist_ok=True)
    for text in dict.fromkeys(it["text"] for it in its):
        rows = []
        for side in ("mix", "up"):
            for pi in range(PROMPTS):
                for m in manifest:
                    if m["side"] == side and m["pi"] == pi and m["text"] == text:
                        rows.append(
                            (
                                Image.open(m["file"]).convert("RGB"),
                                [f"{side} p{pi:02d} σ{m['sigma']:.2f}", m["read"][:14]],
                            )
                        )
        contact_sheet(rows, out / f"traj_{text}.png", thumb=192, cols=len(TRAJ_SIGMAS))
    (root / "traj_reads.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    write_result(
        run_dir,
        script=__file__,
        args={"legs": "traj", "label": label},
        label=f"{label}-traj",
        metrics={"traj": summary},
        artifacts=[str(root)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--label", required=True)
    p.add_argument("--switch", type=float, default=0.75)
    p.add_argument(
        "--legs",
        nargs="+",
        default=["render", "read", "traj"],
        choices=["render", "read", "traj"],
    )
    a = p.parse_args()
    check = render(a.switch) if "render" in a.legs else None
    if "read" in a.legs:
        read(a.switch, check, a.label)
    if "traj" in a.legs:
        traj(a.label)


if __name__ == "__main__":
    main()
