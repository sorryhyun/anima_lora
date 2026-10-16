#!/usr/bin/env python
"""swapword — a row's effect split into what holds whatever the word and what
follows the word (user 10-16: glyphs memorized, templates not; the commit
report's "read what does not change when the word changes", done on one-step
predictions instead of renders).

Held-out items: ``hira_cold_hp``'s table drawn again at ``table.SEED = 1``
(``--frac`` of every tier) into ``OUT/swapword/heldout_s1/data``. An item whose
quoted strings are all among the 81 hiragana is kept. Each item gets ``K``
swapped captions: every quoted glyph replaced by another of the 81, none of
the item's own glyphs (same length, same layout clause), the same swaps for
every arm.

Every arm is ``seed_retrain_0930``'s table with the 81 hiragana rows the
arm's, so arms differ in those rows only; ``ref`` is the same table with the
81 at the pack (Δ 0). Per item × σ in ``SIGMAS``, one x_σ (ε seeded per item ×
σ, shared by every arm), a batch of the true caption and its ``K`` swaps,
prediction v. With d(c) = σ · (v_arm(c) − v_ref(c)) the arm's rows' shift of
x̂₀ under caption c:

- ``M`` = mean_c d(c): the shift that holds whatever the word (template,
  text-ness, canvas) — ``T_in`` / ``T_in_lo`` (its 5×5-cell blur, ≈ 40 px) /
  ``T_out`` its mean squared size in the text box / outside it (dilated 2);
- ``S(c)`` = d(c) − M: the shift that follows the word — ``V_in`` / ``V_out``;
- ``rho`` = T_in / (T_in + V_in): the box's word-free share;
- losses against the flow target in the box / outside it: the true caption
  ``L_true`` and the swaps' ``L_wrong``; ``I`` = L_wrong − L_true (the correct
  word's margin: identity), ``G_wrong`` = L_wrong(ref) − L_wrong(arm) (what
  the rows buy with the wrong word: word-free fit), ``G_out`` = the same
  outside the box under the true caption (canvas fit).

    .venv/bin/python project/cjk_anima_reseed/probes/swapword.py data          # CPU
    make daemon-run ARGS="project/cjk_anima_reseed/probes/swapword.py run --label all"
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
from reseed import HOME, OUT, bootstrap  # noqa: E402

bootstrap()
os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"  # the runs trained routed, read routed

FLOOR = "seed_retrain_0930"
RUN = "hira_cold_hp"
HELD_SEED = 1
ROOT = OUT / "swapword"
HELD = ROOT / f"heldout_s{HELD_SEED}" / "data"
SIGMAS = (0.3, 0.5, 0.7, 0.9)
K = 3
OUT_DIL = 2  # latent cells of slack around the box for the out-box reads
ARMS = [
    "hira_cold_hp_s4050",
    "hira_cold_hp_s8100",
    "hira_cold_hp",
    "hira_cold_up",
    "hira_cold_mix",
    "hira_cold_rand",
    "hira_cold_sball_s4050",
    "hira_cold_sball_s8100",
    "hira_cold_sball",
    "hira_cold_sglyph",
    "hira_cold_spe",
    "hira_cold_slean",
    "hira_cold_sball_lr3_s4050",
    "hira_cold_sball_lr3_s8100",
    "hira_cold_sball_lr3",
    "hira_cold_sball_lr3_x2_s12150",
    "hira_cold_sball_lr3_x2",
]
KEYS = (
    "L_true_in",
    "L_wrong_in",
    "L_true_out",
    "L_wrong_out",
    "T_in",
    "T_in_lo",
    "V_in",
    "T_out",
    "V_out",
)


def data(frac: float) -> None:
    """The held-out draw: ``RUN``'s table at ``HELD_SEED`` into ``HELD``."""
    from reseed import builder
    from reseed import table as T
    from reseed.config import Run, load

    run = load(RUN)
    run.use_pack()
    T.SEED = HELD_SEED
    Run.data = property(lambda self: HELD)
    builder.build(run, None, frac)


def swaps(recs: list, alphabet: str) -> list:
    """``[(rec, [caption_true, swap_1 … swap_K])]`` for the items whose quoted
    strings are all in ``alphabet``."""
    al = set(alphabet)
    out = []
    for i, r in enumerate(recs):
        units = [u.replace(" ", "") for u in r["units"]]
        glyphs = "".join(units)
        if not glyphs or not set(glyphs) <= al:
            continue
        rng = random.Random(f"swap-{i}")
        pool = sorted(al - set(glyphs))
        # K distinct replacements per glyph position: the k-th swap takes the k-th
        draws = [[rng.sample(pool, K) for _ in u] for u in units]
        caps = [r["caption"]]
        for k in range(K):
            cap, at = r["caption"], 0
            for u, du in zip(units, draws):
                q = f'"{u}"'
                j = cap.index(q, at)
                v = "".join(d[k] for d in du)
                cap = cap[:j] + f'"{v}"' + cap[j + len(q) :]
                at = j + len(v) + 2
            caps.append(cap)
        assert len(set(caps)) == K + 1, caps
        out.append((i, r, caps))
    return out


def masks(rec: dict, h: int, w: int, device):
    """``(1, 1, h, w)`` in-box (every box the record carries, flat items too)
    and out-box (outside the box dilated ``OUT_DIL``)."""
    import torch
    import torch.nn.functional as F

    m = torch.zeros(1, 1, h, w, device=device)
    boxes = rec.get("boxes") or ([rec["box"]] if rec.get("box") else [])
    for x0, y0, x1, y1 in boxes:
        m[:, :, y0 // 8 : -(-y1 // 8), x0 // 8 : -(-x1 // 8)] = 1.0
    d = F.max_pool2d(m, 2 * OUT_DIL + 1, stride=1, padding=OUT_DIL)
    return m, 1.0 - d


def arm_tables(names: list, hira_ids: set):
    """The ext ids, and per name the floor's table with the 81 rows the arm's
    (``ref``: at the pack)."""
    from eval.ruler.arms import tables

    ids, tabs = tables([FLOOR, *names])
    pos = [i for i, e in enumerate(ids) if e in hira_ids]
    assert len(pos) == len(hira_ids), (len(pos), len(hira_ids))
    base = tabs[FLOOR]
    out = {"ref": base.clone()}
    out["ref"][pos] = 0.0
    rest = sorted(set(range(len(ids))) - set(pos))
    for a in [FLOOR, *names]:
        t = base.clone()
        t[pos] = tabs[a][pos]
        off = float((tabs[a][rest] - base[rest]).abs().max())
        if off > 1e-3:
            print(
                f"  {a}: rows outside the 81 differ from {FLOOR} by {off:.3g} "
                "(read at the floor's)",
                flush=True,
            )
        out[a] = t
    return ids, out


def run(label: str, max_items: int | None) -> None:
    import torch
    import torch.nn.functional as F

    from common.hooks import ExtDelta
    from common.models import dit_forward, encode_captions, gen_args
    from library.anima.vocab_pack import attached_pack_rows
    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model
    from reseed.trainer import vocab_idx
    from data.inventory import qwen_pieces
    from train.stage import LatentStore

    from eval.ruler.arms import arm_dirs

    names = [a for a in ARMS if a in arm_dirs()]
    missing = sorted(set(ARMS) - set(names))
    if missing:
        print(f"arms not on disk, skipped: {missing}", flush=True)
    recs = [
        json.loads(ln)
        for ln in (HELD / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if ln
    ]
    alphabet = "".join(json.loads((HELD / "vocabs.json").read_text(encoding="utf-8")))
    items = swaps(recs, alphabet)
    if max_items:
        items = items[:max_items]
    print(
        f"items: {len(items)} of {len(recs)} held-out, K {K}, σ {SIGMAS}, "
        f"{len(names) + 2} tables",
        flush=True,
    )
    hira_ids = vocab_idx(list(alphabet), qwen_pieces(char_rows=True))

    args = gen_args(512, 28, 4.0, ROOT / "_tmp")
    device = get_generation_settings(args).device
    cache = encode_captions(
        [c for _i, _r, caps in items for c in caps], device, ROOT / "te_cache"
    )
    ns = argparse.Namespace(seed=0, batch=K + 1, train_size=512)
    lat = LatentStore(ns, HELD, recs, list(range(len(recs))), device)
    anima = load_dit_model(args, device, torch.bfloat16)
    anima.requires_grad_(False)
    anima.eval()
    assert attached_pack_rows(anima), "no vocab pack attached to the DiT"
    ids, tabs = arm_tables(names, hira_ids)
    delta = ExtDelta(anima, ids, tabs["ref"].shape[1], device, row_scale=1.0)
    tabs = {a: t.to(device) for a, t in tabs.items()}
    order = ["ref", FLOOR, *names]

    rows = []  # one per item × σ × table
    t0 = time.time()
    for n_done, (i, r, caps) in enumerate(items):
        x0 = lat[[i]].to(device).float()
        _, _, h, w = x0.shape
        m_in, m_out = masks(r, h, w, device)
        n_in, n_out = m_in.sum().clamp(min=1), m_out.sum().clamp(min=1)

        def mean_in(x):  # (B, C, h, w) → (B,)
            return ((x**2).mean(1, keepdim=True) * m_in).sum((1, 2, 3)) / n_in

        def mean_out(x):
            return ((x**2).mean(1, keepdim=True) * m_out).sum((1, 2, 3)) / n_out

        for s in SIGMAS:
            g = torch.Generator().manual_seed(hash((i, s)) % (2**31))
            eps = torch.randn(x0.shape, generator=g).to(device)
            xs = ((1 - s) * x0 + s * eps).to(torch.bfloat16).repeat(K + 1, 1, 1, 1)
            ts = torch.full((K + 1,), s, device=device)
            tgt = (eps - x0).expand(K + 1, -1, -1, -1)
            v_ref = None
            for a in order:
                delta.raw.data.copy_(tabs[a])
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    v = dit_forward(anima, xs, ts, cache, caps, device).float()
                li, lo = mean_in(v - tgt), mean_out(v - tgt)
                rec = {
                    "item": i,
                    "tier": r["tier"],
                    "src": r["src"],
                    "glyphs": r["glyphs"],
                    "sigma": s,
                    "arm": a,
                    "L_true_in": float(li[0]),
                    "L_wrong_in": float(li[1:].mean()),
                    "L_true_out": float(lo[0]),
                    "L_wrong_out": float(lo[1:].mean()),
                }
                if a == "ref":
                    v_ref = v
                    rec.update({k: 0.0 for k in KEYS[4:]})
                else:
                    d = s * (v - v_ref)
                    M = d.mean(0, keepdim=True)
                    S = d - M
                    M_lo = F.avg_pool2d(
                        M, 5, stride=1, padding=2, count_include_pad=False
                    )
                    rec.update(
                        T_in=float(mean_in(M)[0]),
                        T_in_lo=float(mean_in(M_lo)[0]),
                        V_in=float(mean_in(S).mean()),
                        T_out=float(mean_out(M)[0]),
                        V_out=float(mean_out(S).mean()),
                    )
                rows.append(rec)
        if n_done % 20 == 0:
            el = time.time() - t0
            print(
                f"  {n_done + 1}/{len(items)} items, {el / 60:.1f} min "
                f"(eta {el / (n_done + 1) * (len(items) - n_done - 1) / 60:.1f})",
                flush=True,
            )
    del anima
    torch.cuda.empty_cache()
    res = summarize(rows, order)
    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir(
        "cjk_anima_reseed", label=f"swapword-{label}", root=HOME / "results"
    )
    (run_dir / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    (run_dir / "table.md").write_text(res.pop("md"), encoding="utf-8")
    write_result(
        run_dir,
        script="project/cjk_anima_reseed/probes/swapword.py",
        args={
            "label": label,
            "max_items": max_items,
            "K": K,
            "sigmas": SIGMAS,
            "held": str(HELD),
            "n_items": len(items),
        },
        metrics=res,
    )
    print((run_dir / "table.md").read_text(encoding="utf-8"), flush=True)
    print(f"swapword → {run_dir}", flush=True)


def derived(rs: list, ref: dict) -> dict:
    """Per-item means over ``rs`` (one arm's rows) with the ref's rows
    (``ref[(item, σ)]``) → the read's quantities."""
    import statistics as st

    def m(f):
        return st.fmean(f(x) for x in rs)

    T, V = m(lambda x: x["T_in"]), m(lambda x: x["V_in"])
    return {
        "I": m(lambda x: x["L_wrong_in"] - x["L_true_in"]),
        "G_true": m(lambda x: ref[x["item"], x["sigma"]]["L_true_in"] - x["L_true_in"]),
        "G_wrong": m(
            lambda x: ref[x["item"], x["sigma"]]["L_wrong_in"] - x["L_wrong_in"]
        ),
        "G_out": m(
            lambda x: ref[x["item"], x["sigma"]]["L_true_out"] - x["L_true_out"]
        ),
        "T_in": T,
        "T_in_lo": m(lambda x: x["T_in_lo"]),
        "V_in": V,
        "rho": T / (T + V) if T + V else 0.0,
        "T_out": m(lambda x: x["T_out"]),
        "V_out": m(lambda x: x["V_out"]),
        "n": len(rs),
    }


def paired(rs_a: list, rs_b: list, ref: dict, key: str) -> dict:
    """Item-level (σ pooled) ``key`` of arm a against arm b: mean Δ, a > b /
    a < b, Wilcoxon p."""
    import statistics as st

    from scipy.stats import wilcoxon

    def per_item(rs):
        by: dict = {}
        for x in rs:
            if key == "I":
                v = x["L_wrong_in"] - x["L_true_in"]
            elif key == "G_wrong":
                v = ref[x["item"], x["sigma"]]["L_wrong_in"] - x["L_wrong_in"]
            else:
                v = x[key]
            by.setdefault(x["item"], []).append(v)
        return {k: st.fmean(v) for k, v in by.items()}

    a, b = per_item(rs_a), per_item(rs_b)
    ks = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in ks]
    p = float(wilcoxon(d).pvalue) if any(d) else 1.0
    return {
        "mean_d": st.fmean(d),
        "up": sum(x > 0 for x in d),
        "down": sum(x < 0 for x in d),
        "p": p,
    }


def summarize(rows: list, order: list) -> dict:
    ref = {(x["item"], x["sigma"]): x for x in rows if x["arm"] == "ref"}
    by_arm = {a: [x for x in rows if x["arm"] == a] for a in order}
    res: dict = {"pooled": {}, "by_sigma": {}, "by_src": {}, "vs_floor": {}}
    for a in order[1:]:
        rs = by_arm[a]
        res["pooled"][a] = derived(rs, ref)
        res["by_sigma"][a] = {
            str(s): derived([x for x in rs if x["sigma"] == s], ref) for s in SIGMAS
        }
        res["by_src"][a] = {
            t: derived([x for x in rs if x["tier"].split("_")[0] == t], ref)
            for t in sorted({x["tier"].split("_")[0] for x in rs})
        }
        if a != FLOOR:
            res["vs_floor"][a] = {
                k: paired(rs, by_arm[FLOOR], ref, k)
                for k in ("I", "T_in", "T_in_lo", "V_in", "G_wrong", "T_out")
            }
    rr = by_arm["ref"]
    res["ref"] = {
        "I": sum(x["L_wrong_in"] - x["L_true_in"] for x in rr) / len(rr),
        "L_true_in": sum(x["L_true_in"] for x in rr) / len(rr),
    }
    cols = ("I", "G_wrong", "G_out", "T_in", "T_in_lo", "V_in", "rho", "T_out")
    lines = [
        f"ref (pack rows): I {res['ref']['I']:.4f}, L_true_in {res['ref']['L_true_in']:.4f}",
        "",
        "pooled over σ " + str(SIGMAS),
        "",
        "| arm | " + " | ".join(cols) + " |",
        "|---" * (len(cols) + 1) + "|",
    ]
    for a in order[1:]:
        p = res["pooled"][a]
        lines.append(f"| {a} | " + " | ".join(f"{p[c]:.4f}" for c in cols) + " |")
    for key in ("I", "T_in", "T_in_lo", "rho", "V_in"):
        lines += [
            "",
            f"{key} by σ",
            "",
            "| arm | " + " | ".join(str(s) for s in SIGMAS) + " |",
            "|---" * (len(SIGMAS) + 1) + "|",
        ]
        for a in order[1:]:
            b = res["by_sigma"][a]
            lines.append(
                f"| {a} | " + " | ".join(f"{b[str(s)][key]:.4f}" for s in SIGMAS) + " |"
            )
    srcs = sorted({t for a in order[1:] for t in res["by_src"][a]})
    for key in ("I", "T_in_lo", "rho"):
        lines += [
            "",
            f"{key} by tier family",
            "",
            "| arm | " + " | ".join(srcs) + " |",
            "|---" * (len(srcs) + 1) + "|",
        ]
        for a in order[1:]:
            b = res["by_src"][a]
            lines.append(
                f"| {a} | "
                + " | ".join(f"{b[t][key]:.4f}" if t in b else "–" for t in srcs)
                + " |"
            )
    lines += [
        "",
        f"paired against {FLOOR} (items, σ pooled): mean Δ, up / down, p",
        "",
        "| arm | I | T_in | T_in_lo | V_in | G_wrong | T_out |",
        "|---|---|---|---|---|---|---|",
    ]
    for a, v in res["vs_floor"].items():
        lines.append(
            f"| {a} | "
            + " | ".join(
                f"{v[k]['mean_d']:+.4f}, {v[k]['up']}/{v[k]['down']}, {v[k]['p']:.2g}"
                for k in ("I", "T_in", "T_in_lo", "V_in", "G_wrong", "T_out")
            )
            + " |"
        )
    res["md"] = "\n".join(lines) + "\n"
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("verb", choices=["data", "run"])
    p.add_argument(
        "--frac", type=float, default=0.025, help="data: share of every tier"
    )
    p.add_argument("--label", default="all")
    p.add_argument("--max_items", type=int, default=None, help="run: a smoke")
    a = p.parse_args()
    if a.verb == "data":
        data(a.frac)
    else:
        run(a.label, a.max_items)


if __name__ == "__main__":
    main()
