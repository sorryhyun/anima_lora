#!/usr/bin/env python
"""influence_target — does the dev target see what piecenat saw? (value-only)

Step 1 of the influence line after the smoke (`../influence_smoke`,
`reports/influence_smoke_2026_09_25.md`) and the piecenat read
(`reports/piecenat_2026_09_25.md`). No gradients, no bank: the in-box FM
loss is *measured* on the 8 piecenat pieces at three table points of
run0925_300f (seed / step 5 000 / final), and read against the ruler that
now exists for those pieces.

Two reads on one set of matched renders:

T1  identity — ΔL_in(correct)[s] = L_seed − L_table on the correct render.
    piecenat says 2-glyph pieces were bought (った / です / すごい / メン)
    and the 3+-glyph ones were not (ちょっと / ありがとう / こんにちは).
    The read is whether the per-piece loss gain ranks like the piecenat
    gains (Spearman against the lenient and the official en gains; the
    2-glyph vs 3+-glyph group means). A pass means the existing target sees
    identity at the piece level; a fail on ありがとう (largest loss gain in
    the smoke, 0 → 0 official) means the loss rewards length, not identity.

T2  doubling — M[s] = L_in(doubled) − L_in(correct) on a matched pair (the
    same scene, font, fill, orientation, tilt and colour draw; only the
    glyph string differs: すごい vs すごいい), the diffusion-classifier
    margin under the correct caption. ΔM = M_table − M_seed: positive, the
    table prefers the correct render more than the seed did; negative, it
    moved toward the doubled one. piecenat read すごい / メン doubled in
    `word` (すごいい / メンン). If ΔM does not move at the scale of T1, or
    moves against those reads, the FM-loss family cannot see the
    acceptance axis and the influence line closes for Axis 1.

Matched pairs: both variants are re-rendered from the dev item's scene with
the same `random.Random(seed_k)` (font, stroke, tilt, colour draws consume
the stream identically; only `fit_text` sees the extra glyph, so the doubled
string is drawn a step smaller to fit the same bubble). Noise and σ
are shared across the pair and across the tables (paired reads). Bootstrap
CIs are over items (one scene per item).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, bootstrap, data_dir  # noqa: E402

bootstrap()

from bench._common import make_run_dir, write_result  # noqa: E402

BINS = ((0.3, 0.5), (0.5, 0.7), (0.7, 0.9))
STAGES = ("stage0507", "stage0305")

# reports/piecenat_2026_09_25.md — `native` ruler, piece alone in a native
# scene, `en` clause, of 16 renders: (seed, 300f) official = hit_sfx ∧ hit_vl,
# lenient = the piece string in any box read.
PIECENAT = {
    "った": {"official": (0, 3), "lenient": (0, 12)},
    "です": {"official": (0, 0), "lenient": (3, 14)},
    "すごい": {"official": (1, 6), "lenient": (4, 10)},
    "メン": {"official": (1, 0), "lenient": (8, 11)},
    "しい": {"official": (3, 2), "lenient": (13, 12)},
    "ちょっと": {"official": (0, 0), "lenient": (0, 2)},
    "ありがとう": {"official": (0, 0), "lenient": (0, 3)},
    "こんにちは": {"official": (0, 0), "lenient": (0, 0)},
}
# the doubled forms piecenat read in `word` (§ The other two layers)
DOUBLED_SEEN = {"すごい": "すごいい", "メン": "メンン"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--tag", default="run0925_300f", help="data dirs' tag")
    p.add_argument("--pieces", default=",".join(PIECENAT))
    p.add_argument(
        "--doubled",
        default="",
        help="s:doubled,… overrides; default doubles the last glyph "
        "(すごい → すごいい, the piecenat form)",
    )
    run = OUT / "rows_scale_joint0507_0305_run0925_300f"
    p.add_argument(
        "--tables",
        default=(
            f"seed={OUT / 'rows_step1_0921_merged' / 'trained.pt'},"
            f"k5={run / 'intermediate' / 'trained_step5000.pt'},"
            f"final={run / 'trained.pt'}"
        ),
        help="name=path,… ; the first is the base table, the rest overlay it",
    )
    p.add_argument("--dev_items", type=int, default=24, help="per piece, both tiers")
    p.add_argument(
        "--min_glyph",
        type=int,
        default=10,
        help="px floor for the re-render, both variants (the stage's 28 / 12 floor "
        "rejects the doubled string in half the bubbles; the correct render's size is "
        "set by the dev item's fill, not the floor)",
    )
    p.add_argument("--noises", type=int, default=2, help="per item per σ bin")
    p.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def doubled_form(s: str, overrides: dict) -> str:
    return overrides.get(s) or (s + s[-1])


def load_recs(stage: str, tag: str) -> list[dict]:
    d = data_dir(stage, tag)
    assert (d / "train.jsonl").exists(), f"no data dir {d}"
    return [
        json.loads(ln)
        for ln in (d / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if ln
    ]


def sample_plan(args, recs_by_stage, rng):
    """dev_pick[piece] = [(stage, index)] — scene_piece items whose text is
    the piece, over both tiers."""
    pieces = [s for s in args.pieces.split(",") if s]
    dev_pick = {}
    for s in pieces:
        cands = [
            (st, i)
            for st, recs in recs_by_stage.items()
            for i, r in enumerate(recs)
            if r.get("recipe") == "scene_piece" and r.get("text") == s
        ]
        assert len(cands) >= 4, f"piece {s!r}: only {len(cands)} scene_piece items"
        dev_pick[s] = sorted(rng.sample(cands, min(args.dev_items, len(cands))))
    return dev_pick


def load_overlay(path: Path, ext_ids: list[int], row_scale: float, base):
    import torch

    src = torch.load(path, map_location="cpu", weights_only=False)
    raw = src["delta"]["raw"].float()
    idx = {int(e): i for i, e in enumerate(src["delta"]["ext_ids"])}
    rs = src["delta"].get("row_scale")
    k = float(rs) / row_scale if rs is not None else 1.0
    out = base.clone()
    n = 0
    for i, e in enumerate(ext_ids):
        j = idx.get(int(e))
        if j is not None:
            out[i] = (raw[j] * k).to(out.device)
            n += 1
    return out, n


def render_pairs(args, dev_pick, recs_by_stage, cfgs, work, overrides):
    """Re-render every dev item's scene with the correct and the doubled
    string under one rng. Returns items: [{piece, k, stage, variant, rec}]
    with rec['file'] / ['box'] / ['text'] of the re-render (caption = the
    dev item's, which names the correct string)."""
    from common.render.flat import find_fonts, pick_font
    from common.render.scene import render_into_scene
    from data.synth import load_scenes

    fonts = find_fonts()
    scenes = {}
    for st, cfg in cfgs.items():
        d = cfg.data
        got = load_scenes(d["scenes"], 0.0, 0, "", d["scene_one_bubble"])
        scenes[st] = {(sc["pool"], sc["i"]): sc for sc in got}
    img_dir = work / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    items, dropped = [], defaultdict(int)
    k = 0
    for s, pick in dev_pick.items():
        forms = {"correct": s, "doubled": doubled_form(s, overrides)}
        for st, i in pick:
            rec = recs_by_stage[st][i]
            sc = scenes[st][(rec["scene_pool"], rec["scene"])]
            horiz = bool(rec.get("horizontal", False))
            cfg_d = cfgs[st].data
            base = random.Random(args.seed * 1_000_003 + k)
            font = pick_font(s, fonts, base)
            stroke = base.random() < float(cfg_d["stroke"])
            draw_seed = base.random()
            out = {}
            for variant, text in forms.items():
                drawn = render_into_scene(
                    scene=sc,
                    text=text,
                    font_path=font,
                    rng=random.Random(draw_seed),
                    min_glyph=int(args.min_glyph),
                    stroke=stroke,
                    fill_frac=float(rec["fill"]),
                    max_lines=1,
                    cuts=None,
                    vertical_only=bool(cfg_d["vertical"]) and not horiz,
                    fewest_lines=False,
                    horizontal=horiz,
                )
                if drawn is None:
                    break
                out[variant] = drawn
            if len(out) < 2:
                dropped[s] += 1
                k += 1
                continue
            for variant, (im, box) in out.items():
                f = img_dir / f"{k:04d}_{variant}.png"
                im.save(f)
                r = dict(rec)
                r.update(file=str(f), box=box, text=forms[variant], units=[forms[variant]])
                items.append(
                    {"piece": s, "k": k, "stage": st, "variant": variant, "rec": r}
                )
            k += 1
    (work / "items.jsonl").write_text(
        "\n".join(json.dumps(it, ensure_ascii=False) for it in items) + "\n",
        encoding="utf-8",
    )
    return items, dict(dropped)


def encode_latents(items, device, work):
    """One latent per item file, at the scene's own shape; cached under work."""
    import torch
    from common.models import encode_images, load_vae

    lat_file = work / "latents.pt"
    if lat_file.exists():
        lat = torch.load(lat_file)
        if len(lat) == len(items):
            return lat
    vae = load_vae(device)
    by_shape = defaultdict(list)
    for n, it in enumerate(items):
        by_shape[tuple(it["rec"]["shape"])].append(n)
    lat = [None] * len(items)
    for shape, idxs in by_shape.items():
        got = encode_images(vae, [items[n]["rec"]["file"] for n in idxs], device, shape)
        for n, t in zip(idxs, got):
            lat[n] = t
    del vae
    torch.cuda.empty_cache()
    torch.save(lat, lat_file)
    return lat


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2 + 1
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


def boot_ci(per_item: dict, rng, n: int):
    """per_item: item k → list of paired values; the item mean is the unit.
    Returns (mean, lo, hi) of the grand mean over item means."""
    means = [sum(v) / len(v) for v in per_item.values() if v]
    if not means:
        return 0.0, 0.0, 0.0
    m = sum(means) / len(means)
    if len(means) < 2:
        return m, m, m
    bs = []
    for _ in range(n):
        pick = [means[rng.randrange(len(means))] for _ in means]
        bs.append(sum(pick) / len(pick))
    bs.sort()
    return m, bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    overrides = dict(kv.split(":") for kv in args.doubled.split(",") if kv)
    tables = [kv.split("=", 1) for kv in args.tables.split(",") if kv]
    assert len(tables) >= 2, "--tables needs the base and at least one moved table"
    recs_by_stage = {st: load_recs(st, args.tag) for st in STAGES}
    dev_pick = sample_plan(args, recs_by_stage, rng)

    print(f"influence_target {args.label}: tag {args.tag}")
    for s, pick in dev_pick.items():
        by = defaultdict(int)
        for st, _ in pick:
            by[st] += 1
        print(f"  {s} → {doubled_form(s, overrides)}: {len(pick)} items {dict(by)}")
    n_items = sum(len(v) for v in dev_pick.values())
    n_fwd = n_items * 2 * len(BINS) * args.noises * len(tables)
    print(
        f"  tables: {', '.join(n for n, _ in tables)}; reads ≤ {n_items} items × 2 variants × "
        f"{len(BINS)} bins × {args.noises} noises × {len(tables)} tables = {n_fwd} forwards"
    )
    if args.dry_run:
        return

    import torch
    from common.models import checkpoints, dit_forward, gen_args
    from library.anima.vocab_pack import attached_pack_rows, strategy_pack
    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model
    from library.inference.text import ensure_text_strategies
    from library.runtime.noise import fm_training_batch
    from train.stage import _encode_text

    from cjk_scale.config import load
    from cjk_scale.loss import box_mask
    from cjk_scale.rows import RowTable

    t_start = time.time()
    work = OUT / f"influence_target_{args.label}"
    work.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(
        "influence_target",
        label=args.label,
        root=LINE / "experiments" / "influence_target" / "results",
    )
    cfgs = {st: load(st, None) for st in STAGES}
    args_gen = gen_args(512, 1000, 4.0, work)
    device = get_generation_settings(args_gen).device

    items, dropped = render_pairs(args, dev_pick, recs_by_stage, cfgs, work, overrides)
    n_pairs = len(items) // 2
    print(f"rendered {n_pairs} pairs; dropped {dropped}", flush=True)
    lat = encode_latents(items, device, work)
    print(f"latents: {len(lat)} in {(time.time() - t_start) / 60:.1f} min", flush=True)

    recs = [it["rec"] for it in items]
    cache, train_ext, _ev = _encode_text(recs, [], device, work, te_cache=work / "te_cache")

    anima = load_dit_model(args_gen, device, torch.bfloat16)
    anima.requires_grad_(False)
    assert attached_pack_rows(anima), "no vocab pack attached — set ANIMA_VOCAB_PACK"
    tok, _ = ensure_text_strategies(checkpoints().text_encoder, vocab_pack=None)
    rows = RowTable(
        anima,
        device,
        set(train_ext),
        strategy_pack(tok),
        warm=Path(tables[0][1]),
        init_anchor=0.0,
        free_residual=0.0,
        lr=0.0,
    )
    raw = rows.delta.raw
    ext_ids = [int(e) for e in rows.delta.ext_ids]
    base_tab = raw.detach().clone()
    tabs = {tables[0][0]: base_tab}
    for name, path in tables[1:]:
        tabs[name], n_ov = load_overlay(Path(path), ext_ids, rows.row_scale, base_tab)
        print(
            f"table {name}: overlays {n_ov} rows, ‖Δ‖ {float((tabs[name] - base_tab).norm()):.3f}",
            flush=True,
        )
    anima.eval()
    g = torch.Generator(device=device).manual_seed(args.seed)

    def set_table(t):
        with torch.no_grad():
            raw.copy_(t)

    # pair index: k → {variant: item index}
    pair = defaultdict(dict)
    for n, it in enumerate(items):
        pair[it["k"]][it["variant"]] = n

    reads = []  # one row per (k, bin, noise, variant, table)
    t0 = time.time()
    with torch.no_grad():
        for j, (k, var) in enumerate(sorted(pair.items())):
            nc, nd = var["correct"], var["doubled"]
            piece = items[nc]["piece"]
            lc = lat[nc][None].to(device)
            ld = lat[nd][None].to(device)
            for b, (t_min, t_max) in enumerate(BINS):
                for nz in range(args.noises):
                    noise = torch.randn(lc.shape, generator=g, device=device, dtype=lc.dtype)
                    noisy_c, ts, target_c = fm_training_batch(
                        lc, noise, dtype=torch.bfloat16, device=device, t_min=t_min, t_max=t_max
                    )
                    sig = ts.float().view(-1, 1, 1, 1)
                    noisy_d = ((1.0 - sig) * ld + sig * noise).to(torch.bfloat16)
                    target_d = noise - ld
                    for variant, n_it, noisy, target in (
                        ("correct", nc, noisy_c, target_c),
                        ("doubled", nd, noisy_d, target_d),
                    ):
                        r = items[n_it]["rec"]
                        for tname, tab in tabs.items():
                            set_table(tab)
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                pred = dit_forward(anima, noisy, ts, cache, [r["caption"]], device)
                            se = (pred.float() - target.float()) ** 2
                            m = box_mask(se.shape, [r], se.device)
                            per_cell = se.mean(dim=1, keepdim=True)
                            loss_in = float((per_cell * m).sum() / m.sum().clamp(min=1))
                            reads.append(
                                {
                                    "piece": piece,
                                    "k": k,
                                    "bin": b,
                                    "noise": nz,
                                    "sigma": float(ts[0]),
                                    "variant": variant,
                                    "table": tname,
                                    "in": loss_in,
                                    "plain": float(se.mean()),
                                }
                            )
            if (j + 1) % 16 == 0 or j + 1 == len(pair):
                print(
                    f"  {j + 1}/{len(pair)} pairs, {(time.time() - t0) / 60:.1f} min",
                    flush=True,
                )
    set_table(base_tab)
    (work / "reads.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in reads) + "\n", encoding="utf-8"
    )

    # -- compute -------------------------------------------------------------
    base = tables[0][0]
    moved = [n for n, _ in tables[1:]]
    val = {}
    for r in reads:
        val[(r["k"], r["bin"], r["noise"], r["variant"], r["table"])] = r["in"]
    keys = sorted({(r["k"], r["bin"], r["noise"]) for r in reads})
    piece_of = {r["k"]: r["piece"] for r in reads}
    pieces = list(dev_pick)
    brng = random.Random(args.seed + 1)

    t1 = {s: {} for s in pieces}  # identity gain per table (+ per bin)
    t2 = {s: {} for s in pieces}  # margin per table, Δ margin per moved table
    for s in pieces:
        ks = [k for k in piece_of if piece_of[k] == s]
        for tn in moved:
            gain = defaultdict(list)
            gain_bin = {b: defaultdict(list) for b in range(len(BINS))}
            for k, b, nz in keys:
                if piece_of[k] != s:
                    continue
                d = val[(k, b, nz, "correct", base)] - val[(k, b, nz, "correct", tn)]
                gain[k].append(d)
                gain_bin[b][k].append(d)
            m, lo, hi = boot_ci(gain, brng, args.boot)
            t1[s][tn] = {
                "gain": m,
                "lo": lo,
                "hi": hi,
                "bins": {
                    f"{BINS[b][0]}-{BINS[b][1]}": boot_ci(gain_bin[b], brng, args.boot)[0]
                    for b in range(len(BINS))
                },
            }
        for tn in [base] + moved:
            marg = defaultdict(list)
            for k, b, nz in keys:
                if piece_of[k] != s:
                    continue
                marg[k].append(val[(k, b, nz, "doubled", tn)] - val[(k, b, nz, "correct", tn)])
            t2[s][tn] = {"margin": boot_ci(marg, brng, args.boot)[0]}
        for tn in moved:
            dm = defaultdict(list)
            dm_bin = {b: defaultdict(list) for b in range(len(BINS))}
            for k, b, nz in keys:
                if piece_of[k] != s:
                    continue
                m_t = val[(k, b, nz, "doubled", tn)] - val[(k, b, nz, "correct", tn)]
                m_0 = val[(k, b, nz, "doubled", base)] - val[(k, b, nz, "correct", base)]
                dm[k].append(m_t - m_0)
                dm_bin[b][k].append(m_t - m_0)
            m, lo, hi = boot_ci(dm, brng, args.boot)
            t2[s][tn].update(
                dmargin=m,
                lo=lo,
                hi=hi,
                bins={
                    f"{BINS[b][0]}-{BINS[b][1]}": boot_ci(dm_bin[b], brng, args.boot)[0]
                    for b in range(len(BINS))
                },
            )
        t2[s]["n_items"] = len(ks)

    # ranking against piecenat, per moved table
    rank = {}
    for tn in moved:
        gains = [t1[s][tn]["gain"] for s in pieces]
        ruler = {
            key: [PIECENAT[s][key][1] - PIECENAT[s][key][0] for s in pieces]
            for key in ("lenient", "official")
        }
        two = [t1[s][tn]["gain"] for s in pieces if len(s) == 2]
        more = [t1[s][tn]["gain"] for s in pieces if len(s) >= 3]
        dm_two = [t2[s][tn]["dmargin"] for s in pieces if len(s) == 2]
        dm_more = [t2[s][tn]["dmargin"] for s in pieces if len(s) >= 3]
        rank[tn] = {
            "spearman_lenient": spearman(gains, ruler["lenient"]),
            "spearman_official": spearman(gains, ruler["official"]),
            "gain_2glyph_mean": sum(two) / max(len(two), 1),
            "gain_3plus_mean": sum(more) / max(len(more), 1),
            "dmargin_2glyph_mean": sum(dm_two) / max(len(dm_two), 1),
            "dmargin_3plus_mean": sum(dm_more) / max(len(dm_more), 1),
            "dmargin_spearman_official": spearman(
                [t2[s][tn]["dmargin"] for s in pieces], ruler["official"]
            ),
        }

    # -- report ----------------------------------------------------------------
    fin = moved[-1]
    lines = [
        f"# influence_target — {args.label} ({time.strftime('%Y-%m-%d')})",
        "",
        f"{n_pairs} matched pairs ({', '.join(f'{s} {t2[s]['n_items']}' for s in pieces)}; "
        f"dropped {dropped or 'none'}), {len(BINS)} σ bins × {args.noises} noises, tables "
        f"{' / '.join(tabs)}; {len(reads)} reads, {(time.time() - t_start) / 60:.0f} min. "
        "CIs: 95 % bootstrap over items.",
        "",
        "## T1 — identity: in-box loss gain on the correct render (seed − table; + = improved)",
        "",
        "| piece | glyphs | " + " | ".join(f"gain @{tn} [CI]" for tn in moved)
        + f" | bins @{fin} (0.3-0.5 / 0.5-0.7 / 0.7-0.9) | piecenat lenient | official |",
        "|---|---|" + "---|" * len(moved) + "---|---|---|",
    ]
    for s in sorted(pieces, key=lambda s: -t1[s][fin]["gain"]):
        cells = " | ".join(
            f"{t1[s][tn]['gain']:+.4f} [{t1[s][tn]['lo']:+.4f}, {t1[s][tn]['hi']:+.4f}]"
            for tn in moved
        )
        bins = " / ".join(f"{v:+.4f}" for v in t1[s][fin]["bins"].values())
        pn = PIECENAT[s]
        lines.append(
            f"| {s} | {len(s)} | {cells} | {bins} | {pn['lenient'][0]} → {pn['lenient'][1]} | "
            f"{pn['official'][0]} → {pn['official'][1]} |"
        )
    lines += ["", "| table | ρ vs lenient | ρ vs official | 2-glyph mean | 3+-glyph mean |", "|---|---|---|---|---|"]
    for tn in moved:
        r = rank[tn]
        lines.append(
            f"| {tn} | {r['spearman_lenient']:+.2f} | {r['spearman_official']:+.2f} | "
            f"{r['gain_2glyph_mean']:+.4f} | {r['gain_3plus_mean']:+.4f} |"
        )
    lines += [
        "",
        "## T2 — doubling: margin M = L_in(doubled) − L_in(correct); ΔM = M_table − M_seed (+ = prefers correct more)",
        "",
        "| piece | doubled | M @seed | " + " | ".join(f"ΔM @{tn} [CI]" for tn in moved)
        + f" | bins @{fin} | piecenat doubled |",
        "|---|---|---|" + "---|" * len(moved) + "---|---|",
    ]
    for s in sorted(pieces, key=lambda s: t2[s][fin]["dmargin"]):
        cells = " | ".join(
            f"{t2[s][tn]['dmargin']:+.4f} [{t2[s][tn]['lo']:+.4f}, {t2[s][tn]['hi']:+.4f}]"
            for tn in moved
        )
        bins = " / ".join(f"{v:+.4f}" for v in t2[s][fin]["bins"].values())
        lines.append(
            f"| {s} | {doubled_form(s, overrides)} | {t2[s][base]['margin']:+.4f} | {cells} | {bins} | "
            f"{'yes (' + DOUBLED_SEEN[s] + ')' if s in DOUBLED_SEEN else ''} |"
        )
    lines += ["", "| table | ΔM 2-glyph mean | ΔM 3+-glyph mean | ρ(ΔM, official gain) |", "|---|---|---|---|"]
    for tn in moved:
        r = rank[tn]
        lines.append(
            f"| {tn} | {r['dmargin_2glyph_mean']:+.4f} | {r['dmargin_3plus_mean']:+.4f} | "
            f"{r['dmargin_spearman_official']:+.2f} |"
        )
    lines += [
        "",
        "Reads: T1 passes if the gains rank like piecenat (ρ vs lenient clearly positive, "
        "2-glyph mean above 3+-glyph) — the existing target sees piece identity. T2 is "
        "the acceptance-axis read: a ΔM that sits at zero (CI straddles 0 on every piece) "
        "or goes negative on the pieces piecenat saw doubled means the delta did not buy "
        "the correct-over-doubled preference the rulers need; the sign and size against "
        "T1 says whether the FM-loss family can see doubling at all.",
        "",
        f"Renders / latents / reads: `{work}`.",
    ]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)

    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics={
            "t1_identity": t1,
            "t2_doubling": t2,
            "rank": rank,
            "n_pairs": n_pairs,
            "dropped": dropped,
            "n_reads": len(reads),
            "minutes": (time.time() - t_start) / 60,
        },
        artifacts=["report.md"],
        extra={"work": str(work), "reads": str(work / "reads.jsonl")},
    )
    print(f"result: {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
