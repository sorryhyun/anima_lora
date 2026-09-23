"""boxprobe — what the box share actually controls, read without training.

A scene item's ext rows see the gradient ``s · g_in + (1 − s) · g_out``:
``g_in`` from the latent cells under the text box, ``g_out`` from the
background. The box share ``s`` is a knob on that mix and nothing else, so
the curve ``s(n_glyphs)`` can be read off the two gradients directly. For
every scene item of a built data dir, at ``K`` σ draws inside the stage
band, this backprops ``mean_in`` and ``mean_out`` separately onto the
warm table's rows (no optimizer step) and records, per item and draw:
``‖g_in‖``, ``‖g_out‖``, their cosine, and ``s*`` = the share at which the
two terms have equal magnitude (``r / (1 + r)``, ``r = ‖g_out‖ / ‖g_in‖``).

``report.md`` bins the reads by glyph count and puts the three candidate
curves beside the measured ``s*``: flat (``box_share`` everywhere), the
probe's linear ``min(ρ n, 0.75)`` and the configured log curve
(``loss.py``) — with each curve's in-box fraction of the resolved row
gradient, ``φ = s‖g_in‖ / (s‖g_in‖ + (1 − s)‖g_out‖)``. What it cannot say:
whether equalising φ across glyph counts is what makes rows learn — that
stays a micro arm.

Outputs ``output/cjk_anima_scale/boxprobe_scale_<stage>_<tag>/{reads.jsonl,
report.md}``.
"""

from __future__ import annotations

import json
import statistics as st
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

from .config import StageConfig
from .loss import box_share_of, glyph_count
from .paths import OUT, data_dir, run_tag
from .rows import RowTable

BINS = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 7), (8, 10**9))


def _bin(n: int) -> str:
    for lo, hi in BINS:
        if lo <= n <= hi:
            return f"{lo}" if lo == hi else (f"{lo}–{hi}" if hi < 10**9 else f"{lo}+")
    return "?"


def probe_dir(stage: str, tag: str) -> Path:
    return OUT / f"boxprobe_{run_tag(stage, tag)}"


def probe(
    cfg: StageConfig,
    tag: str,
    *,
    warm: Path | None,
    draws: int = 3,
    max_items: int = 0,
    seed: int = 0,
) -> Path:
    from common.models import checkpoints, dit_forward, gen_args
    from library.anima.vocab_pack import attached_pack_rows, strategy_pack
    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model
    from library.inference.text import ensure_text_strategies
    from library.runtime.noise import fm_training_batch
    from train.stage import LatentStore, _box_mask, _encode_text

    t = cfg.train
    t_min, t_max = cfg.band
    data = data_dir(cfg.stage, tag)
    out = probe_dir(cfg.stage, tag)
    out.mkdir(parents=True, exist_ok=True)
    assert (data / "train.jsonl").exists(), f"no data dir {data} — run the data step"
    recs = [
        json.loads(ln)
        for ln in (data / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if ln
    ]
    ev = json.loads((data / "eval.json").read_text(encoding="utf-8"))
    scene = [i for i, r in enumerate(recs) if r.get("src") == "scene" and "box" in r]
    if max_items:
        scene = scene[:max_items]
    assert scene, "no scene items with a box in the data dir"
    args = gen_args(512, int(t["steps"]), float(t["cfg"]), out)
    device = get_generation_settings(args).device
    cache, train_ext, _ = _encode_text(
        recs, ev, device, out, te_cache=data / "te_cache"
    )
    ns = SimpleNamespace(
        seed=seed, batch=1, train_size=512, row_blocks=0, row_boost="", arm="rows"
    )
    lat = LatentStore(ns, data, recs, list(range(len(recs))), device)
    anima = load_dit_model(args, device, torch.bfloat16)
    anima.requires_grad_(False)
    assert attached_pack_rows(anima), "no vocab pack attached to the DiT"
    tok, _ = ensure_text_strategies(checkpoints().text_encoder, vocab_pack=None)
    rows = RowTable(
        anima,
        device,
        train_ext,
        strategy_pack(tok),
        warm=warm,
        init_anchor=0.0,
        free_residual=0.0,
        lr=0.0,
    )
    raw = rows.delta.raw
    anima.train()
    print(
        f"boxprobe {cfg.stage} ({run_tag(cfg.stage, tag)}): σ [{t_min}, {t_max}], "
        f"{len(scene)} scene items × {draws} draws, warm {warm}",
        flush=True,
    )
    g = torch.Generator(device=device).manual_seed(seed)
    reads = []
    t0 = time.time()
    with (out / "reads.jsonl").open("w", encoding="utf-8") as f:
        for k, i in enumerate(scene):
            r = recs[i]
            latents = lat[[i]].to(device)
            for d in range(draws):
                noise = torch.randn(
                    latents.shape, generator=g, device=device, dtype=latents.dtype
                )
                noisy, ts, target = fm_training_batch(
                    latents,
                    noise,
                    dtype=torch.bfloat16,
                    device=device,
                    t_min=t_min,
                    t_max=t_max,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = dit_forward(anima, noisy, ts, cache, [r["caption"]], device)
                se = ((pred.float() - target.float()) ** 2).mean(dim=1, keepdim=True)
                m = _box_mask(se.shape, [r], se.device)
                n_in = m.sum()
                mean_in = (se * m).sum() / n_in.clamp(min=1)
                mean_out = (se * (1 - m)).sum() / (1 - m).sum().clamp(min=1)
                (g_in,) = torch.autograd.grad(mean_in, raw, retain_graph=True)
                (g_out,) = torch.autograd.grad(mean_out, raw)
                touched = (g_in.abs().sum(1) > 0) | (g_out.abs().sum(1) > 0)
                gi, go = g_in[touched].flatten(), g_out[touched].flatten()
                ni, no = float(gi.norm()), float(go.norm())
                cos = float((gi @ go) / max(ni * no, 1e-30))
                ratio = no / max(ni, 1e-30)
                rec = {
                    "i": i,
                    "text": r["text"],
                    "n_glyphs": glyph_count(r["text"]),
                    "kind": r.get("kind"),
                    "law_kind": r.get("law_kind"),
                    "px": r.get("px"),
                    "rows": int(touched.sum()),
                    "sigma": float(ts.float().view(-1)[0]),
                    "box_cells": int(n_in),
                    "mean_in": float(mean_in),
                    "mean_out": float(mean_out),
                    "g_in": ni,
                    "g_out": no,
                    "cos": cos,
                    "ratio": ratio,
                    "s_star": ratio / (1.0 + ratio),
                }
                reads.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (k + 1) % 20 == 0 or k + 1 == len(scene):
                print(
                    f"  {k + 1}/{len(scene)} items, {(time.time() - t0) / 60:.1f} min",
                    flush=True,
                )
    report(cfg, tag, reads, out)
    del anima
    torch.cuda.empty_cache()
    return out


def _phi(s: float, gi: float, go: float) -> float:
    a, b = s * gi, (1 - s) * go
    return a / max(a + b, 1e-30)


def report(cfg: StageConfig, tag: str, reads: list, out: Path) -> None:
    t = cfg.train
    s1, s_cap, n_cap = (
        float(t["box_share"]),
        float(t["box_share_cap"]),
        float(t["box_share_glyphs"]),
    )
    curves = {
        "flat": lambda n: s1,
        "linear": lambda n: min(s1 * n, 0.75),
        "log": lambda n: box_share_of(n, s1, s_cap, n_cap),
    }
    by = defaultdict(list)
    for r in reads:
        by[_bin(r["n_glyphs"])].append(r)
    med = st.median
    lines = [
        f"# boxprobe — {cfg.stage} / {tag} ({time.strftime('%Y-%m-%d')})",
        "",
        f"σ band [{cfg.band[0]}, {cfg.band[1]}], {len(reads)} reads over "
        f"{len({r['i'] for r in reads})} scene items; curves: flat {s1:g}, "
        f"linear min({s1:g}·n, 0.75), log {s1:g} → {s_cap:g} at {n_cap:g} glyphs.",
        "",
        "`s*` = share at which the in-box and out-of-box row gradients have equal "
        "magnitude; `φ` = in-box fraction of the resolved row gradient under each curve. "
        "Medians per bin; `cos` = cos(g_in, g_out).",
        "",
        "| glyphs | n | px | rows | ‖g_in‖ | ‖g_out‖ | ratio | cos | s* | "
        "s flat / lin / log | φ flat | φ lin | φ log |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    order = sorted(by, key=lambda b: int(b.split("–")[0].rstrip("+")))
    for b in order:
        rs = by[b]
        n_med = int(med([r["n_glyphs"] for r in rs]))
        s_of = {k: c(n_med) for k, c in curves.items()}
        phi = {
            k: med([_phi(s_of[k], r["g_in"], r["g_out"]) for r in rs]) for k in curves
        }
        lines.append(
            f"| {b} | {len(rs)} | {med([r['px'] or 0 for r in rs]):.0f} | "
            f"{med([r['rows'] for r in rs]):.0f} | {med([r['g_in'] for r in rs]):.3g} | "
            f"{med([r['g_out'] for r in rs]):.3g} | {med([r['ratio'] for r in rs]):.2f} | "
            f"{med([r['cos'] for r in rs]):+.2f} | {med([r['s_star'] for r in rs]):.2f} | "
            f"{s_of['flat']:.2f} / {s_of['linear']:.2f} / {s_of['log']:.2f} | "
            f"{phi['flat']:.2f} | {phi['linear']:.2f} | {phi['log']:.2f} |"
        )
    # σ split inside the band
    lo, hi = cfg.band
    mid = (lo + hi) / 2
    lines += [
        "",
        "By σ half (s* medians per glyph bin):",
        "",
        "| glyphs | σ < mid | σ ≥ mid |",
        "|---|---|---|",
    ]
    for b in order:
        rs = by[b]
        a = [r["s_star"] for r in rs if r["sigma"] < mid]
        c = [r["s_star"] for r in rs if r["sigma"] >= mid]
        lines.append(
            f"| {b} | {med(a) if a else float('nan'):.2f} ({len(a)}) | "
            f"{med(c) if c else float('nan'):.2f} ({len(c)}) |"
        )
    lines += [
        "",
        "Reading: a curve whose `φ` is flat across the bins keeps the in-box term's "
        "share of the row gradient independent of the glyph count; `s*` is the "
        "measured share that would make it exactly one half. `cos` near 0 says the "
        "background gradient is noise on the rows, positive says it already pulls "
        "the same way (harmless), negative says it fights the box.",
        "",
        f"Reads: `{out / 'reads.jsonl'}`.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
