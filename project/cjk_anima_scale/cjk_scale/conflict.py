"""conflict — do a run's band groups pull a row the same way? Read without training.

A row is its own parameter: the only thing stage B can do to what stage A
bought is push the same row somewhere else. So "chain the bands" vs "one run
with per-item bands" is a per-row question about the two data gradients,
and it can be read at one table point with no optimizer step — the
boxprobe primitive with a different split.

``scale.py <run> conflict`` (2026-09-25: the run's own data, no flags). A
"stage" below is one of the run's **band groups** (``builder.TABLE``:
``b0709`` / ``b0507`` / ``b0305`` — the old stages' data, now one dir with a
band per item). For every group, on a sample of its items, this draws σ
inside the group's band, backprops the *trained* loss (the box-share FM
loss on scene items and — ``train.GRID_BOX`` — grid items' cells, plain MSE
on flat) onto the seed table's rows, and accumulates per row, per group, the mean per-draw
gradient ``ḡ_S`` (two halves by item parity, for a split-half reliability).
Per row it then reports:

- ``n``, ``‖ḡ‖`` (per-draw norm — an ex-ante exposure: a row the band
  barely moves will not move at any step count), ``coh`` = ‖Σg‖ / Σ‖g‖
  (are the draws pulling one way), ``half`` = cos of the two halves (the
  noise floor every cross-stage cosine is read against);
- per stage pair, ``cos(ḡ_A, ḡ_B)`` and the cancellation
  ``‖ḡ_A + ḡ_B‖ / (‖ḡ_A‖ + ‖ḡ_B‖)`` — what a joint run's summed pull keeps;
- if the run's ``trained.pt`` exists, ``cos(Δ, −ḡ_B)`` with ``Δ`` = what the
  run moved the row by (its table minus the seed), keyed on the first group
  (``Δ<first>·−g<B>``): negative says B's descent runs against what the run
  bought.

Reading: ``cos`` at or below ``−half`` on the rows both stages touch → the
bands fight over the row, the anchor is a truce and no per-stage budget
fixes it; ``cos`` inside ``±half`` → orthogonal, the chain is free either
way and one mixed run with per-item bands is the same thing; ``cos`` at or
above ``half`` → they agree, sequencing is unnecessary. Gradients are taken
at one table point, the seed table.

Outputs ``<run>/conflict/{rows.json, report.md}``. The math below the data
plumbing is the 2026-09-25 probe's, unchanged
(``reports/conflict_joint_2026_09_25.md``).
"""

from __future__ import annotations

import json
import random
import statistics as st
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .config import RunConfig
from .paths import SEED_TABLE, data_dir, run_dir, table_path
from .rows import RowTable

KINDS = ("single", "piece", "multi")


def probe_dir(run: str) -> Path:
    return run_dir(run) / "conflict"


def _load_table(path: Path, ext_ids: list[int], row_scale: float) -> torch.Tensor:
    """A ``trained.pt``'s rows in this table's units, aligned to ``ext_ids``
    (rows it lacks are zero)."""
    src = torch.load(path, map_location="cpu", weights_only=False)
    raw = src["delta"]["raw"].float()
    idx = {int(e): i for i, e in enumerate(src["delta"]["ext_ids"])}
    rs = src["delta"].get("row_scale")
    k = float(rs) / row_scale if rs is not None else 1.0
    out = torch.zeros(len(ext_ids), raw.shape[1])
    for i, e in enumerate(ext_ids):
        j = idx.get(int(e))
        if j is not None:
            out[i] = raw[j] * k
    return out


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=0))


def spec_key(stage: str, band) -> str:
    return stage if band is None else f"{stage}@{band[0]:g}-{band[1]:g}"


def probe(
    rc: RunConfig,
    *,
    items: int = 600,
    draws: int = 1,
    seed: int = 0,
) -> Path:
    """Per band group of the run's data, ``items`` items × ``draws`` σ draws
    (the 2026-09-25 reads: 600 × 1). Per (group, recipe) reads go in the
    report's price table."""
    from common.models import checkpoints, dit_forward, gen_args
    from data.inventory import qwen_pieces
    from library.anima.vocab_pack import attached_pack_rows, strategy_pack
    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model
    from library.inference.text import ensure_text_strategies
    from library.runtime.noise import fm_training_batch
    from train.stage import LatentStore, _encode_text

    from . import train as T
    from .loss import box_share_fm_loss
    from .train import load_items, vocab_idx

    tag = rc.name
    warm = SEED_TABLE
    recipes = None
    data = data_dir(rc.name)
    all_recs, ev, vocabs = load_items(data)
    band_of: dict[str, tuple] = {}
    by_group: dict[str, list] = defaultdict(list)
    for i, r in enumerate(all_recs):
        g = r.get("group") or f"{r['band'][0]:g}-{r['band'][1]:g}"
        by_group[g].append(i)
        band_of.setdefault(g, tuple(r["band"]))
        assert band_of[g] == tuple(r["band"]), f"group {g}: two bands"
    stages = [g for g in ("b0709", "b0507", "b0305") if g in by_group] + sorted(
        g for g in by_group if g not in ("b0709", "b0507", "b0305")
    )
    assert len(stages) >= 2, (
        f"{rc.name}: a conflict needs two band groups, got {stages}"
    )
    out = probe_dir(rc.name)
    out.mkdir(parents=True, exist_ok=True)
    args = gen_args(512, T.GEN_STEPS, 4.0, out)
    device = get_generation_settings(args).device
    rng = random.Random(seed)

    # -- captions, latents, the union table -----------------------------------
    cache, train_ext, ev_ext = _encode_text(
        all_recs, ev, device, out, te_cache=data / "te_cache"
    )
    table_ext: set[int] = train_ext | vocab_idx(vocabs, qwen_pieces())
    label: dict[int, str] = {}
    for text, ids in ev_ext.items():
        if len(ids) == 1:
            label.setdefault(int(ids[0]), text)
    ns = SimpleNamespace(
        seed=seed, batch=1, train_size=512, row_blocks=0, row_boost="", arm="rows"
    )
    lat = LatentStore(ns, data, all_recs, list(range(len(all_recs))), device)
    per: dict[str, dict] = {}
    for s in stages:
        pool = by_group[s]
        pick = sorted(rng.sample(pool, min(items, len(pool)))) if items else pool
        per[s] = {"recs": all_recs, "cache": cache, "pick": pick, "lat": lat}
        print(
            f"conflict {s}: σ {list(band_of[s])}, {len(pick)} of {len(pool)} items × {draws} draws",
            flush=True,
        )

    anima = load_dit_model(args, device, torch.bfloat16)
    anima.requires_grad_(False)
    assert attached_pack_rows(anima), "no vocab pack attached to the DiT"
    tok, _ = ensure_text_strategies(checkpoints().text_encoder, vocab_pack=None)
    rows = RowTable(
        anima,
        device,
        table_ext,
        strategy_pack(tok),
        warm=warm,
        init_anchor=0.0,
        free_residual=0.0,
        lr=0.0,
    )
    raw = rows.delta.raw
    ext_ids = [int(e) for e in rows.delta.ext_ids]
    R, D = raw.shape
    anima.train()

    # -- accumulate -------------------------------------------------------------
    G = {s: torch.zeros(2, R, D, device=device) for s in stages}  # Σg per half
    N = {s: torch.zeros(R, device=device) for s in stages}  # draws touching the row
    S = {s: torch.zeros(R, device=device) for s in stages}  # Σ‖g_row‖
    GR: dict = {}  # (key, recipe) → [Σg half0, Σg half1, N, Σ‖g‖]
    kind_of: dict[int, Counter] = defaultdict(Counter)
    g = torch.Generator(device=device).manual_seed(seed)
    t0 = time.time()
    n_reads = 0
    for s in stages:
        p = per[s]
        bs_cfg, cap, n_cap = T.BOX_SHARE, T.BOX_SHARE_CAP, float(T.BOX_SHARE_GLYPHS)
        gbox = bool(T.GRID_BOX)
        t_min, t_max = band_of[s]
        for k, i in enumerate(p["pick"]):
            r = p["recs"][i]
            latents = p["lat"][[i]].to(device)
            for _ in range(draws):
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
                    pred = dit_forward(
                        anima, noisy, ts, p["cache"], [r["caption"]], device
                    )
                bs = (
                    bs_cfg
                    if r["src"] == "scene" or (gbox and r["src"] == "grid")
                    else 0.0
                )
                loss = box_share_fm_loss(pred, target, [r], bs, cap, n_cap, gbox)
                (gr,) = torch.autograd.grad(loss, raw)
                nrm = gr.norm(dim=1)
                touched = nrm > 0
                G[s][k % 2] += gr
                N[s] += touched.float()
                S[s] += nrm
                rk = (s, r.get("recipe") or "?")
                if rk not in GR:
                    GR[rk] = [
                        torch.zeros(R, D, device=device),
                        torch.zeros(R, D, device=device),
                        torch.zeros(R, device=device),
                        torch.zeros(R, device=device),
                    ]
                GR[rk][k % 2] += gr
                GR[rk][2] += touched.float()
                GR[rk][3] += nrm
                lk = r.get("law_kind") or r.get("kind") or "?"
                for j in torch.nonzero(touched).flatten().tolist():
                    kind_of[j][lk] += 1
                n_reads += 1
            if (k + 1) % 100 == 0 or k + 1 == len(p["pick"]):
                print(
                    f"  {s}: {k + 1}/{len(p['pick'])} items, {(time.time() - t0) / 60:.1f} min",
                    flush=True,
                )
    print(f"conflict: {n_reads} reads in {(time.time() - t0) / 60:.1f} min", flush=True)

    # -- what the run moved (Δ, keyed on the first group) -----------------------
    delta: dict[str, torch.Tensor] = {}
    tp = table_path(rc.name)
    if tp.exists():
        delta[stages[0]] = _load_table(tp, ext_ids, rows.row_scale) - _load_table(
            warm, ext_ids, rows.row_scale
        )
        print(f"Δ (as {stages[0]}): {tp} − {warm}", flush=True)

    # -- per row ------------------------------------------------------------------
    Gc = {s: G[s].cpu() for s in stages}
    Nc = {s: N[s].cpu() for s in stages}
    Sc = {s: S[s].cpu() for s in stages}
    gbar = {s: (Gc[s][0] + Gc[s][1]) / Nc[s].clamp(min=1)[:, None] for s in stages}
    pairs = [(a, b) for i, a in enumerate(stages) for b in stages[i + 1 :]]
    min_n = 8
    reads = []
    for j, e in enumerate(ext_ids):
        kc = kind_of.get(j)
        rec = {
            "ext": e,
            "label": label.get(e, ""),
            "kind": kc.most_common(1)[0][0] if kc else "",
            "stage": {},
            "pair": {},
            "delta": {},
        }
        for s in stages:
            n = int(Nc[s][j])
            if n == 0:
                continue
            gsum = Gc[s][0][j] + Gc[s][1][j]
            h0, h1 = Gc[s][0][j], Gc[s][1][j]
            rec["stage"][s] = {
                "n": n,
                "g": float(Sc[s][j]) / n,
                "coh": float(gsum.norm()) / max(float(Sc[s][j]), 1e-30),
                "half": _cos(h0, h1) if (h0.norm() > 0 and h1.norm() > 0) else None,
            }
        for a, b in pairs:
            if int(Nc[a][j]) >= min_n and int(Nc[b][j]) >= min_n:
                ga, gb = gbar[a][j], gbar[b][j]
                rec["pair"][f"{a}|{b}"] = {
                    "cos": _cos(ga, gb),
                    "keep": float((ga + gb).norm())
                    / max(float(ga.norm() + gb.norm()), 1e-30),
                }
        for a in stages:
            if a not in delta or float(delta[a][j].norm()) == 0:
                continue
            for b in stages:
                if int(Nc[b][j]) >= min_n:
                    rec["delta"][f"{a}|{b}"] = _cos(delta[a][j], -gbar[b][j])
        if rec["stage"]:
            reads.append(rec)
    by_recipe = []
    for (s, rcp), (h0, h1, n_, s_) in GR.items():
        h0, h1, n_, s_ = h0.cpu(), h1.cpu(), n_.cpu(), s_.cpu()
        for j, e in enumerate(ext_ids):
            n = int(n_[j])
            if n < min_n:
                continue
            gsum = h0[j] + h1[j]
            kc = kind_of.get(j)
            by_recipe.append(
                {
                    "stage": s,
                    "recipe": rcp,
                    "ext": e,
                    "label": label.get(e, ""),
                    "kind": kc.most_common(1)[0][0] if kc else "",
                    "n": n,
                    "g": float(s_[j]) / n,
                    "coh": float(gsum.norm()) / max(float(s_[j]), 1e-30),
                    "half": _cos(h0[j], h1[j])
                    if (h0[j].norm() > 0 and h1[j].norm() > 0)
                    else None,
                }
            )
    (out / "rows.json").write_text(
        json.dumps(
            {
                "stages": stages,
                "tag": tag,
                "run": rc.name,
                "warm": str(warm),
                "items": items,
                "draws": draws,
                "seed": seed,
                "min_n": min_n,
                "bands": {s: list(b) for s, b in band_of.items()},
                "recipes": recipes,
                "rows": reads,
                "by_recipe": by_recipe,
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    report(stages, tag, warm, reads, pairs, delta.keys(), out, by_recipe, band_of)
    del anima
    torch.cuda.empty_cache()
    return out


def _med(xs):
    xs = [x for x in xs if x is not None]
    return f"{st.median(xs):+.2f}" if xs else "–"


def report(
    stages, tag, warm, reads, pairs, deltas, out: Path, by_recipe=(), band_of=None
) -> None:
    short = {s: s.replace("stage", "") for s in stages}
    band_of = band_of or {}
    pcols = [f"{short[a]}×{short[b]}" for a, b in pairs]
    dcols = [
        (a, b, f"Δ{short[a]}·−g{short[b]}")
        for a in stages
        if a in deltas
        for b in stages
        if b != a
    ]
    lines = [
        f"# conflict — {tag} ({time.strftime('%Y-%m-%d')})",
        "",
        f"Stages {' → '.join(stages)}"
        + (
            " (σ "
            + ", ".join(f"{short[s]} {list(b)}" for s, b in band_of.items())
            + ")"
            if band_of
            else ""
        )
        + f"; gradients at `{warm}`; {len(reads)} rows.",
        "",
        "Per stage: `n` draws touching the row, `‖ḡ‖` mean per-draw gradient norm, "
        "`coh` ‖Σg‖/Σ‖g‖, `half` split-half cos (the noise floor). Per pair: "
        "cos(ḡ_A, ḡ_B) / `keep` ‖ḡ_A+ḡ_B‖/(‖ḡ_A‖+‖ḡ_B‖) on rows both touch. "
        "`Δa·−gb`: cos of what stage a moved the row by against stage b's descent "
        "(negative = b undoes a).",
        "",
        "| row | kind | "
        + " | ".join(f"{short[s]} n / ‖ḡ‖ / coh / half" for s in stages)
        + " | "
        + " | ".join(pcols)
        + (" | " + " | ".join(c for _, _, c in dcols) if dcols else "")
        + " |",
        "|---|---|" + "---|" * (len(stages) + len(pcols) + len(dcols)),
    ]
    order = {k: i for i, k in enumerate(KINDS)}
    for r in sorted(reads, key=lambda r: (order.get(r["kind"], 9), r["label"])):
        cells = []
        for s in stages:
            x = r["stage"].get(s)
            cells.append(
                "–"
                if x is None
                else f"{x['n']} / {x['g']:.3g} / {x['coh']:.2f} / "
                + ("–" if x["half"] is None else f"{x['half']:+.2f}")
            )
        for a, b in pairs:
            x = r["pair"].get(f"{a}|{b}")
            cells.append("–" if x is None else f"{x['cos']:+.2f} / {x['keep']:.2f}")
        for a, b, _ in dcols:
            x = r["delta"].get(f"{a}|{b}")
            cells.append("–" if x is None else f"{x:+.2f}")
        lines.append(
            f"| {r['label'] or r['ext']} | {r['kind']} | " + " | ".join(cells) + " |"
        )

    lines += ["", "## By kind (medians)", ""]
    lines += [
        "| kind | rows | "
        + " | ".join(f"{short[s]} ‖ḡ‖ / coh / half" for s in stages)
        + " | "
        + " | ".join(f"{c} cos / keep" for c in pcols)
        + (" | " + " | ".join(c for _, _, c in dcols) if dcols else "")
        + " |",
        "|---|---|" + "---|" * (len(stages) + len(pcols) + len(dcols)),
    ]
    for kind in KINDS:
        rs = [r for r in reads if r["kind"] == kind]
        if not rs:
            continue
        cells = []
        for s in stages:
            xs = [r["stage"][s] for r in rs if s in r["stage"]]
            cells.append(
                "–"
                if not xs
                else f"{st.median([x['g'] for x in xs]):.3g} / "
                f"{st.median([x['coh'] for x in xs]):.2f} / "
                f"{_med([x['half'] for x in xs])} ({len(xs)})"
            )
        for a, b in pairs:
            xs = [r["pair"][f"{a}|{b}"] for r in rs if f"{a}|{b}" in r["pair"]]
            cells.append(
                "–"
                if not xs
                else f"{_med([x['cos'] for x in xs])} / "
                f"{st.median([x['keep'] for x in xs]):.2f} ({len(xs)})"
            )
        for a, b, _ in dcols:
            xs = [r["delta"][f"{a}|{b}"] for r in rs if f"{a}|{b}" in r["delta"]]
            cells.append("–" if not xs else f"{_med(xs)} ({len(xs)})")
        lines.append(f"| {kind} | {len(rs)} | " + " | ".join(cells) + " |")
    if by_recipe:
        lines += [
            "",
            "## By recipe (medians over rows with n ≥ min_n) — the price table",
            "",
            "`‖ḡ‖·coh` = coherent movement per draw: what one draw of this recipe at this "
            "band buys the row, noise removed.",
            "",
            "| stage | recipe | kind | rows | n | ‖ḡ‖ | coh | half | ‖ḡ‖·coh |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        groups: dict = defaultdict(list)
        for r in by_recipe:
            groups[(r["stage"], r["recipe"], r["kind"])].append(r)
        order = {k: i for i, k in enumerate(KINDS)}
        for (s, rcp, kind), rs in sorted(
            groups.items(),
            key=lambda kv: (stages.index(kv[0][0]), kv[0][1], order.get(kv[0][2], 9)),
        ):
            g = st.median([r["g"] for r in rs])
            coh = st.median([r["coh"] for r in rs])
            lines.append(
                f"| {short[s]} | {rcp} | {kind} | {len(rs)} | {st.median([r['n'] for r in rs]):.0f} | "
                f"{g:.3g} | {coh:.2f} | {_med([r['half'] for r in rs])} | {g * coh:.3g} |"
            )
    lines += [
        "",
        "Reading: a pair's cos at or below −`half` on the rows both stages touch means the "
        "bands fight over the row (the anchor is a truce, no budget fixes it); inside "
        "±`half` they are orthogonal (chain and joint per-item-band run are the same "
        "thing); at or above `half` they agree (sequencing buys nothing). `keep` is the "
        "fraction of the summed pull a joint run retains; `‖ḡ‖` is the ex-ante exposure "
        "a band gives the row per draw.",
        "",
        f"Rows: `{out / 'rows.json'}`.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
