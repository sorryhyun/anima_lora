"""trainer — the run's rows, plain FM, σ drawn per item inside its band.

Ported from the scale line's ``cjk_scale/train.py`` (2026-10-09). The
numerics are scale's; what changed is the plumbing: the run is a reseed
``Run`` (its name, path and rows), the caller gives ``data``, ``out``,
``context``, ``cold`` and the steps (``steps_per_row`` or ``steps``), and
the scale line's step budget (``budget`` → ``builder`` → ``recipes``) is gone
with the ``budget_factor`` / ``mix_factor`` it wrote into
``train_record.json``.

The stages' ``train/stage.py`` rows-arm path with the levers left behind
(design § 5): no pair loss, no counterfactual input, no ``c_flat``, no
out-vec, no row blocks / boosts, no encoder. What stays is exactly what the
reads of record used — the box-share FM loss (scene items, and grid items'
cells under ``GRID_BOX``), cosine decay with warmup, the seed as the warm
start and the frozen context.

What trains is the run's vocabs' rows (their idx through the Qwen
tokenizer, from ``vocabs.json``), warm from the seed rows; every other row a
training caption touches rides frozen at its seed value. ``trained.pt`` is
the whole merged rows — the seed's rows with the run's on top
(``rows.Rows.state_dict``). A vocab the seed lacks starts cold (``Rows``
prints it).

Reused from the stage packages, unchanged: ``LatentStore`` (per-shape latent
cache in the data dir), ``Batcher`` (one shape × one source per batch),
``BoxSplit`` (in / out split logging), ``_encode_text`` (TE cache +
``eval_coverage.json``), ``dit_forward``.

Order (lazy loading): captions → TE cache → VAE latents → DiT → rows →
compile → loop → ``<run>/trained.pt``.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from . import PUNCT_PACK_SHA, RAW_PACK_SHA
from .rows import Rows

# The trainer is fixed (scale's plan.md § 2); a different trainer is a code change with
# a new report beside it, never a flag. Each value names the read that set it.
INIT_ANCHOR = (
    0.0  # μ 0: the vocabs need the displacement; the rest is frozen, not anchored
)
#                    (reports/conflict_joint_2026_09_25.md § 4 verdict 3; run0925_300f)
LR = 1e-3  # the lr that moved pieces (micro_warm_0923; conflict_joint report § 3)
BATCH = 4
LR_DECAY = "cosine"
WARMUP_RATIO = 0.1  # of the run's steps (micro_warm_0923: 100 / 640, 200 / 1280)
GRID_BOX = True  # grid cells' union as the loss box (reports/grid_box_2026_09_25.md)
BOX_SHARE = 0.25  # a single glyph's in-box share, log up to …
BOX_SHARE_CAP = 0.5  # … the cap, at …
BOX_SHARE_GLYPHS = 8  # … this many glyphs (loss.py; every stage file of record)
FREE_RESIDUAL = (
    1e-3  # μ‖f‖² on the touched rows when there is no anchor (a constant, not a lever)
)
SEED = 0
COMPILE = True
SAVE_EVERY = 5000
PRES_SEED = 7_000  # the pres pass's generator, apart from the data term's draws
GEN_STEPS, GEN_CFG = 28, 4.0  # the generation settings the stage helpers want


def vocab_idx(vocabs: list, tokq) -> set[int]:
    """The run's vocabs → their idx: the Qwen tokenizer maps each vocab to
    its pieces, every piece with a pack row trains (scale's plan.md § 4:
    the vocabs file is the inventory, the tokenizer maps it, done)."""
    from data.inventory import pieces as qpieces

    tok, qmap = tokq
    return {int(e) for v in vocabs for _p, e in qpieces(tok, qmap, v) if e is not None}


def noisy_by_band(latents, noise, bands, device):
    """``fm_training_batch`` with σ drawn per item inside its own band: the
    batch is split by band, each part drawn with its ``t_min`` / ``t_max``,
    and the parts put back in batch order."""
    from library.runtime.noise import fm_training_batch

    groups: dict = {}
    for k, b in enumerate(bands):
        groups.setdefault((float(b[0]), float(b[1])), []).append(k)
    if len(groups) == 1:
        (lo, hi), _ = next(iter(groups.items()))
        return fm_training_batch(
            latents, noise, dtype=torch.bfloat16, device=device, t_min=lo, t_max=hi
        )
    B = latents.shape[0]
    noisy = [None] * B
    ts = [None] * B
    target = [None] * B
    for (lo, hi), ks in groups.items():
        n_, t_, g_ = fm_training_batch(
            latents[ks],
            noise[ks],
            dtype=torch.bfloat16,
            device=device,
            t_min=lo,
            t_max=hi,
        )
        for j, k in enumerate(ks):
            noisy[k], ts[k], target[k] = n_[j], t_[j], g_[j]
    return torch.stack(noisy), torch.stack(ts), torch.stack(target)


def load_items(data: Path) -> tuple[list, list, list]:
    """``train.jsonl`` / ``eval.json`` / ``vocabs.json`` of a run's data dir;
    every item must carry its band (stamped at build time)."""
    assert (data / "train.jsonl").exists(), (
        f"no data dir {data} — run `run.py <run> data` first"
    )
    recs = [
        json.loads(ln)
        for ln in (data / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if ln
    ]
    assert all("shape" in r for r in recs), "every record needs a shape"
    assert all(r.get("band") for r in recs), "every record needs its band (builder)"
    ev = json.loads((data / "eval.json").read_text(encoding="utf-8"))
    vocabs = json.loads((data / "vocabs.json").read_text(encoding="utf-8"))
    return recs, ev, vocabs


def plan(
    run,
    data: Path,
    recs: list,
    vocabs: list,
    touched: set,
    ctx: Path,
    steps_per_row: int | None = None,
    steps: int | None = None,
):
    """Everything ``train`` fixes before the model loads: the rows split
    (what trains, what rides frozen) and the schedule, with the record
    ``train_record.json`` carries. CPU only — the Qwen tokenizer and the
    pack's mapping; ``touched`` = the ext rows the training captions carry
    (``ext_ids_of`` over the TE cache); ``ctx`` = the rows the run sits on.
    The steps are ``steps_per_row`` × the trained rows, or ``steps``."""
    from data.inventory import qwen_pieces

    assert bool(steps_per_row) != bool(steps), "steps_per_row or steps, one of them"
    tokq = qwen_pieces(char_rows=True)
    idx = vocab_idx(vocabs, tokq)
    # captions carry rows outside the vocabs (corpus lines): they ride frozen
    # at the context; what trains (and what the steps count) is the vocabs' rows
    frozen = touched - idx
    touched = touched & idx
    steps = int(steps) if steps else int(steps_per_row) * len(idx)
    steps_per_row = int(steps_per_row) if steps_per_row else round(steps / len(idx), 2)
    warmup = int(round(WARMUP_RATIO * steps))
    bands = sorted({tuple(r["band"]) for r in recs})
    record = {
        "run": run.name,
        "run_config": str(run.path),
        "vocabs": list(run.rows),
        "data": str(data),
        "bands": [list(b) for b in bands],
        "train_steps": steps,
        "steps_per_row": steps_per_row,
        "lr_warmup": warmup,
        "lr_warmup_ratio": WARMUP_RATIO,
        "lr_rows": LR,
        "lr_decay": LR_DECAY,
        "init_anchor": INIT_ANCHOR,
        "free_residual": FREE_RESIDUAL,
        "batch": BATCH,
        "box_share": BOX_SHARE,
        "box_share_cap": BOX_SHARE_CAP,
        "box_share_glyphs": BOX_SHARE_GLYPHS,
        "grid_box": int(GRID_BOX),
        "seed": SEED,
        "n_rows": len(idx),
        "n_touched": len(touched),
        "context": str(ctx),
        "n_context": len(frozen),
        "arm": "rows",
    }
    return SimpleNamespace(
        idx=idx,
        touched=touched,
        frozen=frozen,
        steps=steps,
        steps_per_row=steps_per_row,
        warmup=warmup,
        bands=bands,
        record=record,
    )


def train(
    run,
    *,
    data: Path,
    out: Path,
    context: Path,
    cold: bool,
    max_steps: int | None = None,
    steps_per_row: int | None = None,
    steps: int | None = None,
    row_step_scale: dict | None = None,
    lr: float | None = None,
    free_residual: float | None = None,
    pres: tuple | None = None,
    factor: str = "",
) -> Path:
    """Train ``run`` (a ``reseed.config.Run``: its name, path and rows) on
    ``data`` into ``out``. ``context`` is the warm-from / frozen-context /
    merge file (the run's seed rows); ``cold`` starts the trained rows at the
    pack rows (Δ 0) instead of ``context``'s; the steps are
    ``steps_per_row`` × the trained rows, or ``steps`` (a run whose step count
    is set by some of its rows, ``focus``) — one of them;
    ``max_steps`` stops the loop early with the full-length schedule;
    ``row_step_scale`` = {vocab:
    factor} multiplies that vocab's rows' update each step — a per-row lr
    (AdamW normalizes a gradient scale away, so the step is scaled, not the
    gradient; ``experiments/garble_replace`` inverse frequency);
    ``lr`` replaces ``LR`` (the rows' peak lr);
    ``free_residual`` replaces ``FREE_RESIDUAL`` (0: no norm pull — under
    AdamW the pull alone steps a row absent from the batch by ~lr toward 0,
    so a warm run's rare rows go back to the pack row, ``cjk_anima_reseed``
    ``sent_kanji``);
    ``pres`` = (λ, σ_lo, σ_hi, every) adds λ · L_pres (``loss.pres_loss``,
    ``cjk_anima_reseed`` probe_pres_train) on every ``every``-th step's scene
    batch: the student under the item's caption against the same DiT under
    its EN caption (``loss.en_caption``, detached), at one shared x_σ with
    σ ~ U(σ_lo, σ_hi), outside the dilated text box. The pass draws σ and ε
    from its own generator, so the data term's draws stay the run's without
    it; the EN captions are TE-cached in ``out/te_en``.
    A data dir built with windows (``build.json`` ``glyph_route``) is
    trained routed: ``ANIMA_VOCAB_GLYPH_ROUTE=1`` is set in-process before
    the TE cache (whose key carries it); ``glyph_route_ko`` likewise sets
    ``ANIMA_VOCAB_GLYPH_ROUTE_KO=1`` (Hangul pieces per syllable).
    ``factor = "jamo"`` (``jamo.Jamo``): the trained rows are Hangul
    syllables composed from jamo factors each step; ``trained.pt`` carries
    every syllable's composed row and the factors (``jamo``)."""
    from common.models import checkpoints, dit_forward, gen_args
    from library.anima.ext_vocab import pack_digest
    from library.anima.vocab_pack import attached_pack_rows, strategy_pack
    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model
    from library.inference.text import ensure_text_strategies
    from train.stage import Batcher, BoxSplit, LatentStore, _encode_text

    from .loss import box_share_fm_loss

    torch.manual_seed(SEED)
    out.mkdir(parents=True, exist_ok=True)
    recs, ev, vocabs = load_items(data)
    bj = data / "build.json"
    build = json.loads(bj.read_text(encoding="utf-8")) if bj.exists() else {}
    route, route_ko = (
        build.get("glyph_route", False),
        build.get("glyph_route_ko", False),
    )
    if route:
        os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
    if route_ko:
        os.environ["ANIMA_VOCAB_GLYPH_ROUTE_KO"] = "1"
    args = gen_args(512, GEN_STEPS, GEN_CFG, out)
    device = get_generation_settings(args).device

    cache, touched, _ev_idx = _encode_text(
        recs, ev, device, out, te_cache=data / "te_cache"
    )
    en_of = None
    if pres:
        from common.models import encode_captions, ext_ids_of

        from .loss import en_caption

        lam_p, lo_p, hi_p, every_p = (
            float(pres[0]),
            float(pres[1]),
            float(pres[2]),
            int(pres[3]),
        )
        assert lam_p > 0 and 0 <= lo_p < hi_p < 1 and every_p >= 1, f"pres {pres}"
        en_of = {i: en_caption(r, i) for i, r in enumerate(recs) if r["src"] == "scene"}
        en_cache = encode_captions(list(en_of.values()), device, out / "te_en")
        # a prompt's own marks may route to a frozen row (``~`` in a series
        # tag → the 〜 row, on both captions alike); a trained row may not
        en_ext = ext_ids_of(en_cache)
        cache = {**cache, **en_cache}
        print(
            f"pres: λ {lam_p:g} · L_pres at σ {lo_p:g}–{hi_p:g} every {every_p} "
            f"step(s); {len(en_cache)} EN captions for {len(en_of)} scene items",
            flush=True,
        )
    ctx = Path(context)
    p = plan(run, data, recs, vocabs, touched, ctx, steps_per_row, steps)
    if pres:
        hit = en_ext & set(p.idx)
        assert not hit, f"pres: EN captions hold trained rows {sorted(hit)[:10]}"
        if en_ext:
            print(f"pres: EN captions hold frozen rows {sorted(en_ext)}", flush=True)
    if route:
        p.record["glyph_route"] = True
    if route_ko:
        p.record["glyph_route_ko"] = True
    lr = lr or LR
    if lr != LR:
        p.record.update(lr_rows=lr, lr_override=True)
    print(
        f"rows: {len(p.idx)} ({len(vocabs)} vocabs) — {len(p.touched)} touched "
        f"by the captions, {len(p.idx - p.touched)} with no draw; {len(p.frozen)} "
        f"context rows frozen at {ctx}",
        flush=True,
    )
    ns = SimpleNamespace(seed=SEED, batch=BATCH, train_size=512)
    lat = LatentStore(ns, data, recs, list(range(len(recs))), device)

    anima = load_dit_model(args, device, torch.bfloat16)
    anima.requires_grad_(False)
    assert attached_pack_rows(anima), "no vocab pack attached to the DiT"
    tok, _ = ensure_text_strategies(checkpoints().text_encoder, vocab_pack=None)
    pack = strategy_pack(tok)
    # the rows and ids must be the raw pack's (or the punct pack's: the raw
    # pack's plus an appended row); its encode fold (plan_retrain § 2c) is an
    # encode rule on top, outside this check
    raw_sha = (
        pack_digest(pack.table, {k: v for k, v in pack.mapping.items() if k != "fold"})
        if pack is not None
        else ""
    )
    assert raw_sha.startswith((RAW_PACK_SHA, PUNCT_PACK_SHA)), (
        f"attached pack {getattr(pack, 'name', None)} (sha without fold "
        f"{raw_sha[:12]}…) is not the raw pack ({RAW_PACK_SHA}…) or the punct "
        f"pack ({PUNCT_PACK_SHA}…): "
        "rows are deltas over it and cold rows start at it — "
        "ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack"
    )
    if cold:
        p.record["cold"] = True
    if free_residual is not None:
        p.record.update(free_residual=float(free_residual), free_residual_override=True)
    rows = Rows(
        anima,
        device,
        p.idx,
        pack,
        warm=None if cold else ctx,
        init_anchor=INIT_ANCHOR,
        free_residual=FREE_RESIDUAL if free_residual is None else free_residual,
        lr=lr,
        touched=p.touched,
        frozen=p.frozen,
        context=ctx,
    )
    assert rows.n_rows == len(p.idx), (rows.n_rows, len(p.idx))
    jamo = None
    if factor:
        from .jamo import Jamo, syllable_rows

        assert factor == "jamo", factor
        assert not row_step_scale, "factor rows take no row_lr"
        assert cold, "factor rows start cold (Δ 0)"
        syl_ext = syllable_rows()
        ext_syl = {e: s for s, e in syl_ext.items()}
        jamo = Jamo(rows, {e: ext_syl[e] for e in p.idx}, lr)
        p.record.update(factor=factor, factor_vectors=Jamo.N_VECTORS)
        print(
            f"factor: {len(p.idx)} syllable rows composed from {Jamo.N_VECTORS} "
            f"jamo vectors (b + C[cho, cls] + V + F); trained.pt carries all "
            f"{len(syl_ext)} syllables' rows",
            flush=True,
        )

    def save(path: Path, at: int | None) -> None:
        if jamo is None:
            torch.save(rows.state_dict(record, at), path)
            return
        with torch.no_grad():
            jamo.apply()
        sd = rows.state_dict(record, at)
        n = jamo.merge_all(sd, syl_ext)
        sd["jamo"] = {**jamo.state(), "composed_added": n}
        torch.save(sd, path)

    if pres:
        from .loss import PRES_DIL

        p.record["pres"] = {
            "lam": lam_p,
            "band": [lo_p, hi_p],
            "every": every_p,
            "dil": PRES_DIL,
            "seed": PRES_SEED,
        }
    steps, warmup, record = p.steps, p.warmup, p.record
    spr = (
        p.steps_per_row
        if not isinstance(p.steps_per_row, dict)
        else " + ".join(f"{n} × {k}" for k, n in p.steps_per_row.items())
    )
    print(
        f"train {run.name}: σ per item in {p.bands}, {rows.n_rows} rows, {steps} steps "
        f"({spr}/row) × batch {BATCH}, lr {lr:g} {LR_DECAY} warmup {warmup} "
        f"({WARMUP_RATIO:g}), μ {INIT_ANCHOR:g}, box_share {BOX_SHARE} → cap "
        f"{BOX_SHARE_CAP} at {BOX_SHARE_GLYPHS} glyphs (log), grid_box {int(GRID_BOX)}, "
        + ("cold (pack rows)" if cold else f"warm {ctx}")
        + (f"; stopping at step {max_steps}" if max_steps else ""),
        flush=True,
    )
    opt = torch.optim.AdamW(rows.params, weight_decay=0.0, betas=(0.9, 0.99))
    step_scale = None
    if row_step_scale:
        from data.inventory import pieces as qpieces
        from data.inventory import qwen_pieces

        tokq = qwen_pieces(char_rows=True)
        by_id = {}
        for v, f in row_step_scale.items():
            for _p, e in qpieces(*tokq, v):
                if e is not None:
                    by_id[int(e)] = float(f)
        assert set(by_id) <= set(p.idx), "row_step_scale: vocabs outside the run"
        step_scale = torch.tensor(
            [by_id.get(int(e), 1.0) for e in rows.delta.ext_ids], device=device
        )[:, None]
        p.record["row_step_scale"] = {str(k): v for k, v in sorted(by_id.items())}
        print(
            f"row step scale: {len(by_id)} rows, {int((step_scale < 1).sum())} below 1, "
            f"min {float(step_scale.min()):.3f}",
            flush=True,
        )

    def lr_mult(st):
        m = 1.0
        if LR_DECAY == "cosine":
            m = 0.5 * (1 + math.cos(math.pi * min(st / steps, 1.0)))
        if warmup > 0:
            m *= min((st + 1) / warmup, 1.0)
        return m

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)
    anima.train()
    if COMPILE:
        from library.runtime.harness import compile_blocks_for_training

        compile_blocks_for_training(
            anima, None, backend="inductor", n_token_families=lat.n_families
        )
    batcher = Batcher(ns, recs, lat)
    split = BoxSplit()
    if pres:
        from .loss import pres_loss

        pgen = torch.Generator(device=device).manual_seed(PRES_SEED)
        pres_acc = [0.0, 0]  # Σ L_pres, passes since the last log
    log: list = []
    last = min(steps, max_steps or steps)
    t0 = time.time()
    for step in range(1, last + 1):
        if jamo is not None:
            jamo.apply()
        idx = batcher.next(step)
        latents = lat[idx].to(device)
        noise = torch.randn_like(latents)
        brecs = [recs[i] for i in idx]
        caps = [r["caption"] for r in brecs]
        noisy, ts, target = noisy_by_band(
            latents, noise, [tuple(r["band"]) for r in brecs], device
        )
        is_scene = brecs[0]["src"] == "scene"
        bs = BOX_SHARE if is_scene or (GRID_BOX and brecs[0]["src"] == "grid") else 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = dit_forward(anima, noisy, ts, cache, caps, device)
        loss_fm = box_share_fm_loss(
            pred, target, brecs, bs, BOX_SHARE_CAP, BOX_SHARE_GLYPHS, GRID_BOX
        )
        loss = rows.regularized(loss_fm)
        if is_scene:
            split.add(pred, target, brecs, ts)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if pres and is_scene and step % every_p == 0:
            # a second graph after the first is freed: the same peak memory
            del pred
            s = lo_p + (hi_p - lo_p) * torch.rand(
                len(idx), generator=pgen, device=device
            )
            sv = s.view(-1, *([1] * (latents.dim() - 1)))
            eps = torch.randn(
                latents.shape, generator=pgen, device=device, dtype=torch.float32
            )
            x_s = ((1.0 - sv) * latents.float() + sv * eps).to(torch.bfloat16)
            # grad mode on (a no_grad forward guards its own compiled graphs):
            # detach is the stop-gradient; the EN caption holds no ext id
            if jamo is not None:
                jamo.apply()  # the data term's backward freed the composition
            with torch.autocast("cuda", dtype=torch.bfloat16):
                teach = dit_forward(
                    anima, x_s, s, cache, [en_of[i] for i in idx], device
                ).detach()
                pred_p = dit_forward(anima, x_s, s, cache, caps, device)
            l_pres = pres_loss(pred_p, teach, brecs, GRID_BOX).mean()
            (lam_p * l_pres).backward()
            pres_acc[0] += float(l_pres.detach())
            pres_acc[1] += 1
            del teach, pred_p
        if step_scale is not None:
            before = rows.delta.raw.detach().clone()
        opt.step()
        if step_scale is not None:
            with torch.no_grad():
                rows.delta.raw.copy_(before + step_scale * (rows.delta.raw - before))
        rows.project()
        sched.step()
        if step % 25 == 0 or step == 1:
            rec = rows.log_record(step, loss_fm, loss, t0, split.pop())
            rec["lr"] = opt.param_groups[0]["lr"]
            if pres:
                rec["pres"] = pres_acc[0] / pres_acc[1] if pres_acc[1] else None
                rec["pres_n"] = pres_acc[1]
                pres_acc[:] = [0.0, 0]
            log.append(rec)
            print(json.dumps(rec), flush=True)
        if SAVE_EVERY and step % SAVE_EVERY == 0 and step < steps:
            save(out / "trained_partial.tmp", step)
            os.replace(out / "trained_partial.tmp", out / "trained_partial.pt")
            (out / "train_log.json").write_text(json.dumps(log, indent=1))
    # a stopped-early loop marks its rows with the step it reached
    save(out / "trained.pt", last if last < steps else None)
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    (out / "train_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1)
    )
    print(
        f"train: {last} steps in {(time.time() - t0) / 60:.1f} min → {out / 'trained.pt'}",
        flush=True,
    )
    del anima
    torch.cuda.empty_cache()
    return out
