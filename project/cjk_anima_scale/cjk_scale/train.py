"""train — the run's vocabs' rows, plain FM, σ drawn per item inside its band.

The stages' ``train/stage.py`` rows-arm path with the levers left behind
(design § 5): no pair loss, no counterfactual input, no ``c_flat``, no
out-vec, no row blocks / boosts, no encoder. What stays is exactly what the
reads of record used — the box-share FM loss (scene items, and grid items'
cells under ``GRID_BOX``), cosine decay with warmup, the seed as the warm
start and the frozen context.

The table is the run's vocabs (their idx through the Qwen tokenizer, from
``vocabs.json``), warm from the seed table; every other row a training
caption touches rides frozen at its seed value and is stripped from
``trained.pt``. A vocab the seed lacks starts cold (``RowTable`` prints it).

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

from .config import RunConfig
from .paths import SEED_TABLE, data_dir, run_dir
from .rows import RowTable

# The trainer is fixed (plan.md § 2); a different trainer is a code change with
# a new report beside it, never a flag. Each value names the read that set it.
INIT_ANCHOR = (
    0.0  # μ 0: the vocabs need the displacement; the rest is frozen, not anchored
)
#                    (reports/conflict_joint_2026_09_25.md § 4 verdict 3; run0925_300f)
LR = 1e-3  # the lr that moved pieces (micro_warm_0923; conflict_joint report § 3)
BATCH = 4
LR_DECAY = "cosine"
WARMUP_RATIO = 0.1  # of the run's steps (micro_warm_0923: 100 / 640, 200 / 1280)
STEPS_PER_VOCAB = (
    90  # run0925_300f's joint budget (joint0507_0305 = 90; conflict_joint report)
)
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
GEN_STEPS, GEN_CFG = 28, 4.0  # the generation settings the stage helpers want


def vocab_idx(vocabs: list, tokq) -> set[int]:
    """The run's vocabs → their idx: the Qwen tokenizer maps each vocab to
    its pieces, every piece with a pack row is in the table (plan.md § 4:
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
        f"no data dir {data} — run `scale.py <run> data` first"
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


def train(rc: RunConfig) -> Path:
    from common.models import checkpoints, dit_forward, gen_args
    from data.inventory import qwen_pieces
    from library.anima.vocab_pack import attached_pack_rows, strategy_pack
    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model
    from library.inference.text import ensure_text_strategies
    from train.stage import Batcher, BoxSplit, LatentStore, _encode_text

    from .loss import box_share_fm_loss

    torch.manual_seed(SEED)
    data = data_dir(rc.name)
    out = run_dir(rc.name)
    out.mkdir(parents=True, exist_ok=True)
    recs, ev, vocabs = load_items(data)
    args = gen_args(512, GEN_STEPS, GEN_CFG, out)
    device = get_generation_settings(args).device

    cache, touched, _ev_idx = _encode_text(
        recs, ev, device, out, te_cache=data / "te_cache"
    )
    table = vocab_idx(vocabs, qwen_pieces())
    # captions carry rows outside the vocabs (corpus lines): they ride frozen
    # at the seed; the table (what trains, what the steps count) is the vocabs'
    frozen = touched - table
    touched = touched & table
    print(
        f"table: {len(table)} rows ({len(vocabs)} vocabs) — {len(touched)} touched "
        f"by the captions, {len(table - touched)} with no draw; {len(frozen)} context "
        f"rows frozen at {SEED_TABLE}",
        flush=True,
    )
    ns = SimpleNamespace(
        seed=SEED,
        batch=BATCH,
        train_size=512,
        row_blocks=0,
        row_boost="",
        arm="rows",
    )
    lat = LatentStore(ns, data, recs, list(range(len(recs))), device)

    anima = load_dit_model(args, device, torch.bfloat16)
    anima.requires_grad_(False)
    assert attached_pack_rows(anima), "no vocab pack attached to the DiT"
    tok, _ = ensure_text_strategies(checkpoints().text_encoder, vocab_pack=None)
    rows = RowTable(
        anima,
        device,
        table,
        strategy_pack(tok),
        warm=SEED_TABLE,
        init_anchor=INIT_ANCHOR,
        free_residual=FREE_RESIDUAL,
        lr=LR,
        touched=touched,
        frozen=frozen,
        context=SEED_TABLE,
    )
    n_rows = rows.n_rows
    steps = STEPS_PER_VOCAB * n_rows
    warmup = int(round(WARMUP_RATIO * steps))
    bands = sorted({tuple(r["band"]) for r in recs})
    print(
        f"train {rc.name}: σ per item in {bands}, {n_rows} rows, {steps} steps "
        f"({STEPS_PER_VOCAB}/row) × batch {BATCH}, lr {LR:g} {LR_DECAY} warmup {warmup} "
        f"({WARMUP_RATIO:g}), μ {INIT_ANCHOR:g}, box_share {BOX_SHARE} → cap "
        f"{BOX_SHARE_CAP} at {BOX_SHARE_GLYPHS} glyphs (log), grid_box {int(GRID_BOX)}, "
        f"warm {SEED_TABLE}",
        flush=True,
    )
    opt = torch.optim.AdamW(rows.params, weight_decay=0.0, betas=(0.9, 0.99))

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
    log: list = []
    record = {
        "run": rc.name,
        "run_config": str(rc.path),
        "vocabs": rc.vocabs if isinstance(rc.vocabs, str) else list(rc.vocabs),
        "data": str(data),
        "bands": [list(b) for b in bands],
        "train_steps": steps,
        "steps_per_row": STEPS_PER_VOCAB,
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
        "n_rows": n_rows,
        "n_touched": len(touched),
        "context": str(SEED_TABLE),
        "n_context": len(frozen),
        "arm": "rows",
    }
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = batcher.next(step)
        latents = lat[idx].to(device)
        noise = torch.randn_like(latents)
        brecs = [recs[i] for i in idx]
        noisy, ts, target = noisy_by_band(
            latents, noise, [tuple(r["band"]) for r in brecs], device
        )
        is_scene = brecs[0]["src"] == "scene"
        bs = BOX_SHARE if is_scene or (GRID_BOX and brecs[0]["src"] == "grid") else 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = dit_forward(
                anima, noisy, ts, cache, [r["caption"] for r in brecs], device
            )
        loss_fm = box_share_fm_loss(
            pred, target, brecs, bs, BOX_SHARE_CAP, BOX_SHARE_GLYPHS, GRID_BOX
        )
        loss = rows.regularized(loss_fm)
        if is_scene:
            split.add(pred, target, brecs, ts)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 25 == 0 or step == 1:
            rec = rows.log_record(step, loss_fm, loss, t0, split.pop())
            rec["lr"] = opt.param_groups[0]["lr"]
            log.append(rec)
            print(json.dumps(rec), flush=True)
        if SAVE_EVERY and step % SAVE_EVERY == 0 and step < steps:
            torch.save(rows.state_dict(record, step), out / "trained_partial.tmp")
            os.replace(out / "trained_partial.tmp", out / "trained_partial.pt")
            (out / "train_log.json").write_text(json.dumps(log, indent=1))
    torch.save(rows.state_dict(record), out / "trained.pt")
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    (out / "train_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1)
    )
    print(
        f"train: {steps} steps in {(time.time() - t0) / 60:.1f} min → {out / 'trained.pt'}",
        flush=True,
    )
    del anima
    torch.cuda.empty_cache()
    return out
