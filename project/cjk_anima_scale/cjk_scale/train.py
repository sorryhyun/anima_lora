"""train — rows-only plain FM on one σ band, warm from the previous stage.

The probe's ``train/stage.py`` rows-arm path with the levers left behind
(design § 5): no pair loss, no counterfactual input, no ``c_flat``, no
out-vec, no row blocks / boosts, no encoder. What stays is exactly what
every read of record used — the box-share FM loss on scene batches, plain
MSE on grid / flat batches, cosine decay with warmup, the warm anchor.

Reused from the probe, unchanged: ``LatentStore`` (per-shape latent cache in
the data dir), ``Batcher`` (one shape × one source per batch),
``BoxSplit`` (in / out split logging), ``_encode_text`` (TE cache +
``eval_coverage.json``), ``dit_forward``.

Order (lazy loading): captions → TE cache → VAE latents → DiT → rows →
compile → loop → ``trained.pt``.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from .config import StageConfig
from .paths import arm_dir, data_dir, run_tag
from .rows import RowTable


def inventory_ext(words: dict, ev_ext: dict) -> set[int]:
    """The ext rows of every unit the run names (``words.json``), read off
    the eval strings' captions (``_encode_text``'s ``ev_ext``: text → ext ids;
    the eval set carries every inventory unit by construction). The stage
    table is these ∪ the rows the training captions touch, so a unit the
    band gives no draw stays in the table at its warm value."""
    inventory = {t for v in words.values() for t in v}
    out: set[int] = set()
    for text, ids in ev_ext.items():
        if text in inventory:
            out.update(int(i) for i in ids)
    return out


def train(
    cfg: StageConfig, tag: str, *, warm: Path | None, overrides: dict | None = None
) -> Path:
    from common.models import checkpoints, dit_forward, gen_args
    from library.anima.vocab_pack import attached_pack_rows, strategy_pack
    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model
    from library.inference.text import ensure_text_strategies
    from library.runtime.noise import fm_training_batch
    from train.stage import Batcher, BoxSplit, LatentStore, _encode_text

    from .loss import box_share_fm_loss

    t = {**cfg.train, **(overrides or {})}
    t_min, t_max = cfg.band
    torch.manual_seed(int(t["seed"]))
    data = data_dir(cfg.stage, tag)
    out = arm_dir(cfg.stage, tag)
    out.mkdir(parents=True, exist_ok=True)
    assert (data / "train.jsonl").exists(), (
        f"no data dir {data} — run the data step first"
    )
    recs = [
        json.loads(ln)
        for ln in (data / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if ln
    ]
    ev = json.loads((data / "eval.json").read_text(encoding="utf-8"))
    assert all("shape" in r for r in recs), "every record needs a shape"
    args = gen_args(512, int(t["steps"]), float(t["cfg"]), out)
    device = get_generation_settings(args).device

    cache, train_ext, ev_ext = _encode_text(
        recs, ev, device, out, te_cache=data / "te_cache"
    )
    words = json.loads((data / "words.json").read_text(encoding="utf-8"))
    table_ext = train_ext | inventory_ext(words, ev_ext)
    print(
        f"table: {len(table_ext)} rows = {len(train_ext)} touched by the captions "
        f"+ {len(table_ext - train_ext)} inventory rows this band draws nothing on",
        flush=True,
    )
    ns = SimpleNamespace(
        seed=int(t["seed"]),
        batch=int(t["batch"]),
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
        table_ext,
        strategy_pack(tok),
        warm=warm,
        init_anchor=float(t["init_anchor"]),
        free_residual=float(t["free_residual"]),
        lr=float(t["lr_rows"]),
        touched=train_ext,
    )
    n_rows = len(rows.delta.ext_ids)
    steps = int(t["train_steps"]) or int(t["steps_per_row"]) * n_rows
    warmup, decay = cfg.warmup_steps(steps), t["lr_decay"]
    print(
        f"train {cfg.stage} ({run_tag(cfg.stage, tag)}): σ [{t_min}, {t_max}], {n_rows} rows, "
        f"{steps} steps ({steps / n_rows:.0f}/row) × batch {t['batch']}, lr {t['lr_rows']:g} "
        f"{decay} warmup {warmup} ({float(t['lr_warmup_ratio']):g}), box_share {t['box_share']} → cap {t['box_share_cap']} "
        f"at {t['box_share_glyphs']} glyphs (log), "
        f"warm {'cold' if warm is None else warm}",
        flush=True,
    )
    opt = torch.optim.AdamW(rows.params, weight_decay=0.0, betas=(0.9, 0.99))

    def lr_mult(st):
        m = 1.0
        if decay == "cosine":
            m = 0.5 * (1 + math.cos(math.pi * min(st / steps, 1.0)))
        if warmup > 0:
            m *= min((st + 1) / warmup, 1.0)
        return m

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)
    anima.train()
    if int(t["compile"]):
        from library.runtime.harness import compile_blocks_for_training

        compile_blocks_for_training(
            anima, None, backend="inductor", n_token_families=lat.n_families
        )
    batcher = Batcher(ns, recs, lat)
    split = BoxSplit()
    log: list = []
    record = {
        "stage": cfg.stage,
        "tag": tag,
        "data_tag": run_tag(cfg.stage, tag),
        "band": [t_min, t_max],
        "t_min": t_min,
        "t_max": t_max,
        "train_steps": steps,
        "lr_warmup": warmup,
        "n_rows": n_rows,
        "n_touched": len(train_ext),
        "run": cfg.run.name if cfg.run else None,
        "run_config": str(cfg.run.path) if cfg.run else None,
        **{k: t[k] for k in sorted(t)},
        "config": str(cfg.path),
        "arm": "rows",
    }
    bs_cfg, cap, n_cap = (
        float(t["box_share"]),
        float(t["box_share_cap"]),
        float(t["box_share_glyphs"]),
    )
    save_every = int(t["save_every"])
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = batcher.next(step)
        latents = lat[idx].to(device)
        noise = torch.randn_like(latents)
        noisy, ts, target = fm_training_batch(
            latents,
            noise,
            dtype=torch.bfloat16,
            device=device,
            t_min=t_min,
            t_max=t_max,
        )
        brecs = [recs[i] for i in idx]
        is_scene = brecs[0]["src"] == "scene"
        bs = bs_cfg if is_scene else 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = dit_forward(
                anima, noisy, ts, cache, [r["caption"] for r in brecs], device
            )
        loss_fm = box_share_fm_loss(pred, target, brecs, bs, cap, n_cap)
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
        if save_every and step % save_every == 0 and step < steps:
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
