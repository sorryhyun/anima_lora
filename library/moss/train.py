"""Train a LoRA on a precached folder: flow matching, batch 1.

Reads only what ``cache.py`` wrote (latents + text embeddings), so neither the
Qwen3.5-9B TE nor the VAE is loaded. ``--caption_dropout`` reads the empty prompt
from the shipped ``library/moss/asset/empty_prompt.safetensors`` (key ``embeds``;
one file for every cache, re-encoded only if it is missing).

A layout-JSON cache also holds each image's plain Anima caption
(``embeds_plain``); a step trains on it with probability ``plain_caption_rate``,
else on one of the TE file's caption variants (``embeds_v{k}``) with probability
``caption_variant_rate``, then on the empty prompt with ``caption_dropout``
(``choose_embeds``). Each of the three draws has its own RNG, so the shuffle and
σ sequence match a run with none of them. Samples stay in RAM up to
``PRELOAD_LIMIT_GB``; a larger set is read from disk each step (variants always are).
With ``--fit_report``, step 1 is the largest sample with its longest caption (full,
plain or a variant) and no dropout, so the report bounds the worst case.

Objective (report App. A.2): ``z_σ = (1−σ)·z0 + σ·ε``, the DiT predicts
``ε − z0`` at timestep ``1000·σ``, MSE averaged over image tokens and channels
(text tokens never reach the head), with no σ/SNR weighting. With
``region_weight`` / ``rest_weight`` off 1 (i2i, full-size reference) each token's
error is multiplied before the mean: ``region_weight`` on the tokens the edit
changes (``edit_region``, moving linearly to ``region_weight_end`` by the last
step), ``rest_weight`` on the copied ones.

Two ways to push the training signal through the DiT's caption-row reads instead
of the prompt-independent shift a style set invites (``project/moss_te_sae``):
``delta_rows`` confines the adapter delta to the text / caption / trigger rows of
the joint sequence (``row_mask``, a per-step mask on ``LoRANetwork``; the trigger
rows are found with the tokenizer, ``loader.trigger_rows``), and
``preserve_weight`` adds, on the same noisy latent, the MSE between the LoRA's and
the base model's prediction for the trigger-stripped caption (``preserve_loss``;
the pair comes from ``{stem}.moss_te_neg.safetensors``, ``load_neg``).

σ follows the final Stage-3 interval exactly: ``Z ~ N(μ, s²)``, ``S = 1 − sigmoid(Z)``,
truncated to ``S ≤ cutoff`` and rescaled onto the grid ``{1/1000 … 1}``
(``σ = ⌈1000·S/cutoff⌉ / 1000``), mixed with 1 % discrete uniform. Shift 1, no
resolution shift: that interval trained at area ≤ 1024², the size the cache
targets.

Default memory plan: the DiT's block linears on int8 (``int8`` stream, GPTQ forward
copies, ``library/moss/dit_int8/``), every block resident, the compiled blocks' saved
set capped by ``activation_memory_budget`` and the EMA shadow on the card.
``configs/moss/lora_bf16.toml`` is the bf16 plan.
``--grad_checkpointing`` wraps each block's ``forward`` (``checkpoint_blocks``;
not ``__call__``), so the block swapper's forward hooks fire once per step and
never again during the recompute.

    make daemon-run ARGS="scripts/moss/train.py --cache post_image_dataset/moss/<set> --epochs 8"
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import torch

from library.env import resolve_under_home
from library.moss.loader import (
    compile_blocks,
    empty_cache,
    free_vram_gb,
    load_transformer,
    stream_fp32_,
)
from library.moss.lora import LoRANetwork, load_channel_stats
from library.moss.requests import LORA_ALPHA, TrainRequest
from library.moss.training import setup
from library.moss.training.data import (  # noqa: F401  (this module is their import path)
    EMPTY_PROMPT_PATH,
    NEG_SUFFIX,
    NUM_TRAIN_TIMESTEPS,
    Sample,
    choose_embeds,
    delta_row_ids,
    empty_prompt_embeds,
    load_neg,
    row_mask,
    sample_captions,
    sample_sigma,
    samples_from_cache,
)
from library.moss.training.ema import ema_decay, ema_step, ema_step_card  # noqa: F401
from library.moss.training.losses import (  # noqa: F401
    edit_region,
    preserve_loss,
    region_mse,
    tone_loss,
    tone_projection,
    training_loss,
)
from library.moss.training.memory import (  # noqa: F401
    budget_for_free,
    checkpoint_blocks,
    place,
    report_fit,
    set_activation_memory_budget,
    step_reserve_gb,
)
from library.training.progress import ProgressSink, command_job_path, run_scope

PROGRESS_EVERY = 2  # steps per "  progress i/n" line (the GUI bar reads them)


def run_train(req: TrainRequest) -> Path:
    cache_dir = resolve_under_home(req.cache)
    samples = samples_from_cache(cache_dir)
    if not samples:
        raise SystemExit(f"no cached pairs under {cache_dir}")
    empty = empty_prompt_embeds(req) if req.caption_dropout > 0 else None
    return train(req, samples, empty)


def train(
    req: TrainRequest, samples: list[Sample], empty: torch.Tensor | None = None
) -> Path:
    """The loop, over already-built ``samples`` (``run_train`` reads the cache;
    a fit smoke passes synthetic ones)."""
    if req.int8 != "off":
        from library.moss.dit_int8 import base as dit_int8

        dit_int8.check_plan(req)
    torch.manual_seed(req.seed)
    # Its own RNG: a rich log record draws from the global one (rich's Style link ids).
    shuffle_rng = random.Random(req.seed)
    gen = torch.Generator().manual_seed(req.seed)
    device = torch.device("cuda")

    img = sorted(s.n_img for s in samples)
    txt = sorted(s.n_text for s in samples)
    print(
        f"{len(samples)} samples, image {img[0]}-{img[-1]} tokens, text {txt[0]}-{txt[-1]} tokens",
        flush=True,
    )
    n_refs = sum(s.load_ref is not None for s in samples)
    if n_refs:
        ref_tok = sorted(s.n_ref for s in samples if s.load_ref is not None)
        print(
            f"reference latents: {n_refs}/{len(samples)} samples, "
            f"{ref_tok[0]}-{ref_tok[-1]} tokens",
            flush=True,
        )
    joint = max(s.n_img + s.n_ref + s.n_text for s in samples)
    use_plain = req.plain_caption_rate > 0
    n_plain = sum(s.load_plain is not None for s in samples) if use_plain else 0
    if n_plain:
        print(
            f"plain Anima captions: {n_plain}/{len(samples)} samples, "
            f"rate {req.plain_caption_rate}",
            flush=True,
        )

    get = setup.preload(samples, use_plain)
    regions, region_end = setup.edit_regions(req, samples, get)
    rests, tone_proj, tone_proj_b = setup.tone_setup(req, samples, get, device)
    use_variants = req.caption_variant_rate > 0
    n_variant_samples = sum(1 for s in samples if s.n_variants) if use_variants else 0
    if n_variant_samples:
        print(
            f"caption variants: {n_variant_samples}/{len(samples)} samples, "
            f"rate {req.caption_variant_rate}",
            flush=True,
        )
    rates = dict(
        plain_rate=req.plain_caption_rate,
        variant_rate=req.caption_variant_rate,
        dropout=req.caption_dropout,
        plain_rng=random.Random(req.seed + 1),
        drop_rng=random.Random(req.seed + 2),
        variant_rng=random.Random(req.seed + 3),
    )
    kinds = {"plain": 0, "variant": 0, "empty": 0, "preserve": 0}

    rows_of = setup.delta_rows(req, samples, get)
    negs = setup.load_negs(req, samples)

    transformer = load_transformer(req.model_dir, req.model)
    transformer.requires_grad_(False)
    if req.stream == "fp32":
        stream_fp32_(transformer)
    transformer.ref_timestep = req.ref_timestep if req.ref_timestep >= 0 else None
    transformer.ref_isolate = req.ref_isolate
    alpha = req.alpha if req.alpha is not None or req.algo != "lora" else LORA_ALPHA
    network = LoRANetwork(
        transformer,
        rank=req.rank,
        alpha=alpha,
        targets=req.targets,
        dtype=torch.float32,
        down_init=req.down_init,
        algo=req.algo,
        factor=req.factor,
        channel_stats=load_channel_stats(resolve_under_home(req.channel_stats))
        if req.algo == "lora" and req.svd_drop_loud
        else None,
        channel_mean=load_channel_stats(resolve_under_home(req.channel_mean))
        if req.algo == "lora" and req.down_init == "weight_svd_nodc"
        else None,
        svd_slice=req.svd_slice,
        svd_slice_count=req.svd_slice_count,
        svd_drop_loud=req.svd_drop_loud,
        rank_pattern=setup.rank_pattern(req, alpha),
    )
    int8_base = None
    if req.int8 != "off":
        int8_base = dit_int8.convert_for_training(transformer, req)
    patched = network.apply_to()
    network.to(device)
    print(
        f"{req.algo}: rank {req.rank} alpha {network.alpha} ({setup.adapter_detail(req, network)}) on {patched} linears, "
        f"{network.num_parameters / 1e6:.1f}M params (fp32 masters, saved {req.lora_dtype})",
        flush=True,
    )

    budgeted = req.compile and not req.grad_checkpointing
    budget = req.activation_memory_budget if budgeted else 1.0
    reserve_gb = step_reserve_gb(
        joint, grad_checkpointing=req.grad_checkpointing, budget=budget
    )
    attached = place(
        transformer, device, req.blocks_to_swap, reserve_gb, req.swap_h2d_only
    )
    if budgeted and budget < 1.0:
        set_activation_memory_budget(budget)
        print(f"activation memory budget {budget}", flush=True)
    elif not req.compile and not req.grad_checkpointing:
        print(
            "warning: eager without checkpointing saves ~10 MB per joint token "
            "per step; add --grad_checkpointing unless the card has room",
            flush=True,
        )
    repa, repa_targets = setup.repa_setup(req, samples, transformer)
    if req.compile:

        def shortest(s):
            n = s.n_text_min or s.n_text
            for _pos, neg in negs.get(s.stem, {}).values():
                n = min(n, neg.shape[0])
            return n if empty is None else min(n, len(empty))

        seq_range = (min(s.n_img + s.n_ref + shortest(s) for s in samples), joint)
        print(f"compile: token axis marked {seq_range[0]}-{seq_range[1]}", flush=True)
        compile_blocks(transformer.layers, seq_range)
    else:
        seq_range = None

    def set_rows(n_text: int, rows: list[int], s: Sample) -> None:
        """The step's row mask on the network; its token axis marked like the
        block inputs' (``compile_blocks``), or it would specialise against them."""
        mask = row_mask(s.n_img, n_text, rows, s.n_ref, device)
        if seq_range is not None and seq_range[0] <= mask.shape[1] <= seq_range[1]:
            torch._dynamo.mark_dynamic(mask, 1, min=seq_range[0], max=seq_range[1])
        network.set_row_mask(mask)

    if req.grad_checkpointing:
        checkpoint_blocks(transformer.layers)
    empty_cache()
    print(f"free VRAM before training: {free_vram_gb():.2f} GB", flush=True)

    params = list(network.parameters())
    optimizer, optimizer_name = setup.make_optimizer(req, params)
    total = req.epochs * len(samples)
    if req.max_steps:
        total = min(total, req.max_steps) if req.epochs else req.max_steps
    if req.grad_accum < 1:
        raise SystemExit(f"grad_accum must be >= 1, got {req.grad_accum}")
    accum = req.grad_accum
    updates = math.ceil(total / accum)
    warmup = max(1, int(updates * req.warmup_ratio))
    lr_sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: min(1.0, (update + 1) / warmup)
    )
    epochs = math.ceil(total / len(samples))
    print(
        f"{total} steps ({epochs} epochs x {len(samples)}), "
        + (f"{updates} updates of {accum}, " if accum > 1 else "")
        + f"warmup {warmup}",
        flush=True,
    )
    spans, decays = setup.ema_plan(req, updates)
    shadows = [[] for _ in spans]
    main_shadow = shadows[0] if req.ema_window else None
    update_ema = ema_step_card if req.ema_card else ema_step
    staging, n_updates = [], 0
    if spans:
        staging = [
            torch.empty(p.shape, dtype=torch.float32, pin_memory=True) for p in params
        ]
        print(
            f"EMA from update {warmup}, {'card' if req.ema_card else 'CPU'} shadow: "
            + ", ".join(
                f"span {s:g} = decay {d:.4f} (~{1 / (1 - d):.0f} updates)"
                for s, d in zip(spans, decays)
            ),
            flush=True,
        )

    out_path = resolve_under_home(req.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = setup.run_metadata(
        req,
        n_samples=len(samples),
        total=total,
        image_tokens=f"{img[0]}-{img[-1]}",
        n_plain=n_plain,
        n_variant_samples=n_variant_samples,
        n_refs=n_refs,
        optimizer=optimizer_name,
        has_repa=repa is not None,
        has_regions=bool(regions),
        region_end=region_end,
    )
    if int8_base is not None:
        meta.update(int8_base.metadata())

    def save(tag: str | None = None, shadow: list | None = None) -> Path:
        """The final file at ``out_path``; an epoch save under ``<dir>/<stem>/``.
        A started EMA ``shadow`` is saved in place of the live weights, swapped in
        through the staging buffers."""
        if tag is None:
            path = out_path
        else:
            folder = out_path.parent / out_path.stem
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{out_path.stem}_{tag}{out_path.suffix}"
        if not shadow:
            network.save(path, req.lora_dtype, **meta)
            return path
        k = next(i for i, s in enumerate(shadows) if s is shadow)
        with torch.no_grad():
            for buf, s, p in zip(staging, shadow, params):
                buf.copy_(p)
                p.copy_(s)
            network.save(
                path,
                req.lora_dtype,
                **meta,
                ema_window=spans[k],
                ema_decay=round(decays[k], 6),
            )
            for buf, p in zip(staging, params):
                p.copy_(buf)
        return path

    sink_path = command_job_path(
        str(out_path.parent.parent / "logs" / f"{out_path.stem}.progress.jsonl")
    )
    sink = ProgressSink(sink_path, run=out_path.stem, method="moss", preset=None)
    sink.run_start(total_steps=total, total_epochs=epochs, pid=os.getpid())
    print(f"progress sink: {sink_path}", flush=True)

    largest = max(samples, key=lambda s: s.n_img + s.n_ref + s.n_text)
    history, losses_all, fit = [], [], None
    step, grad_norm = 0, float("nan")
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    with run_scope(sink, final_step=lambda: step):
        for epoch in range(epochs):
            order = shuffle_rng.sample(samples, len(samples))
            if epoch == 0 and req.fit_report:
                # Only the largest sample moves to step 1, the worst case the fit
                # report bounds; the rest of epoch 1 stays shuffled.
                order.remove(largest)
                order.insert(0, largest)
            losses, aligns, t_epoch = [], [], time.time()
            for s in order:
                if step >= total:
                    break
                latent, embeds, alt, ref = get(s)
                embeds, kind = choose_embeds(
                    embeds,
                    alt,
                    s.n_variants if use_variants else 0,
                    s.load_variant,
                    empty,
                    worst_case=step == 0 and req.fit_report,
                    **rates,
                )
                neg = negs.get(s.stem, {}).get(kind)
                if neg is not None:
                    embeds, neg = neg  # the sidecar's own encode of the full caption
                    kinds["preserve"] += 1
                text_rows = rows_of.get((s.stem, kind))
                kind = "variant" if kind.startswith("v") else kind
                if kind in kinds:
                    kinds[kind] += 1
                latent = latent.unsqueeze(0).to(device, torch.float32)
                embeds = embeds.unsqueeze(0).to(device)
                if ref is not None:
                    ref = ref.unsqueeze(0).to(device, torch.float32)
                if rows_of:
                    set_rows(embeds.shape[1], text_rows or [], s)
                sigma = sample_sigma(
                    gen, req.sigma_mu, req.sigma_s, req.sigma_cutoff, req.sigma_uniform
                )
                if repa is not None:
                    target, grid = repa_targets.get(s.stem, (None, None))
                    repa.prime(
                        None if target is None else target.to(device),
                        grid,
                        tuple(latent.shape[-2:]),
                    )
                parts, state = {}, {}
                tone = None
                if s.stem in rests:
                    tone = dict(
                        rest=rests[s.stem].to(device),
                        pool=req.tone_pool,
                        proj=tone_proj_b
                        if step >= total * req.tone_b_start
                        else tone_proj,
                        weight=req.tone_weight
                        if step >= total * req.tone_start
                        else 0.0,
                        log_weight=req.tone_weight,
                    )
                if s.stem in regions or tone is not None:
                    loss = training_loss(
                        transformer,
                        latent,
                        embeds,
                        sigma,
                        ref=ref,
                        region=regions[s.stem].to(device)
                        if s.stem in regions
                        else None,
                        region_weight=req.region_weight
                        + (region_end - req.region_weight) * step / max(total - 1, 1),
                        rest_weight=req.rest_weight,
                        parts=parts,
                        tone=tone,
                        state=state,
                    )
                else:
                    loss = training_loss(
                        transformer, latent, embeds, sigma, ref=ref, state=state
                    )
                align = repa.loss() if repa is not None else None
                if align is not None:
                    ((loss + req.repa_weight * align) / accum).backward()
                    aligns.append(align.item())
                else:
                    (loss / accum).backward()
                if neg is not None:
                    # Its own backward: the two graphs never sit on the card together.
                    neg = neg.unsqueeze(0).to(device)
                    if rows_of:
                        # The stripped caption has no trigger rows.
                        set_rows(
                            neg.shape[1],
                            delta_row_ids(req.delta_rows, neg.shape[1], []),
                            s,
                        )
                    pres = preserve_loss(
                        transformer,
                        network,
                        state["noisy"],
                        state["timestep"],
                        neg,
                        ref,
                        attached,
                    )
                    (req.preserve_weight * pres / accum).backward()
                    parts["pres"] = pres.item()
                step += 1
                updated = step % accum == 0 or step == total
                if updated:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        params, req.max_grad_norm
                    ).item()
                    optimizer.step()
                    lr_sched.step()
                    optimizer.zero_grad(set_to_none=True)
                    n_updates += 1
                    if shadows and n_updates >= warmup:
                        update_ema(shadows, staging, params, decays)
                value = loss.item()
                sink.log(
                    {
                        "loss": value,
                        "sigma": sigma,
                        "lr": lr_sched.get_last_lr()[0],
                    }
                    | ({"grad_norm": grad_norm} if updated else {})
                    | ({"repa": aligns[-1]} if align is not None else {})
                    | {f"loss_{k}": v for k, v in parts.items()},
                    global_step=step,
                    epoch=epoch + 1,
                )
                losses.append(value)
                losses_all.append(
                    {"step": step, "loss": value, "sigma": sigma}
                    | ({"repa": aligns[-1]} if align is not None else {})
                    | {f"loss_{k}": v for k, v in parts.items()}
                )
                if step == 1:
                    if req.fit_report:
                        fit = report_fit(
                            transformer, attached, s.n_img, embeds.shape[1]
                        )
                    t_start = time.time()  # s/step excludes step 1's allocator warm-up
                    t_window, step_window, per_step = t_start, 1, None
                eta_rate = (
                    (time.time() - t_start) / max(1, step - 1)
                    if per_step is None
                    else per_step
                )
                eta = int(eta_rate * (total - step))
                if step % PROGRESS_EVERY == 0 or step == total:
                    print(
                        f"  progress {step}/{total} epoch {epoch + 1}/{epochs} "
                        f"loss {sum(losses) / len(losses):.4f} "
                        f"eta {eta // 3600}:{eta % 3600 // 60:02d}:{eta % 60:02d}",
                        flush=True,
                    )
                if step % req.log_every == 0:
                    # s/step over the steps since the last log line, so a recompile
                    # stays in the window it landed in. Step 1 (compile) has none.
                    now = time.time()
                    if step > step_window:
                        per_step = (now - t_window) / (step - step_window)
                        t_window, step_window = now, step
                    rate = f" {per_step:.2f}s/step" if per_step is not None else ""
                    if align is not None:
                        rate += f" repa {aligns[-1]:.4f}"
                    rate += "".join(f" {k} {v:.4f}" for k, v in parts.items())
                    print(
                        f"  step {step}/{total} loss {value:.4f} |g| {grad_norm:.3f} "
                        f"sigma {sigma:.3f} lr {lr_sched.get_last_lr()[0]:.2e} "
                        f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
                        f"{rate}",
                        flush=True,
                    )
            if not losses:
                break
            mean = sum(losses) / len(losses)
            history.append(
                {"epoch": epoch + 1, "loss": mean, "seconds": time.time() - t_epoch}
                | ({"repa": sum(aligns) / len(aligns)} if aligns else {})
            )
            repa_mean = f" repa {sum(aligns) / len(aligns):.4f}" if aligns else ""
            print(
                f"epoch {epoch + 1}/{epochs}: loss {mean:.4f}{repa_mean} {time.time() - t_epoch:.0f}s "
                f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB",
                flush=True,
            )
            if (
                req.save_every_epochs
                and (epoch + 1) % req.save_every_epochs == 0
                and epoch + 1 < epochs
            ):
                saved = save(f"e{epoch + 1}", main_shadow)
                sink.ckpt(global_step=step, path=str(saved))
                print(f"  saved {saved}", flush=True)

        final = save(shadow=main_shadow)
        sink.ckpt(global_step=step, path=str(final))
        if any(shadows):
            print(f"  saved raw weights {save('raw')}", flush=True)
        for span, shadow in zip(spans, shadows):
            if shadow is not main_shadow and shadow:
                print(
                    f"  saved EMA {span:g} {save(f'ema{span:g}', shadow)}", flush=True
                )
        seconds = time.time() - t_start
        summary = {
            "output": str(final),
            "request": {k: getattr(req, k) for k in req.__dataclass_fields__},
            "fit": fit,
            "steps": step,
            "plain_steps": kinds["plain"],
            "variant_steps": kinds["variant"],
            "dropout_steps": kinds["empty"],
            "preserve_steps": kinds["preserve"],
            "minutes": seconds / 60,
            "s_per_step": seconds / max(1, step - 1),
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
            "history": history,
            "losses": losses_all,
        }
        report = out_path.with_suffix(".json")
        report.write_text(json.dumps(summary, indent=2))
        print(f"wrote {final} and {report} — {seconds / 60:.1f} min", flush=True)
    if int8_base is not None:
        int8_base.close()
    if attached is not None:
        attached.detach()
    return final
