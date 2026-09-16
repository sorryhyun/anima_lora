"""Validation pass: CMMD (primary) with per-sigma FM-MSE fallback.

The trainer class owns only the hook points (``process_batch``, ``on_step_start``, ``on_validation_step_end``,
``_switch_rng_state`` / ``_restore_rng_state``). The PE encoder is cached on
the trainer as ``trainer._cmmd_pe_bundle`` to avoid reloading PE-Core each pass.

CMMD is the primary signal — FM-MSE does not track sample quality on Anima.
FM-MSE still runs as the silent-loss fallback when CMMD can't (no PE/TE cache, sampling
error, missing references)."""

from __future__ import annotations

import logging
import os

import torch
from safetensors.torch import load_file as _load_safetensors
from tqdm import tqdm

from library.anima import training as anima_train_utils
from library.runtime.device import clean_memory_on_device
from library.training.cmmd import (
    cmmd_from_pools,
    load_reference_features,
    pool_and_normalize,
    resolve_pe_sidecar,
)
from library.vision.encoder import encode_pe_from_imageminus1to1, load_pe_encoder

logger = logging.getLogger(__name__)


def run_validation(
    trainer,
    ctx,
    val,
    *,
    val_loss_recorder,
    epoch,
    global_step,
    progress_bar,
    progress_desc,
    postfix_label,
    log_avg_key,
    log_div_key,
    logging_fn,
) -> None:
    """Validation = CMMD between the live model's samples and the held-out
    reference's cached PE features, falling back to per-sigma FM-MSE on
    ``val.dataloader`` if CMMD can't run (no PE/TE cache, sampling error)."""
    args = ctx.args
    accelerator = ctx.accelerator

    ctx.optimizer_eval_fn()
    accelerator.unwrap_model(ctx.network).eval()
    unwrapped_unet = accelerator.unwrap_model(ctx.unet)
    if hasattr(unwrapped_unet, "switch_block_swap_for_inference"):
        unwrapped_unet.switch_block_swap_for_inference()
    rng_states = trainer._switch_rng_state(
        args.validation_seed if args.validation_seed is not None else args.seed
    )

    try:
        use_cmmd = getattr(args, "use_cmmd", True)
        want_fm = getattr(args, "validation_fm_with_cmmd", True)
        val_seed = (
            args.validation_seed if args.validation_seed is not None else args.seed
        )

        # FM-MSE forwards run FIRST, on the DiT as it sits at the end of the
        # training step — resident and block-swap-prepared — so they don't pay
        # CMMD's DiT→CPU→GPU unload/reload, nor forward a DiT left in the
        # all-blocks-on-GPU state that CMMD's `.to(device)` reload leaves under
        # block swap. The result's role (aux vs primary) is decided after CMMD,
        # so the forward sweep happens at most once. Re-seed for determinism.
        fm_result = None
        if use_cmmd and want_fm:
            trainer._switch_rng_state(val_seed)
            fm_result = _compute_fm_validation(
                trainer,
                ctx=ctx,
                val=val,
                progress_desc=progress_desc,
                postfix_label=postfix_label,
                primary=False,
            )

        cmmd_ok = False
        if use_cmmd:
            cmmd_ok = _try_cmmd_validation(
                trainer,
                ctx=ctx,
                val=val,
                unwrapped_unet=unwrapped_unet,
                val_loss_recorder=val_loss_recorder,
                epoch=epoch,
                global_step=global_step,
                progress_desc=progress_desc,
                log_avg_key=log_avg_key,
                log_div_key=log_div_key,
                logging_fn=logging_fn,
            )

        # CMMD owns the primary signal when it ran; the FM sweep (if done) is
        # logged as a diagnostic under distinct `loss/validation/fm_*` keys. If
        # CMMD is off or failed, FM is the primary signal instead — reusing the
        # sweep already done above, or running it now (the DiT is back on the
        # GPU via _try_cmmd_validation's finally, or never left it).
        if cmmd_ok:
            if fm_result is not None:
                _log_fm_validation(
                    fm_result,
                    ctx=ctx,
                    val=val,
                    epoch=epoch,
                    global_step=global_step,
                    log_avg_key=log_avg_key,
                    log_div_key=log_div_key,
                    logging_fn=logging_fn,
                    val_loss_recorder=val_loss_recorder,
                    primary=False,
                )
        else:
            if fm_result is None:
                trainer._switch_rng_state(val_seed)
                fm_result = _compute_fm_validation(
                    trainer,
                    ctx=ctx,
                    val=val,
                    progress_desc=progress_desc,
                    postfix_label=postfix_label,
                    primary=True,
                )
            if fm_result is not None:
                _log_fm_validation(
                    fm_result,
                    ctx=ctx,
                    val=val,
                    epoch=epoch,
                    global_step=global_step,
                    log_avg_key=log_avg_key,
                    log_div_key=log_div_key,
                    logging_fn=logging_fn,
                    val_loss_recorder=val_loss_recorder,
                    primary=True,
                )
        # Method-adapter baseline deltas (e.g. IP-Adapter no_ip / shuffled_ref).
        # Runs independently of CMMD/FM above — these are FM-MSE re-forwards on
        # the same (batch, sigma, noise) with the adapter perturbed, so the
        # delta isolates the adapter's contribution. Gated by
        # ``--validation_baselines`` (default on): each baseline is a full extra
        # val forward per (batch, sigma), so skipping them roughly halves
        # IP-Adapter validation time.
        if getattr(args, "validation_baselines", True):
            _run_validation_baselines(
                trainer,
                ctx=ctx,
                val=val,
                epoch=epoch,
                global_step=global_step,
                progress_desc=progress_desc,
                logging_fn=logging_fn,
            )
    finally:
        trainer._restore_rng_state(rng_states)
        args.t_min = val.original_t_min
        args.t_max = val.original_t_max
        ctx.optimizer_train_fn()
        accelerator.unwrap_model(ctx.network).train()
        if hasattr(unwrapped_unet, "switch_block_swap_for_training"):
            unwrapped_unet.switch_block_swap_for_training()
        clean_memory_on_device(accelerator.device)


def _try_cmmd_validation(
    trainer,
    *,
    ctx,
    val,
    unwrapped_unet,
    val_loss_recorder,
    epoch,
    global_step,
    progress_desc,
    log_avg_key,
    log_div_key,
    logging_fn,
) -> bool:
    """Run CMMD-based validation. Returns True if it logged a value, False
    if the caller should fall back to FM-MSE (no dataset group, no PE/TE
    cache, ``load_reference_features`` failure, or any sampling exception)."""
    args = ctx.args
    accelerator = ctx.accelerator

    if val.dataset_group is None:
        return False

    val_items: list = []
    for ds in val.dataset_group.datasets:
        val_items.extend(ds.image_data.values())
    if not val_items:
        return False

    # Reference PE features sit next to each val item's cached TE
    # output (both produced by `make preprocess-pe` / `-te`).
    ref_sidecars = []
    ref_items = []
    for item in val_items:
        te_path = item.text_encoder_outputs_npz
        if te_path is None:
            continue
        cache_dir = os.path.dirname(te_path)
        ref_sidecars.append(
            resolve_pe_sidecar(
                item.absolute_path, encoder="pe", cache_dir=cache_dir
            )
        )
        ref_items.append(item)
    if not ref_sidecars:
        logger.warning(
            "CMMD val: no items had cached TE outputs; falling back to FM-MSE."
        )
        return False
    try:
        ref_pool = load_reference_features(ref_sidecars).to(accelerator.device)
    except RuntimeError as exc:
        logger.warning(f"CMMD val ref load failed ({exc}); falling back to FM-MSE.")
        return False

    if getattr(trainer, "_cmmd_pe_bundle", None) is None:
        trainer._cmmd_pe_bundle = load_pe_encoder(accelerator.device)
        # Park PE-Core (~600 MB bf16) on CPU between encodes so the DiT
        # sample step has the full GPU budget. Bundle keeps device=cuda
        # so encode_pe_from_imageminus1to1 still routes inputs correctly;
        # we shuttle the underlying model to GPU only for the encode call.
        trainer._cmmd_pe_bundle.encoder.inner.to("cpu")
    bundle = trainer._cmmd_pe_bundle

    sample_steps = int(getattr(args, "validation_sample_steps", 20))
    cfg_scale = float(getattr(args, "validation_cfg_scale", 1.0))
    flow_shift = float(getattr(args, "discrete_flow_shift", 1.0))

    val_progress_bar = tqdm(
        range(len(ref_items)),
        smoothing=0,
        disable=not accelerator.is_local_main_process,
        desc=progress_desc,
    )

    gen_pooled: list[torch.Tensor] = []
    seed_base = (
        args.validation_seed if args.validation_seed is not None else args.seed
    )

    # Three-phase val so the DiT, VAE and PE-Core never share the GPU — the
    # peak is one model + its working set (per-sample VAE decode with the DiT
    # still resident OOMs):
    #   phase 1: DiT resident → sample every item's latents, park on CPU.
    #   phase 2: DiT → CPU, VAE → GPU → decode all latents to pixels (CPU).
    #   phase 3: VAE → CPU, PE → GPU → encode all pixels to features.
    # One DiT/VAE/PE round-trip per val pass instead of N VAE shuttles.
    latents_cpu: list[torch.Tensor] = []
    pixel_images: list[torch.Tensor] = []
    try:
        try:
            with torch.no_grad(), accelerator.autocast():
                unwrapped_unet.prepare_block_swap_before_forward()
                # PHASE 1 — sample latents with the DiT on the GPU; keep the
                # decode out so the VAE never loads while the DiT is resident.
                for i, item in enumerate(ref_items):
                    sd = _load_safetensors(item.text_encoder_outputs_npz)
                    crossattn_emb = _build_val_crossattn_emb(
                        unwrapped_unet, sd, accelerator
                    )

                    bucket_w, bucket_h = item.bucket_reso

                    latents = anima_train_utils.sample_image_latents(
                        accelerator=accelerator,
                        dit=unwrapped_unet,
                        height=int(bucket_h),
                        width=int(bucket_w),
                        crossattn_emb=crossattn_emb,
                        sample_steps=sample_steps,
                        guidance_scale=cfg_scale,
                        flow_shift=flow_shift,
                        seed=seed_base + i,
                        show_progress=False,
                        # Keep the DiT's allocator pool warm across items — no
                        # per-item empty_cache; the boundary cleans (before the
                        # VAE / PE swaps) are what actually free the GPU.
                        clean_cache=False,
                    )
                    latents_cpu.append(latents.detach().to("cpu"))
                    del latents, crossattn_emb
                    val_progress_bar.update(1)

                    trainer.on_validation_step_end(ctx, {})

                # PHASE 2 — park DiT on CPU, bring VAE on, decode every latent.
                unwrapped_unet.to("cpu")
                clean_memory_on_device(accelerator.device)
                org_vae_device = ctx.vae.device
                ctx.vae.to(accelerator.device)
                try:
                    for latents in latents_cpu:
                        image = anima_train_utils.decode_latents_to_image(
                            ctx.vae, latents, accelerator.device
                        )
                        pixel_images.append(image.detach().cpu())
                        del image
                finally:
                    ctx.vae.to(org_vae_device)
                    clean_memory_on_device(accelerator.device)
                latents_cpu.clear()

                # PHASE 3 — VAE off, PE on, encode every decoded pixel batch.
                bundle.encoder.inner.to(accelerator.device)
                try:
                    # Batch PE encoding by bucket: same-shape images go through
                    # one same_bucket=True forward instead of N. Original order
                    # is preserved so gen_pooled[i] still pairs with ref_pool[i].
                    bucket_groups: dict[tuple[int, int], list[int]] = {}
                    for idx, img in enumerate(pixel_images):
                        key = (int(img.shape[-2]), int(img.shape[-1]))
                        bucket_groups.setdefault(key, []).append(idx)

                    pooled_slots: list[torch.Tensor | None] = [None] * len(
                        pixel_images
                    )
                    for indices in bucket_groups.values():
                        batch = torch.stack(
                            [pixel_images[idx] for idx in indices], dim=0
                        ).to(accelerator.device)
                        feats_list = encode_pe_from_imageminus1to1(
                            bundle, batch, same_bucket=True
                        )
                        for idx, feats in zip(indices, feats_list):
                            pooled_slots[idx] = pool_and_normalize(feats).cpu()
                        del batch, feats_list
                    gen_pooled = [t for t in pooled_slots if t is not None]
                finally:
                    bundle.encoder.inner.to("cpu")
                    clean_memory_on_device(accelerator.device)
        finally:
            # The DiT must end on the GPU regardless of how the phases exit:
            # FM-aux / the FM-MSE fallback / resumed training all forward it.
            unwrapped_unet.to(accelerator.device)
            clean_memory_on_device(accelerator.device)
    except (KeyError, RuntimeError, FileNotFoundError) as exc:
        val_progress_bar.close()
        logger.warning(
            f"CMMD val sampling failed ({type(exc).__name__}: {exc}); "
            "falling back to FM-MSE."
        )
        return False

    val_progress_bar.close()

    gen_pool = torch.stack(gen_pooled, dim=0).to(accelerator.device)
    cmmd_value = cmmd_from_pools(ref_pool, gen_pool)
    val_loss_recorder.add(epoch=epoch, step=global_step, loss=cmmd_value)

    # Always surface the score to the console — the tracker logging below is
    # gated on `is_tracking`, so without this the CMMD value never shows.
    logger.info(
        f"CMMD validation @ step {global_step} (epoch {epoch + 1}): "
        f"{cmmd_value:.4f}  (n={len(ref_items)}, lower is better)"
    )

    if ctx.is_tracking:
        logs = {
            log_avg_key: cmmd_value,
            log_div_key: cmmd_value - val.train_loss_recorder.moving_average,
            log_avg_key.removesuffix("_average") + "_cmmd": cmmd_value,
            log_avg_key.removesuffix("_average") + "_n": len(ref_items),
        }
        logging_fn(accelerator, logs, global_step, epoch + 1)
    return True


def _compute_fm_validation(
    trainer,
    *,
    ctx,
    val,
    progress_desc,
    postfix_label,
    primary: bool,
) -> dict | None:
    """Run the per-sigma FM-MSE forward sweep over ``val.dataloader`` (pinning
    ``args.t_{min,max}`` to each sigma in ``val.sigmas``) and return the raw
    losses for :func:`_log_fm_validation` to record/log once CMMD's outcome is
    known — so the role decision (primary vs aux) never costs a second sweep.

    Returns ``{"per_sigma_losses", "ordered_losses"}`` or ``None`` when there's
    nothing to validate. Must run with the DiT resident (the caller schedules
    it before CMMD's unload). Mutates ``args.t_{min,max}``; the caller restores
    them. ``primary`` only selects the progress-bar label here."""
    args = ctx.args
    accelerator = ctx.accelerator

    if val.dataloader is None or len(val.dataloader) == 0 or not val.sigmas:
        return None

    val_progress_bar = tqdm(
        range(val.total_steps),
        smoothing=0,
        disable=not accelerator.is_local_main_process,
        desc=f"{progress_desc} ({'fm-mse' if primary else 'fm-aux'})",
    )
    per_sigma_losses = {s: [] for s in val.sigmas}
    ordered_losses: list[float] = []

    try:
        for val_step, batch in enumerate(val.dataloader):
            if val_step >= val.steps:
                break

            for sigma in val.sigmas:
                trainer.on_step_start(ctx, batch, is_train=False)
                args.t_min = args.t_max = sigma

                loss = trainer.process_batch(ctx, batch, is_train=False)
                current_loss = loss.detach().item()
                ordered_losses.append(current_loss)
                per_sigma_losses[sigma].append(current_loss)
                val_progress_bar.update(1)
                val_progress_bar.set_postfix(
                    {
                        postfix_label: sum(ordered_losses) / len(ordered_losses),
                        "sigma": f"{sigma:.2f}",
                    }
                )
                trainer.on_validation_step_end(ctx, batch)
    finally:
        val_progress_bar.close()

    if not ordered_losses:
        return None
    return {"per_sigma_losses": per_sigma_losses, "ordered_losses": ordered_losses}


def _log_fm_validation(
    fm: dict,
    *,
    ctx,
    val,
    epoch,
    global_step,
    log_avg_key,
    log_div_key,
    logging_fn,
    val_loss_recorder,
    primary: bool,
) -> None:
    """Record/log a precomputed FM-MSE sweep from :func:`_compute_fm_validation`.

    ``primary=True`` (CMMD off / failed): FM-MSE is the validation signal — it
    feeds ``val_loss_recorder`` (the best-ckpt selector, populated even when not
    tracking) and logs to ``log_avg_key``/``log_div_key`` (tagged
    ``_fm_fallback``). ``primary=False`` (alongside a successful CMMD pass):
    purely diagnostic — it never touches the CMMD ``val_loss_recorder`` and logs
    to separate ``loss/validation/fm_*`` keys so the two signals don't collide.
    """
    ordered_losses = fm["ordered_losses"]
    per_sigma_losses = fm["per_sigma_losses"]

    # Primary FM feeds the recorder regardless of tracking (best-ckpt needs it).
    if primary:
        for step, loss in enumerate(ordered_losses):
            val_loss_recorder.add(epoch=epoch, step=step, loss=loss)

    if not ctx.is_tracking:
        return

    accelerator = ctx.accelerator
    if primary:
        logs = {
            log_avg_key: val_loss_recorder.moving_average,
            log_div_key: val_loss_recorder.moving_average
            - val.train_loss_recorder.moving_average,
            log_avg_key.removesuffix("_average") + "_fm_fallback": 1.0,
        }
        for s, losses in per_sigma_losses.items():
            if losses:
                logs[f"loss/validation/sigma_{s:.2f}"] = sum(losses) / len(losses)
    else:
        fm_mean = sum(ordered_losses) / len(ordered_losses)
        logs = {
            "loss/validation/fm_average": fm_mean,
            "loss/validation/fm_div": fm_mean
            - val.train_loss_recorder.moving_average,
        }
        for s, losses in per_sigma_losses.items():
            if losses:
                logs[f"loss/validation/fm_sigma_{s:.2f}"] = sum(losses) / len(losses)
    logging_fn(accelerator, logs, global_step, epoch + 1)


def _run_validation_baselines(
    trainer,
    *,
    ctx,
    val,
    epoch,
    global_step,
    progress_desc,
    logging_fn,
) -> None:
    """Run each method adapter's ``validation_baselines`` and log the FM-MSE
    delta vs the (adapter-active) primary forward.

    For every (val batch, sigma): re-seed to a per-item deterministic point,
    run the primary forward, then for each baseline re-seed to the *same*
    point, ``enter()`` the perturbation, re-forward, ``exit()``. Identical
    noise + sigma means ``delta = baseline_loss − primary_loss`` isolates the
    adapter's contribution (positive ⇒ the adapter is helping).

    Logged as ``loss/validation/baseline_<name>`` and ``..._delta``. Runs only
    when at least one adapter exposes a baseline (others no-op). This is the
    FM-MSE signal — necessary-not-sufficient on Anima; pair with CMMD."""
    args = ctx.args
    accelerator = ctx.accelerator

    adapters = getattr(trainer, "_adapters", None) or []
    pairs = []  # (baseline,)
    for adapter in adapters:
        for baseline in adapter.validation_baselines():
            pairs.append(baseline)
    if not pairs:
        return
    if val.dataloader is None or len(val.dataloader) == 0 or not val.sigmas:
        return

    seed = args.validation_seed if args.validation_seed is not None else args.seed
    primary_losses: list[float] = []
    base_losses: dict[str, list[float]] = {b.name: [] for b in pairs}
    base_deltas: dict[str, list[float]] = {b.name: [] for b in pairs}

    n_forwards = val.total_steps * (1 + len(pairs))
    bar = tqdm(
        range(n_forwards),
        smoothing=0,
        disable=not accelerator.is_local_main_process,
        desc=f"{progress_desc} (baselines)",
    )
    try:
        for val_step, batch in enumerate(val.dataloader):
            if val_step >= val.steps:
                break
            for sigma in val.sigmas:
                args.t_min = args.t_max = sigma
                item_seed = seed + val_step * 1009 + int(sigma * 997)

                # Seed to a deterministic point and leave it seeded; the outer
                # run_validation snapshotted the true RNG and restores it.
                trainer._switch_rng_state(item_seed)
                trainer.on_step_start(ctx, batch, is_train=False)
                primary = trainer.process_batch(ctx, batch, is_train=False)
                primary_loss = primary.detach().item()
                trainer.on_validation_step_end(ctx, batch)
                primary_losses.append(primary_loss)
                bar.update(1)

                for baseline in pairs:
                    # Re-seed to the SAME starting point so the baseline forward
                    # sees identical noise; the only difference is the perturbation.
                    trainer._switch_rng_state(item_seed)
                    baseline.enter()
                    try:
                        trainer.on_step_start(ctx, batch, is_train=False)
                        b_loss = (
                            trainer.process_batch(ctx, batch, is_train=False)
                            .detach()
                            .item()
                        )
                        trainer.on_validation_step_end(ctx, batch)
                    finally:
                        baseline.exit()
                    base_losses[baseline.name].append(b_loss)
                    base_deltas[baseline.name].append(b_loss - primary_loss)
                    bar.update(1)
    finally:
        bar.close()

    if ctx.is_tracking and primary_losses:
        logs = {
            "loss/validation/baseline_primary": sum(primary_losses)
            / len(primary_losses)
        }
        for name in base_losses:
            losses = base_losses[name]
            deltas = base_deltas[name]
            if losses:
                logs[f"loss/validation/baseline_{name}"] = sum(losses) / len(losses)
                logs[f"loss/validation/baseline_{name}_delta"] = sum(deltas) / len(
                    deltas
                )
        logging_fn(accelerator, logs, global_step, epoch + 1)


def _build_val_crossattn_emb(dit, sd, accelerator):
    """Construct the cross-attention embedding the DiT expects from a
    cached TE sidecar — using the saved post-LLM-adapter ``crossattn_emb``
    when present, otherwise running ``llm_adapter`` exactly like
    ``_sample_image_inference`` does. Pads to 512 tokens (the model's
    fixed context length). Multi-variant caches expose `<key>_v0` (pristine
    caption) instead of `<key>`; pin to v0 for deterministic validation."""
    device = accelerator.device
    dtype = dit.dtype
    suffix = "" if "prompt_embeds" in sd or "crossattn_emb" in sd else "_v0"
    ce_key = f"crossattn_emb{suffix}"
    if ce_key in sd:
        ce = sd[ce_key].unsqueeze(0).to(device, dtype=dtype)
        if ce.shape[1] < 512:
            ce = torch.nn.functional.pad(ce, (0, 0, 0, 512 - ce.shape[1]))
        return ce

    prompt_embeds = sd[f"prompt_embeds{suffix}"].unsqueeze(0).to(device, dtype=dtype)
    attn_mask = sd[f"attn_mask{suffix}"].unsqueeze(0).to(device)
    t5_ids = sd[f"t5_input_ids{suffix}"].unsqueeze(0).to(device, dtype=torch.long)
    t5_attn_mask = sd[f"t5_attn_mask{suffix}"].unsqueeze(0).to(device)

    if getattr(dit, "use_llm_adapter", False):
        ce = dit.llm_adapter(
            source_hidden_states=prompt_embeds,
            target_input_ids=t5_ids,
            target_attention_mask=t5_attn_mask,
            source_attention_mask=attn_mask,
        )
        ce[~t5_attn_mask.bool()] = 0
    else:
        ce = prompt_embeds
    if ce.shape[1] < 512:
        ce = torch.nn.functional.pad(ce, (0, 0, 0, 512 - ce.shape[1]))
    return ce
