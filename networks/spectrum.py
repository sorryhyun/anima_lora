"""Spectrum: Adaptive Spectral Feature Forecasting for Anima inference acceleration.

Implements the Spectrum method (Han et al., CVPR 2026, https://github.com/yangheng95/Spectrum)
for training-free diffusion sampling acceleration via Chebyshev polynomial
feature forecasting: observe block outputs at a subset of steps (actual
forwards), fit Chebyshev coefficients via ridge regression, forecast block
outputs at skipped steps, and run only t_embedder + final_layer + unpatchify
on cached steps. See docs/inference/spectrum.md.
"""

import math
import logging
from typing import Optional

import torch
from tqdm import tqdm

from library.inference.adapters import (
    clear_hydra_sigma,
    set_hydra_sigma,
    set_xattn_boost_state,
)
from library.inference import sampling as inference_utils
from library.inference.sampler_context import SamplerSideChannels
from networks.spectrum_sea import (
    l1rel,
    sea_filter,
    solve_delta_for_refresh_ratio,
    window_decision_fraction,
)

# The Chebyshev forecasters live in the pure-compute core shared with the
# ComfyUI node. Re-exported here: bench/spd and tests import them from
# ``networks.spectrum``.
from networks.spectrum_forecast import (  # noqa: F401
    DTYPE,
    ChebyshevForecaster,
    SpectrumPredictor,
    _flatten,
    _unflatten,
)

logger = logging.getLogger(__name__)

# Alias: tests import the spectrum_sea window-fraction helper under this name.
_window_decision_fraction = window_decision_fraction

# Auto-δ calibration cache for the SEA schedule. Keyed by the schedule geometry;
# the first generate with --spectrum_delta auto runs the window schedule while
# recording the SEA distance trace, derives δ to match the target refresh
# fraction, and caches it so subsequent generates use the SEA trigger at
# matched compute. Mirrored to disk (output/spectrum_sea_delta.json) so the
# calibration survives across CLI processes. See docs/inference/spectrum.md
# §"SEA schedule" ("The δ knob").
_AUTO_DELTA_CACHE: dict = {}


def _auto_delta_store_path():
    from library.env import anima_home

    return anima_home() / "output" / "spectrum_sea_delta.json"


def _auto_delta_lookup(key: tuple) -> Optional[float]:
    """In-memory then on-disk lookup of a calibrated δ for ``key``."""
    if key in _AUTO_DELTA_CACHE:
        return _AUTO_DELTA_CACHE[key]
    import json

    path = _auto_delta_store_path()
    try:
        with open(path) as f:
            disk = json.load(f)
    except (OSError, ValueError):
        return None
    val = disk.get("_".join(str(k) for k in key))
    if val is not None:
        val = float(val)
        _AUTO_DELTA_CACHE[key] = val  # promote to in-memory for this process
    return val


def _auto_delta_save(key: tuple, value: float) -> None:
    """Persist a calibrated δ to both the in-memory and on-disk caches."""
    _AUTO_DELTA_CACHE[key] = value
    import json

    path = _auto_delta_store_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path) as f:
                disk = json.load(f)
        except (OSError, ValueError):
            disk = {}
        disk["_".join(str(k) for k in key)] = value
        with open(path, "w") as f:
            json.dump(disk, f, indent=2)
    except OSError as e:
        logger.warning("Spectrum SEA: could not persist auto-delta to %s (%s)", path, e)


def _spectrum_fast_forward(
    model, timesteps_B_T: torch.Tensor, predicted_feature: torch.Tensor
) -> torch.Tensor:
    """Fast path: t_embedder -> final_layer -> unpatchify (skips all blocks)."""
    if timesteps_B_T.ndim == 1:
        timesteps_B_T = timesteps_B_T.unsqueeze(1)
    t_emb, adaln = model.t_embedder(timesteps_B_T)
    t_emb = model.t_embedding_norm(t_emb)
    # Zeros when mod guidance is disabled, so this collapses to identity
    t_emb = t_emb + model._mod_guidance_delta.unsqueeze(1)
    x = model.final_layer(predicted_feature, t_emb, adaln_lora_B_T_3D=adaln)
    return model.unpatchify(x)


def _combine_guided(
    cond_pred,
    uncond_pred,
    *,
    cfgpp_w_eff: Optional[float] = None,
    smc_cfg=None,
    guidance_scale: float = 1.0,
):
    """Merge cond/uncond predictions — plain CFG, CFG++ reweight, or SMC-CFG.

    CFG++ (``cfgpp_w_eff`` set) and SMC-CFG (``smc_cfg`` set) are mutually
    exclusive (generation.py refuses both at once). CFG++ is a pure
    σ-scheduled reweight of the same combine, so it composes with the
    spectrum cache — the forecaster stores raw cond/uncond features and only
    this merge weight differs from plain CFG.
    """
    if cfgpp_w_eff is not None:
        return uncond_pred + cfgpp_w_eff * (cond_pred - uncond_pred)
    if smc_cfg is not None:
        return smc_cfg.combine(cond_pred, uncond_pred, guidance_scale)
    return uncond_pred + guidance_scale * (cond_pred - uncond_pred)


def spectrum_denoise(
    anima,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    embed: torch.Tensor,
    negative_embed: torch.Tensor,
    padding_mask: torch.Tensor,
    guidance_scale: float,
    sampler,  # ERSDESampler / LCMSampler / None — anything with .step(latents, denoised, i)
    device: torch.device,
    ctx: SamplerSideChannels,
    *,
    window_size: float = 2.0,
    flex_window: float = 0.25,
    warmup_steps: int = 6,
    w: float = 0.3,
    m: int = 3,
    lam: float = 0.1,
    stop_caching_step: int = -1,
    calibration_strength: float = 0.0,
    schedule: str = "window",
    delta: Optional[float] = None,
    refresh_ratio: float = -1.0,
    sea_beta: float = 2.0,
    foveation=None,
) -> torch.Tensor:
    """Spectrum-accelerated denoising loop.

    Replaces step-by-step denoising with adaptive scheduling: early (high-noise)
    steps get more actual forwards; later steps are increasingly predicted via
    Chebyshev polynomial fitting.

    Args:
        window_size/flex_window: initial window N (actual forward every
            floor(N) steps) and its growth rate after each actual forward.
        warmup_steps: initial steps that always run actual forwards.
        w/m/lam: Chebyshev/Taylor blend weight, # basis functions, ridge
            regularization strength.
        stop_caching_step: force actual forwards from here on (-1 = auto:
            total_steps - 3).
        calibration_strength: residual calibration (0 = disabled) — on actual
            forwards computes residual = actual - predicted; cached steps add
            residual * strength.
        schedule: "window" (default, content-blind growing window) or "sea"
            (accumulate SEA-filtered relative-L1 distance of the input latent,
            refresh when it crosses delta — only the decision changes, the
            forecast+reuse path is unaffected; window_size/flex_window unused).
        delta: SEA threshold. None/<=0 auto-calibrates to refresh_ratio (first
            generate runs the window schedule while recording the distance
            trace, derives + caches δ; subsequent generates use SEA directly).
        refresh_ratio: target post-warmup refresh fraction for auto-δ. <=0
            matches the window schedule's own fraction at this exact geometry
            (like-for-like at matched compute) — don't hard-code, it varies
            with step count.
        sea_beta: natural-image power-law exponent for the SEA filter.
        ctx: shared conditioning side-channels (SMC-CFG/soft-tokens/P-GRAFT/
            pooled-text/FSG, see library.inference.sampler_context). ctx.fsg,
            when set, forces its scheduled σ-band steps to actual forwards and
            calibrates the latent before each.
        foveation: optional velocity-foveation adapter, duck-typed:
            force_actual/eval_view/pool_velocity/final_readout. Unvalidated
            against SMC-CFG — ignored (with a warning) while foveation is active.
    """
    # Unpack the shared side-channels into the locals the loop body uses.
    pgraft_network = ctx.pgraft_network
    lora_cutoff_step = ctx.lora_cutoff_step
    pooled_text_pos = ctx.pooled_text_pos
    pooled_text_neg = ctx.pooled_text_neg
    smc_cfg = ctx.smc_cfg
    if foveation is not None and smc_cfg is not None:
        logger.warning(
            "Spectrum foveation does not compose with SMC-CFG yet "
            "(unvalidated against pooled periphery velocities) — ignoring it."
        )
        smc_cfg = None
    soft_tokens_net = ctx.soft_tokens_net
    soft_tokens_embed_seqlens = ctx.soft_tokens_embed_seqlens
    soft_tokens_neg_seqlens = ctx.soft_tokens_neg_seqlens
    fsg = ctx.fsg
    cfgpp_lambda = ctx.cfgpp_lambda
    # --xattn_boost: applied to actual cond forwards at σ >= band (reset before
    # uncond). Forecast steps skip the blocks entirely, so in-band cached
    # predictions extrapolate from boosted cond features — consistent with the
    # boost.
    xattn_boost = ctx.xattn_boost
    xattn_boost_band = ctx.xattn_boost_band
    xattn_renorm = ctx.xattn_boost_renorm
    xattn_renorm_frac = ctx.xattn_boost_renorm_frac
    # --traj_stats: passive per-step recorder (actual AND forecast steps,
    # post-combine); generate_body owns the flush.
    traj_stats = getattr(ctx, "traj_stats", None)

    do_cfg = guidance_scale != 1.0
    num_steps = len(timesteps)

    # CFG / CFG++ / SMC-CFG combine — the single cond/uncond merge used by both
    # the actual-forward and cached-prediction branches. cfgpp_lambda and
    # smc_cfg are mutually exclusive (generation.py refuses both).
    def _combine_cfg(cond_pred, uncond_pred, i):
        w_eff = (
            inference_utils.cfgpp_guidance_weight(
                float(sigmas[i]), float(sigmas[i + 1]), cfgpp_lambda
            )
            if cfgpp_lambda is not None
            else None
        )
        return _combine_guided(
            cond_pred,
            uncond_pred,
            cfgpp_w_eff=w_eff,
            smc_cfg=smc_cfg,
            guidance_scale=guidance_scale,
        )

    # FSG carves its scheduled steps out of the cache scheduler: a calibrated
    # step is forced to an actual forward (re-observing keeps the Chebyshev
    # basis honest across the calibration-induced kink) and excluded from the
    # SEA decision domain, same as warmup/tail steps. FSG runs on the
    # cond/uncond gap, so generation.py only sets ctx.fsg under CFG.
    fsg_steps = (
        frozenset(i for i in range(num_steps) if fsg.scheduled(float(sigmas[i])))
        if fsg is not None
        else frozenset()
    )

    curr_ws = window_size
    consec_cached = 0
    fwd_count = 0
    stop_at = num_steps - 3 if stop_caching_step < 0 else stop_caching_step

    # SEA schedule (SeaCache decision metric). delta_val is None while the
    # accumulator is uncalibrated (auto mode, first generate) — the loop then
    # falls back to the window rule and records the distance trace for δ tuning.
    use_sea = schedule == "sea"
    auto_delta = use_sea and (delta is None or float(delta) <= 0.0)
    # Default the auto-δ target to the window schedule's own refresh fraction
    # at this geometry (like-for-like at matched compute).
    if use_sea and float(refresh_ratio) <= 0.0:
        refresh_ratio = _window_decision_fraction(
            num_steps, warmup_steps, stop_at, window_size, flex_window, fsg_steps
        )
    # δ rides the input-latent trajectory, so it must be re-calibrated whenever
    # step count, CFG scale, sampler rule, or resolution changes. Prompt is
    # excluded: a fixed δ yields a per-prompt refresh pattern.
    sampler_label = type(sampler).__name__ if sampler is not None else "euler"
    sea_cache_key = (
        num_steps,
        warmup_steps,
        stop_at,
        round(float(refresh_ratio), 4),
        round(float(guidance_scale), 3),
        sampler_label,
        int(latents.shape[-2]),
        int(latents.shape[-1]),
        # FSG moves the trajectory (and the forced-step set), so δ must
        # recalibrate when the band/K/Δσ/γ changes.
        (tuple(sorted(fsg_steps)), fsg.k, round(fsg.d_sigma, 3), fsg.gamma)
        if fsg is not None
        else None,
        # CFG++ reweights the combine, so it changes the cached trajectory too.
        round(cfgpp_lambda, 4) if cfgpp_lambda is not None else None,
    )
    if not use_sea:
        delta_val: Optional[float] = None
    elif auto_delta:
        delta_val = _auto_delta_lookup(sea_cache_key)
    else:
        delta_val = float(delta)
    sea_prev: Optional[torch.Tensor] = None
    sea_accum = 0.0
    sea_dists: list = []  # per-step distances over decision steps, for auto-δ

    # Forecasters, created lazily on first actual forward
    cond_fc: Optional[SpectrumPredictor] = None
    uncond_fc: Optional[SpectrumPredictor] = None

    cond_residual: Optional[torch.Tensor] = None
    uncond_residual: Optional[torch.Tensor] = None

    # Register hook on final_layer to capture block output (its input)
    captured = {}

    def _capture_pre_hook(module, args):
        # args[0] = x_B_T_H_W_D (block output, after static unpadding)
        captured["feat"] = args[0].detach().clone()

    hook = anima.final_layer.register_forward_pre_hook(_capture_pre_hook)

    try:
        with tqdm(total=num_steps, desc="Spectrum") as pbar:
            for i, t in enumerate(timesteps):
                # P-GRAFT cutoff
                if (
                    pgraft_network is not None
                    and lora_cutoff_step is not None
                    and i == lora_cutoff_step
                ):
                    pgraft_network.set_enabled(False)
                    logger.info(f"P-GRAFT: Disabled LoRA at step {i}/{num_steps}")

                # FSG: pre-step latent calibration toward the golden path, run
                # *before* the SEA metric so both the accumulator and the
                # forecaster see the calibrated trajectory. The calibration's own
                # internal anima() forwards transiently overwrite captured["feat"],
                # but the real per-step forward below is the last anima() call
                # before we read it. Forced to an actual forward below.
                fsg_forced = fsg is not None and i in fsg_steps
                if fsg_forced:
                    latents = fsg.calibrate(
                        anima,
                        latents,
                        float(sigmas[i]),
                        i,
                        embed,
                        negative_embed,
                        padding_mask,
                        guidance_scale,
                        pooled_pos=pooled_text_pos,
                        pooled_neg=pooled_text_neg,
                    )

                # Foveation: one forced actual forward at the σ_c crossing — the
                # composite eval-view kinks the feature trajectory, so the
                # Chebyshev basis must re-observe there. Treated like FSG: no
                # window advance, excluded from the SEA trace.
                fov_forced = foveation is not None and foveation.force_actual(
                    i, float(sigmas[i])
                )

                # SEA: accumulate the SEA-filtered relative-L1 distance of the
                # input latent (shared across cond/uncond). One FFT/iFFT per
                # step — negligible vs a block forward.
                if use_sea:
                    sea_now = sea_filter(latents[:, :, 0], float(sigmas[i]), sea_beta)
                    if sea_prev is not None:
                        d = l1rel(sea_now, sea_prev)
                        sea_accum += d
                        # FSG steps are forced actual; exclude their distance
                        # from the auto-δ decision trace (matches the window baseline).
                        if warmup_steps <= i < stop_at and not (
                            fsg_forced or fov_forced
                        ):
                            sea_dists.append(d)
                    sea_prev = sea_now

                # Decide: actual forward or cached prediction? FSG-scheduled
                # steps are forced actual regardless of window/SEA rule.
                if i < warmup_steps or i >= stop_at or fsg_forced or fov_forced:
                    actual = True
                elif use_sea and delta_val is not None:
                    actual = sea_accum >= delta_val
                else:
                    actual = (consec_cached + 1) % max(1, math.floor(curr_ws)) == 0

                t_exp = t.expand(latents.shape[0])
                set_hydra_sigma(anima, t_exp)

                if actual:
                    # Foveation eval-view: below σ_c the DiT evaluates on the
                    # merged-token composite; the latent itself is never
                    # rewritten (stays on-manifold).
                    model_x = (
                        foveation.eval_view(latents, float(sigmas[i]))
                        if foveation is not None
                        else latents
                    )
                    if soft_tokens_net is not None:
                        soft_tokens_net.append_postfix(
                            embed, soft_tokens_embed_seqlens, timesteps=t_exp
                        )
                    _boost_step = (
                        xattn_boost is not None and float(sigmas[i]) >= xattn_boost_band
                    )
                    if _boost_step:
                        set_xattn_boost_state(
                            anima,
                            xattn_boost,
                            renorm_mode=xattn_renorm,
                            frac=xattn_renorm_frac,
                        )
                    with torch.no_grad():
                        _pos_kw = (
                            {"pooled_text_override": pooled_text_pos}
                            if pooled_text_pos is not None
                            else {}
                        )
                        noise_pred = anima(
                            model_x, t_exp, embed, padding_mask=padding_mask, **_pos_kw
                        )
                    if _boost_step:
                        set_xattn_boost_state(anima, 1.0)  # uncond runs at identity
                    feat = captured["feat"]
                    if cond_fc is None:
                        cond_fc = SpectrumPredictor(
                            m, lam, w, device, feat.shape[1:], num_steps
                        )
                    # Residual calibration: measure prediction error before updating
                    if calibration_strength > 0 and cond_fc.cheb.t_buf.numel() >= 2:
                        cond_residual = feat - cond_fc.predict(float(i))
                    cond_fc.update(float(i), feat)

                    if do_cfg:
                        if soft_tokens_net is not None:
                            soft_tokens_net.append_postfix(
                                negative_embed,
                                soft_tokens_neg_seqlens,
                                timesteps=t_exp,
                            )
                        with torch.no_grad():
                            _neg_kw = (
                                {"pooled_text_override": pooled_text_neg}
                                if pooled_text_neg is not None
                                else {}
                            )
                            uncond_noise_pred = anima(
                                model_x,
                                t_exp,
                                negative_embed,
                                padding_mask=padding_mask,
                                **_neg_kw,
                            )
                        ufeat = captured["feat"]
                        if uncond_fc is None:
                            uncond_fc = SpectrumPredictor(
                                m, lam, w, device, ufeat.shape[1:], num_steps
                            )
                        if (
                            calibration_strength > 0
                            and uncond_fc.cheb.t_buf.numel() >= 2
                        ):
                            uncond_residual = ufeat - uncond_fc.predict(float(i))
                        uncond_fc.update(float(i), ufeat)
                        noise_pred = _combine_cfg(noise_pred, uncond_noise_pred, i)

                    # Advance schedule (only post-warmup to avoid inflating
                    # window). FSG/foveation-forced steps don't advance it —
                    # they're external forcings, not window-driven refreshes.
                    if i >= warmup_steps and not (fsg_forced or fov_forced):
                        curr_ws = round(curr_ws + flex_window, 3)
                    consec_cached = 0
                    sea_accum = 0.0  # refresh resets the SEA accumulator (Eq. 8)
                    fwd_count += 1
                    pbar.set_postfix(mode="fwd", ws=f"{curr_ws:.1f}", n=fwd_count)

                else:
                    # Cached step: predict features, skip all blocks.
                    with torch.no_grad():
                        pred_feat = cond_fc.predict(float(i))
                        if cond_residual is not None:
                            pred_feat = pred_feat + calibration_strength * cond_residual
                        noise_pred = _spectrum_fast_forward(anima, t_exp, pred_feat)

                        if do_cfg:
                            upred_feat = uncond_fc.predict(float(i))
                            if uncond_residual is not None:
                                upred_feat = (
                                    upred_feat + calibration_strength * uncond_residual
                                )
                            uncond_noise_pred = _spectrum_fast_forward(
                                anima, t_exp, upred_feat
                            )
                            noise_pred = _combine_cfg(noise_pred, uncond_noise_pred, i)

                    consec_cached += 1
                    pbar.set_postfix(mode="cached", n=fwd_count)

                # Foveation: below σ_c every emitted v (actual and forecast) is
                # periphery-pooled. Post-unpatchify, so forecaster state is untouched.
                if foveation is not None:
                    noise_pred = foveation.pool_velocity(noise_pred, float(sigmas[i]))

                if traj_stats is not None:
                    traj_stats.record(
                        i,
                        sigmas[i],  # tensor, NOT float() — float() = stream sync
                        latents,
                        noise_pred,
                        uncond_noise_pred if do_cfg else None,
                    )
                denoised = latents.float() - sigmas[i] * noise_pred.float()
                if sampler is not None:
                    new_latents = sampler.step(latents, denoised, i)
                else:
                    new_latents = inference_utils.step(latents, noise_pred, sigmas, i)

                latents = new_latents.to(latents.dtype)

                pbar.update()

        # Foveation final readout: periphery read from the merged
        # representation once before decode, stripping never-denoised HF that
        # pooled velocities cannot remove.
        if foveation is not None:
            latents = foveation.final_readout(latents)

        # Auto-δ: derive the δ matching the target refresh fraction from this
        # generate's window-scheduled trace, and cache it for subsequent generates.
        if auto_delta and delta_val is None and sea_dists:
            new_delta = solve_delta_for_refresh_ratio(sea_dists, refresh_ratio)
            _auto_delta_save(sea_cache_key, new_delta)
            logger.info(
                "Spectrum SEA: auto-calibrated delta=%.4g (target refresh_ratio="
                "%.2f over %d decision steps); this generate used the window "
                "schedule — subsequent generates (incl. new processes, via "
                "%s) use the SEA trigger.",
                new_delta,
                refresh_ratio,
                len(sea_dists),
                _auto_delta_store_path().name,
            )

        speedup = num_steps / max(1, fwd_count)
        cfg_label = " (x2 for CFG)" if do_cfg else ""
        sched_label = (
            f", schedule=sea (delta={delta_val:.4g})"
            if use_sea and delta_val is not None
            else (", schedule=sea (calibrating)" if use_sea else "")
        )
        # FSG spends 3·K extra forwards per scheduled step — call it out so
        # the speedup isn't misread.
        fsg_label = (
            f", fsg={len(fsg_steps)} steps×K{fsg.k} (+{3 * fsg.k * len(fsg_steps)} fwd)"
            if fsg is not None and fsg_steps
            else ""
        )
        cfgpp_label = f", cfg++ λ={cfgpp_lambda}" if cfgpp_lambda is not None else ""
        logger.info(
            f"Spectrum: {fwd_count}/{num_steps} actual forwards "
            f"({speedup:.2f}x theoretical speedup{cfg_label})"
            f"{sched_label}{fsg_label}{cfgpp_label}"
        )

    finally:
        if xattn_boost is not None:
            set_xattn_boost_state(anima, 1.0)
        clear_hydra_sigma(anima)
        if pgraft_network is not None and lora_cutoff_step is not None:
            pgraft_network.set_enabled(True)
        hook.remove()

    return latents


# Register with library.inference.generation so generate() can dispatch here
# without a hard import edge from generation.py back into this file.
from library.inference.generation import register_spectrum_runner  # noqa: E402

register_spectrum_runner(spectrum_denoise)
