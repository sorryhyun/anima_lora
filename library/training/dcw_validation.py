"""Validation-time SNR-t bias diagnostic and end-of-training DCW calibration.

Owns three pieces of trainer-adjacent logic that previously lived inline in
``train.py``:

* ``BiasMetrics.run_validation`` — paired forward/reverse velocity-norm gap
  across the inference schedule, averaged over the val dataloader. The
  late-half integrated signed gap correlates with sample quality where
  FM-MSE val loss does not.
* ``BiasMetrics.calibrate_recipe`` — end-of-training: solve a per-LoRA
  LL-band DCW recipe (anchor-selection over a small λ grid) and stuff it
  into the safetensors metadata.
* ``BiasMetrics._run_LL_pass`` — shared val-dataloader sweep used by the
  baseline + each anchor pass during calibration.

Cross-module state shared with the trainer is kept narrow: ``adapters``
(read-only view of resolved method adapters), ``padding_mask_cache``
(reused across all DiT forwards in the trainer to keep one cache key
tuple), and the trainer's static RNG switch / restore helpers. The bias
state itself (``last_bias_validation`` for cross-method baseline reuse,
plus one-shot warning flags) lives on the instance.

See ``docs/methods/dcw.md`` and ``docs/proposal/lora-dcw-proposal.md``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Callable

import torch
from tqdm import tqdm

from library.inference.adapters import clear_hydra_sigma, set_hydra_sigma
from library.runtime.device import clean_memory_on_device
from library.training.bias_metric import measure_bias_trajectory

if TYPE_CHECKING:
    from train import TrainCtx, ValCtx


BIAS_INCOMPATIBLE_ADAPTERS = frozenset({"apex", "ip_adapter", "easycontrol"})


class BiasMetrics:
    """Owns the SNR-t bias validation pass and the end-of-training DCW
    recipe calibration. See module docstring for the split of state."""

    def __init__(
        self,
        *,
        adapters: list,
        padding_mask_cache: dict,
        switch_rng: Callable[[int], tuple],
        restore_rng: Callable[[tuple], None],
    ) -> None:
        self._adapters = adapters
        self._padding_mask_cache = padding_mask_cache
        self._switch_rng = switch_rng
        self._restore_rng = restore_rng
        # Set by run_validation, consumed by calibrate_recipe to avoid an
        # extra non-probe pass when num_steps matches.
        self.last_bias_validation: dict | None = None
        self._skip_warned = False
        self._no_crossattn_warned = False

    def _incompatible_adapter_names(self) -> list[str]:
        return sorted(
            getattr(a, "name", "?")
            for a in self._adapters
            if getattr(a, "name", None) in BIAS_INCOMPATIBLE_ADAPTERS
        )

    def run_validation(
        self,
        ctx: "TrainCtx",
        val: "ValCtx",
        *,
        epoch: int,
        global_step: int,
        progress_bar,
        log_prefix: str,
        logging_fn,
    ) -> None:
        """SNR-t bias diagnostic — paired forward/reverse velocity-norm gap
        across the inference schedule, averaged over the val dataloader.

        The headline scalar (``<prefix>/integrated_signed_gap`` over the late
        half) tracks the same bias DCW patches at sampler-time, so it
        correlates with sample quality where FM-MSE val loss does not. See
        ``docs/methods/dcw.md``.

        LoRA family only — incompatible adapters short-circuit with a one-time
        warning. Requires cached ``crossattn_emb`` (LoRA training default);
        batches without it are skipped.
        """
        args = ctx.args
        if not getattr(args, "validation_bias_metric", False):
            return

        skipped = self._incompatible_adapter_names()
        if skipped:
            if not self._skip_warned:
                ctx.accelerator.print(
                    f"[bias-metric] disabled — incompatible adapter(s) active: {skipped}"
                )
                self._skip_warned = True
            return

        accelerator = ctx.accelerator
        device = accelerator.device
        weight_dtype = ctx.weight_dtype
        unwrapped_net = accelerator.unwrap_model(ctx.network)
        unwrapped_unet = accelerator.unwrap_model(ctx.unet)

        progress_bar.pause() if hasattr(progress_bar, "pause") else None
        ctx.optimizer_eval_fn()
        unwrapped_net.eval()
        if hasattr(unwrapped_unet, "switch_block_swap_for_inference"):
            unwrapped_unet.switch_block_swap_for_inference()
        # Inference semantics: full-rank T-LoRA / ReFT (training masks are
        # not applied at sampling time; see networks/CLAUDE.md).
        if hasattr(unwrapped_net, "clear_timestep_mask"):
            unwrapped_net.clear_timestep_mask()
        rng_states = self._switch_rng(
            args.validation_seed if args.validation_seed is not None else args.seed
        )

        num_steps = int(getattr(args, "validation_bias_steps", 12))
        flow_shift = float(getattr(args, "flow_shift", 1.0))
        val_seed_base = int(
            args.validation_seed if args.validation_seed is not None else args.seed
        )

        sum_v_fwd = torch.zeros(num_steps, dtype=torch.float64)
        sum_v_rev = torch.zeros(num_steps, dtype=torch.float64)
        sum_v_fwd_LL = torch.zeros(num_steps, dtype=torch.float64)
        sum_v_rev_LL = torch.zeros(num_steps, dtype=torch.float64)
        n_batches = 0
        skipped_no_crossattn = 0

        bias_pbar = tqdm(
            total=val.steps,
            smoothing=0,
            disable=not accelerator.is_local_main_process,
            desc=f"{log_prefix.split('/')[-1]} bias",
        )
        try:
            for val_step, batch in enumerate(val.dataloader):
                if val_step >= val.steps:
                    break

                latents = batch["latents"]
                if latents is None:
                    bias_pbar.update(1)
                    continue
                latents = latents.to(device, dtype=weight_dtype)
                if latents.ndim == 4:
                    latents = latents.unsqueeze(2)  # (B, C, 1, H, W)

                teo = batch.get("text_encoder_outputs_list")
                if teo is None or len(teo) < 6 or teo[4] is None:
                    skipped_no_crossattn += 1
                    bias_pbar.update(1)
                    continue
                crossattn_emb = teo[4].to(device, dtype=weight_dtype)
                t5_attn_mask = teo[3]
                if t5_attn_mask is not None:
                    t5_attn_mask = t5_attn_mask.to(device)

                B, _, _, h_lat, w_lat = latents.shape
                pm_key = (B, h_lat, w_lat, weight_dtype, device)
                padding_mask = self._padding_mask_cache.get(pm_key)
                if padding_mask is None:
                    padding_mask = torch.zeros(
                        B, 1, h_lat, w_lat, dtype=weight_dtype, device=device
                    )
                    self._padding_mask_cache[pm_key] = padding_mask

                fwd_kwargs: dict = {}
                if (
                    getattr(args, "trim_crossattn_kv", False)
                    and t5_attn_mask is not None
                ):
                    seqlens = t5_attn_mask.sum(dim=-1).to(torch.int32)
                    fwd_kwargs["crossattn_seqlens"] = seqlens
                    fwd_kwargs["max_crossattn_seqlen"] = int(seqlens.max())

                def forward_fn(
                    x_t, sigma_b, _emb=crossattn_emb, _pm=padding_mask, _kw=fwd_kwargs
                ):
                    set_hydra_sigma(unwrapped_unet, sigma_b)
                    with accelerator.autocast():
                        return unwrapped_unet(
                            x_t, sigma_b, _emb, padding_mask=_pm, **_kw
                        )

                seed = val_seed_base + 1000 * val_step
                traj = measure_bias_trajectory(
                    forward_fn,
                    latents,
                    num_steps=num_steps,
                    flow_shift=flow_shift,
                    noise_seed=seed,
                )
                sum_v_fwd += traj["v_fwd"]
                sum_v_rev += traj["v_rev"]
                sum_v_fwd_LL += traj["v_fwd_LL"]
                sum_v_rev_LL += traj["v_rev_LL"]
                n_batches += 1
                bias_pbar.update(1)
                bias_pbar.set_postfix_str(f"|gap|={float(traj['gap'].abs().sum()):.2f}")
        finally:
            bias_pbar.close()
            clear_hydra_sigma(unwrapped_unet)

        self._restore_rng(rng_states)
        ctx.optimizer_train_fn()
        unwrapped_net.train()
        if hasattr(unwrapped_unet, "switch_block_swap_for_training"):
            unwrapped_unet.switch_block_swap_for_training()
        clean_memory_on_device(device)
        if hasattr(progress_bar, "unpause"):
            progress_bar.unpause()

        if skipped_no_crossattn and not self._no_crossattn_warned:
            accelerator.print(
                f"[bias-metric] skipped {skipped_no_crossattn} batch(es) without "
                "cached crossattn_emb — set cache_text_encoder_outputs=true "
                "(default for LoRA) to enable."
            )
            self._no_crossattn_warned = True

        if n_batches == 0 or not ctx.is_tracking:
            return

        v_fwd = sum_v_fwd / n_batches
        v_rev = sum_v_rev / n_batches
        v_fwd_LL = sum_v_fwd_LL / n_batches
        v_rev_LL = sum_v_rev_LL / n_batches
        gap = v_rev - v_fwd
        gap_LL = v_rev_LL - v_fwd_LL
        late = num_steps // 2

        # Stash the final aggregated trajectory so calibrate_recipe can
        # reuse it without re-running validation. Sigmas come from the
        # same get_timesteps_sigmas call used inside measure_bias_trajectory.
        from library.inference.sampling import get_timesteps_sigmas

        _, sig_t = get_timesteps_sigmas(num_steps, flow_shift, device)
        self.last_bias_validation = {
            "gap_LL": gap_LL.tolist(),
            "sigmas": sig_t.cpu()[:num_steps].to(torch.float64).tolist(),
            "num_steps": num_steps,
            "flow_shift": flow_shift,
            "n_batches": n_batches,
        }

        logs: dict = {
            f"{log_prefix}/integrated_signed_gap": float(gap.sum()),
            f"{log_prefix}/integrated_abs_gap": float(gap.abs().sum()),
            f"{log_prefix}/integrated_signed_gap_late": float(gap[late:].sum()),
            f"{log_prefix}/integrated_abs_gap_late": float(gap[late:].abs().sum()),
            f"{log_prefix}/integrated_signed_gap_LL_late": float(gap_LL[late:].sum()),
            f"{log_prefix}/integrated_abs_gap_LL_late": float(
                gap_LL[late:].abs().sum()
            ),
        }
        for i in range(num_steps):
            logs[f"{log_prefix}/gap_step_{i:02d}"] = float(gap[i])
            logs[f"{log_prefix}/gap_LL_step_{i:02d}"] = float(gap_LL[i])
            logs[f"{log_prefix}/v_fwd_step_{i:02d}"] = float(v_fwd[i])
            logs[f"{log_prefix}/v_rev_step_{i:02d}"] = float(v_rev[i])
        logging_fn(accelerator, logs, global_step, epoch + 1)

    def _run_LL_pass(
        self,
        ctx: "TrainCtx",
        val: "ValCtx",
        *,
        num_steps: int,
        flow_shift: float,
        val_seed_base: int,
        dcw_probe_lambda: float,
        dcw_probe_schedule: str,
        pbar: "tqdm | None" = None,
        pbar_label: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """One sweep of the val dataloader returning summed LL-band
        ``v_fwd`` / ``v_rev`` trajectories (length ``num_steps``, float64
        on CPU) and the count of usable batches. With
        ``dcw_probe_lambda=0.0`` this is a baseline pass; with a nonzero λ
        it's the calibration probe pass.

        When ``pbar`` is provided, advances by 1 per val batch (skipped or
        not) and updates the postfix with ``pbar_label`` plus the running
        mean late-half ``|gap_LL|`` from this pass so far. Caller owns the
        bar's lifecycle.

        Caller is responsible for putting models in eval mode, switching
        block-swap, clearing timestep masks, and restoring RNG / train
        state — this just iterates and accumulates.
        """
        args = ctx.args
        accelerator = ctx.accelerator
        device = accelerator.device
        weight_dtype = ctx.weight_dtype
        unwrapped_unet = accelerator.unwrap_model(ctx.unet)

        sum_v_fwd_LL = torch.zeros(num_steps, dtype=torch.float64)
        sum_v_rev_LL = torch.zeros(num_steps, dtype=torch.float64)
        n_batches = 0
        late = num_steps // 2
        try:
            for val_step, batch in enumerate(val.dataloader):
                if val_step >= val.steps:
                    break
                latents = batch["latents"]
                if latents is None:
                    if pbar is not None:
                        pbar.update(1)
                    continue
                latents = latents.to(device, dtype=weight_dtype)
                if latents.ndim == 4:
                    latents = latents.unsqueeze(2)

                teo = batch.get("text_encoder_outputs_list")
                if teo is None or len(teo) < 6 or teo[4] is None:
                    if pbar is not None:
                        pbar.update(1)
                    continue
                crossattn_emb = teo[4].to(device, dtype=weight_dtype)
                t5_attn_mask = teo[3]
                if t5_attn_mask is not None:
                    t5_attn_mask = t5_attn_mask.to(device)

                B, _, _, h_lat, w_lat = latents.shape
                pm_key = (B, h_lat, w_lat, weight_dtype, device)
                padding_mask = self._padding_mask_cache.get(pm_key)
                if padding_mask is None:
                    padding_mask = torch.zeros(
                        B, 1, h_lat, w_lat, dtype=weight_dtype, device=device
                    )
                    self._padding_mask_cache[pm_key] = padding_mask

                fwd_kwargs: dict = {}
                if (
                    getattr(args, "trim_crossattn_kv", False)
                    and t5_attn_mask is not None
                ):
                    seqlens = t5_attn_mask.sum(dim=-1).to(torch.int32)
                    fwd_kwargs["crossattn_seqlens"] = seqlens
                    fwd_kwargs["max_crossattn_seqlen"] = int(seqlens.max())

                def forward_fn(
                    x_t, sigma_b, _emb=crossattn_emb, _pm=padding_mask, _kw=fwd_kwargs
                ):
                    set_hydra_sigma(unwrapped_unet, sigma_b)
                    with accelerator.autocast():
                        return unwrapped_unet(
                            x_t, sigma_b, _emb, padding_mask=_pm, **_kw
                        )

                seed = val_seed_base + 1000 * val_step
                traj = measure_bias_trajectory(
                    forward_fn,
                    latents,
                    num_steps=num_steps,
                    flow_shift=flow_shift,
                    noise_seed=seed,
                    dcw_probe_lambda=dcw_probe_lambda,
                    dcw_probe_schedule=dcw_probe_schedule,
                )
                sum_v_fwd_LL += traj["v_fwd_LL"]
                sum_v_rev_LL += traj["v_rev_LL"]
                n_batches += 1
                if pbar is not None:
                    pbar.update(1)
                    mean_gap_LL = (sum_v_rev_LL - sum_v_fwd_LL) / n_batches
                    abs_gap_late = float(mean_gap_LL[late:].abs().sum())
                    pbar.set_postfix_str(
                        f"{pbar_label} |gap_LL|_late={abs_gap_late:.2f} "
                        f"({n_batches} batch{'es' if n_batches != 1 else ''})"
                    )
        finally:
            clear_hydra_sigma(unwrapped_unet)

        return sum_v_fwd_LL, sum_v_rev_LL, n_batches

    def calibrate_recipe(
        self,
        ctx: "TrainCtx",
        val: "ValCtx",
        *,
        metadata: dict,
    ) -> None:
        """End-of-training: solve a per-LoRA LL-band DCW recipe and stuff
        it into the safetensors metadata.

        Iterates the val dataloader once with a small λ_LL=−ε DCW probe
        injected into the reverse rollout, computes per-step
        ``gap_LL_probe``, then closed-form solves for the late-half
        optimum λ_LL* against a baseline ``gap_LL``. The baseline is
        reused from the most recent ``--validation_bias_metric`` pass
        when available, otherwise an extra non-probe pass is run inline
        — so calibration works whether or not bias-metric validation was
        enabled during training.

        Skipped silently when the active adapter is bias-incompatible
        (APEX / IP-Adapter / EasyControl) or the val pass produces zero
        usable batches. See ``docs/proposal/lora-dcw-proposal.md``.
        """
        args = ctx.args
        if not getattr(args, "calibrate_dcw_recipe", True):
            return

        if self._incompatible_adapter_names():
            return

        accelerator = ctx.accelerator
        if not accelerator.is_main_process:
            return

        device = accelerator.device
        unwrapped_net = accelerator.unwrap_model(ctx.network)
        unwrapped_unet = accelerator.unwrap_model(ctx.unet)

        ctx.optimizer_eval_fn()
        unwrapped_net.eval()
        if hasattr(unwrapped_unet, "switch_block_swap_for_inference"):
            unwrapped_unet.switch_block_swap_for_inference()
        if hasattr(unwrapped_net, "clear_timestep_mask"):
            unwrapped_net.clear_timestep_mask()
        rng_states = self._switch_rng(
            args.validation_seed if args.validation_seed is not None else args.seed
        )

        anchors = list(
            getattr(args, "dcw_calibration_anchors", [-0.015, -0.022, -0.030])
        )
        anchors = [float(a) for a in anchors if float(a) != 0.0]
        if not anchors:
            accelerator.print(
                "[dcw-calibration] no nonzero anchors configured — skipping"
            )
            return
        schedule = getattr(args, "dcw_calibration_schedule", "one_minus_sigma")
        val_seed_base = int(
            args.validation_seed if args.validation_seed is not None else args.seed
        )

        flow_shift = float(getattr(args, "flow_shift", 1.0))
        num_steps = int(
            getattr(args, "dcw_calibration_steps", None)
            or getattr(args, "validation_bias_steps", 12)
        )

        # Cached bias-metric baseline (cheap per-cycle diagnostic) is only
        # reusable when its step count matches calibration's. Otherwise σ
        # samples differ and the late-half gap is not comparable across
        # the baseline-vs-anchor passes the selector reads.
        baseline = self.last_bias_validation
        if baseline is not None and baseline["num_steps"] != num_steps:
            baseline = None

        anchor_gaps: list[tuple[float, list[float]]] = []
        n_batches = 0
        n_passes = (1 if baseline is None else 0) + len(anchors)
        cal_pbar = tqdm(
            total=n_passes * val.steps,
            smoothing=0,
            disable=not accelerator.is_local_main_process,
            desc=f"dcw-calib (steps={num_steps})",
        )
        try:
            if baseline is None:
                accelerator.print(
                    f"[dcw-calibration] running baseline val pass at "
                    f"num_steps={num_steps} before {len(anchors)} anchor pass(es)"
                )
                fwd_LL_base, rev_LL_base, n_batches_base = self._run_LL_pass(
                    ctx,
                    val,
                    num_steps=num_steps,
                    flow_shift=flow_shift,
                    val_seed_base=val_seed_base,
                    dcw_probe_lambda=0.0,
                    dcw_probe_schedule=schedule,
                    pbar=cal_pbar,
                    pbar_label="baseline",
                )
                if n_batches_base == 0:
                    return
                gap_LL_base_list = (
                    (rev_LL_base - fwd_LL_base) / n_batches_base
                ).tolist()
                from library.inference.sampling import get_timesteps_sigmas

                _, sig_t = get_timesteps_sigmas(num_steps, flow_shift, device)
                baseline = {
                    "gap_LL": gap_LL_base_list,
                    "sigmas": sig_t.cpu()[:num_steps].to(torch.float64).tolist(),
                    "num_steps": num_steps,
                    "flow_shift": flow_shift,
                    "n_batches": n_batches_base,
                }

            for lam in anchors:
                sum_v_fwd_LL, sum_v_rev_LL, n_batches = self._run_LL_pass(
                    ctx,
                    val,
                    num_steps=num_steps,
                    flow_shift=flow_shift,
                    val_seed_base=val_seed_base,
                    dcw_probe_lambda=lam,
                    dcw_probe_schedule=schedule,
                    pbar=cal_pbar,
                    pbar_label=f"λ={lam:+.4f}",
                )
                if n_batches == 0:
                    accelerator.print(
                        f"[dcw-calibration] anchor λ={lam:+.4f} produced no usable "
                        "batches — skipping recipe"
                    )
                    return
                gap_LL_at_lam = ((sum_v_rev_LL - sum_v_fwd_LL) / n_batches).tolist()
                anchor_gaps.append((lam, gap_LL_at_lam))
        finally:
            cal_pbar.close()
            self._restore_rng(rng_states)
            ctx.optimizer_train_fn()
            unwrapped_net.train()
            if hasattr(unwrapped_unet, "switch_block_swap_for_training"):
                unwrapped_unet.switch_block_swap_for_training()
            clean_memory_on_device(device)

        from library.inference.dcw_calibration import select_best_anchor_LL

        try:
            selected = select_best_anchor_LL(
                baseline["gap_LL"],
                anchor_gaps,
                baseline["sigmas"],
                schedule=schedule,
            )
        except ValueError as exc:
            accelerator.print(
                f"[dcw-calibration] selector failed: {exc} — skipping recipe"
            )
            return

        recipe_payload = {
            "lambda_LL": selected["lambda_LL"],
            "schedule_LL": selected["schedule_LL"],
        }
        late_idx = num_steps // 2
        gap_LL_base = baseline["gap_LL"]
        provenance = {
            "version": 2,  # bump: was linear-response solver, now anchor selection
            "band": "LL",
            "anchors": anchors,
            "num_steps": num_steps,
            "n_batches": n_batches,
            "n_batches_baseline": baseline.get("n_batches", n_batches),
            "baseline_signed_gap_LL_late": float(sum(gap_LL_base[late_idx:])),
            "baseline_abs_gap_LL_late": float(
                sum(abs(g) for g in gap_LL_base[late_idx:])
            ),
            "baseline_gap_LL_late_loss": selected["baseline_gap_LL_late"],
            "best_gap_LL_late_loss": selected["best_gap_LL_late"],
            "candidate_losses": [
                {"lambda": c["lambda"], "loss": c["loss"]}
                for c in selected["candidates"]
            ],
            "num_late_steps": selected["num_late_steps"],
        }

        metadata["ss_dcw_recipe"] = json.dumps(recipe_payload, separators=(",", ":"))
        metadata["ss_dcw_calibration"] = json.dumps(provenance, separators=(",", ":"))

        cand_str = " ".join(
            f"λ={c['lambda']:+.4f}→{c['loss']:.1f}" for c in selected["candidates"]
        )
        winner_note = (
            "baseline (no DCW)"
            if selected["baseline_was_best"]
            else f"λ={selected['lambda_LL']:+.4f}"
        )
        accelerator.print(
            f"[dcw-calibration] selected {winner_note} ({selected['schedule_LL']}); "
            f"candidates: {cand_str}; n_batches={n_batches}"
        )
