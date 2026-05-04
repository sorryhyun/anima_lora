"""Validation-time SNR-t bias diagnostic.

Owns ``BiasMetrics.run_validation`` — paired forward/reverse velocity-norm
gap across the inference schedule, averaged over the val dataloader. The
late-half integrated signed gap correlates with sample quality where
FM-MSE val loss does not.

Cross-module state shared with the trainer is kept narrow: ``adapters``
(read-only view of resolved method adapters), ``padding_mask_cache``
(reused across all DiT forwards in the trainer to keep one cache key
tuple), and the trainer's static RNG switch / restore helpers. One-shot
warning flags live on the instance.

See ``docs/methods/dcw.md``.
"""

from __future__ import annotations

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
    """Owns the SNR-t bias validation pass."""

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
