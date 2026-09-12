"""Loss registry + composer.

The composer calls active handlers in three phases, in this reduction order
(changing the order shifts ortho/multiscale numerics):
  1. Per-sample [B]: flow_match — base FM (weighting + masked + loss_weights).
  2. Per-sample += scalar broadcast: ortho_reg, hydra_balance, functional.
  3. Scalar (after `.mean()`): multiscale — avg_pool2d MSE on pred/target.

The composer does not own forward passes — those happen in the trainer, which
stashes aux tensors on `LossContext.aux` for the composer to consume.

Aux-loss gating convention: a stage-2 handler gates on
``ctx.network._<name>_weight`` (or a documented network attr), stamped by
whoever owns the knob (network factory, network itself, or trainer). Handlers
must NOT read ``ctx.args.*`` for their weight — a knob readable from two
places is how a feature works from TOML and silently no-ops from
``--network_args`` (or vice versa). ``build_loss_composer`` and the stage-3
multiscale blend are the only spots that consult ``args``, to decide
activation only.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


def add_custom_train_arguments(
    parser: argparse.ArgumentParser, support_weighted_captions: bool = True
):
    parser.add_argument(
        "--min_snr_gamma",
        type=float,
        default=None,
        help="gamma for weighting_scheme=min_snr (None = 5.0). The loss weight "
        "peaks where SNR((1-sigma)/sigma)^2 = gamma (sigma ~0.31 at 5.0) and "
        "rolls off above it; smaller gamma flattens the low-sigma half, which "
        "after mean-1 normalization shifts weight toward higher sigma. Inert "
        "for every other weighting_scheme.",
    )
    parser.add_argument(
        "--debiased_estimation_loss",
        action="store_true",
        help="debiased estimation loss",
    )
    if support_weighted_captions:
        parser.add_argument(
            "--weighted_captions",
            action="store_true",
            default=False,
            help="Enable weighted captions in the standard style (token:1.3).",
        )


def apply_masked_loss(loss, batch) -> torch.FloatTensor:
    if "conditioning_images" in batch:
        mask_image = (
            batch["conditioning_images"].to(dtype=loss.dtype)[:, 0].unsqueeze(1)
        )
        mask_image = mask_image / 2 + 0.5
    elif "alpha_masks" in batch and batch["alpha_masks"] is not None:
        mask_image = (
            batch["alpha_masks"].to(dtype=loss.dtype).unsqueeze(1)
        )  # add channel dim
    else:
        return loss

    mask_image = torch.nn.functional.interpolate(
        mask_image, size=loss.shape[2:], mode="area"
    )
    loss = loss * mask_image
    return loss


def compute_cond_diff_weight(
    latents: torch.Tensor,
    cond_latents: torch.Tensor,
    *,
    floor: float = 0.2,
    blur_sigma: float = 1.5,
    quantile: float = 0.9,
) -> torch.Tensor:
    """Per-pixel loss weight from the cond↔target latent difference.

    For paired cond≠target tasks (near-identical except the edit region) this
    reallocates gradient toward the region that actually changes::

        d = ‖z_cond − z_target‖₂  (channel)      # (B, 1, H, W)
        d = gaussian_blur(d, blur_sigma)         # cover bubble interiors + halo
        w = floor + (1 − floor) · min(d / q_quantile(d), 1)
        w = w / mean(w)                          # per-image; effective LR unchanged

    The floor keeps a "copy everything else faithfully" anchor — w=0 outside
    the edit region would license drift there. A zero-diff pair degrades to a
    uniform all-ones map. Both inputs are 4D ``(B, C, H, W)`` latents at the
    same bucket shape.
    """
    from library.runtime.fei import gaussian_blur_2d

    eps = 1e-8
    d = (cond_latents.float() - latents.float()).pow(2).sum(1, keepdim=True).sqrt()
    d = gaussian_blur_2d(d, blur_sigma)
    scale = torch.quantile(d.flatten(1), quantile, dim=1).view(-1, 1, 1, 1)
    m = (d / scale.clamp_min(eps)).clamp(max=1.0)
    w = floor + (1.0 - floor) * m
    return w / w.flatten(1).mean(1).view(-1, 1, 1, 1).clamp_min(eps)


def apply_cond_diff_loss(loss: torch.Tensor, ctx: "LossContext") -> torch.Tensor:
    """Apply ``compute_cond_diff_weight`` to the per-element FM loss.

    No-op unless ``--cond_diff_loss`` is set AND the batch carries paired
    ``cond_latents``. Mirrors ``apply_masked_loss`` — multiplies the unreduced
    ``(B, C, H, W)`` loss before the spatial mean.
    """
    if not bool(getattr(ctx.args, "cond_diff_loss", False)):
        return loss
    cond_latents = ctx.batch.get("cond_latents")
    latents = ctx.batch.get("latents")
    if cond_latents is None or latents is None:
        return loss
    if cond_latents.ndim == 5:
        cond_latents = cond_latents.squeeze(2)
    if latents.ndim == 5:
        latents = latents.squeeze(2)
    # Needs pixel-aligned cond/target. Under free-fit a paired cond can land at
    # a different shape than the target — diff is undefined there, so skip
    # the reallocation for those samples (copy-through still trains). Warn once.
    if cond_latents.shape[-2:] != latents.shape[-2:]:
        if not getattr(apply_cond_diff_loss, "_warned_shape_mismatch", False):
            logger.warning(
                "cond_diff_loss: cond latent shape %s != target %s (free-fit "
                "cross-shape pair) — skipping diff weighting for mismatched pairs.",
                tuple(cond_latents.shape[-2:]),
                tuple(latents.shape[-2:]),
            )
            apply_cond_diff_loss._warned_shape_mismatch = True
        return loss
    w = compute_cond_diff_weight(
        latents.to(loss.device),
        cond_latents.to(loss.device),
        floor=float(getattr(ctx.args, "cond_diff_loss_floor", 0.2)),
        blur_sigma=float(getattr(ctx.args, "cond_diff_loss_blur", 1.5)),
        quantile=float(getattr(ctx.args, "cond_diff_loss_quantile", 0.9)),
    )
    if loss.ndim == 5:  # (B, C, 1, H, W) — singleton frame axis at dim 2
        w = w.unsqueeze(2)
    return loss * w.to(loss.dtype)


def get_huber_threshold_if_needed(
    args, timesteps: torch.Tensor, noise_scheduler
) -> Optional[torch.Tensor]:
    if args.loss_type == "pseudo_huber":
        b_size = timesteps.shape[0]
        return torch.full((b_size,), args.pseudo_huber_c, device=timesteps.device)
    if not (args.loss_type == "huber" or args.loss_type == "smooth_l1"):
        return None

    b_size = timesteps.shape[0]
    if args.huber_schedule == "exponential":
        # `timesteps` is σ∈[0,1] (Anima feeds the DiT time arg directly, not
        # the sd-scripts [0,1000] scale) — do NOT divide alpha by
        # num_train_timesteps, that pins the threshold flat at huber_scale.
        # Intended decay: huber_c**σ · huber_scale.
        alpha = -math.log(args.huber_c)
        result = torch.exp(-alpha * timesteps) * args.huber_scale
    elif args.huber_schedule == "snr":
        if not hasattr(noise_scheduler, "alphas_cumprod"):
            raise NotImplementedError(
                "Huber schedule 'snr' is not supported with the current model."
            )
        alphas_cumprod = torch.index_select(
            noise_scheduler.alphas_cumprod, 0, timesteps.cpu()
        )
        sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod) ** 0.5
        result = (1 - args.huber_c) / (1 + sigmas) ** 2 + args.huber_c
        result = result.to(timesteps.device)
    elif args.huber_schedule == "constant":
        result = torch.full(
            (b_size,), args.huber_c * args.huber_scale, device=timesteps.device
        )
    else:
        raise NotImplementedError(f"Unknown Huber loss schedule {args.huber_schedule}!")

    return result


def conditional_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
    reduction: str,
    huber_c: Optional[torch.Tensor] = None,
):
    if loss_type == "l2":
        loss = torch.nn.functional.mse_loss(model_pred, target, reduction=reduction)
    elif loss_type == "l1":
        loss = torch.nn.functional.l1_loss(model_pred, target, reduction=reduction)
    elif loss_type == "huber":
        if huber_c is None:
            raise NotImplementedError("huber_c not implemented correctly")
        huber_c = huber_c.view(-1, *([1] * (model_pred.ndim - 1)))
        loss = (
            2
            * huber_c
            * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        )
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    elif loss_type == "smooth_l1":
        if huber_c is None:
            raise NotImplementedError("huber_c not implemented correctly")
        huber_c = huber_c.view(-1, *([1] * (model_pred.ndim - 1)))
        loss = 2 * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    elif loss_type == "pseudo_huber":
        if huber_c is None:
            raise ValueError("pseudo_huber_c is required for pseudo_huber loss")
        huber_c = huber_c.view(-1, *([1] * (model_pred.ndim - 1)))
        loss = torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    else:
        raise NotImplementedError(f"Unsupported Loss Type: {loss_type}")
    return loss


# Internal alias — still referenced below by the composer stages.
_conditional_loss = conditional_loss


@dataclass
class LossContext:
    args: argparse.Namespace
    batch: dict
    model_pred: torch.Tensor
    target: torch.Tensor
    timesteps: torch.Tensor
    weighting: Optional[torch.Tensor]
    huber_c: Optional[torch.Tensor]
    loss_weights: torch.Tensor
    network: object
    aux: dict = field(default_factory=dict)
    is_train: bool = True


LossFn = Callable[[LossContext], torch.Tensor]


def _flow_match_loss(ctx: LossContext) -> torch.Tensor:
    """Base rectified-flow MSE with weighting, masked loss, per-sample weight.
    Returns a [B] tensor."""
    loss = _conditional_loss(
        ctx.model_pred.float(),
        ctx.target.float(),
        ctx.args.loss_type,
        "none",
        ctx.huber_c,
    )
    if ctx.weighting is not None:
        loss = loss * ctx.weighting
    if ctx.args.masked_loss or (
        "alpha_masks" in ctx.batch and ctx.batch["alpha_masks"] is not None
    ):
        loss = apply_masked_loss(loss, ctx.batch)
    loss = apply_cond_diff_loss(loss, ctx)
    loss = loss.mean(dim=list(range(1, loss.ndim)))
    loss = loss * ctx.loss_weights
    return loss


def _flow_matching_vr_loss(ctx: LossContext) -> torch.Tensor:
    """AsymFlow §5.2 control-variate FM loss: ``y² → (y + λ·z)²`` per element,
    where ``y = model_pred − target`` (grad flows here) and
    ``z = ref_pred_L − (noise − x_0^L)`` (no_grad, supplied by trainer). λ is
    estimated online as ``λ* = −Cov(y, z) / Var(z)`` on detached residuals,
    EMA'd across batches (β default 0.01); always squared-error regardless of
    ``args.loss_type`` (theory only applies there).

    Trainer contract: ``train.py::get_noise_pred_and_target`` stashes
    ``ctx.aux['vr'] = {'z': Tensor, 'state': mutable_dict}``; this handler
    updates ``state['lambda_ema']`` in place. Falls back to standard
    flow-match if the aux entry is missing (e.g. validation step).
    """
    vr_aux = ctx.aux.get("vr") or {}
    z = vr_aux.get("z")
    weight = float(getattr(ctx.args, "vr_loss_weight", 0.0) or 0.0)
    if weight <= 0.0 or z is None:
        return _flow_match_loss(ctx)

    y = ctx.model_pred.float() - ctx.target.float()
    z_f = z.float()

    # Per-batch λ_batch on detached residuals, then EMA across batches.
    with torch.no_grad():
        y_d = y.detach()
        cov = (y_d * z_f).sum()
        var = (z_f * z_f).sum().clamp_min(1e-12)
        lambda_batch = float(-(cov / var).item())

    beta = float(getattr(ctx.args, "vr_lambda_beta", 0.01) or 0.0)
    state = vr_aux.get("state")
    prev = state.get("lambda_ema") if isinstance(state, dict) else None
    if prev is None or not isinstance(prev, float):
        lambda_ema = lambda_batch
    else:
        lambda_ema = (1.0 - beta) * prev + beta * lambda_batch
    if isinstance(state, dict):
        state["lambda_ema"] = lambda_ema
        state["lambda_batch"] = lambda_batch

    diff = y + lambda_ema * z_f
    loss = diff.pow(2)
    if ctx.weighting is not None:
        loss = loss * ctx.weighting
    if ctx.args.masked_loss or (
        "alpha_masks" in ctx.batch and ctx.batch["alpha_masks"] is not None
    ):
        loss = apply_masked_loss(loss, ctx.batch)
    loss = apply_cond_diff_loss(loss, ctx)
    loss = loss.mean(dim=list(range(1, loss.ndim)))
    loss = loss * ctx.loss_weights
    return weight * loss


def _ortho_reg_loss(ctx: LossContext) -> torch.Tensor:
    weight = float(getattr(ctx.network, "_ortho_reg_weight", 0.0) or 0.0)
    if weight <= 0.0:
        return ctx.model_pred.new_zeros(())
    return weight * ctx.network.get_ortho_regularization()


def _hydra_balance_loss(ctx: LossContext) -> torch.Tensor:
    # Chimera bakes the warmup gate into its own per-pool sum (freq fires from
    # step 0); consume directly — the weight<=0 early-exit below would
    # otherwise zero the freq term during warmup.
    if getattr(ctx.network, "_use_chimera_hydra", False):
        return ctx.network.get_balance_loss()
    weight = float(getattr(ctx.network, "_balance_loss_weight", 0.0) or 0.0)
    if weight <= 0.0:
        return ctx.model_pred.new_zeros(())
    return weight * ctx.network.get_balance_loss()


def _functional_loss(ctx: LossContext) -> torch.Tensor:
    # Stamped by train.py::post_process_network (top-level training arg) —
    # see the gating convention in the module docstring.
    weight = float(getattr(ctx.network, "_functional_loss_weight", 0.0) or 0.0)
    func_loss = ctx.aux.get("func_loss")
    if weight <= 0.0 or func_loss is None:
        return ctx.model_pred.new_zeros(())
    return weight * func_loss.float()


def _repa_loss(ctx: LossContext) -> torch.Tensor:
    """REPA v2 alignment term (absolute patchwise / relational Gram).

    Computed by ``REPAMethodAdapter`` and stashed under ``aux["repa"]``; this
    handler just applies ``network._repa_weight``. Training-only.
    """
    if not ctx.is_train:
        return ctx.model_pred.new_zeros(())
    weight = float(getattr(ctx.network, "_repa_weight", 0.0) or 0.0)
    if weight <= 0.0:
        return ctx.model_pred.new_zeros(())
    repa = ctx.aux.get("repa")
    if repa is None:
        return ctx.model_pred.new_zeros(())
    return weight * repa.float()


def _soft_tokens_contrastive_loss(ctx: LossContext) -> torch.Tensor:
    """SoftREPA-style contrastive term on the soft-tokens bank.

    Computed by ``SoftTokensMethodAdapter`` and stashed under
    ``aux["soft_tokens_contrastive"]``; applies the warmup-gated weight
    ``network._contrastive_weight``. Training-only — gated on ``ctx.is_train``
    so validation FM-MSE stays a clean per-token regression metric.
    """
    if not ctx.is_train:
        return ctx.model_pred.new_zeros(())
    weight = float(getattr(ctx.network, "_contrastive_weight", 0.0) or 0.0)
    if weight <= 0.0:
        return ctx.model_pred.new_zeros(())
    con_loss = ctx.aux.get("soft_tokens_contrastive")
    if con_loss is None:
        return ctx.model_pred.new_zeros(())
    return weight * con_loss.float()


def _fera_fecl_bands(
    z: torch.Tensor, num_bands: int, fei_sigma_low_div: float
) -> list[torch.Tensor]:
    """Decompose ``z (B, C, H, W)`` into ``num_bands`` Laplacian-pyramid
    components (high → low), fp32 internally so bf16 latents don't underflow.
    ``σ_low = min(H_lat, W_lat) / fei_sigma_low_div`` keeps band semantics
    aspect-invariant; subsequent σ's double outward.
    """
    if num_bands < 2:
        raise ValueError(f"num_bands must be >= 2, got {num_bands}")
    from library.runtime.fei import gaussian_blur_2d

    z = z.float()
    h_lat, w_lat = int(z.shape[-2]), int(z.shape[-1])
    sigma_low = float(min(h_lat, w_lat)) / float(fei_sigma_low_div)
    sigmas = [sigma_low * (2.0**k) for k in range(num_bands - 1)]
    pyr = [z]
    for s in sigmas:
        pyr.append(gaussian_blur_2d(pyr[-1], s))
    bands = [pyr[k] - pyr[k + 1] for k in range(num_bands - 1)]
    bands.append(pyr[-1])
    return bands


def _fera_fecl_loss(ctx: LossContext) -> torch.Tensor:
    """FeRA Frequency-Energy Consistency Loss (Yin et al. eq. 10).

    Bandwise consistency between adapter correction ``δ = z_fera − z_base``
    and residual ``r = z_fera − z_target``, weighted by the residual's
    per-band energy share. Trainer stashes ``z_base`` (no-grad base-pass,
    routing zeroed) in ``ctx.aux['fera']``.

    NOTE: 2-band collapses Eq. 10 to a content-free scalar (two ratios summing
    to 1) — keep ``fera_fecl_weight = 0.0`` until bench-validated at 3 bands.
    """
    weight = float(
        getattr(ctx.network, "fecl_weight", None)
        or getattr(getattr(ctx.network, "cfg", None), "fera_fecl_weight", 0.0)
        or 0.0
    )
    if weight <= 0.0:
        return ctx.model_pred.new_zeros(())

    fera_aux = ctx.aux.get("fera") or {}
    z_base = fera_aux.get("z_base")
    if z_base is None:
        return ctx.model_pred.new_zeros(())

    cfg = getattr(ctx.network, "cfg", None)
    num_bands = int(
        getattr(cfg, "fera_num_bands", None) or fera_aux.get("num_bands", 3)
    )
    fei_sigma_low_div = float(
        getattr(cfg, "fei_sigma_low_div", None)
        or fera_aux.get("fei_sigma_low_div", 4.0)
    )

    def _to4(x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(2) if x.dim() == 5 else x

    z_base_4 = _to4(z_base).float()
    z_fera = _to4(ctx.model_pred).float()
    z_target = _to4(ctx.target).float()

    delta = z_fera - z_base_4
    resid = z_fera - z_target
    delta_bands = _fera_fecl_bands(delta, num_bands, fei_sigma_low_div)
    resid_bands = _fera_fecl_bands(resid, num_bands, fei_sigma_low_div)

    eps = 1e-8
    d_total = delta.flatten(1).pow(2).sum(-1).sqrt().clamp_min(eps)
    r_total = resid.flatten(1).pow(2).sum(-1).sqrt().clamp_min(eps)
    r_band_e = torch.stack([b.flatten(1).pow(2).sum(-1) for b in resid_bands], dim=-1)
    r_share = r_band_e / r_band_e.sum(-1, keepdim=True).clamp_min(eps)

    loss = z_target.new_zeros(z_target.shape[0])
    for k in range(num_bands):
        d_band = delta_bands[k].flatten(1).pow(2).sum(-1).sqrt()
        r_band = resid_bands[k].flatten(1).pow(2).sum(-1).sqrt()
        term = (d_band / d_total - r_band / r_total).pow(2)
        loss = loss + r_share[:, k] * term

    return weight * loss.mean()


def _multiscale_loss(ctx: LossContext) -> torch.Tensor:
    """Additional MSE term at 2x-downsampled resolution; the composer blends
    it via `(scalar + ms*ms_w) / (1 + ms_w)`. Returns the raw MSE."""
    ms_weight = float(getattr(ctx.args, "multiscale_loss_weight", 0.0) or 0.0)
    if ms_weight <= 0.0:
        return ctx.model_pred.new_zeros(())
    h, w = ctx.model_pred.shape[-2:]
    side_length = math.sqrt(h * w) * 8
    if side_length < 1024 * 0.9 or h < 2 or w < 2:
        return ctx.model_pred.new_zeros(())
    pred_ds = torch.nn.functional.avg_pool2d(ctx.model_pred.float(), 2)
    target_ds = torch.nn.functional.avg_pool2d(ctx.target.float(), 2)
    return torch.nn.functional.mse_loss(pred_ds, target_ds)


LOSS_REGISTRY: dict[str, LossFn] = {
    "flow_match": _flow_match_loss,
    "flow_matching_vr": _flow_matching_vr_loss,
    "ortho_reg": _ortho_reg_loss,
    "hydra_balance": _hydra_balance_loss,
    "functional": _functional_loss,
    "multiscale": _multiscale_loss,
    "fera_fecl": _fera_fecl_loss,
    "soft_tokens_contrastive": _soft_tokens_contrastive_loss,
    "repa": _repa_loss,
}


# Liveness ledger: every handler consuming a trainer/adapter-supplied aux key
# skips silently when it's missing (partial sidecar coverage, validation
# steps), so "configured ON and 100% skipped" looks identical to "working" in
# the loss curve. The composer records, per skip-if-missing loss, whether its
# aux input was present each train batch; the loop audits counts (step-N +
# run end) with a greppable ``LIVENESS:`` prefix.
#
# Contract: every LOSS_REGISTRY entry that reads ``ctx.aux`` MUST have a probe
# here mirroring its aux gate. Losses computed purely from network attrs/the
# prediction (flow_match, ortho_reg, hydra_balance, multiscale) stay out.
_LIVENESS_PROBES: dict[str, Callable[[dict], bool]] = {
    "flow_matching_vr": lambda aux: (aux.get("vr") or {}).get("z") is not None,
    "functional": lambda aux: aux.get("func_loss") is not None,
    "fera_fecl": lambda aux: (aux.get("fera") or {}).get("z_base") is not None,
    "soft_tokens_contrastive": lambda aux: (
        aux.get("soft_tokens_contrastive") is not None
    ),
    "repa": lambda aux: aux.get("repa") is not None,
}


@dataclass
class LivenessLedger:
    """Per-run consumption counts for skip-if-missing aux losses. ``seen[name]``
    counts train batches composed while ``name`` was ON; ``live[name]`` counts
    those where its aux input was consumed. Emits ``liveness/<name>`` coverage
    fractions at log cadence.
    """

    seen: dict[str, int] = field(default_factory=dict)
    live: dict[str, int] = field(default_factory=dict)

    def record(self, name: str, is_live: bool) -> None:
        self.seen[name] = self.seen.get(name, 0) + 1
        if is_live:
            self.live[name] = self.live.get(name, 0) + 1

    def dead_features(self) -> list[str]:
        """Names configured ON for at least one train batch that never fired."""
        return [
            name
            for name, seen in self.seen.items()
            if seen > 0 and self.live.get(name, 0) == 0
        ]

    def audit(self, *, where: str) -> list[str]:
        """Log one ``LIVENESS:`` line per not-fully-live feature.

        Dead (0 consumptions) → ERROR; partial coverage → WARNING with the
        percentage. Returns the dead names so the caller can decide whether
        to abort (``--liveness_strict``).
        """
        dead: list[str] = []
        for name, seen in self.seen.items():
            if seen <= 0:
                continue
            live = self.live.get(name, 0)
            if live == 0:
                dead.append(name)
                logger.error(
                    "LIVENESS: loss '%s' is configured ON but consumed its aux "
                    "input on 0/%d train batches (%s) — the producing forward/"
                    "dispatch never ran; the run is training as if the feature "
                    "were off.",
                    name,
                    seen,
                    where,
                )
            elif live < seen:
                logger.warning(
                    "LIVENESS: loss '%s' active on %d/%d train batches "
                    "(%.1f%%, %s) — partial aux coverage.",
                    name,
                    live,
                    seen,
                    100.0 * live / seen,
                    where,
                )
        return dead

    def metrics(self, ctx) -> dict[str, float]:
        del ctx
        return {
            f"liveness/{name}": self.live.get(name, 0) / seen
            for name, seen in self.seen.items()
            if seen > 0
        }

    def run_end_fields(self) -> dict:
        """Compact summary for the ``run_end`` progress event ({} when empty)."""
        if not self.seen:
            return {}
        return {
            "liveness": {
                name: {"seen": seen, "live": self.live.get(name, 0)}
                for name, seen in self.seen.items()
            }
        }


# Which stage each registered loss runs in (see module docstring).
# `flow_match` and `flow_matching_vr` are mutually exclusive — both produce
# the per-sample [B] tensor that downstream stages add into.
_STAGE_PER_SAMPLE = ("flow_match", "flow_matching_vr")
_STAGE_SCALAR_BROADCAST = (
    "ortho_reg",
    "hydra_balance",
    "functional",
    "fera_fecl",
    "soft_tokens_contrastive",
    "repa",
)
_STAGE_SCALAR_POST = ("multiscale",)
__all__ = [
    "LivenessLedger",
    "LossContext",
    "LossComposer",
    "LossFn",
    "LOSS_REGISTRY",
    "apply_cond_diff_loss",
    "build_loss_composer",
    "compute_cond_diff_weight",
    "_STAGE_PER_SAMPLE",
    "_STAGE_SCALAR_BROADCAST",
    "_STAGE_SCALAR_POST",
]


@dataclass
class LossComposer:
    """Holds the active loss entries (names in LOSS_REGISTRY) and composes
    them in-order by stage. `build_loss_composer` decides which names to
    include based on `args` + `network`.
    """

    active_losses: list[str]
    # Trainer-owned, survives across per-step composer rebuilds. None
    # (benches/tests) disables recording.
    ledger: Optional[LivenessLedger] = None

    def compose(self, ctx: LossContext) -> torch.Tensor:
        if self.ledger is not None and ctx.is_train:
            for name in self.active_losses:
                probe = _LIVENESS_PROBES.get(name)
                if probe is not None:
                    self.ledger.record(name, bool(probe(ctx.aux)))

        per_sample = ctx.model_pred.new_zeros(ctx.model_pred.shape[0])

        first = True
        for name in _STAGE_PER_SAMPLE:
            if name not in self.active_losses:
                continue
            contribution = LOSS_REGISTRY[name](ctx)
            per_sample = contribution if first else (per_sample + contribution)
            first = False
        if first:
            # exactly one of {flow_match, flow_matching_vr} must always be
            # present; defend against a caller passing an empty composer.
            raise RuntimeError(
                "LossComposer: no per-sample loss registered; "
                "one of {'flow_match', 'flow_matching_vr'} must be in active_losses"
            )

        for name in _STAGE_SCALAR_BROADCAST:
            if name not in self.active_losses:
                continue
            reg = LOSS_REGISTRY[name](ctx)
            if reg is None:
                continue
            per_sample = per_sample + reg  # broadcast scalar -> [B]

        scalar = per_sample.mean()

        if "multiscale" in self.active_losses:
            ms_weight = float(getattr(ctx.args, "multiscale_loss_weight", 0.0) or 0.0)
            if ms_weight > 0.0:
                ms_loss = LOSS_REGISTRY["multiscale"](ctx)
                if ms_loss is not None and torch.is_tensor(ms_loss) and ms_loss.numel():
                    # _multiscale_loss returns 0 when its side-length guard
                    # doesn't apply — check against zero to skip the blend then.
                    if not (ms_loss == 0).all():
                        scalar = (scalar + ms_loss * ms_weight) / (1.0 + ms_weight)

        return scalar


def build_loss_composer(
    args: argparse.Namespace,
    network: object,
    *,
    ledger: Optional[LivenessLedger] = None,
) -> LossComposer:
    """Inspect args + network and return the active LossComposer.

    Exactly one of flow_match / flow_matching_vr is active (VR wins when
    args.vr_loss_weight > 0). Others activate on their respective weight > 0;
    soft_tokens_contrastive gates on the *target* weight, not the live
    warmup-held value.
    """
    fm_name = (
        "flow_matching_vr"
        if float(getattr(args, "vr_loss_weight", 0.0) or 0.0) > 0.0
        else "flow_match"
    )
    active: list[str] = [fm_name]

    if float(getattr(network, "_ortho_reg_weight", 0.0) or 0.0) > 0.0:
        active.append("ortho_reg")
    # Chimera always activates hydra_balance — the freq pool's term fires
    # from step 0 (bypasses warmup), so we can't gate composer activation
    # on the warmup-held ``_balance_loss_weight``.
    if float(getattr(network, "_balance_loss_weight", 0.0) or 0.0) > 0.0 or bool(
        getattr(network, "_use_chimera_hydra", False)
    ):
        active.append("hydra_balance")
    if float(getattr(args, "functional_loss_weight", 0.0) or 0.0) > 0.0:
        active.append("functional")
    if float(getattr(args, "multiscale_loss_weight", 0.0) or 0.0) > 0.0:
        active.append("multiscale")
    # Mirrors the trainer's base-pass forward gate in get_noise_pred_and_target.
    fecl_weight = float(getattr(network, "fecl_weight", 0.0) or 0.0)
    if (
        fecl_weight > 0.0
        and getattr(getattr(network, "cfg", None), "use_moe_style", False)
        == "independent_A"
    ):
        active.append("fera_fecl")
    # Gate on the *target* weight — warmup may hold the live weight at 0.
    if float(getattr(network, "_contrastive_target_weight", 0.0) or 0.0) > 0.0:
        active.append("soft_tokens_contrastive")
    if float(getattr(network, "_repa_weight", 0.0) or 0.0) > 0.0:
        active.append("repa")

    return LossComposer(active_losses=active, ledger=ledger)
