"""The flow-matching loss and its terms: edit-region weights, tone, preservation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from library.moss.training.data import NUM_TRAIN_TIMESTEPS


def preserve_loss(
    transformer, network, noisy, timestep, neg_embeds, ref=None, attached=None
):
    """MSE between the adapter's and the base model's prediction on ``noisy`` for
    the trigger-stripped ``neg_embeds (1, S', 4096)``: the base one under
    ``multiplier`` 0 without grad, the adapter's with. ``attached`` (the block
    swapper): the no-grad forward has no backward to walk the swapped tail back,
    so its layout is restored by hand before the next forward."""
    network.set_multiplier(0.0)
    with torch.no_grad():
        base = transformer(noisy, timestep, neg_embeds, return_dict=False, ref=ref)[0]
    if attached is not None:
        attached.prepare(free_cache=False)
    network.set_multiplier(1.0)
    pred = transformer(noisy, timestep, neg_embeds, return_dict=False, ref=ref)[0]
    return F.mse_loss(pred.float(), base.float())


def edit_region(
    latent: torch.Tensor,
    ref: torch.Tensor,
    threshold: float,
    min_tokens: int = 5,
    dilate: int = 1,
) -> torch.Tensor:
    """``(h, w)`` bool: the tokens where ``latent`` differs from a same-size
    ``ref`` by more than ``threshold`` (RMS over channels), components under
    ``min_tokens`` dropped, grown by ``dilate``."""
    import numpy as np
    from scipy import ndimage

    diff = (latent.float() - ref.float()).pow(2).mean(0).sqrt()
    m = (diff > threshold).numpy()
    eight = np.ones((3, 3), dtype=bool)
    labels, n = ndimage.label(m, structure=eight)
    sizes = ndimage.sum(m, labels, range(1, n + 1))
    keep = np.isin(labels, 1 + np.flatnonzero(np.asarray(sizes) >= min_tokens))
    if dilate and keep.any():
        keep = ndimage.binary_dilation(keep, structure=eight, iterations=dilate)
    return torch.from_numpy(keep)


def region_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    region: torch.Tensor,
    region_weight: float,
    rest_weight: float,
    parts: dict | None = None,
) -> torch.Tensor:
    """The mean over tokens of the per-token MSE times ``region_weight`` on
    ``region`` and ``rest_weight`` elsewhere (not renormalised). ``parts``
    receives the plain mean over each side (``region`` / ``rest``) when both are
    non-empty."""
    err = (pred.float() - target.float()).pow(2).mean(1)[0]
    if parts is not None and region.any() and not region.all():
        parts.update(region=err[region].mean().item(), rest=err[~region].mean().item())
    return (torch.where(region, region_weight, rest_weight) * err).mean()


def tone_projection(path, b_weight: float = 1.0) -> torch.Tensor:
    """``(3, C)`` least-squares map from a per-channel latent offset to Lab
    ``(L, a, b)``, from the per-unit directions in ``path``; the b row scaled by
    ``b_weight``."""
    from safetensors.torch import load_file

    dirs = load_file(str(path))
    proj = torch.linalg.pinv(
        torch.stack([dirs[k] for k in ("L", "a", "b")], 1).double()
    ).float()
    proj[2] *= b_weight
    return proj


def tone_loss(
    err: torch.Tensor, rest: torch.Tensor, pool: int, proj: torch.Tensor | None = None
) -> torch.Tensor:
    """``err (1,C,h,w)`` (x̂₀ − x₀) averaged over ``rest (h,w)`` per ``pool``×``pool``
    cell (``pool`` 0 = one cell), cells under half coverage skipped. With ``proj``
    the cell means map to Lab and the term is their squared norm (ΔE²); without it,
    the mean square over channels. Mean over cells; 0 when no cell qualifies."""
    m = rest[None, None].to(err.dtype)
    if pool == 0:
        num, cov = (err * m).mean((2, 3), keepdim=True), m.mean((2, 3), keepdim=True)
        ok = cov[0, 0] > 0
    else:
        num = F.avg_pool2d(err * m, pool, ceil_mode=True)
        cov = F.avg_pool2d(m, pool, ceil_mode=True)
        ok = cov[0, 0] >= 0.5
    if not ok.any():
        return err.sum() * 0
    cells = (num / cov.clamp(min=1e-6))[0][:, ok]
    if proj is None:
        return cells.pow(2).mean()
    return (proj.to(cells) @ cells).pow(2).sum(0).mean()


def training_loss(
    transformer,
    latent,
    embeds,
    sigma: float,
    gen=None,
    ref=None,
    region=None,
    region_weight: float = 1.0,
    rest_weight: float = 1.0,
    parts: dict | None = None,
    tone: dict | None = None,
    state: dict | None = None,
):
    """``latent (1,128,h,w)`` fp32, ``embeds (1,S,4096)`` at true length,
    ``ref (1,128,h',w')`` an optional clean reference latent (never noised),
    ``region (h,w)`` bool on the latent's device: :func:`region_mse` instead of
    the plain MSE. ``tone``: ``rest`` mask, ``pool``, ``proj``, ``weight`` for
    :func:`tone_loss` on ``x̂₀ − x₀ = σ·(target − pred)``, added at ``weight``
    and logged in ``parts`` as ``tone``. ``state`` receives ``noisy`` and
    ``timestep`` (:func:`preserve_loss` runs on the same ones)."""
    noise = torch.randn(
        latent.shape, generator=gen, device=latent.device, dtype=latent.dtype
    )
    noisy = (1.0 - sigma) * latent + sigma * noise
    target = noise - latent
    timestep = torch.tensor([NUM_TRAIN_TIMESTEPS * sigma], device=latent.device)
    if state is not None:
        state.update(noisy=noisy, timestep=timestep)
    pred = transformer(noisy, timestep, embeds, return_dict=False, ref=ref)[0]
    if region is not None:
        loss = region_mse(pred, target, region, region_weight, rest_weight, parts)
    else:
        loss = F.mse_loss(pred.float(), target.float())
    if tone is None:
        return loss
    term = tone_loss(
        sigma * (target.float() - pred.float()),
        tone["rest"],
        tone["pool"],
        tone["proj"],
    )
    if parts is not None:
        parts["tone"] = term.item()
        if pred.requires_grad:
            # Output-space gradient norms, the tone one at the configured weight
            # even before tone_start: one backward cannot split them per param.
            g_fm, g_tone = (
                torch.autograd.grad(x, pred, retain_graph=True)[0].float()
                for x in (loss, term)
            )
            g_tone = g_tone * tone.get("log_weight", tone["weight"])
            parts.update(
                g_fm=g_fm.norm().item(),
                g_tone=g_tone.norm().item(),
                g_cos=F.cosine_similarity(g_fm.flatten(), g_tone.flatten(), 0).item(),
            )
    return loss + tone["weight"] * term if tone["weight"] else loss
