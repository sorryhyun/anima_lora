"""Trajectory measurement: forward and reverse velocity passes."""

from __future__ import annotations

import numpy as np
import torch

from library.inference.adapters import set_hydra_sigma
from scripts.dcw.haar import BANDS, apply_dcw_LL_only_batched, haar_band_norms_batched


def _padding_mask(x_0: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.zeros(
        1, 1, x_0.shape[-2], x_0.shape[-1], dtype=torch.bfloat16, device=device
    )


@torch.no_grad()
def encode_uncond_embed(
    anima,
    text_encoder,
    negative_prompt: str,
    device: torch.device,
) -> torch.Tensor:
    """Encode the unconditional crossattn embed for CFG.

    Mirrors ``library/inference/text.py:80-94`` — tokenize → encode →
    ``anima._preprocess_text_embeds(...)`` → zero-pad to 512. Returns
    a single (1, 512, 1024) bf16 tensor on ``device``.

    Strategies are assumed primed by the caller (the existing
    transient text-encoder block in ``main`` does this).
    """
    from library.anima import text_strategies

    tok = text_strategies.TokenizeStrategy.get_strategy()
    enc = text_strategies.TextEncodingStrategy.get_strategy()
    tokens = tok.tokenize(negative_prompt)
    embed = enc.encode_tokens(tok, [text_encoder], tokens)
    crossattn, _ = anima._preprocess_text_embeds(
        source_hidden_states=embed[0].to(anima.device),
        target_input_ids=embed[2].to(anima.device),
        target_attention_mask=embed[3].to(anima.device),
        source_attention_mask=embed[1].to(anima.device),
    )
    crossattn[~embed[3].bool()] = 0
    if crossattn.shape[1] < 512:
        crossattn = torch.nn.functional.pad(
            crossattn, (0, 0, 0, 512 - crossattn.shape[1])
        )
    return crossattn.to(device, dtype=torch.bfloat16)


@torch.no_grad()
def _cfg_velocity(
    anima,
    x: torch.Tensor,
    t: torch.Tensor,
    embed: torch.Tensor,
    pad: torch.Tensor,
    *,
    embed_uncond: torch.Tensor | None,
    cfg_scale: float,
) -> torch.Tensor:
    """One DiT forward (CFG=1) or **batched** uncond+cond forward (CFG > 1).

    Combination matches ``library/inference/generation.py``:
        v = v_uncond + s · (v_cond − v_uncond).

    Under CFG > 1 the [uncond, cond] pair is concatenated along the
    batch axis and run as a single forward at batch = 2·B; this halves
    the per-step kernel-launch + attention setup overhead vs two
    separate calls and is the dominant speedup for prod-env (CFG=4)
    bench runs. ``embed_uncond`` is broadcast to match the cond batch
    when needed.
    """
    if cfg_scale == 1.0 or embed_uncond is None:
        return anima(x, t, embed, padding_mask=pad)
    B = x.shape[0]
    embed_u = (
        embed_uncond.expand(B, -1, -1).contiguous()
        if embed_uncond.shape[0] != B
        else embed_uncond
    )
    x2 = torch.cat([x, x], dim=0)
    t2 = torch.cat([t, t], dim=0)
    e2 = torch.cat([embed_u, embed], dim=0)
    p2 = torch.cat([pad, pad], dim=0)
    v = anima(x2, t2, e2, padding_mask=p2)
    v_uncond = v[:B]
    v_cond = v[B:]
    return v_uncond + cfg_scale * (v_cond - v_uncond)


@torch.no_grad()
def measure_forward_norms(
    anima,
    x_0: torch.Tensor,
    embed: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    noise_seeds: list[int],
    device: torch.device,
    embed_uncond: torch.Tensor | None = None,
    cfg_scale: float = 1.0,
) -> list[tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Forward branch only: ‖v_θ((1−σ)x_0 + σε, σ)‖ at every step.

    Runs all ``len(noise_seeds)`` trajectories as a single DiT forward
    per step at batch B = ``len(noise_seeds)`` (or 2·B under CFG > 1).
    The same (x_0, embed) is shared across rows — only the per-step
    noise ε differs. Bit-equivalent to a per-seed serial loop because
    each row uses its own CPU generator (same RNG sequence as before).

    Under CFG > 1 each step runs the cond+uncond pair as a single
    batched forward (see ``_cfg_velocity``); norms / bands are taken on
    the combined velocity. Per-step ``‖v‖`` and band norms accumulate
    on-device and sync once at trajectory end.

    Returns one (norms, bands) tuple per seed, in input order.
    """
    B = len(noise_seeds)
    n_steps = len(sigmas) - 1
    pad_one = _padding_mask(x_0, device)
    pad = pad_one.expand(B, -1, -1, -1).contiguous()
    embed_b = embed.expand(B, -1, -1).contiguous()
    x_0_b = x_0.expand(B, -1, -1, -1, -1).contiguous()
    gens = [torch.Generator(device="cpu").manual_seed(s + 10_000) for s in noise_seeds]

    norms_gpu = torch.zeros(B, n_steps, dtype=torch.float32, device=device)
    bands_gpu = torch.zeros(B, n_steps, 4, dtype=torch.float32, device=device)
    for i in range(n_steps):
        sigma_i = float(sigmas[i])
        # σ is shared across rows; pass shape (1,) to the Hydra router
        # (state is a scalar) and shape (B,) to the model forward.
        t_one = torch.full((1,), sigma_i, device=device, dtype=torch.bfloat16)
        t_b = torch.full((B,), sigma_i, device=device, dtype=torch.bfloat16)
        eps_rows = [torch.randn(x_0.shape, generator=g) for g in gens]
        eps = torch.cat(eps_rows, dim=0).to(device, torch.bfloat16)
        x_t = (1.0 - sigma_i) * x_0_b + sigma_i * eps
        set_hydra_sigma(anima, t_one)
        v = _cfg_velocity(
            anima,
            x_t,
            t_b,
            embed_b,
            pad,
            embed_uncond=embed_uncond,
            cfg_scale=cfg_scale,
        )
        norms_gpu[:, i] = v.float().flatten(start_dim=1).norm(dim=1)
        bands_gpu[:, i] = haar_band_norms_batched(v)

    norms_np = norms_gpu.cpu().numpy().astype(np.float64)
    bands_np = bands_gpu.cpu().numpy().astype(np.float64)

    out: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
    for r in range(B):
        bands_dict = {b: bands_np[r, :, j] for j, b in enumerate(BANDS)}
        out.append((norms_np[r], bands_dict))
    return out


@torch.no_grad()
def run_reverse_batched(
    anima,
    x_0: torch.Tensor,
    embed: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    noise_seeds: list[int],
    dcw_lams: list[float],
    device: torch.device,
    embed_uncond: torch.Tensor | None = None,
    cfg_scale: float = 1.0,
    return_final: bool = False,
) -> (
    list[tuple[np.ndarray, dict[str, np.ndarray]]]
    | tuple[list[tuple[np.ndarray, dict[str, np.ndarray]]], torch.Tensor]
):
    """Run N reverse trajectories in parallel along batch, where each row
    is the (seed, λ) pair ``(noise_seeds[r], dcw_lams[r])``.

    All rows share (x_0, embed, schedule); they diverge via per-row
    initial noise ε₀ (from ``noise_seeds[r]``) and DCW correction λ
    (from ``dcw_lams[r]``). Each step does **one** DiT forward at
    batch = N (or 2×N under CFG > 1, see ``_cfg_velocity``).

    Two production patterns share this signature:

    * **λ-sweep mode** (``--dcw_sweep``): N = #λ values, all rows
      share the same seed (caller passes ``noise_seeds=[s]*N``).
      Bit-equivalent to the previous shared-noise implementation —
      same generator state across rows yields identical ε₀.
    * **Seed-batched mode** (``make dcw``, no sweep): N = ``--n_seeds``,
      ``dcw_lams = [0.0]*N`` (no correction), per-row seeds. Replaces
      the per-seed serial loop with a single batched forward per step.

    DCW correction (when ``λ != 0``): LL-only with
    ``scalar_i = λ · (1 − σ_i)``, applied to ``(prev − x0_pred)``
    independently per row.

    Returns one (norms, bands) tuple per row, in input order.
    ``norms`` shape: (n_steps,). ``bands[b]`` shape: (n_steps,).

    If ``return_final`` is True, additionally returns the final
    reverse-trajectory latent per row (the σ → 0 endpoint), shape
    ``(n_rows, *x_0.shape[1:])``, float32 on CPU. Suitable for VAE
    decode.
    """
    if len(noise_seeds) != len(dcw_lams):
        raise ValueError(
            f"noise_seeds (len={len(noise_seeds)}) must match dcw_lams "
            f"(len={len(dcw_lams)}) — each row is one (seed, λ) pair."
        )
    n_rows = len(dcw_lams)
    n_steps = len(sigmas) - 1
    pad_one = _padding_mask(x_0, device)
    pad = pad_one.expand(n_rows, -1, -1, -1).contiguous()
    embed_b = embed.expand(n_rows, -1, -1).contiguous()

    gens = [torch.Generator(device="cpu").manual_seed(s) for s in noise_seeds]
    x_hat0_rows = [torch.randn(x_0.shape, generator=g) for g in gens]
    x_hat = torch.cat(x_hat0_rows, dim=0).to(device, torch.bfloat16)

    lams_t = torch.tensor(dcw_lams, dtype=torch.float32, device=device)

    norms_gpu = torch.zeros(n_steps, n_rows, dtype=torch.float32, device=device)
    bands_gpu = torch.zeros(n_steps, n_rows, 4, dtype=torch.float32, device=device)

    for i in range(n_steps):
        sigma_i = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])
        # σ is shared across rows; pass shape (1,) to the Hydra router
        # (state is a scalar) and shape (n_rows,) to the model forward
        # to match the batch.
        t_one = torch.full((1,), sigma_i, device=device, dtype=torch.bfloat16)
        t_b = torch.full((n_rows,), sigma_i, device=device, dtype=torch.bfloat16)

        set_hydra_sigma(anima, t_one)
        v = _cfg_velocity(
            anima,
            x_hat,
            t_b,
            embed_b,
            pad,
            embed_uncond=embed_uncond,
            cfg_scale=cfg_scale,
        )
        norms_gpu[i] = v.float().flatten(start_dim=1).norm(dim=1)
        bands_gpu[i] = haar_band_norms_batched(v)

        v_f = v.float()
        prev = x_hat.float() + (sigma_next - sigma_i) * v_f
        if sigma_next > 0.0:
            x0_pred = x_hat.float() - sigma_i * v_f
            scalars = lams_t * (1.0 - sigma_i)
            prev = apply_dcw_LL_only_batched(prev, x0_pred, scalars)
        x_hat = prev.to(torch.bfloat16)

    norms_np = norms_gpu.cpu().numpy().astype(np.float64)
    bands_np = bands_gpu.cpu().numpy().astype(np.float64)

    out: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
    for j in range(n_rows):
        bands_dict = {b: bands_np[:, j, k] for k, b in enumerate(BANDS)}
        out.append((norms_np[:, j], bands_dict))

    if return_final:
        return out, x_hat.float().cpu()
    return out
