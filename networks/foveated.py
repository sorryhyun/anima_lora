"""Deferred-foveated merge — training-free inference acceleration.

The foveated line is **ARCHIVED** (the periphery soft blur is constitutive and
unrecoverable; digest in
`_archive/bench/foveated/report.md`, retired ship plan in
`_archive/proposals/foveated_denoise.md`, durable lessons in
`docs/findings/foveated_denoise.md`, user doc in `docs/inference/foveated.md`).
The runner stays functional and off by default (`--fovea_sigma_c 0`):

  * Above σ_c: baseline full-grid sampling — composition/identity decided
    identically to no-foveation ("deferred" is load-bearing: text drive is
    front-loaded and low bands lock by σ≈0.75, so foveating inside the
    authority window makes a *different image*).
  * During the pre-crossing steps, accumulate two free endogenous signals:
    **cfgdelta** (per-cell |v_cond − v_uncond|, prompt-aware) and **x0var**
    (x̂₀ Laplacian energy at the crossing).
  * At the σ_c crossing, build the **combo** mask: normalized signal sum →
    threshold → morphological open → close → dilate-1 → re-solve the threshold
    so the final fraction hits target. Compact, subject-following,
    per (prompt, seed).
  * Below σ_c: the DiT block stack runs on a reduced sequence — fovea tokens
    1:1, each 2×2-token periphery group averaged to one token; merged rope =
    renormalized elementwise mean of the members' (cos, sin) rows (exact
    mean-position rope for symmetric groups); broadcast before ``final_layer``.
  * Final readout: periphery read from the merged representation (avg-pool →
    bicubic up), fovea untouched. **Part of the quality contract** — skipping
    it leaves never-denoised HF detail in the periphery at decode.

Measured (bench, bare DiT, eager, 1024², 28 steps, CFG 4): fwd ×2.15,
e2e ×1.37, fovea visually baseline-identical, subject RMSE ties-or-beats a
static center rect on every prompt tested.

Invariants:
  * **Never rewrite the latent.** It stays full-res end-to-end; merging is
    what the *compute* sees.
  * The mask lives on the pooled cell grid — every merge-cell is uniformly
    fovea or periphery (no group straddles the boundary).
  * Masks must be compact blobs — scattered fovea cells lose their whole
    attention neighborhood (+47 % subject error).

Architecturally this mirrors ``networks/spd.py``: a sampler-level runner that
replaces the denoise loop and self-registers with
``library.inference.generation`` at import time. Dispatched from
``generate_body`` on ``--fovea_sigma_c > 0``. The merged forward itself runs
through the model's own ``forward`` via the ``token_merger`` kwarg
(``library/anima/models.py``) — no duplicated forward path.

Scope:
  * **Euler only.** The bench calibrated the Euler CFG loop; er_sde noise
    injection into group-shared periphery states is unvalidated. A stochastic
    sampler request falls back to Euler with a one-time warning.
  * **No SMC-CFG / FSG / CFG++ composition** — unvalidated against
    group-shared periphery velocities; warn-and-ignore (SPD posture).
    Spectrum composition is measured safe but buys ~5 % at visible periphery
    cost — mutually exclusive at dispatch, do not re-propose the compose.
  * **Composes with LoRA / Hydra / soft-tokens / P-GRAFT** — per-step adapter
    setters mirror the standard loop, and the per-Linear LoRA delta is
    token-count-agnostic.
  * ``compile_blocks`` paths are unvalidated: the merged stack introduces one
    extra token count per mask (outside the tier's ``dynamic_seq`` band), so a
    compiled model gets a one-time warning — expect a recompile per mask if
    you proceed.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from library.inference.adapters import (
    compute_and_set_hydra_fei,
    set_hydra_content,
    set_hydra_crossattn,
    set_hydra_sigma,
    set_xattn_boost_state,
)
from library.inference.sampler_context import SamplerSideChannels

log = logging.getLogger(__name__)

MASK_SOURCES = ("combo", "cfgdelta", "x0var", "rect")

# Static-rect fallback placement (cy, cx) in cell-grid fractions — the bench
# oracle default (subject faces sit upper-center on Anima's data).
RECT_CENTER = (0.38, 0.5)


class FoveatedTokenMerge:
    """Fixed-mask foveated token merge: index maps + merged rope for one
    (grid, mask) pair, applied at the block-stack boundary.

    ``mask4`` is the latent-pixel fovea mask ``(1, 1, h_lat, w_lat)``, built on
    the pool grid so every ``merge_edge``×``merge_edge``-token cell is uniformly
    fovea or periphery. ``merge_edge=2`` is the benched default; the merged-rope
    exactness argument (renormalized elementwise mean of the members'
    (cos, sin) rows = exact mean-position rope) holds for any symmetric m×m
    group — per axis the member angles pair off symmetrically about the cell
    center, so the mean vector keeps the center angle and renorm restores unit
    norm. ``merge_edge>2`` is a speed escalation only (unbenched).

    The object satisfies the ``token_merger`` protocol consumed by
    ``Anima.forward_mini_train_dit``: ``merge`` / ``unmerge`` / ``merge_rope``.
    """

    def __init__(
        self, anima, mask4: torch.Tensor, device: torch.device, merge_edge: int = 2
    ):
        p = anima.patch_spatial  # latent px per token edge (2)
        m = merge_edge
        self.m = m
        h_lat, w_lat = mask4.shape[-2:]
        self.H_t, self.W_t = h_lat // p, w_lat // p
        if self.H_t % m or self.W_t % m:
            raise ValueError(
                f"token grid {self.H_t}x{self.W_t} must be divisible by "
                f"merge_edge={m} for merge cells"
            )
        self.H_c, self.W_c = self.H_t // m, self.W_t // m

        tok_mask = mask4[0, 0, ::p, ::p] > 0.5  # (H_t, W_t) — uniform per token
        cell_fovea = tok_mask[::m, ::m]  # uniform per m×m-token cell
        tokf = tok_mask.flatten()
        cellp = (~cell_fovea).flatten()

        self.fovea_tok_idx = torch.nonzero(tokf).flatten().to(device)
        self.periph_cell_idx = torch.nonzero(cellp).flatten().to(device)
        self.n_f = int(self.fovea_tok_idx.numel())
        self.L_red = self.n_f + int(self.periph_cell_idx.numel())
        self.L_full = self.H_t * self.W_t

        # full token index → reduced index (fovea rank, or n_f + its cell's rank)
        fov_rank = torch.cumsum(tokf.long(), 0) - 1
        cell_rank = torch.cumsum(cellp.long(), 0) - 1
        ti = torch.arange(self.L_full, device=tokf.device)
        cell_of_tok = (ti // self.W_t // m) * self.W_c + (ti % self.W_t) // m
        self.gather_idx = torch.where(
            tokf, fov_rank, self.n_f + cell_rank[cell_of_tok]
        ).to(device)
        self._rope_cache: tuple | None = None

    def _cell_mean(self, flat_tok: torch.Tensor) -> torch.Tensor:
        """(..., L_full, D*) row-major token rows → (..., H_c*W_c, D*) cell means."""
        lead = flat_tok.shape[:-2]  # () for 2D rope rows, (B,) for token batches
        rest = flat_tok.shape[-1:]
        x = flat_tok.reshape(-1, self.H_t, self.W_t, *rest)
        x = x.reshape(-1, self.H_c, self.m, self.W_c, self.m, *rest).mean(dim=(2, 4))
        return x.reshape(*lead, self.H_c * self.W_c, *rest)

    def merge(self, x_BTHWD: torch.Tensor) -> torch.Tensor:
        """(B,1,H_t,W_t,D) grid → fake-5D (B,1,L_red,1,D) merged sequence."""
        B, T, H, W, D = x_BTHWD.shape
        flat = x_BTHWD.reshape(B, H * W, D)
        fov = flat[:, self.fovea_tok_idx]
        per = self._cell_mean(flat)[:, self.periph_cell_idx]
        return torch.cat([fov, per], dim=1).unsqueeze(1).unsqueeze(3)

    def unmerge(self, x_red: torch.Tensor, B: int, D: int) -> torch.Tensor:
        """fake-5D (B,1,L_red,1,D) → (B,1,H_t,W_t,D) with group-broadcast periphery."""
        flat = x_red.reshape(B, self.L_red, D)
        return flat[:, self.gather_idx].reshape(B, 1, self.H_t, self.W_t, D)

    def merge_rope(self, rope_cos_sin: tuple) -> tuple:
        """Merged (cos, sin): fovea rows pass through; periphery cell rows are the
        renormalized elementwise mean of their members — exact mean-position
        rope for a symmetric m×m group. Cached (rope is constant across steps)."""
        if self._rope_cache is not None:
            return self._rope_cache
        cos_, sin_ = rope_cos_sin  # (L_full, 1, 1, D_head), seq on dim 0
        c32, s32 = cos_.float(), sin_.float()
        cm = self._cell_mean(c32.reshape(self.L_full, -1)).reshape(
            self.H_c * self.W_c, *cos_.shape[1:]
        )[self.periph_cell_idx]
        sm = self._cell_mean(s32.reshape(self.L_full, -1)).reshape(
            self.H_c * self.W_c, *sin_.shape[1:]
        )[self.periph_cell_idx]
        norm = torch.sqrt(cm * cm + sm * sm).clamp_min(1e-6)
        cos_red = torch.cat([c32[self.fovea_tok_idx], cm / norm], dim=0).to(cos_.dtype)
        sin_red = torch.cat([s32[self.fovea_tok_idx], sm / norm], dim=0).to(sin_.dtype)
        self._rope_cache = (cos_red, sin_red)
        return self._rope_cache


# ── Score → compact mask ────────────────────────────────────────────────────────


def _shift(m: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """np.roll without wraparound (zero fill)."""
    out = np.zeros_like(m)
    ys = slice(max(dy, 0), m.shape[0] + min(dy, 0))
    xs = slice(max(dx, 0), m.shape[1] + min(dx, 0))
    yd = slice(max(-dy, 0), m.shape[0] + min(-dy, 0))
    xd = slice(max(-dx, 0), m.shape[1] + min(-dx, 0))
    out[ys, xs] = m[yd, xd]
    return out


def dilate(m: np.ndarray) -> np.ndarray:
    return m | _shift(m, 1, 0) | _shift(m, -1, 0) | _shift(m, 0, 1) | _shift(m, 0, -1)


def erode(m: np.ndarray) -> np.ndarray:
    return m & _shift(m, 1, 0) & _shift(m, -1, 0) & _shift(m, 0, 1) & _shift(m, 0, -1)


def score_to_cells(
    score: np.ndarray, target: float, tol: float = 0.02
) -> tuple[torch.Tensor, float]:
    """Bisect the pre-morphology keep-fraction so the FINAL mask (after open →
    close → dilate-1 margin) lands on the target fraction.

    Morphology is the compactness contract: open drops specks/thin structures
    (isolated fovea cells lose their attention neighborhood), close fills gaps, the dilate is the 1-cell boundary margin.
    Returns ``(cells bool (hp, wp), achieved fraction)``.
    """
    lo, hi = 0.005, 0.95
    cells = None
    for _ in range(16):
        f0 = (lo + hi) / 2
        m = score >= np.quantile(score, 1.0 - f0)
        m = dilate(erode(m))  # open: drop specks / thin structures
        m = erode(dilate(m))  # close: fill gaps
        m = dilate(m)  # 1-cell boundary margin (subject-clipping contract)
        frac = float(m.mean())
        cells = m
        if abs(frac - target) <= tol:
            break
        if frac > target:
            hi = f0
        else:
            lo = f0
    return torch.from_numpy(cells), float(cells.mean())


def rect_cells(hp: int, wp: int, frac: float, cy: float, cx: float) -> torch.Tensor:
    """Static center-rect fovea on the cell grid (the CFG=1 / degenerate-score
    fallback and the bench's placement oracle)."""
    import math

    s = math.sqrt(frac)
    rh = min(hp, max(1, round(hp * s)))
    rw = min(wp, max(1, round(wp * s)))
    top = min(max(0, round(cy * hp - rh / 2)), hp - rh)
    left = min(max(0, round(cx * wp - rw / 2)), wp - rw)
    m = torch.zeros(hp, wp, dtype=torch.bool)
    m[top : top + rh, left : left + rw] = True
    return m


def cells_to_mask4(cells: torch.Tensor, cell_px: int, device) -> torch.Tensor:
    """(hp, wp) bool cell grid → (1, 1, h_lat, w_lat) float latent-pixel mask."""
    m = cells.float()[None, None].to(device)
    return m.repeat_interleave(cell_px, -2).repeat_interleave(cell_px, -1)


# ── Runner ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def foveated_denoise(
    anima,
    latents: torch.Tensor,
    timesteps: torch.Tensor,  # unused (the loop builds t from the live σ); kept for runner-signature parity
    sigmas: torch.Tensor,
    embed: torch.Tensor,
    negative_embed: torch.Tensor,
    padding_mask: torch.Tensor,
    guidance_scale: float,
    sampler,  # ERSDESampler / LCMSampler / None — foveated forces Euler (see module docstring)
    device: torch.device,
    ctx: SamplerSideChannels,
    *,
    sigma_c: float,
    fovea_frac: float = 0.35,
    mask_source: str = "combo",
    merge_edge: int = 2,
) -> torch.Tensor:
    """Deferred-foveated Euler CFG loop.

    Above ``sigma_c``: baseline full-grid forwards, accumulating the per-cell
    mask signals. At the crossing: build the mask (``mask_source``), build the
    ``FoveatedTokenMerge``, continue merged. After the loop: bicubic merged
    readout on the periphery.

    ``fovea_frac`` floor is 0.25; ≤0.15 fails on multi-subject prompts. ``sigma_c=0.75`` is load-bearing — see module note.
    """
    if mask_source not in MASK_SOURCES:
        raise ValueError(
            f"fovea_mask_source must be one of {MASK_SOURCES}, got {mask_source!r}"
        )
    if fovea_frac < 0.25:
        log.warning(
            "--fovea_frac %.2f is below the benched floor 0.25 (subject masks "
            "start dropping faces ≤0.15 on multi-subject prompts).",
            fovea_frac,
        )
    if sampler is not None:
        log.warning(
            "--fovea_sigma_c forces Euler; the requested stochastic sampler is "
            "ignored (noise injection into group-shared periphery states is "
            "unvalidated)."
        )
    if ctx.smc_cfg is not None or ctx.fsg is not None or ctx.cfgpp_lambda is not None:
        log.warning(
            "--fovea_sigma_c does not compose with SMC-CFG / FSG / CFG++ "
            "(sampler-boundary plug-ins are unvalidated against group-shared "
            "periphery velocities); ignoring them. See docs/inference/foveated.md."
        )
    if getattr(anima, "_native_flatten", False):
        log.warning(
            "foveated merge on a compiled model is unvalidated: the merged "
            "sequence adds one token count per mask (outside the tier's "
            "dynamic-seq band) — expect a recompile pause per generation."
        )

    do_cfg = guidance_scale != 1.0
    if not do_cfg and mask_source in ("combo", "cfgdelta"):
        fallback = "x0var"
        log.warning(
            "fovea_mask_source=%s needs CFG (guidance_scale != 1.0) for the "
            "cfgdelta signal; falling back to %s.",
            mask_source,
            fallback,
        )
        mask_source = fallback

    pgraft_network = ctx.pgraft_network
    lora_cutoff_step = ctx.lora_cutoff_step
    soft_tokens_net = ctx.soft_tokens_net
    # --traj_stats: passive per-step recorder. Post-combine, pre-Euler-step,
    # on the full-grid x5 (periphery tokens carry their group-shared values
    # after the σ_c crossing — exactly the trace the gauge must flag).
    traj_stats = getattr(ctx, "traj_stats", None)
    # --xattn_boost: cond-forward-only cross-attn gain at σ ≥ band. The band
    # (default σ ≥ 0.85) sits entirely above any sane σ_c crossing, so the
    # boost acts on full-grid steps and composes trivially with the merge.
    xattn_boost = ctx.xattn_boost
    xattn_boost_band = ctx.xattn_boost_band
    xattn_renorm = ctx.xattn_boost_renorm
    xattn_renorm_frac = ctx.xattn_boost_renorm_frac

    h_lat, w_lat = int(latents.shape[-2]), int(latents.shape[-1])
    cell_px = int(anima.patch_spatial) * int(merge_edge)  # latent px per merge cell
    enabled = True
    if h_lat % cell_px or w_lat % cell_px:
        log.warning(
            "latent grid %dx%d is not divisible by the %d-px merge cell; "
            "foveation disabled for this generation (baseline loop).",
            h_lat,
            w_lat,
            cell_px,
        )
        enabled = False
    hp, wp = h_lat // cell_px, w_lat // cell_px

    def velocity(x: torch.Tensor, sigma_scalar: float, merger):
        # timestep == σ in [0,1] for Anima flow-matching. Setters mirror the
        # standard loop / networks/spd.py.
        t = x.new_full((x.shape[0],), float(sigma_scalar))
        set_hydra_sigma(anima, t)
        compute_and_set_hydra_fei(anima, x)
        merge_kw = {"token_merger": merger} if merger is not None else {}
        set_hydra_content(anima, embed)
        set_hydra_crossattn(anima, embed)
        if soft_tokens_net is not None:
            soft_tokens_net.append_postfix(
                embed, ctx.soft_tokens_embed_seqlens, timesteps=t
            )
        _pos_kw = (
            {"pooled_text_override": ctx.pooled_text_pos}
            if ctx.pooled_text_pos is not None
            else {}
        )
        _boost_step = (
            xattn_boost is not None and float(sigma_scalar) >= xattn_boost_band
        )
        if _boost_step:
            set_xattn_boost_state(
                anima, xattn_boost, renorm_mode=xattn_renorm, frac=xattn_renorm_frac
            )
        try:
            v_c = anima(x, t, embed, padding_mask=padding_mask, **_pos_kw, **merge_kw)
        finally:
            if _boost_step:
                set_xattn_boost_state(anima, 1.0)  # uncond runs at identity
        if not do_cfg:
            return v_c, None
        set_hydra_content(anima, negative_embed)
        set_hydra_crossattn(anima, negative_embed)
        if soft_tokens_net is not None:
            soft_tokens_net.append_postfix(
                negative_embed, ctx.soft_tokens_neg_seqlens, timesteps=t
            )
        _neg_kw = (
            {"pooled_text_override": ctx.pooled_text_neg}
            if ctx.pooled_text_neg is not None
            else {}
        )
        v_u = anima(
            x, t, negative_embed, padding_mask=padding_mask, **_neg_kw, **merge_kw
        )
        return v_c, v_u

    def cell_mean(px_map: torch.Tensor) -> np.ndarray:
        return F.avg_pool2d(px_map[None, None], cell_px)[0, 0].float().cpu().numpy()

    cfg_acc = torch.zeros(h_lat, w_lat, device=device)
    x0_lap: torch.Tensor | None = None
    merger: FoveatedTokenMerge | None = None
    mask4: torch.Tensor | None = None

    x5 = latents
    lat_dtype = latents.dtype  # bf16 in production; fp32 on CPU test models
    n = len(sigmas) - 1
    with tqdm(total=n, desc=f"Foveated denoising ({x5.shape[0]}x)") as pbar:
        for i in range(n):
            # P-GRAFT: disable LoRA at cutoff step (reference model takes over).
            if (
                pgraft_network is not None
                and lora_cutoff_step is not None
                and i == lora_cutoff_step
            ):
                pgraft_network.set_enabled(False)
                log.info("P-GRAFT: Disabled LoRA at step %d/%d", i, n)

            sigma = float(sigmas[i])
            if enabled and merger is None and sigma <= sigma_c:
                # ── σ_c crossing: build the mask from the accumulated signals ──
                if mask_source == "rect":
                    cells = rect_cells(hp, wp, fovea_frac, *RECT_CENTER)
                else:
                    s_cfg = cell_mean(cfg_acc)
                    s_x0 = (
                        cell_mean(x0_lap)
                        if x0_lap is not None
                        else np.zeros((hp, wp), np.float32)
                    )
                    if mask_source == "cfgdelta":
                        score = s_cfg
                    elif mask_source == "x0var":
                        score = s_x0
                    else:  # combo
                        score = s_cfg / max(s_cfg.sum(), 1e-9) + s_x0 / max(
                            s_x0.sum(), 1e-9
                        )
                    if float(score.max()) <= 0.0:
                        # No pre-crossing signal (σ_c ≥ first σ) — static fallback.
                        log.warning(
                            "no accumulated mask signal at the σ_c crossing "
                            "(sigma_c ≥ first step σ?); using static rect."
                        )
                        cells = rect_cells(hp, wp, fovea_frac, *RECT_CENTER)
                    else:
                        cells, _ = score_to_cells(score, fovea_frac)
                mask4 = cells_to_mask4(cells, cell_px, device)
                merger = FoveatedTokenMerge(anima, mask4, device, merge_edge=merge_edge)
                log.info(
                    "foveated crossing at σ=%.3f (step %d/%d, source=%s): "
                    "frac %.3f, tokens %d → %d (%.0f%%)",
                    sigma,
                    i,
                    n,
                    mask_source,
                    float(cells.float().mean()),
                    merger.L_full,
                    merger.L_red,
                    100.0 * merger.L_red / merger.L_full,
                )

            v_c, v_u = velocity(x5, sigma, merger)
            v = (
                (v_u + guidance_scale * (v_c - v_u)) if v_u is not None else v_c
            ).float()

            if traj_stats is not None:
                # sigmas[i] as the 0-d tensor (recorder contract; this loop
                # already syncs on float(sigmas[i]) but the rule is uniform).
                traj_stats.record(i, sigmas[i], x5, v, v_u)

            if enabled and merger is None:
                # Accumulate mask signals from the full-res forwards.
                if v_u is not None:
                    cfg_acc += (v_c.float() - v_u.float()).abs().mean(dim=(0, 1, 2))
                x0 = (x5.float() - sigma * v).squeeze(2)  # (B, C, H, W)
                lap = (
                    -4 * x0[..., 1:-1, 1:-1]
                    + x0[..., :-2, 1:-1]
                    + x0[..., 2:, 1:-1]
                    + x0[..., 1:-1, :-2]
                    + x0[..., 1:-1, 2:]
                )
                e = torch.zeros(h_lat, w_lat, device=device)
                e[1:-1, 1:-1] = (lap**2).mean(dim=(0, 1))
                x0_lap = e  # keep the LAST pre-crossing estimate

            x5 = (x5.float() + v * (float(sigmas[i + 1]) - sigma)).to(lat_dtype)
            pbar.update(1)

    if merger is not None:
        # Final readout: periphery read from the merged representation. Part of
        # the quality contract (bicubic, NOT nearest — nearest reads as mosaic).
        x4 = x5.float().squeeze(2)
        low = F.avg_pool2d(x4, cell_px)
        up = F.interpolate(low, size=x4.shape[-2:], mode="bicubic", align_corners=False)
        x4 = mask4 * x4 + (1.0 - mask4) * up
        x5 = x4.unsqueeze(2).to(lat_dtype)
    return x5


# Side-effect registration (mirrors the register_spd_runner call in networks/spd.py).
from library.inference.generation import register_foveated_runner  # noqa: E402

register_foveated_runner(foveated_denoise)
