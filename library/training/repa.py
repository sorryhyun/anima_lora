"""REPA v2 — patchwise / relational alignment against a vision encoder.

Two loss forms selectable per run (see ``docs/methods/repa.md``):

- **Arm A — absolute patchwise**: project DiT block tokens → encoder dim with
  a small MLP head, cosine per token in fp32, mean.
- **Arm B — relational (Gram)**: L2-normalize tokens on each side, ``G =
  F̂F̂ᵀ``, ``loss = MSE(G_dit, G_pe)``. No head needed — similarities live
  within each space, so dims never need to match (relational KD).

Both arms align a noisy-input mid-block feature to the *clean*-image encoder
target, captured from the **primary** forward (no second DiT forward → no
block-swap offloader desync).

Two non-obvious mechanics:

1. **Grid match.** The encoder token grid ``(gh, gw)`` varies per aspect
   bucket and is ambiguous from token count alone (the bucket table is
   aspect-symmetric), so it's disambiguated by the latent's orientation
   (portrait vs landscape). The DiT side is adaptive-avg-pooled down to that
   grid so both sides are ``N = gh*gw`` tokens in the same row-major order.

2. **native_flatten layout.** Under ``compile_blocks()`` the captured block
   output is fake-5D ``(B, 1, seq, 1, D)``, not eager ``(B, 1, H, W, D)`` — the
   hook still fires (it sits on ``block.__call__``, outside the compiled
   ``block._forward``) and both layouts reshape to the same row-major
   ``(B, N_dit, D)`` from the latent's patch grid.

Config rides the LoRA network kwargs (``use_repa`` / ``repa_mode`` /
``repa_weight`` / ``repa_layer`` / ``repa_encoder``), stashed on the network by
the factory. The scalar loss returns under ``aux["repa"]``, weighted by
``LossComposer`` stage 2 (``losses._repa_loss``).

Operating-point levers (``docs/methods/repa.md`` §"Annealing plan";
both default-off):

- ``repa_anneal_steps`` — hard cutoff (HASTE, arXiv:2505.16792: alignment
  helps early, degrades late). (0, 1] = fraction of ``max_train_steps``; > 1 =
  absolute optimizer steps.
- ``repa_spatial_norm`` — iREPA-style (arXiv:2512.10794) standardization of
  target tokens before per-token L2-norm + Gram, cancelling the shared global
  component that compresses pairwise cosines. Relational mode only.

Diagnostic (``repa_grad_heatmap = N`` = probe every N train micro-steps, 0 =
off): MaskAlign (arXiv:2606.08788) shows full-token alignment concentrates
gradient norm at stable spatial positions — the probe takes
``autograd.grad`` of the alignment scalar w.r.t. pooled DiT tokens, resizes
the per-token grad-norm map to a 32×32 grid, and accumulates top-10%
membership counts across steps into
``<output_name>_repa_grad_heatmap.npz`` / ``repa/heatmap_conc``.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from library.training.method_adapter import (
    ForwardArtifacts,
    MethodAdapter,
    SetupCtx,
    StepCtx,
)
from library.vision.buckets import get_bucket_spec

logger = logging.getLogger(__name__)

# Grad-heatmap diagnostic canonical grid + MaskAlign's top-k membership
# fraction (their Fig. 2a statistic uses the top-10% positions).
_HEATMAP_GRID = 32
_HEATMAP_TOPK_FRAC = 0.10


# Grid-match / pooling / Gram math factored out so no-training probes
# (bench/turbo/) measure the identical quantity the training term optimizes.


def resolve_pe_grid(spec, n_pe: int, h_lat: int, w_lat: int) -> tuple[int, int]:
    """Resolve the encoder ``(gh, gw)`` patch grid for ``n_pe`` patch tokens.
    Token count alone is ambiguous (the bucket table is aspect-symmetric:
    46×23 and 23×46 both have 1058 patches), so disambiguate by latent
    orientation. Returns the unique bucket whose patch product == ``n_pe``.
    """
    cands = [(h, w) for (h, w) in spec.buckets if h * w == n_pe]
    if not cands:
        raise RuntimeError(
            f"REPA: no {spec.encoder} bucket has {n_pe} patches "
            f"(buckets={spec.buckets}). Cache encoder/grid mismatch?"
        )
    if len(cands) > 1:
        portrait = h_lat >= w_lat
        cands = [(h, w) for (h, w) in cands if (h >= w) == portrait] or cands
    return cands[0]


def pool_dit_tokens_to_grid(
    captured: torch.Tensor,
    latent_hw: tuple[int, int],
    patch: int,
    gh: int,
    gw: int,
    trim_tail: int = 0,
) -> torch.Tensor:
    """Captured block output → ``(B, gh*gw, D)`` fp32 tokens on the encoder grid.

    Layout-agnostic over the two block-output shapes (eager ``(B,1,H,W,D)`` /
    native-flatten ``(B,1,seq,1,D)``) — both flatten to row-major
    ``(B, N_dit, D)``, verified against the latent patch grid and
    adaptive-avg-pooled to ``(gh, gw)``.

    ``trim_tail`` drops that many trailing register tokens before the grid
    check — they're non-decoded scratchpad appended at the END of the
    self-attn sequence and must not enter the alignment loss.
    """
    b = captured.shape[0]
    d_dit = captured.shape[-1]
    tokens = captured.reshape(b, -1, d_dit)
    if trim_tail > 0:
        tokens = tokens[:, :-trim_tail]

    h_lat, w_lat = latent_hw
    h_dit, w_dit = h_lat // patch, w_lat // patch
    if tokens.shape[1] != h_dit * w_dit:
        raise RuntimeError(
            f"REPA: captured {tokens.shape[1]} DiT tokens but latent grid is "
            f"{h_dit}x{w_dit}={h_dit * w_dit} (patch={patch}). "
            "Block/patch-grid mismatch."
        )

    dit_grid = tokens.reshape(b, h_dit, w_dit, d_dit).permute(0, 3, 1, 2)
    dit_pooled = F.adaptive_avg_pool2d(dit_grid.float(), (gh, gw))
    return dit_pooled.flatten(2).transpose(1, 2)  # (B, gh*gw, D) fp32


def relational_gram_loss(
    dit_tok: torch.Tensor,
    pe: torch.Tensor,
    *,
    spatial_norm: bool = False,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Arm-B relational loss: ``MSE(Gram(dit_tok), Gram(pe))`` in fp32.

    Per-token L2-norm within each space, then match the N×N affinity
    structure. ``spatial_norm`` applies iREPA target standardization:
    standardize ``pe`` across the token axis before per-token L2-norm,
    removing the shared global component that compresses pairwise cosines.

    ``sample_weights`` (``[B]`` or None): per-sample multipliers applied
    before the batch mean (timestep reweighting); None is bit-exact to the
    unweighted ``F.mse_loss`` reduction.
    """
    if spatial_norm:
        pe = (pe - pe.mean(dim=1, keepdim=True)) / (pe.std(dim=1, keepdim=True) + 1e-6)
    dit_hat = F.normalize(dit_tok, dim=-1)
    pe_hat = F.normalize(pe, dim=-1)
    g_dit = torch.bmm(dit_hat, dit_hat.transpose(1, 2))  # (B, N, N)
    g_pe = torch.bmm(pe_hat, pe_hat.transpose(1, 2))
    if sample_weights is None:
        return F.mse_loss(g_dit, g_pe)
    per_sample = (g_dit - g_pe).pow(2).mean(dim=(1, 2))  # [B]
    return (per_sample * sample_weights.to(per_sample.dtype)).mean()


def _safe_blur(grid: torch.Tensor, sigma: float, gh: int, gw: int) -> torch.Tensor:
    """``gaussian_blur_2d`` guarded against a kernel wider than the grid.

    ``gaussian_blur_2d`` reflect-pads by ``ceil(3σ)``, which needs
    ``pad ≤ min_dim − 1``. Clamp σ defensively so an aggressive small divisor
    degrades to the widest valid blur instead of crashing.
    """
    from library.runtime.fei import gaussian_blur_2d

    if sigma <= 0:
        return grid
    max_pad = min(gh, gw) - 1
    if math.ceil(3.0 * sigma) > max_pad:
        sigma = max(1e-3, (max_pad - 0.01) / 3.0)
    return gaussian_blur_2d(grid, sigma)


def dog_standardize(
    pe: torch.Tensor,
    gh: int,
    gw: int,
    sigma1_div: float,
    sigma2_div: float = 0.0,
    norm_std: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Difference-of-Gaussians band-pass of the target tokens (REPA-DoG).

    Generalizes ``spatial_norm``'s DC removal to a broader low-band strip
    (arXiv:2603.14645v1 §3.5; best ``σ₁ = min/16`` in bench). ``pe``
    is ``(B, N, d)`` with ``N == gh*gw`` (CLS dropped); reshaped to
    ``(gh, gw)`` row-major, band-passed, standardized, flattened back — slots
    in **instead of** ``relational_gram_loss``'s ``spatial_norm`` block.

    Filter ``H(Z)`` on ``Z (B, d, gh, gw)``:

    * ``sigma2_div <= 0`` → ``Z − LP(Z, σ₁)``          high-pass
    * ``sigma2_div > 0``  → ``LP(Z, σ₂) − LP(Z, σ₁)``  band-pass

    with ``σ₁ = min(gh,gw)/sigma1_div``, ``σ₂ = min(gh,gw)/sigma2_div``. At
    ``σ₁→0`` this reduces to the shipped DC removal (``spatial_norm`` is its
    degenerate special case).

    ``norm_std``: ``0`` (default) divides by the empirical per-channel
    spatial std (identical to ``spatial_norm``'s, so an A/B isolates the
    band-pass); ``> 0`` divides by that fixed constant instead (paper's
    ``normalization_std`` regime).
    """
    b, n, d = pe.shape
    grid = pe.transpose(1, 2).reshape(b, d, gh, gw)  # row-major (B, d, gh, gw)
    s1 = float(min(gh, gw)) / float(sigma1_div)
    if sigma2_div > 0:
        s2 = float(min(gh, gw)) / float(sigma2_div)
        h = _safe_blur(grid, s2, gh, gw) - _safe_blur(grid, s1, gh, gw)
    else:
        h = grid - _safe_blur(grid, s1, gh, gw)
    denom = (
        (h.std(dim=(2, 3), keepdim=True) + eps) if norm_std <= 0 else float(norm_std)
    )
    h = h / denom
    return h.reshape(b, d, n).transpose(1, 2)  # back to (B, N, d)


def relational_align_loss(
    captured: torch.Tensor,
    pe: torch.Tensor,
    latent_hw: tuple[int, int],
    patch: int,
    spec,
    *,
    spatial_norm: bool = False,
    dog: bool = False,
    dog_sigma1_div: float = 16.0,
    dog_sigma2_div: float = 0.0,
    dog_norm_std: float = 0.0,
) -> torch.Tensor:
    """One-shot relational alignment: CLS-drop → grid-match → pool → Gram MSE.

    ``captured`` is a raw block output (either layout), ``pe`` the cached
    encoder features ``(B, T, d_enc)`` with CLS still at index 0 when the spec
    carries one. This is the probe entry point (``bench/turbo/``); the
    training adapter composes the same pieces itself (it also needs the
    pooled tokens for the absolute-arm head / grad-heatmap probe).

    ``dog`` applies the REPA-DoG band-pass instead of ``spatial_norm``
    (mutually exclusive; ``dog`` wins).
    """
    n_pe = pe.shape[1] - (1 if spec.use_cls else 0)
    gh, gw = resolve_pe_grid(spec, n_pe, latent_hw[0], latent_hw[1])
    if spec.use_cls:
        pe = pe[:, 1:, :]
    dit_tok = pool_dit_tokens_to_grid(captured, latent_hw, patch, gh, gw)
    if dog:
        pe = dog_standardize(pe, gh, gw, dog_sigma1_div, dog_sigma2_div, dog_norm_std)
        return relational_gram_loss(dit_tok, pe, spatial_norm=False)
    return relational_gram_loss(dit_tok, pe, spatial_norm=spatial_norm)


class REPAHead(nn.Module):
    """3-layer MLP projecting DiT hidden dim → vision-encoder feature dim
    (``h_phi`` from the REPA paper). Last layer init near-zero so step-0
    cosine is unbiased. Only used by Arm A (absolute); Arm B has no head.
    """

    def __init__(self, dit_dim: int, hidden_dim: int, encoder_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dit_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, encoder_dim)
        nn.init.normal_(self.fc3.weight, std=1e-3)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        return self.fc3(x)


class REPAMethodAdapter(MethodAdapter):
    """Bridges REPA into AnimaTrainer's adapter dispatch.

    Setup: install a forward post-hook on ``unet.blocks[repa_layer]``, read
    REPA config off the network. Prime: stash this step's cached PE-Spatial
    features. Extra forward: pool DiT tokens to the encoder grid, compute the
    alignment scalar, return it under ``aux["repa"]``.
    """

    name = "repa"
    # Grid-agnostic: adaptive pooling means a σ-demoted latent grid pools the
    # same way (aspect/orientation preserved), so sigma_lowres is safe here.
    sigma_demote_safe = True

    def __init__(self) -> None:
        self._captured: Optional[torch.Tensor] = None
        self._pe_features: Optional[torch.Tensor] = None
        self._latent_hw: Optional[tuple[int, int]] = None
        self._hook_handle = None
        # Filled in on_network_built.
        self._mode = "relational"
        self._layer = 8
        self._patch = 2
        self._spec = None
        self._anneal_steps = 0.0
        self._spatial_norm = False
        # REPA-DoG target band-pass: when on, replaces the spatial_norm block in
        # the relational loss (off ⇒ inert).
        self._dog = False
        self._dog_sigma1_div = 16.0
        self._dog_sigma2_div = 0.0
        self._dog_norm_std = 0.0
        # Timestep reweighting of the alignment term (0 = uniform).
        self._timestep_weighting = 0.0
        # Trailing register tokens to drop from the capture (0 = none).
        self._trim_tokens = 0
        # Optimizer-step clock: train micro-batches, converted with
        # gradient_accumulation_steps at the anneal gate.
        self._train_micro_steps = 0
        self._grad_heatmap_every = 0
        self._heat_counts: Optional[torch.Tensor] = None
        self._heat_samples = 0
        self._heat_runs = 0
        self._metrics: dict[str, float] = {}

    def on_network_built(self, ctx: SetupCtx) -> None:
        net = ctx.network
        self._mode = str(getattr(net, "_repa_mode", "relational")).lower()
        self._layer = int(getattr(net, "_repa_layer", 8))
        self._anneal_steps = float(getattr(net, "_repa_anneal_steps", 0.0) or 0.0)
        self._spatial_norm = bool(getattr(net, "_repa_spatial_norm", False))
        self._grad_heatmap_every = int(
            float(getattr(net, "_repa_grad_heatmap", 0) or 0)
        )
        if self._grad_heatmap_every > 0 and self._mode != "relational":
            logger.warning(
                "REPA: repa_grad_heatmap targets the relational loss (lever-3 "
                "gate); ignored in %s mode.",
                self._mode,
            )
            self._grad_heatmap_every = 0
        encoder = str(getattr(net, "_repa_encoder", "pe_spatial"))
        self._spec = get_bucket_spec(encoder)
        self._patch = int(ctx.unet.patch_spatial)

        # REPA-DoG target band-pass: replaces the spatial_norm DC-removal block
        # in the relational loss when on (relational mode only).
        self._dog = bool(getattr(net, "_repa_target_dog", True))
        self._dog_sigma1_div = float(getattr(net, "_repa_dog_sigma1_div", 16.0) or 16.0)
        self._dog_sigma2_div = float(getattr(net, "_repa_dog_sigma2_div", 0.0) or 0.0)
        self._dog_norm_std = float(getattr(net, "_repa_dog_norm_std", 0.0) or 0.0)
        if self._dog and self._mode != "relational":
            logger.warning(
                "REPA: repa_target_dog is a relational target-preprocessing "
                "lever; ignored in %s mode.",
                self._mode,
            )
            self._dog = False

        # Timestep reweighting: tilt the alignment term across σ (0 = uniform).
        # See _timestep_weights for the parameterization.
        self._timestep_weighting = float(
            getattr(net, "_repa_timestep_weighting", 0.0) or 0.0
        )

        # At/past the register insert block, the capture carries K extra
        # trailing scratchpad tokens that must be trimmed before grid pooling
        # or pool_dit_tokens_to_grid's patch-grid check fails.
        reg_k = int(getattr(net, "extra_seq_tokens", 0) or 0)
        reg_insert = int(
            getattr(getattr(net, "cfg", None), "register_insert_block", 0) or 0
        )
        self._trim_tokens = reg_k if (reg_k > 0 and self._layer >= reg_insert) else 0
        if self._trim_tokens:
            logger.info(
                "REPA: trimming %d register tokens off the block-%d capture "
                "(registers enter at block %d).",
                self._trim_tokens,
                self._layer,
                reg_insert,
            )

        if self._mode == "absolute" and getattr(net, "repa_head", None) is None:
            raise ValueError(
                "repa_mode='absolute' requires the network to expose a "
                "'repa_head' submodule (attached by networks.lora_anima.factory "
                "when use_repa=true and repa_mode=absolute)."
            )

        blocks = ctx.unet.blocks
        if not (0 <= self._layer < len(blocks)):
            raise ValueError(
                f"repa_layer={self._layer} out of range (DiT has {len(blocks)} blocks)"
            )

        def _hook(_module, _inputs, output: torch.Tensor) -> None:
            # Keep grad (flows back into LoRA modules in blocks <= repa_layer).
            # Runs in block.__call__, outside the compiled block._forward, so it
            # fires under compile_blocks().
            self._captured = output

        self._hook_handle = blocks[self._layer].register_forward_hook(_hook)

        weight = float(getattr(net, "_repa_weight", 0.0))
        if self._mode == "absolute":
            head = net.repa_head
            head_desc = (
                f"; head {head.fc1.in_features}→{head.fc2.out_features}"
                f"→{head.fc3.out_features}"
            )
        else:
            head_desc = "; no head (Gram)"
        anneal_desc = (
            f", anneal={self._anneal_steps:g}"
            f"{' (fraction)' if 0 < self._anneal_steps <= 1.0 else ' steps' if self._anneal_steps > 1 else ''}"
            if self._anneal_steps > 0
            else ""
        )
        ctx.accelerator.print(
            f"REPA[{self._mode}]: hook on block {self._layer}/{len(blocks)}, "
            f"encoder={encoder} grid≤{self._spec.t_max_patches}tok, "
            f"weight={weight}{anneal_desc}"
            f"{', spatial_norm' if self._spatial_norm and not self._dog else ''}"
            f"{f', grad_heatmap/{self._grad_heatmap_every}' if self._grad_heatmap_every else ''}"
            f"{f', dog(σ1=min/{self._dog_sigma1_div:g}, σ2={"off" if self._dog_sigma2_div <= 0 else f"min/{self._dog_sigma2_div:g}"}, norm_std={"empirical" if self._dog_norm_std <= 0 else self._dog_norm_std})' if self._dog else ''}"
            f"{f', tsw={self._timestep_weighting:g}' if self._timestep_weighting else ''}"
            f"{head_desc}"
        )

    def _timestep_weights(self, sigma: torch.Tensor) -> Optional[torch.Tensor]:
        """Per-sample alignment weight vs noise level σ∈[0,1] (``primary.timesteps``
        *is* σ). ``repa_timestep_weighting`` g (signed; 0 ⇒ uniform):

          g > 0 ⇒ w(σ) = (g+1)·σ**g          — emphasize HIGH noise
          g < 0 ⇒ w(σ) = (|g|+1)·(1−σ)**|g|  — emphasize LOW noise

        Both integrate to 1 under uniform σ, so this reshapes *where* REPA acts
        without changing its expected magnitude. Returns None at g == 0
        (bit-exact to the unweighted loss)."""
        g = self._timestep_weighting
        if g == 0.0:
            return None
        s = sigma.detach().float().clamp(0.0, 1.0)
        if g > 0.0:
            return (g + 1.0) * s.pow(g)
        p = -g
        return (p + 1.0) * (1.0 - s).pow(p)

    def prime_for_forward(
        self, ctx: StepCtx, batch, latents: torch.Tensor, *, is_train: bool
    ) -> None:
        # Drop any stash from a step that didn't trigger the hook.
        self._captured = None
        self._pe_features = None
        self._latent_hw = None
        if not is_train:
            return
        # Advance the optimizer-step clock on every train micro-batch (even
        # ones missing PE features) so it stays in lockstep with the trainer's
        # global_step; validation passes don't tick it.
        micro_step = self._train_micro_steps
        self._train_micro_steps += 1
        if self._past_anneal_cutoff(ctx.args, micro_step):
            self._metrics["repa/active"] = 0.0
            return
        feats = batch.get("repa_pe_features") if isinstance(batch, dict) else None
        if feats is None:
            # Batch lacks PE features (some sample missing its sidecar, or the
            # loader was off this step) — skip the term rather than crash.
            self._metrics["repa/active"] = 0.0
            return
        self._metrics["repa/active"] = 1.0
        self._pe_features = feats.to(ctx.accelerator.device, dtype=torch.float32)
        self._latent_hw = (int(latents.shape[-2]), int(latents.shape[-1]))

    def _past_anneal_cutoff(self, args, micro_step: int) -> bool:
        """Hard anneal cutoff: True once the optimizer-step clock
        passes ``repa_anneal_steps`` — (0, 1] is a fraction of
        ``max_train_steps``, > 1 is absolute optimizer steps. 0 = off."""
        if self._anneal_steps <= 0:
            return False
        cutoff = self._anneal_steps
        if cutoff <= 1.0:
            max_steps = int(getattr(args, "max_train_steps", 0) or 0)
            if max_steps <= 0:
                if not getattr(self, "_warned_no_max_steps", False):
                    logger.warning(
                        "REPA: repa_anneal_steps=%g is a fraction but "
                        "max_train_steps is unset — anneal disabled.",
                        self._anneal_steps,
                    )
                    self._warned_no_max_steps = True
                return False
            cutoff = cutoff * max_steps
        accum = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
        if micro_step // accum < cutoff:
            return False
        if not getattr(self, "_anneal_cutoff_logged", False):
            logger.info(
                "REPA: anneal cutoff reached at optimizer step %d "
                "(repa_anneal_steps=%g) — alignment loss off for the rest of the run.",
                micro_step // accum,
                self._anneal_steps,
            )
            self._anneal_cutoff_logged = True
        return True

    def _pe_grid(self, n_pe: int, h_lat: int, w_lat: int) -> tuple[int, int]:
        """Thin delegate to :func:`resolve_pe_grid` over this run's spec."""
        return resolve_pe_grid(self._spec, n_pe, h_lat, w_lat)

    def extra_forwards(self, ctx: StepCtx, primary: ForwardArtifacts) -> Optional[dict]:
        if not primary.is_train or self._pe_features is None or self._latent_hw is None:
            return None
        if self._captured is None:
            # PE features loaded + train step, but the block hook never fired —
            # REPA would be silently inert. Warn once so a mis-wired hook is
            # visible.
            if not getattr(self, "_warned_no_capture", False):
                logger.warning(
                    "REPA: block %d hook did not fire on a train step — alignment "
                    "loss is inert. Check the forward hook under compile_blocks().",
                    self._layer,
                )
                self._warned_no_capture = True
            return None

        h_lat, w_lat = self._latent_hw
        pe = self._pe_features  # (B, T_pe, d_enc) fp32
        n_pe = pe.shape[1] - (1 if self._spec.use_cls else 0)
        gh, gw = resolve_pe_grid(self._spec, n_pe, h_lat, w_lat)
        if self._spec.use_cls:
            pe = pe[:, 1:, :]  # drop CLS → (B, gh*gw, d_enc)

        dit_tok = pool_dit_tokens_to_grid(
            self._captured,
            (h_lat, w_lat),
            self._patch,
            gh,
            gw,
            trim_tail=self._trim_tokens,
        )  # (B, gh*gw, D) fp32

        w = None
        if self._timestep_weighting != 0.0:
            w = self._timestep_weights(primary.timesteps)
            if w is not None:
                self._metrics["repa/tsw_w_mean"] = float(w.mean().detach())

        if self._mode == "absolute":
            head = ctx.network.repa_head
            head_dtype = next(head.parameters()).dtype
            proj = head(dit_tok.to(head_dtype)).float()  # (B, N, d_enc)
            cos = F.cosine_similarity(proj, pe, dim=-1)  # (B, N)
            if w is None:
                loss = (1.0 - cos).mean()
            else:
                loss = ((1.0 - cos).mean(dim=1) * w.to(cos.dtype)).mean()
        else:
            if self._dog:
                # REPA-DoG: band-pass the target instead of spatial_norm's DC
                # removal (mutually exclusive — DoG at σ₁→0 *is* DC removal).
                pe_t = dog_standardize(
                    pe,
                    gh,
                    gw,
                    self._dog_sigma1_div,
                    self._dog_sigma2_div,
                    self._dog_norm_std,
                )
                loss = relational_gram_loss(
                    dit_tok, pe_t, spatial_norm=False, sample_weights=w
                )
            else:
                loss = relational_gram_loss(
                    dit_tok, pe, spatial_norm=self._spatial_norm, sample_weights=w
                )

            if self._grad_heatmap_every > 0:
                if self._heat_runs % self._grad_heatmap_every == 0:
                    self._accumulate_grad_heatmap(loss, dit_tok, gh, gw)
                self._heat_runs += 1

        self._metrics["repa/align_loss"] = float(loss.detach())
        return {"repa": loss}

    def _accumulate_grad_heatmap(
        self, loss: torch.Tensor, dit_tok: torch.Tensor, gh: int, gw: int
    ) -> None:
        """MaskAlign Fig. 2a statistic: top-10% alignment-gradient positions.
        ``autograd.grad`` w.r.t. the pooled tokens only traverses normalize →
        Gram → MSE (near-free); per-token grad-norm maps are resized to the
        canonical ``_HEATMAP_GRID``² so counts accumulate across aspect
        buckets with a fixed top-k per sample.
        """
        if not (loss.requires_grad and dit_tok.requires_grad):
            # No grad path (e.g. captured feature detached) — diagnostic only,
            # never fail the step over it.
            if not getattr(self, "_warned_heatmap_no_grad", False):
                logger.warning(
                    "REPA: grad-heatmap probe skipped — alignment loss has no "
                    "grad path to the pooled tokens."
                )
                self._warned_heatmap_no_grad = True
            return
        (g,) = torch.autograd.grad(loss, dit_tok, retain_graph=True)
        with torch.no_grad():
            b = g.shape[0]
            norms = g.norm(dim=-1).reshape(b, 1, gh, gw)
            canon = F.interpolate(
                norms,
                size=(_HEATMAP_GRID, _HEATMAP_GRID),
                mode="bilinear",
                align_corners=False,
            ).flatten(1)  # (B, 32*32)
            k = max(1, round(_HEATMAP_TOPK_FRAC * canon.shape[1]))
            hits = torch.zeros_like(canon)
            hits.scatter_(1, canon.topk(k, dim=1).indices, 1.0)
            if self._heat_counts is None:
                self._heat_counts = torch.zeros(canon.shape[1], dtype=torch.float64)
            self._heat_counts += hits.sum(dim=0).double().cpu()
            self._heat_samples += b
            freq = self._heat_counts / self._heat_samples
            # Recurrence of the most-hit position vs the uniform expectation
            # (every position lands in the top-10% a fraction topk_frac of the
            # time under no concentration). MaskAlign's pathology is ~21×.
            self._metrics["repa/heatmap_conc"] = float(freq.max() / _HEATMAP_TOPK_FRAC)

    def on_epoch_end(self, ctx: StepCtx) -> None:
        """Dump the accumulated grad-heatmap (overwritten each epoch)."""
        if self._heat_counts is None or self._heat_samples == 0:
            return
        if not ctx.accelerator.is_main_process:
            return
        import numpy as np

        out_dir = str(getattr(ctx.args, "output_dir", "") or ".")
        name = str(getattr(ctx.args, "output_name", "") or "anima")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}_repa_grad_heatmap.npz")
        counts = self._heat_counts.reshape(_HEATMAP_GRID, _HEATMAP_GRID).cpu().numpy()
        freq = counts / self._heat_samples
        conc = float(freq.max() / _HEATMAP_TOPK_FRAC)
        np.savez(
            path,
            counts=counts,
            freq=freq,
            n_samples=self._heat_samples,
            grid=_HEATMAP_GRID,
            topk_frac=_HEATMAP_TOPK_FRAC,
            concentration=conc,
        )
        logger.info(
            "REPA: grad-heatmap → %s (n=%d, concentration=%.2fx vs uniform; "
            "MaskAlign pathology ~21x, ≲3x ⇒ close lever 3)",
            path,
            self._heat_samples,
            conc,
        )

    def metrics(self, ctx) -> dict[str, float]:
        """Surface the last train-step alignment scalar to the loggers.

        ``repa/align_loss`` is *unweighted* (weighted contribution is
        ``align_loss * repa_weight``); not updated on inactive steps, so read
        alongside ``repa/active``.
        """
        del ctx
        return dict(self._metrics)
