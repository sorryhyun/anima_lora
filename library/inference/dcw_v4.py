"""DCW v4 online fusion-head controller.

Wraps the fusion_head safetensors artifact (per-aspect bucket prior +
shared MLP head + standardization stats) and produces a per-step λ for
the post-step pixel-mode DCW correction. See
``docs/proposal/dcw-learnable-calibrator-v4.md`` for the math.

State machine:

* warmup steps ``i < k``: λ_i = λ_scalar·(1−σ_i) + μ_g[i]·λ_scalar/S_pop[i].
  Caller hands the post-CFG ``noise_pred`` to ``record(i, noise_pred)`` so
  the LL band is observed and stashed.
* step ``i == k``: ``fire_head_if_due`` runs the MLP once, producing
  ``α_eff`` (with optional shrinkage / caption-length backstop).
* tail steps ``i ≥ k``: λ_i adds ``α_eff · μ_g[i] / Σ_{j≥k} μ_g[j]`` on
  top of the warmup formula.

The controller is **inactive** (``is_active == False``) when the artifact
fails to load or the request's aspect bucket isn't in the artifact's
table — callers should fall back to scalar ``--dcw`` in that case.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open

from networks.dcw import FusionHead, haar_LL_norm

logger = logging.getLogger(__name__)

ASPECT_TABLE: dict[tuple[int, int], int] = {
    (832, 1248): 0,
    (896, 1152): 1,
    (768, 1344): 2,
    (1152, 896): 3,
    (1248, 832): 4,
}
ASPECT_NAMES = ["832x1248", "896x1152", "768x1344", "1152x896", "1248x832"]


@dataclass
class _Profile:
    mu_g: torch.Tensor          # (n_steps,)
    s_pop: torch.Tensor         # (n_steps,)
    lam_scalar: float
    sigma2_prior: float
    mu_g_S_pop_tail_dot: float  # Σ_{i≥k}(mu_g[i] · S_pop[i]); LSQ denominator
    # Precomputed once per aspect at load time.


class OnlineFusionDCWController:
    def __init__(
        self,
        head: FusionHead,
        profiles: list[_Profile],
        centroid: torch.Tensor,
        aux_mean: torch.Tensor,
        aux_std: torch.Tensor,
        g_obs_mean: torch.Tensor,
        g_obs_std: torch.Tensor,
        k_warmup: int,
        n_steps: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.head = head.to(device=device, dtype=dtype).eval()
        self.profiles = profiles
        self.centroid = centroid.to(device=device, dtype=dtype)
        self.aux_mean = aux_mean.to(device=device, dtype=dtype)
        self.aux_std = aux_std.to(device=device, dtype=dtype)
        self.g_obs_mean = g_obs_mean.to(device=device, dtype=dtype)
        self.g_obs_std = g_obs_std.to(device=device, dtype=dtype)
        self.k_warmup = int(k_warmup)
        self.n_steps = int(n_steps)
        self.device = device
        self.dtype = dtype
        # set per request via setup()
        self.is_active: bool = False
        self.aspect_id: Optional[int] = None
        self.profile: Optional[_Profile] = None
        self.c_pool: Optional[torch.Tensor] = None
        self.aux: Optional[torch.Tensor] = None
        self.g_obs_buf: list[float] = []
        self.alpha_eff: float = 0.0
        self.tail_norm: float = 1.0
        self.disable_shrinkage: bool = False
        self.disable_backstop: bool = False

    @classmethod
    def from_safetensors(
        cls, path: str | Path, *, device: torch.device
    ) -> "OnlineFusionDCWController":
        path = Path(path)
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
            tensors = {k: f.get_tensor(k) for k in f.keys()}

        if meta.get("schema") != "dcw_v4_fusion_head":
            raise ValueError(
                f"{path}: unexpected schema {meta.get('schema')!r}, "
                f"expected 'dcw_v4_fusion_head'"
            )
        k_warmup = int(meta.get("k_warmup", 7))
        n_aspects = int(meta.get("n_aspects", 3))
        n_steps = int(meta.get("n_steps", 28))
        sigma2_pop = float(meta.get("sigma2_pop", 1.0))

        head_sd = {
            k[len("head."):]: v for k, v in tensors.items() if k.startswith("head.")
        }
        # mlp.0 is LayerNorm — its weight shape is (in_dim,)
        in_dim = int(head_sd["mlp.0.weight"].shape[0])
        aspect_emb_dim = int(head_sd["aspect_emb.weight"].shape[1])
        aux_dim = 3
        c_pool_dim = in_dim - (aspect_emb_dim + k_warmup + aux_dim)
        head = FusionHead(
            c_pool_dim=c_pool_dim,
            n_aspects=n_aspects,
            aspect_emb_dim=aspect_emb_dim,
            k=k_warmup,
            aux_dim=aux_dim,
            log_sigma2_init=math.log(max(sigma2_pop, 1e-6)),
        )
        head.load_state_dict(head_sd)

        profiles = []
        for a in range(n_aspects):
            mu_g_a = tensors["bucket_prior_mu_g"][a].clone()
            s_pop_a = tensors["bucket_prior_S_pop"][a].clone()
            tail_dot = float((mu_g_a[k_warmup:] * s_pop_a[k_warmup:]).sum().item())
            profiles.append(
                _Profile(
                    mu_g=mu_g_a,
                    s_pop=s_pop_a,
                    lam_scalar=float(tensors["bucket_prior_lam_scalar"][a]),
                    sigma2_prior=float(tensors["sigma2_prior"][a]),
                    mu_g_S_pop_tail_dot=tail_dot,
                )
            )

        ctrl = cls(
            head=head,
            profiles=profiles,
            centroid=tensors["centroid_c_pool"],
            aux_mean=tensors["aux_mean"],
            aux_std=tensors["aux_std"],
            g_obs_mean=tensors["g_obs_mean"],
            g_obs_std=tensors["g_obs_std"],
            k_warmup=k_warmup,
            n_steps=n_steps,
            device=device,
        )
        logger.info(
            "DCW v4: loaded %s (k=%d, %d aspects, %d steps, sigma2_pop=%.2f)",
            path.name, k_warmup, n_aspects, n_steps, sigma2_pop,
        )
        return ctrl

    def setup(
        self,
        H: int,
        W: int,
        embed: torch.Tensor,
        embed_mask: Optional[torch.Tensor],
        *,
        disable_shrinkage: bool = False,
        disable_backstop: bool = False,
    ) -> None:
        """Resolve aspect, compute c_pool + aux for this generation. Idempotent."""
        self.is_active = False
        self.g_obs_buf = []
        self.alpha_eff = 0.0
        self.disable_shrinkage = disable_shrinkage
        self.disable_backstop = disable_backstop

        aspect_id = ASPECT_TABLE.get((H, W))
        if aspect_id is None:
            logger.warning(
                "DCW v4: aspect (%d, %d) not in table %s — disabling, "
                "fall back to scalar --dcw if you set it",
                H, W, ASPECT_NAMES,
            )
            return
        if aspect_id >= len(self.profiles):
            logger.warning("DCW v4: aspect_id %d out of range — disabling", aspect_id)
            return

        # Pool the first batch row's embed (single-prompt assumption — matches the
        # prototype's per-prompt training format). embed: (B, L, 1024).
        e = embed[0].to(self.device, dtype=self.dtype)
        if embed_mask is not None:
            mask = embed_mask[0].to(self.device, dtype=torch.bool)
            valid = e[mask]
            cap_len = int(mask.sum().item())
        else:
            valid = e
            cap_len = e.shape[0]
        if valid.numel() == 0:
            logger.warning("DCW v4: empty embed mask — disabling")
            return

        c_pool = valid.mean(dim=0)                                         # (1024,)
        token_l2 = valid.norm(dim=-1)                                      # (L,)
        cos_centroid = float(
            torch.dot(c_pool, self.centroid)
            / (c_pool.norm() * self.centroid.norm() + 1e-9)
        )
        aux_raw = torch.tensor(
            [float(cap_len), cos_centroid, float(token_l2.std().item())],
            device=self.device, dtype=self.dtype,
        )
        aux_n = (aux_raw - self.aux_mean) / self.aux_std

        self.aspect_id = aspect_id
        self.profile = self.profiles[aspect_id]
        self.c_pool = c_pool
        self.aux = aux_n
        self.tail_norm = float(self.profile.mu_g[self.k_warmup:].sum().item())
        if abs(self.tail_norm) < 1e-6:
            self.tail_norm = 1.0  # head_corr will be ~0; harmless
        self.is_active = True
        logger.info(
            "DCW v4: setup aspect=%s lam_scalar=%+.4f cap_len=%d cos_centroid=%.3f",
            ASPECT_NAMES[aspect_id], self.profile.lam_scalar, cap_len, cos_centroid,
        )

    def record(self, step_i: int, noise_pred: torch.Tensor) -> None:
        """Observe LL-band norm of the post-CFG velocity at warmup steps."""
        if not self.is_active or step_i >= self.k_warmup:
            return
        self.g_obs_buf.append(haar_LL_norm(noise_pred))

    def fire_head_if_due(self, step_i: int) -> None:
        """Run the MLP at i == k. Sets self.alpha_eff for the tail."""
        if not self.is_active or step_i != self.k_warmup:
            return
        if len(self.g_obs_buf) < self.k_warmup:
            logger.warning(
                "DCW v4: only %d/%d warmup obs collected — disabling head, "
                "tail will use bucket prior only",
                len(self.g_obs_buf), self.k_warmup,
            )
            self.alpha_eff = 0.0
            return

        g_obs = torch.tensor(
            self.g_obs_buf[: self.k_warmup], device=self.device, dtype=self.dtype
        )
        g_obs_n = (g_obs - self.g_obs_mean) / self.g_obs_std

        with torch.no_grad():
            alpha_hat, log_sigma2 = self.head(
                self.c_pool.unsqueeze(0),
                torch.tensor([self.aspect_id], device=self.device, dtype=torch.long),
                g_obs_n.unsqueeze(0),
                self.aux.unsqueeze(0),
            )
        alpha_hat_v = float(alpha_hat[0].item())
        sigma2 = float(torch.exp(log_sigma2[0]).item())

        if self.disable_shrinkage:
            self.alpha_eff = alpha_hat_v
        else:
            sp = self.profile.sigma2_prior
            shrinkage = sp / (sp + sigma2) if (sp + sigma2) > 1e-9 else 0.0
            self.alpha_eff = alpha_hat_v * shrinkage

        # Caption-length backstop: tau_short not yet shipped in the artifact; skip.
        # Hook left here so I10 can wire it in without changing call sites.
        # if (not self.disable_backstop) and self.aux_raw_cap_len < tau_short[a]:
        #     self.alpha_eff = 0.0

        logger.info(
            "DCW v4: head fired at step %d — α̂=%+.2f, σ̂²=%.2f, α_eff=%+.2f",
            step_i, alpha_hat_v, sigma2, self.alpha_eff,
        )

    def lambda_for_step(self, step_i: int, sigma_i: float) -> float:
        """Per-step λ for ``apply_dcw(..., schedule='const', lam=λ_i)``.

        Tail formula deviates from the proposal pseudocode to fix a dimensional
        bug: α̂ is in gap units (the trainer's target is integrated tail gap
        residual), so the LSQ-projected per-step λ that cancels α̂ under the
        mu_g-proportional distribution is

            Δλ_i = −α̂ · mu_g[i] / Σ_{j∈tail}(mu_g[j] · S_pop[j])

        not the proposal's ``α̂ · mu_g[i] / tail_norm``. Output is then clamped
        to ±3·|lam_scalar| as an overshoot guard.
        """
        if not self.is_active:
            return 0.0
        if step_i >= self.n_steps:
            return 0.0
        p = self.profile
        base = p.lam_scalar * (1.0 - sigma_i)
        i = min(step_i, p.mu_g.shape[0] - 1)
        mu_g_i = float(p.mu_g[i].item())
        s_pop_i = float(p.s_pop[i].item())
        bucket_corr = (
            mu_g_i * p.lam_scalar / s_pop_i if abs(s_pop_i) > 1e-6 else 0.0
        )
        if step_i < self.k_warmup:
            lam_i = base + bucket_corr
        else:
            denom = p.mu_g_S_pop_tail_dot
            head_corr = (
                -self.alpha_eff * mu_g_i / denom if abs(denom) > 1e-6 else 0.0
            )
            lam_i = base + bucket_corr + head_corr
        lam_max = 3.0 * abs(p.lam_scalar) if p.lam_scalar != 0.0 else 0.05
        return max(-lam_max, min(lam_max, lam_i))
