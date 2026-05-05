"""DCW online calibrator — produces a per-step λ for the post-step DCW correction.

Loads a fusion-head safetensors artifact (head weights + standardization stats),
observes the LL-band Haar norm of the post-CFG ``noise_pred`` over the first
``k_warmup`` steps, fires the MLP at step ``k_warmup`` to predict the per-prompt
LSQ-optimal scalar λ̂*_p, then applies::

    λ_i = α_eff · gain · (1 − σ_i)        for target_start ≤ i < target_end
    λ_i = 0                                 otherwise

clamped to ±0.05.

Schema compat: accepts both ``dcw_v5_lambda_scalar`` (post-cleanup) and the
legacy ``dcw_v4_fusion_head`` schemas. Pre-``lambda_scalar`` v4 artifacts
(``target_kind=alpha_residual``) are rejected at load time — they need either
a retrain or a ``git checkout`` to the pre-cleanup controller.

The calibrator is **inactive** (``is_active == False``) when the artifact
fails to load or ``setup`` hits an empty embed mask.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open

from networks.dcw import FusionHead, haar_LL_norm

logger = logging.getLogger(__name__)

_VALID_SCHEMAS = ("dcw_v5_lambda_scalar", "dcw_v4_fusion_head")
_LAMBDA_CLAMP = 0.05


class OnlineDCWCalibrator:
    def __init__(
        self,
        head: FusionHead,
        centroid: torch.Tensor,
        aux_mean: torch.Tensor,
        aux_std: torch.Tensor,
        g_obs_mean: torch.Tensor,
        g_obs_std: torch.Tensor,
        k_warmup: int,
        n_steps: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        target_start: Optional[int] = None,
        target_end: Optional[int] = None,
        c_pool_norm: str = "none",
        c_pool_mean: Optional[torch.Tensor] = None,
        c_pool_std: Optional[torch.Tensor] = None,
    ):
        self.head = head.to(device=device, dtype=dtype).eval()
        self.centroid = centroid.to(device=device, dtype=dtype)
        self.aux_mean = aux_mean.to(device=device, dtype=dtype)
        self.aux_std = aux_std.to(device=device, dtype=dtype)
        self.g_obs_mean = g_obs_mean.to(device=device, dtype=dtype)
        self.g_obs_std = g_obs_std.to(device=device, dtype=dtype)
        self.k_warmup = int(k_warmup)
        self.n_steps = int(n_steps)
        self.target_start = int(k_warmup if target_start is None else target_start)
        self.target_end = int(n_steps if target_end is None else target_end)
        self.device = device
        self.dtype = dtype
        self.c_pool_norm = c_pool_norm
        self.c_pool_mean = (
            c_pool_mean.to(device=device, dtype=dtype)
            if c_pool_mean is not None
            else None
        )
        self.c_pool_std = (
            c_pool_std.to(device=device, dtype=dtype)
            if c_pool_std is not None
            else None
        )
        self.is_active: bool = False
        self.c_pool: Optional[torch.Tensor] = None
        self.aux: Optional[torch.Tensor] = None
        self.g_obs_buf: list[float] = []
        self.alpha_eff: float = 0.0
        self.gain: float = 1.0

    @classmethod
    def from_safetensors(
        cls, path: str | Path, *, device: torch.device
    ) -> "OnlineDCWCalibrator":
        path = Path(path)
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
            tensors = {k: f.get_tensor(k) for k in f.keys()}

        schema = meta.get("schema")
        if schema not in _VALID_SCHEMAS:
            raise ValueError(
                f"{path}: unexpected schema {schema!r}, expected one of "
                f"{_VALID_SCHEMAS}"
            )
        # Pre-lambda_scalar v4 artifacts default to alpha_residual. The new
        # controller only does lambda_scalar — refuse to silently misinterpret.
        target_kind = meta.get("target_kind", "lambda_scalar")
        if target_kind != "lambda_scalar":
            raise ValueError(
                f"{path}: target_kind={target_kind!r} is no longer supported. "
                "Either retrain with the current trainer (always lambda_scalar) "
                "or `git checkout` to the pre-cleanup controller for compat."
            )
        k_warmup = int(meta.get("k_warmup", 7))
        n_steps = int(meta.get("n_steps", 28))
        target_start = int(meta.get("target_start", k_warmup))
        target_end = int(meta.get("target_end", n_steps))

        head_sd = {
            k[len("head.") :]: v for k, v in tensors.items() if k.startswith("head.")
        }
        if "alpha_mlp.0.weight" not in head_sd:
            raise ValueError(
                f"{path}: missing 'head.alpha_mlp.*' keys — artifact predates "
                "the alpha/sigma trunk split. Retrain with `make dcw-train`."
            )
        in_dim = int(head_sd["alpha_mlp.0.weight"].shape[0])
        # FusionHead's training-time in_dim = cat_dim + aspect_emb_dim + k + aux_dim.
        # Old v4 artifacts carry aspect_emb tensor (per-aspect rows, all zeroed
        # by the trainer's `aspect_emb.weight.data.zero_()`); new v5 artifacts
        # drop it. In both cases we pass aspect_id=0 at forward and zero the
        # row so its contribution is exactly 0.
        if "aspect_emb.weight" in head_sd:
            n_aspects = int(head_sd["aspect_emb.weight"].shape[0])
            aspect_emb_dim = int(head_sd["aspect_emb.weight"].shape[1])
        else:
            n_aspects = 1
            aspect_emb_dim = 16  # FusionHead's default — must match trainer
        aux_dim = 3
        if "c_proj.1.weight" in head_sd:
            c_proj_w = head_sd["c_proj.1.weight"]
            c_proj_dim = int(c_proj_w.shape[0])
            c_pool_dim = int(c_proj_w.shape[1])
            cat_dim = c_proj_dim
        else:
            c_proj_dim = 0
            c_pool_dim = in_dim - (aspect_emb_dim + k_warmup + aux_dim)
            cat_dim = c_pool_dim
        if cat_dim + aspect_emb_dim + k_warmup + aux_dim != in_dim:
            raise ValueError(
                f"{path}: shape mismatch — cat({cat_dim}) + aspect_emb"
                f"({aspect_emb_dim}) + k({k_warmup}) + aux({aux_dim}) != "
                f"alpha_mlp.0 in_dim({in_dim})"
            )
        head = FusionHead(
            c_pool_dim=c_pool_dim,
            n_aspects=n_aspects,
            aspect_emb_dim=aspect_emb_dim,
            k=k_warmup,
            aux_dim=aux_dim,
            c_proj_dim=c_proj_dim,
        )
        # New v5 artifacts skip aspect_emb + sigma_mlp keys → load with
        # strict=False; sigma_mlp is never called at inference (σ̂² path is
        # gone), aspect_emb is zeroed below to neutralise its contribution.
        head.load_state_dict(head_sd, strict=False)
        head.aspect_emb.weight.data.zero_()

        c_pool_norm = meta.get("c_pool_norm", "none")
        if c_pool_norm not in ("none", "l2", "standardize", "l2_then_standardize"):
            raise ValueError(
                f"{path}: unknown c_pool_norm={c_pool_norm!r}. "
                "Either retrain with the current trainer or update the calibrator."
            )
        ctrl = cls(
            head=head,
            centroid=tensors["centroid_c_pool"],
            aux_mean=tensors["aux_mean"],
            aux_std=tensors["aux_std"],
            g_obs_mean=tensors["g_obs_mean"],
            g_obs_std=tensors["g_obs_std"],
            k_warmup=k_warmup,
            n_steps=n_steps,
            device=device,
            target_start=target_start,
            target_end=target_end,
            c_pool_norm=c_pool_norm,
            c_pool_mean=tensors.get("c_pool_mean"),
            c_pool_std=tensors.get("c_pool_std"),
        )
        logger.info(
            "DCW calibrator: loaded %s (schema=%s, k=%d, target=[%d:%d], "
            "%d steps, c_pool_norm=%s)",
            path.name,
            schema,
            k_warmup,
            target_start,
            target_end,
            n_steps,
            c_pool_norm,
        )
        return ctrl

    def setup(
        self,
        embed: torch.Tensor,
        embed_mask: Optional[torch.Tensor],
        *,
        gain: float = 1.0,
    ) -> None:
        """Compute c_pool + aux for this generation. Idempotent."""
        self.is_active = False
        self.g_obs_buf = []
        self.alpha_eff = 0.0
        self.gain = float(gain)

        # Pool the first batch row's embed (single-prompt assumption — matches
        # the trainer's per-prompt format). embed: (B, L, 1024).
        e = embed[0].to(self.device, dtype=self.dtype)
        if embed_mask is not None:
            mask = embed_mask[0].to(self.device, dtype=torch.bool)
            valid = e[mask]
            cap_len = int(mask.sum().item())
        else:
            valid = e
            cap_len = e.shape[0]
        if valid.numel() == 0:
            logger.warning("DCW calibrator: empty embed mask — disabling")
            return

        c_pool_raw = valid.mean(dim=0)
        token_l2 = valid.norm(dim=-1)
        # cos_centroid stays raw — the trainer's centroid was computed on raw
        # c_pool, and the cos itself is the aux feature, not the head input.
        cos_centroid = float(
            torch.dot(c_pool_raw, self.centroid)
            / (c_pool_raw.norm() * self.centroid.norm() + 1e-9)
        )
        aux_raw = torch.tensor(
            [float(cap_len), cos_centroid, float(token_l2.std().item())],
            device=self.device,
            dtype=self.dtype,
        )
        # Apply the same preprocessing the trainer used to the head's c_pool input.
        c_pool = c_pool_raw
        if self.c_pool_norm in ("l2", "l2_then_standardize"):
            c_pool = c_pool / (c_pool.norm() + 1e-9)
        if self.c_pool_norm in ("standardize", "l2_then_standardize"):
            if self.c_pool_mean is None or self.c_pool_std is None:
                raise RuntimeError(
                    "c_pool_norm requests standardize but artifact has no "
                    "c_pool_mean / c_pool_std tensors — retrain to ship them."
                )
            c_pool = (c_pool - self.c_pool_mean) / self.c_pool_std
        self.c_pool = c_pool
        self.aux = (aux_raw - self.aux_mean) / self.aux_std
        self.is_active = True
        logger.info(
            "DCW calibrator: setup target=[%d:%d] gain=%.4g cap_len=%d "
            "cos_centroid=%.3f c_pool_norm=%s",
            self.target_start,
            self.target_end,
            self.gain,
            cap_len,
            cos_centroid,
            self.c_pool_norm,
        )

    def record(self, step_i: int, noise_pred: torch.Tensor) -> None:
        """Observe LL-band norm of the post-CFG velocity at warmup steps."""
        if not self.is_active or step_i >= self.k_warmup:
            return
        self.g_obs_buf.append(haar_LL_norm(noise_pred))

    def fire_head_if_due(self, step_i: int) -> None:
        """Run the MLP at i == k_warmup. Sets self.alpha_eff for the tail."""
        if not self.is_active or step_i != self.k_warmup:
            return
        if len(self.g_obs_buf) < self.k_warmup:
            logger.warning(
                "DCW calibrator: only %d/%d warmup obs collected — disabling",
                len(self.g_obs_buf),
                self.k_warmup,
            )
            self.alpha_eff = 0.0
            return

        g_obs = torch.tensor(
            self.g_obs_buf[: self.k_warmup], device=self.device, dtype=self.dtype
        )
        g_obs_n = (g_obs - self.g_obs_mean) / self.g_obs_std

        with torch.no_grad():
            alpha_hat, _ = self.head(
                self.c_pool.unsqueeze(0),
                torch.zeros(1, device=self.device, dtype=torch.long),
                g_obs_n.unsqueeze(0),
                self.aux.unsqueeze(0),
            )
        self.alpha_eff = float(alpha_hat[0].item())
        logger.info(
            "DCW calibrator: head fired at step %d — α̂=%+.4g",
            step_i,
            self.alpha_eff,
        )

    def lambda_for_step(self, step_i: int, sigma_i: float) -> float:
        """Per-step λ for ``apply_dcw(..., schedule='const', lam=λ_i)``."""
        if not self.is_active or not (self.target_start <= step_i < self.target_end):
            return 0.0
        lam_i = self.alpha_eff * self.gain * (1.0 - sigma_i)
        return max(-_LAMBDA_CLAMP, min(_LAMBDA_CLAMP, lam_i))
