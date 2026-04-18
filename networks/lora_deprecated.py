# Deprecated LoRA module variants
# These modules are kept for checkpoint compatibility but are no longer actively used.
# - DoRAModule: superseded by OrthoLoRA + postfix tuning (lower VRAM, better quality).
# - OrthoLoRAModule: superseded by OrthoLoRAExpModule (Cayley + SVD-init, hard
#   orthogonality without a reg hyperparameter). See docs/methods/psoft-integrated-ortholora.md.

import math
import random
import warnings

import torch

from networks.lora_modules import BaseLoRAModule, LoRAModule


class DoRAModule(LoRAModule):
    """
    DoRA: Weight-Decomposed Low-Rank Adaptation.
    Decomposes pretrained weight into magnitude and direction, applies LoRA to the direction only.
    Reference: https://arxiv.org/abs/2402.09353

    DEPRECATED: Use OrthoLoRA + postfix tuning instead.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        channel_scale=None,
    ):
        warnings.warn(
            "DoRAModule is deprecated. Use OrthoLoRA + postfix tuning instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            lora_dim,
            alpha,
            dropout,
            rank_dropout,
            module_dropout,
            channel_scale=channel_scale,
        )

        # Initialize magnitude to per-row L2 norms of the original weight
        weight = org_module.weight.detach().float()
        self.magnitude = torch.nn.Parameter(weight.norm(p=2, dim=1))  # [out_features]

    def apply_to(self):
        # Compute and cache the frozen original weight row-norms before parent deletes org_module
        org_weight = self.org_module.weight.detach().float()
        self.register_buffer(
            "_org_weight_norm",
            org_weight.norm(p=2, dim=1).clamp(
                min=torch.finfo(org_weight.dtype).eps
            ),  # [out_features]
        )
        super().apply_to()

    def forward(self, x):
        org_out = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if random.random() < self.module_dropout:
                return org_out

        # per-channel input rebalancing (SmoothQuant-style, see LoRAModule.forward)
        x_lora = x * self.inv_scale if self._has_channel_scale else x

        # Bottleneck matmuls run in fp32 (see LoRAModule.forward).
        lx = torch.nn.functional.linear(
            x_lora.float(), self.lora_down.weight.float()
        )

        # timestep-dependent rank masking
        if self._timestep_mask is not None and self.training:
            lx = lx * self._timestep_mask

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # rank dropout
        if self.rank_dropout is not None and self.training:
            mask = (
                torch.rand((lx.size(0), self.lora_dim), device=lx.device)
                > self.rank_dropout
            )
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)
            lx = lx * mask
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale

        lx = torch.nn.functional.linear(lx, self.lora_up.weight.float())
        lora_out = (lx * self.multiplier * scale).to(org_out.dtype)

        # DoRA: scale by (magnitude / ||W_original||)
        mag_scale = (self.magnitude / self._org_weight_norm).to(org_out.dtype)

        # Broadcast mag_scale to match output shape
        if len(org_out.shape) == 3:
            mag_scale = mag_scale.unsqueeze(0).unsqueeze(0)  # [1, 1, out_dim]
        elif len(org_out.shape) == 2:
            mag_scale = mag_scale.unsqueeze(0)  # [1, out_dim]

        return mag_scale * (org_out + lora_out)


class OrthoLoRAModule(BaseLoRAModule):
    """
    Orthogonal LoRA: QR-based weight parameterization for orthonormal initialization.
    Uses P @ diag(lambda) @ Q decomposition with frozen base copies to ensure zero output at init.
    Orthogonality regularization keeps P and Q close to orthonormal bases during training.
    Reference: T-LoRA (AAAI 2026)

    DEPRECATED: Use OrthoLoRAExpModule (``use_ortho_exp = true``). The exp
    variant replaces soft orthogonality regularization with a hard Cayley
    constraint and initializes bases from the pretrained weight's SVD —
    fewer trainable params, no reg hyperparameter, and equivalent or better
    quality in our benchmarks. Kept here for checkpoint compatibility.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        sig_type="last",
        channel_scale=None,
    ):
        warnings.warn(
            "OrthoLoRAModule is deprecated. Use use_ortho_exp=true (OrthoLoRAExpModule) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            lora_name,
            org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
        )

        in_dim = org_module.in_features
        out_dim = org_module.out_features

        # Q: in_dim -> rank, P: rank -> out_dim, lambda: [1, rank]
        self.q_layer = torch.nn.Linear(in_dim, lora_dim, bias=False)
        self.p_layer = torch.nn.Linear(lora_dim, out_dim, bias=False)
        self.lambda_layer = torch.nn.Parameter(torch.ones(1, lora_dim))

        # Orthogonal initialization via QR decomposition
        # QR on (n, r) is O(n·r²) vs SVD on (n, m) which is O(n²·m) — orders of magnitude faster
        # for typical DiT dimensions (n,m ~ 2048-8192, r ~ 4-64).
        # QR of a random Gaussian matrix produces Haar-distributed orthonormal bases,
        # identical in distribution to SVD left/right singular vectors of a random matrix.
        init_device = "cuda" if torch.cuda.is_available() else "cpu"

        q_rand = torch.randn(in_dim, lora_dim, device=init_device)
        q_orth, _ = torch.linalg.qr(q_rand)  # (in_dim, lora_dim)
        self.q_layer.weight.data = q_orth.T.cpu().clone().contiguous()

        p_rand = torch.randn(out_dim, lora_dim, device=init_device)
        p_orth, _ = torch.linalg.qr(p_rand)  # (out_dim, lora_dim)
        self.p_layer.weight.data = p_orth.cpu().clone().contiguous()

        # Lambda: scale to match expected singular values of random Gaussian(0, 1/r) matrix
        std = 1.0 / lora_dim
        sv_max = std * (math.sqrt(in_dim) + math.sqrt(out_dim))
        sv_min = std * abs(math.sqrt(in_dim) - math.sqrt(out_dim))
        if sig_type == "principal":
            self.lambda_layer.data = torch.linspace(
                sv_max, (sv_max + sv_min) / 2, lora_dim
            ).unsqueeze(0)
        elif sig_type == "last":
            self.lambda_layer.data = torch.linspace(
                (sv_max + sv_min) / 2, sv_min + 1e-6, lora_dim
            ).unsqueeze(0)
        elif sig_type == "middle":
            mid = (sv_max + sv_min) / 2
            spread = (sv_max - sv_min) / 4
            self.lambda_layer.data = torch.linspace(
                mid + spread, mid - spread, lora_dim
            ).unsqueeze(0)

        del q_rand, q_orth, p_rand, p_orth

        # Per-channel input pre-scaling: absorb s into q_layer columns BEFORE
        # cloning base_q_weight, so both trainable and frozen paths see the same
        # rebalanced input. The residual invariant (p_out - base_out == 0 at init)
        # is preserved because q_layer.weight and base_q_weight remain identical.
        self._register_channel_scale(self.q_layer.weight.data, channel_scale)

        # Frozen base copies for residual (ensures zero output at init)
        self.register_buffer(
            "base_q_weight", self.q_layer.weight.data.clone().contiguous()
        )
        self.register_buffer(
            "base_p_weight", self.p_layer.weight.data.clone().contiguous()
        )
        self.register_buffer("base_lambda", self.lambda_layer.data.clone().contiguous())

        # Cached identity for regularization (non-persistent: not saved to state_dict)
        self.register_buffer("_eye_r", torch.eye(lora_dim), persistent=False)

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if self._skip_module():
            return org_forwarded

        dtype = self.q_layer.weight.dtype

        # per-channel input rebalancing (SmoothQuant-style). Applied to both the
        # trainable path and the frozen base path so the init residual cancellation
        # still holds exactly.
        x_lora = self._rebalance(x.to(dtype))

        # timestep mask
        mask = self._timestep_mask  # None when not using timestep masking

        # trainable path
        if mask is not None:
            q_out = self.q_layer(x_lora) * self.lambda_layer * mask
        else:
            q_out = self.q_layer(x_lora) * self.lambda_layer

        # normal dropout
        if self.dropout is not None and self.training:
            q_out = torch.nn.functional.dropout(q_out, p=self.dropout)

        q_out, scale = self._apply_rank_dropout(q_out)

        p_out = self.p_layer(q_out)

        # frozen base path (residual subtraction ensures zero output at init)
        base_out = torch.nn.functional.linear(
            torch.nn.functional.linear(x_lora, self.base_q_weight)
            * self.base_lambda
            * (mask if mask is not None else 1.0),
            self.base_p_weight,
        )

        lora_out = (p_out - base_out) * self.multiplier * scale
        return org_forwarded + lora_out.to(org_forwarded.dtype)

    def regularization(self):
        """Orthogonality regularization: ||P^T P - I||^2 + ||Q Q^T - I||^2"""
        eye = self._eye_r
        p_reg = torch.sum((self.p_layer.weight.T @ self.p_layer.weight - eye) ** 2)
        q_reg = torch.sum((self.q_layer.weight @ self.q_layer.weight.T - eye) ** 2)
        return p_reg, q_reg
