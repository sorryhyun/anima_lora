# LoRA module building blocks
# Extracted from lora_flux.py — generic LoRA, DoRA, OrthoLoRA, and inference modules
#
# Reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import math
from typing import List, Optional
import torch
from torch import Tensor
from library.utils import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
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
        split_dims: Optional[List[int]] = None,
        ggpo_beta: Optional[float] = None,
        ggpo_sigma: Optional[float] = None,
    ):
        """
        if alpha == 0 or None, alpha is rank (no scaling).

        split_dims is used to mimic the split qkv as same as Diffusers
        """
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim
        self.split_dims = split_dims

        if split_dims is None:
            if org_module.__class__.__name__ == "Conv2d":
                kernel_size = org_module.kernel_size
                stride = org_module.stride
                padding = org_module.padding
                self.lora_down = torch.nn.Conv2d(
                    in_dim, self.lora_dim, kernel_size, stride, padding, bias=False
                )
                self.lora_up = torch.nn.Conv2d(
                    self.lora_dim, out_dim, (1, 1), (1, 1), bias=False
                )
            else:
                self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
                self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

            torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_up.weight)
        else:
            # conv2d not supported
            assert sum(split_dims) == out_dim, (
                "sum of split_dims must be equal to out_dim"
            )
            assert org_module.__class__.__name__ == "Linear", (
                "split_dims is only supported for Linear"
            )
            self.lora_down = torch.nn.ModuleList(
                [
                    torch.nn.Linear(in_dim, self.lora_dim, bias=False)
                    for _ in range(len(split_dims))
                ]
            )
            self.lora_up = torch.nn.ModuleList(
                [
                    torch.nn.Linear(self.lora_dim, split_dim, bias=False)
                    for split_dim in split_dims
                ]
            )
            for lora_down in self.lora_down:
                torch.nn.init.kaiming_uniform_(lora_down.weight, a=math.sqrt(5))
            for lora_up in self.lora_up:
                torch.nn.init.zeros_(lora_up.weight)

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        # same as microsoft's
        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        self.fp32_accumulation = False

        self._timestep_mask = None

        self.ggpo_sigma = ggpo_sigma
        self.ggpo_beta = ggpo_beta

        if self.ggpo_beta is not None and self.ggpo_sigma is not None:
            self.combined_weight_norms = None
            self.grad_norms = None
            self.perturbation_norm_factor = 1.0 / math.sqrt(org_module.weight.shape[0])
            self.initialize_norm_cache(org_module.weight)
            self.org_module_shape: tuple[int] = org_module.weight.shape

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward

        del self.org_module

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        if self.split_dims is None:
            # fp32 accumulation: compute LoRA delta in fp32 for better precision
            if self.fp32_accumulation:
                lx = torch.nn.functional.linear(
                    x.float(), self.lora_down.weight.float()
                )
            else:
                lx = self.lora_down(x)

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
                    mask = mask.unsqueeze(1)  # for Text Encoder
                elif len(lx.size()) == 4:
                    mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
                lx = lx * mask

                # scaling for rank dropout: treat as if the rank is changed
                scale = self.scale * (
                    1.0 / (1.0 - self.rank_dropout)
                )  # redundant for readability
            else:
                scale = self.scale

            if self.fp32_accumulation:
                lx = torch.nn.functional.linear(lx, self.lora_up.weight.float())
                lx = (lx * self.multiplier * scale).to(org_forwarded.dtype)
            else:
                lx = self.lora_up(lx)

            # LoRA Gradient-Guided Perturbation Optimization
            if (
                self.training
                and self.ggpo_sigma is not None
                and self.ggpo_beta is not None
                and self.combined_weight_norms is not None
                and self.grad_norms is not None
            ):
                with torch.no_grad():
                    perturbation_scale = (
                        self.ggpo_sigma * torch.sqrt(self.combined_weight_norms**2)
                    ) + (self.ggpo_beta * (self.grad_norms**2))
                    perturbation_scale_factor = (
                        perturbation_scale * self.perturbation_norm_factor
                    ).to(self.device)
                    perturbation = torch.randn(
                        self.org_module_shape, dtype=self.dtype, device=self.device
                    )
                    perturbation.mul_(perturbation_scale_factor)
                    perturbation_output = x @ perturbation.T  # Result: (batch × n)
                if self.fp32_accumulation:
                    return org_forwarded + lx + perturbation_output
                return (
                    org_forwarded + (self.multiplier * scale * lx) + perturbation_output
                )
            else:
                if self.fp32_accumulation:
                    return org_forwarded + lx
                return org_forwarded + lx * self.multiplier * scale
        else:
            lxs = [lora_down(x) for lora_down in self.lora_down]

            # timestep-dependent rank masking
            if self._timestep_mask is not None and self.training:
                lxs = [lx * self._timestep_mask for lx in lxs]

            # normal dropout
            if self.dropout is not None and self.training:
                lxs = [torch.nn.functional.dropout(lx, p=self.dropout) for lx in lxs]

            # rank dropout
            if self.rank_dropout is not None and self.training:
                masks = [
                    torch.rand((lx.size(0), self.lora_dim), device=lx.device)
                    > self.rank_dropout
                    for lx in lxs
                ]
                for i in range(len(lxs)):
                    if len(lx.size()) == 3:
                        masks[i] = masks[i].unsqueeze(1)
                    elif len(lx.size()) == 4:
                        masks[i] = masks[i].unsqueeze(-1).unsqueeze(-1)
                    lxs[i] = lxs[i] * masks[i]

                # scaling for rank dropout: treat as if the rank is changed
                scale = self.scale * (
                    1.0 / (1.0 - self.rank_dropout)
                )  # redundant for readability
            else:
                scale = self.scale

            lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]

            return org_forwarded + torch.cat(lxs, dim=-1) * self.multiplier * scale

    @torch.no_grad()
    def initialize_norm_cache(self, org_module_weight: Tensor):
        n_rows = org_module_weight.shape[0]
        sample_size = min(1000, n_rows)

        indices = torch.randperm(n_rows)[:sample_size]

        weights_float32 = org_module_weight.to(dtype=torch.float32)
        sampled_weights = weights_float32[indices].to(device=self.device)

        sampled_norms = torch.norm(sampled_weights, dim=1, keepdim=True)

        self.org_weight_norm_estimate = sampled_norms.mean()
        self.org_weight_norm_std = sampled_norms.std()

        del sampled_weights, weights_float32

    @torch.no_grad()
    def update_norms(self):
        if self.ggpo_beta is None or self.ggpo_sigma is None:
            return

        if self.training is False:
            return

        module_weights = self.lora_up.weight @ self.lora_down.weight
        module_weights.mul(self.scale)

        self.weight_norms = torch.norm(module_weights, dim=1, keepdim=True)
        self.combined_weight_norms = torch.sqrt(
            (self.org_weight_norm_estimate**2)
            + torch.sum(module_weights**2, dim=1, keepdim=True)
        )

    @torch.no_grad()
    def update_grad_norms(self):
        if self.training is False:
            return

        lora_down_grad = None
        lora_up_grad = None

        for name, param in self.named_parameters():
            if name == "lora_down.weight":
                lora_down_grad = param.grad
            elif name == "lora_up.weight":
                lora_up_grad = param.grad

        if lora_down_grad is not None and lora_up_grad is not None:
            with torch.autocast(self.device.type):
                approx_grad = self.scale * (
                    (self.lora_up.weight @ lora_down_grad)
                    + (lora_up_grad @ self.lora_down.weight)
                )
                self.grad_norms = torch.norm(approx_grad, dim=1, keepdim=True)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class HydraLoRAModule(torch.nn.Module):
    """
    HydraLoRA: MoE-style multi-head LoRA with automatic style routing.
    Shared lora_down captures common features; per-expert lora_up_i heads specialize by style.
    Gate weights (from an external router on crossattn_emb) select expert contributions.
    Reference: docs/hydra-lora.md
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
        num_experts=12,
    ):
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            raise ValueError("HydraLoRAModule does not support Conv2d")

        in_dim = org_module.in_features
        out_dim = org_module.out_features

        self.lora_dim = lora_dim
        self.num_experts = num_experts

        # Shared down projection
        self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))

        # Fused per-expert up projections: (num_experts, out_dim, lora_dim)
        self.lora_up_weight = torch.nn.Parameter(
            torch.zeros(num_experts, out_dim, self.lora_dim)
        )

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        self.multiplier = multiplier
        self.org_module = org_module
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        self.fp32_accumulation = False
        self._timestep_mask = None
        self._hydra_gate = None  # (B, num_experts), set by LoRANetwork.set_hydra_gate()

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        if self.fp32_accumulation:
            lx = torch.nn.functional.linear(
                x.float(), self.lora_down.weight.float()
            )
        else:
            lx = self.lora_down(x)

        # timestep-dependent rank masking (T-LoRA compatibility)
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

        # Expert-weighted output via gate
        if self._hydra_gate is not None:
            gate = self._hydra_gate  # (B, num_experts)
            # Gate-weighted combined weight per batch element: (B, out_dim, lora_dim)
            combined = torch.einsum("be,eod->bod", gate, self.lora_up_weight)
            # Apply: lx is (B, ..., lora_dim), combined is (B, out_dim, lora_dim)
            orig_shape = lx.shape
            B = orig_shape[0]
            lx_3d = lx.reshape(B, -1, orig_shape[-1])  # (B, *, lora_dim)
            out = torch.bmm(lx_3d, combined.transpose(1, 2))  # (B, *, out_dim)
            out = out.reshape(*orig_shape[:-1], -1)  # restore prefix dims
        else:
            # Fallback: uniform average (inference without router)
            out = torch.nn.functional.linear(lx, self.lora_up_weight.mean(dim=0))

        if self.fp32_accumulation:
            return org_forwarded + (out * self.multiplier * scale).to(
                org_forwarded.dtype
            )
        return org_forwarded + out * self.multiplier * scale


class DoRAModule(LoRAModule):
    """
    DoRA: Weight-Decomposed Low-Rank Adaptation.
    Decomposes pretrained weight into magnitude and direction, applies LoRA to the direction only.
    Reference: https://arxiv.org/abs/2402.09353
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
    ):
        # split_dims and ggpo not supported for DoRA
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            lora_dim,
            alpha,
            dropout,
            rank_dropout,
            module_dropout,
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
            if torch.rand(1) < self.module_dropout:
                return org_out

        # fp32 accumulation: compute LoRA delta in fp32 for better precision
        if self.fp32_accumulation:
            lx = torch.nn.functional.linear(x.float(), self.lora_down.weight.float())
        else:
            lx = self.lora_down(x)

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

        if self.fp32_accumulation:
            lx = torch.nn.functional.linear(lx, self.lora_up.weight.float())
            # Cast LoRA delta back to native dtype early
            lora_out = (lx * self.multiplier * scale).to(org_out.dtype)
        else:
            lx = self.lora_up(lx)
            lora_out = lx * self.multiplier * scale

        # DoRA: scale by (magnitude / ||W_original||)
        mag_scale = (self.magnitude / self._org_weight_norm).to(org_out.dtype)

        # Broadcast mag_scale to match output shape
        if len(org_out.shape) == 3:
            mag_scale = mag_scale.unsqueeze(0).unsqueeze(0)  # [1, 1, out_dim]
        elif len(org_out.shape) == 2:
            mag_scale = mag_scale.unsqueeze(0)  # [1, out_dim]

        return mag_scale * (org_out + lora_out)


class OrthoLoRAModule(torch.nn.Module):
    """
    Orthogonal LoRA: QR-based weight parameterization for orthonormal initialization.
    Uses P @ diag(lambda) @ Q decomposition with frozen base copies to ensure zero output at init.
    Orthogonality regularization keeps P and Q close to orthonormal bases during training.
    Reference: T-LoRA (AAAI 2026)
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
    ):
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            raise ValueError("OrthoLoRAModule does not support Conv2d")

        in_dim = org_module.in_features
        out_dim = org_module.out_features

        self.lora_dim = lora_dim

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

        # Frozen base copies for residual (ensures zero output at init)
        self.register_buffer(
            "base_q_weight", self.q_layer.weight.data.clone().contiguous()
        )
        self.register_buffer(
            "base_p_weight", self.p_layer.weight.data.clone().contiguous()
        )
        self.register_buffer("base_lambda", self.lambda_layer.data.clone().contiguous())

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        self.multiplier = multiplier
        self.org_module = org_module
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        self._timestep_mask = None

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        dtype = self.q_layer.weight.dtype

        # timestep mask
        mask = self._timestep_mask if self._timestep_mask is not None else 1.0

        # trainable path
        q_out = self.q_layer(x.to(dtype)) * self.lambda_layer * mask

        # normal dropout
        if self.dropout is not None and self.training:
            q_out = torch.nn.functional.dropout(q_out, p=self.dropout)

        # rank dropout
        if self.rank_dropout is not None and self.training:
            rd_mask = (
                torch.rand((q_out.size(0), self.lora_dim), device=q_out.device)
                > self.rank_dropout
            )
            if len(q_out.size()) == 3:
                rd_mask = rd_mask.unsqueeze(1)
            q_out = q_out * rd_mask
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale

        p_out = self.p_layer(q_out)

        # frozen base path (residual subtraction ensures zero output at init)
        base_out = torch.nn.functional.linear(
            torch.nn.functional.linear(x.to(dtype), self.base_q_weight)
            * self.base_lambda
            * mask,
            self.base_p_weight,
        )

        lora_out = (p_out - base_out) * self.multiplier * scale
        return org_forwarded + lora_out.to(org_forwarded.dtype)

    def regularization(self):
        """Orthogonality regularization: ||P^T P - I||^2 + ||Q Q^T - I||^2"""
        p_reg = torch.sum(
            (
                self.p_layer.weight.T @ self.p_layer.weight
                - torch.eye(self.lora_dim, device=self.p_layer.weight.device)
            )
            ** 2
        )
        q_reg = torch.sum(
            (
                self.q_layer.weight @ self.q_layer.weight.T
                - torch.eye(self.lora_dim, device=self.q_layer.weight.device)
            )
            ** 2
        )
        return p_reg, q_reg


class ReFTModule(torch.nn.Module):
    """
    LoReFT: Low-Rank Representation Fine-Tuning.
    Applies a learned low-rank subspace edit to the output representation:
        h_new = h + R^T(Wh + b − Rh) * scale * multiplier
    where R is an orthogonal rotation selecting the intervention subspace,
    and W is a learned source projection.

    Zero-init: learned_source is initialized to match rotate_layer so delta=0 at init.
    Reference: Wu et al., "ReFT: Representation Finetuning for Language Models" (NeurIPS 2024)
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        reft_dim=4,
        alpha=1,
        dropout=None,
        module_dropout=None,
    ):
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            raise ValueError("ReFTModule does not support Conv2d")

        embed_dim = org_module.out_features
        self.reft_dim = reft_dim

        # R: orthogonal rotation (projects to intervention subspace)
        self.rotate_layer = torch.nn.Linear(embed_dim, reft_dim, bias=False)
        init_device = "cuda" if torch.cuda.is_available() else "cpu"
        r_rand = torch.randn(embed_dim, reft_dim, device=init_device)
        r_orth, _ = torch.linalg.qr(r_rand)  # (embed_dim, reft_dim)
        self.rotate_layer.weight.data = r_orth.T.cpu().clone().contiguous()
        del r_rand, r_orth

        # W: learned source projection — initialized to match R for zero output at init
        self.learned_source = torch.nn.Linear(embed_dim, reft_dim)
        self.learned_source.weight.data = self.rotate_layer.weight.data.clone()
        torch.nn.init.zeros_(self.learned_source.bias)

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()
        alpha = reft_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / reft_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        self.multiplier = multiplier
        self.org_module = org_module
        self.dropout = dropout
        self.module_dropout = module_dropout

        self._timestep_mask = None

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        h = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return h

        # Rh
        rotated = torch.nn.functional.linear(h, self.rotate_layer.weight)
        # Wh + b
        source = self.learned_source(h)

        # timestep-dependent masking
        if self._timestep_mask is not None and self.training:
            rotated = rotated * self._timestep_mask
            source = source * self._timestep_mask

        delta = source - rotated  # (Wh + b) - Rh

        if self.dropout is not None and self.training:
            delta = torch.nn.functional.dropout(delta, p=self.dropout)

        # R^T @ delta — project back to full space
        edit = torch.nn.functional.linear(delta, self.rotate_layer.weight.T)

        return h + edit * self.multiplier * self.scale

    def regularization(self):
        """Orthogonality regularization: ||R R^T - I||^2"""
        R = self.rotate_layer.weight  # (reft_dim, embed_dim)
        reg = torch.sum(
            (
                R @ R.T
                - torch.eye(self.reft_dim, device=R.device)
            )
            ** 2
        )
        return reg


class LoRAInfModule(LoRAModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        **kwargs,
    ):
        # no dropout for inference
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha)

        self.org_module_ref = [org_module]
        self.enabled = True
        self.network = None

    def set_network(self, network):
        self.network = network

    def merge_to(self, sd, dtype, device):
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"]
        org_dtype = weight.dtype
        org_device = weight.device
        weight = weight.to(torch.float)

        if dtype is None:
            dtype = org_dtype
        if device is None:
            device = org_device

        if self.split_dims is None:
            down_weight = sd["lora_down.weight"].to(torch.float).to(device)
            up_weight = sd["lora_up.weight"].to(torch.float).to(device)

            if len(weight.size()) == 2:
                weight = (
                    weight + self.multiplier * (up_weight @ down_weight) * self.scale
                )
            elif down_weight.size()[2:4] == (1, 1):
                weight = (
                    weight
                    + self.multiplier
                    * (
                        up_weight.squeeze(3).squeeze(2)
                        @ down_weight.squeeze(3).squeeze(2)
                    )
                    .unsqueeze(2)
                    .unsqueeze(3)
                    * self.scale
                )
            else:
                conved = torch.nn.functional.conv2d(
                    down_weight.permute(1, 0, 2, 3), up_weight
                ).permute(1, 0, 2, 3)
                weight = weight + self.multiplier * conved * self.scale

            org_sd["weight"] = weight.to(dtype)
            self.org_module.load_state_dict(org_sd)
        else:
            total_dims = sum(self.split_dims)
            for i in range(len(self.split_dims)):
                down_weight = sd[f"lora_down.{i}.weight"].to(torch.float).to(device)
                up_weight = sd[f"lora_up.{i}.weight"].to(torch.float).to(device)

                padded_up_weight = torch.zeros(
                    (total_dims, up_weight.size(0)), device=device, dtype=torch.float
                )
                padded_up_weight[
                    sum(self.split_dims[:i]) : sum(self.split_dims[: i + 1])
                ] = up_weight

                weight = (
                    weight + self.multiplier * (up_weight @ down_weight) * self.scale
                )

            org_sd["weight"] = weight.to(dtype)
            self.org_module.load_state_dict(org_sd)

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        up_weight = self.lora_up.weight.to(torch.float)
        down_weight = self.lora_down.weight.to(torch.float)

        if len(down_weight.size()) == 2:
            weight = self.multiplier * (up_weight @ down_weight) * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            weight = (
                self.multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2))
                .unsqueeze(2)
                .unsqueeze(3)
                * self.scale
            )
        else:
            conved = torch.nn.functional.conv2d(
                down_weight.permute(1, 0, 2, 3), up_weight
            ).permute(1, 0, 2, 3)
            weight = self.multiplier * conved * self.scale

        return weight

    def set_region(self, region):
        self.region = region
        self.region_mask = None

    def default_forward(self, x):
        if self.split_dims is None:
            lx = self.lora_down(x)
            lx = self.lora_up(lx)
            return self.org_forward(x) + lx * self.multiplier * self.scale
        else:
            lxs = [lora_down(x) for lora_down in self.lora_down]
            lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]
            return (
                self.org_forward(x)
                + torch.cat(lxs, dim=-1) * self.multiplier * self.scale
            )

    def fuse_weight(self):
        """Merge LoRA delta into org_module weight. Forward becomes a no-op (just org_forward)."""
        if getattr(self, "_fused", False):
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data += delta
        self._fused = True

    def unfuse_weight(self):
        """Remove LoRA delta from org_module weight."""
        if not getattr(self, "_fused", False):
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data -= delta
        self._fused = False

    def forward(self, x):
        if not self.enabled or self._fused:
            return self.org_forward(x)
        return self.default_forward(x)
