# HydraLoRA: MoE-style multi-head LoRA with layer-local routing.

import math

import torch

from networks.lora_modules.base import BaseLoRAModule


def _sigma_sinusoidal_features(
    sigma: torch.Tensor, sigma_feature_dim: int
) -> torch.Tensor:
    """Sinusoidal σ features matching the DiT t_embedder functional form.

    Shared helper (also used by postfix-sigma, inlined there for historical
    self-containedness). Kept here so HydraLoRAModule / OrthoHydraLoRAExpModule
    can reuse the identical spectrum without cross-module coupling.
    """
    t = sigma.flatten().float()
    half_dim = sigma_feature_dim // 2
    exponent = (
        -math.log(10000)
        * torch.arange(half_dim, dtype=torch.float32, device=t.device)
        / max(half_dim, 1)
    )
    freqs = torch.exp(exponent)
    angles = t[:, None] * freqs[None, :]  # [B, half_dim]
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


class HydraLoRAModule(BaseLoRAModule):
    """
    HydraLoRA: MoE-style multi-head LoRA with layer-local routing.
    Shared lora_down captures common features; per-expert lora_up heads specialize.
    Each module owns its own router that reads the layer input and emits per-sample gates,
    so specialization is learned per-layer rather than globally.
    Reference: docs/methods/hydra-lora.md

    Optional σ-conditional routing (Track B, timestep-hydra.md): when
    ``sigma_feature_dim > 0``, ``sinusoidal(σ)`` is **concatenated** to the
    pooled rank-R router input, making ``self.router`` a
    ``Linear(r + sigma_feat, E)``. The σ-feature columns of the router
    weight are zero-init, so step-0 behavior is identical to the no-σ router
    and σ-dependence only emerges as those columns accumulate gradient.

    Why direct-input rather than the previous additive-bias ``sigma_mlp``: a
    bias-only σ path's gradient is ``dL/d logits · d_sigma_feat``, which
    vanishes whenever experts are undifferentiated (all ``score_e`` near
    equal → ``dL/d logit_e ≈ 0``). Feeding σ into the router's input avoids
    this chicken-and-egg problem — the σ columns train alongside the
    content columns on the same chain rule, so σ routing emerges as soon as
    the router learns anything at all.
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
        num_experts=4,
        channel_scale=None,
        sigma_feature_dim: int = 0,
        sigma_hidden_dim: int = 128,
    ):
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

        self.num_experts = num_experts
        self.in_dim = in_dim

        # Shared down projection
        self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))

        # Fused per-expert up projections: (num_experts, out_dim, lora_dim).
        # Zero-init keeps ΔW = 0 at step 0 (LoRA-safe). Per-expert symmetry is
        # broken by `expert_warmup_ratio` (random per-step expert-gradient
        # masking — see LoRANetwork.step_expert_warmup), not by an init perturb.
        self.lora_up_weight = torch.nn.Parameter(
            torch.zeros(num_experts, out_dim, self.lora_dim)
        )

        # Local router: reads [pooled rank-R signal | sinusoidal(σ)] → per-sample
        # expert gates. Operating in rank-R space (not raw in_dim) is load-bearing:
        # raw DiT inputs have 80–96× DC-bias outlier channels and ~4096 tokens, so
        # mean-pooling raw inputs collapsed the signal to near-constant DC noise and
        # left the router with no trainable gradient (see docs/methods/hydra-lora.md
        # §Fixes). `lora_down` is trained jointly, so signal-carrying directions
        # accumulate here and there are no large outliers to saturate softmax in bf16.
        self.sigma_feature_dim = int(sigma_feature_dim)
        # sigma_hidden_dim retained as an attribute for API compatibility but
        # is no longer used — the sigma_mlp hidden layer is gone.
        self.sigma_hidden_dim = int(sigma_hidden_dim)
        router_in_dim = self.lora_dim + self.sigma_feature_dim
        self.router = torch.nn.Linear(router_in_dim, num_experts, bias=True)
        # Split init: small-std on the pooled rank-R columns, zeros on the
        # σ-feature columns. Step 0 gate is then identical to the σ=off
        # router, and σ influence emerges only as those columns train.
        with torch.no_grad():
            self.router.weight.zero_()
            torch.nn.init.normal_(
                self.router.weight[:, : self.lora_dim], std=0.01
            )
            self.router.bias.zero_()

        self._register_channel_scale(self.lora_down.weight.data, channel_scale)

        self._last_gate = None  # (B, num_experts), cached each forward for balance loss
        self._sigma = None  # (B,) σ tensor; set externally by LoRANetwork.set_sigma
        # Expert-warmup gradient masking. Split into a Python bool gate and a
        # buffer holding the one-hot mask so torch.compile doesn't blow its
        # recompile limit every time the sampled expert rotates:
        #   * ``_warmup_active`` toggles only twice per run (entering and
        #     leaving the warmup window) — dynamo recompiles on transitions,
        #     not per step.
        #   * ``_expert_grad_mask`` is a buffer; value mutations are treated as
        #     dynamic by dynamo, so per-step re-sampling of the active expert
        #     does not recompile.
        # Set externally by LoRANetwork.step_expert_warmup. Default (all-ones
        # mask, gate off) is a no-op — every expert trains normally.
        self._warmup_active: bool = False
        self.register_buffer(
            "_expert_grad_mask",
            torch.ones(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def _compute_gate(self, lx: torch.Tensor) -> torch.Tensor:
        """Pool the rank-R `lora_down` output over the sequence dim, optionally
        concatenate sinusoidal(σ), run router, softmax.

        RMS (L2-norm) pool per channel: ``sqrt(mean(lx**2))``. Unlike mean pool,
        RMS does not cancel zero-mean activations by √N, so the pooled vector
        retains sample-level content over long sequences (L≈4096). Raw DiT inputs
        have DC-bias outliers that would break this aggregator in bf16, but
        rank-R space (post `lora_down`) is bounded by ``‖lora_down‖·‖x‖`` and
        has no such outliers, so RMS is safe here (see
        ``docs/methods/hydra-lora.md`` §Fixes).

        When ``sigma_feature_dim > 0``, ``sinusoidal(σ)`` is concatenated to
        the pooled vector. If ``_sigma`` is None (eval without set_sigma,
        pretraining warmup), zero-pad the σ slice so the router-input shape
        stays constant.
        """
        if lx.dim() >= 3:
            B = lx.shape[0]
            pooled = lx.reshape(B, -1, lx.shape[-1]).pow(2).mean(dim=1).sqrt()
        else:
            pooled = lx
        B = pooled.shape[0]
        # lx is fp32 (bottleneck policy) but router weights follow the adapter's
        # storage dtype (bf16 at inference) — align before matmul.
        pooled = pooled.to(self.router.weight.dtype)
        if self.sigma_feature_dim > 0:
            if self._sigma is not None:
                sigma_feat = _sigma_sinusoidal_features(
                    self._sigma, self.sigma_feature_dim
                ).to(pooled.dtype)
            else:
                sigma_feat = torch.zeros(
                    B,
                    self.sigma_feature_dim,
                    dtype=pooled.dtype,
                    device=pooled.device,
                )
            router_in = torch.cat([pooled, sigma_feat], dim=-1)
        else:
            router_in = pooled
        logits = self.router(router_in)  # (B, num_experts)
        return torch.softmax(logits, dim=-1)

    def forward(self, x):
        # Policy: bf16 storage, fp32 for the bottleneck matmuls. See
        # LoRAModule.forward for rationale. Gate/router stays in autocast
        # dtype — softmax over num_experts is fine in bf16 with the
        # small-std router init.
        org_forwarded = self.org_forward(x)

        if not self.enabled:
            return org_forwarded

        if self._skip_module():
            return org_forwarded

        # per-channel input rebalancing (SmoothQuant-style, see LoRAModule.forward)
        x_lora = self._rebalance(x)

        lx = torch.nn.functional.linear(
            x_lora.float(), self.lora_down.weight.float()
        )

        # Layer-local routing: gate is computed from the rank-R signal *before*
        # timestep masking / dropout — those are training-time perturbations and
        # the gate should behave identically at train and inference.
        gate = self._compute_gate(lx)  # (B, num_experts)
        if self.training:
            self._last_gate = gate  # cache for network-level balance loss

        # timestep-dependent rank masking (T-LoRA compatibility)
        if self._timestep_mask is not None and self.training:
            lx = lx * self._timestep_mask

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        lx, scale = self._apply_rank_dropout(lx)

        # Expert-warmup masking: keep full MoE inference (all experts contribute
        # via the learned gate) but let gradient flow only into the randomly-
        # chosen expert's up-weight slice. Breaks the cold-start deadlock where
        # zero-init experts receive identical gradients under a near-uniform
        # router. ``_warmup_active`` is a python bool that toggles twice per
        # run (enter/leave warmup); the per-step sampled expert is carried in
        # the ``_expert_grad_mask`` buffer, whose value changes don't trigger
        # dynamo recompiles.
        up_weight = self.lora_up_weight
        if self.training and self._warmup_active:
            expert_mask = self._expert_grad_mask.to(up_weight.dtype).view(-1, 1, 1)
            up_weight = (
                up_weight * expert_mask + up_weight.detach() * (1.0 - expert_mask)
            )

        # Gate-weighted combined weight per batch element: (B, out_dim, lora_dim)
        combined = torch.einsum(
            "be,eod->bod", gate.float(), up_weight.float()
        )
        # Apply: lx is (B, ..., lora_dim), combined is (B, out_dim, lora_dim)
        orig_shape = lx.shape
        B = orig_shape[0]
        lx_3d = lx.reshape(B, -1, orig_shape[-1])  # (B, *, lora_dim)
        out = torch.bmm(lx_3d, combined.transpose(1, 2))  # (B, *, out_dim)
        out = out.reshape(*orig_shape[:-1], -1)  # restore prefix dims

        return org_forwarded + (out * self.multiplier * scale).to(
            org_forwarded.dtype
        )
