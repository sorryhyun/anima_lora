"""APEX condition-space shifting.

Implements c_fake = A*c + b (Eq. 9 of the APEX paper, arXiv:2604.12322).
The shifted condition is fed into the same DiT under the fake branch to
provide an endogenous adversarial reference for the real branch — no
discriminator, no precomputed teacher, no architectural change.

Three parameterizations:
  scalar  : A = a*I, b = beta*1                (2 params)
  diag    : A = diag(a), b                     (2D params)
  full    : A in R^{DxD}, b                    (D^2 + D params)

Default init follows Table 7 of the paper: a = -1.0, b = 0.5 — a moderate
negative scaling that sits inside the stable region observed empirically.
Phase 0 on a 2D toy reproduced convergence to this neighborhood on its own
(a -> -1.08, b -> 0.62) so it is a safe starting point for Phase 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from library.training.method_adapter import (
    ForwardArtifacts,
    MethodAdapter,
    SetupCtx,
    StepCtx,
)


class ConditionShift(nn.Module):
    MODES = ("scalar", "diag", "full")

    def __init__(
        self,
        dim: int,
        mode: str = "scalar",
        init_a: float = -1.0,
        init_b: float = 0.5,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(
                f"ConditionShift mode must be one of {self.MODES}, got {mode!r}"
            )
        self.dim = int(dim)
        self.mode = mode

        if mode == "scalar":
            self.a = nn.Parameter(torch.tensor(float(init_a)))
            self.b = nn.Parameter(torch.tensor(float(init_b)))
        elif mode == "diag":
            self.a = nn.Parameter(torch.full((self.dim,), float(init_a)))
            self.b = nn.Parameter(torch.full((self.dim,), float(init_b)))
        else:  # full
            self.A = nn.Parameter(float(init_a) * torch.eye(self.dim))
            self.b = nn.Parameter(torch.full((self.dim,), float(init_b)))

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """Apply the shift to a cross-attention embedding tensor.

        Args:
            c: [B, S, D] text conditioning.

        Returns:
            [B, S, D] shifted conditioning c_fake, matching c.dtype.
        """
        dt = c.dtype
        if self.mode == "scalar":
            return self.a.to(dt) * c + self.b.to(dt)
        if self.mode == "diag":
            return c * self.a.to(dt).view(1, 1, -1) + self.b.to(dt).view(1, 1, -1)
        # full
        return c @ self.A.to(dt).t() + self.b.to(dt).view(1, 1, -1)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, mode={self.mode}"


# ----------------------------------------------------------------- trainer integration


class ApexMethodAdapter(MethodAdapter):
    """APEX self-adversarial distillation (arXiv:2604.12322 §3) integration
    with the trainer adapter dispatch.

    Setup: enforce the warmup-or-warmstart precondition (cold-start regresses,
    proposal §7.3).
    Per step: increment the step counter consumed by the rampup schedule and
    surfaced to TensorBoard via ``state_for_metrics``.
    Extra forwards: run the two fake-branch DiT calls (eq. 11 + eq. 23) and
    one stop-gradient call for the L_mix target, then mix in the warmup/rampup
    weights so the LossComposer's APEX term sees ``lam_c_eff`` / ``lam_f_eff``
    already resolved. No-op outside training and on steps where the network
    has no ``apex_condition_shift`` attached.
    """

    name = "apex"

    def __init__(self) -> None:
        self.step = 0  # exposed via state_for_metrics under "apex_step"

    def on_network_built(self, ctx: SetupCtx) -> None:
        args = ctx.args
        has_warmup = (
            int(getattr(args, "apex_warmup_steps", 0) or 0) > 0
            or float(getattr(args, "apex_warmup_ratio", 0.0) or 0.0) > 0.0
        )
        has_weights = args.network_weights is not None
        if not (has_warmup or has_weights):
            raise ValueError(
                "APEX training requires either --apex_warmup_ratio > 0 "
                "(or --apex_warmup_steps > 0) or --network_weights <path> "
                "(warm-start). Cold-start training is known to regress vs. "
                "plain FM on the one-step objective; see proposal.md §7.3."
            )

    def on_step_start(self, ctx: StepCtx, batch, *, is_train: bool) -> None:
        # Counts at process_batch granularity; aligns with global_step under
        # gradient_accumulation_steps=1 and is close enough under larger
        # accumulation.
        if is_train:
            self.step += 1

    def extra_forwards(self, ctx: StepCtx, primary: ForwardArtifacts):
        if not primary.is_train:
            return None
        network = ctx.network
        if getattr(network, "apex_condition_shift", None) is None:
            return None
        if primary.crossattn_emb is None:
            return None  # APEX needs the cross-attn embedding to shift

        args = ctx.args
        anima = primary.anima_call
        noisy_model_input = primary.noisy_model_input  # 5D
        model_pred = primary.model_pred  # 5D
        timesteps = primary.timesteps
        crossattn_emb = primary.crossattn_emb
        padding_mask = primary.padding_mask
        kw = primary.forward_kwargs

        # Endpoint predictor (Eq. 11): x_fake = x_t - t * F_real, sg.
        t_bcast = timesteps.view(-1, 1, 1, 1, 1).to(model_pred.dtype)
        with torch.no_grad():
            x_fake = noisy_model_input - t_bcast * model_pred.detach()
        # Fresh noise + fresh t for the fake OT trajectory.
        z_fake = torch.randn_like(x_fake)
        t_fake = torch.rand(
            noisy_model_input.shape[0],
            device=noisy_model_input.device,
            dtype=timesteps.dtype,
        )
        t_fake_bcast = t_fake.view(-1, 1, 1, 1, 1).to(model_pred.dtype)
        x_fake_t = t_fake_bcast * z_fake + (1.0 - t_fake_bcast) * x_fake
        # target_fake = z_fake - x_fake (OT velocity on the fake traj)
        target_fake = z_fake - x_fake

        # Shifted condition (grad flows into ConditionShift via L_fake).
        c_fake = network.apex_condition_shift(crossattn_emb)

        # (1) Fake branch at real (x_t, t, c_fake) — stop-gradient target for
        #     L_mix. Paper §3.2: "v_fake := sg(F_theta(x_t, t, c_fake))". Under
        #     no_grad so the fake call doesn't contribute to the real-branch
        #     gradient path.
        with torch.no_grad():
            v_fake_sg = anima(
                noisy_model_input,
                timesteps,
                c_fake.detach(),
                padding_mask=padding_mask,
                **kw,
            )

        # T_mix_v in velocity space (Eq. 23 after Prop. 3 conversion):
        #   T_mix_v = (1-lam)*v_data + lam*v_fake_sg
        lam = float(getattr(args, "apex_lambda", 1.0))
        v_data_5d = (primary.noise - primary.latents).unsqueeze(2)  # [B,C,1,H,W]
        T_mix_v = ((1.0 - lam) * v_data_5d + lam * v_fake_sg).detach()

        # (2) Fake branch at (x_fake_t, t_fake, c_fake) — L_fake target. Grad
        #     flows back through both LoRA and ConditionShift.
        F_fake_on_fake_xt = anima(
            x_fake_t,
            t_fake,
            c_fake,
            padding_mask=padding_mask,
            **kw,
        )

        # Weighting for the L_fake term at its own timestep. Imported lazily so
        # this module doesn't pull in library.anima at definition time.
        from library.anima.training import compute_loss_weighting_for_anima
        from library.training.apex_loss import apex_schedule_weights

        weighting_fake = compute_loss_weighting_for_anima(
            weighting_scheme=args.weighting_scheme, sigmas=t_fake
        )

        # Resolve warmup/rampup schedule for L_c / L_f mixing weights.
        total_steps = int(getattr(args, "max_train_steps", 0) or 0)
        warmup_abs = int(getattr(args, "apex_warmup_steps", 0) or 0)
        rampup_abs = int(getattr(args, "apex_rampup_steps", 0) or 0)
        if warmup_abs <= 0:
            warmup_abs = int(
                float(getattr(args, "apex_warmup_ratio", 0.0) or 0.0) * total_steps
            )
        if rampup_abs <= 0:
            rampup_abs = int(
                float(getattr(args, "apex_rampup_ratio", 0.0) or 0.0) * total_steps
            )
        lam_c_eff, lam_f_eff = apex_schedule_weights(
            step=self.step,
            warmup_steps=warmup_abs,
            rampup_steps=rampup_abs,
            lam_c_target=float(getattr(args, "apex_lambda_c", 1.0)),
            lam_f_target=float(getattr(args, "apex_lambda_f", 1.0)),
        )

        return {
            "apex": {
                "T_mix_v": T_mix_v.squeeze(2),
                "F_fake_on_fake_xt": F_fake_on_fake_xt.squeeze(2),
                "target_fake": target_fake.squeeze(2),
                "weighting_fake": weighting_fake,
                "t_fake": t_fake,
                "lam_c_eff": lam_c_eff,
                "lam_f_eff": lam_f_eff,
            }
        }

    def state_for_metrics(self) -> dict:
        return {"apex_step": self.step}
