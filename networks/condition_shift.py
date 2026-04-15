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
