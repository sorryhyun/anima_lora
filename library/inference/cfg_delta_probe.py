"""Tier-0 cross-attn-drive probe: log the CFG guidance-delta per denoising step.

``||v_cond - v_uncond|| / ||v_cond||`` over sigma is a zero-hook proxy for how
much the text pathway drives the update; the cos(v_cond, v_uncond) says whether
text only rescales the base velocity (cos→1) or rotates it. Whole-network only;
per-block resolution needs the cross-attn knockout probe.

Enable with ``ANIMA_CFG_DELTA_LOG=<path.json>``. Norms are per-sample, then
mean-pooled, so the curve is batch-size invariant.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch


class CfgDeltaProbe:
    """Accumulates per-step CFG guidance-delta diagnostics and dumps them to JSON.

    Created via :meth:`maybe_create` (returns ``None`` unless the env var is set),
    fed one row per step via :meth:`record`, and flushed once via :meth:`flush`.

    Each denoise loop (``--from_file`` runs one per prompt) flushes as a separate
    generation appended to the JSON; the first flush of a process truncates any
    stale file.
    """

    # generations flushed so far in this process
    _gen_counter = 0

    def __init__(self, path: str) -> None:
        self.path = path
        self.rows: list[dict] = []

    @classmethod
    def maybe_create(cls) -> Optional["CfgDeltaProbe"]:
        path = os.environ.get("ANIMA_CFG_DELTA_LOG")
        return cls(path) if path else None

    @staticmethod
    def _per_sample_norm(x: torch.Tensor) -> torch.Tensor:
        return x.float().flatten(1).norm(dim=1)

    def record(
        self,
        step: int,
        sigma: float,
        v_cond: torch.Tensor,
        v_uncond: torch.Tensor,
    ) -> None:
        vc = self._per_sample_norm(v_cond)
        vu = self._per_sample_norm(v_uncond)
        delta = self._per_sample_norm(v_cond - v_uncond)
        cos = torch.nn.functional.cosine_similarity(
            v_cond.float().flatten(1), v_uncond.float().flatten(1), dim=1
        )
        ratio = delta / vc.clamp_min(1e-12)
        self.rows.append(
            {
                "step": int(step),
                "sigma": float(sigma),
                "v_cond_norm": float(vc.mean()),
                "v_uncond_norm": float(vu.mean()),
                "delta_norm": float(delta.mean()),
                "ratio": float(ratio.mean()),
                # 1.0 => text only rescales the base velocity; <1 => it rotates it.
                "cos_cond_uncond": float(cos.mean()),
            }
        )

    def flush(self) -> None:
        if not self.rows:
            return
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        payload = {"generations": []}
        if CfgDeltaProbe._gen_counter > 0 and os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError):
                payload = {"generations": []}
        payload["generations"].append(
            {"gen": CfgDeltaProbe._gen_counter, "steps": self.rows}
        )
        CfgDeltaProbe._gen_counter += 1
        with open(self.path, "w") as f:
            json.dump(payload, f, indent=2)
