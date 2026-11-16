"""``RunContext``: the run state ``setup.build_run`` builds and the loop reads."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from library.anima.models import Anima
from library.runtime.progress import ProgressSink
from library.training.ema import AdapterEMA
from networks.methods.turbo_dmd import TurboDMDNetwork

from .anchor_cache import AnchorDataset
from .metrics import StepGradLog, TauBinCriticLoss, TurboMetrics


@dataclass
class Batch:
    latents: torch.Tensor  # (B, C, H, W), sampler-state dtype
    crossattn_emb: torch.Tensor
    anchor_eps: torch.Tensor | None = None  # dpdmd.anchor_cache only
    anchor_v: torch.Tensor | None = None


@dataclass
class RunContext:
    model: Anima
    turbo: TurboDMDNetwork
    device: torch.device
    dtype: torch.dtype  # sampler state: latents, noise, σ
    cond_dtype: torch.dtype  # text embeddings

    # The disc pair is None with the GAN off.
    student_opt: torch.optim.Optimizer
    fake_opt: torch.optim.Optimizer
    disc_opt: torch.optim.Optimizer | None
    student_sched: object
    fake_sched: object
    disc_sched: object | None

    dataloader: torch.utils.data.DataLoader
    data_iter: object
    # dpdmd.anchor_cache: the dataset that adds (ε, v_target) to each item.
    anchor_cache: AnchorDataset | None

    # (view, x, t, c, *, no_grad, ...) → velocity; (x, t, c, c_null) → CFG'd teacher.
    forward: Callable[..., torch.Tensor]
    teacher_cfg_velocity: Callable[..., torch.Tensor]

    student_sigmas: list[float]
    teacher_anchor_sigmas: list[float]
    uncond_base: torch.Tensor  # T5("") crossattn

    writer: object | None
    progress_sink: ProgressSink | None
    grad_log: StepGradLog | None  # <run log>/grad_steps.jsonl
    console_steps: bool  # off a TTY: a logger line per log_interval
    metrics: TurboMetrics
    tau_profile: TauBinCriticLoss

    # Diversity probe on one held-out sample (None when off).
    val_cond: torch.Tensor | None
    val_latent_shape: tuple | None
    val_clean: torch.Tensor | None

    start_step: int
    int8_base_id: str
    ema: AdapterEMA | None
    # Mean student LR over the EMA's updates (0 under the constant schedule): the
    # loop steps the EMA at rate lr / ema_lr_mean, so its span is a share of summed LR.
    ema_lr_mean: float
    # time.monotonic() of the last console step line (off a TTY).
    last_console: float = 0.0

    def next_batch(self) -> Batch:
        """The next batch on the device; a new epoch (and anchor draw) at the end."""
        try:
            raw = next(self.data_iter)
        except StopIteration:
            if self.anchor_cache is not None:
                self.anchor_cache.next_epoch()  # before iter(): workers copy it
            self.data_iter = iter(self.dataloader)
            raw = next(self.data_iter)
        b = Batch(
            latents=raw["latents"].to(self.device, dtype=self.dtype, non_blocking=True),
            crossattn_emb=raw["crossattn_emb"].to(
                self.device, dtype=self.cond_dtype, non_blocking=True
            ),
        )
        if "anchor_eps" in raw:
            b.anchor_eps = raw["anchor_eps"].to(
                self.device, dtype=self.dtype, non_blocking=True
            )
            b.anchor_v = raw["anchor_v"].to(self.device, non_blocking=True)
        return b
