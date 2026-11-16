"""Per-method extension protocol for AnimaTrainer.

Concrete adapters live next to their network module (e.g.
``networks/methods/easycontrol.py::EasyControlMethodAdapter``) and are
instantiated by ``resolve_adapters`` based on ``args`` + the built network.

The trainer holds ``self._adapters: list[MethodAdapter]`` and dispatches
each lifecycle event to all of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from accelerate import Accelerator


@dataclass(frozen=True)
class SetupCtx:
    """One-time context handed to ``on_network_built`` after the network has
    been created and applied to the DiT."""

    args: Any
    accelerator: Accelerator
    network: Any
    unet: Any
    text_encoders: list
    weight_dtype: torch.dtype


@dataclass(frozen=True)
class StepCtx:
    """Per-step context handed to ``prime_for_forward`` / ``on_step_start``."""

    args: Any
    accelerator: Accelerator
    network: Any
    weight_dtype: torch.dtype


@dataclass(frozen=True)
class ValidationBaseline:
    """A "without-this-method" reference forward run alongside the primary
    validation pass. The trainer:
      1. snapshots the RNG before the primary forward,
      2. runs the primary (with the adapter fully active),
      3. for each baseline: rewinds the RNG, calls ``enter`` (mutates adapter
         state), re-runs ``process_batch``, then calls ``exit`` to restore.

    Same noise + same batch + same sigma → the delta isolates the adapter's
    contribution. Logged as ``<prefix>_baseline_<name>`` (avg) and
    ``<prefix>_baseline_<name>_delta`` (baseline − primary; positive ⇒ the
    method is helping)."""

    name: str
    enter: Callable[[], None]
    exit: Callable[[], None]


@dataclass(frozen=True)
class ForwardArtifacts:
    """Inputs and outputs of the primary DiT forward, handed to adapters
    that need to run additional forwards inside ``extra_forwards``.

    All tensors are in the layout the DiT call expects:
      - ``noisy_model_input``: 5D ``[B, C, 1, H, W]``
      - ``model_pred``:        5D ``[B, C, 1, H, W]`` (pre-squeeze)
      - ``timesteps``:         ``[B]`` in ``[0, 1]``
      - ``crossattn_emb``:     ``[B, S, D]`` after any prefix/postfix injection
      - ``forward_kwargs``:    extra kwargs the trainer passed to ``anima(...)``
                                (e.g. ``crossattn_seqlens``)

    ``noise`` and ``latents`` are 4D ``[B, C, H, W]`` (post-shift-scale, post-squeeze).
    ``anima_call`` invokes the DiT with the same patched-network state the primary
    forward used; pass it positional inputs as ``anima_call(x_5d, t, c, padding_mask=..., **kw)``.
    ``is_train``: True when called from the training loop, False from validation.
    """

    anima_call: Callable
    noisy_model_input: torch.Tensor
    timesteps: torch.Tensor
    crossattn_emb: Optional[torch.Tensor]
    padding_mask: torch.Tensor
    forward_kwargs: dict
    model_pred: torch.Tensor
    noise: torch.Tensor
    latents: torch.Tensor
    is_train: bool


class MethodAdapter:
    """Base class for per-method trainer extensions.

    Defaults are no-ops — subclasses override only the hooks they need.
    """

    name: str = "base"

    # sigma_lowres (σ>threshold → demote-tier latent swap) compatibility. An
    # adapter that runs its own forwards / cond streams at a fixed grid
    # (EasyControl, soft-tokens) stays False; True only for grid-agnostic ones
    # (REPA pools the primary tokens to the encoder grid adaptively).
    sigma_demote_safe: bool = False

    def on_network_built(self, ctx: SetupCtx) -> None:
        """Called once after ``network.apply_to``. Validate runtime contract,
        load auxiliary encoders, install forward hooks, assert preconditions."""

    def on_step_start(self, ctx: StepCtx, batch, *, is_train: bool) -> None:
        """Called at the start of each train/val step (before forward)."""

    def prime_for_forward(
        self, ctx: StepCtx, batch, latents: torch.Tensor, *, is_train: bool
    ) -> None:
        """Push per-step state onto the network before the DiT forward.

        ``latents`` is the 4D ``[B, C, H, W]`` VAE latent (post-shift-scale,
        post-squeeze)."""

    def extra_forwards(self, ctx: StepCtx, primary: ForwardArtifacts) -> Optional[dict]:
        """Run additional DiT forwards and return aux loss tensors.

        Called once per step, AFTER the primary forward, INSIDE the same
        ``set_grad_enabled`` / ``autocast`` scope. Returns a dict that the
        trainer merges into ``loss_aux`` for the LossComposer (e.g.
        ``{"func_loss": tensor}``). Return ``None`` (or omit the override)
        when inactive for this step."""
        return None

    def after_backward(self, ctx: StepCtx) -> None:
        """Called once per train micro-step, AFTER ``accelerator.backward`` and
        BEFORE gradient clipping / the optimizer step.

        Lets an adapter inject extra gradient contributions that can't share the
        primary forward/backward (e.g. soft-tokens gradient-cached contrastive
        negatives, which can't reuse the anchor's block-swap cycle). Manual
        backwards here accumulate into the trainable params' ``.grad`` alongside
        the primary loss and are clipped/stepped with it. No-op by default."""

    def validation_baselines(self) -> list[ValidationBaseline]:
        """Return baselines to evaluate alongside the primary validation
        forward (see :class:`ValidationBaseline`). For when the primary FM
        loss can be met through a path the method shortcuts, so it does not
        show the adapter's contribution. Default empty."""
        return []

    def on_epoch_end(self, ctx: StepCtx) -> None:
        """Called once at the end of each epoch on the main process."""

    def metrics(self, ctx) -> dict:
        """Return log-step keys for this adapter (TensorBoard / W&B keys).

        Adapters that surface internal counters or auxiliary losses override
        this. ``ctx`` is the ``MetricContext`` from ``library.training.metrics``;
        it carries ``args`` and ``network``. Default empty.
        """
        return {}


def resolve_adapters(args, network) -> list[MethodAdapter]:
    """Sniff ``args`` + ``network`` and return the adapters that apply.

    Imports each adapter lazily so this module stays cheap to import.
    """
    adapters: list[MethodAdapter] = []
    if getattr(args, "use_easycontrol", False):
        from networks.methods.easycontrol import EasyControlMethodAdapter

        adapters.append(EasyControlMethodAdapter())
    # Soft-tokens contrastive and REPA are detected off weights the factory
    # stamps on the network (their config rides the network kwargs, not args).
    if float(getattr(network, "_contrastive_target_weight", 0.0) or 0.0) > 0.0:
        from networks.methods.soft_tokens import SoftTokensMethodAdapter

        adapters.append(SoftTokensMethodAdapter())
    if float(getattr(network, "_repa_weight", 0.0) or 0.0) > 0.0:
        from library.training.repa import REPAMethodAdapter

        adapters.append(REPAMethodAdapter())
    return adapters
