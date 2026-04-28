"""Per-method extension protocol for AnimaTrainer.

Concrete adapters live next to their network module (e.g.
``networks/methods/easycontrol.py::EasyControlMethodAdapter``) and are
instantiated by ``resolve_adapters`` based on ``args`` + the built network.

The trainer holds ``self._adapters: list[MethodAdapter]`` and dispatches
each lifecycle event to all of them. This replaces the per-method ``if
args.use_X:`` branches that used to live throughout ``train.py``.
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
class ForwardArtifacts:
    """Inputs and outputs of the primary DiT forward, handed to adapters
    that need to run additional forwards inside ``extra_forwards``.

    All tensors are in the layout the DiT call expects:
      - ``noisy_model_input``: 5D ``[B, C, 1, H, W]``
      - ``model_pred``:        5D ``[B, C, 1, H, W]`` (pre-squeeze)
      - ``timesteps``:         ``[B]`` in ``[0, 1]``
      - ``crossattn_emb``:     ``[B, S, D]`` after any prefix/postfix injection
      - ``forward_kwargs``:    extra kwargs the trainer passed to ``anima(...)``
                                (``crossattn_seqlens``, ``max_crossattn_seqlen``)

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
        post-squeeze). Adapters that don't need latents (e.g. IP-Adapter,
        which works off ``batch['images']``) can ignore the argument."""

    def extra_forwards(
        self, ctx: StepCtx, primary: ForwardArtifacts
    ) -> Optional[dict]:
        """Run additional DiT forwards and return aux loss tensors.

        Called once per step, AFTER the primary forward, INSIDE the same
        ``set_grad_enabled`` / ``autocast`` scope. Returns a dict that the
        trainer merges into ``loss_aux`` for the LossComposer (e.g.
        ``{"apex": {...}}``, ``{"func_loss": tensor}``). Return ``None`` (or
        omit the override) when inactive for this step."""
        return None

    def wants_split_backward(self, *, is_train: bool) -> bool:
        """If True, the trainer composes / backwards the real branch first
        (freeing forward-1 activations), then calls ``extra_forwards_fake``
        before composing / backwarding the fake branch. Used by APEX, where
        the two grad-tracked DiT forwards are autograd-disjoint and keeping
        both graphs live until a single backward roughly doubles peak
        activation memory.

        Default False — adapters that don't need this opt out and the trainer
        runs the legacy single-pass compose+backward.
        """
        return False

    def extra_forwards_fake(self, ctx: StepCtx) -> Optional[dict]:
        """Deferred fake-branch forwards. Called only when the adapter
        returned True from ``wants_split_backward`` and only after the
        trainer has backwarded the real-branch loss (forward-1 activations
        are gone by this point). Inside the trainer's own
        ``set_grad_enabled`` / ``autocast`` scope. Returns a dict the trainer
        merges into ``loss_aux`` before composing the fake-branch loss."""
        return None

    def on_epoch_end(self, ctx: StepCtx) -> None:
        """Called once at the end of each epoch on the main process."""

    def state_for_metrics(self) -> dict:
        """Return state the metrics layer should see in ``MetricContext.trainer_state``.

        Adapters that own internal counters / flags surfaced to TensorBoard
        (APEX step counter, …) override this. Default empty.
        """
        return {}


def resolve_adapters(args, network) -> list[MethodAdapter]:
    """Sniff ``args`` + ``network`` and return the adapters that apply.

    Imports each adapter lazily so this module stays cheap to import.
    """
    adapters: list[MethodAdapter] = []
    if getattr(args, "use_ip_adapter", False):
        from networks.methods.ip_adapter import IPAdapterMethodAdapter

        adapters.append(IPAdapterMethodAdapter())
    if getattr(args, "use_easycontrol", False):
        from networks.methods.easycontrol import EasyControlMethodAdapter

        adapters.append(EasyControlMethodAdapter())
    method = getattr(args, "method", None) or ""
    if method == "apex" or method.startswith("apex_"):
        from networks.methods.apex import ApexMethodAdapter

        adapters.append(ApexMethodAdapter())
    return adapters
