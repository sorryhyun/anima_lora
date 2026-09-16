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
    method is helping).

    Used by IP-Adapter where reference == target leaks information into the
    primary FM loss; the "no_ip" baseline reveals whether the IP path is
    actually contributing or just shortcutting via the copy path."""

    name: str
    enter: Callable[[], None]
    exit: Callable[[], None]


@dataclass(frozen=True)
class ComputeLossCtx:
    """Context for an adapter that *owns the whole training step*.

    Most methods ride the standard path (one DiT forward → ``flow_match`` base
    loss + additive aux from ``extra_forwards``). A method that has no
    ``target = noise - latents`` and runs its own multi-forward objective
    (e.g. BYG's bootstrap rollout + prior + cycle + identity) instead returns
    ``True`` from ``owns_training_step`` and computes the entire scalar loss in
    ``compute_loss``, bypassing ``get_noise_pred_and_target`` and the
    ``LossComposer``.

    Tensors are in the layout the trainer holds at the override point in
    ``_process_batch_inner``: ``latents`` is the 4D ``(B,C,H,W)`` VAE latent
    (post-shift-scale, post-squeeze); ``text_encoder_conds`` is the raw list of
    cached/encoded text-encoder outputs for the batch's primary caption.
    ``unet`` is the (adapter-patched) DiT; ``network`` is the trainable LoRA.
    The owner may run any number of ``unet(...)`` forwards and stage its own
    ``accelerator.backward`` calls (e.g. an independent identity graph) before
    returning the coupled scalar the training loop backwards once.
    """

    args: Any
    accelerator: Accelerator
    network: Any
    unet: Any
    noise_scheduler: Any
    weight_dtype: torch.dtype
    batch: dict
    latents: torch.Tensor
    text_encoder_conds: list
    is_train: bool


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

    # sigma_lowres (σ>threshold → demote-tier latent swap) compatibility.
    # Default False: an adapter that runs its own forwards / cond streams at a
    # fixed grid (EasyControl, BYG, soft-tokens) needs its own operating-point
    # probe before demotion is allowed under it. Set True
    # only for adapters that are grid-agnostic by construction (REPA: pools
    # the primary forward's tokens to the encoder grid adaptively).
    sigma_demote_safe: bool = False

    def on_network_built(self, ctx: SetupCtx) -> None:
        """Called once after ``network.apply_to``. Validate runtime contract,
        load auxiliary encoders, install forward hooks, assert preconditions."""

    def owns_training_step(self, args) -> bool:
        """Return True if this adapter computes the *entire* step loss itself.

        An owning adapter replaces the standard ``get_noise_pred_and_target`` +
        ``LossComposer`` path — the trainer delegates to ``compute_loss`` and
        returns its scalar directly. At most one active adapter may own the
        step (the trainer asserts this). Default: ride the standard path."""
        return False

    def compute_loss(self, ctx: "ComputeLossCtx") -> torch.Tensor:
        """Compute and return the full step loss (owning adapters only).

        Called from ``_process_batch_inner`` in place of the standard
        forward+composer when ``owns_training_step`` is True. The returned
        scalar is backwarded once by the training loop; the adapter may run
        additional ``ctx.accelerator.backward(...)`` calls internally for
        independent loss graphs (e.g. a separately-staged identity term).
        Only invoked when this adapter owns the step, so the base raises."""
        raise NotImplementedError(
            f"{type(self).__name__}.owns_training_step is True but "
            "compute_loss is not implemented"
        )

    def on_step_start(self, ctx: StepCtx, batch, *, is_train: bool) -> None:
        """Called at the start of each train/val step (before forward)."""

    def prime_for_forward(
        self, ctx: StepCtx, batch, latents: torch.Tensor, *, is_train: bool
    ) -> None:
        """Push per-step state onto the network before the DiT forward.

        ``latents`` is the 4D ``[B, C, H, W]`` VAE latent (post-shift-scale,
        post-squeeze). Adapters that don't need latents (e.g. IP-Adapter,
        which works off ``batch['images']``) can ignore the argument."""

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
        forward. For each baseline, the trainer re-runs ``process_batch`` on
        the same (batch, sigma) with the adapter perturbed (e.g. IP-Adapter
        zeroing its image conditioning) and logs the delta. Used when the
        primary FM loss is dominated by paths the method shortcircuits and
        therefore doesn't reflect the adapter's actual contribution. Default
        empty — adapters opt in."""
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
    if getattr(args, "use_byg", False):
        from networks.methods.byg import BYGMethodAdapter

        adapters.append(BYGMethodAdapter())
    # Soft-tokens contrastive: opt-in via a positive contrastive weight on the
    # built network (the objective leaves no learned params, so it's detected
    # off the network's target weight rather than an args flag).
    if float(getattr(network, "_contrastive_target_weight", 0.0) or 0.0) > 0.0:
        from networks.methods.soft_tokens import SoftTokensMethodAdapter

        adapters.append(SoftTokensMethodAdapter())
    # REPA v2 auxiliary alignment: opt-in via a positive weight stamped on the
    # network by the factory (use_repa=true). Detected off the network (the
    # config rides the network kwargs, not args).
    if float(getattr(network, "_repa_weight", 0.0) or 0.0) > 0.0:
        from library.training.repa import REPAMethodAdapter

        adapters.append(REPAMethodAdapter())
    return adapters
