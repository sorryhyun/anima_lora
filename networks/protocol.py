"""Structural typing for the adapter-network surface (proposal Part B3).

The trainer and the inference engine already talk to adapter networks through a
de facto interface — they never import a concrete network class, they
``hasattr``-probe and call. Three concrete networks implement it today:

  * ``networks.lora_anima.network.LoRANetwork`` (the whole LoRA family),
  * ``networks.methods.easycontrol.EasyControlNetwork``,
  * ``networks.methods.soft_tokens.SoftTokensNetwork``

(the latter two via ``networks.methods.base.AdapterNetworkBase``).

This module writes that interface down as a ``typing.Protocol`` so it is
greppable, documented, and guarded by a contract test
(``tests/test_adapter_protocol.py``). It is a *description*, not an enforced
base class — the consumers keep duck-typing (``apply_router_conditioning``'s
``hasattr`` probes are the runtime contract). Both protocols are
``@runtime_checkable`` and non-data (every member is a method), so
``issubclass(SomeNetwork, AdapterNetwork)`` works without constructing a
network (which would need a live DiT).

Why two protocols: the core lifecycle is implemented by *every* adapter
network; the per-step router setters live only on the LoRA family (the method
networks have frozen-DiT splice forwards with no σ/FEI/timestep routing). The
optional surface is split out as ``RouterConditionableNetwork`` so a checker
can require the core everywhere and the routing surface only where it applies.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class AdapterNetwork(Protocol):
    """Core trainer-facing surface common to every adapter network.

    Mirrors what ``train.py`` calls on the network it builds, regardless of
    family: attach to the DiT, split optimizer param groups, toggle multiplier
    / gradient checkpointing, drive the epoch lifecycle, and load weights.
    ``prepare_optimizer_params_with_multiple_te_lrs`` is the always-present
    optimizer entry (``train.py`` falls back to it; the plain
    ``prepare_optimizer_params`` exists only on the method networks).
    """

    def apply_to(
        self,
        text_encoders,
        unet,
        apply_text_encoder: bool = ...,
        apply_unet: bool = ...,
    ) -> None: ...

    def load_weights(self, file): ...

    def prepare_optimizer_params_with_multiple_te_lrs(
        self, text_encoder_lr, unet_lr, default_lr
    ): ...

    def set_multiplier(self, multiplier: float) -> None: ...

    def is_mergeable(self) -> bool: ...

    def enable_gradient_checkpointing(self) -> None: ...

    def prepare_grad_etc(self, text_encoder, unet) -> None: ...

    def on_epoch_start(self, text_encoder, unet) -> None: ...

    def get_trainable_params(self): ...


@runtime_checkable
class RouterConditionableNetwork(Protocol):
    """Optional per-step conditioning surface (LoRA family only).

    These are the setters ``library.training.forward.router_conditioning``
    ``hasattr``-probes each step (timestep_mask → sigma → fei) plus the
    cross-attn content router fired by the inference engine. The method
    networks (EasyControl / SoftTokens) deliberately do NOT implement this —
    their probes no-op. Kept separate from :class:`AdapterNetwork` so a
    consumer can require the core lifecycle without forcing the routing surface
    onto frozen-DiT methods.
    """

    def set_timestep_mask(
        self, timesteps: torch.Tensor, max_timestep: float = ...
    ) -> None: ...

    def set_sigma(self, sigmas: torch.Tensor) -> None: ...

    def set_fei(self, fei: torch.Tensor) -> None: ...

    def set_crossattn_routing(self, crossattn_emb: torch.Tensor) -> None: ...
