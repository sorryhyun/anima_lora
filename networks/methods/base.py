"""Shared lifecycle base for non-LoRA adapter networks.

The ``AdapterNetworkBase`` subclasses under ``networks/methods/`` (easycontrol,
soft_tokens, register) expose the same trainer-facing protocol:

  - ``set_multiplier`` / ``is_mergeable`` / ``enable_gradient_checkpointing``
  - ``prepare_grad_etc`` / ``on_epoch_start`` / ``get_trainable_params``
  - ``prepare_optimizer_params(_with_multiple_te_lrs)``
  - ``save_weights`` / ``load_weights``

``AdapterNetworkBase`` owns the shared scaffolding; each method file
overrides ``metadata_fields``, ``state_dict_for_save``, and optimizer groups
when it needs more than one.

LoRA-family networks under ``networks/lora_anima/`` are *not* subclasses
here; they duck-type the same protocol (``networks/protocol.py``).
"""

from __future__ import annotations

import os
from typing import ClassVar, Optional

import torch
import torch.nn as nn

from library.training.hashing import precalculate_safetensors_hashes


def save_safetensors_with_hashes(
    state_dict: dict[str, torch.Tensor],
    file: str,
    metadata: Optional[dict[str, str]] = None,
) -> None:
    """Write ``state_dict`` to ``file`` as safetensors (or .pt fallback).

    For ``.safetensors`` paths, stamps ``sshs_model_hash`` / ``sshs_legacy_hash``
    into the metadata via ``precalculate_safetensors_hashes``. The hash precalc
    only sees ``ss_``-prefixed keys, so callers can add non-prefixed keys after
    without invalidating the hash.
    """
    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import save_file

        meta = dict(metadata or {})
        model_hash, legacy_hash = precalculate_safetensors_hashes(state_dict, meta)
        meta["sshs_model_hash"] = model_hash
        meta["sshs_legacy_hash"] = legacy_hash
        save_file(state_dict, file, meta)
    else:
        torch.save(state_dict, file)


class AdapterNetworkBase(nn.Module):
    """Base for ``networks/methods/*`` adapter networks.

    Subclasses set the two class attributes and override the small hooks
    they need:

      - ``network_module`` / ``network_spec`` (required) — stamped into
        ``ss_network_module`` / ``ss_network_spec`` save metadata.
      - ``mergeable`` (default ``False``) — return value of ``is_mergeable``.
      - ``metadata_fields()`` — extra ``ss_*`` stamps for save metadata.
      - ``state_dict_for_save(dtype)`` — bytes to write (default: full
        ``self.state_dict()`` detached, CPU-side, cast to ``dtype``).
      - ``prepare_optimizer_params_with_multiple_te_lrs(...)`` — only when
        more than one param group is needed.
      - ``load_weights(file)`` — only when the default strict-ish load can't
        do the validation the method needs.
    """

    network_module: ClassVar[str] = ""
    network_spec: ClassVar[str] = ""
    mergeable: ClassVar[bool] = False

    def __init__(self) -> None:
        super().__init__()
        self.multiplier: float = 1.0

    def set_multiplier(self, multiplier: float) -> None:
        self.multiplier = multiplier

    def is_mergeable(self) -> bool:
        return self.mergeable

    def enable_gradient_checkpointing(self) -> None:
        # DiT handles its own block-level checkpointing; nothing to do here.
        pass

    def prepare_grad_etc(self, text_encoder, unet) -> None:
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet) -> None:
        self.train()

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def prepare_optimizer_params_with_multiple_te_lrs(
        self, text_encoder_lr, unet_lr, default_lr
    ):
        del text_encoder_lr
        lr = unet_lr or default_lr
        params = [{"params": self.get_trainable_params(), "lr": lr}]
        descriptions = [self.network_spec or "adapter"]
        return params, descriptions

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr=None):
        params, _ = self.prepare_optimizer_params_with_multiple_te_lrs(
            text_encoder_lr, unet_lr, default_lr
        )
        return params

    def metadata_fields(self) -> dict[str, str]:
        """Method-specific ``ss_*`` keys to stamp into the safetensors metadata."""
        return {}

    def state_dict_for_save(self, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        """State dict actually written to disk (CPU-side, cast to ``dtype``)."""
        return {k: v.detach().cpu().to(dtype) for k, v in self.state_dict().items()}

    def save_weights(self, file, dtype, metadata) -> None:
        dtype = dtype or torch.bfloat16
        sd = self.state_dict_for_save(dtype)
        meta: dict[str, str] = dict(metadata or {})
        if self.network_module:
            meta["ss_network_module"] = self.network_module
        if self.network_spec:
            meta["ss_network_spec"] = self.network_spec
        meta.update(self.metadata_fields())
        save_safetensors_with_hashes(sd, file, meta)

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
        return self.load_state_dict(weights_sd, strict=False)
