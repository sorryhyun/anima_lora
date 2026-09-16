# Attention-projection fuse/split spec shared by `loading.py` and
# `lora_save.py`. The DiT runtime fuses self-attn ``q/k/v`` into
# ``qkv_proj`` and cross-attn ``k/v`` into ``kv_proj``; ComfyUI checkpoints
# store the unfused per-component projections. Save and load both walk the
# same component lists.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class AttnFuseSpec:
    """One fused-attention projection group.

    Attributes:
        attn_type: ``"self_attn"`` or ``"cross_attn"`` — the namespace in
            the lora_name where the projection lives.
        fused_letters: ``"qkv"`` or ``"kv"`` — name of the fused runtime
            projection (e.g. ``self_attn_qkv_proj``).
        component_letters: Per-component letter tuple (e.g. ``("q","k","v")``).
            Order is load-bearing: save-side splits concat in this order and
            load-side fuses re-concat in this order.
    """

    attn_type: str
    fused_letters: str
    component_letters: Tuple[str, ...]

    @property
    def fused_frag(self) -> str:
        """Fragment of the lora_name that identifies the fused projection,
        e.g. ``"self_attn_qkv_proj"``.
        """
        return f"{self.attn_type}_{self.fused_letters}_proj"

    def component_frag(self, letter: str) -> str:
        """Fragment of the lora_name for one component, e.g. ``"self_attn_q_proj"``."""
        return f"{self.attn_type}_{letter}_proj"


# Save-side splits ``self_attn_qkv_proj`` → q/k/v and ``cross_attn_kv_proj``
# → k/v; load-side re-fuses the inverse. A new fused projection only needs an
# entry here.
ATTN_FUSE_SPECS: Tuple[AttnFuseSpec, ...] = (
    AttnFuseSpec("self_attn", "qkv", ("q", "k", "v")),
    AttnFuseSpec("cross_attn", "kv", ("k", "v")),
)


def match_fused_spec(prefix: str) -> Optional[AttnFuseSpec]:
    """Return the AttnFuseSpec whose ``fused_frag`` ends ``prefix``, else None.

    Save-side dual of :func:`iter_split_groups` — the loader walks split
    component keys to detect groups that need re-fusing, while the saver
    walks fused prefixes (e.g. ``…self_attn_qkv_proj``) to detect groups
    that need splitting.
    """
    for spec in ATTN_FUSE_SPECS:
        if prefix.endswith(spec.fused_frag):
            return spec
    return None


def iter_split_groups(
    state_dict: Dict[str, torch.Tensor],
    sentinel_suffix: str,
) -> Iterator[Tuple[str, AttnFuseSpec]]:
    """Yield ``(shared_prefix, spec)`` for every split q/k/v group.

    A "group" is detected by the presence of the first component's
    ``{shared_prefix}{first_letter}_proj{sentinel_suffix}`` key — the
    caller then knows the surrounding component keys follow the same
    ``{shared_prefix}{letter}_proj.*`` shape and can refuse them. The
    discriminator is the sentinel suffix:

      * ``".lora_down.weight"`` — plain LoRA (each component carries an
        independent down projection).
      * ``".lora_up_weight"`` — Hydra-form (stacked per-expert ups, shared
        per-component down written under ``.lora_down.weight``).
      * ``".lora_ups.0.weight"`` — pre-stack Hydra-form on disk (one key
        per expert; ``_stack_lora_ups`` re-stacks before any iter call
        in practice).
      * ``".lora_downs.0.weight"`` — independent-A stacked experts on
        disk (per-expert downs, distinct from Hydra's shared down).

    ``shared_prefix`` ends with the underscore-separated attn namespace
    (e.g. ``"lora_unet_blocks_0_self_attn_"``) so callers can build any
    of the per-component key paths by appending ``{letter}_proj{...}``.
    """
    for spec in ATTN_FUSE_SPECS:
        first_letter = spec.component_letters[0]
        first_suffix = f"_{spec.attn_type}_{first_letter}_proj{sentinel_suffix}"
        seen: List[str] = []
        for key in list(state_dict.keys()):
            if not key.endswith(first_suffix):
                continue
            # Slice the component-projection suffix off so callers can rebuild
            # any letter's key by appending "{letter}_proj{...}".
            shared_prefix = key[: -len(f"{first_letter}_proj{sentinel_suffix}")]
            seen.append(shared_prefix)
        for shared_prefix in seen:
            yield shared_prefix, spec
