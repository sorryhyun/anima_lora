# Classic LoRA. `merge_to` bakes a checkpoint slice into the base Linear/Conv2d
# weight; `fuse_weight` bakes the live delta and turns forward into a no-op.

import logging
import math
from typing import Dict, List

import torch

from networks.attn_fuse import match_fused_spec
from networks.lora_modules.base import BaseLoRAModule

logger = logging.getLogger(__name__)


class LoRAModule(BaseLoRAModule):
    supports_conv2d = True

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        channel_scale=None,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__(
            lora_name,
            org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
        )

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(
                in_dim, self.lora_dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = torch.nn.Conv2d(
                self.lora_dim, out_dim, (1, 1), (1, 1), bias=False
            )
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self._register_channel_scale(self.lora_down.weight.data, channel_scale)

        # List wrapping prevents nn.Module from registering org_module as a
        # submodule (would double-count params). apply_to() deletes
        # self.org_module after rerouting forward, leaving this as the only
        # handle for fuse/unfuse.
        self.org_module_ref = [org_module]
        self._fused = False

    # Forward is the shared BaseLoRAModule scaffold; this class supplies the
    # down / up GEMMs (Linear-or-Conv2d dispatch) and the eval delta. The
    # T-LoRA gate is the inherited default (``lx * _timestep_mask``). The
    # ``_fused`` short-circuit + the dtype-policy commentary live in the base.

    def _down(self, x_lora, work):
        if isinstance(self.lora_down, torch.nn.Linear):
            return torch.nn.functional.linear(x_lora, self.lora_down.weight.to(work))
        return self.lora_down(x_lora)  # Conv2d

    def _up(self, lx, work):
        if isinstance(self.lora_up, torch.nn.Linear):
            return torch.nn.functional.linear(lx, self.lora_up.weight.to(work))
        return self.lora_up(lx)  # Conv2d

    def _eval_delta(self, x, org_forwarded):
        # Full-rank inference path: no T-LoRA mask, plain module calls (params
        # already in the model dtype), so no work-dtype cast dance.
        x_lora = self._rebalance(x)
        lx = self.lora_up(self.lora_down(x_lora))
        return lx * self.multiplier * self.scale

    def get_weight(self, multiplier=None):
        """Return the LoRA delta as a tensor matching org_module.weight shape."""
        if multiplier is None:
            multiplier = self.multiplier

        up_weight = self.lora_up.weight.to(torch.float)
        down_weight = self.lora_down.weight.to(torch.float)

        # Undo channel absorption so the merged delta applies to raw inputs.
        if self._has_channel_scale and down_weight.dim() == 2:
            down_weight = down_weight * self.inv_scale.to(down_weight).unsqueeze(0)

        if len(down_weight.size()) == 2:
            weight = multiplier * (up_weight @ down_weight) * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            weight = (
                multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2))
                .unsqueeze(2)
                .unsqueeze(3)
                * self.scale
            )
        else:
            conved = torch.nn.functional.conv2d(
                down_weight.permute(1, 0, 2, 3), up_weight
            ).permute(1, 0, 2, 3)
            weight = multiplier * conved * self.scale

        return weight

    def merge_to(self, sd, dtype, device):
        """Merge a per-LoRA state-dict slice into org_module.weight in-place.

        Alternative to apply_to: delta lands on the base weight, no forward
        hook. `sd` is the raw checkpoint slice — the LoRA module hasn't been
        load_state_dict'd at this point.
        """
        with torch.no_grad():
            weight = self.org_module.weight
            org_dtype = weight.dtype
            if dtype is None:
                dtype = org_dtype
            if device is None:
                device = weight.device

            w = weight.data.float()

            down_weight = sd["lora_down.weight"].to(torch.float).to(device)
            up_weight = sd["lora_up.weight"].to(torch.float).to(device)

            # Merged forward has no x rebalancing — undo absorption first.
            if "inv_scale" in sd:
                inv_scale = sd["inv_scale"].to(torch.float).to(device)
                if down_weight.dim() == 2:
                    down_weight = down_weight * inv_scale.unsqueeze(0)

            if len(w.size()) == 2:
                w += self.multiplier * (up_weight @ down_weight) * self.scale
            elif down_weight.size()[2:4] == (1, 1):
                w += (
                    self.multiplier
                    * (
                        up_weight.squeeze(3).squeeze(2)
                        @ down_weight.squeeze(3).squeeze(2)
                    )
                    .unsqueeze(2)
                    .unsqueeze(3)
                    * self.scale
                )
            else:
                conved = torch.nn.functional.conv2d(
                    down_weight.permute(1, 0, 2, 3), up_weight
                ).permute(1, 0, 2, 3)
                w += self.multiplier * conved * self.scale

            weight.data.copy_(w.to(dtype))

    def fuse_weight(self):
        """Bake LoRA delta into org_module.weight; subsequent forwards no-op."""
        if self._fused:
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data += delta
        self._fused = True

    def unfuse_weight(self):
        """Subtract a previously fused LoRA delta back out of org_module.weight."""
        if not self._fused:
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data -= delta
        self._fused = False


# ---------------------------------------------------------------------------
# Save-pipeline helpers (state_dict-level, no module instance required).
#
# Co-located with LoRAModule because they operate on the layout this class
# writes (``.lora_down.weight`` / ``.lora_up.weight`` / ``.alpha`` /
# optional ``.inv_scale``). The standard variant write fires these; the
# Hydra and Chimera writers also defuse their plain-LoRA legs by calling
# :func:`defuse_standard_qkv` directly.
# ---------------------------------------------------------------------------


def defuse_standard_qkv(state_dict: Dict[str, torch.Tensor]) -> None:
    """Split runtime-fused ``…_qkv_proj`` / ``…_kv_proj`` keys per-component.

    Operates on the plain LoRA layout (single ``.lora_down.weight`` +
    single ``.lora_up.weight`` per fused Linear). The down projection is
    cloned per component; the up projection (rows = concatenated output
    channels) is chunked along dim 0. ``.alpha`` / ``.inv_scale``
    (per_channel_scaling) get cloned/chunked alongside — ``inv_scale`` is
    shape ``[in_dim]`` and identical for q/k/v which all see the same Linear
    input, so it clones rather than chunks.

    Used by:
      * the standard write path,
      * the Hydra write path's "plain-LoRA leg" (modules excluded from
        ``router_targets`` save under the plain layout),
      * the Chimera write path's plain-LoRA leg (router_targets excludes
        attention projections by default — OrthoLoRA fallback lands as
        plain LoRA after the ortho distill step).
    """
    fused_groups: List[tuple] = []
    for key in list(state_dict.keys()):
        if not key.endswith(".lora_down.weight"):
            continue
        prefix = key.removesuffix(".lora_down.weight")
        spec = match_fused_spec(prefix)
        if spec is not None:
            fused_groups.append((prefix, spec))

    for prefix, spec in fused_groups:
        suffixes = spec.component_letters
        n = len(suffixes)
        down = state_dict.pop(f"{prefix}.lora_down.weight")
        up = state_dict.pop(f"{prefix}.lora_up.weight")
        alpha = state_dict.pop(f"{prefix}.alpha", None)
        inv_scale = state_dict.pop(f"{prefix}.inv_scale", None)

        up_chunks = up.chunk(n, dim=0)

        base_prefix = prefix.removesuffix(spec.fused_frag)
        for letter, up_chunk in zip(suffixes, up_chunks):
            new_prefix = base_prefix + spec.component_frag(letter)
            state_dict[f"{new_prefix}.lora_down.weight"] = down.clone()
            state_dict[f"{new_prefix}.lora_up.weight"] = up_chunk
            if alpha is not None:
                state_dict[f"{new_prefix}.alpha"] = alpha.clone()
            if inv_scale is not None:
                state_dict[f"{new_prefix}.inv_scale"] = inv_scale.clone()


def bake_inv_scale(state_dict: Dict[str, torch.Tensor]) -> None:
    """Fold per_channel_scaling ``inv_scale`` into ``lora_down`` and drop the key.

    ``per_channel_scaling`` (SmoothQuant-style channel absorption) bakes
    ``s_norm`` into the saved ``lora_down`` (``W[:,c] *= s_norm[c]``) and stores
    ``inv_scale = 1/s_norm`` separately; the trained forward is
    ``F.linear(x * inv_scale, down)``. Pre-folding ``down *= inv_scale`` makes
    the on-disk delta act on raw inputs — a standard LoRA that any consumer
    (stock ComfyUI, ``merge_to_dit``, third-party loaders) applies correctly
    without knowing the ``.inv_scale`` convention. This is exactly what every
    loader does on load (``LoRAModule.merge_to`` / ``get_weight`` / the inference
    factory's ``inv_scale``-keyed reconstruction), precomputed once at save.

    Operates on the split (post-defuse) layout: each ``<prefix>.inv_scale`` has
    a sibling ``<prefix>.lora_down.weight``. Run AFTER ``defuse_standard_qkv``.
    Mutates ``state_dict`` in place. Resume re-derives the reparameterization
    from calibration — see ``LoRANetwork.load_weights``'s re-absorb guard.
    """
    for key in list(state_dict.keys()):
        if not key.endswith(".inv_scale"):
            continue
        prefix = key.removesuffix(".inv_scale")
        down_key = f"{prefix}.lora_down.weight"
        inv_scale = state_dict.pop(key)
        down = state_dict.get(down_key)
        if down is None or down.dim() != 2:
            logger.warning(
                f"bake_inv_scale: no 2D sibling lora_down for {key}; "
                "dropping inv_scale unbaked (delta will be wrong)."
            )
            continue
        orig_dtype = down.dtype
        state_dict[down_key] = (
            down.to(torch.float)
            * inv_scale.to(device=down.device, dtype=torch.float).unsqueeze(0)
        ).to(orig_dtype)


def defuse_and_bake_standard(
    state_dict: Dict[str, torch.Tensor],
) -> None:
    """Standard write pipeline: qkv defuse + inv_scale bake."""
    defuse_standard_qkv(state_dict)
    bake_inv_scale(state_dict)
