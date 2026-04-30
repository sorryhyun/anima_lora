"""REPA-style auxiliary alignment loss (Yu et al., arXiv:2410.06940).

Adds a single auxiliary loss alongside the main flow-matching objective:
  - Hook the output of one mid-block of the DiT during training.
  - Pool the [B, T, H, W, D] feature tensor spatially to [B, D].
  - Project D -> d_enc via a small trainable MLP head.
  - Cosine-align against the spatial mean of the cached PE-Core feature
    [B, T_pe, d_enc] -> [B, d_enc].
  - L_repa = repa_weight * (1 - cos_sim).mean(), broadcast across the
    per-sample flow-matching loss in the LossComposer's stage 2.

v0 design choices:
  - Global mean-pool both sides. The REPA paper aligns per-token (assumes a
    matching token grid between SiT and the vision encoder). DiT patch grid
    (~64x64 at 1024px) and PE-Core grid (24x24 at 336px) don't match, so v0
    pools both to a single global vector. Upgrading to per-token (adaptive
    avg_pool2d to a common grid) is a self-contained change in
    ``_repa_loss`` if v0 shows traction.
  - Hook on ``unet.blocks[args.repa_layer]`` (default 8 of 28) — REPA paper
    uses block 8 for SiT-XL (28 blocks).
  - PE features come from ``batch['ip_features']``. ``train.py`` forces
    ``dataset.ip_features_cache_to_disk = True`` when ``--use_repa`` is set
    so ``{stem}_anima_pe.safetensors`` sidecars (produced by
    ``make preprocess-pe``) are loaded into the batch — no new
    preprocessing pipeline.
  - The trainable MLP head is attached to the LoRA network as
    ``network.repa_head`` (mirrors ``network.apex_condition_shift``);
    ``prepare_optimizer_params_with_multiple_te_lrs`` registers it as its
    own param group with LR ``repa_lr_scale * unet_lr``.

Resume-from-checkpoint: the LoRA save pipeline currently does not include
``repa_head`` weights, so warm-starting from a previously-saved adapter
re-inits the head from scratch. The head is small and re-converges in a few
hundred steps; acceptable for v0.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from library.training.method_adapter import (
    ForwardArtifacts,
    MethodAdapter,
    SetupCtx,
    StepCtx,
)

logger = logging.getLogger(__name__)


class REPAHead(nn.Module):
    """3-layer MLP projecting DiT hidden dim -> vision-encoder feature dim.

    Matches the ``h_phi`` shape from the REPA paper (3 linear + SiLU). Last
    layer initialised near-zero so step 0 cosine is unbiased and the head
    starts learning the projection from a small-norm output (the alignment
    signal then comes from gradient pulling the head toward PE-aligned
    directions, not from arbitrary random initial directions).
    """

    def __init__(self, dit_dim: int, hidden_dim: int, encoder_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dit_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, encoder_dim)
        nn.init.normal_(self.fc3.weight, std=1e-3)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        return self.fc3(x)


class REPAMethodAdapter(MethodAdapter):
    """Bridges REPA into AnimaTrainer's adapter dispatch.

    Setup: assert the LoRA network has ``repa_head`` and PE feature caching
    is on; install a forward post-hook on ``unet.blocks[repa_layer]``.
    Step: clear the captured-output slot and stash batch['ip_features'].
    Extra forward: project + pool + cosine, return the scalar in
    ``loss_aux['repa']['loss']`` for ``_repa_loss`` to consume.
    """

    name = "repa"

    def __init__(self) -> None:
        self._captured: Optional[torch.Tensor] = None
        self._pe_features: Optional[torch.Tensor] = None
        self._hook_handle = None

    def on_network_built(self, ctx: SetupCtx) -> None:
        net = ctx.network
        if getattr(net, "repa_head", None) is None:
            raise ValueError(
                "--use_repa requires the network module to expose a 'repa_head' "
                "submodule (attached automatically by networks.lora_anima.factory "
                "when use_repa=true is in the network kwargs)."
            )
        if not getattr(ctx.args, "ip_features_cache_to_disk", False):
            raise ValueError(
                "--use_repa requires --ip_features_cache_to_disk so that "
                "batch['ip_features'] is populated from {stem}_anima_pe.safetensors. "
                "Run `make preprocess-pe` first; the trainer auto-enables this "
                "flag when --use_repa is set, so reaching this error means the "
                "auto-propagation in train.py was bypassed."
            )

        layer = int(getattr(ctx.args, "repa_layer", 8))
        blocks = ctx.unet.blocks
        if not (0 <= layer < len(blocks)):
            raise ValueError(
                f"--repa_layer={layer} out of range (DiT has {len(blocks)} blocks)"
            )

        def _hook(_module, _inputs, output: torch.Tensor) -> None:
            # Block._forward returns x_B_T_H_W_D (5D). Capture as-is; loss
            # function pools spatially. Don't .detach(): we want grad to flow
            # back through the DiT into the LoRA modules in earlier blocks.
            self._captured = output

        self._hook_handle = blocks[layer].register_forward_hook(_hook)

        head = net.repa_head
        ctx.accelerator.print(
            f"REPA: forward hook on block {layer}/{len(blocks)}; "
            f"head {head.fc1.in_features} -> {head.fc2.out_features} -> "
            f"{head.fc3.out_features}; weight={float(getattr(ctx.args, 'repa_weight', 0.5))}"
        )

    def prime_for_forward(
        self, ctx: StepCtx, batch, latents: torch.Tensor, *, is_train: bool
    ) -> None:
        # Drop any stash from a previous step before this step's forward runs.
        # Without this, a step that doesn't trigger the hook (e.g. a future
        # validation path that bypasses self.blocks) would silently reuse the
        # last training batch's hidden state.
        self._captured = None

        feats = batch.get("ip_features") if isinstance(batch, dict) else None
        if feats is None:
            raise RuntimeError(
                "REPA expected batch['ip_features'] but got None. Check that "
                "--ip_features_cache_to_disk is on (auto-enabled by --use_repa) "
                "and the {stem}_anima_pe.safetensors sidecars exist for every "
                "image in the dataset (run `make preprocess-pe`)."
            )
        self._pe_features = feats.to(ctx.accelerator.device, dtype=ctx.weight_dtype)

    def extra_forwards(
        self, ctx: StepCtx, primary: ForwardArtifacts
    ) -> Optional[dict]:
        if self._captured is None or self._pe_features is None:
            return None
        head = ctx.network.repa_head
        # Layout [B, T, H, W, D]; T=1 for images. Mean over T,H,W -> [B, D].
        h_pooled = self._captured.mean(dim=(1, 2, 3))
        # Run the head in its parameter dtype (bf16 typical) so backward is
        # numerically consistent with the weight dtype the optimizer steps on.
        head_dtype = next(head.parameters()).dtype
        proj = head(h_pooled.to(head_dtype))  # [B, d_enc]
        # PE features [B, T_pe, d_enc] -> [B, d_enc].
        pe_pooled = self._pe_features.mean(dim=1).to(proj.dtype)
        # cosine in fp32 — cos_sim is sensitive to small norms in low precision.
        cos = F.cosine_similarity(proj.float(), pe_pooled.float(), dim=-1)
        loss = (1.0 - cos).mean()
        return {"repa": {"loss": loss}}
