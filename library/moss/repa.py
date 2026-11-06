"""Relational REPA for the MOSS trainer: Anima's DoG-Gram term on one block.

One block's image-token output, pooled to the PE-Spatial grid, is matched to the
image's PE-Spatial patch tokens by Gram MSE after per-token L2-norm, with the
target DoG band-passed (``σ₁ = min(gh, gw)/16``): Anima's shipped form
(``_archive/repa/repa.md``), math from ``library/vision/repa.py``. No head, no
inference-time parameters; the gradient reaches the LoRA in blocks ≤ the layer.

MOSS-specific DiT side: before the Gram the pooled tokens drop the massive-activation
channels (``calibration/massive_channels.npz``, the set DAVE leaves out) and are
centered over tokens. Uncentered, those channels make every image token point one
way, the Gram is all ones, and the loss would mostly ask the LoRA to cancel the
massive-activation DC.

The capture rides ``model.layer_hook`` (called in ``run_layers`` outside the
compiled block, after the swapper has the block on the card), and only the
leading ``n_img`` rows are image tokens. Targets are the Anima PE sidecars of the
resized PNG each latent was encoded from (``pe_sidecar``); a latent that was
resampled from its source has no valid target and trains without the term.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from library.moss.dave import massive_channels
from library.vision.repa import (
    dog_standardize,
    pool_dit_tokens_to_grid,
    relational_gram_loss,
    resolve_pe_grid,
)
from library.vision.buckets import get_bucket_spec

ENCODER = "pe_spatial"
DOG_SIGMA1_DIV = 16.0


def pe_sidecar(latent_path: Path, stem: str, width: int, height: int, pe_root: Path):
    """``pe_root/<source dir name>/{stem}_anima_pe_spatial.safetensors`` for the
    PNG this latent was encoded from, or None (resampled, or no sidecar)."""
    with safe_open(str(latent_path), "pt") as f:
        meta = f.metadata() or {}
    if meta.get("resampled") != "False":
        return None
    if meta.get("source_size") != f"{width}x{height}":
        return None
    src = Path(meta.get("source", ""))
    path = pe_root / src.parent.name / f"{stem}_anima_{ENCODER}.safetensors"
    return path if path.exists() else None


def load_target(path: Path, height: int, width: int) -> tuple[torch.Tensor, tuple]:
    """DoG-standardised PE tokens ``(1, gh·gw, 768)`` fp32 on CPU and ``(gh, gw)``,
    for a latent of ``height × width`` tokens."""
    spec = get_bucket_spec(ENCODER)
    pe = load_file(str(path))["image_features"].float()[None]
    n_pe = pe.shape[1] - (1 if spec.use_cls else 0)
    gh, gw = resolve_pe_grid(spec, n_pe, height, width)
    if spec.use_cls:
        pe = pe[:, 1:]
    return dog_standardize(pe, gh, gw, DOG_SIGMA1_DIV).contiguous(), (gh, gw)


class REPATerm:
    """Captures block ``layer``'s image tokens on the next forward; ``loss()``
    returns the DoG-Gram MSE against the target set by ``prime``."""

    def __init__(self, transformer, layer: int):
        if not 0 <= layer < len(transformer.layers):
            raise ValueError(
                f"repa_layer {layer} outside 0..{len(transformer.layers) - 1}"
            )
        self.layer = layer
        massive = set(massive_channels().tolist())
        self._keep = torch.tensor(
            [c for c in range(transformer.config.dim) if c not in massive]
        )
        self._target = None
        self._captured = None
        transformer.layer_hook = self._hook

    def _hook(self, index, x, n_img):
        if index == self.layer and self._target is not None:
            self._captured = x[:, :n_img]
        return x

    def prime(self, target: torch.Tensor | None, grid, hw) -> None:
        """``target`` on the device (or None to skip this step), PE ``grid``
        ``(gh, gw)``, latent token grid ``hw`` ``(h, w)``."""
        self._target, self._grid, self._hw = target, grid, hw
        self._captured = None

    def loss(self) -> torch.Tensor | None:
        if self._target is None:
            return None
        if self._captured is None:
            raise RuntimeError(f"REPA: block {self.layer} hook did not fire")
        gh, gw = self._grid
        tok = pool_dit_tokens_to_grid(self._captured, self._hw, 1, gh, gw)
        tok = tok[..., self._keep.to(tok.device)]
        tok = tok - tok.mean(1, keepdim=True)
        self._captured = None
        return relational_gram_loss(tok, self._target)
