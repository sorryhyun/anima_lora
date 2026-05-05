"""Cache loading: one (latent, crossattn_emb) pair per sample."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

# post_image_dataset/<stem>_<HHHH>x<WWWW>_anima.npz  (latent is C×H×W, already H/8×W/8)
_LATENT_RE = re.compile(r"^(?P<stem>.+)_(?P<h>\d+)x(?P<w>\d+)_anima\.npz$")


def pick_cached_samples(
    dataset_dir: Path,
    n: int,
    image_h: int | None = None,
    image_w: int | None = None,
    shuffle_seed: int | None = None,
) -> list[tuple[str, Path, Path]]:
    """Return list of (stem, latent_npz_path, text_safetensors_path).

    When ``image_h`` and ``image_w`` are both set, restricts to samples whose
    cache filename encodes exactly that resolution (filename format:
    ``<stem>_<H>x<W>_anima.npz``). Required for ``--compile`` to converge to
    a single graph, and for direct cross-run comparability of v_fwd / v_rev
    norms (different bucket resolutions → different patch counts → different
    norms — see CLAUDE.md "Constant-token bucketing").

    With ``shuffle_seed=None`` (default) the bucket-matched stems are
    returned in alphabetical order — deterministic and reproducible.
    Pass an int to deterministically shuffle the candidate pool before
    truncating to ``n`` (used by ``make dcw`` to widen prompt diversity
    across the 14× cache headroom we previously ignored).
    """
    candidates: list[tuple[str, Path, Path]] = []
    for npz_path in sorted(dataset_dir.glob("*_anima.npz")):
        m = _LATENT_RE.match(npz_path.name)
        if not m:
            continue
        if image_h is not None and int(m.group("h")) != image_h:
            continue
        if image_w is not None and int(m.group("w")) != image_w:
            continue
        stem = m.group("stem")
        te_path = dataset_dir / f"{stem}_anima_te.safetensors"
        if not te_path.exists():
            continue
        candidates.append((stem, npz_path, te_path))

    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        rng.shuffle(candidates)

    return candidates[:n]


def load_cached(
    npz_path: Path, te_path: Path, text_variant: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (x_0 as (1,16,1,H,W) bf16, embed as (1,512,1024) bf16)."""
    with np.load(npz_path) as z:
        latent_keys = [k for k in z.keys() if k.startswith("latents_")]
        if not latent_keys:
            raise RuntimeError(f"no latents_* key in {npz_path}")
        lat = torch.from_numpy(z[latent_keys[0]])
    x_0 = lat.unsqueeze(0).unsqueeze(2).to(device, dtype=torch.bfloat16)

    sd = load_file(str(te_path))
    key = f"crossattn_emb_v{text_variant}"
    if key not in sd:
        raise KeyError(
            f"{key} not in {te_path}; available: "
            f"{[k for k in sd if k.startswith('crossattn_emb_')]}"
        )
    embed = sd[key].to(device, dtype=torch.bfloat16).unsqueeze(0)
    return x_0, embed
