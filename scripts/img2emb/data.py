"""Shared data-loading + loss utilities for the img2emb resampler pipeline.

Every function/class here is consumed by production training stages
(``phase1_5_anchored`` for pretrain, ``phase2_flow`` for finetune) and/or the
phase-0/phase-1 bench trainers under ``bench/img2emb/``. The analysis that used
to share the same file lives back at ``bench/img2emb/phase0_probes.py`` and
imports from here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from library.io.cache import discover_cached_images  # noqa: E402


def active_slice(active_lengths: list[int]) -> torch.Tensor:
    """(N, S) boolean mask — True where slot < that row's active length."""
    N = len(active_lengths)
    S = 512
    L = torch.tensor(active_lengths, dtype=torch.long)
    arange = torch.arange(S).unsqueeze(0).expand(N, -1)
    return arange < L.unsqueeze(1)  # (N, S)


class _VariantMeanDataset(Dataset):
    """Load one TE file, zero-clamp the padded tail across every variant,
    return the variant-mean ``(S, D)`` slice. Parallelizable via DataLoader."""

    def __init__(self, te_paths: list[str], active_lengths: list[int]):
        self.te_paths = te_paths
        self.active_lengths = active_lengths

    def __len__(self) -> int:
        return len(self.te_paths)

    def __getitem__(self, idx: int):
        sd = load_file(self.te_paths[idx])
        variant_keys = sorted(k for k in sd.keys() if k.startswith("crossattn_emb_v"))
        if variant_keys:
            variants = [sd[k].float() for k in variant_keys]
        elif "crossattn_emb" in sd:
            variants = [sd["crossattn_emb"].float()]
        else:
            raise RuntimeError(f"No crossattn_emb in {self.te_paths[idx]}")
        stacked = torch.stack(variants, dim=0)  # (V, S, D)
        L = int(self.active_lengths[idx])
        if L < stacked.shape[1]:
            stacked[:, L:] = 0
        return idx, stacked.mean(dim=0), len(variants)


def _resolve_te_paths(image_dir: str, stems: list[str]) -> list[str]:
    """Align the TE file paths in ``image_dir`` to the cached ``stems`` order."""
    te_by_stem = {
        img.stem: img.te_path
        for img in discover_cached_images(image_dir)
        if img.te_path is not None
    }
    missing = [s for s in stems if s not in te_by_stem]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} stems listed in the cache have no TE file under "
            f"{image_dir} (first missing: {missing[0]}). Point --image_dir at "
            "the same dataset extract_features.py was run against."
        )
    return [te_by_stem[s] for s in stems]


def _load_targets_mean(
    te_paths: list[str],
    active_lengths: list[int],
    num_workers: int,
) -> tuple[torch.Tensor, int]:
    """Materialize the variant-mean targets ``(N, S, D)`` fp32 in RAM.

    Pre-allocated + filled by index via DataLoader workers — peak RAM is the
    final 2 GB tensor, not the old 8 GB per-variant stack.
    """
    ds = _VariantMeanDataset(te_paths, active_lengths)
    loader = DataLoader(
        ds,
        batch_size=16,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=num_workers > 0,
    )
    N = len(ds)
    out: torch.Tensor | None = None
    V_seen: int | None = None
    for idx, mean_var, n_vs in tqdm(loader, desc="targets_mean"):
        batch_V = int(n_vs[0].item())
        if V_seen is None:
            V_seen = batch_V
            S, D = int(mean_var.shape[1]), int(mean_var.shape[2])
            out = torch.empty((N, S, D), dtype=torch.float32)
        elif batch_V != V_seen:
            raise RuntimeError(
                f"Variant count differs across images: {V_seen} vs {batch_V}"
            )
        out[idx] = mean_var.float()
    assert out is not None and V_seen is not None
    return out, V_seen


def load_cache(cache_dir: Path, image_dir: str, encoder: str, num_workers: int):
    """Return dict with pooled / tokens / targets_mean / active_lengths / split /
    te_paths.

    Targets are no longer cached in one big safetensors; variant-mean is
    materialized on demand from the per-image ``*_anima_te.safetensors`` in
    ``image_dir``, and ``te_paths`` is passed through for the resampler to
    stream per-sample variants during training.
    """
    stems = json.loads((cache_dir / "stems.json").read_text())
    split = json.loads((cache_dir / "split.json").read_text())
    act = json.loads((cache_dir / "active_lengths.json").read_text())

    te_paths = _resolve_te_paths(image_dir, stems)
    targets_mean, V = _load_targets_mean(te_paths, act["active_lengths"], num_workers)
    V = int(act.get("num_variants", V))

    pooled = load_file(str(cache_dir / "features" / f"{encoder}_pooled.safetensors"))[
        "pooled"
    ]
    tokens = load_file(str(cache_dir / "features" / f"{encoder}_tokens.safetensors"))[
        "tokens"
    ]
    return {
        "stems": stems,
        "split": split,
        "active_lengths": act["active_lengths"],
        "num_variants": V,
        "te_paths": te_paths,           # list[str] aligned to stems order
        "targets_mean": targets_mean,   # (N, S, D) fp32
        "pooled": pooled,               # (N, D_enc) fp32
        "tokens": tokens,               # (N, T, D_enc) bf16
    }


def _resampler_loss(
    pred: torch.Tensor,      # (B, S, D)
    target: torch.Tensor,    # (B, S, D)
    mask: torch.Tensor,      # (B, S) bool, True in active region
    cos_w: float,
    zero_w: float,
):
    pred_f = pred.float()
    target_f = target.float()
    active = mask.unsqueeze(-1).float()        # (B, S, 1)
    inactive = (1.0 - active)

    # MSE on active region (normalized by active element count)
    diff = (pred_f - target_f) ** 2
    denom_act = active.sum().clamp_min(1.0) * pred_f.shape[-1]
    mse_active = (diff * active).sum() / denom_act

    # Cosine loss on active slots
    if cos_w > 0:
        pred_n = F.normalize(pred_f, dim=-1, eps=1e-8)
        target_n = F.normalize(target_f, dim=-1, eps=1e-8)
        cos = (pred_n * target_n).sum(dim=-1)  # (B, S)
        cos_loss_num = ((1.0 - cos) * mask.float()).sum()
        cos_loss = cos_loss_num / mask.float().sum().clamp_min(1.0)
    else:
        cos_loss = torch.tensor(0.0, device=pred.device)

    # Zero-pad penalty
    if zero_w > 0:
        pad_sq = (pred_f ** 2) * inactive
        denom_inact = inactive.sum().clamp_min(1.0) * pred_f.shape[-1]
        pad_loss = pad_sq.sum() / denom_inact
    else:
        pad_loss = torch.tensor(0.0, device=pred.device)

    total = mse_active + cos_w * cos_loss + zero_w * pad_loss
    return total, {
        "mse_active": float(mse_active.detach().item()),
        "cos_loss": float(cos_loss.detach().item()),
        "pad_loss": float(pad_loss.detach().item()),
    }


class _ResamplerTrainDataset(Dataset):
    """Streams per-sample training targets on demand.

    Yields ``(full_idx, target[S, D] bf16, L)`` where ``full_idx`` is the index
    into the original (N-long) stems list and ``target`` is one uniformly-
    sampled variant (of V) with its padded tail zero-clamped. The caller
    looks up tokens via ``tokens_all[full_idx]`` in the main process.

    DataLoader re-seeds ``torch.initial_seed()`` per worker per epoch, so
    ``torch.randint`` below gives independent variant draws across workers.
    """

    def __init__(
        self,
        train_indices: list[int],
        te_paths: list[str],
        active_lengths: list[int],
        num_variants: int,
    ):
        self.train_indices = train_indices
        self.te_paths = te_paths
        self.active_lengths = active_lengths
        self.V = num_variants

    def __len__(self) -> int:
        return len(self.train_indices)

    def __getitem__(self, pos: int):
        full_idx = self.train_indices[pos]
        sd = load_file(self.te_paths[full_idx])
        if self.V > 1:
            v = int(torch.randint(0, self.V, (1,)).item())
            target = sd[f"crossattn_emb_v{v}"].float()
        else:
            target = sd["crossattn_emb"].float()
        L = int(self.active_lengths[full_idx])
        if L < target.shape[0]:
            target = target.clone()
            target[L:] = 0
        return full_idx, target.to(torch.bfloat16), L
