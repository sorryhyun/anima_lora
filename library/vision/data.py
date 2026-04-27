"""Shared data-loading + loss utilities for the img2emb resampler pipeline.

Consumed by the phase-0/phase-1 bench trainers under ``bench/img2emb/`` and by
the archived img2emb training stages under ``archive/img2emb/``. Extracted
here so live consumers don't have to depend on the archived training code.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from library.io.cache import discover_cached_images


def active_slice(active_lengths: list[int], S: int = 512) -> torch.Tensor:
    """(N, S) boolean mask — True where slot < that row's active length."""
    N = len(active_lengths)
    L = torch.tensor(active_lengths, dtype=torch.long)
    arange = torch.arange(S).unsqueeze(0).expand(N, -1)
    return arange < L.unsqueeze(1)  # (N, S)


class _VariantMeanDataset(Dataset):
    """Load one TE file, zero-clamp the padded tail across every variant,
    return the variant-mean ``(S, D)`` slice. Parallelizable via DataLoader.

    Only used by the phase-0 diagnostic probes (``bench/img2emb/phase0_probes``)
    where the analytic OLS solution requires the per-image mean. All production
    training stages (phase 1 / 1.5 / 2) sample one variant per step via
    ``_ResamplerTrainDataset`` and never touch this class.
    """

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


def load_targets_mean(
    te_paths: list[str],
    active_lengths: list[int],
    num_workers: int,
) -> torch.Tensor:
    """Materialize the variant-mean targets ``(N, S, D)`` fp32 in RAM.

    Opt-in: only phase-0 diagnostic probes call this. The mean is a valid
    *diagnostic* target (order-invariant summary of the caption distribution)
    but a poor *training* target — it shrinks the norm under triangle
    inequality and sits off the T5 manifold. Production training stages use
    ``_ResamplerTrainDataset`` for per-step per-variant sampling instead.

    Pre-allocated + filled by index via DataLoader workers — peak RAM is the
    final ~2 GB tensor, not the old per-variant stack.
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
    assert out is not None
    return out


def _peek_target_shape(te_paths: list[str]) -> tuple[int, int]:
    """Return ``(S, D)`` of the cross-attn target by reading one TE file.

    Cheap stand-in for ``cache["targets_mean"].shape[1:]`` now that we don't
    materialize the mean up-front.
    """
    sd = load_file(te_paths[0])
    variant_keys = sorted(k for k in sd.keys() if k.startswith("crossattn_emb_v"))
    key = variant_keys[0] if variant_keys else "crossattn_emb"
    if key not in sd:
        raise RuntimeError(
            f"No crossattn_emb / crossattn_emb_v* key in {te_paths[0]}"
        )
    t = sd[key]
    return int(t.shape[0]), int(t.shape[1])


def load_cache(cache_dir: Path, image_dir: str, encoder: str, num_workers: int = 0):
    """Return dict with pooled / tokens / target_shape / active_lengths / split /
    te_paths.

    The mean target is no longer materialized eagerly — it's useless for
    training (see ``load_targets_mean`` docstring) and was a ~2 GB RAM tax.
    Phase-0 diagnostics that still want it call ``load_targets_mean`` directly.
    ``num_workers`` is kept in the signature for call-site compatibility but
    is unused here; pass it to ``load_targets_mean`` instead.
    """
    _ = num_workers
    stems = json.loads((cache_dir / "stems.json").read_text())
    split = json.loads((cache_dir / "split.json").read_text())
    act = json.loads((cache_dir / "active_lengths.json").read_text())

    te_paths = _resolve_te_paths(image_dir, stems)
    if "num_variants" not in act:
        raise RuntimeError(
            f"active_lengths.json at {cache_dir} missing 'num_variants'; "
            "re-run extract_features.py to regenerate."
        )
    V = int(act["num_variants"])
    S, D_y = _peek_target_shape(te_paths)

    pooled = load_file(str(cache_dir / "features" / f"{encoder}_pooled.safetensors"))[
        "pooled"
    ]
    tokens = load_file(str(cache_dir / "features" / f"{encoder}_tokens.safetensors"))[
        "tokens"
    ]

    # Optional per-variant pooled T5 targets for InfoNCE (see
    # ``compute_target_pooled`` in extract_features.py). Shape: (N, V, D).
    target_pooled: torch.Tensor | None = None
    tgt_pooled_path = cache_dir / "features" / "target_pooled.safetensors"
    if tgt_pooled_path.exists():
        target_pooled = load_file(str(tgt_pooled_path))["pooled"]

    return {
        "stems": stems,
        "split": split,
        "active_lengths": act["active_lengths"],
        "num_variants": V,
        "te_paths": te_paths,           # list[str] aligned to stems order
        "target_shape": (S, D_y),       # (S, D_y) int tuple
        "pooled": pooled,               # (N, D_enc) fp32
        "tokens": tokens,               # (N, T, D_enc) bf16
        "target_pooled": target_pooled, # (N, V, D_y) fp32 or None
    }


@torch.no_grad()
def match_target_to_pred(
    pred: torch.Tensor,         # (B, K, D)
    target: torch.Tensor,       # (B, K, D)
    mask_active: torch.Tensor,  # (B, K) bool
    anchor_mask: torch.Tensor,  # (B, K) bool
) -> tuple[torch.Tensor, dict[str, float]]:
    """Hungarian-permute the *content* rows of ``target`` (active & non-anchor)
    to minimize cosine distance against ``pred``; leave anchor + inactive rows
    in place. Returns ``(tgt_perm, diag)``.

    Why: DiT cross-attention has no K-side RoPE, so Σⱼ softmax(QK)ⱼ Vⱼ is
    permutation-invariant over K. The pretrain MSE/cos is the only stage that
    couples to slot order. Permuting the target to the prediction's order
    before MSE removes a supervision artifact without changing what the
    downstream consumer sees. Anchor slots stay positional because their
    identity is sample-specific and used by post-injection.
    """
    from scipy.optimize import linear_sum_assignment

    B, K, _ = pred.shape
    pred_n = F.normalize(pred.detach().float(), dim=-1, eps=1e-8)
    tgt_n = F.normalize(target.float(), dim=-1, eps=1e-8)
    tgt_perm = target.clone()
    rows_total = 0
    rows_moved = 0
    for b in range(B):
        content = mask_active[b] & ~anchor_mask[b]
        idx = content.nonzero(as_tuple=True)[0]
        M = int(idx.numel())
        if M < 2:
            continue
        cost = (1.0 - pred_n[b, idx] @ tgt_n[b, idx].T).cpu().numpy()
        row, col = linear_sum_assignment(cost)
        col_t = torch.as_tensor(col, device=target.device, dtype=torch.long)
        tgt_perm[b, idx] = target[b, idx[col_t]]
        rows_total += M
        rows_moved += int((col != row).sum())
    return tgt_perm, {
        "match_rows_total": float(rows_total),
        "match_rows_moved": float(rows_moved),
        "match_move_frac": float(rows_moved) / max(1.0, float(rows_total)),
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


def _pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool ``x`` (B, S, D) over active slots (``mask`` True).

    Zero-length rows (mask all False) return zero to avoid NaN; L2-normalize
    at the call site if needed. Denominator is the true active count, so
    length-normalized across variable-L samples.
    """
    m = mask.unsqueeze(-1).float()
    return (x.float() * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def _infonce_loss(
    pred_pool: torch.Tensor,      # (B, D)  resampler pooled output
    tgt_pooled: torch.Tensor,     # (B, V, D) per-variant pooled targets
    tau: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """SupCon-style multi-positive InfoNCE across in-batch variants.

    Every one of the B×V other-image variants is a negative; the V variants
    belonging to each anchor are positives (summed-then-averaged inside the
    log). Returns scalar loss + {"infonce_loss", "infonce_acc"} where
    ``infonce_acc`` is recall@1 on the anchor's correct image.
    """
    B, V, D = tgt_pooled.shape
    device = pred_pool.device
    pred_n = F.normalize(pred_pool.float(), dim=-1, eps=1e-8)          # (B, D)
    tgt_n = F.normalize(tgt_pooled.float(), dim=-1, eps=1e-8)          # (B, V, D)
    tgt_flat = tgt_n.reshape(B * V, D)                                  # (B*V, D)

    sim = (pred_n @ tgt_flat.T) / max(tau, 1e-4)                        # (B, B*V)
    # Build positive mask: row b has 1 at columns [b*V:(b+1)*V].
    pos_mask = (
        torch.eye(B, device=device).repeat_interleave(V, dim=1).bool()  # (B, B*V)
    )

    log_prob = sim - sim.logsumexp(dim=1, keepdim=True)
    per_anchor = -(log_prob * pos_mask.float()).sum(dim=1) / float(V)
    loss = per_anchor.mean()

    with torch.no_grad():
        # recall@1: top-1 column belongs to the correct image?
        top1 = sim.argmax(dim=1)
        correct = (top1 // V) == torch.arange(B, device=device)
        acc = correct.float().mean().item()
    return loss, {
        "infonce_loss": float(loss.detach().item()),
        "infonce_acc": float(acc),
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
