"""General cached-pair train dataset (VAE latents + text encoder outputs).

Loads pre-cached VAE latents + text encoder outputs from disk, grouped by
latent resolution so each batch has uniform spatial dims. Despite the name
it is not distill-specific — used by ``project/finished/mod_guidance/distill.py`` and
``scripts/distill_turbo/distill.py``.
"""

from __future__ import annotations

import glob
import logging
import os
import random

import numpy as np
import torch
from PIL import Image

from library.io.cache import (
    LATENT_CACHE_SUFFIX,
    TE_CACHE_SUFFIX,
    discover_cached_pairs,
    get_latent_resolution,
    load_cached_latents,
    load_cached_text_features,
)

logger = logging.getLogger(__name__)


def _cached_collate_impl(batch):
    """Stack a list of per-sample :class:`CachedDataset` dicts into a batch dict.

    Self-describing: optional keys (``pooled_text`` / ``mask``) are stacked only
    when present in the sample dict, so there's no collate-vs-dataset flag to
    keep in sync. ``idx`` stays a Python list; everything else is
    ``torch.stack``-ed (uniform spatial dims guaranteed by
    :class:`BucketBatchSampler`).
    """
    keys = batch[0].keys()
    out: dict = {
        "idx": [b["idx"] for b in batch],
        "latents": torch.stack([b["latents"] for b in batch]),
        "crossattn_emb": torch.stack([b["crossattn_emb"] for b in batch]),
    }
    if "pooled_text" in keys:
        out["pooled_text"] = torch.stack([b["pooled_text"] for b in batch])
    if "mask" in keys:
        out["mask"] = torch.stack([b["mask"] for b in batch])  # [B, 1, H, W]
    return out


def make_cached_collate():
    """Stacking collate for :class:`CachedDataset` batches (``idx`` /
    ``latents`` / ``crossattn_emb`` plus optional ``pooled_text`` / ``mask``).

    Bypasses the default ``collate_tensor_fn`` (its ``_new_shared_filename_cpu``
    makes non-resizable storage on some PyTorch / Python 3.13 builds). Returns
    a module-level function, not a closure, so DataLoader workers can pickle
    it under spawn.
    """
    return _cached_collate_impl


class BucketBatchSampler(torch.utils.data.Sampler):
    """Yields batches of sample indices grouped by resolution bucket.

    Every batch is same-resolution, so a plain tensor-stacking collate still
    works at ``batch_size > 1`` (mixed resolutions would crash the stack).
    When ``shuffle``, batch order is reshuffled per epoch, but the
    largest-token-count bucket's first batch is always pinned to step 0 so
    torch.compile's biggest graph + peak VRAM land up front (fail-fast instead
    of a mid-run OOM). ``shuffle=False`` keeps deterministic largest-first order.
    """

    def __init__(self, batches, warmup_idxs, *, shuffle=True, seed=0):
        self._batches = batches  # list[list[int]]
        # Batch indices pinned to the front, largest-token first. Accepts a
        # single int / None for back-compat.
        if warmup_idxs is None:
            warmup_idxs = []
        elif isinstance(warmup_idxs, int):
            warmup_idxs = [warmup_idxs]
        self._warmup_idxs = list(warmup_idxs)
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        order = list(range(len(self._batches)))
        if self._shuffle:
            random.Random(self._seed + self._epoch).shuffle(order)
            self._epoch += 1
            # Insert smallest-first at 0 so the largest family lands at step 0.
            for idx in reversed(self._warmup_idxs):
                order.remove(idx)
                order.insert(0, idx)
        for bi in order:
            yield self._batches[bi]


class CachedDataset(torch.utils.data.Dataset):
    """Loads pre-cached latents and text encoder outputs for distillation.

    Samples are grouped by latent resolution so that each batch has uniform
    spatial dimensions (matching the bucket-based batching used in training).
    A deterministic per-bucket split (seeded by ``validation_seed``) carves off
    the last ``validation_split`` fraction for the val set, mirroring the
    LoRA training convention.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1,
        *,
        split: str = "train",
        validation_split: float = 0.0,
        validation_seed: int = 42,
        sample_ratio: float = 1.0,
        synth_data_dir: str | None = None,
        mask_dir: str | None = None,
        keep_list: set[str] | None = None,
        need_pooled: bool = True,
    ):
        assert split in ("train", "val")
        self.data_dir = data_dir
        self.synth_data_dir = synth_data_dir
        # Optional keys are included in __getitem__'s dict only when requested,
        # so consumers select features at construction time.
        self.need_pooled = need_pooled  # emit `pooled_text` (mod-guidance needs it)
        self.mask_dir = mask_dir  # emit `mask`; see _resolve_mask_path
        cached = discover_cached_pairs(data_dir)

        # Drop every cached pair whose stem isn't in keep_list, BEFORE bucketing.
        if keep_list is not None:
            n_before = len(cached)
            cached = [img for img in cached if img.stem in keep_list]
            logger.info(
                f"[{split}] keep_list filter: {len(cached)}/{n_before} pairs "
                f"survive (keep_list has {len(keep_list)} stems)"
            )

        # --synth_data_dir: rewrite each sample's latent path to the synthetic
        # NPZ for the same stem (TE paths stay in data_dir); samples without a
        # synthetic counterpart are dropped. Lookup is stem-keyed (not
        # basename-keyed) because the lora cache uses WxH pixel dims (e.g.
        # `0896x1152`) while the synth writer uses HxW latent dims (e.g.
        # `144x112`) — same logical pair, different basenames.
        n_dropped_no_synth = 0
        if synth_data_dir is not None:
            synth_by_stem: dict[str, str] = {}
            for path in glob.glob(
                os.path.join(synth_data_dir, "**", f"*{LATENT_CACHE_SUFFIX}"),
                recursive=True,
            ):
                # `{stem}_{HxW}_anima.npz`: strip suffix, drop trailing `_HxW`
                without_suffix = os.path.basename(path).removesuffix(
                    LATENT_CACHE_SUFFIX
                )
                stem = without_suffix.rsplit("_", 1)[0]
                synth_by_stem.setdefault(stem, path)
            remapped: list = []
            for img in cached:
                if img.te_path is None:
                    continue
                synth_path = synth_by_stem.get(img.stem)
                if synth_path is None:
                    n_dropped_no_synth += 1
                    continue
                remapped.append(img._replace(npz_path=synth_path))
            cached = remapped
            if n_dropped_no_synth:
                logger.warning(
                    f"[{split}] {n_dropped_no_synth} samples have no synthetic "
                    f"latent under {synth_data_dir}; dropped."
                )

        buckets: dict[str, list[tuple[str, str]]] = {}
        for img in cached:
            if img.te_path is None:
                continue
            res = get_latent_resolution(img.npz_path)
            buckets.setdefault(res, []).append((img.npz_path, img.te_path))

        # Per-bucket deterministic shuffle, then carve last `validation_split`
        # off as val (train/val never overlap, stay bucket-grouped). sample_ratio
        # applies per-bucket, keeping >=1 sample per non-empty bucket.
        #
        # Emit buckets largest-token-count first: DataLoader runs shuffle=False,
        # so this order == iteration order, and front-loading the biggest
        # resolution turns a mid-run compile/VRAM spike into a fail-fast at start.
        def _tok_count(res: str) -> int:
            a, b = res.split("x")
            return int(a) * int(b)

        rng = random.Random(validation_seed)
        self.batch_size = batch_size
        self.samples: list[tuple[str, str]] = []
        # Same-resolution batches, built bucket-by-bucket (remainder dropped to
        # a multiple of batch_size, so a contiguous chunk is always one bucket).
        self._batches: list[list[int]] = []
        self._batch_tok: list[int] = []
        n_train = n_val = 0
        for _res, items in sorted(
            buckets.items(), key=lambda kv: _tok_count(kv[0]), reverse=True
        ):
            items = list(items)
            rng.shuffle(items)
            n = len(items)
            n_v = int(round(n * validation_split)) if validation_split > 0.0 else 0
            n_t = n - n_v
            train_items = items[:n_t]
            val_items = items[n_t:]
            n_train += n_t
            n_val += n_v
            picked = train_items if split == "train" else val_items
            if sample_ratio < 1.0 and picked:
                n_keep = max(1, int(round(len(picked) * sample_ratio)))
                picked = picked[:n_keep]
            full = (len(picked) // batch_size) * batch_size
            start = len(self.samples)
            self.samples.extend(picked[:full])
            tok = _tok_count(_res)
            for j in range(start, start + full, batch_size):
                self._batches.append(list(range(j, j + batch_size)))
                self._batch_tok.append(tok)

        sr_note = f", sample_ratio={sample_ratio}" if sample_ratio < 1.0 else ""
        source = (
            f"latents={synth_data_dir} (synth), te={data_dir}"
            if synth_data_dir is not None
            else data_dir
        )
        logger.info(
            f"[{split}] {len(self.samples)} samples from {source} "
            f"({len(buckets)} buckets; pre-drop train={n_train}, val={n_val}{sr_note})"
        )

        # Masked loss coverage check up front (mask_dir set but 0 masks found
        # would otherwise silently degrade to a no-op — every mask all-ones).
        if self.mask_dir is not None and self.samples:
            n_found = sum(
                1 for _, te in self.samples if self._resolve_mask_path(te) is not None
            )
            logger.info(
                f"[{split}] masked loss: {n_found}/{len(self.samples)} samples have a "
                f"{{stem}}_mask.png under {self.mask_dir} "
                f"(missing → all-ones mask, full loss)"
            )
            if n_found == 0:
                logger.warning(
                    f"[{split}] mask_dir={self.mask_dir} resolved 0 masks — masked "
                    "loss is a no-op. Did you run `make mask`?"
                )

    def __len__(self):
        return len(self.samples)

    def make_batch_sampler(
        self, *, shuffle: bool = True, seed: int = 0
    ) -> BucketBatchSampler:
        """Build a bucket-grouped batch sampler over this dataset.

        Pass to ``DataLoader(batch_sampler=...)`` (not ``batch_size=``). Each
        batch is one resolution; pins one batch per token-count family to the
        front (largest first) so all torch.compile graphs warm up early. See
        ``BucketBatchSampler``.
        """
        first_by_tok: dict[int, int] = {}
        for i in range(len(self._batches)):
            first_by_tok.setdefault(self._batch_tok[i], i)
        warmup = [first_by_tok[t] for t in sorted(first_by_tok, reverse=True)]
        return BucketBatchSampler(self._batches, warmup, shuffle=shuffle, seed=seed)

    def __getitem__(self, idx):
        latent_path, te_path = self.samples[idx]
        latents, _res, _h, _w = load_cached_latents(latent_path)  # (16, H, W)
        # Fixed variant=0: the teacher cache keys on (sample_idx, sigma_idx) only,
        # so a random variant per visit would return a teacher pred computed
        # under a different caption than the student is conditioned on.
        crossattn_emb, pooled_text = load_cached_text_features(te_path, variant=0)
        out: dict = {
            "idx": idx,
            "latents": latents,
            "crossattn_emb": crossattn_emb,
        }
        if self.need_pooled:
            out["pooled_text"] = pooled_text
        if self.mask_dir is not None:
            out["mask"] = self._load_mask(te_path, latents.shape[-2], latents.shape[-1])
        return out

    def _resolve_mask_path(self, te_path: str) -> str | None:
        """Map a TE cache path to its ``{stem}_mask.png``, or None if absent.
        Prefers the nested path, falls back to flat — matching
        ``dreambooth.py``'s mask resolution.
        """
        stem = os.path.basename(te_path).removesuffix(TE_CACHE_SUFFIX)
        rel = os.path.relpath(os.path.dirname(te_path), self.data_dir)
        candidates: list[str] = []
        if rel and rel != "." and not rel.startswith(".."):
            candidates.append(os.path.join(self.mask_dir, rel, f"{stem}_mask.png"))
        candidates.append(os.path.join(self.mask_dir, f"{stem}_mask.png"))
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _load_mask(self, te_path: str, h: int, w: int) -> torch.Tensor:
        """Foreground mask at latent resolution ``(h, w)``, ``[1, h, w]`` in
        ``[0, 1]``. Falls back to all-ones (full loss) when no mask exists."""
        path = self._resolve_mask_path(te_path)
        if path is None:
            return torch.ones(1, h, w, dtype=torch.float32)
        img = Image.open(path).convert("L")
        m = torch.from_numpy(np.asarray(img, dtype=np.float32))[None, None] / 255.0
        m = torch.nn.functional.interpolate(m, size=(h, w), mode="area")
        return m[0]  # [1, h, w]
