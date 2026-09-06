"""Shared training data for the O2 fine-tunes (``plan_ocr.md``): the O1 manifest as a
torch ``Dataset`` of (BGR crop, target string) with train-time augmentation.

* **Mix** — decision 2: COO ``sfx`` : Manga109 ``speech`` 1 : 1 *by count* (the
  manifest is already count-matched per book; ``--speech_ratio`` rescales the
  speech draw). Both kinds come from the same book split.
* **Target rule** (findings § O1) — NFKC-fold + strip all whitespace: Manga109's
  ``<text>`` keeps line breaks and full-width punctuation that manga-ocr's
  vocab lacks; the scorer's ``exact`` applies the same fold.
* **Augmentation** — ``augment.Augment`` on the train split only; per-worker
  seeding so DataLoader workers do not replay one RNG stream.
"""

from __future__ import annotations

import random
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manga109 as m109  # noqa: E402
from augment import Augment  # noqa: E402

MAX_TARGET_CHARS = 96


def normalize_target(s: str) -> str:
    return "".join(unicodedata.normalize("NFKC", s).split())


def load_split(
    split: str,
    *,
    speech_ratio: float = 1.0,
    limit: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Manifest rows of one split; speech drawn at ``speech_ratio`` × the SFX count."""
    df = pd.read_parquet(m109.derived_root() / "manifest.parquet")
    df = df[df.split == split]
    sfx = df[df.kind == "sfx"]
    sp = df[df.kind == "speech"]
    n_sp = min(len(sp), int(round(len(sfx) * speech_ratio)))
    sp = sp.sample(n_sp, random_state=seed) if n_sp < len(sp) else sp
    out = pd.concat([sfx, sp])
    if limit:
        out = pd.concat(
            [
                g.sample(min(limit, len(g)), random_state=seed)
                for _, g in out.groupby("kind")
            ]
        )
    out = out.copy()
    out["target"] = out.text.map(normalize_target).str.slice(0, MAX_TARGET_CHARS)
    out = out[out.target.str.len() > 0]
    return out.sort_values(["kind", "book", "page", "id"]).reset_index(drop=True)


class CropDataset(Dataset):
    def __init__(self, df: pd.DataFrame, *, augment: bool, seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.derived = m109.derived_root()
        self.paths = [str(self.derived / p) for p in self.df.path]
        self.targets = list(self.df.target)
        self.orients = list(self.df.orient)
        self.area = (self.df.w * self.df.h).to_numpy()
        self.augment = augment
        self.seed = seed
        self._aug: Augment | None = None

    def __len__(self):
        return len(self.df)

    def _get_aug(self) -> Augment:
        if self._aug is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info else 0
            self._aug = Augment(seed=self.seed * 1000 + wid)
        return self._aug

    def __getitem__(self, i: int):
        img = cv2.imread(self.paths[i])
        if img is None:
            raise FileNotFoundError(self.paths[i])
        if self.augment:
            img = self._get_aug()(img)
        return img, self.targets[i], i


def area_batches(
    area: np.ndarray, batch_size: int, rng: random.Random
) -> list[list[int]]:
    """Batches of similar crop area (the VL batching rule); batch order shuffled,
    membership jittered by a random tie-break so epochs differ."""
    key = area * np.exp(np.array([rng.gauss(0, 0.15) for _ in range(len(area))]))
    order = np.argsort(key)
    batches = [
        order[s : s + batch_size].tolist() for s in range(0, len(order), batch_size)
    ]
    rng.shuffle(batches)
    return batches


def collate_raw(items):
    imgs, targets, idx = zip(*items)
    return list(imgs), list(targets), list(idx)
