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
    extra: list[str] | None = None,
    extra_repeat: int = 1,
    extra_replace: bool = False,
) -> pd.DataFrame:
    """Manifest rows of one split; speech drawn at ``speech_ratio`` × the SFX count.

    ``extra`` = names of sibling manifests (``manifest_<name>.parquet`` — O3's
    colorized crops) whose rows of the same split are **appended** before the
    speech draw / ``limit``, tagged by their ``source`` column, each row
    ``extra_repeat`` times (oversampling a small colorized set).
    ``extra_replace`` instead **swaps** them in: every grey row whose polygon
    (``kind, book, page, id``) an extra row re-cuts is dropped first, so the
    colorized copy stands in for its original and the total stays put.
    """
    df = pd.read_parquet(m109.derived_root() / "manifest.parquet")
    if "source" not in df.columns:
        df["source"] = "grey"
    df = df[df.split == split]
    key = ["kind", "book", "page", "id"]
    for name in extra or []:
        ex = pd.read_parquet(m109.derived_root() / f"manifest_{name}.parquet")
        ex = ex[ex.split == split]
        if extra_replace:
            drop = set(map(tuple, ex[key].values))
            df = df[[t not in drop for t in map(tuple, df[key].values)]]
        df = pd.concat([df] + [ex] * max(1, extra_repeat), ignore_index=True)
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
    return out.sort_values(["source", "kind", "book", "page", "id"]).reset_index(
        drop=True
    )


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


def vl_tokens(
    w: np.ndarray, h: np.ndarray, min_pixels: int, max_pixels: int, factor: int = 28
) -> np.ndarray:
    """Exact patch-token count each crop will cost the VL tower, from its (w, h) alone.

    Mirrors ``transformers…paddleocr_vl.smart_resize``: the processor rounds each
    edge to a multiple of ``factor`` (= patch 14 × merge 2) and then rescales so the
    pixel budget lands inside ``[min_pixels, max_pixels]``. Tokens = pixels / 14².

    The **min_pixels floor is why crop area is a bad batching key** — every crop
    below it is upscaled to the same cost, so sorting by area does not sort by cost.
    """
    w = np.asarray(w, dtype=np.float64).copy()
    h = np.asarray(h, dtype=np.float64).copy()
    small_h = h < factor
    w[small_h] = np.round(w[small_h] * factor / h[small_h])
    h[small_h] = factor
    small_w = w < factor
    h[small_w] = np.round(h[small_w] * factor / w[small_w])
    w[small_w] = factor
    hb = np.round(h / factor) * factor
    wb = np.round(w / factor) * factor
    px = hb * wb
    over = px > max_pixels
    if over.any():
        beta = np.sqrt(h[over] * w[over] / max_pixels)
        hb[over] = np.maximum(factor, np.floor(h[over] / beta / factor) * factor)
        wb[over] = np.maximum(factor, np.floor(w[over] / beta / factor) * factor)
    under = px < min_pixels
    if under.any():
        beta = np.sqrt(min_pixels / (h[under] * w[under]))
        hb[under] = np.ceil(h[under] * beta / factor) * factor
        wb[under] = np.ceil(w[under] * beta / factor) * factor
    return ((hb * wb) / 196).astype(np.int64)


def token_batches(
    tokens: np.ndarray, budget: int, batch_size: int, rng: random.Random
) -> list[list[int]]:
    """Batches capped by **total packed tokens**, not crop count.

    ``area_batches`` fixes the crop count, so a batch drawn from the large-crop tail
    can cost several times the median and OOM the tower on a spike (measured on
    ``manifest_all``: median 20.1k tokens, max 122.6k, 34 batches over 60k). Sorting
    by token cost and cutting on a budget bounds the peak; because ~99 % of crops sit
    at the processor's min_pixels floor, a 24k budget costs only +0.6 % more steps.

    ``batch_size`` stays the upper bound on crops per batch, so a budget wider than
    ``batch_size`` × median simply reproduces the old behaviour.
    """
    key = tokens * np.exp(np.array([rng.gauss(0, 0.15) for _ in range(len(tokens))]))
    order = np.argsort(key)
    batches: list[list[int]] = []
    cur: list[int] = []
    run = 0
    for i in order:
        t = int(tokens[i])
        if cur and (run + t > budget or len(cur) >= batch_size):
            batches.append(cur)
            cur, run = [], 0
        cur.append(int(i))
        run += t
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


def collate_raw(items):
    imgs, targets, idx = zip(*items)
    return list(imgs), list(targets), list(idx)
