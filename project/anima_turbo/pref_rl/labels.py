"""Label sets for the preference head (pref_rl/proposal.md § 3.3). Torch-free.

- ``gelcrawl``: the selected / retrieved split, grouped by artist.
- a session name: the renders of its answered train rounds, grouped by round.

Features of a set are cached by ``features.py`` under ``feat_dir(name)``, one
``<basis>.npz`` (``keys``, ``X`` float16) per basis.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import store
from store import Session

GELCRAWL = "gelcrawl"
IMAGE_EXTS = {".webp", ".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class Item:
    key: str
    path: Path
    label: int
    group: str


def gelcrawl_root() -> Path:
    return Path(os.environ.get("GELCRAWL_HOME", "~/gelcrawl")).expanduser()


def _images(d: Path) -> dict[str, Path]:
    return {
        f.stem: f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS and f.is_file()
    }


def gelcrawl_items() -> list[Item]:
    """Artists with both folders: selected = 1, retrieved and not selected = 0
    (the manifest of gelcrawl's ``classify.py``)."""
    root = gelcrawl_root()
    ret, sel = root / "retrieved", root / "selected"
    both = {d.name for d in ret.iterdir() if d.is_dir()} & {
        d.name for d in sel.iterdir() if d.is_dir()
    }
    out = []
    for artist in sorted(both):
        r, s = _images(ret / artist), _images(sel / artist)
        for stem in sorted(s):
            out.append(Item(f"{artist}/{stem}", r.get(stem, s[stem]), 1, artist))
        for stem in sorted(set(r) - set(s)):
            out.append(Item(f"{artist}/{stem}", r[stem], 0, artist))
    return out


def session_items(session: Session) -> list[Item]:
    out = []
    for row in session.picks():
        if row.get("kind", "train") != "train" or row.get("skipped"):
            continue
        rid, picks = row["rid"], set(row["picks"])
        for slot in range(len(row.get("seeds") or [])):
            path = session.image_path(rid, slot)
            if path.exists():
                out.append(Item(f"{rid}/{slot:02d}", path, int(slot in picks), rid))
    return out


def items_of(name: str) -> list[Item]:
    if name == GELCRAWL:
        return gelcrawl_items()
    return session_items(Session(store.SESSIONS_ROOT / name))


def feat_dir(name: str) -> Path:
    return store.SESSIONS_ROOT / ("_gelcrawl" if name == GELCRAWL else name) / "feat"


def bases_of(name: str) -> list[str]:
    d = feat_dir(name)
    return sorted(p.stem for p in d.glob("*.npz")) if d.is_dir() else []


def load_feats(name: str, basis: str) -> dict[str, np.ndarray]:
    path = feat_dir(name) / f"{basis}.npz"
    if not path.exists():
        return {}
    z = np.load(path)
    return dict(zip(z["keys"].tolist(), z["X"]))


def save_feats(name: str, basis: str, feats: dict[str, np.ndarray]) -> None:
    d = feat_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    keys = sorted(feats)
    tmp = d / f"{basis}.tmp.npz"
    np.savez(
        tmp,
        keys=np.array(keys),
        X=np.stack([feats[k] for k in keys]).astype(np.float16),
    )
    tmp.replace(d / f"{basis}.npz")


def aligned(
    items: list[Item], feats: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X float32, y, group) over the items that have a feature row."""
    have = [it for it in items if it.key in feats]
    X = np.stack([feats[it.key] for it in have]).astype(np.float32)
    y = np.array([it.label for it in have], dtype=np.int64)
    g = np.array([it.group for it in have])
    return X, y, g
