"""The alpha head's feature cache: what one DiT forward on a composite leaves for training.

``<root>/manifest.json`` lists the rows, ``<root>/<set>/<name>.safetensors`` holds one:

    f       (gh, gw, D) bf16     the input of final_layer.linear
    lat     (gh, gw, 64) fp32    x_t's latent patches
    alpha   (H, W) uint8         the target
    fg, comp (H, W, 3) uint8     read sets only: the straight colour, the composite
    bg      (H, W, 3) fp16       read sets only: the background

Sets: ``train`` (every cutout outside the read images, ``draws`` times on each background
kind), ``new_bg`` (R1's fit images on backgrounds of their own, read) and ``other``
(images that never train, read).

Every file is stamped with a digest of what made it (models, σ range, seed, stream,
palette, cutout, caption). The builder re-encodes a file whose stamp differs, and drops
the manifest until it is done.
"""

from __future__ import annotations

import hashlib
import json
import os
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from library.anima.pixel_alpha_data import BG_KINDS, Cutout, bg_caption, draw_bg

FORMAT = 1
MANIFEST = "manifest.json"
READ_SETS = ("new_bg", "other")


@dataclass
class Row:
    set: str
    key: str
    group: str  # the stress kind a read image was picked for
    kind: str  # background kind
    draw: int
    noise: int  # DiTRollout's noise index
    caption: str
    size: tuple[int, int]  # (h, w), pixels
    stamp: str = ""

    @property
    def rel(self) -> str:
        return f"{self.set}/{self.key}__{self.kind}{self.draw}.safetensors"

    @property
    def tokens(self) -> int:
        return (self.size[0] // 16) * (self.size[1] // 16)


def _digest(*parts) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:20]


def file_id(path: str | Path | None) -> list | None:
    """A weight file as (name, size, mtime)."""
    if not path:
        return None
    st = Path(path).stat()
    return [Path(path).name, st.st_size, st.st_mtime_ns]


def plan_rows(
    cuts: list[Cutout],
    fit: list[Cutout],
    read: list[Cutout],
    draws: int,
    recipe: dict,
) -> list[Row]:
    """Every file of the cache, stamped for ``recipe``."""
    held = {c.key for c in read}
    specs = [
        (c, "train", kind, d, zlib.crc32(f"{c.key}|{kind}|{d}".encode()))
        for d in range(draws)
        for c in cuts
        if c.key not in held
        for kind in BG_KINDS
    ]
    # the read sets keep R1's backgrounds, σ and noise (rgbar1_1122_1515)
    for set_name, some, offset in (("new_bg", fit, 0), ("other", read, len(fit))):
        specs += [
            (c, set_name, kind, 1, 10000 + 3 * (offset + i) + j)
            for i, c in enumerate(some)
            for j, kind in enumerate(BG_KINDS)
        ]
    out = []
    for c, set_name, kind, draw, noise in specs:
        row = Row(
            set=set_name,
            key=c.key,
            group=c.group,
            kind=kind,
            draw=draw,
            noise=noise,
            caption=bg_caption(c.caption, kind),
            size=c.a.shape,
        )
        row.stamp = _digest(FORMAT, recipe, c.sha, asdict(row))
        out.append(row)
    return out


def compose(row: Row, cut: Cutout, palette: list, recipe: dict):
    """→ (composite, background, σ) of one row."""
    if row.set == "train":
        rng = cut.rng("train", recipe["seed"], row.kind, row.draw)
    else:
        rng = cut.rng(row.kind, row.draw)
    bg = draw_bg(row.kind, palette, *cut.a.shape, rng)
    sigma = float(rng.uniform(*recipe["sigma"]))
    return cut.composite(bg), bg, sigma


def is_current(root: Path, row: Row) -> bool:
    from safetensors import SafetensorError, safe_open

    path = root / row.rel
    if not path.exists():
        return False
    try:
        with safe_open(str(path), "pt") as fh:
            return (fh.metadata() or {}).get("stamp") == row.stamp
    except SafetensorError:
        return False


def row_bytes(row: Row, d_feat: int = 2048) -> int:
    h, w = row.size
    n = row.tokens * (2 * d_feat + 4 * 64) + h * w
    return n + (12 * h * w if row.set in READ_SETS else 0)


def write_row(
    root: Path,
    row: Row,
    cut: Cutout,
    f: torch.Tensor,
    lat: torch.Tensor,
    sigma: float,
    comp: np.ndarray,
    bg: np.ndarray,
) -> None:
    from safetensors.torch import save_file

    tensors = {
        "f": f.contiguous(),
        "lat": lat.contiguous(),
        "alpha": torch.from_numpy(cut.a).contiguous(),
    }
    if row.set in READ_SETS:
        tensors |= {
            "fg": torch.from_numpy(cut.fg).contiguous(),
            "comp": torch.from_numpy(np.round(comp * 255).astype(np.uint8)),
            "bg": torch.from_numpy(bg.astype(np.float16)).contiguous(),
        }
    path = root / row.rel
    tmp = path.with_suffix(".tmp")
    save_file(tensors, str(tmp), metadata={"stamp": row.stamp, "sigma": repr(sigma)})
    os.replace(tmp, path)


def read_row(root: Path, row: Row) -> tuple[dict[str, torch.Tensor], float]:
    """→ (tensors, σ), the bytes read here (a loader thread), not at the first use."""
    from safetensors import safe_open
    from safetensors.torch import load

    path = root / row.rel
    with safe_open(str(path), "pt") as fh:
        meta = fh.metadata() or {}
    return load(path.read_bytes()), float(meta["sigma"])


def drop_manifest(root: Path) -> None:
    (root / MANIFEST).unlink(missing_ok=True)


def write_manifest(root: Path, recipe: dict, rows: list[Row]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    body = {"recipe": recipe, "rows": [asdict(r) for r in rows]}
    tmp = root / f"{MANIFEST}.tmp"
    tmp.write_text(json.dumps(body, indent=1))
    os.replace(tmp, root / MANIFEST)


def read_manifest(root: Path) -> tuple[dict, dict[str, list[Row]]]:
    """→ (recipe, rows by set)."""
    path = root / MANIFEST
    if not path.exists():
        raise FileNotFoundError(
            f"{path}: no finished cache; run scripts/pixel_head/cache_alpha.py first"
        )
    body = json.loads(path.read_text())
    sets: dict[str, list[Row]] = {"train": [], **{s: [] for s in READ_SETS}}
    for r in body["rows"]:
        row = Row(**{**r, "size": tuple(r["size"])})
        sets[row.set].append(row)
    return body["recipe"], sets
