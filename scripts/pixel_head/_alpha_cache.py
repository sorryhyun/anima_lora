"""The alpha head's input cache: what the DiT and the head's loss take, no DiT output.

``<root>/manifest.json`` lists the rows.

    <set>/<row>.safetensors     latent (16, H/8, W/8) bf16, the VAE's, of the composite
    text/<digest>.safetensors   cross (n, 1024) bf16: the caption after llm_adapter, its
                                zero padding cut off; :func:`read_inputs` pads it back
    cutout/<key>.png            the alpha (L); RGBA for the images that are read

A composite and its background are redrawn from the palette and the seed
(:func:`compose`), so neither is stored. Sets: ``train`` (every cutout outside the read
images, ``draws`` times on each background kind), ``new_bg`` (R1's fit images on
backgrounds of their own, read) and ``other`` (images that never train, read).

Every latent is stamped with a digest of what made it (models, seed, palette, cutout,
caption). The builder re-encodes a file whose stamp differs, and drops the manifest
until it is done.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.pixel_head._alpha_data import BG_KINDS, Cutout, bg_caption, draw_bg

FORMAT = 2
MANIFEST = "manifest.json"
READ_SETS = ("new_bg", "other")
TEXT_TOKENS = 512  # Anima takes the TE max-padded


@dataclass
class Row:
    set: str
    key: str
    group: str  # the stress kind a read image was picked for
    kind: str  # background kind
    draw: int
    noise: int  # a read row's noise index; training draws its own
    caption: str
    size: tuple[int, int]  # (h, w), pixels
    text: str = ""  # its caption's file
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
        (c, "train", kind, d, 0)
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
    te = [recipe.get(k) for k in ("dit", "text_encoder", "vocab_pack")]
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
        row.text = _digest(FORMAT, te, row.caption)
        row.stamp = _digest(FORMAT, recipe, c.sha, asdict(row))
        out.append(row)
    return out


def compose(row: Row, cut: Cutout, palette: list, seed: int):
    """→ (composite, background, the row's rng after the background's draws)."""
    if row.set == "train":
        rng = cut.rng("train", seed, row.kind, row.draw)
    else:
        rng = cut.rng(row.kind, row.draw)
    bg = draw_bg(row.kind, palette, *cut.a.shape, rng)
    return cut.composite(bg), bg, rng


def is_current(root: Path, row: Row) -> bool:
    from safetensors import SafetensorError, safe_open

    path = root / row.rel
    if not (path.exists() and text_path(root, row).exists()):
        return False
    if not cutout_path(root, row.key).exists():
        return False
    try:
        with safe_open(str(path), "pt") as fh:
            return (fh.metadata() or {}).get("stamp") == row.stamp
    except SafetensorError:
        return False


def text_path(root: Path, row: Row) -> Path:
    return root / "text" / f"{row.text}.safetensors"


def cutout_path(root: Path, key: str) -> Path:
    return root / "cutout" / f"{key}.png"


def _save(tensors: dict, path: Path, metadata: dict | None = None) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    save_file(tensors, str(tmp), metadata=metadata)
    os.replace(tmp, path)


def write_cutout(root: Path, cut: Cutout, colour: bool) -> None:
    path = cutout_path(root, cut.key)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.dstack([cut.fg, cut.a]) if colour else cut.a
    tmp = path.with_suffix(".tmp")
    Image.fromarray(x).save(tmp, format="PNG", compress_level=3)
    os.replace(tmp, path)


def write_text(root: Path, row: Row, cross: torch.Tensor) -> None:
    """``cross`` (N, 1024), zero-padded: kept up to its last non-zero position."""
    used = cross.ne(0).any(-1).nonzero()
    n = int(used[-1]) + 1 if len(used) else 1
    _save({"cross": cross[:n].clone().contiguous()}, text_path(root, row))


def write_latent(root: Path, row: Row, latent: torch.Tensor) -> None:
    _save({"latent": latent.contiguous()}, root / row.rel, {"stamp": row.stamp})


def read_inputs(root: Path, row: Row) -> tuple[torch.Tensor, torch.Tensor]:
    """→ (latent (1, 16, h, w), crossattn embed (1, 512, 1024) max-padded), the bytes
    read here (a loader thread)."""
    from safetensors.torch import load

    latent = load((root / row.rel).read_bytes())["latent"][None]
    cross = load(text_path(root, row).read_bytes())["cross"]
    out = cross.new_zeros(1, TEXT_TOKENS, cross.shape[-1])
    out[0, : len(cross)] = cross
    return latent, out


def read_alpha(root: Path, key: str) -> np.ndarray:
    x = np.array(Image.open(cutout_path(root, key)))
    return x if x.ndim == 2 else x[..., 3]


def read_cutout(root: Path, row: Row) -> Cutout:
    """A read row's cutout."""
    x = np.array(Image.open(cutout_path(root, row.key)))
    fg, a = x[..., :3].copy(), x[..., 3].copy()
    return Cutout(row.key, fg, a, row.caption, "", {}, row.group)


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
