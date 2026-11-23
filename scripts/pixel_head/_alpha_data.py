"""The alpha head's data: RGBA cutouts and the light backgrounds they are put on.

Numpy on the CPU: images are (H, W, 3) and alphas (H, W), float32 in [0, 1]. A background
is an affine colour field ``coef`` (3 × 3: colour at the centre, change across x, change
across y) plus grain, both from the image set's own simple backgrounds
(:func:`fit_palette`). Plan: ``project/pixel_decoder/rgba_plan.md``.
"""

from __future__ import annotations

import hashlib
import math
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from library.anima.pixel_rgba import affine_grid, border_mask, fit_field
from library.anima.pixel_head import PATCH

BG_KINDS = ("white", "flat", "field")
BG_TAGS = ("simple background", "white background", "gradient background")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")
TARGET_TOKENS = 4100
LUMA = np.array([0.299, 0.587, 0.114], np.float32)


# ---------------------------------------------------------------- backgrounds


def _border_fit(txt: Path):
    cap = txt.read_text(encoding="utf-8", errors="ignore").lower()
    if not any(t in cap for t in BG_TAGS):
        return None
    src = next(
        (p for e in IMAGE_EXT if (p := txt.with_suffix(e)).exists()),
        None,
    )
    if src is None:
        return None
    im = Image.open(src).convert("RGBA")
    s = 512 / max(im.size)
    im = im.resize(
        (max(64, round(im.width * s)), max(64, round(im.height * s))), Image.BOX
    )
    x = np.asarray(im, np.float32) / 255
    if (x[..., 3] < 0.98).mean() > 0.01:
        return None
    return fit_field(x[..., :3], border_mask(*x.shape[:2]))


def fit_palette(root: Path) -> dict:
    """The light backgrounds of ``root``'s simple-background images.

    An image counts when one affine field holds at least 55 % of its border; it is
    light when the field's centre is not white (a channel under 250) and its luma ≥ 180.
    """
    txts = sorted(root.rglob("*.txt"))
    with ThreadPoolExecutor(6) as ex:
        fits = [f for f in ex.map(_border_fit, txts) if f and f[1] >= 0.55]
    white, light = 0, []
    for coef, _, res in fits:
        c = coef[0]
        if c.min() >= 250 / 255:
            white += 1
        elif c @ LUMA >= 180 / 255:
            light.append({"coef": coef.tolist(), "res": res})
    return {"found": len(fits), "white": white, "light": light}


def draw_bg(kind: str, palette: list, h: int, w: int, rng: np.random.Generator):
    """→ the background (H, W, 3).

    ``flat``: a light colour of the palette (half of the draws from the tinted ones),
    ±6 levels of jitter. ``field``: a palette entry's gradient (a sloped one when the
    draw allows), flipped at random, with its measured grain.
    """
    if kind == "white":
        return np.ones((h, w, 3), np.float32)
    coefs = np.array([p["coef"] for p in palette], np.float32)
    if kind == "flat":
        tinted = np.ptp(coefs[:, 0], axis=1) >= 8 / 255
        pool = np.flatnonzero(tinted if rng.random() < 0.5 else ~tinted)
        c = coefs[rng.choice(pool), 0] + rng.uniform(-6, 6, 3) / 255
        return np.broadcast_to(c.clip(0, 1).astype(np.float32), (h, w, 3)).copy()
    assert kind == "field", kind
    slope = np.abs(coefs[:, 1:]).sum(1).max(1)
    i = int(rng.choice(np.flatnonzero(slope >= 4 / 255)))
    k = coefs[i].copy()
    k[1:] *= rng.choice([-1.0, 1.0], size=(2, 1))
    grain = rng.normal(0, palette[i]["res"], (h, w, 1)).astype(np.float32)
    return (affine_grid(h, w) @ k + grain).clip(0, 1).astype(np.float32)


# -------------------------------------------------------------------- cutouts


def fit_size(w: int, h: int) -> tuple[int, int]:
    """About ``TARGET_TOKENS`` DiT tokens at native aspect."""
    s = math.sqrt(TARGET_TOKENS * PATCH * PATCH / (w * h))
    return tuple(max(PATCH, round(v * s / PATCH) * PATCH) for v in (w, h))


@dataclass
class Cutout:
    key: str  # <artist>_<stem>
    fg: np.ndarray  # straight colour, uint8
    a: np.ndarray  # uint8
    caption: str
    sha: str
    score: dict
    group: str = ""  # the stress kind it was picked for

    @property
    def alpha(self) -> np.ndarray:
        return self.a.astype(np.float32) / 255

    @property
    def premul(self) -> np.ndarray:
        return self.fg.astype(np.float32) / 255 * self.alpha[..., None]

    def composite(self, bg: np.ndarray) -> np.ndarray:
        x = self.premul + (1 - self.alpha[..., None]) * bg
        return np.round(x.clip(0, 1) * 255).astype(np.float32) / 255

    def rng(self, *salt) -> np.random.Generator:
        return np.random.default_rng(
            zlib.crc32("|".join(map(str, (self.key, *salt))).encode())
        )


def outer_bg(near: np.ndarray) -> np.ndarray:
    """The part of ``near`` that is connected to the image border."""
    _, lab = cv2.connectedComponents(near.astype(np.uint8), connectivity=4)
    edge = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    return near & np.isin(lab, np.setdiff1d(edge, [0]))


def base_regions(a: np.ndarray) -> dict[str, np.ndarray]:
    """The regions that depend on alpha alone."""
    bg, solid = a <= 0.02, a >= 0.98
    soft = ~bg & ~solid
    far = cv2.distanceTransform((~bg).astype(np.uint8), cv2.DIST_L2, 3) > 4
    fg = (a > 0.25).astype(np.uint8)
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(fg, cv2.MORPH_OPEN, el)
    return {
        "transition": soft,
        "translucent": soft & far,  # sheets and effects, not the antialiased edge
        "thin": (fg > 0) & (opened == 0),  # strokes narrower than 7 px
        "holes": bg & ~outer_bg(bg),
    }


def _load_cutout(png: Path) -> Cutout:
    im = Image.open(png).convert("RGBA")
    x = np.array(im.resize(fit_size(*im.size), Image.LANCZOS))  # premultiplied inside
    a = x[..., 3].astype(np.float32) / 255
    r = base_regions(a)
    score = {
        "thin": float(r["thin"].mean()),
        "translucent": float(r["translucent"].mean()),
        "white_fg": float(((a >= 0.98) & (x[..., :3].min(2) >= 240)).mean()),
        "holes": float(r["holes"].mean()),
    }
    return Cutout(
        key=f"{png.parent.name}_{png.stem}",
        fg=x[..., :3].copy(),
        a=x[..., 3].copy(),
        caption=png.with_suffix(".txt").read_text(encoding="utf-8").strip(),
        sha=hashlib.sha1(x.tobytes()).hexdigest(),
        score=score,
    )


def load_cutouts(root: Path, limit: int = 0) -> list[Cutout]:
    """``root``'s cutouts at decode size, reposts dropped; ``limit``: the first N files.

    ``root`` holds ``<artist>/<id>.png``, or, flat, ``<id>.png`` (key ``<root name>_<id>``).
    """
    files = sorted(root.glob("*/*.png")) or sorted(root.glob("*.png"))
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"{root}: no <artist>/<id>.png or <id>.png")
    with ThreadPoolExecutor(10) as ex:
        return dedup(list(ex.map(_load_cutout, files)))


def dedup(cuts: list[Cutout]) -> list[Cutout]:
    """Reposts and near-identical variants dropped: the first one stays."""
    out, small, seen = [], [], set()
    for c in cuts:  # reposts and near-identical variants: the first one stays
        if c.sha in seen:
            continue
        seen.add(c.sha)
        m = cv2.resize(c.a, (128, 128), interpolation=cv2.INTER_AREA) > 127
        if any((m & q).sum() > 0.99 * (m | q).sum() for q in small):
            continue
        small.append(m)
        out.append(c)
    return out


def thin_keep(key: str, artist: str, frac: float) -> bool:
    """A stable ``frac`` share of ``artist``'s keys (``<artist>_<stem>``)."""
    if not key.startswith(artist + "_"):
        return True
    return zlib.crc32(key.encode()) / 2**32 < frac


def pick_stress(cuts: list[Cutout], n: int) -> tuple[list[Cutout], list[Cutout]]:
    """``n`` to fit and ``n`` to only read, no split by artist: per stress kind the top
    images by that kind's area, taken alternately."""
    kinds = ("translucent", "thin", "white_fg", "holes")
    fit, read, used = [], [], set()
    for kind in kinds:
        ranked = sorted(cuts, key=lambda c: -c.score[kind])
        top = [c for c in ranked if c.key not in used][: 2 * n // len(kinds)]
        for i, c in enumerate(top):
            used.add(c.key)
            c.group = kind
            (read if i % 2 else fit).append(c)
    return fit, read


def bg_caption(caption: str, kind: str) -> str:
    from anime_tools.captions.position_clauses import compose_caption, parse_caption

    parsed = parse_caption(caption)
    drop = {"transparent background", "white background", "simple background"}
    tags = [t for t in parsed.flat_tags if t not in drop]
    add = ["white background"] if kind == "white" else []
    return compose_caption([*tags, "simple background", *add], parsed.clauses)
