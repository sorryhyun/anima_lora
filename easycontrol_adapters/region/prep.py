#!/usr/bin/env python3
"""Select + mask + paint the condition tree for the *region* EasyControl task
("regenerate what is under the paint, coherently with everything around it").

Cond = a solid-color paint blob composited over the real image (or a white
canvas, ``--cond_background white`` — the v1/v2 era); target = the original
image (full scene, full caption). Image mode is the character-inpaint framing:
the blob occludes a region, so the model learns "fill the paint, keep the
surrounding scene" — no LaMa-style background recovery is ever needed, because
neither training nor inference sees the pixels behind the paint.

Three data **slices** share one paint semantics (paint = region to regenerate;
everything outside is context):

* **solo** (``count == ['1girl']``, ``solo``) — the paint is the character's
  own SAM3 mask, augmented exact→rough (levels 0-3) so hand-painted blobs work
  at inference, **plus level 4 "slack"**: the paint is a much larger box /
  ellipse that contains the character off-center, and the real background is
  the target under the slack part — this is what teaches the model that the
  paint is an *allowed region*, not a silhouette to fill edge-to-edge, and how
  to continue/harmonize the background inside the paint (the v3 adapter,
  trained on tight supersets only, filled every paint with character; bench
  ``area_ratio ≈ iou``). Level 5 "face" paints only the head (SAM3 ``head`` — face + hair, what a
  user actually paints over; the bare ``face`` concept is inconsistent about
  the fringe), the body stays visible — the inpaint-subset "concentrate" case.
* **pair** (``count == ['1boy','1girl']``) — *one-figure paint*: the girl's
  instance mask (SAM3 ``girl`` minus ``boy``) is painted, the boy stays
  visible as real context, so the generated girl must match a visible
  partner's lighting / scale / eye-line. Gated on the partner staying mostly
  visible (``--min_partner_visible``).
* Captions get two variant families per image (``stage_captions``): the
  **flat** rows (the resized-tree ``{stem}.variants.txt`` verbatim) and
  **positioned** copies carrying a trailing clause ``On the <pos>, <girl
  character name>`` where ``<pos>`` is the girl's reading-order position
  (``assign_positions`` over the girl + partner boxes), so the paint position
  and the caption's position vocabulary reinforce each other. Encoded into
  ``{base}/text`` (``text_cache_dir``) by ``stage_text``; captions that already
  carry hand/v2 clauses are kept as-is.

Every level keeps the paint a strict superset of the painted subject's mask:
on a real-background cond any character pixel left outside the paint (hair
wisps, feet) would leak pose/identity the model could cheat off — a channel
that is empty at inference time.

Idempotent stages (run in order; the cond stage prunes gate-failed symlinks
that a select-only re-run would resurrect without cond latents):

1. **Select** — read the caption index, keep solo-1girl (+ 1girl1boy with
   ``--pairs``) images (``multiple views`` absent — a view sheet would union
   several figures into one mask), mirror them as a relative-symlink target
   tree under ``{base}/staging`` and per-slice scoping trees under
   ``{base}/select/{solo,pair}``; record slice + character names in
   ``{base}/select.json``. Stale symlinks are pruned.
2. **Sam** (GPU) — SAM3 concept segmentation via
   ``anime_tools.masking.cli.generate_masks`` (low threshold: 0.7 misses ~45%
   of solo images on nude/close-up/stylized art; 0.4 recovers them with no
   measured over-inclusion, probed 2026-08-23): ``{base}/masks`` = girl
   (pairs: girl minus boy), ``{base}/masks_boy`` = boy/man/male and
   ``{base}/masks_person`` = person (pairs only; the partner is boy ∪
   (person − girl) — SAM3's male prompts alone miss most anime partners),
   ``{base}/masks_head`` = head, face + hair (all, ``--no_face`` to skip). Existing masks
   are skipped (``--sam_force`` regenerates); "focus not found" images get no
   mask file and drop out at the cond stage.
3. **Cond** — gate each raw mask on coverage / fragmentation / hole-iness
   (a dirty mask drops the image: its staging symlink is unlinked so it
   leaves the dataset), draw one augment level per image (seeded
   ``crc32(stem)`` so re-runs are bit-identical) and paint the condition
   image into ``{base}/cond_images``. Per-image records (slice, level,
   boxes, position word, coverages) + drop reasons land in
   ``{base}/report.json``.
4. **Captions** — write ``{stem}.txt`` + ``{stem}.variants.txt`` next to each
   staged symlink (flat + positioned rows, see above).
5. **Encode** (GPU) — VAE-encode the cond images at native size into
   ``{base}/cond`` so cond latent shape == target latent shape.
6. **Text** (GPU) — TE-encode the staged caption sidecars into ``{base}/text``.

Target latents and PE come from the shared LoRA cache (``cache_dir =
post_image_dataset/lora`` in the blueprint) — nothing is re-encoded for
targets. Run from the repo root::

    python easycontrol_adapters/region/prep.py --skip_encode --skip_text   # staging (GPU: SAM)
    python easycontrol_adapters/region/prep.py \
        --skip_select --skip_sam --skip_cond --skip_captions               # preprocess (GPU: VAE+TE)

Via the task runner (knobs from ``configs/easycontrol/region.toml``)::

    make easycontrol-staging    EASYADAPTER=region   # stages 1-4
    make easycontrol-preprocess EASYADAPTER=region   # stages 5-6

Contact sheets of the masks / paints / captions: ``contact_sheet.py`` next to
this file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]

LEVEL_NAMES = ("exact", "smooth", "blob", "wobble", "slack", "face")
SLICE_SOLO = "solo"
SLICE_PAIR = "pair"
BOY_PROMPTS = ("boy", "man", "male")
PERSON_PROMPTS = ("person", "human")  # partner fallback: person minus girl


def _stable_seed(name: str) -> int:
    """Process-stable per-stem seed (Python's hash() is salted; crc32 isn't)."""
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


def _save_png_atomic(arr: np.ndarray, out: Path) -> None:
    """Temp file in the same dir + os.replace so the final name is never truncated."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, suffix=".tmp.png")
    os.close(fd)
    try:
        Image.fromarray(arr).save(tmp)
        os.replace(tmp, out)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _mirror_symlinks(keys: list[str], resized_dir: Path, tree: Path) -> tuple[int, int]:
    """Relative symlinks ``tree/{key}.png -> resized/{key}.png``; prune the rest."""
    kept_set = set(keys)
    written = 0
    for key in keys:
        dst = tree / f"{key}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        target = os.path.relpath(resized_dir / f"{key}.png", dst.parent)
        if dst.is_symlink():
            if os.readlink(dst) == target:
                continue
            dst.unlink()
        dst.symlink_to(target)
        written += 1
    pruned = 0
    if tree.is_dir():
        for p in tree.rglob("*.png"):
            if p.relative_to(tree).with_suffix("").as_posix() not in kept_set:
                p.unlink()
                pruned += 1
    return written, pruned


# ── Stage 1: select ──────────────────────────────────────────────────────────


def stage_select(
    index_path: Path,
    resized_dir: Path,
    caption_src: Path,
    base: Path,
    *,
    pairs: bool,
    limit: int | None,
) -> dict[str, dict]:
    from anime_tools.captions.position_clauses import parse_caption

    if not index_path.is_file():
        raise SystemExit(f"{index_path} not found — run `make caption-index` first.")
    image_meta = json.loads(index_path.read_text(encoding="utf-8"))["image_meta"]

    select: dict[str, dict] = {}
    for key, meta in sorted(image_meta.items()):
        count = meta.get("count")
        if count == ["1girl"]:
            slice_ = SLICE_SOLO
        elif pairs and count == ["1boy", "1girl"]:
            slice_ = SLICE_PAIR
        else:
            continue
        if not (resized_dir / f"{key}.png").exists():
            continue
        # Prefer the resized caption (what training's TE cache encodes), fall
        # back to the caption master for a not-yet-mirrored image.
        cap_path = resized_dir / f"{key}.txt"
        if not cap_path.is_file():
            cap_path = caption_src / meta["path"]
            if not cap_path.is_file():
                continue
        parsed = parse_caption(cap_path.read_text(encoding="utf-8"))
        tags = parsed.tag_keys
        if "multiple views" in tags:
            continue
        if slice_ == SLICE_SOLO and "solo" not in tags:
            continue
        select[key] = {
            "slice": slice_,
            "character": list(meta.get("character") or []),
            "has_clauses": parsed.has_clauses,
        }
    if limit:
        # Keep the slice mix representative under a QA cap.
        by_slice: dict[str, list[str]] = {}
        for k, v in select.items():
            by_slice.setdefault(v["slice"], []).append(k)
        keep: set[str] = set()
        for ks in by_slice.values():
            keep.update(ks[: max(1, limit // max(1, len(by_slice)))])
        select = {k: v for k, v in select.items() if k in keep}

    staging = base / "staging"
    written, pruned = _mirror_symlinks(list(select), resized_dir, staging)
    for slice_ in (SLICE_SOLO, SLICE_PAIR):
        keys = [k for k, v in select.items() if v["slice"] == slice_]
        _mirror_symlinks(keys, resized_dir, base / "select" / slice_)
    (base / "select.json").write_text(json.dumps(select, indent=1), encoding="utf-8")
    n_pair = sum(1 for v in select.values() if v["slice"] == SLICE_PAIR)
    print(
        f"Select: {len(select)} targets ({len(select) - n_pair} solo, {n_pair} pair; "
        f"{written} linked, {pruned} pruned)"
    )
    return select


def load_select(base: Path) -> dict[str, dict]:
    p = base / "select.json"
    if not p.is_file():
        raise SystemExit(f"{p} not found — run the select stage first.")
    return json.loads(p.read_text(encoding="utf-8"))


# ── Stage 2: SAM masks ───────────────────────────────────────────────────────


def _run_sam(
    image_dir: Path,
    masks_dir: Path,
    *,
    focus: list[str],
    ignore: list[str],
    threshold: float,
    sam_dilate: int,
    batch_size: int,
    checkpoint: str,
    force: bool,
    tag: str,
) -> None:
    if not image_dir.is_dir() or not any(image_dir.rglob("*.png")):
        print(f"SAM[{tag}]: nothing under {image_dir}, skipped")
        return
    cfg = {
        "prompts": list(ignore),
        "focus_prompts": list(focus),
        "threshold": threshold,
        "dilate": sam_dilate,
        "path_pattern": "*",
    }
    cfg_path = masks_dir.parent / f"sam_region_{tag}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")  # JSON is valid YAML
    cmd = [
        sys.executable,
        "-m",
        "anime_tools.masking.cli.generate_masks",
        "--config",
        str(cfg_path),
        "--image-dir",
        str(image_dir),
        "--mask-dir",
        str(masks_dir),
        "--checkpoint",
        checkpoint,
        "--batch-size",
        str(batch_size),
        "--recursive",
    ]
    if force:
        cmd.append("--force")
    print(f"SAM[{tag}]: focus={focus} ignore={ignore} → {masks_dir}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def stage_sam(base: Path, *, face: bool, **kw) -> None:
    """Girl masks (pairs: girl minus boy), boy masks (pairs), face masks (all).

    The pair pass runs first into the shared ``masks/`` so the later solo pass
    (non-force) never overwrites a girl-minus-boy mask with a plain girl one;
    each pass is scoped by its ``select/<slice>`` symlink tree."""
    sel = base / "select"
    _run_sam(
        sel / SLICE_PAIR,
        base / "masks",
        focus=["girl"],
        ignore=list(BOY_PROMPTS),
        tag="pair_girl",
        **kw,
    )
    # A bare "boy" prompt finds ~2% of anime partners (18/860, 2026-08-25);
    # the union with "man"/"male" is what actually recovers them.
    _run_sam(
        sel / SLICE_PAIR,
        base / "masks_boy",
        focus=list(BOY_PROMPTS),
        ignore=[],
        tag="pair_boy",
        **kw,
    )
    _run_sam(
        sel / SLICE_SOLO,
        base / "masks",
        focus=["girl"],
        ignore=[],
        tag="solo_girl",
        **kw,
    )
    _run_sam(
        sel / SLICE_PAIR,
        base / "masks_person",
        focus=list(PERSON_PROMPTS),
        ignore=[],
        tag="pair_person",
        **kw,
    )
    if face:
        _run_sam(
            base / "staging",
            base / "masks_head",
            focus=["head"],
            ignore=[],
            tag="head",
            **kw,
        )


# ── Stage 3: gate + augment + paint ──────────────────────────────────────────


def _mask_gates(
    binary: np.ndarray,
    *,
    min_coverage: float,
    max_coverage: float,
    min_dominance: float,
    max_hole_frac: float,
    min_comp_frac: float,
    max_raggedness: float,
) -> tuple[np.ndarray | None, str | None]:
    """Return ``(speck-cleaned + pinhole-closed mask, None)`` or ``(None, drop_reason)``."""
    h, w = binary.shape
    area = h * w
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comps = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, n)]
    if not comps:
        return None, "empty"
    kept = [(a, i) for a, i in comps if a >= min_comp_frac * area]
    if not kept:
        return None, "specks_only"
    total = sum(a for a, _ in kept)
    cov = total / area
    if cov < min_coverage:
        return None, "too_small"
    if cov > max_coverage:
        return None, "too_large"
    if max(a for a, _ in kept) / total < min_dominance:
        return None, "fragmented"
    clean = np.isin(labels, [i for _, i in kept]).astype(np.uint8)
    # Raggedness: contour length / sqrt(area). A circle scores ~3.5, a clean
    # figure silhouette (hair spikes included) ~10-30; a salt-noise mask whose
    # speckle background connects outward — invisible to the hole gate — scores
    # in the hundreds.
    contours, _ = cv2.findContours(clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, closed=True) for c in contours)
    if perimeter / max(1.0, float(total) ** 0.5) > max_raggedness:
        return None, "ragged"
    # Hole fraction: background flood from a padded ring; unreached non-mask
    # pixels are enclosed holes (the ragged-tights signature).
    padded = np.pad((1 - clean).astype(np.uint8), 1, constant_values=1)
    ffmask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, ffmask, (0, 0), 2)
    holes = int((padded[1:-1, 1:-1] == 1).sum())
    if holes / (total + holes) > max_hole_frac:
        return None, "holey"
    # Close pinholes so the exact/smooth augment levels don't inherit
    # salt-noise texture from an otherwise-clean mask.
    k = _odd(int(max(h, w) * 0.008))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return clean, None


def _blur(m: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur; large sigmas go through a downscale round-trip for speed."""
    if sigma <= 8:
        return cv2.GaussianBlur(m, (0, 0), sigma)
    s = max(2, int(sigma / 8))
    small = cv2.resize(
        m,
        (max(1, m.shape[1] // s), max(1, m.shape[0] // s)),
        interpolation=cv2.INTER_AREA,
    )
    small = cv2.GaussianBlur(small, (0, 0), sigma / s)
    return cv2.resize(small, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_LINEAR)


def _odd(k: int) -> int:
    return max(3, k | 1)


def _elastic_warp(
    m: np.ndarray, rng: random.Random, edge: int, amplitude_frac: float
) -> np.ndarray:
    """Smooth random displacement field — hand-painted wobble.

    Bends the contour without bridging concavities, so limb structure (spread
    legs, raised arms) survives where a heavy blur would absorb it."""
    h, w = m.shape
    grid = rng.randint(6, 10)
    amp = edge * amplitude_frac
    field = [
        cv2.resize(
            np.array(
                [[rng.uniform(-1, 1) for _ in range(grid)] for _ in range(grid)],
                dtype=np.float32,
            ),
            (w, h),
            interpolation=cv2.INTER_CUBIC,
        )
        * amp
        for _ in range(2)
    ]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(
        m,
        xs + field[0],
        ys + field[1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _slack_region(
    mask: np.ndarray,
    rng: random.Random,
    edge: int,
    *,
    grow: tuple[float, float],
    max_coverage: float,
) -> np.ndarray | None:
    """A loose box / ellipse that *contains* the mask off-center.

    Targets a paint area of ``U(grow)`` × the character bbox (capped at
    ``max_coverage`` of the image) and distributes the growth over the four
    sides with random per-side weights, clamped to the image — so a
    full-height figure still gets slack sideways, and the character sits
    off-center in the paint (a centered halo would just teach "character in
    the middle of the paint"). The edge gets a light wobble so it reads like a
    hand-painted region rather than a rectangle. Returns ``None`` when there
    is no room to grow meaningfully (< 1.25× bbox): the paint is what a user
    draws when they mean "somewhere around here", and the model has to place
    the character inside it AND reproduce the background on the rest."""
    h, w = mask.shape
    x0, y0, x1, y1 = _bbox(mask)
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    target = min(max_coverage * h * w, bw * bh * rng.uniform(*grow))
    if target < 1.25 * bw * bh:
        return None
    room = (x0, y0, w - x1, h - y1)  # left, top, right, bottom
    wts = [rng.uniform(0.2, 1.0) for _ in range(4)]

    def box(scale: float) -> tuple[int, int, int, int]:
        g = [
            min(r, scale * wt * (bw if i % 2 == 0 else bh))
            for i, (r, wt) in enumerate(zip(room, wts))
        ]
        return (int(x0 - g[0]), int(y0 - g[1]), int(x1 + g[2]), int(y1 + g[3]))

    lo, hi = 0.0, 4.0
    for _ in range(24):  # binary-search the growth scale to hit the target area
        mid = (lo + hi) / 2
        bx0, by0, bx1, by1 = box(mid)
        if (bx1 - bx0) * (by1 - by0) < target:
            lo = mid
        else:
            hi = mid
    x0, y0, x1, y1 = box(hi)
    if (x1 - x0) * (y1 - y0) < 1.25 * bw * bh:
        return None
    region = np.zeros((h, w), np.float32)
    if rng.random() < 0.5:
        cv2.rectangle(region, (x0, y0), (x1 - 1, y1 - 1), 1.0, thickness=-1)
        region = _blur(region, edge * 0.02) >= 0.5  # rounded corners
    else:
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        # An ellipse inscribed in the box would clip the bbox corners; grow
        # the axes by sqrt(2) so the box (hence the mask) is contained.
        ax, ay = int((x1 - x0) * 0.5 * 1.4142) + 1, int((y1 - y0) * 0.5 * 1.4142) + 1
        cv2.ellipse(region, (cx, cy), (ax, ay), 0, 0, 360, 1.0, thickness=-1)
        region = region >= 0.5
    region = _elastic_warp(
        region.astype(np.float32), rng, edge, rng.uniform(0.005, 0.015)
    )
    return (region >= 0.5).astype(np.uint8)


def _one_sided_slack(
    mask: np.ndarray,
    m: np.ndarray,
    rng: random.Random,
    edge: int,
    *,
    max_ratio: float,
) -> np.ndarray | None:
    """Thin halo ∪ silhouette shifted one way, bounded to ``max_ratio`` × area."""
    h, w = mask.shape
    ang = rng.uniform(0, 2 * np.pi)
    for shift_frac, sigma_frac in ((0.12, 0.03), (0.08, 0.025), (0.05, 0.02)):
        dx, dy = (
            int(np.cos(ang) * edge * shift_frac),
            int(np.sin(ang) * edge * shift_frac),
        )
        shifted = cv2.warpAffine(
            m, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), borderValue=0
        )
        halo = _blur(np.maximum(m, shifted), edge * sigma_frac) >= 0.4
        halo = (
            _elastic_warp(halo.astype(np.float32), rng, edge, rng.uniform(0.01, 0.02))
            >= 0.5
        )
        if halo.sum() <= max_ratio * mask.sum():
            return halo if halo.sum() >= 1.2 * mask.sum() else None
    return None


def _augment_mask(
    mask: np.ndarray,
    rng: random.Random,
    level: int,
    *,
    slack_grow: tuple[float, float] = (1.4, 2.2),
    max_slack_coverage: float = 0.7,
    slack_max_ratio: float = 1.6,
) -> np.ndarray | None:
    """Exact→rough augmentation spectrum, plus slack.

    0 exact, 1 smoothed, 2 blob, 3 wobble — every one keeps the pose readable:
    the paint cue carries position AND rough silhouette, so bridging the gap
    between spread legs into one fat blob over-simplifies the cond (user
    feedback 2026-08-23). Blur sigmas stay small enough that only sub-limb
    concavities close. 4 slack — see :func:`_slack_region`. (Level 5 "face" is
    the smooth level applied to the face mask by the caller.) Returns ``None``
    only for a slack draw with no room to grow.

    The output is always a strict superset of the (slightly dilated) source
    mask: levels 1/3 can locally dip inside the true silhouette, and on a
    real-background cond any character pixel left outside the paint leaks
    pose/identity that vanishes at inference (nothing under the paint)."""
    h, w = mask.shape
    edge = max(h, w)
    m = mask.astype(np.float32)
    guard_k = _odd(int(edge * 0.006))
    guard = cv2.dilate(mask, np.ones((guard_k, guard_k), np.uint8))
    if level == 0:
        out = guard > 0
    elif level == 1:
        out = _blur(m, edge * rng.uniform(0.008, 0.02)) >= 0.5
    elif level == 2:
        # Rounder + slightly grown, but sigma/threshold bounded so wide limb
        # gaps survive (a sub-0.5 threshold grows the region as it smooths).
        out = _blur(m, edge * rng.uniform(0.015, 0.04)) >= rng.uniform(0.40, 0.50)
    elif level == 3:
        smoothed = _blur(m, edge * 0.012)
        out = _elastic_warp(smoothed, rng, edge, rng.uniform(0.01, 0.03)) >= 0.5
    elif level == 4:
        slack = _slack_region(
            mask, rng, edge, grow=slack_grow, max_coverage=max_slack_coverage
        )
        if slack is None:
            # Large figure (bbox already most of the frame): a thin halo plus
            # the silhouette shifted toward ONE random side — room on one side
            # only, so the girl sits off-centre and the paint stays ≤ ~1.6×
            # her area (an all-round fat halo took 0.3-coverage girls to 0.7
            # paints — user feedback 2026-08-25).
            out = _one_sided_slack(mask, m, rng, edge, max_ratio=slack_max_ratio)
            if out is None or out.mean() > max(max_slack_coverage, 0.85):
                return None  # no room — caller falls back to a tight level
        else:
            out = slack > 0
    else:
        raise ValueError(f"unknown augment level {level}")
    out = (out | (guard > 0)).astype(np.uint8)
    return out if out.any() else mask


def _load_mask(path: Path, size: tuple[int, int]) -> np.ndarray | None:
    if not path.exists():
        return None
    binary = (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)
    if binary.shape != (size[1], size[0]):
        binary = cv2.resize(binary, size, interpolation=cv2.INTER_NEAREST)
    return binary


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 2:
        return mask
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == best).astype(np.uint8)


def _position_word(girl_box, partner_box, size, *, side_frac: float = 0.4) -> str:
    """Caption position word for the girl.

    Pairs go through the grammar's own ``assign_positions`` (left/right by
    reading order). A lone box always reads ``center`` there (the vocabulary
    qualifies a lone subject only within a multi-row sheet), so solo images
    take the side her bbox centre falls on: ``left`` / ``right`` when it is
    outside the middle ``side_frac``..``1-side_frac`` band, else ``center``."""
    from anime_tools.captions.position_clauses import assign_positions, horizontal_names

    if partner_box is not None:
        return assign_positions([girl_box, partner_box], size)[0]
    cx = (girl_box[0] + girl_box[2]) / 2 / max(1, size[0])
    if cx < side_frac:
        return horizontal_names(2)[0]
    if cx > 1 - side_frac:
        return horizontal_names(2)[1]
    # The corpus says "On the middle" for a centred subject (never "center").
    return horizontal_names(3)[1]


def stage_cond(
    base: Path,
    *,
    gate_kwargs: dict,
    paint_color: tuple[int, int, int],
    cond_background: str,
    aug_weights: list[float],
    slack_grow: tuple[float, float],
    min_partner_visible: float,
    min_face_frac: float,
    pair_max_hole_frac: float,
    max_slack_coverage: float,
    overwrite: bool,
) -> None:
    from library.preprocess import tqdm_progress
    from library.preprocess._dataset import walk_images

    staging = base / "staging"
    masks_dir = base / "masks"
    cond_images = base / "cond_images"
    select = load_select(base)
    images = walk_images(staging, recursive=True)
    if not images:
        raise SystemExit(f"No staged images under {staging} — run the select stage.")
    if not any(masks_dir.rglob("*_mask.png")):
        raise SystemExit(f"No masks under {masks_dir} — run the SAM stage first.")
    if len(aug_weights) != len(LEVEL_NAMES):
        raise SystemExit(f"--aug_weights needs {len(LEVEL_NAMES)} values {LEVEL_NAMES}")

    report_path = base / "report.json"
    old_records: dict[str, dict] = {}
    if report_path.is_file() and not overwrite:
        try:
            old_records = {
                r["image"]: r for r in json.loads(report_path.read_text())["records"]
            }
        except (KeyError, ValueError):
            old_records = {}
    report: dict = {"kept": 0, "dropped": {}, "drops": [], "records": [], "levels": {}}

    def drop(img: Path, rel: Path, reason: str) -> None:
        img.unlink()  # leaves the dataset (staging tree = dataset scope)
        for tree in (
            cond_images,
            base / "select" / SLICE_SOLO,
            base / "select" / SLICE_PAIR,
        ):
            (tree / rel).with_suffix(".png").unlink(missing_ok=True)
        for suffix in (".txt", ".variants.txt"):
            (staging / rel.parent / f"{rel.stem}{suffix}").unlink(missing_ok=True)
        report["dropped"][reason] = report["dropped"].get(reason, 0) + 1
        report["drops"].append(
            {"image": rel.with_suffix("").as_posix(), "reason": reason}
        )

    levels = list(range(len(LEVEL_NAMES)))
    progress = tqdm_progress("Painting cond")
    progress(0, total=len(images))
    for p in images:
        rel = p.relative_to(staging)
        key = rel.with_suffix("").as_posix()
        meta = select.get(key)
        if meta is None:
            drop(p, rel, "unselected")
            progress(1, detail=f"drop {rel.stem}")
            continue
        slice_ = meta["slice"]
        mask_name = f"{rel.stem}_mask.png"
        mask_path = masks_dir / rel.parent / mask_name
        out = (cond_images / rel).with_suffix(".png")
        if not mask_path.exists():
            drop(p, rel, "no_mask")  # SAM "focus not found"
            progress(1, detail=f"drop {rel.stem}")
            continue
        if out.exists() and not overwrite and key in old_records:
            report["kept"] += 1
            rec = old_records[key]
            report["records"].append(rec)
            report["levels"][rec["level"]] = report["levels"].get(rec["level"], 0) + 1
            progress(1, detail=f"skip {rel.stem}")
            continue
        with Image.open(p) as im:
            iw, ih = im.size
            rgb = np.array(im.convert("RGB")) if cond_background == "image" else None
        binary = _load_mask(mask_path, (iw, ih))
        gk = dict(gate_kwargs)
        partner_raw = None
        if slice_ == SLICE_PAIR:
            # Subtract the partner here too (not only via SAM's ignore
            # prompt): the girl prompt alone often claims the boy as well.
            partner_raw = _load_mask(
                base / "masks_boy" / rel.parent / mask_name, (iw, ih)
            )
            # Fallback: SAM3's male prompts miss most anime partners; a
            # "person" pass minus the girl recovers the rest (only helps when
            # the girl prompt did not itself swallow the boy).
            person = _load_mask(
                base / "masks_person" / rel.parent / mask_name, (iw, ih)
            )
            if person is not None and person.any():
                extra = (person & (1 - binary)).astype(np.uint8)
                extra = _largest_component(extra) if extra.any() else extra
                if extra.sum() >= 0.01 * iw * ih:
                    partner_raw = (
                        extra if partner_raw is None else (partner_raw | extra)
                    )
            if partner_raw is None or not partner_raw.any():
                drop(p, rel, "no_partner_mask")
                progress(1, detail=f"drop {rel.stem} (no_partner_mask)")
                continue
            binary = (binary & (1 - partner_raw)).astype(np.uint8)
            # girl-minus-boy legitimately carves holes where the partner
            # overlaps her; the hole gate is a salt-noise detector, not a
            # composition rule, so loosen it here.
            gk["max_hole_frac"] = max(gk["max_hole_frac"], pair_max_hole_frac)
        clean, reason = _mask_gates(binary, **gk)
        if reason:
            drop(p, rel, reason)
            progress(1, detail=f"drop {rel.stem} ({reason})")
            continue

        partner = None
        if partner_raw is not None:
            partner = (partner_raw & (1 - clean)).astype(np.uint8)
            if not partner.any():
                drop(p, rel, "partner_inside_girl")
                progress(1, detail=f"drop {rel.stem} (partner_inside_girl)")
                continue

        rng = random.Random(_stable_seed(rel.stem))
        level = rng.choices(levels, weights=aug_weights)[0]
        face = None
        if level == 5:
            # "head" (face + hair) — what a user paints over; SAM3's "face"
            # concept is inconsistent about the fringe (2026-08-25 sheets).
            face = _load_mask(base / "masks_head" / rel.parent / mask_name, (iw, ih))
            if face is None:
                face = _load_mask(
                    base / "masks_face" / rel.parent / mask_name, (iw, ih)
                )
            if face is not None:
                # The girl's face only (a pair image's face pass also finds the
                # boy's): intersect with her slightly-dilated mask, keep the
                # largest blob.
                k = _odd(int(max(iw, ih) * 0.01))
                face = face & cv2.dilate(clean, np.ones((k, k), np.uint8))
                face = _largest_component(face.astype(np.uint8))
            if (
                face is None
                or face.sum() < min_face_frac * iw * ih
                or face.sum() > 0.7 * clean.sum()
            ):
                level = 4  # no usable face → slack (the other "regenerate" level)
                face = None
        if face is not None:
            blob = _augment_mask(face, rng, 1)
        else:
            blob = _augment_mask(
                clean,
                rng,
                level,
                slack_grow=slack_grow,
                max_slack_coverage=max_slack_coverage,
            )
            if blob is None:  # slack with no room → a tight level instead
                level = rng.choice((1, 2, 3))
                blob = _augment_mask(clean, rng, level)

        partner_visible = None
        if partner is not None:
            partner_visible = float(
                (partner & (1 - blob)).sum() / max(1, partner.sum())
            )
            if partner_visible < min_partner_visible:
                drop(p, rel, "partner_occluded")
                progress(1, detail=f"drop {rel.stem} (partner_occluded)")
                continue

        canvas = rgb if rgb is not None else np.full((ih, iw, 3), 255, np.uint8)
        canvas[blob > 0] = paint_color
        _save_png_atomic(canvas, out)

        girl_box = _bbox(clean)
        partner_box = _bbox(partner) if partner is not None else None
        rec = {
            "image": key,
            "slice": slice_,
            "level": LEVEL_NAMES[level],
            "position": _position_word(girl_box, partner_box, (iw, ih)),
            "girl_box": list(girl_box),
            "partner_box": list(partner_box) if partner_box else None,
            "girl_coverage": round(float(clean.mean()), 4),
            "paint_coverage": round(float(blob.mean()), 4),
            "girl_in_paint_area_ratio": round(
                float(clean.sum() / max(1, blob.sum())), 4
            ),
            "partner_visible": None
            if partner_visible is None
            else round(partner_visible, 4),
            "size": [iw, ih],
        }
        report["records"].append(rec)
        report["levels"][rec["level"]] = report["levels"].get(rec["level"], 0) + 1
        report["kept"] += 1
        progress(1, detail=f"{rel.stem} [{rec['level']}]")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(
        f"\nCond: {report['kept']} kept, "
        f"{sum(report['dropped'].values())} dropped {report['dropped']}; "
        f"levels {report['levels']} (details: {report_path})"
    )


# ── Stage 4: captions (flat + positioned variants) ───────────────────────────


def stage_captions(base: Path, resized_dir: Path, caption_src: Path) -> None:
    """Write ``{stem}.txt`` + ``{stem}.variants.txt`` next to each staged symlink.

    Rows = the resized tree's flat variant rows verbatim (``v0`` pristine first;
    a resized image without a sidecar contributes just its ``.txt``), then a
    positioned copy of each flat ``v*`` row with a trailing ``On the <pos>,
    <name>`` clause when the caption has no clauses yet and the girl has
    exactly one character name (an unnamed girl gets flat rows only — a clause
    needs a subject tag to bind). The position word comes from the cond
    stage's record (girl + partner boxes through the caption grammar's own
    ``assign_positions``), so the caption position and the paint agree.
    """
    from anime_tools.captions.position_clauses import (
        PositionClause,
        compose_caption,
        parse_caption,
    )
    from anime_tools.captions.variants import (
        read_variants_sidecar,
        variants_sidecar_path,
        write_variants_sidecar,
    )

    staging = base / "staging"
    select = load_select(base)
    report = json.loads((base / "report.json").read_text(encoding="utf-8"))
    records = {r["image"]: r for r in report["records"]}
    n_pos = n_flat_only = n_kept_clauses = 0
    for key, rec in sorted(records.items()):
        meta = select.get(key) or {}
        img = staging / f"{key}.png"
        if not img.exists():
            continue
        cap_path = resized_dir / f"{key}.txt"
        if not cap_path.is_file():
            cap_path = caption_src / f"{key}.txt"
        caption = cap_path.read_text(encoding="utf-8").strip().split("\n")[0]
        sidecar = variants_sidecar_path(resized_dir / f"{key}.png")
        rows = read_variants_sidecar(sidecar) if sidecar.is_file() else []
        if not rows or rows[0][0] != "v0":
            rows = [("v0", caption)] + [r for r in rows if r[0] != "v0"]
        parsed = parse_caption(rows[0][1])
        names = meta.get("character") or []
        if parsed.has_clauses:
            n_kept_clauses += 1
        elif len(names) == 1:
            clause = PositionClause(position=rec["position"], tags=(names[0],))
            flat_rows = [(lab, txt) for lab, txt in rows if lab.startswith("v")]
            n_v = len(flat_rows)
            rows = rows + [
                (
                    f"v{n_v + i}",
                    compose_caption(parse_caption(txt).flat_tags, (clause,)),
                )
                for i, (_, txt) in enumerate(flat_rows)
            ]
            n_pos += 1
        else:
            n_flat_only += 1
        (staging / f"{key}.txt").write_text(rows[0][1] + "\n", encoding="utf-8")
        write_variants_sidecar(variants_sidecar_path(img), rows)
    print(
        f"Captions: {n_pos} with positioned variants, {n_flat_only} flat-only "
        f"(no single character name), {n_kept_clauses} already clause-bearing"
    )


# ── Stage 5: encode cond ─────────────────────────────────────────────────────


def stage_encode(
    cond_images: Path,
    cond_cache_dir: Path,
    *,
    vae_path: str,
    batch_size: int,
    chunk_size: int,
    vae_2d: bool,
    overwrite: bool = False,
):
    """VAE-encode cond images at native size (cond latent shape == target's).

    ``overwrite`` re-encodes every cond — REQUIRED after a repaint: the cache is
    keyed ``{stem}_{WxH}`` and a same-stem latent from an earlier paint recipe
    is silently kept otherwise (2026-08-25: v4 solo latents survived the v5
    repaint until this flag existed).

    ``vae_2d`` default ON — the same fold the target latents in
    post_image_dataset/lora/ were cached with, so cond and target stay
    numerically consistent."""
    import torch

    from library.models import qwen_vae
    from library.preprocess import cache_latents, tqdm_progress

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading VAE from {vae_path} (2d_fold={vae_2d}) ...")
    vae = qwen_vae.load_vae(
        vae_path,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=chunk_size,
        disable_cache=True,
        vae_2d=vae_2d,
    )
    vae.to(device, dtype=torch.bfloat16)
    vae.requires_grad_(False)
    vae.eval()

    stats = cache_latents(
        cond_images,
        vae,
        cache_dir=cond_cache_dir,
        recursive=True,
        batch_size=batch_size,
        overwrite=overwrite,
        progress=tqdm_progress("Caching cond latents"),
    )

    vae.to("cpu")
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats


# ── Stage 6: encode text ─────────────────────────────────────────────────────


def stage_text(
    staging: Path,
    text_cache_dir: Path,
    *,
    qwen3_path: str,
    dit_path: str,
    t5_tokenizer_path: str | None,
    batch_size: int,
    overwrite: bool,
):
    """TE-encode the staged ``{stem}.variants.txt`` sidecars into ``text_cache_dir``.

    The sidecar is the source of truth (``cache_text_embeddings`` encodes its
    rows verbatim when present), so the flat + positioned rows written by
    :func:`stage_captions` land as ``v0..vN`` variants the loader samples from."""
    import torch

    from library.anima import weights as anima_utils
    from library.anima.strategy import AnimaTextEncodingStrategy, AnimaTokenizeStrategy
    from library.preprocess import cache_text_embeddings, tqdm_progress

    text_cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Qwen3 text encoder from {qwen3_path} ...")
    text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(
        qwen3_path, dtype=torch.bfloat16, device=str(device)
    )
    t5_tokenizer = anima_utils.load_t5_tokenizer(t5_tokenizer_path)
    print(f"Loading LLM adapter from {dit_path} ...")
    llm_adapter = anima_utils.load_llm_adapter(
        dit_path, dtype=torch.bfloat16, device=str(device)
    )
    tokenize_strategy = AnimaTokenizeStrategy(
        qwen3_tokenizer=qwen3_tokenizer, t5_tokenizer=t5_tokenizer
    )
    encoding_strategy = AnimaTextEncodingStrategy()

    from library.preprocess.uncond import (
        DEFAULT_UNCOND_DIR,
        stage_uncond_sidecar_with_models,
    )

    stage_uncond_sidecar_with_models(
        DEFAULT_UNCOND_DIR,
        text_encoder,
        tokenize_strategy,
        encoding_strategy,
        llm_adapter,
        device=device,
        overwrite=False,
    )
    stats = cache_text_embeddings(
        staging,
        tokenize_strategy,
        encoding_strategy,
        text_encoder,
        llm_adapter=llm_adapter,
        device=device,
        cache_dir=text_cache_dir,
        recursive=True,
        batch_size=batch_size,
        overwrite=overwrite,
        progress=tqdm_progress("Caching caption text"),
    )
    text_encoder.to("cpu")
    del text_encoder, llm_adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="post_image_dataset/easycontrol/region")
    parser.add_argument(
        "--caption_index", default="post_image_dataset/captions/caption_index.json"
    )
    parser.add_argument("--resized_dir", default="post_image_dataset/resized")
    parser.add_argument("--caption_src", default="image_dataset")
    parser.add_argument("--limit", type=int, default=None, help="cap #images (QA)")
    parser.add_argument(
        "--pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also stage 1girl1boy images (girl painted, boy visible)",
    )
    # ── SAM stage ──
    parser.add_argument(
        "--threshold", type=float, default=0.4, help="SAM concept threshold"
    )
    parser.add_argument("--sam_dilate", type=int, default=0)
    parser.add_argument("--sam_batch_size", type=int, default=4)
    parser.add_argument("--checkpoint", default="models/sam3/sam3.pt")
    parser.add_argument(
        "--sam_force", action="store_true", help="regenerate existing masks"
    )
    parser.add_argument(
        "--face",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the SAM face pass (needed for the face augment level)",
    )
    # ── cond stage: quality gates ──
    parser.add_argument("--min_coverage", type=float, default=0.03)
    parser.add_argument("--max_coverage", type=float, default=0.85)
    parser.add_argument(
        "--min_dominance",
        type=float,
        default=0.75,
        help="largest component / total mask area floor (fragmentation gate)",
    )
    parser.add_argument("--max_hole_frac", type=float, default=0.05)
    parser.add_argument(
        "--pair_max_hole_frac",
        type=float,
        default=0.25,
        help="hole-gate ceiling for pair images (girl minus boy carves real holes)",
    )
    parser.add_argument(
        "--max_raggedness",
        type=float,
        default=20.0,
        help="contour-length / sqrt(mask-area) ceiling (salt-noise mask gate)",
    )
    parser.add_argument(
        "--min_comp_frac",
        type=float,
        default=0.002,
        help="components below this image-area fraction are specks (removed)",
    )
    parser.add_argument(
        "--min_partner_visible",
        type=float,
        default=0.5,
        help="pair images: fraction of the boy mask that must stay outside the paint",
    )
    parser.add_argument(
        "--min_face_frac",
        type=float,
        default=0.004,
        help="face level: minimum face area as a fraction of the image (else → slack)",
    )
    # ── cond stage: paint ──
    parser.add_argument(
        "--paint_color", nargs=3, type=int, default=[0, 0, 0], metavar=("R", "G", "B")
    )
    parser.add_argument(
        "--cond_background",
        choices=("white", "image"),
        default="white",
        help="paint the blob over a white canvas or over the real image "
        "(character-inpaint framing: fill the paint, keep the scene)",
    )
    parser.add_argument(
        "--aug_weights",
        nargs=len(LEVEL_NAMES),
        type=float,
        default=[0.15, 0.20, 0.15, 0.10, 0.25, 0.15],
        help="sampling weights for augment levels " + " / ".join(LEVEL_NAMES),
    )
    parser.add_argument(
        "--slack_grow",
        nargs=2,
        type=float,
        default=[1.5, 3.0],
        metavar=("LO", "HI"),
        help="slack level: target paint area as a multiple of the character bbox",
    )
    parser.add_argument(
        "--max_slack_coverage",
        type=float,
        default=0.7,
        help="slack level: paint may cover at most this image fraction",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="re-paint staged cond PNGs"
    )
    # ── encode stage ──
    parser.add_argument("--vae", default="models/vae/qwen_image_vae.safetensors")
    parser.add_argument("--vae_2d", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument(
        "--cond_overwrite",
        action="store_true",
        help="re-encode every cond latent (needed after any repaint)",
    )
    # ── text stage ──
    parser.add_argument(
        "--qwen3", default="models/text_encoders/qwen_3_06b_base.safetensors"
    )
    parser.add_argument(
        "--dit", default="models/diffusion_models/anima-base-v1.0.safetensors"
    )
    parser.add_argument("--t5_tokenizer_path", default=None)
    parser.add_argument("--text_batch_size", type=int, default=16)
    parser.add_argument(
        "--text_overwrite", action="store_true", help="re-encode existing TE caches"
    )
    # ── stage toggles ──
    parser.add_argument("--skip_select", action="store_true")
    parser.add_argument("--skip_sam", action="store_true")
    parser.add_argument("--skip_cond", action="store_true")
    parser.add_argument("--skip_captions", action="store_true")
    parser.add_argument("--skip_encode", action="store_true")
    parser.add_argument("--skip_text", action="store_true")
    args = parser.parse_args()

    base = Path(args.base)
    staging = base / "staging"

    if not args.skip_select:
        stage_select(
            Path(args.caption_index),
            Path(args.resized_dir),
            Path(args.caption_src),
            base,
            pairs=args.pairs,
            limit=args.limit,
        )

    if not args.skip_sam:
        stage_sam(
            base,
            face=args.face,
            threshold=args.threshold,
            sam_dilate=args.sam_dilate,
            batch_size=args.sam_batch_size,
            checkpoint=args.checkpoint,
            force=args.sam_force,
        )

    if not args.skip_cond:
        stage_cond(
            base,
            gate_kwargs=dict(
                min_coverage=args.min_coverage,
                max_coverage=args.max_coverage,
                min_dominance=args.min_dominance,
                max_hole_frac=args.max_hole_frac,
                min_comp_frac=args.min_comp_frac,
                max_raggedness=args.max_raggedness,
            ),
            paint_color=tuple(args.paint_color),
            cond_background=args.cond_background,
            aug_weights=args.aug_weights,
            slack_grow=tuple(args.slack_grow),
            min_partner_visible=args.min_partner_visible,
            min_face_frac=args.min_face_frac,
            pair_max_hole_frac=args.pair_max_hole_frac,
            max_slack_coverage=args.max_slack_coverage,
            overwrite=args.overwrite,
        )

    if not args.skip_captions:
        stage_captions(base, Path(args.resized_dir), Path(args.caption_src))

    if not args.skip_encode:
        stats = stage_encode(
            base / "cond_images",
            base / "cond",
            vae_path=args.vae,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            vae_2d=args.vae_2d,
            overwrite=args.cond_overwrite,
        )
        print(
            f"\nCond latent caching complete: {stats.written} cached, "
            f"{stats.skipped} skipped (already existed)"
        )

    if not args.skip_text:
        tstats = stage_text(
            staging,
            base / "text",
            qwen3_path=args.qwen3,
            dit_path=args.dit,
            t5_tokenizer_path=args.t5_tokenizer_path,
            batch_size=args.text_batch_size,
            overwrite=args.text_overwrite,
        )
        print(
            f"\nCaption text caching complete: {tstats.written} cached, "
            f"{tstats.skipped} skipped (already existed)"
        )


if __name__ == "__main__":
    main()
