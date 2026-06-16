#!/usr/bin/env python3
"""near_twin engine — pairing gates, Stage-B grid match, discriminators.

The pair-mining core of the near-twin tag-gap miner: the same-size/tag-pivot
prune, the dense grid match, the discriminators, and ``run_artist`` (Stages
A/B). The reusable embedding half — member discovery, PE-Spatial encoding, and
the per-image feature cache — was promoted to ``library.vision.pe_features`` and
is re-imported below (names preserved for backward compatibility). Rendering and
export live in ``near_twin.outputs``; the CLI + config layering in
``near_twin.__main__`` (see that module's docstring for the full pipeline).
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch  # noqa: F401  (kept for API parity / discriminate_signal device handling)
from PIL import Image

# Run from the repo root; `library` is installed editable (`uv sync`).
from library.vision.encoder import encode_pe_from_imageminus1to1  # noqa: F401  (kept for API parity)

# Embedding primitive + member discovery promoted to the library. Re-exported
# here so ``near_twin.outputs`` / ``__main__`` and any external importer keep
# pulling these names from ``.engine`` unchanged.
from library.vision.pe_features import (  # noqa: F401
    CACHE_ROOT,
    GRID_CACHE,
    GRID_NATIVE,
    IMAGE_EXTS,
    PE_NATIVE,
    Feature,
    Member,
    _cache_path,
    _dir_hash,
    _image_size,
    caption_text,
    embed_members,
    gather_members,
    keep_size_cohabiting,
    normalize_tag,
    read_tags,
)

# Stage-B dense grid match promoted to the library so the dataset-grouping
# curation tool shares the exact same near-twin gate. Re-exported here so
# ``near_twin.outputs`` / ``__main__`` keep importing them from ``.engine``.
from library.vision.pe_matching import (  # noqa: F401
    MatchResult,
    _geom_filter,
    match_grids,
    pool_cells,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------- pairing prune


def prune_for_pairing(
    members: list[Member], mode: str, target: set[str]
) -> list[Member]:
    """Members that could still form an accepted pair → embed only these.

    Region/signal modes: just the same-size gate (``keep_size_cohabiting``).

    Tag mode adds the **tag pivot**: an accepted pair has the target tag in
    *exactly one* member, so a same-size group is only useful when it holds BOTH
    a tagged and an untagged member. An all-tagged or all-untagged size group can
    never produce an exactly-one gap, so none of it is worth embedding.
    """
    if mode != "tag":
        return keep_size_cohabiting(members)
    by_size: dict[tuple[int, int], list[Member]] = {}
    for m in members:
        if m.wh != (0, 0):
            by_size.setdefault(m.wh, []).append(m)
    kept: list[Member] = []
    for grp in by_size.values():
        flags = [bool(target & read_tags(m.txt_path)) for m in grp]
        if any(flags) and not all(flags):  # both a tagged and an untagged member
            kept.extend(grp)
    return kept


# ---------------------------------------------------------------------------- identity pairs


def _mean_saturation(image_path: Path) -> float:
    """Mean HSV saturation in [0, 1] over a small thumbnail (cheap, size-robust)."""
    try:
        with Image.open(image_path) as im:
            thumb = im.convert("RGB")
            thumb.thumbnail((256, 256))
            s = np.asarray(thumb.convert("HSV"))[..., 1]
    except (
        Exception
    ):  # unreadable / truncated → treat as fully desaturated (won't pass a floor)
        return 0.0
    return float(s.mean()) / 255.0


def select_identity_members(
    by_artist: dict[str, list[Member]],
    target_tags: set[str],
    n: int,
    saturation_min: float,
    seed: int,
    exclude_stems: set[str] | None = None,
) -> list[Member]:
    """Pick ``n`` clean singles to stage as identity (cond==target) pairs.

    Draws from the **full** gathered pool (not the same-size/twin-gated set) —
    every member carrying NONE of ``target_tags`` is a case whose correct sanitize
    output is "do nothing", so staging it as cond==target teaches the no-op and
    counters the adapter's standing bias to globally transform every input (the
    wash-out failure mode). With ``saturation_min`` > 0 the pool is filtered to
    vivid images, which *also* injects the saturated clean targets the mined twins
    under-represent. Selection is random but ``seed``-fixed for run-to-run
    idempotency, and lazy — tags/saturation are read only until ``n`` are
    collected, so a saturation floor never decodes the whole corpus.
    """
    if n <= 0:
        return []
    exclude = exclude_stems or set()
    pool = [
        m
        for members in by_artist.values()
        for m in members
        if m.wh != (0, 0) and m.stem not in exclude
    ]
    random.Random(seed).shuffle(pool)
    picked: list[Member] = []
    for m in pool:
        if target_tags & read_tags(m.txt_path):
            continue  # carries the attribute we remove — not an identity case
        if saturation_min > 0.0 and _mean_saturation(m.image_path) < saturation_min:
            continue
        picked.append(m)
        if len(picked) >= n:
            break
    return picked


# ---------------------------------------------------------------------------- Stage B match
# ``match_grids`` / ``MatchResult`` / ``_geom_filter`` now live in
# ``library.vision.pe_matching`` (re-exported at the top of this module).


def _largest_blob(cells: set[int], G: int) -> tuple[int, set[int]]:
    """Largest 4-connected component of ``cells`` on the G×G grid → (size, blob)."""
    remaining = set(cells)
    best: set[int] = set()
    while remaining:
        seed = next(iter(remaining))
        comp: set[int] = set()
        q = deque([seed])
        remaining.discard(seed)
        while q:
            c = q.popleft()
            comp.add(c)
            r, col = divmod(c, G)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, col + dc
                if 0 <= nr < G and 0 <= nc < G:
                    n = nr * G + nc
                    if n in remaining:
                        remaining.discard(n)
                        q.append(n)
        if len(comp) > len(best):
            best = comp
    return len(best), best


def diff_bbox_norm(cells: set[int], G: int) -> tuple[float, float, float, float]:
    """Bounding box over ``cells`` (grid idx) → normalized (x0, y0, x1, y1) in [0,1]."""
    if not cells:
        return (0.0, 0.0, 0.0, 0.0)
    rs = [c // G for c in cells]
    cs = [c % G for c in cells]
    return (min(cs) / G, min(rs) / G, (max(cs) + 1) / G, (max(rs) + 1) / G)


# ---------------------------------------------------------------------------- discriminator


@dataclass
class PairVerdict:
    accept: bool
    gap_holder: str  # "a" | "b" | "?"
    n_extra_diff: int
    extra_diff_tags: list[str]
    reason: str = ""


def discriminate_tag(
    tags_a: set[str],
    tags_b: set[str],
    target: set[str],
    max_extra: int,
    rest_jaccard_min: float,
) -> PairVerdict:
    """Keep pairs where the target tag is present in **exactly one** member."""
    in_a = bool(target & tags_a)
    in_b = bool(target & tags_b)
    if in_a == in_b:  # both or neither → not a single-attribute gap on this tag
        return PairVerdict(False, "?", 0, [], "tag present in both/neither")
    gap_holder = "a" if in_a else "b"
    rest_a, rest_b = tags_a - target, tags_b - target
    extra = sorted(rest_a ^ rest_b)
    if max_extra >= 0 and len(extra) > max_extra:
        return PairVerdict(
            False, gap_holder, len(extra), extra, "too many other tag differences"
        )
    if rest_jaccard_min > 0:
        union = rest_a | rest_b
        jac = len(rest_a & rest_b) / len(union) if union else 1.0
        if jac < rest_jaccard_min:
            return PairVerdict(
                False, gap_holder, len(extra), extra, f"rest-jaccard {jac:.2f} < min"
            )
    return PairVerdict(True, gap_holder, len(extra), extra)


def discriminate_region(
    match: MatchResult, region_min_frac: float, region_max_frac: float, scatter_max: int
) -> PairVerdict:
    """Tagless: a single compact diff region of the right rough size IS the gap.

    ``gap_holder`` is a best-effort guess (the member with more unmatched cells
    in the diff region — the side carrying the added content); cross-check by
    eyeballing the HTML.
    """
    N = match.G * match.G
    size, blob = _largest_blob(match.diff_cells, match.G)
    frac = size / N
    scatter = len(match.diff_cells) - size  # diff cells outside the main blob
    if not (region_min_frac <= frac <= region_max_frac):
        return PairVerdict(
            False, "?", scatter, [], f"region frac {frac:.2f} out of range"
        )
    if scatter > scatter_max:
        return PairVerdict(False, "?", scatter, [], f"diff scatter {scatter} > max")
    gap_holder = "a" if len(match.diff_a & blob) >= len(match.diff_b & blob) else "b"
    return PairVerdict(True, gap_holder, scatter, [])


# ---------------------------------------------------------------------------- ranked pair record


@dataclass
class PairRecord:
    artist: str
    a: Member
    b: Member
    cosine: float
    match: MatchResult
    verdict: PairVerdict
    extra_fields: dict = field(default_factory=dict)

    @property
    def ids(self) -> tuple[str, str]:
        return tuple(sorted((self.a.stem, self.b.stem)))

    @property
    def pair_id(self) -> str:
        x, y = self.ids
        return f"{x}-{y}"

    def holder_member(self) -> Member:
        return self.a if self.verdict.gap_holder == "a" else self.b

    def clean_member(self) -> Member:
        return self.b if self.verdict.gap_holder == "a" else self.a


# ---------------------------------------------------------------------------- core run


def run_artist(
    artist: str,
    members: list[Member],
    feats: dict[str, Feature],
    args: argparse.Namespace,
    target_tags: set[str],
) -> list[PairRecord]:
    out: list[PairRecord] = []
    stems = [m.stem for m in members if m.stem in feats]
    cls = np.stack([feats[s].cls for s in stems]) if stems else np.empty((0, 768))
    idx = {m.stem: m for m in members}
    # Tag mode: read each sidecar once and cache the target-presence flag + full
    # tagset (reused by discriminate_tag below, so no per-pair re-read).
    tagsets = (
        {s: read_tags(idx[s].txt_path) for s in stems} if args.mode == "tag" else {}
    )
    n = len(stems)
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = stems[i], stems[j]
            if idx[si].wh != idx[sj].wh:  # same-size gate: exact (W, H) only
                continue
            if args.mode == "tag":  # tag pivot: skip same-status pairs before the match
                if bool(target_tags & tagsets[si]) == bool(target_tags & tagsets[sj]):
                    continue
            if args.id_window and si.isdigit() and sj.isdigit():
                if abs(int(si) - int(sj)) > args.id_window:
                    continue
            cos = float(cls[i] @ cls[j])
            if cos < args.sim_min:  # Stage A prefilter
                continue
            match = match_grids(
                feats[si],
                feats[sj],
                args.grid,
                args.cell_match_min,
                args.ratio,
                args.geom_check,
            )
            if match.match_frac < args.match_frac_min:  # Stage B near-twin gate
                continue
            ma, mb = idx[si], idx[sj]
            if args.mode == "tag":
                verdict = discriminate_tag(
                    tagsets[si],
                    tagsets[sj],
                    target_tags,
                    args.max_extra_diff,
                    args.rest_jaccard_min,
                )
            elif args.mode == "region":
                verdict = discriminate_region(
                    match,
                    args.region_min_frac,
                    args.region_max_frac,
                    args.region_scatter_max,
                )
            else:  # signal
                verdict = discriminate_signal(ma, mb, args)
            if not verdict.accept:
                continue
            out.append(PairRecord(artist, ma, mb, cos, match, verdict))

    # Rank by edit-cleanliness: fewest other differences first, then tightest twin.
    out.sort(key=lambda p: (p.verdict.n_extra_diff, -p.cosine))
    if args.per_artist_topk:
        out = out[: args.per_artist_topk]
    return out


# ---------------------------------------------------------------------------- signal mode (MIT text)


_SIGNAL_STATE: dict = {}


def _mit_text_fraction(member: Member, device: str) -> float:
    """Per-image MIT text-area fraction — the detector behind post_image_dataset/masks/."""
    if "mit_model" not in _SIGNAL_STATE:
        sys.path.insert(0, str(REPO_ROOT / "scripts" / "preprocess"))
        from generate_masks_mit import _detect_mask, _load_model  # type: ignore

        _SIGNAL_STATE["mit_model"] = _load_model(None, device=device)
        _SIGNAL_STATE["mit_detect"] = _detect_mask
    cache = CACHE_ROOT / _dir_hash(member.image_path.parent) / f"{member.stem}.mit_text"
    if cache.is_file():
        return float(cache.read_text())
    with Image.open(member.image_path) as im:
        arr = np.asarray(im.convert("RGB"))
    mask = _SIGNAL_STATE["mit_detect"](
        _SIGNAL_STATE["mit_model"], arr, device=device, text_threshold=0.8
    )
    frac = float(np.count_nonzero(mask) / mask.size)
    cache.write_text(f"{frac:.6f}")
    return frac


def discriminate_signal(a: Member, b: Member, args: argparse.Namespace) -> PairVerdict:
    if args.signal != "mit_text":
        raise ValueError(f"unknown signal {args.signal!r}")
    fa = _mit_text_fraction(a, args.device)
    fb = _mit_text_fraction(b, args.device)
    hi, lo = (fa, fb) if fa >= fb else (fb, fa)
    if (
        hi - lo < args.signal_delta or lo > args.signal_delta
    ):  # gap present + low side ≈ 0
        return PairVerdict(False, "?", 0, [], f"signal gap {hi - lo:.3f} insufficient")
    gap_holder = "a" if fa >= fb else "b"
    return PairVerdict(True, gap_holder, 0, [])
