"""Identity-pair reference sampler for IP-Adapter distinct-pair training.

Consumes the *method-agnostic* caption index built by
``python -m anime_tools.captions.index`` (``make caption-index``) (``post_image_dataset/captions/
caption_index.json``) and owns the IP-Adapter **policy** on top of it: a
tiered character → copyright → artist back-off that, given a target image,
returns a *different* image of the same identity to feed the IP path.

The index encodes no policy; level priority, the cross-artist constraint and
the candidate-pool restriction all live here.

Distinct pairs, not self-pairing: with reference == VAE target the IP path can
lower the loss by copying the target's pixels instead of learning identity. See
``_archive/proposals/ip-adapter-identity-pairs.md``.
"""

from __future__ import annotations

import json
import os
import random
from typing import Iterable, Optional

# Tightest → loosest. ``resolve`` walks this in order and returns the first
# tier that has a distinct same-group image available.
LEVELS = ("character", "copyright", "artist")


class IdentityPairSampler:
    """Picks a distinct same-identity (or unrelated) reference stem.

    Parameters
    ----------
    index_path:
        Path to ``caption_index.json``.
    min_level:
        Loosest tier allowed before falling back to self. One of
        ``character`` / ``copyright`` / ``artist``. ``artist`` (default) allows
        the full back-off; ``character`` restricts pairing to same-character
        only (no franchise/artist fallback).
    cross_artist:
        When True, character/copyright-tier matches must come from a *different*
        artist — forces the IP path to carry identity without the source
        artist's style (the ``identity_cross_artist`` mode). Artist-tier
        matches are unaffected (they share the artist by definition).
    restrict_stems:
        Optional set of stems the *reference* may be drawn from. Used to keep
        training references inside the training split (no validation-image
        leakage). ``None`` ⇒ any stem in the index is eligible.
    """

    def __init__(
        self,
        index_path: str,
        *,
        min_level: str = "artist",
        cross_artist: bool = False,
        restrict_stems: Optional[Iterable[str]] = None,
    ) -> None:
        if min_level not in LEVELS:
            raise ValueError(f"min_level must be one of {LEVELS}, got {min_level!r}")
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        self.index_path = index_path
        self.image_meta: dict[str, dict] = index["image_meta"]
        self.groups: dict[str, dict[str, list[str]]] = index["groups"]
        self.cross_artist = bool(cross_artist)
        # Active tiers = prefix of LEVELS up to and including min_level.
        self.levels = LEVELS[: LEVELS.index(min_level) + 1]

        self._restrict: Optional[set[str]] = (
            set(restrict_stems) if restrict_stems is not None else None
        )
        # Candidate pool for shuffled() — every reference-eligible stem.
        self._all_stems: list[str] = sorted(
            s for s in self.image_meta if self._eligible(s)
        )

    # -- helpers ----------------------------------------------------------

    def _eligible(self, stem: str) -> bool:
        return self._restrict is None or stem in self._restrict

    def has(self, stem: str) -> bool:
        """True when ``stem`` is present in the index (so we can pair it)."""
        return stem in self.image_meta

    def rel_dir(self, stem: str) -> str:
        """Directory of the index's recorded path for ``stem`` (e.g. the
        artist subdir), used to resolve the nested PE-feature cache."""
        meta = self.image_meta.get(stem)
        if meta is None:
            return ""
        return os.path.dirname(meta.get("path", ""))

    def _identity_tags(self, stem: str) -> set[str]:
        """character + copyright tags, used to test 'unrelated' for shuffled."""
        meta = self.image_meta.get(stem, {})
        return set(meta.get("character", [])) | set(meta.get("copyright", []))

    # -- policy -----------------------------------------------------------

    def resolve(self, target_stem: str, rng: random.Random) -> tuple[str, str]:
        """Return ``(reference_stem, level)`` for a *distinct* same-identity
        reference, walking character → copyright → artist. Falls back to
        ``(target_stem, "self")`` when no distinct positive is reachable."""
        meta = self.image_meta.get(target_stem)
        if meta is None:
            return target_stem, "self"
        target_artists = set(meta.get("artist", []))

        for level in self.levels:
            candidates: set[str] = set()
            for tag in meta.get(level, []):
                for s in self.groups.get(level, {}).get(tag, []):
                    if s == target_stem or not self._eligible(s):
                        continue
                    if (
                        self.cross_artist
                        and level in ("character", "copyright")
                        and target_artists
                        and target_artists
                        & set(self.image_meta.get(s, {}).get("artist", []))
                    ):
                        # Same artist — rejected in cross-artist mode so the
                        # match can't preserve the source artist's style.
                        continue
                    candidates.add(s)
            if candidates:
                return rng.choice(sorted(candidates)), level
        return target_stem, "self"

    def shuffled(self, target_stem: str, rng: random.Random) -> tuple[str, str]:
        """Return ``(reference_stem, "shuffled")`` for an *unrelated* image —
        a different identity, used as the validation negative baseline. Tries
        to avoid sharing any character/copyright tag; accepts any distinct stem
        after a few rejections (small datasets may be densely connected)."""
        if not self._all_stems:
            return target_stem, "self"
        target_identity = self._identity_tags(target_stem)
        for _ in range(8):
            cand = rng.choice(self._all_stems)
            if cand == target_stem:
                continue
            if target_identity and (self._identity_tags(cand) & target_identity):
                continue
            return cand, "shuffled"
        # Fall back to any distinct stem.
        for _ in range(8):
            cand = rng.choice(self._all_stems)
            if cand != target_stem:
                return cand, "shuffled"
        return target_stem, "self"

    def _diff_char_candidates(
        self, target_stem: str, level: str, target_chars: set[str]
    ) -> set[str]:
        """Stems sharing a ``level`` tag (``artist``/``copyright``) with the
        target but whose ``character`` tags are non-empty and **disjoint** from
        the target's — a genuine same-context / different-identity negative.
        Empty when the target's ``level`` tags are unpopulated or every sibling
        shares a character. Shared by ``hard_negative`` (artist tier) and
        ``hard_negative_backoff`` (artist then copyright)."""
        meta = self.image_meta.get(target_stem, {})
        candidates: set[str] = set()
        for tag in meta.get(level, []):
            for s in self.groups.get(level, {}).get(tag, []):
                if s == target_stem or not self._eligible(s):
                    continue
                cand_chars = set(self.image_meta.get(s, {}).get("character", []))
                if cand_chars and not (cand_chars & target_chars):
                    candidates.add(s)
        return candidates

    def hard_negative(self, target_stem: str, rng: random.Random) -> tuple[str, str]:
        """Return ``(reference_stem, level)`` for a *hard* negative — a
        same-artist image whose ``character`` tags are **disjoint** from the
        target's (style-matched, content-different).

        Both sides must be character-tagged for the contrast to be genuine
        (otherwise "different character" is vacuous). When no such sibling
        exists — orphan artist, untagged target, or a dataset where the artist's
        images all share characters — falls back to ``shuffled()`` (returning
        its ``"shuffled"`` level so callers can see the degradation).
        ``hard_negative_backoff`` adds copyright / original tiers before that
        fallback."""
        meta = self.image_meta.get(target_stem)
        if meta is None:
            return self.shuffled(target_stem, rng)
        target_chars = set(meta.get("character", []))
        target_artists = set(meta.get("artist", []))
        if not target_chars or not target_artists:
            return self.shuffled(target_stem, rng)

        candidates = self._diff_char_candidates(target_stem, "artist", target_chars)
        if candidates:
            return rng.choice(sorted(candidates)), "hard"
        return self.shuffled(target_stem, rng)

    def _same_artist_original_candidates(self, target_stem: str) -> set[str]:
        """Same-artist stems also tagged ``original`` — a *different original
        work* by the same artist. For OC targets (``copyright == ["original"]``)
        there is no character tag to disjoint on, so this is the only
        style-matched hard negative available. May occasionally surface the
        artist's own recurring OC (the ``original`` tag does not disambiguate
        which OC), which is a weaker — but still meaningfully harder than
        ``shuffled`` — negative. Empty when the target is not ``original`` or
        has no same-artist ``original`` sibling."""
        meta = self.image_meta.get(target_stem, {})
        if "original" not in set(meta.get("copyright", [])):
            return set()
        candidates: set[str] = set()
        for artist in meta.get("artist", []):
            for s in self.groups.get("artist", {}).get(artist, []):
                if s == target_stem or not self._eligible(s):
                    continue
                if "original" in set(self.image_meta.get(s, {}).get("copyright", [])):
                    candidates.add(s)
        return candidates

    def hard_negative_backoff(
        self, target_stem: str, rng: random.Random
    ) -> tuple[str, str]:
        """Tiered hard negative: same-artist/different-character, then
        same-copyright/different-character, then (for ``original`` targets) a
        different same-artist ``original`` work, then ``shuffled()``. Mirrors
        ``resolve``'s back-off but on the *negative* side.

        The copyright tier rescues character-tagged targets whose artist has no
        different-character sibling (franchises are densely populated). The
        ``original`` tier rescues OC targets — which carry no character tag at
        all and so skip both character tiers — with a style-matched
        same-artist/different-work negative (the bulk of the residual
        ``shuffled`` fallback on OC-heavy datasets). Each tier trades a little
        style/identity control for far more genuine, non-``shuffled`` negatives.

        Returns ``(reference_stem, level)`` with ``level`` in
        ``{"hard_artist", "hard_copyright", "hard_original", "shuffled"}`` so
        callers can log the hardness mix. Falls back to ``shuffled()`` only when
        no tier is reachable."""
        meta = self.image_meta.get(target_stem)
        if meta is None:
            return self.shuffled(target_stem, rng)
        target_chars = set(meta.get("character", []))
        if target_chars:
            for level, label in (
                ("artist", "hard_artist"),
                ("copyright", "hard_copyright"),
            ):
                candidates = self._diff_char_candidates(target_stem, level, target_chars)
                if candidates:
                    return rng.choice(sorted(candidates)), label

        original = self._same_artist_original_candidates(target_stem)
        if original:
            return rng.choice(sorted(original)), "hard_original"
        return self.shuffled(target_stem, rng)

    def draw(
        self, target_stem: str, mode: str, rng: random.Random
    ) -> tuple[str, str]:
        """Dispatch a single negative draw by ``mode`` string, returning
        ``(reference_stem, level)``. One path shared by the dataset's
        ``__getitem__`` and the setup-time level histogram. ``shuffled`` and
        ``jaccard`` both source ``shuffled()`` (jaccard adds a weight, not a
        different stem)."""
        if mode == "hard":
            return self.hard_negative(target_stem, rng)
        if mode == "hard_backoff":
            return self.hard_negative_backoff(target_stem, rng)
        return self.shuffled(target_stem, rng)

    def tag_jaccard(self, stem_a: str, stem_b: str) -> float:
        """Jaccard overlap of the two stems' identity+style tag sets
        (``character ∪ copyright ∪ artist``), used by the ``jaccard`` negative
        mode to down-weight less-surprising mismatches. Returns ``0.0`` when
        either stem is unknown or both tag sets are empty."""

        def _tags(stem: str) -> set[str]:
            m = self.image_meta.get(stem, {})
            return (
                set(m.get("character", []))
                | set(m.get("copyright", []))
                | set(m.get("artist", []))
            )

        a, b = _tags(stem_a), _tags(stem_b)
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)
