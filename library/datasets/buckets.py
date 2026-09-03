import math
import random
from typing import NamedTuple, Tuple

# The free-fit tier / band / solver geometry is OWNED by ``anime_tools.buckets``
# (curation split, API-first T5 — 2026-09-03) and re-exported here, the way
# ``library/models/pe.py`` re-exports the PE tower: the package's resize stage
# and ``make preprocess-resize`` (a ``ResizeRequest`` itself now) land an image
# on the same ``(W, H)`` by construction. Trainer-only helpers — token-family
# budgets for compile, per-band clustering, σ-demote siblings, the frozen DCW
# aspect set, ``BucketManager`` — stay below. Torch-free and numpy-free on the
# import path (numpy is imported lazily inside BucketManager) so the GUI can
# read the tier table on the UI thread.
from anime_tools.buckets import (  # noqa: F401 — re-exports
    ALLOWED_TARGET_RES,
    DEFAULT_FREEFIT_MAX_RATIO,
    DEFAULT_TARGET_RES,
    EDGE_TOKEN_BANDS,
    FREEFIT_BAND_TOLERANCE,
    FREEFIT_FROZEN_EDGES,
    band_for_tier,
    choose_edge,
    freefit_band_for_edge,
    freefit_bucket,
)

# ``EDGE_TOKEN_BANDS`` — per-tier token-count bands, the free-fit search range
# for each tier edge (token count = (W//16)*(H//16), capped at 256/axis by
# rope). ``--target_res`` selects active tiers; ``choose_edge`` picks the one
# that resizes an image least; ``freefit_bucket`` lands the native-aspect
# (W, H) inside the band. Design: _archive/proposals/free_aspect_token_band_resize.md.

_band = band_for_tier


def token_count_families(target_res) -> int:
    """Number of distinct band-endpoint token counts across the active tiers
    (deduped) — drives the dynamo cache-size budget in ``compile_blocks``."""
    counts: set = set()
    for edge in target_res:
        lo, hi = _band(edge)
        counts.add(lo)
        counts.add(hi)
    return len(counts)


def token_count_range(target_res) -> tuple[int, int]:
    """(min, max) token count across the active tiers — bounds the
    ``mark_dynamic`` seq-length hint in ``compile_blocks``."""
    los = [_band(edge)[0] for edge in target_res]
    his = [_band(edge)[1] for edge in target_res]
    if not los:
        raise ValueError("token_count_range requires at least one tier")
    return min(los), max(his)


def token_counts_for_resos(resos) -> set:
    """Distinct token counts ``(W//16)*(H//16)`` over a set of (W, H) resolutions."""
    return {(w // 16) * (h // 16) for w, h in resos}


def snap_sample_size(width: int, height: int) -> Tuple[int, int]:
    """Snap a requested sample (W, H) to the DiT's 16px pixel grid — the same
    snap ``_sample_image_inference`` applies, shared so the compile token
    budget agrees with the seq len a sample prompt actually runs at."""
    return max(64, width - width % 16), max(64, height - height % 16)


def token_counts_for_sample_prompts(prompts) -> set:
    """Distinct DiT token counts the training sample prompts will request
    (``prompts`` = ``train_util.load_prompts`` dicts, width/height default 512).
    Folded into the compile token budget so a sample resolution outside the
    training buckets widens the compiled range instead of raising a
    dynamic-seq ConstraintViolationError mid-training (issue #42)."""
    counts: set = set()
    for prompt_dict in prompts:
        w, h = snap_sample_size(
            int(prompt_dict.get("width", 512)),
            int(prompt_dict.get("height", 512)),
        )
        counts.add((w // 16) * (h // 16))
    return counts


def cluster_token_bands(counts, rel_gap: float = 0.10) -> "list[tuple[int, int]]":
    """Cluster a token-count set into per-tier ``(lo, hi)`` bands.

    Data-driven (_archive/proposals/perband_dynamic_seq.md): sort the *actual*
    counts and split wherever the relative gap to the previous count exceeds
    ``rel_gap``. Handles every count source uniformly — tier buckets, sample
    prompts, σ-demote siblings — with no ``EDGE_TOKEN_BANDS`` special-casing:
    a sample prompt landing between tiers becomes its own singleton band, a
    single-tier pool degenerates to one band == the old union range.
    """
    ordered = sorted({int(c) for c in counts})
    if not ordered:
        return []
    bands: list[tuple[int, int]] = []
    lo = prev = ordered[0]
    for c in ordered[1:]:
        if (c - prev) > rel_gap * prev:
            bands.append((lo, prev))
            lo = c
        prev = c
    bands.append((lo, prev))
    return bands


def band_for_seq(bands, seq: int) -> "tuple[int, int] | None":
    """The band containing ``seq``, or None (gap / out of range). ``bands``
    must be sorted and non-overlapping (``cluster_token_bands`` output)."""
    import bisect

    if not bands:
        return None
    i = bisect.bisect_right([b[0] for b in bands], seq) - 1
    if i < 0 or seq > bands[i][1]:
        return None
    return bands[i]


def widen_bands(bands, extra: int) -> "list[tuple[int, int]]":
    """Widen each band's hi by ``extra`` (register tokens grow seq by a
    constant K; mid-stack insertion runs pre-insert blocks at the bare seq,
    so lo stays). Raises if widening would make adjacent bands touch/overlap
    — silent band merging would un-tighten the per-band graphs."""
    if extra <= 0:
        return list(bands)
    for (_, hi), (next_lo, _) in zip(bands, bands[1:]):
        if hi + extra >= next_lo:
            raise ValueError(
                f"extra_seq_tokens={extra} >= inter-band gap "
                f"({next_lo - hi} between hi={hi} and next lo={next_lo}); "
                "bands would merge — widen the clustering gap or drop "
                "--compile_seq_bands for this run"
            )
    return [(lo, hi + extra) for lo, hi in bands]


# The single measured-safe σ-demote route (1024→896). Other candidate routes
# (896→768, 1280→1024) failed/differ per their own gradient probe — do not add
# routes without one (_archive/sigma_lowres/bench/run_sigma_probe.py).
SIGMA_DEMOTE_ROUTE: Tuple[int, int] = (1024, 896)


def demote_bucket_for(
    width: int,
    height: int,
    native_edge: int,
    demote_edge: int,
    max_ratio: "float | None" = None,
) -> "tuple[int, int] | None":
    """σ-demote sibling grid ``(W', H')`` for a native free-fit bucket, or None
    if ``(width, height)`` isn't native to ``native_edge``'s token band. Pure
    and deterministic — preprocess emit and trainer fetch both call this so
    they derive the identical demoted grid.
    """
    if max_ratio is None:
        max_ratio = DEFAULT_FREEFIT_MAX_RATIO
    lo, hi = freefit_band_for_edge(native_edge)
    tok = (width // 16) * (height // 16)
    if not (lo <= tok <= hi):
        return None
    return freefit_bucket(width, height, freefit_band_for_edge(demote_edge), max_ratio)


def demoted_token_counts(resos, native_edge: int, demote_edge: int) -> set:
    """Distinct token counts the σ-demote siblings of ``resos`` will run at.
    Unioned into the compile token budget (``train.py::_derive_token_budget``)
    when sigma_lowres is on, so demoted forwards stay inside the compiled range.
    """
    counts: set = set()
    for w, h in resos:
        bucket = demote_bucket_for(w, h, native_edge, demote_edge)
        if bucket is not None:
            counts.add((bucket[0] // 16) * (bucket[1] // 16))
    return counts


# Frozen literal: dataset's top-5 (H, W) resolutions by frequency from the old
# discrete 1024-tier bucket pool (pre-free-fit). Kept for CNS calibration and
# mod-guidance distillation which still key off it (DCW name only because those
# consumers import the symbol; the DCW line itself is retired, _archive/dcw/).
DCW_ASPECT_BUCKETS: Tuple[Tuple[int, int], ...] = (
    (1200, 896),  # 896x1200 portrait, most common, 4200-tok
    (1344, 800),  # 800x1344 tall portrait, 4200-tok
    (896, 1200),  # 1200x896 landscape, 4200-tok
    (1344, 768),  # 768x1344 tall portrait, 4032-tok
    (1152, 896),  # 896x1152 portrait, 4032-tok
)
DCW_ASPECT_NAMES: Tuple[str, ...] = tuple(f"{h}x{w}" for h, w in DCW_ASPECT_BUCKETS)
DCW_ASPECT_TABLE: dict = {hw: i for i, hw in enumerate(DCW_ASPECT_BUCKETS)}
N_DCW_ASPECTS: int = len(DCW_ASPECT_BUCKETS)


def make_bucket_resolutions(max_reso, min_size=256, max_size=1024, divisible=64):
    """Generate bucket resolutions for multi-aspect-ratio training.
    Moved from model_util.py to avoid dependency."""
    max_width, max_height = max_reso
    max_area = max_width * max_height

    resos = set()

    width = int(math.sqrt(max_area) // divisible) * divisible
    resos.add((width, width))

    width = min_size
    while width <= max_size:
        height = min(max_size, int((max_area // width) // divisible) * divisible)
        if height >= min_size:
            resos.add((width, height))
            resos.add((height, width))

        width += divisible

    resos = list(resos)
    resos.sort()
    return resos


class BucketManager:
    def __init__(
        self, max_reso=None, min_size=None, max_size=None, reso_steps=None
    ) -> None:
        if max_size is not None:
            if max_reso is not None:
                assert max_size >= max_reso[0], (
                    "the max_size should be larger than the width of max_reso"
                )
                assert max_size >= max_reso[1], (
                    "the max_size should be larger than the height of max_reso"
                )
            if min_size is not None:
                assert max_size >= min_size, (
                    "the max_size should be larger than the min_size"
                )

        if max_reso is None:
            self.max_reso = None
            self.max_area = None
        else:
            self.max_reso = max_reso
            self.max_area = max_reso[0] * max_reso[1]
        self.min_size = min_size
        self.max_size = max_size
        self.reso_steps = reso_steps

        self.resos = []
        self.reso_to_id = {}
        self.buckets = []

    def add_image(self, reso, image_or_info):
        bucket_id = self.reso_to_id[reso]
        self.buckets[bucket_id].append(image_or_info)

    def shuffle(self):
        for bucket in self.buckets:
            random.shuffle(bucket)

    def sort(self):
        sorted_resos = self.resos.copy()
        sorted_resos.sort()

        sorted_buckets = []
        sorted_reso_to_id = {}
        for i, reso in enumerate(sorted_resos):
            bucket_id = self.reso_to_id[reso]
            sorted_buckets.append(self.buckets[bucket_id])
            sorted_reso_to_id[reso] = i

        self.resos = sorted_resos
        self.buckets = sorted_buckets
        self.reso_to_id = sorted_reso_to_id

    def make_buckets(
        self,
        freefit_resos=None,
        target_res=None,
    ):
        if freefit_resos is not None:
            # Free-fit: the on-disk cached (W, H) set IS the source of truth for
            # buckets — every cached latent exact-matches in select_bucket, nothing
            # AR-snaps at load. target_res is preprocess-only and inert here.
            resos = sorted(set(tuple(r) for r in freefit_resos))
        else:
            resos = make_bucket_resolutions(
                self.max_reso, self.min_size, self.max_size, self.reso_steps
            )
        self.set_predefined_resos(resos)

    def set_predefined_resos(self, resos):
        import numpy as np

        self.predefined_resos = resos.copy()
        self.predefined_resos_set = set(resos)
        self.predefined_aspect_ratios = np.array([w / h for w, h in resos])

    def add_if_new_reso(self, reso):
        if reso not in self.reso_to_id:
            bucket_id = len(self.resos)
            self.reso_to_id[reso] = bucket_id
            self.resos.append(reso)
            self.buckets.append([])

    def select_bucket(self, image_width, image_height):
        aspect_ratio = image_width / image_height
        reso = (image_width, image_height)
        if reso in self.predefined_resos_set:
            pass
        else:
            import numpy as np

            ar_errors = self.predefined_aspect_ratios - aspect_ratio
            predefined_bucket_id = np.abs(ar_errors).argmin()
            reso = self.predefined_resos[predefined_bucket_id]

        ar_reso = reso[0] / reso[1]
        if aspect_ratio > ar_reso:
            scale = reso[1] / image_height
        else:
            scale = reso[0] / image_width

        resized_size = (
            int(image_width * scale + 0.5),
            int(image_height * scale + 0.5),
        )

        self.add_if_new_reso(reso)

        ar_error = (reso[0] / reso[1]) - aspect_ratio
        return reso, resized_size, ar_error

    @staticmethod
    def get_crop_ltrb(bucket_reso: Tuple[int, int], image_size: Tuple[int, int]):
        # Calculate crop left/top according to the preprocessing of Stability AI. Crop right is calculated for flip augmentation.

        bucket_ar = bucket_reso[0] / bucket_reso[1]
        image_ar = image_size[0] / image_size[1]
        if bucket_ar > image_ar:
            resized_width = bucket_reso[1] * image_ar
            resized_height = bucket_reso[1]
        else:
            resized_width = bucket_reso[0]
            resized_height = bucket_reso[0] / image_ar
        crop_left = (bucket_reso[0] - resized_width) // 2
        crop_top = (bucket_reso[1] - resized_height) // 2
        crop_right = crop_left + resized_width
        crop_bottom = crop_top + resized_height
        return crop_left, crop_top, crop_right, crop_bottom


class BucketBatchIndex(NamedTuple):
    bucket_index: int
    bucket_batch_size: int
    batch_index: int
