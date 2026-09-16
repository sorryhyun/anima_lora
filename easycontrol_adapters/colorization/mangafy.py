"""Mangafication: color illustration → synthetic B&W manga (lineart + screentone).

Pure ``cv2`` / ``numpy``, no model downloads. Produces a 3-channel grayscale
RGB image approximating a screentoned manga page, used as the *condition*
(target = original color image) for the colorization adapter.

Pipeline: XDoG lineart (crisp ink contours) → luminance-banded screentone
(each band gets its own seeded dot/line/cross pattern + period/angle, so
texture changes where value changes) → composite line over tone as 3-channel
RGB. Per-image jitter (seeded by stem) keeps the model from keying on one
fixed screen operator.
"""

from __future__ import annotations

import cv2
import numpy as np

# ── Default knobs (overridable; jittered per-stem when ``seed`` is given) ─────
_XDOG_SIGMA = 0.5  # base blur scale (px) -> stroke width, scales with image below
_XDOG_K = 1.6  # second-Gaussian scale ratio
_XDOG_SHARP = 12.0  # p - DoG sharpening; higher = thinner/harder lines
_XDOG_EPS = 0.10  # soft-threshold midpoint
_XDOG_PHI = 12.0  # soft-threshold slope
# Screen pitch (px @ ~900px short side; scales with image). Keep comfortably above
# the native Nyquist (~2px) or a rotated screen beats into wavy moiré across smooth
# tone; line/cross need a larger pitch than dots since a 1-bit stripe aliases worse.
_TONE_PERIOD = 2  # dot screen period - fine manga screentone pitch, moiré-safe
_HATCH_PERIOD = 1  # line/cross period (resolves cleanly under the ss=4+ supersample)
# Screen is rendered at `ss`x, hard-thresholded, then resolved by an area-average
# downscale (cv2.INTER_AREA = exact box mean) so fine/rotated screens anti-alias
# into smooth gray instead of folding into moiré. ss sets the tone-smoothness
# (ss^2 gray levels); area-average beats Gaussian/Lanczos/soft-threshold here.
_TONE_SS = 8
_TONE_ANGLE = 45.0  # screen rotation (deg)
_TONE_WHITE_CUT = 0.90  # luminance >= this -> pure white (highlights, no tone)
_TONE_BLACK_CUT = 0.10  # luminance <= this -> solid black (deep shadow)
# Real manga mixes clustered-dot, parallel-line/hatch, and cross-hatch tone; each
# luminance band (see below) draws its own pattern so texture changes where value does.
_TONE_KINDS = ("dot", "line", "cross")
# Weighted draw per band. Hatch over large flat regions of smooth-shaded
# illustrations reads as dense stripes, so line is down-weighted and cross is off;
# every kind stays in the table (the draw consumes one rng slot per band either way).
_TONE_KIND_WEIGHTS = (0.8, 0.2, 0.0)  # P(dot, line, cross)
# The toned luminance range is split into this many bands per image; each gets its
# own pattern/angle/period. 1 = single tone whole image.
_TONE_BAND_COUNTS = (1, 2, 3, 4)
_TONE_BAND_WEIGHTS = (0.10, 0.40, 0.35, 0.15)
# Shadow-detail lift: a shadow-gated unsharp mask on the tone-fill luminance so dark
# fabric/hair keeps its fold/weave relief instead of crushing to flat black, without
# lifting the darkness itself.
# Gated off above `_DETAIL_GATE_HI` to keep skin/highlights faithful. XDoG still runs
# on raw luminance. `_DETAIL_AMOUNT = 0` disables.
_DETAIL_AMOUNT = 1.0  # unsharp strength on shadow local-contrast; 0 -> no lift
_DETAIL_SIGMA = 3.0  # high-pass blur radius (px @ ~900px short side; scales with image)
_DETAIL_GATE_LO = 0.15  # luminance <= this -> full lift
_DETAIL_GATE_HI = 0.45  # luminance >= this -> no lift (skin / highlights untouched)


def _luminance(img_rgb: np.ndarray, weights: tuple[float, float, float]) -> np.ndarray:
    """RGB uint8 → float32 luminance in [0, 1] with (jitterable) channel weights."""
    rgb = img_rgb.astype(np.float32) / 255.0
    w = np.asarray(weights, dtype=np.float32)
    w = w / w.sum()
    return np.clip(rgb @ w, 0.0, 1.0)


def _enhance_shadow_detail(
    gray: np.ndarray,
    *,
    amount: float,
    sigma: float,
    gate_lo: float,
    gate_hi: float,
) -> np.ndarray:
    """Shadow-gated unsharp mask on luminance (ink = 0, paper = 1).

    Amplifies ``gray - blur(gray)`` around the region's own mean so dark
    fabric gains fold/weave relief without lifting its darkness. Gate fades
    the effect from full at/below ``gate_lo`` to off at/above ``gate_hi``.
    Returns ``gray`` unchanged when ``amount <= 0``. Used only for the
    tone-fill input; XDoG keeps the raw luminance."""
    if amount <= 0:
        return gray
    lp = cv2.GaussianBlur(gray, (0, 0), float(sigma))
    w = np.clip((gate_hi - gray) / max(gate_hi - gate_lo, 1e-6), 0.0, 1.0)
    enhanced = gray + (w * amount) * (gray - lp)
    return np.clip(enhanced, 0.0, 1.0).astype(np.float32)


def _xdog(
    gray: np.ndarray,
    *,
    sigma: float,
    k: float,
    sharp: float,
    eps: float,
    phi: float,
) -> np.ndarray:
    """Winnemöller XDoG. Returns float32 in [0, 1]; ink ≈ 0, paper ≈ 1."""
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k)
    dog = (1.0 + sharp) * g1 - sharp * g2
    out = np.ones_like(dog)
    below = dog < eps
    out[below] = 1.0 + np.tanh(phi * (dog[below] - eps))
    return np.clip(out, 0.0, 1.0)


def _uniformize(field: np.ndarray) -> np.ndarray:
    """Rank-normalize a screen field to uniform (0, 1) — the ordered-dither fix.

    Ink coverage equals ``1 - gray`` only when the field is uniform on
    [0,1]; the raw ``cross`` sinusoid has mean ~0.70 and over-inks bright
    tones (the "cow-print" failure). Rank-normalizing to a uniform CDF is
    monotonic, so texture (which pixel inks first) is untouched — only
    coverage is corrected."""
    flat = field.reshape(-1)
    ranks = np.empty(flat.shape[0], dtype=np.float32)
    order = np.argsort(flat, kind="stable")
    ranks[order] = (np.arange(flat.shape[0], dtype=np.float32) + 0.5) / flat.shape[0]
    return ranks.reshape(field.shape)


def _screen_field(
    h: int, w: int, *, period: float, angle: float, kind: str
) -> np.ndarray:
    """Periodic screen field in [0, 1] used as a spatially-varying ink threshold.

    ``dot`` = rotated clustered-dot screen (sin x sin); ``line`` = a single
    rotated sinusoid; ``cross`` = two perpendicular line screens unioned.
    Rank-normalized via :func:`_uniformize` so ink coverage equals
    ``1 - luminance`` for every pattern."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.deg2rad(angle)
    xr = xs * np.cos(a) - ys * np.sin(a)
    yr = xs * np.sin(a) + ys * np.cos(a)
    wx = 2.0 * np.pi * xr / period
    wy = 2.0 * np.pi * yr / period
    if kind == "line":
        field = (np.sin(wx) + 1.0) * 0.5
    elif kind == "cross":
        # union of two orthogonal line screens
        field = np.maximum((np.sin(wx) + 1.0) * 0.5, (np.sin(wy) + 1.0) * 0.5)
    else:
        field = (np.sin(wx) * np.sin(wy) + 1.0) * 0.5  # "dot" (default)
    return _uniformize(field)


def _pick_band_count(rng: np.random.Generator) -> int:
    """Seeded number of luminance bands (``_TONE_BAND_COUNTS``)."""
    return int(rng.choice(_TONE_BAND_COUNTS, p=_TONE_BAND_WEIGHTS))


def _band_kinds(n: int, rng: np.random.Generator) -> list[str]:
    """``n`` screen patterns, one per luminance band, ``dot``-weighted.

    One draw per band, in order, so any backend handed the same ``rng``
    picks the same structure."""
    return [str(rng.choice(_TONE_KINDS, p=_TONE_KIND_WEIGHTS)) for _ in range(n)]


def _luma_band_labels(
    gray: np.ndarray, *, black_cut: float, white_cut: float, n: int
) -> np.ndarray:
    """Per-pixel luminance-band index in ``[0, n)`` (0 = darkest toned band).

    Band edges are quantiles of the toned pixels (between the cuts), so
    every band is populated regardless of the image's histogram. Pixels
    past the cuts are clamped here but overwritten by the solid-black /
    white flatten in :func:`_render_tone`."""
    toned = gray[(gray > black_cut) & (gray < white_cut)]
    if n <= 1 or toned.size == 0:
        return np.zeros(gray.shape, np.int32)
    edges = np.percentile(toned, np.linspace(0.0, 100.0, n + 1)[1:-1])
    return np.clip(np.digitize(gray, edges), 0, n - 1).astype(np.int32)


def _band_plan(
    rng: np.random.Generator,
    *,
    n_bands: int | None,
    kind: str | None,
    angle: float | None,
    period: float,
    hatch_period: float,
) -> tuple[int, list[tuple[str, float, float]]]:
    """Seeded per-band screen plan: ``(n, [(kind, angle_deg, base_period), …])``.

    Consumes ``rng`` in the exact order the rendering loop needs (band
    count -> per-band patterns -> per-band angle then period), so a CPU and
    a GPU renderer fed the same seed pick the same structure."""
    n = max(1, n_bands if n_bands is not None else _pick_band_count(rng))
    kinds = [kind] * n if kind is not None else _band_kinds(n, rng)
    plan: list[tuple[str, float, float]] = []
    for i in range(n):
        ki = kinds[i]
        base = period if ki == "dot" else hatch_period
        ra = angle if angle is not None else float(rng.uniform(0.0, 90.0))
        # jitter period per band (n>1); range kept tight to avoid a coarse screen
        rp = base if n == 1 else base * float(rng.uniform(0.85, 1.3))
        plan.append((str(ki), ra, rp))
    return n, plan


def _render_tone(
    gray: np.ndarray,
    *,
    rng: np.random.Generator,
    period: float,
    white_cut: float,
    black_cut: float,
    hatch_period: float | None = None,
    ss: int = _TONE_SS,
    kind: str | None = None,
    angle: float | None = None,
    n_bands: int | None = None,
) -> np.ndarray:
    """Luminance-band screentone. float32 in [0,1]; ink = 0, paper = 1.

    Splits the toned value range into ``n_bands`` bands (random unless
    pinned), each screened with its own pattern/angle/period. Dots use
    ``period``; line/cross use the coarser ``hatch_period`` (defaults to 2x
    the dot period). Rendered at ``ss``x resolution then area-downscaled
    (anti-aliased gray, not 1-bit). ``kind``/``angle``/``n_bands`` pin those
    choices when given; otherwise drawn from the seeded ``rng``."""
    if hatch_period is None:
        hatch_period = period * 2.0
    h, w = gray.shape
    ss = max(1, int(ss))
    gs = (
        cv2.resize(gray, (w * ss, h * ss), interpolation=cv2.INTER_LINEAR)
        if ss > 1
        else gray
    )
    hs, ws = gs.shape
    n, plan = _band_plan(
        rng,
        n_bands=n_bands,
        kind=kind,
        angle=angle,
        period=period,
        hatch_period=hatch_period,
    )
    labels = _luma_band_labels(gs, black_cut=black_cut, white_cut=white_cut, n=n)
    tone = np.ones_like(gs)  # 1 = paper, 0 = ink
    for i, (ki, ra, rp) in enumerate(plan):
        screen = _screen_field(hs, ws, period=rp * ss, angle=ra, kind=ki)
        sel = labels == i
        tone[sel] = (gs[sel] >= screen[sel]).astype(np.float32)
    tone[gs >= white_cut] = 1.0
    tone[gs <= black_cut] = 0.0
    if ss > 1:  # area-average downscale → anti-aliased gray tone (kills moiré)
        tone = cv2.resize(tone, (w, h), interpolation=cv2.INTER_AREA)
    return tone


def resolve_params(
    img_rgb: np.ndarray, rng: np.random.Generator, overrides: dict
) -> dict:
    """Resolve all seeded/overridable knobs into a flat dict (backend-agnostic).

    Draws jitter from ``rng`` so any backend handed the same generator
    picks identical knobs. Returns scalars plus the raw ``overrides``
    passthrough for ``ss``/``tone_kind``/``angle``/``n_bands``."""
    # Three resolution curves, keyed on the short side: `scale` (linear) tracks
    # image size 1:1 for the dot pitch; `hatch_scale` (ratio**0.6) is gentler so
    # line/cross stripes don't balloon into black bands on large images;
    # `line_scale` (sqrt) keeps XDoG stroke width close to constant pixel width
    # rather than scaling 1:1. Floors keep all three off zero.
    short = float(min(img_rgb.shape[:2]))
    ratio = short / 900.0
    scale = max(0.5, ratio)
    hatch_scale = max(0.6, ratio**0.6)
    line_scale = max(0.6, ratio**0.5)
    return dict(
        sigma=overrides.get(
            "sigma", _XDOG_SIGMA * line_scale * float(rng.uniform(0.8, 1.25))
        ),
        k=overrides.get("k", _XDOG_K),
        sharp=overrides.get("sharp", _XDOG_SHARP * float(rng.uniform(0.8, 1.2))),
        eps=overrides.get("eps", _XDOG_EPS),
        phi=overrides.get("phi", _XDOG_PHI),
        period=overrides.get(
            "period", _TONE_PERIOD * scale * float(rng.uniform(0.9, 1.2))
        ),
        hatch_period=overrides.get(
            "hatch_period", _HATCH_PERIOD * hatch_scale * float(rng.uniform(0.9, 1.2))
        ),
        white_cut=overrides.get("white_cut", _TONE_WHITE_CUT),
        black_cut=overrides.get("black_cut", _TONE_BLACK_CUT),
        # jitter around BT.601 so the cond isn't a single fixed operator
        luma_weights=overrides.get(
            "luma_weights",
            (
                0.299 * float(rng.uniform(0.85, 1.15)),
                0.587 * float(rng.uniform(0.9, 1.1)),
                0.114 * float(rng.uniform(0.7, 1.3)),
            ),
        ),
        ss=int(overrides.get("ss", _TONE_SS)),
        tone_kind=overrides.get("tone_kind"),
        angle=overrides.get("angle"),
        n_bands=overrides.get("n_bands"),
        # shadow-detail lift is deterministic (no rng draw), so seed->structure holds
        detail_amount=float(overrides.get("detail_amount", _DETAIL_AMOUNT)),
        detail_sigma=float(overrides.get("detail_sigma", _DETAIL_SIGMA * scale)),
        detail_gate_lo=float(overrides.get("detail_gate_lo", _DETAIL_GATE_LO)),
        detail_gate_hi=float(overrides.get("detail_gate_hi", _DETAIL_GATE_HI)),
    )


def mangafy_array(
    img_rgb: np.ndarray,
    *,
    seed: int | None = None,
    **overrides,
) -> np.ndarray:
    """Color RGB uint8 ``(H, W, 3)`` → mangafied B&W RGB uint8 ``(H, W, 3)``.

    ``seed`` drives reproducible per-image jitter of the XDoG/screen knobs.
    Any knob can be pinned via ``overrides`` (see the module constants);
    pinning ``tone_kind``/``angle``/``n_bands`` forces a single fixed
    screen (e.g. ``n_bands=1, tone_kind="line"`` for QA)."""
    if img_rgb.ndim == 2:
        img_rgb = np.stack([img_rgb] * 3, axis=-1)
    img_rgb = img_rgb[:, :, :3]

    rng = np.random.default_rng(seed)
    p = resolve_params(img_rgb, rng, overrides)

    gray = _luminance(img_rgb, p["luma_weights"])
    line = _xdog(
        gray, sigma=p["sigma"], k=p["k"], sharp=p["sharp"], eps=p["eps"], phi=p["phi"]
    )
    tone_gray = _enhance_shadow_detail(
        gray,
        amount=p["detail_amount"],
        sigma=p["detail_sigma"],
        gate_lo=p["detail_gate_lo"],
        gate_hi=p["detail_gate_hi"],
    )
    tone = _render_tone(
        tone_gray,
        rng=rng,
        period=p["period"],
        hatch_period=p["hatch_period"],
        ss=p["ss"],
        white_cut=p["white_cut"],
        black_cut=p["black_cut"],
        kind=p["tone_kind"],
        angle=p["angle"],
        n_bands=p["n_bands"],
    )
    # ink wherever either the line or the screen is dark
    manga = np.minimum(line, tone)
    out = (np.clip(manga, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.stack([out] * 3, axis=-1)


def mangafy_pil(pil_image, *, seed: int | None = None, **overrides):
    """PIL convenience wrapper around :func:`mangafy_array`."""
    from PIL import Image

    arr = np.array(pil_image.convert("RGB"))
    return Image.fromarray(mangafy_array(arr, seed=seed, **overrides))
