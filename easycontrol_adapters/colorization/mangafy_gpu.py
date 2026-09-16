"""GPU (torch) backend for :mod:`mangafy` — same screentone, on the device.

Reimplements the CPU-bound parts of the ``cv2``/``numpy`` mangafier (the
supersampled per-band screen fields and the two XDoG blurs) in torch.

It reuses :func:`mangafy.resolve_params` and :func:`mangafy._band_plan`
**verbatim**, fed the same NumPy ``default_rng(seed)``. So the *structure* of every
page — band count, which pattern (dot/line/cross) lands on which value band, each
band's angle/period, the XDoG knobs, the luma weights — is bit-identical to the CPU
path; only the sub-pixel rendering (Gaussian kernel, bilinear up / area down, float
trig) differs, the same way two image libraries differ. Output is the same
3-channel grayscale RGB uint8 the colorization cond expects.

cv2 parity notes:
  * Gaussian: separable conv with a radius-``round(4σ)`` kernel and ``reflect``
    padding (cv2's ``BORDER_REFLECT_101``) ≈ ``cv2.GaussianBlur(ksize=0)`` on float.
  * supersample resolve: ``bilinear`` up then ``area`` down == ``cv2.INTER_LINEAR`` /
    ``cv2.INTER_AREA`` (area = exact ss×ss box mean at integer ratio).
  * band labels: ``torch.quantile`` (linear) + ``bucketize(right=True)`` mirror
    ``np.percentile`` + ``np.digitize`` (only exactly-on-edge pixels can differ).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from mangafy import _band_plan, _enhance_shadow_detail, resolve_params


def _gaussian_blur_t(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian on a 2-D ``(H, W)`` tensor; ≈ ``cv2.GaussianBlur(ksize=0)``."""
    r = max(1, int(round(4.0 * float(sigma))))
    xs = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(xs * xs) / (2.0 * sigma * sigma))
    k = k / k.sum()
    x = x[None, None]
    x = F.pad(x, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, -1, 1))
    return x[0, 0]


def _xdog_t(
    gray: torch.Tensor, *, sigma: float, k: float, sharp: float, eps: float, phi: float
) -> torch.Tensor:
    """Winnemöller XDoG on the GPU. float in [0, 1]; ink ≈ 0, paper ≈ 1."""
    g1 = _gaussian_blur_t(gray, sigma)
    g2 = _gaussian_blur_t(gray, sigma * k)
    dog = (1.0 + sharp) * g1 - sharp * g2
    out = torch.ones_like(dog)
    soft = 1.0 + torch.tanh(phi * (dog - eps))
    return torch.where(dog < eps, soft, out).clamp_(0.0, 1.0)


def _uniformize_t(field: torch.Tensor) -> torch.Tensor:
    """Rank-normalize a screen field to uniform (0, 1) — torch port of
    :func:`mangafy._uniformize`. Makes ink coverage track luminance for every
    pattern (``cross`` is mean ~0.70 raw and would over-ink bright skin into black
    blotches). Monotonic, so the texture is preserved; only coverage is corrected."""
    flat = field.reshape(-1)
    n = flat.numel()
    order = torch.argsort(flat)
    ranks = torch.empty_like(flat)
    ranks[order] = (torch.arange(n, device=field.device, dtype=field.dtype) + 0.5) / n
    return ranks.reshape(field.shape)


def _screen_field_t(
    h: int, w: int, *, period: float, angle: float, kind: str, device, dtype
) -> torch.Tensor:
    """Periodic screen field in [0, 1] — torch port of :func:`mangafy._screen_field`.

    Rank-normalized to a uniform CDF (:func:`_uniformize_t`) so ink coverage equals
    ``1 - luminance`` for every pattern, matching the NumPy backend."""
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    a = math.radians(angle)
    xr = xs * math.cos(a) - ys * math.sin(a)
    yr = xs * math.sin(a) + ys * math.cos(a)
    wx = (2.0 * math.pi / period) * xr
    if kind == "line":
        field = (torch.sin(wx) + 1.0) * 0.5
    else:
        wy = (2.0 * math.pi / period) * yr
        if kind == "cross":
            field = torch.maximum(
                (torch.sin(wx) + 1.0) * 0.5, (torch.sin(wy) + 1.0) * 0.5
            )
        else:
            field = (torch.sin(wx) * torch.sin(wy) + 1.0) * 0.5  # "dot"
    return _uniformize_t(field)


def _quantile_sorted(x: torch.Tensor, qs: torch.Tensor) -> torch.Tensor:
    """Linear-interpolation quantiles via ``sort`` — matches ``np.percentile`` and,
    unlike ``torch.quantile``, has no ~16M-element input cap (a supersampled page's
    toned-pixel count overflows it)."""
    s, _ = torch.sort(x)
    pos = qs * (s.numel() - 1)
    lo = pos.floor().long()
    hi = pos.ceil().long()
    frac = pos - lo.to(pos.dtype)
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _luma_band_labels_t(
    gray: torch.Tensor, *, black_cut: float, white_cut: float, n: int
) -> torch.Tensor:
    """Per-pixel value-band index in ``[0, n)`` — torch port of the NumPy labeller."""
    if n <= 1:
        return torch.zeros_like(gray, dtype=torch.long)
    toned = gray[(gray > black_cut) & (gray < white_cut)]
    if toned.numel() == 0:
        return torch.zeros_like(gray, dtype=torch.long)
    qs = torch.linspace(0.0, 1.0, n + 1, device=gray.device, dtype=gray.dtype)[1:-1]
    edges = _quantile_sorted(toned, qs)
    # np.digitize(..., right=False) == torch.bucketize(..., right=True)
    return torch.bucketize(gray, edges, right=True).clamp_(0, n - 1)


def _render_tone_t(
    gray: torch.Tensor,
    *,
    rng: np.random.Generator,
    period: float,
    hatch_period: float,
    ss: int,
    white_cut: float,
    black_cut: float,
    kind: str | None,
    angle: float | None,
    n_bands: int | None,
) -> torch.Tensor:
    """Luminance-band screentone on the GPU; float in [0, 1], ink = 0, paper = 1."""
    h, w = gray.shape
    ss = max(1, int(ss))
    gs = (
        F.interpolate(
            gray[None, None],
            size=(h * ss, w * ss),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
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
    labels = _luma_band_labels_t(gs, black_cut=black_cut, white_cut=white_cut, n=n)
    tone = torch.ones_like(gs)  # 1 = paper, 0 = ink
    for i, (ki, ra, rp) in enumerate(plan):
        screen = _screen_field_t(
            hs, ws, period=rp * ss, angle=ra, kind=ki, device=gs.device, dtype=gs.dtype
        )
        tone = torch.where(labels == i, (gs >= screen).to(tone.dtype), tone)
    tone = torch.where(gs >= white_cut, torch.ones_like(tone), tone)
    tone = torch.where(gs <= black_cut, torch.zeros_like(tone), tone)
    if ss > 1:  # area-average downscale → anti-aliased gray tone (kills moiré)
        tone = F.interpolate(tone[None, None], size=(h, w), mode="area")[0, 0]
    return tone


def mangafy_array_gpu(
    img_rgb: np.ndarray,
    *,
    seed: int | None = None,
    device: "torch.device | str | None" = None,
    **overrides,
) -> np.ndarray:
    """GPU twin of :func:`mangafy.mangafy_array` — same args, same output contract.

    Color RGB uint8 ``(H, W, 3)`` → mangafied B&W RGB uint8 ``(H, W, 3)``. Falls
    back to CPU torch if no CUDA device is available (still vectorized, just not on
    the GPU). Structurally identical to the NumPy backend for a given ``seed``."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    if img_rgb.ndim == 2:
        img_rgb = np.stack([img_rgb] * 3, axis=-1)
    img_rgb = img_rgb[:, :, :3]

    rng = np.random.default_rng(seed)
    p = resolve_params(img_rgb, rng, overrides)

    rgb = (
        torch.from_numpy(np.ascontiguousarray(img_rgb)).to(
            device=device, dtype=torch.float32
        )
        / 255.0
    )
    wts = torch.tensor(p["luma_weights"], device=device, dtype=torch.float32)
    gray = (rgb @ (wts / wts.sum())).clamp_(0.0, 1.0)  # (H, W)

    line = _xdog_t(
        gray, sigma=p["sigma"], k=p["k"], sharp=p["sharp"], eps=p["eps"], phi=p["phi"]
    )
    # Shadow-detail lift runs on the native-res luminance via the shared cv2 helper
    # (cheap host-side op, not supersampled), so it's bit-identical to the CPU backend.
    tone_gray = gray
    if p["detail_amount"] > 0:
        enhanced = _enhance_shadow_detail(
            gray.detach().cpu().numpy(),
            amount=p["detail_amount"],
            sigma=p["detail_sigma"],
            gate_lo=p["detail_gate_lo"],
            gate_hi=p["detail_gate_hi"],
        )
        tone_gray = torch.from_numpy(enhanced).to(device=gray.device, dtype=gray.dtype)
    tone = _render_tone_t(
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
    # ink wherever either the line or the screen is dark; truncate to uint8 (matches NumPy)
    manga = torch.minimum(line, tone).clamp_(0.0, 1.0)
    out = (manga * 255.0).to(torch.uint8).cpu().numpy()
    return np.stack([out] * 3, axis=-1)
