"""The measurement kernel of ``run_sigma_probe.py`` — everything that touches
the model or the σ grid.

σ grids (:func:`build_sigmas`), the flat-gradient group map used by the Q2
J-decomposition (:func:`build_groups`), the two RoPE-alignment context
managers (:func:`pi_rope`, :func:`yarn_rope`), and the per-bin gradient
estimator itself (:func:`grad_estimate_binned`). No statistics live here —
those are in :mod:`.stats`.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from project.sigma_lowres.bench.tier_routing.run_grad_probe import (  # noqa: E402,F401
    DIT,
    VAE,
    cosine,
    encode_probe_latents,
    spearman,
)

log = logging.getLogger(__name__)


def enable_deterministic() -> None:
    """train.py-mirrored deterministic knobs. Must run before any CUDA/cublas
    init — kills the atomics-order run-to-run noise so runs sharing a warm
    inductor cache are bit-comparable (it does NOT pin the kernel set)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from networks import attention_dispatch

    attention_dispatch.set_deterministic(True)
    log.info("deterministic mode on (train.py-mirrored knobs)")


def bin_sigmas(bins: int, draws: int, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    """(bins, draws) σ grid: uniform bins on (lo, hi), stratified midpoints
    inside each bin. Uniform (not training-density) — per-bin means make the
    marginal density irrelevant, and σ is the axis under test. The window
    concentrates all bins in a sub-interval (crossover localization)."""
    b = torch.arange(bins, dtype=torch.float64).view(-1, 1)
    j = (torch.arange(draws, dtype=torch.float64) + 0.5).view(1, -1)
    u = (b + j / draws) / bins
    return (lo + (hi - lo) * u).to(torch.float32)


def build_sigmas(
    bins: int, draws: int, endpoint: bool, lo: float = 0.0, hi: float = 1.0
) -> torch.Tensor:
    """Uniform-bin grid over the (lo, hi) window, optionally with an exact
    σ=1.0 bin appended. ``--bins 0 --endpoint_bin`` gives an endpoint-only
    grid."""
    parts = []
    if bins > 0:
        parts.append(bin_sigmas(bins, draws, lo, hi))
    if endpoint:
        parts.append(torch.ones(1, draws, dtype=torch.float32))
    if not parts:
        raise SystemExit("need --bins > 0 and/or --endpoint_bin")
    return torch.cat(parts, dim=0)


GROUP_RE = re.compile(r"^lora_unet_blocks_(\d+)_(.+)$")


def build_groups(network) -> dict[str, list[tuple[int, int]]]:
    """Group -> flat-vector ranges (sorted-name order — must match
    ``grad_estimate_binned``'s flatten), at two levels: ``type:<module minus
    block prefix>`` and ``block:<idx>``.

    Fused projections additionally get row-block sub-groups on ``lora_up``
    (rows are contiguous in the row-major flatten): ``self_attn_qkv_proj`` →
    ``type:self_attn_up_{q,k,v}`` and ``cross_attn_kv_proj`` →
    ``type:cross_attn_up_{k,v}``. RoPE touches self-attn q/k only, so the
    q/k-vs-v contrast is the RoPE discriminator; ``lora_down`` is shared
    across the fused heads and stays only in the module-level group."""
    named = [(n, p) for n, p in sorted(network.named_parameters()) if p.requires_grad]
    groups: dict[str, list[tuple[int, int]]] = {}
    pos = 0
    for name, p in named:
        s, e = pos, pos + p.numel()
        pos = e
        m = GROUP_RE.match(name.split(".")[0])
        keys = (
            (f"type:{m.group(2)}", f"block:{int(m.group(1)):02d}")
            if m
            else ("type:other", "block:other")
        )
        for k in keys:
            groups.setdefault(k, []).append((s, e))
        if m and name.endswith(".lora_up.weight"):
            typ = m.group(2)
            if typ == "self_attn_qkv_proj":
                third = p.numel() // 3  # rows are [q; k; v] blocks
                for j, sub in enumerate(("q", "k", "v")):
                    groups.setdefault(f"type:self_attn_up_{sub}", []).append(
                        (s + j * third, s + (j + 1) * third)
                    )
            elif typ == "cross_attn_kv_proj":
                half = p.numel() // 2  # rows are [k; v] blocks
                for j, sub in enumerate(("k", "v")):
                    groups.setdefault(f"type:cross_attn_up_{sub}", []).append(
                        (s + j * half, s + (j + 1) * half)
                    )
    return groups


def grouped_cosine(
    a: torch.Tensor,
    b: torch.Tensor,
    groups: dict[str, list[tuple[int, int]]],
) -> dict[str, float]:
    """Per-group cosine of two flat gradient vectors over each group's
    flat-vector ranges."""
    out = {}
    for g, ranges in groups.items():
        d = na = nb = 0.0
        for s, e in ranges:
            va, vb = a[s:e], b[s:e]
            d += float(va.dot(vb))
            na += float(va.dot(va))
            nb += float(vb.dot(vb))
        out[g] = d / (na**0.5 * nb**0.5) if na > 0 and nb > 0 else 0.0
    return out


@contextmanager
def pi_rope(anima, h_scale: float, w_scale: float):
    """PI-stretch the main-stream RoPE for the duration: every spatial patch
    ``i`` sits at fractional position ``i * scale`` (EasyControl's
    ``generate_embeddings_scaled`` — exact at fractional positions, distinct
    cache key). Instance-level ``forward`` patch on the pos_embedder; RoPE is
    built OUTSIDE the compiled block graph (``prepare_embedded_sequence``), so
    there is no dynamo interaction — the blocks just receive different cos/sin
    input tensors at the same token count. Scaled cache entries are dropped on
    exit (per-image scales would otherwise accrete VRAM across the run)."""
    pe = anima.pos_embedder

    def fwd(x_B_T_H_W_C, fps=None):
        return pe.generate_embeddings_scaled(
            x_B_T_H_W_C.shape, h_scale=h_scale, w_scale=w_scale, fps=fps
        )

    pe.forward = fwd
    try:
        yield
    finally:
        del pe.forward  # restore the class-level forward
        for k in [k for k in pe._cos_sin_cache if k and k[0] == "scaled"]:
            del pe._cos_sin_cache[k]


@contextmanager
def yarn_rope(
    anima,
    h_scale: float,
    w_scale: float,
    alpha: float,
    beta: float,
    gate: tuple[float, float] | None = None,
):
    """Frequency-banded position alignment (YaRN/NTK-by-parts style).

    Per spatial RoPE frequency ``f_d`` (rad/patch), the rotation count across
    the demoted extent is ``r_d = N * f_d / 2π``. Bands with ``r_d < alpha``
    (global-extent carriers) get the full PI stretch toward native
    coordinates; ``r_d > beta`` (local content-precision carriers) keep the
    native integer spacing; linear ramp between. Implemented as a per-dim
    frequency rescale (phase ``i·f_d·s_d`` ≡ position scale per band), built
    probe-locally from the pos_embedder's own buffers — models.py untouched.
    Same instance-``forward`` patch + no-dynamo-interaction reasoning as
    ``pi_rope``; local per-shape cache, dropped on exit.

    With ``gate=(center, gamma)`` (the ``<edge>yarnsig`` arm), both band
    thresholds are scaled by ``μ(σ) = sigmoid(gamma·[logit(σ) −
    logit(center)])`` — SigMa's dynamic boundary gating (Eq. 21) adapted to
    demotion: at low σ the thresholds shrink toward 0, every band lands above
    ``beta·μ`` and keeps native spacing (intervention off); μ→1 recovers the
    static yarn arm. Yields a handle with ``set_sigma`` that the estimator
    calls per draw; cos/sin cache keyed on (shape, μ)."""
    from einops import repeat as _repeat

    pe = anima.pos_embedder
    cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
    state = {"mu": 1.0}

    def s_vec(freqs: torch.Tensor, n: int, scale: float) -> torch.Tensor:
        mu = state["mu"]
        if mu < 1e-9:
            return torch.ones_like(freqs)
        r = n * freqs / (2 * math.pi)
        g = ((r - alpha * mu) / ((beta - alpha) * mu)).clamp(0.0, 1.0)
        return (1.0 - g) * scale + g

    def fwd(x_B_T_H_W_C, fps=None):
        B, T, H, W, _ = x_B_T_H_W_C.shape
        key = (T, H, W, round(state["mu"], 6))
        hit = cache.get(key)
        if hit is not None:
            return hit
        h_freqs = 1.0 / ((10000.0 * pe.h_ntk_factor) ** pe.dim_spatial_range)
        w_freqs = 1.0 / ((10000.0 * pe.w_ntk_factor) ** pe.dim_spatial_range)
        t_freqs = 1.0 / ((10000.0 * pe.t_ntk_factor) ** pe.dim_temporal_range)
        half_h = torch.outer(pe.seq[:H], h_freqs * s_vec(h_freqs, H, h_scale))
        half_w = torch.outer(pe.seq[:W], w_freqs * s_vec(w_freqs, W, w_scale))
        half_t = torch.outer(pe.seq[:T], t_freqs)
        em = torch.cat(
            [
                _repeat(half_t, "t d -> t h w d", h=H, w=W),
                _repeat(half_h, "h d -> t h w d", t=T, w=W),
                _repeat(half_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )
        freqs = em.flatten(0, 2).unsqueeze(1).unsqueeze(1).float()
        out = (torch.cos(freqs), torch.sin(freqs))
        cache[key] = out
        return out

    handle = None
    if gate is not None:
        center, gamma = gate
        logit_c = math.log(center / (1.0 - center))

        def set_sigma(sigma: float) -> None:
            s = min(max(sigma, 1e-6), 1.0 - 1e-6)
            state["mu"] = 1.0 / (
                1.0 + math.exp(-gamma * (math.log(s / (1.0 - s)) - logit_c))
            )

        handle = argparse.Namespace(set_sigma=set_sigma)
    pe.forward = fwd
    try:
        yield handle
    finally:
        del pe.forward
        cache.clear()


def build_probe_bundle(args, probe, extra_latents):
    """Load DiT + adapter and compile the block graph over exactly the token
    counts this run touches (native probe tiers + every demoted/derived grid).
    compile-after-apply: ``build_anima`` attaches the adapter first."""
    from library.runtime.harness import build_anima, compile_blocks_for_training

    args.gradient_checkpointing = args.grad_ckpt
    args.compile = False  # compile is wired below (needs dynamic-seq marks)
    bundle = build_anima(args, dit_path=args.dit, adapter=args.adapter, train_mode=True)

    if not args.grad_ckpt:
        counts = {r.tokens for r in probe}
        counts.update(
            (t.shape[-2] // 2) * (t.shape[-1] // 2) for t in extra_latents.values()
        )
        compile_blocks_for_training(
            bundle.anima,
            bundle.network,
            backend="inductor",
            mode=None,
            n_token_families=len(counts),
            seq_range=(min(counts), max(counts)),
            dynamic_seq=True,
            activation_memory_budget=args.activation_memory_budget,
            partitioner_aggressive_recomputation=True,
            grad_ckpt=False,
        )
    return bundle


def grad_estimate_binned(
    bundle,
    latents: torch.Tensor,
    crossattn: torch.Tensor,
    sigmas: torch.Tensor,  # (bins, draws)
    seeds: list[int],  # len == bins * draws
    rope_patch=None,  # no-arg callable returning a rope-patch context manager
    prefix_draws: list[int] | None = None,
    batch_draws: int = 1,
    target_alpha: float = 1.0,
) -> tuple[list[torch.Tensor], list[float]]:
    """Per-σ-bin accumulated-gradient estimates.

    Returns ``(vecs, norms)``: per bin, the flattened LoRA gradient summed
    over that bin's draws (float32, CPU) and its L2 norm. Same forward/
    backward cost as the pooled 3a estimator at equal total draws — only the
    accumulator is split.

    ``prefix_draws`` (draw-sweep mode, single-bin grids only): instead of one
    vector per bin, snapshot the accumulating gradient after each listed draw
    count — nested-seed estimates at D ∈ prefix_draws from one pass; the
    returned lists are indexed by prefix.

    ``batch_draws`` > 1 runs that many draws per forward/backward: per-draw
    noise is generated from the same per-draw seeds and stacked, σ rides the
    batch axis, and the loss is ``B * mse_mean`` — exactly the accumulated
    sum of per-draw mean losses (same-shape elements), so the resulting bin
    gradient is unchanged up to float reduction order. Chunks are clamped at
    prefix-snapshot boundaries; a σ-gated rope arm falls back to per-draw
    stepping inside any chunk whose σ values differ.
    """
    device = bundle.device
    params = [
        p for _, p in sorted(bundle.network.named_parameters()) if p.requires_grad
    ]
    lat = latents.unsqueeze(0).to(device)  # (1, 16, H, W) float32
    pad = torch.zeros(
        1, 1, lat.shape[-2], lat.shape[-1], dtype=torch.bfloat16, device=device
    )
    vecs: list[torch.Tensor] = []
    norms: list[float] = []
    n_bins, n_draws = sigmas.shape
    snap = set(prefix_draws) if prefix_draws else None
    if snap is not None:
        assert n_bins == 1, "prefix_draws needs a single-bin (endpoint-only) grid"
        assert max(snap) == n_draws, "largest prefix must equal draws_per_bin"

    def flat_grad() -> torch.Tensor:
        return torch.cat([p.grad.detach().float().flatten().cpu() for p in params])

    ctx = rope_patch() if rope_patch else nullcontext()
    with ctx as rope_handle:
        for b in range(n_bins):
            for p in params:
                p.grad = None
            j = 0
            while j < n_draws:
                nb = min(batch_draws, n_draws - j)
                if snap is not None:  # never straddle a snapshot boundary
                    nb = min(nb, min(s for s in snap if s > j) - j)
                sig = sigmas[b, j : j + nb]
                if rope_handle is not None and nb > 1 and len(set(sig.tolist())) > 1:
                    nb, sig = 1, sigmas[b, j : j + 1]  # σ-gated rope: uniform σ only
                noise = torch.cat(
                    [
                        torch.randn(
                            lat.shape,
                            generator=torch.Generator(device=device).manual_seed(
                                seeds[b * n_draws + j + k]
                            ),
                            device=device,
                            dtype=lat.dtype,
                        )
                        for k in range(nb)
                    ],
                    dim=0,
                )
                sigma_b = sig.to(device)  # (nb,)
                if rope_handle is not None:  # σ-gated rope (yarnsig arm)
                    rope_handle.set_sigma(float(sig[0]))
                sview = sigma_b.view(-1, 1, 1, 1)
                noisy = (1.0 - sview) * lat + sview * noise
                # target_alpha scales the image in the TARGET only (input
                # untouched); 1.0 * lat is exact, so the default is
                # bit-identical to the historical target
                target = noise - target_alpha * lat
                noisy_5d = noisy.unsqueeze(2).to(torch.bfloat16)
                # nb == 1 must hand the compiled blocks the EXACT tensors the
                # pre-batching code did — an expand() view changes strides,
                # which changes dynamo guards → re-autotuned kernels → fp
                # drift vs. earlier runs' cached graphs
                ca = crossattn if nb == 1 else crossattn.expand(nb, -1, -1)
                pm = pad if nb == 1 else pad.expand(nb, -1, -1, -1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = bundle.anima(noisy_5d, sigma_b, ca, padding_mask=pm)
                pred = pred.squeeze(2).float()
                loss = torch.nn.functional.mse_loss(pred, target)
                if nb > 1:  # recover the accumulated sum of per-draw means
                    loss = loss * nb
                loss.backward()
                j += nb
                if snap is not None and j in snap:
                    vec = flat_grad()
                    vecs.append(vec)
                    norms.append(float(vec.norm()))
            if snap is None:
                vec = flat_grad()
                vecs.append(vec)
                norms.append(float(vec.norm()))
    for p in params:
        p.grad = None
    return vecs, norms
