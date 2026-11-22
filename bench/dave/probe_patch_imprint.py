#!/usr/bin/env python
"""DAVE Phase 3 — analytic dot-predictor: per-block patch-grid imprint probe.

Measures, per block, whether attenuating that block alone produces the
patch-grid **dot** artifact, for comparison against the shipped mask's hand-set
``block ≤ 18`` cap.

The mechanism (README Phase 2c): subtracting a block's per-channel spatial DC
makes the next block's LayerNorm/adaLN renormalize and **boost the AC residual**;
the highest spatial frequency the token stream carries is the per-patch grid
(period = ``spatial_patch_size`` in the latent → the latent Nyquist for p=2), so
the boosted HF aliases onto that grid as stippling. Late blocks imprint it
straight onto ``final_layer → unpatchify`` (no downstream attention left to
diffuse it); mid blocks let ~15 blocks of attention smear it out. So the analytic
signature of a dot-causing block is: **attenuating it alone concentrates the
output-latent perturbation at the patch-grid harmonic frequencies.**

What it does
------------
For a fixed seed, generate a baseline final latent, then re-generate with DAVE
attenuating **one block at a time** (a one-hot mask through the *real* intervention
path inside ``generate()`` — same τ-gate, same σ window, same dose as production).
The per-block perturbation is ``Δ_ℓ = lat_ℓ − lat_base``; we score it by

    imprint(ℓ) = power(Δ_ℓ) at the patch-grid harmonics  /  total power(Δ_ℓ)

(fraction in the periodic-at-patch-stride bands of the 2D spectrum, DC dropped).
A high fraction = the block moves the output mostly as a patch-grid pattern = a
dot-causer. A low fraction = it moves the output as broadband/low-freq
recomposition = safe. Averaged over a few seeds. We also report ``hf_frac`` (a
generic top-radius HF fraction, cross-check) and ``delta_energy`` (does the block
move the output at all). The verdict: which blocks self-exclude as dot-causers,
and whether they agree with the shipped ``block ≤ 18`` cap.

Read-only w.r.t. the model state (uses the production hook, removed each call).
Eager 5D forwards, compile off. Outputs ``result.json``, a per-block bar chart, a
spectrum montage, and ``imprint.npz``.

Usage
-----
    .venv/bin/python bench/dave/probe_patch_imprint.py                    # all blocks, 3 seeds
    .venv/bin/python bench/dave/probe_patch_imprint.py --seeds 5 --steps 28
    .venv/bin/python bench/dave/probe_patch_imprint.py --blocks 14 18 19 22 --decode 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from anima_lora import (
    GenerationRequest,
    decode_to_pil,
    default_checkpoints,
    generate,
    get_generation_settings,
    load_dit_model,
    load_vae,
    prepare_text_inputs,
)
from bench._anima import DEFAULT_NEG as DEFAULT_NEGATIVE
from bench._anima import DEFAULT_PROMPT
from bench._common import make_run_dir, write_result

# One-hot mask scratch file (in-repo so setup_dave's resolve_under_home finds it).
ONEHOT_REL = "output/tests/dave/_imprint_onehot.npz"


# --------------------------------------------------------------------------- #
# Latent shaping + spectral metric                                             #
# --------------------------------------------------------------------------- #
def _to_chw(latent: torch.Tensor) -> torch.Tensor:
    """Collapse a (B,C,[T,]H,W) latent to a single (C,H,W) float tensor.

    Per the DiT layout the temporal/frame axis is a singleton; we squeeze any
    size-1 leading/middle dims until 3 axes remain (C,H,W are all > 1).
    """
    x = latent.detach().float().cpu()
    while x.dim() > 3:
        # squeeze the first size-1 axis (batch, then the singleton frame dim).
        sq = next((d for d in range(x.dim()) if x.shape[d] == 1), None)
        if sq is None:
            x = x[0]  # no singleton left — drop the leading axis defensively
        else:
            x = x.squeeze(sq)
    return x


def _patch_grid_masks(H: int, Wf: int, W: int, p: int, tol_bins: float):
    """Boolean (H, Wf) mask of the patch-grid harmonic frequencies + a radial HF mask.

    The patch grid (period p in the latent) has spectral support along each axis at
    the harmonics f = k/p (k ≥ 1, up to Nyquist 0.5). The 2D dot lattice is "fy is a
    patch harmonic OR fx is a patch harmonic" — capturing horizontal/vertical patch
    banding and the (0.5, 0.5) checkerboard corner for p=2.
    """
    fy = torch.fft.fftfreq(H).abs()  # (H,) normalized [0, 0.5]
    fx = torch.fft.rfftfreq(W)  # (Wf,) normalized [0, 0.5]
    tol_y = tol_bins / H
    tol_x = tol_bins / W

    def axis_harmonics(f, tol):
        m = torch.zeros_like(f, dtype=torch.bool)
        k = 1
        while k / p <= 0.5 + 1e-6:
            m |= (f - k / p).abs() <= tol
            k += 1
        return m

    hy = axis_harmonics(fy, tol_y)
    hx = axis_harmonics(fx, tol_x)
    patch2d = hy[:, None] | hx[None, :]  # (H, Wf)
    rad = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)  # (H, Wf)
    hf = rad >= 0.25  # generic top-radius HF (cross-check)
    return patch2d, hf


def imprint_metrics(
    delta: torch.Tensor, p: int, tol_bins: float
) -> tuple[float, float, float]:
    """(patch-imprint fraction, generic-HF fraction, total Δ energy) for a (C,H,W) Δ."""
    C, H, W = delta.shape
    spec = torch.fft.rfft2(delta, norm="ortho")  # (C, H, W//2+1)
    power = (spec.real**2 + spec.imag**2).mean(0)  # (H, Wf) avg over channels
    power[0, 0] = 0.0  # drop DC (a constant shift is not a dot)
    total = float(power.sum().clamp_min(1e-20))
    patch2d, hf = _patch_grid_masks(H, power.shape[1], W, p, tol_bins)
    imprint = float(power[patch2d].sum()) / total
    hf_frac = float(power[hf].sum()) / total
    energy = float((delta * delta).sum())
    return imprint, hf_frac, energy


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE)
    p.add_argument("--seeds", type=int, default=3, help="number of seeds to average")
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument(
        "--size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W")
    )
    p.add_argument(
        "--strength", type=float, default=0.5, help="DAVE dose to test at (paper α=0.5)"
    )
    p.add_argument("--tau", type=float, default=0.15, help="DAVE temporal cutoff τ")
    p.add_argument(
        "--blocks",
        type=int,
        nargs="+",
        default=None,
        help="subset of block indices to test (default: all). Cheap dry-run lever.",
    )
    p.add_argument(
        "--tol_bins",
        type=float,
        default=1.5,
        help="patch-harmonic match tolerance in FFT bins (default 1.5).",
    )
    p.add_argument(
        "--cap",
        type=int,
        default=18,
        help="shipped mask block cap, for the agreement check (default 18).",
    )
    p.add_argument(
        "--decode",
        type=int,
        default=0,
        help="also VAE-decode the baseline + the N highest- and N lowest-imprint "
        "blocks to PNG for visual confirmation (default 0 = latent-only).",
    )
    p.add_argument("--label", default=None)
    opts = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    onehot_path = Path(__file__).resolve().parents[2] / ONEHOT_REL
    onehot_path.parent.mkdir(parents=True, exist_ok=True)

    ckpts = default_checkpoints()
    req = GenerationRequest(
        dit=ckpts.dit,
        vae=ckpts.vae,
        text_encoder=ckpts.text_encoder,
        prompt=opts.prompt,
        negative_prompt=opts.negative_prompt,
        save_path="output/tests/dave/_imprint_unused.png",
        infer_steps=opts.steps,
        guidance_scale=opts.cfg,
        image_size=tuple(opts.size),
        sampler="euler",
        flow_shift=3.0,
        seed=opts.seed0,
    )
    args = req.to_args()
    args.device = device
    args.compile = False  # eager 5D — the production hook fires around each block
    args.compile_blocks = True

    gen_settings = get_generation_settings(args)
    print("[imprint] loading DiT…")
    anima = load_dit_model(args, device, torch.bfloat16)
    num_blocks = len(anima.blocks)
    patch = int(getattr(anima, "spatial_patch_size", 2))
    blocks = opts.blocks if opts.blocks is not None else list(range(num_blocks))
    print(
        f"[imprint] {num_blocks} blocks (patch_size={patch}), testing {len(blocks)} "
        f"× {opts.seeds} seeds @ strength={opts.strength}, τ={opts.tau}, steps={opts.steps}"
    )
    print(
        f"[imprint] ≈{(len(blocks) + 1) * opts.seeds} generations — this is a heavy offline probe."
    )

    context, context_null = prepare_text_inputs(args, device, anima)
    text_data = {"context": context, "context_null": context_null}
    shared = {"model": anima}

    def gen(seed: int, block: int | None) -> torch.Tensor:
        """One generation's raw latent; block=None → baseline, else one-hot DAVE at block."""
        args.seed = seed
        args.dave_strength = opts.strength
        args.dave_tau = opts.tau
        args.dave_block_lo = 0
        args.dave_block_hi = -1
        args.dave_sigma_lo = 0.0
        args.dave_sigma_hi = 1.0
        if block is None:
            args.dave = None
        else:
            w = np.zeros(num_blocks, dtype=np.float32)
            w[block] = 1.0
            np.savez(onehot_path, weight=w)
            args.dave = ONEHOT_REL
        return generate(
            args, gen_settings, shared_models=shared, precomputed_text_data=text_data
        )

    # (len(blocks), seeds) per-metric matrices.
    n = len(blocks)
    imp = np.full((n, opts.seeds), np.nan)
    hff = np.full((n, opts.seeds), np.nan)
    den = np.full((n, opts.seeds), np.nan)
    base_lat: dict[int, torch.Tensor] = {}  # raw baseline latent (for decode)
    block_lat: dict[tuple[int, int], torch.Tensor] = {}  # raw (seed, block) for decode

    for si in range(opts.seeds):
        seed = opts.seed0 + si
        print(f"[imprint] seed {si + 1}/{opts.seeds} (seed={seed}) — baseline")
        base_raw = gen(seed, None)
        base = _to_chw(base_raw)
        if opts.decode:
            base_lat[seed] = base_raw
        for bi, bl in enumerate(blocks):
            print(f"[imprint]   block {bl}")
            lat_raw = gen(seed, bl)
            delta = _to_chw(lat_raw) - base
            imp[bi, si], hff[bi, si], den[bi, si] = imprint_metrics(
                delta, patch, opts.tol_bins
            )
            if opts.decode:
                block_lat[(seed, bl)] = lat_raw

    imprint = np.nanmean(imp, axis=1)  # (n,)
    hf_frac = np.nanmean(hff, axis=1)
    energy = np.nanmean(den, axis=1)

    # Dot-causers: blocks whose imprint fraction clears the knee. Use a robust
    # threshold (the gap between the safe-band median and the worst block); fall
    # back to mean+1·std. The point is the *ranking* + the agreement check.
    med = float(np.median(imprint))
    std = float(np.std(imprint))
    thr = max(med + std, 0.5 * (med + float(np.max(imprint))))
    dot_blocks = [int(blocks[i]) for i in range(n) if imprint[i] >= thr]
    safe_blocks = [int(blocks[i]) for i in range(n) if imprint[i] < thr]
    # Agreement with the shipped cap: are all flagged dot-blocks above the cap,
    # and are all at-or-below-cap blocks safe? (the cap's analytic justification)
    above_cap = [b for b in dot_blocks if b > opts.cap]
    cap_violations = [b for b in dot_blocks if b <= opts.cap]
    safe_above_cap = [b for b in safe_blocks if b > opts.cap]

    order = np.argsort(imprint)[::-1]
    print("\n" + "=" * 70)
    print(f"  patch_size={patch}  threshold={thr:.3f} (median {med:.3f})")
    print("  block | imprint | hf_frac | Δenergy   (sorted by imprint)")
    print("  " + "-" * 56)
    for i in order:
        flag = " DOT" if imprint[i] >= thr else ""
        print(
            f"  {blocks[i]:5d} | {imprint[i]:.4f}  | {hf_frac[i]:.4f}  | "
            f"{energy[i]:.3e}{flag}"
        )
    print("  " + "-" * 56)
    print(f"  dot-causers (imprint≥thr): {dot_blocks}")
    print(
        f"  shipped cap = {opts.cap}: "
        f"{'AGREES' if not cap_violations else 'DISAGREES'} "
        f"(flagged>cap={above_cap}, flagged≤cap={cap_violations}, safe>cap={safe_above_cap})"
    )
    print("=" * 70)

    # ---- plots ----
    artifacts: list[str] = []
    run_dir = make_run_dir("dave", label=opts.label or "imprint")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(7, n * 0.32), 4.2))
        colors = ["tab:red" if imprint[i] >= thr else "tab:blue" for i in range(n)]
        ax.bar([str(b) for b in blocks], imprint, color=colors)
        ax.axhline(thr, ls="--", c="gray", lw=1, label=f"dot threshold {thr:.2f}")
        ax.axvline(
            x=_cap_xpos(blocks, opts.cap),
            ls=":",
            c="green",
            lw=1.4,
            label=f"shipped cap ≤{opts.cap}",
        )
        ax.set_xlabel("block index")
        ax.set_ylabel("patch-grid imprint fraction")
        ax.set_title(
            f"DAVE per-block patch-imprint (strength={opts.strength}, τ={opts.tau}) — red = dot-causer"
        )
        ax.legend()
        fig.tight_layout()
        bar = run_dir / "imprint_per_block.png"
        fig.savefig(bar, dpi=130)
        plt.close(fig)
        artifacts.append(bar.name)
    except ImportError:
        print("[warn] matplotlib unavailable — skipping bar chart")

    # ---- optional VAE-decode for visual confirmation ----
    if opts.decode:
        seed = opts.seed0
        topN = [int(blocks[i]) for i in order[: opts.decode]]
        botN = [int(blocks[i]) for i in order[::-1][: opts.decode]]
        vae = load_vae(
            args.vae,
            device="cpu",
            disable_mmap=True,
            spatial_chunk_size=args.vae_chunk_size,
            disable_cache=args.vae_disable_cache,
            dtype=torch.bfloat16,
            eval=True,
        )
        dec_dir = run_dir / "decoded"
        dec_dir.mkdir(exist_ok=True)
        decode_to_pil(vae, base_lat[seed], device).save(dec_dir / "baseline.png")
        for tag, bls in (("dot", topN), ("safe", botN)):
            for bl in bls:
                img = decode_to_pil(vae, block_lat[(seed, bl)], device)
                img.save(
                    dec_dir / f"{tag}_blk{bl}_imp{imprint[blocks.index(bl)]:.3f}.png"
                )
        print(f"[imprint] decoded baseline + {tag}/dot/safe blocks → {dec_dir}")
        artifacts.append("decoded/")

    np.savez(
        run_dir / "imprint.npz",
        blocks=np.array(blocks, dtype=np.int64),
        imprint=imprint.astype(np.float32),
        hf_frac=hf_frac.astype(np.float32),
        energy=energy.astype(np.float32),
        imprint_per_seed=imp.astype(np.float32),
        patch_size=np.int64(patch),
        threshold=np.float32(thr),
        dot_blocks=np.array(dot_blocks, dtype=np.int64),
    )
    artifacts.append("imprint.npz")

    metrics = {
        "num_blocks": num_blocks,
        "patch_size": patch,
        "tested_blocks": [int(b) for b in blocks],
        "num_seeds": opts.seeds,
        "steps": opts.steps,
        "cfg": opts.cfg,
        "strength": opts.strength,
        "tau": opts.tau,
        "threshold": thr,
        "imprint_per_block": {int(blocks[i]): float(imprint[i]) for i in range(n)},
        "hf_frac_per_block": {int(blocks[i]): float(hf_frac[i]) for i in range(n)},
        "dot_blocks": dot_blocks,
        "safe_blocks": safe_blocks,
        "shipped_cap": opts.cap,
        "cap_agrees": not cap_violations,
        "flagged_above_cap": above_cap,
        "flagged_at_or_below_cap": cap_violations,
        "safe_above_cap": safe_above_cap,
    }
    write_result(
        run_dir,
        script=__file__,
        args=opts,
        metrics=metrics,
        label=opts.label or "imprint",
        artifacts=artifacts,
        device=device,
    )
    print(f"→ {run_dir}")


def _cap_xpos(blocks: list[int], cap: int) -> float:
    """x position (bar units) of the cap boundary for the axvline."""
    # bars are categorical at 0..n-1; place the line just past the last block ≤ cap.
    le = [i for i, b in enumerate(blocks) if b <= cap]
    return (max(le) + 0.5) if le else -0.5


if __name__ == "__main__":
    main()
