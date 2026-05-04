#!/usr/bin/env python
"""Feasibility check: can a K-step probe classify a prompt as stable
(seed-reproducible) vs unstable (seed-noise-dominated), AND do c_pool
features correlate with per-prompt amplitude / seed-variance?

Background
----------
v2 ships per-aspect-bucket profiles only. The candidate v3 architecture
adds a prompt-conditioned residual head h(c_pool) → (δ̂_p, log σ̂²_p)
trained with Gaussian NLL — heteroscedastic regression. At inference,
the prediction is shrunk toward bucket prior by σ̂² / (σ̂² + σ²_target),
so seed-noise-dominated prompts are conservatively predicted near zero
without needing an explicit probe-based stability gate.

The 1406-small_invhd analysis showed:

- ICC on integrated per-prompt δ (1248×832, CFG=4) = 0.70 — prompt
  signal is well above seed noise on average.
- Per-prompt seed-stability is bimodal: 50% of prompts have SNR < 1
  (seed-noise-dominated); 50% have SNR > 1.5 (clear signal). Almost no
  middle.

For the heteroscedastic head to work, c_pool features must correlate
with **both** per-prompt amplitude AND per-prompt seed-variance. If
only amplitude correlates, σ̂ collapses to a constant and the model
reduces to a flat regressor.

Tests on a paired (P prompts × S=2 seeds × T steps) baseline artifact
(gap_<band> from gaps_per_sample.npz):

Test 1 — OFFLINE CALIBRATION QUESTION
    Early-window seed-pair diff² ↔ late-window seed-pair diff².
    "With 2 seeds at calibration, do K early steps rank-correlate with
    which prompts will be unstable?"

Test 2 — INFERENCE-TIME QUESTION
    Early-window single-seed distance from bucket mean ↔
    late-window seed-pair |diff|.
    "From 1 seed's K-step probe, can we predict the prompt's noisiness?"

Test 3 — PROBE-AS-AMPLITUDE-PREDICTOR (cross-check)
    Early-window single-seed |g - μ_bucket| ↔
    late-window |seed-mean(g) - μ_bucket|.
    Sign-flip-tolerant amplitude check.

Test 4 — C_POOL FEATURES vs per-prompt {amplitude, seed-variance}
    For each simple scalar c_pool feature (mean-pooled crossattn_emb_v0
    over tokens), compute Spearman r against signed amplitude, |amp|,
    seed variance σ²_p, and |seed-diff|. Heteroscedastic head needs
    both an amplitude signal AND a seed-variance signal.

Test 5 — BIMODAL SEPARATION
    For each c_pool feature, do stable (SNR>τ) and unstable (SNR<τ)
    prompts separate? Mann-Whitney U vs feature value.

Outputs
-------
Per-K table for each test (Pearson and Spearman r + p-value), per-prompt
SNR classification, c_pool feature correlations and group-separation
tests. No artifacts written — pure stdout. Pipe to a file:

    uv run python bench/dcw/stability_predictor_check.py \\
        --gaps_npz bench/dcw/results/20260504-1406-small_invhd/gaps_per_sample.npz \\
        --n_prompts 16 --n_seeds 2 \\
        --dataset_dir post_image_dataset/lora \\
        | tee bench/dcw/results/20260504-1406-small_invhd/stability_check.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

BANDS = ("LL", "LH", "HL", "HH")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--gaps_npz",
        type=str,
        required=True,
        help="Path to gaps_per_sample.npz from a baseline measure_bias.py run "
        "(--dump_per_sample_gaps). Must have S>=2 seeds for seed-pair tests.",
    )
    p.add_argument(
        "--n_prompts",
        type=int,
        required=True,
        help="Number of unique prompts in the run. Rows in gap_<band> are "
        "interleaved (prompt0_seed0, prompt0_seed1, prompt1_seed0, ...).",
    )
    p.add_argument(
        "--n_seeds",
        type=int,
        default=2,
        help="Seeds per prompt (default 2; tests assume exactly 2).",
    )
    p.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[2, 4, 6, 8],
        help="Probe-window lengths to test.",
    )
    p.add_argument(
        "--band",
        type=str,
        default="LL",
        choices=BANDS,
        help="Haar subband (default LL — the band v2/v3 corrects).",
    )
    p.add_argument(
        "--snr_threshold",
        type=float,
        default=1.0,
        help="Late-window SNR (|amp|/|seed_diff|) threshold for the "
        "stable/unstable split in the per-prompt summary.",
    )
    p.add_argument(
        "--dataset_dir",
        type=str,
        default="post_image_dataset/lora",
        help="Directory with cached *_anima_te.safetensors files. Used by "
        "Test 4/5 to load c_pool features. Pass empty '' to skip.",
    )
    p.add_argument(
        "--text_variant",
        type=int,
        default=0,
        help="crossattn_emb_v<N> variant (default 0 = canonical).",
    )
    return p.parse_args()


def load_c_pool_features(
    dataset_dir: Path, stems: list[str], text_variant: int
) -> dict[str, np.ndarray]:
    """Compute simple scalar features from per-prompt crossattn embeds.

    Reads ``<stem>_anima_te.safetensors`` from ``dataset_dir`` for each
    stem. Mean-pools ``crossattn_emb_v<N>`` across tokens to get one
    (D,) c_pool per prompt, then computes scalar features that don't
    require fitting a model (n=16 is far too small for that).

    Returned dict maps feature_name → (P,) float64 array.
    """
    from safetensors.torch import load_file

    embeds: list[np.ndarray] = []
    for stem in stems:
        path = dataset_dir / f"{stem}_anima_te.safetensors"
        if not path.exists():
            raise FileNotFoundError(f"missing text-emb cache: {path}")
        sd = load_file(str(path))
        key = f"crossattn_emb_v{text_variant}"
        if key not in sd:
            raise KeyError(
                f"{key} not in {path}. Available: "
                f"{[k for k in sd if k.startswith('crossattn_emb_')]}"
            )
        embeds.append(sd[key].float().numpy())  # (T, D)

    pools = np.stack([e.mean(axis=0) for e in embeds])  # (P, D)
    centroid = pools.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid) + 1e-9
    pool_norms = np.linalg.norm(pools, axis=1)

    feats: dict[str, np.ndarray] = {
        "||c_pool||_2": pool_norms,
        "||c_pool - μ_centroid||_2": np.linalg.norm(pools - centroid, axis=1),
        "cos(c_pool, μ_centroid)": (pools @ centroid) / (pool_norms * centroid_norm),
        # Token-level: how "spread out" is the prompt's content?
        "mean_token_l2": np.array([np.linalg.norm(e, axis=1).mean() for e in embeds]),
        "token_l2_std": np.array([np.linalg.norm(e, axis=1).std() for e in embeds]),
        # Effective non-padded token count (rough — captions of different
        # lengths give different effective T even if the tensor is 512-padded).
        "nonpad_tokens": np.array(
            [(np.linalg.norm(e, axis=1) > 0.05).sum() for e in embeds],
            dtype=np.float64,
        ),
    }
    return feats


def fmt_corr_row(K: int, rp: float, pp: float, rs: float, ps: float) -> str:
    return f"  {K:>3d}  {rp:>+8.3f}  {pp:>6.3f}  {rs:>+8.3f}  {ps:>6.3f}"


def main() -> None:
    args = parse_args()
    if args.n_seeds != 2:
        raise SystemExit("Tests assume n_seeds=2; rerun calibration with --n_seeds 2.")

    z = np.load(args.gaps_npz, allow_pickle=True)
    key = f"gap_{args.band}"
    if key not in z:
        raise SystemExit(
            f"{key} not in {args.gaps_npz}. Available: "
            f"{[k for k in z.files if k.startswith('gap_')]}"
        )
    g = z[key].astype(np.float64)  # (P*S, T)
    sigmas = z["sigmas"]
    stems = list(z["stems"]) if "stems" in z.files else None

    P, S = args.n_prompts, args.n_seeds
    if g.shape[0] != P * S:
        raise SystemExit(
            f"gap shape {g.shape} inconsistent with n_prompts={P}, n_seeds={S}"
        )

    G = g.reshape(P, S, -1)  # (P, S, T)
    T = G.shape[2]
    bucket_mean = G.mean(axis=(0, 1))  # (T,)
    seed_mean = G.mean(axis=1)  # (P, T)
    seed_diff = G[:, 0, :] - G[:, 1, :]  # (P, T)

    # ---- Header ----
    print("=" * 72)
    print(f"stability predictor feasibility — {args.band} band")
    print("=" * 72)
    print(f"  source     : {args.gaps_npz}")
    print(f"  shape      : P={P} prompts × S={S} seeds × T={T} steps")
    print(f"  bucket-mean integrated {args.band} gap : {bucket_mean.sum():+.1f}")
    print(f"  σ₀ = {float(sigmas[0]):.3f}, σₙ = {float(sigmas[-1]):.3f}")
    print()

    valid_ks = [K for K in args.ks if 1 <= K < T - 1]
    if not valid_ks:
        raise SystemExit(f"No valid K in {args.ks} for T={T}")

    # ---- Test 1: offline calibration question ----
    print("Test 1 — OFFLINE: early seed-pair diff² ↔ late seed-pair diff²")
    print("  question : with 2 seeds at calibration, do K early steps")
    print("             rank-correlate with which prompts will be unstable?")
    print(f"  {'K':>3s}  {'r_pear':>8s}  {'p':>6s}  {'r_spear':>8s}  {'p':>6s}")
    for K in valid_ks:
        early = (seed_diff[:, :K] ** 2).sum(axis=1)
        late = (seed_diff[:, K:] ** 2).sum(axis=1)
        rp, pp = pearsonr(early, late)
        rs, ps = spearmanr(early, late)
        print(fmt_corr_row(K, rp, pp, rs, ps))
    print()

    # ---- Test 2: inference-time question ----
    print(
        "Test 2 — INFERENCE-TIME: |early single-seed − bucket_mean| ↔ "
        "late seed-pair |diff|"
    )
    print("  question : from 1 seed's K-step probe, can we predict")
    print("             whether the prompt is seed-stable?")
    print(
        f"  {'K':>3s}  {'r_s0_p':>8s}  {'p':>6s}  {'r_s0_sp':>8s}  {'p':>6s}  "
        f"{'r_s1_sp':>8s}  {'p':>6s}"
    )
    late_diff_abs = np.abs(seed_diff[:, max(valid_ks) :]).sum(axis=1)  # placeholder
    for K in valid_ks:
        late = np.abs(seed_diff[:, K:]).sum(axis=1)
        rows = []
        for s in range(S):
            early_s = np.abs(G[:, s, :K] - bucket_mean[:K]).sum(axis=1)
            rs, ps = spearmanr(early_s, late)
            rows.append((rs, ps))
        early_s0 = np.abs(G[:, 0, :K] - bucket_mean[:K]).sum(axis=1)
        rp0, pp0 = pearsonr(early_s0, late)
        print(
            f"  {K:>3d}  {rp0:>+8.3f}  {pp0:>6.3f}  "
            f"{rows[0][0]:>+8.3f}  {rows[0][1]:>6.3f}  "
            f"{rows[1][0]:>+8.3f}  {rows[1][1]:>6.3f}"
        )
    print(
        "  (r_s0_p = Pearson on seed0; r_s0_sp / r_s1_sp = Spearman on each seed)"
    )
    print()

    # ---- Test 3: amplitude cross-check ----
    print(
        "Test 3 — AMPLITUDE: |early single-seed − bucket_mean| ↔ "
        "|late seed-mean − bucket_mean|"
    )
    print("  question : does early amplitude predict late amplitude")
    print("             (sign-flip-tolerant; uses |·|)?")
    print(f"  {'K':>3s}  {'r_pear':>8s}  {'p':>6s}  {'r_spear':>8s}  {'p':>6s}")
    for K in valid_ks:
        early = np.abs(G[:, 0, :K] - bucket_mean[:K]).sum(axis=1)
        late_amp = np.abs(seed_mean[:, K:] - bucket_mean[K:]).sum(axis=1)
        rp, pp = pearsonr(early, late_amp)
        rs, ps = spearmanr(early, late_amp)
        print(fmt_corr_row(K, rp, pp, rs, ps))
    print()

    # ---- Per-prompt SNR classification ----
    K_split = 4 if 4 in valid_ks else valid_ks[len(valid_ks) // 2]
    print(
        f"Per-prompt SNR (late-window K={K_split}+: "
        f"|seed-mean − bucket| / |seed-pair-diff|)"
    )
    print(f"  threshold = {args.snr_threshold:.2f}")
    if stems:
        late_amp = np.abs(seed_mean[:, K_split:] - bucket_mean[K_split:]).sum(axis=1)
        late_inst = np.abs(seed_diff[:, K_split:]).sum(axis=1)
        snr = late_amp / (late_inst + 1e-9)
        for p in range(P):
            tag = "stable  " if snr[p] >= args.snr_threshold else "unstable"
            print(f"  {stems[p * S]:>12s}  snr={snr[p]:>6.2f}  {tag}")
        n_stable = int((snr >= args.snr_threshold).sum())
        print(
            f"\n  stable: {n_stable}/{P} ({100 * n_stable / P:.0f}%)"
            f"  unstable: {P - n_stable}/{P} ({100 * (P - n_stable) / P:.0f}%)"
        )

        # ---- Test 1' — does the K=K_split probe identify the unstable subset? ----
        print()
        print(
            f"Test 1' — does early seed-pair diff² (K={K_split}) "
            f"separate stable vs unstable?"
        )
        early_inst = (seed_diff[:, :K_split] ** 2).sum(axis=1)
        stable_mask = snr >= args.snr_threshold
        if stable_mask.any() and (~stable_mask).any():
            mst = float(np.median(early_inst[stable_mask]))
            mun = float(np.median(early_inst[~stable_mask]))
            print(f"  median early seed-diff² | stable   = {mst:.1f}")
            print(f"  median early seed-diff² | unstable = {mun:.1f}")
            print(f"  ratio (unstable / stable)          = {mun / mst:.2f}")
        else:
            print("  classification too imbalanced for split.")

    # ---- Test 4: c_pool features vs per-prompt amplitude / seed-variance ----
    if stems and args.dataset_dir:
        unique_stems = [stems[p * S] for p in range(P)]
        try:
            feats = load_c_pool_features(
                Path(args.dataset_dir), unique_stems, args.text_variant
            )
        except (FileNotFoundError, KeyError) as e:
            print()
            print(f"Test 4 — c_pool features: SKIPPED ({e})")
            feats = None

        if feats is not None:
            integ_per_seed = (G - bucket_mean[None, None, :]).sum(axis=2)  # (P, S)
            y_signed = integ_per_seed.mean(axis=1)
            y_abs = np.abs(y_signed)
            sigma2_seed = integ_per_seed.var(axis=1, ddof=1)
            seed_diff_int_abs = np.abs(integ_per_seed[:, 0] - integ_per_seed[:, 1])

            targets: dict[str, np.ndarray] = {
                "δ_signed": y_signed,
                "|δ|     ": y_abs,
                "σ²_seed ": sigma2_seed,
                "|s0-s1| ": seed_diff_int_abs,
            }

            print()
            print("Test 4 — c_pool features ↔ per-prompt {amplitude, seed-noise}")
            print(
                "  question : do simple c_pool scalars correlate with per-prompt "
                "amplitude AND seed variance?"
            )
            print("  (heteroscedastic head needs both signals to be present)")
            print()
            header = (
                f"  {'feature':<28s}  "
                + "  ".join(f"{tn:>12s}" for tn in targets)
            )
            print(header)
            for fname, fvals in feats.items():
                cells = []
                for tn, tv in targets.items():
                    rs, ps = spearmanr(fvals, tv)
                    star = "*" if ps < 0.05 else " "
                    cells.append(f"{rs:>+5.2f}{star}({ps:>4.2f})")
                print(f"  {fname:<28s}  " + "  ".join(cells))
            print("  (cell = Spearman r * = p<0.05; (p))")

            # ---- Test 5: bimodal separation (Mann-Whitney U) ----
            from scipy.stats import mannwhitneyu

            print()
            print(
                f"Test 5 — does any c_pool feature separate stable (SNR≥"
                f"{args.snr_threshold}) from unstable?"
            )
            stable_mask = snr >= args.snr_threshold
            print(
                f"  groups: stable n={int(stable_mask.sum())}, "
                f"unstable n={int((~stable_mask).sum())}"
            )
            if stable_mask.any() and (~stable_mask).any():
                print(
                    f"  {'feature':<28s}  {'med_stable':>12s}  "
                    f"{'med_unstable':>14s}  {'U':>6s}  {'p':>6s}"
                )
                for fname, fvals in feats.items():
                    a = fvals[stable_mask]
                    b = fvals[~stable_mask]
                    u, p = mannwhitneyu(a, b, alternative="two-sided")
                    star = "*" if p < 0.05 else " "
                    print(
                        f"  {fname:<28s}  {np.median(a):>+12.3f}  "
                        f"{np.median(b):>+14.3f}  {u:>6.1f}  {p:>5.3f}{star}"
                    )

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
