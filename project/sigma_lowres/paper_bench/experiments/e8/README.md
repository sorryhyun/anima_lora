# E8 — the gap-native restructure: ε\*, the guarantee region, and the null→gap bridge

| | |
|---|---|
| **Status** | **DONE** — E8.1 + E8.2 written into `main.tex` 2026-07-29; **E8.3 done 2026-07-31** |
| **Verdict (E8.3)** | The transported spectral family **fails in both directions at every tolerance**: it cannot produce the measured mid-σ structure (RMSE 0.147/0.355 on 768/512 vs our 0.093), and on each δ's committed region the floor persists. δ is *inert at the curve level*. |
| **Runs** | `runs/20260731-1939-e83-bridge/`; figure `paper/figs/e83_overlay.png` |
| **Scripts** | `e83_bridge.py` (CPU-only; imports [E5](../e5/)'s fit/paired-stats helpers) |
| **Depends on** | [E1](../e1/) (self-floors + per-bin SEM), [E4](../e4/)'s `reenc_noise_floor.py` run (`bench/results/20260730-0940-reenc-floor/`) |
| **In the paper** | §epsstar (Eq. epsstar / safe(ε)); §4.6 Fig. `fig:e83`. The **[pending]** overlay marker has been stripped — `fig:e83` ships in the appendix (`paper_suggestion/appendix.tex`). |

Origin: the gap-native restructure proposal (E8, 2026-07-28).

## E8.1 — ε\*, the minimum detectable gap [DONE 2026-07-29]

The instrument's detectability threshold as a function of (N, D, floor
cosine): below ε\*, a demotion is indistinguishable from redraw noise.
Sources: [E1](../e1/) self-floors (bias) + per-bin SEM (variance).
"Safe" ≡ one-sided 95% CI of the debiased gap below ε\* —
[E3](../e3/)'s non-inferiority criterion promoted from an ad-hoc
reenc±0.04 band to the *definition*. Landed as main.tex §epsstar
(Eq. epsstar / safe(ε)).

Renamed in the theory rewrite to **median certification resolution**,
with the power-adjusted margin ε\*\_{α,β} = b_D + (z\_{1−α}+z\_{1−β})·SE
in a footnote (at margin exactly 1.645·SE a truly-safe route certifies
with only ~50% power).

## E8.2 — the guarantee region [DONE 2026-07-29]

The safety map restated as the (route, σ) region where debiased gap ≤
ε\* at one-sided 95% — [E1](../e1/)(b) + [E3](../e3/) output verbatim.
Wording is "statistical non-inferiority at instrument resolution", never
a hard bound; the per-example vs batch-aggregate map split is retained
(E3/R3).

## E8.3 — null→gap bridge [DONE 2026-07-31]

Convert each published tolerance δ into a predicted *gap curve*, not
just a boundary: under the diagonal model on the measured P(f), compute
the destroyed-band Bayes-residual mismatch m_null,e(σ; δ), map through
the measured G(σ)^p with the route gain A calibrated on the one safe
route, and overlay predicted vs measured curves per (δ, route). This is
the Table-1 confrontation restated in gap units; it subsumes the
continuous t\*(δ) sweep figure (family spread ≤ 0.13) and the
δ_reenc-anchored row (`reenc_noise_floor.py`, ran 2026-07-30).

**Unit honesty:** the null emits residual units only — the bridge
(G^p, calibrated A) belongs to the two-term account, so the paper must
say "the null read through our bridge", which also makes §3 load-bearing
for the confrontation rather than decorative.

### Results

The transported family fails in both directions at every tolerance.

1. **In-window**, the model's destroyed-band mean-residual (diagonal
   Wiener on measured P(f)) is concentrated below σ ≈ 0.3 —
   high-frequency-quiet latents — so it cannot produce the measured
   mid-σ structure: RMSE 0.147 (768) / 0.355 (512) vs our account's
   0.093/0.093; **δ is inert at the curve level** (eq / 0.01 / δ_reenc
   all within 0.01 RMSE of each other — the gates land where the curve
   has already died).
2. **On each δ's committed region** (predicted gap ≡ 0) the floor
   persists: RMSE 0.027/0.049/0.218 for 896/768/512 at SPD's default.

Robust to the transport-shape choice (per-draw vs mean-residual ≤ 10%,
identical committed reads). Consistency: reproduces Table 1's t\* rows
exactly, [E5](../e5/)'s 768 committed-region 0.164, and the ≤ 0.13
family-spread claim (measured 0.125). Bridge convention: single gain
calibrated floorless on 1024→896; unit honesty stated in the figure
caption.
