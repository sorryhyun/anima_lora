# Trajectory-resolved latent statistics — anime-domain commitment is front-loaded and lives in a ~3-dim channel subspace, but aggregate structure never converts into a per-token/per-image intervention basis

> **LINE CLOSED (2026-07-24).** The proposal is archived
> (`_archive/proposals/traj_latent_stats.md`, originally PR #74); the
> `--traj_stats` recorder (`library/inference/traj_stats.py`) stays shipped.
> The bench (atlas, subspace analysis, trajectory-intactness gauge) was
> briefly archived, then re-adopted 2026-07-24 into
> `_archive/sigma_lowres/bench/traj_stats/` as that line's observability
> harness (oracle-replay sidecars first for any new σ-conditioned
> intervention). Run-by-run evidence:
> `_archive/sigma_lowres/bench/traj_stats/report.md` (Phases 0–3) and
> `_archive/sigma_lowres/bench/tier_routing/report.md` (Phase 3a). All four
> Phase 3 intervention candidates are dead: tier routing (3a probe), committed-token
> compute reuse (oracle replay), channel truncation (no compute target), decode
> probe (closed unrun — payoff bounded to ~3 GB of latent caches).

## The question

The anime domain is latent-sparse (flat cel shading, line art, uniform
backgrounds), so static corpus statistics scream redundancy — but the one
intervention previously built on redundancy signals, the deferred-foveated
merge, died because its damage lived in the *process*, not the endpoint
([[project_foveated_denoise_p0]]: the periphery blur was constitutive, P4t).
This line inverted the order: record per-step / per-token / per-channel
statistics of x̂₀ = z − σ·v during generation (bit-exact, 1.35 % overhead
at 1024²), build the domain atlas, build a process-intactness gauge, and only
then audition efficiency interventions against it.

## Finding 1 — token commitment is front-loaded, tightly and consistently

Aggregate over 32 generation traces (8 prompts × 4 seeds, 1024², 28-step
er_sde CFG 4) and 8 real-image DirectEdit inversions:

| σ | 0.86 | 0.80 | 0.625 | 0.50 | 0.333 |
|---|---|---|---|---|---|
| commit-CDF, generation | 0.12 | 0.18 | 0.34 | 0.47 | 0.65 |
| commit-CDF, inversion | 0.06 | 0.11 | 0.30 | 0.48 | 0.74 |

~Half of all tokens take their final k=4 quantization code by σ=0.5, ~two
thirds by σ=1/3, with tight cross-seed/prompt IQR. The activity side agrees
(active-token fraction 1.00 → 0.29 → 0.12 over σ = 1 → 0.50 → 0.26; shape
τ-robust), and effective guidance is even more front-loaded (post-combine
‖v_final − v_uncond‖ falls 2.14 → 0.44 → 0.26 by σ=0.80/0.33), consistent
with the cross-attn front-loading finding
([crossattn_self_attn_dominance.md](crossattn_self_attn_dominance.md)).
Sparse prompts commit earlier (less text signal to integrate). This is the
trajectory-side companion of the σ resolution staircase
(`sigma_signal_where_anima_resolves.md`).

## Finding 2 — the domain uses a fixed ~3-dim slice of the 16-dim VAE channel space, at every σ

Per-channel code entropy spans ~4× (high: 13, 4, 11, 15, 5; near-idle: 8,
14, 6, 10), and the marginal skew is the axis-aligned shadow of a subspace
fact: channel-covariance effective rank of token-level x̂₀ is 2.7–3.4 / 16
(top-4 directions ≈ 90–94 % var), and the subspace is the same one —
principal angles ≈ 1 through the eighth direction — for the static corpus,
final generations, and mid-trajectory x̂₀ at σ=0.5. Inference never leaves
the corpus subspace, and the channel profile is frozen from σ≈0.92 down:
channels don't take turns, the domain just uses a fixed subset hard.
(No compute lever hides here: the 16 channels mix into 1024-dim tokens at
patch embed. Whether decode quality survives projection to the subspace
was never tested — that probe closed unrun.)

## Finding 3 — generation and inversion are statistically the same process below σ≈0.92

Commit-CDF max gap 0.086, per-knot channel-profile Pearson ≥ 0.89 below
σ=0.92, final cbits Spearman 0.94–0.98 against the static corpus column.
Only the σ→1 knots diverge, mechanically (prior guess from noise vs
destroyed-image estimate). So corpus statistics license generation-side
claims, not just img2img ones. Side result with standing value: the
generated corpus is measurably smoother than the training corpus (final
token-Laplacian hf 0.13 vs 0.20 under identical normalization) — a usable
baseline for any future "does X restore real-image texture" question.

## Finding 4 — process-intactness is measurable, and quality-neutral ≠ process-transparent

The gauge (per-σ divergence curves; verdicts from *distributional* metrics
only) rediscovered P4t from traces alone — foveation flagged process-broken
via a commit-CDF hole (0.17) plus an in-loop hf blow-up (~30×, the
blocky group-shared periphery), not the predicted flatline, which only
exists after the bicubic readout. Calibration's two interpretation rules
outlive the line:

1. Quality-neutral ≠ process-transparent. Spectrum passes its own
   quality benches yet its forecast steps are process-visible (ΔE −0.65 on
   forecast knots — information genuinely doesn't flow through the DiT
   there). "Perturbed" is the correct reading; bands were not re-tuned to
   flatter a shipped method.
2. Pointwise divergence is chaos, not damage. SMC-CFG reaches 65 %
   token code mismatch through ordinary trajectory divergence while every
   process statistic stays intact. Never gate on pointwise x̂₀ RMSE.

## Finding 5 — the closure: aggregate structure ≠ intervention basis (failed twice, independently)

Both exploitation attempts died at cheap pre-quality gates, with the same
shape:

- Training-side (tier routing, Phase 3a): demote-one-tier gradient cost
  is real (gap 0.074–0.147 vs re-encode control ≈ 0) but flat in static
  redundancy (quartile means indistinguishable, bootstrap P = 0.60), and
  per-image gap ranking has ~zero split-half reliability at K≤32. The
  redundancy scalar predicts nothing about demotion safety
  ([[project_tier_routing_phase3a_failed]]).
- Inference-side (committed-token compute reuse): killed by a free
  offline oracle replay of the 32 atlas sidecars
  (`_archive/sigma_lowres/bench/traj_stats/run_reuse_oracle.py`,
  `results/20260724-0001-phase3-reuse-oracle/`) — no implementation, no
  renders. Two independent kill shots: (a) even a perfect oracle
  freezing each token at its true retrospective commit step skips only
  25.5 % of token-steps at σ<1.0 (13.6 % at σ<0.5) — the ceiling is a
  property of the trajectories, detector-independent, and frozen tokens
  must stay in the K/V stream so wall-clock savings sit below it; (b) the
  realizable stable-for-m detector is unreliable everywhere — codes
  flicker, stable-for-m ≠ stable-forever; at the only tolerable-staleness
  cell (m=4, σ<0.5, ~14 % skip) 26 % of frozen tokens end in a different
  final code cell. And per-token commit time is not predictable a priori
  either (commit vs final-hf Spearman ≈ 0.17). All replay numbers are
  open-loop, i.e. the intervention's best case.

The population-level curves (Findings 1–2) are tight and reproducible; the
per-token / per-image draws from them are noise. Don't re-propose
token-level cache-skip / delta-token reuse on this DiT, and don't
re-propose redundancy-routed preprocessing — the refutations are
mechanism-independent.

## Reusable traps

- Oracle-replay first. The recorded sidecars (codes + activity + commit
  per step) are rich enough to adjudicate an intervention idea offline
  before any implementation — compute the perfect-oracle ceiling and the
  realizable-detector error from `traces_gen/` alone. Compute reuse died in
  an afternoon for zero GPU; price future ideas the same way.
- "Commit" is retrospective. The commit trace is *last change*, only
  knowable at the end — any online detector is a prediction and must be
  scored against it, never conflated with it.
- Recorder hook sites: pass `sigmas[i]` as the 0-d tensor
  (`float()` is a stream sync, ~28 ms/step) and write sidecars with
  uncompressed `np.savez` (zlib >100 ms inside generation wall time).

## Repro (bench home: `_archive/sigma_lowres/bench/traj_stats/`)

    uv run python _archive/sigma_lowres/bench/traj_stats/run_atlas.py --label phase1
    uv run python _archive/sigma_lowres/bench/traj_stats/analyze_subspace.py
    uv run python _archive/sigma_lowres/bench/traj_stats/run_gauge_calibration.py --label phase2
    uv run python _archive/sigma_lowres/bench/traj_stats/run_reuse_oracle.py
