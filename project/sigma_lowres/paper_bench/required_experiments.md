# paper_bench — required experiments before the sigma_lowres paper ships

Open runs only. Completed items — the 2026-07-28 review triage (R1–R5),
E1 (all of a/b/c + instrument changes), E2 (target-α sweep, slope ≈ 0
confirmed), E4 (4/5-arm grid: measured −14.6% wall, seed-noise
yardstick; remaining full-band-CMMD residual retired 2026-07-30), E5
(held-out validation, qualified PASS — Eq. 3 predicts, governors
upgrade to measured), E7 (controlled-LoRA factorial + probes), and
E8.1/E8.2 — live in `completed_experiments.md` with their results;
`paper_plan.md` is the manuscript plan.

**Context: E1 landed 2026-07-29 and decision rule 1 fired (512 debiased
gap_∞ +0.304 → token-count floor confirmed → Branch A).** Everything
below consumes E1's debiased numbers; the paired per-image object and
per-route debiased floors are in `completed_experiments.md`.

Status legend: **[FIX]** = paper/repo edit, no GPU; **[OWED]** = already
owed by `roadmap.md` (Phase 1b); **[STRETCH]** = raises acceptance
ceiling, not required for correctness.

---

## E3 — aggregation-conditioned safety map

Mostly already measured (`--pool`); what's missing is the **framing** and
one run at the real operating point:

- One verdict-grid run with `--pool <actual train batch × accum>` (read
  from the shipped LoRA config) + `--self_floor` so pooled arms get
  self-floors too.
- Paper: publish **two maps** — per-example (worst case, what a
  batch-1 user sees) and batch-aggregate (what the shipped trainer
  consumes) — and define "safe" as the pre-specified non-inferiority
  test (now the ε\* definition, E8.1). The CI itself is free (per-bin
  mean + 1.645·SEM from existing `per_image.jsonl` rows); the debiasing
  inside it is E1's paired object. This resolves the report-vs-paper 768
  contradiction the review found instead of hiding it.

## E6 [STRETCH] — one generalization arm each

- **One extra DiT** (any open flow-matching DiT with a different VAE),
  endpoint + 3-bin grid, routes {×0.875, ×0.5}, N=12: turns the case
  study into a phenomenon.
- **Full-FT probe** (all-param grads, N small, grad-ckpt): does the floor
  live in LoRA geometry or the model? One run answers it.
- Second LoRA checkpoint on Anima (different corpus): near-free, uses the
  existing instrument verbatim — largely discharged by E7's two
  controlled adapters (done; see `completed_experiments.md` §E7).

## E8.3 [FIX] — null→gap bridge (remaining part of E8)

E8.1 (ε\*) and E8.2 (guarantee region) are written into main.tex — see
`completed_experiments.md`. Remaining: convert each published tolerance
δ into a predicted *gap curve*, not just a boundary: under the diagonal
model on the measured P(f), compute the destroyed-band Bayes-residual
mismatch m_null,e(σ; δ), map through the measured G(σ)^p with the route
gain A calibrated on the one safe route, and overlay predicted vs
measured curves per (δ, route). This is the Table-1 confrontation
restated in gap units; it subsumes the continuous t\*(δ) sweep figure
(family spread ≤ 0.13) and the δ_reenc-anchored row
(`reenc_noise_floor.py`, ran 2026-07-30). Unit honesty: the null emits
residual units only — the bridge (G^p, calibrated A) belongs to the
two-term account, so the paper must say "the null read through our
bridge", which also makes §3 load-bearing for the confrontation rather
than decorative.

**Depends on:** E4's `reenc_noise_floor` run for the anchored row —
satisfied 2026-07-30 (`bench/results/20260730-0940-reenc-floor/`).
**Cost:** analysis + figures only.

## E9–E11 — response.md-driven vector instruments (launched 2026-07-30)

The `question.md` review (`paper/response.md`) found the pre-registered
I_768 scalar probe confounded (it estimates I(σ) + [F(σ) − F(1)], not
I(σ)) and the "banded ≤ 0.10 erasure" bound invalid (cross-terms). The
manuscript consequences are in `paper/action.md`; the replacement
instruments, all launched 2026-07-30:

- **E9 — interventional B/C ledger** [STOPPED before start 2026-07-30
  (hold); relaunch with the same command when GPU frees]
  (`run_sigma_probe.py --repromote
  --keep_arm_sums --self_floor`, routes 896/768/512, σ ∈ [0.5,1.0] 4 bins
  + endpoint, D=8, N=24, deterministic): B = ḡ_rp − ḡ₀ (data, native
  graph), C = ḡ_dem − ḡ_rp (graph, demoted data); cross-set-debiased
  S/F/I per (route, bin) via `vector_ledger.py`. Reads F(σ) directly (no
  σ-flat assumption), settles the 768 window two ways (I < 0 vs F
  collapse — pre-registered branches in `action.md`), localizes the
  Goldilocks prediction (window center at |B⊥| ≈ |C⊥|), and closes the
  §4.5 reenc-proxy [pending] (B vs native vs B vs reenc).
- **E10 — exact target-content vectors** (same instrument,
  `--target_alpha 0,1 --target_kappa --keep_arm_sums`, endpoint-only,
  N=40): the forward pass is α-independent, so t = ḡ(1) − ḡ(0) is exact
  at shared seeds; per-image + aggregate κ∥/κ⊥ of δt = t_dem − t_src
  decide parallel-landing vs J^T-attenuation for the §4.3
  "unresolvable share" paragraph. **[DONE 2026-07-30 — verdict:
  parallel landing** (`bench/results/20260730-2116-e10-kappa/`):
  |t_src|/G ≈ 2.23 aggregate; δt κ∥ −0.75/−1.18/−1.86 vs κ⊥
  0.09/0.14/0.20 on 896/768/512, rel ≥ 0.995, reenc control at noise
  floor; demotion shortens t along ĝ_src, rotation only at 512
  (cos 0.74). E2's flat α-slope explained (angular gap blind to
  parallel rescaling; κ⊥ second-order). Manuscript consequence in
  `paper/action.md` §4.3.]
- **E11 — Δr̄ direction structure** (`run_prior_distance.py
  --save_residuals` on the 1280 probe cache + `resid_structure.py`):
  split-half-corrected pairwise cosines + normalized stacked-SVD
  top-mode share across route pairs per σ — rank-one common mismatch
  direction vs norm-only uniformity. **[DONE 2026-07-30 — verdict:
  norm-only** (`bench/results/20260730-2054-resvec/`): non-adjacent
  corrected cos ≈ 0 at low σ, +0.2–0.33 at high σ; SVD top share
  0.33–0.36 vs 0.25; cross-image direction consistency ≈ 0 everywhere
  (image-specific directions — grid-conditional composition prior
  refuted under full captions); Δr̄ goes low-frequency as σ rises.
  Manuscript consequence in `paper/action.md` §4.4 ("universal m" →
  universal amplitude law). `--uncond` rerun implemented; first launch
  stopped mid-run 2026-07-30 (hold) — relaunch to close the
  caption-conditioning caveat.]

Instrument deltas live in `run_sigma_probe.py` (`--repromote`,
`--keep_arm_sums`, `--target_kappa`; arm-sum memmaps under
`<run>/arm_sums/`) and `run_prior_distance.py` (`--save_residuals`).
Vector stores are large (fp32 flat LoRA grads ≈ 311 MB × arms × bins) —
runs stay under the gitignored `bench/results/`; only the derived
`ledger.json` / `resid_structure.json` get copied into `paper_bench/`.

## Reproducibility deliverables (paper_bench/ contents) [FIX]

- `make_all.py` — single entry that regenerates every table/figure in
  `paper/` from run dirs (table → run-id mapping frozen in a
  `MANIFEST.toml` with commit hashes, adapter path, seeds).
  `plot_debiased_map.py` (Fig 1c) is the first piece.
- Results archive: `paper_bench/runs/` is already gitignore-exempt and
  in-repo for the E1 verdict runs; the older `bench/results/` raw runs
  still need either the same carve-out or a published tarball (HF
  dataset) — the reproducibility statement must match what's actually
  public.
- Manuscript pass: strip pending markers (E9/E10 clear the last two —
  the §4.5 reenc-proxy and §4.6 probe pendings), shorten
  abstract, enlarge Fig. 1, drop colored link boxes, label every claim
  pre-registered / confirmatory / post-hoc (the freeze dates exist in
  `questions.md` — link the commits), and narrow the headline to
  "spectral sufficiency of the noisy input does not guarantee gradient
  equivalence under resolution substitution."

---

## Order of operations

E1 + E2 + E4 + E5 + E7 + E8.1/8.2 done → E9–E11 (in flight 2026-07-30;
their ledgers gate the `paper/action.md` §4.3/§4.5/§4.6 rewrites) →
E3 pooled-arm run + a/(b/B) batch-size fit (`action.md` estimand fix) →
E8.3 analysis (anchored row satisfied) → reproducibility deliverables →
E6 if targeting a top venue.
