# paper_bench — required experiments before the sigma_lowres paper ships

Open runs only. Completed items — the 2026-07-28 review triage (R1–R5),
E1 (all of a/b/c + instrument changes), E2 (target-α sweep, slope ≈ 0
confirmed), **E4 core (4-arm × 3-seed grid: measured −14.6% wall,
seed-noise yardstick; residuals below)**, E5 (held-out validation,
qualified PASS — Eq. 3 predicts, governors upgrade to measured), and
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

## E4 [CORE DISCHARGED 2026-07-30] — residuals only

The 4-arm × 3-seed × 2-artist grid ran and the throughput claim is now
**measured** (sigma896: −14.6% wall / −15.1% FLOPs; full results +
seed-noise yardstick verdicts in `completed_experiments.md` §E4).
Still owed before the paper's E4 story closes:

- **Full-band CMMD rescoring** — the exercise scored on the SFW prompt
  amendment (9/12 prompts vs rating-mismatched all-band pools) and the
  negative control exposed the resulting lack of power (unsafe768 ≈
  native). Re-run `e4_render_eval.py` with the manifest's frozen 24/15
  full-band prompts (metrics stay private; SFW sheets remain the
  figures) and state the non-inferiority test in ε\* terms (E8.1).
  This is also the empirical handle on the **σ-gate vs 896only**
  question (gate is endpoint-invisible so far — see the tension noted
  in `completed_experiments.md`).
- **val loss + peak mem** — not captured in the exercise runs (no
  validation split configured); add both to the full-band pass.
- `reenc_noise_floor.py` run (unchanged — script exists, queued;
  δ_reenc row + D(f) numbers for the manuscript).
- Manuscript: clear the E4 `[pending]` markers and upgrade every
  "projected ceiling of ~14%" to the measured numbers.

Why the A/B is epistemically load-bearing (the instrument is
valence-blind; the residual risk is accumulated sub-band bias):
`claim_accumulated_bias.md`.

## E6 [STRETCH] — one generalization arm each

- **One extra DiT** (any open flow-matching DiT with a different VAE),
  endpoint + 3-bin grid, routes {×0.875, ×0.5}, N=12: turns the case
  study into a phenomenon.
- **Full-FT probe** (all-param grads, N small, grad-ckpt): does the floor
  live in LoRA geometry or the model? One run answers it.
- Second LoRA checkpoint on Anima (different corpus): near-free, uses the
  existing instrument verbatim — largely subsumed by E7's controlled
  adapter if that runs.

## E7 — controlled-LoRA 2×2 style factorial: data relationship to the adapter

**Question.** Does the safety map depend on the probe data's
relationship to the adapter — is the map a property of (model, route)
or of (model, route, *adapter's training distribution*)? Post-E5 this
carries the deployment half of the calibration-recipe claim:
adapter-agnostic ⇒ "calibrate once per model, use with any adapter";
adapter-dependent ⇒ recalibrate per adapter. Also cleans up the known
wart (2/40 verdict stems were in the shipped probe adapter's fine-tune
set — membership was never a controlled axis).

**Design (upgraded 2026-07-29 — factorial, two adapters).** Two
controlled LoRAs on opposite **style clusters**, probed on both
clusters. The reverse arm converts the style-statistics confound into
an estimable main effect: probe-style main effect = image-statistics
dependence; adapter×probe-style **interaction** = the pure
distribution-relationship effect.

- **Frozen style axis (operational, decided before any training):**
  per-artist median latent redundancy (the established tier_routing
  scalar, high = flat), over qualifying images (1024-tier cache +
  complete sidecars). Inventory: `paper_bench/e7_inventory.py`, run
  `runs/20260729-1147-e7-inventory/e7_inventory.tsv` — 2901 qualifying
  images, 70 artists with ≥8. **Clusters = top-12 ("flat", median
  ≥ 0.787) and bottom-12 ("dirty", ≤ 0.707) artist medians among
  eligible artists.** `sincos` (shipped adapter's artist) falls
  mid-pack and is excluded by construction. Known correlate stated up
  front: redundancy anti-correlates with tag count (ρ = −0.40), so
  supervision density rides the style axis — acknowledged, not
  controlled.
- **Pool composition per cluster:** train on **8 of the 12 artists**
  (4 reserved entirely → the same-style-never-seen cell), splitting
  **within** each trained artist ~65/35 (held-out side ≥3 images);
  per-artist train contribution capped (≤16 images) so large-n artists
  don't dominate; total train set ~100–150 images ≈ a realistic
  targeted fine-tune. Within-artist split random-seeded; **stem-level
  manifest frozen and committed before training** (`path_pattern`
  can't express within-artist splits — needs a stem-list knob or a
  dedicated symlink subtree).
- **Training recipe (frozen 2026-07-29, `configs/gui-methods/custom/e7.toml`):**
  verbatim the shipped probe adapter's recipe
  (`anima_soup_sincos.snapshot.toml` is the authority — NOT
  `configs/methods/lora.toml`, whose default stacks Ortho/T-LoRA/Hydra
  and changes both checkpoint keys and gradient geometry, and NOT
  `configs/soup/soup.toml`, whose defaults drifted from what actually
  trained the shipped adapter): plain-LoRA soup, dim 32 / alpha 128,
  weight_svd init, 6 epochs, per-ingredient `lr_pool 1e-5,2e-5,5e-5`,
  adaln 16/90, timestep mask, REPA relational 0.05@8, caption dropout
  0.05, full pipeline (uncond init → 3 seeded fine-tunes 1001–1003 →
  SVD soup @32). **Pre-registered deviations from shipped:** (1)
  Phase-1 uncond pool = the cluster's own train manifest, NOT the
  corpus-wide `anima_uncond_df58248c` (which saw every image
  caption-dropped and would contaminate S2/S3 membership); (2) train
  totals equalized across clusters (124 images each); (3) identical
  seeds both clusters. **Frozen split manifest:**
  `paper_bench/e7_manifest.py` (seed 20260729) →
  `runs/20260729-1158-e7-manifest/` — train 124 = 8 largest-n artists
  × ≤16 (65/35 within-artist), holdout 418 (dirty) / 236 (flat),
  reserved-artist cells belko/dikko/nvl/yomiji292 (dirty) and
  ebifurya/hamao/madana/moursho (flat); patterns verified stem-exact
  via `filter_paths_by_glob`. Launch:
  `make soup CUSTOM=e7 PATH_PATTERN="$(cat path_pattern_<c>.txt)"
  POOL_PATH_PATTERN=<same> NAME=e7_<c> --queue`. Bonus read: two fresh
  checkpoints' floors = the G5 checkpoint-dependence ("carving") test
  for free. **Adapters TRAINED 2026-07-29**: daemon jobs
  20260729-124257-{8ec9e2 (flat), 6f3dc8 (dirty)} both done →
  `output/ckpt/anima_soup_e7_flat.safetensors` /
  `anima_soup_e7_dirty.safetensors` (+ per-seed s1001–s1003 siblings +
  snapshots; snapshots verified — dim 32/alpha 128, 6 epochs,
  stem-exact 124-stem patterns over the 8 manifest artists per
  cluster, cluster-own uncond init `anima_uncond_abd30619`, not the
  corpus-wide pool). **Probe runs DONE 2026-07-29** (daemon jobs
  20260729-134926-{a4a4ba (flat), 52e01d (dirty)}, frozen probe lists
  `runs/20260729-1349-e7-probe-lists/`): flat →
  `runs/20260729-1349/`, dirty → `runs/20260729-1702/` (48 stems each,
  cell/membership/style tags in `per_image.jsonl`, endpoint debiased
  gaps ≈ 0 both arms). **Bin-mean readout** (`e7_cells.py`; A_e/F_e
  per-cell fits still owed for the formal ledger): (1) map shape
  replicates on both adapters — 896 reaches the ±ε\* band in-window,
  768 stays above until the top; (2) probe-style main effect ≈ 0 and
  the adapter×probe-style **interaction** (verdict quantity) null
  below resolution — raw-gap paired −0.022±0.027 (896), −0.015±0.059
  (768); (3) membership S1−S2 does not replicate in sign across
  adapters (no trajectory variant needed); (4) NEW FACT: redraw-floor
  *level* is checkpoint-dependent (in-window cos_floor ≈0.73 flat vs
  ≈0.50 dirty; gnorm endpoint 11.98 vs 7.12) while cell-invariant
  within each adapter incl. opposite-cluster stems — the G5 carving
  read came back positive. **In the paper**: Appendix `app:e7`
  (figure `figs/gap_debiased_e7_{flat,dirty}.png` via
  `plot_debiased_map.py --run … --title …`) + Limitations `[pending]`
  marker resolved; one outlier stem (dirty S1 `10541215`, debiased-896
  ≈ −3.05) noted for the formal fit. Naming: the paper calls the
  clusters **high-redundancy** (= `flat` in every artifact/ckpt/run
  name) and **low-redundancy** (= `dirty`); figs
  `gap_debiased_e7_route{896,768}.png` (`plot_e7_routes.py` — one
  panel per route, color = adapter, solid ID own-cluster S1/S2/S3s vs
  dashed OOD S3x, shared axes; the per-adapter variant stays available
  via `plot_debiased_map.py --split-ood-cell S3x`). **Correlation
  post-hoc (2026-07-29, not in paper — guard against re-proposing a
  redundancy→gap law):** (a) per-stem redundancy vs in-window gap
  within-run stays null across all three runs (flat/dirty/E1b; one
  nominal hit in 10 tests, sign not replicated); (b) per-stem gaps do
  NOT correlate across adapters on the 24 shared stems (gap_896
  ρ=+0.30 ns, gap_768 ρ=−0.45 nominal-negative, floor ρ=+0.15 ns) —
  gap magnitude is not an image property; (c) adapter-level train-red
  → floor is NOT monotonic once shipped-sincos/E1b is added as a third
  point (0.670→0.499, 0.773→0.783, 0.806→0.735; cross-run caveats:
  different probe pool/grid, gnorm@1 37.5 vs 12.0/7.1), and the
  adapter ordering of "bigger gap" flips between raw and debiased
  units — so the checkpoint effect stays "adapter-dependent floor,
  mechanism unresolved" (supervision-density confound ρ=−0.40 still
  rides the style axis).
- **Probe cells per adapter, N≈12 each** (S1/S2 redundancy-matched
  within cluster; one probe list per adapter, cells as row tags —
  cross-cluster stems shared between the two adapters so the
  interaction reads paired per-stem):

| cell | relationship to adapter |
|---|---|
| S1 | trained-on (in manifest) |
| S2 | held-out images, trained artists |
| S3s | never-seen artists, same style cluster |
| S3x | never-seen artists, opposite cluster |
| S4 (optional) | outside the illustration corpus — model-level OOD, NOT part of the factorial (probes the backbone's manifold, not the adapter relationship) |

- **Runs:** ONE probe run per adapter (≈48 stems, endpoint + 4-bin
  high-σ grid, routes {896, 768}, `--self_floor`, D=8) — all
  within-adapter cell contrasts stay inside one process (kernel-path
  rule); cross-adapter reads are of paired gaps. Instrument delta:
  `--probe_list <file>` (explicit stems) + membership/style tags in
  `per_image.jsonl` rows.

**Pre-registered readout (E5-upgraded: fit A_e, F_e per cell).**

| effect | prediction |
|---|---|
| Floor_e (and exp-law τ) across all cells | **invariant** — graph property (strong, falsifiable) |
| A_e: S1 vs S2 (membership, style-matched) | may shrink on trained-on (memorized → small residual) |
| A_e: probe-style main effect (flat vs dirty pools) | ≈ 0 — the designed-contrast version of the per-image spectral-predictor question (correlational null so far; first time tested by design at the cell level, where bin means are reliable) |
| A_e: adapter×probe-style interaction | the E7 verdict quantity |

Power, stated so a null is bounded: at N=12/cell the resolvable
bin-mean effect is ~0.03–0.05 gap units; effects below that are
"bounded by instrument resolution", not absent.

- S1 vs S2 shift ⇒ the map drifts *during* training (the trainer
  demotes its own fine-tune set) → add the trajectory variant:
  re-probe S1/S2 at an early-epoch checkpoint (free — `save_every`
  artifacts exist).
- No shift anywhere ⇒ the Limitations operating-point paragraph
  becomes a supported robustness claim: the map is adapter- AND
  content-agnostic on this model, and the recipe is "calibrate once
  per model".

Cost: two standard LoRA trains + two probe runs (~E1b-scale each) +
the `--probe_list` delta. Still the cheapest generalization item;
kick off both trains alongside E3's GPU time as already planned.

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
(`reenc_noise_floor.py`, owed in E4). Unit honesty: the null emits
residual units only — the bridge (G^p, calibrated A) belongs to the
two-term account, so the paper must say "the null read through our
bridge", which also makes §3 load-bearing for the confrontation rather
than decorative.

**Depends on:** E4's `reenc_noise_floor` run for the anchored row.
**Cost:** analysis + figures only.

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
- Manuscript pass: strip pending markers (after E4's floor run), shorten
  abstract, enlarge Fig. 1, drop colored link boxes, label every claim
  pre-registered / confirmatory / post-hoc (the freeze dates exist in
  `questions.md` — link the commits), and narrow the headline to
  "spectral sufficiency of the noisy input does not guarantee gradient
  equivalence under resolution substitution."

---

## Order of operations

E1 + E2 + E5 done → main.tex restructure per `paper_plan.md` (E5's
qualified PASS sets the voice) → E3 (kick off E7's controlled-LoRA
train here — it's GPU-cheap and its probes need E1's debiased
instrument anyway; two trains now — the E7 factorial) → E8.3 analysis (its anchored row waits on E4's
`reenc_noise_floor` run) → E4 (Phase 1b as owed) → E7 probes → E6 if
targeting a top venue.
