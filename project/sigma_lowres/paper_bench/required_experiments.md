# paper_bench — required experiments before the sigma_lowres paper ships

Open runs only. Completed items — the 2026-07-28 review triage (R1–R5),
E1 (all of a/b/c + instrument changes), E2 (target-α sweep, slope ≈ 0
confirmed), E5 (held-out validation, qualified PASS — Eq. 3 predicts,
governors upgrade to measured), and E8.1/E8.2 — live in
`completed_experiments.md` with their results; `paper_plan.md` is the
manuscript plan.

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

## E4 [OWED] — the end-to-end A/B (Phase 1b, already gated in roadmap.md)

Unchanged from `roadmap.md` but now with the review's negative control:
native baseline vs σ-conditional demotion (map from E1/E3, yarnsig
refinement per the 2026-07-27 PASS) vs **an unsafe-route arm**
(e.g. 1024→768 unconditional) as the negative control; identical data
order/noise/init, 3 seeds; measure realized wall-clock (incl. cache
load), examples/s, peak mem, CMMD, val loss, matched renders. Until this
lands, every "14%" in the paper reads "a projected ceiling of ~14%".
Also run `reenc_noise_floor.py` (script exists, queued) and clear the
three pending markers.

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
