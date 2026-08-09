# E7 — controlled-LoRA 2×2 style factorial

| | |
|---|---|
| **Status** | **DONE 2026-07-29/30** (A_e/F_e per-cell fits still owed for the formal ledger) |
| **Verdict** | The map shape **replicates on both adapters**; probe-style main effect ≈ 0 and the adapter×probe-style interaction is null below resolution → **"calibrate once per model"** survives. NEW FACT: the redraw-floor *level* is checkpoint-dependent (the G5 "carving" read came back positive). |
| **Runs** | `runs/20260729-1147-e7-inventory/` · `runs/20260729-1158-e7-manifest/` · `runs/20260729-1349-e7-probe-lists/` · probes: `runs/20260729-1349/` (flat) and `runs/20260729-1702/` (dirty) |
| **Scripts** | `e7_inventory.py` (style axis) · `e7_manifest.py` (frozen splits) · `e7_probe_lists.py` (frozen probe cells) · `e7_cells.py` (bin-mean readout) · `plot_e7_routes.py` (per-route figures) |
| **Adapters** | `output/ckpt/anima_soup_e7_flat.safetensors` / `anima_soup_e7_dirty.safetensors` (+ per-seed s1001–s1003 siblings + snapshots); config `configs/gui-methods/custom/e7.toml` |
| **In the paper** | Appendix `app:e7` + figures `figs/gap_debiased_e7_route{896,768}.png`; the Limitations `[pending]` marker is resolved. Naming: **high-redundancy** = `flat` in every artifact, **low-redundancy** = `dirty`. |

**Question.** Does the safety map depend on the probe data's
relationship to the adapter — is the map a property of (model, route)
or of (model, route, *adapter's training distribution*)? Post-[E5](../e5/)
this carries the deployment half of the calibration-recipe claim:
adapter-agnostic ⇒ "calibrate once per model, use with any adapter";
adapter-dependent ⇒ recalibrate per adapter. Also cleans up the known
wart (2/40 verdict stems were in the shipped probe adapter's fine-tune
set — membership was never a controlled axis).

## Design (upgraded 2026-07-29 — factorial, two adapters)

Two controlled LoRAs on opposite **style clusters**, probed on both
clusters. The reverse arm converts the style-statistics confound into
an estimable main effect: probe-style main effect = image-statistics
dependence; adapter×probe-style **interaction** = the pure
distribution-relationship effect.

- **Frozen style axis (operational, decided before any training):**
  per-artist median latent redundancy (the established tier_routing
  scalar, high = flat), over qualifying images (1024-tier cache +
  complete sidecars). Inventory: `e7_inventory.py`, run
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
  seeds both clusters.
- **Frozen split manifest:** `e7_manifest.py` (seed 20260729) →
  `runs/20260729-1158-e7-manifest/` — train 124 = 8 largest-n artists
  × ≤16 (65/35 within-artist), holdout 418 (dirty) / 236 (flat),
  reserved-artist cells belko/dikko/nvl/yomiji292 (dirty) and
  ebifurya/hamao/madana/moursho (flat); patterns verified stem-exact
  via `filter_paths_by_glob`. Launch:
  `make soup CUSTOM=e7 PATH_PATTERN="$(cat path_pattern_<c>.txt)"
  POOL_PATH_PATTERN=<same> NAME=e7_<c> --queue`. Bonus read: two fresh
  checkpoints' floors = the G5 checkpoint-dependence ("carving") test
  for free.
- **Adapters TRAINED 2026-07-29**: daemon jobs
  20260729-124257-{8ec9e2 (flat), 6f3dc8 (dirty)} both done →
  `output/ckpt/anima_soup_e7_flat.safetensors` /
  `anima_soup_e7_dirty.safetensors` (+ per-seed s1001–s1003 siblings +
  snapshots; snapshots verified — dim 32/alpha 128, 6 epochs,
  stem-exact 124-stem patterns over the 8 manifest artists per
  cluster, cluster-own uncond init `anima_uncond_abd30619`, not the
  corpus-wide pool).

## Probe cells per adapter, N≈12 each

S1/S2 redundancy-matched within cluster; one probe list per adapter,
cells as row tags — cross-cluster stems shared between the two adapters
so the interaction reads paired per-stem.

| cell | relationship to adapter |
|---|---|
| S1 | trained-on (in manifest) |
| S2 | held-out images, trained artists |
| S3s | never-seen artists, same style cluster |
| S3x | never-seen artists, opposite cluster |
| S4 (optional) | outside the illustration corpus — model-level OOD, NOT part of the factorial (probes the backbone's manifold, not the adapter relationship) |

**Runs:** ONE probe run per adapter (≈48 stems, endpoint + 4-bin
high-σ grid, routes {896, 768}, `--self_floor`, D=8) — all
within-adapter cell contrasts stay inside one process (kernel-path
rule); cross-adapter reads are of paired gaps. Instrument delta:
`--probe_list <file>` (explicit stems) + membership/style tags in
`per_image.jsonl` rows.

## Pre-registered readout (E5-upgraded: fit A_e, F_e per cell)

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

## Results — probe runs DONE 2026-07-29

Daemon jobs 20260729-134926-{a4a4ba (flat), 52e01d (dirty)}, frozen
probe lists `runs/20260729-1349-e7-probe-lists/`: flat →
`runs/20260729-1349/`, dirty → `runs/20260729-1702/` (48 stems each,
cell/membership/style tags in `per_image.jsonl`, endpoint debiased gaps
≈ 0 both arms). **Bin-mean readout** (`e7_cells.py`; A_e/F_e per-cell
fits still owed for the formal ledger — NB before running them, E13:
raw `A` carries a per-run G normalization, so any cross-adapter
comparison needs a stated normalization convention):

1. Map shape **replicates on both adapters** — 896 reaches the ±ε\*
   band in-window, 768 stays above until the top.
2. Probe-style main effect ≈ 0 and the adapter×probe-style
   **interaction** (verdict quantity) null below resolution — raw-gap
   paired −0.022±0.027 (896), −0.015±0.059 (768).
3. Membership S1−S2 does not replicate in sign across adapters (no
   trajectory variant needed).
4. **NEW FACT:** redraw-floor *level* is checkpoint-dependent
   (in-window cos_floor ≈0.73 flat vs ≈0.50 dirty; gnorm endpoint 11.98
   vs 7.12) while cell-invariant within each adapter incl.
   opposite-cluster stems — the G5 carving read came back positive.

One outlier stem (dirty S1 `10541215`, debiased-896 ≈ −3.05) noted for
the formal fit.

### Correlation post-hoc (2026-07-29, not in paper)

Guard against re-proposing a redundancy→gap law:

- (a) per-stem redundancy vs in-window gap within-run stays null across
  all three runs (flat/dirty/E1b; one nominal hit in 10 tests, sign not
  replicated);
- (b) per-stem gaps do NOT correlate across adapters on the 24 shared
  stems (gap_896 ρ=+0.30 ns, gap_768 ρ=−0.45 nominal-negative, floor
  ρ=+0.15 ns) — gap magnitude is not an image property;
- (c) adapter-level train-red → floor is NOT monotonic once
  shipped-sincos/E1b is added as a third point (0.670→0.499,
  0.773→0.783, 0.806→0.735; cross-run caveats: different probe
  pool/grid, gnorm@1 37.5 vs 12.0/7.1), and the adapter ordering of
  "bigger gap" flips between raw and debiased units — so the checkpoint
  effect stays "adapter-dependent floor, mechanism unresolved"
  (supervision-density confound ρ=−0.40 still rides the style axis).
  E19.6 later narrowed the mechanism space: with the sincos adapter
  merged, both ledger legs and the native residual field are unmoved
  (dircos ≈ 1, amp within ~3 %) — LoRA-moves-B refuted, the map-level
  null extends to the leg level, so the floor-*level* dependence is not
  carried by the B leg. Mechanism otherwise still open.

## Figures

`plot_e7_routes.py` — one panel per route, color = adapter, solid ID
own-cluster S1/S2/S3s vs dashed OOD S3x, shared axes. The per-adapter
variant stays available via
`plot_debiased_map.py --run … --title … --split-ood-cell S3x`
(→ `figs/gap_debiased_e7_{flat,dirty}.png`).

**Cost note (historical):** two standard LoRA trains + two probe runs
(~E1b-scale each) + the `--probe_list` delta — the cheapest
generalization item.
