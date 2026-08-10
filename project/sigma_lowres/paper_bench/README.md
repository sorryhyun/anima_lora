# paper_bench — experiment index for the sigma_lowres paper

One directory per experiment under `experiments/<eN>/`, each holding its
own `README.md` (question, design, pre-registration, results, verdict)
plus the scripts that only it uses. Run artifacts stay flat in `runs/`
(gitignore-exempt, committable) so run IDs quoted in the manuscript keep
resolving; heavy vector stores live centrally under the gitignored
`arm_sums/<run-name>/` (moved 2026-08-10 from per-run `bench/results/`
dirs, which keep symlinks — see `arm_sums/README.md` for the lifecycle
policy).

*Reorganized 2026-07-31: `completed_experiments.md` and
`required_experiments.md` were split into these per-experiment records.
`../record/review_triage_20260728.md` holds the R1–R5 external-review
triage that spawned E1–E4/E6; `../record/paper_plan.md` remains the
manuscript plan (both moved to `record/` 2026-08-09). The records of
the two refuted/killed lines — [E12](../record/e12/) (REFUTED on both
probes) and [E15](../record/e15/) (priced out, pre-registered kill) —
also live in `../record/` since 2026-08-09; their status rows stay in
the table below so the numbering and verdicts remain visible here.*

## Status

| exp | title | status | verdict in one line |
|---|---|---|---|
| [E1](experiments/e1/) | debiased gaps: self-floors + draw-count extrapolation **[GATE]** | DONE 2026-07-29 | Rule 1 fired — token-count floor confirmed debiased (512 gap_∞ +0.304); endpoint gap **is** the graph floor |
| [E2](experiments/e2/) | target-strength sweep at the endpoint | DONE 2026-07-29 | α-flat at the anchors — no resolvable target-content share |
| [E3](experiments/e3/) | aggregation-conditioned safety map | DONE 2026-08-04 | intercept ≤ 0.02/0.03 (896/768) at σ ≥ 0.44, persists at low σ; shipped operating point = the per-example map |
| [E4](experiments/e4/) | the end-to-end A/B | DONE 2026-07-30 | measured **−14.6% wall / −15.1% FLOPs**; render deltas inside the seed lottery |
| [E5](experiments/e5/) | Eq. 3 held-out validation + three-form refit | DONE 2026-07-29 | qualified PASS at ~0.09 RMSE; exact angular link X headlines |
| [E6](experiments/e6/) | generalization arms | **OPEN [STRETCH]** | one extra DiT + one full-FT probe |
| [E7](experiments/e7/) | controlled-LoRA 2×2 style factorial | DONE 2026-07-29/30 | map is adapter-agnostic → "calibrate once per model"; floor *level* is checkpoint-dependent |
| [E8](experiments/e8/) | ε\*, guarantee region, null→gap bridge | DONE (8.1/8.2 07-29, 8.3 07-31) | the transported spectral family fails in both directions; δ inert at the curve level |
| [E9](experiments/e9/) | interventional B/C ledger | DONE 2026-07-31 | branch (i) — negative interference, I_768 < 0, window center σ ≈ 0.69 |
| [E10](experiments/e10/) | exact target-content vectors | DONE 2026-07-30 | parallel landing (κ∥ ≫ κ⊥); explains E2's flat α-slope |
| [E11](experiments/e11/) | Δr̄ direction structure | DONE 2026-07-30 (`--uncond` rerun pending) | norm-only — universal amplitude law, not a universal direction |
| [E12](../record/e12/) | posterior-budget probes | DONE 2026-07-31 · **moved to `../record/` 2026-08-09** | REFUTED on both probes; native-only pre-screening is void |
| [E13](experiments/e13/) | resolving both ends of the σ curve (segmented grid) | DONE 2026-08-01 | H1 falsified (high-σ flat + a *correction-regime* step at σ=1), H2 falsified (the mid-σ peak is a plateau ≈0.09–0.43, not an artifact), H3 confirmed (896's approach is an instrument limit); E5's prediction survives the refit, its ratio-governor does not (A carries a per-run G normalization) |
| [E14](experiments/e14/) | low-σ vector ledger: decomposing the 896 bump (B/C below σ=0.5) | **DONE 2026-08-02** | headline: the 896 bump is a **two-large-opposing-terms regime** (ρ ≈ −0.7…−0.9 at every σ), not a starved data term; H-a FAILS (no 1120 twin), H-b fired (the 768-sibling half later fell probe-matched — E19 19.0), H-c σ ≤ 0.062 only, H-d substantive; carried the e13b probe-matched rerun E13 owed (`runs/20260801-2304-e14-ledger-probematched`) |
| [E15](../record/e15/) | two-level unbiased demotion (MLMC roulette, model-priced coin) | DONE 2026-07-31 (15.0 only; **E14 until 2026-08-01**, run dir keeps the `e14` label) · **moved to `../record/` 2026-08-09** | **priced out** — per-sample correction 2nd moment ≈ V_total/2 ⇒ q̄≈0.45 at the 1.5× cap ⇒ net +2.6–4.1% vs ≥20% gate; yield = the aggregate-coherence finding (per-sample ‖Δ‖≈0.7–1.6‖g‖ vs aggregate 0.15–0.35); redesign lever = anchored control variate (own record if pursued) |
| [E16](experiments/e16/) | placement vs dilution: demotion scheduling (trajectory-propagator probe) | **DONE 2026-08-03** (E15 until 2026-08-01) | 16.0 verdict **AMPLIFICATION** — placement dominates (ΔW cos vs native: late 0.906 ≫ early 0.193); 16.1: **combo** stacked router = throughput frontier, **−18.3 % wall inside the seed lottery on both corpora**; win768late = max-margin arm; scheduling unnecessary on certified routes |
| [E17](experiments/e17/) | Gaussian-closure test of the posterior residual identity | DONE 2026-08-01 | all three closures FAIL the amplitude bar (low-σ ~40% over-prediction) but reproduce shape (r 0.94–0.97), route-uniformity, reenc ≈ 0, endpoint — measured ‖Δr̄(σ)‖ certified as the minimal honest closure |
| [E19](experiments/e19/) | locating the birth of the B/C anti-alignment | **19.0+19.1 DONE 2026-08-06** | 19.0: completeness ordering confirmed (512 least complete), decoherence refuted (ρ route-uniform, amplitudes break), headline not a shared-arm artifact, 768 has no in-window crossing probe-matched; 19.1: closure predicts ρ_r < 0 everywhere but **weak** (−0.06…−0.14 at verdict bins vs measured g-level −0.7…−0.9) — prediction frozen, 19.2 scores it |
| [E20](experiments/e20/) | the cancellation-aware account under the paper's own lanes | DONE 2026-08-08 | 20.1/20.2 PARTIAL (geometry right, amplitude law open, no 20.3 spend); **20.4 NEGATIVE — derived data term fails at the estimand level** (the standing "estimand bridge" gap; no ledger-derived objective term) |
| [E21](experiments/e21/) | cell-level g-ledger: the cancellation drawn over the network | DONE 2026-08-08 | **LOCAL** (both routes, every verdict σ — the cancellation is pointwise per (depth×type) cell); π-swing dual read: direction delocalized (19.4 language stands), amplitude 86–87 % adaln (descriptive) |
| [E22](experiments/e22/) | per-image g-ledger: does the cancellation hold per sample? | DONE 2026-08-09 (22.4) | **PER-SAMPLE HOLDS at σ = 0.7** (22.4, D = 96: 14/16 gated ≥ 8 floor, median ρ_i −0.820, 93 % ≤ −0.7; strata null stands at healthy counts, no E23b hypothesis) — E23a drafting licensed, single-σ caveat carried. 22.1 (D = 24, three σ) remains INSTRUMENT-LIMITED on record |
| [E24](experiments/e24/) | cancellation-axis geometry: is the cancelled direction one global mode? | DONE 2026-08-08 | **STRUCTURED** — one axis across route/store/corpus at fixed σ (0.87–1.00), smooth σ-rotation (0.44–0.97; the "tracks the anchor's rotation" reading retired by E25.0-2 — matched-angle, not planar); knob read: residual is angle-borne (R_angle/R ≤ 0.30 vs R_amp/R ≥ 0.55 everywhere; frozen label text self-contradictory, numbers cited); E23a/b names reserved by E22's gated lever sketch |
| [E26](experiments/e26/) | cross-adapter cancellation geometry (the preserved E7 pair) | **E26.0 SMOKE DONE 2026-08-09** | **both adapters PASS** (flat ρ −0.939, dirty −0.959; depth orders with leg size; residual level mirrors E7's floor-level fact) → full-grid amendment licensed; rel_cos_R passes everywhere (dirty 0.51 marginal); cross-adapter direction cosines **frame-confounded** (≈ ĝ baseline 0.38) — frame-free estimand owed in the amendment if axis-sharing is to carry weight |
| [E25b](experiments/e25b/) | explicit resolution conditioning (adaln micro-conditioning arm) | **FROZEN 2026-08-11** | Stage 0 instrument + Stage 1 geometry gate pre-registered (h(B+C) ratio primary, 0.9/1.1 branch table, tenth-scale deterministic twins + 768 5-bin probe pair ≈ 10 GPU-h, go required); Stage 2 ship read gated on IMPROVES; E20.4-adjacency paragraph discharged in-file |
| [E25](experiments/e25/) | σ-local angle lever (population-level exploitation) | SKETCH · **E25.0 DONE 2026-08-09** | prerequisite read: **25.0-1 PARTIAL** — pooled (B+C)̂ direction reliable at 11/12 verdict conditions (hole: 768/σ = 0.4333; route-shared per σ, across-route 0.95, across-store ≈ 1) so E25a alive but restricted; **25.0-2 NO-GAIN** — E24's co-rotation is matched-angle, not planar (ĝ-frame transport buys Δ ≈ 0), so E25a's lookup must interpolate per σ bin, normalized-frame single direction refuted; arms still unfrozen, freeze owes the E20.4-adjacency paragraph; per-sample variants stay E23a-gated |

## Layout

```
paper_bench/
  README.md                    this index
  plot_debiased_map.py         Fig 1c / per-adapter maps (cross-experiment)
  vector_ledger.py             S/F/I ledger from arm sums (E9 + E10)
  runs/                        run artifacts, flat, gitignore-exempt
  experiments/<eN>/README.md   per-experiment record + its own scripts
```

Scripts resolve paths off their own location: `PAPER_BENCH` (=this dir,
where `runs/` lives), `SIGMA` (=`project/sigma_lowres`, for `bench/` and
`paper/`), and `REPO` (=repo root). Runs written by a moved script still
land in `paper_bench/runs/`.

## Open work

Ordering: ~~[E3](experiments/e3/) pooled-arm run + the a + b/B
batch-size fit~~ (DONE 2026-08-04, §4 map paragraph rewritten) →
reproducibility deliverables (below) → **[E6](experiments/e6/)** if
targeting a top venue. **[E13](experiments/e13/)** is closed as a run; what remains is
its manuscript write-in (Fig. 1a redraw off `runs/20260801-0125`, dense-end
rows into Table 2, §4.4 re-derived rather than assumed, §4.7's shape
claims re-scored against a plateau instead of a peak, 896 stated as an
instrument limit, and the per-run G normalization of A written down).
~~The **e13b probe-matched rerun** reserved to carry
**[E14](experiments/e14/)**'s B/C ledger arms~~ (DONE 2026-08-02 — one
run carried both: the owed §4.7 probe-matched refit is discharged, and
the 896 low-σ plateau decomposed as a two-large-opposing-terms regime,
not a starved data term). [E11](experiments/e11/)'s `--uncond`
rerun closes the caption-conditioning caveat whenever GPU is free.
**[E15](../record/e15/)** closed at 15.0 (priced out, pre-registered
kill); what remains is writing its §5 paragraph + the
aggregate-coherence sharpening into the manuscript, and deciding
whether the anchored-control-variate redesign gets its own record.
~~**[E16](experiments/e16/)** is E15's zero-cost successor~~ (DONE
2026-08-03 — 16.0 AMPLIFICATION, placement dominates; 16.1's combo
router is the new throughput frontier at −18.3 % wall inside the seed
lottery).

### Reproducibility deliverables [FIX]

- `make_all.py` — single entry that regenerates every table/figure in
  `paper/` from run dirs (table → run-id mapping frozen in a
  `MANIFEST.toml` with commit hashes, adapter path, seeds).
  `plot_debiased_map.py` (Fig 1c) is the first piece.
- Results archive: `paper_bench/runs/` is already gitignore-exempt and
  in-repo for the E1 verdict runs; the older `bench/results/` raw runs
  still need either the same carve-out or a published tarball (HF
  dataset) — the reproducibility statement must match what's actually
  public.
- Manuscript pass: ~~strip pending markers~~ (§4.5 reenc-proxy + §4.6
  probe pendings stripped 2026-07-31 with the E9 write-in; ~~E8.3
  overlay~~ (in the appendix as `fig:e83`), ~~E3 batch-aggregate grid~~
  (written in 2026-08-04), ~~raw-run tarball marker~~ (marker removed
  2026-08-04, repo link added to the abstract — publishing the tarball
  itself stays under Reproducibility deliverables above); **still
  pending in-manuscript: the whole E13 write-in
  (Fig. 1a / Table 2 / §4.4 / §4.7 / A-normalization)**), shorten abstract, enlarge Fig. 1, drop colored link boxes,
  label every claim pre-registered / confirmatory / post-hoc (the freeze
  dates exist in `record/questions.md` — link the commits), and narrow the
  headline to "spectral sufficiency of the noisy input does not
  guarantee gradient equivalence under resolution substitution."

## Conventions worth not relearning

- **Kernel-path chaos.** Per-bin cosines at low D differ by up to
  |Δcos| ≈ 0.3 across processes with different inductor kernel sets.
  Never compare cosines across runs — every gap/floor/debias pairing
  must stay inside one process. See [E1](experiments/e1/).
- **Probe sets bound the claim, not just kernel paths.** Two runs with
  different image sets share no *levels* — only within-run shape claims
  transfer. E13 overlaps E1b by 2 of 24 images, so E1a/E1b keep every
  endpoint level. Match `--probe_list` when levels are the point.
- **Non-uniform σ grids need bin-width weights.** Any WLS over bins
  (E5's fits) must carry a bin-width term alongside `1/sem²`, or a
  segmented grid silently concentrates the fit where the bins are thin
  (E13: 45% of the weight in 10% of the axis, which flipped p\* 2→1).
  `bin_widths()` is normalized to mean 1, so it is exactly inert on a
  uniform grid. See [E13](experiments/e13/).
- **`A` in E5's fits carries a per-run G normalization.** `m(σ)` comes
  from a fixed run while `G(σ)` floats with the probe run, so raw `A`
  is not comparable across runs and the ratio-governor z is a
  scale-dependent statistic. Predictions (`A·x`) are unaffected.
- **Debiased units only.** After the instrument-validation block, every
  main-text number is the paired per-image debiased gap (arm − reenc,
  |Δ|>1.5 trimmed). Raw numbers live in the manuscript appendix.
- **Segmented σ grids ship.** `--sigma_window` takes `LO,HI,BINS`
  segments joined by `:` (`'0,0.1,4 : 0.1,0.9,6 : 0.9,1.0,4'`) at a
  global `--draws_per_bin`; the single-interval form still works.
  Per-segment draws-per-bin is *not* possible — the estimator iterates a
  rectangular `(bins, draws)`; vary bin density instead.
- **Results root.** Paper-bench runs pass
  `--results_root project/sigma_lowres/paper_bench/runs`.
- **`--label` passes through since 2026-08-01.** `make daemon-run`'s own
  flags are scoped to the prefix before the script path; everything after
  the script (incl. `--label`) reaches it verbatim, and the child's label
  is mirrored into the job display name. (Before the fix the wrapper ate
  `--label` anywhere in ARGS — the label-less run dirs E2/E13 note.)
