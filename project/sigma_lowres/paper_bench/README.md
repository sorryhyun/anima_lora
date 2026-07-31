# paper_bench — experiment index for the sigma_lowres paper

One directory per experiment under `experiments/<eN>/`, each holding its
own `README.md` (question, design, pre-registration, results, verdict)
plus the scripts that only it uses. Run artifacts stay flat in `runs/`
(gitignore-exempt, committable) so run IDs quoted in the manuscript keep
resolving; heavy vector stores stay under the gitignored
`bench/results/`.

*Reorganized 2026-07-31: `completed_experiments.md` and
`required_experiments.md` were split into these per-experiment records.
`review_triage_20260728.md` holds the R1–R5 external-review triage that
spawned E1–E4/E6; `paper_plan.md` remains the manuscript plan.*

## Status

| exp | title | status | verdict in one line |
|---|---|---|---|
| [E1](experiments/e1/) | debiased gaps: self-floors + draw-count extrapolation **[GATE]** | DONE 2026-07-29 | Rule 1 fired — token-count floor confirmed debiased (512 gap_∞ +0.304); endpoint gap **is** the graph floor |
| [E2](experiments/e2/) | target-strength sweep at the endpoint | DONE 2026-07-29 | α-flat at the anchors — no resolvable target-content share |
| [E3](experiments/e3/) | aggregation-conditioned safety map | **OPEN** | one pooled `--self_floor` grid + the two-maps framing |
| [E4](experiments/e4/) | the end-to-end A/B | DONE 2026-07-30 | measured **−14.6% wall / −15.1% FLOPs**; render deltas inside the seed lottery |
| [E5](experiments/e5/) | Eq. 3 held-out validation + three-form refit | DONE 2026-07-29 | qualified PASS at ~0.09 RMSE; exact angular link X headlines |
| [E6](experiments/e6/) | generalization arms | **OPEN [STRETCH]** | one extra DiT + one full-FT probe |
| [E7](experiments/e7/) | controlled-LoRA 2×2 style factorial | DONE 2026-07-29/30 | map is adapter-agnostic → "calibrate once per model"; floor *level* is checkpoint-dependent |
| [E8](experiments/e8/) | ε\*, guarantee region, null→gap bridge | DONE (8.1/8.2 07-29, 8.3 07-31) | the transported spectral family fails in both directions; δ inert at the curve level |
| [E9](experiments/e9/) | interventional B/C ledger | DONE 2026-07-31 | branch (i) — negative interference, I_768 < 0, window center σ ≈ 0.69 |
| [E10](experiments/e10/) | exact target-content vectors | DONE 2026-07-30 | parallel landing (κ∥ ≫ κ⊥); explains E2's flat α-slope |
| [E11](experiments/e11/) | Δr̄ direction structure | DONE 2026-07-30 (`--uncond` rerun pending) | norm-only — universal amplitude law, not a universal direction |
| [E12](experiments/e12/) | posterior-budget probes | DONE 2026-07-31 | REFUTED on both probes; native-only pre-screening is void |
| [E13](experiments/e13/) | resolving both ends of the σ curve (segmented grid) | **PLANNED** | segmented `--sigma_window` + one run; σ→1 rise is real in raw units but its shape — and the mid-σ peak — are unresolved |
| [E14](experiments/e14/) | two-level unbiased demotion (MLMC roulette, model-priced coin) | DONE 2026-07-31 (14.0 only) | **priced out** — per-sample correction 2nd moment ≈ V_total/2 ⇒ q̄≈0.45 at the 1.5× cap ⇒ net +2.6–4.1% vs ≥20% gate; yield = the aggregate-coherence finding (per-sample ‖Δ‖≈0.7–1.6‖g‖ vs aggregate 0.15–0.35); redesign lever = anchored control variate (own record if pursued) |
| [E15](experiments/e15/) | placement vs dilution: demotion scheduling (trajectory-propagator probe) | **PROPOSED** | zero-cost successor to E14 — same-mass early/late/spread 768 twins → ΔW ordering picks washout / linear / amplification regime; washout ⇒ hybrid768 ("768 early, clean finish") targets ~−21–24% wall inside the yardstick |

## Layout

```
paper_bench/
  README.md                    this index
  paper_plan.md                what the manuscript becomes (+ manuscript status)
  review_triage_20260728.md    R1–R5 external-review triage (origin of E1–E4, E6)
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

Ordering: **[E3](experiments/e3/)** pooled-arm run + the a/(b/B)
batch-size fit (`paper/action.md` estimand fix) → reproducibility
deliverables (below) → **[E6](experiments/e6/)** if targeting a top
venue. **[E13](experiments/e13/)** is orthogonal and can go whenever GPU
is free, but note it can move §4.7: its H2 tests whether the mid-σ peak
E5's data-term fit is calibrated against is partly an artifact of the
attenuation correction at the floor minimum. [E11](experiments/e11/)'s `--uncond` rerun closes the
caption-conditioning caveat whenever GPU is free.
**[E14](experiments/e14/)** closed at 14.0 (priced out, pre-registered
kill); what remains is writing its §5 paragraph + the
aggregate-coherence sharpening into the manuscript, and deciding
whether the anchored-control-variate redesign gets its own record.
**[E15](experiments/e15/)** is E14's zero-cost successor (scheduling
instead of roulette); its 15.0 ΔW-ordering probe is ~1 h of
deterministic twins and self-contained.

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
  probe pendings stripped 2026-07-31 with the E9 write-in; **still
  pending in-manuscript: E8.3 overlay, E3 batch-aggregate grid, raw-run
  tarball**), shorten abstract, enlarge Fig. 1, drop colored link boxes,
  label every claim pre-registered / confirmatory / post-hoc (the freeze
  dates exist in `record/questions.md` — link the commits), and narrow the
  headline to "spectral sufficiency of the noisy input does not
  guarantee gradient equivalence under resolution substitution."

## Conventions worth not relearning

- **Kernel-path chaos.** Per-bin cosines at low D differ by up to
  |Δcos| ≈ 0.3 across processes with different inductor kernel sets.
  Never compare cosines across runs — every gap/floor/debias pairing
  must stay inside one process. See [E1](experiments/e1/).
- **Debiased units only.** After the instrument-validation block, every
  main-text number is the paired per-image debiased gap (arm − reenc,
  |Δ|>1.5 trimmed). Raw numbers live in the manuscript appendix.
- **Results root.** Paper-bench runs pass
  `--results_root project/sigma_lowres/paper_bench/runs`.
- **`make daemon-run` eats `--label`** from ARGS as the *job* label, so
  a script never sees it — name the run dir another way.
