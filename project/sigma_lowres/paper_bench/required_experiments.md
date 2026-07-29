# paper_bench — required experiments before the sigma_lowres paper ships

Open runs only. Completed items — the 2026-07-28 review triage (R1–R5),
E1 (all of a/b/c + instrument changes), and E8.1/E8.2 — live in
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

## E2 — target-strength sweep at the endpoint (relabel the floor)

`--target_alpha 0,0.25,0.5,0.75,1`: at σ=1, input = ε unchanged, target
= ε − α·x (per-arm x). Decomposes the endpoint gap into graph share
(α=0, ≡ x-zero-in-target-only), target-content share (slope in α), and
interaction. N=12, D=16, routes {768, 512} (896 is ≈0 already). Cheap —
one afternoon run. **Post-E1(c) status: demoted from gate-adjacent to
cheap confirmation** — x-zero ≡ endpoint at every route predicts
α-slope ≈ 0. Paper edits regardless of outcome: rename "endpoint gap @
σ=1" → "high-noise endpoint gap"; x-zero is the graph-only control;
768's text stops claiming the plateau *is* the floor.

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

## E5 — Eq. 3 held-out validation, or demotion to "conceptual"

Unlocked by E1 (floors confirmed): fit A_e, Floor_e on {1024→896,
1024→512, 1280→1120}, predict {1024→768, 1280→1024} from measured m(σ),
G(σ); compare vs spectral-null and an unconstrained smooth fit. If it
fails to predict, keep Eq. 3 but present it as a decomposition, and
change "ratio sets / token count sets" to "consistent with". No new
instrument needed — this is analysis over existing + E1 rows.

## E6 [STRETCH] — one generalization arm each

- **One extra DiT** (any open flow-matching DiT with a different VAE),
  endpoint + 3-bin grid, routes {×0.875, ×0.5}, N=12: turns the case
  study into a phenomenon.
- **Full-FT probe** (all-param grads, N small, grad-ckpt): does the floor
  live in LoRA geometry or the model? One run answers it.
- Second LoRA checkpoint on Anima (different corpus): near-free, uses the
  existing instrument verbatim — largely subsumed by E7's controlled
  adapter if that runs.

## E7 — controlled-LoRA 2×2: data relationship to the adapter

**Question.** Does the safety map depend on the probe data's relationship
to the adapter — i.e. is the map a property of the (model, route) or of
the (model, route, *adapter's training distribution*)? The paper already
concedes operating-point dependence in Limitations; this measures it
instead of conceding it. It also cleans up a known wart: the current
verdict pool contains 2/40 stems from the probe adapter's own fine-tune
set (`report.md` in-pool overlap check, 2026-07-26) — membership was
never a controlled axis.

**Design.** Train a **controlled LoRA** with a frozen, manifest-recorded
fine-tune set (a dedicated `path_pattern` slice; standard `make lora`,
same recipe as the shipped probe adapter), holding out a matched split of
the same artists. Probe cells, N≈12 each, redundancy-matched across
cells:

| cell | membership | domain |
|---|---|---|
| S1 | trained-on (in the fine-tune manifest) | in-domain |
| S2 | held-out, same artists | in-domain |
| S3 | held-out, different artists / style cluster | near-OOD |
| S4 | (optional) far-OOD — outside the illustration corpus | OOD |

Run the E1-debiased instrument (endpoint + the 4-bin high-σ grid, routes
{896, 768}) per cell. Instrument delta: probe selection currently goes
through redundancy scoring with `--artists`/`--max_per_artist` only — add
a `--probe_list <file>` override (explicit stem list per cell) plus a
membership tag in `per_image.jsonl` rows, so cells are exact rather than
glob-approximate.

**Pre-registered readout.**
- Two-term prediction: the **floor** is a Jacobian/graph property →
  invariant across all four cells (strong, falsifiable). The **input
  branch** rides the residual and gradient norm, which plausibly shrink
  on trained-on images (memorized → small residual) → A_e and possibly
  the measured σ* may shift between S1 and S2–S4.
- S1 vs S2 shift ⇒ the map drifts *during* training as images become
  trained-on — directly relevant to deployment, since the trainer demotes
  its own fine-tune set. If this fires, add the trajectory variant:
  re-probe S1/S2 at an early-epoch checkpoint of the same controlled run
  (checkpoints are free — they already exist as `save_every` artifacts).
- No shift anywhere ⇒ one Limitations paragraph becomes a supported
  robustness claim; the safety map is adapter-agnostic on this model.

Cost: one standard LoRA train + ~4 probe runs at reduced grid; cheapest
item on the generalization axis and the only one that addresses a stated
limitation with existing hardware.

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

E1 done (Branch A written) → E2 + E3 in parallel (kick off E7's
controlled-LoRA train here — it's GPU-cheap and its probes need E1's
debiased instrument anyway) → E8.3 analysis (its anchored row waits on
E4's `reenc_noise_floor` run) → E4 (Phase 1b as owed) → E7 probes → E5 →
E6 if targeting a top venue.
