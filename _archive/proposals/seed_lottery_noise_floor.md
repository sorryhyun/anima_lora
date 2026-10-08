# Seed-lottery noise floor — a robust A/B framework for Anima

Status: **CLOSED 2026-07-12 — blocked on the instrument; probe archived unrun**
(see the closing section at the bottom). Motivated by *The FID Lottery:
Quantifying Hidden Randomness in Generative-Model Evaluation* (Dufour, Efros,
Pérez, arXiv:2606.20536) — paper PDF in repo root `2606.20536v1.pdf`.

## The problem this fixes

We make a *lot* of method calls on **single training runs** with small effect
sizes — REPA relational-vs-absolute "won ~6:4–7:3" and the layer-8-vs-26 anchor,
channel-scaling quality "never A/B'd",
chimera content-half gate std 0.004,
turbo lr "non-monotonic ckpt quality 1k>4k>2k>3k".

The FID Lottery paper measures, on several hundred SiT models, that **retraining
the same recipe with a different seed moves FID 3.2× more than resampling a fixed
model**, and that this between-seed spread sits on a **scale-invariant ~1.3% CoV
floor** that does *not* shrink with compute or model size — because the per-step
flow-matching noise (the largest single source, ~77% of the spread; init ~67%,
data-order ~51%) is regenerated every batch and never averages out. A
"non-monotonic quality curve" across runs is the lottery's *signature*, not
necessarily a real effect.

Two of their findings we already arrived at independently — FM val loss doesn't
track quality so we moved to CMMD, and we rank turbo by rendered samples not fm_mse.
But we have **never measured our own floor**, so we can't tell which banked A/B
conclusions are real and which are inside it.

**We are exposed worse than the paper's authors.** They hold ImageNet fixed
across ~100k GPU-hours. We change the *data pool* constantly — new artists,
re-preprocess, free-fit tier changes,
caption edits. The paper's D.3 factorial
(10 init seeds × 15 data/noise pairings) shows "good init" transfers only weakly
even under data-*ordering* changes (avg pairwise Spearman ρ = 0.36; same init
top-3 under one pairing, bottom-3 under another). Changing the data *pool* is a
strictly larger perturbation → **a lucky seed is not bankable across our dataset
iterations, and two A/Bs run weeks apart are confounded by method × seed × data
pool all at once.**

## The one load-bearing principle

**A gap is only real if it survives reseeding.** Everything below serves one
output: a per-experiment **noise floor** (the between-seed CoV of our pipeline),
and a verdict on every reported A/B — `REAL` / `INCONCLUSIVE (inside floor)`.
This is a *measurement and reporting* discipline, not a new training method.

## How the paper's two-axis panel maps onto our actual code

The paper's `N × K` panel is `N` training seeds × `K` sampling seeds. Our seeding
surface already cleanly separates the two axes — no new knobs needed for v1:

| Paper axis | Drives | Our control |
|---|---|---|
| **Training lottery** (`N`) | init + data order + per-step FM noise | `--seed` → `set_seed(args.seed)` (`train.py:1973`); already bundles all three exactly as the paper's "training seed" does. `args.seed=None` → random (`train.py:1971`). |
| **Evaluation lottery** (`K`) | the sampling noise of CMMD validation | `validation_seed` (`configs/base.toml:81`, currently fixed `42`) + the CMMD `validation_split_num` path. |

So `σ_between` = spread of per-seed mean CMMD across `N` values of `--seed`;
`σ_within` = spread across `K` validation seeds for one fixed checkpoint. The
floor we care about for A/Bs is `σ_between`; `σ_within` tells us whether our
selection metric can even *see* between-seed differences.

## Why a frozen-DiT LoRA floor is genuinely unknown (not just "assume 1.3%")

The paper's 1.3% is full-model SiT-on-ImageNet. A LoRA on a **frozen** DiT is a
different variance regime, and it could go either way:

- **Lower:** the frozen backbone removes the backbone-init source (the paper's
  ~67% contributor). LoRA init is just the small A/B matrices (B=0, A~gaussian) —
  far less init entropy.
- **Higher:** ~200 images is a small, high-variance regime; the per-step FM noise
  (the dominant source) is fully present and a short run sees each image few
  times, so data-order and noise interact more violently than at ImageNet scale.

The number must be **measured**, and it is *per dataset class* — a 200-image
artist LoRA, a colorize EasyControl run, and a turbo distill almost certainly
have different floors. The framework measures it where you're about to make a
claim, not once globally.

## Phase 0 — does CMMD even separate seeds? (the gate)

Before any of this is worth building, one question decides everything:
**is `σ_between` larger than `σ_within`?** If the between-seed signal is buried
under sampling noise, selection is hopeless and the answer to "how many seeds"
is "one — seed-fishing is pointless, just train and ship." This is the real risk
flagged by the null-TTA conclusion that **no reliable quality reward exists for
Anima** and that CMMD is a
"global-tone lever" — CMMD may simply not rank LoRAs by the quality we care about.

**Run:** pick one representative config (suggest plain LoRA on a stable ~200-image
artist set). Train **N=5** seeds (`--seed 0..4`), each via the daemon
(`make lora --queue`, [[project_daemon_wiring_pattern]]) so they pipeline. For
each checkpoint, evaluate CMMD over **K≥5** validation seeds.

**Pre-registered gate:**
- Compute `σ_between` (across the 5 per-seed mean CMMDs) and `σ_within` (median
  across-validation-seed σ for a fixed checkpoint).
- **GATE FIRES (build Phase 1)** iff `σ_between / σ_within > ~1.5` *and* the
  rendered-sample ranking of the 5 seeds is not visibly random (same
  sample-grid discipline we trust for turbo). Report the CoV `σ_between / μ` as
  the headline floor number.
- **GATE FAILS** → write the finding ("CMMD cannot resolve our seed lottery; A/Bs
  on this pipeline need rendered-sample adjudication or a better reward, not more
  seeds"), and the framework stops here. That negative result is *itself* worth
  the run — it would retroactively flag every CMMD-ranked single-run A/B as
  unverified.

**Deliverable:** `bench/seed_lottery/probe.py` writing the standard envelope via
`bench/_common.py::write_result` (schema_version, git sha, env auto-captured),
metrics = `{sigma_between, sigma_within, cov_between, cov_ratio, n_train, k_val,
per_seed_cmmd: [...]}`. Drops into `bench/seed_lottery/results/<TS>/`.

## Phase 1 — the floor as a reusable A/B verdict

Only if Phase 0 gates. Two thin deliverables, no new training infra:

1. **`floor` block in the bench envelope.** Extend `write_result`'s `extra=` (or a
   first-class optional field) so any bench comparing two configs can attach
   `{"floor_cov": <measured>, "delta_cov": <observed gap>, "verdict":
   "REAL"|"INCONCLUSIVE"}`. The rule, straight from the paper: a gap **below the
   measured CoV** is `INCONCLUSIVE` and must not be reported as a win without
   multi-seed confirmation. This is the spectral-fraction-metric-inverts lesson generalized: "is this gap above
   the noise floor?" becomes a standard question every `result.json` answers.

2. **`make seed-floor METHOD=… PRESET=… DATA=…`** — a task wrapper that runs the
   Phase-0 probe on demand for a given pipeline, so the floor is re-measurable
   whenever the data pool changes (which, for us, is the trigger that invalidates
   the previous number). Mirrors the bespoke-loop wiring pattern — turbo/spd
   loops won't get it free; note as a known
   gap, not silent.

## How many seeds — the order-statistics answer (for the doc/help)

The paper confirms the per-seed distribution is **Gaussian, not heavy-tailed**
((max−min)/σ ∈ [3.1, 5.0]). So picking best-of-K (given a *reliable* ranker) buys
the expected-max-of-K-normals gain over a single draw:

| K | P(best beats mean) | E[gain] over avg |
|---|---|---|
| 2 | 75% | 0.56 σ |
| 3 | 87.5% | 0.85 σ |
| **4** | **94%** | **1.03 σ** |
| **5** | **97%** | **1.16 σ** |
| 10 | 99.9% | 1.54 σ |

**Recommendation: N=5** — it matches the paper's own detectability threshold
(`2σ/√N` ⇒ N=5 resolves a ~0.25-FID gap) and sits exactly where order-statistics
returns flatten (5→10 adds only ~0.4σ for double the compute). **N=3** is the
budget floor for "just beat average." Past 5 is for chasing records, not robust
selection. **Caveat that dominates the table:** every number assumes you can
*identify* the best of K — i.e. Phase 0 gated. If `σ_within ≳ σ_between`, the
realized gain collapses toward zero regardless of K.

## Explicit non-goals

- **Not a new training method, optimizer, or loss.** Pure measurement/reporting.
- **No per-step DB writes / no training-hot-path coupling** — the probe reads the
  CMMD *summary* at run end, same as the run-ledger discipline
  (`sqlite_run_ledger.md`); if that ledger lands, `per_seed_cmmd` rows live there
  naturally.
- **Not claiming a universal constant.** The floor is measured per (method,
  preset, data pool) and re-measured when the pool changes — the whole point is
  that it does *not* transfer.
- **DDP/numerical noise not chased** — the paper shows it's below the sampling
  floor (σ_between collapses to 0.047 even with EMA weights 5–6% apart); not a
  meaningful source, so we don't isolate it.

## Why now / why not

**For:** it's cheap relative to what it protects. One N=5 probe (~5 short LoRA
runs, pipelined via `--queue`) tells us whether *any* of our banked CMMD-ranked
A/B conclusions are trustworthy, and gives a reusable `REAL/INCONCLUSIVE` stamp
for every future bench. Given how often we change the data pool, this is the
single missing guardrail against confounded cross-time comparisons.

**Against (honest):** if Phase 0's gate fails — plausible given the "no reliable
quality reward for Anima" finding — the framework can't *select* seeds, only
*flag* uncertainty. That's still useful (it prices the noise into every claim),
but it's a smaller win than "train 5, pick the best." So the real deliverable is
the Phase-0 measurement; Phase 1 is contingent on it gating. Worth a verifier
pass before building Phase 1, since "CMMD ranks LoRAs by quality" is exactly the
kind of plausible-but-unproven premise this project has been burned by before.

## Open questions

- Is one global floor per *method* enough, or does it drift with dataset size
  (200 vs 2000 images) enough to need re-measurement per dataset? Phase 0 on two
  dataset sizes answers this cheaply.
- Selection denoising: averaging K validation seeds shrinks `σ_within` by √K —
  is there a cheaper variance-reduction on the eval side (fixed sampling-noise
  across seeds, paired evaluation) that sharpens selection without 5× the CMMD
  cost?
- Should the floor verdict gate `make test-unit` / CI for new methods (Tier
  1.5/2 in CONTRIBUTING already require a bench + invariant test)? Natural home,
  but only after the floor number is trusted on more than one pipeline.

## Closing (2026-07-12) — the gate question was answered before Phase 0 ran

The Phase-0 probe (`_archive/bench/seed_lottery/probe.py`) was built but never
run: `bench/seed_floor` (now `_archive/bench/seed_floor/report.md`) answered
the gate question for free — using the ready-made uncond-soup seed trio instead
of training new checkpoints — and answered it **negatively at the instrument
level**. At the eval size we can afford (24 prompts / 96 refs), paired-holdout
CMMD cannot separate between-seed spread from its own sampling noise: several
checkpoints (plain, s1002, the real-vs-real floor itself) cluster within ~1.1×
of each other, the cfg-1 "s1002 = 1.02 disaster draw" flipped to *best* at
cfg-4, and median-heuristic bandwidth changed nothing (on L2-normalized
features MMD ordering is bandwidth-invariant — the fragility is sample size,
not σ). Running this probe as designed would have measured noise with a noise
generator.

The paired-eval line (`paired_gram_eval.md`) — the designated instrument fix —
then failed its own Phase 0 (token-Gram is content-dominated) and was archived
the same day. So the framework is closed **blocked-on-instrument**, not
refuted: the question (which banked single-run A/Bs are real?) stands, and the
durable output is a guard, not a floor number:

- **Never gate an A/B on a paired-holdout CMMD scalar at n≈24/96;** treat any
  gap under ~2× the real-vs-real floor as unresolved.
- **cfg-1 CMMD misranks within a family across seeds**, not just across
  adapter families (the reft GOTCHA generalizes). Rendered comparisons that
  must rank models use 28-step / CFG-4.
- CMMD stays fine as the *coarse within-run* val signal — the fragility is seed-level resolution at
  small n, not the metric wholesale.

**Reopening gate:** a per-item paired metric that clears a plain-vs-base
known-difference gate (CSD was the named next candidate), or an eval budget in
the hundreds of prompts/refs. Given that soup's value proposition moved to the
memorization axis (interference × amplification — `bench/memorization` is the
live bench there), neither is currently scheduled.
