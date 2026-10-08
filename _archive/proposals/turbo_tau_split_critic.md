# Turbo τ-split critic — dual fake LoRAs specialized by noise level

Status: **CLOSED 2026-07-20 — Phase 0 verdict: G1_FAIL at peak LR; Phase 1
never gated.** Findings write-up (the durable record):
`docs/findings/turbo_tau_critic_interference_lr_artifact.md`.

**The P0b gate fired, then failed its own robustness check.** Same bundle,
same seed, only the LR differs:

- At the bundle's **annealed tail LR** (4.04e-6,
  `bench/turbo/results/20260714-2032-p0b-superturbo-B/`): G1 headroom 17–19×
  the ctl/ctl2 noise floor on both bands, G2 symmetric forgetting — verdict
  `FIRE — Phase 1 unlocked`. This is the read the old header cited.
- At the **peak LR the critic actually trains at** (3e-5,
  `bench/turbo/results/20260714-2109-p0b-superturbo-B-peaklr/`):
  `improvement_hi_band` **−6.15e-5** against a noise floor of **3.54e-4**;
  `degradation_lo_arm_on_hi_band` **−6.99e-5** (the lo-specialist *improved*
  the hi band → bands share one solution, G2 inverted). `G1_headroom: false`,
  `G2_forgetting: false`, verdict **`G1_FAIL — close the line`**.

The 17–19× was a linear-regime artifact — at 4e-6 the arms barely leave their
init, collapsing the ctl/ctl2 denominator. Per the pre-registered rule below
(§"If G1 fails"), the line is closed. Caveat: the peak-LR floor exceeds the
tail-LR run's entire signal, so that arm bounds the effect below seed variance
rather than proving absence; reopening means **re-running P0b at peak LR with
more updates**, not a Phase-1 A/B.

- **P0a telemetry KEPT** (`TauBinCriticLoss`, rides every run) — always
  scoped to survive the verdict.
- **Phase 1 wiring ships INERT**: `fake_tau_banks` / `fake_tau_boundary`
  in `configs/methods/turbo.toml [network]` default to `1` / `0.5`;
  `TurboDMDNetwork.fake_hi` + `set_fake_bank`; both routing sites via
  `primitives.sample_t_routed`; warmup routed; resume bundle carries both
  banks (bank-count mismatch refused); per-bank τ profiles. Invariant tests:
  `tests/test_turbo_tau_critic.py`. `banks=1` is byte-identical (same RNG
  stream, second stack never constructed). Wired in commit `9d3a8438`
  *before* the peak-LR probe landed — which is why the tree briefly looked
  like an open line.
- **Phase 1 was NEVER evaluated** — none of the four gates below (rendered
  grids, within-run CMMD, profile-flattening, diversity) were run. The only
  long `banks=2` run is `anima_superturbo_C`, the null NFE=2 arm, confounded
  three ways. Every turbo run since 2026-07-16 is `fake_tau_banks = 1`.

## Premise

The fake/critic LoRA has the hardest job in the DP-DMD loop: model the
student's *entire* output distribution across the *whole* τ range with one
rank-96 LoRA, while that distribution moves every step. The student, by
contrast, only has to represent a few-step denoising map at 2–4 fixed σ's.
When the critic underfits, the repulsive half of the DMD gradient
(`delta_dm = v_real − v_fake`, `scripts/distill_turbo/distill.py:971`) is
miscalibrated exactly where the student misbehaves — the "fake teacher exerts
insufficient repulsive force" failure that AMD (arXiv:2602.07345) names the
*Forbidden Zone* and addresses with reward-weighted critic focus (their
"Repulsive Landscape Sharpening"). We have no trustworthy reward model for
this domain ([[project_shelved_explorations]]: no quality reward exists), but
the reward-free analogue of "critic must specialize where it's weak" is
**capacity specialization by τ**: two fake LoRAs, one owning high-σ, one
low-σ, each updated and queried only on its band.

Why this could be free: τ-routing partitions the existing
`fake_steps_per_student_step` updates instead of adding any — each bank sees
~half the data but has ~half the task. Total critic compute is unchanged by
construction, which makes the Phase-1 A/B inherently matched-compute
(the [[project_spectrum_sea_schedule_prompt_gen]] lesson: reallocation claims
are only meaningful at matched compute).

Why it could also do nothing: the student-side analogue of "more/split
capacity" has repeatedly not been the lever ([[project_superturbo_nfe2_line]]:
subspace-lock refuted, rank ≠ lever). If the single critic's per-band error
is *not* dominated by cross-band gradient interference, splitting it buys
nothing. That is a cheaply measurable premise — hence the Phase-0 gate below.

**What this line is NOT for**: basin/mode selection. The critic re-tracks the
student within tens of updates, so critic-side changes shape stability and
gradient quality *within* the basin the student init picked
(cos(S, turboV10) ≈ 0.002 — init picks the mode). Don't evaluate this arm on
"did asvd96 converge toward the _S family" — it can't.

## Phase 0 — measure the premise before building anything

Two parts; **P0b is the gate**, P0a is telemetry that ships regardless.

### P0a — τ-binned critic-loss telemetry (~free, keep forever)

Bucket the per-update fake loss (the inner loop at `distill.py:1068`) by the
drawn `tau_fake` into 8 uniform bins; log per-bin interval means as
`train/fake_loss_tau{i}` (TB smoothing recovers the EMA view). Same for the
warmup loop (`warmup/fake_loss_tau{i}`). Piggyback on the next run — no
dedicated run needed.

> **Implemented**: `TauBinCriticLoss` in `scripts/distill_turbo/metrics.py`
> (GPU-resident, one extra sync per log boundary), wired at
> `distill.py:1092`/`:1202` and in `warmup.py::run_fake_warmup`; unit tests in
> `tests/test_turbo_metrics.py`.

> **Trap, pre-registered**: raw per-τ loss is structurally nonuniform — the
> FM target `ε − x0` is intrinsically harder to predict at some τ regardless
> of critic capacity, so a lumpy profile is NOT evidence of a capacity
> deficit. P0a is a *baseline profile* for the Phase-1 mechanism check
> ("does the split flatten the *excess*?"), never the gate.

### P0b — specialization-headroom (interference) probe — THE GATE

`bench/turbo/tau_critic_probe.py` (implemented; `bench/_common.py` envelope).
Offline, no training-loop changes. Uses a crash-resume bundle from a real run
(the bundle carries the fake weights + Adam moments —
`scripts/distill_turbo/resume.py`; the probe restores BOTH per arm, so each
clone is exactly "the run continues, τ restricted"). Substrate:
`output/ckpt/anima_superturbo_B/anima_superturbo_B_resume.pt` — _B rendered
better than _C, and the probe should measure interference in the best
student's regime. Run:

```
python bench/turbo/tau_critic_probe.py \
    --bundle output/ckpt/anima_superturbo_B/anima_superturbo_B_resume.pt
```

Design — a controlled fine-tune that cancels the structural τ-dependence:

1. Load student + fake from the bundle; freeze the student. Build a fixed
   held-out probe set: ~256 tuples `(x_pred, τ, ε)` with x_pred generated by
   the frozen student, τ stratified over 8 bins, fixed seed.
2. Clone the trained fake **four** ways and fine-tune each for N=400
   fake-only updates (the `run_fake_warmup` machinery, student frozen), same
   data stream, same LR (current post-warmup value, no schedule):
   - **lo**: τ drawn only from [0, 0.5)
   - **hi**: τ drawn only from [0.5, 1]
   - **ctl**: τ uniform [0, 1] (the status quo)
   - **ctl2**: same as ctl, different seed → the noise floor for every
     comparison below.
3. Measure per-bin probe loss before/after for each clone.

Pre-registered gate (fires → Phase 1 unlocked):

- **G1 (headroom)**: `lo` improves its own band's probe loss over `ctl` by
  more than the `ctl` vs `ctl2` spread, AND same for `hi` on its band.
  Band-restricted training beating uniform training *on the restricted band
  at equal update count* is the definition of cross-band gradient
  interference — the only mechanism a τ-split can harvest.
- **G2 (sanity)**: `lo` degrades on the hi band (and vice versa). If the
  restricted clones *don't* forget the other band, the bands share one
  solution and the split is pointless even if G1 numerically passes.

If G1 fails: **close the line** and record it next to the student-side
"rank ≠ lever" finding. The telemetry (P0a) stays.

Optional third arm while the probe is open (cheap, sharpens the writeup
either way): repeat with the fake truncated to r48 (`warm_start_plain_lora`
SVD-truncates to any rank, `networks/methods/turbo_dmd.py:234`). If r48 ≈ r96
per-band, capacity is slack and a negative G1 is unsurprising; if r48 is
clearly worse, capacity binds and a negative G1 means the binding constraint
is something else (data freshness, LR), which is worth knowing.

## Phase 1 — `fake_tau_banks = 2` (only if P0b fires)

### Wiring (the real seams)

- **`networks/methods/turbo_dmd.py`** — `TurboDMDNetwork` grows an optional
  second fake stack (`fake_hi`); runtime chain becomes
  `linear → fake_hi → fake_lo → student → original` (same additive
  `apply_to` chaining, `:393-406`). Bank selection extends the existing view
  mechanism: a `set_fake_bank(i)` that `set_view("fake")` consults, toggling
  `enabled` per stack exactly like today's teacher/student/fake switching
  (`set_view`, `:483`).
  - **Compile note**: `enabled` is a Python attr read inside
    `LoRAModule.forward`, i.e. a dynamo guard — today's 3 views are 3 graph
    specializations per block; a second bank adds a 4th. Bounded and
    identical in kind to what already runs, but budget for it: the pinned
    recompile limit is a ContextVar and
    the cache is isolated per run.
    No new buffer machinery needed — do NOT invent a fused dual-weight module
    unless recompile counts actually blow up.
- **`scripts/distill_turbo/distill.py`** — two routing sites, both by the
  scalar τ of the batch:
  1. the fake update loop `:1068-1091` (`tau_fake` → owner bank trains);
  2. the DMD-gradient fake query `:962-971` (`tau_dm` → owner bank
     answers). Routing only the updates and letting one bank answer all
     queries would train a specialist and then ignore it.
  - v0 requires `batch_size == 1` (the shipped turbo config): τ is per-sample
    and B=1 makes it a scalar. Assert loudly; B>1 would need a split-batch
    double forward — out of scope.
- **Optimizer**: keep ONE `fake_opt` over both banks' params. With
  `zero_grad(set_to_none=True)`, the inactive bank's grads are `None` and
  AdamW skips them; Adam's per-param `step` state keeps bias correction
  correct under sparse updates. Total fake updates are unchanged, so
  `fake_sched` sizing (`distill.py:447-451`) is untouched.
- **Warmup** (`warmup.py::run_fake_warmup`): route each head-start update by
  its drawn τ — both banks enter the main loop calibrated on their own band.
  Consider `fake_warmup_steps` 200 → 400 when banks=2 (each bank gets ~200).
- **Warm start**: `fake_init_weights` seeds BOTH banks from the same file
  (SVD-truncated per bank rank). Keeps the matched-critic-at-init property.
- **Resume** (`resume.py`): bundle carries both banks + the shared opt state.
  Bump the bundle schema tag so old bundles refuse rather than half-load.
- **Save**: unchanged — both fakes are discarded; output stays a plain LoRA.
- **Metrics**: `train/fake_loss` per bank + the P0a per-bin profile per bank.

### Config (`configs/methods/turbo.toml` `[network]`)

| key | default | note |
|---|---|---|
| `fake_tau_banks` | `1` | **1 = second stack never constructed → byte-identical loop** |
| `fake_tau_boundary` | `0.5` | raw-τ split point; uniform `t_distribution` → even update split |

Bank ranks reuse `fake_rank`. **Do not pre-shrink to r64 for VRAM**: the
second bank's marginal memory is params+grads+Adam moments only
(≈ 8 bytes/param at bf16; read `n_fake` from the startup log, `distill.py:422`)
— activations don't double since one bank is enabled per forward. Measure
first. If VRAM does force r64, the Phase-1 arms must include a single-r64
control (below), or rank confounds the verdict.

### Decision gate (Phase 1)

A/B at fixed seed/data/iterations, total critic updates equal by
construction. Arms: single-r96 (shipped) vs dual-r96; if r64 was forced,
add single-r64. Ship only on all four:

1. **Rendered grids at the student's NFE** (`--infer_steps` = `student_steps`,
   `--cfg 1.0`) — the primary ranking, per
   [[project_turbo_lr_instability_threshold]] (rank by rendered output, not
   scalars).
2. **Within-run CMMD non-regressing** ([[project_cmmd_val_signal]] — trend
   only, never cross-run).
3. **Mechanism check**: the dual arm's per-bank excess over the P0a baseline
   profile visibly flattens. A quality win *without* this is a lucky seed,
   not the mechanism — treat as unproven.
4. **No diversity collapse** (`scripts/distill_turbo/diversity.py` grids —
   pose diversity is the known turbo failure axis,
   [[project_turbo_teacher_gap_2026_06_29]]).

### Invariant tests (Tier 1.5 requirement)

`tests/test_turbo_tau_critic.py`:
- `fake_tau_banks=1` constructs no second network and leaves the update/query
  paths byte-identical (compare a few steps of fake-loss values against a
  pinned pre-change run at fixed seed).
- Degenerate boundary: `banks=2, boundary=1.0`, both banks warm-started from
  the same file → bank-lo answers/trains everything; probe-batch outputs
  match a single warm-started critic. (Functional equality, not RNG-stream
  equality — constructing the second network advances the init RNG.)
- Config validation: `banks=2` requires `batch_size=1`; boundary ∈ (0, 1].

## Cost

- Phase 0: metrics patch + one bench script + ~4×400 fake-only updates
  offline (minutes-to-hours on one GPU, no training run consumed).
- Phase 1: zero extra per-step compute (routing partitions existing updates;
  the DMD query cost is identical). Memory: +1 fake LoRA's params/grads/Adam
  states, measure via the `n_fake` log line. One extra dynamo graph
  specialization per block.

## Sequencing

```
P0a τ-binned telemetry (ships regardless, rides next run)
P0b interference probe on a resume bundle
   ├─ G1/G2 fail → CLOSE (record beside student-side "rank ≠ lever")
   └─ fire → Phase 1: fake_tau_banks=2 (off-by-default)
         └─ matched A/B: grids + CMMD + profile-flattening + diversity
               └─ ship / kill
```

## Contributing tier

Phase 1 changes training numerics → **Tier 1.5**: bench script
(`bench/turbo/tau_critic_probe.py`, which doubles as the standing probe) +
invariant test (`fake_tau_banks=1` byte-identical).

## References

- AMD, "Optimizing Few-Step Generation with Adaptive Matching Distillation"
  (arXiv:2602.07345) — the Forbidden-Zone framing; §3.3 Repulsive Landscape
  Sharpening is the reward-weighted version of "critic must specialize where
  the student misbehaves". This proposal is the reward-free, τ-structured
  analogue (their reward-proxy machinery is inapplicable here — no domain
  reward model, and their Fig. 7 mode-concentration behavior is
  anti-diversity, which DP-DMD exists to prevent).
- `scripts/distill_turbo/distill.py:962-971` (DMD fake query),
  `:1068-1091` (fake update loop), `:422` (`n_fake` log), `:447-451`
  (fake scheduler sizing).
- `networks/methods/turbo_dmd.py:288` (`TurboDMDNetwork`), `:393-406`
  (additive chain), `:483` (`set_view`), `:234` (`warm_start_plain_lora`
  SVD-truncation).
- `scripts/distill_turbo/warmup.py` — head-start loop the probe reuses.
- [[project_superturbo_nfe2_line]] — the student-side capacity null result
  this line must not silently contradict.
- [[project_turbo_R_plateau]] — the standing lever list (f-distill /
  identity aux / per_step_expert) this arm competes with for run budget.
