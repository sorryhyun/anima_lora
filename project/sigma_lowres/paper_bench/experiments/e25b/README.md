# E25b — explicit resolution conditioning (the adaln micro-conditioning arm, frozen)

| | |
|---|---|
| **Status** | **PRE-REGISTERED (FROZEN) 2026-08-11** — the E25 sketch's arm (b), frozen per roadmap §1(d) after E28-F1 landed the rationale fact (see Question). Zero GPU is spent by this registration: Stage 0 (instrument) is CPU/code work, **Stage 1 (~10 GPU-h) and Stage 2 (full-scale ship read) each require an explicit go**. Submission order: Stage 1 waits for E30's 30-B slice read to land (context row, NOT a gate — see "30-B relation"). |
| **Question** | The line knows the demoted-step residual is angle-borne (E24.2), locally enforced (E21), with the phase-response **amplitude carried 86–87 % by the adaln band** (E21, descriptive), and — since E28-F1 — that a **conditioning input demonstrably organizes the axis field** (MISMATCH-CARRIED: sign(σ_noise − σ_cond) is the organizing variable, tested at two pin values). E25b asks the lever form of that fact: if the resolution dimension the network currently absorbs *implicitly* is made an **explicit conditioning input** on the adaln trunk (micro-conditioning precedent: SDXL size/crop; in-architecture precedent: `enable_fps_modulation` in `library/anima/models.py`), does the demoted-step cancellation geometry improve — **h(B+C) shrinks** (primary; ρ deepening reported alongside) — at the trained arm's own operating point? |
| **Licensed by** | E24 STRUCTURED (σ-local lever mandate); E24.2 (target = angle/residual, never amplitude rebalancing); E21 (adaln amplitude concentration — the injection-site prior); E28-F1 MISMATCH-CARRIED + E29 NATIVE-SMOOTH (conditioning input organizes the field — the roadmap-(d) rationale fact); E25.0 (pre-registered as not gating E25b: 25.0-1 PARTIAL and 25.0-2 NO-GAIN touch only E25a's lookup). |
| **Explicitly NOT this** | **The E20.4-adjacency paragraph (the freeze requirement) is below** and is part of this table's contract. Not per-sample (E22 → 22.4 → E23a unchanged): the conditioning input is a *known per-step discrete scalar* (the route), not an estimated quantity — there is no estimand anywhere in the training path. Not E25a (no lookup, no gradient projection, no probe-built object enters training). No PI/RoPE change (G11; yarnsig is untouched). No σ_cond probe-pin interaction (E28-F1's `--cond_sigma` seam is a *probe* intervention; E25b touches the *trainer*). Not a paper-1 item (revision_plan §8); paper-2/paper-3 material per roadmap §2.B. |
| **Depends on** | E4/E16 deterministic twin harness (tenth scale, 480 steps, bs 1, hews, `--deterministic --paired_step_rng`); `run_sigma_probe.py` + `bench/sigma_probe/` (the probe); `vector_ledger.py` h/ρ conventions (`h()` at vector_ledger.py:207); E26 frozen per-bin criteria (readable: rel_B ∧ rel_C ≥ 0.5; passing: I < 0, ρ ≤ −0.5, h(B+C) < min(h(B), h(C))); T0 boot-family law (roadmap §3). |
| **In the paper** | Nothing in the paper-1 revision. Paper 2: the lever-form consequence of the conditioning-organizes-the-field fact. Paper 3 (roadmap §2.B): if it lands, the generalization is "conditioning-injection absorbs a compute-graph substitution, certified at the gradient level". |

## The E20.4-adjacency paragraph (freeze requirement, e25 kill criterion 3)

E20.4 closed the **derived-data-term family**: replacing a measured
account input with a ledger-derived quantity, or re-deriving any
ledger term as an objective correction — the failure was
estimand-level. E25b is outside that family on three grounds, stated
so the distinction is checkable: (1) **nothing enters the objective**
— the loss, target, σ-draw, and demote router are bit-unchanged; the
lever is a forward-pass *input*, trained by ordinary backprop like the
timestep embedding it rides next to. (2) **No ledger quantity appears
anywhere in the training path** — the injected scalar is the route's
known scale (a config fact), not B, C, ρ, an amplitude curve, or any
probe-built object; the ledger appears only in the *readout* (the
Stage-1 probe), exactly as it does for every trained-arm comparison
the line has run. (3) **No estimand bridge is required** — E20.4's
failure was that the ledger legs are not the account's x(σ); E25b
makes no claim connecting them: its prediction is a paired scalar
comparison (twin arms, same instrument), not a derivation. If a future
amendment ever proposes feeding a measured/estimated quantity (rather
than the known route scale) into the embedding, that amendment
re-enters the closed family and must not run.

## Frozen design (Stage 0 — instrument; no GPU)

**Trainer** (`--sigma_lowres_res_cond`, opt-in, default off):

- **Input scalar**: s = log2(step_edge / native_edge) of the grid the
  step *actually trains on* — 0 on native steps, log2(896/1024) ≈
  −0.193, log2(768/1024) ≈ −0.415 on demoted steps. Known per step
  from the router's decision; no estimation.
- **Embedding**: the model's own sinusoidal timestep-embedding
  function applied to s (dim 256), projected by one
  `nn.Linear(256, t_emb_dim, bias=False)` initialized to **zeros**,
  added to the timestep embedding output before the adaln modulation
  trunk (the same summation point fps modulation uses). Trainable
  params: the projection only (~0.5 MB fp32).
- **Zero-init identity invariant**: with the projection at init (or the
  flag off), the forward is **bit-exact** to a control arm — pinned by
  a new test in `tests/test_sigma_lowres.py` alongside the yarnsig
  identities. Checkpoint carries the projection under a dedicated key
  prefix + `ss_sigma_lowres_res_cond` metadata; `make merge` refuses
  checkpoints carrying it (the projection is not a ΔW).
- Native steps run the module too (s = 0) — the conditioning is a
  total function of the step, not a demoted-branch patch, so the
  network can place native as a point on the resolution axis rather
  than an unconditioned default.

**Probe extension** (`--res_cond` on `run_sigma_probe.py`): when the
`--adapter` checkpoint carries the res-cond keys, install the trained
projection and apply, per arm, the embedding of the grid that arm
actually runs at — native / redraw-floor / reenc arms get s = 0, the
demote arm at edge e gets s = log2(e/1024). That per-arm application
*is* the explicit-conditioning semantics under probe; a control
checkpoint (no keys) with `--res_cond` is a setup error (hard fail,
not silent no-op). Instrument diff reviewed against this paragraph
before any Stage-1 submission.

## Stage 1 — the geometry gate (~10 GPU-h; go required)

**Training arms** — E16-pattern deterministic seed-twins at E4 tenth
scale (480 steps, bs 1, hews corpus, stock **combo** recipe: stacked
router 768@σ∈(0.65,0.95) / 896+yarnsig@σ>0.5, `--deterministic
--paired_step_rng`, seed 42), identical argv except the flag:

- **ctrl** — combo verbatim (the shipped recipe, E16.1's combo arm).
- **rescond** — combo + `--sigma_lowres_res_cond`.

Zero-init means the pair starts bit-identical; every divergence is
gradient-borne through the projection. Both runs daemon-queued
back-to-back.

**Probes** — one `run_sigma_probe.py` run per arm, back-to-back in
one boot (T0; if a reboot splits the pair, both probes resubmit):
route **768 only**, segmented window
`0.23333333333333334,0.7666666666666667,4 : 0.7666666666666667,0.9,1`
(verdict bins {0.3, 0.4333, 0.5667, 0.7, 0.8333} — the E26/E28 grid),
standing 40-stem probe list, 12 draws/bin, `--repromote
--keep_arm_sums --self_floor --deterministic --seed 42`, fp32 sums;
the rescond probe adds `--res_cond`. All verdict reads are
**scalars** (h, ρ, rel — boot-portable within Δρ ≈ 0.03–0.05); no
cross-adapter vector claim of any kind (the E26 frame lesson).

### Validation gates (before the verdict row is read)

1. **Twin-start identity**: rescond step-0 checkpoint ΔW ≡ 0 vs ctrl
   (zero-init invariant, in vivo).
2. **Ledger reproduction**: within-condition scalars through the
   independent `bc_ledger` path on all 5 bins of both stores (the e28
   gate-1 convention).
3. **Readability**: E26 frozen readable criterion (rel_B ∧ rel_C ≥
   0.5) on ≥ 4/5 verdict bins in **both** stores; fewer ⇒
   **INCONCLUSIVE-UNRELIABLE**, read nothing else. (Tenth-scale
   adapters are an untested probe operating point — this gate is where
   that risk lands, recorded up front.)
4. **Same-family**: `vector_ledger.assert_same_family` on the probe
   pair (T0; scalars would survive a wall breach, but the pair
   resubmits rather than mixing families).

### Pre-registered reading (25b-1, the discriminator)

Primary estimand: per-bin paired ratio **r_h = h(B+C)_rescond /
h(B+C)_ctrl** over co-readable verdict bins; Δρ = ρ_rescond − ρ_ctrl
reported alongside (the E24.2 coupling: deepening ρ and shrinking
h(B+C) are the same improvement seen from two sides; h carries the
verdict because it is the quantity the lever exists to shrink).
Constants 0.9 / 1.1 are judgment values, recorded as such at freeze.

| outcome (median over co-readable verdict bins) | verdict |
|---|---|
| median r_h ≤ **0.9** AND median Δρ ≤ +0.02 | **IMPROVES** — explicit conditioning absorbs part of the demoted-step residual; Stage 2 (ship read) becomes registrable. |
| median r_h ≥ **1.1** OR median Δρ ≥ +0.05 | **WORSENS** — the conditioning input disrupts the geometry; E25b closes (record; no Stage 2; the paper-2 sentence states the negative). |
| otherwise | **NULL** — inside the drift/judgment band; E25b closes as no-effect-at-this-scale unless a specific amendment argues a bigger training scale, which then re-registers Stage 1 at that scale. |

Descriptive rows (no verdict weight): per-bin r_h and Δρ profiles
(does any improvement concentrate at the deep-σ bins where ρ deepens
on every adapter — the E26 shape); rel_R of both stores vs the E25.0
hole at 768/σ = 0.4333; h(B)/h(C) ratios (does conditioning move a
*leg* or only the residual); ctrl-vs-sincos scalar comparison (context:
where a tenth-scale combo adapter sits relative to the line's standing
operating point).

### 30-B relation (context, not a gate)

E30.1's 30-B measures whether the adaln *slice* carries the
cancellation geometry in base-weight frame. E25b's site choice does
not stand on that result — the t-embedding→adaln trunk is where the
architecture takes conditioning inputs (timestep, fps), full stop —
but the rationale paragraph of any Stage-2 registration must cite
30-B's outcome either way (SUFFICIENT strengthens the site story;
INSUFFICIENT means the lever's site is chosen by architecture, not by
the base-frame geometry, and the wording must say so).

## Stage 2 — ship read (gated on 25b-1 IMPROVES + explicit go; own amendment)

Not frozen here beyond its shape (it re-registers with constants when
licensed): full-scale E16.1-pattern grid (3 seeds × 2 artists,
rescond vs combo), quality gate = the E4 5-arm seed-lottery render
yardstick (non-regression: rescond inside the per-corpus band),
throughput gate = wall-clock at fixed steps within **+1 %** of combo
(the lever must not spend the −18.3 % win), plus the Tier 1.5
packaging (bench script + invariant test per CONTRIBUTING.md — the
Stage-0 tests and the Stage-1 bench script are the basis). Output on a
ship decision would be an ordinary LoRA + the res-cond projection keys,
loader-gated exactly like the training flag.

## Kill switches / honesty

- One embedding form, one injection site, no sweep — any variant
  (learned Fourier scales, per-block projections, demoted-only
  conditioning) is a new amendment with its own multiplicity
  accounting.
- The known-input rule (adjacency paragraph) is a standing kill: any
  measured/estimated quantity proposed as embedding input kills the
  amendment proposing it.
- Stage-1 arms are tenth-scale: a NULL at this scale is recorded as
  scale-qualified, not as a universal negative (the branch table's
  NULL row wording is the frozen form).
- Pooled scalars only; per-sample variants stay E23a-gated; wording
  "population-level conditioning change at this operating point".
- Storage per roadmap §3: the two probe stores (~24 GB) live until the
  Stage-1 read commits its tables (`e25b_read.json`), then raw
  arm_sums are reclaimed (manifests retained).
- If thresholds go stale before submission (instrument redesign,
  reboot mid-pair), re-freeze via amendment; no post-hoc renegotiation.

## Cost ladder

| item | cost |
|---|---|
| Stage 0 — trainer flag + probe extension + invariant tests | CPU/code only |
| Stage 1 — 2 × tenth-scale deterministic twins | ≈ 2–3 GPU-h (480 steps, ~1.5× deterministic overhead) |
| Stage 1 — 2 × 768 5-bin probes | ≈ 7.6 GPU-h (e28f1 actuals) |
| Stage 1 — CPU read (`e25b_read.py`) | minutes |
| Stage 2 — full-scale grid + yardstick | own amendment; E16.1-class |
