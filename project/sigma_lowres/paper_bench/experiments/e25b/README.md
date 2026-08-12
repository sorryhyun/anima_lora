# E25b — explicit resolution conditioning (the adaln micro-conditioning arm, frozen)

| | |
|---|---|
| **Status** | **PRE-REGISTERED (FROZEN) 2026-08-11; STAGE 0 BUILT + STAGE 1 RUN + READ 2026-08-12 — 25b-1 IMPROVES** (median r_h **0.572**, median Δρ **+0.005**; 5/5 bins readable both stores, all gates pass; the improvement concentrates at the shallow bins and in the **graph leg** h(C)) — **STAGE 2 RUN + READ 2026-08-12 — 25b-2 FAIL (quality), 25b-3 NULL.** rescond renders below the in-batch yardstick on BOTH corpora (hews −0.013, channel −0.003; near-miss = miss as frozen); throughput was free (paired wall 1.0024 — cost was never the problem); ΔW secondary opposite-direction 12/12; rescue NULL/not-rescued. One consistent story: the conditioning absorbs the substitution per step (25b-1) but converges to a *different* model, not a closer recovery of native. Recorded per the frozen branch — no paper-method update, no ship, flag stays experimental opt-in; paper-2 gets the gradient-level positive with the render-level negative alongside. E25c (inference knob) is independent and unaffected. See "Stage 2 result" below. |
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

## Stage 0 as-built (2026-08-12, commit 96d2eb10) — deviations recorded

- **Injection-site correction**: the frozen design said "the same
  summation point fps modulation uses" — **that summation point does
  not exist**: `enable_fps_modulation` lives only in the rotary
  positional embedding (temporal-frequency rescale) and never touches
  the t-embedding. The as-built site is the **pooled-text-delta seam**
  (`models.py`, after `t_embedding_norm`, before the adaln trunk) — the
  only additive-conditioning precedent on the trunk input. Like that
  precedent, the delta modulates the trunk input but **not** the
  `adaln_lora` bypass term (computed inside `t_embedder` from the
  unmodified sinusoid). Design intent (sinusoid dim 256 → zero-init
  `Linear(256, 2048, bias=False)`, added before the trunk) unchanged.
- **Probe freeze rule** (instrument note): the kernel's flat gradient
  vector is built from `requires_grad` params, so the projection is
  **always frozen under the probe** (with or without `--res_cond`) —
  the rescond store's vector layout stays identical to its control
  twin's.
- Checkpoint carries the projection under the dot-free key
  `sigma_lowres_res_cond_proj` (register_tokens precedent) +
  `ss_sigma_lowres_res_cond` stamp; `make merge` refuses it
  (non-bakeable); factory key-sniff rebuilds the param before
  `load_state_dict`. Zero-init identity + non-degeneracy + route-scalar
  + merge-refusal + probe-hard-fail pinned in
  `tests/test_sigma_lowres.py::TestResCond`.

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

## Stage 1 result (2026-08-12) — 25b-1 IMPROVES

Twins `e25b_hews_{ctrl,rescond}_s42` (jobs 20260812-012606-{43c1ce,c9d225},
E16.1 combo argv verbatim + seed 42; rescond adds the flag; ~6 min each);
probes `20260812-0138-e25b-ctrl-768` / `20260812-0141-e25b-rescond-768`
(jobs 20260812-0138{19-f55622,20-c773cf}, back-to-back one boot; ~3.8 h
each — actual Stage-1 total ≈ 7.7 GPU-h vs the ~10 estimate). Read:
`e25b_read.{py,json}`.

**Gates.** (1) Twin-start identity — in-vivo evidence from five gate-1
micro-runs: first-forward loss bit-identical across
ctrl/ctrl2/rescond/{clip,noclip} (0.421300232410) and a determinism
control (ctrl vs ctrl2: 0/1092 checkpoint keys differ). The micro-runs'
post-training checkpoints diverge on 728/1092 keys — that is the lever
training (the projection has zero value but nonzero gradient at step 1),
i.e. gradient-borne per the registration's premise. Micro-run instrument
note: `--max_train_steps` is overridden by the method config's
`max_train_epochs`, so the micros ran full-length — the loss-stream
evidence is what carries the gate. (2) Ledger reproduction — independent
`bc_ledger` (data_ref reenc) on all 5 bins, both stores (tables in the
read json). (3) Readability — **5/5 bins readable in BOTH stores**
(ctrl rel_B 0.97–0.98 / rel_C 0.85–0.94; rescond 0.91–0.97 / 0.79–0.88):
the registered tenth-scale-operating-point risk did not materialize —
these are the cleanest reliability legs the line has recorded.
(4) Same-family — pass.

**25b-1 (frozen thresholds 0.9 / +0.02):**

| σ | h(B+C) ctrl | h(B+C) rescond | r_h | Δρ |
|---|---|---|---|---|
| 0.3 | 0.2298 | 0.0702 | **0.305** | +0.005 |
| 0.4333 | 0.2917 | 0.0971 | **0.333** | +0.005 |
| 0.5667 | 0.2903 | 0.1660 | **0.572** | +0.011 |
| 0.7 | 0.2203 | 0.1421 | **0.645** | −0.006 |
| 0.8333 | 0.1077 | 0.1016 | 0.943 | −0.004 |

Median r_h = **0.572** ≤ 0.9 AND median Δρ = **+0.005** ≤ +0.02 ⇒
**IMPROVES** — explicit conditioning absorbs part of the demoted-step
residual; **Stage 2 becomes registrable**.

**Descriptives (no verdict weight).** The improvement is **shallow-σ
concentrated** (r_h 0.31/0.33 at σ = 0.3/0.43, decaying to 0.94 at
σ = 0.83) — the lever helps most exactly where the demote gap is
largest, opposite to the deep-σ concentration the E26 ρ-shape might
have suggested. Leg split: **h(C) (the graph leg) shrinks at every
bin** (r_h_C 0.32–0.66) while h(B) is mixed (0.43 at σ = 0.3, then
1.28–1.41 at deep bins) — the conditioning input absorbs precisely the
compute-graph substitution it announces, at the cost of a somewhat
larger data leg at deep σ; the net residual shrinks everywhere
(r_h < 1 at all bins). rel_R (E25.0 hole context): ctrl 0.55–0.70 with
no σ = 0.4333 hole at this scale; rescond 0.35–0.68 (lower where the
residual is smallest — less signal, same noise). Ctrl-vs-sincos
context in the read json (cross-boot scalar caveat): tenth-scale combo
legs are ~3–10× the full-scale sincos legs relative to G, with the
same deep-ρ shape.

## Stage 2 — ship read (REGISTERED (FROZEN) 2026-08-12; licensed by 25b-1 IMPROVES + explicit go)

The pre-registered shape (full-scale E16.1 grid, rescond vs combo,
yardstick quality gate + +1 % throughput gate, Tier 1.5 packaging) is
kept verbatim and re-registered here with constants. **One recorded
extension** beyond the frozen shape, at the user's direction: a
**rescue pair** on the below-yardstick single-route arm (`sigma768`,
E4's 5th arm — named "the rescue target" in E16's depends-on row),
asking whether explicit conditioning widens the certified route
envelope. The floated `combolate` comparator is **not** retrained:
its E16.1 verdict (below the hews band at −14.6 %, i.e. *worse*
throughput than combo) means a rescue there wins nothing the combo
comparison doesn't already decide, and its ΔW-closeness question is
answered in-batch by this grid's own ΔW column (no cross-batch
numeric comparison to the recorded 0.753/0.771 — the chaos-floor and
cross-boot guards forbid it).

**Site rationale (required 30-B citation).** E30.1's 30-B landed
**INSUFFICIENT** (adaln gradient share 0.36–0.42 < 0.5 on both legs):
the base-frame adaln *slice* does not carry the cancellation geometry
on its own. The injection site therefore stands on **architecture,
not base-frame slice geometry** — the t-embedding → adaln trunk is
where this architecture takes its conditioning inputs (timestep,
pooled-text delta), and 25b-1's leg split (the graph leg h(C) shrinks
at every bin) is the site's empirical vindication at the gradient
level. No base-frame claim supports the site; none is needed.

### Stage 2.0 — instrument (CPU/code only; blocks any GPU submission)

The Stage-1 instrument stops at the trainer + probe. **The inference
path does not apply the res-cond delta**: the delta fires only when
`(proj, s)` is attached per forward (`models.py:2191`), which only the
trainer does — and worse, the keep-live adapter loader **filters the
state dict** to `lora_unet_*`/`register_tokens`
(`library/inference/models.py:159,218`), so
`sigma_lowres_res_cond_proj` is dropped before the factory key-sniff
ever sees it. A rescond checkpoint rendered today would silently run
**without** its trained s = 0 offset (native training steps trained
W·φ(0) ≠ 0 — dropping it at inference is off-distribution), which
would corrupt the yardstick read. Stage 2.0 therefore ships, with
invariant tests, before any run is queued:

1. **Loader pass-through**: the keep-live filter admits
   `sigma_lowres_res_cond_proj`; hard fail (not silent no-op) if the
   `ss_sigma_lowres_res_cond` stamp is present but the key is missing.
2. **Inference attach**: when the loaded adapter carries the key,
   every inference forward attaches `(proj, s = 0)` (native-grid
   inference is the trained s = 0 point) via the same try/finally
   idiom the trainer uses. Control checkpoints stay bit-exact
   untouched.
3. **Tests** (`TestResCond` additions): rescond render forward differs
   from a delta-stripped control; control checkpoint bit-exact
   unaffected; inference s = 0 delta bit-exact equal to the trainer's
   native-step delta; stamp-without-key hard fail.
4. Harness = the E4 scripts as-is (`e4_render_eval.py`,
   `e4_seed_yardstick.py`, frozen prompt/gen_seed grid) + a Stage-2
   read script (`e25b_ship_read.py`) emitting the tables below;
   demote-mass accounting via `token_step_hist`.

### Frozen grid (E16.1 protocol)

2 corpora (hews 60-stem / 8 ep; channel\_(caststation) 15-stem /
32 ep) × 3 seeds (1001–1003) × 480 steps bs 1, `--deterministic
--paired_step_rng`, stock lora recipe — identical argv within a pair
except the flag:

| arm | routing | flag | role |
|---|---|---|---|
| native | — | — | yardstick anchor |
| combo | 768@σ∈(0.65,0.95) / 896+yarnsig@σ>0.5 | — | ctrl (shipped recipe) |
| **rescond** | combo | `--sigma_lowres_res_cond` | **25b-2 primary** |
| sigma768 | 768+yarnsig@σ>0.5, no stack | — | rescue ctrl (E4 5th arm) |
| **rescond768** | sigma768 | `--sigma_lowres_res_cond` | **25b-3 rescue** |

Plus one determinism-control duplicate (combo, hews, s1001) = **31
runs**, one daemon batch. The yardstick is **recomputed in-batch**
(cross-seed native~native render cos per corpus) — the recorded
0.9547/0.9541 are context, not the bar (same-env principle). Training
runs may span boots (checkpoints are portable); the render + yardstick
pass runs in **one boot** — if a reboot splits it, all arms re-render.

### Validation gates (before the verdict rows are read)

1. **Twin-start identity**: first-forward loss bit-identical within
   both flag pairs (rescond vs combo, rescond768 vs sigma768), per the
   Stage-1 convention.
2. **Determinism control**: the duplicate combo run's checkpoint is
   key-identical (0 differing keys) to its twin.
3. **Demote-mass accounting**: `token_step_hist` per arm matches the
   configured gate, and realized demoted step sets are identical
   within each flag pair per seed (seed-keyed σ stream, the E4 CRN
   property).
4. **Yardstick sanity**: in-batch native lottery within ±0.02 of the
   recorded 0.9547/0.9541 per corpus — outside that, record the drift
   and proceed on the in-batch numbers (they carry regardless).

### 25b-2 — the ship read (primary)

*(Amended pre-submission 2026-08-12, user decision: the throughput
gate is demoted to a measurement — at this scale ±1 % is ~3 s of wall
against run-to-run noise, and the projection's cost (one 256×2048
matmul per forward) is architecturally negligible; a gate there would
be a coin flip. The paired wall ratio and `token_step_hist` are still
measured and recorded for the arm-table row. Same amendment registers
the ΔW directional read below as a secondary.)*

- **Quality gate (the only ship gate)**: rescond mean within-seed
  render cos vs its native twin **at-or-inside the in-batch yardstick
  on BOTH corpora** (the bar combo passed in E16.1).
- **Throughput (measured, not gated)**: paired per-seed wall ratio
  rescond/combo over the 6 pairs + `token_step_hist`; recorded in the
  read json and the docs arm table. A large regression would be
  visible here and argued about in the open, not silently gated.

| outcome | verdict |
|---|---|
| quality gate passes | **PASS — no product ship (user decision 2026-08-12)**: the result updates the **paper's method treatment** (the lever-form consequence per the In-the-paper row; which paper text absorbs it is decided at write time, not here). The flag stays experimental opt-in, default recipe unchanged; a product ship (docs arm-table promotion + full Tier 1.5 packaging) is a separate later decision with this grid as its evidence. |
| quality gate fails | **FAIL (quality)** — recorded; the Stage-1 gradient-geometry result stands as paper-2 material with the render-level negative alongside. |

One grid, no reruns on a miss; a near-miss is a miss (seed-lottery
wording from Stage 1 carries).

**Secondary (registered directional read, NOT the ship bar): ΔW
toward native.** Per corpus, median over seeds of in-batch ΔW
cos(arm~native twin): direction = rescond > combo on both corpora —
the read that rescond buys for free what `combolate` bought with 4 pp
of throughput. Registered with its limits stated: ΔW closeness ≠
render closeness is a *measured* lesson of this line (`combolate` at
ΔW 0.75 rendered below the hews band on all three seeds; combo at
0.37 rendered inside both — do not read quality off this column), and
the res-cond projection is **not part of ΔW** at all, so this read is
blind to exactly the lever's new component. It can corroborate a SHIP
story; it cannot carry or veto one.

### 25b-3 — the rescue read (secondary; NOT a ship gate)

Margin honesty, recorded at freeze: E4's sigma768 missed the hews band
by **0.0055** and was already inside on channel — band membership at
this margin is near seed noise. The verdict-bearing read is therefore
the **paired within-seed comparison**, which the deterministic twin
design exists to support; band membership is context.

- **Primary estimand**: per-seed paired Δcos = cos(rescond768~native)
  − cos(sigma768~native), within seed, per corpus (6 pairs).
  **RESCUE-DIRECTION** iff median Δcos > 0 on both corpora; NULL
  otherwise.
- **Band claim** (context row, reads only if licensed): if the
  in-batch sigma768 ctrl is below the band on ≥ 1 corpus AND
  rescond768 is at-or-inside on both AND its wall is within +1 % of
  sigma768 ⇒ **RESCUED** (recorded). If the in-batch sigma768 is
  itself inside both bands, the band claim is
  **NOT-TESTABLE-AT-THIS-BATCH** (the lottery moved) and only the
  directional read reports.
- RESCUED does **not** change any shipped default and does not touch
  paper-1 (sigma768 remains its off-map control; the rescue is
  paper-2/3 material per roadmap §2.B). Promoting 768@σ>0.5 to a
  shipped recipe would be its own registration.

### Descriptives (no verdict weight)

- ΔW cos vs native twin per arm — does conditioning move the endpoint
  toward native (the question `combolate`'s schedule bought with
  throughput)? Qualitative context only vs E16.1's scheduling rows; no
  cross-batch numeric comparison.
- ‖W·φ(0)‖ per rescond checkpoint — the size of the trained native
  offset (the stake of the Stage-2.0 wiring).
- CMMD recorded, no verdict at this N (the E4 lesson). Render fig
  sheets per the E4 candidate format.

### Launch record (2026-08-12)

Grid submitted via `e25b_stage2_launch.py` — 31 lora jobs, ids in
`e25b_stage2_jobs.json` (fail-fast cell = `e25b2_hews_rescond768_s1001`,
the never-run flag/route combination: **rc 0**, checkpoint carries
`sigma_lowres_res_cond_proj` + stamp). Stage 2.0 wiring landed the same
day: `_attach_res_cond` in `library/inference/models.py` (loader-gated
persistent `(proj, s=0)` attach after every adapter route's
`lora_unet_*` filter; stamp-without-key hard fail; multi-carrier
refusal; unscaled by `lora_multiplier` — a conditioning input, not a
ΔW) + 4 `TestResCond` additions; full fast suite green.

## Stage 2 result (2026-08-12) — 25b-2 FAIL (quality); 25b-3 NULL

Grid + evals + yardstick all in one day (31 train jobs rc=0; evals
`20260812-e25b2-eval-sfw-s100{1,2,3}` + `20260812-e25b2-yardstick`,
one boot). Read: `e25b_ship_read.{py,json}`.

**Gates.** (1) Twin-start identity — 12/12 flag pairs first-loss
bit-identical. (2) Determinism control — ctrl2 vs combo twin **0/1092
keys differ**. (3) Demote-mass pair-identity — 6/6 cells, hist
identical within both flag pairs. (4) Yardstick sanity — in-batch
0.9578/0.9541 vs recorded 0.9547/0.9541, inside ±0.02. **Recorded
observation**: realized σ>0.5 mass is 180/480 (37.5 %) vs the
E16.1-era 244/480 (50.8 %) at the same seeds — the σ-draw distribution
shifted between 2026-08-02 and 2026-08-12 (cause not chased here);
in-batch comparisons are unaffected (CRN holds), but wall deltas vs
native run smaller than the recorded −18.3 % (combo −13.6…−17.1 %),
and this batch's ΔW-to-native levels sit far above E16.1's recorded
ones (combo 0.74–0.77 vs 0.365/0.434) — cross-batch scalar comparisons
are dead, again.

**25b-2 (the ship read).** In-batch yardstick (cross-seed native
lottery): hews **0.9578**, channel **0.9541**. Mean within-seed render
cos vs native twin:

| arm | hews | channel | vs bar |
|---|---|---|---|
| combo | 0.9575 | 0.9650 | boundary hews (−0.0004), inside channel |
| **rescond** | **0.9450** | **0.9515** | **below BOTH** (−0.0128 / −0.0026) |
| sigma768 | 0.9532 | 0.9674 | below hews, inside channel (E4 shape reproduced) |
| rescond768 | 0.9575 | 0.9484 | boundary hews, below channel |

Quality gate (rescond at-or-inside on BOTH corpora): **FAIL** — hews
is not close (−0.013), channel is a near-miss (−0.003) and a near-miss
is a miss as frozen. Throughput (measured, not gated): paired wall
rescond/combo **1.0024** — the projection is free, as predicted; cost
was never the problem. Per the frozen branch: **recorded; the Stage-1
gradient-geometry result stands as paper-2 material with the
render-level negative alongside.** No paper-method update (the PASS
action), no ship; the flag stays an experimental opt-in.

**25b-3 (rescue read).** Paired per-seed Δcos(rescond768 − sigma768 vs
native twin): hews median **+0.0057** (2/3 positive), channel median
**−0.0206** (0/3) ⇒ direction **NULL** (both corpora required). Band
claim: sigma768 below the band on hews (testable), rescond768 at
0.9575 hews (hair below bar) / 0.9484 channel (below) ⇒ **not
rescued**.

**Secondary (ΔW toward native, corroborate-only).** Direction NOT
confirmed — the opposite on 12/12 pairs: rescond median 0.407/0.465 vs
combo 0.744/0.770 (rescond768 vs sigma768 the same). The conditioning
moves the endpoint *away* from the native twin, coherently (well above
the 0.26 nondet-retrain floor).

**CMMD (descriptive, no verdict).** Incoherent at this N exactly as E4
recorded (rescond is the best hews arm at s1001 and the worst at
s1002; several demote arms "beat" native) — the E4 lesson reproduces;
numbers in the read json.

**Mechanistic reading (descriptive).** The three reads tell one
consistent story: explicit conditioning *does* absorb demoted-step
residual per step (25b-1, gradient level), but the training then
converges to a **different model** — ΔW far from the native twin, and
renders outside the native seed lottery on hews — rather than a closer
recovery of the native run. The lever changes the solution; it does
not recover the native one. That is a legitimate paper-2 finding
(conditioning-absorbs-the-substitution at the gradient level; the
endpoint diverges at the model level) and a NO on shipping it as a
quality-neutral throughput recipe. E25c (inference knob on the trained
axis) is registered independently and its material (the rescond
checkpoints) exists — it reads the trained axis, not the ship
question, and is unaffected by this FAIL.

### Cost (Stage 2)

| item | cost |
|---|---|
| Stage 2.0 — inference wiring + tests + read script | CPU/code only |
| grid — 31 deterministic runs (≈ 5.5–6.5 min each) | ≈ 3–3.5 GPU-h |
| renders + yardstick (one boot, E4 harness) | ≈ 1.5–2.5 GPU-h |
| read (`e25b_ship_read.py`) | CPU minutes |
| **total** | **≈ 5–6 GPU-h** |

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
- **Stage-2 additions**: the rescue read never gates the ship read
  (a 25b-3 NULL cannot block a 25b-2 SHIP, and vice versa); the
  in-batch yardstick is the bar even if it drifts from the recorded
  numbers; no arm additions after freeze (`combolate` exclusion is
  recorded in the Stage-2 preamble — re-adding it is an amendment);
  Stage 2.0 wiring must not change training-path semantics (trainer
  forward bit-unchanged — pinned by the existing zero-init identity
  test).

## Cost ladder

| item | cost |
|---|---|
| Stage 0 — trainer flag + probe extension + invariant tests | CPU/code only |
| Stage 1 — 2 × tenth-scale deterministic twins | ≈ 2–3 GPU-h (480 steps, ~1.5× deterministic overhead) |
| Stage 1 — 2 × 768 5-bin probes | ≈ 7.6 GPU-h (e28f1 actuals) |
| Stage 1 — CPU read (`e25b_read.py`) | minutes |
| Stage 2 — full-scale grid + yardstick | own amendment; E16.1-class |
