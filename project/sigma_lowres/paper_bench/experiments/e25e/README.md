# E25e — re-centered resolution conditioning (REGISTERED (FROZEN) 2026-08-12)

| | |
|---|---|
| **Status** | **REGISTERED (FROZEN) 2026-08-12; Stage 0 landed same day.** Not yet run. |
| **Question** | E25b's post-close decomposition (`e25b/e25b_native_offset.{py,json}`) measured that the trained conditioning lever was **~90 % common-mode**: ‖W·φ(0)‖ 0.130–0.192 vs resolution-differential ‖W·φ(s) − W·φ(0)‖ 0.005–0.024, cos ≥ 0.994, across all 12 carrier checkpoints. SGD used the projection as a free global bias on the adaln trunk; the bias shifted the native operating point, the LoRA co-adapted, and the endpoint diverged (25b-2 FAIL, ΔW ANTI 12/12). User hypothesis (2026-08-12): telling the model the resolution narrows the per-step gap but *also pulls the 1024 side* — so remove the 1024-side channel structurally. E25e asks: with the delta re-centered to **W·(φ(s) − φ(0))** — native forwards bit-identical to control for ANY projection value, zero gradient to the projection on native steps — does the endpoint stay near the native twin (mechanism read), and does the trained model render inside the native yardstick (ship read)? |
| **Licensed by** | E25b Stage 1 IMPROVES (the conditioning *does* absorb the per-step graph-leg substitution — the lever idea is alive at the gradient level); E25b Stage 2 FAIL + the post-close common-mode decomposition (the divergence channel is the parameterization's common-mode freedom, not the conditioning idea); E25d ANTI coherence (the s-sweep moves only the tiny differential — consistent with common-mode dominance). |
| **Explicitly NOT this** | No probe stage (deliberate — the 25b lesson: the Stage-1 within-model geometry read did not predict the endpoint; E25e gates on the endpoint estimand directly, user decision 2026-08-12). No tenth-scale gate (user decision: paper-equivalent scale directly). E20.4 known-input rule unchanged: s is still the router's config fact; the loss/target/σ-draw/router are bit-unchanged; centering is a forward-parameterization change only. Not a rescue arm (sigma768 family stays closed at 25b-3 NULL). No cross-batch numeric comparison to the 25b grid (σ-draw shift recorded there; all reads in-batch). |
| **Depends on** | E16.1/25b Stage-2 harness verbatim (`e4_render_eval.py`, `e4_seed_yardstick.py`, frozen prompt/gen_seed grid, `e25b_stage2_launch.py` pattern); Stage 0 below. |

## The lever (Stage 0, landed 2026-08-12)

`--sigma_lowres_res_cond_centered` (requires `--sigma_lowres_res_cond`;
setup-time error otherwise, `_validate_res_cond_flags`):

- **Delta**: W·(φ(s) − φ(0)) instead of W·φ(s) — same zero-init
  projection, same injection site, same known-input scalar. At s = 0
  the sinusoid difference is exactly the zero vector, so the forward
  delta **and the projection's gradient are exactly zero** for any
  projection value: native steps are bit-identical to a control arm
  throughout training and train nothing through the module (centered ≡
  demoted-only, structurally). The common-mode channel E25b measured
  cannot exist.
- **Checkpoint**: same `sigma_lowres_res_cond_proj` tensor; the variant
  rides the metadata stamp `ss_sigma_lowres_res_cond = "centered"` (no
  tensor footprint). Factory/probe/inference all read the stamp and
  apply centered semantics; the inference attach at the default s = 0
  is then a mathematical no-op (kept for `--res_cond_s` uniformity —
  a centered checkpoint has **no** native offset to preserve, so the
  25b Stage-2.0 off-distribution hazard does not exist here).
- Invariants pinned in `tests/test_sigma_lowres.py::TestResCondCentered`
  (s=0 delta exactly zero for random proj, zero grad to proj at s=0 +
  nonzero at demote s, centered = uncentered − native-offset, zero-init
  identity, cfg/stamp round-trip, flag-pair validation, inference
  attach + stripped-stamp hard fail); 77 sigma tests, full fast suite
  1351 green.

## Frozen design — one grid, paper-equivalent scale (no pre-gate)

E16.1/25b-Stage-2 protocol verbatim: 2 corpora (hews 60-stem / 8 ep;
channel\_(caststation) 15-stem / 32 ep) × 3 seeds (1001–1003) × 480
steps bs 1, `--deterministic --paired_step_rng`, stock lora recipe,
identical argv within the flag pair:

| arm | routing | flags | role |
|---|---|---|---|
| native | — | — | yardstick anchor + ΔW/render reference |
| combo | 768@σ∈(0.65,0.95) / 896+yarnsig@σ>0.5 | — | ctrl (shipped recipe) |
| **rescond_c** | combo | `--sigma_lowres_res_cond --sigma_lowres_res_cond_centered` | **treatment** |

Plus one determinism-control duplicate (combo, hews, s1001) = **19
runs**, one daemon batch. Renders + in-batch yardstick in **one boot**
(reboot splits ⇒ all arms re-render). Yardstick recomputed in-batch
(the 25b numbers are context, not the bar).

### Validation gates (before the verdict rows are read)

1. **Twin-start identity**: first-forward loss bit-identical within the
   flag pair (rescond_c vs combo), 6/6 cells.
2. **Determinism control**: duplicate combo checkpoint key-identical to
   its twin (0 differing keys).
3. **Demote-mass pair-identity**: `token_step_hist` identical within
   the flag pair per seed (CRN).
4. **Yardstick sanity**: in-batch native lottery within ±0.02 of
   0.9578/0.9541 (this batch's own numbers carry regardless).
5. **Native-step identity (the centered invariant, in vivo)**: covered
   by gate 1 at start; the structural claim (native forwards identical
   at every step) is pinned by the Stage-0 tests, not re-measured
   per-run.

### 25e-1 — mechanism read (primary): does removing common-mode stop the endpoint divergence?

Per corpus, median over seeds of **in-batch ΔW cos(arm ~ native
twin)**. Context constants from the 25b batch (recorded, not compared
numerically): combo sat at 0.74–0.77, rescond collapsed to 0.41–0.47.
Judgment constants 0.05 / 0.15 recorded at freeze.

| outcome (both corpora) | verdict |
|---|---|
| median ΔW cos(rescond_c) ≥ median(combo) − **0.05** | **COMMON-MODE-CONFIRMED** — the collapse does not reproduce; the 25b divergence channel was the common-mode offset. |
| median ΔW cos(rescond_c) ≤ median(combo) − **0.15** | **COLLAPSE-PERSISTS** — the divergence is carried by the demote-step co-adaptation itself; the conditioning-lever family closes for ship purposes (paper-2 records both decompositions). |
| otherwise / corpora disagree | **NULL-MIXED** — recorded; no re-cut. |

Registered with the standing limit: ΔW closeness ≠ render closeness
(measured, twice). 25e-1 reads the *mechanism* question — where the
endpoint went — for which ΔW is the direct estimand; it does not carry
the ship question.

### 25e-2 — ship read: the 25b-2 bar verbatim

- **Quality gate (the only ship gate)**: rescond_c mean within-seed
  render cos vs native twin at-or-inside the **in-batch yardstick on
  BOTH corpora**. Near-miss = miss.
- Throughput measured, not gated (paired wall ratio + token_step_hist;
  the projection is architecturally free — 25b measured 1.0024).

| outcome | verdict |
|---|---|
| quality gate passes | **PASS — paper-method update, no product ship** (the 25b PASS action carries verbatim: flag stays experimental opt-in; product ship is a separate later decision). |
| quality gate fails | **FAIL (quality)** — recorded; with 25e-1 alongside, paper-2 gets the full decomposition story either way. |

25e-1 and 25e-2 are independent rows: CONFIRMED + FAIL is a legal and
informative outcome (endpoint recovered, renders still outside — would
localize the residual damage in the demote steps' data leg), as is
COLLAPSE-PERSISTS + PASS (renders fine off a different endpoint — the
yardstick, not ΔW, is the ship bar). Each reads as recorded.

### Descriptives (no verdict weight)

- ‖W·(φ(s) − φ(0))‖ per checkpoint — the learned differential when it
  is the *only* channel (25b prior: 0.005–0.024 while common-mode ate
  ~90 %; does it grow?). Emitted by the read script this time (the
  25b native-offset descriptive was registered and then not emitted —
  not repeating that).
- CMMD recorded, no verdict at this N (E4/25b lesson). Render fig
  sheets per the E4 format.

## Kill switches / honesty

- One parameterization change, no sweep — any further variant
  (per-block centering, learned φ scales, demote-branch-only module)
  is a new amendment with its own multiplicity accounting.
- The E20.4 known-input kill stands unchanged: any measured/estimated
  quantity proposed as embedding input kills the amendment proposing
  it.
- No post-hoc arms; no reruns on a miss; a near-miss is a miss.
- A centered checkpoint under the E25c-style `--res_cond_s` knob
  sweeps the differential axis only — descriptive if ever rendered,
  own registration before any claim (E25c itself stays on the 25b
  checkpoints).
- If thresholds go stale before submission (instrument redesign,
  reboot mid-batch), re-freeze via amendment; no post-hoc
  renegotiation.

## Cost ladder

| item | cost |
|---|---|
| Stage 0 — flag + centered delta + stamp plumbing + tests | CPU/code only (landed) |
| grid — 19 deterministic runs (≈ 5.5–6.5 min each) | ≈ 2–2.5 GPU-h |
| renders + in-batch yardstick (one boot, E4 harness) | ≈ 1.5–2 GPU-h |
| read script (`e25e_read.py`, 25e-1 + 25e-2 + descriptives) | CPU minutes |
| **total** | **≈ 4–4.5 GPU-h** |
