# E25e — re-centered resolution conditioning (REGISTERED (FROZEN) 2026-08-12)

| | |
|---|---|
| **Status** | **RUN + READ 2026-08-12.** 25e-1 **COLLAPSE-PERSISTS** (both corpora — the divergence was never the common-mode offset), 25e-2 **PASS** (rescond_c inside the in-batch yardstick on both corpora). The registration's "COLLAPSE + PASS" branch: renders fine off a different endpoint; conditioning-lever family closes for ship purposes per the frozen wording, paper-method update only, flag stays experimental opt-in. See Result below. |
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

## Launch record (2026-08-12)

Grid submitted via `e25e_launch.py` — 19 lora jobs, ids in
`e25e_jobs.json`. Fail-fast cell `e25e_hews_rescond_c_s1001` (the
never-run centered flag combination, queued first): **rc 0**,
checkpoint carries `sigma_lowres_res_cond_proj` + the **"centered"**
stamp (proj param norm 0.109 — nonzero, trained). Remaining 18 cells
queued behind it.

## Result (2026-08-12, `e25e_read.{py,json}`)

Renders + in-batch yardstick one boot (eval jobs `20260812-211155-*`,
runs `20260812-e25e-eval-sfw-s{1001..1003}` + `20260812-e25e-yardstick`).
**All 4 gates pass**: (1) twin-start identity — TB `loss/current` at step 2
(earliest logged step) bit-identical within the flag pair, 6/6 cells;
(2) determinism control — ctrl2 vs combo 0/1092 differing keys;
(3) demote-mass pair-identity — `token_step_hist` identical, 6/6;
(4) yardstick sanity — in-batch native lottery 0.9571 / 0.9541 vs recorded
0.9578 / 0.9541, within ±0.02.

### 25e-1 — COLLAPSE-PERSISTS (both corpora)

In-batch ΔW cos to the native twin (`stage2_dw/`, lora modules only):

| corpus | combo per seed | rescond_c per seed | medians | Δ |
|---|---|---|---|---|
| hews | 0.718 / 0.772 / 0.744 | 0.428 / 0.447 / 0.440 | 0.744 vs 0.440 | **−0.303** |
| channel | 0.760 / 0.770 / 0.773 | 0.515 / 0.527 / 0.492 | 0.770 vs 0.515 | **−0.254** |

Both far past the −0.15 constant. Removing the common-mode channel did
**not** recover the endpoint — rescond_c collapses to the same band the
25b uncentered arm sat in (0.41–0.47, context not compared numerically).
The divergence is carried by the demote-step co-adaptation itself: the
LoRA co-adapts to *any* live conditioning delta on demote steps,
regardless of whether the native operating point moves. Per the frozen
wording, the conditioning-lever family closes for ship purposes; paper-2
records both decompositions (the ~90 % common-mode measurement stands as
a *descriptive* of what SGD does with the freedom, not as the divergence
mechanism).

### 25e-2 — PASS (quality gate, both corpora)

| corpus | in-batch yardstick | combo mean | rescond_c mean | rescond_c margin |
|---|---|---|---|---|
| hews | 0.9571 | 0.9546 (outside, −0.0026) | **0.9587** | **+0.0016** |
| channel | 0.9541 | 0.9650 | **0.9654** | **+0.0113** |

PASS per the frozen bar (rescond_c at-or-inside on BOTH corpora) =
paper-method update, no product ship; flag stays experimental opt-in.
Descriptive footnote: combo itself lands *outside* on hews this batch
(−0.0026) — the in-batch seed lottery; rescond_c out-renders its own
control on both corpora, off a ΔW-divergent endpoint. With 25e-1
alongside: ΔW closeness ≠ render closeness, now measured a third time,
in the opposite direction (25b: close-ish ΔW / bad renders per arm
rescond; here: collapsed ΔW / good renders).

Throughput measured-not-gated: paired wall rescond_c/combo mean 0.9997
(the projection is architecturally free, replicating 25b's 1.0024);
wall vs native −14.6…−15.2 % both arms both corpora (σ-draw-shift
context as recorded in the 25b read).

### Descriptives (no verdict weight)

- **Learned differential norms** (emitted this time): per rescond_c
  checkpoint ‖W·(φ(s)−φ(0))‖ = 0.003–0.035 (s896) / 0.007–0.077 (s768),
  strongly seed-variable (hews s1001 0.077 vs s1002 0.007). When the
  differential is the *only* channel it can grow past the 25b band
  (0.005–0.024) — but doesn't reliably.
- **CMMD** (recorded, no verdict at this N): rescond_c ≤ combo on 5/6
  cells, sitting at native-level values.
- **Cross-grid determinism**: e25e native/combo checkpoints bit-identical
  to the e25b2 grid's (0/1092 keys, same boot + argv) — the shared arms
  replicate exactly; the fail-fast cell trained one commit earlier
  (7f17421b vs 53d68448) but the inter-commit diff touches no training
  code.

## Conceptual figure

`e25e_concept_fig.py` → `e25e_concept.png` — two 3D scenes
(e24_bc_comb_3d idiom: a comb planted on the σ axis), **same
gap-closing, different rail**. Each scene: per-verdict-bin h(B+C)
chord pairs (grey ctrl vs black conditioned, exact Stage-1 768-probe
values) standing on the rail the model trains on, with the green
label arrows W·(φ(s)−φ(0)) doing the closing (the demote↔native pair
comparison cancels the common-mode, so the label is the only part the
gap sees). **E25b (left, measured)**: the trained W·φ(0) pushes the
whole rail off the dashed native anchor — every step trains there,
endpoint diverges (ΔW 0.74→0.41). **E25e (right, registered)**:
W·(φ(0)−φ(0)) = 0 exactly, the rail *is* the native anchor; chords
drawn faded + "?" as the registered expectation (grid running).
Honesty strip: rail offset ×2 and label arrows ×6 for visibility;
h-units vs delta-units heterogeneous — conceptual composition; legs
not drawn (h-units don't Euclidean-compose). Envelope in
`runs/<stamp>-fig-e25e-concept/`.

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
