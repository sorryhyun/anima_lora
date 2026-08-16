# sigma_lowres — roadmap

Status: Phase 0 DONE (spectral mechanism refuted, σ-map measured), Phase 1a
DONE (ratio transfer FAILED), pooled-gradient probe DONE 2026-07-25,
**1280→1024 probe DONE 2026-07-26** (Q1 answered: ratio refuted as governor,
capacity predicts the ordering, but the safe threshold is route-dependent —
1280→1024 floors at σ ≥ ~0.75–0.875, not 0.5; report.md "1280→1024 probe").
Safe set = {1024→896 @ σ>0.5, 1280→1024 @ σ>σ\*∈(0.625,0.875)}. The cheap
harness for off-corpus tiers now exists (`prep_1280_probe.py` probe-local
cache + `--data_root` — no corpus re-preprocess, ~47 min/probe).

## Next: σ-window refinement (localize σ\* for 1280→1024)

`run_sigma_probe.py --sigma_window 0.5,1.0 --bins 5` on the same probe-local
cache — all bins in the crossover region (centers 0.55…0.95). **Started
2026-07-26, deprioritized at 5/24 images** (partial rows were run
`20260726-2109`, envelope since pruned; same command re-runs) in favor of the
prior-distance discriminator, which landed the same day (groundings.md G6:
no 1280 discontinuity, prior ↮ Floor — the Floor is graph-side). Payoff of resuming is gate-position-sensitive: σ\* ≈ 0.65 →
~9% epoch saving on 1280-tier data, σ\* ≈ 0.75 → ~5%. Then the decision
point:

- The corpus has no 1280 tier today (`target_res = [1024, 896]`), so the new
  route's practical value is conditional on adopting one; the map/paper value
  (3-route boundary σ\*(route)) is already banked.
- Phase 1b remains gated on judging {1024→896 @ σ>0.5} (~13–14%) + any
  adopted-1280 increment worth the dual-cache complexity.

## PI-aligned 768 route **[CLOSED 2026-07-27, same day — G11]**

G10 decomposed the Floor (`Floor_e = RoPE_e + Resid_e`; 768's Floor is
mostly RoPE_e, erasable at the σ=1 endpoint by PI-stretched rope) and
conditionally reopened a PI-aligned 1024→768 @ high-σ route (0.56× token
cost). The pre-registered σ-resolved gate (G11, `results/20260727-1234`)
**closed it the same day**, twice over: (1) the stretch is off-manifold
once content is in the input — 768pi is *worse* than plain 768 through
σ 0.56–0.81, better only at σ ≥ 0.94; (2) S1(1024→768) alone stays
+0.09–0.22 over reenc across the whole window (ratio 0.75, A_e ~ ratio).
Do NOT re-propose trainer-side pi-rope: training through it voids the
zero-adaptation substitution premise. G10 survives as mechanism (Q6
material). 512 was already closed (Resid_512 ≈ 0.22). Safe set unchanged:
{1024→896 @ σ>0.5, 1280→1024 @ σ\*∈(0.625, 0.875)}.

## YaRN-banded gate-widening probe for 1024→896 **[RUN 2026-07-27 — no gate widening; the owed retune was superseded by the SigMa gate probe below]**

Outcome (run `20260727-1421`, envelope since pruned; full read in
`bench/report.md`):
between the pre-registered branches. (1)'s improvement leg passed at
σ=0.59 (paired yarn−896 −0.050, 2.1 SEM) but the **in-band leg failed
everywhere** (yarn−reenc +0.06–0.19, ≥3.4 SEM out at every bin σ ≤ 0.65)
— gate stays σ>0.5. (2)'s premise refuted: yarn ≪ pi (5 SEM) — frequency
selectivity DOES rescue alignment from the G11 off-manifold penalty. (3)
no regression above the gate; yarn is best + most stable at σ≥0.59 and
the endpoint. Low-σ liability found (+0.064 over plain at σ=0.21; ramp
bands, no attention-temperature term) — never apply ungated. **The one
allowed α,β retune is now owed** (0.5,2 / 2,8, optionally + attn
temperature) before the alignment family closes; independently, yarn at
the existing gate is a −0.04 paired refinement candidate, shippable only
via the Phase-1b CMMD A/B.

Target: move the safe gate of the LIVE route left (σ>0.5 → ~0.3?), not
revive dead routes. Phase-0: gap_896 ≈ 0.14–0.16 through σ ≈ 0.44 is
S1-dominated (Floor_896 ≈ 0); G11 showed *uniform* PI stretch adds an
off-manifold penalty in the contentful window, but the mechanism reading
(content lobes keyed to high-freq integer spacing; global extent in low
freqs) predicts a **frequency-selective** remap — YaRN/NTK-by-parts: full
stretch below α rotations across the demoted extent, native spacing above
β, ramp between — may cut the RoPE-mediated share of S1's transmission
without the penalty. `--yarn_align A,B` arm; window bins σ∈(0.15, 0.65) +
endpoint, arms reenc/896/896pi/896yarn (pi = uniform-stretch baseline).

Pre-registered read: (1) gap_896yarn < gap_896 by ≥2 combined SEM in any
bin σ ≥ 0.3 AND in-band there → gate-widening candidate; then a
`--sigma_window` refinement localizes the new σ\*, and ONLY the Phase-1b
CMMD A/B (rerun at the new gate) ships it. (2) 896yarn ≈ 896pi ≥ 896
everywhere → frequency-selectivity doesn't rescue alignment; RoPE share
of A_e absent or bands wrong — one α,β retune allowed, then close the
alignment family for good. (3) Regression above σ 0.5 → discard
regardless of mid-σ behavior.

## SigMa σ-gated YaRN boundaries **[RUN 2026-07-27 — PASS both legs; yarnsig = the Phase-1b refinement candidate]**

Outcome (run `20260727-1639`, envelope since pruned; full read in `bench/report.md`
§"SigMa σ-gated YaRN boundaries"): liability leg PASS (+0.033 ± 0.025 at
σ=0.21, within 2 SEM; static yarn replicated +0.079 in-pool → gate cut it
~58%, residual trend noted), preservation leg PASS (yarnsig ≈ yarn at
σ=0.59 + endpoint; best/most-stable arm at the endpoint). Per
pre-registration: yarnsig replaces static-yarn-at-gate as the Phase-1b
refinement candidate, ungated on demoted steps, shippable only via the
Phase-1b CMMD A/B. The α,β retune is no longer owed (conditional reserve
iff the A/B surfaces a low-σ regression) — **the alignment family is
otherwise closed**. Not a gate-widener (in-band vs reenc fails 4–7 SEM at
every window bin); gate stays σ>0.5.

Motivation: SigMa (σ: Sigmoid Modulation for Ultra High Resolution
Diffusion, ICML 2026, github.com/bxuanz/SigMa) independently derives the
same ordering law the line measured (harsher scale change → later
handover) and parameterizes YaRN intervention as a **σ-dependent sigmoid
gate on the ramp boundaries** (their Eq. 21: thresholds × μ(t), μ a
log-odds sigmoid). We adopt the **functional form only** — their scale
laws (t_c = 1/s, γ = √s) are inference-side spectral reasoning (the class
Phase 0 showed mispredicts the gradient crossover 3.5×) and off-scale at
s = 8/7 (t_c = 0.875 vs measured crossover ≈ 0.35). This is NOT the α,β
retune (bands stay 1,4) and NOT a re-proposal of uniform PI (closed G11):
it tests whether **timing** alone removes the yarn low-σ liability that
forced the hard gate.

Arm: `--yarn_sigma_gate 0.35,2` adds `896yarnsig` — same banded rescale,
thresholds α·μ(σ), β·μ(σ), μ = sigmoid(2·[logit(σ) − logit(0.35)]). At
σ=0.21 μ≈0.20 (ramp bands → native spacing, liability mechanism removed);
at σ=0.59 μ≈0.88 (≈ static yarn); μ→1 at the endpoint. Run mirrors
20260727-1421 bin-for-bin, PI arm dropped:
`--bins 4 --sigma_window 0.15,0.65 --endpoint_bin --demote_edges 896
--yarn_align 1,4 --yarn_sigma_gate 0.35,2`.

Pre-registered read (one shot at (σ_c, γ) = (0.35, 2) — no iteration on
the gate params; comparator is **static yarn**, not PI):

1. **Liability leg**: paired yarnsig−896 at σ=0.21 within 2 combined SEM
   of 0 (the +0.064 liability erased), and no new liability at 0.34.
2. **Preservation leg**: paired yarnsig−yarn within 2 combined SEM at
   σ=0.59 AND the endpoint (keeps the −0.050/−0.048 wins).
3. **PASS both** → yarnsig replaces "yarn at the existing gate" as the
   Phase-1b refinement candidate — applicable **ungated on demoted steps**
   (no second σ-threshold in the trainer); still only shippable via the
   Phase-1b CMMD A/B.
4. **FAIL liability leg** → timing doesn't fix ramp-band damage; the
   remaining alignment-family shot stays the α,β retune (0.5,2 / 2,8,
   optionally + attn temperature). **FAIL preservation only** → fall back
   to hard-gated static yarn as the candidate; no γ/σ_c search.
5. **Explicitly NOT a gate-widener**: yarnsig → plain 896 as μ→0, so it
   cannot beat 896's own S1 below the gate (in-band vs reenc at σ≤0.46 is
   reported but not a pass criterion). Gate stays σ>0.5 regardless.

## YaRN/yarnsig on the 1024→768 window **[RUN 2026-08-15 — all three legs FAIL ship criteria; 768 stays plain, family closed at 768]**

Outcome (run `20260815-1041-yarn768`, envelope since pruned; full read in
`bench/report.md` §"yarnsig on the 1024→768 window"): in-window = no harm
but no win (both arms within 2 SEM of plain; yarnsig 2.8 SEM *worse* than
static yarn at σ=0.875 — near-unity μ still discretely reassigns boundary
bands); endpoint = improvement borderline (yarnsig−768 −0.030, 2.0 SEM)
but in-band FAIL at 8 SEM (+0.070 over reenc) — `threshold2_max` stays
0.95; downward = the family's largest paired win (yarn −0.082 / yarnsig
−0.093 at σ=0.425, ~3 SEM) yet still +0.16–0.18 over reenc (~7 SEM) —
gap reduction ≠ certification. No rule-2 wiring change; no window change;
safe set unchanged. Original pre-registration below.

Extends the (otherwise closed) alignment family to the deep route: the
combo recipe's 1024→768 window (0.65, 0.95) was certified on **plain**
demotion, and `train.py` deliberately keeps yarnsig primary-only. This
probe measures whether the banded rope helps 768 at all. NOT a PI
re-proposal (G11 closed uniform PI at 768 — pi *worse* than plain through
σ 0.56–0.81; yarn beat PI by 5 SEM at 896 precisely via frequency
selectivity), and NOT a param search: α,β = 1,4 and (σ_c, γ) = (0.35, 2)
transported verbatim (rotation counts self-adapt to the 768 extent;
inside the window μ ≥ ~0.85, near-static — the 768 crossover is *read*
from this run's paired yarn−768 curve, not searched).

Three σ regions, three separate legs — bins
`--bins 4 --sigma_window 0.35,0.95 --endpoint_bin` (centers 0.425, 0.575,
0.725, 0.875, 1.0), arms reenc/768/768yarn/768yarnsig, mirroring
20260727-1639 otherwise (40 img × 8 draws/bin, soup_sincos adapter):

1. **In-window leg (0.725, 0.875)**: paired {yarn,yarnsig}−768 ≤ 0 or
   within 2 combined SEM → alignment does no harm where the route is
   certified (a Phase-1b-style refinement candidate needs a further
   ≥2 SEM *win* in-window). Any ≥2 SEM regression in-window → **discard
   for 768 entirely**, route stays plain, family stays closed.
2. **Endpoint leg (σ=1)**: the window's upper bound exists because the
   σ=1 gap re-elevates (+0.130). 896yarn/yarnsig both drove the endpoint
   negative (−0.013/−0.014, best+most-stable arms), so this is the one
   leg with a favorable precedent. gap_768yarn(sig) at σ=1 in-band vs
   reenc AND ≥2 SEM better than plain 768 → **upward-widening candidate**
   (`threshold2_max` → ~1.0); a `--sigma_window 0.85,1.0` refinement
   localizes the new upper σ\*, and only a rerun Phase-1b CMMD A/B ships
   it.
3. **Downward leg (0.425, 0.575)**: expectation LOW — the same leg
   FAILED at 896 (in-band vs reenc +0.06–0.19, ≥3.4 SEM out, everywhere
   below the gate). Pre-registered read: in-band vs reenc at either bin →
   downward-widening candidate; else **downward widening for 768 closes
   with the family** — no retune owed on this leg.

Ship-gating unchanged regardless of outcome: yarnsig is wired
primary-only (`train.py` choice==1), so even a pass ships nothing by
itself — it buys a trainer wiring change (rule-2 yarnsig with its own
certified window) plus the CMMD A/B, in that order.

## Phase 1b — trainer wiring **[BUILT 2026-07-26]** + the gate **[OPEN]**

Wiring shipped opt-in (`--sigma_lowres`, route pinned to 1024→896 @ σ>0.5) —
full description in `methods.md` §"Phase 1b trainer wiring". Key deviations
from the sketch below: σ is drawn trainer-side (σ-first via
`draw_flat_sigmas`, single source of density truth) rather than at batch
assembly, and the sibling cache is an **in-npz key** (`demoted_{H}x{W}`, `make
preprocess-demote`) rather than a stem-suffixed file — reconcile and bucket
discovery needed no changes at all.

**yarnsig wiring [BUILT 2026-07-27] + in-vivo arm [RUN 2026-07-27 — benign]**:
`--sigma_lowres_yarnsig` (bare = the probe's `1,4,0.35,2`) applies the
SigMa-gated banded rope on demoted forwards only. Fifth paired tenth4s arm +
rank-space ΔW read (`bench/report.md` §"yarnsig in-vivo arm",
`bench/compare_ckpt_dw.py` now the permanent instrument): base↔yarnsig
0.319 ≈ base↔sigma 0.320 (no added displacement), sigma↔yarnsig 0.402 with
the best late-block agreement of any pair — the rope footprint sits in the
low-signal early/mid blocks. No red flag; not a gate substitute.

- **Gate (still owed)**: fixed-steps A/B — CMMD non-inferior (within-run
  usage only, per `project_cmmd_val_signal`) + rendered spot-check + realized
  wall-clock logged. Now **three-armed**: baseline vs `--sigma_lowres` vs
  `--sigma_lowres --sigma_lowres_yarnsig` (`tenth` preset × 4 epochs; paired
  checkpoints already exist for weight-space reads). Pitch is wall-clock at
  fixed steps, never "more steps in the same time" (autoscale lesson). CMMD
  regression → close the line (pre-committed).

## Phase 1c — bespoke loops (EC / turbo) — gated on separate probes

Each needs its own operating-point probe before any wiring (questions.md Q5).
Do not schedule until Phase 1b has shipped and survived its gate.

## Kill criteria

- 1280→1024 probe fails AND ~14% is judged below the dual-cache complexity
  bar → close; keep `rapsd.py` + `run_sigma_probe.py` as reusable
  instruments and the "spectral sufficiency ≠ gradient equivalence" finding.
- Phase 1b CMMD regression → close (pre-committed in the proposal).

## Pointers

Design: `project/sigma_lowres/initial_proposal.md` · Data:
`project/sigma_lowres/bench/report.md` · Memory: `project_sigma_lowres_phase0`,
`project_tier_routing_phase3a_failed` (split-half check mandatory).
