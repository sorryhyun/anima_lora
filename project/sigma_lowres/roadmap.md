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
2026-07-26, deprioritized at 5/24 images** (partial rows
`bench/results/20260726-2109/`; same command re-runs) in favor of the
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

Outcome (`bench/results/20260727-1421/`, full read in
`bench/yarnsig_report.md`):
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

Outcome (`bench/results/20260727-1639/`, full read in
`bench/yarnsig_report.md` §"SigMa σ-gated YaRN boundaries"): liability leg PASS (+0.033 ± 0.025 at
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

## yarnsig 1024→768 rescue probe **[PRE-REGISTERED 2026-07-27]**

Motivation: G11 closed the 768 route against **uniform PI** only; the
YaRN-banded probe then refuted that premise's generalization at 896 —
yarn beats PI by −0.090 ± 0.018 (5 SEM) in-window, i.e. frequency
selectivity avoids exactly the off-manifold penalty that closed G11. And
G10 puts the largest erasable RoPE share of any edge at 768 (Floor_768
≈ RoPE-dominated at σ=1). Frequency-selective alignment has never been
measured at 768. This is NOT a PI re-proposal (G11's "do not re-propose
trainer-side pi-rope" stands — yarn keeps high-frequency bands at native
spacing, the property uniform PI lacked).

Priors are against a full rescue: yarn's paired shave at 896 was
−0.03…−0.05 vs a 768 in-window excess of +0.09…+0.22 over reenc (G11
leg 2). The probe either closes the alignment family at 768 with the
right comparator, or finds the shave scales with the RoPE share.

Run (mirrors G11's σ-resolved window bin-for-bin, + alignment arms;
7 arms ≈ 2 h): `run_sigma_probe.py --adapter
output/ckpt/anima_soup_sincos.safetensors --bins 4 --sigma_window 0.5,1.0
--endpoint_bin --demote_edges 768 --pi_align --yarn_align 1,4
--yarn_sigma_gate 0.35,2` (40 images × 8 draws/bin; bins ≈ 0.56 / 0.69 /
0.81 / 0.94 + σ=1 endpoint; pi kept as the in-pool comparator; yarnsig ≈
yarn in this window — μ ≥ 0.78 — included because it is the rope the
trainer would actually ship).

Pre-registered read (one shot; no α,β / γ,σ_c iteration):

1. **Rescue leg**: gap_768yarn within 2 combined SEM of gap_reenc at any
   bin σ < 0.875 AND paired yarn−768 negative at ≥ 2 SEM there → the 768
   route conditionally REOPENS at that bin's σ (route 1024→768 @
   σ>σ\*_yarn with yarnsig rope mandatory; 0.56× token cost), still
   shippable only via the Phase-1b CMMD A/B. Both sub-legs required.
2. **Mechanism leg** (no route implication): paired yarn−pi ≤ −2 SEM
   in-window confirms frequency-selectivity generalizes to ratio 0.75;
   paired yarn−768 ≤ −0.08 at any window bin (a shave that scales with
   768's larger RoPE share instead of staying at 896's −0.05) is the
   RoPE-share-scaling read even if leg 1 fails.
3. **FAIL both** → the alignment family closes at 768 for good; safe set
   unchanged; 768's in-window excess is attributed to capacity
   (S1/Resid), completing the G11 leg-2 account.

## yarnsig 1280→1024 gate probe **[PRE-REGISTERED 2026-07-27]**

Resumes the deprioritized σ\*-localization run (partial rows
`bench/results/20260726-2109/`) with alignment arms added: does
frequency-selective alignment move the 1280→1024 crossover left from
(0.625, 0.875) — ideally to ≤ 0.5625, letting the route share the
896-route's single σ>0.5 threshold? At 896 yarn was NOT a gate-widener
(its in-band leg failed vs reenc below the gate), but 1280→1024 is the
opposite regime: the largest absolute target capacity ever probed (4116
tokens — capacity, not ratio, governs the threshold per the 20260726
probe), so plain-1024's S1 budget near σ\* is small and a −0.05-class
alignment shave could plausibly flip a bin.

Run (probe-local 1280 cache, production chains, TE symlinked; 6 arms ×
8 draws ≈ 2.5 h): `run_sigma_probe.py --adapter
output/ckpt/anima_soup_sincos.safetensors --tier 1280 --demote_edges 1024
--data_root project/sigma_lowres/bench/probe1280_cache --bins 4
--sigma_window 0.5,1.0 --endpoint_bin --draws_per_bin 8 --yarn_align 1,4
--yarn_sigma_gate 0.35,2 --pool 8` (24 images; bins ≈ 0.56 / 0.69 / 0.81 /
0.94 bracket the known σ\* interval; same (0.35, 2) gate — as-shipped
rope, μ ≥ 0.78 in-window).

Pre-registered read:

1. **σ\* localization** (plain 1024 arm, banked regardless of yarn):
   the first bin where gap_1024 sits within 2 combined SEM of gap_reenc
   narrows σ\* from (0.625, 0.875) to ~0.125 resolution. Payoff map:
   σ\* ≈ 0.65 → ~9% epoch saving on 1280-tier data, ≈ 0.75 → ~5%
   (conditional on ever adopting a 1280 tier — corpus has none today).
2. **Gate-widening leg**: at the lowest bin where plain 1024 is OUT of
   band, paired yarn(sig)−1024 ≤ −2 SEM AND gap_1024yarn within 2 SEM of
   gap_reenc → the route's gate moves left one bin with yarnsig rope
   mandatory; reaching 0.5625 = single-threshold trainer simplification.
3. **FAIL** (no bin flips) → yarnsig is not a gate-widener at 1280
   either (consistent with 896); the run still banks read 1 and the
   route keeps σ\*∈(0.625, 0.875) with plain rope.

## Phase 1b — trainer wiring **[BUILT 2026-07-26]** + the gate **[E4 core RUN 2026-07-30 — CMMD read owed]**

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
rank-space ΔW read (`bench/yarnsig_report.md` §"yarnsig in-vivo arm",
`bench/compare_ckpt_dw.py` now the permanent instrument): base↔yarnsig
0.319 ≈ base↔sigma 0.320 (no added displacement), sigma↔yarnsig 0.402 with
the best late-block agreement of any pair — the rope footprint sits in the
low-signal early/mid blocks. No red flag; not a gate substitute.

- **Gate**: the E4 grid (`paper_bench/completed_experiments.md` §E4 — 4 arms
  × 3 seeds × 2 artists, superseding the three-armed sketch) ran 2026-07-30:
  wall-clock **measured −14.6% / −15.1% FLOPs** at fixed steps, and the
  seed-noise yardstick puts sigma896's render deltas inside the seed lottery.
  **Still owed**: the CMMD non-inferiority read — the exercise pass had no
  metric power (negative control unsafe768 ≈ native at exercise N; full-band
  rescoring queued, `paper_bench/required_experiments.md` §E4) — plus val
  loss + peak mem. Pitch is wall-clock at fixed steps, never "more steps in
  the same time" (autoscale lesson). CMMD regression → close the line
  (pre-committed).

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
