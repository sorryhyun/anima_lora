# yarnsig — σ-gated YaRN-banded RoPE alignment (probe log)

> **Frozen 2026-07-31, moved here from `../bench/yarnsig_report.md`** along
> with `report.md`, which it was extracted from. Last live 2026-07-27; the
> open probes in the closing section were never run. Bare `results/<run-id>/`
> paths below are relative to **`../bench/results/`**.

The yarnsig sub-line of sigma_lowres, extracted from `report.md`
2026-07-27: frequency-selective (YaRN/NTK-by-parts) RoPE alignment on
demoted steps, with SigMa-style σ-gated band boundaries. Status:
**Phase-1b refinement candidate for the live 1024→896 route** — wired as
`train.py --sigma_lowres_yarnsig` (mechanics in `../methods.md` §"Phase 1b
trainer wiring"), shippable only via the Phase-1b fixed-steps CMMD A/B.

Lineage (full context stays in `report.md`): G10 decomposed the demote
Floor as `Floor_e = RoPE_e + Resid_e` (768's Floor mostly RoPE at the
σ=1 endpoint); G11 closed uniform PI-stretch — off-manifold once content
is in the input. The yarn probes below are the frequency-*selective*
alignment family that survived where PI sank: full PI stretch only for
bands with < α rotations across the demoted extent, native spacing above
β, linear ramp between; yarnsig scales both thresholds by
μ(σ) = sigmoid(γ·[logit σ − logit σ_c]).

Sections below are moved verbatim from `report.md` (stubs left in place
there). New yarnsig probes (the 1024→768 rescue probe and the 1280→1024
gate probe, pre-registered in `../roadmap.md` 2026-07-27) land here.

## YaRN-banded alignment probe (2026-07-27): gate does NOT widen — but frequency-selectivity rescues alignment from the PI penalty

Run: `results/20260727-1421/` (40 images × 8 draws/bin, `--bins 4
--sigma_window 0.15,0.65 --endpoint_bin --demote_edges 896 --pi_align
--yarn_align 1,4`; 110 min; smoke `results/20260727-1415`). The roadmap's
pre-registered gate-widening probe for the live 1024→896 route: `896yarn` =
NTK-by-parts per-dim frequency rescale (`yarn_rope` in `run_sigma_probe.py`
— full PI stretch for bands with < α=1 rotation across the demoted extent,
native integer spacing above β=4, linear ramp between), `896pi` = the
uniform-stretch baseline. Instrument valid: split-half 0.79–0.87 for the
demote arms; gap_reenc |mean| ≤ 0.037.

| σ bin | 0.21 | 0.34 | 0.46 | 0.59 | 1.0 |
|---|---|---|---|---|---|
| gap_896 | .163 | .187 | .103 | .118 | .035 |
| gap_896pi | .368 | .220 | .147 | .174 | .014 |
| gap_896yarn | .227 | .163 | **.074** | **.068** | **−.013** |
| paired yarn−896 | +.064 (2.8σ) | −.025 | −.029 | **−.050 (2.1σ)** | −.048 |
| paired yarn−reenc | +.190 | +.138 | +.060 | **+.089 (5.9σ)** | −.021 |

Scored against the pre-registered outcomes:

1. **Gate-widening: FAIL.** The improvement leg passes at σ=0.59 (paired
   −0.050 ± 0.024, 2.1 combined SEM) but the in-band leg fails everywhere:
   yarn's excess over reenc is +0.060–0.190 (3.4–7.7 SEM out) at every bin
   σ ≤ 0.65 (medians agree: +0.077/+0.082 at 0.46/0.59). No bin below the
   gate becomes safe. **Gate stays σ > 0.5.**
2. **The close-the-family outcome did NOT occur either** — its premise
   (yarn ≈ pi ≥ 896) is refuted at 5 SEM: yarn beats PI by −0.090 ± 0.018
   paired over σ 0.46–0.59 (33/40 images) and PI is the worst arm in every
   window bin (re-confirming G11's off-manifold verdict at 896). Frequency
   selectivity is exactly what rescues alignment: keeping high-freq bands
   at native spacing avoids the penalty that sank uniform PI.
3. **No regression above σ 0.5** — yarn is the best arm at 0.59 and at the
   endpoint (−0.013, and most stable: cross-image SD 0.236 vs 0.359 plain).
   Its gap is perfectly monotone in σ (Spearman −1.0).

Secondary structure worth keeping: yarn has a genuine **low-σ liability**
(+0.064 ± 0.023 over plain at σ=0.21, only 15/40 wins) — the ramp bands
(1 < r < 4 rotations) sit in neither coordinate system, and at low σ the
score leans on exactly the fine positional consistency they perturb; the
LLM-side attention-temperature compensation (~0.1·ln s + 1) is not
implemented. The improvement-vs-σ crossover sits near σ ≈ 0.35. Never
apply yarn ungated.

Verdict: **as a gate-widener, closed at α,β = 1,4** — the roadmap's one
allowed retune remains (wider full-stretch band 0.5,2 / stricter 2,8 are
the natural probes, optionally + attention temperature) before the
alignment family closes for good. Independently, yarn at the *existing*
gate is a small paired win over plain demotion (−0.042 ± 0.018 over
σ ≥ 0.46 incl. endpoint) — a refinement candidate that only the Phase-1b
CMMD A/B could ship, not a probe-level decision. Safe set unchanged:
{1024→896 @ σ>0.5, 1280→1024 @ σ\*∈(0.625, 0.875)}.

Pool note: this pool reads hotter than Phase-0's above the gate (plain-896
excess +0.139 at σ=0.59 vs Phase-0's ~0.05 at 0.56) — pool composition,
not a gate re-litigation; paired arm-vs-arm differences are the verdict
objects here.

## SigMa σ-gated YaRN boundaries (2026-07-27): PASS both legs — yarnsig is the Phase-1b refinement candidate

Run: `results/20260727-1639/` (110 min; 40 images × 8 draws/bin, `--bins 4
--sigma_window 0.15,0.65 --endpoint_bin --demote_edges 896 --yarn_align
1,4 --yarn_sigma_gate 0.35,2`; smoke `results/20260727-1633`;
pre-registration in `roadmap.md` §"SigMa σ-gated YaRN boundaries").
`896yarnsig` = the static (1,4) banded rescale with both thresholds scaled
per-draw by μ(σ) = sigmoid(2·[logit(σ) − logit(0.35)]) — SigMa's dynamic
boundary gating (Eq. 21, github.com/bxuanz/SigMa), functional form only
(their scale laws rejected as inference-side, center from our measured
crossover). μ at the bins: 0.20 / 0.48 / 0.72 / 0.88 / 1.00. Instrument
valid: demote-arm split-half 0.58–0.88; gap_reenc |mean| ≤ 0.038.

| σ bin | 0.21 | 0.34 | 0.46 | 0.59 | 1.0 |
|---|---|---|---|---|---|
| gap_896 | .163 | .187 | .102 | .118 | .035 |
| gap_896yarn | .243 | .149 | .083 | .077 | .045 |
| gap_896yarnsig | .197 | .155 | .112 | .086 | **−.014** |
| paired yarnsig−896 | **+.033 (1.4σ)** | −.032 | +.010 | −.032 | −.049 |
| paired yarnsig−yarn | −.046 | +.007 | +.029 | **+.009 (0.4σ)** | **−.058 (1.4σ)** |
| paired yarn−896 (ref) | +.079 (2.6σ) | −.038 | −.019 | −.041 | +.010 |

Scored against the pre-registered legs:

1. **Liability leg: PASS.** Paired yarnsig−896 at σ=0.21 is +0.033 ±
   0.025 — within 2 combined SEM of 0, and no new liability at 0.34
   (−0.032, yarnsig *better* than plain). Static yarn replicated its
   liability in the same pool (+0.079, 2.6σ — prior run +0.064), so the
   σ-gate cut it by ~58%. Honesty note: the residual is a trend, not
   zero — median +0.023, 16/40 wins, split-half of the bin-0 diff
   unstable (+0.070/−0.003). "Attenuated below significance", not
   "proven erased".
2. **Preservation leg: PASS.** Paired yarnsig−yarn is +0.009 ± 0.022
   (0.4σ) at σ=0.59 and −0.058 ± 0.043 (1.4σ, favoring yarnsig) at the
   endpoint — the σ≥0.46 alignment wins survive the gate. yarnsig keeps
   the paired win over plain demotion where it matters (−0.032 @ 0.59,
   −0.049 @ endpoint) and is the best + most stable arm at the endpoint
   (gap −0.014, SEM 0.037 vs 0.057–0.059 for the others). The endpoint
   yarnsig-vs-yarn diff is seed noise by construction (μ=1 there makes
   the arms rope-identical).
3. **Not a gate-widener, as pre-committed**: yarnsig excess over reenc is
   +0.099–0.159 (4.1–7.2 SEM) at every window bin — plain-896's S1 owns
   that budget. **Gate stays σ > 0.5.**

Verdict: **yarnsig replaces "static yarn at the existing gate" as the
Phase-1b refinement candidate** — applicable ungated on demoted steps (no
second σ-threshold in the trainer; the gate is inside the rope schedule),
shippable ONLY via the Phase-1b CMMD A/B. The owed α,β retune is **no
longer owed** — timing did the job the retune was reserved for; it stays
available as a conditional reserve iff the Phase-1b A/B surfaces a low-σ
regression attributable to the residual +0.033 trend. Mechanism footnote:
at μ=0.20 the (1,4) ramp bands are already at native spacing, yet a
~40% liability residual trends — so the prior run's ramp-band account is
at most partial; some share plausibly rides the still-stretched r < α·μ
global carriers. Mechanism-value only; no route implication. Safe set
unchanged: {1024→896 @ σ>0.5, 1280→1024 @ σ\*∈(0.625, 0.875)}.

## yarnsig in-vivo arm (2026-07-27): rope footprint is benign — no extra displacement from base, divergence-from-sigma lands in the low-signal blocks

Trainer wiring for the SigMa-probe winner landed the same day
(`--sigma_lowres_yarnsig`, bare flag = the probe's `1,4,0.35,2`;
`methods.md` §"Phase 1b trainer wiring" for mechanics — μ from batch-min σ,
rope swapped only on demoted forwards, invariants pinned in
`tests/test_sigma_lowres.py::TestYarnsigRope`). Fifth paired arm trained
into the tenth4s sweep (`anima_lora_tenth4s_yarnsig`, seed 42 CRN, 1200
steps 10:10, 245/500 = 49% eligible steps demoted — lockstep with the
sigma arm's demote set by construction; first demote σ=0.525 → μ=0.808;
final avr_loss 0.0913 vs sigma's 0.0912). Existing base/sigma/σ0.75/896only
checkpoints reused as pair partners (CRN makes them valid; the wiring
change touches no RNG path). Comparison: `results/20260727-1944-dw-yarnsig/`.

| pair | cos | differs by |
|---|---|---|
| base ↔ yarnsig | **0.319** | demotion + rope on ~48% of steps |
| base ↔ sigma | 0.320 | demotion on ~48% (reference) |
| sigma ↔ yarnsig | **0.402** | ONLY rope on the shared demoted ~48% |
| yarnsig ↔ σ>0.75 | 0.340 | (sigma ↔ σ>0.75 = 0.343) |
| yarnsig ↔ 896only | 0.240 | (sigma ↔ 896only = 0.245) |

- **No added displacement from base**: base↔yarnsig 0.319 ≈ base↔sigma
  0.320, per-block profiles overlapping everywhere (late blocks 0.49–0.73
  vs 0.48–0.72) — switching the demoted steps' rope to yarnsig moves the
  final weights no further from never-demote than plain demotion already
  does.
- **The rope intervention's footprint is below noise resolution**
  [corrected same day by the twin-floor measurement (`report.md` §"Twin
  controls + `--deterministic`"); then RESOLVED by the deterministic
  re-run (`report.md` §"Deterministic three-arm table") — footprint real,
  0.396 pure-treatment]: sigma↔yarnsig 0.402 ≈ the identical-command twin
  floor 0.413 — at this instrument's nondeterministic resolution the rope
  change displaces weights no more than hardware nondeterminism does, so
  nothing could be attributed from this pair.
- **Location in weight space unchanged**: yarnsig's cosines to every third
  arm track sigma's within 0.005 — it occupies the same neighborhood, i.e.
  the σ-gated rope is a refinement of the sigma arm, not a third regime.

Verdict: no in-vivo red flag for the yarnsig refinement — weight-space
behavior is indistinguishable-from-plain-demotion where that matters
(distance from base, late-block structure) while carrying the probe's
alignment wins on demoted steps. The Phase-1b fixed-steps CMMD A/B remains
the only shipping gate, now three-armed: base vs `--sigma_lowres` vs
`--sigma_lowres --sigma_lowres_yarnsig`.

Deterministic attribution note (full sections stay in `report.md`): under
`--deterministic`, sigma↔yarnsig = **0.396** with zero noise — the rope
change's weight-space footprint is real, landing in the same low-signal
early/mid blocks as every trajectory perturbation; endpoint ΔW cosine
detects separation but cannot rank treatment magnitudes.

## Open probes (pre-registered 2026-07-27, `../roadmap.md`)

1. **yarnsig 1024→768 rescue probe** — the G11 closure of 768 tested only
   uniform PI; frequency-selective alignment was never measured at 768,
   where G10 puts the largest RoPE share of any edge. σ-window 0.5–1.0
   mirror of G11 with yarn/yarnsig (+ pi in-pool comparator) arms.
2. **yarnsig 1280→1024 gate probe** — resumes the deprioritized σ\*
   localization on the probe-local 1280 cache with yarn/yarnsig arms: does
   frequency-selective alignment move the route's crossover left from
   (0.625, 0.875) toward the 896-route's σ>0.5?

Results land in this file when the runs complete.
