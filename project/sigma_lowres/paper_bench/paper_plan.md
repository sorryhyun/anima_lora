# paper_plan — restructure direction for the sigma_lowres paper

Companion to `required_experiments.md` (open runs) and
`completed_experiments.md` (discharge record): those say *what must run*
and *what ran*; this file says *what the manuscript becomes*. Rewritten
2026-07-29, superseding the Branch-A plan (E1 fired rule 1, the
gap-native restructure is already in `paper/main.tex` — that plan is
discharged; its claim-level revisions all landed).

**Status:** the manuscript is currently the Branch-A spine (metrology →
account → measurements → null-in-gap-units → application). This plan
restructures it once more, into **theory → evidence → application**,
to fix the remaining structural problem: it reads as an exploration
log — evidence organized by run/question instead of by claim, hedges
scattered per-sentence instead of consolidated.

---

## 1. Direction in one paragraph

Claim-first. The paper states the two-term account as its theoretical
object up front — the **form** (additive two terms, G^p denominator,
p ∈ [1,2], σ-flat floor) *is* derived, from the perturbation expansion
plus cosine geometry, conditional on A1–A3; the **coefficients** (A_e,
m(σ), G(σ), Floor_e) are measured, not derived. That ledger is stated
once, prominently, in the theory section — and then the rest of the
paper is written assertively, without re-hedging each claim. Every
assumption and every discriminating prediction made in the theory
section names its designated probe; the evidence section discharges
them one term at a time, all in debiased units. The spectral account
appears in the theory section as what it is — the null model of the
input branch — and is scored in the evidence section. Application
(safety maps, trainer, cost) comes last and consumes the evidence.

Headline unchanged: *spectral sufficiency of the noisy input does not
guarantee gradient equivalence under resolution substitution* — plus
the converse-asymmetry sentence (reenc: input-level error real,
gradient cost below instrument resolution; input-level error neither
implies nor excuses gradient-level cost).

## 2. Voice and hedging policy

Distinguish structural hedging from protective hedging:

- **Structural (remove):** the exploration narrative ("the spectral
  prediction failed, we then…"), raw-vs-debiased double bookkeeping in
  the main text, per-sentence "at our operating point" qualifiers.
  Raw/historical tables move to the appendix wholesale; after the
  instrument-validation block (§4.1), every main-text number is
  debiased and the text never mentions raw again.
- **Protective (keep, consolidate):** pre-registered / confirmatory /
  post-hoc labels, ε\*-relative "safe" wording, "projected ceiling
  until E4". These survived the external review for a reason.
  Consolidate into **one epistemic-status ledger** — a paragraph or a
  status column in the claims table — instead of inline hedges.

**E5 decided the voice (qualified PASS, 2026-07-29 — see
`completed_experiments.md`).** Eq. 3 headlines as a **predictive**
first-order account with stated resolution: held-out routes at ~0.09
RMSE (768 beats the oracle smooth on its own data; 1280→1024 within
the 2× gate), governors upgraded from "consistent with" to measured —
ratio-twin amplitudes agree at z=0.14, and the floor exp-law
F(n) = 0.70·exp(−n/1041 tok) hits the held-out 768 floor dead-on
(+0.088 predicted vs +0.092±0.012 measured). Stated limits, verbatim
in the theory/evidence sections: the prediction is NOT within ε\*
anywhere (χ²/bin 7.6–9.2); the 768 mid-σ window (measured ≈ 0,
predicted ~0.1 — the form cannot dip below Floor_e) is the account's
one structural failure; A(r) interpolation rests on two distinct
ratios and is evidently convex (the 1280→1024 peak overshoot).

## 3. New section spine (current tex → new)

| new | content | built from (current tex) |
|---|---|---|
| §1 Intro | claim-first; narrowed headline; 14% as projected ceiling until E4 | §1, tightened |
| §2 Related | unchanged framing ("we test the tempting extension, not their methods") | §2 |
| §3 Theory: the demotion gap and a two-term account | **3.1 Setup**: gap definition, redraw floor, reenc control, ε\*(N, D, floor), safe(ε) — *definitions only*, no demonstrations. **3.2 The two perturbations and the first-order form**: input branch vs graph term, A1–A3, Eq. 3, the derivation ledger. **3.3 The spectral null**: tolerance family as the null model of the input branch. **3.4 Discriminating predictions**: where the accounts diverge; each prediction and each assumption names its §4 probe | §3.1 + §3.3 (defs only) + §4 (all) + §6.1 (family statement) |
| §4 Evidence, per term | **4.1 The instrument, validated**: finite draws manufacture floors (c/D decay, native floor 0.85→1.005), self-floor debiasing is D-flat, det-twins; from here on, debiased units only. **4.2 The phenomenon**: gap curves vs σ; the null scored against them. **4.3 The floor is graph**: endpoint ≡ x-zero ≡ α-flat — three independent probes converging (E1a/c + E2); σ-flatness; the strongest single figure. **4.4 Input-branch anatomy**: m(σ) route-uniform, G(σ) U-shape and small-G amplification, A_e governors (ratio vs absolute size, 1280→1024). **4.5 Floor decomposition**: depth band + RoPE_e/Resid_e waterfall (mechanism, not a lever). **4.6 The account confronted**: E5 held-out prediction; null→gap overlay + t\*(δ) sweep (E8.3) | §3.2 (as 4.1) + §5 (all) + §6.2–6.3 |
| §5 Application | **two safety maps** — per-example (batch-1 worst case) and batch-aggregate (what the shipped trainer consumes; E3), verdicts at ε\*; σ-conditional training on the measured-safe route; E4 A/B turns 14% from projected to measured; cost accounting | §7 + E3/E4 |
| §6 Limitations | ε\*-relative coverage, debiased-coverage, operating point (measured by E7 if it runs) | §8 |
| §7 Conclusion | metrology + counterexample + map, one epistemic-status recap | §9 |
| App | instrument details; **raw/historical tables** (moved out of main text); extra figures | App A/B + §5's raw tables |

Ordering rationale kept from the previous plan: the null's *scoring*
(4.2/4.6) stays after the instrument and curves because the bridge
consumes measured G(σ) and a calibrated A — confronting before the
curves would hide that dependence. Only the null's *statement* moves
forward into theory.

## 4. The floor waterfall (§4.5)

Floor_e as an explicit additive ledger, one stacked bar or 4-column
table per route, components measured by designated probes:

    Floor_e = reenc (≤ band, by control)
            + target-content share (≈ 0 — E2 α-flat)
            + RoPE_e (erased by PI at endpoint)
            + Resid_e (remainder; carries the absolute-size governor)

Debiased anchors in hand: 896 ≈ +0.02–0.04 total; 768 ≈ +0.056
(RoPE_e the large majority — G10 tempered, "not exactly all");
512 ≈ +0.30 (RoPE_e ~0.096 raw-paired, Resid_e the bulk). Caveat kept:
reenc control is a *proxy* for the pipeline cost demotion actually
pays; the optional demote→re-promote arm (§7) closes it empirically.

## 5. Figure/table plan

- **Fig 1** (enlarged, first page): measured debiased gap curves +
  the null's predicted curves at its published δ — the confrontation
  visible before any prose. (Fig 1c debiased map exists,
  `plot_debiased_map.py`; RAPSD σ\* vlines removed — the null curves
  come from E8.3.)
- **Fig: floor convergence** (§4.3): endpoint vs x-zero vs α-sweep per
  route — three probes, one number.
- **Fig: floor waterfall** (§4.4 above).
- **Fig: t\*(δ) sweep** — x: δ (log), y: t\*, three near-coincident
  route curves (family spread ≤ 0.13), measured boundaries as markers,
  δ_reenc as vertical anchor (needs `reenc_noise_floor.py`, E4).
- **Table: two safety maps** (per-example / batch-aggregate), columns:
  route, Floor_e (debiased), σ\*, verdict at ε\*.
- **Table: claims ledger** — claim, status label (pre-registered /
  confirmatory / post-hoc), probe, section. This is the consolidated
  hedge (§2 policy).

## 6. Order of work

1. ~~E5 analysis~~ **DONE 2026-07-29** (`paper_bench/e5_holdout.py`,
   run `runs/20260729-1130-e5-holdout/`) — qualified PASS, voice set
   (§2). The overlay figure is a §4.6 candidate as-is.
2. Restructure `main.tex` to the §3 spine — mostly *moving* existing
   debiased blocks, not rewriting; raw tables to appendix; hedging
   pass per §2 policy (structural out, ledger in).
3. **E8.3 analysis** → overlay + t\*(δ) figures (δ_reenc row waits on
   E4's `reenc_noise_floor.py`).
4. **E3** pooled-with-self-floors run → two-maps table (§5). Kick off
   E7's controlled-LoRA train alongside (GPU-cheap).
5. **E4** (Phase 1b A/B + `reenc_noise_floor.py`) → 14% measured,
   δ_reenc anchor lands, clear remaining [pending] markers.
6. Hygiene pass per `required_experiments.md` [FIX] list: `make_all.py`
   + MANIFEST, results archive matching the repro statement, abstract
   shortened, claims-ledger labels linked to freeze commits.
7. E7 probes / E6 arms if targeting a top venue.

## 7. Optional items (not gating)

- **Demote→re-promote arm** (pixel down→up at native grid, encode):
  Floor = 0 by construction, isolates the input branch across the full
  σ axis, tests two-term additivity per bin (gap_demote ≈
  gap_repromote + Floor_e); built-in falsification (endpoint must be
  in band). The cleanest independent check of the form — worth one run
  if E5 is attempted. Add as E9 in `required_experiments.md` if
  adopted.
- **E6/E7 generalization arms** per their existing entries.
