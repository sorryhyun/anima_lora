# paper_plan — restructure direction for the sigma_lowres paper

Companion to `README.md` (the experiment index) and
`experiments/<eN>/README.md` (per-experiment records): those say *what
ran and what it showed*; this file says *what the manuscript becomes*,
and §9 records where the manuscript currently stands. Rewritten
2026-07-29, superseding the Branch-A plan (E1 fired rule 1, the
gap-native restructure is already in `paper/main.tex` — that plan is
discharged; its claim-level revisions all landed).

**Status:** the manuscript is currently the Branch-A spine (metrology →
account → measurements → null-in-gap-units → application). This plan
restructures it once more, into **theory → evidence → application**,
to fix the remaining structural problem: it reads as an exploration
log — evidence organized by run/question instead of by claim, hedges
scattered per-sentence instead of consolidated.

**Updated 2026-07-29 (same day) after an external theory review of §3.**
The review's core findings are adopted: the *derived* object is the
four-term angular expansion d = S + F + I + R (data branch, graph
branch, interaction, remainder); the two-term account is its
*reduction* under named assumptions; the local law cosine geometry
fixes is quadratic, so the current `A·m/G^p` form is neither derived
nor honestly-labeled empirical — an analysis-only E5 refit decides the
reported form. Concrete text fixes in §8 below; E9 promoted in §7.

---

## 1. Direction in one paragraph

Claim-first. The paper states the account as its theoretical object up
front, with a two-level ledger. **Derived unconditionally**: the
four-term angular expansion gap = S_e + F_e + I_e + R_e (data-branch
share, graph share, projected interaction, higher-order remainder),
from the perturbation expansion plus the local cosine identity
1−cos ≈ ½‖δg⊥‖²/G². **Derived conditionally (the reduced two-term
account)**: S_e ≈ A_e·(M(σ)/G(σ))² + Floor_e under named assumptions
A1–A4 (small remainder; graph-relative stationarity ‖P⊥C‖/G ≈ const;
data-branch factorization; negligible projected interaction). The
**coefficients** (A_e, M(σ), G(σ), Floor_e) are measured, not derived;
the *fitted* power form (m/G)^p with explicit p is labeled empirical,
never derived — cosine geometry fixes quadratic locally and nothing
else. That ledger is stated once, prominently, in the theory section —
and then the rest of the paper is written assertively, without
re-hedging each claim. Every
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
`experiments/e5/`).** Eq. 3 headlines as a **predictive**
first-order account with stated resolution: held-out routes at ~0.09
RMSE (768 beats the oracle smooth on its own data; 1280→1024 within
the 2× gate), governors upgraded from "consistent with" to measured —
ratio-twin amplitudes agree at z=0.14, and the floor exp-law
F(n) = 0.70·exp(−n/1041 tok) hits the held-out 768 floor dead-on
(+0.088 predicted vs +0.092±0.012 measured). Stated limits, verbatim
in the theory/evidence sections: the prediction is NOT within ε\*
anywhere (χ²/bin 7.6–9.2); A(r) interpolation rests on two distinct
ratios and is evidently convex (the 1280→1024 peak overshoot).

**The 768 mid-σ window is reframed (external review).** Previously
booked as "the account's one structural failure" (measured ≈ 0 below
the predicted floor — a positive two-term form cannot do that). Under
the four-term expansion it is the interaction term's signature:
I_768(σ) < 0 in that window (equivalently, graph-relative
stationarity A2 fails there). The reduced account fails exactly where
its assumptions say it can; the fuller derived form represents it.
Present it as delimiting the reduction's domain, not as an anomaly —
and E9 is the designated direct probe (§7).

## 3. New section spine (current tex → new)

| new | content | built from (current tex) |
|---|---|---|
| §1 Intro | claim-first; narrowed headline; 14% as projected ceiling until E4 | §1, tightened |
| §2 Related | unchanged framing ("we test the tempting extension, not their methods") | §2 |
| §3 Theory: the demotion gap and a two-term account | **3.1 Estimand and resolution**: population objects g_q(σ), d_e(σ) = 1−cos = ½‖ĝ_nat−ĝ_e‖², then *separately* the reported quantity — the reenc-excess Δ_e = d_e − d_reenc (Δ_e can be negative; d_e cannot); the aggregation operator stated as part of the estimand (per-example mean vs batch-gradient cosine — coefficients need not transfer, so E3 is estimand definition, not application detail); redraw floor, reenc control; ε\* renamed **median certification resolution** with the power-adjusted margin (z_α+z_β)·SE in a footnote. *Definitions only, no demonstrations.* **3.2 Data and graph perturbations**: δg = B_e + C_e + R_e ("data branch" — demotion changes input *and* FM target; the residual derivative D_x r[δx] = D_z f[(1−σ)δx] + δx keeps target-content dependence alive at σ=1, so the endpoint is NOT graph-only by construction — that equality is an *empirical* E1/E2 result). **3.3 Angular geometry**: derived d = S + F + I + R; then A1–A4 → the reduced two-term account with quadratic local law A·(M/G)²; fitted-p form labeled empirical (E5 refit decides which is reported). **3.4 The spectral null**: tolerance family as the null of the *input-mediated part of B_e only* — it does not constrain the target share, C_e, I_e, or the Jacobian gain; E8.3 is a "spectral baseline transported into gap units by one-route calibration," not parameter-free. **3.5 Discriminating predictions and falsifiers**: endpoint/x-zero/α convergence, graph stationarity, route factorization, low-G amplification, and the interaction test (E9); each names its §4 probe | §3.1 + §3.3 (defs only) + §4 (all) + §6.1 (family statement) |
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

1. ~~E5 analysis~~ **DONE 2026-07-29** (`experiments/e5/e5_holdout.py`,
   run `runs/20260729-1130-e5-holdout/`) — qualified PASS, voice set
   (§2). The overlay figure is a §4.6 candidate as-is.
2. ~~E5 three-form refit~~ **DONE 2026-07-29**
   (`experiments/e5/e5_refit.py`, run `runs/20260729-1322-e5-refit/`) —
   the exact angular link `1−1/√(1+(c·m/G)²)` headlines (held-out mean
   RMSE 0.071 vs 0.093/0.094; cures the 1280→1024 overshoot, retiring
   E5 miss #2), labeled empirical via its `‖δg⊥‖ = c·m` loading; the
   derived quadratic is its small-κ limit (p\* scan lands on 2.00
   exactly). Floor law form-invariant. See `experiments/e5/`.
3. ~~Restructure `main.tex` to the §3 spine~~ **DONE 2026-07-29**,
   landed together with the question-first reframe (new title "When
   Does Training on Downscaled Images Yield the Same Gradients?";
   §5 = "the answer, in deployable form"). §8 fixes + data-branch
   rename + claims-ledger table all in; raw tables in a dedicated
   appendix; abstract deliberately untouched (revised after review of
   the rewrite). See §9 below.
4. **E8.3 analysis** → overlay + t\*(δ) figures (δ_reenc row waits on
   E4's `reenc_noise_floor.py`).
5. **E3** pooled-with-self-floors run → two-maps table (§5). Kick off
   E7's controlled-LoRA train alongside (GPU-cheap).
6. **E4** (Phase 1b A/B + `reenc_noise_floor.py`) → 14% measured,
   δ_reenc anchor lands, clear remaining [pending] markers.
7. Hygiene pass per the README's open-work [FIX] list: `make_all.py`
   + MANIFEST, results archive matching the repro statement, abstract
   shortened, claims-ledger labels linked to freeze commits.
8. E7 probes / E9 / E6 arms if targeting a top venue (E9 first among
   these — see §7).

## 7. Optional items (not gating a first complete draft)

- **E9: demote→re-promote arm** (pixel down→up at native grid,
  encode) — **promoted 2026-07-29: run before any top-venue
  submission** (still not gating a first complete draft, because the
  four-term framing carries the "derived" claim without it — only the
  *reduction* is assumption-conditional). Floor = 0 by construction,
  isolates the data branch across the full σ axis, and is the direct
  interaction probe: I_e ≈ Δ_demote − Δ_repromote − Floor_e per bin.
  The 768 mid-σ dip predicts I_768 < 0 there — a pre-registrable
  sign. Built-in falsification (endpoint must be in band). One
  verdict-scale run. **Done 2026-07-31 — see `experiments/e9/`.** If I_e is bounded over the claimed region, the two-term
  reduction keeps "derived (conditional)"; if not, the manuscript
  keeps I_e explicit in Eq. 3 and the reduction is demoted to a
  semi-empirical account — either way the paper is consistent.
- **E6/E7 generalization arms** per their existing entries.

## 8. Text fixes from the external review (apply during step 3)

Concrete, evidence-already-in-hand; none change any measurement:

- **Endpoint claim (`paper/main.tex:501` region):** replace "each
  isolating the graph term by construction" with: at σ=1
  *input*-content dependence vanishes by construction, but
  *target*-content dependence need not (the +δx term); the x-zero and
  target-α probes establish empirically that the remaining
  contribution is below resolution at our operating point. E2's
  α-slope ≈ 0 is exactly this discharge — the claim gets *stronger*,
  not weaker.
- **"Expected gradients are never noise-masked" (§3.1(a)):** too
  absolute. Replace with: input noise masking alone provides no bound
  on expected-gradient disagreement, because systematic target and
  graph terms survive noise averaging.
- **ε\* (`§3.3/eq:epsstar`):** rename to "median certification
  resolution" — at margin exactly 1.645·SE a truly-safe route
  certifies with only ~50% power. Footnote the general margin
  ε\*_{α,β} = b_D + (z_{1−α}+z_{1−β})·SE (b_D = residual finite-draw
  bias allowance).
- **Estimand blur:** Eq. 3 is population demotion distance; E5 fits
  the reenc-excess Δ_e. State both objects in §3.1 and say which every
  figure/table reports (Δ_e can be negative; d_e cannot).
- **Rename "input branch" → "data branch"** throughout (text +
  figures): the branch includes the FM-target change, and the paper
  already defines it that way — the current name is a mislabel.
- **Spectral-null scope sentence (§3.4):** the null constrains only
  the input-mediated part of B_e — not the target share, not C_e, not
  I_e, not the Jacobian gain; E8.3 is a calibrated transport into gap
  units, not a parameter-free prediction.

## 9. Manuscript status

*(theory→evidence→application + question-first reframe, written
2026-07-29; moved here from the old `completed_experiments.md`.)*

Superseded the Branch-A spine the same day: the §3 spine above landed
together with the question-first reframe (title now "When Does Training
on Downscaled Images Yield the Same Gradients?"; §1 leads with the
practical question, the map is presented as the answer). New structure:
§3 Theory (3.1 estimand — d_e vs reenc-excess Δ_e both stated,
aggregation operator part of the estimand, ε\* renamed "median
certification resolution" with the power footnote; 3.2 data/graph
branches — endpoint NOT data-free by construction, "data branch" rename
throughout; 3.3 four-term expansion d = S+F+I+R derived + A1–A4
reduction + exact angular link labeled empirical; 3.4 spectral null
with exact scope sentence; 3.5 seven discriminating predictions each
naming its probe), §4 Evidence (4.1 instrument+debiasing+"debiased
units only from here" + consolidated coverage statement; 4.2 phenomenon
+ Table-null boundary scoring; 4.3 endpoint≡x-zero≡α-flat; 4.4 data
branch + governors; 4.5 depth + RoPE/Resid waterfall; 4.6 the account
confronted — E5 held-out + three-form refit + 768 mid-σ dip as I_e<0
interaction signature delimiting the reduction's domain + E9 designated
+ claims-ledger table as the consolidated hedge), §5 Application
("the answer, in deployable form"), §6/§7 updated. Raw/historical
tables all moved to a dedicated appendix (incl. new raw-vs-debiased
endpoint revision-record table); per-sentence hedges replaced by the
ledger. E5's refit figure `figs/e5_refit.png` is the §4.6 figure
(`figs/e5_overlay.png` also staged). Abstract intentionally NOT
revised yet (user will revise after reading the rewrite; it still says
"input term"). Compiles clean under tectonic (0 overfull, no broken
refs, 22 pp).

**Still open in the manuscript**: none — the last **[pending]**
marker (results tarball) was removed 2026-08-04; the abstract now
carries the repo link, and *publishing* the raw-run tarball itself
stays on the reproducibility-deliverables list in the README. Cleared
since: E8.3 overlay + t\*(δ) figure (in the appendix as `fig:e83`), E3 pooled-with-self-floors run + a + b/B fit (2026-08-04,
`runs/20260804-0002-e3-pooled-verdict/`, §4 map paragraph rewritten), E2's
marker (2026-07-29, measured sweep), E4's A/B + `reenc_noise_floor.py`
(2026-07-30), E7's membership probe (2026-07-29), §4.5 reenc-proxy +
§4.6 probe pendings (2026-07-31, with the E9 write-in).

**Figures:** Fig 1c regenerated in debiased units 2026-07-29
(`plot_debiased_map.py` → `figs/gap_debiased.png` — paired per-image
map with the bin-level ±ε\* band; RAPSD σ\* vlines removed, to be
addressed separately). Remaining raw figures are marked as raw in
captions; Fig-1 enlargement + waterfall figures are still owed.
