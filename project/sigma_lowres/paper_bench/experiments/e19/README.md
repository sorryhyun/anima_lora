# E19 — why are B and C anti-aligned? Locating the birth of the near-cancellation

| | |
|---|---|
| **Status** | **DONE 2026-08-08 (19.0–19.6)**. Verdict chain: the anti-alignment's mid-σ depth is **Jᵀ-born** (19.2), with a weak residual-level seed the Gaussian closure derives in sign/shape/ordering (19.1); the Jᵀ mechanism is **global** — depth-uniform, type-uniform, magnitude ∝ branch energy (19.3, early-band localization refuted); and at the noise-dominated bins it is **partially RoPE-phase-mediated** — PI-aligning phases rotates C (cos(C, C_pi) 0.50–0.67 on 768) and halves the anti-alignment, growing h(C) ~2× (19.4). The whole read is **operating-point clean**: rerunning the 19.2 probe bit-matched with `anima_soup_sincos` merged leaves both legs and the native residual field unmoved (dircos ≈ 1, amp within ~3% at every gated bin, all routes) — leg-level operating-point invariance, extending E7's map-level null (19.6). Working account: model-level scale covariance along the demotion diagonal, with a **phase-borne component of the cancellation** the account must now carry |
| **Question** | [E14](../e14/)'s headline turned the measured gap curve into the *residual of a near-cancellation*: at every σ the data-branch and graph-branch perturbations are strongly anti-aligned (ρ ≈ −0.7…−0.9), both individually far larger than the realized gap, and the σ≈0.4–0.7 cliffs are the \|B⊥\|/\|C⊥\| = 1 crossings. This is why the spectral account misses by ~4× — it transports amplitude without the interference structure. So the theory-facing question is no longer "a law per branch" but: **where is the anti-alignment born — in the residual field r (a property of the trained denoiser + data statistics) or in the pull-back through Jᵀ (a property of the graph/Jacobian)? And is it derivable?** |
| **Depends on** | [E14](../e14/) (headline + `vector_ledger.py`; unit-honesty rule), [E9](../e9/) (crossing↔window localization), [E17](../e17/) (Gaussian-closure machinery), [E11](../e11/) (residual directions norm-only; `--save_residuals`), Q2/G10/G11 in `record/questions.md`, [E7](../e7/)+Q3 (adapter axis), [E18](../e18/proposal.md) (per-draw projection hook — the storage-free alternative 19.3 didn't need) |
| **Instruments** | 19.0 `reads_190.py` (committed-ledger re-reads); 19.1 `bench/run_closure_rho.py`; 19.2 `run_prior_distance.py --repromote --save_residuals` + `bench/ledger_rho_r.py`; 19.3 `run_sigma_probe.py --repromote --keep_arm_sums` (now dumps `groups.json`) + `paper_bench/ledger_depth.py`; 19.4 `--pi_align --repromote` in one process + `paper_bench/ledger_pi.py`; 19.5 `bench/ledger_b_scoreshift.py` (CPU, 19.2 store × 19.1 closure); 19.6 `run_prior_distance.py --adapter` + `bench/ledger_operating_point.py` |
| **In the paper** | The cancellation account is the theoretical spine of the follow-up paper (§3/§4.6 of paper 2); the current draft's Fig. `ledgergeom` + "why the legs anti-align" paragraph stand, now with measured backing (see 19.3 item 4). [appendix.md](appendix.md) holds the per-σ geometry figure |

## The reframe

E14 established *that* the curve is a cancellation residual; nothing yet
says *why* the two interventions oppose. One observation orders the
hypothesis space: near-cancellation is exactly what the safe-route
phenomenon **requires**. If demoting data+graph *together* is
quasi-equivalent (small realized gap) while each half-intervention is
large, then B ≈ −C is forced wherever the route is safe — the joint
demotion direction is a near-flat direction of g(data, graph) even
though neither axis direction is. That reading — approximate **scale
covariance of the trained model along the demotion diagonal** — makes
an immediate prediction: cancellation completeness should track the
safety map (best on 896, degraded on 512).

The discriminator ladder, cheap→expensive, with verdicts:

| level | question | verdict |
|---|---|---|
| **L0** (19.0) | what do the committed ledgers already pin? | completeness tracks the safety map; unsafe = mismatched magnitudes, not decoherence; headline not a shared-arm artifact |
| **L1** (19.1+19.2) | residual-level or Jᵀ-born? | **Jᵀ-born**; weak derivable r-level seed (closure gets sign/shape/ordering, misses level ~1.5–2×) |
| **L2** (19.3) | depth-localized in the Q2 band 3–8, or uniform? | **depth-uniform and type-uniform**; interaction magnitude ∝ branch energy at every depth |
| **L3** (19.4) | is the graph side RoPE-causal — does PI-aligning phases rotate C? | **C ROTATES, partially** — cos(C⊥, C_pi⊥) 0.50–0.67 (768), ρ −0.89…−0.92 → −0.34…−0.60; residual anti-alignment survives, and h(C_pi) ≈ 2×h(C) at σ 0.7/0.83 — the near-cancellation is partly phase-borne |

## 19.0 — re-reads from the committed E14 ledgers (0 GPU; `reads_190.py` → `frozen_target_190.json`, DONE 2026-08-06)

All numbers reliability-gated at rel_cos ≥ 0.5, both data refs
(reenc = primary, matching `ledger.json`).

1. **Completeness ordering CONFIRMED** under both refs. Mean cancellation
   fraction 1 − h(B+C)/(h(B)+h(C)) over reliable bins, 896/768/512:
   **0.770 / 0.725 / 0.631** (reenc); 0.638 / 0.593 / 0.343 (native).
   512 least complete — the safety-map ordering scale covariance predicts.
2. **Decoherence REFUTED.** 512 keeps mean |ρ| = 0.864/0.824
   (reenc/native) — indistinguishable from 896 (0.860/0.727) and 768
   (0.891/0.802) — while its amplitude ratio √(S/F) never crosses 1
   (0.63–0.95). The angle is route-uniform; what breaks on the unsafe
   route is magnitude matching. **Unsafe = mismatched magnitudes** — the
   open half is the amplitude law.
3. **Shared-arm validity closed.** B and C share the repromote arm with
   opposite signs, biasing a naive ρ negative; the instrument's cross-set
   products handle it. Mean |I_same − I_cross| = 0.062, inflation up to
   ~2.3× at a few low/mid-σ bins — the artifact is real and the debias
   load-bearing — but the headline uses cross-set only and stays
   −0.7…−0.9. **Not a shared-arm artifact.**
4. **Crossings vs E9.** 896: σ ≈ 0.47 (reenc) / 0.53 (native), consistent
   with E9's ≈ 0.5. 768: **no in-window crossing** on E14's grid (ratio
   peaks ≈ 0.99 at σ = 0.0875, 0.76–0.88 through the E9 window) — the
   crossing↔window-center localization survives cleanly on 896 only; on
   768 the window sits where the ratio is closest to 1 from below.
   Frozen target for 19.1 = E14's table, not E9's.
5. Endpoint σ=1 and gate-failed cells are non-verdict-bearing
   (ratio-of-small-numbers; 896's endpoint fails both rel gates).

## 19.1 — closure-predicted ρ_r (theory first; `run_closure_rho.py`, FROZEN 2026-08-06 before any 19.2 run)

E17's fitted per-arm Gaussian closures, extended to paired branch
predictions on the demoted grid: B_r = D_e r̂_rp − D_e r̂_ref,
C_r = r̂_dem − D_e r̂_rp (so B_r + C_r is the exact measured mismatch
object; D_e = the instrument's area-downsample, pinned over the
proposal's up-sampling A_e which would carry the promote operator's
smoothing in one branch). ⊥ projects out D_e r̂_native per image;
pooled inner products (E11's norm-only verdict). Probe-matched to E14.
Controls: cross-fit split (shared-rp-model bias), bicubic vs area,
both refs. **The wager**: E17 failed on *amplitude* but reproduced
shape — ρ is scale-free, so the closure gets a second shot where its
failure mode doesn't bite. Run `20260806-2223-e19-closure-rho`,
summary `prediction_191.json`:

1. **Sign: ρ_r < 0 at every (route, bin), all three closures, both
   refs** — the pre-registered coherence criterion passes everywhere.
2. **Magnitude: weak.** Verdict bins (σ ≥ 0.3, area, reenc):
   ρ_r ≈ −0.06…−0.14, deepening to −0.22/−0.31/−0.39 (896/768/512) only
   at σ = 1. Far shallower than the measured g-level −0.7…−0.9.
   Per-image cosines equally weak (−0.10 ± 0.06 at σ=0.57).
3. **|B_r⊥|/|C_r⊥| ≈ 0.21–0.61, no crossings** — misses the measured
   896 crossing. |ρ_r| is U-shaped in σ (weakest σ ≈ 0.17–0.43), unlike
   the deep-flat g-level profile.
4. **Controls clean**: cross-fit ≈ same-fit (Δρ ≤ 0.01); bicubic 2–3×
   more negative with stable sign (operator owns magnitude, not sign);
   holdout ≈ fit.

**Pre-registered reading rule for 19.2** (verdict bins, reenc, area):
measured |ρ_r| ≲ 0.35 → seed derivable-and-weak, depth is Jᵀ-made
(L2/L3 take over); |ρ_r| ≥ 0.5 with the g-profile → residual level owns
it, beyond-Gaussian closure next; ρ_r ≥ 0 → closure sign falsified,
Jᵀ-born outright.

## 19.2 — the r-level ledger (measured; run `20260806-2342-e192-rho-r-measured`, `measured_192.json`, DONE 2026-08-07)

`run_prior_distance.py --repromote --save_residuals` on E14's 40-image
probe list (arm latents bit-identical to 19.1's cache), 16 draws, E14's
15 σ centers; `ledger_rho_r.py` mirrors `bc_ledger`'s cross-half debias
(shared-rp + shared-ref corrections). Estimand = aggregate mean residual
contrast (E11: per-image directions are idiosyncratic). All verdict
cells pass rel ≥ 0.5 (rel_B 0.73–0.99, rel_C 0.86–1.0).

1. **L1 verdict: the mid-σ anti-alignment depth is created through Jᵀ.**
   At σ = 0.3–0.83 (reenc, area), measured ρ_r = **−0.12…−0.22** on all
   three routes while ρ_g ≈ −0.83…−0.96 — every gated mid-σ cell
   classifies `weak_derivable_seed` under the pre-registered rule.
2. **The seed is real, and the closure derives its structure.** ρ_r < 0
   everywhere (frozen sign prediction holds); measured σ-profile is the
   predicted U-shape, with the |ρ_r| minimum exactly where ρ_g is
   deepest; endpoint deepening route-ordered as predicted (−0.33/−0.49/
   −0.61 vs closure −0.22/−0.30/−0.39 — same ordering, uniform ~1.5–1.6×
   scale miss). Magnitude under-predicted ~1.5–2× throughout — the same
   failure axis E17 recorded, opposite direction.
3. **No r-level amplitude crossings** (ratio 0.34–0.76, rising, never 1):
   the g-ledger's 896 crossing at σ ≈ 0.47 — the cliff mechanism — is
   also Jᵀ-made.
4. **σ → 1 tail: the one regime the residual level partly owns.** 512
   reaches ρ_r −0.53…−0.62 at σ ≥ 0.9625; 768 tops at −0.49; steepest on
   the unsafe route (r̄ → x − x̄_prior limit). The paper-facing mid-σ
   claim stays Jᵀ-born.
5. **Controls**: shared-arm correction load-bearing (naive −0.18…−0.67
   vs cross-half −0.12…−0.26); bicubic 1.5–2.5× deeper, sign stable;
   native ≈ reenc; per-image cosines weak (−0.14…−0.19 ± 0.04).

## 19.3 — depth-resolved ρ_ℓ (run `results/20260807-0745-e193-depth-ledger`, `depth_193.json`, DONE 2026-08-07)

E14's `arm_sums/` store was reclaimed → reduced-grid rerun: E14's 40
images, routes {896, 768}, σ = E14's four crossing-region bins
**bit-exactly** (centers 0.3/0.4333/0.5667/0.7) + endpoint,
`--repromote --keep_arm_sums --self_floor --deterministic`; 5.7 GPU-h,
18 GB store. Instrument (committed first, `fa3b0352`): stores carry
`groups.json` (28 blocks + 15 module types incl. `self_attn_up_{q,k,v}`
row splits); `ledger_depth.py` runs the E14 estimator per slice —
slice-local ρ_ℓ (rel-gated 0.5) **and** a global-⊥ partition S/F/I_part
that resums to the global cross-set S/F/I exactly (verified ≤ 1e-5).
The rerun replicates E14's globals at the shared bins (ρ_g −0.87…−0.95;
896's endpoint gate-fails as E14 recorded).

1. **L2 verdict: early-band (3–8) localization REFUTED — ρ_ℓ is
   depth-uniform.** Every gated block reads deep at the mid-σ bins, both
   routes/refs: ρ_ℓ ∈ [−0.99, −0.56], median ≈ −0.93 (14–18/28 blocks
   read per bin; gated-out cells are the low-energy early/mid blocks).
   Gate-free cross-check: the parts-derived per-block ratio is
   deep-negative for **all 28 blocks**, band 3–8 included (896:
   −0.86…−0.98; 768: −0.74…−0.97).
2. **Interaction magnitude tracks branch energy exactly.** Per-block
   mid-σ I shares match S and F shares block-by-block: block:27 owns
   43%/52% (896/768) of I but equally 36–55% of S/F; block:01 14–19% of
   all three; band 3–8 4–6% of all three. **Zero excess interaction at
   any depth.** Only depth texture: block:27 is C-heavy, block:01
   B-heavy — the crossing structure is depth-textured even though the
   angle is not.
3. **Type-uniform too.** All 15 module types deep where gated
   (−0.8…−0.97 at mid σ) — including cross-attn and MLP, which RoPE
   never touches. The fused-qkv row splits show **no q/k-vs-v excess**
   anywhere they read — the RoPE-row discriminator comes back negative.
4. **Joint reading with 19.2.** The pre-registered tree said
   depth-uniform "corroborates the r-level account" — but 19.2 measured
   that seed weak. Joint verdict: Jᵀ-born yet **not localized**, so the
   theory target is not an early-block operator's shared range but
   model-level scale covariance along the demotion diagonal. If the
   joint demotion direction is a near-flat direction of the trained
   function, B ≈ −C holds at the function level and every parameter
   slice inherits the angle through the chain rule — which is what the
   ledger shows, down to the energy-proportional magnitude split.
5. **Endpoint (non-verdict).** σ=1 is ratio-of-small-numbers; 896 fails
   both gates. Directionally, 768's endpoint reads deep across blocks
   6–27 while 896's early blocks flip weakly positive — consistent with
   19.2's σ→1 tail finding, same route ordering.

## 19.4 — the PI-align causal arm (DONE 2026-08-07; run `bench/results/20260807-1400-e194-pi-causal`, ledger `pi_194.json`, instrument committed `610c8958`)

`--pi_align --repromote` in ONE process (kernel-path rule): the `<e>pi`
arm re-runs the demoted-graph forward with PI-stretched RoPE phases, so
C_pi = ḡ_dem,pi − ḡ_rp isolates "graph minus its phase geometry". If the
graph branch's anti-aligned component is RoPE-mediated, C must
**rotate** (ρ_pi toward 0, debiased cos(C⊥, C_pi⊥) well below 1, the
crossing moves), not merely shrink.

Pre-registered, with G11's qualifier built in: the clean read is
**noise-dominated bins only** — grid = E14 bins 0.7 / 0.8333 / 0.9625
(bit-exact) + endpoint; at mid σ PI is off-manifold with content
(768pi measured *worse* through σ 0.56–0.81), so a "worse" result there
is expected and not a falsification. Route 768 primary (largest
RoPE_e), 896 as small-floor control. After 19.3's null q/k-row read and
type-uniformity, the PE-mediation prior is weakened: a no-rotation
result now coheres with the global scale-covariance account rather than
contradicting Q2/G10.

Verdict (reenc ref; all three verdict bins gated on both routes):

1. **C ROTATES.** 768: cos(C⊥, C_pi⊥) = **0.504 / 0.648 / 0.671** at
   σ = 0.7 / 0.8333 / 0.9625, with ρ → ρ_pi = −0.891→−0.344,
   −0.924→−0.534, −0.845→−0.600. The pre-registered rotation
   criterion (direction change, not mere shrinkage) is met — the
   graph leg's anti-aligned component is **phase-geometry-mediated**.
2. **Partially.** ρ_pi does not reach 0 at any verdict bin
   (−0.34…−0.60): a phase-independent share of the anti-alignment
   survives PI-stretching — mediation, not sole ownership.
3. **The near-cancellation is partly phase-borne.** h(C_pi) ≈ 2× h(C)
   at σ 0.7/0.8333 on 768 (0.178→0.307, 0.084→0.176) and
   \|B\|/\|C_pi\| drops to 0.28–0.30 (vs 0.76–0.87 for C): with phases
   PI-aligned the graph leg *grows* and stops magnitude-matching B —
   the phase geometry is part of what lets the legs cancel on the
   safe route.
4. **896 control, same direction, weaker**: cos 0.572/0.733/0.872,
   ρ −0.893→−0.413 at σ = 0.7 — ordering consistent with the smaller
   RoPE_e floor; rotation fades toward the endpoint on both routes.
5. **Endpoint non-verdict as expected**: 896 relC = 0.48 gate-fails
   and its ρ = −1.066 is the small-denominator artifact; 768's
   ρ_pi = −0.003 at σ = 1 is directionally striking but unreadable at
   these reliabilities.

Joint with 19.3: phase-mediated yet depth/type-uniform — the phase
share of the anti-alignment is carried globally (every block/module
inherits it through the chain rule), not by an early-band RoPE
operator. The scale-covariance account keeps the flat-diagonal claim
but must now carry an explicitly phase-borne component of the
cancellation; Q2/G10's PE-mediation reading survives in this weakened,
delocalized form.

## 19.5 — measured-vs-closure leg DIRECTIONS (run `20260807-1928-e195-dircos`, `dircos_195.json`, DONE 2026-08-07)

`bench/ledger_b_scoreshift.py` (committed `68c87d57` before the run):
pooled direction cosine between the 19.2 measured r-level legs and the
19.1 closure-predicted legs, per (route, σ, closure), common
⊥-to-measured-native subspace, measured second moments cross-half
debiased (shared-ref correction), reenc/area primary. The question the
Yang-Song picture poses: is B literally the *derivable score-shift* of
demotion, with the beyond-Gaussian structure living in the graph leg?

1. **The closure owns BOTH leg directions at mid/high σ.** On gated
   verdict bins σ ≥ 0.4333, all three routes: dircos_B 0.64–0.87,
   dircos_C 0.76–0.95, and the resultant's direction is nearly exact
   near the endpoint (dircos_net 0.94–0.99). The 19.1/19.2 magnitude
   miss is mostly a **scale story, not a direction story**.
2. **The pre-registered "B is the derivable score-shift" contrast comes
   back REVERSED**: gap = dircos_B − dircos_C has gated median
   **−0.104 / −0.111 / −0.109** (896/768/512; 30 gated bins each,
   closures near-identical) — the *data* leg is the less-predicted one
   at every mid/high-σ bin. Read with the score-field picture: the
   manifold (beyond-Gaussian) content of p_σ lives on the data side;
   the graph leg is operator-like and second-order statistics capture
   its direction better.
3. **σ-profile matches the score-field intuition**: every leg's dircos
   falls monotonically toward low σ (≈ 0.20–0.25 at σ = 0.0125) —
   where the score is manifold-dominated the Gaussian closure loses
   the *direction* first, long before the verdict window. (Low-σ bins
   non-verdict as usual.)
4. **Per-leg amplitude resolves the level miss**: the closure
   over-predicts leg energy at mid σ (amp pred/meas ≈ 1.15–1.31 for B,
   1.39–1.60 for C at σ 0.3–0.57) and under-predicts toward the
   endpoint (B ≈ 0.68–0.86) — the 19.2 σ-uniform ~1.5–2× ρ-level miss
   decomposes into a σ-*dependent* per-leg scale miss.
5. **Controls tight**: bicubic / native-ref / predicted-native
   projector / no-projection move dircos by ≤ 0.04 at the checked
   bins; diag/block/octave agree to ~0.01.

Operating-point caveat, now explicit: the 19.2 store (and hence this
read) is **base-DiT**, while the E14 g-ledger is **sincos-attached** —
the two ledgers the line compares straddle operating points. 19.6
closed exactly this: leg-level invariance, so the straddle is benign.

## 19.6 — adapter operating-point arm (DONE 2026-08-08 — **leg-level operating-point invariance**; run `20260808-0102-e196-rho-r-sincos`, ledger `20260808-0625` → [opdiff_196.json](opdiff_196.json))

`run_prior_distance.py --adapter` (default None keeps the 19.2
estimand) reruns the 19.2 probe bit-matched (same probe list, latent
cache, seed, draws) at the shipped `anima_soup_sincos` operating point;
`bench/ledger_operating_point.py` then diffs the two stores in x-space
— pooled dircos + amplitude ratio per leg and for the native residual
field itself, common ⊥-to-base-native subspace, both sides cross-half
debiased. Pre-registered: the LoRA-moves-B account predicts
dircos_B < dircos_C − 0.2 (and/or |log amp_B| ≫ |log amp_C|); dircos ≈ 1
with amp ≈ 1 for both legs = leg-level operating-point invariance,
extending E7's map-level null and retroactively cleaning 19.5 and the
r(base)-vs-g(sincos) comparisons throughout E19.

**Verdict — the invariance branch of the pre-registration** (40 images,
16 draws, 15 σ, all 3 routes; all 10 verdict bins pass the rel gate on
every route):

1. **Both legs are operating-point invariant**: dircos_C ≈ 1.02–1.07
   with amp_C 0.98–1.00 across every gated bin and route; dircos_B sits
   at or *above* 1 (1.02–1.42) with amp_B 0.93–1.02. The gated gap
   dircos_C − dircos_B has median **−0.284 / −0.153 / −0.081**
   (896/768/512) — the *opposite sign* of the LoRA-moves-B prediction
   (dircos_B < dircos_C − 0.2), and B's systematic >1 overshoot cannot
   be rotation (rotation lowers cos; overshoot = cross-half debiasing
   dividing by an underestimated B reliability, the noisier leg). The
   debias-free confirmation: **per-image raw cos(base, sincos) is
   0.95–0.98 (B) / 0.93–0.99 (C)** at every gated bin, all routes.
   Read: **LoRA-moves-B refuted; E7's map-level null extends to the leg
   level**, and 19.5 plus every r(base)-vs-g(sincos) comparison in E19
   is retroactively operating-point clean.
2. **The native residual field itself is also unmoved** at the verdict
   band: natres dircos 0.95–1.03, amp 0.99–1.02 at σ ≥ 0.3 (low-σ bins
   overshoot to 1.09–1.22 / amp 1.15 — non-verdict as usual). This
   *supersedes* the smoke's suggestive per-arm raw cos ≈ 0.16–0.24 (2
   images, 4 draws): that was parity noise at small N, not real field
   movement — the full cross-half-debiased pooled read finds none.
3. **Controls tight**: own-projector / no-projection / native-ref
   variants move dircos by ≤ 0.06 at the gated bins (all routes);
   bicubic moves B by up to 0.19 on 896 (the known-noisier control —
   its C stays ≤ 0.03), never in the direction of the refuted branch.

## Decision tree (resolution marked)

| observation | conclusion | status |
|---|---|---|
| 19.1 predicts ρ_r < 0 AND 19.2 measures it | interaction derivable from second-order data statistics | **partial** — sign/shape/ordering derivable; depth is Jᵀ's (closure = seed account, level miss committed) |
| 19.2: ρ_r < 0, but 19.1's closure misses it | residual-level but beyond-Gaussian | not taken (closure did not miss sign/shape) |
| 19.2: ρ_r ≈ 0 while ρ_g ≈ −0.9 | Jᵀ creates it | **✓ (weak-seed variant)** — L2/L3 took over |
| 19.3: ρ_ℓ concentrated in blocks ~3–8 + 19.4 rotates C | graph side is PE-geometry-mediated | **19.3 half REFUTED** (depth-uniform); **19.4 half CONFIRMED** (C rotates, partially) — phase-mediated but **not** depth-localized |
| 19.0: 512 keeps ρ ≈ −0.9 with broken amplitude ratio | unsafe = mismatched magnitudes, not decoherence | **✓** — amplitude law is the open half |
| 19.0: `I_sameset` − cross-set I large | part of the headline is shared-arm noise | **✗** — artifact real but headline cross-set-only |

## Kill switches / honesty

- **Unit-honesty inherited from E9/E14**: at plateau magnitudes
  κ ≈ 0.7–0.9 the S/F/I quadratic is out of its truncation domain —
  magnitudes read only via h(·); ρ, signs, and localization are the
  licensed reads (ρ is an angle statistic and stays in-domain).
- **Reliability gate**: any (route, bin) cell with rel_cos_B/C < 0.5
  does not read (E14's reproducibility floor).
- **Low-σ caution transfers**: 19.2's ρ_r verdict leans on mid/high-σ
  bins first (E17: the measured low-σ excess is itself the least
  certain point), low-σ reported but not verdict-bearing.
- **Conventions**: probe-matched lists when levels are compared; every
  cross-arm cosine inside one process (kernel-path chaos); bin-width
  weights on any WLS over a segmented grid.
- 19.1's prediction was committed (this file + run record) **before**
  19.2 was submitted — the theory-first claim stands.

## Groundings

- **E14** (`runs/20260801-2304-e14-ledger-probematched`): ρ ≈ −0.7…−0.9
  at every σ, cross-set debiased; crossings 896 ≈ 0.5, 768 = 0.688 (E9);
  H-d assumption (iii) fails below σ ≈ 0.45.
- **E17**: all three closures fail the amplitude bar but reproduce shape
  (Pearson 0.94–0.97), route-uniformity, reenc ≈ 0, endpoint — the basis
  of 19.1's scale-free wager.
- **E11**: mismatch directions image-specific (norm-only) — fixes 19.2's
  estimand at the aggregate level.
- **E15**: per-sample ‖Δ‖ ≈ 0.7–1.6‖g‖ vs aggregate 0.15–0.35 — the
  cancellation is an *aggregate* phenomenon.
- **Q2/G10/G11**: depth band 3–8 ≈ 3× (the *floor's* depth profile —
  distinct from 19.3's cancellation-angle read, see appendix.md); RoPE_e
  erased by PI at the endpoint on 768; PI off-manifold at mid σ.
- **E7 + Q3**: map adapter-agnostic, floor level checkpoint-dependent;
  Q3's mixed-res-trained adapter remains the designated
  training-distribution falsifier — scale covariance along the demotion
  diagonal is *trained*, so an adapter trained off-diagonal should
  degrade the cancellation.
- **E18**: the per-draw projection hook would have made 19.3's storage
  question moot; the reduced-grid rerun landed first, so E18's hook is
  now optional rather than shared-with-19.3.

## Cost ladder (actuals)

| item | GPU | note |
|---|---|---|
| 19.0 | none | committed JSON only |
| 19.1 | one VAE encode pass | then CPU on stored latents |
| 19.2 | ~4.5 h | forward-only, latent-sized vectors |
| 19.3 | 5.7 h | reduced grid, 18 GB store (kept on the training box) |
| 19.4 | ~4.8 h | 2 routes × 4 bins, one process |
| 19.5 | none | CPU: 19.2 residual store × warm 19.1 closure cache |
| 19.6 | 5.3 h | 19.2 mirror at the sincos operating point; ledger CPU |

No verdict-grid-scale spend anywhere; the full σ-map stays reserved for
confirming whatever theory survives 19.4.

## Appendix

[appendix.md](appendix.md) — the per-σ B/C geometry figure
(`fig_bc_comb.py` → `bc_comb_768.png`): the E14 legs planted along one
σ axis at true debiased scale, and the reads it makes visible.
