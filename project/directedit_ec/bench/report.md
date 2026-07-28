# directedit_ec — EasyControl cond stream as a learned preservation prior for DirectEdit

## In-place instruction editing probes (2026-07-26)

**Runs:** `results/20260726-{1129-inplace-edit-probe-sfw, 1145-inplace-lambda-sweep,
1213-inplace-src0-probe, 1826-inplace-ecinvonly-probe}` (subject_edit adapter)
plus twin_edit-e3 capability checks `…-1753-phase2p5-edit-probe-twinedit-e3` /
`…-1758-inplace-probe-twinedit-e3` · **Bench:** `run_inplace_probe.py` (new —
`--arms_spec` grammar: `kind/ec/off/lam/src/ioff/tar`; per-pass b via
`--easycontrol_invert_b_offset`, `tar=caption` for base-model edit passes) ·
3 same-artist train pairs (izuna / yanami / hoshino), seed 42, judgeability
cap `max_delta 24`, rating safe+sensitive. All upper bounds: train pairs,
single seed.

### Verdict ladder

1. **src="" fixes the wash-out** (1213): ψ_src=caption during inversion caused
   the cross-pass conditioning mismatch; the shipped zero-training in-place
   recipe is `ec=both, src="", λ0.5, b0` — instruction-only, composition-held.
   Fails only on the trivially-copyable class (aligned-copy lock).
2. **EC-inversion-only REFUTED** (1826): cond-engaged inversion (net b0) +
   ~base edit pass (b −12) ≈ promptless base inversion on 2/3 pairs
   (fm_error 0.088 vs 0.102; mse_vs_src within noise) and **wash-out on the
   copyable pair** — EC-field Δz anchors don't replay under the base field
   where cond dominated. Mismatch absorption is one-directional
   (`--easycontrol_edit_only` = safe; the reverse = not).
3. **The 1826 control is a new recipe**: base inversion (ψ_src="") +
   full-caption ψ_tar + λ0.5 lands in-place edits with zero adapter, ~2–3×
   closer to source than feed-forward EasyEdit. The old `inv_noec` failure
   was ψ_tar=delta, not inversion.
4. **twin_edit e3** (killed at step 7210/13380, epoch-3 save): feed-forward
   instruction probe **passes at b0** on all 3 pairs (+2 = copy regime) —
   open-gate recipe carried over, pairs off its training manifest. Under
   inversion it copy-locks **harder** than subject_edit (yanami 0.0010 vs
   0.0069) — aligned twins + empty-instruction identity no-ops teach exact
   copy-through for precisely the src="" inversion configuration.

5. **Minimal-edit probes** (`…-1847-inplace-glasses-min-edit`,
   `…-1857-inplace-glasses-twinedit-e3`; `--delta_override glasses`,
   `tar=srccap` = source caption + ", glasses"): NO mask-free recipe lands a
   1-tag edit. subject_edit ff **loses identity entirely** (rich instructions
   were load-bearing for cond retrieval; 1 tag = off its length distribution);
   twin_edit-e3 ff is the mirror image — composition holds (mse 0.015–0.082
   vs subject's 0.09–0.12) but the edit is ignored, output = slightly
   degraded near-copy (the 25% empty-instruction no-ops plausibly teach
   "short instruction ≈ copy"). Base-inv srccap arms re-confirm phase-1a:
   the global anchor suppresses small edits at λ1 AND λ0.5 — minimal
   in-place edits remain the mask recipe's (outcome 1) territory. Caveats:
   e3 = half-trained; "glasses" may be off twin's pivot-tag distribution
   (its home minimal edits are expression/text-removal pivots); hoshino is a
   bad glasses case (source already wears eyewear on head).

Roll-up table + division of labor: `../outcomes.md` §In-place editing surface
map.

## Phase 0 (aligned-pair arm): pair census (2026-07-26)

**Run:** `results/20260726-1337-pair-census-full` (full corpus; smoke =
`…-1315-pair-census-smoke`) · **Bench:** `run_pair_census.py` · **Spec:**
`../easycontrol_request.md` §Phase 0 — the volume blocker nobody recorded for
sanitize.

One shared pass over `$CAPTION_CORPUS_DIR/retrieved` (15,780 images, 98
artists): same-size prune → 8,671 embeddable → Stage-A/B twin match at
sanitize's gates (sim 0.85 / match_frac 0.3 / cell 0.9) → **4,264 accepted
twins**, then every twin scored against all four slices at once with
**per-tag pivot** semantics (a slice tag counts when it's in exactly one
member — i.e. exactly when it lands in the delta caption); sanitize's
stricter set-level `tag_any` count recorded alongside.

### Verdict: PASS — 2,349 usable twins → 4,698 post-doubling vs the ~600 floor

| Slice | pairs | strict `tag_any` | artists | top pivots |
|---|---|---|---|---|
| expression | 1,828 | 303 | 66 | open mouth 745 · closed eyes 437 · tongue out 370 |
| clothing_state | 650 | 371 | 60 | nude 326 · bottomless 125 · clothes lift 98 |
| text_bubbles | 229 | 202 | 33 | speech bubble 132 · english text 71 · sound effects 51 |
| object | 169 | 93 | 45 | holding 38 · gloves 27 · halo 26 |

Findings against the request doc's priors:

- **The slice priority inverts.** The doc ranked text/bubbles first ("known
  to exist in volume") — at *pair* level it's third (229; still enough for
  the Q9 removal probe at 458 post-doubling). Expression variants dominate
  (1,828), clothing/state second (650) — the highest-value user slice is well
  fed.
- **Per-tag pivot is the volume unlock.** Set-level `tag_any` semantics would
  have seen 969 pairs total vs 2,349 (expression 6× undercounted: both twins
  usually carry *some* expression tag, so the set-level pivot fails exactly
  where the per-tag delta is cleanest).
- **Delta lengths sit inside the band, not far below it.** Median 13, p90 24,
  92.2% ≤ `max_delta` 24 — the "twins should sit far below the corpus median
  31" hope is only half-true; keep the cap and accept ~8% attrition.
- **Saturation skew replicates sanitize.** 96.5% of pair members < 0.4 mean
  HSV saturation (hist peak at 0.2) — `identity_saturation_min 0.4` for the
  identity fill stays justified, unchanged.
- Direction doubling is counted, not staged: doubling is a staging-time knob;
  all counts above are per-twin.

**Remaining Phase-0 items:** the by-eye pass over
`results/20260726-1337-pair-census-full/spotcheck/<slice>/index.html` (20
sheets per slice: alignment quality + delta sanity), which is a human gate;
caption generation itself is exercised and recorded per-pair in
`pairs_manifest.json`. If the eyeball pass holds, the aligned-pair arm is
unblocked for staging + training per the request doc's objective table.

## Phase 2.5: delta-caption edit descriptor — instruction probe (2026-07-26)

**Train:** `anima_easycontrol_subject_edit`, 7860 steps / 12 epochs over the
655-pair set (2h55m, `loss/epoch_average` 0.0781, clean). Arm-2 open-gate
recipe: `b_cond_init=-4.0`, `cond_res_scale=1.0`, `apply_ffn_lora=1`,
drop_p 0.05, lr 2e-5. Probed checkpoint = the epoch-12 final.

**Runs:** `results/20260726-1023-phase2p5-edit-probe` (first draw — all three
pairs landed nsfw/explicit, kept for the record, not judged) ·
`results/20260726-1033-phase2p5-edit-probe-sfw` (judged: `--rating
safe,sensitive --pair_scope same_artist --max_delta 24`; izuna (swimsuit) /
yanami anna / hoshino (swimsuit), offsets 0,2,3,4,6, seed 42). Bench:
`run_edit_probe.py` — replays the training task (cond = A, prompt = mined
delta caption), noec control + one arm per offset.

### Verdict: instruction probe PASSES at the TRAINED point — b0 is the operating band

The decisive difference from the subject descriptor: **the adapter is engaged
at b_offset 0**, where subject-v2's probe was inert. The delta objective
forces identity through the cond stream (the name tag cancels out of every
prompt), and it shows:

- **noec control** — proves both starvations at once. Renders miss the
  character (izuna: blue eyes/no halo; hoshino: no heterochromia/no halo), and
  the `-`-prefixed removals act as *attractors* (noec izuna holds the ramune
  the prompt says to remove; noec hoshino keeps orca + sunglasses + wading):
  the base TE reads `-ramune` as "ramune". Negation semantics exist only in
  the adapter.
- **ec_b0** — identity retrieval + instruction following simultaneously.
  Izuna: correct eyes/halo/scarf/visor/ear coloring, scene moved
  chair→wading-in-water per the instruction. Hoshino: exact heterochromia
  (orange/blue), halo, and the *clothing-state* instruction "jacket partially
  removed" lands (cond wears it on; render slips it off the shoulders), plus
  tinted eyewear + frilled bikini + skin fang. Yanami: artist style + additions
  land (arm up / armpits / white shirt / long hair).
- **ec_b2 → b3** — the copy regime arrives almost immediately: +2 drifts back
  toward the cond composition, +3 is a near-verbatim cond copy (izuna b3 =
  cond's chair/ramune/signboard scene). The band is *narrow and centered at
  the trained point* — no offset hunting needed, which is exactly what
  shipping wants.

**Systematic weakness: removals.** Additions and state-changes land; removals
of *objects present in the cond* mostly fail across all three rows (ramune
bottles, inflatable orca, hair beads survive their `-` tags). Plausible
levers, untested: removal-heavy pair mining, higher removal weighting, or
`cond_noise_max > 0`.

**Caveats:** train pairs (upper bound — held-out draw still owed, e.g.
`--seed` change or a held-out character split), single seed, render-judged
n=3. The `-`-prefix syntax is TE-blind (noec shows it), so all negation logic
is adapter-side — a fresh-vocabulary removal token is an open alternative if
removal performance needs a lever.

## Phase 2: cross-image subject descriptor (2026-07-25)

**Train:** `anima_easycontrol_subject`, 8928 steps / 8 epochs over the 1116-pair
set, 1h45m, `loss/epoch_average` 0.0797, no anomalies. Pair data verified: cond
latents symlink to a *different* image of the same character (73% cross-artist).

**Runs:** `results/20260725-0930-phase2a-boffset` (offsets 0..+4),
`-0949-…-hi` (+6/+8), `-0953-…-fine` (+5/+7) — one b_offset curve, same
seed/config, cross-run comparable · `-1000-phase2-retrieval` (geometry,
`--phase 2`) · `-1014-phase2-subject-probe-engaged` (DirectEdit-free retrieval).

### Verdict: both gates FAIL — but the run does NOT test the Phase-2 hypothesis

The gates fail, and the kill criterion's *conclusion* ("pairing wasn't the
constraint") is **not** supported: the cond stream was closed for the entire
train, so cross-image pairing was never actually exercised.

**The gate never opened during training.** `b_cond` is empirically non-learning
— the checkpoint saved exactly `-8.0` on all 28 blocks, as the inpaint
checkpoint saved exactly `-6.0` (bf16 resolution at that magnitude is 0.0625,
so |drift| < 0.03 over 8928 AdamW steps at lr 2e-5). It *is* wired to train
(in the optimizer via `get_trainable_params`, analytical gradient in
`easycontrol_attention.py`), it simply sits at a near-stationary point. With
`b_cond=-8` and `cond_res_scale=0.5` (S_c/S_t ≈ 0.25) the cond keys carried

    cond attention mass ≈ 0.25·e⁻⁸ ≈ 8.4e-5   (0.008%)

versus inpaint's `2.5e-3` at `b=-6, cond_res_scale=1` — **29.5× more**. The
weights corroborate it: cond-LoRA up-projections (zero-init) reached only
|w| ≈ 1.2e-4, while the adaln LoRA — which feeds the *target* stream and is not
gated by `b_cond` — moved 8× further. The cond path got almost no gradient
because the gate was shut.

**Gate (a) — sweet-spot width: FAIL (0 usable units, vs inpaint's ~1).** The
preservation band exists but is displaced ~7 units; MSE vs source, edit = +glasses:

| arm | dan_9596032 | 10473210 | 7538087 |
|---|---|---|---|
| base_t0 (pure anchor) | 0.13988 | 0.05723 | 0.02324 |
| vinj_t6 | 0.01594 | 0.00811 | 0.00549 |
| ec_s1 (offset 0 = trained point) | 0.14454 | 0.06598 | 0.03024 |
| ec_b4 | 0.15791 | 0.05846 | 0.02013 |
| ec_b5 | 0.15001 | 0.05391 | 0.01850 |
| ec_b6 | 0.09146 | 0.01666 | 0.00881 |
| ec_b7 | **0.01068** | **0.00498** | **0.00147** |
| ec_b8 | 0.00451 | 0.00236 | 0.00101 |

At +7 preservation beats `vinj_t6` on all three images — but the **edit is
suppressed across the whole band**. Render-judged (face crops), +glasses lands
only at +5, where preservation is nil (0.150). Preserve-and-land is empty:
usable width **0**. Inpaint had ~1 unit (b−1 on dan). The width did not
improve; it got worse.

**Gate (b) — geometry parity: FAIL to demonstrate retrieval.** `--phase 2` adds
the arm 1b could not express: cond left **whole** (identity available
position-free) with only the Δz anchor released full-frame. 1b's geometry row
gray-fills the *entire* cond, so a subject descriptor gets zero identity to
retrieve — it degenerates to unanchored generation for any adapter.

| arm | dan (squatting→standing) | 10473210 (+arms up) | 7538087 (→sitting) |
|---|---|---|---|
| vinj_t6 | pose unchanged | pose unchanged | pose unchanged |
| ec_anch (offset 0) | stands, keeps **nothing** | no change | no change |
| ec_anch_b6 | pose unchanged, source kept | no change | no change |
| ec_anch_b8 | ≈ source (clamped) | ≈ source | ≈ source |

Preservation and edit stay mutually exclusive — the same cliff as inpaint. EC
ties `vinj_t6` only by both failing. No arm lands the pose *and* keeps identity.

**The decisive check: no position-free retrieval was learned.**
`run_subject_probe.py` drops DirectEdit entirely and replays the adapter's own
training task as plain generation — cond = image A, prompt = caption of image B
(different image, same character), against a no-EC control at the same seed (the
control matters: the prompt already carries the character name as a tag). On
train-set pairs, i.e. an upper bound:

- offset 0 (trained point): `ec_b0` ≈ `noec` on all 3 pairs — the adapter
  contributes essentially nothing.
- offsets +6/+7/+8: the image **degrades** — washed out at +6, muddy at +7,
  collapsed to noise at +8. Identity does not transfer.

So what appears at +7/+8 in the edit bench is not learned retrieval, it is the
**architectural** copy path: extended self-attention over cond K/V reproduces
the cond whenever it is spatially aligned with the target (in the edit bench
cond *is* the source), and floods the attention with mismatched features when it
is not. That path exists without any training; the subject pairs contributed
almost nothing to it.

### What this does and does not settle

- **Does not settle** whether cross-image pairing can teach position-free
  retrieval — the mechanism that would carry it was closed at ~8e-5 mass for
  the whole run. This is a training-configuration failure, not a refutation.
- **Does settle** that `b_cond_init` is a load-bearing hyperparameter that does
  *not* self-correct, and that `-8` (with `cond_res_scale=0.5`) is far too
  closed to train through. Next arm: `b_cond_init ≈ -2` (mass 3.3e-2) and/or
  `cond_res_scale=1.0`, same cost as this run (~1h45m).
- **Untested combination:** the shipped 1a/1b mask recipe (`ec_mask_anch`) at
  the subject adapter's engaged point (+7). Given the probe result it would at
  best reproduce inpaint behavior, but it was not run.
- Q4 (hole-style artifact) is **not** answered — it needs the mask recipe at an
  engaged offset.

Wiring shipped alongside: `inference.py --easycontrol_b_offset` (the dial existed
only in `scripts/edit.py`; this checkpoint is unusable from the main inference
path without it), `EASYADAPTER=subject` for `test-easycontrol`, and
`run_bench.py --phase 2`.

## Phase 1a: masked-cond probe (2026-07-24)

**Runs:** `results/20260724-1827-phase1a` (3 img × 7 arms), `results/20260724-1844-phase1a-anchmask` (+2 arms, same seed/config — cross-run comparable) · Edit: caption + ", glasses", CFG 4, 28 steps, seed 42, b_offset 0 everywhere.

### Verdict: PASS, amended — the hole needs punching in BOTH preservation mechanisms

Feeding the inpaint prior its trained input (cond = source with a gray hole over the face box) gives exception-driven preservation exactly as proposed — but the cond hole alone landed the edit on only **1/3** images. The missing piece is the **Δz anchor**: it is global, so it keeps pulling the hole content back to the source after the EC prior has released it. Dropping the anchor inside the edit region (`--mask`, the never-implemented paper-Eq.-12 anchor-side half — now wired in `directedit.edit_forward`) fixes this: **`ec_mask_anch` (EC cond hole + anchor mask) lands the edit on 3/3 images — including 10473210, where every Phase-0 recipe at b−1 failed — at b_offset 0, no per-image tuning, still zero training.**

Outside-hole MSE vs source (recon_base level in parens):

| arm | dan_9596032 | 10473210 (hard) | 7538087 |
|---|---|---|---|
| recon_base | (0.00019) | (0.00015) | (0.00005) |
| base_t0 | 0.15114 | 0.05904 | 0.01915 |
| vinj_t6 | 0.01441 | 0.00781 | 0.00502 |
| ec_b-1 | 0.05394 | 0.00618 | 0.00850 |
| ec_b-2 | 0.15057 | 0.05029 | 0.02639 |
| ec_mask | 0.00239 | **0.00038** | 0.00301 |
| anch_only | 0.15340 | 0.05804 | 0.01728 |
| **ec_mask_anch** | **0.00238** | **0.00036** | **0.00310** |
| edit lands (ec_mask_anch) | ✓ | ✓ | ✓ |

- **Best-of-alternatives comparison** (the gate's real question): ec_mask_anch beats best-of-{vinj_t6, ec_b-1, ec_b-2} on outside-hole preservation by 2.6–17× on every image, and is the only arm landing the edit on all three.
- **The two controls split the blame cleanly.** `ec_mask` (cond hole only): preservation identical, edit lands 1/3 — the anchor suppresses the edit inside the hole. `anch_only` (anchor mask only, no EC): edit lands but outside-MSE ≈ base_t0 (composition destroyed) — at CFG 4 the anchor never was the preservation mechanism; the EC prior is.
- **The literal "≤ 2× recon" gate FAILS as written** (ratios 12.6 / 2.4 / 61.5) and is mis-calibrated: recon is near-pixel-exact, so the denominator is ~0.0001 and any visible-but-negligible drift (0.002–0.003 absolute, far below every alternative) explodes the ratio. Judged on renders + vs-alternatives, the probe achieves what the gate was written to test.
- **Known artifact:** on 7538087 the hole regenerates with a flat, saturated style (present in *every* EC arm on that image, masked or not — an inpaint-prior artifact on this simple gray-background style, not a mask effect). Edit still lands.

Wiring shipped: `scripts/edit.py --easycontrol_mask <png>` (gray-fills the cond image pre-VAE, matching the training distribution) and `--mask <png>` (drops Δz inside the region, latent-resolution). The recipe: pass the same mask to both.

## Phase 1b: edit-type generalization (2026-07-24)

**Run:** `results/20260724-1850-phase1b` (3 img × 4 edit types × {base_t0, vinj_t6, ec_b-1, ec_b-2, ec_mask_anch}; `EDITS_1B` in `run_bench.py` defines the per-image concrete edits + hole boxes). Same seed/CFG/steps as 1a.

### Verdict: PASS — ec_mask_anch ≥ vinj_t6 on all 3 in-place edit types

Render-judged (edit lands + composition held), per type across images:

| edit type | dan_9596032 | 10473210 (hard) | 7538087 | verdict |
|---|---|---|---|---|
| REMOVE (kanzashi / halo / blush) | **EC lands, vinj fails** (ornaments erased clean) | both fail (halo survives everything) | **EC lands, vinj fails** (blush gone; style-drift caveat) | **EC > vinj** |
| REPLACE hair color | **EC lands** (pale blue; vinj stays pink) | both fail (stays white) | both fail (EC goes *black*, not blonde) | **EC > vinj** |
| expression | both land; EC preserves better (0.0024 vs 0.0142 outside) | both ambiguous | both land; vinj cleaner in-box, EC better outside | **EC ≥ vinj** |
| geometry (control) | EC: pose DOES change (standing) but composition fully released; vinj: no edit | same pattern | same pattern | expected fail, recorded |

- Outside-hole preservation: ec_mask_anch is best-in-class on **every** in-place row (0.0004–0.0034), 2–6× ahead of vinj_t6, while being the only recipe that lands REMOVE/REPLACE edits at all.
- **Geometry nuance:** with a full-frame box the recipe degenerates to unanchored generation (gray cond + no anchor) — it *does* produce the pose, proving the suppression was preservation-owned, but keeps nothing. Position-locked prior confirmed; this is Phase 2's associative-retrieval target.
- **Hard-image ceiling:** 10473210's in-place edits (halo removal, white→black hair) fail for every method — beyond the current teacher regardless of preservation mechanism.
- **Failure colors:** 7538087 brown→blonde came out black in both EC arms (prior's dark-line style bias + "black bikini" in-caption attractor, plausibly); the flat-saturated hole-style artifact from 1a persists on this image.

Gate ("EC ≥ vinj_t6 for ≥ 2 of 3 in-place types"): **3/3. PASS.** Phase 2 (cross-image subject descriptor) is unblocked, with the geometry row as its falsifiable target.

---

# Phase 0: EasyControl cond stream as a learned preservation prior for DirectEdit

**Date:** 2026-07-24 · **Runs:** `results/20260724-1731-phase0-full` (3 img × 8 arms), `results/20260724-1749-phase0b-boffset` (2 img × 10 arms) · **Adapter:** `output/ckpt/methods/anima_inpaint.safetensors` (hole-free cond = "copy everything" reference) · **Edit:** caption + ", glasses" (in-place attribute edit), CFG 4, 28 steps, seed 42.

## Verdict: PASS, with the dial moved from `cond_scale` to `b_cond`

The zero-training composition works and, at the right gate offset, **beats V-injection on composition preservation while still landing the edit** (image-dependent sweet spot). Wiring: `scripts/edit.py --easycontrol_weight … --easycontrol_b_offset …`.

## Findings

1. **Exact composition with the Δz anchor.** With the cond KV cache active through BOTH inversion and edit passes, ψ_tar == ψ_src reconstructs the source pixel-exactly (recon gate recon_ec/recon_base = 0.85–0.97 ≤ 2.0 on all images). The EC prior does not perturb the anchor.
2. **`cond_scale` is near-binary on the inpaint prior.** 0.25/0.5 ≈ no-EC baseline (prior disengages — scaled-down cond-LoRA deltas move cond K/V off the distribution the learned gate retrieves); 1.0 = total clamp (pixel-level source copy, edit fully suppressed). No usable middle regime on this axis.
3. **`b_cond` offset is the continuous dial.** It's applied live as a logit bias in the LSE-extended attention (`easycontrol.py::_target_only_with_cached_cond_kv`), NOT baked into the KV cache, so an additive offset after `load_weights` shifts cond softmax mass ~e× per −1. Useful range on the inpaint prior: **−1 to −2**; −3/−4 ≈ disengaged.
4. **Head-to-head vs V-injection** (face crops `faces_*.png`):
   - dan_9596032 @ b−1: thin red glasses land AND the full source composition survives (pond reflections, obi, ornaments) — `vinj_t6` landed bolder glasses but invented a fireworks background. **EC wins.**
   - 10473210 @ b−1: edit fails to land (so does `vinj_t6`); @ b−2 glasses land with partial divergence (slightly better than pure anchor). **Tie-ish; sweet spot shifted.**
   - Pure anchor (`t_inj=0`) at CFG 4 loses the composition entirely on 2/3 images — consistent with why `t_inj` exists.
5. **Hyperparameter surface shrinks but doesn't vanish:** one interpretable scalar (b_offset, per-image sweet spot within −1..−2) vs V-injection's step count × block set. Also EC costs one KV prefill instead of a parallel src forward per injected step.

## Caveats / next levers

- The inpaint adapter is used off-label (hole-free cond; trained "trust cond fully", `b_cond_init=-6`, `drop_p=0`). The narrow/binary operating point is plausibly inpaint-specific. A purpose-trained prior should widen the sweet spot:
  - **cross-image subject descriptor** (cond = image A of a character, target = image B; mine pairs via `caption_index.json`) — trains position-free appearance retrieval, the thing no shipped aligned-pair adapter has;
  - **DirectEdit-synthesized edit pairs** (the feed-forward-editor distillation route).
- Compile is disabled under EC in `edit.py` (matches the inference engine's eager EC path); the compile-compat claim is untested.
- MSE-vs-source is a preservation proxy only; edit success was judged on renders (no tagger checkpoint available for readback at run time).
