# directedit_ec — open questions

## Q1 — Is the cliff-shaped operating point data-owned or architectural?

The inpaint prior's usable band is narrow because it was trained "cond is
authoritative" (`b_cond_init=-6`, `drop_p=0`, aligned pairs). Phase 2's
cross-image subject descriptor is the falsifier: if sweet-spot width does
NOT improve over inpaint's (~1 b_offset unit), the pairing wasn't the binding
constraint — the cliff is architectural (gate granularity), and the next
lever is per-block/per-σ gate schedules (a different, smaller proposal).

**Phase-2 arm 1 (2026-07-25): still OPEN, and the arm-1 answer does not
count.** Measured width was 0 usable units (worse than inpaint's ~1), but the
adapter trained with its cond gate shut (~8.4e-5 attention mass), so the
"data-owned" branch was never exercised. Re-ask after an arm with the gate
open. One genuinely new datum: **the cliff shape survived** — across the whole
+6…+8 preservation band the edit stayed suppressed, and the only offset that
landed the edit (+5) preserved nothing. Two adapters, two very different
training sets, same mutually-exclusive shape — weak evidence for the
architectural branch, not yet decisive.

**Arm 2 (2026-07-25): first evidence for the data-owned branch.** At the
subject-probe level (no DirectEdit in the loop) the open-gate adapter's
response is *graded*, not cliff-shaped: inert at the trained point →
position-free retrieval at +2/+3 → global-appearance leak at +4 → verbatim
copy at +6…+8, a ≥2-unit-wide qualitatively distinct middle regime. Whether
that width survives composition with inversion + anchor is exactly gate (a),
still to run. If it does, the cliff was data-owned (arm 1's shut-gate training)
and the per-block/per-σ gate-schedule proposal stays shelved.

## Q2 — Can a trained prior do associative (position-free) retrieval?

The geometry row proved the inpaint prior is position-locked: with a
full-frame hole it produces the pose but keeps nothing. Phase 2 trains
retrieval that positional copying cannot satisfy (cond = image A, target =
image B of the same character). Falsifiable target: parity with vinj_t6 on
the 1b geometry edit.

**Phase-2 arm 1 (2026-07-25): NO for this checkpoint, hypothesis still open.**
`bench/run_subject_probe.py` tests it without DirectEdit in the loop (cond =
image A, prompt = caption of image B, vs a no-EC control at the same seed —
the control is essential, since the prompt already carries the character name
as a tag). Result on train-set pairs: inert at the trained point (`ec_b0` ≈
`noec`), and progressive image *degradation* at +6/+7/+8 rather than identity
transfer. Mechanistic reading: the aligned-cond copy path that engages at
+7/+8 in the edit bench is **architectural** — extended self-attention over
cond K/V reproduces a spatially-aligned cond and floods on a mismatched one —
and needs no training, so arm 1's cond stream contributed essentially nothing.
The retrieval question needs an arm trained with the gate open.

**Arm 2 (2026-07-25): YES — at the probe level.** The open-gate retrain
(`b_cond_init=-4`, `cond_res_scale=1.0`, epoch-4 checkpoint of 8) shows
position-free retrieval at b_offset +2/+3: the prompt's composition is kept
while cond-specific attributes (hairstyle, hair ornaments absent from both
prompt and noec control) migrate into it, artifact-free, consistent across
pairs. Above that band the mechanism degrades into the architectural copy
path arm 1 exposed (+4 global leak, +6…+8 verbatim patch-copy) — both
behaviors coexist in one adapter, selected by operating point. Caveats:
train-set pairs (upper bound), so a held-out check is owed; and the
vinj_t6-parity target on the 1b geometry edit is still gate (b)'s to answer.
Run dirs: `bench/results/20260725-2318-…-engaged/` and
`…2326-…-midgate/`.

## Q8 — Does `b_cond` ever learn? (new, 2026-07-25)

Both trained checkpoints saved `b_cond` at exactly their init (subject −8.0,
inpaint −6.0, all 28 blocks; bf16 resolution 0.0625 there, so |drift| < 0.03
over 8928 AdamW steps). It is in the optimizer and has an analytical gradient
(`easycontrol_attention.py`), so this is a near-stationary point, not missing
wiring — plausibly because the gradient scales with the cond mass the gate
itself controls, making a shut gate self-sustaining. Consequences if true:
`b_cond_init` is a *fixed* hyperparameter that must be chosen correctly up
front, and a gate warmup/schedule (or a mass-normalized parameterization)
would be the general fix. Worth a direct instrumented check — log
`b_cond.grad` over the first ~100 steps of the next arm.

**Arm 2 datum (2026-07-25): the answer is no even with the gate open.** The
arm-2 epoch-4 checkpoint saved all 28 `b_cond` at exactly the −4.0 init,
despite ~220× arm 1's cond attention mass (1.8e-2) — so the "shut gate
starves its own gradient" story is insufficient on its own; drift is below
bf16 resolution (~0.03 at −4) even when the cond stream demonstrably carries
content. Treat `b_cond_init` as fully fixed. The instrumented `b_cond.grad`
check is still the open item if we ever want the mechanism.

## Q3 — What owns the hard-image ceiling?

10473210's in-place edits (halo removal, white→black hair) fail for every
method — the limit is the teacher (base model + inversion), not preservation.
Is it caption-attractor strength, inversion quality at CFG 4, or model prior?
Matters for Phase 3: a feed-forward editor can only be as good as the data
the teacher can label.

## Q4 — The hole-style artifact

Flat, saturated regeneration inside the hole on simple flat-background images
(7538087; all EC arms, mask-independent). Inpaint-prior-owned. Does it
persist under the Phase-2 subject prior, or is it an artifact of the inpaint
training data specifically? Watch it in the Phase-2 gate re-run.

## Q5 — Automatic mask source

The recipe needs a hole box. Manual today; the cfgdelta subject localizer
(foveation line's reusable artifact, `project_foveated_denoise_p0`) is the
planned automatic source. Open: does a loose auto-box degrade the recipe
(the anchor mask drops Δz everywhere inside it), i.e. how tight must masks be?

## Q6 — Edit-success metric beyond renders

Render judging caps phase gates at small n. Tag-readback
(`docs/proposal/tag_readback_reward.md`, Phase 0a passed) would give a
scalable edit-lands metric — blocked on a trained tagger checkpoint at bench
time. Wire it into `run_bench.py` when available. **Paper-critical as of
2026-07-26**: the delta-caption editor's natural metric IS tag readback
(instructed additions present, removals absent, off-instruction tags
unchanged) — per-tag precision/recall over the instruction, plus an identity
metric on the untouched tags. Every quantitative table in the paper (Q7)
wants this; prioritize the tagger checkpoint accordingly.

## Q7 — Paper bar (GO as of 2026-07-26 — Phase 2.5 completes the story)

The claim is now two connected contributions, stronger than the original
single-dial story:

1. *A pretrained image-conditioning adapter's attention gate is a continuous
   preservation dial for flow-inversion editing, composing exactly with
   residual-anchored inversion* (Phases 0–1b, zero-training).
2. *Retraining the same adapter on mined tag-delta instructions turns it into
   a feed-forward instruction editor whose engaged band sits at the trained
   operating point* (Phase 2.5) — inversion removed entirely, 1× NFE.

The subject-vs-subject_edit contrast is the key ablation: same architecture,
same open-gate recipe, same pair source — full-caption objective leaves the
trained point inert (engagement only at hunted offsets), delta-caption
objective (name-tag cancellation forcing identity through cond) opens it at
b0. That isolates the *objective* as the mechanism. Supporting mechanism
findings worth a section: `b_cond` never trains (Q8), the architectural
aligned-cond copy path, removals-as-attractors in the base TE (noec
controls).

Missing, all evaluation (matched-NFE table FIRST, per the FSG lesson
`project_fsg_golden_path_phase0`):

- Inversion side: RF-Inversion / RF-Solver / FireFlow / FlowEdit at matched
  NFE; PIE-Bench.
- Instruction side: InstructPix2Pix-family baselines (IP2P, MagicBrush-tuned)
  — note domain mismatch (photo-centric, natural-language instructions) and
  handle honestly: report on their benches AND on an anima-domain tag bench.
- Quantitative metrics via Q6 tag-readback + an identity metric; held-out
  splits (Q10). Single-seed render-judged n=3 probes do not go in a paper.

## Q9 — Why do object removals fail, and what is the lever? (new, 2026-07-26)

Phase 2.5's one systematic weakness: `-tag` removals of cond-present objects
mostly fail (ramune/orca/beads survive), while additions and state-changes
land. The noec control shows the base TE reads `-x` as "x" — negation is
entirely adapter-learned and is fighting both the text attractor and the cond
stream's copy of the object. Candidate levers, none tested: removal-heavy
pair mining (weight pairs by n_removals), a dedicated removal token in fresh
vocabulary instead of the `-` prefix (TE-blind syntax may be the bottleneck),
`cond_noise_max > 0` (weaken the cond copy), or instruction-side loss
weighting on removal regions. Also open: do removals fail uniformly, or only
for objects the cond renders saliently?

## Q10 — Held-out generalization (new, 2026-07-26; gates the ship AND the paper)

Every 2.5 verdict is on train pairs — an upper bound. Three widening rings,
in order: (a) corpus images outside the pair manifest, (b) hand-written
instructions (the user distribution — includes instruction styles mining
never produces, e.g. removal-only), (c) out-of-corpus anime images. Ring (b)
is the ship gate (`docs/proposal/easyedit_comfy_node.md` Phase 0); rings
(a)+(c) with Q6 metrics are the paper's held-out split. A collapse at (a)
means pair memorization — retrain with more pairs before anything ships.
