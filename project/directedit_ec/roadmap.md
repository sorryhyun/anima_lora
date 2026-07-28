# directedit_ec — roadmap

Status: Phases 0, 1a, 1b all PASSED (2026-07-24, zero training). The shipped
recipe (cond hole + anchor mask, b_offset 0) beats V-injection on every
in-place edit type. Phase 2 arm 1 ran 2026-07-25 and both gates failed — but
on a run that never exercised the hypothesis. **Arm 2 trained the same day
with the gate open, and gate (c) — the decisive DirectEdit-free probe —
PASSED: position-free identity retrieval demonstrated at b_offset +2/+3.**
Next work item is gates (a) and (b) against the arm-2 checkpoint, engaged
offsets ≈ +2…+4. **Phase 2.5 (subject_edit) trained and its instruction
probe PASSED 2026-07-26 — engaged at b_offset 0, see below.**

## Phase 2 — cross-image subject descriptor (one standard EasyControl train)

**Status: arm 1 RUN, gates FAILED, hypothesis UNTESTED (2026-07-25).** The
train itself was clean (8928 steps, loss 0.0797, pair data verified), but
`b_cond_init=-8` with `cond_res_scale=0.5` left the cond stream at ~8.4e-5
attention mass for all 8928 steps, and **`b_cond` does not learn** (saved
exactly at init, as inpaint's did). So the cond path got almost no gradient and
cross-image pairing was never actually exercised. Full data:
`bench/report.md#phase-2`.

- Gate (a) sweet-spot width: **FAIL** — 0 usable units (preserve *and* land the
  edit) vs inpaint's ~1. The preservation band exists but is displaced ~7
  offset units (+6…+8) and the edit is suppressed throughout it.
- Gate (b) geometry parity: **FAIL to demonstrate** — EC ties `vinj_t6` only by
  both failing to land the pose.
- Decisive: `bench/run_subject_probe.py` (DirectEdit-free, cond = image A,
  prompt = caption of image B) shows **no identity transfer at any offset** —
  inert at the trained point, image degradation at +6/+7/+8. What engages at
  +7/+8 is the *architectural* aligned-cond copy path, not learned retrieval.

**Do NOT invoke the kill criterion on this run** — it does not discriminate the
hypothesis. Arm 2 first: retrain with the gate open (`b_cond_init ≈ -2`, mass
3.3e-2, and/or `cond_res_scale=1.0`), same cost (~1h45m), then re-run the same
three gate benches. If arm 2 also shows no retrieval, the kill criterion applies
with evidence behind it.

**Arm 2: RUN (2026-07-25), gate (c) PASSED — the adapter does learn
position-free retrieval.** Config: `b_cond_init=-4.0`, `cond_res_scale=1.0`
(mass e⁻⁴ ≈ 1.8e-2, ~220× arm 1's), `apply_ffn_lora=1`, drop_p 0.05,
`cond_noise_max=0.0`, lr 2e-5. The run was stopped at **epoch 4 of 8**; the
probed checkpoint is the epoch-4 save
(`output/ckpt/anima_easycontrol_subjectv2/anima_easycontrol_subjectv2-000004.safetensors`;
resume bundle exists). `run_subject_probe.py` (3 cross-artist train pairs,
seed 42), two sweeps:

- `+6/+7/+8` (effective b +2…+4 — arm 1's band): near-**verbatim copy of the
  cond image**, prompt scene discarded, patch-grid artifacts. The cond stream
  now carries full image content — the opposite failure of arm 1's inert
  stream — but this operating point is off-manifold (trained mass 1.8e-2,
  driven to σ≈0.88–0.98). Run dir
  `bench/results/20260725-2318-phase2-subject-probe-v2e4-engaged/`.
- `+2/+3/+4` (effective b −2…0): **retrieval band.** At +2/+3, artifact-free
  renders that keep the prompt's composition while cond-specific attributes
  migrate in position-free (hachimiya pair: cond's long low twintails + star
  hair ties appear, absent from both prompt and noec control). At +4 the
  cond's *global* appearance starts leaking (background tone/texture) — the
  onset of the copy regime. Run dir
  `bench/results/20260725-2326-phase2-subject-probe-v2e4-midgate/`.

So the dose-response is graded, not cliff-shaped, at the probe level:
inert (trained point) → position-free retrieval (+2/+3) → global leak (+4) →
verbatim copy (+6…+8). Caveats: train pairs (upper bound — held-out check
still owed if the gates pass), and epoch 4/8. Next: gates (a) and (b) with
engaged offsets ≈ +2…+4; retraining is **optional** — only if the gates want
a wider band would a fresh arm with `cond_noise_max > 0` (suppress the copy
end) or the epoch-8 resume be justified.

Surface as shipped: `configs/easycontrol/subject.toml` (descriptor with knobs +
generated blueprint tail), `easycontrol_adapters/tools/subject_pairs.py`
(the miner — near_twins contract, CPU-only), `EASYADAPTER=subject` registered
in `scripts/tasks/training.py`. First mining run: **1116 pairs over 283
characters, 813 (73%) cross-artist** (solo 1girl/1boy single-character only,
cap 16 targets/character, seed 42; manifest at
`post_image_dataset/easycontrol/subject/pairs.json`).

```bash
make easycontrol-staging    EASYADAPTER=subject   # mine pairs → staging/ + cond/ (CPU, done)
make easycontrol-preprocess EASYADAPTER=subject   # rebuild cond/ only (after corpus re-preprocess)
make easycontrol            EASYADAPTER=subject   # the Phase-2 train (--queue for daemon)
```

Design (as proposed, now implemented):

- **Pairs**: cond = image A of a character, target = image B of the same
  character, mined from `caption_index.json` character tags. Staging emits a
  pair manifest; cond + target latents/TE reuse the shared LoRA cache — no
  synthetic tree and **no encode pass** (both steps are pure symlinks;
  cheapest descriptor staging yet). The same-artist + shared-tags fallback
  was skipped — character tags alone cover ~1.1k targets. Each target's cond
  partner prefers a different artist dir (starves the style shortcut too).
- **Anti-shortcut knobs** (in `[training]`): `cond_res_scale = 0.5` (starves
  the positional shortcut, and is faster); `easycontrol_drop_p = 0.05`;
  `b_cond_init = -8`. **Arm 1 showed these two compose into a trap**: they
  multiply into ~8.4e-5 cond attention mass (0.25·e⁻⁸), 29.5× below inpaint's
  2.5e-3, and `b_cond` does not self-correct — it stays at init. The
  anti-shortcut intent was right, the dose closed the mechanism. `b_cond_init`
  is NOT a "softer learned operating point"; it is a fixed hyperparameter.
- **Text**: full captions of the *target* — prompt keeps owning layout/pose,
  cond owns identity/appearance only.
- Cost: inpaint-recipe scale (8 epochs over the 1.1k pair set ≈ inpaint's
  4 epochs over the 3k corpus in optimizer steps; 16 GiB-friendly).
- **Gate** — three benches, all re-runnable against any future arm:

```bash
W=output/ckpt/anima_easycontrol_subject.safetensors
# (a) b_offset band: sweep WIDE — the band's location depends on b_cond_init,
#     and the arm-1 band sat at +6..+8, outside every inpaint-era offset.
python project/directedit_ec/bench/run_bench.py --ec_weight $W --ec_scales 1.0 \
    --ec_b_offsets "-2,-1,1,2,4,6,8"
# (b) geometry / associative retrieval (whole cond, anchor released)
python project/directedit_ec/bench/run_bench.py --phase 2 --ec_weight $W \
    --ec_b_offsets "<engaged offsets from (a)>"
# (c) DECISIVE: retrieval with no DirectEdit in the loop
python project/directedit_ec/bench/run_subject_probe.py --n_pairs 3 \
    --b_offsets "<engaged offsets from (a)>"
```

  (a) sweet-spot width ≥ 2 b_offset units where the edit *both* preserves and
  lands (inpaint: ~1) — answers Q1; (b) ≥ parity with vinj_t6 on the geometry
  edit — answers Q2. **Run (c) first on any new arm**: it is cheap, has no
  composition confound, and a null there makes (a)/(b) uninterpretable.

## Phase 2.5 — delta-caption edit descriptor (subject_edit; instruction probe PASSED 2026-07-26)

Mined, teacher-free path to *edit-instruction* semantics: same cross-image
pairs as Phase 2, but the prompt is the **tag delta** between the captions —
additions in the target's order plus `-`-prefixed removals — instead of the
target's full caption. The prompt stops describing and starts instructing.
Three structural wins over the Phase-3 synthesis path: real targets (no
teacher ceiling — Q3 does not bound this), no tagger-readback dependency
(not blocked on Q6), and the character-name tag cancels out of every prompt
(shared between the pair), so identity *must* come from the cond — the
name-tag shortcut the subject probe had to control for is starved by
construction.

What it does NOT learn on its own: in-place editing. Mined pairs aren't
pixel-aligned, so the operator is "re-render with these changes"; composition
preservation stays owned by the shipped Phase-1b recipe (inversion + anchor)
at composition time.

Surface: `configs/easycontrol/subject_edit.toml` (descriptor),
`easycontrol_adapters/tools/subject_edit_pairs.py` (miner — subject_pairs
contract + delta captions + min-delta partner policy),
`EASYADAPTER=subject_edit` registered in `scripts/tasks/training.py`. Staging
writes REAL `.txt` files (delta captions are new text), so unlike subject the
preprocess step runs a TE encode pass into the descriptor-owned
`text_cache_dir` (inpaint-style); VAE latents + PE still ride the shared
cache. Tag *dropout* stays 0 in that encode — dropping part of an instruction
leaves the target unexplained (shuffle variants are fine).

First mining run (2026-07-25): corpus median caption 37 tags, median
nearest-partner symmetric delta 37 — true "small edits" barely exist between
distinct booru images, so the delta band is a purity-vs-size dial
(max_delta 24/36/44 → 165/566/845 pairs). Shipped at `max_delta=40`:
**662 pairs, 177 characters, 71% same-artist, median delta 31 tags**; known
label noise accepted (caption inconsistency shows up as spurious delta tags,
e.g. eye-color flips on the same character). Training config = the arm-2
open-gate recipe (`b_cond_init=-4`, `cond_res_scale=1.0`, ffn LoRA), 12
epochs ≈ subject's optimizer-step count.

Gate: its own edit-instruction probe (cond = A, prompt = mined delta, judged
on whether the *instructed* changes land while identity holds — the subject
probe only exercises retrieval). Bench artifact: `bench/run_edit_probe.py`
(replays the training task; `--rating` filters pair draws by the Anima rating
band, `--max_delta` caps instruction length for judgeability).

**First arm RUN + probe PASSED (2026-07-26).** Train: 7860 steps / 12 epochs,
loss 0.0781, clean; probed ckpt = epoch-12 final. Judged run:
`bench/results/20260726-1033-phase2p5-edit-probe-sfw` (3 same-artist
safe/sensitive pairs, offsets 0,2,3,4,6). Headline: **the adapter is engaged
at the TRAINED point (b_offset 0)** — identity retrieval (halo, heterochromia,
accessories) and instruction following (scene moves, clothing-state changes
like "jacket partially removed") land simultaneously at b0, with the noec
control proving both that tags alone miss the character and that the base TE
reads `-tag` removals as attractors. Copy regime starts at +2, near-verbatim
cond copy by +3 — the band is narrow but centered where inference runs by
default. Systematic weakness: object *removals* mostly fail (cond objects
survive their `-` tags). Full data: `bench/report.md#phase-25`. Owed next:
held-out draw (train pairs = upper bound); removal-lever arm only if removal
performance matters for the use case.

## Phase 3 — feed-forward editor (endgame, gated on Phase 2)

Distill DirectEdit itself: synthesize `(source, edited, edit-caption)` pairs
at scale with the Phase-1b recipe + tagger readback as the label filter,
train an EasyControl editor descriptor (cond = source, target = edited).
Inference becomes one cached-cond generation — no inversion, no anchor, no
patching. InstructPix2Pix's recipe with our tag vocabulary and a
training-free teacher. **Separate proposal when reached**; only if Phase 2
shows the teacher reliable enough to label its own data (the hard-image
ceiling, questions.md Q3, bounds this).

## Parallel / opportunistic

- Wire tag-readback edit-success into `run_bench.py` once a trained tagger
  checkpoint exists (Q6) — retro-scores existing result dirs too.
- Swap the manual hole box for the cfgdelta subject localizer (Q5) — small
  edit.py flag, benchable on the existing 1a set.
- Paper: **GO** (2026-07-26, Q7) — the Phase-2.5 result completes the
  two-contribution story; matched-NFE baseline table is the first work item,
  not the last. Ship track runs in parallel:
  `docs/proposal/easyedit_comfy_node.md` (held-out validation = Phase 0 gate,
  shared with the paper's Q10).

## Kill criteria

- Phase 2 sweet-spot width ≤ inpaint's AND geometry parity fails → pairing
  wasn't the constraint; close this proposal's training arc, keep the
  zero-training recipe as the shipped artifact, spin the gate-schedule idea
  (per-block/per-σ) as a separate smaller proposal only if demand exists.
- Phase 3 never starts unless Phase 2's adapter demonstrably widens the
  operating band — a feed-forward editor distilled from a cliff-shaped
  teacher inherits the cliff.

## Relation to BYG

BYG (demoted 2026-07-02, validation-gated) keeps its niche: free-form
*instruction* edits beyond tag space. This line's Phases 0–2 stay tag-edit
scoped. If BYG's Phase-0 gate ever passes, a BYG arm belongs in this bench.

## Pointers

Proposal: `project/directedit_ec/initial_proposal.md` · Data:
`project/directedit_ec/bench/report.md` · Memory: `project_directedit_ec_phase0` ·
Recipe + component map: `methods.md` · Shippable artifacts: `outcomes.md` ·
EasyEdit ship proposal: `docs/proposal/easyedit_comfy_node.md`.
