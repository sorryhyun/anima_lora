# EasyEdit — ship the delta-caption instruction editor to ComfyUI

Status: **PHASES 1 + 2 IMPLEMENTED** (2026-07-26), shipped as an **alpha with
disclosed limits**. Phase 0 (held-out validation) was **skipped, not passed** —
see the Phase 0 section for what that costs and what would close it.

- Checkpoint published: `anima_subject_edit_alpha.safetensors` in
  [`sorryhyun/anima-easycontrol-adapters`](https://huggingface.co/sorryhyun/anima-easycontrol-adapters)
  (sha256 `948722bd…` = the epoch-12 / 7860-step final, i.e. the checkpoint the
  Phase-2.5 probe actually judged).
- Node repo `~/ComfyUI-EasyControl-KSamplerCompat` at **v0.3.0**: EasyEdit
  README section, `workflows/easyedit.json`, and the two Phase-2 builder nodes
  (`AnimaEasyEditInstruction`, `AnimaEasyEditDelta`) with
  `tests/test_delta_caption.py` pinning miner equivalence.

The `_alpha` suffix and the README/model-card limits sections are what carries
the missing Phase-0 evidence: the published claim is scoped to "probe on train
pairs, three characters, single seed — an upper bound, not a held-out score."

## TL;DR

Phase 2.5 of directedit_ec produced an editor with a genuinely simple UX:
give it a reference image and a tag instruction (`additions, -removals`), get
one plain generation with the reference's identity and the instructed changes
applied. No inversion, no anchor, no mask, no offset tuning — the engaged
band sits at the trained operating point (b_offset 0), so the default just
works. That is node-shaped. Ship it as **EasyEdit**:

1. **Phase 0 (kill-shot, inference only):** held-out validation — does it
   survive images outside the mined pair manifest and hand-written
   instructions? If identity or instruction-following collapses off-manifest,
   do not ship; return to training levers.
2. **Phase 1:** publish the checkpoint + a workflow riding the **existing**
   EasyControl KSampler node (zero new node code), plus README.
3. **Phase 2 (UX, gated on demand):** a small DeltaCaption builder node and/or
   tagger integration so users don't hand-compose delta captions.

Naming: EasyEdit (see `project/directedit_ec/outcomes.md` §2 — DirectControl
rejected; paper uses a distinct method name).

## What ships (mechanically)

- **Checkpoint**: local `output/ckpt/anima_easycontrol_subject_edit.safetensors`
  → published as **`anima_subject_edit_alpha.safetensors`** in
  `sorryhyun/anima-easycontrol-adapters` — a standard EasyControl adapter
  (per-block cond LoRA + `b_cond` gate, ffn LoRA on).
- **Inference path**: exactly the EasyControl path that already ships —
  `~/ComfyUI-EasyControl-KSamplerCompat` (latent match CONFIRMED,
  `project_easycontrol_ksampler_node`). EasyEdit is *checkpoint + prompting
  convention*, not a new architecture: cond image in, delta caption as the
  prompt, b_offset 0. Node work for Phase 1 was **none** beyond the bundled
  workflow JSON + README section, as predicted — the b_offset check passed
  (no offset is applied anywhere in the node).
- **Prompting convention** (must be documented verbatim): additions as bare
  tags first, then `-`-prefixed removals; Anima tag vocabulary; the rating
  band participates (e.g. `safe, …, -nsfw` converts rating) — that is a
  feature but must be stated. Instruction ≈ what changes, never a full
  caption — a full target caption is the *subject* task and will behave
  differently.

## Positioning vs the DirectEdit node

Complementary, not competing — document both in one story:

| | DirectEdit node (shipped) | EasyEdit (this) |
|---|---|---|
| Semantics | edit THIS image in place | re-render with these changes |
| Preserves | composition + unchanged pixels | identity/appearance |
| Cost | 2 passes (invert + edit) | 1 generation |
| Inputs | image + src/tgt captions | image + delta instruction |
| Weak spot | hard-image teacher ceiling | object removals |

Follow-up (separate, after this ships): the Phase-1b EC-hole recipe as an
"in-place mode" upgrade to the DirectEdit node (needs EC support in that
node's `_vendor` tree — regenerate via `make vendor-sync`, never hand-cp).

## Phase 0 — held-out validation (the gate) — **SKIPPED, still owed**

Shipping happened without this. The mitigation is disclosure, not evidence: the
build is labelled `_alpha`, and both the model card and the node README state
that the only validation is a train-pair probe (n=3 characters, single seed,
render-judged) which is an upper bound. That is honest but it is not the gate —
if the adapter turns out to be memorizing pairs, users find out before we do.
Running it remains the highest-value follow-up, and its kill criterion still
applies retroactively: a collapse off-manifold means **unpublish**, not
re-document.

All inference, ~1 GPU-hour, using `run_edit_probe.py` variants:

- **Held-out draw**: different `--seed` + pairs excluded from training would
  require a retrain, so approximate: probe characters whose pairs were capped
  out of the manifest, and cond images from the corpus that are NOT any
  pair's cond. Checks identity retrieval off the exact trained pairs.
- **Hand-written instructions**: 10–15 instructions written by hand (not
  mined deltas) over corpus cond images — the actual user distribution.
  Include addition-only, removal-only, and mixed instructions; score
  removals separately (expected weaker — the ship decision is whether
  additions/state-changes hold, with removals disclosed as a limit).
- **Non-corpus cond images** (stretch): a few out-of-corpus anime images —
  the adapter never saw them and the base model's coverage varies; failure
  here narrows the README claim ("corpus-style images"), it does not kill.

Kill criterion: if held-out identity transfer or addition-landing visibly
collapses vs the train-pair probe (render-judged, same contact-sheet format),
the adapter is memorizing pairs — do not publish; levers are more pairs
(lower `min_group`), cross-artist scope, or `cond_noise_max > 0`.

## Phase 1 — publish — **DONE**

- [x] HF checkpoint upload + model card. Limits section covers removals, the
      narrow band ("do not raise b_offset; +2 begins verbatim copying"), the
      rating-band behavior, and composition-not-preserved.
- [x] Workflow JSON (`workflows/easyedit.json`, UI graph format, core nodes +
      this repo only) + README section in the EasyControl KSampler repo.
- [x] Verified the KSampler node applies **no** b_cond offset — it uses the
      checkpoint's trained gate verbatim (`easycontrol_patch.py` reads
      `b_cond.{i}` straight from the state dict), which is exactly `b_offset 0`.
      Deliberately left unexposed: an offset knob is a footgun here.
- [ ] Announce surface (Arca Live / Civitai — mind
      `project_user_community_audience`).

Sampling defaults land for free: ComfyUI's `Anima` config already carries
`shift = 3.0` (`comfy/supported_models.py`), the probe's `flow_shift`, so the
shipped workflow needs no sampling-shift node — stock KSampler at euler /
simple / 28 steps / cfg 4.0 reproduces the validated operating point.

## Phase 2 — UX — **DONE** (shipped with Phase 1 rather than demand-gated)

- [x] **DeltaCaption builder nodes**, both in the EC KSampler repo
      (`delta_caption.py`, CPU-only, zero deps):
      `AnimaEasyEditInstruction` (add / remove fields → instruction) and
      `AnimaEasyEditDelta` (two captions → computed delta, plus separate
      `additions`/`removals` outputs for debugging an edit that misfired).
- [x] Equivalence with `subject_edit_pairs.delta_caption` is **pinned by test**
      (`tests/test_delta_caption.py::test_matches_miner` reimplements the miner
      inline and asserts byte-equality) — the node cannot silently drift from
      what the adapter was trained on.
- [x] **Tagger integration** needs no code: `AnimaEasyEditDelta.source_caption`
      is a plain STRING, so the in-tree anima-tagger node's caption output wires
      straight in. Closes the loop to "edit by editing the caption" without
      inversion, and without a cross-repo dependency.
- Tag hygiene the nodes handle: underscore→space normalization (skipping
  3-char Danbooru emoticons like `^_^` / `x_x`), comma **or** newline
  separators, de-duplication, no double `-` prefix, and add/remove conflicts
  resolved to the addition.

## Open items folded into the paper, not this ship

Removal-mechanism lever (why `-tag` negation is weak; removal-heavy mining /
removal token), quantitative edit-success metric (tag-readback, Q6), and the
subject-vs-subject_edit operating-point ablation. None of them block a ship
whose README states the limits honestly.
