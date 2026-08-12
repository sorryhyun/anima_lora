# E25c — the res-cond projection as an inference-time resolution knob (frozen)

| | |
|---|---|
| **Status** | **REGISTERED (FROZEN) 2026-08-12** — blocked on E25b Stage-2 checkpoints (grid running, `e25b_stage2_jobs.json`) + the 25c.0 flag (CPU). Zero additional training; renders only. |
| **Question** | E25b trains a projection that places every training step on an explicit resolution axis (s = log2(edge/1024): 0 native, −0.193 @896, −0.415 @768). The 1024-tier training images are themselves downscales of higher-res originals, so the learned axis correlates grid coarseness with detail loss. E25c asks the inference-side pair of questions: **(25c-1)** is the learned axis a coherent *direction* (monotone response of render statistics to s) or a 3-point lookup? **(25c-2)** does any s ≠ 0 at inference *shrink the resized-native gap* — i.e. does conditioning the render on a point other than the trained default s = 0 move outputs closer to the native twin (or measurably finer), the SDXL size-conditioning inference trick transposed to this axis? |
| **Category** | **Mod-guidance-class**, not Spectrum-class: requires a rescond-trained checkpoint (the projection), loader-gated like `--pooled_text_proj`. Not training-free, not composable onto arbitrary checkpoints. |
| **Licensed by** | E25b 25b-1 IMPROVES (the projection demonstrably organizes the demoted-step geometry — there is something on the axis to steer); Stage 2.0 wiring (`_attach_res_cond`, the (proj, s) attach seam this knob rides). Precedent: SDXL micro-conditioning steered at inference away from low-res modes. |
| **Explicitly NOT this** | **No training change of any kind** — the trainer is bit-untouched, and no 25c result may be cited to amend the trainer (that would be a new E25-family amendment with its own accounting). s stays a **user-chosen known scalar** (a generation setting, like cfg) — no estimated/measured quantity is ever computed *into* s (the E25b known-input kill applies unchanged at this seam). The transplant cell (projection onto a foreign checkpoint) is a single descriptive curiosity, claim-free. Nothing here touches paper-1; paper-2/3 material alongside E25b. |
| **Depends on** | E25b Stage-2 grid checkpoints (rescond / rescond768 + their native twins); `e4_render_eval.py` frozen prompt/gen_seed grid; `bench/rapsd.py` (spectral read); Stage 2.0 `_attach_res_cond`. |

## Honesty paragraph (recorded at freeze)

The projection was trained on **three points** of the axis (rescond:
{0, −0.193, −0.415}; rescond768: two). SDXL's size-conditioning became
a steerable direction because training spanned a near-continuous size
distribution; ours may have learned a lookup with arbitrary behavior
between and beyond its points. **s > 0 is pure zero-shot
extrapolation** — the sinusoid embedding of an unseen s is
well-defined and the projection is linear, so the delta is defined,
but nothing constrains the model's response to it. A LOOKUP verdict on
25c-1 is a fully respectable outcome and closes the knob idea at this
operating point; it must be recorded, not retried with a finer s grid
(one sweep, no post-hoc grid refinement without amendment).

## 25c.0 — instrument (CPU only)

- `--res_cond_s <float>` on inference (default **0.0** = current
  behavior): plumbed into `_attach_res_cond` as the attached s.
  Default bit-identical to today's attach (pinned); flag with a
  non-carrier checkpoint = **hard fail** (the probe `--res_cond`
  convention — a knob that silently does nothing is the failure mode
  this line keeps closing).
- Tests (`TestResCond` additions): default-0 tuple identity;
  s-propagation to the attached tuple; non-carrier hard fail.
- Reads reuse `e4_render_eval.py` (renders) + `rapsd.py` (spectra) +
  a small `e25c_read.py`.

## Frozen design — one render sweep (no training)

**S = {−0.415, −0.193, 0, +0.193, +0.415}** — the two trained demote
points, the trained native point, and their positive mirror images
(the extrapolation pair chosen by symmetry, not tuned).

Primary material: the 6 Stage-2 **rescond** checkpoints (2 corpora × 3
seeds) rendered at every s ∈ S on the frozen E4 prompt/gen_seed grid.
Reference renders (native twins, s = 0 ctrl) come from the Stage-2
yardstick pass — re-rendered in the same boot if the passes don't
share one (renders compare within one boot only). Descriptive
material, same sweep, no verdict weight: the 6 **rescond768**
checkpoints; one **transplant cell** (the hews-s1001 rescond
projection attached onto the hews-s1001 **combo** ctrl checkpoint,
s ∈ S, one corpus one seed — curiosity only, the projection is
co-trained with its ΔW and is expected to be an arbitrary perturbation
elsewhere).

### 25c-1 — axis coherence (primary)

Per (checkpoint, prompt): Spearman ρ_s of high-band RAPSD energy
(band frozen in `e25c_read.py` before the sweep runs; the `rapsd.py`
convention) against s over the 5 sweep points.

| outcome (median ρ_s over the 6 rescond ckpts × prompts) | verdict |
|---|---|
| median ρ_s ≥ **+0.8** | **AXIS(+)** — finer-conditioning ⇒ more high-band energy; the learned axis is a direction with the expected sign. |
| median ρ_s ≤ **−0.8** | **AXIS(−)** — a direction with inverted sign (recorded as such; still a direction). |
| otherwise | **LOOKUP** — no coherent inference-time axis at this operating point; the knob idea closes (E25b's training-time verdict is untouched). |

0.8 is a judgment constant, recorded as such at freeze (5-point
Spearman: 0.8 admits one inversion under ties, a clean monotone run
scores 1.0).

### 25c-2 — gap shrink (reads only if 25c-1 ≠ LOOKUP)

Paired per-seed read on the rescond checkpoints: Δcos(s) =
cos(render(s) ~ native twin) − cos(render(0) ~ native twin), within
seed and prompt grid, per corpus.

| outcome | verdict |
|---|---|
| some s > 0 has Δcos > 0 on **≥ 5/6** twin pairs (both corpora represented) | **GAP-SHRINK** — conditioning finer than trained moves renders toward the native twin; the resized-native gap is partially recoverable at inference. |
| some s < 0 has Δcos > 0 on ≥ 5/6 pairs instead | **ANTI** — matching the resized distribution (not fighting it) is what helps; recorded, mechanistically interesting, not a ship story. |
| otherwise | **NULL** — s = 0 (the trained operating point) is the optimum; the knob is a no-op for quality. |

s* (the argmax) and the full Δcos(s) profile per corpus are recorded
regardless. Eyeball sheets (E4 fig-candidate format) accompany every
verdict — a GAP-SHRINK with visible artifacts at s* is recorded as
caveated, the sheets are the artifact check.

### Descriptives (no verdict weight)

Full RAPSD curves per s; rescond768 sweep (2-point axis — does the
shallower training sampling degrade coherence?); the transplant cell;
per-corpus divergence of the profiles (hews vs channel).

## Kill switches / honesty

- One sweep, frozen S; refining or extending the s grid after seeing
  results is an amendment.
- No 25c result feeds back into training or into any trainer flag
  default.
- Renders of a comparison pair share a boot; a reboot mid-sweep
  resubmits the sweep (render-level reads only — no vector claims of
  any kind, so no arm-store/family machinery is needed).
- LOOKUP closes the line at this operating point; the recorded
  reopening condition is a *training-side* change to axis sampling
  (e.g. more routes ⇒ more points on the axis), which is an E25-family
  training amendment first, an E25c rerun second.
- `--res_cond_s` ships (if at all) as an experimental inference flag
  documented alongside mod-guidance, never a default; any default
  change is its own decision.

## Cost

| item | cost |
|---|---|
| 25c.0 — flag + tests + read script | CPU only |
| render sweep — (6 rescond + 6 rescond768 + 1 transplant) ckpts × 5 s × 24-prompt grid, one boot | ≈ 1.5–2.5 GPU-h |
| read (`e25c_read.py`: RAPSD + paired cos + sheets) | CPU minutes |
