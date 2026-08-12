# E25d — truthful low-res inference labeling: the res-cond intended-use transfer read (frozen)

| | |
|---|---|
| **Status** | **DONE 2026-08-12 — 25d-1 ANTI** (5/6 checkpoints median Δcos < 0; signal carried by hews, channel a prompt-level coin flip). Registered frozen earlier the same day (bcc85e69); run + read below. |
| **Question** | E25b trained (896-grid step, s = −0.193) as a *truthful label*: the conditioning absorbs the native↔demoted compute-graph gap per step (25b-1). E25d asks whether that absorption **transfers to inference**: when the rescond model renders at an 896-grid size, does labeling the render truthfully (s = −0.193) move its output **toward the same model's own 1024-grid behavior**, versus rendering the same 896 grid with the (false) native label s = 0? This is the SDXL micro-conditioning *intended use* (truthful size labels at inference), transposed to the learned s-axis — distinct from E25c, which sweeps s at **native** resolution (labels deliberately false there). |
| **Category** | Mod-guidance-class (needs a rescond checkpoint; loader-gated). Functional read: "does truthful labeling help at inference", NOT a claim that 896-grid inference visits the training distribution (it doesn't — see honesty). |
| **Licensed by** | E25b 25b-1 IMPROVES (the conditioning demonstrably absorbs the per-step grid substitution — there is an absorption to transfer); Stage 2.0 `_attach_res_cond` + 25c.0 `--res_cond_s` (s-propagation pinned in TestResCond). Context only, no license weight: the 2026-08-12 base-model eyeball (`runs/20260812-e25c-eyeball-base`, claim-free) showed the projection is live at inference. |
| **Explicitly NOT this** | No training change; no feedback into any trainer flag or default (known-input kill unchanged). Not a re-read of E25b (its FAIL/NULL verdicts stand on their own material) and **not a pre-read of E25c** — E25c's frozen native-resolution sweep is untouched: no E25d render is an input to any 25c-1/25c-2 statistic, and E25d does not refine or extend E25c's S grid. Constant s only — a σ-gated per-step s schedule at inference (matching training's σ>0.5 gate) is a recorded follow-up that needs sampler-boundary code and its own registration. |
| **Depends on** | Stage-2 rescond checkpoints `e25b2_{hews,channel}_rescond_s{1001,1002,1003}`; `e4_prompts_sfw.json` (frozen SFW paper prompts + gen_seeds); `demote_bucket_for` (the exact trained 896 sibling grid); PE-Core pooled cosine (`pool_and_normalize`, the E4/E25b render-read convention). |

## Honesty paragraph (recorded at freeze)

Training paired the 896 grid with s = −0.193 **only on σ > 0.5 steps**; a
full 896-grid render runs every step — including low-σ steps — on a
(grid, σ) combination the run never trained, whatever s is. Neither 896
arm is in-distribution; the read is strictly *comparative*: same grid,
same seed, same init noise, the only difference the truthful-vs-false
label. PE-Core cosine is resolution-robust but not resolution-blind, so
the absolute cos(896-arm ~ 1024-ref) carries a resolution confound —
**the gate reads only the paired within-seed delta between the two 896
arms**, where that confound cancels. The SFW prompt subset (the paper's
eval-sfw amendment; 9 hews / 12 channel) is frozen here as the prompt
grid — chosen before any E25d render existed. Judgment constants (≥5/6
checkpoints, per-checkpoint median over prompts) mirror 25c-2's
conventions. Note also: the Stage-2 rescond arms trained with
`late:0.75` spans active via the base.toml default (recorded 2026-08-12)
— "rescond" here means rescond-on-late-combo, as trained; intended (the
goal is per-sample gradient similarity), recorded for the label's sake.

## Frozen design — one render pass (no training)

Per checkpoint (6 = 2 corpora × 3 seeds), per frozen SFW prompt at its
frozen `gen_seed`, **three arms in one boot**:

| arm | size | s |
|---|---|---|
| `ref1024` | the prompt's native free-fit bucket | 0 (trained native point) |
| `plain896` | `demote_bucket_for(W, H, 1024, 896)` — the exact trained sibling grid | 0 (**false** label) |
| `cond896` | same 896 bucket | **−0.193** (**truthful** label) |

`plain896`/`cond896` share latent shape and seed ⇒ identical init noise
(CRN pair); the checkpoint loads once per cell through the shipped
attach (`--lora_weight` merge + `_attach_res_cond` at s = 0) and the s
mutation between arms uses the same eager `(proj, s)` seam the trainer
uses per-step. 20 steps / CFG 1.0 (the paper render convention). A
reboot mid-run discards prior latents (boot-id stamped; no cross-boot
render comparisons).

### 25d-1 — transfer (primary)

Per (checkpoint, prompt): Δcos = cos(cond896 ~ ref1024) −
cos(plain896 ~ ref1024), PE-Core pooled unit-norm dot. Per checkpoint:
median Δcos over its prompts.

| outcome (6 checkpoints) | verdict |
|---|---|
| median Δcos > 0 on **≥ 5/6** (both corpora among the positives) | **TRANSFER** — truthful labeling moves 896-grid renders toward the model's own native-grid behavior; the absorption is live at inference. |
| median Δcos < 0 on ≥ 5/6 | **ANTI** — the truthful label pushes renders *away*; recorded, mechanistically interesting (the delta acts as a style perturbation, not a grid compensator). |
| otherwise | **NULL** — labeling is inert at inference; absorption is a training-dynamics phenomenon only. |

Recorded regardless: per-checkpoint baseline gap median
cos(plain896 ~ ref1024) (how far 896 sits at all), per-prompt Δcos sign
counts, per-corpus split.

### Descriptives (no verdict weight)

Radial power spectrum (luma, normalized frequency; high band frozen at
f/f_nyq ≥ 0.5) of the three arms — does `cond896` move the high-band
mass toward `ref1024`'s? Eyeball sheets per (corpus, seed): rows =
prompts, cols = ref1024 / plain896 / cond896 — sheets accompany the
verdict; a TRANSFER with visible artifacts is recorded as caveated.

## Kill switches / honesty

- One render pass, frozen arm set — no other s values at 896, no
  post-hoc size or prompt refinement, no retry on a near-miss.
- No E25d result feeds back into training or any default; `--res_cond_s`
  remains experimental opt-in regardless of outcome.
- All comparisons within one boot; resume across a reboot re-renders.
- NULL/ANTI closes the intended-use knob at this operating point; the
  recorded reopening is training-side (σ-unrestricted or denser-axis
  conditioning), an E25-family amendment first.
- E25c's frozen sweep remains its own single sweep; nothing here
  substitutes for it.

## Cost

| item | cost |
|---|---|
| renders — each ckpt renders its own corpus's prompts: (3×9 + 3×12) cells × 3 arms = 189 renders, one boot | ≈ 1–1.5 GPU-h |
| read (PE-cos + spectra + sheets) | CPU/GPU minutes, same job |

## Result (2026-08-12) — 25d-1 ANTI

Run `runs/20260812-e25d/` (daemon job 20260812-180631-822164, rc = 0,
one boot, boot-id stamped); all 189 renders landed, read in the same
job. Summary JSON committed as `e25d_result.json` (full per-prompt
table); eyeball sheets + RAPSD descriptives in the run dir.

Per checkpoint (median Δcos = cos(cond896 ~ ref1024) −
cos(plain896 ~ ref1024), paired within seed):

| ckpt | median Δcos | prompts Δcos > 0 | baseline gap (median cos plain896 ~ ref1024) |
|---|---|---|---|
| hews_s1001 | −0.0009 | 1/9 | 0.909 |
| hews_s1002 | −0.0167 | 2/9 | 0.882 |
| hews_s1003 | −0.0066 | 2/9 | 0.933 |
| channel_s1001 | −0.0004 | 6/12 | 0.927 |
| channel_s1002 | −0.0040 | 4/12 | 0.929 |
| channel_s1003 | **+0.0071** | 8/12 | 0.929 |

5/6 medians negative ⇒ **ANTI** as frozen. Honest characterization of
what the sign count is made of:

- **The ANTI signal is hews-borne.** hews is 3/3 negative with clearly
  negative prompt-level mass (5/27 prompts positive across its three
  seeds). channel is 18/36 positive at the prompt level — an exact coin
  flip — with two tiny negative medians (−0.0004, −0.0040) and one
  positive (+0.0071); read alone, channel would be NULL. The 6-ckpt
  sign-count gate lands ANTI because both near-zero channel medians
  fall on the negative side.
- **Effect sizes are small.** Only hews_s1002 (−0.0167) exceeds 0.01;
  the baseline gap the label was supposed to close is ~0.07–0.12 of
  cosine (plain896 sits at 0.88–0.93 from ref1024). The truthful label
  closes none of it and on hews consistently costs a little.
- **No spectral compensation either** (descriptive): cond896's
  high-band RAPSD mass is closer to ref1024's than plain896's on only
  34/63 cells (hews 18/27, channel 16/36) — no coherent high-band move
  toward native.
- The corpus asymmetry echoes the same-day base-model eyeball
  (`runs/20260812-e25c-eyeball-base`: hews projection ≈ s-inert offset,
  channel projection the live axis) — recorded as a descriptive rhyme
  only, no claim.

**Reading (per the frozen ANTI branch):** the truthful label acts as a
weak style perturbation at inference, not a grid compensator — the
per-step absorption 25b-1 demonstrated during training does not
transfer to full-trajectory 896-grid renders, consistent with 25b-2/3's
"converges to a different model" story and with the honesty paragraph's
warning that full renders are off the trained (grid, σ) joint for both
arms. The intended-use knob is **closed at this operating point**; the
recorded reopening is training-side (σ-unrestricted or denser-axis
conditioning), an E25-family amendment first. Kill switches honored:
one pass, no retries, nothing feeds training, `--res_cond_s` stays
experimental opt-in, E25c's frozen sweep untouched.
