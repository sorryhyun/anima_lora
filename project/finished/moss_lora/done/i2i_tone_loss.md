# MOSS i2i: a tone term on the unedited tokens

Status: **shipped 2026-10-28** as sanitize v3 (`tone_weight 0.003`, `tone_start 0.5`, b ×1
on the w0005 lr 5e-5 e8 recipe; Hub `sorryhyun/MOSS-I2I-LORAs`, `sanitize/moss_i2i_sanitize_v3`).
Mean over five pages × three seeds: ΔL −0.03, Δb −0.23. User doc: `docs/moss/methods/tone_loss.md`.
Open: the ~20-page held-out check, Δa +0.4 (unchanged in every run), swap_text.

The sanitize LoRA `moss_i2i_sanitize_wm` (e6) brightens, cools and slightly desaturates
the whole page, including far from any text it removes. The training pairs do not teach
this, so the shift is something the run produced on its own. This proposal adds a loss
term that compares the coarse tone of the unedited tokens' x̂₀ with the target's, so
"keep the page's tone" becomes an explicit constraint rather than a by-product of MSE.

## Measurements

Five held-out pages (`output/moss_i2i_eval/sanitize/inputs/{hard_to_sanitize1-3,usual_case1-2}.png`), all
rendered with the same CLI settings: `scripts/moss/generate.py`, the sanitize
`generate.toml`, `--sampler er_sde --cns auto --spectrum`, seed 42. Each output is
compared with its input in Lab after a σ=3 px blur, on one shared mask per image: the
pixels **neither** model changed by ΔE > 8, dilated by 20 px (26-71 % of the page).

| image | 200p ΔL / Δb / sat× | e6 ΔL / Δb / sat× |
|---|---|---|
| hard_to_sanitize1 | +0.24 / −0.23 / 1.07 | **+1.95 / −0.63** / 0.99 |
| hard_to_sanitize2 | −0.45 / −0.14 / 0.97 | **+0.90 / −0.68** / 1.01 |
| hard_to_sanitize3 | −0.32 / +0.10 / 0.92 | **+1.98 / −1.06 / 0.80** |
| usual_case1 | −0.28 / +0.07 / 1.04 | +0.57 / −0.23 / 0.92 |
| usual_case2 | +0.19 / +0.33 / 1.02 | **+2.28 / −0.52** / 0.97 |

- 200p (`moss_i2i_sanitize_200p`: 200 pairs, AdamW + `dc_isolate`) moves lightness by at
  most ±0.5, in no consistent direction.
- e6 (`moss_i2i_sanitize_wm`: 280 pairs incl. 80 `wm_*`, Muon, `dc_isolate` off, 6 ep) is
  brighter on all five (+0.6 to +2.3) and bluer on all five (Δb −0.2 to −1.1). A median
  ΔE ≈ 2.5 is about where people start to notice.
- The renderer matters too: the same 200p LoRA moved +0.6 to +1.0 in L from ComfyUI to the
  CLI. Compare runs only within one renderer.

**The data is tone-neutral.** Target − condition over the unedited pixels of every staged
pair (same blur, ΔE > 8 mask, 1024 px long side):

| pair kind | n | ΔL mean | Δb mean | sat× median |
|---|---|---|---|---|
| textless | 160 | −0.00 | −0.00 | 1.000 |
| mined twin | 39 | −0.00 | −0.01 | 1.000 |
| wm | 80 | −0.05 | +0.02 | 1.005 |

**The set's overall tone points the other way.** Absolute means (Lab, HSV saturation, 512 px
thumbnails):

| group | n | L | b | chroma | sat | L > 90 |
|---|---|---|---|---|---|---|
| staged targets | 200 | 62.0 | +6.22 | 13.0 | 0.212 | 14.4 % |
| `wm_*` sources (counted twice) | 80 | 63.2 | +6.48 | 13.0 | 0.212 | 17.6 % |
| the five test pages | 5 | 63.9 | +1.46 | 10.8 | 0.169 | 26.2 % |
| corpus sample (`retrieved/`) | 400 | 68.8 | +4.01 | 12.5 | 0.195 | 29.8 % |

The targets are darker, yellower and more saturated than the corpus. A pull toward the
set's mean would darken and warm the output; e6 brightens and cools it. The duplicated
`wm_*` sources sit within 1.2 L of the rest.

The lossless-WebP change does not bear on this: the converted PNGs were already lossless,
so no training pixel changed.

## Why plain MSE lets it through (hypothesis)

The flow-matching loss is the per-token MSE on v. A tone offset on the copied area is
one more residual in that mean, with no separate weight. At ΔL ≈ 2 it is small next to
the per-step loss (0.09 average in the last epoch of e6), and the LoRA's capacity goes
to the edit. `rest_weight > 1` (the region loss already in `train.py`) does not fix
this: it scales every frequency on the rest tokens equally, so the offset stays the same
share of the loss.

Two other causes are not ruled out, and Phase 0 checks the cheap one:
- the shift accumulates with training (e3 vs e6);
- Muon / `dc_isolate` off vs AdamW, or the `wm_*` pairs. Only a 2×2 of runs separates
  these, so it is left out unless Phase 1 fails.

## The term

With `x_t = (1−σ)x₀ + σε` and `v = ε − x₀`, the prediction's clean estimate is
`x̂₀ = x_t − σ·v̂`, so `x̂₀ − x₀ = σ(v − v̂)`. On the rest tokens `R` (the complement of
`edit_region`, grown by an extra margin), pool the error coarsely per channel and
penalise the pooled values:

```
e      = (x̂₀ − x₀) ⊙ 1_R                                  # (1, 128, h, w), packed latent
pooled = avgpool_k(e) / avgpool_k(1_R)                     # masked mean per k×k cell
L      = L_mse + tone_weight · mean over cells with coverage ≥ ½ of pooled²
```

- `tone_pool = 0` means one cell (the per-channel mean over all of `R`); `k > 0` pools
  k×k tokens (k = 8 is 128 px at 16 px per token), which also catches tone drifting
  between regions.
- The term reads x̂₀, not v, because tone is a property of the image. Its σ factor
  down-weights high σ, where x̂₀ is still mostly a guess.
- The mask comes from `edit_region(latent, ref, …)` as the region loss computes it: it
  needs a full-size reference (`ref_scale 1`), which sanitize and swap_text have, and
  pixel-aligned pairs. The staged mined twins pass (unedited share 82 %, ΔL 0.00).
  `tone_dilate` grows the rest mask's margin beyond `region_dilate`, so seams stay out
  of `R`.
- Logged separately as `tone` in progress events, like the region / rest parts.

New `TrainRequest` fields (advanced): `tone_weight` (0 = off, the default),
`tone_pool` (0 | k), `tone_dilate` (tokens, default 2), `tone_b_weight` (scale on the
Lab projection's b row, 1), `tone_start` / `tone_b_start` (fraction of all steps before
which the term is only logged / the b scale is 1; 0).

## Plan

**Phase 0: does the shift grow with training? (about 10 min, no code)**
Render `moss_i2i_sanitize_wm_e3` with the settings above and run the tone script. If e3
is already near e6, the shift sets in early and the term needs to act from the start; if
e3 is clean, stopping earlier is a second fix to keep in mind.

**Phase 1: implement + invariant tests**
- `library/moss/train.py`: `tone_loss(pred, target, latent, noisy, sigma, rest_mask, pool)`
  next to `region_mse`; precompute the rest masks once per sample like `regions`.
- Tests (`tests/test_moss_train.py`):
  - `tone_weight = 0` gives a loss bit-identical to today's.
  - A constant per-channel offset on x̂₀ over `R` gives exactly offset² at `tone_pool = 0`.
  - Errors inside the edit region contribute nothing.
  - Cells under half coverage are skipped.
- Bench (CONTRIBUTING Tier 2): promote the tone script to a bench on `bench/_common.py`.
  Input: a LoRA plus a page folder. Output: per-image ΔL / Δb / sat× / ΔE on the
  shared-mask method, in the `result.json` envelope. (It moved as is, without the
  envelope, to `bench/moss_i2i_eval/tone_map.py`.)

**Phase 2: arms (e6 recipe, 280 pairs, Muon, local `blocks_to_swap 28` or a pod)**
Micro arms first, `max_steps 560` (2 epochs), compared with an equal-step control:

| arm | tone_weight | tone_pool |
|---|---|---|
| control | 0 | — |
| DC | λ | 0 |
| pooled | λ | 8 |

Pick λ from the logged magnitudes of the control (`tone` logged but not added), so the
term starts at about 10 % of `L_mse`. Extend the winner to 6 epochs.

**Gates (on the bench page set; widen to ~20 held-out sanitize pages before shipping)**
- |ΔL| ≤ 0.5 and |Δb| ≤ 0.3 on average, sat× within 0.97–1.03, with no page past
  |ΔL| 1.0.
- Edit quality no worse than the control by eye on hard_to_sanitize2 (red caption boxes,
  dense SFX) and usual_case2 (faint tiled watermark).
- The edit-region part of the loss (logged `region`) no worse than the control at equal
  steps.

## Results (2026-10-25)

Renders as in [Measurements](#measurements), mean over the five pages (scripts:
`bench/moss_i2i_eval/tone_map.py`; the DC-bias analysis and ablation that were here are
in git history):

| run | recipe | ΔL | Δb | sat× | max \|ΔL\| |
|---|---|---|---|---|---|
| 200p | AdamW + `dc_isolate`, 200 pairs | −0.07 | −0.02 | 0.99 | 0.33 |
| cs0 | Muon lr 1e-4, 280 pairs, 600 steps | +1.37 | −0.64 | 0.93 | 1.74 |
| w0005 | cs0 + `tone_weight 0.005` | −0.75 | +0.45 | 1.03 | 1.42 |
| w0002 raw | cs0 + `tone_weight 0.002` | −0.36 | −0.96 | 1.08 | 1.41 |
| w0002 EMA | same run, EMA 0.97 | +0.55 | −0.69 | 1.02 | 1.10 |

- **No run passes every gate.** 200p is the only one inside the tone gates, but it edits
  worse than the Muon runs (user read, by eye). The drift shows in every Muon run and not
  in 200p, but optimizer, `dc_isolate`, the `wm_*` pairs and step count all differ
  between them; no run separates them.
- **The drift grows with training**: e6's e3 → e6 checkpoints, ΔL +0.48 → +1.74.
- **`tone_weight 0.005` over-corrects**: darker and warmer on 4 of 5 pages, and it leaves
  the 「無断転載禁止」 watermark on hard_to_sanitize1 (both w0002 files remove it).
- **Checkpoint noise is gate-sized**: EMA and raw of one run differ by +0.6 to +1.3 L, the
  same sign on all five pages. So the arms above, one checkpoint each, do not resolve the
  term's effect at 0.002.
- **Muon + EMA on i2i**: the transfer evidence is colorize, not sanitize. Hub v4 (Muon,
  kaiming, `ema_window 0.1`) beat the AdamW v3 on the 12-page holdout (chroma err 13.3 →
  11.2), with the TE (fp8 → int8) and steps (8k → 10k) also changed
  (`project/finished/moss_lora/README.md` § colorize). On sanitize, one run's EMA filled
  hard_to_sanitize2's red box cleaner than its raw weights, by eye.
- **CNS and CFG are not the cause (2026-10-26, settled — do not re-raise).**
  `moss_i2i_sanitize_w0002_ema_e6` (Muon lr 5e-5, 6 ep, `tone_weight 0.002`,
  `tone_pool 8`, `ema_window 0.1`), same five pages, mean over pages:

  | render | ΔL | Δb |
  |---|---|---|
  | cfg 3, CNS auto | +0.34 | −0.99 |
  | cfg 3, CNS off | +0.35 | −1.05 |
  | cfg 1, CNS auto | −0.15 | −0.69 |

  CNS off changes no page by more than 0.2. At cfg 1 every page is still bluer
  (Δb −0.4 to −1.5) and hard_to_sanitize2 keeps its contrast compression (input L 0–20
  +2.0, 90–101 −1.8), so the drift is in the LoRA's conditional prediction, not the
  sampler. Spectrum stayed on in all three.
- **Identity pairs do not fix the drift (2026-10-26).** `moss_i2i_sanitize_idt_cs0_600`:
  the cs0 recipe on a 277-pair tree with the 42 stacked `wm_*` pairs swapped for 39
  clean → clean pairs (`staging/_identical/`, textless pages the textless miner had
  dropped). Same renders, one shared mask with cs0, mean over the five pages:

  | run | ΔL | Δb | sat× |
  |---|---|---|---|
  | cs0 | +1.19 | −0.64 | 0.96 |
  | idt raw | +2.09 | −0.47 | 0.94 |
  | idt EMA (0.1) | +0.63 | −1.08 | 0.96 |

  The raw weights are brighter than cs0 on all five pages (+0.4 to +1.8), most in the
  dark and mid tones (input L 0–60: +2 to +4). The per-pixel ΔL maps show the same
  page-wide lift outside the edits, not a local effect. The EMA's lower ΔL is within the
  raw-vs-EMA spread seen before. Edit quality is unchanged by eye: 「無断転載禁止」,
  the handle and usual_case2's tiled watermark all go.
- **`tone_weight 0.005` at lr 5e-5 fixes ΔL, not Δb (2026-10-27).**
  `moss_i2i_sanitize_idt_w0005_lr5e5_e8`: the identity tree, Muon lr 5e-5,
  `tone_weight 0.005`, 8 epochs (2216 steps), EMA saved every 2 epochs. Mean over the five
  pages and seeds 42 / 43 / 44, one shared mask per row:

  | checkpoint | ΔL | Δb | max \|ΔL\| |
  |---|---|---|---|
  | e2 EMA | −0.03 | −0.19 | 0.94 |
  | e4 EMA | −0.08 | −0.44 | 1.26 |
  | e6 EMA | −0.04 | −0.47 | 1.01 |
  | e8 EMA | +0.02 | −0.63 | 0.57 |
  | e8 raw | +0.10 | +0.34 | 1.41 |

  e8 EMA passes the ΔL, max-|ΔL| and sat× gates and fails Δb: bluer on all 15
  page × seed renders. Unlike `tone_weight 0.005` at lr 1e-4 it does not over-correct, and
  it edits best of all runs so far: it removes hard_to_sanitize2's red caption boxes and
  repaints under them (cs0 and e2 leave red blank shapes), and removes
  「無断転載禁止」 and the handle. Seeds move a checkpoint's mean Δb by about ±0.1, so
  the Δb is real. The raw and EMA files differ by ~1.0 in b with opposite signs, so b
  wanders with the weights. Δb grows with training (e2 → e8).
- **Tone jitter did not help (2026-10-27; code reverted).** Each step shifted the target
  and the reference latent by the same random Lab offset (L / a / b each uniform in ±3,
  along `tone_dirs`), so that the unedited tone would pass through. Same recipe and seeds
  as the run above (`moss_i2i_sanitize_idt_w0005_jit3_e8`), ΔL / Δb:

  | checkpoint | control | jitter ±3 |
  |---|---|---|
  | e2 EMA | −0.03 / −0.19 | +0.22 / −0.03 |
  | e4 EMA | −0.08 / −0.44 | +0.02 / −0.30 |
  | e6 EMA | −0.04 / −0.47 | +0.21 / −0.17 |
  | e8 EMA | +0.02 / −0.63 | +0.42 / −0.47 |
  | e8 raw | +0.10 / +0.34 | +1.30 / −0.33 |

  Jitter moves Δb 0.14–0.30 toward zero at every checkpoint. Δb grows from e2 to e8 by
  the same ~0.45 in both arms, so only the starting point moves. The costs:
  - e8 EMA is still past the Δb gate.
  - The brightening returns: ΔL +0.42 for e8 EMA, and +1.30 for raw, with one page at
    +2.45.
  - Edits get worse. e8 EMA leaves a faint 「禁止」 on hard_to_sanitize1 and red-box
    debris on hard_to_sanitize2, while the control's e8 EMA removes both.
- **The small VAE decoder is not the cause.** MOSS ships the FLUX.2 small decoder: the
  same encoder and BN, with a 3/4-width decoder. Decoding the same latents with the
  full-width decoder (FLUX.2-klein-4B's VAE) changes the five pages' Δb by about +0.05
  on average. In a round trip the small decoder is bluer (Δb −0.26 vs −0.09 over 20
  pages), mostly on the more saturated staged pages. PSNR is equal (38.0 dB), and small
  Hangul is indistinguishable between the two decoders.

- **The term fixed L, not b (2026-10-27).** 200p and four Muon runs on one
  shared mask (seed 42, mean over the five pages; Δb warm = input b > 5, i.e. skin, red,
  orange):

  | run | ΔL | Δa | Δb | Δb warm | median ΔE | max \|ΔL\| |
  |---|---|---|---|---|---|---|
  | 200p | −0.12 | +0.30 | −0.01 | −0.04 | 1.09 | 0.39 |
  | e6 | +1.67 | +0.40 | −0.67 | −1.30 | 1.98 | 2.32 |
  | cs0 | +1.35 | +0.28 | −0.66 | −1.17 | 1.76 | 1.66 |
  | idt cs0 EMA | +0.81 | +0.49 | −1.10 | −2.00 | 1.77 | 1.02 |
  | idt w0005 lr 5e-5 e8 EMA | +0.19 | +0.36 | −0.72 | −1.17 | 1.26 | 0.45 |

  The e8 EMA's Δb is cs0's. The four Muon runs share the spatial Δb pattern (on
  hard_to_sanitize3 the stockings and skirt bluest, hair and background faint), while 200p
  goes warmer. Δb is largest in mid and dark tones and at high chroma; paper white is near 0.
  Δa ≈ +0.3 is in 200p too, so it is not the LoRA's.
- **The b error is in the single-step x̂₀, on training pairs too (2026-10-27).**
  `tone_probe.py` (job `20261027-122703-f762ce`, 24 pairs of the identity tree + the five
  pages; `project/finished/moss_lora/space/tone_probe_idt_w0005.json`, untracked), decoded x̂₀ vs
  x₀ on the rest pixels, Δb at σ 0.1 / 0.3:

  | LoRA | training pairs | held-out pages | render Δb |
  |---|---|---|---|
  | base | +0.13 / +0.26 | +0.12 / +0.33 | — |
  | e2 EMA | −0.06 / 0.00 | −0.05 / +0.03 | −0.19 |
  | e8 EMA | −0.09 / −0.10 | +0.01 / −0.10 | −0.63 |
  | e8 raw | +0.38 / +0.60 | +0.42 / +0.46 | +0.34 |

  Pairs and pages agree, so the term sees the b error during training; it is not a
  held-out gap. The render follows the single-step sign (raw warm, EMA cool), about 6× for
  the EMA, like the 5–8× seen for L. A b error of 0.1 is 0.01 in ΔE², 4–25× less gradient
  than the L error the term did fix (single-step 0.2–0.5). The latent DC projected on the
  tone directions agrees (e8 EMA at σ 0.3: b −0.15 pairs, −0.19 pages). At σ 0.6 the b
  error averages 0, while the term's magnitude is dominated by high σ.

- **A late b ×3 term overshoots b (2026-10-27).** The idea was to scale the b row of the
  Lab projection ×3, so that a b error (single-step ~0.1) weighs like the L error the term
  removed (0.2–0.5). Two runs on the identity tree:
  - On the w0005 e8 recipe from step 0 (stopped at step 245 of 2216): the tone term ran
    about 5× b ×1's, the gradient norm about 4× (2.8 / 5.2 at σ < 0.5 / ≥ 0.5, against
    0.7 / 1.2), clipped on nearly every step.
  - `moss_i2i_sanitize_idt_cs0_600_late_w0005_b3`: the `idt_cs0_600` recipe with
    `tone_weight 0.005`, `tone_b_weight 3`, `tone_start 0.5`, and the same σ sequence as
    the control. Until step 300 the logged term fell with FM alone (28 → 7.2) and FM
    matched the control (0.091 / 0.092). When the term switched on, the gradient norm went
    0.20 → 11.2 and the term *rose* to 29, settling at ~21. FM ended 15–25 % above the
    control.

  Final EMA, mean over the five pages:

  | run | seed | ΔL | Δa | Δb | Δb warm | median ΔE |
  |---|---|---|---|---|---|---|
  | idt cs0 EMA (control) | 42, shared mask | +0.81 | +0.49 | −1.10 | −2.00 | 1.77 |
  | late b ×3 EMA | 42, shared mask | +0.58 | +0.61 | +1.20 | +0.89 | 2.01 |
  | idt w0005 lr 5e-5 e8 EMA | 42 / 43 / 44 mean, two-run mask | +0.02 | +0.32 | −0.63 | −0.98 | 1.18 |
  | late b ×3 EMA | 42 / 43 / 44 mean, two-run mask | +0.36 | +0.59 | +1.29 | +1.08 | 2.03 |

  Every page at every seed is warmer (Δb +0.4 to +2.2, most on hard_to_sanitize1 / 2).
  hard_to_sanitize2 also goes darker (ΔL −0.4 to −1.5) while the other pages go brighter.
  300 steps moved b by 2.4, so b responds to the term; the weight is too strong.
  Two-run mask: pixels neither of the two runs changed at that seed (`idt_cs0_600` has
  only seed 42).

- **A late `tone_weight 0.001` term passes the mean gates (2026-10-27).**
  `moss_i2i_sanitize_idt_cs0_600_late_w0001`: the `idt_cs0_600` recipe (Muon lr 1e-4,
  600 steps, identity tree) with `tone_start 0.5`, `tone_weight 0.001`, b ×1. Final EMA,
  mean over the five pages and seeds 42 / 43 / 44, mask from the b ×3 arm and this one:

  | run | ΔL | Δa | Δb | Δb warm | median ΔE | sat× |
  |---|---|---|---|---|---|---|
  | idt cs0 EMA (control, seed 42 only) | +0.84 | +0.48 | −1.10 | −2.01 | 1.77 | — |
  | late b ×3 EMA | +0.36 | +0.58 | +1.29 | +1.09 | 2.03 | — |
  | **late w0001 EMA** | **−0.31** | +0.53 | **+0.04** | −0.45 | 1.33 | 0.99 |

  - Mean |ΔL| and |Δb| are inside the gates, sat× 0.98–1.00 per seed. One page breaks the
    per-page gate: hard_to_sanitize2 at seed 43, ΔL −1.04. L now errs dark
    (hard_to_sanitize1 / 3: −0.35 to −0.85), and the warm regions stay cool. Δa +0.5 has
    not moved in any run.
  - Edits match the control by eye: 「無断転載禁止」 and the handle go on
    hard_to_sanitize1 (seed 42 keeps the empty top-left bubble outline), and
    hard_to_sanitize2 keeps the same red-box debris as the control.
  - Training: the gradient norm went 0.22 → 0.56 when the term switched on (b ×3:
    0.20 → 11.2), and FM stayed at the control's. Output-space gradient norms (logged
    `g_fm` / `g_tone` / `g_cos`, ∂/∂pred, the tone one at the configured weight):
    tone / FM 1.4 before the switch, 2.0–2.3 after, cos +0.03 to +0.07.
  - The logged term *rises* when it is added (2.6 → 7.2 over steps 200–300 → 301–350,
    ~5.7 after), as it did under b ×3, while the renders move toward the gate. Not
    explained yet.

- **The late w0001 term does not carry to full length (2026-10-28).**
  `moss_i2i_sanitize_idt_w0001_late_lr5e5_e8`: the w0005 lr 5e-5 e8 recipe (2216 steps)
  with `tone_start 0.5`, `tone_weight 0.001`, b ×1. Final EMA, mean over the five pages
  and seeds 42 / 43 / 44, mask from the three runs below:

  | run | ΔL | Δa | Δb | median ΔE | sat× | pages past \|ΔL\| 1 |
  |---|---|---|---|---|---|---|
  | **late w0001 e8 EMA** | −0.39 | +0.41 | **−0.87** | 1.41 | 0.92–1.04 | hard_to_sanitize2 at all three seeds (−1.07 / −2.10 / −1.48) |
  | late w0001 600 EMA | −0.30 | +0.53 | +0.03 | 1.33 | 0.92–1.05 | hard_to_sanitize2 seed 43 (−1.05) |
  | w0005 e8 EMA | +0.03 | +0.33 | −0.63 | 1.18 | 0.93–1.04 | none |

  - Δb fails the gate and is cooler than w0005 e8 from step 0; every seed is −0.83 to
    −0.92. The 600-step arm's Δb ≈ 0 does not survive 1100 more steps at this weight.
  - hard_to_sanitize2 goes dark at every seed, most in the bright input (L 60–101:
    −1.5 to −2.9) while L < 20 brightens.
  - Training: |g| 0.20 → 0.65 when the term switched on, back to 0.25 by step 1700;
    tone / FM output-space gradient 1.3 before, 3.1 just after, 2.0 at the end.

- **A late `tone_weight 0.003` term passes the mean gates at full length (2026-10-28).**
  `moss_i2i_sanitize_idt_w0003_late_lr5e5_e8`: the w0001 e8 run above with
  `tone_weight 0.003`. Final EMA, mean over the five pages and seeds 42 / 43 / 44, mask
  from the three runs:

  | run | ΔL | Δa | Δb | median ΔE | sat× per seed | pages past \|ΔL\| 1 |
  |---|---|---|---|---|---|---|
  | **late w0003 e8 EMA** | **−0.03** | +0.41 | **−0.23** | **1.07** | 0.966 / 0.986 / 0.978 | hard_to_sanitize2 seed 43 (−1.07) |
  | late w0001 e8 EMA | −0.39 | +0.41 | −0.87 | 1.42 | — | hard_to_sanitize2 at all three seeds |
  | w0005 e8 EMA | +0.03 | +0.33 | −0.63 | 1.19 | — | none |

  - Per page (ΔL / Δb, seed mean): hard_to_sanitize1 −0.07 / +0.12, hard_to_sanitize2
    −0.53 / −0.12, hard_to_sanitize3 −0.09 / −0.26, usual_case1 +0.13 / −0.59,
    usual_case2 +0.38 / −0.32. Every seed is inside the mean ΔL and Δb gates (Δb −0.19 to
    −0.29). sat× at seed 42 is 0.966, just under 0.97. Δa +0.4 is unchanged again.
  - Edits match w0005 e8 by eye at seed 42: hard_to_sanitize2 loses every caption box and
    SFX with the same red debris top-left, usual_case2 loses the tiled watermark, and
    hard_to_sanitize1 loses 「無断転載禁止」 and the handle.
  - Training, against the w0001 e8 run at equal steps (same σ sequence): FM 1.00× before
    the switch and 1.02–1.05× after. The term rose on the switch as before (0.83 → 3.7),
    then kept falling to 0.85 by step 1900, where the w0001 term stalled at ~2.4.
    |g| 0.17 → 1.23 (clipped at 1.0 on part of the steps), back to 0.31 by step 1700.

## Out of scope

- **A correction at render time.** Fit a per-channel tone shift on the unedited region
  (Lab or latent) against the full-size reference and apply it to the output. It works on
  LoRAs already trained and stays available as a fallback, but it hides the cause.
- The other i2i tasks. Colorize changes tone on purpose, so the term does not apply;
  swap_text could take it unchanged once sanitize shows it works.
