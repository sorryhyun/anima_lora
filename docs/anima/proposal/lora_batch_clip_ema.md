# Anima LoRA: what is left after the clip / EMA recipe

Status: OWED (2026-11-06). §0 ran (s33): B lost its lead once the mask was off
and ||dW|| matched. Its follow-up (B at its own ||dW||) and §1–§2 stay open. The settled work is in
[`../findings/lora_gradient_noise_recipe.md`](../findings/lora_gradient_noise_recipe.md):
the grad-norm and noise-scale measurements, the median and per-σ-band clip and how
they are implemented, the three blind A/Bs (s29 recipe B, s30 band clip, s31 T-LoRA
mask), and the σ gate renders.

Ground rules carried over: several training seeds per arm, blind pairs at the `make
test` recipe (28 steps euler, cfg 4, shift 3, 1024²) graded in `anima-blind-pairs`,
and `bench/anima_abclip/run_ab.py` for train / render / make / score / answer. The
images that cause gradient spikes stay in the set. Each run is hour-scale and gets an
explicit go first.

## 0. Redo s29 without the T-LoRA mask

Every LoRA in s29 and s30 trained with `use_timestep_mask` on (`min_rank` 1) and
rendered at full rank, so each arm was trained as one model and rendered as another.
Most likely this, more than anything else, is what s29's 41–16 measured. The recipe
verdict needs a rerun on the maskless default:

- Arms as in s29, both with `use_timestep_mask=false`: **A** = 2e-5 cosine, bs 1,
  `max_grad_norm` 1.0 (fixed clip); **B** = 5e-5 `constant_with_warmup`, GA 2 (effective
  bs 2), median clip `clip_k` 2 per micro-batch, one window (no σ bands).
- Three artists: ama mitsuki, sweetonedollar, channel (caststation). Seeds 42–45,
  render seeds 9 / 10 (1 / 2 were planned; one check render at seed 1 was seen, so
  the set moved off them). Both arms retrain: the s29 LoRAs were deleted on 2026-11-05, and
  they carry the mask anyway.
- **Control ‖dW‖.** In s29, B moved 2.3–2.5× as far as A (median ‖dW‖), so a win can be
  strength, not the recipe. Rescale B to A's median ‖dW‖ of the same (artist, seed)
  before rendering (`match_dw`, as in s30–s32), so the pairs compare direction.
- s30 (band clip, 39–21) and the shipped recipe rest on B and also ran under the mask.
  Decide whether to redo s30 after this result.

**Result, s33 (2026-11-06, `run_ab.py --exp abclip2`,
[`report_abclip2.md`](../../../bench/anima_abclip/report_abclip2.md)).** B moved
2.2–2.5× as far as A, the same ratio as s29, and was rescaled by ×0.40–0.46. **A 52–42**, 2
ties; one-sided p for A 0.18 (pairs) and 0.17 (12 clusters); for B 0.87 / 0.88.

| | A | B |
|---|---:|---:|
| ama mitsuki | 17 | 13 |
| sweetonedollar | 19 | 13 |
| caststation | 16 | 16 |

Per (artist, seed): 6 A, 4 B, 2 even. Single seeds swing 6–2 either way (cast s45 to
B, ama s44 to A). At equal step length, B's direction is no better than A's. So s29's
41–16 came from the mask mismatch, from B's larger ||dW||, or from both. Owed next:

- **B at its own ||dW|| against A**, both without the mask. If B wins there, the lever
  is the step length (lr / schedule), not the clip, and the clip / GA part of the recipe
  has no blind win of its own.
- s30 rested on the masked B; redo it only if the clip is kept.

## 1. Training σ against inference σ

Training draws σ from logit-normal(0, 1) at shift 1 (`timestep_sampling = "sigmoid"`;
`discrete_flow_shift` only acts under `timestep_sampling = "shift"`). Inference runs
shift 3:

| | σ < 0.4 | σ >= 0.8 | σ >= 0.9 |
|---|---:|---:|---:|
| training draws, shift 1 | 30 % | 7.4 % | 0.6–1.4 % |
| inference steps (28, shift 3) | 5 steps | 12 steps | ~25 % |
| training draws, shift 2 | — | 24 % | — |
| training draws, shift 3 | 6.7 % | 38.7 % | — |

The σ gate (findings, sweetonedollar under B) puts nearly all of the LoRA's visible
effect at σ >= 0.8. With the LoRA off there, the render is the base's (MAD 8.5–8.9
against 50–59 with it on). With it off below 0.4, nothing changes past the noise floor.
The band also holds the seed-to-seed variation (40.9 between training seeds, 7.4 with
the LoRA off there). It is fit from about 18 micro-batches per run.

Two readings, and the A/B does not separate them in advance:

- More draws at high σ fit that band from more samples, so training seeds should agree
  more.
- More draws at high σ also learn more composition from a 30-image set, so leakage
  (uniform, pose, bubbles) can grow. Shift 3 also cuts σ < 0.4 to 6.7 % of draws.

s32 (2026-11-05, `run_ab.py --exp sigshift`, [`report_sigshift.md`](../../../bench/anima_abclip/report_sigshift.md)):
the shipped recipe (no mask) against `timestep_sampling = "shift"`,
`discrete_flow_shift = 2`, seeds 42–44, shift2 rescaled to shift1's median ‖dW‖
(×0.91–1.04), render seeds 7 / 8. **Tied overall, 22–23 (3 ties), cluster p 0.50**, and
the two artists split:

| | shift1 wins | shift2 wins | cross-seed MAD shift1 → shift2 |
|---|---:|---:|---:|
| caststation | 8 | 14 (every seed) | 49.1 → 43.4 |
| sweetonedollar | 14 | 9 (s42 6–2, s43 5–3) | 40.6 → 52.9 |

For each artist, the arm that won is also the one whose training seeds agreed more.
sweetonedollar's hug prompt went 5–0 to shift1. One reading: a tag the base already
renders (sweetonedollar, painterly at base) loses from the extra high-σ fit, and
caststation gains. Whether the base knows caststation is unchecked, and two artists
cannot separate this reading from artist noise. The shipped default stays at shift 1.

## 2. The T-LoRA mask

`use_timestep_mask` (on in `configs/methods/lora.toml` until 2026-11-05, `min_rank` 1)
masked ranks in training only. Inference ran every rank at every σ: the static merge
cannot vary with σ, and the eval forward skipped the mask. The T-LoRA paper (§3.1,
arXiv 2507.05964) and its code mask at inference as well. `inference.py
--lora_timestep_mask` now does that on the P-GRAFT hooks. There are three ways to run
a LoRA:

| | training | inference | status |
|---|---|---|---|
| T-LoRA as published | mask | mask (hooks, a custom node in ComfyUI) | s31: tied plain LoRA, 11–13 |
| plain LoRA | none | full rank | the default since 2026-11-05 (mask commented out) |
| what shipped until 2026-11-05 | mask | full rank | untested against either; every s29 / s30 LoRA |

In s31 (sweetonedollar, seeds 42–44) the two arms of a pair shared composition and
differed in detail. Wins followed the training seed (1–7, 6–2, 4–4). The published method
costs a custom node at inference and gained nothing here, so the default dropped the
mask.

Owed:

- **Mismatched against plain**: replaced by §0. The s31 band LoRAs were deleted on
  2026-11-05; §0 retrains without the mask instead of measuring the mismatch.
- **A second artist** before treating "mask = no mask" as general. The masked
  caststation LoRAs were deleted, so this now means training both arms.
- If the mask returns, stamp `use_timestep_mask` / `min_rank` / `alpha_rank_scale` in
  the LoRA so inference can apply it without flags. They are not stamped now.
