# Anima LoRA: what is left after the clip / EMA recipe

Status: OWED (2026-11-05). Two questions stay open. The settled work is in
[`../findings/lora_gradient_noise_recipe.md`](../findings/lora_gradient_noise_recipe.md):
the grad-norm and noise-scale measurements, the median and per-σ-band clip and how
they are implemented, the three blind A/Bs (s29 recipe B, s30 band clip, s31 T-LoRA
mask), and the σ gate renders.

Ground rules carried over: several training seeds per arm, blind pairs at the `make
test` recipe (28 steps euler, cfg 4, shift 3, 1024²) graded in `anima-blind-pairs`,
and `bench/anima_abclip/run_ab.py` for train / render / make / score / answer. The
images that cause gradient spikes stay in the set. Each run is hour-scale and gets an
explicit go first.

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

Next: the shipped recipe against it with `timestep_sampling = "shift"` and
`discrete_flow_shift = 2` (a gentler start than 3), seeds 42–45. Read blind wins and
cross-seed MAD (`bench/anima_abclip/sigma_gate.py` style) side by side. The band clip
matters here: σ >= 0.8 keeps its own median, so the extra draws there are not clipped
toward mid-σ norms. Test after question 2 is settled, because under the mask the extra
high-σ draws would train only 1–8 ranks.

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

- **Mismatched against plain.** Render the s31 band LoRAs at full rank against the
  nomask LoRAs (both on disk, render seeds 5 and 6; renders only). If the mismatch loses,
  it may have cost s29 / s30 their arms equally, so their verdicts stand, but the
  absolute quality of those LoRAs was below what their weights could give.
- **A second artist** before treating "mask = no mask" as general; caststation has
  B and band LoRAs on disk to train against.
- If the mask returns, stamp `use_timestep_mask` / `min_rank` / `alpha_rank_scale` in
  the LoRA so inference can apply it without flags. They are not stamped now.
