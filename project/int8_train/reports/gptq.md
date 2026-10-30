# Offline GPTQ base (2026-10-30)

**Verdict: a GPTQ base cuts per-prediction error by about 10% and CFG velocity error by
2–7%, the same on the base's own trajectories and on held-out real images.**
Calibrating on forward-noised real images is enough. The error that survives into the
CFG velocity is the activation term, which GPTQ does not touch.

## Calibration and per-linear error

`src/dit_gptq.py`, job `20261030-114754-5e5507`, 17.6 min. Two images per artist from
all 91 folders of `post_image_dataset/lora` (182 images, 3000–4608 tokens), each at one σ
from the inference schedule (28 steps, shift 3), run with its own caption and with the
negative prompt: 364 samples, 1.49M tokens per linear. Blocks quantized in order, each
feeding the next through the W8A8 kernel.

Output: `models/diffusion_models/anima-base-v1.0_int8_gptq.safetensors` (NVMe), 2.37 GiB, the full
DiT in ComfyUI's `int8_tensorwise` + `convrot` layout, `int8_id 9dad0c30a67e4d48…`.
Per-linear errors: `anima-base-v1.0_int8_gptq.report.json` next to it, copied to
[`gptq_linears.json`](gptq_linears.json).

Mean over the 168 linears, relative output error energy:

| | RTN | GPTQ | act-aware |
|---|---|---|---|
| W8A8 | 6.40e-5 | 4.12e-5 | 4.10e-5 |
| W8A16 (weight only) | 2.79e-5 | 0.52e-5 | — |

GPTQ removes 81% of the weight term. The W8A8 error drops 36%, and what is left is
mostly the activation term. Act-aware adds nothing over plain GPTQ. Its ridge had to be
raised above 1e-4 for some linears (mean 1.2e-4), where `H̃` is not positive definite.

## End to end: trajectories against held-out real images

`src/traj_eval.py`, job `20261030-120640-c9289f`. Raw numbers:
[`traj_eval.json`](traj_eval.json).

- `traj`: bf16-base Euler trajectories (CFG 4, 28 steps, shift 3, 1024²), the 4 A/B
  prompts × seeds 11, 22, all 224 states.
- `real`: 28 cached images outside the calibration pick, forward-noised at schedule σ.

Each arm is scored against bf16 at the same state as `‖v − v_bf16‖ / ‖v_bf16‖`, mean
per σ band:

| states | σ | n | cond RTN → GPTQ | uncond RTN → GPTQ | CFG velocity RTN → GPTQ |
|---|---|---|---|---|---|
| traj | > 0.7 | 128 | 0.0071 → 0.0066 (0.92x) | 0.0068 → 0.0060 (0.89x) | 0.0319 → 0.0306 (0.96x) |
| traj | 0.3–0.7 | 72 | 0.0082 → 0.0074 (0.90x) | 0.0081 → 0.0072 (0.90x) | 0.0368 → 0.0357 (0.97x) |
| traj | < 0.3 | 24 | 0.0152 → 0.0137 (0.90x) | 0.0152 → 0.0137 (0.90x) | 0.0663 → 0.0651 (0.98x) |
| real | > 0.7 | 16 | 0.0085 → 0.0077 (0.91x) | 0.0076 → 0.0066 (0.87x) | 0.0356 → 0.0332 (0.93x) |
| real | 0.3–0.7 | 9 | 0.0093 → 0.0085 (0.91x) | 0.0091 → 0.0079 (0.87x) | 0.0419 → 0.0402 (0.96x) |
| real | < 0.3 | 3 | 0.0168 → 0.0151 (0.90x) | 0.0168 → 0.0150 (0.89x) | 0.0735 → 0.0724 (0.99x) |

- **The calibration source does not matter.** On the base's own trajectories the gain
  is 0.89–0.92x, and on held-out real images it is 0.87–0.91x. Base-generated
  calibration data would not buy more.
- **The CFG velocity gains much less than either prediction.** The weight term is
  common to the cond and uncond passes (same weight, similar inputs), so it mostly
  cancels in `v_c − v_u`, the part CFG multiplies by 4. The activation rounding is
  not common, and it is what CFG amplifies. GPTQ removes the cancelling part.
- LoRA training uses the cond prediction without CFG, so the training-side gain is the
  ~10% column.

## What this means

- The GPTQ file is the better base for both uses, since it is closer to bf16 at every
  σ. The gain is small: a 2–4% cut in the CFG velocity error, which was already
  4/255 in pixels ([split_and_infer.md](split_and_infer.md)).
- Any further gain at inference has to come from the activation side: SmoothQuant
  scales, or finer activation scales. For the cross-attention `output_proj`, scales fold
  into the bf16 cross `v_proj`. For self-attention `output_proj`, they fold into the
  int8 `v` rows. The block-input linears (`qkv`, cross `q`, `mlp.layer1`) read an adaLN-
  modulated norm, so their fold point is still open. `mlp.layer2` follows a GELU and
  cannot fold.
