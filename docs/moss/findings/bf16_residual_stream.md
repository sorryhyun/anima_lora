# The bf16 residual stream is most of MOSS's bf16 error

Measured 2026-10-30 on MOSS-Image Base and SFT. The measurements come from the int8 line:
[`project/int8_train/reports/moss_grad_error.md`](../../../project/int8_train/reports/moss_grad_error.md)
(script `project/int8_train/src/grad_error_moss.py`). The renders come from
`bench/moss_ab/lora_ab.py --stream fp32`.

## Setting

MOSS runs bf16 end to end. The DiT weights are bf16 and nothing runs under autocast, so
the residual stream `x`, the modulation, the gates and the norm outputs are all bf16.
Each block's `x = x + gate·proj(...)` rounds to an 8-bit mantissa, twice per block over
42 blocks. The upstream inference (`inference_runtime.py`, `torch_dtype` bf16) and our
`generate` / trainer do the same. The technical report does not say how pretraining
handled precision. The code comments ("fp32 weights + autocast") suggest bf16
activations there as well, since linear outputs are bf16 under autocast.

The fp32-stream arm (`bench/moss_ab/_render.py::stream_fp32_`) casts every parameter
except the 252 block linears' weights to fp32. The stream, modulation, gates, norms,
embedders and head then run in fp32. The block GEMMs stay bf16, and so do FA2 and any
LoRA adapter, which computes in its base output's dtype.

## Observations

**Prediction error.** Six cached images at σ ∈ {0.1, 0.5, 0.9}, one fixed ε each,
scored as ‖v − v_fp32‖ / ‖v_fp32‖ against a true fp32 forward (all GEMMs fp32, TF32 off):

| σ | bf16 | fp32 stream | loss bf16 / fp32 stream / fp32 |
|---|---|---|---|
| 0.1 | 4.81% | 1.64% | 0.5635 / 0.5612 / 0.5609 |
| 0.5 | 1.93% | 0.81% | 0.2253 / 0.2249 / 0.2249 |
| 0.9 | 2.08% | 0.59% | 0.3819 / 0.3814 / 0.3814 |

- The fp32 stream cuts the prediction error 2.4–3.5x.
- The loss matches fp32 to the fourth decimal.
- For scale: int8 W8A8 linears add about 2% at σ 0.5.

**Training gradient.** The LoRA gradient of blocks 28–41 moves 1.5–3x closer to fp32.
That of blocks 0–27 barely moves (0.75–1.37x bf16's distance at σ 0.5 and 0.9). Those
gradients still pass back through every later block's bf16 dX GEMMs and the FA2
backward. Turning off cuBLAS reduced-precision reduction is not a substitute: it helps
at σ 0.1, does nothing at 0.5 and is worse at 0.9.

**Renders.** The usual A/B ran on `ama_mitsuki_muon_ema`: 3 prompts × seeds 1, 2, Base
and SFT, LoRA ×1.0 and ×0.0, ER-SDE 50 steps + Spectrum, CFG 3, int8 TE. The cached
bf16 latents carry the same settings digest. Mean |Δ| per cell, bf16 against fp32
stream, out of 255:

| | LoRA ×1.0 | LoRA ×0.0 |
|---|---|---|
| Base | 1.9–6.0 (mean 4.3) | 5.0–16.4 (mean 9.5) |
| SFT | 2.8–31.0 (mean 12.1) | 3.5–14.0 (mean 7.3) |

- Base keeps the composition and changes the details. These include accessory designs,
  eyes, the "01" shoulder tattoo, and the Akita Neru hair tie with the LoRA.
- SFT can change the composition. On the sailor-uniform prompt, seed 1, the pose, bag
  and ribbon colour all change (31/255).
- The stretched near-white background of Base p0 seed 2 (LoRA ×1.0) shows faint grey
  blotches in bf16 that the fp32 stream does not have. Corner mean is 254.68 against
  254.98.
- Across the 24 cells, neither arm wins consistently. Both are clean images.

Grids: `output/tests/ab_ama_mitsuki_muon_ema_50st{,_fp32s}/`.

**Cost.** At LoRA ×0.0, 1024², 50 steps, Spectrum and compiled blocks, the fp32 stream
takes 20.8 s per image against bf16's 20.5 s (+1.5%). The extra memory is about 0.6 GB
for the fp32 non-block parameters. In training, the checkpoint boundaries would add
about 1.2 GB at 4.5k tokens (not run).

## Reading

- bf16 rounding in the stream is not small next to the model's own output. It moves a
  render by 4–12/255 on average and sometimes changes the composition. A bf16 sample is
  one draw of that rounding noise. On Anima, an int8 base moved samples 4/255.
- Closer to fp32 is not shown to be better. If pretraining ran a bf16 stream, the
  network learned with that rounding. The renders show detail changes, not a quality
  shift, apart from the one background case.
- At +1.5% inference time it is cheap to keep as an option. Making it the default
  needs a blind preference pass over more prompts.

## Training with an fp32 stream

`ama_mitsuki_muon_ema_fp32s` is `ama_mitsuki_muon_ema` retrained with `--stream fp32`,
same request and seed. Its 848 σ draws and sample order match the bf16 run exactly.
The memory plan moved from swap 36 / budget 0.5 to swap 38 / budget 0.45, because the
first run OOMed at the worst-case sample (4200 + 340 tokens) with 72 MB short.
Those two settings change speed, not numerics.

| | bf16 | fp32 stream |
|---|---|---|
| s/step | 1.80 | 1.85 |
| peak reserved | 13.73 GB | 13.98 GB |
| mean loss, all steps / last epoch | 0.2832 / 0.2833 | 0.2829 / 0.2829 |

- The loss is 0.0002–0.0004 lower throughout, including step 1 (−0.0003), before any
  update. It is the forward's own error, not better learning.
- The renders form a 2×2 of train stream × inference stream. They use the A/B above at
  LoRA ×1.0; grids are in `output/tests/ab_ama_mitsuki_muon_ema{,_fp32s}_50st{,_fp32s}/`.
  Mean |Δ| out of 255:

| changed | Base | SFT |
|---|---|---|
| inference stream, bf16-trained LoRA | 4.3 | 12.1 |
| inference stream, fp32-trained LoRA | 5.5 | 8.9 |
| training stream, bf16 inference | 14.6 | 26.6 |
| training stream, fp32 inference | 14.8 | 24.2 |

- Changing the training stream moves a render about 3x more than changing the inference
  stream. The LoRA itself changes. Matching the inference stream to the training one
  shows no extra effect.
- The style is learned equally in both. On the 8 cells with visible hands, cropped,
  the fp32-trained LoRA looked better in 4, tied in 3 and worse in 1. Fewer fused or
  smeared fingers.
- This is one run per arm, with no bf16 retrain to show the run-to-run spread, and the
  judging was not blind. The hand lean is a lead, not a result.

## Not tested

- int8 linears with an fp32 stream. In quadrature this lands near today's bf16
  prediction error.
- A bf16 retrain of the same request, for the run-to-run spread the training-stream
  difference has to beat; then a blind pass on hands over more seeds.
- i2i tasks that hinge on fine detail (`swap_text`, `sanitize`). There the training
  stream may matter more than on a style LoRA.
- Text rendering (the `bench/_moss.py` zh / ja / ko prompts), where small errors could
  show first.
