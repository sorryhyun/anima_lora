# SmoothQuant fold from `o` into `v`, MOSS-Image (2026-10-31)

**Verdict: no gain, so no file was built.** Folding SmoothQuant scales from `o`'s columns
into `v`'s row scales raises `o`'s mean W8A8 error at every α tried. ConvRot has already
flattened the input: the scales leave its crest near 4.6 and enlarge `‖W·S‖`, which the
activation noise scales with.

`src/dit_fold_vo_moss.py --probe 7 --per_set 8 --alphas 0.25,0.5,1`, job
`20261031-141623-406cb8`, 4 min, peak 11.0 GB. 64 images, 128 samples, a subset of the GPTQ
base's picks at its seed. Blocks 0, 7, …, 35 are scored; the stream stays on the file's
copies. Errors are in-sample, on `o`'s W8A8 inputs. `none` is the same fit with the fold off.

## Results

Mean over the six blocks, relative output error energy, ×1e-5:

| arm | W8A8 RTN | W8A8 GPTQ | t2i | i2i | W8A16 GPTQ | crest |
|---|---|---|---|---|---|---|
| `none` | 18.12 | **11.18** | 15.44 | 9.80 | 0.69 | 4.60 |
| α 0.25 | 19.31 | 12.16 | 16.25 | 10.82 | 0.71 | 4.70 |
| α 0.5 | 20.36 | 12.11 | 15.94 | 10.88 | 0.82 | 4.55 |
| α 1 | 42.66 | 20.89 | 26.69 | 18.96 | 1.80 | 4.44 |

W8A8 GPTQ per block, ×1e-5:

| block | `none` | α 0.25 | α 0.5 | α 1 |
|---|---|---|---|---|
| 0 | 1.49 | 1.91 | 1.74 | 5.11 |
| 7 | 8.22 | 11.31 | 14.47 | 44.58 |
| 14 | 9.06 | 10.47 | 9.15 | 12.16 |
| 21 | 13.41 | 14.92 | 14.09 | 20.33 |
| 28 | 18.25 | 17.33 | 16.93 | 21.01 |
| 35 | 16.63 | 17.03 | 16.32 | 22.13 |

Crest is the rms per-token amax of the rotated input over its rms.

- The best case is block 28 at α 0.5, −7%. Blocks 0 and 7 lose 16% and 76% at the same α.
- A per-block α would take the wins alone, and `o` is about 30% of the mean activation term
  over the six linears. That is under 1% of that term.
- Even α 1 only moves crest from 4.60 to 4.44, about 7% of the step's energy, and the error
  doubles.

## What this means

- The activation step follows the per-token amax in the rotated basis. Channel scaling does
  not reach it, because the rotation has already mixed the channels. The earlier estimate of
  halving `o`'s noise had no such mechanism behind it.
- The fold's other costs (`wq_t`, LoRA adapters, ComfyUI) are not worth paying, and neither
  is a runtime diagonal for the other linears, which has the same limit.
- Clipping the per-token amax does not work either; see [moss_act_scale.md](moss_act_scale.md).
