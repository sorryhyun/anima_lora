# pixel_decoder run configs

One file per run, named as the run; each header holds the run's settings, question and read. Kept
here: the configs behind the files on the Hub and those the open plans (`../rgba_plan.md`,
`../x1_plan.md`) read. The other finished runs' configs are in `../archive/configs/` (local,
gitignored; in git under this folder up to 7cc7cbbf). The trainers' field defaults:
`configs/pixel_head/` at the repo root.

| config | what | verdict | record |
|---|---|---|---|
| `x1dll_1123_0821` | the ×1 base on the DINO term, 12,000 rollouts at LR 5e-5 | shipped | Hub `x1/anima_pixel_decoder_base` |
| `x1dgan512_1123_1214` | GAN finish of the base, PE term on a 512 crop | shipped | Hub `x1/anima_pixel_decoder_sharp` |
| `x1fit_1124_0955` | overfit of the ×1 form on 16 fine-stroke windows | | `../reports/x1_fit_2026_11_24.md` |
| `x1read_1124_1053` | the sharp file on the same windows, no fit | | same |
| `x1sig_1124_1250` | decode-point read over σ 0 – 0.1364, two named images | | same |
| `x1sig_1124_1252` | the same on a third image | | same |
| `rgbacache_1123_2254` | the RGBA cache `pixel_alpha_r4` with every cutout's colour | | `../rgba_plan.md`, Data |
| `rgbar5_1123_2305` | R5, the first four-channel run | baseline | `../reports/rgba_r2_r6_2026_11_24.md` |
| `rgbar6_1124_0034` | R6, R5 at LR 5e-5 | shipped | Hub `x1/anima_pixel_decoder_rgba_preview` |
| `x2ovll_1124_0139` | the ×2 B head with the overlapping output, 12,000 rollouts | baseline | `../reports/x2_overlap_12k_2026_11_24.md` |
| `x2r10gf_1125_0104` | GAN finish of the B head at ratio 1.0, 4,000 rollouts | adopted on the user's read | `../reports/x2_ratio_2026_11_24.md` |
| `x2lens_1126_2225` | pattern directions at `norm2`'s output and 12 edits of `out2`, no training | not passed | `../reports/x2_lens_2026_11_26.md` |
| `x2f8_1126_2227` | the edit `f8_k4` as head files on two finishes | shipped | Hub `x2/anima_pixel_head_x2_v3` (`x2r10gf_1125_0104` + `f8_k4`) |

The first ×2 file on the Hub (`x2/anima_pixel_head_x2`, the catalog's row) is `x2bgan2_1121_1340`; it
has no config file here or in the archive, and its record is
`../reports/x2_full_run_deploy_2026_11_21.md`.
