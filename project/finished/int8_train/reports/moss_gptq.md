# MOSS-Image offline GPTQ base, t2i and i2i (2026-10-31)

**Verdict: on held-out images a GPTQ base cuts int8's excess prediction error by 28–35%
in t2i and 31–32% in i2i, and takes the late-block gradient from 2.59x to 1.20x bf16 at
σ = 0.5.** With an fp32 stream, int8 GPTQ sits below today's bf16 prediction error at every
σ tested, at 1.47 s/step. Early blocks do not improve. Generation runs on the file at
6.3 GiB instead of 10.7; no LoRA has been trained on it and no A/B has been rendered.

## Calibration

`src/dit_gptq_moss.py`, job `20261031-005234-3a1bc3`, 64.6 min, peak 14.1 GB. 233 images
(30 each from `ama_mitsuki_layout`, `group_wsw_layout`, `hews_layout`, `sincos_layout`,
23 from `_calib_mixed`, 30 each from `i2i_sanitize_idt`, `i2i_swap_text`,
`i2i_uncensor_combined`), each at one σ of the inference schedule (50 steps, shift 5),
run as cond and as the skip-22 empty-prompt uncond: 466 samples, 1.20M t2i and 1.48M i2i
tokens per linear. i2i samples carry reference tokens at `ref_timestep` 0, isolated; in
the uncond's block 22 only those tokens run. Plain GPTQ only: act-aware added nothing on
Anima. `wq_t` stays round to nearest. `channel_(caststation)_layout` stays out of
calibration.

Output: `models/moss_image/moss_image_base_int8_gptq.safetensors` (NVMe), 5.04 GiB,
`int8_id e98d2cd810ffe731…`. Per-linear errors and the calibration stems: its
`.report.json`. Base only; SFT has no file.

Mean over the 252 linears, relative output error energy, on the calibration samples:

| | RTN | GPTQ | change |
|---|---|---|---|
| W8A16 (weight only) | 5.92e-5 | 0.75e-5 | −87% |
| W8A8, all | 12.8e-5 | 7.6e-5 | −41% |
| W8A8, t2i tokens | 13.0e-5 | 8.4e-5 | −36% |
| W8A8, i2i tokens | 12.8e-5 | 7.4e-5 | −42% |

i2i is 55% of the tokens, so the shared weights lean toward it, but t2i still gains about
what Anima did (−36%).

## Held-out prediction error

`src/grad_error_moss.py` against an fp32 forward. t2i: the six images of
[moss_grad_error.md](moss_grad_error.md) (job `20261031-015731-2b0f96`), at σ = 0.5 also 30
images (job `20261031-021944-99d1bc`). i2i: 12 uncalibrated `i2i_sanitize_idt` pairs
(job `20261031-023638-f2679a`), latent and reference centre-cropped to 48×48 so the
joint ~4.6k tokens fit: full size OOMs in the fp32 reference. Raw numbers:
[`moss_gptq_grad_error_t2i.json`](../assets/moss_gptq_grad_error_t2i.json),
[`_t2i_n30`](../assets/moss_gptq_grad_error_t2i_n30.json),
[`_i2i`](../assets/moss_gptq_grad_error_i2i.json).

`‖v − v_fp32‖ / ‖v_fp32‖`, %:

| | σ | bf16 | int8 RTN | int8 GPTQ | RTN, fp32 stream | GPTQ, fp32 stream | bf16, fp32 stream |
|---|---|---|---|---|---|---|---|
| t2i | 0.1 | 4.81 | 6.99 | 5.93 | 5.31 | **3.85** | 1.64 |
| t2i | 0.5 | 1.93 | 2.82 | 2.43 | 2.24 | **1.71** | 0.81 |
| t2i | 0.9 | 2.08 | 2.58 | 2.30 | 1.69 | **1.34** | 0.59 |
| t2i, 30 images | 0.5 | 1.92 | 2.75 | 2.38 | 2.16 | **1.65** | – |
| i2i | 0.1 | 3.61 | 5.36 | 4.50 | 4.52 | **3.52** | 2.13 |
| i2i | 0.5 | 2.30 | 3.32 | 2.84 | 2.78 | **2.17** | 1.34 |
| i2i | 0.9 | 2.05 | 2.44 | 2.24 | 1.53 | **1.19** | 0.78 |

Excess over bf16 in quadrature, RTN → GPTQ: t2i 5.06 → 3.46, 2.05 → 1.47, 1.54 → 0.99
(−32%, −28%, −35%); i2i 3.96 → 2.69, 2.40 → 1.65, 1.33 → 0.90 (−32%, −31%, −32%). The
prediction from [moss_grad_error.md](moss_grad_error.md) was about a quarter. Anima's
held-out gain was ~10%, measured from bf16 rather than as the excess over it.

## Gradient

Each image's distance from the fp32 gradient over bf16's, per group of seven blocks,
median over images [p90].

| arm | 0–6 | 7–13 | 14–20 | 21–27 | 28–34 | 35–41 |
|---|---|---|---|---|---|---|
| t2i σ 0.5 (30), RTN | 1.07 [1.89] | 1.04 [2.07] | 1.04 [2.06] | 1.03 [1.92] | 1.56 [1.64] | 2.59 [2.78] |
| t2i σ 0.5 (30), GPTQ | 1.04 [2.13] | 1.10 [2.08] | 1.08 [2.26] | 1.05 [2.36] | 1.23 [1.42] | 1.20 [1.29] |
| t2i σ 0.5 (30), GPTQ fp32 stream | 1.01 [1.48] | 1.05 [1.57] | 1.07 [1.79] | 0.98 [1.69] | 1.12 [1.38] | 0.92 [1.04] |
| i2i σ 0.5 (12), RTN | 1.02 [4.28] | 1.03 [3.09] | 1.41 [3.36] | 1.41 [3.16] | 1.47 [2.43] | 1.65 [2.73] |
| i2i σ 0.5 (12), GPTQ | 1.00 [1.56] | 0.98 [1.72] | 0.98 [1.72] | 1.07 [1.62] | 1.06 [1.65] | 1.00 [1.88] |

At σ = 0.1 the six-image t2i medians over modules go 0.52 → 0.39 of bf16's distance;
at σ = 0.9, 0.98 → 1.03. The i2i σ = 0.1 and 0.9 rows in the JSON read the same way.

- GPTQ does what the weight share predicted in late blocks, where bf16 is accurate and
  int8's excess is the weight term.
- In blocks 0–27 the medians stay at bf16's level with RTN or GPTQ. Those gradients are
  dominated by what the depth report already found there.

## Single-token gradient outliers

The six-image pooled t2i numbers at σ = 0.5 read the other way (GPTQ 1.53 against RTN
1.15, and `gptq_w8a16` 4.96, 13x in blocks 7–13). This is a tail, not a GPTQ cost: the
pooled figure is a sum of squares and a few (image, arm) pairs dominate it. Over 30
images the medians tie and the p90s are close.

`src/grad_spike.py` traced the largest one, `channel4` at σ = 0.5 (jobs
`20261031-021701-27bb18`, `20261031-021836-d89616`):

- One token's `dY` grows about 1.3x per block going back from block ~25 to 0 and holds up
  to 98% of the energy, entering through `q`. `‖dY_o‖` at block 0 is 5.5e-4 in fp32, 5.4e-4
  in bf16 and 4.2e-2 in `gptq_w8a16`; the same weights through the int8 kernel (`gptq`) give
  3.8e-4.
- Every arm has a hot token, and which token it is changes from arm to arm (761 in fp32,
  1536 and 4120 in bf16, 29 with an fp32 stream). Its share of `‖dY_o‖²` is 0.11–0.24 in
  the fp32 and fp32-stream arms, up to 0.35 in bf16 and up to 0.93 in `gptq_w8a16`.
- It is not `_RMSNorm`'s ε: the smallest per-token input RMS is 0.15, a gain of about 7.

**The mechanism is unresolved.** An fp32 stream lowers the GPTQ p90s in t2i (table above)
but not in i2i at σ = 0.5 (1.47–2.22 against 1.56–1.88), on 12 images. A step like this
would reach LoRA training as one outlier gradient; whether that matters was not measured.

## Trainer

`src/train_int8_moss.py --int8 full --int8_file <file>` checks the fingerprint and stamps
`int8_base_id`. Twelve steps on `hews_layout` with `--stream fp32`, swap 0, checkpointing
(job `20261031-025829-feb16e`): 1.47 s/step, peak 13.52 GB, the same speed as RTN on bf16
stream.

## Inference

`src/infer_int8_moss.py --int8 on --int8_file <file>` runs `library.moss.generate` on the
converted DiT, forward copies only. One render per arm, ER-SDE 50 steps + Spectrum, CFG 3,
seed 7, compiled, all blocks resident.

| job | arm | denoise | peak |
|---|---|---|---|
| t2i 1024² | bf16 | 21.6 s | 10.7 GiB |
| t2i 1024² | bf16, fp32 stream | 21.9 s | 12.3 GiB |
| t2i 1024² | int8 RTN | 15.0 s | 6.3 GiB |
| t2i 1024² | int8 GPTQ | 11.3 s | 6.3 GiB |
| t2i 1024² | int8 GPTQ, fp32 stream | 15.1 s | 7.8 GiB |
| i2i 1216×864, full-size reference, sanitize LoRA | bf16 | 29.1 s | 13.0 GiB |
| i2i 〃 | int8 GPTQ | 27.7 s | 8.6 GiB |
| i2i 〃 | int8 GPTQ, fp32 stream | 27.7 s | 11.2 GiB |

- RTN and GPTQ run the same kernels, so their 3.7 s gap is run to run, not a GPTQ gain.
- Removing the 4.5 GiB is the larger change: a full-size reference on an fp32 stream fits
  resident, where bf16 OOMs and needs `blocks_to_swap 10`.
- i2i gains little time (29.1 → 27.7 s); not profiled.
- All arms render cleanly, and the sanitize LoRA (trained on bf16) removes the page's text
  on int8. Mean |Δ| from bf16 on that page is 0.73 (GPTQ) and 0.68 (with an fp32 stream)
  out of 255.
- The t2i seed is chaotic: 12.4–13.9 from bf16 for the int8 arms, against 11.7 between
  bf16 and its own fp32 stream. One seed orders nothing, so this is a smoke test, not an A/B.

## Not done

- No LoRA trained on the file; no multi-seed render A/B.
- i2i scored on `sanitize` crops only, not `swap_text` or full-size pairs.
- The SFT DiT.
