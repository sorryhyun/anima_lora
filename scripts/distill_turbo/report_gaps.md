# Pooled moment gaps: dataset, CFG 1, CFG 4, turbo student

Measured 2026-10-31 on the int8 base, as a follow-up to reading Liu et al., *Intrinsic
Distribution Matching* (IDM, arXiv:2610.39633). The question was how far a dataset reference
sits from the CFG 4 teacher the student targets, before any feature-moment loss (`L_pop`,
Eq. 13) is considered here.

Script: `bench/turbo/moment_gap.py`. Run:
`bench/turbo/results/20261031-1834-moment-gap-int8/` (job `20261031-183425-cc4523`, 55 min,
eager, ~22 s/item).

## Setup

128 items drawn uniformly from `post_image_dataset/lora`. All five sets share each item's
resolution and start noise:

| Set | What |
|---|---|
| D | the cached dataset latent |
| T1 | teacher, CFG 1, 28 Euler steps, dataset caption |
| T4 | teacher, CFG 4, 28 Euler steps, dataset caption |
| T1na | T1 with the artist tag removed from the caption |
| S | the v1.1 delta `anima_turbo_v1.1_delta_r96_asvd_adaln`, 4 steps, CFG 1, on the int8 base as a P-GRAFT hook |

Every set goes through one feature map: the teacher's block 27, mean-pooled, on the item's
caption and a fixed noise per item, at τ 0.8 and 0.5. A gap between two sets therefore comes
from the images. Latent channel means and decoded pixels are read the same way.

`nW2²` below is `‖μ_a − μ_b‖² + ‖σ_a − σ_b‖²` over the pooled per-coordinate moments, divided
by D's summed per-coordinate variance. `null p95` swaps a and b per item at random; a pair
below it is not separable at this N. The bootstrap ranges in `result.json` run high against
the point estimate, so the null is the floor to read.

## Results

| Pair | nW2² τ 0.8 | nW2² τ 0.5 | null p95 τ 0.8 / 0.5 |
|---|---|---|---|
| T1–T4 (CFG) | 0.186 | 0.189 | 0.011 / 0.011 |
| D–T1 | 0.078 | 0.156 | 0.015 / 0.021 |
| D–T4 | 0.048 | 0.026 | 0.013 / 0.014 |
| T1–T1na (artist tag) | 0.022 | 0.010 | 0.009 / 0.009 |
| S–T4 | 0.032 | 0.060 | 0.006 / 0.009 |
| S–D | 0.088 | 0.053 | 0.012 / 0.013 |

Decoded pixels, set means:

| | T1 | D | T4 | S |
|---|---|---|---|---|
| Saturation | 0.179 | 0.198 | 0.212 | 0.207 |
| Luma std | 0.204 | 0.230 | 0.250 | 0.250 |
| Luma p99 − p1 | 0.747 | 0.821 | 0.858 | 0.860 |

Pooled spread against T4, `sqrt(Σvar_set / Σvar_T4)`, with the 90% bootstrap range for S:

| Space | D | T1 | S | S range |
|---|---|---|---|---|
| τ 0.8 features | 1.013 | 0.993 | 0.885 | 0.86–0.91 |
| τ 0.5 features | 0.964 | 0.923 | 0.858 | 0.82–0.89 |
| Latent channel means | 0.960 | 0.945 | 0.798 | 0.76–0.83 |

Cosine from S between "toward D" and "toward T4":

| Space | Mean | Std |
|---|---|---|
| τ 0.8 features | 0.67 | 0.80 |
| τ 0.5 features | 0.74 | 0.85 |
| Latent channel means | 0.69 | 0.97 |

## Reads

- **The CFG gap is large and sits in the mean.** T1–T4 is about 17× the null at both τ, and
  its std part is 2–5% of the total.
- **The dataset sits nearer CFG 4 than CFG 1.** D–T4 is smaller than D–T1 by 1.6× at τ 0.8 and
  6× at τ 0.5. In pixels D lies about 60% of the way from T1 to T4. The expected picture, where
  the data matches CFG 1 and CFG 4 is the outlier, does not hold: base Anima at CFG 1 and 28
  Euler steps renders washed out.
- **A dataset reference is usable but not neutral.** Moving S toward D mostly moves it toward
  T4 (cos 0.67–0.97), and D's saturation and contrast are slightly below T4's (paired
  saturation −0.014, se 0.006), so it would pull a little against the baked guidance.
- **The artist tag moves the teacher little.** T1–T1na is 1/9 to 1/18 of the CFG gap, and at
  τ 0.5 it reaches only the null.
- **The student is under-dispersed.** S has 11–14% less pooled feature spread than T4 and 20%
  less in latent channel means. 89–93% of feature coordinates have a smaller std than T4. D,
  T1 and T1na all stay near T4 at τ 0.8. S's gap to T4 is small in the mean and carries a large
  share in the std (0.018 of 0.032 at τ 0.8), which is the term `L_pop` constrains explicitly.

## Limits

- S is the v1.1 delta, the official release the docs describe as mode-collapsed, not a DP-DMD
  student. The spread deficit may not carry over.
- The spread is across prompts. It says nothing about seed-to-seed pose diversity under one
  prompt.
- T1na is a proxy for the dataset's narrowness. It removes the artist tag only, so the content
  range of the 3.2k images is untouched, and Anima's training distribution is not measured.
- Everything ran on the int8 base. The delta was extracted on bf16, and no bf16 run backs these
  numbers.
- 128 of about 3.2k items, one seed per item.

## Consequences

- `L_pop`, if tried, should take T4 samples as its reference. A dataset reference would not be
  ruinous, but T4 is the target and costs no argument.
- Before spending an arm on `L_pop`, read the same spread ratio on a DP-DMD student. If it stays
  well below 1, the std term has a concrete job; if it closes, the case for `L_pop` is back to
  the paper's +0.35 GenEval points.
- `idea.md` ideas 1 and 2 stay ahead of this.
- The next run of the script should enable `compile_dynamic_seq`; this one ran eager by
  oversight, which affects speed only.
