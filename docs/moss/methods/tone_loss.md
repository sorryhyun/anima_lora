# i2i tone term on MOSS-Image

An i2i LoRA whose output should keep the input's look outside the edit (sanitize,
swap_text) can still drift the whole image's tone: Muon sanitize runs brightened the
untouched page by up to +2 L and cooled it by up to −1 b, though every training pair is
tone-neutral. The flow-matching MSE treats that offset as one more small residual. The tone
term adds an explicit penalty on it. Off by default. Code: `tone_loss` in
`library/moss/train.py`.

## Use

```bash
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --config configs/moss/i2i/sanitize/train.toml"
```

The shipped config sets `tone_weight 0.003`, `tone_start 0.5`.

The shipped sanitize v3 (`sorryhyun/MOSS-I2I-LORAs`, `sanitize/moss_i2i_sanitize_v3`) is
this setting on the 277-pair identity tree (`post_image_dataset/moss/i2i_sanitize_idt`:
the near-twin pairs plus 39 clean → clean pairs), Muon lr 5e-5, 8 epochs = 2,216 steps,
`ema_window 0.1`. The settings are stamped in the LoRA's metadata.

| Flag | Default | What it does |
|---|---|---|
| `--tone_weight` | `0` (off) | Weight of the term. The shipped value is 0.003 |
| `--tone_start` | `0` | Fraction of the run before which the term is only logged, not added. The shipped value is 0.5 |
| `--tone_pool` | `8` | Cell side in latent tokens (8 = 128 px); `0` = one cell over the whole unedited area |
| `--tone_dilate` | `2` | Tokens the edit region grows by for the term, beyond `region_dilate`, so seams stay out |
| `--tone_dirs` | `library/moss/calibration/tone_dirs.safetensors` | Latent → Lab projection; `''` = the raw 128 channels |
| `--tone_b_weight` / `--tone_b_start` | `1` / `0` | Scale on the b row of the projection, and when it starts |
| `--tone_log` | off | Compute and log the term without adding it (to size the weight) |

It needs a full-size reference (`ref_scale 1`) and pixel-aligned pairs: the unedited
tokens are found by comparing the target latent with the reference (`edit_region`, the
same mask the region loss uses). The run stops at start if any pair's reference is not
the latent's size; a pair whose edit covers the whole image gets no term.

## The term

With $`x_t = (1-\sigma)x_0 + \sigma\epsilon`$ and $`v = \epsilon - x_0`$, the prediction's
clean estimate is $`\hat{x}_0 = x_t - \sigma\hat{v}`$, so $`\hat{x}_0 - x_0 = \sigma(v - \hat{v})`$.
On the rest tokens $`R`$ (outside the dilated edit region), with $`P_k`$ a mean pool over
k×k token cells and $`D`$ the `tone_dirs` projection:

```math
\begin{aligned}
e &= (\hat{x}_0 - x_0) \odot \mathbf{1}_R
  && \text{(1, 128, h, w), packed latent} \\
c &= P_k(e) \,/\, P_k(\mathbf{1}_R)
  && \text{masked mean per cell, coverage} \ge \tfrac{1}{2} \\
(\Delta L, \Delta a, \Delta b) &= D\,c
  && \text{per cell} \\
\mathcal{L} &= \mathcal{L}_\text{fm} + w_\text{tone} \cdot \operatorname{mean}_\text{cells} \lVert D\,c \rVert^2
\end{aligned}
```

- It reads $`\hat{x}_0`$, so it penalises a tone error in the image, and the σ factor
  down-weights high σ, where $`\hat{x}_0`$ is mostly a guess.
- Pooling keeps it blind to texture: only each cell's mean colour counts.
- `tone_dirs` is each Lab axis's per-channel response in the packed latent, measured by
  encoding pages shifted in L, a and b (`project/finished/moss_lora/src/tone_dirs.py`). With it the
  term is ΔE² in Lab units.

## Why late and at 0.003

- **From step 0** (`tone_weight 0.005`, lr 5e-5) it fixes L (ΔL +0.02) but leaves b
  (Δb −0.63): the single-step b error is 4–25× less gradient than the L error.
- **Late and weak** (`tone_start 0.5`, 0.001) passes at 600 steps but not over a full
  run (Δb −0.87 at 2,216 steps): the logged term stalls at ~2.4 once on.
- **Late at 0.003** passes on L and b. The logged term keeps falling to the level it had
  before switching on, and the flow-matching loss stays within 2–5 % of the run without
  it at equal steps.
- **Scaling b instead** (`tone_b_weight 3`) overshoots to warm (Δb +1.3).

Five held-out pages × seeds 42 / 43 / 44, final EMA, Lab on the pixels no run changed:

| run | ΔL | Δb | median ΔE | pages past \|ΔL\| 1 |
|---|---|---|---|---|
| no term (`idt_cs0_600`, seed 42) | +0.84 | −1.10 | 1.77 | — |
| 0.005 from step 0 | +0.03 | −0.63 | 1.19 | 0 |
| 0.001 late | −0.39 | −0.87 | 1.42 | 3 |
| **0.003 late (v3)** | **−0.03** | **−0.23** | **1.07** | 1 (ΔL −1.07) |

Edit quality matches the 0.005 run by eye: caption boxes, SFX, tiled and stamped
watermarks and handles all go, and non-text art (hearts, a halo's symbols) stays.

## Limits

- Δa ≈ +0.4 (slightly redder) is the same with and without the term in every run.
- One dark, high-contrast page (hard_to_sanitize2) still loses up to 1 L at one seed.

Research log (measurements, the hypotheses ruled out — CNS, CFG, identity pairs, tone
jitter, the small VAE decoder — and every arm): `project/finished/moss_lora/done/i2i_tone_loss.md`.
