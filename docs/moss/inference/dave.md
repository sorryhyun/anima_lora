# DAVE on MOSS-Image

DAVE (`docs/anima/inference/dave.md`) adds same-prompt variety across seeds: during the first
steps it removes part of the image tokens' per-channel mean on a pool of blocks. Framing,
pose and panel layout vary more; hands and small objects get a little looser. Off by
default. Code: `library/moss/dave.py`.

## Use

```bash
make daemon-run ARGS="--stall-timeout 900 scripts/moss/generate.py --lora output/ckpt_moss/moss_lora.safetensors --spectrum --dave auto"
```

GUI: the Test box's `dave` and `dave_strength` fields, `dave_tau` under Advanced.

| Flag | Default | What it does |
|---|---|---|
| `--dave` | `''` (off) | `auto` = the shipped MOSS pool, blocks 19–30 of 42 (`library/moss/calibration/dave_alpha.npz`); `lo-hi` = blocks lo..hi; or a path to a `dave_alpha.npz` |
| `--dave_strength` | `0.3` | Fraction of the mean removed on a pooled block (0 = off, 1 = all) |
| `--dave_tau` | `0.10` | Fraction of the steps it runs on, from the first: steps `0 .. round(τ·steps)−1`, at least one. 5 steps at 50 |

The DAVE settings are part of the saved latent's fingerprint, so a resumed job does not
reuse a latent rendered with other settings.

## How it differs from Anima's DAVE

- **Image tokens only.** A MOSS block outputs `[image | text]`; the mean is taken over
  the image tokens and only they are edited. The text tokens pass through.
- **Massive channels left out.** About 99 % of the image tokens' mean sits in a few
  massive-activation channels (1356 is ~1900× the median channel at block 0, 42 next),
  the same on every seed and prompt. Removing part of them would shift the block output,
  so 18 channels (`library/moss/calibration/massive_channels.npz`) are left out of the
  mean with every pool. Derived on Base; whether SFT has the same channels is unchecked.
- **Pool 19–30, not Anima's 8–18.** Measured on Base (3 prompts × 8 seeds × 30 steps,
  strength 0.3), cross-seed similarity at the last step: vanilla 0.406, blocks 8–18 0.379,
  19–30 0.209, 8–30 0.205.
- **Skip-22 uncond.** The edit runs after each block on the rows that ran it, so at block
  22 (in the pool) only the cond pass is edited.
- **τ is a step count**, not a σ boundary.
- **Spectrum.** Forecast steps run no blocks, so DAVE lands on full steps only. At 50
  steps τ 0.10 sits inside Spectrum's 6 warmup steps; `generate.py` prints a note when τ
  reaches past them.

## What to expect

Base, 50 ER-SDE steps, `auto` · 0.3 · τ 0.10, 4 seeds of 3 prompts against vanilla: text
stays legible, bubble placement and panel framing vary where vanilla repeats one layout,
and the character's attributes hold. One seed of four turned a sunset sky blue. Hand
close-ups recompose at a visible cost (a hand half through a mug, a softer render).
τ 0.06 for hands, higher strength and SFT are unmeasured.

## ComfyUI

`ComfyUI-Anima-DAVE` 0.2.0 and later detects a MOSS-Image model: `mask=auto` loads the
same pool and massive-channel list (`moss_dave_alpha.npz`). Its default `tau` is 0.06
(3 of 50 steps), below `generate.py`'s 0.10.

Probe runs and the pool sweep: `project/finished/moss_lora/done/dave.md`.
