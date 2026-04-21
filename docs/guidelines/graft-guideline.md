# GRAFT Step Guideline

## Quick Start

```bash
make graft-step    # train + generate candidates
# review candidates, delete bad ones
make graft-step    # ingest survivors, retrain, generate new candidates
# repeat
```

## What Happens Each Step

1. **Holdout**: 20% of `image_dataset/` images are randomly held out (their captions become generation prompts)
2. **Train**: LoRA trains for 2 epochs on the remaining 80% + any previously survived candidates
3. **Generate**: 4 candidate images per held-out caption, saved to `graft/candidates/iter_NNN/`
4. **Review**: You delete bad candidates, then run `make graft-step` again
5. **Ingest**: Surviving candidates move to `graft/survivors/` and join the training set

## How Many Candidates to Keep

The GRAFT paper uses rejection sampling -- the selectivity of your filtering IS the reward signal.

- **Keep ~25-50%** of candidates (strict but not too sparse)
  - Too lenient (keeping 90%+): weak signal, model barely improves
  - Too strict (keeping <10%): too few survivors to influence training
  - Sweet spot: keep the ones that look genuinely good, delete the rest
- **Quality over quantity**: 3 great images beat 10 mediocre ones
- **It's OK to keep 0**: if an iteration produces nothing good, just delete everything and `make graft-step` — the model retrains on original data and tries again with new seeds

## What to Look For When Curating

- **Keep**: correct anatomy, good composition, matches the caption intent, pleasing style
- **Delete**: distorted faces/hands, artifacts, wrong subject, off-topic compositions
- Trust your gut -- you're the reward function

## Directory Layout

```
graft/
  graft_config.toml     # tune epochs, sample ratio, candidate count, etc.
  state.json            # auto-managed (don't edit)
  candidates/iter_NNN/  # current round's candidates (review here)
  survivors/            # accumulated good generations (auto-managed)
  train_images/         # symlinks to image_dataset minus holdout (auto-managed)
```

## Configuration (`graft/graft_config.toml`)

| Key | Default | What it does |
|-----|---------|--------------|
| `epochs_per_step` | 2 | Training epochs per iteration |
| `candidates_per_prompt` | 4 | Images generated per held-out caption |
| `pgraft_sample_ratio` | 0.2 | Fraction of dataset held out for generation |
| `pgraft_enabled` | true | P-GRAFT: LoRA disabled for last 25% of denoising |
| `pgraft_cutoff_ratio` | 0.75 | When to switch from fine-tuned to reference model |
| `inference_steps` | 50 | Denoising steps for generation |
| `guidance_scale` | 3.5 | CFG scale |

## P-GRAFT

When `pgraft_enabled = true`, candidate generation uses LoRA for the first 75% of denoising steps, then switches to the base model for the last 25%. This produces images that reflect the fine-tuned style but stay closer to the base model's distribution -- more stable and less prone to overfitting artifacts.

## Resetting

To start fresh, delete `graft/state.json` and `graft/survivors/`. The next `make graft-step` starts from iteration 0.
