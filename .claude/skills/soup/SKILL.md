---
name: soup
description: Run or modify the uncond-init soup pipeline (make soup, scripts/soup/, configs/soup/soup.toml) — selection globs, uncond-init reuse and naming, env knobs, sigma_lowres interaction, and constraints. Load before running a soup, changing its knobs, or editing scripts/soup/.
---

# Uncond-init soup pipeline

`make soup` runs the uncond-init soup pipeline (`scripts/soup/`; GUI: Experimental tab →
soup). Recipe from `bench/memorization/report.md`; deep-dive: `docs/experimental/soup.md`.

## What it does

Three stages, submitted as one daemon command job (attach-by-default; `--queue` detaches,
`--inline` bypasses the daemon):

1. **Uncond inter-train** — a short `caption_dropout_rate 1.0` train on a **diluted pool**: `POOL_PATH_PATTERN` (default `*` = whole dataset), diluted by `--sample_ratio`, with the fine-tune set always unioned in. The checkpoint name is deterministic and the checkpoint is **reused if it exists** — one shared uncond init serves every soup drawing the same pool.
2. **Fine-tunes** — 3 seeded fine-tunes on `PATH_PATTERN` starting from that init.
3. **Soup** — exact ΔW average, SVD-truncated back to `network_dim` → `anima_soup_<slug>.safetensors` + snapshot (slug derived from the pattern or `NAME`).

## Selection

`make soup PATH_PATTERN="<glob>"`, or the `TARGET=<dir>` shorthand ⇒
`PATH_PATTERN="<dir>/*" NAME=<dir>`. **Selection is an fnmatch `path_pattern`, not an
artist dir** (`|` = alternatives) — a soup can span any glob slice.

## Knobs

env: `NAME`, `POOL_PATH_PATTERN`, `UNCOND_RATIO`, `UNCOND_EPOCHS`, `UNCOND_INIT`,
`NUM_SOUP`, `RANK`, `ARTISTS_SHARD`, `CUSTOM`, plus `LR_POOL`/`LR_INTERVAL` for opt-in
per-ingredient LR diversity. By default ingredients vary by **seed only**; the soup is a
uniform average with **no greedy-selection gate**, so a bad LR draw is averaged in, not
dropped.

`NO_UNCOND=1` / `[soup] no_uncond = true` skips Phase 1 — ingredients init from
`down_init` (`weight_svd` [+ top-level `svd_slice`]); **required** for `svd_slice != 0`
(the uncond `--network_weights` warm-start would overwrite the slice; the pipeline
refuses). All ingredients of one soup share one slice; the slug gets
`_nouncond[_k<slice>]`.

`ARGS` reaches the fine-tunes — **except `--sigma_lowres*`**, a whole-pipeline routing
knob that is also replayed onto the uncond run and folded into its name as an
`_sl<digest>` tag, so a σ-demoted soup can't silently reuse a natively-trained init (and
vice versa). Names are unchanged when σ-demote is off.

## Config

Ingredients train `--method soup` → `configs/soup/soup.toml`, a dedicated plain-LoRA
stack. Its `[soup]` table holds the pipeline knob defaults (incl. `pool_path_pattern`)
and is stripped from the train.py merge.

## Constraints

- Plain-LoRA checkpoints only — hydra/chimera refused.
- Quality win + seed-lottery insurance, **NOT a memorization fix**.
