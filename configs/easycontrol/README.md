# configs/easycontrol/

This directory holds two kinds of file:

- **`easycontrol.toml`** — the **shipped, self-contained method config**
  (method settings + inline dataset blueprint, no `dataset_config` reference).
  Auto-discovered by `--method easycontrol` / `make easycontrol` via
  `_resolve_method_path` (the per-method-dir layout takes precedence over the
  flat `configs/methods/` folder). This is the EasyControl pilot for the
  consolidated per-method layout — see `configs/CLAUDE.md`.
- **EasyControl descriptors** (`near_twins.toml`, `colorize.toml`, …) — the
  EasyControl half of an i2i task (`[preprocess]` / `[training]` knobs, the
  miner-written blueprint tail, the GUI `[variant]` block), selected via
  `EASYADAPTER=<task>`. The model-agnostic half (`name`, `stage`, `[staging]`)
  is `i2i_tasks/configs/<task>.toml`; see `i2i_tasks/README.md`.

The `gui-methods` variant (`configs/gui-methods/easycontrol.toml`) still points
at the standalone `configs/datasets/easycontrol.toml` blueprint; keep the inline
subset in `easycontrol.toml` in sync with it until gui-methods is migrated.

| File | Producer | Points at |
|---|---|---|
| `near_twins.toml` | `python -m i2i_tasks.tools.near_twins` | the materialized `_tags`/`_no_tags` near-twin pair tree under `post_image_dataset/i2i/near_twins/staging/` (VAE/TE caches land beside it under `near_twins/cache/`) |
| `subject.toml` | `python -m i2i_tasks.tools.subject_pairs` | cross-image same-character pairs (directedit_ec Phase 2): `subject/staging/` target symlinks + `pairs.json` + `subject/cond/`; **no encode pass** — target latents/TE reused from the shared LoRA cache (`cache_dir = post_image_dataset/lora`), staging + preprocess are both CPU-only symlink steps |

`near_twins.toml` carries two hand-edited tables above the generated
blueprint: `[preprocess]` (VAE/TE caching knobs, read by
`make easycontrol-preprocess EASYADAPTER=near_twins`) and the optional
`[training]` table (folded into `--key value` CLI overrides by
`make easycontrol EASYADAPTER=near_twins`). Both survive re-runs of the miner
verbatim; only the blueprint tail (below the sentinel) is rewritten. The mining
knobs (`name` + `[staging]`) are in `i2i_tasks/configs/near_twins.toml`, which the
miner reads via `--config` and writes this file via `--config-out`.

Train on the mined tree with:

```bash
make i2i-stage TASK=near_twins   # mine pair tree
make easycontrol-preprocess EASYADAPTER=near_twins   # VAE/TE caches
make easycontrol            EASYADAPTER=near_twins   # train (easycontrol method
                                                    #   + near_twins blueprint
                                                    #   + [training] overrides)
```

The training run uses the shipped `configs/easycontrol/easycontrol.toml` method.
train.py's dataset-config validator only accepts `[general]`/`[[datasets]]`, so
the train step first extracts the blueprint into a generated sidecar
(`post_image_dataset/i2i/near_twins/dataset_config.toml`, regenerated
each run) and points `--dataset_config` there; keys in `[training]` override the
method defaults (CLI wins the merge chain).

**It's a text-removal control task.** The generated blueprint pairs each twin via
`cond_cache_dir` (EasyControl roles: `cache_dir` = denoising target,
`cond_cache_dir` = `set_cond` reference):

- **target** = the clean `_no_tags` member (+ its caption) — what the model
  generates. `path_pattern = '*_no_tags.*'` keeps only these as targets.
- **cond** = the paired `_tags` latent (the text-bearing reference), symlinked
  into `cond/` under the `_no_tags` stem by the preprocess step (step 4 of
  `make easycontrol-preprocess EASYADAPTER=near_twins` — `_near_twins_build_cond`).
  Same-bucket twins only; a member that bucketed to a different resolution is
  skipped with a warning.

So the adapter learns: given a text-bearing panel as the reference + a clean
caption, regenerate the clean version. At inference, feed a `_tags`-style page as
the EasyControl condition. The `cond/` tree is pure symlinks over `cache/` and is
rebuilt from scratch on every preprocess run.

These start as **seed / eval** pair sets: each accepted pair is a `{id}_tags` /
`{id}_no_tags` couple (the `_tags` side holds the discriminator attribute). See
`i2i_tasks/tools/near_twins/README.md`.

Source images stay user-facing; VAE/TE/PE caches land under
`post_image_dataset/` via the subset `cache_dir` (the IP-Adapter / EasyControl
pattern). Re-running the miner regenerates the matching `.toml` here.
