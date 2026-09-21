# configs/ — the merge chain

Key-by-key semantics of `base.toml` for users: `docs/guidelines/base-config.md`.

`base.toml → presets.toml[<preset>] → methods/<method>.toml → CLI args`, **method
settings win over preset settings on overlap**.
`library.config.io.load_method_preset(method, preset, methods_subdir=...)` is the
reusable merge helper (also re-exported from `library.train_util`). `make print-config
METHOD=… PRESET=…` dumps the merged chain. All config paths are relative to the repo
root.

## `base.toml` — infra + the default dataset blueprint

Shared infra (model paths, optimizer, compile) **and** the default LoRA dataset blueprint
(`[general]` + `[[datasets]]` + `[[datasets.subsets]]`), consumed by
`BlueprintGenerator` and skipped by the flat method+preset merge (see
`_DATASET_CONFIG_SECTIONS`). Three ways to override the blueprint:

1. `--dataset_config` for a separate file;
2. a **scalar-only** `[general]` / `[[datasets]]` block in the method TOML to
   *shallow-override* top-level scalars (`_apply_dataset_overrides`; subset-level
   overrides are not supported this way);
3. a method TOML carrying a **full** `[[datasets]]` with `subsets`, which *fully
   replaces* base's blueprint inline (the self-contained per-method layout below).

Full-vs-shallow is decided by `load_dataset_config_from_base` on whether the method's
`[[datasets]]` has a subset.

**`base.toml` is overwritten on `make update`** — never park user-owned state in it.

## `preprocess.toml` — user-owned, preserved across updates

Preprocess knobs split out of `base.toml` (`source_image_dir`, **`target_res`**,
**`mask_dir`**). Read by the preprocess pipeline via
`load_path_overrides`, layered **`preprocess.toml → base.toml → preset → method`** —
preprocess.toml is read *first*, so a legacy copy of any of these keys still in
`base.toml` keeps winning.

`train.py` never reads the filter knobs, **but `target_res` and `mask_dir` are dual-use**:
`load_method_preset` seeds both from preprocess.toml at lowest priority (preset / method /
CLI still override).

- `target_res` is **inert at train time** (seeded for the snapshot only).
- **`mask_dir` is load-bearing.** It is the single mask-root knob for `make mask` /
  `make mask-clean`, `make preprocess-reconcile`, the GUI mask counter/overlay, the turbo
  loop, and training — where it reaches every subset that doesn't name its own `mask_dir`
  via the BlueprintGenerator argparse fallback (`--mask_dir` overrides).

Two gates keep a mask tree from enabling masking by itself:

- training gates `mask_dir` on **the directory existing** (`resolve_configured_mask_dir`),
  so a maskless checkout falls back to the legacy `masks/{merged,sam}` auto-resolution
  instead of enabling masked loss over an empty tree;
- **`masked_loss` (off by default since v2) is the one switch** — when it is off,
  `train.py` strips `mask_dir` from every subset and logs one line.

The rest of the **shared** path contract (`resized_image_dir`, `lora_cache_dir`, model
paths) stays in `base.toml` because the dataset blueprint interpolates
`{resized_image_dir}` / `{lora_cache_dir}`.

## `presets.toml` — hardware profiles

Sections `[default]`, `[low_vram]` (also Windows 8GB), `[graft]`, `[half]`, `[quarter]`,
`[tenth]`, `[debug]`. Holds `blocks_to_swap`, gradient/offload checkpointing, etc.

## `methods/` — one flat file per family

Read by `train.py` (`lora`, `chimera`, `soft_tokens`, `byg`, `register`), each holding
rank + routing knobs + opinionated LR/epochs/output_name. Variants inside `lora.toml` are
comment-toggle blocks; the default stacks LoRA + OrthoLoRA + T-LoRA + shared_A FEI-routed
Hydra (routing surface: the `lora-routing` skill).

`turbo.toml` uses a bespoke sectioned schema read only by `scripts/distill_turbo/`;
don't `print-config METHOD=turbo`.

## Self-contained per-method dir — `<method>/<method>.toml`

Method config **+** full inline dataset blueprint in one file, no `dataset_config`
cross-reference. `_resolve_method_path` (`library/config/io.py`) **prefers**
`configs/<method>/<method>.toml` over the flat `configs/methods/<method>.toml` when
present (default `methods` subdir only — `gui-methods` stays flat); `--method <m>` picks
it up with no extra flag.

EasyControl uses it: `configs/easycontrol/easycontrol.toml`, with the miner-generated
descriptor blueprints `near_twins.toml` / `colorize.toml` in the same dir.
`configs/gui-methods/easycontrol.toml` still points at the standalone
`configs/datasets/easycontrol.toml` — keep the inline subset in sync until gui-methods is
migrated.

## `gui-methods/` — clean per-variant parallel tree

One file per variant, no comment-toggle blocks. Selected via `--methods_subdir gui-methods`
(wrapped by `make lora-gui`). `ls` for the live list; custom ones live in
`gui-methods/custom/`.

**Hardware composes via preset** — the GUI's Hardware dropdown picks a
`presets.toml` section tagged `[<name>.gui] group="hardware"` (display metadata, stripped
from the merge like `[variant]`). So variant files must **NOT** pin
`gradient_checkpointing` / `unsloth_offload_checkpointing` — method wins over preset, so
pinning silently defeats the picker. Pinned by a test in `tests/test_config.py`.

Data-scope is plain flat keys (`sample_ratio`, `artists_shard` — defaults in `base.toml`;
`sample_ratio=1.0` is inert so per-subset ratios stay authoritative), surfaced as GUI form
fields; the `[half]` / `[quarter]` presets are CLI shorthand for the same key.

## Subset knobs

Subsets accept `cache_dir` — redirects all VAE/TE/PE caches to that dir with
stem-mirrored names (EasyControl uses this to keep source dirs user-facing while caches
live under `post_image_dataset/`).

Outputs split by kind: checkpoints (+ `.snapshot.toml` + `_moe` siblings) in
`output/ckpt/`, inference images in `output/tests/`.
