# comfyui-anima-trainer

ComfyUI custom node that trains an Anima LoRA and immediately applies it to a `MODEL`
downstream, so a single workflow run covers **curate → train → generate**.

Complements `custom_nodes/comfyui-hydralora/` (which only loads already-trained
adapters). The trainer imports `AnimaTrainer` from the anima_lora workspace
and calls it in-process — ComfyUI must be run from a Python env that has
anima_lora's dependencies installed. The `__init__.py` adds
`<anima_lora_root>` to `sys.path` automatically.

## Nodes

### Anima LoRA Trainer (simple)

Inputs:
- `model` — DiT MODEL from your existing Anima loader.
- `train` — on: run training. off: pure adapter-loader.
- `rank` — `16` or `32`. lr is set per rank: r=16 → 1e-4, r=32 → 5e-5.
- `gpu` — hardware tier:
  - `8GB`  → `[low_vram]` preset (gradient checkpointing + unsloth offload)
  - `16GB` → `[default]` preset + `blocks_to_swap=12`
  - `high` → `[32gb]` preset (no swap, no checkpointing)
- `dataset_dir` — directory of images with `.txt` caption sidecars.
- `prompt` — caption used when an IMAGE input is connected (single-image mode).
- `use_adapter` / `adapter` / `strength` — when `train` is off, optionally apply
  an existing adapter to the MODEL.
- `image` (optional socket) — IMAGE tensor. When connected, switches to
  single-image mode; each frame is written to a temp dir with the prompt.

Locks the method to `configs/gui-methods/tlora.toml` (T-LoRA + OrthoLoRA,
no ReFT). Trains for 25 epochs at batch_size=1. Saves to
`<ComfyUI>/models/loras/anima_trainer_<timestamp>.safetensors`.

### Anima LoRA Trainer (Advanced)

Same flow plus:
- `method_variant` — any stem under `configs/gui-methods/`.
- `preset` — any section in `configs/presets.toml`.
- `learning_rate` / `max_train_epochs` / `network_dim` / `blocks_to_swap` —
  zero/negative sentinels mean "use the method/preset default".
- `warm_start` + `warm_start_adapter` — sets `network_weights=<path>` and
  `dim_from_weights=true` for resuming from a prior LoRA.

## Notes

- **Training is long** (minutes to tens of minutes). The ComfyUI worker blocks
  until it finishes. Watch the console for per-step logs.
- **Memory** — the node calls `comfy.model_management.unload_all_models()` before
  training so there is room for the trainer's own DiT + optimizer state.
  The input MODEL gets re-instantiated after training and patched with the
  freshly-saved LoRA before being returned.
- **Single-image mode** writes PNGs + `.txt` under
  `<anima_lora_root>/output/tmp_trainer/<timestamp>/`. Prune periodically.
- **Plain LoRA output** — tlora + ortholora saves as pure LoRA (SVD collapse at
  save time), so the output is usable by any ComfyUI LoRA loader as well. Paths
  that enable ReFT (Advanced → `method_variant=tlora_ortho_reft`) produce
  safetensors that need the sibling `comfyui-hydralora` node for inference.

## Out of scope (v1)

- Per-step progress bar via `comfy.utils.ProgressBar` (needs a
  `progress_callback=` kwarg on `AnimaTrainer.train`).
- Baking the trained LoRA into DiT weights as a standalone checkpoint
  (see `scripts/merge_to_dit.py` for the CLI equivalent).
- GRAFT-style iterative curation loops.
- Postfix / prefix training (separate config family).
