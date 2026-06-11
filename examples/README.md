# examples/

Runnable scripts showing the Anima programmatic API for library embedders —
the Python you write when you `import anima_lora` into your own code instead of
going through `make` targets. Each script is self-contained and runs from the
repo root (`anima_lora/`).

After `uv sync` (which installs this repo editable), the front-door package is
importable from anywhere — the curated entry points live on `anima_lora`:

```python
import anima_lora
settings = anima_lora.get_generation_settings(args)
latent = anima_lora.generate(args, settings)
image = anima_lora.decode_to_pil(vae, latent, device)
```

`anima_lora` is a thin lazy re-export of `library.inference` /
`library.config.io` / `library.anima.weights` / `library.models.qwen_vae` (see
`anima_lora/__init__.py` for the full map). Repo-relative model/config paths
resolve against the repo home, not the CWD — so `import anima_lora` works from
any directory; set `ANIMA_HOME` to point at a relocated checkout. The high-level flows
(`01`–`03`) import the curated entry points from `anima_lora`; the building-block
scripts (`04`–`06`) reach into the `library.*` homes directly, since their point
is to show the raw primitives. Either way each script keeps a `sys.path` shim so
`python examples/<script>.py` runs straight from the repo without an install.

**High-level flows** — the supported entry points:

| Script | Shows | Needs |
|---|---|---|
| [`01_generate.py`](01_generate.py) | Text-to-image: `get_generation_settings` → `generate` → `save_output`, optionally with one or more LoRA adapters attached at DiT load | DiT + VAE + text encoder (+ adapter `.safetensors` for LoRA) |
| [`02_config_and_train.py`](02_config_and_train.py) | `load_method_preset` merge chain + `create_network` (three-axis routing) + in-process training via `AnimaTrainer().train(args)` | config part: nothing; `--build-network`: DiT; `--train`: preprocessed cache |
| [`03_generate_with_correction.py`](03_generate_with_correction.py) | Training-free sampler correction (DCW / Spectrum) via the `GenerationRequest.extra_argv` escape hatch for long-tail method flags | DiT + VAE + text encoder |

**Building blocks** — the raw primitives for writing your own `scripts/` tool:

| Script | Shows | Needs |
|---|---|---|
| [`04_load_models.py`](04_load_models.py) | Load DiT / VAE / text encoder directly; encode a prompt to the DiT-ready cross-attn embedding | DiT + VAE + text encoder |
| [`05_vae_and_dataset.py`](05_vae_and_dataset.py) | VAE pixel↔latent round-trip; iterate the on-disk training cache (`CachedDataset`) | VAE (+ cache for part B) |
| [`06_frozen_dit_training_build.py`](06_frozen_dit_training_build.py) | Frozen DiT + fresh adapter build for *training* via the `harness` helpers (`place_dit_for_training` / `compile_dit_blocks` / `enable_training_grad_ckpt`) — the `scripts/distill_*` model-build sequence | DiT |
| [`07_stack_ortho_init_tlora.py`](07_stack_ortho_init_tlora.py) | Stacking LoRA-family **variants** from Python — `create_network(use_ortho_init=True, use_timestep_mask=True, …)` (kwargs → `resolve_network_spec`, no TOML) + the one per-step `apply_router_conditioning` hook; prints the live T-LoRA mask rank per step | DiT |

## Setup

```bash
uv sync
hf auth login
make download-models      # DiT, text encoder, VAE, …
# `02 --train` also needs the training cache:
make preprocess
```

Model paths default to the `configs/base.toml` locations. To point at weights
stored elsewhere, set `ANIMA_DIT` / `ANIMA_VAE` / `ANIMA_TEXT_ENCODER` — either
as real env vars (one-off override) or in a project-root `.env` file (persistent,
gitignored). Copy the template and edit:

```bash
cp .env.example .env       # then uncomment ANIMA_DIT / ANIMA_VAE / … as needed
```

Every script here resolves its paths through `default_checkpoints()`, which
loads `.env` automatically (env vars win → `.env` → `configs/base.toml` →
built-in fallbacks), so you never have to export them in your shell. Set
`ANIMA_HOME` in the same `.env` if you `import anima_lora` from another project.

## Quick start

```bash
python examples/01_generate.py --prompt "a red fox in a snowy forest"
python examples/01_generate.py --lora_weight output/ckpt/my_lora.safetensors --prompt "…"
python examples/02_config_and_train.py --method lora --preset default
python examples/02_config_and_train.py --train --max_train_epochs 8
python examples/03_generate_with_correction.py --correction dcw   # extra_argv method knobs
python examples/04_load_models.py --prompt "a lighthouse at dusk"
python examples/05_vae_and_dataset.py                       # iterate the cache
python examples/05_vae_and_dataset.py --image some/photo.png  # VAE round-trip
python examples/06_frozen_dit_training_build.py             # build a trainable adapter
python examples/07_stack_ortho_init_tlora.py --steps 3      # stack OrthoInit + T-LoRA
```

## Notes for embedders

- **`anima_lora` is the stable API; `library.*` / `networks.* `/ `scripts.*` are internal.**
  The curated `anima_lora` façade is the surface we keep stable across releases.
  The underlying trees are installed and importable for advanced use (`04`/`05`
  reach into `library.*` on purpose), but they may move or change signature
  without a deprecation cycle — pin a tag (`ANIMA_VERSION`) if you depend on them.
- **Inference is request-driven.** `01`/`03` build a typed
  `anima_lora.GenerationRequest` and call `.to_args()` — which feeds the request
  through `inference.parse_args` under the hood, so every optional knob the
  generation code reads via `getattr()` still gets a value. The long tail of
  method knobs (spectrum/dcw/ip-adapter) rides through the request's `extra_argv`,
  or you can build the `argparse.Namespace` straight from `inference.parse_args(argv)`.
- **Adapter family is in the checkpoint, not the call.** `01 --lora_weight` passes
  any LoRA / OrthoLoRA / T-LoRA / Hydra / FeRA `.safetensors`; the DiT loader reads
  the metadata and merges-or-keeps-live accordingly.
- **Variant stacking is kwargs, not config.** The toggle blocks in
  `configs/methods/lora.toml` are just one way to pick variants — the same keys
  pass straight through `create_network(**kwargs)` to `resolve_network_spec`. `07`
  stacks OrthoInit + T-LoRA that way and shows the single per-step driving hook
  (`apply_router_conditioning`). Which combos exist is the three-axis matrix in
  `networks/CLAUDE.md`; impossible combos raise at build, they don't silently
  degrade.
- **Prompt encoding uses two process-global strategy singletons.** `generate()` /
  `prepare_text_inputs()` lazily install them from `args.text_encoder` (via
  `anima_lora.ensure_text_strategies`), so the high-level flows just work; `04`
  shows the explicit one-liner. Encoding also needs the DiT — the encoder hidden
  states are projected by `Anima._preprocess_text_embeds`.
- **Multi-GPU training** must go through `accelerate launch train.py …`
  (`make lora`). `02 --train` is the single-process equivalent.
- The text-encoder padding and constant-token bucketing invariants in
  `../CLAUDE.md` apply — they're handled inside the called functions, but worth
  reading before you deviate from these flows.
