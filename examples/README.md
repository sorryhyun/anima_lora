# examples/

Runnable scripts showing the Anima programmatic API for embedders — driving
the pipeline from `import anima_lora` instead of `make` targets. Each script is
self-contained.

After `uv sync` (which installs this repo editable), the front-door package is
importable from anywhere:

```python
import anima_lora
settings = anima_lora.inference.get_generation_settings(args)
latent = anima_lora.inference.generate(args, settings)
image = anima_lora.inference.decode_to_pil(vae, latent, device)
```

`anima_lora` is a lazy re-export grouped into `anima_lora.{models, inference,
config, training, captioning}`; the namespace → export → canonical-home map is
in `anima_lora/__init__.py`. Repo-relative model/config paths resolve against
the repo home, not the CWD; set `ANIMA_HOME` for a relocated checkout.

The high-level flows (`01`–`03`, `08`, `09`) import from `anima_lora`; `10`
drives diffusers; the building blocks (`04`–`07`) import the `library.*`
primitives directly. None needs a `sys.path` bootstrap.

**High-level flows** — the supported entry points:

| Script | Shows | Needs |
|---|---|---|
| [`01_generate.py`](01_generate.py) | Text-to-image: `get_generation_settings` → `generate` → `save_output`, optionally with one or more LoRA adapters attached at DiT load | DiT + VAE + text encoder (+ adapter `.safetensors` for LoRA) |
| [`02_config_and_train.py`](02_config_and_train.py) | `load_method_preset` merge chain + `create_network` (three-axis routing) + in-process training via `AnimaTrainer().train(args)` | config part: nothing; `--build-network`: DiT; `--train`: preprocessed cache |
| [`03_generate_with_correction.py`](03_generate_with_correction.py) | Training-free sampler correction (SMC-CFG / Spectrum) via the `GenerationRequest.extra_argv` escape hatch for long-tail method flags | DiT + VAE + text encoder |
| [`08_easycontrol_train_and_infer.py`](08_easycontrol_train_and_infer.py) | **Image-conditioned** end-to-end: `--method easycontrol` training (same merge-chain + `AnimaTrainer().train()` as `02`) and image-conditioned inference via the typed `easycontrol_weight` / `easycontrol_image` request fields | config: nothing; `--train`: paired cache in `easycontrol-dataset/`; `--infer`: adapter + ref image |
| [`09_cjk_vocab_pack.py`](09_cjk_vocab_pack.py) | **Prompt in JA / KO / ZH** through the [CJK vocab pack](https://huggingface.co/sorryhyun/anima-vocab-pack-cjk) (not a LoRA) on the front door: `GenerationRequest(vocab_pack=…)` — `generate()` installs the pack-routing tokenizer and `load_dit_model` hooks the rows onto `llm_adapter.embed` (state dict untouched). Default follows base.toml `vocab_pack` (`make download-vocab-pack`); `no_vocab_pack=True` forces the stock tokenizer. `--dry_run` reports the routed id stream with no weights | DiT + VAE + text encoder + the downloaded pack; `--dry_run`: text-encoder tokenizer only |
| [`10_cjk_vocab_pack_diffusers.py`](10_cjk_vocab_pack_diffusers.py) | The same pack on the **diffusers** Anima `ModularPipeline` (≥ 0.39, `circlestone-labs/Anima-Base-v1.0-Diffusers`): swap the `text_encoder` block for a subclass whose T5 ids come from `HybridT5Encoder`, widen `pipe.text_conditioner.embed` once — nothing else from this repo's engine. `--dry_run` checks both patch points on CPU | diffusers pipeline weights (auto-fetched, ~5.6 GB); `--dry_run`: text side only |

**Building blocks** — the raw primitives for writing your own `scripts/` tool:

| Script | Shows | Needs |
|---|---|---|
| [`04_load_models.py`](04_load_models.py) | Load DiT / VAE / text encoder directly; encode a prompt to the DiT-ready cross-attn embedding | DiT + VAE + text encoder |
| [`05_vae_and_dataset.py`](05_vae_and_dataset.py) | VAE pixel↔latent round-trip; iterate the on-disk training cache (`CachedDataset`) | VAE (+ cache for part B) |
| [`06_frozen_dit_training_build.py`](06_frozen_dit_training_build.py) | Frozen DiT + fresh adapter build for *training* via the `harness` helpers (`place_dit_for_training` / `compile_dit_blocks` / `enable_training_grad_ckpt`) — the `project/finished/mod_guidance` / `scripts/distill_turbo` model-build sequence | DiT |

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
python examples/03_generate_with_correction.py --correction spectrum   # extra_argv method knobs
python examples/08_easycontrol_train_and_infer.py                 # print easycontrol config
python examples/08_easycontrol_train_and_infer.py --train --max_train_epochs 6
python examples/08_easycontrol_train_and_infer.py --infer --ref path/to/ref.png
python examples/04_load_models.py --prompt "a lighthouse at dusk"
python examples/05_vae_and_dataset.py                       # iterate the cache
python examples/05_vae_and_dataset.py --image some/photo.png  # VAE round-trip
python examples/06_frozen_dit_training_build.py             # build a trainable adapter
```

## Notes for embedders

- **`anima_lora` is the stable API; `library.*` / `networks.*` / `scripts.*` are internal.**
  The underlying trees are importable (`04`–`07` use `library.*`), but may move
  or change signature without a deprecation cycle — pin a tag (`ANIMA_VERSION`)
  if you depend on them.
- **Inference is request-driven.** `01`/`03` build a typed
  `anima_lora.GenerationRequest` and call `.to_args()`, which feeds the request
  through `inference.parse_args` so every optional knob gets its default. The long tail of
  method knobs (spectrum/smc-cfg/ip-adapter) rides through the request's `extra_argv`,
  or you can build the `argparse.Namespace` straight from `inference.parse_args(argv)`.
- **Adapter family is in the checkpoint, not the call.** `01 --lora_weight` passes
  any LoRA / T-LoRA / Hydra `.safetensors`; the DiT loader reads
  the metadata and merges-or-keeps-live accordingly.
- **Image-conditioning is typed, not `extra_argv`.** EasyControl (`08`) feeds a
  reference image, so `GenerationRequest` models `easycontrol_weight` /
  `easycontrol_image` (and `pooled_text_proj` for mod-guidance) as first-class
  fields — only the leftover scalar knobs (`--easycontrol_scale`,
  `--easycontrol_image_match_size`) ride `extra_argv`. Training is *not* special:
  EasyControl is a plain `--method easycontrol` over the same merge-chain +
  `AnimaTrainer` as `02`, with an inline paired-dataset blueprint in
  `configs/easycontrol/easycontrol.toml`.
- **Variant stacking is kwargs, not config.** The toggle flags in
  `configs/methods/lora.toml` are just one way to pick variants — the same keys
  pass straight through `create_network(**kwargs)` to `resolve_network_spec`.
  Which combos exist is the three-axis matrix in `networks/CLAUDE.md`; impossible
  combos raise at build.
- **Prompt encoding uses two process-global strategy singletons.** `generate()` /
  `prepare_text_inputs()` lazily install them from `args.text_encoder` (via
  `anima_lora.ensure_text_strategies`), so the high-level flows just work; `04`
  shows the explicit one-liner. Encoding also needs the DiT — the encoder hidden
  states are projected by `Anima._preprocess_text_embeds`.
- **Multi-GPU training** needs `accelerate launch` (`ANIMA_ACCELERATE_LAUNCH=1
  make lora`). `02 --train` runs single-process, like the default `make lora`.
- The text-encoder padding and free-fit bucketing invariants in `../CLAUDE.md`
  apply — the called functions handle them; read them before deviating from
  these flows.
