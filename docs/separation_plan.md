# Plan: split inference into `../anima_inference`

Working document. Captures the agreed-upon shape of the split before any
moves happen, so the refactor is reversible and the final state is checked
against this file rather than reconstructed from chat logs.

## Why

`anima_lora` has accumulated features on both the training and inference
sides — spectrum (purely inference), mod-guidance (inference + a small
distillation script), p-graft, plus the inference CLI itself. Inference
researchers don't need the training pipeline, training researchers don't
need the inference CLI, and shipping spectrum/mod-guidance as inference
research from a "training repo" is awkward at the user-facing level.

The split is for **decongestion and audience separation**, not for
shrinking installed dependencies — `anima_inference` will depend on
`anima_lora` as an installed package, so the dep surface is unchanged.

## Final layout

### `anima_lora` after split (training repo)

Stays:

- All training code: `train.py`, `library/training/`, `library/datasets/`,
  `library/anima/`, `library/models/`, `library/io/`, `library/runtime/`,
  `library/config/`, `library/log.py`.
- All adapter code: `networks/{lora_anima, lora_modules, methods,
  attention_dispatch.py, lora_save.py, lora_utils.py}`.
- DCW (ongoing experiment): `networks/dcw.py`,
  `library/training/{dcw_validation, bias_metric}.py`, `bench/dcw/`.
- `library/inference/*` — the package stays. Training and bench import
  from `sampling.py`, `adapters.py`, `models.py`, `mod_guidance.py`.
- `networks/spectrum.py` — stays as a leaf module. Decoupled from
  `library/inference/generation.py` via a callback hook (see Phase 0
  below) so neither repo holds a hard import edge.
- Method training configs: `configs/methods/`, `configs/gui-methods/`,
  `configs/presets.toml`, `configs/base.toml`.
- All training `make` targets, GUI, masking, preprocessing.
- `custom_nodes/comfyui-hydralora/`.

Removed (moved to `anima_inference`):

- `inference.py` (top-level).
- `scripts/distill_modulation.py`.
- All `cmd_test*` task commands in `scripts/tasks/inference.py` —
  `test`, `test-mod`, `test-spectrum`, `test-spectrum-dcw`, `test-apex`,
  `test-hydra`, `test-prefix`, `test-postfix*`, `test-ip`,
  `test-easycontrol`, `test-merge`, `test-ref`.
- `make distill-mod` target.
- `docs/methods/spectrum.md`, `docs/methods/mod-guidance.md`.

### `anima_inference` (new sibling repo)

Owns:

- `inference.py` (the CLI).
- Mod-guidance distillation script.
- All `test*` task commands (the entire user-facing inference surface).
- Inference research docs (spectrum, mod-guidance).
- Spectrum runner registration shim — imports
  `anima_lora.networks.spectrum.spectrum_denoise` and registers it via
  `library.inference.generation.register_spectrum_runner` at startup.

Depends on: `anima_lora` installed as a Python package
(`uv pip install -e ../anima_lora`).

### Stays in their own repos

- `ComfyUI-Spectrum-KSampler` — independent, published to Comfy registry
  on its own cadence.
- `custom_nodes/comfyui-hydralora/` — stays inside `anima_lora` because
  it is consumed against training output.

## Phase 0 — groundings (DONE)

- [x] Write this `plan.md`.
- [x] Refactor `library/inference/generation.py` to use a
      `_SPECTRUM_RUNNER` callback. The direct
      `from networks.spectrum import spectrum_denoise` at the spectrum
      branch is replaced with a call through a module-level runner.
      `register_spectrum_runner(fn)` is the API; if no runner is
      registered when `args.spectrum=True`, raise a clear error.
- [x] `networks/spectrum.py` registers itself on import (calls
      `register_spectrum_runner(spectrum_denoise)` at module bottom).
- [x] `inference.py` imports `networks.spectrum` near the top so the
      registration happens before `generate()` runs.
- [x] `make test-spectrum` works end-to-end after the refactor
      (verified by user).

After Phase 0, anima_lora's `library/inference/generation.py` has no
direct dependency on `networks.spectrum`. The runner is plugged in by
whichever entry point owns the CLI — anima_lora's `inference.py` today,
`anima_inference`'s `inference.py` after the move.

Decision (Phase 0 follow-up): `networks/` will **not** move under
`library/`. Blast radius is wide (32 Python files + 11 method/gui-method
TOML configs + checkpoint `.snapshot.toml` files all reference
`networks.*` paths via the user-visible `network_module` config key),
the framework/plugins boundary between `library/` and `networks/` is
conceptually load-bearing, and the inference split's packaging story
("declare `library` and `networks` as two top-level packages") works
fine without the move. Revisit only as its own focused refactor.

## Phase 1 — make `anima_lora` importable as a library

`pyproject.toml` currently has `[tool.setuptools] packages = []` — the
project is set up as an app, not a library. Nothing is exposed for
`import library.*` from a sibling project.

- [ ] Set `[tool.setuptools.packages.find]` (or explicit
      `packages = ["library", "networks"]`) so both packages are
      discoverable when installed.
- [ ] Sanity check from a sibling venv:
      `uv pip install -e ../anima_lora`, then
      `python -c "from library.anima import models; from library.inference import generation; from networks import spectrum, dcw"`.
- [ ] Tag a baseline commit so `anima_inference` can pin it.

## Phase 2 — bootstrap `../anima_inference`

- [ ] `git init ../anima_inference`. Skeleton: `pyproject.toml`,
      `README.md`, `CLAUDE.md`, `.gitignore`, `ruff.toml`.
- [ ] `pyproject.toml` declares `anima-lora` as a path dep:
      `dependencies = ["anima-lora @ file:///../anima_lora"]` for dev.
- [ ] Layout:
      ```
      anima_inference/
        anima_inference/
          __init__.py
          inference.py
          spectrum_register.py        # imports networks.spectrum, registers runner
          mod_guidance/
            __init__.py
            distill.py                # was scripts/distill_modulation.py
        scripts/
          tasks/
            inference.py              # cmd_test, cmd_test_spectrum, ...
        tasks.py
        pyproject.toml
        README.md
        CLAUDE.md
      ```
- [ ] `anima_inference`'s `__init__.py` imports `spectrum_register` so
      `import anima_inference` is enough to wire spectrum.

## Phase 3 — moves (one PR per move, validate after each)

In dependency order:

- [ ] Move `inference.py` → `anima_inference/anima_inference/inference.py`
      via `git mv`. Imports keep working unchanged because anima_lora is
      pip-installed.
- [ ] Move `scripts/distill_modulation.py` →
      `anima_inference/anima_inference/mod_guidance/distill.py`. Imports
      stay the same (`library.anima.{models, weights}`,
      `library.io.cache`).
- [ ] Move all `cmd_test*` from `scripts/tasks/inference.py` to
      `anima_inference/scripts/tasks/inference.py`. Drop the originals
      from `anima_lora`.
- [ ] Move `docs/methods/spectrum.md` and `docs/methods/mod-guidance.md`
      to `anima_inference/docs/`.
- [ ] Update `anima_inference`'s `tasks.py` to expose `test`, `test-*`,
      `distill-mod`. Default checkpoint search path
      `../anima_lora/output/ckpt/`, env-var-overridable.

## Phase 4 — docs cleanup

- [ ] `anima_lora/CLAUDE.md`: drop the spectrum and mod-guidance sections,
      replace with a one-paragraph cross-reference to `anima_inference`.
- [ ] `networks/CLAUDE.md`: keep the `spectrum.py` row in the layout
      table (the file stays here) but note the registration boundary —
      `library/inference/generation.py` no longer imports it directly,
      the entry-point CLI plugs it in.
- [ ] `anima_inference/CLAUDE.md`: fresh draft. Narrower than ours —
      training pipeline is "see `../anima_lora/CLAUDE.md`".

## Phase 5 — release coordination

- [ ] Tag both repos.
- [ ] `anima_inference/pyproject.toml` pins `anima-lora` by SHA or tag
      until anima_lora cuts numbered releases.

## Risks (carry forward, revisit if they bite)

1. **Shared dep surface is large.** `anima_inference` imports
   `library.{anima, models, io, runtime, datasets, inference, log}`.
   Renames in anima_lora will break anima_inference. Coordination tax —
   accepted price of the split.
2. **Bench cross-talk on mod-guidance.**
   `library/inference/mod_guidance.py` stays in anima_lora because
   `bench/dcw/measure_bias.py` imports `setup_mod_guidance`.
   `anima_inference` re-exports it for callers that want a stable name.
3. **DCW + spectrum composition (`test-spectrum-dcw`) crosses repos.**
   The composition test lives in `anima_inference`, DCW logic stays in
   anima_lora. DCW changes can break the test.
4. **Workflow change.** `make test*` moves out of anima_lora. Iterating
   on a LoRA becomes:
   ```
   cd anima_lora && make lora
   cd ../anima_inference && python tasks.py test
   ```
   Two `cd` hops instead of one `make`.

## Open decisions

- None as of writing this plan. The Phase 0 refactor uses Option I
  (callback registration) so spectrum.py can stay where it is without
  anchoring the cross-repo dependency direction.
