# Anima LoRA — `library/` Refactor Plan

Goal: kill the 9-way `*_util(s).py` sprawl at `library/` root. Readers shouldn't have to open a file to discover what domain it covers. Group by **domain**, mirroring what `library/datasets/`, `library/training/`, and `library/inference/` already did.

Constraints:
- `library/train_util.py` is a **public re-exporting facade** used by external scripts (`scripts/convert_lora_to_comfy.py`, `scripts/graft_step.py`, `preprocess/*`) and 3 test files. Its import surface must not change.
- `make lora | tlora | hydralora | postfix | test` must pass after every milestone.
- Checkpoints stay byte-identical. No numerical drift.

---

## Progress

- [x] **M1** — `library/config/` (schema, loader). Shims at old paths. 4 internal caller sites rewritten.
- [x] **M2** — `library/runtime/` (device, offloading, noise). Shims at old paths. 14 internal caller sites rewritten.
- [x] **M3** — `library/io/` (cache, safetensors). Shims at old paths. 3 internal safetensors callers rewritten; `cache_utils` has no internal callers (shim alone carries the 6 external sites).
- [x] **M4** — `library/models/` (qwen_vae, sai_spec). Shims at old paths. 3 internal callers rewritten; cross-module imports inside moved `qwen_vae.py` updated to `runtime/` + `io/`.
- [ ] **M5** — `library/anima/` (deferred; biggest churn).
- [ ] **M6** — absorb `custom_train_functions`, rename `utils.py → log.py` (stdlib-safe rename).
- [ ] **M7** — external caller migration + shim deletion.

Gate status for M1–M4: `py_compile` clean across all 31 touched files; torch-free shim re-exports verified (`config_schema`, `sai_model_spec`). Behavioral gates (`make test-unit`, `--print-config`, one `make lora` step) deferred — this venv hits a Python 3.13 × torch `@_overload_method` AST bug on `import torch` that is unrelated to the refactor (reproduced on clean `HEAD`).

---

## Target tree

```
library/
├── __init__.py
├── train_util.py                 # facade — unchanged public API, internal imports updated
├── strategy_base.py              # abstract strategies (not anima-specific) — stays at root
├── custom_train_functions.py     # absorbed into training/losses.py in M6
├── anima/
│   ├── models.py                 # was anima_models.py                (2226 LOC)
│   ├── training.py               # was anima_train_utils.py           ( 977 LOC)
│   ├── weights.py                # was anima_utils.py                 ( 514 LOC; weight xforms, fused-proj remap)
│   └── strategy.py               # was strategy_anima.py              ( 658 LOC)
├── models/
│   ├── qwen_vae.py               # was qwen_image_autoencoder_kl.py   (2043 LOC)
│   └── sai_spec.py               # was sai_model_spec.py              ( 210 LOC)
├── config/
│   ├── schema.py                 # was config_schema.py
│   └── loader.py                 # was config_util.py
├── io/
│   ├── cache.py                  # was cache_utils.py        (external: 6 script sites)
│   └── safetensors.py            # was safetensors_utils.py
├── runtime/
│   ├── device.py                 # was device_utils.py                (17 import sites)
│   ├── offloading.py             # was custom_offloading_utils.py
│   └── noise.py                  # was noise_utils.py                 (flow-matching sampling)
├── logging.py                    # was utils.py (84 LOC, purely logging + fire_in_thread; 31 sites)
├── datasets/                     # unchanged
├── training/                     # unchanged (absorbs custom_train_functions.py in M6)
└── inference/                    # unchanged
```

Everything that moves keeps a short-lived **shim module** at the old path: `from library.<new_path> import *` + a `# DEPRECATED: import from library.<new_path>` line. Shims are deleted in M7 once all internal callers are updated.

---

## Migration principles

1. **One domain per milestone.** Each milestone moves a cohesive group, ships, and is landable alone.
2. **Shim, then sweep.** Move file → write shim at old path → run `make test-unit` + one `make lora` epoch → update internal callers → delete shim in M7.
3. **Facade stays stable.** `library/train_util.py` updates its internal `from library.<x>` paths but keeps the same exported names. No caller of `train_util` notices anything.
4. **Scripts come last.** `scripts/*` and `preprocess/*` are user-facing — they migrate to new paths only in M7, after every shim has proven harmless.
5. **No content changes.** This is pure `git mv` + import rewrites. Splitting a big file (e.g., slicing `anima_models.py`) is out of scope; do it in a separate plan.

---

## Milestones

### M1 — `library/config/` (lowest risk, ~30min)
- `git mv library/config_schema.py library/config/schema.py`
- `git mv library/config_util.py library/config/loader.py`
- Shim: `library/config_schema.py` re-exports from `library.config.schema`; same for `config_util`.
- Callers today: `train.py` (both), `library/train_util.py` (schema), 1 test. Update these 4 sites.
- **Gate:** `make test-unit` + `python train.py --print-config --method lora`.

### M2 — `library/runtime/` (device/offload/noise)
- Move `device_utils.py → runtime/device.py`, `custom_offloading_utils.py → runtime/offloading.py`, `noise_utils.py → runtime/noise.py`.
- `device_utils` is the hot one: 17 sites incl. `library/train_util.py` re-export. Update facade to import from new path; re-export name `clean_memory_on_device` unchanged.
- Shims kept for all three.
- **Gate:** `make test-unit` + one `make lora` step (verifies device+noise path end-to-end).

### M3 — `library/io/` (cache + safetensors)
- Move `cache_utils.py → io/cache.py`, `safetensors_utils.py → io/safetensors.py`.
- `cache_utils` has 6 **external-ish** script sites (`scripts/graft_step`, `scripts/distill_modulation`, `scripts/invert_embedding`, `scripts/interpret_inversion`, `preprocess/cache_latents`, `preprocess/cache_text_embeddings`). Shim keeps them working; they migrate in M7.
- **Gate:** `make preprocess` round-trips (writes + reads disk caches).

### M4 — `library/models/` (ancillary model defs)
- Move `qwen_image_autoencoder_kl.py → models/qwen_vae.py`, `sai_model_spec.py → models/sai_spec.py`.
- `sai_model_spec` is re-exported by `train_util` as a module (`from library import sai_model_spec`) — keep that alias working via shim.
- **Gate:** `make test-unit` + one LoRA save (checks `sai_spec` metadata write).

### M5 — `library/anima/` (anima-specific bundle)
- Move `anima_models.py → anima/models.py`, `anima_train_utils.py → anima/training.py`, `anima_utils.py → anima/weights.py`, `strategy_anima.py → anima/strategy.py`.
- `strategy_base.py` stays at root (it's domain-agnostic).
- Biggest churn: 20+ internal + bench import sites. Update `train.py`, `inference.py`, `networks/*`, `bench/*` in one commit per file.
- **Gate:** `make lora` 1 epoch + `make test` generates an image.

### M6 — Small absorptions & rename
- Inline `custom_train_functions.py` (66 LOC, 1 caller) into `library/training/losses.py`. Delete the file.
- Rename `utils.py → logging.py` with shim. Contents are purely logging + `fire_in_thread` — the rename is a readability win worth the 31-site sweep.
- **Gate:** `make test-unit`.

### M7 — Shim deletion & external callers
- Update `scripts/*` and `preprocess/*` to new paths.
- Delete every shim module introduced in M1–M6.
- Update `CLAUDE.md` "Architecture" section to point at the new tree.
- **Gate:** `grep -r 'from library\.\(config_schema\|config_util\|device_utils\|custom_offloading_utils\|noise_utils\|cache_utils\|safetensors_utils\|qwen_image_autoencoder_kl\|sai_model_spec\|anima_models\|anima_train_utils\|anima_utils\|strategy_anima\|custom_train_functions\)' .` returns nothing.

---

## Sequencing

Recommended order: **M1 → M2 → M3 → M4 → M5 → M6 → M7**.

- M1 first because config has the fewest importers — proves the shim pattern works.
- M5 is the biggest churn; do it once earlier milestones have ironed out the migration recipe.
- M7 gates on M1–M6 being in; otherwise the shim deletions regress scripts.

M1–M4 are parallelizable on separate branches if needed.

---

## Out of scope (deliberate)

- **No file splits.** `anima_models.py` (2226 LOC) and `qwen_image_autoencoder_kl.py` (2043 LOC) are moved whole. Slicing them is a separate plan.
- **No API redesign.** `train_util.py`'s re-export list is frozen — change it in a different milestone.
- **No new abstractions.** This is renaming + `mkdir`, not an architecture refactor.
- **No coverage additions.** Existing `make test-unit` (69 tests) is the parity check; if it stays green, we shipped.
- **No Windows-specific work.** Cross-platform behavior is unchanged because paths inside `library/` don't touch the shell layer.

---

## Critical files

- `/home/sorryhyun/anima_lora/library/train_util.py` — facade; its internal imports change every milestone, its export surface never does.
- `/home/sorryhyun/anima_lora/library/__init__.py` — currently empty; may need a handful of re-exports in M7 if we want `from library import sai_spec` to keep working.
- `/home/sorryhyun/anima_lora/train.py` — touched in M1, M2, M5 (biggest single caller).
- `/home/sorryhyun/anima_lora/inference.py` — touched in M2, M4, M5.
- `/home/sorryhyun/anima_lora/scripts/convert_lora_to_comfy.py`, `scripts/graft_step.py` — external-surface callers; moved in M7 only.
- `/home/sorryhyun/anima_lora/CLAUDE.md` — architecture section updated in M7.
