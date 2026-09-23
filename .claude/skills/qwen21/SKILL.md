---
name: qwen21
description: Qwen-Image-2.1 LoRA line (NOT Anima) — running cache/train through the daemon, `make gui-qwen`, the CacheRequest/TrainRequest flag surface and how to add a field, model-dir resolution, cache layout, swap sizing, and the gotchas (flat stem-keyed cache, --no-x flag spelling, stall budget, GUI-side chain, Test A/B generation). Load before running, changing, or debugging anything under library/qwen21/, scripts/qwen21/, gui/qwen21/ or project/qwen21_lora/.
---

# Qwen-Image-2.1 LoRA (`library/qwen21/`)

**Not Anima.** Read `library/qwen21/CLAUDE.md` first — the root invariants (5D latents,
max-padded text, free-fit bucketing, block-compile first) are wrong here. Research state
and measured numbers: `project/qwen21_lora/README.md` + `report.md`.

## Layout

| Where | What |
|---|---|
| `library/qwen21/` | core: `loader` / `blockswap` / `lora` / `accel`, `cache.run_cache(req)`, `train.run_train(req)`, `generate.run_generate(req)` |
| `library/qwen21/requests.py` | **torch-free** `CacheRequest` / `TrainRequest` / `GenerateRequest` — the one flag definition |
| `library/qwen21/scan.py` | torch-free folder/cache counts (GUI) |
| `scripts/qwen21/{cache,train,generate}.py` | sidecar CLIs: `Request.from_argv()` → `run_*` |
| `gui/qwen21/` | `make gui-qwen` window (en/cn) |
| `project/qwen21_lora/src/` | one-off research scripts (`backward_smoke`, `smoke_t2i`, `bench_accel`) |

`tests/test_qwen21_boundary.py` enforces the island: nothing Anima-side imports
`library.qwen21`; it imports only `library.env` / `library.runtime.{offloading,device}`;
`requests` and `gui.qwen21.app` stay torch-free.

## Running

GPU work goes through the daemon:

```bash
make daemon-run ARGS="--stall-timeout 900 scripts/qwen21/cache.py --src 'post_image_dataset/resized/channel_(caststation)'"
make daemon-run ARGS="--stall-timeout 900 scripts/qwen21/train.py --epochs 8 --save_every_epochs 4"
make daemon-run ARGS="--stall-timeout 900 scripts/qwen21/generate.py --lora output/qwen21/qwen21_lora.safetensors"
make gui-qwen   # the same jobs from a window
```

**Test / A-B generation** (`GenerateRequest`, the Train tab's Test button): renders each
prompt at every `--multipliers` scale (default `1.0,0.0`) from one loaded model with the
same seed, so the pair differs by the adapter only. `set_multiplier(0)` short-circuits
every adapter, so no second model copy is loaded. The default prompt is `BOCCHI_PROMPT` —
natural language, a character the base model already knows. `--prompts_file` (one per
line, e.g. `project/qwen21_lora/eval_prompts.txt`) overrides it. Without
`--width/--height` it renders square at `--resolution`, which is off-distribution for a
portrait dataset. The GUI fills an empty `lora` with the Train tab's `output`, writes
each run to `out_dir/<timestamp>/`, and shows the pair from its `manifest.json`.

Defaults: `--src post_image_dataset/resized` (walks subfolders; captions = the revised
`{stem}.txt`, `.variants.txt` ignored), cache and LoRA under `output/qwen21/`. The
research cache is `project/qwen21_lora/cache` — pass `--out` / `--cache` explicitly.
Relative paths resolve under the repo home, not the CWD.

**Model dir** (diffusers layout: `transformer/ text_encoder/ vae/ scheduler/`):
`--model_dir` → `$ANIMA_QWEN21_MODEL_DIR` (env or `.env`) → `models/qwen_image_2.1`. On
the dev box that last one is a symlink to the NVMe copy. There is no catalog row yet.

## Flags — one definition

A field on `CacheRequest` / `TrainRequest` / `GenerateRequest` is the CLI flag, its default, its help and the
GUI widget at once. Adding one:

1. Add the field with `_f(default, help, choices=…, advanced=…, multiline=…)`.
2. Read it in `run_cache` / `run_train` / `run_generate`.
3. For the Chinese GUI, add `FIELDS_CN[name]` in `gui/qwen21/strings.py`
   (`FIELDS_CN_GENERATE` for a `GenerateRequest` field whose name means something else
   elsewhere, e.g. `resolution`).
   English needs nothing — the label is the field name, the help comes from the metadata.

Widget type follows the field: `bool` → checkbox, `choices` → combo, `int` with a
non-None default → spin box, `multiline` → text box, everything else → line edit (empty =
`None` for `… | None` fields). `src/out/cache/model_dir/out_dir` get a folder picker,
`output` a save-file picker, `lora/prompts_file` an open-file picker.

`to_argv()` writes only non-default values, so a daemon job's `argv` reads as the
choices made.

## Gotchas

- **Bool flags are `--x` / `--no-x`** (`BooleanOptionalAction`), keeping the underscore:
  `--no-grad_checkpointing`, `--no-save_crops`. The pre-move `--no_grad_checkpointing`
  is gone.
- **The cache is flat, keyed by file stem.** Caching refuses a source tree with the same
  stem in two subfolders; the GUI warns before submitting.
- **Caching skips existing files.** An edited caption keeps its old embedding until
  `--overwrite` (or the `.te` file is deleted). The GUI counts these as stale.
- **Resolution is an area, not a crop.** `calculate_dimensions(res², aspect)` keeps the
  native aspect with both edges on a multiple of 32. The image is resized, not cropped, so
  aspect drifts up to ~2 %. `post_image_dataset/resized` input is already Anima-resized,
  so it gets resampled twice.
- **Stall budget.** The daemon's default command watchdog is 120 s. The text-encoder load
  and long passes can be quieter than that, so the GUI submits with 900 s; pass
  `--stall-timeout 900` on the CLI too.
- **The GUI chains preprocess → train itself.** Training is submitted only on cache
  success, so a failed cache never trains on a stale folder. Closing the window
  mid-chain drops the pending train job; the running cache job continues. The GUI
  re-attaches to a running `qwen21-cache` / `qwen21-train` / `qwen21-test` job on reopen.
- **Progress** comes from stdout lines, not `progress.jsonl`: `  text i/n`,
  `  latents i/n`, `  step i/n`, `  image i/n`. Keep that shape if you change the prints
  — the GUI's bar parses it (tqdm bars too, but tqdm's `\r` redraws only reach the log
  at the next newline, so the denoise bar arrives in one burst).
- **Swap sizing.** `report_fit` prints after step 1 with a `--blocks_to_swap`
  suggestion measured against `mem_get_info` free. At 1024² the step is compute-bound,
  so fewer swaps doesn't make it faster (see `library/qwen21/CLAUDE.md`). Default
  `--activation_reserve_gb 3.5` is the measured floor with gradient checkpointing
  (2026-09-23, 30 samples up to 4096+346 tokens): swap 6, peak 13.55 GB, 0.5 GB spare,
  4.08 s/step. 2.0 OOMs at step 1 (swap 2); 7.0 swapped 14 at the same speed.
- **RAM:** the whole checkpoint is ~33 GB bf16 and sits in page cache on the 64 GB box.
  Reloads are ~free, so there is no resident-model worker; VRAM is the constraint.
