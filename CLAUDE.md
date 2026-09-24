# CLAUDE.md

Orientation and invariants for this repository. Task-shaped detail lives in skills (`ls
.claude/skills`) and in the nested `configs/CLAUDE.md`, `networks/CLAUDE.md`,
`gui/CLAUDE.md`.

## Project Overview

Anima — LoRA/T-LoRA training and inference pipeline for the Anima diffusion model
(DiT-based, flow-matching). Supports several adapter families (LoRA / T-LoRA / HydraLoRA
/ EasyControl) selectable via method config + hardware preset. The LoRA family is
routed via a three-axis surface — `use_moe_style` /
`route_per_layer` / `router_source` — see `configs/methods/lora.toml` and the
`lora-routing` skill.

## Setup

```bash
uv sync                    # Install dependencies (Python 3.13)
hf auth login              # Authenticate for model downloads
make download-models       # first-run set: DiT, TE, VAE, PE, CJK vocab pack, tagger, tag DB
# Training images go in image_dataset/ with .txt caption sidecars
make preprocess            # Resize → post_image_dataset/resized/, cache → post_image_dataset/lora/
```

Weights are catalog rows, not commands (`library/downloads.py`; `make download-list`
shows every row, `make download-model <pack|alias|row>` fetches one — SAM3 and OCR are
opt-in). **Load the `model-catalog` skill** before adding or moving a weight or changing
a loader's default path.

## Commands

Both `make` (Unix) and `python tasks.py` (cross-platform/Windows) work — the `Makefile`
is a thin dispatcher forwarding every target to `python tasks.py <target> $(ARGS)`.
**`tasks.py` is the source of truth**; command bodies live in
`scripts/tasks/{training,inference,preprocess,masking,gui,downloads,utilities,tagger}.py`
and `scripts/experimental_tasks/` (for `exp-*`). `make help` lists every target.

All training runs `train.py --method <name> --preset <name>`, invoked directly
(single-process `Accelerator()`, `mixed_precision` from the config chain);
`ANIMA_ACCELERATE_LAUNCH=1` wraps it in `accelerate launch` for multi-GPU
(`build_launch_cmd` in `scripts/tasks/_common.py`). Override any config value from CLI
(`--network_dim 32 --max_train_epochs 64`) or the preset via `PRESET=low_vram make lora`.
`exp-*` targets are experimental and may break or be removed.

Knobs and gotchas:

- **Training**: `make lora PRESET=low_vram|half|quarter` (half → `sample_ratio=0.5`;
  full list in `configs/presets.toml`);
  `make lora-gui GUI_PRESETS=tlora` runs the clean per-variant `configs/gui-methods/`
  tree (`ls` it for the live list). `make turbo` is the DP-DMD distiller;
  `exp-soft-tokens` is the experimental method.
- **`make soup PATH_PATTERN="<glob>"`** (or `TARGET=<dir>` shorthand) — uncond-init soup
  pipeline (`scripts/soup/`; GUI: Experimental tab → soup). Plain-LoRA only. **Load the
  `soup` skill** before running or modifying it; deep-dive `docs/experimental/soup.md`.
- **Inference compose flags**: `SPECTRUM=1` / `MOD=1` / `NOLORA=1` compose into
  **every** `test-*` target (`make test`, `test-hydra`, `test-merge`, `test-smc-cfg`,
  `test-easycontrol REF_IMAGE=…`, `exp-test-*`).
- **`make gen`** — daemon-routed batch generation (same argv + env levers as `make
  test`, submitted as a GPU command job). Eval grids / seed sweeps go here; interactive
  single images stay on `make test` or the resident inference server.
- **Daemon** (local FIFO job queue, auto-starts on first submit): **agent-launched GPU
  work must go through it** — GPU processes started from a Claude Code background Bash
  get killed by the harness sandbox layer after ~1 min (silent SIGKILL, no trace). Front
  door `make daemon-run ARGS="<script.py> [flags]"`; append `--queue` to any
  train/distill target to enqueue. **Load the `daemon` skill** for the full surface;
  contract in `anima_daemon/README.md`.
- **σ-demoted training** (`--sigma_lowres`, opt-in): high-σ steps train on a lower-res
  sibling latent. Needs sibling latents precached; output is an ordinary LoRA. **Load
  the `sigma-lowres` skill before enabling/tuning it**; contract in
  `docs/optimizations/sigma_lowres.md`.
- **Gotchas**: `make merge ADAPTER_DIR=… [MULTIPLIER=0.8]` bakes LoRA into the DiT
  (LoRA/T-LoRA only) and refuses Hydra-moe / postfix unless `--allow-partial`.
  `turbo` output is a normal LoRA — infer with `--infer_steps` matched to the DP-DMD
  `student_steps` rollout (currently 4) and `--cfg 1.0`. `make print-config METHOD=…
  PRESET=…` dumps the merged chain; `make test-unit` runs pytest; `ruff check . --fix &&
  ruff format .` (touched files only — see [[feedback_ruff_scope_collateral]]).
- **Run the test suite at most twice per task** (here or in `../anime_tools`): once
  after the change, once after fixing what it caught; scope the re-run to the affected
  test file. If a third run is required, say why.

## Key entry points

| File | Purpose |
|------|---------|
| `anima_lora/__init__.py` | **Programmatic front door** — lazy re-export of the curated embedder entry points, namespaced `anima_lora.{models, inference, config, training, captioning}`. `import anima_lora` instead of reverse-engineering `main()`s; see the `embedder-api` skill |
| `examples/` | Runnable API scripts (high-level flows + raw primitives). `examples/README.md` is the embedder guide |
| `train.py` | `AnimaTrainer` — main training loop via HF Accelerate |
| `inference.py` | Standalone image generation (`--help` for all flags) |
| `networks/spectrum.py` | Spectrum inference acceleration |
| `gui/` | PySide6 GUI package |
| `tasks.py` | Cross-platform task runner — source of truth for every `make` target |
| `library/downloads.py` | **Model catalog** — one `Asset` per weight, grouped into packs; loaders import their default paths from here. Add a weight by adding a row, not a command (`model-catalog` skill) |
| `scripts/tasks/` + `scripts/experimental_tasks/` | Where command bodies actually live (`_common.py` = shared helpers) |

Docs: shipped method deep-dives in `docs/methods/`, experimental in
`docs/experimental/`, active proposals in `docs/proposal/`, retired material under
`_archive/`. Active lines with open phases live under `project/<line>/` (see
`project/README.md`); completed lines move to `project/finished/<line>/` (e.g. the
ResShift SR sidecar — no `make sr-*` targets, run its scripts directly); killed or
superseded lines go to `_archive/`.

## Programmatic API (embedders)

`uv sync` installs the repo editable, so `anima_lora` is importable anywhere. Inference
is **request-driven** (build a typed `GenerationRequest`, call `.to_args()`), and
repo-relative model/config paths resolve against the **repo home**, not the CWD — **new
code opening a repo-relative path must call `resolve_under_home()`**
(`library.env.anima_home()`; `ANIMA_HOME` for a relocated checkout). Namespaces, the
strategy singletons and the per-model path env vars are in the **`embedder-api` skill**.

## Config flow

`base.toml → presets.toml[<preset>] → methods/<method>.toml → CLI args`. **Method
settings win over preset settings on overlap** (so a method can force e.g.
`blocks_to_swap=0`). `configs/base.toml` is overwritten on `make update`;
`configs/preprocess.toml` is user-owned. **`masked_loss` (off by default) is the one
masking switch** — off, `train.py` strips `mask_dir` from every subset, so a mask tree on
disk never enables masking by itself. Override modes, preprocess.toml layering, the
self-contained `configs/<method>/<method>.toml` layout, gui-methods rules:
**`configs/CLAUDE.md`**. Key-by-key semantics: `docs/guidelines/base-config.md`.

## Architecture

- **Modular `library/`** (`train_util.py` is a re-exporting facade): domain subpackages
  `anima/` (DiT model, weights, strategy), `datasets/` (`cache.py` = `CachedDataset`),
  `training/` (optimizer/scheduler/checkpoint + loss/sampler/metric registries),
  `inference/` (engine + `request.py` typed `GenerationRequest`; plug-ins split
  `corrections/` — SMC-CFG / mod-guidance / CNS — vs `editing/` — DirectEdit + postfix
  inversion), `preprocess/` (caching orchestration), `models/`,
  `vision/`, `config/`, `io/` (cache-path resolution), `runtime/` (device/offloading +
  `cli.py` argparse + `harness.py` `build_anima`).
- **Tooling layering contract**: **primitives** (`library/*` — load a model, encode a
  batch, resolve a cache path) → **façade** (`anima_lora/` — embedder entry points) →
  **orchestration** (`library/preprocess/`, `library/runtime/harness.py` — drive
  primitives over a whole dataset/run) → **entry points** (`scripts/preprocess/*.py`,
  `bench/**/run_bench.py`, `scripts/**`, `tasks.py` — thin argparse wrappers).
  `bench/`, `scripts/` are **not** installed packages (only
  `anima_lora`/`library`/`networks` are) — they keep a `sys.path` bootstrap to import
  siblings.
- **`library/qwen21/`** is the Qwen-Image-2.1 LoRA line's core — **not Anima**, none of
  the invariants below apply; read `library/qwen21/CLAUDE.md` and **load the `qwen21` skill** before touching it.
- **Strategy pattern** for tokenization/encoding (`library/anima/strategy.py`,
  `library/anima/text_strategies.py`).
- **Pluggable adapters** under `networks/`, selected via `network_module` + (LoRA family)
  the three-axis routing cfg — per-module map and dispatch invariants in
  **`networks/CLAUDE.md`**.

## Critical invariants

### Text encoder padding
The pretrained model expects max-padded text encoder outputs — zero-padded positions act
as attention sinks in cross-attention softmax. Trimming to actual text length produces
**black images**. Both training and inference must pad to `max_length` and must NOT mask
out padding via `crossattn_seqlens`. Regenerate disk-cached `.npz` after any
tokenizer/padding change.

### Free-fit native-shape bucketing — the only resize mode
Each image keeps its **native aspect ratio** and lands its token count anywhere inside
its edge tier's band (`EDGE_TOKEN_BANDS`; edges 512 768 896 1024 1280 1536); there is no
flag for it. It **requires `compile_dynamic_seq`** (auto-enabled by `train.py` with
`torch_compile`, forced in the distill loops) — without it the band becomes a static
N-graph cascade. **On-disk caches are the source of truth** for which buckets exist
(`make_buckets()` uses the cached `(W,H)`), and **`--target_res` is preprocess-only**.
Bands, tier choice, graph budget: **`bucketing` skill**.

### Lazy model loading
DiT loads AFTER text-encoder/VAE caching and unloading, to avoid OOM: text encoder →
cache → free → VAE → cache → free → load DiT → attach adapter → train.

### DiT depth is read from the checkpoint, not assumed
`load_anima_model` calls `probe_dit_arch()` (`library/anima/weights.py`) to count
`blocks.N.` in the safetensors **header** and read `model_channels` off
`x_embedder.proj.1.weight`, then builds the matching module list — so a depth-expanded
derivative loads with no flag (**Anima-2.9B** is 40 blocks vs base's 28, same TE/VAE, so
caches and preprocessing are unchanged). The count is anchored to top-level `blocks.N.`
so `llm_adapter.blocks.0..5` never inflates depth.

Consequences: **a LoRA is depth-specific** — module names carry the block index, so a
40-block adapter merged onto the 28-block base silently drops its tail blocks behind a
`not all LoRA keys are used` warning (`save_weights` stamps `ss_num_blocks` so the
mismatch is machine-detectable). Two `networks/calibration/` artifacts are depth-baked
(`channel_stats.safetensors`, `dave_alpha.npz`); **CNS γ is not**. Depth-relative knobs
use `num_blocks`, not literals. Envelope + regeneration recipes:
`docs/methods/anima-2.9b.md`.

### compile-after-apply (`build_anima`)
`torch.compile` traces the adapter's monkey-patched forward, so `compile_blocks()` MUST
run **after** `network.apply_to` + `load_weights`.
`library/runtime/harness.py::build_anima` is the shared harness encoding this ordering;
use it from `bench`/`scripts`/`preprocess` rather than open-coding load→apply→compile.

### The DiT operates on 5D latents `(B, C, T=1, H, W)` — the singleton is **dim 2**
The DiT forward (and `PatchEmbed`, which asserts `x.dim() == 5`) takes a **5D** latent
with a singleton frame axis at **dim 2**. Everything around it is 4D `(B, C, H, W)`: VAE
`encode_pixels_to_latents`, cached `.npz` latents, the training inner loop, FFT/spectral
helpers (Spectrum, CNS γ, Log-Gabor; 3D/4D), and the PE vision tower
(`encode_pe_from_imageminus1to1`, `(B,3,H,W)`). **Always `unsqueeze(2)` into the DiT and
`squeeze(2)` out** — never `squeeze()`/`squeeze(0)`, which hits batch when B=1. Bite
points: **`vae.decode_to_pixels` returns 5D `(B,3,1,H,W)` for a 5D latent** (squeeze
dim 2 before a vision tower / `F.interpolate`), and **sampler-boundary plug-ins
(SMC/CNS/SGMI/etc.) receive 5D** while reference latents they blend against are often 4D
(match ndim first — see the archived FreeText `_match_latent_ndim`).

## Methods

Read the linked deep-dive before working on a method.

**Training-free inference stacks** (Spectrum, SPD, foveated merge, SMC-CFG, CNS,
mod-guidance, embedding inversion, DAVE): [`docs/inference/`](docs/inference/README.md).
Most ride the sampler boundary and compose with any checkpoint; DAVE is a block-forward
hook. Channel scaling (per-channel LoRA gradient rebalance, on by default):
[`docs/optimizations/channel_scaling.md`](docs/optimizations/channel_scaling.md).

| Method | What it is | Gotcha / pointer |
|---|---|---|
| **DirectEdit + Anima Tagger** | Inversion + edit-conditioning swap; Tagger (`anime_tools.tagger`) maps image → Anima-format tags for ψ_src. | Edit leverage collapses if ψ_src is off-manifold — verify with `exp-test-directedit-dry`. `docs/experimental/directedit_editing_v3.md` |
| **EasyControl** | Extended self-attn image conditioning; frozen DiT, per-block cond LoRA + scalar `b_cond` gate. Source `easycontrol-dataset/`. | `docs/experimental/easycontrol.md` |
| **Soft Tokens** | SoftREPA per-layer × per-t soft text tokens (~1M params); frozen DiT, per-block `Block.forward` splice into `crossattn_emb`. | Contrastive term (`infonce` default in code; shipped config uses `softrank`, weight 0.15). `configs/methods/soft_tokens.toml` |
| **Turbo** | DP-DMD (diversity-preserved DMD) distillation; output is a normal LoRA. Shipped as `make turbo` / `make test-turbo`; published 4-step student at `huggingface.co/sorryhyun/anima-turbo-4step`. | Bespoke sectioned schema + two-optimizer loop under `scripts/distill_turbo/`, kept out of `train.py` — don't `print-config METHOD=turbo`. Honors `--queue`, writes a canonical `.snapshot.toml`. `docs/methods/turbo.md` (ops), `docs/structure/turbo.md` (structure) |
| **CJK vocab pack** | Text-encoder asset (not a LoRA): extra T5-side rows for JA / KO / ZH spans. One key — `vocab_pack` in `configs/base.toml` (**on by default since v2**; `""` = off) — drives training, TE caching, `inference.py` and `GenerationRequest`; `library/anima/vocab_pack.py` owns the strategy subclass + `llm_adapter.embed` hooks (state dict stays 32128 rows). | TE caches skip on existence only — enabling/changing a pack needs `make preprocess-te ARGS=--overwrite` for CJK captions; caches and LoRAs carry the pack digest and warn on mismatch. EN is bit-exact either way. `docs/methods/cjk_vocab_pack.md` |

## Preprocessing & scripts

Data-prep scripts in `scripts/preprocess/` are thin argparse wrappers (resize → VAE
latents → text embeddings → PE features → masks); **the caching logic lives in
`library/preprocess/`** — edit orchestration there, flags in the script. `make
preprocess-{resize,vae,te,pe}` / `make mask`. Resize is **idempotent + size-aware**
(skips images already at the correct bucket; `--overwrite` forces all).
After a `target_res` tier change, run `make preprocess-reconcile` (dry-run;
`ARGS="--delete"` to act) to drop orphaned latent npz / stale resized PNG / PE sidecar /
mask for every image whose bucket moved — TE caches are text-only and never touched.
Other utility scripts: `edit.py`, plus the `scripts/toolkits/` bundle
(`export_logs_json.py`, `merge_to_dit.py`, `merge_loras.py`, `extract_delta_lora.py`,
`comfy_batch.py`).

Caches live under `post_image_dataset/lora/`: `{stem}_{WxH}_anima.npz` (VAE),
`{stem}_anima_te.safetensors` (text), `{stem}_anima_pe.safetensors` (PE). σ-demote
siblings are **keys inside** the native VAE npz (`demoted_{H}x{W}`, one per route,
outside the latents namespace), emitted by `make preprocess-demote`. TE caching reads
**only** the revised caption beside the resized image
(`post_image_dataset/resized/**/{stem}.txt`) — no fallback to the `image_dataset/`
master. Resize moves images only; the caption stages read revised-first (master as
fallback) and always write the revised caption, so a master hand-edit reaches training
only through a caption stage run, and a dataset that skips every caption stage caches
empty prompts.

Curation **exclusion** (Image tab **Exclude (D)** / **Restore…**) is `anime_tools.exclude`
on the trainer's trees (`library/datasets/curation_actions.py`): workspace files (resized
copy, caption sidecars, mask, OCR) move under `post_image_dataset/_excluded/`, the source
under `image_dataset/` stays. That ledger, the package GUI's `workspace/_excluded` ledger
and the `skip` / `move` marks in `curation_decisions.json` are unioned into
`ResizeRequest.skip`. A `post_image_dataset/moved/` tree (pre-0.6) is inert.

### Curation lives in `anime_tools`

Caption grammar, tag taxonomy, the **Anima Tagger**, caption-master stages, **masking**
and **grouping** live in the sibling repo **https://github.com/sorryhyun/anime_tools**
(checkout `../anime_tools`). Dependency direction is **trainer → `anime_tools`, never the
reverse** (`tests/test_curation_boundary.py`). The package is a **git dependency pinned
by release tag** — an edit in `../anime_tools` is invisible to `make` targets and daemon
jobs until a tag is cut and the pin moves. **Load the `anime-tools` skill** before
importing the package, editing `scripts/tasks/` (wrappers build a request, never spell a
flag), adding or changing a stage, or bumping the pin.

### Captions

A caption may carry trailing **position clauses** (`<flat tag bag>. On the left, akita
neru, yellow eyes.`) and text clauses (`… Japanese text reads as "…"`, attached only by
the export stage's `--combine_ocr` from `post_image_dataset/ocr/{stem}.ocr.txt`). **Never
hand-split a caption** (`caption.split(",")` corrupts clauses): use
`anime_tools.captions.position_clauses` (`parse_caption` / `compose_caption`).

`make caption-autotag` and `make caption-position` are dry-run by default (`ARGS="--apply"`
writes); `make caption-full` (position → OCR read → OCR clause) writes by default
(`ARGS="--dry_run"` to plan). All three **must be followed by `make preprocess-te`**.
**Load the `captions` skill** before parsing/editing captions or running any of them.

## Custom nodes

ComfyUI nodes mostly live in standalone repos symlinked into `../comfy/custom_nodes/` —
edit the source repo, not the symlink. `_vendor/` subsets inside nodes: **regenerate with
`make vendor-sync`, never `cp` by hand**, and re-run before every node publish (see
[[feedback_vendor_sync]]). Repo map and in-tree nodes: **`custom-nodes` skill**.

## External tools

ComfyUI, SAM3, and manga-image-translator live in the parent directory (`../comfy/`,
`../sam3/`, etc.).

## Contributing

PRs follow a tier system — see `CONTRIBUTING.md`. Key constraint for code work:
numerics/efficiency changes (Tier 1.5) and new methods (Tier 2) **require a bench script +
invariant test**. Bench scripts share `bench/_common.py` and drop a `result.json`
envelope into `bench/<method>/results/<YYYYMMDD-HHMM>[-label]/`.
