# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

It carries **orientation** (where things live) and **invariants** (rules you would break
without knowing you were in that territory). Task-shaped detail lives in skills — `ls
.claude/skills` — and in the nested `configs/CLAUDE.md`, `networks/CLAUDE.md`,
`gui/CLAUDE.md`. Each section below names the skill that holds the rest.

## Project Overview

Anima — LoRA/T-LoRA training and inference pipeline for the Anima diffusion model
(DiT-based, flow-matching). Supports several adapter families (LoRA / OrthoLoRA / T-LoRA
/ HydraLoRA / FeRA / ChimeraHydra / EasyControl) selectable via method config + hardware
preset. The LoRA family is routed via a three-axis surface — `use_moe_style` /
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
and `scripts/experimental_tasks/` (for `exp-*`). Don't grep the Makefile for a recipe —
look there. `make help` lists every target.

All training runs `train.py --method <name> --preset <name>`. By default it's invoked
**directly** (single-GPU fast path — skips the ~5s accelerate launcher bootstrap;
`train.py` builds its own single-process `Accelerator()` and reads `mixed_precision`
from the config chain). Set `ANIMA_ACCELERATE_LAUNCH=1` to wrap it in `accelerate
launch` for multi-GPU / distributed runs (see `build_launch_cmd` in
`scripts/tasks/_common.py`). Override any config value from CLI (`--network_dim 32
--max_train_epochs 64`) or the preset via `PRESET=low_vram make lora`. `exp-*` targets
are experimental — may break or be removed.

Non-obvious knobs and gotchas worth knowing up front:

- **Training**: `make lora PRESET=low_vram|fast_16gb|half` (half → `sample_ratio=0.5`);
  `make lora-gui GUI_PRESETS=tlora` runs the clean per-variant `configs/gui-methods/`
  tree (`ls` it for the live list). `make turbo` is the shipped DP-DMD distiller
  (promoted from `exp-turbo`); `exp-soft-tokens | exp-chimera` are the experimental
  methods.
- **`make soup PATH_PATTERN="<glob>"`** (or `TARGET=<dir>` shorthand) — uncond-init soup
  pipeline (`scripts/soup/`; GUI: Experimental tab → soup). Plain-LoRA only; quality win
  + seed-lottery insurance, NOT a memorization fix. **Load the `soup` skill** before
  running or modifying it; deep-dive `docs/experimental/soup.md`.
- **Inference compose flags**: `SPECTRUM=1` / `MOD=1` / `NOLORA=1` compose into
  **every** `test-*` target (`make test`, `test-hydra`, `test-merge`, `test-smc-cfg`,
  `test-easycontrol REF_IMAGE=…`, `exp-test-*`).
- **`make gen`** — daemon-routed batch generation (same argv + env levers as `make
  test`, submitted as a GPU command job). Eval grids / seed sweeps go here; interactive
  single images stay on `make test` or the resident inference server.
- **Daemon** (local FIFO job queue, auto-starts on first submit): **agent-launched GPU
  work must go through it** — GPU processes started from a Claude Code background Bash
  get killed by the harness sandbox layer after ~1 min (silent SIGKILL, no trace;
  observed 2026-07-25). Front door `make daemon-run ARGS="<script.py> [flags]"`, `make
  daemon-wait [JOB=<id>]` to block; append `--queue` to any train/distill target to
  enqueue. Discovery is pidfile-based — never hardcode 8765. **Load the `daemon` skill**
  for the full surface; contract in `anima_daemon/README.md`.
- **σ-demoted training** (`--sigma_lowres`, opt-in): routes each train step's latent
  grid by noise level — high-σ steps train on a lower-res sibling latent, ~−14% wall
  with the shipped **combolate** recipe (what `configs/base.toml` sets). Needs sibling
  latents precached; output is an ordinary LoRA. **Load the `sigma-lowres` skill before
  enabling/tuning it**; contract in `docs/optimizations/sigma_lowres.md`.
- **Gotchas**: `make merge ADAPTER_DIR=… [MULTIPLIER=0.8]` bakes LoRA into the DiT
  (LoRA/Ortho/T-LoRA only) and refuses Hydra-moe / postfix unless `--allow-partial`.
  `turbo` output is a normal LoRA — infer with `--infer_steps` matched to the DP-DMD
  `student_steps` rollout (currently 4) and `--cfg 1.0`. `make print-config METHOD=…
  PRESET=…` dumps the merged chain; `make test-unit` runs pytest; `ruff check . --fix &&
  ruff format .` (touched files only — see [[feedback_ruff_scope_collateral]]).
- **Run the test suite at most twice per task** (here or in `../anime_tools`): once
  after the change, once after fixing what it caught. Re-running it as a progress
  check is noise — read the failure and fix it. Needing a third run means the change
  wants rethinking, not another loop; if a run is genuinely required beyond that, say
  why. Scope a re-run to the affected file (`pytest tests/test_x.py`) rather than
  sweeping the whole suite again.

## Key entry points

| File | Purpose |
|------|---------|
| `anima_lora/__init__.py` | **Programmatic front door** — lazy re-export of the curated embedder entry points, namespaced `anima_lora.{models, inference, config, training, captioning}`. `import anima_lora` instead of reverse-engineering `main()`s; see the `embedder-api` skill |
| `examples/` | Runnable API scripts (`01`–`04` high-level flows, `05`–`06` raw primitives). `examples/README.md` is the embedder guide |
| `train.py` | `AnimaTrainer` — main training loop via HF Accelerate |
| `inference.py` | Standalone image generation (`--help` for all flags) |
| `networks/spectrum.py` | Spectrum inference acceleration |
| `gui/` | PySide6 GUI package |
| `tasks.py` | Cross-platform task runner — source of truth for every `make` target |
| `library/downloads.py` | **Model catalog** — one `Asset` per weight, grouped into packs; loaders import their default paths from here. Add a weight by adding a row, not a command (`model-catalog` skill) |
| `scripts/tasks/` + `scripts/experimental_tasks/` | Where command bodies actually live (`_common.py` = shared helpers) |

Docs: shipped method deep-dives in `docs/methods/`, experimental in
`docs/experimental/`, active proposals in `docs/proposal/`, retired material under
`_archive/`. Active promoted lines with open phases get a home under `project/<line>/`
(methods/bench/questions/roadmap digests — see `project/README.md`); successfully
completed lines move to the tracked `project/finished/<line>/` tier (verdict digest +
working tree; e.g. the ResShift SR sidecar, whose `make sr-*` targets were removed — run
its scripts directly), while killed/superseded lines go to `_archive/`.

## Programmatic API (embedders)

`uv sync` installs the repo editable, so `anima_lora` is importable anywhere. Two things
to carry everywhere: inference is **request-driven** (build a typed `GenerationRequest`,
call `.to_args()`), and repo-relative model/config paths resolve against the **repo
home**, not the CWD — **new code opening a repo-relative path must call
`resolve_under_home()`** (`library.env.anima_home()`; `ANIMA_HOME` for a relocated
checkout). The rest — namespaces, the strategy singletons, per-model path env vars — is
in the **`embedder-api` skill**.

## Config flow

Three-layer merge chain: `base.toml → presets.toml[<preset>] → methods/<method>.toml →
CLI args`. **Method settings win over preset settings on overlap**, so a method can force
its own hardware requirements (e.g. a frozen-DiT method forcing `blocks_to_swap=0`).
`configs/<method>/<method>.toml` (self-contained: method + inline dataset blueprint) is
**preferred** over the flat `configs/methods/<method>.toml` when present.

Two facts that bite from outside `configs/`:

- **`configs/base.toml` is overwritten on `make update`**; `configs/preprocess.toml` is
  user-owned and preserved — that's why `target_res` / `mask_dir` live there.
- **`masked_loss` (off by default since v2) is the one masking switch**: when it is off,
  `train.py` strips `mask_dir` from every subset and logs one line, so a mask tree left
  on disk never re-enables masking by itself. `mask_dir` is additionally gated on the
  directory existing (`resolve_configured_mask_dir`).

Everything else — the dataset-blueprint override modes, the preprocess.toml layering,
the gui-methods hardware-picker rule, per-subset `cache_dir` — is in
**`configs/CLAUDE.md`**. Key-by-key semantics for users: `docs/guidelines/base-config.md`.

## Architecture

- **Modular `library/`** (`train_util.py` is a re-exporting facade): domain subpackages
  `anima/` (DiT model, weights, strategy), `datasets/` (`cache.py` = `CachedDataset`),
  `training/` (optimizer/scheduler/checkpoint + loss/sampler/metric registries),
  `inference/` (engine + `request.py` typed `GenerationRequest`; plug-ins split
  `corrections/` — SMC-CFG / mod-guidance / CNS — vs `editing/` — DirectEdit + postfix
  inversion), `preprocess/` (caching orchestration), `models/`, `captioning/`,
  `vision/`, `config/`, `io/` (cache-path resolution), `runtime/` (device/offloading +
  `cli.py` argparse + `harness.py` `build_anima`). Full per-subpackage map in
  `docs/structure/`.
- **Tooling layering contract**: **primitives** (`library/*` — load a model, encode a
  batch, resolve a cache path) → **façade** (`anima_lora/` — embedder entry points) →
  **orchestration** (`library/preprocess/`, `library/runtime/harness.py` — drive
  primitives over a whole dataset/run) → **entry points** (`scripts/preprocess/*.py`,
  `bench/**/run_bench.py`, `scripts/**`, `tasks.py` — thin argparse wrappers).
  `scripts/preprocess/*.py` are now thin CLI shells over `library/preprocess/`.
  `bench/`, `scripts/` are **not** installed packages (only
  `anima_lora`/`library`/`networks` are) — they keep a `sys.path` bootstrap to import
  siblings.
- **Strategy pattern** for tokenization/encoding (`library/anima/strategy.py`,
  `library/anima/text_strategies.py`).
- **Pluggable adapters** under `networks/` — selected via `network_module` + (for LoRA
  family) the three-axis routing cfg. LoRA modules in `networks/lora_modules/`
  coordinated by `networks/lora_anima/`; EasyControl in `networks/methods/`; attention
  dispatcher `networks/attention_dispatch.py`; Spectrum `networks/spectrum.py`; SPD
  `networks/spd.py`. **See `networks/CLAUDE.md`** for the per-module map, three-axis
  surface, and dispatch invariants.

## Critical invariants

### Text encoder padding
The pretrained model expects max-padded text encoder outputs — zero-padded positions act
as attention sinks in cross-attention softmax. Trimming to actual text length produces
**black images**. Both training and inference must pad to `max_length` and must NOT mask
out padding via `crossattn_seqlens`. Regenerate disk-cached `.npz` after any
tokenizer/padding change.

### Free-fit native-shape bucketing — the only resize mode
Free-fit is the sole resize mode: each image keeps its **native aspect ratio** and lands
its patch-grid token count anywhere inside its edge tier's band (`EDGE_TOKEN_BANDS`;
edges 512 768 896 1024 1280 1536). There is no `freefit` flag — it's implicit, and the
discrete bucket pool + the pad-to-static path are both gone.

Three consequences that reach outside the resize code:

- **It requires `compile_dynamic_seq`** — auto-enabled by `train.py` whenever
  `torch_compile` is on, and forced in the bespoke distill loops. Without it the band
  explodes into a static N-graph cascade.
- **The on-disk caches are the source of truth** for which buckets exist —
  `make_buckets()` uses the cached `(W,H)`, nothing AR-snaps at load, and snap-era caches
  still train fine.
- **`--target_res` is preprocess-only; training never needs it.** The dynamo budget is
  derived from the buckets the filtered images actually populate, plus sample-prompt
  resolutions when sampling is on.

Bands, tier choice, the frozen 1024 band, `_native_flatten`: **`bucketing` skill**.

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
`library/runtime/harness.py::build_anima` is the shared harness encoding this ordering
(promoted from `bench/_anima.py`); use it from `bench`/`scripts`/`preprocess` rather
than open-coding load→apply→compile.

### The DiT operates on 5D latents `(B, C, T=1, H, W)` — the singleton is **dim 2**
The DiT forward (and `PatchEmbed`, which `assert x.dim() == 5`) takes a **5D** latent
with a singleton temporal/frame axis at **dim 2** (`T=1` for images — Anima reuses a
video-shaped layout). Everything *around* the DiT is 4D `(B, C, H, W)`: VAE
`encode_pixels_to_latents` returns 4D, cached `.npz` latents are 4D, the training inner
loop works in 4D, FFT/spectral helpers (Spectrum, CNS γ, Log-Gabor) want 3D/4D
`(C,H,W)`/`(B,C,H,W)`, and the vision tower (PE-Core `encode_pe_from_imageminus1to1`)
wants 4D `(B,3,H,W)`. So the boundary dance is **always `unsqueeze(2)` going into the
DiT and `squeeze(2)` coming out** — target **dim 2 explicitly**, never
`squeeze()`/`squeeze(0)` (which silently hits batch when B=1 and corrupts the layout).
Two recurring bite points: **`vae.decode_to_pixels` returns 5D `(B,3,1,H,W)` when fed a
5D latent** (squeeze dim 2 before handing RGB to a vision tower / `F.interpolate`), and
**sampler-boundary plug-ins (SMC/CNS/SGMI/etc.) receive 5D** while any reference latent
they blend against is often 4D (match ndim first — see the archived FreeText
`_match_latent_ndim`). Mishandling dim 2 was a repeated source of subtle freetext bugs.

## Methods

Adapter families (training methods) below — one-line orientation plus the load-bearing
gotcha; read the linked deep-dive before working on one.

**Training-free inference stacks** (Spectrum, SPD, foveated merge, SMC-CFG, CNS,
mod-guidance, embedding inversion, DAVE) are documented separately under
[`docs/inference/`](docs/inference/README.md) — read the relevant doc when you touch one
rather than carrying their details here. Most ride on the sampler boundary and compose
with any checkpoint (DAVE is the exception — a block-forward hook for same-prompt
diversity). Channel scaling (per-channel LoRA gradient rebalance, on by default) is a
training-time feature — see
[`docs/optimizations/channel_scaling.md`](docs/optimizations/channel_scaling.md); note
it's exactly inert on frozen-basis ortho variants.

| Method | What it is | Gotcha / pointer |
|---|---|---|
| **DirectEdit + Anima Tagger** | Inversion + edit-conditioning swap; Tagger (`anime_tools.tagger`) maps image → Anima-format tags for ψ_src. | Edit leverage collapses if ψ_src is off-manifold — verify with `exp-test-directedit-dry`. `docs/experimental/directedit_editing_v3.md` |
| **EasyControl** | Extended self-attn image conditioning; frozen DiT, per-block cond LoRA + scalar `b_cond` gate. Source `easycontrol-dataset/`. | `docs/experimental/easycontrol.md` |
| **Soft Tokens** | SoftREPA per-layer × per-t soft text tokens (~1M params); frozen DiT, per-block `Block.forward` splice into `crossattn_emb`. | InfoNCE objective intentionally skipped. `configs/methods/soft_tokens.toml` |
| **ChimeraHydra** | Dual-pool additive MoE: content pool (ContentRouter on pooled `crossattn_emb`) + freq pool (FreqRouter on FEI+σ), two A's per Linear off disjoint SVD subspaces. Both pools always centered-gate. | T-LoRA mask hits content branch only. `docs/experimental/chimera-hydra.md`, `networks/lora_modules/chimera.py` |
| **Turbo** | DP-DMD (diversity-preserved DMD) distillation; output is a normal LoRA. Shipped as `make turbo` / `make test-turbo`; published 4-step student at `huggingface.co/sorryhyun/anima-turbo-4step`. | Bespoke sectioned schema + two-optimizer loop under `scripts/distill_turbo/`, kept out of `train.py` — don't `print-config METHOD=turbo`. Honors `--queue`, writes a canonical `.snapshot.toml`. `docs/methods/turbo.md` (ops), `docs/structure/turbo.md` (structure) |
| **CJK vocab pack** | Text-encoder asset (not a LoRA): extra T5-side rows for JA / KO / ZH spans. One key — `vocab_pack` in `configs/base.toml` (**on by default since v2**; `""` = off) — drives training, TE caching, `inference.py` and `GenerationRequest`; `library/anima/vocab_pack.py` owns the strategy subclass + `llm_adapter.embed` hooks (state dict stays 32128 rows). | TE caches skip on existence only — enabling/changing a pack needs `make preprocess-te ARGS=--overwrite` for CJK captions; caches and LoRAs carry the pack digest and warn on mismatch. EN is bit-exact either way. `docs/methods/cjk_vocab_pack.md` |

## Preprocessing & scripts

Data-prep scripts in `scripts/preprocess/` are thin argparse wrappers (resize → VAE
latents → text embeddings → PE features → masks); **the caching logic lives in
`library/preprocess/`** — edit orchestration there, flags in the script. `make
preprocess-{resize,vae,te,pe,pooled}` / `make mask`. Resize is **idempotent +
size-aware** (skips images already at the correct bucket; `--overwrite` forces all).
After a `target_res` tier change, run `make preprocess-reconcile` (dry-run;
`ARGS="--delete"` to act) to drop orphaned latent npz / stale resized PNG / PE sidecar /
mask for every image whose bucket moved — TE caches are text-only and never touched.
Other utility scripts: `edit.py`, plus the `scripts/toolkits/` bundle
(`export_logs_json.py`, `merge_to_dit.py`, `merge_loras.py`, `extract_delta_lora.py`,
`comfy_batch.py`).

Caches live under `post_image_dataset/lora/`: `{stem}_{WxH}_anima.npz` (VAE),
`{stem}_anima_te.safetensors` (text), `{stem}_anima_pe.safetensors` (PE). σ-demote
siblings are **keys inside** the native VAE npz (`demoted_{H}x{W}`, one per route,
outside the latents namespace) — not separate files; `make preprocess-demote` emits
them. TE caching reads the **revised** caption beside the resized image
(`post_image_dataset/resized/**/{stem}.txt`, written by every caption stage and mirrored
from the `image_dataset/` master only while no revised caption exists — anime_tools ≥
0.4.0 is revised-first, so once an image has a revised caption a hand-edit of its master
no longer reaches it: edit the revised caption, or delete it to re-mirror); training
reads only cached embeddings.

### Curation lives in `anime_tools`

The caption grammar, tag taxonomy, the **Anima Tagger**, the caption-master stages,
**masking** (SAM3 / merge) and **grouping** live in the sibling repo
**https://github.com/sorryhyun/anime_tools** (package `anime_tools`, checkout
`../anime_tools`; contract at `../anime_tools/docs/contract.md`). Dependency direction is
**trainer → `anime_tools`, never the reverse** (`tests/test_curation_boundary.py`). The
package is a **git dependency pinned by rev** — an edit in `../anime_tools` is invisible
to `make` targets and daemon jobs until the pin is bumped.

The typed **request API is the front door**: one frozen request dataclass per stage,
registered in `anime_tools.stages.registry`; the `make` targets keep their names and the
wrappers in `scripts/tasks/` **build a request and never spell a flag**.
The Phase-3 import shims are gone — import `anime_tools` directly.

**Load the `anime-tools` skill** before importing the package, editing `scripts/tasks/`,
adding or changing a stage, or bumping the pin.

### Captions: grammar, autotag, position clauses

A caption may bind attributes to subjects with trailing **position clauses** (`<flat tag
bag>. On the left, akita neru, yellow eyes.`) — the period delimits clauses, commas
separate tags inside one, so a plain `caption.split(",")` silently corrupts them.
**Never hand-split a caption**: `anime_tools.captions.position_clauses` (torch-free) is
the single grammar (`parse_caption` / `compose_caption`).

`make caption-autotag` batch-tags the dataset; `make caption-position` generates position
clauses. Both are dry-run by default, and an `ARGS="--apply"` **must** be followed by
`make preprocess-te`. **Load the `captions` skill** before parsing/editing captions or
running either target.

## Custom nodes

ComfyUI nodes mostly live in standalone repos, symlinked into `../comfy/custom_nodes/` —
edit the source repo, not the symlink. In-tree under `custom_nodes/`:
`comfyui-anima-directedit/`, `comfyui-anima-register/`, `comfyui-anima-trainer/`.
Several nodes carry a `_vendor/` subset of the live tree: **regenerate with `make
vendor-sync`, never `cp` by hand** — re-run before every node publish (see
[[feedback_vendor_sync]]). Full repo map: the **`custom-nodes` skill**.

## External tools

ComfyUI, SAM3, and manga-image-translator live in the parent directory (`../comfy/`,
`../sam3/`, etc.).

## Contributing

PRs follow a tier system — see `CONTRIBUTING.md`. Key constraint for code work:
numerics/efficiency changes (Tier 1.5) and new methods (Tier 2) **require a bench script +
invariant test**. Bench scripts share `bench/_common.py` and drop a `result.json`
envelope into `bench/<method>/results/<YYYYMMDD-HHMM>[-label]/`.
