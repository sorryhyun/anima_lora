# CLAUDE.md

Orientation and invariants for this repository. Task-shaped detail lives in skills (`ls
.claude/skills`) and in the nested `CLAUDE.md` files: `library/moss/`, `library/qwen21/`,
`configs/`, `networks/`, `gui/`.

## What this repo is

A LoRA training and inference workbench for several diffusion models, sharing one GPU
job daemon, one curation package (`anime_tools`), one i2i staging tree and one GUI.
Each model is a **line** with its own code, caches and invariants; a LoRA from one line
never loads on another.

| Line | Status | Core | Read first |
|---|---|---|---|
| **MOSS-Image** | **primary** — most active work | `library/moss/`, `scripts/moss/`, `configs/moss/`, `gui/moss/` | `library/moss/CLAUDE.md`, `docs/moss/README.md` |
| **Anima** | experimental — research bed (CJK vocab lines, distillation, inference stacks) | `train.py`, `inference.py`, `library/anima/`, `networks/`, `configs/{base,presets}.toml` + `configs/methods/` | [Anima line](#anima-line-experimental) below |
| **Qwen-Image-2.1** | side line | `library/qwen21/`, `scripts/qwen21/`, `gui/qwen21/` | `library/qwen21/CLAUDE.md` + the `qwen21` skill |

The repo, package (`anima_lora`), daemon and env vars (`ANIMA_HOME`, …) keep the Anima
name. **Check which line a file belongs to before applying an invariant** — Anima's
(max-padded TE, 5D latents, free-fit bands) are wrong for MOSS and Qwen.

## Setup

```bash
uv sync                               # Python 3.13; installs the repo editable
hf auth login
make download-model ARGS=moss-base    # MOSS DiT + Qwen3.5-9B TE + FLUX.2 VAE + tokenizer (~27 GB)
make download-models                  # Anima first-run set (DiT, Qwen3-0.6B TE, VAE, PE, CJK vocab pack, tagger)
```

Run Python as `.venv/bin/python` (or `uv run --no-sync`); a bare `uv run` re-resolves and
can fail on the torch index split. Weights are rows in the model catalog
(`library/downloads.py`; `make download-list`) — **load the `model-catalog` skill**
before adding or moving a weight or changing a loader's default path.

## Running work

- **`tasks.py` is the source of truth** for every `make` target (the `Makefile` only
  forwards); bodies live in `scripts/tasks/` and `scripts/experimental_tasks/` (`exp-*`,
  may break or be removed). `make help` lists them all.
- **Agent-launched GPU work goes through the daemon** (auto-starts on first submit): a
  GPU process started from a Claude Code background Bash is SIGKILLed by the harness
  sandbox after ~1 min, no trace. `make daemon-run ARGS="[--stall-timeout S] <script.py>
  [flags]"` (daemon-run's own flags go *before* the script path); `make daemon-wait`
  blocks until terminal; `make run-status` follows a run's `progress.jsonl`. **Load the
  `daemon` skill** before launching or monitoring a GPU job.
- **Lint only the files you touched**: `ruff check --fix <files>` (a repo-wide F401
  autofix strips the re-exports `library/train_util.py` and `anima_lora/__init__.py`
  exist for). Run `ruff format <files>` only right before a commit — mid-task it reflows
  lines and breaks the next patch.
- **Run the test suite at most twice per task** (here or in `../anime_tools`): once after
  the change, once after fixing what it caught, scoped to the affected test file. Say why
  if a third run is needed. `make test-unit` runs pytest.
- New code opening a repo-relative path calls `resolve_under_home()` (`library.env`;
  `ANIMA_HOME` relocates the checkout) — paths resolve against the repo home, not the CWD.

## MOSS-Image line (primary)

MOSS-Image (OpenMOSS, Apache-2.0): a 5B single-stream DiT (42 blocks) on a frozen
Qwen3.5-9B text encoder and the frozen FLUX.2 VAE. Code map, module roles and the full
"what does not carry over from Anima" table: **`library/moss/CLAUDE.md`**. User doc:
`docs/moss/README.md` (GUI walkthrough: `docs/guidelines/moss-guidebook.md`).

Running it:

- `make gui-moss` (or the Anima / MOSS-Image mode switch in `make gui`) submits cache /
  train / test as daemon jobs. From the CLI the same jobs are
  `make daemon-run ARGS="--stall-timeout 900 scripts/moss/{cache,train,generate,tune_swap}.py …"`.
- **Every flag and default is defined once** in `library/moss/requests.py` (torch-free
  request dataclasses). `--config <file.toml>` takes a flat TOML of field names; order is
  field default → `--config` → CLI. Shipped defaults: `configs/moss/{cache,lora}.toml`.
  None of Anima's `base → preset → method` chain, `make lora` or `masked_loss` applies.
- **i2i**: `make i2i-stage TASK=<task>` → `make i2i-preprocess TASK=<task> MODEL=moss`
  → train / generate with `configs/moss/i2i/<task>/{cache,train,generate}.toml`. Tasks,
  ref flags (`ref_timestep` / `ref_isolate` are stamped in the LoRA) and status:
  `docs/moss/methods/i2i.md`.
- Caches land in `post_image_dataset/moss[/<set>]` (flat, two files per image), LoRAs in
  `output/ckpt_moss/` (i2i: `output/ckpt_moss_i2i/`).

Invariants that bite:

- **Never pad the TE.** Each prompt is encoded alone at its true length (Qwen3.5's GDN
  layers misbehave on padding, and RoPE places the image at frame position = text
  length); `forward_batched` masks padded keys instead.
- **Latents are 4D packed `(B, 128, H/16, W/16)`**, normalised by the VAE's BatchNorm
  buffers — no frame axis. Image sides are multiples of 16 at native aspect.
- **Sampler knobs come from the reference pipeline** (`pipeline_moss_image.py` in the
  model folder): CFG 3, the uncond branch skips block 22, shift 5 snapped to the training
  σ grid, timestep input `1000·σ`. Read the pipeline before changing one; `model.py`'s
  unbatched forward stays op-for-op with it.
- **The default TE is INT8** (`moss_te_int8`; `fp8` / `bf16` selectable). TE caches stamp
  `encoder` and `caption_format` (`layout_json` by default, via
  `anime_tools.captions.layout_json`) and re-encode when either changes — **cache, train
  and generate with the same `text_encoder` and caption/prompt format.**
- **Training defaults to an int8 DiT base** (`int8 = stream`, GPTQ file `moss_dit_int8`,
  fp32 stream, swap 0, budget 0.3, EMA on the card; `library/moss/dit_int8/`). It refuses
  LoHa, and the i2i configs stay on bf16: `configs/moss/lora_bf16.toml` is the bf16 plan.
  Block swap runs on it and trades VRAM, not speed. One LoRA has completed on it (848 steps, ama mitsuki; `docs/moss/optimization/int8.md`).
- Adapters live in `library/moss/lora.py`, held outside the blocks (the block swapper
  moves every `.weight` under a block); `networks/` is bound to Anima module names.
- Decode with the DiT off the card — the fp32 VAE decode at ~2 MP does not fit beside it.

Research log (finished 2026-10-28): `project/finished/moss_lora/README.md` (settled logs in
`done/`, parked ones in `_archive/moss_lora/`); A/B and i2i eval scripts: `bench/moss_ab/`,
`bench/moss_i2i_eval/`; findings: `docs/moss/findings/`. ComfyUI: the MOSS Toolkit and
the Spectrum pack's **KSampler (Spectrum for MOSS)** (`custom-nodes` skill).

## Shared infrastructure

### Layering

**Primitives** (`library/*` — load a model, encode a batch, resolve a cache path) →
**façade** (`anima_lora/`, Anima embedder entry points; `embedder-api` skill) →
**orchestration** (`library/preprocess/`, `library/runtime/harness.py`) → **entry
points** (`scripts/**`, `bench/**/run_bench.py`, `tasks.py` — thin argparse wrappers).
`bench/` and `scripts/` are packages of the editable install: import them as
`bench.<dir>.<mod>` / `scripts.<dir>.<mod>`, no `sys.path` bootstrap. Repo-root modules
(`train.py`, `inference.py`, `tasks.py`) and `i2i_tasks/` are not installed.

Shared across lines: `library.runtime.{offloading,blockswap,dynamo}`, `library.env`,
`library.int8` (the int8 base linears),
the ER-SDE sampler and CNS core (`library.inference.sampling`,
`library.inference.corrections.cns_core`), `library.lines` (request base, LoRA / LoKr /
LoHa adapters and GUI scan counts for MOSS and Qwen). Nothing Anima-side imports
`library.moss` (`tests/test_moss_boundary.py`).

### Curation lives in `anime_tools`

Caption grammar, tag taxonomy, the Anima Tagger, caption stages, layout-JSON conversion,
masking and grouping live in the sibling repo **https://github.com/sorryhyun/anime_tools**
(checkout `../anime_tools`). Dependency direction is **trainer → `anime_tools`, never the
reverse** (`tests/test_curation_boundary.py`). It is a **git dependency pinned by release
tag** (`pyproject.toml`) — an edit in `../anime_tools` is invisible to `make` targets and
daemon jobs until a tag is cut and the pin moves. **Load the `anime-tools` skill** before
importing the package, editing `scripts/tasks/` (wrappers build a request, never spell a
flag), adding or changing a stage, or bumping the pin.

The trainer GUI does not curate: its anime_tools tab embeds the package's web panel,
which curates in `workspace/` and exports captions / masks / the revised master (never
images). Order: Export → preprocess. Exclusions are unioned into `ResizeRequest.skip`
(`library/datasets/curation_actions.py`).

### Captions

A caption may carry trailing **position clauses** (`<tag bag>. On the left, akita neru,
yellow eyes.`) and OCR text clauses (`… Japanese text reads as "…"`). **Never hand-split
a caption** (`caption.split(",")` corrupts clauses) — use
`anime_tools.captions.position_clauses` (`parse_caption` / `compose_caption`). The same
tag captions feed MOSS, mapped onto its layout-JSON prompt at cache time.

`make caption-autotag` / `caption-position` are dry-run by default (`ARGS="--apply"`
writes); `make caption-full` writes by default (`ARGS="--dry_run"` plans). After any of
them, re-cache the TE (`make preprocess-te` for Anima, the MOSS `cache` job). **Load the
`captions` skill** before parsing/editing captions or running these.

### i2i staging

`i2i_tasks/<task>` is model-agnostic: `make i2i-stage TASK=…` writes a pair tree under
`post_image_dataset/i2i/` that both MOSS i2i and Anima's EasyControl read. Configs
`i2i_tasks/configs/<task>.toml`; map `i2i_tasks/README.md`.

### GUI and ComfyUI nodes

PySide6 under `gui/` (`make gui`; `make gui-moss` / `gui-qwen` open the line windows,
`gui/lines/`). Rules and recipes: `gui/CLAUDE.md` and the `gui-changes` skill.
ComfyUI (`../comfy/`), SAM3 etc. live in the parent directory. ComfyUI nodes mostly live
in standalone repos symlinked into `../comfy/custom_nodes/` — edit the source repo, not
the symlink. `_vendor/` subsets: **regenerate with `make vendor-sync`, never `cp` by
hand**, and before every node publish. Repo map: **`custom-nodes` skill**.

## Anima line (experimental)

Anima: DiT-based flow-matching model (28 blocks base; 40-block **Anima-2.9B**
derivative), Qwen3-0.6B TE bridged into the DiT by `llm_adapter`, whose 32128-row
T5-vocab embedding table is what the CJK vocab pack extends. Adapter families (LoRA /
T-LoRA / HydraLoRA / EasyControl) live in `networks/` (dispatch map: `networks/CLAUDE.md`;
routing axes: `lora-routing` skill).

### Commands

- Training: `train.py --method <name> --preset <name>` via `make lora [PRESET=…]`
  (`configs/presets.toml`), `make lora-gui GUI_PRESETS=<variant>`
  (`configs/gui-methods/`). Config chain `base.toml → presets.toml[<preset>] →
  methods/<method>.toml → CLI`; **method wins over preset**. `configs/base.toml` is
  overwritten by `make update`; `configs/preprocess.toml` is user-owned. `masked_loss`
  (off) is the one masking switch — a mask tree on disk never enables it. Details:
  **`configs/CLAUDE.md`**; `make print-config METHOD=… PRESET=…` dumps the merged chain.
  Train targets run through the daemon (attach by default); `--queue` detaches,
  `--inline` bypasses it; `ANIMA_ACCELERATE_LAUNCH=1` wraps in `accelerate launch`
  for multi-GPU (forces inline).
- `make turbo` — DP-DMD distiller: bespoke loop under `scripts/distill_turbo/`, so
  `train.py` infra (compile cache, `--deterministic`, `print-config`) does not reach it.
  Output is a normal LoRA — infer with `--infer_steps 4 --cfg 1.0` (`make test-turbo`).
  `docs/anima/methods/turbo.md`.
- `make soup TARGET=<artist>` — uncond-init soup, plain LoRA only; **load the `soup`
  skill** first.
- Inference: `make test` (interactive) / `make gen` (daemon batch: grids, seed sweeps).
  `SPECTRUM=1` / `MOD=1` / `NOLORA=1` compose into every `test-*` target.
- `--int8 fwd|full|stream` (off by default; `inference.py --int8`, `INT8=1` on `test-*`) runs the
  frozen DiT's large linears as int8 GEMMs: LoRA-family networks, no block swap. Forward
  copies come from `<dit>_int8_gptq.safetensors` beside the DiT (catalog `anima_dit_int8`,
  fetched on first use), else round to nearest. `docs/anima/optimizations/int8.md`.
- `make merge ADAPTER_DIR=… [MULTIPLIER=0.8]` bakes a LoRA / T-LoRA into the DiT;
  refuses Hydra-moe / postfix unless `--allow-partial`.
- `--sigma_lowres` (σ-demoted training, opt-in): the research line is archived but the
  feature is live; needs `make preprocess-demote` siblings. **Load the `sigma-lowres`
  skill** before enabling it.

### Invariants

- **TE padding**: Anima expects **max-padded** TE outputs — zero-padded positions are
  attention sinks in cross-attention. Trimming to text length gives **black images**.
  Pad to `max_length` in training and inference and do not mask padding via
  `crossattn_seqlens`. Regenerate cached TE outputs after any tokenizer/padding change.
- **5D latents `(B, C, T=1, H, W)`, singleton at dim 2.** Everything around the DiT is
  4D (VAE encode, cached latents, the training loop, spectral helpers, the PE tower).
  `unsqueeze(2)` in, `squeeze(2)` out — never `squeeze()` / `squeeze(0)` (hits batch at
  B=1). `vae.decode_to_pixels` returns 5D for a 5D latent; sampler-boundary plug-ins
  receive 5D while reference latents are often 4D (match ndim first).
- **Free-fit native-shape bucketing** is the only resize mode: native aspect, token count
  inside its edge tier's band (`EDGE_TOKEN_BANDS`, edges 512…1536). Requires
  `compile_dynamic_seq` (auto-on with `torch_compile`). On-disk caches decide which
  buckets exist; `--target_res` is preprocess-only. **`bucketing` skill.**
- **Lazy loading**: TE → cache → free → VAE → cache → free → DiT → adapter → train.
- **compile after apply**: `compile_blocks()` runs after `network.apply_to` +
  `load_weights`. Use `library/runtime/harness.py::build_anima` rather than open-coding
  load → apply → compile.
- **Depth comes from the checkpoint** (`probe_dit_arch()` in `library/anima/weights.py`
  counts top-level `blocks.N.`). A LoRA is depth-specific: a 40-block adapter on the
  28-block base drops its tail behind a `not all LoRA keys are used` warning
  (`ss_num_blocks` is stamped). `channel_stats.safetensors` and `dave_alpha.npz` in
  `networks/calibration/` are depth-baked; CNS γ is not. Depth-relative knobs use
  `num_blocks`. `docs/anima/methods/anima-2.9b.md`.

### Preprocessing and caches

`make preprocess` = resize → autotag → VAE → position clauses → TE → caption index
(→ REPA PE with `use_repa`); single stages `make preprocess-{resize,vae,te,pe}`,
`make mask`. Orchestration lives in `library/preprocess/`, flags in
`scripts/preprocess/`. Resize is idempotent and size-aware; after a `target_res` tier
change run `make preprocess-reconcile` (`ARGS="--delete"` to act). Caches in
`post_image_dataset/lora/`: `{stem}_{WxH}_anima.npz`, `{stem}_anima_te.safetensors`,
`{stem}_anima_pe.safetensors`. TE caching reads **only** the revised caption beside the
resized image (`post_image_dataset/resized/**/{stem}.txt`), no fallback to the
`image_dataset/` master, and a missing sidecar encodes an empty caption — so a master
hand-edit reaches training only through a caption stage, and resize + `preprocess-te`
alone caches empty prompts. TE caches invalidate on caption mtime only.

### Methods

Read the deep-dive before working on one (`docs/anima/methods/`, `docs/anima/experimental/`).
Training-free inference stacks (Spectrum, SMC-CFG, CNS, mod-guidance, DAVE, …):
`docs/anima/inference/`. Channel scaling (on by default): `docs/anima/optimizations/channel_scaling.md`.

| Method | Gotcha / pointer |
|---|---|
| **CJK vocab pack** — T5-side rows for JA / KO / ZH spans; `vocab_pack` in `configs/base.toml` (on by default) drives training, TE caching and inference | Changing the pack is invisible to the mtime check — `make preprocess-te ARGS=--overwrite`; TE caches carry the pack digest. `docs/anima/methods/cjk_vocab_pack.md`. Active pack work: `project/cjk_anima_reseed/`, `project/cjk_ko/` (live on branches `cjk-reseed` / `cjk-ko`) |
| **Turbo** — DP-DMD, 4-step student (`huggingface.co/sorryhyun/anima-turbo-4step`) | see Commands; `docs/anima/structure/turbo.md` |
| **EasyControl** — extended self-attn image conditioning, per-block cond LoRA + `b_cond` gate | source `easycontrol-dataset/` or the i2i staging tree. `docs/anima/experimental/easycontrol.md` |
| **Soft Tokens** — per-layer × per-t soft text tokens | shipped config uses `softrank` (code default `infonce`). `configs/methods/soft_tokens.toml` |
| **DirectEdit** — inversion + edit-conditioning swap | collapses if ψ_src is off-manifold; check with `exp-test-directedit-dry`. `docs/anima/experimental/directedit_editing_v3.md` |

## Projects, docs, history

- Active research lines: `project/<line>/` (`project/README.md` lists them). **A line's
  live work can sit on its own branch** — run `git branch -a` and check before editing a
  `project/<line>/`.
- Finished lines: `project/finished/` (code search skips most of it); retired lines and
  benches: the gitignored `_archive/`. Ask the **`archive-explorer`** agent about either
  instead of grepping. Proposals: `docs/{moss,anima}/proposal/`.
- Docs: one folder per line (`docs/moss/`, `docs/anima/`); docs shared across lines
  (installation, guidebooks, CNS) sit at `docs/` — index `docs/README.md`.
- `CONTRIBUTING.md`: numerics/efficiency changes (Tier 1.5) and new methods (Tier 2) need
  a bench script + invariant test. Benches share `bench/_common.py` and write a
  `result.json` envelope to `bench/<name>/results/<YYYYMMDD-HHMM>[-label]/`.
