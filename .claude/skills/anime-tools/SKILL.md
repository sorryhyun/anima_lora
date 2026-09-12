---
name: anime-tools
description: The trainer ↔ anime_tools boundary — what the curation split moved out, the typed request/stage API the make targets build, the git-pin dev loop and its stale-venv trap, and the tests that guard the contract. Load before importing anime_tools, editing scripts/tasks/, adding or changing a stage, bumping the pinned tag, or debugging a stage's argv.
---

# Curation lives in `anime_tools`

Sibling repo **https://github.com/sorryhyun/anime_tools** (package `anime_tools`,
checkout `../anime_tools`). Per-feature contracts: `../anime_tools/docs/`. The package
carries its own skills — read them before editing package code:
`../anime_tools/.claude/skills/{captions,add-stage,model-catalog,release}/SKILL.md`.

## What lives there (split Phases 1–3b, 2026-08-30)

Caption grammar, tag taxonomy/correction, variants sidecars, caption index, the **Anima
Tagger**, the caption-master stages (autotag / position clauses / multiview audit),
**masking** (SAM3 / merge), **grouping** (PE-Spatial near-twin features → `groups.json`),
and the tagger-only benches + their gitignored training history.

**Dependency direction is trainer → `anime_tools`, never the reverse.**
`tests/test_curation_boundary.py` guards the trainer side; the package guards itself.

What stays trainer-side:

- **GUI panels** — they reach the package only via the `autotag_server` stdio protocol,
  daemon jobs, and the torch-free grammar.
- `library/models/pe.py` — a permanent re-export. **The vendored PE vision tower is owned
  by `anime_tools.vision.pe`**, so a standalone `anime_tools` groups with the same
  PE-Spatial-B16-512 the trainer uses. The `library/vision/` encoder/bucket registry
  stays here.
- `configs/clause_vocabulary.yaml` — the user-editable override of the package default.
- `sam3` as a direct dep — now redundant (its last consumer moved to
  `_archive/bench/position_captions/`; `anime-tools` declares sam3 itself).
  `segmentation-models-pytorch` rides only on `anime-tools[masking]`.
- Benches that use the tagger as a *judge of DiT output*.

**Phase 3 deleted the `library._moved` shims and every forwarding shell** —
`library.captioning.*`, `library.preprocess.{caption_variants,autotag,position_captions,…}`,
`library.vision.{pe_features,pe_matching,grouping_embedder}`, `library.datasets.grouping`,
the `scripts.anima_tagger` / `scripts.curate` script dirs, and the
`scripts.preprocess.{autotag_captions,position_captions,correct_captions,generate_masks*,merge_masks,probe_*,build_caption_index,audit_*,…}`
shells no longer exist. Import `anime_tools` directly.

## The typed request API is the front door

API-first migration (T0–T6, landed 2026-09-03; the proposal was retired once
complete):
one **frozen request dataclass per stage** (`ResizeRequest`, `AutotagRequest`,
`PositionRequest`, `CorrectRequest`, `GroupRequest`, `SamMaskRequest`, …), registered in
`anime_tools.stages.registry` with a lazy `Stage.runner()`. `python -m
anime_tools.<pkg>.cli.<name>` is its shell — use it directly for the tagger CLIs that
have no request yet (`make daemon-run ARGS="-m anime_tools.tagger.cli.train_sidecar …"`).

The `make` targets (`preprocess-resize`, `mask`, `curate-group`, `caption-*`, `tagger`,
`autotag`) keep their names and `--queue` routing. The rule for the wrappers in
`scripts/tasks/`: **build a request, never spell a flag.** `_common.execute_stage` runs it

- **in-process** under a daemon job — one interpreter, one tagger/SAM3 load shared across
  consecutive stages, released before a trainer child via `release_models()`; or
- as a **`python -m <stage.module> *req.to_argv()` child** from a shell.

A user's `ARGS` reach a stage through the request's own generated parser
(`request_with_args`), so every flag the stage has still works from `make` and an unknown
one fails with the stage's usage. `run()` and the in-process path both export
`ANIMA_HOME` so the package's bare relative defaults anchor on this checkout
(`ANIME_TOOLS_HOME` → `ANIMA_HOME` → CWD).

Guard: `tests/test_anime_tools_cli_contract.py` re-parses every emitted argv through the
stage's parser and is the drift alarm. There is no contract-version handshake any more —
the release tag is the version, and a surface change shows up as a failing contract row
(or a `TypeError` on a request field) when the pin moves.

Adding a stage or a flag? Follow `../anime_tools/.claude/skills/add-stage/SKILL.md`, then
add the trainer-side wrapper + a contract-test row here.

## The pin, and the trap it sets

It is a **git dependency, not PyPI**. `pyproject.toml` pins a **release tag** (`tag =
"vX.Y.Z"`) under `[tool.uv.sources]` via the default-on `anime-tools-git` group — cut with
the package's `release` skill (version bump → annotated tag → `release.yml`).

**The trainer `.venv` holds the pinned copy, not `../anime_tools`.** An edit in the
sibling checkout is invisible to `make` targets, daemon jobs and the GUI until a tag is
cut and the pin moves — this has silently run stale package code on the GPU before. `python -c "import
anime_tools"` with cwd=`../anime_tools` lies (sys.path[0]); check from the trainer root.

- **Ship a package change**: release it upstream (tag pushed), move the `tag` in
  `pyproject.toml`, then `uv lock --upgrade-package anime-tools && uv sync`.
- **The package's `[tool.uv.sources]` leak into this lock.** uv honors a git dependency's
  own sources, so a torch index pinned upstream (v0.6.1's win32 cu132 source) collides with
  the trainer's `rocm-windows` group at `uv lock`. Any upstream torch source must be
  extra/group-conditioned so a consumer never sees it.
- **Live dev loop** against the checkout: `uv sync --no-group anime-tools-git --group
  anime-tools-dev` (the two groups conflict by design, like `cuda-windows` /
  `rocm-windows`).
- **Smoke an unpushed change on the GPU**: submit with
  `DaemonClient.submit_command(argv=["-m", ...], extra_env={"PYTHONPATH":
  "/home/sorryhyun/anima/anime_tools"})` — `PYTHONPATH` is not in the daemon's
  captured-env whitelist, so `make daemon-run` cannot pass it.
- `uv sync` also **prunes ad-hoc-installed packages** — reinstall anything you added by
  hand after a sync.

## Moving more code into the package

- The package's ruff rules are stricter than the trainer's (SIM115, DTZ005, EXE001…) —
  expect to fix those on any move.
- `tests/test_doc_refs.py` treats any slash-path token in a tracked `.md` as a live path.
  Mention deleted files in **dotted-module form**, not as paths.
- `../anime_tools/bench/_common.py` is a *copy* of the trainer's envelope helper; a probe
  there that reads the trainer's results dir must be run from this checkout.
