---
name: captions
description: Caption pipeline — position-clause grammar (never hand-split a caption), make caption-autotag modes, make caption-position (v2 rewrite rules and gates), and the preprocess-stage wiring for both. Load before parsing/editing captions or caption code, running either target, or touching the caption preprocess stages.
---

# Caption pipeline (trainer-side wiring)

The caption code moved to the **`anime_tools`** package (curation split Phase 1,
2026-08-30 — https://github.com/sorryhyun/anime_tools, sibling checkout
`../anime_tools`). The full skill — grammar details, `--caption_drop_groups`
resolution order, autotag modes, the v2 position-clause move rules and gates,
the tuning defaults — lives there: `../anime_tools/.claude/skills/captions/SKILL.md`,
evidence in `../anime_tools/docs/position_captions.md`. **Read it before
editing caption code.** What stays trainer-side is below.

## The one rule

`<flat tag bag>. On the left, akita neru, yellow eyes. On the right, kasane teto.`
— the **period** delimits clauses, commas separate tags *inside* one. A plain
`caption.split(",")` silently corrupts clauses. **Never hand-split a caption**:
`anime_tools.captions.position_clauses` (`parse_caption` / `compose_caption`)
is the single grammar; `anime_tools.captions.shuffle` is the training-time
shuffle / `@no-artist` grammar (`library.anima.training` re-exports it).

## Trainer targets (each builds an `anime_tools` request object)

The wrappers in `scripts/tasks/preprocess.py` never spell a flag: each target builds the
stage's frozen request (`anime_tools.stages.requests`) from the config chain + GUI env
and runs it through `_common.execute_stage` — in-process under a daemon job (autotag →
position share one tagger load; `release_models()` frees it before the TE child), as a
`python -m <stage.module> *req.to_argv()` child from a shell. `ARGS` are applied through
the request's own generated parser (`request_with_args`), so every flag the stage has
still works from `make`; an unknown one fails with the stage's usage.

| Target | Request (stage id) | Notes |
|---|---|---|
| `make caption-autotag` | `AutotagRequest` (`autotag`) | dry-run default; `ARGS="--mode missing\|merge\|overwrite"`; `ARGS="--apply"` then **`make preprocess-te`**. Writes the **revised** caption (`resized/`), master read-only |
| `make caption-position` | `PositionRequest` (`position`) | SAM3 → tagger → v2 rewrite; dry-run default, GPU — route through the daemon |
| `make preprocess-captions` | `CorrectRequest` (`correct`) | corrects the revised caption in place (mirrors the master only for an image with none) + `.variants.txt` under `post_image_dataset/resized/`; `--caption_drop_groups` |
| `make caption-index` | plain CLI `anime_tools.captions.index` | `post_image_dataset/captions/caption_index.json` (`--out` spelled by the trainer) |
| `make autotag` / `make tagger*` | plain CLIs `anime_tools.tagger.cli.*` | single-image / vocab build / dbv4 ckpt |

Stage wiring (`scripts/tasks/preprocess.py`): autotag runs **first** (right after
resize, `apply=True`), then position clauses, then correction/variants, then TE —
chain order pinned by `tests/test_preprocess_tasks.py`; the request fields the trainer
sets are pinned by `tests/test_anime_tools_cli_contract.py`. Caption edits do **not**
invalidate TE caches (existence-only skip) — always re-run `make preprocess-te`
after an `--apply`; a stale `.variants.txt` keeps training the old caption. Once an
image has a revised caption, a hand-edit of its master no longer reaches it — edit the
revised caption, or delete it to re-mirror.

`configs/clause_vocabulary.yaml` is the user-editable clause policy; the package
ships an identical default used when the file is absent from the curation home.
