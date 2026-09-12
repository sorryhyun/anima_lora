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
| `make caption-full` | `PositionRequest` → `OcrRequest` (`ocr`) → `ExportRequest` (`export`) | the whole derived-caption chain, one daemon job (re-enters `tasks.py caption-full --inline` inside it, so the stages share a process). **Applies by default** (`--dry_run` to plan) — every step writes the derived tree, which the master's dry-run guard exists to protect. `--skip_position` / `--skip_ocr` re-combine from the sidecars already read; `--ocr_min_det` / `--ocr_min_glyph` are the floors |
| `make preprocess-captions` | `CorrectRequest` (`correct`) | corrects the revised caption in place (mirrors the master only for an image with none) + `.variants.txt` under `post_image_dataset/resized/`; `--caption_drop_groups` |
| `make caption-index` | plain CLI `anime_tools.captions.index` | `post_image_dataset/captions/caption_index.json` (`--out` spelled by the trainer) |
| `make autotag` / `make tagger*` | plain CLIs `anime_tools.tagger.cli.*` | single-image / vocab build / dbv4 ckpt |

Stage wiring (`scripts/tasks/preprocess.py`): autotag runs **first** (right after
resize, `apply=True`), then position clauses, then correction/variants, then TE —
chain order pinned by `tests/test_preprocess_tasks.py`; the request fields the trainer
sets are pinned by `tests/test_anime_tools_cli_contract.py`. TE caches **are** mtime-aware
(`library/preprocess/text.py::_cache_is_current` re-encodes a stem whose cache is older
than its caption `.txt` or `.variants.txt`), so a plain `make preprocess-te` after an
`--apply` picks up exactly the stems that changed — no `--overwrite`, which is needed
only for what mtime cannot see (a vocab-pack or variant-count change). Always run it: a
stale `.variants.txt` keeps training the old caption. Once an
image has a revised caption, a hand-edit of its master no longer reaches it — edit the
revised caption, or delete it to re-mirror.

## The OCR clause is a publish, not a caption stage

`with_ocr_clause` (`anime_tools.captions.ocr_sidecar`) is the one place an
`{stem}.ocr.txt` meets a caption, and it is reachable only through the **export**
stage's `--combine_ocr` — written as a workspace→trainer publish. The trainer is its
own workspace (the caption stages write `post_image_dataset/resized` directly), so
`_caption_combine_request` runs that export **in place**: `out` is the resized tree's
*parent*, because an export writes a caption to `out/resized/<rel>.txt`. Every row but
`caption`/`variants` then compares identical and is skipped — no pixel, mask or index
churn — and `master`/`excluded_dir` keep the package's (absent) workspace defaults, so
no row can write back over the hand-written masters under `image_dataset/`. Pinned by
`test_caption_full_combine_publishes_in_place`.

The combine is idempotent: a text clause the caption already carries is replaced, and a
re-run whose sidecar lost its lines *removes* the clause. Two floors decide which lines
reach a caption (`--ocr_min_det` 0.5, `--ocr_min_glyph` 16.0) — the sidecar always keeps
every line. One caveat: a **dry run of the full chain reports the combine against the
sidecars already on disk**, not against what its own OCR step would have written.

`configs/clause_vocabulary.yaml` is the user-editable clause policy; the package
ships an identical default used when the file is absent from the curation home.
