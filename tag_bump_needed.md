# Pending: `anime_tools` tag bump

**Status: blocked on an upstream release. The trainer tree is red until it lands.**

`../anime_tools@da5a93e` ("a caption is never re-split, correct has an undo") made
`correct` an `ApplyRequest` — dry-run by default like its six siblings, so the GUI's
Undo works on it. That crosses the seam: `scripts/tasks/preprocess.py` must pass
`apply=True` at both construction sites (the `request_from_form` path and the
`CorrectRequest(**fields)` path), or the correction silently stops writing and TE
caches the un-corrected caption.

That change is **in the working tree, uncommitted**, together with one updated
assertion in `tests/test_preprocess_tasks.py`. It cannot be committed on its own:
`pyproject.toml:141` pins `tag = "v0.6.4"`, whose `CorrectRequest` has no `apply`
field, and `anime-tools-git` is default-on — so a plain `uv sync` gives
`TypeError: CorrectRequest.__init__() got an unexpected keyword argument 'apply'`
on the first `make preprocess-captions` / `preprocess-te`. Ten tests fail there
today (7 in `tests/test_preprocess_tasks.py`, 3 in
`tests/test_anime_tools_cli_contract.py`); all pass with `PYTHONPATH=../anime_tools`.

## What unblocks it

Upstream has **no `v0.6.5` tag** (only a version-bump commit, `291a523`) and a large
in-flight refactor uncommitted (`stages/run.py` deleted, runners moved into the stage
modules). So:

1. Land the in-flight refactor in `../anime_tools` and cut + push a release tag
   (the package's own `release` skill: version bump → annotated tag → `release.yml`).
2. Here, in **one commit**: move `tag` in `pyproject.toml`, then
   `uv lock --upgrade-package anime-tools && uv sync`, and land the `apply=True`
   change with it. The pin and the flag cannot land separately.
3. Re-run `tests/test_preprocess_tasks.py` + `tests/test_anime_tools_cli_contract.py`.

## Two non-blocking items that ride the same release

From `../anime_tools/issue.md` §"The trainer side (`anima_lora`)":

- **`gui/progress.py:29`** — `TQDM_RE` matches only tqdm's `NN%|bar| cur/tot`, and no
  stage emits tqdm any more (`_progress.py` owns the format). The tabs driving a bar
  off a daemon job's stdout — `_job_mixin.py:90` → `TqdmProgressTracker.feed`, used by
  `tabs/image_tab.py:245` for `curate-group` and `tabs/preprocess/tab.py:1097` for the
  mask stages — will sit in indeterminate "busy" mode for the whole run. One extra
  regex alternative fixes it: `^\s*\[(?P<cur>\d+)/(?P<tot>\d+)\]\s*(?P<label>.*)$`
  (label *after* the counts, unlike tqdm's). Alternative: move those tabs onto
  `JsonlProgressReader` — the mask merge and the grouping pass now write `step` lines
  to `progress.jsonl` where before only `masks_sam` did. `_job_mixin.py:90`'s docstring
  is stale either way.
- **`scripts/tasks/masking.py:191`** — "The SAM3 checkpoint and batch size are the
  request defaults": `SamMaskRequest` has no `batch_size` any more. Prose only.

Nothing else crosses: `contract.py` only gained a `REPLAY_SHAPES` key, `buckets.py` is
untouched, and the `progress.jsonl` line shapes the daemon's reader filters on are
unchanged.

*Written 2026-09-12. Delete this file when the pin moves.*
