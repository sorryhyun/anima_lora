# v2.0.0-beta — branch checklist

Working checklist for the `v2.0.0.beta` branch: cut from `main`, PR-merged, a
release draft published on merge. One checkbox = one commit-sized job. The
**why** and the long-form state live in
[`v2_release_plan.md`](v2_release_plan.md) — that doc stays as the reference
for Tracks A/B/C; this file is what actually gets ticked.

Two things make v2 a major bump (both already on `main`): curation moved to the
`anime_tools` package, and pre-three-axis LoRA checkpoints no longer load. This
branch adds a third: **masking is off by default and the in-image text masker
is gone.**

---

## 0. Branch setup

- [x] Cut `v2.0.0.beta` from `main` (at `e3c2e2e9`).
- [x] Open the PR early as a draft so the checklist is visible while the work
      lands — [#99](https://github.com/sorryhyun/anima_lora/pull/99).
- [x] `.github/workflows/release.yml` publishes with `gh release create
      --generate-notes` and **no `--prerelease`** — a `-beta` tag would become
      "latest" and reach every `make update`. Fix the workflow to pass
      `--prerelease` when the tag contains `-`. **Do this first**, before any
      tag exists. (`1a412dab` — `case "$TAG" in *-*)`; `scripts/update.py`
      resolves `releases/latest`, which excludes prereleases.)

## 1. Masking off by default

The default becomes: no masks, train on the whole image. Masking stays fully
supported, just opt-in.

- [ ] `configs/base.toml`: `masked_loss = true` → `false`, with a comment
      saying `make mask` + this key is the pair that turns it on.
- [ ] `gui/tabs/preprocess/knobs.py`: `DEFAULT_RUN_SAM_MASK = True` → `False`.
- [x] SAM3 out of `make download-models`' component list
      (`scripts/tasks/downloads.py::cmd_download_models`) — keep
      `make download-sam3` as the opt-in target, the way the vocab pack works.
      Removes the gated-repo failure from every first-run install.
      (`af98813e` — the component list is now `DEFAULT_SET` in
      `library/downloads.py`; SAM3, MIT, the OCR stack and the vocab pack are
      all out of it, pinned by a test.)
- [ ] Check nothing silently re-enables it: `resolve_configured_mask_dir`
      already gates on the dir existing, so a maskless checkout is inert — but
      a user who ran `make mask` once before upgrading now has a mask dir *and*
      `masked_loss = false`. Confirm that combination trains unmasked with no
      warning spam (or one clear line).
- [ ] Method configs that pin `masked_loss = true` (`configs/methods/turbo.toml`
      uses `use_masked_loss`) — decide per file whether the pin is intentional.
      Method wins over preset, so a stale pin defeats the new default.

## 2. Remove the manga text detector (MIT)

Text is no longer masked automatically, so the UNet++ / ComicTextDetector
backend goes. Roughly 17 references in `scripts/tasks/masking.py` plus the
surfaces below.

- [ ] `scripts/tasks/masking.py`: drop `_mit_request`, `_mit_model_path`,
      `MIT_MODEL_PATH`, the `run_mit` switch and the MIT half of the merge
      sources. `make mask` becomes SAM-only (the merge step stays — rules still
      compose).
- [ ] `configs/sam_mask.yaml`: delete the `mit:` block and `run_mit` from the
      header comment.
- [ ] GUI: `DEFAULT_RUN_MIT_MASK` + the MIT card in
      `gui/tabs/preprocess/masking.py`, the knob in `knobs.py`, and the 13
      `preprocess_run_mit_mask*` / MIT strings in each of
      `gui/i18n/{en,ko,ja,cn}.py`. `gui/system_dialog.py` lists the MIT weights.
- [x] Downloads: drop `mit_text` / `ctd_onnx` from `DL.GROUPS` in
      `library/downloads.py` and the `download-mit` target in `tasks.py`
      (`cmd_download_mit` is now a two-line lookup). The rows stay in the
      *package* catalog, so they also disappear from the GUI's Curation
      Models panel only if the panel filters them — decide which.
      Note in the release notes that `models/mit/` can be deleted.
      (Done: the `mit` group and `download-mit` are gone; the package's
      `text_mask` pack is hidden on the trainer side via
      `DL.HIDDEN_PACKS` — `curation_catalog()` / `by_id()` / `resolve()` /
      the Curation tab never see its rows; the rows themselves stay in the
      package catalog for its own users. `scripts/tasks/masking.py` still
      reads a leftover `models/mit/model.pth` if present — that goes with
      the MIT masking bullet above.)
- [ ] Tests: `tests/test_masking_task.py`, `tests/test_nested_paths.py` (7
      refs), `tests/test_anime_tools_cli_contract.py` (12 refs — the
      `MitMaskRequest` argv round-trip goes with it).
- [ ] Legacy path triple `masks/{merged,sam,mit}` in
      `library/datasets/subsets.py::_resolve_default_mask_dir` and its mention
      in `library/config/cli_args.py` — keep or trim? Keeping costs nothing and
      preserves old checkouts; if trimmed, do it in the same commit as the docs.
- [ ] Package side: `anime_tools` keeps `MitMaskRequest` (its own users may
      want it) — this is a *trainer* removal only. Confirm no trainer import
      of `anime_tools.masking.mit` survives; the `anime-tools[masking]` extra
      still carries `segmentation-models-pytorch`, decide whether the trainer
      still requests that extra.

## 2b. Model catalog (landed early — `af98813e`)

Not in the original plan; it fell out of §1 and makes §2 smaller.
`anime_tools.downloads` is the curation weights' catalog (`Asset` rows with an
offline installed-probe); `library/downloads.py` now adds the Anima-only rows
and concatenates it, and every `make download-*` plus both GUI Models panels
read that one list. Follow-ups this leaves:

- [x] GUI: one Models modal, two tabs (**Anima** / **Curation
      (anime_tools)**), rows scrolling inside each tab so the log pane stays
      visible. Shared token field, shared log, one QProcess.
- [x] The Curation tab renders every package row, including the OCR stack and
      `tagger_onnx`. Decide whether the trainer filters any of them out (see
      §2 — MIT is the live question). **Decision: filter by pack id**, not
      by row — `DL.HIDDEN_PACKS = ("text_mask",)`; the OCR stack and
      `tagger_onnx` stay visible as opt-in rows.
- [x] Packs + CJK mandatory. The package grew a `Pack` table
      (`anime_tools.downloads.PACKS`, `Asset.pack`); the trainer adds
      `anima` / `pe` / `cjk` and exposes `DL.PACKS` / `by_pack()` /
      `GROUP_ALIASES` (`resolve` = alias → pack → row, so `make download-model
      ocr` expands and `make download-pe` keeps both towers). Both Models
      tabs render one `QGroupBox` per pack with a "Download pack" button;
      `make download-list` prints by pack. `configs/base.toml` now ships
      `vocab_pack = "models/vocab_packs/anima_cjk_vocab_pack"`, `vocab_pack`
      is in `DEFAULT_SET`, and `resolve_pack_prefix` auto-fetches the
      shipped default when missing (custom paths still raise) so a `make
      update` over a pre-v2 checkout does not hard-fail. Cache-stamp
      mismatch already warns once per kind per run, not per file.
- [ ] `docs/guidelines/가이드북.md` / `ガイドブック.md` / `指南书.md`: the
      English `guidebook.md` model-download block changed (first-run set no
      longer includes SAM3/MIT; `download-list` / `download-model` are new).
      **Translator agent**, with the rest of §3.
- [ ] `anime_tools`' `model-catalog` skill says the trainer addresses rows by
      id — now true. Consider a trainer-side skill or a `CLAUDE.md` pointer
      when the surface settles.

## 3. Docs cleanup

- [ ] `CLAUDE.md`: the masking sentences in **Config flow** (mask_dir is
      "load-bearing") and **Preprocessing & scripts**, plus the 4 MIT
      references. State the new default in one line.
- [ ] `README.md` (3 MIT refs) — Setup section: SAM3 is now an opt-in download.
- [ ] `docs/guidelines/guidebook.md` + `가이드북.md` / `ガイドブック.md` /
      `指南书.md` (6 refs each) and `docs/guidelines/training.md`,
      `base-config.md`. **Use the translator agent** — diff-driven, after the
      English is final.
- [ ] `docs/proposal/gui_preprocess_tab_refactor.md` and
      `anime_tools_api_first.md` mention MIT — these are landed proposals;
      either annotate or move to `_archive/`.
- [ ] `docs/v2_release_plan.md`: add a banner pointing here as the live
      checklist.
- [x] `make test-unit` includes `tests/test_doc_refs.py`, which failed on 11
      stale refs in the CJK research tree — sibling-repo paths written as if
      they were this repo's, plus a make target that never existed. Fixed at
      the source rather than allowlisted: the package's files now carry their
      real `../anime_tools/` prefix (which the linter skips, correctly — they
      are not this repo's to verify) and the two prose mentions of an `ocr`
      make target are reworded. `test_doc_refs` is green.

## 4. Scripts / preprocess cleanup

- [ ] Walk `scripts/preprocess/` and `scripts/tasks/` for shells left behind by
      the anime_tools split and by the MIT removal; delete rather than deprecate
      (v2 is the breaking release — this is the moment).
- [ ] `make help` output re-read end to end: every target still exists, every
      description still true after the mask changes.
- [ ] `make preprocess` chain unchanged (it never ran masking) — confirm with a
      dry run that no stage now warns about missing masks.
- [ ] `make preprocess-reconcile` still deletes orphaned masks correctly when
      `mask_dir` is absent.

## 5. Carried over from `v2_release_plan.md` (still open)

Detail and rationale in that doc; listed here so nothing gets lost.

- [ ] **A1** Pin the shipping `anime_tools` rev in `[tool.uv.sources]`, `uv lock`,
      commit the lock. Package is currently ahead of the pin (the `OcrRequest
      --reader vl` work is uncommitted in `../anime_tools` — commit and push it
      first).
- [ ] **A2** Offline/Windows install decision — document that `uv sync` needs
      network for the git dep (the honest default) in `README.md` Setup.
- [ ] **A4** Delete the stub extras `cuda-windows = [] / rocm-windows = []`.
- [ ] **A6** Hygiene gates: `tests/test_repo_hygiene.py` (no tracked symlinks),
      tarball extracts under `tarfile filter="data"`,
      `tests/test_curation_boundary.py`, `tests/test_doc_refs.py`,
      `make test-unit`.
- [ ] **B5** OCR caption stage → `anime_tools` as `caption-ocr` (dry-run by
      default; an `--apply` must be followed by `make preprocess-te`).
      **This is now load-bearing**: with text masking gone, OCR'd captions are
      the only thing making in-image text attributable.
- [ ] **B6** Ship the unmask recipe (masks off + OCR captions + vocab pack) as a
      documented bundle, not a toggle — unmasking *without* the captions
      reproduces the spam (arm B).
- [ ] **B7** Guidebook line + 3 translations, README Setup mention.
- [ ] **B9** ComfyUI parity: `make vendor-sync` (never hand-`cp`), rendered
      same-seed grid through `AnimaVocabPackLoader`, then registry publish.
- [ ] **C3** `pyproject.toml` `version` → `2.0.0b1` (PEP 440) at the beta tag.
- [ ] **C4** Release notes skeleton (v1.17.1 format).

## 6. Merge gates

- [ ] `make test-unit` green (including `test_doc_refs`). *Green as of
      `2026-09-07`: 1647 passed, 1 skipped — re-check before the tag.*
- [ ] `make preprocess` on a small shard from a **fresh** clone with SAM3 never
      downloaded — must complete with no mask-related error.
- [ ] `make lora` on that shard trains unmasked by default.
- [ ] `make mask` still works after opting in (`make download-sam3` first) and
      training with `--masked_loss` picks the masks up.
- [ ] GUI opens, Preprocessing tab has no MIT card and no dead settings key.
- [ ] `make update` from a v1.17.1 checkout preserves `configs/preprocess.toml`,
      datasets and outputs.

## 7. Tag & release draft

- [ ] Squash-or-merge the PR (repo convention: commit directly on `main`, so a
      merge commit is fine).
- [ ] Tag `v2.0.0-beta.1`; CI creates the release **as a prerelease** (§0).
- [ ] `gh release edit v2.0.0-beta.1` to attach the notes — `gh release create`
      after CI hits "already exists".
- [ ] Verify `make update` on a v1.17.x checkout does **not** pick the beta up
      (prereleases are excluded from `releases/latest` — that is the intent).

---

### Explicitly out of scope

Everything `v2_release_plan.md` already listed, plus: the CJK DiT research line
(`project/cjk_aware_anima_dit/`) ships nothing in v2 — the vocab pack and the
unmask recipe do.
