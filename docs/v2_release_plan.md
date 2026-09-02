# v2.0.0 release plan (working draft, 2026-09-02)

Scratch plan for the next release. Two pillars make it a major bump:

1. **Curation is a separate package** — `anime_tools` (tagger, masking,
   grouping, caption polishing) is a git dependency; the in-tree shims and
   forwarding shells are gone. Already on `main` (the `anime-tools-phase1`
   branch is fully merged, 0 commits ahead); what remains is the release
   surface around it.
2. **Vocab-pack-aware training** — the CJK ext-vocab encoder becomes a
   first-class trainer/inference path so manga data trains **with text masks
   off**: in-image text is OCR'd into the caption, the caption is encoded
   through the vocab pack, and the text pixels become attributable instead
   of spam. The unmask A/B (`project/cjk_aware_anima/reports/0901_unmask_ab.md`)
   settled that the *captions* carry the effect and that even the legacy
   `synthjako2` pack is enough — so this does not wait on the glossary-r2
   pack; v5 is a quality upgrade, not a gate.

Ship order: **`v2.0.0-beta.1` first**, then `v2.0.0` once the beta gates below
hold. The tracks are independent; the release track runs last.

## Why a major version

Breaking for existing installs (each already true on `main` or lands here):

- `library.captioning.*`, `library.preprocess.{caption_variants,autotag,
  position_captions,…}`, `library.vision.{pe_features,pe_matching,
  grouping_embedder}`, `library.datasets.grouping`, the `scripts.anima_tagger`
  / `scripts.curate` dirs and the curation `scripts.preprocess.*` shells no
  longer exist — import `anime_tools` (`make` target names are unchanged).
- Pre-three-axis LoRA checkpoints (`ss_use_hydra` / `ss_use_fei_router`
  metadata) no longer load.
- `segmentation-models-pytorch` left the trainer's deps (rides on
  `anime-tools[masking]`).
- The empty stub extras `cuda-windows = [] / rocm-windows = []` kept for old
  updaters (GH #92) can finally go — v2 is the planned cut-off (~v1.18 was the
  estimate).
- Text-encoder caches for captions containing CJK are **not** what the stock
  encoder produced once a pack is active (EN-only captions stay bit-exact).
  Users with JA/KO captions must re-run `make preprocess-te` after enabling a
  pack; existence-only skip means this is a manual overwrite, see B2.
- `pyproject.toml` still says `version = "0.1.0"` (never bumped alongside the
  git tags). Bump to `2.0.0` at the tag so `importlib.metadata` agrees with the
  release for the first time.

## Track A — anime_tools as a shipped dependency

State: repos `sorryhyun/anime_tools` and the Adapter node are **public**;
package rev pinned in `[tool.uv.sources]`; the dev loop
(`uv sync --no-group anime-tools-git --group anime-tools-dev`) works.

- [ ] **A1. Pin the release rev.** Bump `[tool.uv.sources]` to the anime_tools
  rev that will ship (package `0.2.0` + the OCR improvements pushed
  2026-09-02), `uv lock`, commit the lock. The rev must be pushed before
  `uv lock` resolves.
- [ ] **A2. Offline / Windows tarball.** Open since Phase 3: a release install
  with no GitHub access must still satisfy the git dep. Decide between
  (a) vendoring the pinned rev into the release tarball and pointing the
  `anime-tools-git` group at a local path in the shipped `pyproject.toml`, or
  (b) documenting that `uv sync` needs network for the git dep (it already
  needs it for PyPI). (b) is the honest default; pick (a) only if a user
  reports an offline install. Record the decision in `README.md` Setup.
- [ ] **A3. Installer + updater compatibility.** `install.sh` / `install.ps1`
  / `scripts/update.py` run `uv sync` with the default groups — confirm the
  `anime-tools-git` group resolves on a fresh clone without the dev group
  present (the v1.17.1 updater is what beta testers run to get to v2, so the
  sync behaviour must not change under it).
- [ ] **A4. Drop the stub extras** and the `--extra` plumbing in the
  installers once A3 confirms no shipped updater passes `--extra`.
- [ ] **A5. Docs.** `CLAUDE.md` already describes the split; user-facing:
  Setup section of `README.md` (what `anime_tools` is, that curation CLIs are
  `python -m anime_tools.<pkg>.cli.<name>`), guidebook line + the three
  translations (use the translator agent, diff-driven).
- [ ] **A6. Hygiene gates before tagging.** `tests/test_repo_hygiene.py`
  (no tracked symlinks — the recurring v1.16.2 breaker), tarball extracts
  under `tarfile filter="data"`, `tests/test_curation_boundary.py`,
  `tests/test_doc_refs.py`, `make test-unit`.

## Track B — ext-vocab training path

What exists today (all research-grade, under `project/cjk_aware_anima/` and
`bench/cjk_adapter/`):

| Piece | State |
|---|---|
| `library/anima/ext_vocab.py` (`HybridT5Encoder`, `segment_runs`, `load_ext_assets`) | promoted, canonical, EN bit-exact tests green |
| Vocab pack `synthja_v4` | public test release `huggingface.co/sorryhyun/anima-vocab-pack-ja` (self-contained: table + mapping + Qwen3 tokenizer files) |
| Pack-aware TE caching | `project/cjk_aware_anima/datasets/cache_te_ext.py` — bench script, temp caption mirror, sidecar cache dir |
| OCR caption stage | `project/cjk_aware_anima/datasets/ocr_text_captions.py` — mask-complement crops → manga-ocr → quote tags appended via `parse_caption`/`compose_caption` |
| Unmask recipe | `configs/gui-methods/custom/cjk_unmask_c.toml` (`masked_loss=false` + redirected `text_cache_dir`) |
| ComfyUI | `AnimaVocabPackLoader` in the Adapter node v3.9.0 (committed + pushed; comfy-path render grid still owed) |
| Glossary r2 pack (`synthja_v5` / `synthjako3`) | chain re-queued 2026-09-02 after the r2 override merge + reselect guards; gates in `project/cjk_aware_anima/temp_plan.md` step 5 |

To ship, in dependency order:

- [ ] **B1. Pack as a model asset.** A `vocab_pack` path key in
  `configs/base.toml` (empty = off; resolved via `resolve_under_home()` like
  the other model paths), a download target in `scripts/tasks/downloads.py`
  pulling the HF pack into the models dir, and pack identity stamped from the
  pack's JSON (`rows`, `stats`, training label) so caches and checkpoints can
  name which pack they saw.
- [ ] **B2. Strategy shim in the trainer.** `AnimaTokenizeStrategy`
  (`library/anima/strategy.py`) routes the T5-side stream through
  `HybridT5Encoder` when `vocab_pack` is set; `AnimaTextEncoderOutputsCachingStrategy`
  follows automatically. This is the promotion of `cache_te_ext.py`'s
  `ExtTokenizeStrategy` — delete the bench copy once landed. Cache
  invalidation: TE caches skip on existence only (no content hash), so
  document "re-run `make preprocess-te` with overwrite after enabling a
  pack"; a cheap improvement is stamping the pack id into the `.safetensors`
  metadata and warning at load when it differs from the active pack.
  Test: EN-only caption bit-exact with and without the pack (extend
  `tests/test_cjk_distill.py`'s G1 set to the strategy level).
- [ ] **B3. Row append at DiT load.** `load_anima_model` /
  `load_dit_model` appends the pack rows to the adapter's T5 embedding
  (the 32128 hardcode; the node's clamp-pre-hook / embed-forward-hook pair is
  the reference) so sample-during-training and inference see the same table
  the caches were built with.
- [ ] **B4. Inference surface.** `inference.py` + `GenerationRequest` accept
  the pack (default = the config key, flag to disable), so `make test` /
  `make gen` with a JA prompt exercise the same encoder. Adapter family stays
  in checkpoint metadata; the pack is a *text-encoder* asset, not part of the
  LoRA.
- [ ] **B5. OCR caption stage → anime_tools.** `ocr_text_captions.py` is
  curation (needs the masks anime_tools owns, edits the caption master), so it
  moves into the package as a `caption-ocr` stage with the same contract as
  `make caption-autotag` / `make caption-position`: dry-run by default,
  `--apply` writes sidecars, and an apply **must** be followed by
  `make preprocess-te`. Dependency: `manga-ocr` is installed in the venv but
  declared in **neither** `pyproject.toml` — it rides on an anime_tools extra
  (`anime-tools[ocr]`), never on the trainer. Bring `manga_text.MangaOCR`
  along (it is the only consumer).
- [ ] **B6. Unmask recipe as a shipped variant.** `masked_loss` is already a
  GUI-visible key; promote the arm-C config out of `gui-methods/custom/` into
  a documented recipe (variant or guidebook section): masks off + OCR captions
  + pack. Keep `masked_loss=true` as the default — unmasking without the
  captions reproduces the spam (arm B), so the recipe must be presented as a
  bundle, not a toggle.
- [ ] **B7. Docs.** A user-facing `cjk_vocab_pack.md` under `docs/methods/`
  (what works: tags / quotes / mixed names; what does not: rare full-JA
  kanji names; the TE-cache regeneration note; KO status), a line in the
  guidebook + translations, and `docs/inference/README.md` cross-link for the
  inference flag.
- [ ] **B8. Which pack ships.** v5 if the temp_plan step-5 gates pass
  (distill readouts vs v4, same-seed 2c grid ≥3 seeds, EN bit-exact);
  otherwise v4 ships and the regressing override is treated as a review bug.
  Either way the pack is a **HF asset, not a release-tag asset** (241 MB) —
  the release notes link the HF repo and the download target fetches it.
- [ ] **B9. ComfyUI parity.** Rendered same-seed grid through the
  `AnimaVocabPackLoader` path before the node's public publish
  (`make vendor-sync` first — never hand-copy into `_vendor/`).

## Track C — release mechanics

- [ ] **C1. Prerelease semantics.** `scripts/update.py` and `install.sh`
  resolve "latest" via the GitHub `releases/latest` API, which **excludes
  prereleases** — so a beta tag is invisible to `make update` unless the user
  passes the tag explicitly (`ANIMA_VERSION=v2.0.0-beta.1` / update.py's
  version argument). That is the desired behaviour. But
  `.github/workflows/release.yml` creates the release with `gh release create
  --generate-notes` and **no `--prerelease`**, so a `v2.0.0-beta.1` tag would
  be published as a full release and become "latest". Fix before tagging:
  mark prerelease when the tag contains `-` (one-line change in the workflow),
  or push the tag and immediately `gh release edit v2.0.0-beta.1 --prerelease`.
  Prefer the workflow fix.
- [ ] **C2. Tag hygiene.** Tag `v2.0.0-beta.1` (SemVer prerelease; sorts
  below `v2.0.0`). Check the tarball with `tarfile filter="data"` and
  `test_no_tracked_symlinks` first. Release notes: `gh release edit` after CI
  creates the release (`gh release create` afterwards hits "already exists").
- [ ] **C3. Version bump.** `pyproject.toml` → `2.0.0b1` for the beta
  (PEP 440), `2.0.0` at GA.
- [ ] **C4. Release notes skeleton** (v1.17.1 format: "Major updates" bullets
  with weight links): curation split + how to get the CLIs; vocab pack +
  unmask recipe with the HF link; breaking-changes list from above; Anima-2.9B
  and caformer tagger already shipped in 1.17.x so not repeated.

## Beta → GA gates

- One full user flow on a fresh install from the beta tag (installer →
  `make download-models` → pack download → `make preprocess` on a JA-caption
  shard → `make lora` → `make test` with a JA prompt) on Linux **and**
  Windows, no repo clone.
- `make update` from a v1.17.1 checkout to the beta tag preserves
  `configs/preprocess.toml`, datasets, outputs, and re-syncs the git dep.
- Unmask recipe reproduces the arm-C result on a second shard (not `sincos`)
  at 3 seeds; render-level scorer (text-mask area + Tagger recall) reported,
  not eyeballed.
- EN-only users: byte-identical TE caches and renders with the pack key
  empty (the default) — the whole B track must be inert when off.
- No open P0 from beta testers for one week (Arca Live / Civitai / JP
  users — three languages of UX to watch).

## Explicitly out of scope for v2.0.0

- Phase 5b real-co-occurrence corpus for rare kanji names, morpheme-row
  minting, the KO pack as default, the glyph line (rendering JA text) — all
  stay in `project/cjk_aware_anima/plan.md`.
- Unmask arm C2 (PP-OCR captions) ranking — needs the render scorer first.
- PyPI publication of `anime_tools` — git dependency stays.
