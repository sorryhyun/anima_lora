# v2.0.0 release plan (working draft, 2026-09-02; Track A/B state re-checked 2026-09-06)

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
  Users with JA/KO captions must re-run `make preprocess-te ARGS=--overwrite`
  after enabling a pack; existence-only skip means this is a manual overwrite,
  now flagged by the cache stamp warning (B2).
- `pyproject.toml` still says `version = "0.1.0"` (never bumped alongside the
  git tags). Bump to `2.0.0` at the tag so `importlib.metadata` agrees with the
  release for the first time.

## Track A — anime_tools as a shipped dependency

State: repos `sorryhyun/anime_tools` and the Adapter node are **public**;
package rev pinned in `[tool.uv.sources]`; the dev loop
(`uv sync --no-group anime-tools-git --group anime-tools-dev`) works.

- [ ] **A1. Pin the release rev.** Bump `[tool.uv.sources]` to the anime_tools
  rev that will ship, `uv lock`, commit the lock. The rev must be pushed
  before `uv lock` resolves. The D0 blocker is gone: the pin moved to
  `94e6d92` (0.4.0) on 2026-09-02, `8708224` (0.4.1) on 2026-09-03 and
  `46ebbb5` (0.4.2) on 2026-09-06; A1 is now just "pin whatever rev ships"
  (as of 2026-09-06 the package is 2 commits ahead — the PP-OCR VL LoRA
  support — so one more bump is owed at tag time).
- [ ] **A2. Offline / Windows tarball.** Open since Phase 3: a release install
  with no GitHub access must still satisfy the git dep. Decide between
  (a) vendoring the pinned rev into the release tarball and pointing the
  `anime-tools-git` group at a local path in the shipped `pyproject.toml`, or
  (b) documenting that `uv sync` needs network for the git dep (it already
  needs it for PyPI). (b) is the honest default; pick (a) only if a user
  reports an offline install. Record the decision in `README.md` Setup.
- [x] **A3. Installer + updater compatibility** (by inspection 2026-09-06).
  `install.sh` / `install.ps1` / `scripts/update.py` run a plain `uv sync`
  with no `--extra`, and `anime-tools-git` is in `default-groups`, so the git
  dep resolves under the v1.17.1 updater unchanged. The fresh-clone run is the
  beta gate below.
- [ ] **A4. Drop the stub extras.** The `--extra` plumbing is already gone
  from the installers (A3); `cuda-windows = [] / rocm-windows = []` are still
  in `pyproject.toml` — delete at the v2 tag.
- [ ] **A5. Docs.** `CLAUDE.md` describes the split and (since 2026-09-03)
  the request API as the front door; user-facing: Setup section of
  `README.md` (what `anime_tools` is; the typed request API is the front door
  and `python -m anime_tools.<pkg>.cli.<name>` is its shell — Track D landed,
  so state it plainly), guidebook line + the three translations (use the
  translator agent, diff-driven).
- [ ] **A6. Hygiene gates before tagging** (tests exist; run at tag time —
  `test_doc_refs` currently fails on 6 pre-existing stale references in
  `anima_daemon/issues.md` + `project/cjk_aware_anima*/`). `tests/test_repo_hygiene.py`
  (no tracked symlinks — the recurring v1.16.2 breaker), tarball extracts
  under `tarfile filter="data"`, `tests/test_curation_boundary.py`,
  `tests/test_doc_refs.py`, `make test-unit`.

## Track B — ext-vocab training path

What exists today (all research-grade, under `project/cjk_aware_anima/` and
`bench/cjk_adapter/`):

| Piece | State |
|---|---|
| `library/anima/ext_vocab.py` (`HybridT5Encoder`, `segment_runs`, `load_ext_assets`) | promoted, canonical, EN bit-exact tests green |
| `library/anima/vocab_pack.py` (`VocabPack`, `VocabPackTokenizeStrategy`, `attach_vocab_pack`, stamps) | **shipped 2026-09-06** — the one home for both patch points; `tests/test_vocab_pack.py` |
| Vocab pack `synthjakozh1sym_r256` | published as `anima_cjk_vocab_pack.*` at `huggingface.co/sorryhyun/anima-vocab-pack-cjk` (repo renamed from `-ja` 2026-09-06; JA-only files deleted; self-contained: table + mapping + Qwen3 tokenizer files). The quote-partitioned `_isoq` build is research-only. |
| Pack-aware TE caching | `make preprocess-te` forwards base.toml `vocab_pack` (shipped); `project/cjk_aware_anima/datasets/cache_te_ext.py` stays as the research mirror/OCR-caption driver |
| OCR caption stage | `project/cjk_aware_anima/datasets/ocr_text_captions.py` — mask-complement crops → manga-ocr → quote tags appended via `parse_caption`/`compose_caption` |
| Unmask recipe | `configs/gui-methods/custom/cjk_unmask_c.toml` (`masked_loss=false` + redirected `text_cache_dir`) |
| ComfyUI | `AnimaVocabPackLoader` in the Adapter node, 3.10.0 pushed (quote partition + LoRA↔pack digest check); comfy-path render grid still owed |
| Glossary r2 pack (`synthja_v5` / `synthjako3`) | superseded — the JA+KO+ZH joint pack above is what ships (B8) |

To ship, in dependency order:

- [x] **B1. Pack as a model asset** (2026-09-06). `vocab_pack` key in
  `configs/base.toml` (empty = off; `ANIMA_VOCAB_PACK` env override;
  `default_checkpoints().vocab_pack`), `make download-vocab-pack` →
  `models/vocab_packs/anima_cjk_vocab_pack.*` (opt-in, not part of
  `download-models`), identity = `pack_digest` + name + rows
  (`VocabPack.identity()` / `cache_metadata()` / `checkpoint_metadata()`).
- [x] **B2. Strategy shim in the trainer** (2026-09-06).
  `VocabPackTokenizeStrategy(AnimaTokenizeStrategy)` is installed by
  `setup_training_strategies` / `ensure_text_strategies` / `inference.py` /
  `cache_text_embeddings.py` whenever a pack is active (stock class otherwise
  — off is inert). TE caches written through a pack carry `vocab_pack` /
  `vocab_pack_sha` metadata (both writers) and the cache-completeness check
  warns once per mismatch kind. G1 lifted to the strategy level in
  `tests/test_vocab_pack.py` (EN bit-exact, JA lands on ext rows). The bench
  `ExtTokenizeStrategy` in `cache_te_ext.py` is not deleted yet — that file
  is the live OCR-caption driver of the DiT line and is mid-edit.
- [x] **B3. Rows at DiT load** (2026-09-06). `load_anima_model(…,
  vocab_pack=)` / `load_llm_adapter(…, vocab_pack=)` install the node's
  clamp-pre-hook / embed-forward-hook pair (`attach_vocab_pack`) — the state
  dict stays at 32128 rows, so merge / save are unaffected. `train.py`'s lazy
  DiT load and `load_dit_model` pass the same memoised pack the strategy uses;
  the LoRA `ss_ext_pack_sha` stamp is checked against the active pack at load.
- [x] **B4. Inference surface** (2026-09-06). `--vocab_pack` /
  `--no_vocab_pack` on `inference.py` (default = the config key), the same two
  fields on `GenerationRequest`; `examples/09_cjk_vocab_pack.py` rides the front door. Verified
  end-to-end on a daemon job (JA prompt, every tag composed). Not covered: the
  resident inference server keeps one warm DiT, so a per-request pack change
  after the first load is not honoured (restart the server).
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
- [ ] **B7. Docs.** `docs/methods/cjk_vocab_pack.md` + the
  `docs/inference/README.md` cross-link + CLAUDE.md row + GUI field tooltips
  (4 languages) landed 2026-09-06; still owed: the guidebook line + its three
  translations and the README Setup mention.
- [x] **B8. Which pack ships** (settled 2026-09-06). The joint JA+KO+ZH
  `synthjakozh1sym_r256` (symbol block, no iso partition) ships as
  `anima_cjk_vocab_pack.*` from `sorryhyun/anima-vocab-pack-cjk`; the
  v4-vs-v5 question is superseded. It is a **HF asset, not a release-tag
  asset** (~285 MB) — the release notes link the HF repo and
  `make download-vocab-pack` fetches it.
- [ ] **B9. ComfyUI parity.** Node 3.10.0 is pushed; the rendered same-seed
  grid through the `AnimaVocabPackLoader` path is still owed before the
  registry publish (`make vendor-sync` first — never hand-copy into `_vendor/`).

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

## Track D — API-first curation boundary

Plans: `docs/proposal/anime_tools_api_first.md` (trainer side; the package
side's plan was folded into `../anime_tools/CLAUDE.md`). **Landed in full
2026-09-02/03** — the trainer builds one typed request dataclass per stage
(`SamMaskRequest`, `AutotagRequest`, `ResizeRequest`, …) and runs it
in-process under the daemon or as a `python -m` child; nothing spells a flag.

- [x] **D0. Pin-bump prerequisites** (= T0, 2026-09-02). `sam_mask.yaml` →
  the new mask flags, correction pass reads revised-first, revised-caption
  owner recorded in `anime_tools/docs/contract.md`; pin `94e6d92` (0.4.0).
- [x] **D1. Contract test** (= T1, 2026-09-02).
  `tests/test_anime_tools_cli_contract.py` re-parses every emitted argv through
  the stage's generated parser in-process — the drift alarm.
- [x] **D2. `anime_tools.contract` + `CONTRACT_VERSION`** (= T2, 2026-09-03).
- [x] **D3. Masking through requests** (= T3, 2026-09-03).
- [x] **D4. Caption stages + grouping through requests** (= T4, 2026-09-03).
- [x] **D5. Free-fit geometry owned by `anime_tools`** (= T5, 2026-09-03;
  pin `8708224`, 0.4.1). `library/datasets/buckets.py` re-exports;
  `make preprocess-resize` is a `ResizeRequest`.

Beta gate addition (still owed as a live run): `make preprocess` on the JA
shard followed by `make caption-autotag ARGS=--apply` + `make preprocess-te`
keeps the autotag tags in the TE-cached caption (the clash D0 fixes).

## Beta → GA gates

- Track D0 + D1 green (see above).
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
