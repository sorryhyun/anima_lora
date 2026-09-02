# anime_tools API-first boundary — trainer side (2026-09-02)

Status: **approved direction; package half landed, T0–T1 landed 2026-09-02, T2–T6 open**
(re-audited 2026-09-02 against `anime_tools` HEAD `6a225f8`; the package-side
requests below landed the same day on top of it, as `0.4.0`). The package-side
plan (`docs/api_first_plan.md`, deleted in `9ccc655`) is now folded into
`../anime_tools/CLAUDE.md`; Track D of `docs/v2_release_plan.md` says what
ships in v2.0.0.

## What the boundary is today

The trainer never imports `anime_tools` for heavy work. `scripts/tasks/
{preprocess,masking,curate,tagger}.py` hand-assemble argv and shell out to
`python -m anime_tools.<pkg>.cli.<name>`; `gui/tabs/_autotag.py` drives one
stdio worker. Fourteen module strings, no test that the flags we emit exist.
The trainer venv still installs the pin `971e229` (package `0.2.0`,
`__version__` reported `0.1.0`), 102 commits behind HEAD.

### What the package delivered

Every phase of the package plan (P0–P6) is on `main` and green — 851 tests,
ruff clean, and the request / registry / GUI-schema modules import with
`torch`, `cv2` and `sam3` poisoned in `sys.modules`:

| Piece | Where |
|---|---|
| Shared constants, `CONTRACT_VERSION = 1` (stdlib-only) | `anime_tools/contract.py` |
| Request base: `arg()` field metadata → generated parser, `to_argv()` / `from_namespace()` inverses, nested blocks, drawers | `anime_tools/_request.py` |
| One frozen request per stage | `stages/requests.py` (`ResizeRequest`, `AutotagRequest`, `PositionRequest`, `CorrectRequest`, `OcrRequest`, `AuditRequest`, `ExportRequest`), `masking/requests.py` (`SamMaskRequest`, `MitMaskRequest`, `MergeMasksRequest`), `grouping/requests.py` (`GroupRequest`) |
| In-process runners | `stages/run.py::run_<stage>`, `masking/{sam,mit,merge}.py::run_*`, `grouping/groups.py::run_groups`; all lazily exported from the package `__init__`s |
| Stage list, torch-free, `module:Class` resolved lazily | `stages/registry.py` (`STAGES`, `BY_ID`, `Stage.request_class()`) |
| Runner per stage, `module:function` resolved lazily | `Stage.run` + `Stage.runner()` — `anime_tools.stages.run:run_autotag`, `anime_tools.masking.sam:run_sam_masks`, … for all eleven; `tests/test_registry_requests.py::test_every_stage_names_its_runner` |
| One SAM3 / one tagger per process | `masking/_sam3.py::load_sam3` (cached on its args), `stages/_models.py::load_anima_tagger` (cached per checkpoint dir + device) |
| Daemon progress | `_progress.py`: with `ANIMA_DAEMON_JOB_DIR` set, `step()` appends the daemon's own `{"ev": "step", "global_step", "total_steps", "detail"}` line and `phase()` heartbeats every 30 s around a model load. Verified against `anima_daemon/tail.py` (filters on `ev` / `global_step`) and `manager.py` (exports the variable, reads the `progress.jsonl` mtime for stall liveness) |
| Round-trip + torch-free tests | `tests/test_registry_requests.py`, `test_stage_requests.py`, `test_masking_requests.py`, `test_progress.py`, `test_boundary.py::test_contract_is_torch_free` |

### Argv we emit today, parsed by HEAD's generated parsers

Probe of 2026-09-02: every wrapper's argv captured with `run()` stubbed and
fed through `registry.BY_ID[...].request_class().parser()` (this is the T1
test in throwaway form). All 17 `from anime_tools … import` sites in the
trainer also still resolve at HEAD.

| Call site | At `6a225f8` |
|---|---|
| `make mask` SAM step: `--config configs/sam_mask.yaml` | **breaks** — replaced by `--prompts` / `--focus-prompts` / `--threshold` / `--dilate` (the yaml keys map 1:1 onto `SamMaskRequest` fields) |
| `make mask` MIT step (`--model-path`, `--text-threshold`, `--dilate`, `--ctd-gate`) and merge step | OK (`use_mit` defaults on, so no `--use-mit` needed) |
| `caption-autotag`, `caption-position` (`--src/--dst/--path_pattern/--apply/--mode`) | OK |
| `correct_captions` incl. all four `--caption_*` correction flags and the variant flags | OK |
| `curate-group` (`--source-dir`), `caption-index` (`--src`) | OK (`captions.index` is not a registered stage; plain CLI) |
| `make autotag` forwarding `--device` | OK — the flag is still declared (`add_device_arg`); the earlier audit's "removed" claim was wrong |
| `caption-autotag --apply` then `make preprocess` | OK since request 1 landed — `correct` reads the revised caption first. T0's beta-gate check confirms it end to end |

Hand-copied constants across the seam, each with a "duplicated because torch"
comment and no drift test: autotag sentinels (`gui/tabs/_autotag.py`), autotag
modes (`scripts/tasks/preprocess.py::_AUTOTAG_MODES`), dbv4 checkpoint file
set (`scripts/tasks/downloads.py::TAGGER_CKPT_REQUIRED`). All three now exist
in `anime_tools.contract` with the same values. Free-fit geometry is a second
copy (`anime_tools/buckets.py` vs `library/datasets/buckets.py`).

## Target

`anime_tools` exposes one frozen request dataclass per stage in torch-free
modules, with `run_<stage>(req)` in-process and `req.to_argv()` for the
daemon. The trainer builds requests and never spells a flag. **This exists
now**; what is left is moving the trainer onto it.

What the trainer gains beyond drift safety: `make mask` today spawns three
interpreters and loads SAM3 twice (subject masks, then the MIT text masker's
SAM gate). Under the daemon the job's child can run all three stages in one
process with one SAM3 (`load_sam3` is already cached). Same for autotag →
position clauses (one tagger, `load_anima_tagger`). And the `_progress`
heartbeat ends the "quiet SAM3 load killed by the 120 s stall watchdog" class
without `--stall-timeout 0`.

## Package-side requests before the pin bump

All landed on `anime_tools` main on 2026-09-02, on top of `6a225f8`, as
version `0.4.0` (868 tests, ruff clean). Pin the commit that carries them
(tag `v0.4.0`). Recorded here as what was asked and what was decided; nothing
here reopens the design.

1. **Correction pass reads the revised caption first.** *Landed.*
   `stages/captions.py::write_corrected_preprocess_captions` resolves each
   image's caption through `_walk_captions.resolve_caption` — revised first,
   master as read-only fallback — like every other caption stage, and corrects
   the revised caption in place (`correct_caption` reorders the flat bag
   around its clauses). The autotag `merge` → `correct` chain keeps the merged
   tags, and a caption autotag `missing` created for an image with *no* master
   is no longer deleted as stale. The decision left open above was taken the
   first way: **once an image has a revised caption, a hand-edit of its master
   no longer reaches it** — edit the revised caption, or delete it to
   re-mirror. This is the rule `anime_tools/CLAUDE.md`, the contract row (2),
   the `--help` text and the registry note all now state, so the trainer's
   "edit `image_dataset/*.txt`, run `make preprocess`" habit changes for any
   image that already has `post_image_dataset/resized/*.txt`. Consequences the
   trainer sees at the pin bump:
   - `_reattach_clauses` is gone (dead under revised-first: there is no
     destination to reattach from when the master is read).
   - `PreprocessCaptionStats` changed shape: `missing_source` is now
     `no_caption` (neither caption exists; the only thing removed is an orphan
     variants sidecar), `removed_stale` is gone (a revised caption is never
     stale), and `from_master` is new (captions mirrored because no revised
     one existed). `clauses_preserved` now counts captions that carried
     clauses through, not reattachments. The printed epilogue changed with
     them; nothing in the trainer parses it.
   - Tests pinned in `tests/test_correct_captions.py` (package): merge then
     correct keeps the tags, a `missing`-mode caption survives, a clause on the
     revised caption survives re-correction, master-only mirrors, a master
     edit does not reach a revised caption, no caption at all counts and drops
     the orphan sidecar, a second run is a no-op.
   - One fix beyond the request, found while pinning the no-op: any caption
     carrying `@no-artist` rewrote its variants sidecar on every run (the
     sidecar's `v0` is the caption *minus* the sentinel, and the currency check
     compared the raw caption), i.e. a needless TE re-encode per
     `make preprocess-captions`. `_sidecar_is_current` now compares against the
     `v0` the generator would write.
2. **Revised-caption owner in `docs/contract.md`.** *Landed.* New row for
   `post_image_dataset/resized/**/{stem}.txt`: written by every caption stage
   (`correct` in place, autotag / position / audit rewrite, Export publishes
   `workspace/resized/` there), read by `preprocess-te` beside the resized
   image, revised-first for every reader on either side, every write pushed
   onto `{stem}.history.txt`. The master row's consumer is now the mirror
   (read only for an image with no revised caption) and the caption index —
   the TE step never read the master.
3. **`Stage.run` in the registry.** *Landed.* `Stage(…, run="module:function")`
   for all eleven stages, resolved by `Stage.runner()` with the same lazy
   `ImportError` contract as `request_class()`; the runners' model imports are
   inside their bodies, so resolving one loads no weights. T3/T4 can go from
   `BY_ID[id]` to `runner()(request)` without naming a runner.
4. **Version bump.** *Landed* in `pyproject.toml` + `uv.lock` as `0.4.0`;
   the `v0.4.0` tag is the release step (tag-driven `release.yml`).
   `CONTRACT_VERSION` stays `1`.
5. `captions.index` stays a plain CLI, as allowed; T1 special-cases it.

## Steps

Ordered. T0–T2 are the pin bump and can land in one PR once the requests
above are pinned; T3–T5 are the migration proper.

- [x] **T0. Pin bump** (blocks v2 Track A1). *Landed 2026-09-02*, pin `94e6d92` — see "What T0 found" below.
  - `scripts/tasks/masking.py::_run_sam`: translate `sam_mask.yaml`
    (`prompts`, `focus_prompts`, `threshold`, `dilate`, `path_pattern`) into
    the new flags; drop the tempfile yaml round-trip (`_sam_config_path`).
    The GUI's `SAM_MASK_CONFIG_JSON` env snapshot keeps the same keys.
    (Building a `SamMaskRequest` and calling `.to_argv()` here is already
    possible and is the T3 shape; either is acceptable for T0.)
  - `[tool.uv.sources]` rev → the `anime_tools` commit carrying requests 1–4
    (`v0.4.0`), `uv lock`, `make test-unit`, then one `make preprocess` +
    `make mask` on a small shard, and the beta-gate check: `make preprocess` →
    `make caption-autotag ARGS=--apply` → `make preprocess-te` keeps the
    autotag tags in the TE-cached caption.
  - Four trainer tests call `write_corrected_preprocess_captions` directly and
    pin the old master-first behaviour; they change with the pin:
    `tests/test_preprocess_dataset.py::test_write_corrected_preprocess_captions_removes_stale_missing_source`
    (a revised caption with no master is now *the* caption — `no_caption == 0`,
    the file stays; the package's own test covers the no-caption orphan
    sidecar), `tests/test_caption_variant_sidecars.py::test_caption_step_missing_source_removes_sidecar`
    (same: the sidecar is regenerated from the revised caption, not removed —
    drop the `a.txt` to test the orphan case, asserting `no_caption`),
    `::test_mirror_keeps_clauses_the_master_does_not_have` (still true;
    `clauses_preserved == 1` still holds) and
    `::test_a_hand_written_master_clause_wins_over_the_derived_one` (inverts:
    the revised caption wins, `clauses_preserved == 1`, "On the left" stays).
    The `preprocess-captions` docs / `captions` skill lose the "edit the
    master, re-run" phrasing for already-mirrored images.
- [x] **T1. Contract test.** *Landed 2026-09-02.* `tests/test_anime_tools_cli_contract.py`: import
  `anime_tools.stages.registry` **in-process** (it and the request modules are
  torch-free by the package's own test — no child interpreter needed), map
  each wrapper's `-m` module to its `Stage`, and assert the argv parses and
  builds via `request_class().from_namespace(parser().parse_args(argv))`.
  Source the argv from the wrapper functions themselves (`_run_sam`,
  `_run_mit`, `cmd_mask`'s merge call, `_caption_autotag_argv`,
  `_caption_position_argv`, `cmd_preprocess_captions`, `cmd_curate_group`)
  with `run()` stubbed, so the test reads the real argv. `captions.index` is
  checked by `--help` text or skipped. Also assert every trainer
  `from anime_tools … import` name resolves (a module-import sweep).
- [ ] **T2. Constants from `anime_tools.contract`.** Replace the three hand
  copies with imports (values are identical today, so this is mechanical);
  the trainer GUI stays torch-free by construction since the module is.
  Assert `anime_tools.contract.CONTRACT_VERSION == 1` at import in
  `scripts/tasks/_common.py` with a clear "bump the pin" message.
- [ ] **T3. Masking through requests.** `scripts/tasks/masking.py::cmd_mask`
  builds `SamMaskRequest` / `MitMaskRequest` / `MergeMasksRequest`. Under a
  daemon job (`ANIMA_DAEMON_JOB_DIR` set) call the runners in-process
  (`Stage.run` after request 3, else `anime_tools.masking.run_*`) so the
  cached `load_sam3` is shared; without one, keep a subprocess per stage so
  `make mask` from a shell still releases VRAM on exit. Drop `RUN_SAM_MASK` /
  `RUN_MIT_MASK` / `MIT_*` env plumbing in favour of request fields the GUI
  fills. Stop passing `--batch-size 4` / `--checkpoint` / `--model-path`
  literals: the request defaults come from the package's `downloads.py`
  catalog, which is where the weights actually land.
- [ ] **T4. Caption stages + grouping through requests.**
  `_caption_master_argv` → `AutotagRequest.to_argv()` /
  `PositionRequest.to_argv()`; `cmd_preprocess_captions` →
  `CorrectRequest`; `curate.py` → `GroupRequest`. Scope resolution
  (`_resolved_path_pattern_args`) becomes one `path_pattern=` field. Under a
  daemon job, autotag → position in one process shares `load_anima_tagger`.
- [ ] **T5. Free-fit geometry has one owner.** Move `EDGE_TOKEN_BANDS`,
  `freefit_bucket`, `choose_edge` and the resize solver into
  `anime_tools.buckets` / `anime_tools.stages.resize`; `library/datasets/
  buckets.py` re-exports them the way `library/models/pe.py` re-exports the
  PE tower. Delete `FREEFIT_BAND_VERSION` cross-checking as moot. Trainer-only
  helpers (`token_count_families`, `cluster_token_bands`, σ-demote helpers)
  stay here. `make preprocess-resize` then calls `ResizeRequest`. This is the
  one step with real package-side work still open.
- [ ] **T6. Docs.** `CLAUDE.md` "Curation lives in anime_tools" paragraph:
  the request API is the front door, `python -m` is the shell; the `captions`
  and `daemon` skills lose their hand-written argv examples.
  `docs/v2_release_plan.md` Track D: drop the `--device` item from D0, point
  at `../anime_tools/CLAUDE.md` instead of the deleted plan file, and fix A5
  wording.

## What T0 found (2026-09-02)

Landed as one change (trainer + two package commits); the deviations from the
plan above, in the order they bit:

1. **The trainer could not lock at all** — not the pin, the sibling checkout.
   `anime_tools` had grown a `[tool.uv.sources]` torch → cu130 index for win32
   (package commit `ad93fdb`), and uv carries a path dependency's sources into
   the consumer's resolve, so the `anime-tools-dev` editable group made every
   trainer `uv lock` fail with "conflicting indexes for torch (cu130 vs
   ROCm)" — the [[project_uv_run_resolve_broken]] symptom, root-caused. The
   package dropped the block (`d9bbaa3`): the trainer owns the Windows backend
   via its `cuda-windows` / `rocm-windows` groups; a standalone Windows install
   passes `--index` at sync time. `uv lock` / `uv sync` work again.
2. **`rules:` is gone from the package but the GUI still sends one.**
   `sam_section.collect_rules()` → `SAM_MASK_CONFIG_JSON = {"rules": […],
   "path_pattern": …}`. `scripts/tasks/masking.py` now normalizes both schemas
   itself (`_sam_rules`) and builds one `SamMaskRequest` per rule (`to_argv()`,
   the T3 shape) into its own temp dir; the merge step's pixel-min union is the
   old compose (ignore regions unioned). A rule with its own `path_pattern`
   runs on that pattern alone (the package takes one glob per run — the old
   "global scope ∩ rule pattern" is not expressible; global scope still applies
   to every rule without one). An empty `focus_prompts` is spelled
   `--focus-prompts none` (the request defaults it to the subject prompt).
   Per-rule `threshold`/`dilate` fall back to the top-level values, then to the
   package defaults — the trainer no longer carries `0.5`/`5`. Pinned in
   `tests/test_masking_task.py`; the argv round-trips in the T1 test.
3. **One import broke**: `library/anima/training.py` re-exported the private
   `_is_artist_tag`, which became `anime_tools.captions.taxonomy.is_artist_tag`.
   Nothing in the trainer called it beyond the re-export (one GUI comment, one
   test); dropped. The other 42 import names resolve (T1 sweeps them).
4. **`captions.index` crashed at its final write** at 0.4.0 — the CLI resolves
   `--vocab` to a `Path` and stored it in `meta` (`TypeError: PosixPath is not
   JSON serializable`). Fixed in the package (`94e6d92`, with a test) — hence
   the second pin. Its default `--out` also moved to the package's
   `workspace/captions/`; the trainer now passes `--out
   post_image_dataset/captions/caption_index.json` (`CAPTION_INDEX_PATH`) so
   `train.py` / `configs/easycontrol/*.toml` keep reading where they always did.
5. **Five tests inverted, not four**: the listed four plus
   `test_mirror_picks_up_a_master_edit_around_the_clauses` (now
   `test_a_master_edit_does_not_reach_a_revised_caption`, which also pins the
   "delete the revised caption to re-mirror" path via `from_master`).
6. The package folded its `stages` / `grouping` / `masking` extras into the
   base requirements; the trainer's group specs are plain `anime-tools` now.
   The pin drags `transformers` 5.10.1 → 5.16.1 (`anime_tools` requires
   ≥ 5.16); the scratch preprocess + TE encode below ran on it.
7. Validation ran **inline, not through the daemon**: the queue was serial
   behind a running CJK cache job and a queued cold-joint train. On a 5-image
   scratch copy of real images (two without captions) via a `CONFIG_FILE`
   snapshot: `make preprocess` (resize → VAE → captions → TE → index) passed.
   **Still owed**: the beta-gate chain (`caption-autotag --apply` →
   `preprocess-te` keeps the autotag tags in the TE-cached caption) — the
   `caption-autotag` target submits to the daemon itself, and the job queued
   behind the CJK cold-joint runs, so it was cancelled; rerun once the queue is free
   (scratch set + `CONFIG_FILE` snapshot are the recipe; the config lives only
   in the session scratchpad, so rebuild it: five images, two without captions).

## Guards

- `tests/test_curation_boundary.py` is unchanged: dependency direction stays
  trainer → `anime_tools`.
- T1's contract test is the drift alarm until T3/T4 remove the argv entirely;
  after that it still guards `to_argv()` ↔ parser agreement from our side.
- `make daemon-run`, `--queue`, and `make gen` are unaffected: the daemon
  still receives argv, produced by `to_argv()` instead of by hand.
- The `device` field is in the package GUI's `AUTO_FIELDS`: `to_argv()` omits
  it at its default and the child resolves the device itself, matching what
  the wrappers do today. T3/T4 must not start passing it.

## Out of scope

- Routing the `anime_tools` web GUI's own job runner through `anima_daemon`
  (a package-side decision; left open there).
- The resident autotag worker protocol stays a stdio sentinel stream
  (`contract.AUTOTAG_*`).
