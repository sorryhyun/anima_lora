# Preprocess panel drawn from `anime_tools` stage schemas

Status: **P0–P3 landed 2026-09-07** — the live Preprocessing tab draws the
image-prep, auto-tag, caption-rewriting and SAM-mask sections from the
`resize` / `autotag` / `correct` / `masks_sam` schemas (`stage_form.py`,
`tests/test_gui_stage_form.py` 14 tests + the regenerated characterization
fixture); the `CAPTION_*` env ladder for the migrated knobs and
`MASK_CONFIG_JSON` are gone, replaced by one `PREPROCESS_STAGES_JSON` form
payload that `scripts/tasks/_common.request_from_form` turns into requests
through the package's `build_argv`; the MIT masker is removed trainer-side
(v2 checklist §2); the i18n overlay (option 2 in §4) is
`gui/explanations/guides/<lang>/_stage_fields.json`. See **§7 Landed** for
the deltas from the plan. P4 (`ocr` / `groups` / `audit` panels) stays
open. Question this answers (repo owner): *"how can we set the trainer's
Preprocess panel to be anime_tools'? I guess we can make anime_tools emit a
config list or something."*

Short answer: the config list already exists and is torch-free; the trainer
needs a renderer over it, a binding table for its roots, and a decision on
translations.

## 1. The emitter already exists

`../anime_tools/anime_tools/gui/stages.py` is what the package's web GUI draws
its forms from, and it is exactly the "config list":

- `schema(stage)` walks `_request.args_of(Request)` — the same field list the
  CLI parser is generated from — and emits one dict per field; `dump_schemas()`
  / `load_schemas()` do it for every registered stage (`stages/registry.py`,
  eleven today: `resize autotag position correct audit ocr groups masks_sam
  masks_mit masks_merge export`).
- `build_argv(schema, values, *, apply, roots, settings, report_root,
  mask_root)` turns a `{dest: value}` payload back into the request's argv:
  coerced per field kind, read through `Request.from_namespace` (so the
  request's own `__post_init__` raises `ValueError` *before* a job exists),
  spelled by `to_argv()` (a value at the request default is omitted).
- `load_parser(stage)` is the stage's generated argparse parser — the
  round-trip oracle.

Measured in the trainer venv (2026-09-07, `.venv/bin/python`):

| Probe | Result |
|---|---|
| `python -X importtime -c "import anime_tools.gui.stages"` | **28 ms** cumulative; no `torch`, `fastapi`, `starlette`, `pydantic`, `numpy`, `PIL`, `cv2` in the import log. `anime_tools/gui/__init__.py` only defines `main()` and imports the server lazily inside it, so the subpackage import is clean. |
| `dump_schemas()` | **0.13 s** for all 11 stages, 186 fields (2026-09-07 HEAD of the sibling checkout, mid PP-OCR removal); every stage `available: True` in this venv |
| `import gui.tabs.preprocess.stage_form` | 0 torch/fastapi frames (the pilot module) |

So **no move is needed**: a Qt GUI can `from anime_tools.gui.stages import
load_schemas, build_argv, load_parser` as-is. The relocation to
`anime_tools.stages.schema` proposed in the brief is a hygiene option, not a
requirement — the only thing arguing for it is that `anime_tools.gui` *reads*
as "web server" and a future contributor may add an eager fastapi import to
its `__init__`. If the owner wants the guarantee, the cheap form is a
package-side test asserting the import stays server-free (the package already
has `tests/test_boundary.py::create_app()`-without-torch; this is the mirror
image), not a module move.

Field shape as emitted (`ocr.det_conf`, verbatim):

```json
{"dest": "det_conf", "kind": "float", "flags": ["--det_conf", "--det-conf"],
 "default": 0.25, "choices": null,
 "help": "Keep a detected box scored at least this (0-1). The model card's F1 threshold is 0.426; 0.25 boxes ~15%% more, nearly all real text on a manga page",
 "required": false, "path": false, "path_kind": "file", "group": "Detector",
 "negate": null, "label": "--det_conf",
 "root": null, "setting": null, "report": null, "mask": null,
 "auto": false, "overridable": false, "advanced": false, "gate": null}
```

`kind ∈ {bool,int,float,str,enum,list}`; `root`/`setting`/`report`/`mask` are
the **bindings** (§3) — non-null means "hidden, filled by `build_argv`";
`gate` is the dest of the bool this field hangs off (the gate names itself);
`advanced` is the fold; `path`/`path_kind` pick the chooser. Two gaps worth
knowing: a `list` field carries **no item type** (`target_res` is a list of
ints, `resize_crop_margins` of four floats, `skip` of strings — the renderer
can't tell), and a `str` field whose request has a `read`/`write` pair
(`prompts` → `prompt_list`) arrives as its argv spelling (`"none"` for empty),
which the form must show as-is.

## 2. Mapping — today's sections and knobs onto stages

`gui/tabs/preprocess/knobs.py` has 27 rows. Where each goes:

| Trainer section / knob (`knobs.py`) | Stage · field (schema) | Note |
|---|---|---|
| **ImagePrepSection** ↔ **`resize`** (14 fields) | | |
| `source_image_dir` | `resize.src` — bound root `src` | trainer supplies via `roots` |
| `path_scope` | — | **trainer-native**: `ConfigTab._gui_scoped_paths` offsets resized/lora/mask roots; becomes the `roots` the trainer hands `build_argv` |
| `preprocess_path_pattern` | `path_pattern` — bound setting | supplied via `settings` |
| `drop_lowres_images` | — | **trainer-only**: `ResizeRequest` has only `min_pixels`; the switch collapses to `min_pixels=0` (`_min_pixels_args`) |
| `min_pixels` | `resize.min_pixels` | direct |
| `target_res` | `resize.target_res` (list, no item type) | keep `_TargetResWidget` as the domain widget; the schema kind is too weak |
| `resize_bucket_resos` | — | **dead at the request level**: no stage field; only `library/preprocess/resize_preview.py` still accepts it "for signature compatibility" |
| `resize_crop_anchor` | `resize.resize_crop_anchor` (enum, 9 choices) | the 3×3 picker stays a domain widget over the same enum |
| `resize_crop_margins` | `resize.resize_crop_margins` (list of 4) | domain widget |
| `freefit_max_ratio` | `resize.freefit_max_ratio` | direct |
| *(stage has, trainer lacks)* | `recursive` `copy_captions` `overwrite` `workers` `skip` `report_dir` | `overwrite` is the one users ask for |
| **TextCachingSection** — **trainer-native** | | |
| `caption_shuffle_variants`, `caption_tag_dropout_rate` | *(are* `correct.caption_shuffle_variants` / `correct.caption_tag_dropout_rate`*)* | `scripts/tasks/preprocess.py` already builds a `CorrectRequest` from them, but they are consumed by the trainer's TE cache pass (variants sidecar → `make preprocess-te`); stay here until the TE stage itself is schema'd. `caption_tag_randomize_rate` exists on `correct` and not in the GUI. |
| σ-demote (`preprocess-demote`), VAE/TE/PE caches | — | trainer scripts, no request; stay |
| **AutotagSection** ↔ **`autotag`** (10 fields, 2 basic) | | |
| `caption_autotag` | — | **chain gate** (whether the Run chain runs the stage) — trainer-native, not a stage field |
| `caption_autotag_mode` | `autotag.mode` (enum `missing/merge/overwrite`) | direct; `CAPTION_AUTOTAG_MODES` literal in `knobs.py` goes away |
| `caption_autotag_min_confidence` | `autotag.min_confidence` | direct |
| *(lacks)* | `from_report` (replay), `report_dir` (bound), `tagger_dir` (bound setting) | |
| **CaptionEditingSection** ↔ **`correct`** (15, 5 basic) **+ `position`** (45, 7 basic) | | |
| `caption_correct_order` | `correct.no_correct` **inverted** | `scripts/tasks/preprocess.py` sets `fields["no_correct"] = True` when off; the form should show the package's switch and drop the inversion |
| `caption_insert_no_artist`, `caption_trigger_word`, `caption_trigger_at_front` | same dests on `correct` | direct (`trigger_at_front` is `advanced` in the package) |
| `caption_position_clauses` | — | **chain gate** for `position`; the stage's 45 knobs (7 basic: `crops flatten prompt score_threshold min_instances max_instances rewrite`) have no GUI today |
| *(lacks)* | `correct.caption_drop_groups` (GH #95 — **already a basic field**), `caption_tag_randomize_rate`, `tag_csv`, `qwen3`, `t5_tokenizer_path` | Phase 2 of `gui_preprocess_tab_refactor.md` becomes "render the schema" |
| **SamMaskSection** ↔ **`masks_sam`** (14, 5 basic) | | |
| `run_sam_mask` | — | chain gate |
| `mask_path_pattern` | `path_pattern` — bound setting | shared by both mask backends today (`_config_path_pattern`) |
| `mask_rules` (cards: pattern / prompts / focus / threshold / dilate) | **N × `masks_sam`** requests: `prompts` `focus_prompts` `threshold` `dilate` + per-rule `path_pattern` | the *rule list* is the trainer's construct (`cmd_mask` runs one SAM pass per rule into a tempdir, then `masks_merge`); one `StageFormSection` per card, list of value dicts persisted |
| *(lacks)* | `force` `workers` `recursive` `batch_size`; `prompt_embed` `checkpoint` (bound settings) | |
| **MitMaskSection** ↔ **`masks_mit`** (16, 6 basic) | | |
| `run_mit_mask` | — | chain gate |
| `mit_text_threshold`, `mit_dilate` | `masks_mit.text_threshold`, `masks_mit.dilate` | direct |
| *(lacks)* | `use_sam`/`sam_prompts`/`sam_threshold` drawer, `use_mit`/`model_path`/`ctd_gate` drawer | **being removed in v2** (`docs/v2_release_checklist.md` §4 "MIT removal") — do **not** build this section; P2 renders `masks_sam` only |
| **Stages the trainer GUI lacks entirely** | `ocr` (15, 3 basic — shrinking as PP-OCR leaves), `groups` (12, 2 basic), `audit` (32, 7 basic), `export` (11), `position` as a form, `masks_merge` (2, both bound) | all free once the renderer exists |
| **Stays trainer-native** | Run chain + split buttons (`_run_te` / `_run_pe` / `_run_mask`), `preprocess_env` / `MASK_CONFIG_JSON`, `preprocess_config_snapshot`, `path_scope`, `[variant]` persistence, TE/VAE/PE/demote caches | the GUI is the orchestrator; the package is the stage list |

## 3. Bindings — the package's roots and settings vs the trainer's paths

`anime_tools.gui.stages.ROOT_FIELDS` names, per stage, which dests are dataset
roots (`src` / `dst` / `masks` / `master` / `out`); `SETTING_FIELDS` names the
stage-independent knobs (`path_pattern` / `tagger_dir` / `checkpoint` /
`prompt_embed`); `MASK_FIELDS` and the `Stage.report` dest bind each
generator's / stage's own *tail* under one `mask_root` / `report_root`. Every
bound field is hidden and filled by `build_argv`, so **the trainer never lets
a stale saved path win over the checkout's roots** — the same property
`knobs.py` gets from `default_from="preprocess_toml"` today, but in the
package's code.

| Package root / setting | Trainer source |
|---|---|
| `src` | `source_image_dir` (`configs/preprocess.toml`, default `image_dataset`) + `path_scope` offset |
| `dst` | `resized_image_dir` from the merged chain (`post_image_dataset/resized`, scoped) |
| `masks` | `mask_dir` from `configs/preprocess.toml` (`post_image_dataset/masks`, scoped as `_scoped_mask_output_dir` does) |
| `master`, `out` | export-only; `out` = `post_image_dataset` — the trainer *is* the export target, so Export stays off the trainer panel |
| `path_pattern` | `preprocess_path_pattern` / `mask_path_pattern` knobs |
| `tagger_dir`, `checkpoint`, `prompt_embed` | `library/downloads.py` catalog rows (`tagger`, `sam3`, the prompt embed) — the Models panel already knows where they land |
| `report_root` | new: a `post_image_dataset/reports/` (or blank = beside `dst`, the package default) |
| `mask_root` | a tempdir per run, exactly what `cmd_mask` does today — the merge writes to `masks` |

Execution is already there: `scripts/tasks/_common.py::execute_stage(stage,
req)` runs a request **in-process** under a daemon job (`Stage.runner()(req)`,
one interpreter per chain, SAM3/tagger caches shared, `ANIMA_HOME` pinned)
and as a `python -m <stage.module> *req.to_argv()` child from a shell;
`request_with_args(req, extra)` applies a user's `ARGS` through the request's
own parser. So the Qt panel's submit is: `argv = build_argv(...)` →
`req = Stage.request_class().from_argv(load_parser(stage), argv)` → hand
`stage.id` + `argv` to a `tasks.py` entry that calls `execute_stage`. The
`MASK_CONFIG_JSON` env snapshot and the `CAPTION_*` env ladder both retire in
favour of argv the stage's parser validates; a `[variant.stages.*]` dict
(§5) is what gets persisted, and argv is derived from it at submit time.

## 4. The i18n gap — the real design decision

The trainer ships four-language labels and tooltips: `gui/i18n/{en,ko,ja,cn}.py`
(`preprocess_*` keys) and `gui/explanations/guides/<lang>/_preprocess_fields.json`
(28 per-field help entries, English fallback). The audience is KR/CN/JP-heavy
(`project_user_community_audience`). The package's schemas carry **English
`help` only**; its web GUI translates panel and stage *names*
(`../anime_tools/frontend/src/i18n/{en,ko,ja,zh}.ts`, keyed by registry id)
and by design ships argparse labels and help "as they arrive". Rendering
schemas verbatim would therefore be an i18n **regression** for most of the
base — that is the decision, not the widget code.

Options:

1. **Package ships the table.** `anime_tools` adds a translation table keyed
   `<stage_id>.<dest>` → `{label, help}` per language (`en` schema-checked
   like its `Dict` type), served in `schema()` as `i18n: {ko: {...}, ...}` or
   beside it. Both GUIs read one source; a new request field needs one row in
   one repo; the package's own web GUI gains field translations it lacks
   today.
2. **Trainer overlay, same key.** `gui/explanations/guides/<lang>/_stage_fields.json`
   keyed `<stage_id>.<dest>` → `{label, help}`; `StageFormSection.label_for`
   / `help_for` consult it, fall back to the schema's English. Lives with the
   `translator` agent's existing surfaces; drift is the trainer's problem.
3. **Accept English** for the stage forms.

**Recommendation: (1), with (2) as the transition.** The key scheme is the
same in both, so an overlay written now is the seed of the package table
later — copy the JSON across, delete the overlay. The pilot already exposes
the two hooks (`label_for` / `help_for` on `StageFormSection`, module-level
defaults in `stage_form.py`), so (2) is a JSON file plus ~10 lines. (3) is
what the pilot renders today and is acceptable *only* for the stages the GUI
never had (`ocr`, `groups`, `audit`) — not for replacing translated sections.

What breaks `tests/fixtures/gui_preprocess_knobs.json`: the fixture pins
`KNOBS` byte-for-byte — `to_env` output, `[variant]` meta elision, snapshot
keys. A section that reads its rows from a schema **is not in `KNOBS`**, so
every knob a phase migrates off `knobs.py` (P1: `caption_autotag_mode`,
`caption_autotag_min_confidence`, `caption_correct_order`,
`caption_insert_no_artist`, `caption_trigger_word`,
`caption_trigger_at_front`; P2: `mask_*`, `mit_*`; P3: the image knobs) drops
out of `env` / `meta_full` / `overrides` in the fixture, and the
`CAPTION_*` env names `scripts/tasks/preprocess.py` reads go with them. That
is a deliberate `--write` regeneration per phase, with the removed keys listed
in the commit — the fixture's job is to make the change visible, not to
forbid it. `PREPROCESS_ONLY_KEYS` (ConfigTab's training-snapshot strip list)
shrinks the same way; `[variant.stages]` must be added to the strip list so it
never reaches `train.py`.

## 5. Persistence

Today: flat keys in `configs/gui-methods/<variant>.toml` `[variant]`, elided
by `knobs.merge_into_meta` under five `persist` policies, and SAM rules as a
`mask_rules` list; `configs/sam_mask.yaml` is only the CLI fallback.

With schemas: one dict per stage under `[variant.stages.<stage_id>]`,
holding only the form's own values (`anime_tools.gui.stages.form_values`
strips bound/auto dests — a saved root can never resurrect). Elision is
`to_argv()`'s: a value at the request default is not spelled, so saving the
full dict costs nothing at submit and the TOML stays readable; if the owner
wants the "plain checkout keeps an empty meta" property, drop keys equal to
the schema `default` at save. Validation is `build_argv`'s `ValueError` at
Save *and* Run — the SAM "nothing to mask" refusal fires in the dialog, not
after a SAM3 load. `sam_mask.yaml` rules become
`[[variant.stages.masks_sam]]` — a list of `masks_sam` value dicts, each a
complete request minus the bound roots; `mask_path_pattern` folds into each
entry's `path_pattern` (a rule with its own pattern already runs alone, see
`_sam_request`). The chain gates (`caption_autotag`,
`caption_position_clauses`, `run_sam_mask`) stay flat `[variant]` keys: they
are not stage fields.

## 6. Phases

| Phase | Scope | Gate |
|---|---|---|
| **P0** (done) | `stage_form.py` (Qt-free `load_stage_schemas` / `visible_fields` / `argv_for` / `knob_for`; Qt `StageFormSection`) + this doc | `tests/test_gui_stage_form.py`: schemas load for 4 stages, bound roots hidden, argv round-trips through `load_parser` for `masks_sam` + `ocr`, request validation surfaces as `ValueError`, offscreen render of `masks_sam` / `ocr` / `export` (values, argv, fold, gate, browse) |
| **P1** | Replace `AutotagSection` + `CaptionEditingSection` with `StageFormSection(autotag)` / `(correct)`; chain gates stay as checkboxes on the tab; `caption_drop_groups` arrives for free; `[variant.stages.{autotag,correct}]` persistence; the `(2)` overlay JSON for their ~10 fields in 4 languages | fixture regenerated with the 6 removed keys listed; `tests/test_anime_tools_cli_contract.py` still re-parses every emitted argv; a new test that a saved `[variant.stages.correct]` round-trips to the same argv the old env ladder produced for the default variant |
| **P2** | `SamMaskSection` → list of `StageFormSection(masks_sam)` cards; `MitMaskSection` **deleted** (v2 MIT removal), `MASK_CONFIG_JSON` retired for argv; `mask_root` tempdir + merge unchanged | `make mask` from the GUI and from the shell produce identical merged masks on the test dataset; fixture regenerated |
| **P3** | `ImagePrepSection` over `resize` with the three domain widgets kept (`target_res`, anchor, margins) mapped by dest; `drop_lowres_images` becomes a trainer-side sugar over `min_pixels`; `resize_bucket_resos` removed; `overwrite` surfaced | `tests/test_gui_preprocess_knobs.py` shrinks to the trainer-native rows (`path_scope`, text caching, chain gates); launch-speed guard unchanged |
| **P4** (optional) | `ocr` / `groups` / `audit` panels in the Experimental tab, English help (option 3) until the package table exists | render test per stage |

Not in scope: moving `anime_tools.gui.stages` (§1 — unnecessary), Export on
the trainer panel (§3 — the trainer is the export target), the TE/VAE/PE cache
stages (no request exists; a `TextCacheRequest` in the trainer would be the
first *trainer-side* request, a separate proposal).

## 7. Landed (2026-09-07) — deltas from the plan

- **Transport is form values, not argv.** The tab sends every stage form as
  `PREPROCESS_STAGES_JSON = {stage_id: {dest: value}}` (`masks_sam`: a list,
  one per card) inside `preprocess_env()`, so the Train auto-chain gets it
  for free. `scripts/tasks/_common.request_from_form(stage_id, values, *,
  roots, settings, mask_root, apply, **overrides)` runs `build_argv` with the
  **trainer's** roots (`_path(...)`, path_scope-scoped) and then
  `dataclasses.replace` for the dests the chain owns — so a saved form can
  never carry a stale path, and validation runs twice (form → request →
  replace). The GUI validates at Save/Run with placeholder roots
  (`_VALIDATION_ROOTS`) and shows the `ValueError` in a dialog
  (`preprocess_invalid_stage`).
- **Trainer-owned dests are hidden** (`stage_form.TRAINER_FIELDS`):
  `resize.{recursive,copy_captions,skip}`, `autotag.from_report`,
  `correct.{recursive, caption_shuffle_variants, caption_tag_dropout_rate,
  caption_tag_randomize_rate, qwen3, t5_tokenizer_path}`,
  `masks_sam.recursive`. `masks_sam.path_pattern` is a *shown* bound field
  (`SHOWN_BOUND`): each card carries its own scope and `make mask` threads it
  as the run's setting, so the global `mask_path_pattern` knob is gone.
- **`no_correct` is shown as the package's switch**, seeded from
  `preprocess.toml`'s `caption_correct_order` inverted (default: no
  reordering, as before). The trainer keeps its rule that a trigger word /
  `@no-artist` / drop-groups force the correction pass (the package injects
  nothing under `correct=False`); the overlay help says so.
- **Seeds and elision.** `stage_form.seeded_defaults` layers
  `configs/preprocess.toml` over the schema defaults for the dests it names
  (`_PP_SEEDS`); `[variant.stages.<id>]` keeps only values ≠ the seeded
  default (`target_res` always — `ALWAYS_PERSIST`). SAM cards seed from
  `configs/sam_mask.yaml` (`knobs.load_rules`) when the variant has no
  `[[variant.stages.masks_sam]]`; an empty prompt list is the package's
  `none` spelling, shown as-is.
- **Removed**: knob rows `min_pixels target_res resize_bucket_resos
  resize_crop_anchor resize_crop_margins freefit_max_ratio
  caption_correct_order caption_insert_no_artist caption_trigger_word
  caption_trigger_at_front caption_autotag_mode
  caption_autotag_min_confidence run_mit_mask mask_path_pattern mask_rules
  mit_text_threshold mit_dilate`; env `MIN_PIXELS TARGET_RES
  FREEFIT_MAX_RATIO CAPTION_CORRECT_ORDER CAPTION_INSERT_NO_ARTIST
  CAPTION_TRIGGER_WORD CAPTION_TRIGGER_AT_FRONT CAPTION_AUTOTAG_MODE
  CAPTION_AUTOTAG_MIN_CONFIDENCE MASK_CONFIG_JSON`; `MitMaskSection`,
  `_mit_request` / `MIT_MODEL_PATH` / `run_mit` in `scripts/tasks/masking.py`,
  the MIT i18n strings, the `_TargetResWidget` bucket popup. `DROP_LOWRES_IMAGES`
  stays (sugar → `min_pixels=0`); `scripts/tasks/preprocess.py::_min_pixels_args`
  and `_config_target_res` read the resize form when no env is set.
- **Kept flat in `[variant]`**: `source_image_dir path_scope
  preprocess_path_pattern drop_lowres_images caption_shuffle_variants
  caption_tag_dropout_rate caption_position_clauses caption_autotag
  run_sam_mask` (+ the `stages` sub-table, in `PREPROCESS_ONLY_KEYS`).
- **Not done**: the P2 GPU gate (GUI vs shell `make mask` on the test
  dataset) was not run; the argv/request equivalence is pinned by
  `tests/test_anime_tools_cli_contract.py` and `tests/test_masking_task.py`
  instead. P4 untouched.

## Pilot notes

- `StageFormSection` follows `_section.py`'s `add_knob(key, widget, label, *,
  tooltip, field)` contract but keys a **private** knob table
  (`knob_for(field)` → `Knob`) instead of `KNOBS_BY_KEY`, so
  `read_widget` / `set_widget` dispatch is reused unchanged and nothing in
  `knobs.py` or the fixture moves. `gate` → `Knob.enabled_by`; `advanced` →
  a `QCheckBox(t("advanced_section"))` toggling a flat `QGroupBox`;
  `path` → line edit + neutral `…` `QPushButton` (no action-button colour —
  it is a chooser, not a Run).
- The i18n hooks are `label_for(field)` / `help_for(field)`; today they
  humanise the schema `label` (`--min_score` → `min score`) and return the
  English `help`.
- "Qt-free" here means no widget / no `QApplication`: `gui/__init__.py`
  imports PySide6 eagerly (via `gui.dialogs`), so no `gui.*` module imports
  with PySide6 absent — `knobs.py` included. The module's `try: import
  PySide6` guard is defensive only.
- Evidence the round-trip gate earns its keep: while this pilot was written
  the package dropped `OcrRequest.min_score` (PP-OCR removal); the OCR tests
  failed on the *next* run with `Namespace has no attribute 'min_score'` —
  the same alarm `tests/test_anime_tools_cli_contract.py` gives the `make`
  wrappers, now covering the form.
- `int`/`float` fields whose default is `None` render as free text
  (`build_argv` coerces; blank → request default); spins use a 32-bit int
  range and 4 decimals, which is enough for every current default
  (`min_area_frac=0.005`).
