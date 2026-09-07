# Preprocessing tab refactor — declarative knobs + section panels

Status: **Phases 0, 1 and 3 landed 2026-08-28** (`gui/tabs/preprocess/` package; the tab is 2043 → 1019 lines + ~930 lines of section modules). **Phase 2 superseded 2026-09-07**: `caption_drop_groups` (and every other `correct` / `autotag` / `resize` / `masks_sam` knob) now arrives from the stage schema — see `gui_preprocess_from_anime_tools.md` §7; the `tag_groups` chip widget below was not built (a plain text row over the schema's `str` field). Phase 4 open. Motivating change: GH #95 caption-group removal
(`--caption_drop_groups`, landed in the pipeline 2026-08-28, **not yet in the GUI**).

## Why now

`gui/tabs/preprocess_tab.py` is 2043 lines and still growing linearly with every
knob. The cost isn't the widget code — it's that **one knob touches 7–9 sites**,
none of which is checked against the others:

| # | Site | Example (`caption_autotag_mode`) |
|---|------|----------------------------------|
| 1 | `DEFAULT_*` constant | `DEFAULT_CAPTION_AUTOTAG_MODE = "missing"` |
| 2 | Widget build in `__init__` (inside a `QGroupBox` form) | ~25 lines |
| 3 | `set_variant()` load — 2–3 level fallback (`meta → preprocess.toml/gui_settings → DEFAULT`) | 4 lines + 1 `setValue` |
| 4 | `_connect_dirty_signals()` tuple | 1 entry |
| 5 | `preprocess_env()` — env name + serialization | 1–3 lines |
| 6 | `preprocess_overrides()` — snapshot key | 1 line |
| 7 | `_save_variant_preprocess_meta()` — default-elision (`pop` if equal to default, with the `_pp_default` subtlety) | 6 lines |
| 8 | `guides/*/_preprocess_fields.json` ×4 langs + `i18n/*.py` ×4 | 8 files |
| 9 | (sometimes) `_GUI_PREPROCESS_KEYS` in ConfigTab's strip list, `tests/test_gui_snapshot_preprocess_keys.py` | |

Three different "where does the default come from" policies coexist (hardcoded;
`preprocess.toml`; `gui_settings.json`) and the elision rule at site 7 has a
known trap (the four caption-master stages must compare against the
*toml-resolved* default or an unchecked box won't stick). Every new knob is a
chance to get one of those nine sites wrong, and the file is now too long to
see them all on one screen. The masking section (`_RuleCard`, SAM/MIT) is a
separate mini-app living in the same class.

## Target shape

```
gui/tabs/preprocess/
  __init__.py          # re-exports PreprocessingTab (import path unchanged)
  knobs.py             # Qt-FREE: the knob table + load/serialize/elide logic
  tab.py               # PreprocessingTab: top bar, run buttons, job observer, log
  _section.py          # KnobSection base: builds a QGroupBox form from knob specs
  image_prep.py        # source dir / scope / pattern / lowres / target_res / crop / freefit
  text_caching.py      # shuffle variants / tag dropout (/ randomize rate)
  captions.py          # autotag box + caption-editing box (+ NEW drop-groups row)
  masking.py           # SAM rule cards + MIT (moved out of the tab class)
gui/tabs/preprocess_tab.py   # 3-line shim: `from gui.tabs.preprocess import *`
```

### `knobs.py` — the single source of truth (Qt-free, like `config_io.py`)

```python
@dataclass(frozen=True)
class Knob:
    key: str                       # TOML / snapshot / variant-meta key
    section: str                   # "image" | "text" | "autotag" | "captions" | "mask"
    kind: Literal["bool","int","float","str","choice","tier_list","crop_anchor","margins","tag_groups"]
    default: object                # hardcoded fallback
    default_from: Literal["const","preprocess_toml","gui_settings"] = "const"
    env: str | None = None         # CAPTION_DROP_GROUPS …; None = snapshot-only
    choices: tuple[str, ...] = ()  # for kind="choice"
    persist: Literal["variant","variant_if_changed","mask_only"] = "variant_if_changed"
    enabled_by: str | None = None  # key of a bool knob that gates this one
    snapshot: bool = True          # goes into preprocess_overrides()

KNOBS: tuple[Knob, ...] = (
    Knob("source_image_dir", "image", "str", "image_dataset", default_from="preprocess_toml"),
    …
    Knob("caption_autotag", "autotag", "bool", False, default_from="preprocess_toml", env="CAPTION_AUTOTAG"),
    Knob("caption_autotag_mode", "autotag", "choice", "missing", default_from="preprocess_toml",
         env="CAPTION_AUTOTAG_MODE", choices=("missing","merge","overwrite"), enabled_by="caption_autotag"),
    Knob("caption_drop_groups", "captions", "tag_groups", (), default_from="preprocess_toml",
         env="CAPTION_DROP_GROUPS"),                       # ← GH #95, one row
    …
)
```

Pure functions over the table (all unit-testable without Qt):

- `resolve_default(knob, pp_cfg, settings)` — encodes the three default policies once.
- `load_values(meta, pp_cfg, settings) -> dict[key, value]` — replaces the 60-line
  fallback ladder in `set_variant`.
- `to_env(values) -> dict[str,str]` — replaces `preprocess_env()`; bool→"1"/"0",
  float→`:g`, list→space/comma-joined per kind.
- `to_overrides(values)` — replaces `preprocess_overrides()`.
- `elide_for_meta(values, pp_cfg, settings) -> dict` — replaces the 150-line
  `pop`-or-set chain in `_save_variant_preprocess_meta`; the `_pp_default`
  trap becomes `default_from="preprocess_toml"` and cannot be forgotten.
- `PREPROCESS_ONLY_KEYS = {k.key for k in KNOBS if …}` — replaces
  `_GUI_PREPROCESS_KEYS` (ConfigTab's strip list) so the two can't drift.

### `_section.py` — one form builder

`KnobSection(QGroupBox)` takes the knobs for its section and builds the form:
label via `make_field_label` + `_preprocess_fields.json` tooltip keyed by
`knob.key`, widget via `gui.widgets.fields._widget`-style mapping on `kind`
(extend `_widget`/`_read` with the three domain widgets already in the file:
`_ResizeCropAnchorWidget`, margins, target-res), `enabled_by` wiring, and a
`values()` / `set_values()` / `dirty_signal` surface. The tab then does
`for section in self.sections: values.update(section.values())`. Sections
that need bespoke UI (crop preview, masking rule cards) subclass it and add
their extra widgets below the generated form.

### `tab.py` — what's left

Top bar (method/variant combos, Save, Run TE/PE/Mask split buttons, Stop),
progress + status row, the log, `_submit` / reattach / `_on_job_finished`
(DaemonJobMixin), and the four public methods ConfigTab calls
(`set_variant`, `preprocess_env`, `preprocess_overrides`,
`preprocess_config_snapshot`, `persist_preprocess_inputs`) — each now a
one-liner over `knobs.py`. Target ≲ 500 lines.

## Where the new feature lands

| Layer | Status | Work |
|---|---|---|
| `anime_tools.captions.tag_drop_groups` + `correction` | done | — |
| `anime_tools.stages.cli.correct_captions --caption_drop_groups` | done | — |
| `tasks.py preprocess`: env `CAPTION_DROP_GROUPS` / `caption_drop_groups` toml / CLI | done | — |
| `configs/preprocess.toml` commented key | done | — |
| GUI | **pending** | Phase 2 below: one `Knob` row + a `tag_groups` widget |

The `tag_groups` widget: a flow of toggle chips over `drop_group_names()`
(slug label, tooltip = KB path + example tags) plus a free-text line for
literal path prefixes (`효과/연출 > 조명`). Value = tuple of selectors;
serialized comma-joined into `CAPTION_DROP_GROUPS`. Placed in the
**caption-editing** box next to `insert_no_artist` (the two compose: drop
`artist` + insert `@no-artist` is the style-LoRA recipe from #95). It should
carry a one-line hint that the master caption is never edited and that
unknown tags are kept — the two questions the issue author will ask.

## Phases

Each phase is independently shippable and leaves `make gui` working; the
existing tests (`tests/test_gui_preprocess_tab.py`,
`test_gui_snapshot_preprocess_keys.py`, `test_gui_launch_speed.py`) are the
regression net and must stay green after every phase.

### Phase 0 — characterization (½ day) — DONE 2026-08-28

`tests/test_gui_preprocess_characterization.py` + `tests/fixtures/gui_preprocess_knobs.json` (two default-source scenarios × defaults/flipped × env/overrides/meta-with-and-without-mask, plus reload fixed points). Regenerate only via `--write`.

Pin current behavior before moving anything:

- Snapshot test: for every knob, `preprocess_env()`, `preprocess_overrides()`
  and the saved `[variant]` meta for (a) all-defaults, (b) every knob flipped.
  Store as a JSON fixture. This is the contract Phase 1 must reproduce byte-for-byte.
- Import-time guard already exists (`test_gui_launch_speed.py`) — keep it; the
  refactor must not add a torch-importing leaf (`tag_drop_groups.py` is
  stdlib-only, verified).

### Phase 1 — extract `knobs.py`, keep the UI as is (1 day) — DONE 2026-08-28

Landed as designed with two deviations: `elide_for_meta` became `merge_into_meta(meta, values, defaults, include_mask=)` (mutates the loaded `[variant]` table in place, matching the old pop-or-set semantics — a mask-less save leaves earlier mask keys untouched), and `default_from` grew a `sam_yaml` policy for the mask knobs. The fixture surfaced a real policy split the table now declares rather than hides: `drop_lowres_images` / `min_pixels` / `freefit_max_ratio` / crop knobs **load** from `preprocess.toml` but are **elided** against the hardcoded default (`persist="if_changed"`), so under a populated TOML they get written to the variant untouched, and flipping one back to the hardcoded default is popped and reloads as the TOML value — see `test_const_elision_under_populated_toml_is_the_recorded_quirk`. Collapsing that onto `if_changed_resolved` is a one-word change per row, deliberately not made here (Non-goals).

- Write the `KNOBS` table + the five pure functions; unit-test them against
  the Phase 0 fixture.
- Rewire `set_variant` / `preprocess_env` / `preprocess_overrides` /
  `_save_variant_preprocess_meta` to call them. Widgets and layout untouched;
  `self.<widget>` attributes stay (tests reach into them).
- Replace `_GUI_PREPROCESS_KEYS` with the derived set.
- Expected: −300 lines, zero visual change.

### Phase 2 — add `caption_drop_groups` through the new path (½ day)

- One `Knob` row; a `tag_groups` widget (`value()`/`set_value()`/`changed`,
  the domain-widget protocol `_section.read_widget`/`set_widget` already
  dispatch to) added via `CaptionEditingSection.add_knob`; i18n keys ×4 +
  `_preprocess_fields.json` ×4 (use the `translator` agent for ko/ja/cn).
- Round-trip test in `test_gui_preprocess_tab.py` (variant meta ⇄ widget ⇄ env),
  and assert `CAPTION_DROP_GROUPS` reaches `tasks.py`'s
  `_caption_correction_config` and turns the correction pass on by itself.
- This phase doubles as the proof that a new knob is now ~1 row + strings.
- Also surface the same control in the Dataset (image) tab's single-caption
  correction (`image_tab._caption_correction_options`) so the per-image
  "correct" button previews what the batch run will do — reads the same knob
  value from the Preprocessing tab instance it already receives.

### Phase 3 — section panels (1–2 days) — DONE 2026-08-28 (before Phase 2)

Landed as designed: `_section.py` (`KnobSection` — `add_knob(key, widget, label)` registers the editor, wires change→`changed`, the `enabled_by` gate, and the generic `values()`/`set_values()` dispatched on **widget type**, so two `float` knobs may use different editors), `image_prep.py` (+ `_ResizeCropAnchorWidget`, new `_CropMarginsWidget`), `text_caching.py`, `captions.py` (`AutotagSection` + `CaptionEditingSection`), `masking.py` (`_RuleCard`, `SamMaskSection` with `collect_rules()`/`set_rule_cards()`, `MitMaskSection`), `tab.py`. `gui/tabs/preprocess_tab.py` is the shim; `gui/tabs/preprocess/__init__.py` exports `PreprocessingTab` lazily (PEP 562) so `knobs` stays importable without PySide6.

Deviations / notes:

- Masking is two sections (SAM, MIT), not one `MaskingSection` — they were two group boxes and persist independently.
- The `tab.<widget>` aliases are a `_WIDGET_ALIASES` map resolved in `PreprocessingTab.__getattr__`; `_set_target_res_widget` / `_set_rule_cards` / `_autotag_mode` / `_resize_crop_margins` / `_rule_cards` stay as thin delegates. Tests still use them (migration to `values()` deferred); `image_tab` reads `target_res_widget` / `resize_crop_anchor_widget` / `freefit_max_ratio_spin` / `_resize_crop_margins` through the same shim.
- Tests that monkeypatch the default-source loaders now patch `gui.tabs.preprocess.tab` (the shim re-exports the names but patching it no longer reaches the tab).
- Phase 0's fixture reproduced byte-for-byte; no `--write`.

### Phase 4 — polish (optional, ½ day)

- Collapsible sections (`QToolButton` header, like ConfigTab's Advanced fold)
  with a per-section "enabled" summary in the header ("Captions: drop
  artist,lighting · autotag merge") so the tab reads as a pipeline.
- Order sections in **pipeline order** — resize → autotag → position clauses →
  correction/drop → TE → masking — matching `cmd_preprocess`'s chain
  (currently autotag sits *after* text caching in the UI, which is the reverse
  of when it runs).

## Non-goals

- No change to `tasks.py preprocess` contracts (env names, snapshot keys) —
  Phase 0's fixture enforces it.
- No merging of the three default-source policies; the table just makes them
  explicit. Collapsing `gui_settings.json`-backed knobs onto `preprocess.toml`
  is a separate decision (it changes what `make update` preserves).
- ConfigTab's own 1665 lines — out of scope; the `KnobSection` builder is
  designed so it *could* be reused there later, but don't couple the phases.

## Risks

- **Silent default drift** in Phase 1: the elision rule is subtle (three
  sources, `_pp_default` for four keys). The Phase 0 fixture is the only guard
  — don't skip it.
- **Tests reaching into widgets**: 10 tests touch `tab.<widget>` directly.
  Phase 3's `__getattr__` shim keeps them green; migrate them gradually
  (then delete `_WIDGET_ALIASES`).
- **i18n parity is manual** (gui/CLAUDE.md) — Phase 2 adds ~6 strings; run
  the `translator` agent, don't hand-copy English.
