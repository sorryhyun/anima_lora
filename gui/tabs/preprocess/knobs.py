"""The Preprocessing tab's trainer-native knob table — single source of truth,
Qt-free.

This table holds only what is **not** a field of an ``anime_tools`` stage
request: the dataset roots and scope (``source_image_dir``
/ ``path_scope`` / ``preprocess_path_pattern``) and the trainer-side TE-cache
knobs (``caption_shuffle_variants`` / ``caption_tag_dropout_rate``). The
resize geometry and the caption mirror's rewrite knobs are drawn from the stage
schemas by ``stage_form.py`` and persisted under ``[variant.stages.<stage_id>]``.

The tab is cache-only: the curation stages (autotag, position clauses, SAM
masks) run from the ``anime_tools`` panel, and the GUI pins their chain gates
off (:data:`CURATION_GATES_OFF`) so a ``preprocess.toml`` that turns one on
never starts a tagger / SAM3 pass from a cache build.

The pure functions below implement the tab's default resolution, env /
override serialisation and ``[variant]`` elision from the table, so adding a
knob is one row + the widget, and each knob's default source (hardcoded /
``preprocess.toml`` / ``gui_settings.json``) is declared on its row.

The contract is pinned byte-for-byte by
``tests/test_gui_preprocess_characterization.py`` (fixture under
``tests/fixtures/``). The quirks it records — e.g. a knob that *loads* from
``preprocess.toml`` but is *elided* against the hardcoded default — are
reproduced on purpose; collapsing the policies is a separate decision.

Like ``gui/core/config_io.py`` this module must stay importable without PySide6
and without torch (``tests/test_gui_launch_speed.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Defaults match the bare CLI (scripts/tasks/preprocess.py), so a fresh GUI runs
# the same pipeline.
DEFAULT_SOURCE_IMAGE_DIR = "image_dataset"
DEFAULT_PREPROCESS_PATH_PATTERN = "*"
DEFAULT_TE_SHUFFLE_VARIANTS = 4
DEFAULT_TE_TAG_DROPOUT = 0.1

# Chain gates of the curation stages, exported off by every GUI run: the tab
# builds caches, the ``anime_tools`` panel curates. Env beats the config
# chain in ``scripts/tasks/preprocess.py``.
CURATION_GATES_OFF: dict[str, str] = {
    "CAPTION_AUTOTAG": "0",
    "CAPTION_POSITION_CLAUSES": "0",
}

# The key under ``[variant]`` holding the per-stage form values
# (``[variant.stages.<stage_id>]`` — see ``stage_form.py``). Not a knob row:
# listed in ``PREPROCESS_ONLY_KEYS`` so ConfigTab strips it from the training
# snapshot like every other preprocess-owned key.
STAGES_KEY = "stages"

Kind = Literal["bool", "int", "float", "str"]
# Knobs that left the table: a variant saved by an older GUI may still carry
# them; they load as nothing and are dropped on the next save.
RETIRED_KEYS = (
    "drop_lowres_images",  # resize lost its pixel floor (anime_tools 0.7.5)
    # The curation chain gates, gone with the tab's curation sections.
    "caption_position_clauses",
    "caption_autotag",
    "run_sam_mask",
)

DefaultFrom = Literal["const", "preprocess_toml", "gui_settings"]
# How the knob reaches the variant's ``[variant]`` meta on save:
#   if_changed        — written only when it differs from the *hardcoded* default
#   if_changed_resolved — written only when it differs from the *resolved*
#                       default (preprocess.toml-backed knobs; a checkbox left at
#                       the hardcoded default would otherwise not stick, because
#                       the tab always exports the env var and env beats the TOML)
#   if_truthy         — written when non-empty (path_scope)
Persist = Literal["if_changed", "if_changed_resolved", "if_truthy"]


@dataclass(frozen=True)
class Knob:
    key: str
    section: str  # "image" | "text"
    kind: Kind
    default: object
    default_from: DefaultFrom = "const"
    env: str | None = None
    persist: Persist = "if_changed"
    snapshot: bool = False  # goes into preprocess_overrides()
    enabled_by: str | None = None
    # Empty text means "the default": "const" → the hardcoded one, "resolved"
    # → the scenario's resolved one. Applied before env export and elision.
    empty_fallback: Literal["const", "resolved"] | None = None


# Row order == on-disk key order in a freshly written [variant] table.
KNOBS: tuple[Knob, ...] = (
    Knob(
        "source_image_dir",
        "image",
        "str",
        DEFAULT_SOURCE_IMAGE_DIR,
        default_from="preprocess_toml",
        persist="if_changed_resolved",
        empty_fallback="resolved",
    ),
    # Layered on top at submit time by gui.core.submit.scoped_paths; the field
    # edits the *unscoped* root. Normalised (and validated) by the tab before save.
    Knob("path_scope", "image", "str", "", persist="if_truthy"),
    Knob(
        "preprocess_path_pattern",
        "image",
        "str",
        DEFAULT_PREPROCESS_PATH_PATTERN,
        env="PREPROCESS_PATH_PATTERN",
        empty_fallback="const",
    ),
    Knob(
        "caption_shuffle_variants",
        "text",
        "int",
        DEFAULT_TE_SHUFFLE_VARIANTS,
        default_from="gui_settings",
        env="CAPTION_SHUFFLE_VARIANTS",
    ),
    Knob(
        "caption_tag_dropout_rate",
        "text",
        "float",
        DEFAULT_TE_TAG_DROPOUT,
        default_from="gui_settings",
        env="CAPTION_TAG_DROPOUT_RATE",
    ),
)

KNOBS_BY_KEY: dict[str, Knob] = {k.key: k for k in KNOBS}

# Every key the Preprocessing tab owns. ConfigTab strips these from the
# training snapshot (they must not ride into train.py — e.g.
# ``caption_tag_dropout_rate`` collides with a real *live* dataloader arg).
# Retired keys stay listed: an older variant may still carry them.
PREPROCESS_ONLY_KEYS: frozenset[str] = (
    frozenset(k.key for k in KNOBS) | {STAGES_KEY} | frozenset(RETIRED_KEYS)
)
ENV_KNOBS: tuple[Knob, ...] = tuple(k for k in KNOBS if k.env)
SNAPSHOT_KNOBS: tuple[Knob, ...] = tuple(k for k in KNOBS if k.snapshot)


def resolve_default(knob: Knob, pp_cfg: dict, settings: dict):
    """The effective default for one knob under its ``default_from`` policy.

    A ``preprocess.toml``-backed knob resolves *through* the user-owned TOML:
    load-bearing for the caption-master stages, whose env var the tab always
    exports (env beats the TOML in ``tasks.py``), so the widget must start
    from the TOML's answer or a `true` set there would be silently overridden."""
    if knob.default_from == "const":
        return knob.default
    if knob.default_from == "preprocess_toml":
        value = pp_cfg.get(knob.key)
    elif knob.default_from == "gui_settings":
        value = settings.get(knob.key)
    else:  # pragma: no cover — unreachable by construction
        raise ValueError(f"{knob.key}: unsupported default_from")
    return knob.default if value is None else value


def resolved_defaults(pp_cfg: dict, settings: dict) -> dict:
    """``{key: effective default}`` for every knob — the one dict the other
    helpers take, so the tab reads its default sources exactly once."""
    return {k.key: resolve_default(k, pp_cfg, settings) for k in KNOBS}


def load_values(meta: dict, defaults: dict) -> dict:
    """Widget values for a variant: its ``[variant]`` meta over the resolved
    defaults (what ``set_variant`` shows)."""
    return {knob.key: meta.get(knob.key, defaults[knob.key]) for knob in KNOBS}


def _coerce(knob: Knob, value):
    """Normalise a raw widget value to the type the TOML/snapshot carries.
    Strings for float knobs are parsed (the tab hands the dropout rate over as
    the line-edit text so the env export stays byte-exact)."""
    kind = knob.kind
    if kind == "bool":
        return bool(value)
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "str":
        return str(value)
    raise ValueError(f"{knob.key}: unknown kind {kind}")  # pragma: no cover


def _with_empty_fallback(knob: Knob, value, defaults: dict):
    if knob.empty_fallback and (value is None or value == ""):
        return knob.default if knob.empty_fallback == "const" else defaults[knob.key]
    return value


def to_env(values: dict, defaults: dict) -> dict[str, str]:
    """Environment consumed by ``tasks.py preprocess``.

    The trainer-native knobs ride as env, not just the config snapshot,
    because the Train auto-chain hands preprocess a snapshot with the
    preprocess-only keys stripped; env wins over the snapshot in ``tasks.py``.
    A ``str`` for a float/int knob is exported verbatim (already user text).
    The stage forms travel separately (``PREPROCESS_STAGES_JSON``, see
    ``stage_form.STAGE_VALUES_ENV``)."""
    env: dict[str, str] = dict(CURATION_GATES_OFF)
    for knob in ENV_KNOBS:
        value = _with_empty_fallback(knob, values[knob.key], defaults)
        if knob.kind == "bool":
            text = "1" if value else "0"
        elif isinstance(value, str):
            text = value
        elif knob.kind == "int":
            text = str(int(value))
        elif knob.kind == "float":
            text = f"{float(value):g}"
        else:
            text = str(value)
        env[knob.env] = text  # type: ignore[index]
    return env


def to_overrides(values: dict) -> dict[str, object]:
    """Flat config overrides captured in preprocess snapshots
    (``preprocess_overrides``)."""
    return {knob.key: _coerce(knob, values[knob.key]) for knob in SNAPSHOT_KNOBS}


def merge_into_meta(meta: dict, values: dict, defaults: dict) -> dict:
    """Apply the elision rules to a variant's ``[variant]`` table in place:
    each knob is written or popped per its ``persist`` policy, so a plain
    checkout keeps an empty meta. Returns ``meta``. The ``stages`` sub-table is ``stage_form``'s
    (``merge_stages_into_meta``)."""
    for key in RETIRED_KEYS:
        meta.pop(key, None)
    for knob in KNOBS:
        value = _coerce(knob, _with_empty_fallback(knob, values[knob.key], defaults))
        if knob.persist == "if_truthy":
            keep = bool(value)
        elif knob.persist == "if_changed":
            keep = value != _coerce(knob, knob.default)
        elif knob.persist == "if_changed_resolved":
            keep = value != _coerce(knob, defaults[knob.key])
        else:  # pragma: no cover
            raise ValueError(f"{knob.key}: unknown persist {knob.persist}")
        if keep:
            meta[knob.key] = value
        else:
            meta.pop(knob.key, None)
    return meta
