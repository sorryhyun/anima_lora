"""Unit tests for the Qt-free knob table (``gui/tabs/preprocess/knobs.py``).

The table holds only the trainer-native rows — dataset roots / scope and the
TE-cache variant knobs (the curation chain gates left with the tab's curation
sections; the env pins them off); the stage forms are ``stage_form``'s and
tested in ``test_gui_stage_form.py``.
The full contract is pinned by ``test_gui_preprocess_characterization.py``;
this file holds what that fixture cannot show: retired keys in an older
variant, and the table's closure against the stage forms.
"""

from __future__ import annotations

from gui.tabs.preprocess import knobs as K
from tests.test_gui_preprocess_characterization import SCENARIOS

# The widget values `_flip_every_knob` in the characterization test produces
# (trainer knobs only; the stage forms are flipped there too but live under
# ``meta["stages"]`` and the ``PREPROCESS_STAGES_JSON`` env, not here).
FLIPPED_VALUES = {
    "source_image_dir": "flipped_images",
    "path_scope": "artist_a",
    "preprocess_path_pattern": "artist_a/**",
    "caption_shuffle_variants": 11,
    "caption_tag_dropout_rate": "0.45",
}


def _defaults(scenario: str) -> dict:
    s = SCENARIOS[scenario]
    return K.resolved_defaults(s["preprocess_toml"], s["gui_settings"])


def _default_widget_values(scenario: str) -> dict:
    """What the widgets hold right after ``set_variant`` on an empty variant."""
    values = K.load_values({}, _defaults(scenario))
    # Free-text numerics are shown with :g and read back as text.
    values["caption_tag_dropout_rate"] = (
        f"{float(values['caption_tag_dropout_rate']):g}"
    )
    return values


def _flipped_widget_values(scenario: str) -> dict:
    values = _default_widget_values(scenario)
    values.update(FLIPPED_VALUES)
    return values


def _persistable(values: dict) -> dict:
    """The tab validates the free-text numerics before persisting."""
    values = dict(values)
    values["caption_tag_dropout_rate"] = float(values["caption_tag_dropout_rate"])
    return values


def test_retired_knob_loads_as_nothing_and_is_dropped_on_save():
    """A variant saved by an older GUI still carries ``drop_lowres_images`` or
    a curation chain gate: none is a knob any more, so each loads as nothing,
    the next save pops it, and ConfigTab still strips it from training."""
    defaults = _defaults("populated")
    stale = {key: True for key in K.RETIRED_KEYS}
    loaded = K.load_values(stale, defaults)
    values = _persistable(_flipped_widget_values("populated"))
    meta = K.merge_into_meta(dict(stale), values, defaults)
    for key in K.RETIRED_KEYS:
        assert key not in loaded and key not in meta, key
        assert key in K.PREPROCESS_ONLY_KEYS, key


def test_table_invariants():
    keys = [k.key for k in K.KNOBS]
    assert len(keys) == len(set(keys))
    assert K.PREPROCESS_ONLY_KEYS == set(keys) | {K.STAGES_KEY} | set(K.RETIRED_KEYS)
    for knob in K.KNOBS:
        assert knob.enabled_by is None or knob.enabled_by in K.KNOBS_BY_KEY, knob.key
    env_names = [k.env for k in K.ENV_KNOBS]
    assert len(env_names) == len(set(env_names))
    # Nothing a stage form shows is a knob row any more (the trainer-owned
    # dests it hides — the variant sidecar knobs — are exactly the overlap).
    from gui.tabs.preprocess.stage_form import (
        STAGE_IDS,
        load_stage_schemas,
        visible_fields,
    )

    schemas = load_stage_schemas()
    shown = {f["dest"] for sid in STAGE_IDS for f in visible_fields(schemas[sid])}
    assert not shown & set(keys)
