"""Regression tests for GUI preprocess-profile persistence."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def _temporary_custom_variant(name: str) -> Iterator[tuple[str, object]]:
    from gui import variant_path

    variant = f"custom/{name}"
    path = variant_path(variant)
    old_text = path.read_text(encoding="utf-8") if path.exists() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[variant]\nfamily = "lora"\n', encoding="utf-8")
    try:
        yield variant, path
    finally:
        if old_text is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(old_text, encoding="utf-8")


def _make_tab():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from gui.tabs.preprocess_tab import PreprocessingTab

    app = QApplication.instance() or QApplication([])
    assert app is not None
    return PreprocessingTab()


def test_preprocess_tab_persists_default_target_res_to_variant():
    """The GUI profile should show the resolution tiers it will use.

    ``[1024]`` is the default tier, but it must still be written to the active
    gui-method variant when the Preprocess tab saves. Otherwise users see the
    resolution selection reset/vanish from the profile even though the widget
    accepted the value.
    """
    from gui import _load

    tab = None
    with _temporary_custom_variant("__pytest_preprocess_target_res__") as (
        variant,
        path,
    ):
        tab = _make_tab()
        tab.set_variant(variant, method="lora")
        tab._set_target_res_widget([1024])

        assert tab.persist_preprocess_inputs()

        meta = _load(path)["variant"]
        assert meta["target_res"] == [1024]

        if tab is not None:
            tab.deleteLater()


def test_preprocess_tab_source_dir_editable_and_persists():
    """The source image dir must be editable again and round-trip to the variant.

    Regression for the path-scoping change that locked the field read-only
    (setReadOnly(True)) with no save path, leaving the raw image root
    un-revisable from the GUI. It now edits the *base* root (preprocess-owned),
    persists onto the variant, and path_scope is appended on top at submit time.
    """
    from gui import _load

    tab = None
    with _temporary_custom_variant("__pytest_preprocess_source_dir__") as (
        variant,
        path,
    ):
        tab = _make_tab()
        tab.set_variant(variant, method="lora")

        # Editable + dirty-tracked (was read-only, never marked dirty before).
        assert not tab.source_dir_edit.isReadOnly()
        assert not tab._dirty
        tab.source_dir_edit.setText("/data/myset")
        assert tab._dirty

        assert tab._save_all()
        assert _load(path)["variant"]["source_image_dir"] == "/data/myset"

        # path_scope is layered on top of the edited base, not the hard default.
        path.write_text(
            '[variant]\nfamily = "lora"\n'
            'source_image_dir = "/data/myset"\n'
            'path_scope = "group1"\n',
            encoding="utf-8",
        )
        tab.set_variant(variant, method="lora")
        assert tab.source_dir_edit.text() == "/data/myset"
        snapshot = tab.preprocess_config_snapshot()
        assert snapshot["source_image_dir"] == "/data/myset/group1"

        if tab is not None:
            tab.deleteLater()


def test_preprocess_tab_persists_masking_settings_to_variant():
    from gui import _load

    tab = None
    with _temporary_custom_variant("__pytest_preprocess_mask_profile__") as (
        variant,
        path,
    ):
        tab = _make_tab()
        tab.set_variant(variant, method="lora")

        assert not tab._dirty
        tab.mask_path_pattern_edit.setText("character_a/*")
        assert tab._dirty
        assert tab.save_btn.text().endswith(" *")

        tab.run_sam_mask_chk.setChecked(False)
        tab.run_mit_mask_chk.setChecked(True)
        tab.mit_threshold_edit.setText("0.7")
        tab.mit_dilate_spin.setValue(9)
        card = tab._rule_cards[0]
        card.path_pattern_edit.setText("character_a/*")
        card.prompts_edit.setPlainText("speech bubble\nartist")
        card.focus_prompts_edit.setPlainText("girl")
        card.threshold_edit.setText("0.45")
        card.dilate_spin.setValue(7)

        assert tab._save_all()

        meta = _load(path)["variant"]
        assert meta["run_sam_mask"] is False
        assert meta["run_mit_mask"] is True
        assert meta["mask_path_pattern"] == "character_a/*"
        assert meta["mit_text_threshold"] == 0.7
        assert meta["mit_dilate"] == 9
        assert meta["mask_rules"] == [
            {
                "path_pattern": "character_a/*",
                "prompts": ["speech bubble", "artist"],
                "focus_prompts": ["girl"],
                "threshold": 0.45,
                "dilate": 7,
            }
        ]
        assert not tab._dirty
        assert not tab.save_btn.text().endswith(" *")

        assert tab.persist_preprocess_inputs()
        meta_after_preprocess_save = _load(path)["variant"]
        for key in (
            "run_sam_mask",
            "run_mit_mask",
            "mask_path_pattern",
            "mask_rules",
            "mit_text_threshold",
            "mit_dilate",
        ):
            assert meta_after_preprocess_save[key] == meta[key]

        if tab is not None:
            tab.deleteLater()


def test_preprocess_tab_freefit_max_ratio_round_trips_to_variant():
    """Free-fit is always on; a non-default max-ratio must persist and reload."""
    from gui import _load

    tab = None
    with _temporary_custom_variant("__pytest_preprocess_freefit__") as (
        variant,
        path,
    ):
        tab = _make_tab()
        tab.set_variant(variant, method="lora")
        tab.freefit_max_ratio_spin.setValue(3.0)

        assert tab.persist_preprocess_inputs()
        meta = _load(path)["variant"]
        # The freefit toggle is gone (free-fit is the only mode); only the clamp.
        assert "freefit" not in meta
        assert meta["freefit_max_ratio"] == 3.0

        # Reload into a fresh widget and confirm the value comes back.
        tab.freefit_max_ratio_spin.setValue(4.0)
        tab.set_variant(variant, method="lora")
        assert tab.freefit_max_ratio_spin.value() == 3.0

        if tab is not None:
            tab.deleteLater()


def test_preprocess_overrides_carry_freefit_max_ratio():
    """preprocess_overrides feeds the resize CLI; free-fit is always on."""
    tab = None
    with _temporary_custom_variant("__pytest_preprocess_freefit_ovr__") as (
        variant,
        _path,
    ):
        tab = _make_tab()
        tab.set_variant(variant, method="lora")
        tab.freefit_max_ratio_spin.setValue(3.5)

        overrides = tab.preprocess_overrides()
        assert "freefit" not in overrides
        assert overrides["freefit_max_ratio"] == 3.5

        if tab is not None:
            tab.deleteLater()


def test_preprocess_tab_caption_options_round_trip_to_variant():
    from gui import _load

    tab = None
    with _temporary_custom_variant("__pytest_preprocess_caption_options__") as (
        variant,
        path,
    ):
        tab = _make_tab()
        tab.set_variant(variant, method="lora")

        tab.caption_correct_order_chk.setChecked(True)
        tab.caption_insert_no_artist_chk.setChecked(True)
        tab.caption_trigger_word_edit.setText("@dataset-trigger")
        tab.caption_trigger_at_front_chk.setChecked(True)

        assert tab._save_all()
        meta = _load(path)["variant"]
        assert meta["caption_correct_order"] is True
        assert meta["caption_insert_no_artist"] is True
        assert meta["caption_trigger_word"] == "@dataset-trigger"
        assert meta["caption_trigger_at_front"] is True

        env = tab.preprocess_env()
        assert env["CAPTION_CORRECT_ORDER"] == "1"
        assert env["CAPTION_INSERT_NO_ARTIST"] == "1"
        assert env["CAPTION_TRIGGER_WORD"] == "@dataset-trigger"
        assert env["CAPTION_TRIGGER_AT_FRONT"] == "1"

        overrides = tab.preprocess_overrides()
        assert overrides["caption_correct_order"] is True
        assert overrides["caption_insert_no_artist"] is True
        assert overrides["caption_trigger_word"] == "@dataset-trigger"
        assert overrides["caption_trigger_at_front"] is True

        tab.caption_correct_order_chk.setChecked(False)
        tab.caption_insert_no_artist_chk.setChecked(False)
        tab.caption_trigger_word_edit.clear()
        tab.caption_trigger_at_front_chk.setChecked(False)
        tab.set_variant(variant, method="lora")
        assert tab.caption_correct_order_chk.isChecked()
        assert tab.caption_insert_no_artist_chk.isChecked()
        assert tab.caption_trigger_word_edit.text() == "@dataset-trigger"
        assert tab.caption_trigger_at_front_chk.isChecked()

        if tab is not None:
            tab.deleteLater()


def test_masking_task_reads_gui_sam_config_snapshot(monkeypatch):
    from scripts.tasks import masking

    monkeypatch.setenv(
        "SAM_MASK_CONFIG_JSON",
        '{"path_pattern":"character_a/*","rules":[{"prompts":["bubble"]}]}',
    )

    cfg = masking._load_sam_config()

    assert cfg["rules"] == [{"prompts": ["bubble"]}]
    assert masking._config_path_pattern(cfg) == "character_a/*"
