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

    from gui.tabs.preprocess import PreprocessingTab

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
        assert meta["stages"]["resize"]["target_res"] == [1024]

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
    """SAM rule cards are ``masks_sam`` stage forms, persisted as
    ``[[variant.stages.masks_sam]]`` (elided against the package defaults);
    the run switch stays a flat ``[variant]`` key."""
    from gui import _load

    tab = None
    with _temporary_custom_variant("__pytest_preprocess_mask_profile__") as (
        variant,
        path,
    ):
        tab = _make_tab()
        tab.set_variant(variant, method="lora")

        assert not tab._dirty
        tab.run_sam_mask_chk.setChecked(False)
        assert tab._dirty
        assert tab.save_btn.text().endswith(" *")

        card = tab._rule_cards[0]
        card.widgets["path_pattern"].setText("character_a/*")
        card.widgets["masks"].setText("keep:text:girl, ignore:text:speech bubble")
        card.widgets["threshold"].setValue(0.45)
        card.widgets["dilate"].setValue(7)

        assert tab._save_all()

        meta = _load(path)["variant"]
        assert meta["run_sam_mask"] is False
        assert meta["stages"]["masks_sam"] == [
            {
                "path_pattern": "character_a/*",
                "masks": ["keep:text:girl", "ignore:text:speech bubble"],
                "threshold": 0.45,
                "dilate": 7,
            }
        ]
        assert not tab._dirty
        assert not tab.save_btn.text().endswith(" *")

        # The job receives the card as a form the package's build_argv reads.
        import json

        forms = json.loads(tab.preprocess_env()["PREPROCESS_STAGES_JSON"])
        assert forms["masks_sam"][0]["masks"] == [
            "keep:text:girl",
            "ignore:text:speech bubble",
        ]
        assert forms["masks_sam"][0]["path_pattern"] == "character_a/*"

        assert tab.persist_preprocess_inputs()
        meta_after_preprocess_save = _load(path)["variant"]
        assert meta_after_preprocess_save["run_sam_mask"] is False
        assert (
            meta_after_preprocess_save["stages"]["masks_sam"]
            == meta["stages"]["masks_sam"]
        )

        # Reload: the card comes back as saved.
        tab.set_variant(variant, method="lora")
        assert (
            tab._rule_cards[0].widgets["masks"].text()
            == "keep:text:girl, ignore:text:speech bubble"
        )
        assert tab._rule_cards[0].widgets["dilate"].value() == 7

        if tab is not None:
            tab.deleteLater()


def test_preprocess_tab_refuses_a_sam_card_with_a_malformed_mask(monkeypatch):
    """The request's own validation fires in the Save dialog, not after a
    SAM3 load."""
    from PySide6.QtWidgets import QMessageBox

    warned = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *a, **k: warned.append(a[2]) or QMessageBox.Ok
    )
    tab = None
    with _temporary_custom_variant("__pytest_preprocess_mask_invalid__") as (
        variant,
        _path,
    ):
        tab = _make_tab()
        tab.set_variant(variant, method="lora")
        card = tab._rule_cards[0]
        card.widgets["masks"].setText("girl")
        assert not tab._save_all()
        assert warned and "ROLE:KIND:VALUE" in warned[-1]
        # A cache-build save (mask section excluded) is not blocked by it.
        assert tab.persist_preprocess_inputs()
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
        # The freefit toggle is gone (free-fit is the only mode); only the clamp,
        # now a resize stage field.
        assert "freefit" not in meta and "freefit_max_ratio" not in meta
        assert meta["stages"]["resize"]["freefit_max_ratio"] == 3.0

        # Reload into a fresh widget and confirm the value comes back.
        tab.freefit_max_ratio_spin.setValue(4.0)
        tab.set_variant(variant, method="lora")
        assert tab.freefit_max_ratio_spin.value() == 3.0

        if tab is not None:
            tab.deleteLater()


def test_preprocess_stage_values_carry_freefit_max_ratio():
    """The resize form feeds the resize request; free-fit is always on."""
    tab = None
    with _temporary_custom_variant("__pytest_preprocess_freefit_ovr__") as (
        variant,
        _path,
    ):
        tab = _make_tab()
        tab.set_variant(variant, method="lora")
        tab.freefit_max_ratio_spin.setValue(3.5)

        resize = tab.stage_values()["resize"]
        assert "freefit" not in resize
        assert resize["freefit_max_ratio"] == 3.5
        assert "freefit_max_ratio" not in tab.preprocess_overrides()

        if tab is not None:
            tab.deleteLater()


def test_preprocess_tab_caption_options_round_trip_to_variant():
    """The caption-editing box is the ``correct`` stage form: its values
    persist under ``[variant.stages.correct]`` and reach the job as a form."""
    import json

    from gui import _load

    tab = None
    with _temporary_custom_variant("__pytest_preprocess_caption_options__") as (
        variant,
        path,
    ):
        tab = _make_tab()
        tab.set_variant(variant, method="lora")

        tab.caption_no_correct_chk.setChecked(False)
        tab.caption_insert_no_artist_chk.setChecked(True)
        tab.caption_trigger_word_edit.setText("@dataset-trigger")
        tab.caption_trigger_at_front_chk.setChecked(True)
        tab.caption_drop_groups_edit.setText("artist")

        assert tab._save_all()
        meta = _load(path)["variant"]
        assert meta["stages"]["correct"] == {
            "caption_insert_no_artist": True,
            "caption_trigger_word": "@dataset-trigger",
            "caption_drop_groups": "artist",
            "no_correct": False,
            "caption_trigger_at_front": True,
        }
        for key in ("caption_correct_order", "caption_trigger_word"):
            assert key not in meta  # no flat key, no env

        env = tab.preprocess_env()
        assert "CAPTION_TRIGGER_WORD" not in env
        form = json.loads(env["PREPROCESS_STAGES_JSON"])["correct"]
        assert form["caption_trigger_word"] == "@dataset-trigger"
        assert form["no_correct"] is False and form["caption_drop_groups"] == "artist"

        tab.caption_no_correct_chk.setChecked(True)
        tab.caption_insert_no_artist_chk.setChecked(False)
        tab.caption_trigger_word_edit.clear()
        tab.caption_trigger_at_front_chk.setChecked(False)
        tab.set_variant(variant, method="lora")
        assert not tab.caption_no_correct_chk.isChecked()
        assert tab.caption_insert_no_artist_chk.isChecked()
        assert tab.caption_trigger_word_edit.text() == "@dataset-trigger"
        assert tab.caption_trigger_at_front_chk.isChecked()
        assert tab.caption_drop_groups_edit.text() == "artist"

        if tab is not None:
            tab.deleteLater()


def test_caption_master_stages_default_from_preprocess_toml(monkeypatch):
    """`caption_position_clauses = true` in the TOML must reach the checkbox.

    The tab always exports CAPTION_POSITION_CLAUSES / CAPTION_AUTOTAG, and env
    beats the config file in tasks.py — so a checkbox pinned to the hardcoded
    default would export "0" and silently cancel a CLI-side opt-in.
    """
    from gui import _load
    from gui.tabs.preprocess import tab as preprocess_tab

    monkeypatch.setattr(
        preprocess_tab,
        "_load_preprocess_toml",
        lambda: {
            "caption_position_clauses": True,
            "caption_autotag": True,
            "caption_autotag_mode": "merge",
            "caption_autotag_min_confidence": 0.35,
        },
    )

    tab = None
    with _temporary_custom_variant("__pytest_preprocess_master_stages__") as (
        variant,
        path,
    ):
        tab = _make_tab()
        assert tab.caption_position_clauses_chk.isChecked()
        assert tab.caption_autotag_chk.isChecked()
        assert tab._autotag_mode() == "merge"

        # A variant carrying no override keeps the config's answer, and the env
        # the daemon job inherits agrees with it.
        tab.set_variant(variant, method="lora")
        assert tab.caption_position_clauses_chk.isChecked()
        assert tab.caption_autotag_chk.isChecked()
        env = tab.preprocess_env()
        assert env["CAPTION_POSITION_CLAUSES"] == "1"
        assert env["CAPTION_AUTOTAG"] == "1"
        import json

        autotag = json.loads(env["PREPROCESS_STAGES_JSON"])["autotag"]
        assert autotag == {"mode": "merge", "min_confidence": 0.35}

        # Unchecking must PERSIST as false — popping it (the old rule, which
        # compared against the hardcoded default) would let the config's `true`
        # come back on the next load.
        tab.caption_position_clauses_chk.setChecked(False)
        tab.caption_autotag_chk.setChecked(False)
        assert tab._save_all()
        meta = _load(path)["variant"]
        assert meta["caption_position_clauses"] is False
        assert meta["caption_autotag"] is False

        tab.set_variant(variant, method="lora")
        assert not tab.caption_position_clauses_chk.isChecked()
        assert not tab.caption_autotag_chk.isChecked()
        assert tab.preprocess_env()["CAPTION_POSITION_CLAUSES"] == "0"

        if tab is not None:
            tab.deleteLater()


def test_masking_task_reads_the_gui_rule_cards(monkeypatch, tmp_path):
    """What the tab sends is what ``make mask`` builds: one request per card."""
    from scripts.tasks import masking

    tab = None
    with _temporary_custom_variant("__pytest_preprocess_mask_cards__") as (
        variant,
        _path,
    ):
        tab = _make_tab()
        tab.set_variant(variant, method="lora")
        tab._rule_cards[0].widgets["path_pattern"].setText("character_a/*")
        tab._rule_cards[0].widgets["masks"].setText("ignore:text:bubble")
        tab.sam_section.add_rule_card({"masks": "keep:text:girl", "threshold": 0.4})
        env = tab.preprocess_env()
        if tab is not None:
            tab.deleteLater()

    monkeypatch.setenv("PREPROCESS_STAGES_JSON", env["PREPROCESS_STAGES_JSON"])
    a, b = masking._sam_requests(tmp_path / "resized", tmp_path)
    assert [m.spec() for m in a.masks] == ["ignore:text:bubble"]
    assert a.path_pattern == "character_a/*"
    assert [m.spec() for m in b.masks] == ["keep:text:girl"] and b.threshold == 0.4
    assert a.mask_dir != b.mask_dir
