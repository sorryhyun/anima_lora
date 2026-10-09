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
        assert not tab.widget("source_image_dir").isReadOnly()
        assert not tab._dirty
        tab.widget("source_image_dir").setText("/data/myset")
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
        assert tab.widget("source_image_dir").text() == "/data/myset"
        snapshot = tab.preprocess_config_snapshot()
        assert snapshot["source_image_dir"] == "/data/myset/group1"

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
        tab.widget("freefit_max_ratio").setValue(3.0)

        assert tab.persist_preprocess_inputs()
        meta = _load(path)["variant"]
        # The freefit toggle is gone (free-fit is the only mode); only the clamp,
        # now a resize stage field.
        assert "freefit" not in meta and "freefit_max_ratio" not in meta
        assert meta["stages"]["resize"]["freefit_max_ratio"] == 3.0

        # Reload into a fresh widget and confirm the value comes back.
        tab.widget("freefit_max_ratio").setValue(4.0)
        tab.set_variant(variant, method="lora")
        assert tab.widget("freefit_max_ratio").value() == 3.0

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
        tab.widget("freefit_max_ratio").setValue(3.5)

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

        tab.widget("no_correct").setChecked(False)
        tab.widget("caption_insert_no_artist").setChecked(True)
        tab.widget("caption_trigger_word").setText("@dataset-trigger")
        tab.widget("caption_trigger_at_front").setChecked(True)
        tab.widget("caption_drop_groups").setText("artist")

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

        tab.widget("no_correct").setChecked(True)
        tab.widget("caption_insert_no_artist").setChecked(False)
        tab.widget("caption_trigger_word").clear()
        tab.widget("caption_trigger_at_front").setChecked(False)
        tab.set_variant(variant, method="lora")
        assert not tab.widget("no_correct").isChecked()
        assert tab.widget("caption_insert_no_artist").isChecked()
        assert tab.widget("caption_trigger_word").text() == "@dataset-trigger"
        assert tab.widget("caption_trigger_at_front").isChecked()
        assert tab.widget("caption_drop_groups").text() == "artist"

        if tab is not None:
            tab.deleteLater()


def test_preprocess_tab_is_cache_only(monkeypatch):
    """A ``preprocess.toml`` turning the curation stages on never reaches a
    GUI run (env beats the config in ``tasks.py``), the job carries only the
    cache-side stage forms, and an older GUI's curation tables survive a Save
    untouched."""
    import json

    import toml

    from gui import _load
    from gui.tabs.preprocess import tab as preprocess_tab

    monkeypatch.setattr(
        preprocess_tab,
        "_load_preprocess_toml",
        lambda: {"caption_position_clauses": True, "caption_autotag": True},
    )
    tab = None
    with _temporary_custom_variant("__pytest_preprocess_cache_only__") as (
        variant,
        path,
    ):
        data = _load(path)
        data["variant"]["caption_autotag"] = True
        data["variant"]["stages"] = {
            "masks_sam": [{"masks": ["ignore:text:bubble"]}],
            "autotag": {"mode": "merge"},
        }
        path.write_text(toml.dumps(data), encoding="utf-8")

        tab = _make_tab()
        tab.set_variant(variant, method="lora")
        env = tab.preprocess_env()
        assert env["CAPTION_AUTOTAG"] == "0"
        assert env["CAPTION_POSITION_CLAUSES"] == "0"
        assert set(json.loads(env["PREPROCESS_STAGES_JSON"])) == {"resize", "correct"}

        assert tab._save_all()
        meta = _load(path)["variant"]
        assert "caption_autotag" not in meta
        assert meta["stages"]["masks_sam"] == [{"masks": ["ignore:text:bubble"]}]
        assert meta["stages"]["autotag"] == {"mode": "merge"}

        if tab is not None:
            tab.deleteLater()


def test_resize_preview_lists_each_source_with_its_bucket(tmp_path, monkeypatch):
    """The Preprocess tab's resize preview reads the tab's live tiers and
    source dir, and leaves an excluded image out like the resize does."""
    from PySide6.QtGui import QImage

    from gui.i18n import t
    from gui.tabs.preprocess import resize_preview as RP
    from library.preprocess.resize_preview import compute_resize_preview

    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    QImage(1600, 900, QImage.Format_RGB32).save(str(src / "a" / "wide.png"))
    QImage(800, 1200, QImage.Format_RGB32).save(str(src / "tall.png"))
    QImage(640, 640, QImage.Format_RGB32).save(str(src / "gone.png"))
    monkeypatch.setattr(RP, "excluded_rels", lambda: ("gone.png",))
    monkeypatch.setattr(RP, "workspace_excluded_rels", lambda: ())

    tab = _make_tab()
    tab.widget("source_image_dir").setText(str(src))
    tab._set_target_res_widget([768, 1024])
    dlg = RP.ResizePreviewDialog(tab)
    dlg.refresh()

    rows = {
        dlg.tree.topLevelItem(i).text(0): dlg.tree.topLevelItem(i)
        for i in range(dlg.tree.topLevelItemCount())
    }
    assert set(rows) == {"a/wide.png", "tall.png"}
    pv = compute_resize_preview(1600, 900, [768, 1024])
    assert rows["a/wide.png"].text(2) == "{}×{}".format(*pv.bucket_size)
    assert rows["a/wide.png"].text(3) == str(pv.target_edge)
    assert t("preprocess_resize_preview_skipped", n=1) in dlg.summary.text()
