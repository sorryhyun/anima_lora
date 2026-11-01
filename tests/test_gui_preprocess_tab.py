"""Regression tests for GUI preprocess-profile persistence."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import pytest


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


@pytest.mark.core
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
