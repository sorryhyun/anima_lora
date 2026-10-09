"""Regression test: the GUI training snapshot must not carry preprocess-only keys.

Bug: ``_load_base()`` folds ``configs/preprocess.toml`` into the Config-tab
baseline so preprocess-owned fields surface in the form. That baseline also fed
``_queue_config_snapshot``, so the submitted training config inherited
preprocess knobs. Most were harmless ("unknown key" warnings), but
``caption_tag_dropout_rate`` collides with a real ``train.py`` argparse arg whose
meaning is *live dataloader tag dropout* — the dataset blueprint generator
pushed it onto every subset and tripped ``assert_extra_args``:

    AssertionError: when caching Text Encoder output, token_warmup_step or
    caption_tag_dropout_rate cannot be used

The snapshot builder now strips the whole preprocess-only key set.
"""

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


def _make_config_tab():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from gui.tabs.config_tab import ConfigTab

    app = QApplication.instance() or QApplication([])
    assert app is not None
    return ConfigTab()


def _make_config_and_preprocess_tabs():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from gui.tabs.config_tab import ConfigTab
    from gui.tabs.preprocess import PreprocessingTab

    app = QApplication.instance() or QApplication([])
    assert app is not None
    preprocess_tab = PreprocessingTab()
    return ConfigTab(preprocess_tab=preprocess_tab), preprocess_tab


def test_snapshot_drops_preprocess_only_keys():
    from gui.tabs.preprocess.knobs import PREPROCESS_ONLY_KEYS as _GUI_PREPROCESS_KEYS

    tab = _make_config_tab()
    try:
        snapshot = tab._queue_config_snapshot("lora")
    finally:
        tab.deleteLater()

    # The specific key that crashed training must be gone.
    assert "caption_tag_dropout_rate" not in snapshot
    assert "caption_tag_randomize_rate" not in snapshot

    # And no preprocess-owned knob should ride into the training config at all.
    leaked = _GUI_PREPROCESS_KEYS & snapshot.keys()
    assert not leaked, f"preprocess-only keys leaked into training snapshot: {leaked}"

    # Sanity: a genuine training knob still survives (caption_dropout_rate is a
    # legit train arg — it's TE-cache-compatible via cache_supports_dropout).
    assert "caption_dropout_rate" in snapshot


def test_preprocess_queue_snapshot_keeps_scoped_source_separate_from_train_snapshot():
    from gui.tabs.preprocess.knobs import PREPROCESS_ONLY_KEYS as _GUI_PREPROCESS_KEYS

    config_tab = None
    preprocess_tab = None
    with _temporary_custom_variant("__pytest_queue_scope__") as (variant, _path):
        config_tab, preprocess_tab = _make_config_and_preprocess_tabs()
        try:
            preprocess_tab.set_variant(variant, method="lora")
            preprocess_tab.widget("source_image_dir").setText("image_dataset")
            preprocess_tab.widget("path_scope").setText("scope1")

            preprocess_snapshot = config_tab._preprocess_config_snapshot(variant)
            train_snapshot = config_tab._queue_config_snapshot(variant)
            chain_spec = config_tab._chain_train_spec(
                variant, config_snapshot=train_snapshot
            )
        finally:
            if config_tab is not None:
                config_tab.deleteLater()
            if preprocess_tab is not None:
                preprocess_tab.deleteLater()

    assert preprocess_snapshot["source_image_dir"] == "image_dataset/scope1"
    assert (
        preprocess_snapshot["resized_image_dir"] == "post_image_dataset/resized/scope1"
    )
    assert preprocess_snapshot["lora_cache_dir"] == "post_image_dataset/lora/scope1"

    assert "source_image_dir" not in train_snapshot
    leaked = _GUI_PREPROCESS_KEYS & train_snapshot.keys()
    assert not leaked, f"preprocess-only keys leaked into training snapshot: {leaked}"
    assert chain_spec["config_snapshot"] == train_snapshot
