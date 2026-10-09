"""Regression tests for the GUI use_valid / validation_split_num coupling.

Bug: saving ``validation_split_num = 0`` from the Config tab silently became
16. ``use_valid`` (checkbox) and ``validation_split_num`` (spinbox) were
independent widgets, and the save path coerced an enabled-but-zero count to
``_DEFAULT_VALIDATION_SPLIT_NUM`` (16). The two widgets are now coupled: the
spinbox is the source of truth, an explicit 0 reads back as "off", and ticking
the box surfaces a positive default up front instead of at save time.
"""

from __future__ import annotations

import os

from gui.core.validation import _DEFAULT_VALIDATION_SPLIT_NUM, apply_validation_choice


def _make_config_tab():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from gui.tabs.config_tab import ConfigTab

    app = QApplication.instance() or QApplication([])
    assert app is not None
    return ConfigTab()


def test_zero_split_num_unchecks_use_valid():
    """Setting the spinbox to 0 mirrors onto the checkbox (validation off),
    so the existing save path writes 0 instead of coercing to the default."""
    tab = _make_config_tab()
    try:
        use_valid_w = tab._w.get("use_valid")
        vsn_w = tab._w.get("validation_split_num")
        assert use_valid_w is not None and vsn_w is not None

        # Enable + a real count, then zero it out the way a user would.
        vsn_w.setValue(16)
        assert use_valid_w.isChecked()
        vsn_w.setValue(0)
        assert not use_valid_w.isChecked()
    finally:
        tab.deleteLater()


def test_ticking_use_valid_seeds_positive_default():
    """Ticking the box when the count is 0 surfaces a positive default in the
    spinbox up front (no silent 0→16 only at save time)."""
    tab = _make_config_tab()
    try:
        use_valid_w = tab._w.get("use_valid")
        vsn_w = tab._w.get("validation_split_num")
        assert use_valid_w is not None and vsn_w is not None

        vsn_w.setValue(0)
        assert not use_valid_w.isChecked()
        use_valid_w.setChecked(True)
        assert vsn_w.value() > 0
    finally:
        tab.deleteLater()


def test_apply_validation_choice_disabled_writes_zero():
    """When the checkbox follows the spinbox to 0, the save helper writes a
    concrete validation_split_num = 0 (disabled), never the 16 default."""
    out: dict = {}
    apply_validation_choice(out, enabled=False, split_num=0, base_split_num=0)
    assert out["datasets"][0]["validation_split_num"] == 0
    assert out["datasets"][0]["validation_split"] == 0.0
    assert _DEFAULT_VALIDATION_SPLIT_NUM == 16  # guards the historical default
