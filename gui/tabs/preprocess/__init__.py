"""Preprocessing-tab package.

Layout:

- ``knobs.py``      — Qt-free knob table + load/env/overrides/elision logic
- ``_section.py``   — ``KnobSection``: one ``QGroupBox`` form over knob rows
- ``image_prep.py`` / ``text_caching.py`` / ``captions.py`` / ``masking.py``
                    — the section panels (domain widgets live with their section)
- ``tab.py``        — ``PreprocessingTab``: top bar, run buttons, job observer, log

``PreprocessingTab`` is exported lazily so ``import gui.tabs.preprocess.knobs``
stays PySide6-free (the knob unit tests and the launch-speed guard rely on it).
"""

from __future__ import annotations

__all__ = ["PreprocessingTab"]


def __getattr__(name: str):
    if name == "PreprocessingTab":
        from gui.tabs.preprocess.tab import PreprocessingTab

        return PreprocessingTab
    raise AttributeError(name)
