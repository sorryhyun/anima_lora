"""Reusable Qt widgets + the config-form field factory.

Split by concern into sibling modules — ``mixins`` (LazyTabMixin,
DirtyTrackingMixin), ``fields`` (the ``_widget``/``_read`` factory + label
helpers), ``buttons`` (action-button variants + SplitButtonStyle),
``target_res`` (the multi-scale tier row), ``sample_prompts`` (the sample
prompt editor), ``image_view`` (the zoom/pan image label) — but re-exported
here so every existing ``from gui.widgets import <name>`` call site keeps
working unchanged.
"""

from __future__ import annotations

from gui.widgets._qt_utils import _no_wheel
from gui.widgets.buttons import SplitButtonStyle, action_button, apply_variant
from gui.widgets.fields import (
    _ATTN_MODES,
    _LOOKS_RICH,
    _TOOLTIP_WRAP_MIN_LEN,
    TOOLTIP_WRAP_COLS,
    ClickableLabel,
    _display_width,
    _read,
    _widget,
    _wrap_plain,
    hint_label,
    make_field_label,
    wrap_tooltip,
)
from gui.widgets.image_view import ImageViewerDialog, ScaledImageLabel
from gui.widgets.mixins import DirtyTrackingMixin, LazyTabHolder, LazyTabMixin
from gui.widgets.sample_prompts import (
    SamplePromptsDialog,
    _normalize_prompt_lines,
    _SamplePromptRow,
    _SamplePromptsLauncher,
    _SamplePromptsWidget,
)
from gui.widgets.target_res import _target_res_tiers, _TargetResWidget

__all__ = [
    "_ATTN_MODES",
    "LazyTabHolder",
    "LazyTabMixin",
    "_target_res_tiers",
    "_TargetResWidget",
    "_SamplePromptRow",
    "_SamplePromptsWidget",
    "_normalize_prompt_lines",
    "SamplePromptsDialog",
    "_SamplePromptsLauncher",
    "_no_wheel",
    "_widget",
    "_read",
    "ClickableLabel",
    "TOOLTIP_WRAP_COLS",
    "_TOOLTIP_WRAP_MIN_LEN",
    "_LOOKS_RICH",
    "_display_width",
    "_wrap_plain",
    "wrap_tooltip",
    "make_field_label",
    "apply_variant",
    "action_button",
    "hint_label",
    "DirtyTrackingMixin",
    "ScaledImageLabel",
    "ImageViewerDialog",
    "SplitButtonStyle",
]
