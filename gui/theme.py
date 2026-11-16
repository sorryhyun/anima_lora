"""Named visual themes for the GUI — Dark / Light / Sepia.

Each theme is a flat set of **semantic color tokens** (background, panel, text, border, accent, …);
``apply_theme`` turns the active theme into a ``QPalette`` + global stylesheet,
and ``tok()`` lets individual widgets pull the same tokens instead of hardcoding
hex literals.

Design notes
------------
* Only the neutral chrome (window/panel/input/text/border) varies by theme.
  Saturated *action* buttons are white-on-color and theme-independent: the
  ``ACTION_COLORS`` table, surfaced as global ``[variant="…"]`` rules. A call
  site sets ``variant`` via :func:`gui.widgets.action_button` /
  :func:`gui.widgets.apply_variant`, not an inline ``setStyleSheet``.
* Spacing/padding are tokens too: :class:`Pad` / :class:`Gap`.
* Widgets read tokens at *build* time, so a theme switch re-applies the palette +
  global stylesheet and the caller rebuilds the window for ``tok()`` lookups.

This module imports Qt; ``gui/core/`` stays Qt-free.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QPainter,
    QPalette,
    QPixmap,
    QPolygon,
)
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QProxyStyle, QStyle

from gui.core.paths import DEFAULT_THEME, get_setting, set_setting

# Bundled UI font (Pretendard, OFL), three static weights; see gui/fonts/README.md.
_FONT_DIR = Path(__file__).parent / "fonts"
_PRETENDARD_FILES = (
    "Pretendard-Regular.ttf",
    "Pretendard-Medium.ttf",
    "Pretendard-Bold.ttf",
)
# Loaded family name, or None (files missing / refused by Qt): OS fonts only.
_bundled_family: str | None = None
_fonts_loaded = False


@dataclass(frozen=True)
class Theme:
    """A flat palette of semantic color tokens (hex strings).

    Naming is by *role*, not appearance, so the same key means the same thing in
    every theme:

    * ``window`` deepest app background; ``base`` text-entry/list canvas;
      ``panel`` a slightly raised neutral surface (help/log/preview boxes).
    * ``input_bg`` / ``input_hover`` text-field fill; ``surface`` / ``surface_hover``
      neutral button & tab fill.
    * ``text`` primary; ``text_bright`` max-contrast; ``text_dim`` secondary/hint.
    * ``border`` / ``border_dim`` strong / subtle separators.
    * ``accent`` selection/highlight; ``accent_text`` text on accent.
    * ``link`` / ``link_visited``; status ``ok`` / ``warn`` / ``err`` (text) and
      ``ok_bg`` / ``warn_bg`` / ``err_bg`` (banner fills that the matching
      status text reads on).
    * ``scroll_bg`` / ``scroll_handle`` / ``scroll_handle_hover`` scrollbars.
    * ``is_dark`` whether the theme reads as dark (lets callers branch when a
      token isn't enough).
    """

    name: str
    is_dark: bool
    window: str
    base: str
    panel: str
    input_bg: str
    input_hover: str
    surface: str
    surface_hover: str
    text: str
    text_bright: str
    text_dim: str
    border: str
    border_dim: str
    accent: str
    accent_text: str
    tab_selected: str
    tooltip_bg: str
    link: str
    link_visited: str
    ok: str
    warn: str
    err: str
    ok_bg: str
    warn_bg: str
    err_bg: str
    scroll_bg: str
    scroll_handle: str
    scroll_handle_hover: str


_DARK = Theme(
    name="dark",
    is_dark=True,
    window="#1e1e1e",
    base="#191919",
    panel="#2b2b2b",
    input_bg="#2a2a2a",
    input_hover="#2c2c2c",
    surface="#3a3a3a",
    surface_hover="#4a4a4a",
    text="#dcdcdc",
    text_bright="#ffffff",
    text_dim="#888888",
    border="#555555",
    border_dim="#444444",
    accent="#3c78c8",
    accent_text="#ffffff",
    tab_selected="#1e1e1e",
    tooltip_bg="#323232",
    link="#ffb86b",
    link_visited="#e6944e",
    ok="#4ade80",
    warn="#fbbf24",
    err="#f87171",
    ok_bg="#0a3d2a",
    warn_bg="#3d2e0a",
    err_bg="#3d0a0a",
    scroll_bg="#242424",
    scroll_handle="#b8b8b8",
    scroll_handle_hover="#d0d0d0",
)

_LIGHT = Theme(
    name="light",
    is_dark=False,
    window="#f4f4f4",
    base="#ffffff",
    panel="#ececec",
    input_bg="#ffffff",
    input_hover="#f0f0f0",
    surface="#e4e4e4",
    surface_hover="#d6d6d6",
    text="#1f1f1f",
    text_bright="#000000",
    text_dim="#6a6a6a",
    border="#b6b6b6",
    border_dim="#d2d2d2",
    accent="#2f6fb0",
    accent_text="#ffffff",
    tab_selected="#ffffff",
    tooltip_bg="#fafafa",
    link="#1a5fb4",
    link_visited="#7a3fb0",
    ok="#1a7f37",
    warn="#9a6700",
    err="#c01c28",
    ok_bg="#dcf5e3",
    warn_bg="#fff1cc",
    err_bg="#fde2e1",
    scroll_bg="#e0e0e0",
    scroll_handle="#b4b4b4",
    scroll_handle_hover="#909090",
)

_SEPIA = Theme(
    name="sepia",
    is_dark=False,
    window="#f4ecd8",
    base="#fffaf0",
    panel="#ebe0c8",
    input_bg="#fffaf0",
    input_hover="#f7efdd",
    surface="#e6dabe",
    surface_hover="#dccfae",
    text="#4a4036",
    text_bright="#2a231b",
    text_dim="#8a7d68",
    border="#c8b896",
    border_dim="#d8cbae",
    accent="#b4691f",
    accent_text="#ffffff",
    tab_selected="#fffaf0",
    tooltip_bg="#f0e6cf",
    link="#9a4f12",
    link_visited="#7a3f0f",
    ok="#5a7d2a",
    warn="#9a6700",
    err="#b03028",
    ok_bg="#e3ebcc",
    warn_bg="#f5e2b8",
    err_bg="#f2d3c8",
    scroll_bg="#e6dabe",
    scroll_handle="#c0ad88",
    scroll_handle_hover="#a89468",
)

# Saturated action-button palette, theme-independent (white-on-color reads on any
# background). Each entry becomes a global `[variant="<key>"]` rule.
ACTION_COLORS: dict[str, str] = {
    "primary": "#27ae60",  # green — the main go/run action (Train, Run)
    "secondary": "#8e44ad",  # purple — an alternate action (Test)
    "info": "#2980b9",  # blue — neutral-ish action (Group, Preprocess, Delete)
    "success": "#16a085",  # teal — affirmative/apply/download (Merge, Download, Guide)
    "danger": "#c0392b",  # red — destructive/stop (Stop, Delete)
    "warning": "#e67e22",  # orange — needs-attention (unsaved/dirty Save)
    "busy": "#7f8c8d",  # gray — in-progress / temporarily inert
}

# Top-bar nav / overlay-toggle buttons: a muted family, (base, hover) per role.
# Surfaced via nav_button_qss().
NAV_COLORS: dict[str, tuple[str, str]] = {
    "queue": ("#5d6d7e", "#6b7c8c"),  # Queue overlay toggle — idle (slate)
    "queue_on": ("#34495e", "#3d566e"),  # Queue overlay toggle — active (navy)
    "tensorboard": ("#2471a3", "#2e86c1"),  # TensorBoard toggle — idle (blue)
    "tensorboard_on": ("#117864", "#148f77"),  # TensorBoard toggle — active (teal)
    "update": ("#b45309", "#d97706"),  # "Update available" alert (amber)
}


def _blend(hex_a: str, hex_b: str, t: float) -> str:
    """Linear blend of two hex colors — ``t`` of *a* mixed with ``1-t`` of *b*."""
    a, b = QColor(hex_a), QColor(hex_b)
    mix = lambda ca, cb: round(ca * t + cb * (1 - t))  # noqa: E731
    return QColor(
        mix(a.red(), b.red()), mix(a.green(), b.green()), mix(a.blue(), b.blue())
    ).name()


def action_button_qss(
    variant: str, *, selector: str = "QPushButton, QToolButton"
) -> str:
    """Per-widget stylesheet for an action *variant* (same colors as the global
    ``[variant="…"]`` rules).

    The Train / Preprocess split buttons need it: a global app stylesheet bypasses
    the widget's ``setStyle()`` proxy and snaps the menu label to the full-button
    centre, while a per-widget rule composes with it. Don't add a
    ``::menu-button`` rule here; that re-breaks the centring
    (:class:`gui.widgets.buttons.SplitButtonStyle`)."""
    base = ACTION_COLORS[variant]
    return f"{selector}{{background:{base};color:white;font-weight:bold;padding:4px 16px;}}"


def nav_button_qss(role: str) -> str:
    """Per-widget stylesheet for a top-bar nav/toggle button; idle/active
    states are swapped at runtime via ``setStyleSheet``."""
    base, hover = NAV_COLORS[role]
    return (
        f"QPushButton {{ background:{base}; color:white; font-weight:bold; "
        f"padding:4px 12px; border:1px solid {base}; border-radius:3px; }}"
        f"QPushButton:hover {{ background:{hover}; }}"
    )


class Pad:
    """Padding / contents-margin design tokens (px)."""

    NONE = 0
    XS = 4
    SM = 8
    MD = 12
    LG = 16


class Gap:
    """Inter-widget spacing design tokens (px) for ``layout.setSpacing(...)``."""

    NONE = 0
    TIGHT = 4
    ROW = 6
    SECTION = 8


THEMES: dict[str, Theme] = {t.name: t for t in (_DARK, _LIGHT, _SEPIA)}

# Display order + i18n label keys for the Settings selector.
THEME_ORDER = ("dark", "light", "sepia")
THEME_LABEL_KEYS = {
    "dark": "settings_theme_dark",
    "light": "settings_theme_light",
    "sepia": "settings_theme_sepia",
}

# The live theme, set by apply_theme() so tok() never re-reads settings.
_active: Theme = THEMES[DEFAULT_THEME]
_base_style = None
_scroll_handle_style = None


class _ScrollHandleStyle(QProxyStyle):
    """Keep native scrollbars, but repaint only the current-position handle."""

    def __init__(self, base_style):
        super().__init__(base_style)
        self._handle = QColor(_active.scroll_handle)
        self._handle_hover = QColor(_active.scroll_handle_hover)

    def set_scroll_colors(self, handle: str, hover: str) -> None:
        self._handle = QColor(handle)
        self._handle_hover = QColor(hover)

    def drawComplexControl(self, control, option, painter, widget=None):
        super().drawComplexControl(control, option, painter, widget)
        if control != QStyle.CC_ScrollBar:
            return
        rect = self.subControlRect(
            QStyle.CC_ScrollBar, option, QStyle.SC_ScrollBarSlider, widget
        )
        if not rect.isValid() or rect.width() <= 0 or rect.height() <= 0:
            return
        is_hover = (
            option.activeSubControls & QStyle.SC_ScrollBarSlider
            and option.state & QStyle.State_MouseOver
        )
        color = self._handle_hover if is_hover else self._handle
        if rect.height() >= rect.width():
            handle_rect = rect.adjusted(3, 2, -3, -2)
        else:
            handle_rect = rect.adjusted(2, 3, -2, -3)
        if (
            not handle_rect.isValid()
            or handle_rect.width() <= 0
            or handle_rect.height() <= 0
        ):
            handle_rect = rect
        radius = min(handle_rect.width(), handle_rect.height()) / 2
        painter.save()
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(handle_rect, radius, radius)
        painter.restore()


def current_theme_name() -> str:
    """The persisted theme name (falls back to the default)."""
    name = get_setting("theme", DEFAULT_THEME)
    return name if name in THEMES else DEFAULT_THEME


def active_theme() -> Theme:
    """The Theme last passed to apply_theme() (the live look)."""
    return _active


def tok(key: str) -> str:
    """One color token from the active theme, e.g. ``tok("panel")``. Raises on
    an unknown key."""
    return getattr(_active, key)


def set_theme(name: str) -> None:
    """Persist the chosen theme name (does not re-apply — caller does that)."""
    if name in THEMES:
        set_setting("theme", name)


# App font size handed to app.setFont; clamped so a stray override can't shrink/blow up the UI.
DEFAULT_FONT_SIZE = 10
FONT_SIZE_MIN = 8
FONT_SIZE_MAX = 18


def current_font_size() -> int:
    """The persisted app font point size (clamped, falls back to the default)."""
    try:
        size = int(get_setting("font_size", DEFAULT_FONT_SIZE))
    except (TypeError, ValueError):
        return DEFAULT_FONT_SIZE
    return max(FONT_SIZE_MIN, min(FONT_SIZE_MAX, size))


def set_font_size(size: int) -> None:
    """Persist the chosen app font size (does not re-apply — caller does that)."""
    set_setting("font_size", max(FONT_SIZE_MIN, min(FONT_SIZE_MAX, int(size))))


def rich_text_pt(px_at_default: float) -> str:
    """A rich-text ``font-size`` (in pt) that scales with the app font knob.

    Qt rich text honours only ``px``/``pt`` (``em``/``%`` are ignored), so a px
    literal ignores the font knob. Converts a size given "in px at the 10pt
    default" into pt scaled by the current app font. 10pt ≈ 13.33px at 96dpi."""
    return f"{px_at_default * current_font_size() / 13.333:.1f}pt"


def _build_palette(t: Theme) -> QPalette:
    p = QPalette()
    for role, color in [
        (QPalette.Window, QColor(t.window)),
        (QPalette.WindowText, QColor(t.text)),
        (QPalette.Base, QColor(t.base)),
        (QPalette.AlternateBase, QColor(t.panel)),
        (QPalette.ToolTipBase, QColor(t.tooltip_bg)),
        (QPalette.ToolTipText, QColor(t.text)),
        (QPalette.Text, QColor(t.text)),
        (QPalette.Button, QColor(t.surface)),
        (QPalette.ButtonText, QColor(t.text)),
        (QPalette.Highlight, QColor(t.accent)),
        (QPalette.HighlightedText, QColor(t.accent_text)),
        (QPalette.Link, QColor(t.link)),
        (QPalette.LinkVisited, QColor(t.link_visited)),
        (QPalette.PlaceholderText, QColor(t.text_dim)),
    ]:
        p.setColor(role, color)
    # Disabled text needs an explicit dim or it inherits full-contrast text.
    disabled = QColor(t.text_dim)
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, disabled)
    return p


_ARROW_DIR = Path(tempfile.gettempdir()) / "anima_gui_arrows"
_arrow_cache: dict[tuple[str, str], str] = {}


def _arrow_icon(color: str, direction: str) -> str:
    """Render a small filled-triangle arrow PNG in ``color`` and return its path.

    Qt stylesheets paint the CSS border-triangle trick as a filled box, and the
    native spin arrow is dark-on-dark in the dark theme. Cached per
    (color, direction) in a temp dir."""
    key = (color, direction)
    cached = _arrow_cache.get(key)
    if cached:
        return cached
    _ARROW_DIR.mkdir(parents=True, exist_ok=True)
    # 2x size for crisp rendering on HiDPI; the stylesheet draws it at ~11px.
    w, h = 22, 14
    pm = QPixmap(w, h)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    pad_x, pad_y = 4, 3
    if direction == "up":
        tri = QPolygon(
            [
                QPoint(pad_x, h - pad_y),
                QPoint(w - pad_x, h - pad_y),
                QPoint(w // 2, pad_y),
            ]
        )
    else:
        tri = QPolygon(
            [QPoint(pad_x, pad_y), QPoint(w - pad_x, pad_y), QPoint(w // 2, h - pad_y)]
        )
    p.drawPolygon(tri)
    p.end()
    safe = color.lstrip("#")
    path = _ARROW_DIR / f"{direction}_{safe}.png"
    pm.save(str(path), "PNG")
    # Qt CSS wants forward slashes even on Windows.
    url = path.as_posix()
    _arrow_cache[key] = url
    return url


def _action_button_rules() -> str:
    """Global-stylesheet rules for the saturated action-button variants.

    One ``[variant="…"]`` block per :data:`ACTION_COLORS` entry, for both
    ``QPushButton`` and ``QToolButton``; hover/pressed shades derive from the
    base color."""
    blocks = []
    for variant, base in ACTION_COLORS.items():
        hover = QColor(base).lighter(115).name()
        pressed = QColor(base).darker(112).name()
        # Disabled: blended toward gray with dim text so it reads as off, keeping
        # a faint variant tint.
        dis_bg = _blend(base, "#4a4a4a", 0.4)
        dis_fg = "#c8c8c8"
        blocks.append(f"""
        QPushButton[variant="{variant}"], QToolButton[variant="{variant}"] {{
            background: {base}; color: #ffffff; font-weight: bold;
            padding: 4px 16px; border: 1px solid {base}; border-radius: 3px;
        }}
        QPushButton[variant="{variant}"]:hover, QToolButton[variant="{variant}"]:hover {{
            background: {hover}; border-color: {hover};
        }}
        QPushButton[variant="{variant}"]:pressed, QToolButton[variant="{variant}"]:pressed {{
            background: {pressed}; border-color: {pressed};
        }}
        QPushButton[variant="{variant}"]:disabled, QToolButton[variant="{variant}"]:disabled {{
            background: {dis_bg}; border-color: {dis_bg}; color: {dis_fg};
        }}""")
    return "".join(blocks)


# Main-window model modes (``MainWindow`` sets ``mode`` on the top bar). The bar's
# fill is theme-independent with white labels; ``barButton`` buttons keep the
# window fill.
MODES = ("anima", "moss")
MODE_BAR = {"anima": "#000000", "moss": "#3a1859"}


def _mode_rules(t: Theme) -> str:
    rules = "".join(
        f'QFrame#modeBar[mode="{m}"] {{ background: {c}; }}'
        for m, c in MODE_BAR.items()
    )
    return (
        rules + "QFrame#modeBar { border: none; }"
        "QFrame#modeBar QLabel { color: #ffffff; background: transparent; }"
        "QComboBox#modeCombo { font-weight: bold; }"
        f'QPushButton[barButton="true"] {{ background: {t.window}; color: {t.text}; }}'
        f'QPushButton[barButton="true"]:hover {{ background: {t.surface_hover}; }}'
    )


def _build_stylesheet(
    t: Theme, font_family: str = "", font_size: int = DEFAULT_FONT_SIZE
) -> str:
    # Enforce font family+size via stylesheet (not just app.setFont): Windows native style ignores the app QFont, and stylesheet-resolved widgets don't reliably inherit its point size.
    font_rule = (
        f"* {{ font-family: {font_family}; font-size: {font_size}pt; }}\n"
        if font_family
        else f"* {{ font-size: {font_size}pt; }}\n"
    )
    return f"""
        {font_rule}
        QGroupBox {{
            font-weight: bold; border: 1px solid {t.border_dim};
            border-radius: 4px; margin-top: 8px; padding-top: 16px;
        }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
        QPushButton {{ padding: 4px 12px; border: 1px solid {t.border}; border-radius: 3px; }}
        QPushButton:hover {{ background: {t.surface_hover}; }}
        QScrollArea {{ border: none; }}
        QSplitter::handle {{ background: {t.border_dim}; }}
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit, QListWidget {{
            background: {t.input_bg}; color: {t.text}; border: 1px solid {t.border};
            border-radius: 3px; padding: 2px 4px;
        }}
        /* Spin steppers placed side by side on the right edge instead of the cramped native vertical stack. */
        QSpinBox, QDoubleSpinBox {{ padding-right: 34px; }}
        /* height intentionally oversized: Qt clamps the spin-button to field height, so this fills it edge-to-edge at any font size. */
        QSpinBox::up-button, QDoubleSpinBox::up-button {{
            subcontrol-origin: border; subcontrol-position: right;
            width: 16px; height: 200px; border-left: 1px solid {t.border}; background: {t.surface};
        }}
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            subcontrol-origin: border; subcontrol-position: right; margin-right: 16px;
            width: 16px; height: 200px; border-left: 1px solid {t.border}; background: {t.surface};
        }}
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
            background: {t.surface_hover};
        }}
        QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
        QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
            background: {t.accent};
        }}
        /* Theme-coloured arrows painted by _arrow_icon — Qt can't render a CSS border-triangle (paints a white box) and the native arrow is dark-on-dark in the dark theme. */
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
            image: url({_arrow_icon(t.text, "up")}); width: 11px; height: 7px;
        }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
            image: url({_arrow_icon(t.text, "down")}); width: 11px; height: 7px;
        }}
        QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {{
            image: url({_arrow_icon(t.text_bright, "up")});
        }}
        QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {{
            image: url({_arrow_icon(t.text_bright, "down")});
        }}
        QComboBox QAbstractItemView {{
            background: {t.input_bg}; color: {t.text}; selection-background-color: {t.accent};
        }}
        QTabWidget::pane {{ border: 1px solid {t.border_dim}; }}
        QTabBar::tab {{
            background: {t.input_bg}; color: {t.text}; border: 1px solid {t.border_dim};
            padding: 6px 14px;
            font-size: {font_size + 1}pt; font-weight: 500;
            border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px;
        }}
        QTabBar::tab:selected {{ background: {t.tab_selected}; color: {t.text_bright}; }}
        QTabBar::tab:hover {{ background: {t.surface}; }}
        QToolTip {{ max-width: 400px; background: {t.tooltip_bg}; color: {t.text};
            border: 1px solid {t.border}; }}
        QMenu {{
            background: {t.input_bg}; color: {t.text}; border: 1px solid {t.border};
        }}
        QMenu::item {{ padding: 4px 20px; background: transparent; color: {t.text}; }}
        QMenu::item:selected {{ background: {t.accent}; color: {t.accent_text}; }}
        QMenu::item:disabled {{ color: {t.text_dim}; }}
        QMenu::separator {{ height: 1px; background: {t.border_dim}; margin: 4px 8px; }}
        {_action_button_rules()}
        {_mode_rules(t)}
    """


def _load_bundled_fonts() -> str | None:
    """Register the bundled Pretendard weights with Qt (once per process).

    Returns the family name to head the UI font stack, or ``None`` if none
    loaded. All three weights register under one family, so ``setWeight``
    picks the instance."""
    global _bundled_family, _fonts_loaded
    if _fonts_loaded:
        return _bundled_family
    _fonts_loaded = True
    for fname in _PRETENDARD_FILES:
        path = _FONT_DIR / fname
        if not path.exists():
            continue
        fid = QFontDatabase.addApplicationFont(str(path))
        if fid == -1:
            continue
        fams = QFontDatabase.applicationFontFamilies(fid)
        if fams:
            _bundled_family = fams[0]  # "Pretendard"
    return _bundled_family


def _prefer_cleartype_font_engine() -> None:
    """Use Qt's GDI font engine on Windows — but only at integer DPI scaling.

    Qt 6's DirectWrite engine renders small text with grayscale antialiasing,
    soft next to native apps (worse on Pretendard); GDI's ClearType is crisp but
    fringed and undersized at fractional scaling (125% / 150%), so GDI is forced
    only at an integer scale.

    An unaware process always reports 96 DPI, so per-monitor-v2 awareness (the
    context Qt 6 sets anyway) is set before ``GetDpiForSystem``. Must run before
    ``QApplication`` is constructed. Skipped when ``QT_QPA_PLATFORM`` is already
    pinned (user choice, offscreen tests)."""
    if sys.platform != "win32" or "QT_QPA_PLATFORM" in os.environ:
        return
    try:
        import ctypes

        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 == -4. Harmless if it fails (awareness already set).
        ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
        dpi = ctypes.windll.user32.GetDpiForSystem()  # 96 == 100%
    except (OSError, AttributeError):
        return  # pre-1607 Windows / no DPI API — leave Qt on DirectWrite
    if not dpi:
        return
    scale = dpi / 96.0
    if abs(scale - round(scale)) < 0.01:  # integer scaling only
        os.environ["QT_QPA_PLATFORM"] = "windows:fontengine=gdi"


def apply_theme(app: QApplication, name: str | None = None) -> Theme:
    """Resolve + apply a theme to the whole app (palette + global stylesheet).

    ``name`` defaults to the persisted theme. Sets the active theme for ``tok()``
    and the app font; returns the applied Theme."""
    global _active, _base_style, _scroll_handle_style
    resolved = name if (name in THEMES) else current_theme_name()
    t = THEMES[resolved]
    _active = t

    # Pretendard leads but lacks Han ideographs + emoji, so CJK and color-emoji fonts must both be named explicitly (a CJK family alone won't cascade to emoji → tofu).
    if sys.platform == "win32":
        families = ["Malgun Gothic", "Segoe UI Emoji"]
    elif sys.platform == "darwin":
        families = ["Apple SD Gothic Neo", "Apple Color Emoji"]
    else:  # Linux: Noto Sans CJK + Noto Color Emoji ship with most distros.
        families = ["Noto Sans CJK KR", "Noto Color Emoji"]
    bundled = _load_bundled_fonts()
    if bundled:
        families.insert(0, bundled)
    font = QFont()
    font.setFamilies(families)
    # 10pt not the OS-native 9pt: Pretendard's smaller apparent size matches Segoe UI 9pt only at 10pt.
    font.setPointSize(current_font_size())
    font.setStyleHint(QFont.SansSerif)
    app.setFont(font)

    # Same stack for the stylesheet (native-styled Windows controls); Qt CSS needs multi-word families quoted.
    family_css = ", ".join(f'"{f}"' for f in families)

    if _base_style is None:
        _base_style = app.style()
        _scroll_handle_style = _ScrollHandleStyle(_base_style)
        app.setStyle(_scroll_handle_style)
    if _scroll_handle_style is not None:
        _scroll_handle_style.set_scroll_colors(t.scroll_handle, t.scroll_handle_hover)

    app.setPalette(_build_palette(t))
    app.setStyleSheet(_build_stylesheet(t, family_css, current_font_size()))
    return t
