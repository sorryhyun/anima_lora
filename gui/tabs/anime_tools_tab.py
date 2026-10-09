"""AnimeToolsTab — the ``anime_tools`` curation panel, rendered in a tab.

On first open it seeds the panel's settings (``gui.core.anime_tools_panel``), reuses a
panel already serving this checkout or starts one, and loads the page into a
``QWebEngineView``. The server runs with ``--exit-with-window``: the page holds
its ``/api/alive`` stream open, so the server stops a few seconds after this
view (and any browser tab on it) is gone — closing the GUI reaps it. A start that
failed is retried the next time the tab is shown.

``QtWebEngineWidgets`` is imported here, not at module level: the tab sits behind
a ``LazyTabHolder`` and the import costs launch time.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import QLabel, QStackedWidget, QVBoxLayout, QWidget

from gui.core import anime_tools_panel
from gui.i18n import t
from gui.tabs.preprocess.knobs import DEFAULT_SOURCE_IMAGE_DIR
from gui.theme import ACTION_COLORS, Pad, tok

_POLL_MS = 400
_START_TIMEOUT_S = 90.0


class AnimeToolsTab(QWidget):
    def __init__(self, source_image_dir=lambda: None, parent=None) -> None:
        """``source_image_dir``: returns the Preprocess tab's source tree, which
        the panel's settings are seeded with on every (re)start."""
        super().__init__(parent)
        self._source_image_dir = source_image_dir
        self._url: str | None = None
        self._log = None
        self._log_offset = 0
        self._deadline = 0.0
        self._failed = False
        self._poll = QTimer(self)
        self._poll.setInterval(_POLL_MS)
        self._poll.timeout.connect(self._poll_launch)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(Pad.NONE)

        self.warning_lbl = QLabel()
        self.warning_lbl.setWordWrap(True)
        self.warning_lbl.setStyleSheet(
            f"color: {ACTION_COLORS['warning']}; padding: 0 {Pad.XS}px;"
        )
        self.warning_lbl.hide()
        lay.addWidget(self.warning_lbl)

        self._stack = QStackedWidget()
        self._placeholder = QLabel()
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet(f"color: {tok('text_dim')}; padding: 24px;")
        self._stack.addWidget(self._placeholder)
        self._view = self._make_view()
        if self._view is not None:
            self._stack.addWidget(self._view)
        lay.addWidget(self._stack, 1)

        self._start()

    def _make_view(self):
        try:
            from PySide6.QtWebEngineCore import QWebEnginePage
            from PySide6.QtWebEngineWidgets import QWebEngineView
        except ImportError:
            return None

        class _ExternalPage(QWebEnginePage):
            """Target of a ``_blank`` link: hands the URL to the system browser
            instead of opening a window of its own."""

            def acceptNavigationRequest(self, url, _type, _is_main_frame):  # noqa: N802
                QDesktopServices.openUrl(url)
                self.deleteLater()
                return False

        class _Page(QWebEnginePage):
            def createWindow(self, _type):  # noqa: N802
                return _ExternalPage(self)

        view = QWebEngineView()
        view.setPage(_Page(view))
        # Chromium clears to white until the page paints; the view is also kept
        # off-stack until its first load lands (``_on_load_finished``).
        view.page().setBackgroundColor(QColor(tok("window")))
        view.loadFinished.connect(self._on_load_finished)
        return view

    # ------------------------------------------------------------------ start

    def _start(self) -> None:
        self._poll.stop()
        self._url = None
        self._deadline = 0.0
        self._failed = False
        try:
            seeded = anime_tools_panel.seed_settings(
                self._source_image_dir() or DEFAULT_SOURCE_IMAGE_DIR
            )
        except Exception as exc:  # noqa: BLE001 — surface any seed failure
            self._fail(t("anime_tools_failed", err=exc))
            return
        if seeded.warnings:
            self.warning_lbl.setText(
                t(
                    "anime_tools_root_warning",
                    roots=", ".join(seeded.warnings),
                    path=seeded.path,
                )
            )
            self.warning_lbl.show()
        else:
            self.warning_lbl.hide()

        url = anime_tools_panel.find_running()
        if url is not None:
            self._load(url)
            return
        try:
            log = anime_tools_panel.log_path()
            self._log_offset = log.stat().st_size if log.exists() else 0
            self._log = anime_tools_panel.launch()
        except Exception as exc:  # noqa: BLE001 — surface any launch failure
            self._fail(t("anime_tools_failed", err=exc))
            return
        self._show_placeholder(t("anime_tools_starting", log=self._log))
        self._deadline = time.monotonic() + _START_TIMEOUT_S
        self._poll.start()

    def _poll_launch(self) -> None:
        url = anime_tools_panel.url_in_log(self._log, self._log_offset)
        if url is not None:
            self._poll.stop()
            self._load(url)
        elif time.monotonic() > self._deadline:
            self._poll.stop()
            self._fail(t("anime_tools_timeout", log=self._log))

    def _load(self, url: str) -> None:
        self._url = url
        if self._view is None:
            self._show_placeholder(t("anime_tools_no_webengine", url=url))
            return
        self._view.setUrl(QUrl(url))

    def _on_load_finished(self, ok: bool) -> None:
        if ok:
            self._stack.setCurrentWidget(self._view)
            return
        # The start line is printed just before uvicorn binds; a fresh launch
        # retries a refused first load until its deadline.
        if self._url is not None and time.monotonic() < self._deadline:
            url = self._url
            QTimer.singleShot(_POLL_MS, lambda: self._view.setUrl(QUrl(url)))
        else:
            self._stack.setCurrentWidget(self._view)

    def _show_placeholder(self, text: str) -> None:
        self._placeholder.setText(text)
        self._stack.setCurrentWidget(self._placeholder)

    def _fail(self, text: str) -> None:
        self._failed = True
        self._show_placeholder(text)

    def showEvent(self, event):  # noqa: N802 — Qt event handler name
        super().showEvent(event)
        if self._failed:
            self._start()
