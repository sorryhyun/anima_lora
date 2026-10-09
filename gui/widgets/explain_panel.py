"""ExplainPanel — the right-hand help / image-gallery pane of the form tabs.

One ``QTextBrowser`` that shows a method guide, a field's help, or the newest
few images of a directory (test output, training samples). ``mode`` records
which of those is up so a host's poll can leave field help alone.
"""

from __future__ import annotations

import html
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QTextBrowser

from gui.core.paths import IMAGE_EXTS
from gui.explanations import field_help_html
from gui.i18n import t
from gui.theme import rich_text_pt, tok
from gui.widgets.image_view import ImageViewerDialog


def newest_images(d: Path, limit: int = 4, *, since: float | None = None) -> list[Path]:
    """Newest images in ``d`` by mtime. ``since`` (epoch seconds) drops any
    written before it — the training-sample gallery passes the job's start
    time so a fresh run never shows the previous run's stale samples."""
    if not d.is_dir():
        return []
    dated: list[tuple[float, Path]] = []
    for p in d.iterdir():
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            mt = p.stat().st_mtime
        except OSError:  # file vanished mid-scan (e.g. a clobbering re-run)
            continue
        if since is not None and mt < since:
            continue
        dated.append((mt, p))
    dated.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in dated[:limit]]


def _h2(text: str) -> str:
    return (
        f"<h2 style='margin:0 0 10px 0; font-size:{rich_text_pt(18)};'>"
        f"{html.escape(text)}</h2>"
    )


def _dim(text: str, extra: str = "") -> str:
    return (
        f"<p style='color:{tok('text_dim')}; font-style:italic;{extra}'>"
        f"{html.escape(text)}</p>"
    )


class ExplainPanel(QTextBrowser):
    """``mode``: ``None`` (guide / placeholder), ``"help"``, or a gallery's
    ``kind`` (``"test"``, ``"sample"``)."""

    def __init__(self, parent=None, *, min_width: int = 320):
        super().__init__(parent)
        self.mode: str | None = None
        # Identity of the gallery render currently showing (None = not a
        # gallery); lets a poll skip setHtml when nothing changed.
        self._gallery_sig: tuple | None = None
        # Links are dispatched in _on_anchor (gallery zoom, fragments, external).
        self.setOpenLinks(False)
        self.anchorClicked.connect(self._on_anchor)
        self.setStyleSheet(
            f"QTextBrowser {{ font-size: 120%; padding: 12px; "
            f"background: {tok('panel')}; color: {tok('text')}; }}"
        )
        self.setMinimumWidth(min_width)

    def set_html(self, content: str, *, gallery_sig: tuple | None = None) -> None:
        """Every write goes through here so the gallery signature stays true."""
        self._gallery_sig = gallery_sig
        self.setHtml(content)

    def show_guide(self, guide_html: str | None) -> None:
        """A method guide, or the "click a field" hint when there is none."""
        self.mode = None
        self.set_html(guide_html or _dim(t("click_field_for_help")))

    def show_field_help(
        self, field: str, help_text: str | None, notes: tuple[str, ...] = ()
    ) -> None:
        self.mode = "help"
        parts = [_h2(field)]
        if help_text:
            parts.append(
                f"<p style='font-size:{rich_text_pt(15)}; line-height:1.6;'>"
                f"{field_help_html(help_text)}</p>"
            )
        else:
            parts.append(_dim(t("no_help_available")))
        parts.extend(_dim(f"• {note}", " margin-top:12px;") for note in notes)
        self.set_html("".join(parts))

    def show_gallery(
        self, kind: str, title_key: str, empty_key: str, imgs: list[Path]
    ) -> None:
        """Render ``imgs`` as an ``<img>`` stack. Hosts poll this, so an
        unchanged image set skips setHtml (which resets scroll to top); a real
        refresh restores the previous scroll offset."""
        self.mode = kind

        def _mtime(p: Path):
            try:
                return p.stat().st_mtime_ns
            except OSError:
                return None

        sig = (title_key, tuple((str(p), _mtime(p)) for p in imgs))
        if sig == self._gallery_sig:
            return
        if not imgs:
            self.set_html(_h2(t(title_key)) + _dim(t(empty_key)), gallery_sig=sig)
            return
        parts = [_h2(t(title_key))]
        for p in imgs:
            url = p.resolve().as_uri()
            magnify = "magnify" + url[len("file") :]
            parts.append(
                f"<p style='margin:0 0 10px 0;'>"
                f"<a href='{magnify}'><img src='{url}' style='max-width:100%;'/></a><br/>"
                f"<span style='color:{tok('text_dim')}; font-size:{rich_text_pt(11)};'>{html.escape(p.name)}</span> "
                f"<a href='{magnify}' style='text-decoration:none; font-size:{rich_text_pt(12)};'>🔍</a>"
                f"</p>"
            )
        sb = self.verticalScrollBar()
        pos = sb.value()
        self.set_html("".join(parts), gallery_sig=sig)
        sb.setValue(min(pos, sb.maximum()))

    def _on_anchor(self, url: QUrl) -> None:
        """``magnify:`` is the gallery zoom scheme (a file URI with the scheme
        swapped); in-document fragments scroll, everything else opens externally."""
        if url.scheme() == "magnify":
            fileurl = QUrl(url)
            fileurl.setScheme("file")
            ImageViewerDialog(Path(fileurl.toLocalFile()), self.window()).show()
        elif url.isRelative() and url.hasFragment():
            self.scrollToAnchor(url.fragment())
        else:
            QDesktopServices.openUrl(url)
