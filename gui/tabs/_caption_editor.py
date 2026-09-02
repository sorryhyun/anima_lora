"""Caption-editing widgets split out of ``image_tab.py``.

A self-contained cluster (none of it references the owning ``ImageViewerTab``):
the boxed tag editor with autocomplete, char-level diff helpers, the on-disk
caption history sidecar, and the version-browser dialog.
"""

from __future__ import annotations

import difflib
import json
from datetime import datetime
from html import escape
from pathlib import Path

from PySide6.QtCore import QEvent, QRect, QStringListModel, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPen,
    QTextBlockFormat,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QListWidget,
    QSplitter,
    QStyledItemDelegate,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import t
from gui.theme import tok

# Translucent green for inserted spans; deletions aren't rendered inline (they
# surface via the (+X / −Y) summary in the caption header).
_ADD_BG = QColor(60, 130, 70, 120)


def _add_format() -> QTextCharFormat:
    fmt = QTextCharFormat()
    fmt.setBackground(_ADD_BG)
    return fmt


def _diff_spans(old: str, new: str) -> tuple[list[tuple[int, int]], int, int]:
    """Char-level diff between old and new.

    Returns (insert_spans_in_new, total_added_chars, total_removed_chars).
    insert_spans are (j1, j2) ranges in `new` that should be highlighted.
    """
    if old == new:
        return [], 0, 0
    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    spans: list[tuple[int, int]] = []
    add_total = 0
    rem_total = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "insert":
            spans.append((j1, j2))
            add_total += j2 - j1
        elif tag == "replace":
            spans.append((j1, j2))
            add_total += j2 - j1
            rem_total += i2 - i1
        elif tag == "delete":
            rem_total += i2 - i1
    return spans, add_total, rem_total


def _history_path(caption_path: Path) -> Path:
    return caption_path.with_suffix(caption_path.suffix + ".history.jsonl")


def _read_history(caption_path: Path) -> list[dict]:
    """Return history entries (oldest first). Skips malformed lines."""
    hp = _history_path(caption_path)
    if not hp.exists():
        return []
    out: list[dict] = []
    for line in hp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and "ts" in entry and "text" in entry:
            out.append(entry)
    return out


def _append_history(caption_path: Path, prev_text: str) -> None:
    """Append the previous on-disk text as a history entry."""
    hp = _history_path(caption_path)
    entry = {"ts": datetime.now().isoformat(timespec="seconds"), "text": prev_text}
    with hp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# Border colors for inline tag boxes. @artist and "On the …" / "In the …"
# section headers keep warm/cool tints so the trainer's split rules
# (anima_smart_shuffle in library/anima/training.py) stay visible.
_BOX_BORDER_PLAIN = QColor("#e0e0e0")
_BOX_BORDER_ARTIST = QColor("#c9a227")
_BOX_BORDER_SECTION = QColor("#5e8eb0")


def _tag_ranges(text: str):
    """Yield ``(start, end, tag_text)`` for each comma-separated, trimmed tag.

    Whitespace around each tag is excluded from the range so the painted box
    hugs the visible characters, not the surrounding spaces.
    """
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i] in " \t\n":
            i += 1
        start = i
        while i < n and text[i] != ",":
            i += 1
        end = i
        while end > start and text[end - 1] in " \t\n":
            end -= 1
        if end > start:
            yield (start, end, text[start:end])
        if i < n and text[i] == ",":
            i += 1


def _tag_border_color(tag: str) -> QColor:
    # Mirror anime_tools.captions.taxonomy.is_artist_tag: `@<non-space>` is an artist
    # handle, but `@ @` (space-form booru eye-shape) is a general tag and must
    # not steal the artist tint. Inline to keep this module free of library/*.
    if len(tag) >= 2 and tag[0] == "@" and not tag[1].isspace():
        return _BOX_BORDER_ARTIST
    if (
        tag.startswith("On the ")
        or tag.startswith("In the ")
        or ". On the " in tag
        or ". In the " in tag
    ):
        return _BOX_BORDER_SECTION
    return _BOX_BORDER_PLAIN


class _TagCompletionDelegate(QStyledItemDelegate):
    """Autocomplete popup row: tag name on the left, its KB category dimmed
    on the right (e.g. ``long hair            general``)."""

    # Horizontal breathing room reserved between the tag name and its
    # right-aligned category, plus the right-edge inset (kept in sync with the
    # ``adjusted(0, 0, -_CATEGORY_INSET, 0)`` in ``paint``).
    _CATEGORY_GAP = 24
    _CATEGORY_INSET = 8

    def __init__(self, kind_lookup: dict[str, str], parent=None):
        super().__init__(parent)
        self._kind_lookup = kind_lookup

    def paint(self, painter, option, index) -> None:  # noqa: N802 — Qt API
        super().paint(painter, option, index)
        kind = self._kind_lookup.get(index.data(Qt.DisplayRole) or "")
        if not kind:
            return
        painter.save()
        painter.setPen(QColor("#9a9a9a"))
        painter.drawText(
            option.rect.adjusted(0, 0, -self._CATEGORY_INSET, 0),
            Qt.AlignRight | Qt.AlignVCenter,
            kind,
        )
        painter.restore()

    def sizeHint(self, option, index):  # noqa: N802 — Qt API
        # Reserve width for the right-aligned category so it never crowds the
        # tag name. ``sizeHintForColumn(0)`` (which sizes the popup) reads this.
        hint = super().sizeHint(option, index)
        kind = self._kind_lookup.get(index.data(Qt.DisplayRole) or "")
        if kind:
            extra = option.fontMetrics.horizontalAdvance(kind)
            hint.setWidth(
                hint.width() + extra + self._CATEGORY_GAP + self._CATEGORY_INSET
            )
        return hint


class BoxedCaptionEdit(QTextEdit):
    """QTextEdit that paints thin border boxes inline around each
    comma-separated tag.

    Uses ``viewportEvent`` rather than ``QTextCharFormat`` because Qt's
    text framework can set per-character backgrounds and foregrounds but
    not borders. We let Qt render the text normally, then overlay
    rectangles on the viewport by walking ``cursorRect()`` across each
    tag's character range. Boxes follow scroll, wrap, and live edits
    automatically because ``cursorRect()`` is always queried in current
    viewport coordinates.

    The font is configured with extra letter spacing and the document with
    a roomier line height so tag boxes have visible breathing room both
    horizontally (the comma+space between tags is wider) and vertically
    (wrapped lines don't crowd their box borders together).
    """

    # Emitted when the user clicks on an existing comma-separated tag; carries
    # the tag text so the owning tab can show its KB explanation.
    tag_clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        font = self.font()
        font.setPixelSize(14)
        # 115% letter spacing widens the gap between adjacent boxes instead of
        # manufacturing gaps via per-box geometry.
        font.setLetterSpacing(QFont.PercentageSpacing, 115)
        self.setFont(font)
        self._apply_block_format()
        # Tag autocomplete. The model is heavy (~114k rows) so the owning tab
        # builds it on a background thread and hands it over via
        # ``set_completion_data`` — until then the completer is simply absent.
        self._completer: QCompleter | None = None

    def setPlainText(self, text: str) -> None:  # noqa: N802 — Qt API
        # setPlainText replaces the document, so the line-height format we
        # applied earlier gets reset. Reapply after every full replacement.
        super().setPlainText(text)
        self._apply_block_format()

    def _apply_block_format(self) -> None:
        cursor = QTextCursor(self.document())
        cursor.select(QTextCursor.Document)
        fmt = QTextBlockFormat()
        # 140% ProportionalHeight: vertical separation between wrapped lines.
        fmt.setLineHeight(
            140, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value
        )
        cursor.mergeBlockFormat(fmt)

    def viewportEvent(self, event) -> bool:  # noqa: N802 — Qt API
        result = super().viewportEvent(event)
        if event.type() == QEvent.Paint:
            self._paint_boxes()
        return result

    def _paint_boxes(self) -> None:
        text = self.toPlainText()
        if not text.strip():
            return
        painter = QPainter(self.viewport())
        try:
            painter.setBrush(Qt.NoBrush)
            for start, end, tag in _tag_ranges(text):
                pen = QPen(_tag_border_color(tag))
                pen.setWidth(1)
                painter.setPen(pen)
                rects = [r for r in self._tag_rects(start, end) if r.width() > 0]
                for i, r in enumerate(rects):
                    # A tag split across a soft wrap draws "open" boxes: omit the
                    # right edge of every segment but the last and the left edge
                    # of every segment but the first, so the run reads as one
                    # continuous box flowing off the right margin and back in.
                    self._draw_tag_box(
                        painter, r, open_left=i > 0, open_right=i < len(rects) - 1
                    )
        finally:
            painter.end()

    def _draw_tag_box(
        self, painter: QPainter, r: QRect, *, open_left: bool, open_right: bool
    ) -> None:
        if not open_left and not open_right:
            painter.drawRoundedRect(r, 2, 2)
            return
        # Open side gets a square (unrounded) corner so the segment butts flush
        # against the line edge instead of curling back inward.
        left, right, top, bottom = r.left(), r.right(), r.top(), r.bottom()
        painter.drawLine(left, top, right, top)
        painter.drawLine(left, bottom, right, bottom)
        if not open_left:
            painter.drawLine(left, top, left, bottom)
        if not open_right:
            painter.drawLine(right, top, right, bottom)

    def _tag_rects(self, start: int, end: int) -> list[QRect]:
        """Per-line bounding rectangles for char range ``[start, end)``.

        Walks character-by-character so soft wraps (visual line breaks
        without an actual ``\\n``) get their own rectangle. For a typical
        caption (~500 chars) this is a few hundred ``cursorRect`` calls
        per paint — well under the budget for live editing.
        """
        if end <= start:
            return []
        cur = QTextCursor(self.document())
        cur.setPosition(start)
        cr = self.cursorRect(cur)
        line_left = cr.left()
        line_right = cr.left()
        line_top = cr.top()
        line_height = cr.height()
        rects: list[QRect] = []

        # Negative pad → box extends 1px OUTWARD so glyphs sit inside with a
        # halo; small extension leaves the comma+space gap between boxes wide.
        pad_x = -1
        pad_y = -1

        def _emit() -> None:
            w = line_right - line_left - 2 * pad_x
            h = line_height - 2 * pad_y
            if w > 0 and h > 0:
                rects.append(QRect(line_left + pad_x, line_top + pad_y, w, h))

        for pos in range(start + 1, end + 1):
            cur.setPosition(pos)
            cr = self.cursorRect(cur)
            if cr.top() != line_top:
                _emit()
                line_left = cr.left()
                line_right = cr.left()
                line_top = cr.top()
                line_height = cr.height()
            else:
                line_right = cr.left()
        _emit()
        return rects

    # --- Tag click → explanation -------------------------------------------

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 — Qt API
        super().mouseReleaseEvent(event)
        if event.button() != Qt.LeftButton or self.textCursor().hasSelection():
            return
        pos = self.cursorForPosition(event.position().toPoint()).position()
        for start, end, tag in _tag_ranges(self.toPlainText()):
            if start <= pos <= end:
                self.tag_clicked.emit(tag)
                return

    # --- Tag autocomplete ---------------------------------------------------

    # Cap on how many contains-matches the popup offers. Since ``names`` is
    # ordered by popularity, the first N matches are the N most popular.
    _MAX_COMPLETIONS = 30

    def set_completion_data(self, names, kind_lookup) -> None:
        """Attach the autocomplete model (``names`` already ordered by
        popularity, ``kind_lookup`` = name → category). Called on the main
        thread once the owning tab has built the data off-thread.

        We filter to the top ``_MAX_COMPLETIONS`` matches ourselves (refilling
        a small model on each keystroke) rather than handing the full ~114k-row
        list to ``QCompleter`` and letting it scroll endlessly."""
        self._completion_names = names
        self._completion_names_lc = [n.lower() for n in names]
        model = QStringListModel([], self)
        comp = QCompleter(model, self)
        comp.setWidget(self)
        comp.setCaseSensitivity(Qt.CaseInsensitive)
        comp.setCompletionMode(QCompleter.PopupCompletion)
        comp.setMaxVisibleItems(12)
        comp.popup().setItemDelegate(_TagCompletionDelegate(kind_lookup, comp.popup()))
        comp.activated[str].connect(self._insert_completion)
        self._completer = comp
        self._completion_model = model

    def _top_matches(self, prefix: str) -> list[str]:
        """Top ``_MAX_COMPLETIONS`` tags matching ``prefix``, prefix-matches
        first then mid-word substring matches, each ranked by popularity.

        So typing ``ye`` surfaces ``yellow eyes`` ahead of ``eye`` — a tag that
        *starts* with what you typed is a stronger hit than one that merely
        contains it."""
        needle = prefix.lower()
        prefix_hits: list[str] = []
        contains_hits: list[str] = []
        for name, name_lc in zip(self._completion_names, self._completion_names_lc):
            if name_lc.startswith(needle):
                prefix_hits.append(name)
            elif needle in name_lc:
                contains_hits.append(name)
            if len(prefix_hits) >= self._MAX_COMPLETIONS:
                # Already enough prefix matches; nothing later can outrank them.
                return prefix_hits[: self._MAX_COMPLETIONS]
        return (prefix_hits + contains_hits)[: self._MAX_COMPLETIONS]

    def _current_tag_prefix(self) -> tuple[str, bool]:
        """Return ``(prefix, in_tag_context)`` for the token under the cursor.

        Tags are comma-separated; a period boundary means the user is writing a
        prose sentence (the ``On the … . In the …`` caption sections), so the
        helper stays out of the way there.
        """
        pos = self.textCursor().position()
        text = self.toPlainText()
        start = pos
        while start > 0 and text[start - 1] not in ",.\n":
            start -= 1
        if start > 0 and text[start - 1] == ".":
            return "", False
        return text[start:pos].strip(), True

    def _insert_completion(self, completion: str) -> None:
        cursor = self.textCursor()
        pos = cursor.position()
        text = self.toPlainText()
        start = pos
        while start > 0 and text[start - 1] not in ",\n":
            start -= 1
        while start < pos and text[start] == " ":
            start += 1
        cursor.setPosition(start, QTextCursor.MoveAnchor)
        cursor.setPosition(pos, QTextCursor.KeepAnchor)
        cursor.insertText(completion)
        self.setTextCursor(cursor)

    def keyPressEvent(self, event) -> None:  # noqa: N802 — Qt API
        comp = self._completer
        if (
            comp is not None
            and comp.popup().isVisible()
            and event.key()
            in (
                Qt.Key_Enter,
                Qt.Key_Return,
                Qt.Key_Escape,
                Qt.Key_Tab,
                Qt.Key_Backtab,
            )
        ):
            event.ignore()  # let the popup consume navigation / accept keys
            return
        super().keyPressEvent(event)

        if comp is None:
            return
        prefix, in_tag = self._current_tag_prefix()
        if not in_tag or len(prefix) < 2:
            comp.popup().hide()
            return
        matches = self._top_matches(prefix)
        if not matches:
            comp.popup().hide()
            return
        if matches != self._completion_model.stringList():
            self._completion_model.setStringList(matches)
            # Model already holds only the matches; an empty prefix shows them all.
            comp.setCompletionPrefix("")
            comp.popup().setCurrentIndex(comp.completionModel().index(0, 0))
        rect = self.cursorRect()
        rect.setWidth(
            comp.popup().sizeHintForColumn(0)
            + comp.popup().verticalScrollBar().sizeHint().width()
        )
        comp.complete(rect)


def _unified_diff_html(old: str, new: str) -> str:
    """Tiny unified diff with red-/green+ coloring; empty string means no changes."""
    if old == new:
        return ""
    diff = difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        lineterm="",
        n=3,
    )
    rows: list[str] = []
    for line in diff:
        if line.startswith("---") or line.startswith("+++"):
            continue  # filenames are noise here
        if line.startswith("@@"):
            rows.append(f'<span style="color:{tok("link")};">{escape(line)}</span>')
        elif line.startswith("+"):
            rows.append(f'<span style="color:#9ad17a;">{escape(line)}</span>')
        elif line.startswith("-"):
            rows.append(f'<span style="color:#e07a7a;">{escape(line)}</span>')
        else:
            rows.append(f'<span style="color:{tok("text_dim")};">{escape(line)}</span>')
    if not rows:
        return ""
    return (
        '<pre style="font-family:monospace;font-size:11px;margin:0;">'
        + "\n".join(rows)
        + "</pre>"
    )


class CaptionVersionsDialog(QDialog):
    """Browse prior versions of a caption and restore one in-place."""

    def __init__(self, caption_path: Path, current_disk_text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("caption_versions_title", name=caption_path.stem))
        self.resize(820, 520)
        self._caption_path = caption_path
        self._current = current_disk_text
        self._restored: str | None = None  # set on Restore

        history = _read_history(caption_path)
        # Newest first — that's what users want to see at the top.
        self._history = list(reversed(history))

        lay = QVBoxLayout(self)

        sp = QSplitter(Qt.Horizontal)
        self.list = QListWidget()
        if not self._history:
            self.list.addItem(t("caption_versions_empty"))
            self.list.setEnabled(False)
        else:
            for entry in self._history:
                self.list.addItem(entry["ts"])
        self.list.currentRowChanged.connect(self._show_diff)
        sp.addWidget(self.list)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.diff = QTextBrowser()
        self.diff.setStyleSheet(
            f"QTextBrowser {{ background:{tok('base')}; color:{tok('text')}; "
            f"border:1px solid {tok('border_dim')}; padding:6px; }}"
        )
        rl.addWidget(self.diff, 1)
        sp.addWidget(right)
        sp.setSizes([220, 600])
        lay.addWidget(sp, 1)

        btns = QDialogButtonBox()
        self.restore_btn = btns.addButton(
            t("caption_versions_restore"), QDialogButtonBox.AcceptRole
        )
        self.restore_btn.setEnabled(False)
        self.restore_btn.clicked.connect(self._restore)
        close_btn = btns.addButton(
            t("caption_versions_close"), QDialogButtonBox.RejectRole
        )
        close_btn.clicked.connect(self.reject)
        lay.addWidget(btns)

        if self._history:
            self.list.setCurrentRow(0)

    def _show_diff(self, row: int) -> None:
        if not (0 <= row < len(self._history)):
            self.restore_btn.setEnabled(False)
            self.diff.setHtml("")
            return
        prev = self._history[row]["text"]
        html = _unified_diff_html(prev, self._current)
        if not html:
            self.diff.setHtml(
                f'<i style="color:{tok("text_dim")};">{t("caption_diff_clean")}</i>'
            )
        else:
            self.diff.setHtml(html)
        self.restore_btn.setEnabled(True)

    def _restore(self) -> None:
        row = self.list.currentRow()
        if not (0 <= row < len(self._history)):
            return
        self._restored = self._history[row]["text"]
        self.accept()

    def restored_text(self) -> str | None:
        return self._restored
