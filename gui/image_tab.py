"""ImageViewerTab — dataset image browser with caption editor + history."""

from __future__ import annotations

import difflib
import json
from datetime import datetime
from html import escape
from pathlib import Path

from PySide6.QtCore import QEvent, QRect, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
    QTextBlockFormat,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui import ScaledImageLabel, _image_dirs, _imgs
from gui.i18n import t


# Inline-highlight palette for the editor: a translucent green for inserted
# spans (visible on the dark theme without overpowering the text). We don't
# render deletions inline — the user already removed those characters, so we
# surface them via the (+X / −Y) summary in the caption header instead.
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


# Border colors for inline tag boxes. Plain tags get a near-white border per
# the user's request — clearly distinct from the dark editor background and
# the light text. @artist boundary and "On the …" / "In the …" section
# headers keep their warm/cool tints so the trainer's split rules
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
    if tag.startswith("@"):
        return _BOX_BORDER_ARTIST
    if (
        tag.startswith("On the ")
        or tag.startswith("In the ")
        or ". On the " in tag
        or ". In the " in tag
    ):
        return _BOX_BORDER_SECTION
    return _BOX_BORDER_PLAIN


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

    def __init__(self, parent=None):
        super().__init__(parent)
        font = self.font()
        font.setPixelSize(14)
        # 115% letter spacing widens the natural gap between adjacent boxes
        # (the comma+space stretches with the rest of the text), which is
        # cheaper than fiddling with per-box geometry to manufacture gaps.
        font.setLetterSpacing(QFont.PercentageSpacing, 115)
        self.setFont(font)
        self._apply_block_format()

    def setPlainText(self, text: str) -> None:  # noqa: N802 — Qt API
        # setPlainText replaces the document, so the line-height format we
        # applied earlier gets reset. Reapply after every full replacement.
        super().setPlainText(text)
        self._apply_block_format()

    def _apply_block_format(self) -> None:
        cursor = QTextCursor(self.document())
        cursor.select(QTextCursor.Document)
        fmt = QTextBlockFormat()
        # ProportionalHeight = 1 (Qt's QTextBlockFormat.LineHeightTypes).
        # 140% gives clear vertical separation between wrapped lines without
        # making the editor feel stretched.
        fmt.setLineHeight(140, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
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
                for r in self._tag_rects(start, end):
                    if r.width() > 0:
                        painter.drawRoundedRect(r, 2, 2)
        finally:
            painter.end()

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

        # Box pads slightly OUTWARD from the text so the glyphs sit inside
        # with a 1px halo. Negative pad → outward extension. Keeping the
        # outward extension small (1px instead of 2px) leaves more of the
        # comma+space between tags untouched, so adjacent boxes have a
        # visibly wider gap. Going to 0 would put glyph edges right on the
        # border line, which reads as "text escaping the box."
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
            rows.append(f'<span style="color:#7aa6da;">{escape(line)}</span>')
        elif line.startswith("+"):
            rows.append(f'<span style="color:#9ad17a;">{escape(line)}</span>')
        elif line.startswith("-"):
            rows.append(f'<span style="color:#e07a7a;">{escape(line)}</span>')
        else:
            rows.append(f'<span style="color:#aaa;">{escape(line)}</span>')
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
            "QTextBrowser { background:#1e1e1e; color:#dcdcdc; "
            "border:1px solid #444; padding:6px; }"
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
                f'<i style="color:#aaa;">{t("caption_diff_clean")}</i>'
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


class ImageViewerTab(QWidget):
    def __init__(self):
        super().__init__()
        self._images: list[Path] = []
        self._dirs = _image_dirs()
        self._current_caption_path: Path | None = None
        self._disk_text: str = ""  # last value seen on disk (for diff baseline)
        self._suspend_dirty = False  # while we set text programmatically
        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel(t("directory")))
        self.dc = QComboBox()
        self.dc.addItems(self._dirs)
        self.dc.currentTextChanged.connect(self._load_dir)
        top.addWidget(self.dc, 1)
        self.cnt = QLabel()
        top.addWidget(self.cnt)
        lay.addLayout(top)

        sp = QSplitter(Qt.Horizontal)
        self.fl = QListWidget()
        self.fl.currentRowChanged.connect(self._on_row_changed)
        sp.addWidget(self.fl)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.img = ScaledImageLabel()
        self.img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img.setMinimumSize(400, 400)
        rl.addWidget(self.img, 1)

        # Caption header: label + buttons
        cap_head = QHBoxLayout()
        self.cap_label = QLabel(t("caption"))
        cap_head.addWidget(self.cap_label)
        cap_head.addStretch()
        self.save_btn = QPushButton(t("caption_save"))
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save)
        self.revert_btn = QPushButton(t("caption_revert"))
        self.revert_btn.setEnabled(False)
        self.revert_btn.clicked.connect(self._revert)
        self.versions_btn = QPushButton(t("caption_versions"))
        self.versions_btn.clicked.connect(self._open_versions)
        cap_head.addWidget(self.save_btn)
        cap_head.addWidget(self.revert_btn)
        cap_head.addWidget(self.versions_btn)
        rl.addLayout(cap_head)

        # Caption editor with inline tag-box overlay. Each comma-separated
        # tag is outlined by a thin rectangle painted on the viewport;
        # @artist and section headers use accent colors so the trainer's
        # split rules (anima_smart_shuffle in library/anima/training.py)
        # stay visible without a separate preview pane.
        self.cap = BoxedCaptionEdit()
        self.cap.setMaximumHeight(180)
        self.cap.textChanged.connect(self._on_text_changed)
        rl.addWidget(self.cap)

        # One-line grammar reminder, mirrors anima_smart_shuffle's split rules.
        self.guide = QLabel(t("caption_guideline_html"))
        self.guide.setWordWrap(True)
        self.guide.setTextFormat(Qt.RichText)
        self.guide.setStyleSheet(
            "QLabel { color:#888; font-size:11px; padding:2px 4px; }"
        )
        rl.addWidget(self.guide)

        sp.addWidget(right)
        sp.setSizes([220, 750])
        lay.addWidget(sp)

        QShortcut(QKeySequence("Right"), self, lambda: self._nav(1))
        QShortcut(QKeySequence("Left"), self, lambda: self._nav(-1))
        QShortcut(QKeySequence.Save, self, self._save)
        if self._dirs:
            self._load_dir(self.dc.currentText())

    # ── data loading ──────────────────────────────────────────

    def _load_dir(self, name: str):
        if not self._confirm_discard_if_dirty():
            # Roll the combo back without re-firing _load_dir.
            return
        d = self._dirs.get(name)
        if not d:
            return
        self._images = _imgs(d)
        self.fl.clear()
        for p in self._images:
            self.fl.addItem(p.stem)
        self.cnt.setText(t("n_images", n=len(self._images)))
        if self._images:
            self.fl.setCurrentRow(0)
        else:
            self._current_caption_path = None
            self._set_caption_text("")
            self._disk_text = ""
            self._refresh_buttons()
            self._refresh_inline_diff()

    def _on_row_changed(self, row: int):
        if not self._confirm_discard_if_dirty():
            # Snap back to the previous selection without recursing.
            prev = self._row_for_path(self._current_caption_path)
            if prev is not None and prev != row:
                self.fl.blockSignals(True)
                self.fl.setCurrentRow(prev)
                self.fl.blockSignals(False)
            return
        self._show(row)

    def _show(self, row: int):
        if not 0 <= row < len(self._images):
            return
        p = self._images[row]
        pm = QPixmap(str(p))
        if not pm.isNull():
            self.img.set_source(pm)
        cp = p.with_suffix(".txt")
        self._current_caption_path = cp
        if cp.exists():
            text = cp.read_text(encoding="utf-8")
        else:
            text = ""
        self._disk_text = text
        self._set_caption_text(text if text else "")
        self._refresh_buttons()
        self._refresh_inline_diff()

    # ── caption editing ───────────────────────────────────────

    def _set_caption_text(self, text: str) -> None:
        self._suspend_dirty = True
        try:
            self.cap.setPlainText(text)
        finally:
            self._suspend_dirty = False

    def _on_text_changed(self) -> None:
        if self._suspend_dirty:
            return
        self._refresh_buttons()
        self._refresh_inline_diff()

    def _is_dirty(self) -> bool:
        if self._current_caption_path is None:
            return False
        return self.cap.toPlainText() != self._disk_text

    def _refresh_buttons(self) -> None:
        dirty = self._is_dirty()
        self.save_btn.setEnabled(dirty)
        self.revert_btn.setEnabled(dirty)
        marker = t("caption_dirty_marker") if dirty else ""
        label = t("caption") + marker
        if dirty:
            _, add, rem = _diff_spans(self._disk_text, self.cap.toPlainText())
            if add or rem:
                label += "  " + t("caption_diff_stats", add=add, rem=rem)
        self.cap_label.setText(label)
        # Versions button is enabled whenever there's a caption file context;
        # the dialog itself shows "(no prior versions)" when empty.
        self.versions_btn.setEnabled(self._current_caption_path is not None)

    def _refresh_inline_diff(self) -> None:
        """Highlight inserted spans (vs disk) directly in the editor."""
        if self._current_caption_path is None:
            self.cap.setExtraSelections([])
            return
        spans, _, _ = _diff_spans(self._disk_text, self.cap.toPlainText())
        if not spans:
            self.cap.setExtraSelections([])
            return
        fmt = _add_format()
        sels: list[QTextEdit.ExtraSelection] = []
        doc = self.cap.document()
        for j1, j2 in spans:
            cur = QTextCursor(doc)
            cur.setPosition(j1)
            cur.setPosition(j2, QTextCursor.KeepAnchor)
            es = QTextEdit.ExtraSelection()
            es.cursor = cur
            es.format = fmt
            sels.append(es)
        self.cap.setExtraSelections(sels)

    def _save(self) -> None:
        cp = self._current_caption_path
        if cp is None or not self._is_dirty():
            return
        new_text = self.cap.toPlainText()
        try:
            # Snapshot the prior on-disk version into history before overwriting.
            # Skip when the previous file didn't exist (nothing to preserve).
            if cp.exists():
                _append_history(cp, self._disk_text)
            cp.write_text(new_text, encoding="utf-8")
        except OSError as e:
            QMessageBox.warning(
                self, t("error"), t("caption_save_failed", err=str(e))
            )
            return
        self._disk_text = new_text
        self._refresh_buttons()
        self._refresh_inline_diff()

    def _revert(self) -> None:
        if self._current_caption_path is None:
            return
        self._set_caption_text(self._disk_text)
        self._refresh_buttons()
        self._refresh_inline_diff()

    def _open_versions(self) -> None:
        cp = self._current_caption_path
        if cp is None:
            return
        # Diff inside the dialog compares against the on-disk text, so save
        # any pending edits or warn? We keep it simple: dialog always uses
        # disk as the comparison baseline. If user restores a version, it
        # replaces *editor* contents (becomes a pending edit until they Save).
        dlg = CaptionVersionsDialog(cp, self._disk_text, self)
        if dlg.exec() == QDialog.Accepted:
            restored = dlg.restored_text()
            if restored is not None:
                self._set_caption_text(restored)
                self._refresh_buttons()
                self._refresh_inline_diff()

    # ── navigation helpers ────────────────────────────────────

    def _row_for_path(self, cp: Path | None) -> int | None:
        if cp is None:
            return None
        for i, p in enumerate(self._images):
            if p.with_suffix(".txt") == cp:
                return i
        return None

    def _confirm_discard_if_dirty(self) -> bool:
        """Prompt to save if dirty. Returns False if the user cancels."""
        if not self._is_dirty():
            return True
        reply = QMessageBox.question(
            self,
            t("caption_unsaved_title"),
            t("caption_unsaved_body"),
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if reply == QMessageBox.Cancel:
            return False
        if reply == QMessageBox.Save:
            self._save()
            # If the save failed, _is_dirty() is still True — abort the switch.
            return not self._is_dirty()
        # Discard: drop edits silently.
        return True

    def _nav(self, d: int):
        r = self.fl.currentRow() + d
        if 0 <= r < self.fl.count():
            self.fl.setCurrentRow(r)
