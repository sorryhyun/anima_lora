"""Pick-round window for preference RL on the turbo student (proposal/pref_rl.md § 3.5, § 3.6).

    make exp-turbo-pref            # or: .venv/bin/python project/anima_turbo/pref_rl/app.py

One session = one policy and one label log under ``output/turbo_pref/<name>/``. The window
starts a render server (a daemon job, ``server.py``) that keeps the policy loaded and renders
the queued rounds; the window keeps ``PREFETCH`` rounds queued so the next grid is ready when
the current one is answered.

Picking: click a tile or press its key (rows ``1234`` / ``qwer`` / ``asdf`` / ``zxcv``), at
most ``max_picks``; zero is a valid answer. Right-click or Space over a tile opens it large.
Enter submits, Ctrl+K skips (logged, left out of every read).
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from PySide6.QtCore import QSize, Qt, QTimer, Signal  # noqa: E402
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QPixmap, QShortcut  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui import theme as gui_theme  # noqa: E402
from gui.jobs import daemon as gui_daemon  # noqa: E402
from gui.widgets import action_button, apply_variant  # noqa: E402

import store  # noqa: E402
from store import Caption, CaptionPicker, Session, SessionConfig  # noqa: E402

PREFETCH = 2  # rounds kept queued beyond the one on screen
SERVER_SCRIPT = "project/anima_turbo/pref_rl/server.py"
STALL_TIMEOUT = 900
KEYS = "1234qwerasdfzxcv"
COLS = 4


# ── tiles ────────────────────────────────────────────────────────────────


class Tile(QLabel):
    toggled = Signal(int)
    zoom = Signal(int)

    def __init__(self, slot: int):
        super().__init__()
        self.slot = slot
        self.picked_rank: int | None = None
        self.has_focus_ring = False
        self.readonly = False
        self.note = ""
        self._pix: QPixmap | None = None
        self._src: str | None = None
        self.setMinimumSize(120, 120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_image(self, path: Path | None) -> None:
        src = str(path) if path else None
        if src == self._src:
            return
        self._src = src
        self._pix = QPixmap(src) if src else None
        self.update()

    def clear(self) -> None:
        self.set_image(None)
        self.picked_rank = None
        self.note = ""
        self.update()

    def mousePressEvent(self, e):  # noqa: N802
        if (
            e.button() == Qt.MouseButton.LeftButton
            and self._pix is not None
            and not self.readonly
        ):
            self.toggled.emit(self.slot)
        elif e.button() == Qt.MouseButton.RightButton and self._pix is not None:
            self.zoom.emit(self.slot)

    def _scaled(self, size: QSize) -> QPixmap:
        key = (self._src, size.width(), size.height())
        if getattr(self, "_cache_key", None) != key:
            self._cache = self._pix.scaled(
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._cache_key = key
        return self._cache

    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        r = self.rect().adjusted(4, 4, -4, -4)
        p.fillRect(self.rect(), QColor(gui_theme.tok("base")))
        ir = r
        if self._pix is None:
            p.setPen(QColor(gui_theme.tok("text_dim")))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "rendering…")
        else:
            scaled = self._scaled(r.size())
            ir = scaled.rect().translated(
                r.x() + (r.width() - scaled.width()) // 2,
                r.y() + (r.height() - scaled.height()) // 2,
            )
            p.drawPixmap(ir.topLeft(), scaled)
            if self.readonly and self.picked_rank is None:
                p.fillRect(ir, QColor(0, 0, 0, 110))
        p.setBrush(Qt.BrushStyle.NoBrush)
        green = QColor(gui_theme.ACTION_COLORS["primary"])
        if self.picked_rank is not None:
            p.setPen(QPen(green, 5))
            p.drawRect(ir.adjusted(-2, -2, 2, 2))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(green)
            p.drawEllipse(ir.x() + 6, ir.y() + 6, 30, 30)
            p.setPen(QColor("white"))
            p.drawText(
                ir.x() + 6,
                ir.y() + 6,
                30,
                30,
                Qt.AlignmentFlag.AlignCenter,
                str(self.picked_rank + 1),
            )
        elif self.has_focus_ring and self._pix is not None:
            p.setPen(QPen(QColor(gui_theme.tok("accent")), 2))
            p.drawRect(ir.adjusted(-2, -2, 2, 2))
        # key hint (and blind arm in a past round), bottom-left of the image
        hint = KEYS[self.slot] + (f"  {self.note}" if self.note else "")
        p.fillRect(
            ir.x(), ir.bottom() - 21, 14 + 9 * len(hint), 22, QColor(0, 0, 0, 140)
        )
        p.setPen(QColor(255, 255, 255, 220))
        p.drawText(ir.x() + 7, ir.bottom() - 5, hint)
        p.end()

    def enterEvent(self, _e):  # noqa: N802
        self.has_focus_ring = True
        self.update()

    def leaveEvent(self, _e):  # noqa: N802
        self.has_focus_ring = False
        self.update()


class ZoomDialog(QDialog):
    """One render at full size. ←/→ step through the grid, P or click picks, Esc / Space close."""

    def __init__(self, win: "PrefWindow", slot: int):
        super().__init__(win)
        self.win = win
        self.slot = slot
        self.setWindowTitle("render")
        self.label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.status = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        lay = QVBoxLayout(self)
        lay.addWidget(self.label, 1)
        lay.addWidget(self.status)
        screen = self.screen().availableGeometry()
        self.resize(int(screen.width() * 0.7), int(screen.height() * 0.9))
        self._show()

    def _show(self) -> None:
        path = self.win.slot_path(self.slot)
        pix = QPixmap(str(path)) if path else QPixmap()
        avail = self.size() - QSize(24, 60)
        if not pix.isNull():
            pix = pix.scaled(
                avail,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.label.setPixmap(pix)
        picked = self.slot in self.win.picks
        self.status.setText(
            f"[{KEYS[self.slot]}] {self.slot + 1}/{self.win.n_slots}   "
            f"{'PICKED' if picked else 'not picked'}   ←/→ move · P pick · Esc close"
        )

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        self._show()

    def mousePressEvent(self, _e):  # noqa: N802
        self.win.toggle(self.slot)
        self._show()

    def keyPressEvent(self, e):  # noqa: N802
        k = e.key()
        if k in (Qt.Key.Key_Escape, Qt.Key.Key_Space):
            self.accept()
        elif k == Qt.Key.Key_Right:
            self.slot = (self.slot + 1) % self.win.n_slots
            self._show()
        elif k == Qt.Key.Key_Left:
            self.slot = (self.slot - 1) % self.win.n_slots
            self._show()
        elif k == Qt.Key.Key_P:
            self.win.toggle(self.slot)
            self._show()
        else:
            super().keyPressEvent(e)


# ── session dialog ───────────────────────────────────────────────────────


class SessionDialog(QDialog):
    def __init__(self, parent, cfg: SessionConfig, *, name: str | None):
        super().__init__(parent)
        self.setWindowTitle("New session" if name is None else f"Policy — {name}")
        form = QFormLayout(self)
        self.name = QLineEdit(time.strftime("pref_%m%d_%H%M"))
        if name is None:
            form.addRow("Name", self.name)
        self.dit = QLineEdit(cfg.dit)
        self.loras = QLineEdit(" ".join(cfg.loras))
        self.blind = QLineEdit(" ".join(cfg.blind_loras))
        self.blind.setPlaceholderText(
            "arm B LoRA(s) for blind grids; empty = no blind mode"
        )
        self.int8 = QCheckBox("int8 base")
        self.int8.setChecked(cfg.int8)
        self.compile = QCheckBox("compile blocks")
        self.compile.setChecked(cfg.compile)
        self.steps = QSpinBox(minimum=1, maximum=50, value=cfg.infer_steps)
        self.cfg = QDoubleSpinBox(
            minimum=0.0, maximum=20.0, singleStep=0.5, value=cfg.guidance_scale
        )
        self.shift = QDoubleSpinBox(
            minimum=0.1, maximum=20.0, singleStep=0.5, value=cfg.flow_shift
        )
        self.n = QSpinBox(minimum=4, maximum=32, singleStep=4, value=cfg.n_images)
        self.maxp = QSpinBox(minimum=1, maximum=32, value=cfg.max_picks)
        self.tier = QComboBox()
        for t in (768, 896, 1024, 1280):
            self.tier.addItem(str(t), t)
        self.tier.setCurrentIndex(self.tier.findData(cfg.tier))
        self.pool = QLineEdit(cfg.pool)
        self.pool.setToolTip(
            "A directory of captions (*.txt beside images) or a .txt with one caption per line"
        )
        self.rand = QDoubleSpinBox(
            minimum=0.0, maximum=1.0, singleStep=0.1, value=cfg.random_share
        )
        form.addRow("DiT", self.dit)
        form.addRow("LoRA(s), arm A", self.loras)
        form.addRow("LoRA(s), arm B", self.blind)
        row = QHBoxLayout()
        row.addWidget(self.int8)
        row.addWidget(self.compile)
        form.addRow("", row)
        form.addRow("Steps", self.steps)
        form.addRow("CFG", self.cfg)
        form.addRow("Shift", self.shift)
        form.addRow("Images / round", self.n)
        form.addRow("Max picks", self.maxp)
        form.addRow("Edge tier", self.tier)
        form.addRow("Caption pool", self.pool)
        form.addRow("Random share", self.rand)
        note = QLabel(
            "Policy edits apply when the server restarts; each round logs the policy it was rendered with."
        )
        note.setWordWrap(True)
        form.addRow(note)
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self.setMinimumWidth(640)

    def config(self, base: SessionConfig) -> SessionConfig:
        base.dit = self.dit.text().strip()
        base.loras = self.loras.text().split()
        base.blind_loras = self.blind.text().split()
        base.int8 = self.int8.isChecked()
        base.compile = self.compile.isChecked()
        base.infer_steps = self.steps.value()
        base.guidance_scale = self.cfg.value()
        base.flow_shift = self.shift.value()
        base.n_images = self.n.value()
        base.max_picks = self.maxp.value()
        base.tier = self.tier.currentData()
        base.pool = self.pool.text().strip()
        base.random_share = self.rand.value()
        return base


# ── main window ──────────────────────────────────────────────────────────


class PrefWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Turbo preference rounds")
        self.session: Session | None = None
        self.cfg: SessionConfig | None = None
        self.picker: CaptionPicker | None = None
        self.rng = random.Random()
        self.rid: str | None = None  # round on screen
        self.viewing_past: dict | None = None  # a picks.jsonl row shown read-only
        self.picks: list[int] = []
        self.job_id: str | None = None
        self.override: str | None = None
        self._build()
        self._load_sessions()
        self.timer = QTimer(self, interval=500, timeout=self._tick)
        self.timer.start()

    # ── layout ──

    def _build(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(
            gui_theme.Pad.SM, gui_theme.Pad.SM, gui_theme.Pad.SM, gui_theme.Pad.SM
        )

        bar = QHBoxLayout()
        self.session_box = QComboBox()
        self.session_box.setMinimumWidth(220)
        self.session_box.currentIndexChanged.connect(self._on_session_changed)
        bar.addWidget(QLabel("Session"))
        bar.addWidget(self.session_box)
        bar.addWidget(QPushButton("New…", clicked=self._new_session))
        self.policy_btn = QPushButton("Policy…", clicked=self._edit_policy)
        bar.addWidget(self.policy_btn)
        bar.addSpacing(16)
        bar.addWidget(QLabel("Mode"))
        self.mode_box = QComboBox()
        self.mode_box.addItem("Train rounds", "train")
        self.mode_box.addItem("Blind A/B", "blind")
        self.mode_box.currentIndexChanged.connect(self._on_mode_changed)
        bar.addWidget(self.mode_box)
        bar.addStretch(1)
        self.server_label = QLabel("server: off")
        bar.addWidget(self.server_label)
        self.server_btn = action_button(
            "Start server", variant="info", on_click=self._toggle_server
        )
        bar.addWidget(self.server_btn)
        outer.addLayout(bar)

        split = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(split, 1)

        # left: caption + grid + actions
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        cap_frame = QFrame()
        cap_frame.setFrameShape(QFrame.Shape.StyledPanel)
        cv = QVBoxLayout(cap_frame)
        cv.setContentsMargins(
            gui_theme.Pad.SM, gui_theme.Pad.XS, gui_theme.Pad.SM, gui_theme.Pad.XS
        )
        self.caption = QLabel("—")
        self.caption.setWordWrap(True)
        self.caption.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.caption_meta = QLabel("")
        self.caption_meta.setStyleSheet(f"color:{gui_theme.tok('text_dim')}")
        cv.addWidget(self.caption)
        cv.addWidget(self.caption_meta)
        lv.addWidget(cap_frame)

        grid_host = QWidget()
        self.grid = QGridLayout(grid_host)
        self.grid.setSpacing(gui_theme.Gap.TIGHT)
        self.tiles: list[Tile] = []
        lv.addWidget(grid_host, 1)

        act = QHBoxLayout()
        self.count_label = QLabel("")
        f = self.count_label.font()
        f.setPointSize(f.pointSize() + 3)
        f.setBold(True)
        self.count_label.setFont(f)
        act.addWidget(self.count_label)
        act.addStretch(1)
        self.back_btn = QPushButton(
            "Back to current round", clicked=self._back_to_current
        )
        self.back_btn.hide()
        act.addWidget(self.back_btn)
        act.addWidget(QPushButton("Next caption…", clicked=self._set_override))
        act.addWidget(QPushButton("Clear picks", clicked=self._clear_picks))
        self.skip_btn = action_button(
            "Skip (Ctrl+K)", variant="warning", on_click=self._skip
        )
        act.addWidget(self.skip_btn)
        self.submit_btn = action_button(
            "Submit (Enter)", variant="primary", on_click=self._submit
        )
        act.addWidget(self.submit_btn)
        lv.addLayout(act)
        split.addWidget(left)

        # right: reads + history + log
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.stats = QLabel("")
        self.stats.setWordWrap(True)
        self.stats.setTextFormat(Qt.TextFormat.RichText)
        self.stats.setAlignment(Qt.AlignmentFlag.AlignTop)
        rv.addWidget(self.stats)
        rv.addWidget(QLabel("Answered rounds"))
        self.history = QListWidget()
        self.history.itemClicked.connect(self._show_past)
        rv.addWidget(self.history, 2)
        rv.addWidget(QLabel("Server log"))
        self.log = QPlainTextEdit(readOnly=True)
        self.log.setMaximumBlockCount(400)
        rv.addWidget(self.log, 1)
        split.addWidget(right)
        split.setSizes([1200, 360])
        split.setStretchFactor(0, 1)

        self.setCentralWidget(root)

        for i, key in enumerate(KEYS):
            QShortcut(QKeySequence(key), self, activated=lambda i=i: self._key_slot(i))
        QShortcut(QKeySequence(Qt.Key.Key_Return), self, activated=self._submit)
        QShortcut(QKeySequence(Qt.Key.Key_Enter), self, activated=self._submit)
        QShortcut(QKeySequence("Ctrl+K"), self, activated=self._skip)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self._zoom_hovered)

    def _layout_tiles(self, n: int) -> None:
        if len(self.tiles) == n:
            return
        for t in self.tiles:
            t.setParent(None)
        self.tiles = []
        for i in range(n):
            t = Tile(i)
            t.toggled.connect(self.toggle)
            t.zoom.connect(self._zoom)
            self.grid.addWidget(t, i // COLS, i % COLS)
            self.tiles.append(t)

    @property
    def n_slots(self) -> int:
        return len(self.tiles)

    # ── sessions ──

    def _load_sessions(self, select: str | None = None) -> None:
        self.session_box.blockSignals(True)
        self.session_box.clear()
        for s in store.list_sessions():
            self.session_box.addItem(s.name, str(s.root))
        self.session_box.blockSignals(False)
        if select:
            self.session_box.setCurrentIndex(self.session_box.findText(select))
        self._on_session_changed()

    def _new_session(self) -> None:
        cfg = self.cfg or SessionConfig()
        dlg = SessionDialog(
            self, SessionConfig.from_dict(cfg.__dict__.copy()), name=None
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        name = dlg.name.text().strip()
        try:
            Session.create(name, dlg.config(SessionConfig()))
        except FileExistsError as e:
            QMessageBox.warning(self, "Session", str(e))
            return
        self._load_sessions(select=name)

    def _edit_policy(self) -> None:
        if not self.session:
            return
        dlg = SessionDialog(self, self.session.config(), name=self.session.name)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        cfg = dlg.config(self.session.config())
        store._write_json(self.session.root / "session.json", cfg.__dict__)
        self.cfg = cfg
        self._apply_mode_availability()
        if self._server_alive():
            self.log.appendPlainText("policy saved; restart the server to load it")

    def _on_session_changed(self) -> None:
        path = self.session_box.currentData()
        self.policy_btn.setEnabled(bool(path))
        if not path:
            self.session = self.cfg = self.picker = None
            self.caption.setText("Create a session (New…) to start.")
            self._layout_tiles(0)
            return
        self.session = Session(Path(path))
        self.cfg = self.session.config()
        self._layout_tiles(self.cfg.n_images)
        pool = store.load_pool(self.cfg.pool)
        root = store.resolve_under_home(self.cfg.pool)
        self.picker = CaptionPicker(pool, self.cfg.random_share)
        self._pool_root = root if root.is_dir() else None
        self.job_id = self._read_job_id()
        self.log.clear()
        self._log_offset = 0
        self._apply_mode_availability()
        self.log.appendPlainText(f"pool: {len(pool)} captions from {self.cfg.pool}")
        self._refresh_history()
        self._advance()

    def _apply_mode_availability(self) -> None:
        has_b = bool(self.cfg and self.cfg.blind_loras)
        item = self.mode_box.model().item(1)
        item.setEnabled(has_b)
        if not has_b and self.mode_box.currentData() == "blind":
            self.mode_box.setCurrentIndex(0)

    # ── queue ──

    def _mode(self) -> str:
        return self.mode_box.currentData()

    def _enqueue(self, caption_text: str | None = None) -> str:
        assert self.session and self.cfg and self.picker
        if caption_text is not None:
            text, size = store.parse_prompt_line(caption_text.replace("\n", " "))
            cap, how = Caption(text, "", size), "typed"
        else:
            used = {r.get("caption_src") for r in self.session.picks()}
            used |= {
                (self.session.request(r) or {}).get("caption_src")
                for r in self.session.unanswered_rids()
            }
            cap, how = self.picker.pick(used)
        if self._pool_root is not None and cap.size is None:
            cap.size = store.native_size(self._pool_root, cap)
        req = store.make_request(self.cfg, cap, self._mode(), how, self.rng)
        return self.session.write_request(req)

    def _top_up(self) -> None:
        if not self.session or not self.picker or not self.picker.pool:
            return
        while len(self.session.unanswered_rids()) < 1 + PREFETCH:
            self._enqueue()

    def _withdraw_queued(self) -> None:
        """Drop queued rounds the server has not started, beyond the one on screen."""
        for rid in self.session.unanswered_rids():
            if rid != self.rid:
                self.session.withdraw(rid)

    def _on_mode_changed(self) -> None:
        if not self.session:
            return
        self._withdraw_queued()
        self._top_up()
        self.log.appendPlainText(
            f"mode → {self._mode()}; queued rounds not yet started were redrawn"
        )

    def _set_override(self) -> None:
        if not self.session:
            return
        text, ok = QInputDialog.getMultiLineText(
            self, "Next caption", "Caption for the next round:"
        )
        if ok and text.strip():
            self._withdraw_queued()
            self._enqueue(text.strip())
            self._top_up()

    def _advance(self) -> None:
        self.viewing_past = None
        self.back_btn.hide()
        self._top_up()
        pending = self.session.unanswered_rids() if self.session else []
        self.rid = pending[0] if pending else None
        self.picks = []
        for t in self.tiles:
            t.clear()
            t.readonly = False
        self._render_round()

    # ── round view ──

    def _render_round(self) -> None:
        if self.viewing_past is not None:
            row = self.viewing_past
            self.caption.setText(row.get("caption") or "")
            pol = row.get("policy") or {}
            self.caption_meta.setText(
                f"round {row['rid']} · {row.get('kind')} · {row['n_picks']} picked"
                f"{' · skipped' if row.get('skipped') else ''} · {row.get('caption_src') or 'typed'}"
                f" · A={', '.join(Path(p).stem for p in pol.get('A', []))}"
            )
            for t in self.tiles:
                t.set_image(self.session.image_path(row["rid"], t.slot))
                t.picked_rank = (
                    row["picks"].index(t.slot) if t.slot in row["picks"] else None
                )
                t.readonly = True
                arms = row.get("arms") or []
                t.note = (
                    arms[t.slot]
                    if row.get("kind") == "blind" and t.slot < len(arms)
                    else ""
                )
                t.update()
            self._update_count()
            return
        if not self.rid:
            self.caption.setText("No round queued.")
            self.caption_meta.setText("")
            self._update_count()
            return
        req = self.session.request(self.rid) or {}
        self.caption.setText(req.get("caption", ""))
        h, w = req.get("size", [0, 0])
        self.caption_meta.setText(
            f"round {self.rid} · {req.get('kind')} · {w}×{h} · caption: {req.get('caption_pick')}"
            f" · {req.get('caption_src') or 'typed'}"
        )
        self._refresh_tiles()
        self._update_count()

    def _refresh_tiles(self) -> None:
        if not self.rid or self.viewing_past is not None:
            return
        slots = set(self.session.rendered_slots(self.rid))
        for t in self.tiles:
            t.set_image(
                self.session.image_path(self.rid, t.slot) if t.slot in slots else None
            )

    def slot_path(self, slot: int) -> Path | None:
        rid = self.viewing_past["rid"] if self.viewing_past else self.rid
        if not rid:
            return None
        p = self.session.image_path(rid, slot)
        return p if p.exists() else None

    def _round_ready(self) -> bool:
        return bool(self.rid) and self.session.done(self.rid) is not None

    def toggle(self, slot: int) -> None:
        if self.viewing_past is not None or not self.rid or slot >= self.n_slots:
            return
        if self.slot_path(slot) is None:
            return
        if slot in self.picks:
            self.picks.remove(slot)
        elif len(self.picks) >= self.cfg.max_picks:
            self.statusBar().showMessage(f"at most {self.cfg.max_picks} picks", 2000)
            return
        else:
            self.picks.append(slot)
        for t in self.tiles:
            t.picked_rank = self.picks.index(t.slot) if t.slot in self.picks else None
            t.update()
        self._update_count()

    def _key_slot(self, i: int) -> None:
        if QApplication.activeModalWidget() is None:
            self.toggle(i)

    def _zoom(self, slot: int) -> None:
        if slot < self.n_slots and self.slot_path(slot) is not None:
            ZoomDialog(self, slot).exec()
            if self.viewing_past is None:
                self.toggle_refresh()

    def toggle_refresh(self) -> None:
        for t in self.tiles:
            t.picked_rank = self.picks.index(t.slot) if t.slot in self.picks else None
            t.update()
        self._update_count()

    def _zoom_hovered(self) -> None:
        for t in self.tiles:
            if t.has_focus_ring:
                self._zoom(t.slot)
                return

    def _clear_picks(self) -> None:
        self.picks = []
        self.toggle_refresh()

    def _update_count(self) -> None:
        if self.viewing_past is not None:
            self.count_label.setText(
                f"{self.viewing_past['n_picks']} picked (read-only)"
            )
            self.submit_btn.setEnabled(False)
            self.skip_btn.setEnabled(False)
            return
        ready = self._round_ready()
        n_rendered = len(self.session.rendered_slots(self.rid)) if self.rid else 0
        maxp = self.cfg.max_picks if self.cfg else store.MAX_PICKS
        if self.rid and not ready:
            self.count_label.setText(
                f"rendering {n_rendered}/{self.n_slots} · {len(self.picks)} / {maxp} picked"
            )
        else:
            self.count_label.setText(f"{len(self.picks)} / {maxp} picked")
        self.submit_btn.setEnabled(ready)
        self.skip_btn.setEnabled(bool(self.rid))

    def _submit(self) -> None:
        if QApplication.activeModalWidget() is not None:
            return
        if self.viewing_past is not None or not self._round_ready():
            return
        self.session.answer(self.rid, self.picks)
        self._refresh_history()
        self._advance()

    def _skip(self) -> None:
        if self.viewing_past is not None or not self.rid:
            return
        note, ok = QInputDialog.getText(self, "Skip round", "Why (optional):")
        if not ok:
            return
        self.session.answer(self.rid, [], skipped=True, note=note)
        self._refresh_history()
        self._advance()

    # ── history + reads ──

    def _refresh_history(self) -> None:
        self.history.clear()
        rows = self.session.picks() if self.session else []
        for row in reversed(rows):
            mark = "skip" if row.get("skipped") else f"{row['n_picks']:>2}"
            kind = "B" if row.get("kind") == "blind" else " "
            it = QListWidgetItem(
                f"{row['rid']} {kind} [{mark}]  {(row.get('caption') or '')[:70]}"
            )
            it.setData(Qt.ItemDataRole.UserRole, row)
            self.history.addItem(it)
        self._refresh_stats(rows)

    def _refresh_stats(self, rows: list[dict]) -> None:
        if not self.session:
            self.stats.setText("")
            return
        tr = store.train_read(rows)
        hist = " ".join(f"{k}:{v}" for k, v in enumerate(tr["hist"]) if v)
        n_target = store.P1_ROUNDS
        bar = "█" * min(tr["rounds"], n_target) + "·" * max(0, n_target - tr["rounds"])
        lines = [
            f"<b>Train rounds</b> {tr['rounds']} / {n_target} (P1)<br><tt>{bar}</tt>",
            f"labelled {tr['labelled']} · picked {tr['picked']} · zero-pick rounds {tr['zero_rounds']}"
            f" · skipped {tr['skipped']}",
            f"picks per round: {hist or '—'}",
        ]
        if self.cfg and self.cfg.blind_loras:
            br = store.blind_read(rows)
            share = "—" if br["share_b"] is None else f"{br['share_b']:.3f}"
            p = "—" if br["p_sign"] is None else f"{br['p_sign']:.3f}"
            lines += [
                f"<br><b>Blind A/B</b> {br['grids']} grids (P2 gate: 20)",
                f"B share of picks {share} (A {br['picks_a']} · B {br['picks_b']})",
                f"grids B&gt;A {br['wins_b']} · A&gt;B {br['losses_b']} · sign p {p}",
                f"B = {', '.join(Path(x).stem for x in self.cfg.blind_loras)}",
            ]
        self.stats.setText("<br>".join(lines))

    def _show_past(self, item: QListWidgetItem) -> None:
        self.viewing_past = item.data(Qt.ItemDataRole.UserRole)
        self.back_btn.show()
        self._render_round()

    def _back_to_current(self) -> None:
        self.viewing_past = None
        self.back_btn.hide()
        self.history.clearSelection()
        for t in self.tiles:
            t.readonly = False
            t.note = ""
            t.set_image(None)
        self.toggle_refresh()
        self._render_round()

    # ── server ──

    def _job_file(self) -> Path:
        return self.session.root / "server_job"

    def _read_job_id(self) -> str | None:
        try:
            return self._job_file().read_text().strip() or None
        except FileNotFoundError:
            return None

    def _job_state(self) -> str | None:
        return gui_daemon.read_job_state(self.job_id) if self.job_id else None

    def _heartbeat_live(self) -> bool:
        return self.session.server_status().get("state") in (
            "loading",
            "ready",
            "rendering",
        )

    def _server_alive(self) -> bool:
        st = self._job_state()
        if st is not None and not gui_daemon.is_terminal(st):
            return True
        return (
            self._heartbeat_live()
        )  # a server started from the CLI has no job id here

    def _toggle_server(self) -> None:
        if not self.session:
            return
        if self._server_alive():
            self.session.stop_path.touch()
            if self.job_id and self._job_state() == "queued":
                gui_daemon.stop_job(self.job_id)
            self.log.appendPlainText("stop requested")
            return
        try:
            resp = gui_daemon.ensure_daemon().submit_command(
                label="turbo-pref",
                argv=[SERVER_SCRIPT, "--session", str(self.session.root)],
                stall_timeout=STALL_TIMEOUT,
                start=True,
            )
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Server", f"daemon submit failed: {e}")
            return
        self.job_id = resp.get("job_id")
        if self.job_id:
            self._job_file().write_text(self.job_id)
            self._log_offset = 0
            self.log.appendPlainText(f"server job {self.job_id} submitted")

    def _tail_log(self) -> None:
        if not self.job_id:
            return
        path = Path(gui_daemon.stdout_path(self.job_id))
        if not path.exists():
            return
        off = getattr(self, "_log_offset", 0)
        size = path.stat().st_size
        if size < off:
            off = 0
        if size == off:
            return
        with open(path, "rb") as f:
            f.seek(off)
            chunk = f.read().decode("utf-8", "replace")
        self._log_offset = size
        for line in chunk.splitlines():
            if (
                "[pref-server]" in line
                or "Error" in line
                or "Traceback" in line
                or "error" in line.lower()
            ):
                self.log.appendPlainText(line.rstrip())

    def _tick(self) -> None:
        if not self.session:
            return
        state = self._job_state()
        hb = self.session.server_status()
        if (state is None or gui_daemon.is_terminal(state)) and self._heartbeat_live():
            text, alive = f"server: {hb.get('state')} (started elsewhere)", True
        elif state is None:
            text, alive = "server: off", False
        elif gui_daemon.is_terminal(state):
            text, alive = f"server: {state}", False
        elif state == "queued":
            text, alive = "server: queued (daemon busy)", True
        else:
            text, alive = f"server: {hb.get('state', 'starting')}", True
            if hb.get("state") == "rendering":
                text += f" {hb.get('rid')}"
        self.server_label.setText(text)
        self.server_btn.setText("Stop server" if alive else "Start server")
        apply_variant(self.server_btn, "danger" if alive else "info")
        self._tail_log()
        if self.viewing_past is None and self.rid:
            self._refresh_tiles()
            self._update_count()


def main() -> None:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    gui_theme.apply_theme(app)
    win = PrefWindow()
    win.resize(1600, 1000)
    win.show()
    QTimer.singleShot(0, gui_daemon.ensure_daemon_quietly)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
