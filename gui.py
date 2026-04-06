"""Anima LoRA — PySide6 GUI for config editing, GRAFT curation, and dataset browsing."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import toml
from PySide6.QtCore import QProcess, Qt, Signal
from PySide6.QtGui import QFont, QKeySequence, QPixmap, QPalette, QColor, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui_i18n import t, load_language, save_language, available_languages, current_language

ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = ROOT / "configs"
GRAFT_DIR = ROOT / "graft"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

PRESETS = {
    "Windows 8GB VRAM": "training_config_win8gb.toml",
    "Windows 16GB VRAM": "training_config_win16gb.toml",
    "FA4 8GB VRAM": "training_config_fa4_8gb.toml",
    "FA4 16GB VRAM": "training_config_fa4_16gb.toml",
}

_GROUPS = {
    "Architecture": {
        "network_dim", "network_alpha", "network_module", "use_dora",
        "use_timestep_mask", "min_rank", "alpha_rank_scale", "network_train_unet_only",
    },
    "Training": {
        "learning_rate", "max_train_epochs", "save_every_n_epochs",
        "checkpointing_epochs", "gradient_accumulation_steps",
        "caption_shuffle_variants", "optimizer_type", "lr_scheduler",
        "timestep_sampling", "discrete_flow_shift",
    },
    "Performance": {
        "attn_mode", "gradient_checkpointing", "unsloth_offload_checkpointing",
        "blocks_to_swap", "torch_compile", "trim_crossattn_kv",
        "cache_llm_adapter_outputs", "masked_loss", "mixed_precision",
        "lora_fp32_accumulation", "static_token_count", "vae_chunk_size",
        "vae_disable_cache", "cache_latents", "cache_latents_to_disk",
        "cache_text_encoder_outputs", "cache_text_encoder_outputs_to_disk",
        "skip_cache_check",
    },
    "Paths": {
        "pretrained_model_name_or_path", "qwen3", "vae",
        "output_dir", "output_name", "save_model_as",
    },
}
_K2G = {k: g for g, ks in _GROUPS.items() for k in ks}
_SKIP = {"base_config", "dataset_config"}

# Presets that lock the Performance group (users should not tweak these)
_LOCKED_PERFORMANCE = {
    "Windows 8GB VRAM",
    "Windows 16GB VRAM",
    "FA4 8GB VRAM",
    "FA4 16GB VRAM",
}


# ── Helpers ────────────────────────────────────────────────────


def _load(p: Path) -> dict:
    return toml.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save(p: Path, d: dict):
    p.write_text(toml.dumps(d), encoding="utf-8")


def _merged(f: str) -> dict:
    v = _load(CONFIGS_DIR / f)
    b = v.pop("base_config", None)
    base = _load(CONFIGS_DIR / b) if b else {}
    base.update(v)
    return base


def _imgs(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS) if d.exists() else []


_ATTN_MODES = ["flex", "flash"] if sys.platform == "win32" else ["flex", "flash", "flash4"]


def _widget(v: Any, key: str = "") -> QWidget:
    if key == "attn_mode":
        w = QComboBox()
        w.addItems(_ATTN_MODES)
        idx = w.findText(str(v))
        if idx >= 0:
            w.setCurrentIndex(idx)
        return w
    if isinstance(v, bool):
        w = QCheckBox()
        w.setChecked(v)
        return w
    if isinstance(v, int):
        w = QSpinBox()
        w.setRange(0, 10000)
        w.setValue(v)
        w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        w.wheelEvent = lambda e: e.ignore()
        return w
    if isinstance(v, float):
        return QLineEdit(f"{v:g}")
    if isinstance(v, list):
        return QLineEdit(json.dumps(v))
    return QLineEdit(str(v))


def _read(w: QWidget, orig: Any = None) -> Any:
    if isinstance(w, QComboBox):
        return w.currentText()
    if isinstance(w, QCheckBox):
        return w.isChecked()
    if isinstance(w, QSpinBox):
        return w.value()
    t = w.text()
    if isinstance(orig, float):
        try:
            return float(t)
        except ValueError:
            pass
    if isinstance(orig, list):
        try:
            return json.loads(t)
        except (json.JSONDecodeError, ValueError):
            pass
    return t


def _image_dirs() -> dict[str, Path]:
    dirs: dict[str, Path] = {}
    for name, path in [
        ("image_dataset", ROOT / "image_dataset"),
        ("post_image_dataset", ROOT / "post_image_dataset"),
        ("graft/survivors", GRAFT_DIR / "survivors"),
    ]:
        if path.exists():
            dirs[name] = path
    cd = GRAFT_DIR / "candidates"
    if cd.exists():
        for p in sorted(cd.iterdir()):
            if p.is_dir():
                dirs[f"graft/candidates/{p.name}"] = p
    return dirs


# ── ScaledImageLabel ───────────────────────────────────────────


class ScaledImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self._src: QPixmap | None = None
        self.setAlignment(Qt.AlignCenter)

    def set_source(self, pm: QPixmap):
        self._src = pm
        self._rescale()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._rescale()

    def _rescale(self):
        if self._src and not self._src.isNull():
            self.setPixmap(
                self._src.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )


# ── Thumbnail ──────────────────────────────────────────────────

THUMB = 220


class Thumbnail(QLabel):
    clicked = Signal(object)

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.selected = False
        self.setFixedSize(THUMB, THUMB)
        self.setAlignment(Qt.AlignCenter)
        pm = QPixmap(str(path))
        if not pm.isNull():
            self.setPixmap(pm.scaled(THUMB, THUMB, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.setToolTip(path.name)
        self._style()

    def toggle(self):
        self.selected = not self.selected
        self._style()

    def _style(self):
        c = "#e74c3c" if self.selected else "transparent"
        self.setStyleSheet(f"border:3px solid {c};background:#1e1e1e;")

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.toggle()
            self.clicked.emit(self)


# ── ConfigTab ──────────────────────────────────────────────────


class ConfigTab(QWidget):
    def __init__(self):
        super().__init__()
        self._w: dict[str, QWidget] = {}
        self._vkeys: set[str] = set()
        self._ds_edit: QPlainTextEdit | None = None
        self._preprocessed = (ROOT / "post_image_dataset").exists()
        lay = QVBoxLayout(self)

        # Top bar: preset + save + preprocess + train + stop
        top = QHBoxLayout()
        top.addWidget(QLabel(t("preset")))
        self.combo = QComboBox()
        self.combo.addItems(PRESETS)
        self.combo.currentTextChanged.connect(self._load_preset)
        top.addWidget(self.combo, 1)

        save_btn = QPushButton(t("save"))
        save_btn.clicked.connect(self._save_preset)
        top.addWidget(save_btn)

        self.preprocess_btn = QPushButton(t("preprocess"))
        self._preprocess_idle_style = "background:#2980b9;color:white;font-weight:bold;padding:4px 16px;"
        self._preprocess_busy_style = "background:#7f8c8d;color:white;font-weight:bold;padding:4px 16px;"
        self.preprocess_btn.setStyleSheet(self._preprocess_idle_style)
        self.preprocess_btn.clicked.connect(self._start_preprocess)
        top.addWidget(self.preprocess_btn)

        self.train_btn = QPushButton(t("train"))
        self._train_idle_style = "background:#27ae60;color:white;font-weight:bold;padding:4px 16px;"
        self._train_busy_style = "background:#7f8c8d;color:white;font-weight:bold;padding:4px 16px;"
        self.train_btn.setStyleSheet(self._train_idle_style)
        self.train_btn.clicked.connect(self._start_training)
        self.train_btn.setEnabled(self._preprocessed)
        top.addWidget(self.train_btn)

        self.stop_btn = QPushButton(t("stop"))
        self.stop_btn.setStyleSheet(
            "background:#c0392b;color:white;font-weight:bold;padding:4px 16px;"
        )
        self.stop_btn.clicked.connect(self._stop_training)
        self.stop_btn.setEnabled(False)
        top.addWidget(self.stop_btn)

        lay.addLayout(top)

        # Vertical splitter: config form on top, log on bottom
        vsplit = QSplitter(Qt.Vertical)

        sc = QScrollArea()
        sc.setWidgetResizable(True)
        self._form = QWidget()
        self._fl = QVBoxLayout(self._form)
        sc.setWidget(self._form)
        vsplit.addWidget(sc)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family:monospace;font-size:11px;")
        self.log.setPlaceholderText(t("log_placeholder"))
        vsplit.addWidget(self.log)

        vsplit.setSizes([500, 200])
        lay.addWidget(vsplit)

        # QProcess for training
        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(str(ROOT))
        self._proc.readyReadStandardOutput.connect(self._read_stdout)
        self._proc.readyReadStandardError.connect(self._read_stderr)
        self._proc.finished.connect(self._on_finished)

        self._load_preset(self.combo.currentText())

    def _load_preset(self, name: str):
        f = PRESETS[name]
        var = _load(CONFIGS_DIR / f)
        self._vkeys = set(var) - _SKIP
        cfg = {k: v for k, v in _merged(f).items() if k not in _SKIP}

        self._w.clear()
        self._ds_edit = None
        while self._fl.count():
            it = self._fl.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

        # Grouped training config fields
        groups: dict[str, dict] = {g: {} for g in _GROUPS}
        groups["Other"] = {}
        for k, v in cfg.items():
            groups.setdefault(_K2G.get(k, "Other"), {})[k] = v

        lock_perf = name in _LOCKED_PERFORMANCE

        for gn, flds in groups.items():
            if not flds:
                continue
            locked = lock_perf and gn == "Performance"
            box = QGroupBox(gn)
            if locked:
                box.setToolTip(t("locked_by_preset"))
            form = QFormLayout()
            for k in sorted(flds):
                w = _widget(flds[k], key=k)
                if locked:
                    w.setEnabled(False)
                self._w[k] = w
                lbl = QLabel(k)
                if locked:
                    lbl.setStyleSheet("color:#666;")
                elif k not in self._vkeys:
                    lbl.setStyleSheet("color:#888;")
                    lbl.setToolTip(t("from_base"))
                form.addRow(lbl, w)
            box.setLayout(form)
            self._fl.addWidget(box)

        # Dataset config (raw TOML editor)
        ds_path = CONFIGS_DIR / "dataset_config.toml"
        if ds_path.exists():
            box = QGroupBox(t("dataset_config"))
            bl = QVBoxLayout()
            ds_edit = QPlainTextEdit(ds_path.read_text(encoding="utf-8"))
            ds_edit.setStyleSheet("font-family:monospace;")
            ds_edit.setMaximumHeight(180)
            bl.addWidget(ds_edit)
            self._ds_edit = ds_edit
            dsb = QPushButton(t("save_dataset_config"))
            dsb.clicked.connect(self._save_ds)
            bl.addWidget(dsb)
            box.setLayout(bl)
            self._fl.addWidget(box)

        self._fl.addStretch()

    # ── Save ──

    def _build_save_data(self) -> tuple[Path, dict[str, Any]]:
        f = PRESETS[self.combo.currentText()]
        p = CONFIGS_DIR / f
        orig = _load(p)
        bf = orig.get("base_config")
        base = _load(CONFIGS_DIR / bf) if bf else {}
        merged = {**base, **orig}

        out: dict[str, Any] = {}
        if bf:
            out["base_config"] = bf
        ds = orig.get("dataset_config")
        if ds:
            out["dataset_config"] = ds

        for k, w in self._w.items():
            v = _read(w, merged.get(k))
            if k in self._vkeys or (k in base and base[k] != v):
                out[k] = v
        return p, out

    def _save_preset(self):
        p, out = self._build_save_data()
        _save(p, out)
        QMessageBox.information(self, t("saved"), t("saved_file", name=p.name))

    def _save_ds(self):
        if not self._ds_edit:
            return
        p = CONFIGS_DIR / "dataset_config.toml"
        text = self._ds_edit.toPlainText()
        try:
            toml.loads(text)
        except toml.TomlDecodeError as e:
            QMessageBox.warning(self, t("invalid_toml"), str(e))
            return
        p.write_text(text, encoding="utf-8")
        QMessageBox.information(self, t("saved"), t("dataset_saved"))

    # ── Training ──

    def _start_preprocess(self):
        python = sys.executable
        args = [
            "scripts/post_images.py",
            "--src", "image_dataset",
            "--dst", "post_image_dataset",
            "--vae", "models/vae/qwen_image_vae.safetensors",
            "--vae_batch_size", "4",
            "--vae_chunk_size", "64",
        ]

        self.log.clear()
        self._log(f"> python {' '.join(args)}\n")
        self._running_mode = "preprocess"
        self._proc.start(python, args)
        self.preprocess_btn.setText(t("preprocess") + " ...")
        self.preprocess_btn.setStyleSheet(self._preprocess_busy_style)
        self.preprocess_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.combo.setEnabled(False)

    def _start_training(self):
        if not self._preprocessed:
            QMessageBox.warning(self, t("error"), t("preprocess_required"))
            return

        # Auto-save config before training
        p, out = self._build_save_data()
        _save(p, out)

        accelerate = shutil.which("accelerate")
        if not accelerate:
            QMessageBox.warning(self, t("error"), t("accelerate_not_found"))
            return

        f = PRESETS[self.combo.currentText()]
        args = [
            "launch",
            "--num_cpu_threads_per_process", "3",
            "--mixed_precision", "bf16",
            "train.py",
            "--config_file", f"configs/{f}",
        ]

        self.log.clear()
        self._log(f"> accelerate {' '.join(args)}\n")
        self._running_mode = "train"
        self._proc.start(accelerate, args)
        self.train_btn.setText(t("train") + " ...")
        self.train_btn.setStyleSheet(self._train_busy_style)
        self.train_btn.setEnabled(False)
        self.preprocess_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.combo.setEnabled(False)

    def _stop_training(self):
        if self._proc.state() != QProcess.NotRunning:
            self._proc.kill()

    def _read_stdout(self):
        data = self._proc.readAllStandardOutput().data().decode(errors="replace")
        self._log(data)

    def _read_stderr(self):
        data = self._proc.readAllStandardError().data().decode(errors="replace")
        self._log(data)

    def _on_finished(self, exit_code: int, _status: QProcess.ExitStatus):
        self._log(f"\n{t('finished', code=exit_code)}\n")
        mode = getattr(self, "_running_mode", "train")
        if mode == "preprocess" and exit_code == 0:
            self._preprocessed = True
        self.preprocess_btn.setText(t("preprocess"))
        self.preprocess_btn.setStyleSheet(self._preprocess_idle_style)
        self.preprocess_btn.setEnabled(True)
        self.train_btn.setText(t("train"))
        self.train_btn.setStyleSheet(self._train_idle_style)
        self.train_btn.setEnabled(self._preprocessed)
        self.stop_btn.setEnabled(False)
        self.combo.setEnabled(True)

    def _log(self, text: str):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)


# ── GraftTab ───────────────────────────────────────────────────


class GraftTab(QWidget):
    def __init__(self):
        super().__init__()
        self.thumbs: list[Thumbnail] = []
        lay = QVBoxLayout(self)
        sp = QSplitter(Qt.Horizontal)

        # Left panel: iteration list + GRAFT config
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)

        hl = QHBoxLayout()
        hl.addWidget(QLabel(t("iterations")))
        ref = QPushButton("\u21bb")
        ref.setFixedWidth(28)
        ref.setToolTip(t("refresh"))
        ref.clicked.connect(self._refresh)
        hl.addWidget(ref)
        ll.addLayout(hl)

        self.il = QListWidget()
        self.il.currentTextChanged.connect(self._load_iter)
        ll.addWidget(self.il)

        gc = _load(GRAFT_DIR / "graft_config.toml")
        self._gcw: dict[str, QWidget] = {}
        if gc:
            box = QGroupBox(t("graft_config"))
            form = QFormLayout()
            for k, v in gc.items():
                w = _widget(v)
                self._gcw[k] = w
                form.addRow(k, w)
            box.setLayout(form)
            ll.addWidget(box)
            sb = QPushButton(t("save_graft_config"))
            sb.clicked.connect(self._save_gc)
            ll.addWidget(sb)

        sp.addWidget(left)

        # Right panel: toolbar + thumbnail grid + preview
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        bar = QHBoxLayout()
        for lbl, fn in [
            (t("select_all"), self._sel_all),
            (t("invert"), self._inv),
            (t("deselect"), self._desel),
        ]:
            b = QPushButton(lbl)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addStretch()
        self.stat = QLabel(t("n_images", n=0))
        bar.addWidget(self.stat)
        db = QPushButton(t("delete_selected"))
        db.setStyleSheet("background:#c0392b;color:white;font-weight:bold;padding:4px 12px;")
        db.clicked.connect(self._delete)
        bar.addWidget(db)
        rl.addLayout(bar)

        vs = QSplitter(Qt.Vertical)

        sc = QScrollArea()
        sc.setWidgetResizable(True)
        self._gw = QWidget()
        self._gl = QGridLayout(self._gw)
        self._gl.setSpacing(8)
        sc.setWidget(self._gw)
        vs.addWidget(sc)

        pw = QWidget()
        pl = QVBoxLayout(pw)
        pl.setContentsMargins(0, 0, 0, 0)
        self.prev = ScaledImageLabel()
        self.prev.setMinimumHeight(200)
        self.prev.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        pl.addWidget(self.prev)
        self.prev_info = QLabel()
        self.prev_info.setWordWrap(True)
        pl.addWidget(self.prev_info)
        vs.addWidget(pw)
        vs.setSizes([450, 300])

        rl.addWidget(vs)
        sp.addWidget(right)
        sp.setSizes([250, 750])
        lay.addWidget(sp)

        QShortcut(QKeySequence.Delete, self, self._delete)
        QShortcut(QKeySequence("Ctrl+A"), self, self._sel_all)
        self._refresh()

    def _refresh(self):
        self.il.clear()
        cd = GRAFT_DIR / "candidates"
        if not cd.exists():
            return
        ds = sorted(d.name for d in cd.iterdir() if d.is_dir())
        self.il.addItems(ds)
        if ds:
            self.il.setCurrentRow(len(ds) - 1)

    def _load_iter(self, name: str):
        self.thumbs.clear()
        while self._gl.count():
            w = self._gl.takeAt(0).widget()
            if w:
                w.deleteLater()
        if not name:
            return
        for i, p in enumerate(_imgs(GRAFT_DIR / "candidates" / name)):
            t = Thumbnail(p)
            t.clicked.connect(self._on_click)
            self.thumbs.append(t)
            self._gl.addWidget(t, i // 4, i % 4)
        self._upd()

    def _on_click(self, thumb: Thumbnail):
        self._upd()
        pm = QPixmap(str(thumb.path))
        if not pm.isNull():
            self.prev.set_source(pm)
        info = thumb.path.name
        jp = thumb.path.with_suffix(".json")
        if jp.exists():
            try:
                m = json.loads(jp.read_text(encoding="utf-8"))
                info += f"  |  seed: {m.get('seed', '?')}"
                cap = m.get("caption", "")
                if cap:
                    info += f"\n{cap[:200]}"
            except (json.JSONDecodeError, OSError):
                pass
        self.prev_info.setText(info)

    def _upd(self):
        n = len(self.thumbs)
        s = sum(t.selected for t in self.thumbs)
        self.stat.setText(t("n_images_selected", n=n, s=s))

    def _sel_all(self):
        for t in self.thumbs:
            if not t.selected:
                t.toggle()
        self._upd()

    def _desel(self):
        for t in self.thumbs:
            if t.selected:
                t.toggle()
        self._upd()

    def _inv(self):
        for t in self.thumbs:
            t.toggle()
        self._upd()

    def _delete(self):
        sel = [t for t in self.thumbs if t.selected]
        if not sel:
            return
        if (
            QMessageBox.question(
                self,
                t("delete"),
                t("delete_confirm", n=len(sel)),
                QMessageBox.Yes | QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return
        for t in sel:
            t.path.unlink(missing_ok=True)
            t.path.with_suffix(".json").unlink(missing_ok=True)
        cur = self.il.currentItem()
        if cur:
            self._load_iter(cur.text())

    def _save_gc(self):
        gc = _load(GRAFT_DIR / "graft_config.toml")
        _save(GRAFT_DIR / "graft_config.toml", {k: _read(w, gc.get(k)) for k, w in self._gcw.items()})
        QMessageBox.information(self, t("saved"), t("graft_saved"))


# ── ImageViewerTab ─────────────────────────────────────────────


class ImageViewerTab(QWidget):
    def __init__(self):
        super().__init__()
        self._images: list[Path] = []
        self._dirs = _image_dirs()
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
        self.fl.currentRowChanged.connect(self._show)
        sp.addWidget(self.fl)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.img = ScaledImageLabel()
        self.img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img.setMinimumSize(400, 400)
        rl.addWidget(self.img, 1)
        rl.addWidget(QLabel(t("caption")))
        self.cap = QTextEdit()
        self.cap.setReadOnly(True)
        self.cap.setMaximumHeight(120)
        rl.addWidget(self.cap)
        sp.addWidget(right)
        sp.setSizes([220, 750])
        lay.addWidget(sp)

        QShortcut(QKeySequence("Right"), self, lambda: self._nav(1))
        QShortcut(QKeySequence("Left"), self, lambda: self._nav(-1))
        if self._dirs:
            self._load_dir(self.dc.currentText())

    def _load_dir(self, name: str):
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

    def _show(self, row: int):
        if not 0 <= row < len(self._images):
            return
        p = self._images[row]
        pm = QPixmap(str(p))
        if not pm.isNull():
            self.img.set_source(pm)
        cp = p.with_suffix(".txt")
        self.cap.setPlainText(cp.read_text(encoding="utf-8") if cp.exists() else t("no_caption"))

    def _nav(self, d: int):
        r = self.fl.currentRow() + d
        if 0 <= r < self.fl.count():
            self.fl.setCurrentRow(r)


# ── Dark Theme ─────────────────────────────────────────────────


def _dark(app: QApplication):
    # Use a font that supports Korean glyphs on Windows
    font = QFont("Malgun Gothic", 9)
    font.setStyleHint(QFont.SansSerif)
    app.setFont(font)

    p = QPalette()
    for role, color in [
        (QPalette.Window, QColor(30, 30, 30)),
        (QPalette.WindowText, QColor(220, 220, 220)),
        (QPalette.Base, QColor(25, 25, 25)),
        (QPalette.AlternateBase, QColor(35, 35, 35)),
        (QPalette.ToolTipBase, QColor(50, 50, 50)),
        (QPalette.ToolTipText, QColor(220, 220, 220)),
        (QPalette.Text, QColor(220, 220, 220)),
        (QPalette.Button, QColor(45, 45, 45)),
        (QPalette.ButtonText, QColor(220, 220, 220)),
        (QPalette.Highlight, QColor(60, 120, 200)),
        (QPalette.HighlightedText, QColor(255, 255, 255)),
    ]:
        p.setColor(role, color)
    app.setPalette(p)
    app.setStyleSheet("""
        QGroupBox {
            font-weight: bold; border: 1px solid #444;
            border-radius: 4px; margin-top: 8px; padding-top: 16px;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        QPushButton { padding: 4px 12px; border: 1px solid #555; border-radius: 3px; }
        QPushButton:hover { background: #555; }
        QScrollArea { border: none; }
        QSplitter::handle { background: #444; }
        QLineEdit, QSpinBox, QComboBox, QPlainTextEdit, QTextEdit, QListWidget {
            background: #2a2a2a; color: #dcdcdc; border: 1px solid #555; border-radius: 3px;
            padding: 2px 4px;
        }
        QComboBox QAbstractItemView {
            background: #2a2a2a; color: #dcdcdc; selection-background-color: #3c78c8;
        }
        QTabWidget::pane { border: 1px solid #444; }
        QTabBar::tab {
            background: #2a2a2a; color: #dcdcdc; border: 1px solid #444;
            padding: 4px 12px; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px;
        }
        QTabBar::tab:selected { background: #1e1e1e; }
        QTabBar::tab:hover { background: #3a3a3a; }
    """)


# ── Main ───────────────────────────────────────────────────────


LANG_NAMES = {"en": "English", "ko": "\ud55c\uad6d\uc5b4"}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(t("window_title"))
        self.resize(1100, 750)

        central = QWidget()
        main_lay = QVBoxLayout(central)
        main_lay.setContentsMargins(0, 0, 0, 0)

        # Language selector bar
        lang_bar = QHBoxLayout()
        lang_bar.addStretch()
        lang_bar.addWidget(QLabel(t("language")))
        self.lang_combo = QComboBox()
        for code in available_languages():
            self.lang_combo.addItem(LANG_NAMES.get(code, code), code)
        self.lang_combo.setCurrentIndex(
            available_languages().index(current_language())
        )
        self.lang_combo.currentIndexChanged.connect(self._change_lang)
        self.lang_combo.setFixedWidth(100)
        lang_bar.addWidget(self.lang_combo)
        main_lay.addLayout(lang_bar)

        self.tabs = QTabWidget()
        self.tabs.addTab(ConfigTab(), t("tab_config"))
        self.tabs.addTab(GraftTab(), t("tab_graft"))
        self.tabs.addTab(ImageViewerTab(), t("tab_images"))
        main_lay.addWidget(self.tabs)
        self.setCentralWidget(central)

    def _change_lang(self, idx: int):
        lang = self.lang_combo.itemData(idx)
        save_language(lang)
        QMessageBox.information(
            self,
            "Language" if lang == "en" else "\uc5b8\uc5b4",
            "Please restart the application to apply the language change."
            if current_language() == "en"
            else "\uc5b8\uc5b4 \ubcc0\uacbd\uc744 \uc801\uc6a9\ud558\ub824\uba74 \uc571\uc744 \ub2e4\uc2dc \uc2dc\uc791\ud574\uc8fc\uc138\uc694.",
        )


def main():
    load_language()
    app = QApplication(sys.argv)
    _dark(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
