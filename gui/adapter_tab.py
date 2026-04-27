"""IP-Adapter / EasyControl tabs.

Each tab is a slim launcher: it shows the dataset state (source image count,
caption pairing, cache coverage), and gives the user one button to preprocess
the dataset and one button to train the adapter. Heavy config editing still
happens in the Config tab (these methods are listed there as variants).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui import ROOT, ScaledImageLabel, _imgs
from gui.i18n import t

LATENT_SUFFIX = "_anima.npz"
TE_SUFFIX = "_anima_te.safetensors"
PE_SUFFIX = "_anima_pe.safetensors"


def _stem(path: Path) -> str:
    return path.stem


def _count_caches(cache_dir: Path, stems: set[str], require_pe: bool) -> dict[str, int]:
    """Count how many of the given stems have each cache type present."""
    if not cache_dir.exists():
        return {"latents": 0, "te": 0, "pe": 0}
    have_lat: set[str] = set()
    have_te: set[str] = set()
    have_pe: set[str] = set()
    for p in cache_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if name.endswith(TE_SUFFIX):
            have_te.add(name.removesuffix(TE_SUFFIX))
        elif name.endswith(PE_SUFFIX):
            have_pe.add(name.removesuffix(PE_SUFFIX))
        elif name.endswith(LATENT_SUFFIX):
            stripped = name.removesuffix(LATENT_SUFFIX)
            # strip trailing "_WxH"
            parts = stripped.rsplit("_", 1)
            have_lat.add(parts[0] if len(parts) >= 2 else stripped)
    out = {
        "latents": len(have_lat & stems),
        "te": len(have_te & stems),
    }
    if require_pe:
        out["pe"] = len(have_pe & stems)
    return out


class _AdapterTab(QWidget):
    """Shared layout for IP-Adapter / EasyControl launcher tabs."""

    SOURCE_DIR: str = ""        # e.g. "ip-adapter-dataset"
    CACHE_DIR: str = ""         # e.g. "post_image_dataset/ip-adapter"
    PREPROCESS_TASK: str = ""   # tasks.py target, e.g. "ip-adapter-preprocess"
    TRAIN_TASK: str = ""        # tasks.py target, e.g. "ip-adapter"
    REQUIRE_PE: bool = False
    METHOD_LABEL: str = ""

    def __init__(self):
        super().__init__()
        self._source = ROOT / self.SOURCE_DIR
        self._cache = ROOT / self.CACHE_DIR
        self._images: list[Path] = []

        lay = QVBoxLayout(self)

        # ── Header: paths + status + open buttons ──────────────────
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel(t("adapter_source_dir")))
        self.src_lbl = QLabel(str(self._source))
        self.src_lbl.setStyleSheet("color:#dcdcdc;")
        path_row.addWidget(self.src_lbl, 1)
        open_src = QPushButton(t("adapter_open_dir"))
        open_src.clicked.connect(lambda: self._open(self._source))
        path_row.addWidget(open_src)
        lay.addLayout(path_row)

        cache_row = QHBoxLayout()
        cache_row.addWidget(QLabel(t("adapter_cache_dir")))
        self.cache_lbl = QLabel(str(self._cache))
        self.cache_lbl.setStyleSheet("color:#dcdcdc;")
        cache_row.addWidget(self.cache_lbl, 1)
        open_cache = QPushButton(t("adapter_open_dir"))
        open_cache.clicked.connect(lambda: self._open(self._cache))
        cache_row.addWidget(open_cache)
        lay.addLayout(cache_row)

        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color:#bbbbbb;padding:4px 0;")
        lay.addWidget(self.status_lbl)

        # ── Action bar: refresh / preprocess / train / stop ────────
        bar = QHBoxLayout()
        refresh = QPushButton("↻")
        refresh.setFixedWidth(28)
        refresh.clicked.connect(self._reload)
        bar.addWidget(refresh)

        preprocess_label = (
            t("adapter_preprocess_pe") if self.REQUIRE_PE else t("adapter_preprocess")
        )
        self.preprocess_btn = QPushButton(preprocess_label)
        self.preprocess_btn.setStyleSheet(
            "background:#2980b9;color:white;font-weight:bold;padding:4px 16px;"
        )
        self.preprocess_btn.clicked.connect(self._start_preprocess)
        bar.addWidget(self.preprocess_btn)

        self.train_btn = QPushButton(t("adapter_train"))
        self.train_btn.setStyleSheet(
            "background:#27ae60;color:white;font-weight:bold;padding:4px 16px;"
        )
        self.train_btn.clicked.connect(self._start_train)
        bar.addWidget(self.train_btn)

        self.stop_btn = QPushButton(t("adapter_stop"))
        self.stop_btn.setStyleSheet(
            "background:#c0392b;color:white;font-weight:bold;padding:4px 16px;"
        )
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        bar.addWidget(self.stop_btn)

        bar.addStretch()
        lay.addLayout(bar)

        # ── Body: dataset browser on top, log on bottom ────────────
        vsplit = QSplitter(Qt.Vertical)

        browse = QSplitter(Qt.Horizontal)
        self.fl = QListWidget()
        self.fl.currentRowChanged.connect(self._show)
        browse.addWidget(self.fl)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.img = ScaledImageLabel()
        self.img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img.setMinimumSize(320, 320)
        rl.addWidget(self.img, 1)
        rl.addWidget(QLabel(t("caption")))
        self.cap = QTextEdit()
        self.cap.setReadOnly(True)
        self.cap.setMaximumHeight(120)
        rl.addWidget(self.cap)
        browse.addWidget(right)
        browse.setSizes([220, 750])
        vsplit.addWidget(browse)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family:monospace;font-size:11px;")
        self.log.setPlaceholderText(t("adapter_log_placeholder"))
        vsplit.addWidget(self.log)
        vsplit.setSizes([460, 240])
        lay.addWidget(vsplit, 1)

        # ── Subprocess wiring ─────────────────────────────────────
        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(str(ROOT))
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._read_stdout)
        self._proc.finished.connect(self._on_finished)
        self._buf = ""

        self._reload()

    # ── Dataset state ──────────────────────────────────────────────

    def _reload(self):
        self._images = _imgs(self._source) if self._source.exists() else []
        self.fl.clear()
        for p in self._images:
            self.fl.addItem(QListWidgetItem(p.stem))
        if not self._source.exists():
            self.status_lbl.setText(t("adapter_no_dataset"))
            self.train_btn.setEnabled(False)
            return
        # Count caption pairings
        captions = sum(1 for p in self._images if p.with_suffix(".txt").exists())
        stems = {p.stem for p in self._images}
        caches = _count_caches(self._cache, stems, self.REQUIRE_PE)
        line = t("adapter_n_pairs", n=len(self._images), c=captions)
        cache_bits = [f"latents:{caches['latents']}", f"te:{caches['te']}"]
        if self.REQUIRE_PE:
            cache_bits.append(f"pe:{caches.get('pe', 0)}")
        line += "  |  " + ", ".join(cache_bits)
        self.status_lbl.setText(line)
        self.train_btn.setEnabled(len(self._images) > 0)
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
        self.cap.setPlainText(
            cp.read_text(encoding="utf-8") if cp.exists() else t("no_caption")
        )

    def _open(self, p: Path):
        if not p.exists():
            try:
                p.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                QMessageBox.warning(self, t("error"), str(e))
                return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    # ── Subprocess actions ────────────────────────────────────────

    def _start_preprocess(self):
        self._launch([sys.executable, "tasks.py", self.PREPROCESS_TASK])

    def _start_train(self):
        if not self._source.exists() or not any(self._source.iterdir()):
            QMessageBox.warning(
                self, t("error"), t("adapter_no_dataset")
            )
            return
        self._launch([sys.executable, "tasks.py", self.TRAIN_TASK])

    def _launch(self, argv: list[str]):
        if self._proc.state() != QProcess.NotRunning:
            QMessageBox.information(self, "", "A subprocess is already running.")
            return
        self.log.clear()
        self._buf = ""
        self.log.appendPlainText("> " + " ".join(argv))
        self._proc.start(argv[0], argv[1:])
        self.preprocess_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _stop(self):
        if self._proc.state() != QProcess.NotRunning:
            self._proc.kill()

    def _read_stdout(self):
        data = bytes(self._proc.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.log.appendPlainText(line)

    def _on_finished(self, code: int, _status):
        if self._buf:
            self.log.appendPlainText(self._buf)
            self._buf = ""
        self.log.appendPlainText(t("finished", code=code))
        self.preprocess_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._reload()


class IPAdapterTab(_AdapterTab):
    SOURCE_DIR = "ip-adapter-dataset"
    CACHE_DIR = "post_image_dataset/ip-adapter"
    PREPROCESS_TASK = "ip-adapter-preprocess"
    TRAIN_TASK = "ip-adapter"
    REQUIRE_PE = True
    METHOD_LABEL = "IP-Adapter"


class EasyControlTab(_AdapterTab):
    SOURCE_DIR = "easycontrol-dataset"
    CACHE_DIR = "post_image_dataset/easycontrol"
    PREPROCESS_TASK = "easycontrol-preprocess"
    TRAIN_TASK = "easycontrol"
    REQUIRE_PE = False
    METHOD_LABEL = "EasyControl"
