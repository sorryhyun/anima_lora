"""The MOSS window's i2i tabs: stage → preprocess on one, train → test on the
other, for one i2i task.

Tasks are the ``library.moss.requests.I2I_CACHE`` rows whose
``configs/moss/i2i/<task>/cache.toml`` exists; both tabs show the selected task
under a header holding the task picker and the tab's buttons. Each form starts
from the task's
``cache.toml`` / ``train.toml`` / ``generate.toml``, and only what differs from
that seed is saved (``moss_gui_i2i`` in ``gui_settings.json``). Stage is
``tasks.py i2i-stage`` with ``TASK`` in the job's env, a daemon command job like
the rest. Test renders with the LoRA only and shows each reference beside its
output.
"""

from __future__ import annotations

import dataclasses
import html
import json
import time
from pathlib import Path
from typing import ClassVar

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui.core.paths import get_setting, set_setting
from gui.lines.window import LineJob, LinePage, RequestForm, tr
from gui.moss.plans import generate_plans, model_items, train_plans
from gui.theme import Pad, rich_text_pt, tok
from gui.widgets import ExplainPanel, action_button
from library.env import resolve_under_home
from library.moss.requests import (
    I2I_CACHE,
    ColorizeCacheRequest,
    GenerateRequest,
    TrainRequest,
    i2i_config,
)
from library.moss.scan import cache_counts

STAGE, CACHE, TRAIN, TEST = LABELS = (
    "moss-i2i-stage",
    "moss-i2i-cache",
    "moss-i2i-train",
    "moss-i2i-test",
)
SETTINGS_KEY = "moss_gui_i2i"
THUMB_PX = 300


@dataclasses.dataclass
class StageRequest:
    """``make i2i-stage`` as a daemon job; ``TASK`` rides in the env."""

    SCRIPT: ClassVar[str] = "tasks.py"

    def to_argv(self) -> list[str]:
        return ["i2i-stage"]


def tasks() -> list[str]:
    return [t for t in I2I_CACHE if i2i_config(t, "cache").is_file()]


def _seed(cls, task: str, kind: str) -> dict:
    path = i2i_config(task, kind)
    return cls.load_config(path) if path.is_file() else {}


def _staged(req) -> tuple[int | None, Path]:
    """Pairs the cache request would read, ``None`` while its source is missing."""
    if getattr(req, "pairs", ""):
        path = resolve_under_home(req.pairs)
        if not path.is_file():
            return None, path
        rows = path.read_text(encoding="utf-8").splitlines()
        return sum(1 for r in rows if r.strip()), path
    path = resolve_under_home(getattr(req, "staging", "") or req.src)
    if not path.is_dir():
        return None, path
    return sum(1 for _ in path.rglob("*.png")), path


def _records(run_dir: Path) -> list[dict]:
    manifest = run_dir / "manifest.json"
    if not manifest.is_file():
        return []
    return json.loads(manifest.read_text(encoding="utf-8"))["images"]


def _thumb(path: Path, caption: str) -> QWidget:
    cell = QWidget()
    v = QVBoxLayout(cell)
    v.setContentsMargins(0, 0, 0, 0)
    image = QLabel()
    image.setPixmap(
        QPixmap(str(path)).scaled(
            THUMB_PX, THUMB_PX, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
    )
    image.setToolTip(str(path))
    label = QLabel(caption)
    label.setAlignment(Qt.AlignCenter)
    v.addWidget(image)
    v.addWidget(label)
    return cell


def _group(title: str, hint: str | None, *widgets: QWidget) -> QGroupBox:
    box = QGroupBox(tr(title))
    v = QVBoxLayout(box)
    if hint:
        label = QLabel(hint)
        label.setWordWrap(True)
        v.addWidget(label)
    for w in widgets:
        if isinstance(w, QLabel):
            w.setWordWrap(True)
        v.addWidget(w)
    return box


def _body(*groups: QGroupBox) -> QWidget:
    body = QWidget()
    v = QVBoxLayout(body)
    for g in groups:
        v.addWidget(g)
    v.addStretch(1)
    return body


class _Task:
    """One task's forms, seeded from its configs, and the read-outs the two
    tabs show for it."""

    def __init__(self, task: str, saved: dict, prep_help, train_help):
        self.task = task
        self.cache_cls = I2I_CACHE[task]
        self.seeds = {
            "cache": _seed(self.cache_cls, task, "cache"),
            "train": _seed(TrainRequest, task, "train"),
            "test": _seed(GenerateRequest, task, "generate"),
        }
        self.forms = {
            "cache": RequestForm(self.cache_cls, prep_help),
            "train": RequestForm(TrainRequest, train_help),
            "test": RequestForm(GenerateRequest, train_help),
        }
        model_items(
            self.forms["train"], train_plans(self.seeds["train"]), "model_help_i2i"
        )
        model_items(
            self.forms["test"], generate_plans(self.seeds["test"]), "model_help_test"
        )
        for kind, form in self.forms.items():
            form.set_values(self.seeds[kind])
            form.set_values(saved.get(kind, {}))
        self.cache_summary = QLabel()
        self.train_summary = QLabel()
        self.images = QGridLayout()

    def prep_body(self) -> QWidget:
        return _body(
            _group("i2i_stage_group", tr("i2i_stage_hint", task=self.task)),
            _group("i2i_cache_group", None, self.forms["cache"], self.cache_summary),
        )

    def train_body(self) -> QWidget:
        test = _group("i2i_test_group", tr("i2i_test_hint"), self.forms["test"])
        test.layout().addLayout(self.images)
        return _body(
            _group("i2i_train_group", None, self.forms["train"], self.train_summary),
            test,
        )

    def state(self) -> dict:
        """Per form, the values that differ from the task's seed."""
        out = {}
        for kind, form in self.forms.items():
            req = form.request()
            if req is None:
                continue
            out[kind] = {
                f.name: getattr(req, f.name)
                for f in dataclasses.fields(req)
                if getattr(req, f.name) != self.seeds[kind].get(f.name, f.default)
            }
        return out

    def train_output(self) -> Path:
        return resolve_under_home(
            self.forms["train"].text_of("output") or TrainRequest().output
        )

    def show_test(self, run_dir: Path | None) -> None:
        while self.images.count():
            item = self.images.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for row, r in enumerate(_records(run_dir) if run_dir else []):
            if r.get("ref"):
                self.images.addWidget(_thumb(Path(r["ref"]), tr("test_ref")), row, 0)
            m = r.get("lora_multiplier")
            caption = tr("test_lora", m=f"{m:g}") if m else tr("test_base")
            self.images.addWidget(_thumb(Path(r["image"]), caption), row, 1)

    def latest_test_dir(self) -> Path | None:
        base = resolve_under_home(
            self.forms["test"].text_of("out_dir") or GenerateRequest().out_dir
        )
        runs = sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
        return next((p for p in reversed(runs) if _records(p)), None)


class _Tab(LinePage):
    """One i2i tab: the task picker and the tab's buttons pinned on top, the
    selected task's groups below, help on the right."""

    def __init__(self, i2i: I2I, title_key: str, labels: tuple[str, ...], buttons):
        super().__init__()
        self.i2i = i2i
        self.title = tr(title_key)
        self.labels = labels
        self.buttons = buttons
        self.explain = ExplainPanel()
        self.task_combo = QComboBox()
        self.stack = QStackedWidget()

        header = QHBoxLayout()
        header.setContentsMargins(Pad.SM, Pad.SM, Pad.SM, Pad.XS)
        header.addWidget(QLabel(tr("i2i_task")))
        header.addWidget(self.task_combo)
        header.addStretch(1)
        for b in buttons:
            header.addWidget(b)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.stack)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(scroll)
        split.addWidget(self.explain)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([620, 380])
        root = QVBoxLayout(self)
        root.setContentsMargins(Pad.NONE, Pad.NONE, Pad.NONE, Pad.NONE)
        root.addLayout(header)
        root.addWidget(split, 1)

    def add_task(self, task: str, body: QWidget) -> None:
        self.task_combo.addItem(task)
        self.stack.addWidget(body)

    def set_busy(self, busy: bool) -> None:
        for b in self.buttons:
            b.setEnabled(not busy)
        if not busy and self is self.i2i.prep:
            self.i2i.rescan()

    def job_finished(self, label: str, ok: bool) -> None:
        self.i2i.job_finished(label, ok)

    def persist(self) -> None:
        if self is self.i2i.prep:
            self.i2i.persist()


class I2I:
    """What the two tabs share: the tasks, the selected one, the job handlers."""

    def __init__(self, panel):
        self.panel = panel
        self._test_task: _Task | None = None
        self._test_out: Path | None = None
        saved = get_setting(SETTINGS_KEY, {}) or {}

        self.prep = _Tab(
            self,
            "tab_i2i_preprocess",
            (STAGE, CACHE),
            [
                action_button(tr("run_stage"), on_click=self.run_stage),
                action_button(tr("run_cache"), on_click=self.run_cache),
            ],
        )
        self.prep.scripts = {ColorizeCacheRequest.SCRIPT: CACHE}
        self.train = _Tab(
            self,
            "tab_i2i_train",
            (TRAIN, TEST),
            [
                action_button(tr("run_train"), on_click=self.run_train),
                action_button(
                    tr("i2i_chain"), variant="secondary", on_click=self.run_chain
                ),
                action_button(tr("run_test"), variant="info", on_click=self.run_test),
            ],
        )
        self.tabs = (self.prep, self.train)

        self.tasks = [
            _Task(
                task,
                saved.get(task, {}),
                self.prep.explain.show_field_help,
                self.train.explain.show_field_help,
            )
            for task in tasks()
        ]
        for t in self.tasks:
            self.prep.add_task(t.task, t.prep_body())
            self.train.add_task(t.task, t.train_body())
            t.show_test(t.latest_test_dir())
        if not self.tasks:
            for tab in self.tabs:
                tab.stack.addWidget(QLabel(tr("i2i_no_tasks")))

        self._rescan_timer = QTimer(self.prep)
        self._rescan_timer.setSingleShot(True)
        self._rescan_timer.setInterval(300)
        self._rescan_timer.timeout.connect(self.rescan)
        for t in self.tasks:
            for form in t.forms.values():
                form.watch(self._rescan_timer.start)
        names = [t.task for t in self.tasks]
        self.index = names.index(saved["task"]) if saved.get("task") in names else 0
        for tab in self.tabs:
            tab.task_combo.setCurrentIndex(self.index)
            tab.stack.setCurrentIndex(self.index)
            tab.task_combo.currentIndexChanged.connect(self._select)
        self.rescan()

    # ── state ────────────────────────────────────────────────────────────

    @property
    def current(self) -> _Task | None:
        return self.tasks[self.index] if self.tasks else None

    def _select(self, index: int) -> None:
        # Setting the other tab's combo re-enters with the same index.
        if index == self.index:
            return
        self.index = index
        for tab in self.tabs:
            tab.task_combo.setCurrentIndex(index)
            tab.stack.setCurrentIndex(index)
            tab.explain.show_guide(None)
        self.rescan()

    def _guide_html(self, tab: _Tab, task: str) -> str:
        dim = tok("text_dim")
        return (
            f"<h2 style='margin:0 0 10px 0; font-size:{rich_text_pt(18)};'>"
            f"{html.escape(tab.title)} · {html.escape(task)}</h2>"
            f"<p style='font-size:{rich_text_pt(15)}; line-height:1.6;'>"
            f"{html.escape(tr('i2i_task_' + task))}</p>"
            f"<p style='line-height:1.6;'>{html.escape(tr('i2i_guide'))}</p>"
            f"<p style='color:{dim}; font-style:italic;'>"
            f"{html.escape(tr('click_field'))}</p>"
        )

    def rescan(self) -> None:
        t = self.current
        if t is None:
            return
        for tab in self.tabs:
            if tab.explain.mode is None:
                tab.explain.show_guide(self._guide_html(tab, t.task))
        lines = []
        req = t.forms["cache"].request()
        if req is not None:
            n, path = _staged(req)
            lines.append(
                tr("i2i_staged", n=n) if n else tr("i2i_not_staged", path=path)
            )
            c = cache_counts(resolve_under_home(req.out))
            lines.append(
                tr("cache_counts", samples=c.samples, text=c.text, latents=c.latents)
            )
        t.cache_summary.setText("\n".join(lines))
        cache = t.forms["train"].text_of("cache")
        n = cache_counts(resolve_under_home(cache)).samples if cache else 0
        t.train_summary.setText(tr("train_ready", n=n) if n else tr("train_empty"))
        t.forms["test"].widgets["lora"].setPlaceholderText(
            tr("test_lora_placeholder", path=t.train_output())
        )

    def persist(self) -> None:
        data = {t.task: t.state() for t in self.tasks}
        if self.current is not None:
            data["task"] = self.current.task
        set_setting(SETTINGS_KEY, data)

    def job_finished(self, label: str, ok: bool) -> None:
        if label != TEST:
            return
        t = self._test_task
        if ok and t is not None:
            t.show_test(self._test_out or t.latest_test_dir())
        self._test_task = self._test_out = None

    # ── jobs ─────────────────────────────────────────────────────────────

    def _stage_job(self, t: _Task) -> LineJob:
        return LineJob(StageRequest(), STAGE, env={"TASK": t.task})

    def _warn(self, key: str, **kwargs) -> None:
        QMessageBox.warning(self.panel, tr("window_title"), tr(key, **kwargs))

    def run_stage(self) -> None:
        if self.current is None:
            return
        self.persist()
        self.panel.submit([self._stage_job(self.current)])

    def _cache_req(self, t: _Task, staged: bool):
        req = t.forms["cache"].request()
        if req is None or not self.panel._model_ok(req):
            return None
        if staged:
            n, path = _staged(req)
            if not n:
                self._warn("i2i_not_staged", path=path)
                return None
        return req

    def run_cache(self) -> None:
        t = self.current
        req = self._cache_req(t, staged=True) if t else None
        if req is None:
            return
        self.persist()
        self.panel.submit([LineJob(req, CACHE)])

    def run_train(self) -> None:
        t = self.current
        req = t.forms["train"].request() if t else None
        if req is None or not self.panel._model_ok(req):
            return
        self.persist()
        self.panel.submit([LineJob(req, TRAIN)])

    def run_chain(self) -> None:
        t = self.current
        cache_req = self._cache_req(t, staged=False) if t else None
        if cache_req is None:
            return
        # Training reads what this preprocessing run writes.
        t.forms["train"].set_values({"cache": cache_req.out})
        train_req = t.forms["train"].request()
        if train_req is None or not self.panel._model_ok(train_req):
            return
        self.persist()
        self.panel.submit(
            [
                self._stage_job(t),
                LineJob(cache_req, CACHE, note=tr("i2i_stage_next")),
                LineJob(train_req, TRAIN, note=tr("chain_next")),
            ]
        )

    def run_test(self) -> None:
        t = self.current
        req = t.forms["test"].request() if t else None
        if req is None or not self.panel._model_ok(req):
            return
        if not req.ref_image or not resolve_under_home(req.ref_image).exists():
            self._warn("i2i_need_ref")
            return
        if not req.ref_vision and not (req.prompt or req.prompts_file):
            self._warn("i2i_need_prompt")
            return
        lora = resolve_under_home(req.lora) if req.lora else t.train_output()
        if not lora.is_file():
            self._warn("no_lora", path=lora)
            return
        self.persist()
        # One folder per run, so earlier tests stay on disk.
        out = resolve_under_home(req.out_dir) / time.strftime("%Y%m%d-%H%M%S")
        req = dataclasses.replace(req, lora=str(lora), out_dir=str(out))
        if self.panel.submit([LineJob(req, TEST)]):
            self._test_task, self._test_out = t, out


def pages(panel) -> list[LinePage]:
    """The i2i Preprocess and i2i Train tabs."""
    return list(I2I(panel).tabs)
