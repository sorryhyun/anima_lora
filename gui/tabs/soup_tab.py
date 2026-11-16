"""SoupTrainTab — uncond-init ΔW-soup pipeline launcher (experimental).

Edits ``configs/soup/soup.toml`` with the shared distill layout
(``_DistillConfigTab``), plus a run-only box for the per-run ``--path_pattern`` /
``--name`` / fine-tune args, which are not persisted (outside ``_fields``).

Submits ``python -m scripts.soup.pipeline`` directly as one daemon command job,
not through ``tasks.py``: that would re-enter ``run_command`` inside the daemon
and nest-submit a second job, deadlocking the serial queue. The pipeline runs
its ``train.py`` phases as subprocesses within this one job.
"""

from __future__ import annotations

import shlex

from PySide6.QtWidgets import QFormLayout, QGroupBox, QLineEdit, QMessageBox

from gui.i18n import t
from gui.theme import tok
from gui.tabs.distill_tab import _DistillConfigTab
from gui.widgets import make_field_label


class SoupTrainTab(_DistillConfigTab):
    """ΔW-soup pipeline launcher — distill-tab layout + a run-only arg box."""

    CONFIG_PATH = "configs/soup/soup.toml"
    TRAIN_TASK = "soup"
    METHOD_LABEL = "Soup"

    def _load_and_build(self) -> None:
        # Run-only box goes above the TOML-backed form.
        super()._load_and_build()
        self._fl.insertWidget(0, self._build_run_box())

    def _build_run_box(self) -> QGroupBox:
        box = QGroupBox(t("soup_run_section"))
        form = QFormLayout()

        def row(key: str, widget: QLineEdit) -> None:
            tip = t(f"{key}_tip")
            widget.setToolTip(tip)
            lbl = make_field_label(
                t(key),
                style=f"text-decoration: underline dotted; color:{tok('text')};",
                tooltip=tip,
                on_click=lambda _k=key, _t=tip: self._show_explain(t(_k), _t),
            )
            form.addRow(lbl, widget)

        self._path_pattern = QLineEdit()
        self._path_pattern.setPlaceholderText("sincos/*   or   art_a/*|art_b/*")
        row("soup_path_pattern", self._path_pattern)

        self._name = QLineEdit()
        self._name.setPlaceholderText("(blank = derived from path_pattern)")
        row("soup_name", self._name)

        self._ft_args = QLineEdit()
        self._ft_args.setPlaceholderText("--network_dim 32 --max_train_epochs 8")
        row("soup_ft_args", self._ft_args)

        box.setLayout(form)
        return box

    # ------------------------------------------------------------------ launch
    def _start_train(self) -> None:
        # Submits the pipeline module directly (module docstring); `_launch`
        # still flushes unsaved edits so the pipeline reads them.
        argv = self._build_argv()
        if argv is None:
            return
        self._launch(self.TRAIN_TASK, argv)

    def _build_argv(self) -> list[str] | None:
        """The ``python -m scripts.soup.pipeline`` argv from the run-only box
        (everything else comes from the saved TOML). None if path_pattern is
        blank."""
        pattern = self._path_pattern.text().strip()
        if not pattern:
            QMessageBox.warning(self, t("error"), t("soup_path_pattern_required"))
            return None
        argv = ["-m", "scripts.soup.pipeline", "--path_pattern", pattern]
        name = self._name.text().strip()
        if name:
            argv += ["--name", name]
        # Unknown args forward to the Phase-2 fine-tunes (make soup ARGS="…").
        try:
            argv += shlex.split(self._ft_args.text())
        except ValueError as e:
            QMessageBox.warning(self, t("error"), str(e))
            return None
        return argv
