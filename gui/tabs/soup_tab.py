"""SoupTrainTab — uncond-init ΔW-soup pipeline launcher (experimental).

Reuses the shared distill config-editor layout (``_DistillConfigTab``): the
``[soup]`` pipeline knobs *and* the plain-LoRA training stack are edited as the
sectioned form over ``configs/soup/soup.toml`` (Save writes them back through
tomlkit, comments preserved), as in the Turbo tab. The one soup-specific
addition is a small **run-only** box pinned to the
top of the form carrying the per-run ``--path_pattern`` / ``--name`` / fine-tune
args — these are *not* persisted to the TOML (like ``output_name``, they vary
per run), so they live outside ``_fields`` and Save ignores them.

Unlike Turbo, submission does **not** route through ``tasks.py`` — it
submits ``python -m scripts.soup.pipeline`` directly as one daemon command job.
Routing through ``tasks.py`` would re-enter ``run_command`` *inside* the daemon
and nest-submit a second command job, deadlocking the serial queue; the pipeline
instead runs its ``train.py`` phases as direct subprocesses within this one job.
"""

from __future__ import annotations

import shlex

from PySide6.QtWidgets import QFormLayout, QGroupBox, QLineEdit, QMessageBox

from gui.i18n import t
from gui.tabs.distill_tab import _DistillConfigTab
from gui.widgets import make_field_label


class SoupTrainTab(_DistillConfigTab):
    """ΔW-soup pipeline launcher — distill-tab layout + a run-only arg box."""

    CONFIG_PATH = "configs/soup/soup.toml"
    TRAIN_TASK = "soup"
    METHOD_LABEL = "Soup"

    def _load_and_build(self) -> None:
        # Build the TOML-backed section form first, then pin the run-only box
        # (path_pattern / name / fine-tune args — never written to the TOML) to
        # the very top so it reads before the [soup] knobs and training stack.
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
                style="text-decoration: underline dotted; color:#ddd;",
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
        # Override the base's `tasks.py <target>` launch: soup submits the
        # pipeline module directly (see the module docstring). `_launch` still
        # flushes any unsaved [soup]/training edits so the pipeline reads them.
        argv = self._build_argv()
        if argv is None:
            return
        self._launch(self.TRAIN_TASK, argv)

    def _build_argv(self) -> list[str] | None:
        """The ``python -m scripts.soup.pipeline`` argv from the run-only box.
        The pool / dose / seeds / rank knobs come from the (just-saved) TOML, so
        only the per-run selection rides argv. None if path_pattern is blank."""
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
