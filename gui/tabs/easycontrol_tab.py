"""EasyControlTab — config editor for the EasyControl adapter family.

Reuses ConfigTab's grouped form + field-explanation panel + Save UI (so the tab
reads like the LoRA / Methods tabs — config on the left, explanation on the
right). The tab hosts two *kinds* of variant:

  • Method variants — the ``[variant] family = "easycontrol"`` gui-methods files
    (e.g. ``easycontrol``). The variant stem *is* the train.py method; the form
    edits gui-methods/<stem>.toml and training rides ConfigTab's daemon path.

  • Descriptor variants — control-task projects described by a single
    ``configs/easycontrol/<stem>.toml`` (top-level ``name`` + ``[staging]`` /
    ``[preprocess]`` / ``[training]`` tables + blueprint, plus a ``[variant]``
    metadata block). The form renders the descriptor's scalar tables (Save writes
    them back through tomlkit; blueprint and ``[variant]`` are preserved), and the
    Preprocess / Train buttons drive the staging → preprocess → descriptor-folded
    train flow (always training the base
    ``easycontrol`` method with the descriptor's ``[training]`` table folded in as
    CLI overrides). ``colorize`` is one such descriptor variant.

The selected variant implies an ``EASYADAPTER`` value for the preprocess route:

  • EasyControl (ref == target) — EASYADAPTER unset → reads easycontrol-dataset/.
  • Colorize (B&W manga → color) — EASYADAPTER=colorize → mangafy + cond/text
    caching (easycontrol_adapters/colorization/prep.py via the descriptor).

Custom variants don't map onto the easycontrol method selection, so the "+ New"
button and custom entries are suppressed here.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import tomlkit
from PySide6.QtCore import Qt, QProcess, QProcessEnvironment
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QWidget,
)

from gui import (
    ROOT,
    _read,
    _widget,
    confirm_existing_caches,
    confirm_resumable_checkpoint,
    list_gui_variants,
    merged_gui_variant_preset,
)
from gui import daemon as gui_daemon
from gui.explanations import field_help
from gui.i18n import t
from gui.tabs.config_tab import ConfigTab
from gui.theme import action_button_qss, tok
from gui.widgets import action_button, apply_variant, make_field_label

_DESCRIPTOR_DIR = ROOT / "configs" / "easycontrol"


def _descriptor_variants() -> list[tuple[int, str, str]]:
    """``(order, stem, label)`` for every descriptor with a GUI ``[variant]`` block.

    Scans ``configs/easycontrol/*.toml`` for a ``[variant] family = "easycontrol"``
    table (the GUI dropdown metadata). near_twins ships no such block — it stays
    CLI-only — so only colorize surfaces today. Sorted by ``order`` then stem."""
    out: list[tuple[int, str, str]] = []
    if not _DESCRIPTOR_DIR.is_dir():
        return out
    for p in sorted(_DESCRIPTOR_DIR.glob("*.toml")):
        try:
            doc = tomllib.loads(p.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        var = doc.get("variant")
        if isinstance(var, dict) and var.get("family") == "easycontrol":
            out.append(
                (int(var.get("order", 100)), p.stem, str(var.get("label", p.stem)))
            )
    return sorted(out, key=lambda t: (t[0], t[1]))


class EasyControlTab(ConfigTab):
    # variant stem → EASYADAPTER value (absent → default ref==target EasyControl).
    _VARIANT_ENV = {"colorize": "colorize"}
    # variant stem → cache dir the preprocess-reuse prompt inspects.
    _CACHE_DIRS = {
        "easycontrol": "post_image_dataset/easycontrol",
        "colorize": "post_image_dataset/easycontrol/colorize/cond",
    }

    def __init__(self):
        # Created BEFORE super().__init__ because ConfigTab.__init__ ends by
        # calling _try_reattach → _attach_to_job, which disables preprocess_btn.
        # On a GUI reopen mid-train that reattach fires during super().__init__,
        # before the post-super setup below would otherwise create the button.
        self.preprocess_btn = QPushButton(t("preprocess"))

        super().__init__(methods=["easycontrol"])

        # Custom variants can't map onto the easycontrol route — hide "+ New".
        self.new_variant_btn.setVisible(False)
        # test-easycontrol needs a REF_IMAGE the plain Test button can't supply.
        self.test_btn.setVisible(False)

        self.adapter_guide_btn = action_button(
            t("adapter_guide"),
            variant="success",
            tooltip=t("adapter_guide_tooltip"),
            on_click=self._open_adapter_guide,
        )
        self._top_bar.insertWidget(
            self._top_bar.indexOf(self.train_btn), self.adapter_guide_btn
        )

        # EasyControl preprocessing is the bespoke easycontrol-preprocess, so it
        # gets an explicit button before Train rather than ConfigTab's auto-chain.
        apply_variant(self.preprocess_btn, "info")
        self.preprocess_btn.clicked.connect(self._ec_start_preprocess)
        self._top_bar.insertWidget(
            self._top_bar.indexOf(self.train_btn), self.preprocess_btn
        )

        # Slimmer handler than ConfigTab: EasyControl has its own Preprocess
        # button + bespoke caches, so it skips the cache-existence/auto-chain logic.
        self.train_btn.clicked.disconnect()
        self.train_btn.clicked.connect(self._ec_start_train)

    def _open_adapter_guide(self) -> None:
        from gui.dialogs import GuidebookDialog
        from gui.i18n import current_language

        # Localized guide: ADAPTER_GUIDE.<lang>.md, English (ADAPTER_GUIDE.md) fallback.
        base = ROOT / "easycontrol_adapters"
        path = base / f"ADAPTER_GUIDE.{current_language()}.md"
        if not path.exists():
            path = base / "ADAPTER_GUIDE.md"
        GuidebookDialog(path, self).exec()

    @staticmethod
    def _descriptor_stems() -> list[str]:
        return [stem for _o, stem, _l in _descriptor_variants()]

    def _is_descriptor_variant(self, variant: str) -> bool:
        """True for a configs/easycontrol/<variant>.toml descriptor (file-edited
        launcher), vs a gui-methods method variant edited in the form."""
        return variant in self._descriptor_stems()

    def _refresh_variant_row(self, method: str) -> None:
        # gui-methods method variants first, then descriptor variants, de-duped.
        gui_variants = [
            v for v in list_gui_variants(method) if not v.startswith("custom/")
        ]
        variants = gui_variants + [
            s for s in self._descriptor_stems() if s not in gui_variants
        ]
        current = [
            self.variant_combo.itemText(i) for i in range(self.variant_combo.count())
        ]
        if current != variants:
            self.variant_combo.blockSignals(True)
            self.variant_combo.clear()
            if variants:
                self.variant_combo.addItems(variants)
            self.variant_combo.blockSignals(False)

    # Descriptor tables rendered as form groups, in this order. Everything else in
    # the file (blueprint, [variant] metadata) is preserved verbatim but not editable.
    _DESC_TABLE_ORDER = ("staging", "preprocess", "training")
    _DESC_SKIP_TABLES = frozenset({"general", "datasets", "variant"})

    def _reload(self) -> None:
        # Descriptor variants render scalar knob tables as form groups; method
        # variants fall through to ConfigTab's normal form build. Refresh the combo
        # first so the variant row is populated even when we short-circuit early.
        method = self.method_combo.currentText()
        if not method:
            return
        self._refresh_variant_row(method)
        if self._is_descriptor_variant(self._current_variant()):
            self._show_descriptor_form(self._current_variant())
            return
        self._desc_doc = None
        self._desc_widgets = []
        super()._reload()

    @staticmethod
    def _plain(v):
        """tomlkit scalars subclass the native types but carry formatting; unwrap
        to plain Python so _widget/_read see int/float/str/bool/list, not tomlkit
        items (which would round-trip oddly through the widgets)."""
        if isinstance(v, bool):
            return bool(v)
        if isinstance(v, int):
            return int(v)
        if isinstance(v, float):
            return float(v)
        if isinstance(v, str):
            return str(v)
        if isinstance(v, list):
            return [EasyControlTab._plain(x) for x in v]
        return v

    def _show_descriptor_form(self, variant: str) -> None:
        """Render the descriptor's scalar knob tables as grouped form fields.

        Mirrors ConfigTab._reload's teardown (clear self._fl, reset explain, clear
        dirty), then builds one QGroupBox per editable table (top-level scalars like
        ``name`` first, then [staging]/[preprocess]/[training]) reusing ConfigTab's
        _widget / ClickableLabel / dirty wiring. The parsed tomlkit doc is stashed
        on self so Save can write changed values back in place — comments and the
        [[datasets]] blueprint survive untouched."""
        self._origin = {}
        self._w.clear()
        # (table-or-None, key, widget, original-plain-value) so Save can route each
        # value back into the right tomlkit table.
        self._desc_widgets: list[tuple[str | None, str, QWidget, object]] = []
        while self._fl.count():
            it = self._fl.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

        path = _DESCRIPTOR_DIR / f"{variant}.toml"
        rel = path.relative_to(ROOT)
        self._desc_doc = tomlkit.parse(path.read_text(encoding="utf-8"))
        if hasattr(self, "_explain"):
            self._set_explain_html(t("easycontrol_descriptor_note", path=str(rel)))

        header = QLabel(t("easycontrol_descriptor_form_header", path=str(rel)))
        header.setWordWrap(True)
        header.setTextFormat(Qt.RichText)
        header.setStyleSheet(f"color:{tok('text')}; padding:4px 2px 8px 2px;")
        self._fl.addWidget(header)

        top = {
            k: v
            for k, v in self._desc_doc.items()
            if not isinstance(v, (dict, list)) and k not in self._DESC_SKIP_TABLES
        }
        if top:
            self._fl.addWidget(self._desc_group(t("ec_desc_group_top"), top, None))

        tables = [k for k in self._desc_doc if isinstance(self._desc_doc[k], dict)]
        ordered = [k for k in self._DESC_TABLE_ORDER if k in tables] + [
            k
            for k in tables
            if k not in self._DESC_TABLE_ORDER and k not in self._DESC_SKIP_TABLES
        ]
        for tbl in ordered:
            items = {
                k: v
                for k, v in self._desc_doc[tbl].items()
                if not isinstance(v, (dict, list))
                or (isinstance(v, list) and not any(isinstance(x, dict) for x in v))
            }
            if items:
                self._fl.addWidget(self._desc_group(f"[{tbl}]", items, tbl))

        self._fl.addStretch()
        self._clear_dirty()

    def _desc_group(self, title: str, items: dict, table: str | None) -> QGroupBox:
        """One form group for a descriptor table (or the top-level scalars when
        ``table is None``). Registers each field in self._desc_widgets and wires its
        change signal to the dirty flag; clickable labels show field help where the
        key is a known method knob (mostly [training])."""
        box = QGroupBox(title)
        form = QFormLayout()
        for k, raw in items.items():  # tomlkit preserves file order
            plain = self._plain(raw)
            w = _widget(plain, key=k)
            help_text = field_help(k)
            lbl = make_field_label(
                k,
                on_click=lambda _k=k, _h=help_text: self._show_explain(_k, _h, ()),
            )
            form.addRow(lbl, w)
            self._desc_widgets.append((table, k, w, plain))
            self._connect_dirty_signal(w)
        box.setLayout(form)
        return box

    def _save_preset(self, *, silent: bool = False):
        # Method variants write back via ConfigTab; descriptor variants re-dump the
        # tomlkit doc (comments + blueprint preserved).
        variant = self._current_variant()
        if self._is_descriptor_variant(variant):
            self._save_descriptor(variant, silent=silent)
            return
        super()._save_preset(silent=silent)

    def _save_descriptor(self, variant: str, *, silent: bool) -> None:
        doc = getattr(self, "_desc_doc", None)
        if doc is None:
            return
        for table, key, w, orig in self._desc_widgets:
            val = _read(w, orig)
            # Sparse writeback: only touch changed fields so unedited ones keep their
            # original tomlkit formatting (quote style, arrays, float notation).
            if val == orig:
                continue
            target = doc if table is None else doc.get(table)
            if target is not None:
                target[key] = val
        path = _DESCRIPTOR_DIR / f"{variant}.toml"
        path.write_text(tomlkit.dumps(doc), encoding="utf-8")
        self._clear_dirty()
        if not silent:
            QMessageBox.information(self, t("saved"), f"Saved {path.relative_to(ROOT)}")

    def _ec_adapter(self) -> str | None:
        # A descriptor variant's stem IS its EASYADAPTER value (task dispatch loads
        # configs/easycontrol/<stem>.toml), so route those generically.
        variant = self._current_variant()
        if self._is_descriptor_variant(variant):
            return variant
        return self._VARIANT_ENV.get(variant)

    def _ec_proc_env(self) -> QProcessEnvironment:
        """System env with EASYADAPTER set (or cleared) for the active variant.
        Rebuilt each launch so a stale value can't leak across runs (the QProcess
        is reused)."""
        env = QProcessEnvironment.systemEnvironment()
        adapter = self._ec_adapter()
        if adapter:
            env.insert("EASYADAPTER", adapter)
        else:
            env.remove("EASYADAPTER")
        return env

    def _ec_cache_dir(self) -> Path:
        # Descriptor variants cache under their `name` slug, which can differ from
        # the dropdown stem (e.g. near_twins.toml ships name = "sanitize").
        variant = self._current_variant()
        if self._is_descriptor_variant(variant):
            slug = variant
            try:
                doc = tomllib.loads(
                    (_DESCRIPTOR_DIR / f"{variant}.toml").read_text(encoding="utf-8")
                )
                slug = str(doc.get("name") or variant).strip()
            except (OSError, tomllib.TOMLDecodeError):
                pass
            return ROOT / "post_image_dataset" / "easycontrol" / slug / "cond"
        rel = self._CACHE_DIRS.get(variant, "post_image_dataset/easycontrol")
        return ROOT / rel

    def _ec_launch(self, argv: list[str], mode: str) -> None:
        if self._proc.state() != QProcess.NotRunning:
            return
        self.log.clear()
        self._reset_progress()
        self._progress_tracker.mark_starting(t("starting"))
        self._running_mode = mode
        adapter = self._ec_adapter()
        prefix = f"EASYADAPTER={adapter} " if adapter else ""
        self._log(f"> {prefix}python {' '.join(argv)}\n")
        self._proc.setProcessEnvironment(self._ec_proc_env())
        self._ec_set_busy(True)
        self._proc.start(sys.executable, argv)

    def _flush_dirty_descriptor(self) -> bool:
        """Flush unsaved descriptor form edits before a descriptor run (preprocess
        and train both re-read the file from disk). Returns False to abort the run
        if the buffer doesn't parse — _save_descriptor leaves dirty set and has
        already surfaced the parse error."""
        if self._dirty:
            self._save_preset(silent=True)
        return not self._dirty

    def _ec_start_preprocess(self) -> None:
        if self._is_descriptor_variant(self._current_variant()):
            if not self._flush_dirty_descriptor():
                return
        if not confirm_existing_caches(self, self._ec_cache_dir()):
            return
        argv = ["tasks.py", "easycontrol-preprocess"]
        # Single GUI button runs the whole pipeline in one shot; --no-skip_mangafy
        # re-enables the staging stage easycontrol-preprocess otherwise skips.
        if self._is_descriptor_variant(self._current_variant()):
            argv.append("--no-skip_mangafy")
        self._ec_launch(argv, "ec_preprocess")

    def _ec_start_train(self) -> None:
        variant = self._current_variant()
        if self._is_descriptor_variant(variant):
            if not self._flush_dirty_descriptor():
                return
            self._ec_start_train_descriptor(variant)
            return
        # Method variant: flush form edits so train.py reads the same TOML.
        if self._dirty:
            self._save_preset(silent=True)
        merged, _ = merged_gui_variant_preset(variant, self._current_preset())
        if not confirm_resumable_checkpoint(self, merged):
            return
        # The variant stem IS the train.py method (EASYADAPTER read only at dispatch),
        # so submit to the daemon exactly like the LoRA tab.
        self._launch_training(variant)

    def _descriptor_merged(self, variant: str) -> dict:
        """Effective config for a descriptor variant: the base easycontrol method
        chain (base → preset → methods/easycontrol.toml) with the descriptor's
        [training] table folded on top — the same merge train.py will see. Used
        only to locate the output checkpoint for the resume prompt."""
        from library.config.io import load_method_preset

        merged = dict(load_method_preset("easycontrol", self._current_preset()))
        doc = tomllib.loads(
            (_DESCRIPTOR_DIR / f"{variant}.toml").read_text(encoding="utf-8")
        )
        merged.update(doc.get("training") or {})
        return merged

    def _ec_start_train_descriptor(self, variant: str) -> None:
        """Train a descriptor variant via the same flow as the CLI
        `make easycontrol EASYADAPTER=<variant>`: train the base easycontrol method
        with the descriptor's blueprint (--dataset_config sidecar) + [training]
        table folded in as CLI overrides. Routed through the daemon so it survives
        the GUI closing, exactly like a method-variant train."""
        if not confirm_resumable_checkpoint(self, self._descriptor_merged(variant)):
            return
        from scripts.tasks.training import _easy_train_extra

        try:
            extra = _easy_train_extra(variant, [])
        except SystemExit as e:  # descriptor missing a [[datasets]] blueprint, etc.
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, t("error"), str(e))
            return
        # Submit the base easycontrol method (methods/ tree, not gui-methods) with
        # the descriptor argv appended; mirror ConfigTab's busy-UI + attach.
        merged = self._descriptor_merged(variant)
        logging_dir = merged.get("logging_dir")
        if logging_dir and self._tb_panel is not None:
            self._tb_panel.set_log_dir(logging_dir)
        self.train_btn.setText(t("train") + " ...")
        self.train_btn.setStyleSheet(action_button_qss("busy"))
        self._ec_set_busy(True)
        self.log.clear()
        self._reset_progress()
        self._progress_tracker.mark_starting(t("starting"))
        self._log(t("daemon_submitting") + "\n")
        QApplication.processEvents()
        try:
            resp = gui_daemon.submit_training(
                method="easycontrol",
                preset=self._current_preset(),
                methods_subdir=None,
                extra=extra,
            )
        except Exception as e:  # noqa: BLE001 — daemon failed to start / submit
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, t("error"), t("daemon_submit_failed", err=str(e)))
            self._restore_idle_ui()
            return
        job_id = resp.get("job_id") if isinstance(resp, dict) else None
        if not job_id:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(
                self, t("error"), t("daemon_submit_failed", err=str(resp))
            )
            self._restore_idle_ui()
            return
        self._log(t("daemon_queued", job_id=job_id))
        self._attach_to_job(job_id, replay_log=False)

    # Also gray out the EasyControl-only Preprocess button on attach. Overriding
    # _attach_to_job (not just the launch site) covers re-attach on GUI reopen too.
    def _attach_to_job(
        self, job_id: str, *, replay_log: bool, kind: str = "train"
    ) -> None:
        super()._attach_to_job(job_id, replay_log=replay_log, kind=kind)
        self.preprocess_btn.setEnabled(False)

    def _ec_set_busy(self, busy: bool) -> None:
        self.preprocess_btn.setEnabled(not busy)
        self.train_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        self.method_combo.setEnabled(not busy)
        self.variant_combo.setEnabled(not busy)
        self.preset_combo.setEnabled(not busy)

    def _restore_idle_ui(self):
        super()._restore_idle_ui()
        self.preprocess_btn.setEnabled(True)

    def _try_reattach(self) -> None:
        """Re-bind to an easycontrol/colorize daemon training job still running
        when this tab is constructed (close GUI mid-train → reopen → re-attach).

        Discriminate by method so we don't hijack another tab's job (e.g. a
        LoRA-tab training run): the daemon's single active job is shared across
        the ConfigTab subclasses, and this tab only owns its own family's
        variants. EasyControl preprocess runs as a QProcess (not a daemon
        command job), so there's nothing of ours to re-attach but the train."""
        try:
            job_id = gui_daemon.active_job_id()
        except Exception:  # noqa: BLE001 — daemon unreachable → nothing to attach
            return
        if not job_id or gui_daemon.read_job_kind(job_id) != "train":
            return
        family = {
            v for v in list_gui_variants("easycontrol") if not v.startswith("custom/")
        }
        if gui_daemon.read_job_label(job_id) not in family:
            return
        self.log.clear()
        self._reset_progress()
        self._progress_tracker.mark_starting(t("starting"))
        self._log(t("daemon_reattached", job_id=job_id))
        self._attach_to_job(job_id, replay_log=True, kind="train")
