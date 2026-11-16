"""EasyControlTab — config editor for the EasyControl adapter family.

Extends ConfigTab. Two kinds of variant:

  • Method variants: ``[variant] family = "easycontrol"`` gui-methods files. The
    stem *is* the train.py method and training rides ConfigTab's daemon path.

  • Descriptor variants (e.g. ``colorize``): an EasyControl descriptor
    ``configs/easycontrol/<stem>.toml`` beside the i2i task file
    ``i2i_tasks/configs/<stem>.toml``. The form renders both files' scalar
    tables (Save writes back through tomlkit, blueprint and ``[variant]``
    preserved); Train trains the base ``easycontrol`` method with the
    descriptor's ``[training]`` table folded in as CLI overrides.

The variant sets ``EASYADAPTER`` for preprocess: unset reads
easycontrol-dataset/ (ref == target); a descriptor stem selects its prep route.
Custom variants don't map onto this, so "+ New" and custom entries are hidden.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import tomlkit
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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
from gui.jobs import daemon as gui_daemon
from gui.explanations import field_help
from gui.i18n import t
from gui.tabs.config_tab import ConfigTab
from gui.theme import tok
from gui.widgets import action_button, apply_variant, make_field_label

_DESCRIPTOR_DIR = ROOT / "configs" / "easycontrol"
_TASK_DIR = ROOT / "i2i_tasks" / "configs"


def _descriptor_variants() -> list[tuple[int, str, str]]:
    """``(order, stem, label)`` for every descriptor with a GUI ``[variant]`` block.

    Scans ``configs/easycontrol/*.toml`` for a ``[variant] family = "easycontrol"``
    table; a descriptor without one stays CLI-only. Sorted by ``order`` then stem."""
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
        "easycontrol": "post_image_dataset/i2i",
        "colorize": "post_image_dataset/i2i/colorize/cond",
    }

    def __init__(self):
        # Created before super().__init__: its _try_reattach → _attach_to_job
        # disables preprocess_btn when the GUI reopens mid-train.
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

        # Own Preprocess button + bespoke caches, so no cache check / auto-chain.
        self.train_btn.clicked.disconnect()
        self.train_btn.clicked.connect(self._ec_start_train)

    def _open_adapter_guide(self) -> None:
        from gui.dialogs.guidebook import GuidebookDialog
        from gui.i18n import current_language

        # Localized guide: ADAPTER_GUIDE.<lang>.md, English (ADAPTER_GUIDE.md) fallback.
        base = ROOT / "i2i_tasks"
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
        # Refresh the combo first so the variant row is populated on the
        # descriptor short-circuit too.
        method = self.method_combo.currentText()
        if not method:
            return
        self._refresh_variant_row(method)
        if self._is_descriptor_variant(self._current_variant()):
            self._show_descriptor_form(self._current_variant())
            return
        self._desc_docs = None
        self._desc_widgets = []
        super()._reload()

    @staticmethod
    def _plain(v):
        """Unwrap tomlkit scalars (native subclasses carrying formatting) to plain
        Python so they round-trip through _widget/_read."""
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

    def _desc_paths(self, variant: str) -> dict[str, Path]:
        """The files a descriptor variant edits: its i2i task file (``name``,
        ``stage``, [staging]) when there is one, then the EasyControl descriptor."""
        task = _TASK_DIR / f"{variant}.toml"
        paths = {"task": task} if task.is_file() else {}
        paths["ec"] = _DESCRIPTOR_DIR / f"{variant}.toml"
        return paths

    def _show_descriptor_form(self, variant: str) -> None:
        """Render the descriptor's scalar knob tables as grouped form fields.

        One QGroupBox per editable table: the i2i task file's top-level scalars
        and [staging] first, then the descriptor's [preprocess]/[training]. The
        parsed tomlkit docs are kept so Save writes changed values back in place
        (comments and the [[datasets]] blueprint untouched)."""
        self._origin = {}
        self._clear_form()
        # (doc, table-or-None, key, widget, original-plain-value) so Save can route
        # each value back into the right file and tomlkit table.
        self._desc_widgets: list[tuple[str, str | None, str, QWidget, object]] = []

        paths = self._desc_paths(variant)
        rels = " + ".join(str(path.relative_to(ROOT)) for path in paths.values())
        self._desc_docs = {
            doc: tomlkit.parse(path.read_text(encoding="utf-8"))
            for doc, path in paths.items()
        }
        self._explain.show_guide(t("easycontrol_descriptor_note", path=rels))

        header = QLabel(t("easycontrol_descriptor_form_header", path=rels))
        header.setWordWrap(True)
        header.setTextFormat(Qt.RichText)
        header.setStyleSheet(f"color:{tok('text')}; padding:4px 2px 8px 2px;")
        self._fl.addWidget(header)

        for doc_key, doc in self._desc_docs.items():
            top = {
                k: v
                for k, v in doc.items()
                if not isinstance(v, (dict, list)) and k not in self._DESC_SKIP_TABLES
            }
            if top:
                self._fl.addWidget(
                    self._desc_group(t("ec_desc_group_top"), top, doc_key, None)
                )

            tables = [k for k in doc if isinstance(doc[k], dict)]
            ordered = [k for k in self._DESC_TABLE_ORDER if k in tables] + [
                k
                for k in tables
                if k not in self._DESC_TABLE_ORDER and k not in self._DESC_SKIP_TABLES
            ]
            for tbl in ordered:
                items = {
                    k: v
                    for k, v in doc[tbl].items()
                    if not isinstance(v, (dict, list))
                    or (isinstance(v, list) and not any(isinstance(x, dict) for x in v))
                }
                if items:
                    self._fl.addWidget(
                        self._desc_group(f"[{tbl}]", items, doc_key, tbl)
                    )

        self._fl.addStretch()
        self._clear_dirty()

    def _desc_group(
        self, title: str, items: dict, doc: str, table: str | None
    ) -> QGroupBox:
        """One form group for a descriptor table (or the top-level scalars when
        ``table is None``) of ``self._desc_docs[doc]``, registered in
        ``self._desc_widgets``."""
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
            self._desc_widgets.append((doc, table, k, w, plain))
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
        docs = getattr(self, "_desc_docs", None)
        if not docs:
            return
        changed: set[str] = set()
        for doc_key, table, key, w, orig in self._desc_widgets:
            val = _read(w, orig)
            # Sparse writeback: only touch changed fields so unedited ones keep their
            # original tomlkit formatting (quote style, arrays, float notation).
            if val == orig:
                continue
            doc = docs[doc_key]
            target = doc if table is None else doc.get(table)
            if target is not None:
                target[key] = val
                changed.add(doc_key)
        paths = self._desc_paths(variant)
        for doc_key in changed:
            paths[doc_key].write_text(tomlkit.dumps(docs[doc_key]), encoding="utf-8")
        self._clear_dirty()
        if not silent:
            saved = ", ".join(str(paths[d].relative_to(ROOT)) for d in sorted(changed))
            QMessageBox.information(
                self, t("saved"), f"Saved {saved or '(no changes)'}"
            )

    def _ec_adapter(self) -> str | None:
        # A descriptor variant's stem IS its EASYADAPTER value (task dispatch loads
        # configs/easycontrol/<stem>.toml), so route those generically.
        variant = self._current_variant()
        if self._is_descriptor_variant(variant):
            return variant
        return self._VARIANT_ENV.get(variant)

    def _ec_cache_dir(self) -> Path:
        # Descriptor variants cache under their i2i task's `name` slug, which can
        # differ from the dropdown stem (e.g. near_twins ships name = "sanitize").
        variant = self._current_variant()
        if self._is_descriptor_variant(variant):
            slug = variant
            try:
                doc = tomllib.loads(
                    (_TASK_DIR / f"{variant}.toml").read_text(encoding="utf-8")
                )
                slug = str(doc.get("name") or variant).strip()
            except (OSError, tomllib.TOMLDecodeError):
                pass
            return ROOT / "post_image_dataset" / "i2i" / slug / "cond"
        rel = self._CACHE_DIRS.get(variant, "post_image_dataset/i2i")
        return ROOT / rel

    def _ec_launch(self, argv: list[str], mode: str) -> None:
        if self._proc.is_running():
            return
        self.log.clear()
        self._reset_progress()
        self._progress_tracker.mark_starting(t("starting"))
        self._running_mode = mode
        adapter = self._ec_adapter()
        prefix = f"EASYADAPTER={adapter} " if adapter else ""
        self._log(f"> {prefix}python {' '.join(argv)}\n")
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_pickers_enabled(False)
        # Set or cleared every launch, so one variant's adapter never leaks into the next.
        self._proc.start(argv, env={"EASYADAPTER": adapter})

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
        """Train a descriptor variant as `make easycontrol EASYADAPTER=<variant>`
        does: the base easycontrol method with the descriptor's blueprint
        (--dataset_config sidecar) + [training] table as CLI overrides."""
        if not confirm_resumable_checkpoint(self, self._descriptor_merged(variant)):
            return
        from scripts.tasks.training import _easy_train_extra

        try:
            extra = _easy_train_extra(variant, [])
        except SystemExit as e:  # descriptor missing a [[datasets]] blueprint, etc.
            QMessageBox.warning(self, t("error"), str(e))
            return
        # The base easycontrol method (methods/ tree, not gui-methods) with the
        # descriptor argv appended.
        self._submit_training(
            lambda: gui_daemon.submit_training(
                method="easycontrol",
                preset=self._current_preset(),
                methods_subdir=None,
                extra=extra,
            ),
            logging_dir=self._descriptor_merged(variant).get("logging_dir"),
        )

    # Overridden here, not at the launch site, so re-attach on reopen is covered.
    def _attach_to_job(
        self, job_id: str, *, replay_log: bool, kind: str = "train"
    ) -> None:
        super()._attach_to_job(job_id, replay_log=replay_log, kind=kind)
        self.preprocess_btn.setEnabled(False)

    # Preprocess launches like Train, so it locks and unlocks with the pickers.
    def _set_pickers_enabled(self, enabled: bool) -> None:
        super()._set_pickers_enabled(enabled)
        self.preprocess_btn.setEnabled(enabled)

    def _try_reattach(self) -> None:
        """Re-bind to this family's daemon training job still running when the
        tab is constructed. Filtered by label so another tab's run (the daemon's
        active job is shared) is not hijacked; EasyControl preprocess is a direct
        child, so only train jobs apply."""
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
        self._reattach(job_id, kind="train")
