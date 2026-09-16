"""Pre-launch confirmation dialogs + checkpoint/cache discovery.

The Qt-facing half of the train/preprocess launch flow: resume prompts, cache
reassurance popups, and the on-disk probes (``find_resumable_checkpoint`` /
``count_preprocess_caches``) that decide whether to show them. Kept apart from
the Qt-free ``gui.config_io`` so that module doesn't pull QMessageBox in.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from gui._paths import ROOT
from gui.i18n import current_language, t

# Cache discovery lives in the torch-free leaf library/io/cache_names.py, shared with
# the preprocess pipeline. Re-exported for ``gui/__init__.py``.
from library.io.cache_names import count_preprocess_caches  # noqa: F401

_GUIDELINES = ROOT / "docs" / "guidelines"
_GUIDEBOOK_BY_LANG: dict[str, Path] = {
    "en": _GUIDELINES / "guidebook.md",
    "ko": _GUIDELINES / "가이드북.md",
    "cn": _GUIDELINES / "指南书.md",
    "ja": _GUIDELINES / "ガイドブック.md",
}
_GUIDEBOOK_FALLBACK = _GUIDEBOOK_BY_LANG["en"]


def _guidebook_path() -> Path:
    return _GUIDEBOOK_BY_LANG.get(current_language(), _GUIDEBOOK_FALLBACK)


class GuidebookDialog(QDialog):
    """In-app markdown viewer for the guidebook."""

    def __init__(self, md_path: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("guidebook"))
        self.resize(900, 720)
        self._md_path = md_path

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setSearchPaths([str(md_path.parent)])
        self.browser.document().setBaseUrl(
            QUrl.fromLocalFile(str(md_path.parent) + "/")
        )
        # Default anchor color is pure blue — illegible on the dark bg.
        self.browser.document().setDefaultStyleSheet(
            "a { color: #ffb86b; text-decoration: underline; }"
            "a:visited { color: #e6944e; }"
            "code { background:#2a2a2a; padding:1px 4px; border-radius:3px; }"
            "pre { background:#2a2a2a; padding:8px; border-radius:4px; }"
        )
        self.browser.setStyleSheet(
            "QTextBrowser { background:#1e1e1e; color:#dcdcdc; "
            "border:1px solid #444; padding:12px; }"
        )
        try:
            text = md_path.read_text(encoding="utf-8")
        except OSError as e:
            text = f"# Error\n\nCould not read `{md_path}`:\n\n`{e}`"
        self.browser.setMarkdown(text)
        lay.addWidget(self.browser)

        btn_bar = QHBoxLayout()
        btn_bar.addStretch()
        open_ext = QPushButton(t("guidebook_open_external"))
        open_ext.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._md_path)))
        )
        close = QPushButton(t("guidebook_close"))
        close.clicked.connect(self.close)
        btn_bar.addWidget(open_ext)
        btn_bar.addWidget(close)
        lay.addLayout(btn_bar)


def confirm_resumable_checkpoint(parent: QWidget | None, merged: dict) -> bool:
    """Prompt the user when a checkpoint is on disk; return whether to launch.

    Returns True if training should proceed (Yes = let train.py auto-resume,
    No = wipe the state dir + adapter sidecar so train.py starts fresh),
    False if the user cancelled. Returns True with no prompt when there is
    nothing to resume from — the call site can wrap every train launch in
    this helper unconditionally.
    """
    found = find_resumable_checkpoint(merged)
    if found is None:
        return True
    state_dir, step = found
    choice = QMessageBox.question(
        parent,
        t("resume_checkpoint_title"),
        t("resume_checkpoint_question", step=step),
        QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
        QMessageBox.Yes,
    )
    if choice == QMessageBox.Cancel:
        return False
    if choice == QMessageBox.Yes:
        return True
    # No → start fresh: wipe the state dir + sibling adapter so train.py's auto_resume
    # sees nothing. Bail on failure rather than silently resuming against the user's choice.
    import shutil

    sidecar = state_dir.parent / f"{state_dir.name.removesuffix('-state')}.safetensors"
    try:
        shutil.rmtree(state_dir)
        if sidecar.is_file():
            sidecar.unlink()
    except OSError as e:
        QMessageBox.warning(
            parent,
            t("error"),
            t("resume_checkpoint_delete_failed", error=str(e)),
        )
        return False
    return True


def confirm_existing_caches(
    parent: QWidget | None,
    cache_dir: Path,
    require_pe: bool = False,
    pe_encoder: str | None = None,
) -> bool:
    """Reassure the user that existing preprocess caches will be reused, not
    deleted. Returns True to proceed, False if the user cancelled.

    No-op (returns True without prompting) when the cache directory is empty
    or missing, so the call site can wrap every preprocess launch in this.

    ``pe_encoder`` selects which PE sidecar variant is counted (defaults to the
    REPA default ``pe_spatial``) — see :func:`count_preprocess_caches`.
    """
    counts = count_preprocess_caches(cache_dir, pe_encoder=pe_encoder)
    has_any = (
        counts["latents"] > 0 or counts["te"] > 0 or (require_pe and counts["pe"] > 0)
    )
    if not has_any:
        return True

    parts: list[str] = []
    if counts["latents"]:
        parts.append(t("preprocess_cache_count_latents", n=counts["latents"]))
    if counts["te"]:
        parts.append(t("preprocess_cache_count_te", n=counts["te"]))
    if require_pe and counts["pe"]:
        parts.append(t("preprocess_cache_count_pe", n=counts["pe"]))

    body = t(
        "preprocess_existing_caches_body",
        cache_dir=str(cache_dir),
        items="  • " + "\n  • ".join(parts),
    )
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Information)
    box.setWindowTitle(t("preprocess_existing_caches_title"))
    box.setText(body)
    box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    box.setDefaultButton(QMessageBox.Ok)
    return box.exec() == QMessageBox.Ok


def confirm_train_using_cache(
    parent: QWidget | None,
    cache_dir: Path,
    require_pe: bool = False,
    pe_encoder: str | None = None,
) -> bool | None:
    """Train-side cache confirmation: returns True to launch training against
    the existing cache, False if the user cancelled, or None when no cache was
    found on disk (caller should auto-chain a preprocess run instead).

    Distinct from ``confirm_existing_caches`` (which reassures during
    Preprocess) — this gates Train and exposes the empty-cache case as a
    separate ``None`` so the caller can branch into the auto-preprocess flow.

    ``require_pe`` (set when ``use_repa`` is on) makes the PE feature cache
    mandatory: a core latent/TE cache that lacks PE sidecars still returns
    ``None`` so the caller auto-chains a (PE-caching) preprocess pass, rather
    than launching a REPA run whose alignment target is silently absent. This
    is the common "preprocessed before enabling REPA" case. ``pe_encoder`` must
    match the variant's ``repa_encoder`` (defaults to ``pe_spatial``) so the PE
    sidecars REPA will actually read are the ones we look for — otherwise a
    fully-cached PE-Spatial run is misread as cache-missing.
    """
    counts = count_preprocess_caches(cache_dir, pe_encoder=pe_encoder)
    has_core = counts["latents"] > 0 or counts["te"] > 0
    # REPA on + a built core cache but no PE sidecars → treat as cache-missing
    # so Train rebuilds the PE caches. preprocess is idempotent (it skips the
    # latents/TE already on disk), so this only adds the missing PE pass.
    if require_pe and has_core and counts["pe"] == 0:
        return None
    has_any = has_core or (require_pe and counts["pe"] > 0)
    if not has_any:
        return None

    parts: list[str] = []
    if counts["latents"]:
        parts.append(t("preprocess_cache_count_latents", n=counts["latents"]))
    if counts["te"]:
        parts.append(t("preprocess_cache_count_te", n=counts["te"]))
    if require_pe and counts["pe"]:
        parts.append(t("preprocess_cache_count_pe", n=counts["pe"]))

    body = t(
        "train_using_cache_body",
        cache_dir=str(cache_dir),
        items="  • " + "\n  • ".join(parts),
    )
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Question)
    box.setWindowTitle(t("train_using_cache_title"))
    box.setText(body)
    box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
    box.setDefaultButton(QMessageBox.Yes)
    return box.exec() == QMessageBox.Yes


def find_resumable_checkpoint(merged: dict) -> tuple[Path, int] | None:
    """If the merged config has a writable ``checkpointing_epochs`` and an
    on-disk checkpoint state directory exists with a usable ``train_state.json``,
    return ``(state_dir, current_step)``. Returns ``None`` when there is
    nothing to resume — that's the common case and callers should treat it as
    "just launch training normally".

    Mirrors ``library.training.checkpoints.AnimaCheckpointer.auto_resume``: the
    same ``<output_dir>/<output_name>-checkpoint-state/`` path that ``train.py``
    would auto-pick up. We deliberately do NOT enforce ``current_step <
    max_train_steps`` here — that check varies with dataset size and is
    re-evaluated at launch; the GUI prompt only needs to know "is there
    something on disk that train.py would consider resumable".
    """
    if not merged.get("checkpointing_epochs"):
        return None
    output_dir = merged.get("output_dir")
    output_name = merged.get("output_name") or "last"
    if not output_dir:
        return None
    state_dir = ROOT / output_dir / f"{output_name}-checkpoint-state"
    train_state_file = state_dir / "train_state.json"
    if not train_state_file.is_file():
        return None
    try:
        data = json.loads(train_state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    step = int(data.get("current_step", 0))
    return state_dir, step
