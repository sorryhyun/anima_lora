"""Pre-launch confirmation dialogs + checkpoint/cache discovery.

Resume prompts, cache reassurance popups, and the on-disk probes
(``find_resumable_checkpoint`` / ``count_preprocess_caches``) that decide
whether to show them.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import (
    QMessageBox,
    QWidget,
)

from gui.core.paths import ROOT
from gui.i18n import t

# Re-exported for ``gui/__init__.py``.
from library.io.cache_names import count_preprocess_caches  # noqa: F401


def confirm_resumable_checkpoint(parent: QWidget | None, merged: dict) -> bool:
    """Prompt when a checkpoint is on disk; return whether to launch.

    Yes = train.py auto-resumes, No = wipe the state dir + adapter sidecar so it
    starts fresh, Cancel = False. True without a prompt when nothing is resumable.
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
    # No → start fresh. On a failed delete, abort: launching would resume against the user's choice.
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
    """Tell the user existing preprocess caches will be reused, not deleted.
    True to proceed (also when there are no caches), False if cancelled.
    ``pe_encoder`` picks the PE sidecar variant counted.
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
    """Train-side cache confirmation: True to train on the existing cache,
    False if cancelled, None when no cache exists (caller auto-chains a
    preprocess run).

    ``require_pe`` (``use_repa`` on) makes the PE cache mandatory: a core cache
    without PE sidecars returns None, so REPA never trains without its target.
    ``pe_encoder`` must match the variant's ``repa_encoder``, or a fully cached
    run is misread as cache-missing.
    """
    counts = count_preprocess_caches(cache_dir, pe_encoder=pe_encoder)
    has_core = counts["latents"] > 0 or counts["te"] > 0
    # Preprocess skips latents/TE already on disk, so the chain only adds the PE pass.
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
    """``(state_dir, current_step)`` when ``checkpointing_epochs`` is set and
    ``<output_dir>/<output_name>-checkpoint-state/train_state.json`` is readable
    (the path ``AnimaCheckpointer.auto_resume`` picks up), else ``None``.

    ``current_step < max_train_steps`` is not checked: it depends on dataset
    size and train.py re-evaluates it at launch.
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
