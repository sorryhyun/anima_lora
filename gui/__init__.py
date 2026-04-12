"""Anima LoRA — PySide6 GUI package."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import toml
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QWidget,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"
GRAFT_DIR = ROOT / "graft"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

PRESETS = {
    "Windows 8GB VRAM": "training_config_win8gb.toml",
    "Windows 16GB VRAM": "training_config_win16gb.toml",
    # FA4 presets are not supported yet (flash-attention-sm120 disabled)
    # "FA4 8GB VRAM": "training_config_fa4_8gb.toml",
    # "FA4 16GB VRAM": "training_config_fa4_16gb.toml",
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
        "blocks_to_swap", "torch_compile", "compile_mode", "trim_crossattn_kv",
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

_LOCKED_PERFORMANCE = {
    "Windows 8GB VRAM",
    "Windows 16GB VRAM",
    # "FA4 8GB VRAM",   # not supported yet
    # "FA4 16GB VRAM",  # not supported yet
}

# flash4 is not supported yet (flash-attention-sm120 disabled)
_ATTN_MODES = ["flex", "flash"]


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


# ── Public entry point ─────────────────────────────────────────


def main():
    from gui.app import main as _main
    _main()
