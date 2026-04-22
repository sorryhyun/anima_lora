"""Anima LoRA — PySide6 GUI package."""

from __future__ import annotations

import json
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

METHODS_DIR = CONFIGS_DIR / "methods"
GUI_METHODS_DIR = CONFIGS_DIR / "gui-methods"
PRESETS_FILE = CONFIGS_DIR / "presets.toml"


_METHOD_ORDER = ("lora", "postfix", "apex", "graft")

# GUI variant picker maps method families → self-contained gui-methods files.
# Order is display order in the variant combo. Any gui-methods/*.toml not
# listed here is attached to its best-guess family by prefix.
_FAMILY_VARIANTS: dict[str, list[str]] = {
    "lora": [
        "lora",
        "lora-fast",
        "lora-8gb",
        "ortholora",
        "tlora",
        "reft",
        "tlora_ortho_reft",
        "hydralora",
        "hydralora_sigma",
    ],
    "postfix": [
        "postfix",
        "postfix_exp",
        "postfix_func",
        "postfix_sigma",
        "prefix",
    ],
    "apex": ["apex"],
    "graft": ["graft"],
}


def list_methods() -> list[str]:
    """Method families, in a user-friendly order (lora first)."""
    return list(_METHOD_ORDER)


def list_gui_variants(method: str) -> list[str]:
    """gui-methods/*.toml files that belong to the given method family.

    The GUI training path uses gui-methods as the source of truth (each variant
    is a self-contained TOML — no toggle blocks), so the variant combo lists
    actual files, not overlay presets.
    """
    if not GUI_METHODS_DIR.exists():
        return []
    have = {p.stem for p in GUI_METHODS_DIR.glob("*.toml")}
    want = _FAMILY_VARIANTS.get(method, [])
    ordered = [v for v in want if v in have]
    return ordered


def _load_all_presets() -> dict:
    if not PRESETS_FILE.exists():
        return {}
    data = toml.loads(PRESETS_FILE.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def list_presets() -> list[str]:
    return sorted(_load_all_presets())


_GROUPS = {
    "Architecture": {
        "network_dim",
        "network_alpha",
        "network_module",
        "network_args",
        "use_ortho",
        "use_timestep_mask",
        "use_hydra",
        "add_reft",
        "use_sigma_router",
        "min_rank",
        "alpha_rank_scale",
        "num_experts",
        "balance_loss_weight",
        "balance_loss_warmup_ratio",
        "reft_dim",
        "reft_alpha",
        "reft_layers",
        "sigma_feature_dim",
        "sigma_hidden_dim",
        "sigma_router_layers",
        "per_bucket_balance_weight",
        "num_sigma_buckets",
        "network_train_unet_only",
    },
    "Training": {
        "learning_rate",
        "max_train_epochs",
        "save_every_n_epochs",
        "checkpointing_epochs",
        "gradient_accumulation_steps",
        "caption_shuffle_variants",
        "optimizer_type",
        "lr_scheduler",
        "timestep_sampling",
        "discrete_flow_shift",
    },
    "Performance": {
        "attn_mode",
        "gradient_checkpointing",
        "unsloth_offload_checkpointing",
        "blocks_to_swap",
        "torch_compile",
        "compile_mode",
        "trim_crossattn_kv",
        "cache_llm_adapter_outputs",
        "masked_loss",
        "mixed_precision",
        "static_token_count",
        "vae_chunk_size",
        "vae_disable_cache",
        "cache_latents",
        "cache_latents_to_disk",
        "cache_text_encoder_outputs",
        "cache_text_encoder_outputs_to_disk",
        "skip_cache_check",
        "layer_start"
    },
    "Paths": {
        "pretrained_model_name_or_path",
        "qwen3",
        "vae",
        "output_dir",
        "output_name",
        "save_model_as",
    },
}
_K2G = {k: g for g, ks in _GROUPS.items() for k in ks}
_SKIP = {"base_config", "dataset_config", "general", "datasets"}

# flash4 is not supported yet (flash-attention-sm120 disabled)
_ATTN_MODES = ["flex", "flash"]


# ── Helpers ────────────────────────────────────────────────────


def _load(p: Path) -> dict:
    return toml.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save(p: Path, d: dict):
    p.write_text(toml.dumps(d), encoding="utf-8")


def merged_method_preset(method: str, preset: str) -> tuple[dict, dict[str, str]]:
    """Return (merged_dict, origin_map). origin_map[key] is 'base' | 'preset' | 'method'."""
    base = _load(CONFIGS_DIR / "base.toml")
    pset = _load_all_presets().get(preset, {})
    meth = _load(METHODS_DIR / f"{method}.toml")
    merged: dict = {}
    origin: dict[str, str] = {}
    for k, v in base.items():
        merged[k] = v
        origin[k] = "base"
    for k, v in pset.items():
        merged[k] = v
        origin[k] = "preset"
    for k, v in meth.items():
        merged[k] = v
        origin[k] = "method"
    return merged, origin


def merged_gui_variant_preset(
    variant: str, preset: str
) -> tuple[dict, dict[str, str]]:
    """Merge base + preset + gui-methods/<variant>.toml. The GUI uses this
    instead of `merged_method_preset` so edits/training target the clean
    per-variant file, not the toggle-block methods/ tree."""
    base = _load(CONFIGS_DIR / "base.toml")
    pset = _load_all_presets().get(preset, {})
    meth = _load(GUI_METHODS_DIR / f"{variant}.toml")
    merged: dict = {}
    origin: dict[str, str] = {}
    for k, v in base.items():
        merged[k] = v
        origin[k] = "base"
    for k, v in pset.items():
        merged[k] = v
        origin[k] = "preset"
    for k, v in meth.items():
        merged[k] = v
        origin[k] = "method"
    return merged, origin


def _imgs(d: Path) -> list[Path]:
    return (
        sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if d.exists()
        else []
    )


def _safetensors_in(d: Path) -> list[Path]:
    """Return .safetensors files in a directory, newest first."""
    if not d.exists():
        return []
    return sorted(
        (p for p in d.iterdir() if p.is_file() and p.suffix == ".safetensors"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _adapter_dirs() -> dict[str, Path]:
    """Directories likely to contain LoRA adapter checkpoints.

    Mirrors ``_image_dirs``: returns only paths that exist and actually have
    .safetensors files, keyed by a short display name.
    """
    dirs: dict[str, Path] = {}
    for name, path in [
        ("output/ckpt", ROOT / "output" / "ckpt"),
        ("output_temp", ROOT / "output_temp"),
        ("models/diffusion_models", ROOT / "models" / "diffusion_models"),
    ]:
        if path.exists() and any(path.glob("*.safetensors")):
            dirs[name] = path
    # Any subdirectory of output/ckpt/ or output_temp/ with .safetensors (e.g.
    # iteration snapshots). Skip *-checkpoint-state dirs — those are
    # optimizer/state shards, not adapters.
    for parent, label in (
        (ROOT / "output" / "ckpt", "output/ckpt"),
        (ROOT / "output_temp", "output_temp"),
    ):
        if not parent.exists():
            continue
        for p in sorted(parent.iterdir()):
            if (
                p.is_dir()
                and not p.name.endswith("-checkpoint-state")
                and any(p.glob("*.safetensors"))
            ):
                dirs[f"{label}/{p.name}"] = p
    # GRAFT survivors / candidate iterations sometimes hold adapter artifacts.
    graft_dir = GRAFT_DIR
    for candidate in ("survivors",):
        p = graft_dir / candidate
        if p.exists() and any(p.glob("*.safetensors")):
            dirs[f"graft/{candidate}"] = p
    cd = graft_dir / "candidates"
    if cd.exists():
        for p in sorted(cd.iterdir()):
            if p.is_dir() and any(p.glob("*.safetensors")):
                dirs[f"graft/candidates/{p.name}"] = p
    return dirs


def _image_dirs() -> dict[str, Path]:
    dirs: dict[str, Path] = {}
    for name, path in [
        ("image_dataset", ROOT / "image_dataset"),
        ("post_image_dataset", ROOT / "post_image_dataset"),
        ("output/tests", ROOT / "output" / "tests"),
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
                self._src.scaled(
                    self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )


# ── Public entry point ─────────────────────────────────────────


def main():
    from gui.app import main as _main

    _main()
