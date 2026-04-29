"""Anima LoRA Trainer ComfyUI nodes.

Two classes:

- ``AnimaLoRATrainer`` (simple): rank 16/32, gpu tier 8/16/high, IMAGE-or-dir
  dataset, and a `use_adapter` fallback for the train=off case. Locks the
  method to ``gui-methods/tlora.toml`` (T-LoRA + OrthoLoRA, no ReFT).

- ``AnimaLoRATrainerAdvanced``: adds method_variant / preset / lr / epochs /
  dim / warm-start overrides. Lets you pick any ``gui-methods/*.toml`` variant.

Both take a MODEL input and return (MODEL, STRING) — the STRING is the absolute
path of the saved safetensors on training runs, empty string otherwise.
"""

from __future__ import annotations

import datetime as _dt
import os

import folder_paths  # ComfyUI builtin

# Training deps are imported lazily inside `apply()` so that merely loading
# this module in ComfyUI doesn't force `library.*` imports (which pull torch
# extensions and slow startup).


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _anima_lora_root() -> str:
    from .training import find_anima_lora_root

    return find_anima_lora_root(os.path.dirname(__file__))


def _gui_method_variants() -> list[str]:
    """List stems of configs/gui-methods/*.toml files. Defensive: returns a
    minimal fallback if the directory is missing so INPUT_TYPES still works.
    """
    try:
        root = _anima_lora_root()
    except Exception:
        return ["tlora"]
    gui_dir = os.path.join(root, "configs", "gui-methods")
    if not os.path.isdir(gui_dir):
        return ["tlora"]
    stems = [
        os.path.splitext(n)[0]
        for n in sorted(os.listdir(gui_dir))
        if n.endswith(".toml")
    ]
    return stems or ["tlora"]


def _preset_names() -> list[str]:
    try:
        root = _anima_lora_root()
    except Exception:
        return ["default"]
    import toml

    presets_path = os.path.join(root, "configs", "presets.toml")
    try:
        with open(presets_path, "r", encoding="utf-8") as f:
            data = toml.load(f)
        return list(data.keys()) or ["default"]
    except Exception:
        return ["default"]


def _comfy_loras_dir() -> str:
    """Return the first directory registered under ComfyUI's 'loras' folder."""
    entry = folder_paths.folder_names_and_paths.get("loras")
    if not entry or not entry[0]:
        raise RuntimeError("ComfyUI has no 'loras' folder registered.")
    return entry[0][0]


# ---------------------------------------------------------------------------
# Core training-and-apply flow shared by both node classes
# ---------------------------------------------------------------------------


def _train_and_save(
    *,
    method: str,
    preset: str,
    overrides: dict,
    image,
    prompt: str,
    dataset_dir: str,
) -> str:
    """Run training end-to-end; return absolute path of the saved safetensors."""
    import comfy.model_management

    from .dataset_prep import prepare_dataset_dir
    from .training import (
        build_training_namespace,
        run_training,
    )

    root = _anima_lora_root()
    tmp_root = os.path.join(root, "output", "tmp_trainer")

    work_dir, dataset_cfg, n_images = prepare_dataset_dir(
        image, prompt, dataset_dir, tmp_root=tmp_root
    )

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"anima_trainer_{ts}"
    output_dir = _comfy_loras_dir()

    overrides = dict(overrides)
    overrides.setdefault("dataset_config", dataset_cfg)
    overrides.setdefault("output_dir", output_dir)
    overrides.setdefault("output_name", output_name)

    args = build_training_namespace(
        anima_lora_root=root,
        method=method,
        preset=preset,
        methods_subdir="gui-methods",
        overrides=overrides,
    )

    print(
        f"[Anima Trainer] method={method} preset={preset} "
        f"rank={getattr(args, 'network_dim', '?')} "
        f"epochs={getattr(args, 'max_train_epochs', '?')} "
        f"lr={getattr(args, 'learning_rate', '?')} "
        f"blocks_to_swap={getattr(args, 'blocks_to_swap', '?')} "
        f"images={n_images} → {output_name}.safetensors",
        flush=True,
    )

    # Free ComfyUI-held models so the trainer has room for its own DiT +
    # optimizer state. Training then runs in-process.
    comfy.model_management.unload_all_models()
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    saved_path = run_training(args, anima_lora_root=root)
    print(f"[Anima Trainer] saved {saved_path}", flush=True)
    return saved_path


def _apply_lora_to_model(model_clone, file_path: str, strength: float) -> None:
    """Apply an anima-family adapter (LoRA / Hydra / ReFT) to a cloned MODEL.

    Delegates to the sibling ``comfyui-hydralora`` custom node for the actual
    patching so there is one source of truth for key-sniffing + hook install.
    """
    # Import late so that the sibling node is discovered by ComfyUI's normal
    # custom_nodes loader (we don't add it to sys.path ourselves).
    try:
        from custom_nodes.__init__ import (  # noqa: F401  (may not exist)
            _marker,
        )
    except Exception:
        pass
    try:
        from custom_nodes.comfyui_hydralora.adapter import apply_adapter
    except ImportError:
        # ComfyUI often registers custom nodes under module names with a hyphen
        # -> underscore swap is not automatic. Fall back to a direct path import.
        import importlib.util

        sibling = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "comfyui-hydralora",
            "adapter.py",
        )
        if not os.path.exists(sibling):
            raise RuntimeError(
                "Cannot apply adapter: sibling `comfyui-hydralora/adapter.py` "
                f"not found at {sibling}. Install that custom node alongside "
                "this one, or patch the resulting LoRA manually via KSampler."
            )
        spec = importlib.util.spec_from_file_location(
            "_anima_trainer_adapter", sibling
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        apply_adapter = mod.apply_adapter
    apply_adapter(model_clone, file_path, strength, strength)


# ---------------------------------------------------------------------------
# Simple node
# ---------------------------------------------------------------------------


class AnimaLoRATrainer:
    """Train an Anima LoRA (T-LoRA + OrthoLoRA) and apply it to MODEL.

    If ``train`` is off, acts as a pure adapter-loader: optionally patches the
    MODEL with a user-selected existing LoRA.
    """

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        if not loras:
            loras = [""]
        return {
            "required": {
                "model": ("MODEL",),
                "train": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "train new",
                        "label_off": "use existing",
                    },
                ),
                "rank": (["16", "32"], {"default": "16"}),
                "gpu": (["8GB", "16GB", "high"], {"default": "16GB"}),
                "dataset_dir": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Directory of images + matching .txt caption "
                            "sidecars. Ignored when an IMAGE socket is connected."
                        ),
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Caption used when an IMAGE is connected "
                            "(single-image mode)."
                        ),
                    },
                ),
                "use_adapter": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "When `train` is off, apply the selected adapter "
                            "to MODEL."
                        ),
                    },
                ),
                "adapter": (
                    loras,
                    {
                        "tooltip": (
                            "Existing Anima adapter (LoRA / Hydra / ReFT). "
                            "Used only when `train` is off and `use_adapter` is on."
                        )
                    },
                ),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05},
                ),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "lora_path")
    FUNCTION = "apply"
    CATEGORY = "anima/training"
    DESCRIPTION = (
        "Train an Anima T-LoRA + OrthoLoRA from a single image+prompt or a "
        "directory with .txt sidecars, then apply the trained LoRA to MODEL. "
        "With `train` off, optionally applies a pre-existing adapter. "
        "Training runs in-process and blocks the workflow until done; watch "
        "the ComfyUI console for progress."
    )

    def apply(
        self,
        model,
        train: bool,
        rank: str,
        gpu: str,
        dataset_dir: str,
        prompt: str,
        use_adapter: bool,
        adapter: str,
        strength: float,
        image=None,
    ):
        new_model = model.clone()

        if not train:
            if use_adapter and adapter:
                path = folder_paths.get_full_path("loras", adapter)
                _apply_lora_to_model(new_model, path, strength)
                return (new_model, "")
            return (new_model, "")

        from .training import gpu_tier_overrides, rank_overrides

        overrides: dict = {}
        overrides.update(rank_overrides(int(rank)))
        overrides["max_train_epochs"] = 25
        overrides.update(gpu_tier_overrides(gpu))

        saved_path = _train_and_save(
            method="tlora",
            preset={"8GB": "low_vram", "16GB": "default", "high": "32gb"}[gpu],
            overrides=overrides,
            image=image,
            prompt=prompt,
            dataset_dir=dataset_dir,
        )
        _apply_lora_to_model(new_model, saved_path, strength)
        return (new_model, saved_path)


# ---------------------------------------------------------------------------
# Advanced node
# ---------------------------------------------------------------------------


class AnimaLoRATrainerAdvanced:
    """Advanced trainer: exposes method / preset / hyperparameter overrides."""

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras") or [""]
        variants = _gui_method_variants()
        presets = _preset_names()
        return {
            "required": {
                "model": ("MODEL",),
                "train": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "train new",
                        "label_off": "use existing",
                    },
                ),
                "method_variant": (
                    variants,
                    {"default": "tlora" if "tlora" in variants else variants[0]},
                ),
                "preset": (
                    presets,
                    {
                        "default": "default"
                        if "default" in presets
                        else presets[0]
                    },
                ),
                "dataset_dir": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "learning_rate": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.00001,
                        "tooltip": "0 = use method's default",
                    },
                ),
                "max_train_epochs": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10000,
                        "tooltip": "0 = use method's default",
                    },
                ),
                "network_dim": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 256,
                        "tooltip": "0 = use method's default",
                    },
                ),
                "blocks_to_swap": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 60,
                        "tooltip": "-1 = use preset's value",
                    },
                ),
                "warm_start": ("BOOLEAN", {"default": False}),
                "warm_start_adapter": (loras,),
                "use_adapter": ("BOOLEAN", {"default": False}),
                "adapter": (loras,),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05},
                ),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "lora_path")
    FUNCTION = "apply"
    CATEGORY = "anima/training"

    def apply(
        self,
        model,
        train: bool,
        method_variant: str,
        preset: str,
        dataset_dir: str,
        prompt: str,
        learning_rate: float,
        max_train_epochs: int,
        network_dim: int,
        blocks_to_swap: int,
        warm_start: bool,
        warm_start_adapter: str,
        use_adapter: bool,
        adapter: str,
        strength: float,
        image=None,
    ):
        new_model = model.clone()

        if not train:
            if use_adapter and adapter:
                path = folder_paths.get_full_path("loras", adapter)
                _apply_lora_to_model(new_model, path, strength)
            return (new_model, "")

        overrides: dict = {}
        if learning_rate > 0:
            overrides["learning_rate"] = float(learning_rate)
        if max_train_epochs > 0:
            overrides["max_train_epochs"] = int(max_train_epochs)
        if network_dim > 0:
            overrides["network_dim"] = int(network_dim)
            overrides["network_alpha"] = float(network_dim)
        if blocks_to_swap >= 0:
            overrides["blocks_to_swap"] = int(blocks_to_swap)
        if warm_start and warm_start_adapter:
            overrides["network_weights"] = folder_paths.get_full_path(
                "loras", warm_start_adapter
            )
            overrides["dim_from_weights"] = True

        saved_path = _train_and_save(
            method=method_variant,
            preset=preset,
            overrides=overrides,
            image=image,
            prompt=prompt,
            dataset_dir=dataset_dir,
        )
        _apply_lora_to_model(new_model, saved_path, strength)
        return (new_model, saved_path)


NODE_CLASS_MAPPINGS = {
    "AnimaLoRATrainer": AnimaLoRATrainer,
    "AnimaLoRATrainerAdvanced": AnimaLoRATrainerAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaLoRATrainer": "Anima LoRA Trainer",
    "AnimaLoRATrainerAdvanced": "Anima LoRA Trainer (Advanced)",
}
