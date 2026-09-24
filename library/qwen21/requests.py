"""Typed requests for the two Qwen-Image-2.1 jobs — caching and LoRA training.

Torch-free on purpose: the GUI builds its forms from these dataclasses and must
not pull torch in at launch. Each field carries its CLI help in ``metadata``, so
the sidecar CLIs (``scripts/qwen21/``) and the GUI read one definition:

    req = TrainRequest(cache="…", epochs=4)
    argv = req.to_argv()                 # → daemon job argv
    req = TrainRequest.from_argv(argv)   # ← sidecar CLI

``to_argv`` writes only fields that differ from their defaults, so a job record
reads as the choices that were made.

Model directory: ``--model_dir`` if given, else ``ANIMA_QWEN21_MODEL_DIR`` (env or
``.env``), else ``models/qwen_image_2.1`` under the repo home — the diffusers
folder layout (``transformer/``, ``text_encoder/``, ``vae/``, ``scheduler/``).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from library.env import load_dotenv, resolve_under_home

MODEL_DIR_ENV = "ANIMA_QWEN21_MODEL_DIR"
MODEL_DIR_NAME = "qwen_image_2.1"

# The single-stream block's linears, as a fullmatch against a module's
# qualified name. Attention projections plus the SwiGLU's three.
DEFAULT_TARGETS = (
    r"transformer_blocks\.\d+\.(attn\.(to_q|to_k|to_v|to_out\.0)"
    r"|img_mlp\.(proj|gate_layer|out))"
)

LORA_DTYPES = ("bf16", "fp16", "fp32")
COMPILE_SEQ_MODES = ("bounded", "dynamic")

# The default test prompt: a character the base model already knows, described
# in plain language the way Qwen-Image expects (not a tag list).
BOCCHI_PROMPT = (
    "An anime illustration of Hitori Gotoh (Bocchi) from Bocchi the Rock!: a shy "
    "teenage girl with very long pink hair, blue eyes, and yellow and blue "
    "cube-shaped hair clips on one side of her head. She wears a pink tracksuit "
    "jacket over a gray pleated skirt and holds a black electric guitar, glancing "
    "nervously at the viewer with a flustered blush under the colorful stage "
    "lights of a small live house."
)


def default_model_dir() -> Path:
    """``ANIMA_QWEN21_MODEL_DIR`` if set, else ``<repo home>/models/qwen_image_2.1``."""
    load_dotenv()
    override = os.environ.get(MODEL_DIR_ENV)
    if override:
        return resolve_under_home(override)
    return resolve_under_home(Path("models") / MODEL_DIR_NAME)


def resolve_model_dir(model_dir: str | Path | None) -> Path:
    """An explicit ``model_dir`` wins; empty/None falls back to the default."""
    if model_dir:
        return resolve_under_home(model_dir)
    return default_model_dir()


def _f(
    default: Any,
    help: str,
    *,
    choices: tuple | None = None,
    advanced=False,
    multiline=False,
):
    """A request field: ``help`` / ``choices`` / ``advanced`` / ``multiline``
    ride in metadata (the last two are GUI layout only)."""
    meta = {
        "help": help,
        "choices": choices,
        "advanced": advanced,
        "multiline": multiline,
    }
    return field(default=default, metadata=meta)


class _Request:
    """argv round-trip over the dataclass fields.

    Types map to argparse by the field's default: ``bool`` is a flag (``--x`` when
    the default is False, ``--no-x`` when it is True), ``None`` defaults are
    typed from the annotation, the rest from the default's own type.
    """

    SCRIPT: ClassVar[str]

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        ap = argparse.ArgumentParser(description=cls.__doc__)
        for f in dataclasses.fields(cls):
            flag = f"--{f.name}"
            help = f.metadata.get("help")
            if isinstance(f.default, bool):
                ap.add_argument(
                    flag,
                    action=argparse.BooleanOptionalAction,
                    default=f.default,
                    help=help,
                )
                continue
            kind = _field_type(f)
            ap.add_argument(
                flag,
                type=kind,
                default=f.default,
                choices=f.metadata.get("choices"),
                required=f.default is dataclasses.MISSING,
                help=help,
            )
        return ap

    @classmethod
    def from_argv(cls, argv: list[str] | None = None):
        return cls(**vars(cls.parser().parse_args(argv)))

    def to_argv(self) -> list[str]:
        argv: list[str] = []
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if f.default is not dataclasses.MISSING and value == f.default:
                continue
            if isinstance(value, bool):
                argv.append(f"--{f.name}" if value else f"--no-{f.name}")
            elif value is not None:
                argv += [f"--{f.name}", str(value)]
        return argv


def _field_type(f: dataclasses.Field) -> type:
    annotation = str(f.type)
    if "int" in annotation:
        return int
    if "float" in annotation:
        return float
    return str


@dataclass
class CacheRequest(_Request):
    """Precache Qwen3-VL text embeddings and VAE latents for a training folder."""

    SCRIPT: ClassVar[str] = "scripts/qwen21/cache.py"

    src: str = _f(
        "post_image_dataset/resized",
        "folder of images + .txt captions, subfolders included (default: the "
        "Anima resized tree, whose captions are the revised ones)",
    )
    out: str = _f("output/qwen21/cache", "cache folder (both files per image)")
    resolution: int = _f(
        1024,
        "target pixel area as a square edge — 1024 means ~1024^2 pixels at the "
        "image's own aspect ratio, not a 1024x1024 crop",
    )
    overwrite: bool = _f(False, "re-encode even when the cache file exists")
    skip_text: bool = _f(False, "skip the text-encoder pass")
    skip_latents: bool = _f(False, "skip the VAE pass")
    save_crops: bool = _f(
        True, "also write the resized images the latents were made from"
    )
    te_blocks_to_swap: int | None = _f(
        None, "text-encoder blocks to swap (default: sized to free VRAM)", advanced=True
    )
    model_dir: str = _f(
        "",
        f"diffusers folder (default: ${MODEL_DIR_ENV} or models/{MODEL_DIR_NAME})",
        advanced=True,
    )


@dataclass
class TrainRequest(_Request):
    """Train a LoRA on a precached folder — flow matching, batch 1, block swap."""

    SCRIPT: ClassVar[str] = "scripts/qwen21/train.py"

    cache: str = _f("output/qwen21/cache", "cache folder written by caching")
    output: str = _f(
        "output/qwen21/qwen21_lora.safetensors",
        "LoRA output path (a .json run report lands beside it)",
    )
    epochs: int = _f(8, "passes over the cache")
    rank: int = _f(16, "LoRA rank")
    alpha: float | None = _f(None, "LoRA alpha (default = rank)")
    lr: float = _f(1e-4, "AdamW learning rate")
    save_every_epochs: int = _f(0, "also save every N epochs (0 = final only)")
    warmup_ratio: float = _f(0.1, "linear warmup, as a fraction of all steps")
    max_grad_norm: float = _f(1.0, "gradient clip", advanced=True)
    lora_dtype: str = _f(
        "fp32",
        "adapter master-weight dtype (the rank GEMMs run in the model's bf16 "
        "either way); bf16 halves the adapter's optimizer state but rounds "
        "away small updates",
        choices=LORA_DTYPES,
    )
    targets: str = _f(
        DEFAULT_TARGETS, "fullmatch regex over linear names", advanced=True
    )
    logit_mean: float = _f(0.0, "logit-normal sigma sampling mean", advanced=True)
    logit_std: float = _f(1.0, "logit-normal sigma sampling std", advanced=True)
    blocks_to_swap: int | None = _f(
        None,
        "transformer blocks to swap (default: sized from activation_reserve_gb)",
    )
    activation_reserve_gb: float | None = _f(
        None,
        "VRAM kept free for activations when sizing the swap (default: measured "
        "0.3 GB + 0.6 MB x the largest image+text token count in the cache, plus "
        "one block of slack; 3.3 GB at 4096+346 tokens). Fewer swaps are not "
        "faster at 1024²",
        advanced=True,
    )
    grad_checkpointing: bool = _f(True, "activation checkpointing (the VRAM lever)")
    compile: bool = _f(
        False,
        "torch.compile each block; +-0 while the step is PCIe-bound (512², swap "
        "12), -13% where it is compute-bound (1024², swap 7, checkpointing)",
        advanced=True,
    )
    compile_seq: str = _f(
        "bounded",
        "how the joint token count goes symbolic: bounded = automatic dynamic + "
        "mark_dynamic over the cache's [min, max] joint tokens (hidden dims stay "
        "static); dynamic = torch.compile(dynamic=True)",
        choices=COMPILE_SEQ_MODES,
        advanced=True,
    )
    compile_mode: str | None = _f(None, "torch.compile mode", advanced=True)
    seed: int = _f(0, "RNG seed", advanced=True)
    model_dir: str = _f(
        "",
        f"diffusers folder (default: ${MODEL_DIR_ENV} or models/{MODEL_DIR_NAME})",
        advanced=True,
    )


@dataclass
class GenerateRequest(_Request):
    """Render prompts with and without a LoRA — same seed per prompt, so each
    pair differs by the adapter only."""

    SCRIPT: ClassVar[str] = "scripts/qwen21/generate.py"

    prompt: str = _f(
        BOCCHI_PROMPT, "prompt (ignored when prompts_file is set)", multiline=True
    )
    lora: str = _f("", "LoRA to test (empty = base model only)")
    multipliers: str = _f(
        "1.0,0.0", "comma-separated adapter scales; 0.0 is the base model"
    )
    width: int | None = _f(None, "multiple of 32 (default: resolution, square)")
    height: int | None = _f(None, "multiple of 32 (default: resolution, square)")
    steps: int = _f(20, "denoising steps")
    seed: int = _f(1234, "seed of the first prompt (+1 per prompt)")
    out_dir: str = _f("output/qwen21/test", "images + manifest.json land here")
    prompts_file: str = _f("", "one prompt per line; overrides prompt", advanced=True)
    resolution: int = _f(
        1024, "square edge when width/height are not given", advanced=True
    )
    true_cfg_scale: float = _f(
        1.0,
        "the pipeline default — 2.1 has no guidance embedding, >1 costs a second "
        "forward per step",
        advanced=True,
    )
    negative_prompt: str = _f("", "used only when true_cfg_scale > 1", advanced=True)
    blocks_to_swap: int | None = _f(
        None, "transformer blocks to swap (default: sized to free VRAM)", advanced=True
    )
    te_blocks_to_swap: int | None = _f(
        None, "text-encoder blocks to swap (default: sized to free VRAM)", advanced=True
    )
    model_dir: str = _f(
        "",
        f"diffusers folder (default: ${MODEL_DIR_ENV} or models/{MODEL_DIR_NAME})",
        advanced=True,
    )
