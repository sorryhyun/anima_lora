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

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from library.env import resolve_under_home
from library.lines.requests import Request, env_model_dir
from library.lines.requests import request_field as _f
from library.lines.requests import resolve_model_dir as _resolve_model_dir

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
    return env_model_dir(
        MODEL_DIR_ENV, resolve_under_home(Path("models") / MODEL_DIR_NAME)
    )


def resolve_model_dir(model_dir: str | Path | None) -> Path:
    return _resolve_model_dir(model_dir, default_model_dir)


@dataclass
class CacheRequest(Request):
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
class TrainRequest(Request):
    """Train a LoRA on a precached folder — flow matching, batch 1, block swap."""

    SCRIPT: ClassVar[str] = "scripts/qwen21/train.py"

    cache: str = _f("output/qwen21/cache", "cache folder written by caching")
    output: str = _f(
        "output/qwen21/qwen21_lora.safetensors",
        "LoRA output path (a .json run report lands beside it)",
    )
    epochs: int = _f(8, "passes over the cache")
    algo: str = _f(
        "lora",
        "adapter: lora, or LyCORIS lokr (Kronecker product) / loha (Hadamard "
        "product of two rank-r pairs)",
        choices=("lora", "lokr", "loha"),
    )
    rank: int = _f(16, "LoRA rank (lokr: rank of the larger Kronecker factor)")
    alpha: float | None = _f(None, "LoRA alpha (default = rank)")
    factor: int = _f(
        -1,
        "lokr only: cap on the small Kronecker factor's side (-1 = near-square split)",
        advanced=True,
    )
    down_init: str = _f(
        "kaiming",
        "lora only, down-projection init: kaiming or weight_svd (the base "
        "weight's top-r right singular vectors / sqrt 3; MOSS's default, not "
        "measured on Qwen-Image-2.1)",
        choices=("kaiming", "weight_svd"),
        advanced=True,
    )
    lr: float = _f(1e-4, "AdamW learning rate")
    save_every_epochs: int = _f(0, "also save every N epochs (0 = final only)")
    warmup_ratio: float = _f(0.1, "linear warmup, as a fraction of all steps")
    max_grad_norm: float = _f(1.0, "gradient clip", advanced=True)
    lora_dtype: str = _f(
        "bf16",
        "dtype the saved LoRA is written in. Training always keeps fp32 master "
        "weights and runs the rank GEMMs in the model's bf16",
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
        True,
        "torch.compile each block: -11% per step where the step is compute-bound "
        "(1024², swap 7, checkpointing), +-0 where it is PCIe-bound (512², swap "
        "12); ~40 s of compile up front. --no-compile for a quick smoke",
        advanced=True,
    )
    compile_seq: str = _f(
        "dynamic",
        "how the joint token count goes symbolic: dynamic = "
        "torch.compile(dynamic=True), one graph for every sample; bounded = "
        "automatic dynamic + mark_dynamic over the cache's [min, max] joint "
        "tokens (hidden dims stay static). Same speed; bounded pays its compile "
        "as 17 s + one 21 s recompile instead of 42 s up front",
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
class GenerateRequest(Request):
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
