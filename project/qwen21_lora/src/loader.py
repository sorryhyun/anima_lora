"""Qwen-Image-2.1 pipeline loading under a 16 GB VRAM budget.

The checkpoint is 33 GB in bf16 and its three heavy parts are wildly uneven:

    text_encoder  Qwen3-VL-8B       17.5 GB   <- larger than the whole card
    transformer   7B, 32 SS blocks  14.2 GB
    vae           AutoencoderKLQwenImage21   1.35 GB

diffusers' ``enable_model_cpu_offload`` moves one *whole* module at a time, so
its peak is the largest module — 17.5 GB, which OOMs a 16.3 GB 5070 Ti before
the first prompt is encoded. ``enable_sequential_cpu_offload`` fits but streams
every submodule every step, which is unusable for a 40-step sample.

Neither accelerate offload path works on this box (2026-09-21): ``cpu_offload``
and ``device_map="auto"`` both OOM at ~14.4 GB allocated, and the same wall is
hit whether the ``max_memory`` GPU budget is 7 GiB or 11 GiB — the weights an
``AlignDevicesHook`` pages in are never returned to CPU, so the card fills
regardless of the budget.

So both heavy modules run on the GPU under the trainer's own block swapper
(``library.runtime.offloading``, attached by ``blockswap.py``), which is what
``blocks_to_swap`` already does for Anima. Weights stay bf16 — nothing is
quantized.

The phase split is the same one ``train.py`` makes for Anima (text encode →
free → DiT): encode with the text encoder swapping, drop the 17.5 GB of it,
then give the card to the transformer + VAE. Callers that only want embeddings
(latent/TE caching) never pay for the transformer at all.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import torch

DEFAULT_MODEL_DIR = Path("/media/sorryhyun/data/anima_models/qwen_image_2.1")

# Rough bf16 footprints, for picking a strategy without loading anything.
_TRANSFORMER_GB = 14.3
_TEXT_ENCODER_GB = 17.6
_VAE_GB = 1.4


def free_vram_gb(device: int = 0) -> float:
    free, _total = torch.cuda.mem_get_info(device)
    return free / 1024**3


def empty_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


@dataclass(frozen=True)
class Plan:
    """Which strategy the card can afford."""

    name: str
    reason: str

    @property
    def resident_transformer(self) -> bool:
        return self.name in ("full", "split")


def choose_plan(device: int = 0, headroom_gb: float = 2.0) -> Plan:
    free = free_vram_gb(device)
    if free >= _TRANSFORMER_GB + _TEXT_ENCODER_GB + _VAE_GB + headroom_gb:
        return Plan("full", f"{free:.1f} GB free — everything resident")
    if free >= _TRANSFORMER_GB + headroom_gb:
        return Plan(
            "split",
            f"{free:.1f} GB free — transformer resident, text encoder streamed "
            "for the encode pass then dropped",
        )
    return Plan("sequential", f"{free:.1f} GB free — submodule streaming throughout")


TEXT_ENCODER_BLOCKS = "model.language_model.layers"
TRANSFORMER_BLOCKS = "transformer_blocks"


def load_text_encoder(
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    dtype: torch.dtype = torch.bfloat16,
    *,
    attn_implementation: str | None = None,
):
    """Qwen3-VL-8B on CPU — the caller block-swaps it onto the card.

    ``attn_implementation`` is transformers' own switch ("sdpa",
    "flash_attention_2", "eager"). The encode pass is one forward over a short
    right-padded prompt, so this is a small share of a generation; it falls back
    to the default rather than failing the run if the backend is unavailable.
    """
    from transformers import Qwen3VLForConditionalGeneration

    path = Path(model_dir) / "text_encoder"
    if attn_implementation:
        try:
            return Qwen3VLForConditionalGeneration.from_pretrained(
                path, dtype=dtype, attn_implementation=attn_implementation
            )
        except (ImportError, ValueError) as exc:
            print(
                f"text_encoder: {attn_implementation} unavailable ({exc}); "
                "falling back to the default implementation",
                flush=True,
            )
    return Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=dtype)


def place(
    model,
    blocks_path: str,
    device: torch.device,
    *,
    blocks_to_swap: int | None = None,
    supports_backward: bool = False,
    activation_reserve_gb: float = 2.5,
    label: str = "model",
    minimal_schedule: bool = True,
):
    """Move ``model`` onto ``device``, block-swapping only as much as needed.

    ``blocks_to_swap=None`` sizes the swap against what is actually free;
    0 keeps everything resident. Returns the ``Attached`` handle (or None) —
    the caller must ``detach()`` it to give the VRAM back.
    """
    import blockswap

    blocks = blockswap.find_blocks(model, blocks_path)
    if blocks_to_swap is None:
        blocks_to_swap = blockswap.auto_blocks_to_swap(
            model,
            blocks,
            free_vram_gb(),
            activation_reserve_gb=activation_reserve_gb,
        )
    per = blockswap.block_size_gb(blocks)
    resident = blockswap.resident_size_gb(model, blocks)
    print(
        f"{label}: {len(blocks)} blocks x {per:.2f} GB + {resident:.2f} GB resident; "
        f"swapping {blocks_to_swap}, "
        f"{resident + (len(blocks) - blocks_to_swap) * per:.2f} GB on card "
        f"({free_vram_gb():.2f} GB free)",
        flush=True,
    )

    attached, blocks = blockswap.attach(
        model,
        blocks_path,
        blocks_to_swap,
        device,
        supports_backward=supports_backward,
        minimal_schedule=minimal_schedule,
    )
    if attached is None:
        model.to(device)
        return None
    blockswap.to_device_except_blocks(model, blocks, device)
    attached.prepare()
    return attached


def load_pipeline(
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    dtype: torch.dtype = torch.bfloat16,
    *,
    components: tuple[str, ...] | None = None,
    text_encoder=None,
):
    """Build the pipeline on CPU. Nothing is moved to the GPU here.

    ``components`` keeps the named modules and passes ``None`` for the rest —
    ``("text_encoder",)`` loads 17.5 GB instead of 33 GB when all the caller
    wants is prompt embeddings. ``text_encoder`` injects an already-loaded (and
    possibly device-mapped) one instead of reading it off disk again.
    """
    from diffusers import QwenImage21Pipeline

    model_dir = Path(model_dir)
    if not (model_dir / "model_index.json").exists():
        raise FileNotFoundError(f"no Qwen-Image-2.1 checkpoint at {model_dir}")

    kwargs: dict[str, object] = {}
    if components is not None:
        # `processor` and `scheduler` are tiny config-only pieces; always keep
        # them so the pipeline can still build its prompt templates.
        for name in ("vae", "text_encoder", "transformer"):
            if name not in components:
                kwargs[name] = None
    if text_encoder is not None:
        kwargs["text_encoder"] = text_encoder
    return QwenImage21Pipeline.from_pretrained(model_dir, dtype=dtype, **kwargs)


def encode_prompts(
    pipe,
    prompts: list[str],
    *,
    device: str = "cuda",
    images: list | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]]:
    """Encode prompts and bring the results back to CPU.

    ``no_grad`` is what makes the encoder releasable, not just faster:
    ``encode_prompt`` does not disable grad itself, so the returned embedding
    carries a ``grad_fn`` chain whose saved tensors are the encoder's own
    ``Linear`` weights. Keeping the result then pins every resident block on
    the card — 9 GB that survives ``del``, ``gc.collect()`` and
    ``empty_cache()``, because ``SavedVariable`` holds them from C++ where no
    Python reference is visible. Diagnosed 2026-09-21; the giveaway was that a
    probe which *discarded* the embeddings released the memory cleanly.
    """
    out = []
    with torch.no_grad():
        for prompt in prompts:
            embeds, mask, pad_mask = pipe.encode_prompt(
                prompt=prompt, image=images, device=torch.device(device)
            )
            out.append(
                (
                    embeds.detach().to("cpu"),
                    None if mask is None else mask.detach().to("cpu"),
                    pad_mask.detach().to("cpu"),
                )
            )
    return out


def decode_latents(pipe, latents: torch.Tensor, height: int, width: int):
    """The tail of ``QwenImage21Pipeline.__call__``, run separately.

    Generating with ``output_type="latent"`` lets the caller unload the
    transformer before the VAE runs — the decoder is a video-style one with a
    feature cache and wants several GB to itself at 1024.
    """
    # Tiled: the decoder holds a feature cache across its 3D resnet stack, and
    # untiled at 1024 it asks for >1 GiB contiguous on top of ~12 GB already
    # allocated. Tiles keep the peak flat and the output is the same image.
    pipe.vae.enable_tiling()
    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = latents.to(pipe.vae.device, pipe.vae.dtype)
    shape = (1, pipe.vae.config.z_dim, 1, 1, 1)
    mean = torch.tensor(pipe.vae.config.latents_mean).view(shape).to(latents)
    std = torch.tensor(pipe.vae.config.latents_std).view(shape).to(latents)
    decoded = pipe.vae.decode(latents * std + mean, return_dict=False)[0][:, :, 0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")


def drop_text_encoder(pipe) -> None:
    """Release the 17.5 GB text encoder and whatever it held on the card."""
    te = pipe.text_encoder
    pipe.text_encoder = None
    pipe.register_to_config(text_encoder=None)
    del te
    empty_cache()
