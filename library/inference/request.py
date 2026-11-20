"""``GenerationRequest`` — a typed front door for a single generation call.

``generate(args, gen_settings)`` reads ~40 fields off an ``argparse.Namespace``
via ``getattr``; this dataclass builds that namespace::

    from anima_lora import GenerationRequest, generate, get_generation_settings

    req = GenerationRequest(
        prompt="a red fox in a snowy forest",
        dit="models/diffusion_models/anima-base-v1.0.safetensors",
        vae="models/vae/qwen_image_vae.safetensors",
        text_encoder="models/text_encoders/qwen_3_06b_base.safetensors",
        seed=42,
    )
    args = req.to_args()                       # fully-defaulted Namespace
    args.device = "cuda"
    latent = generate(args, get_generation_settings(args))

``to_args()`` runs ``to_argv()`` through ``library.inference.args.build_default_args``,
so the parser owns every default; this dataclass carries the common knobs and
the long tail rides through ``extra_argv``.

One request is safe to reuse (build a fresh namespace per call). With ``seed``
unset, ``generate()`` draws a random seed per call without storing it; call
``resolve_seed(args)`` yourself if you need it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple, Union

# The parser requires --save_path; the placeholder lets non-saving requests
# build a namespace. An unset dit / text_encoder / vae takes the catalog default.
_PLACEHOLDER_SAVE_PATH = "__unset_save_path__"


@dataclass(frozen=True)
class GenerationRequest:
    """A single text-to-image generation request.

    Fields with concrete defaults mirror ``inference.parse_args`` (pinned by
    ``test_generation_request``); ``None`` fields are omitted so the parser's
    default wins.
    """

    prompt: Optional[str] = None
    negative_prompt: str = ""

    image_size: Tuple[int, int] = (1024, 1024)  # (height, width), as --image_size
    infer_steps: int = 50
    guidance_scale: float = 3.5
    flow_shift: float = 3.0
    sigma_tail_power: float = 1.0
    sampler: str = "euler"
    seed: Optional[int] = None

    lora_weight: Optional[Sequence[str]] = None
    lora_multiplier: Optional[Union[float, Sequence[float]]] = None
    soft_tokens_weight: Optional[str] = None
    easycontrol_weight: Optional[str] = None
    easycontrol_image: Optional[str] = None
    pooled_text_proj: Optional[str] = None
    # CJK vocab pack path prefix (``library.anima.vocab_pack``). ``None`` =
    # the `vocab_pack` key in configs/base.toml; ``no_vocab_pack`` forces off.
    vocab_pack: Optional[str] = None
    no_vocab_pack: bool = False
    stream: Optional[str] = None  # ``fp32`` | ``bf16``

    dit: Optional[str] = None
    vae: Optional[str] = None
    # ``pixel_head`` decodes with the pixel head (library.anima.pixel_head) instead
    # of the VAE; ``pixel_head`` names the file (default: the catalog row);
    # ``pixel_scale`` 2 reads a ×2 head file's 2× output.
    decoder: Optional[str] = None  # ``vae`` | ``pixel_head``
    pixel_head: Optional[str] = None
    pixel_scale: Optional[int] = None  # 1 | 2
    text_encoder: Optional[str] = None
    device: Optional[str] = None
    attn_mode: str = "torch"
    vae_chunk_size: Optional[int] = None
    vae_disable_cache: bool = False
    # 2D-fold the VAE for ~2x faster decode; set False for the stock 3D path.
    vae_2d: bool = True
    text_encoder_cpu: bool = False

    save_path: Optional[str] = None
    output_type: str = "images"
    no_metadata: bool = False

    # verbatim CLI tokens (e.g. ["--spectrum"]); override the same flag above
    extra_argv: Sequence[str] = field(default_factory=tuple)

    def to_argv(self) -> list[str]:
        """Render as CLI tokens; ``extra_argv`` goes last so it overrides."""
        argv: list[str] = []

        if self.text_encoder is not None:
            argv += ["--text_encoder", self.text_encoder]
        argv += ["--save_path", self.save_path or _PLACEHOLDER_SAVE_PATH]

        argv += ["--negative_prompt", self.negative_prompt]
        argv += ["--image_size", str(self.image_size[0]), str(self.image_size[1])]
        argv += ["--infer_steps", str(self.infer_steps)]
        argv += ["--guidance_scale", str(self.guidance_scale)]
        argv += ["--flow_shift", str(self.flow_shift)]
        argv += ["--sigma_tail_power", str(self.sigma_tail_power)]
        argv += ["--sampler", self.sampler]
        argv += ["--attn_mode", self.attn_mode]
        argv += ["--output_type", self.output_type]

        if self.prompt is not None:
            argv += ["--prompt", self.prompt]
        if self.seed is not None:
            argv += ["--seed", str(self.seed)]
        if self.dit is not None:
            argv += ["--dit", self.dit]
        if self.vae is not None:
            argv += ["--vae", self.vae]
        if self.decoder is not None:
            argv += ["--decoder", self.decoder]
        if self.pixel_head is not None:
            argv += ["--pixel_head", self.pixel_head]
        if self.pixel_scale is not None:
            argv += ["--pixel_scale", str(self.pixel_scale)]
        if self.device is not None:
            argv += ["--device", self.device]
        if self.vae_chunk_size is not None:
            argv += ["--vae_chunk_size", str(self.vae_chunk_size)]
        if self.soft_tokens_weight is not None:
            argv += ["--soft_tokens_weight", self.soft_tokens_weight]
        if self.easycontrol_weight is not None:
            argv += ["--easycontrol_weight", self.easycontrol_weight]
        if self.easycontrol_image is not None:
            argv += ["--easycontrol_image", self.easycontrol_image]
        if self.pooled_text_proj is not None:
            argv += ["--pooled_text_proj", self.pooled_text_proj]
        if self.vocab_pack is not None:
            argv += ["--vocab_pack", self.vocab_pack]
        if self.stream is not None:
            argv += ["--stream", self.stream]

        if self.lora_weight is not None:
            argv += ["--lora_weight", *self.lora_weight]
        if self.lora_multiplier is not None:
            mults = (
                [self.lora_multiplier]
                if isinstance(self.lora_multiplier, (int, float))
                else list(self.lora_multiplier)
            )
            argv += ["--lora_multiplier", *(str(m) for m in mults)]

        if self.no_metadata:
            argv += ["--no_metadata"]
        if self.vae_disable_cache:
            argv += ["--vae_disable_cache"]
        if not self.vae_2d:
            argv += ["--no_vae_2d"]
        if self.text_encoder_cpu:
            argv += ["--text_encoder_cpu"]
        if self.no_vocab_pack:
            argv += ["--no_vocab_pack"]

        argv += list(self.extra_argv)
        return argv

    def to_args(
        self,
        parse_args: Optional[Callable[[list[str]], argparse.Namespace]] = None,
    ) -> argparse.Namespace:
        """Build a fully-defaulted ``argparse.Namespace`` for ``generate()``.

        Uses ``build_default_args`` unless ``parse_args`` is given. The parser
        validates choices and requireds, so a request with no ``prompt`` raises
        here, as on the CLI.
        """
        if parse_args is None:
            from library.inference.args import build_default_args

            parse_args = build_default_args
        return parse_args(self.to_argv())
