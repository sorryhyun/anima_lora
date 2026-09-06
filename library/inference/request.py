"""``GenerationRequest`` — a typed front door for a single generation call.

``generate(args, gen_settings)`` reads ~40 fields off an ``argparse.Namespace``
via ``getattr``, so historically the only safe way to build one was to call the
CLI parser (what ``examples/01`` and ``03`` do). This dataclass makes the request
the canonical constructor and turns the CLI parser into *one* consumer instead of
the only one::

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

``to_args()`` feeds ``to_argv()`` through ``library.inference.args.build_default_args``
(the parser ``inference.parse_args`` itself delegates to) so the **parser
stays the single source of truth for every default** — this dataclass only
carries the knobs an embedder commonly sets, and the long tail
(spectrum / ip-adapter / … sub-knobs) rides through ``extra_argv``.

The request is frozen, and ``generate()`` doesn't write back to the namespace
``to_args()`` returns, so one ``GenerationRequest`` is safe to reuse across
calls (build a fresh namespace per call). When ``seed`` is unset, ``generate()``
resolves a fresh random seed per call without storing it — call
``resolve_seed(args)`` yourself if you need the concrete seed for saving.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple, Union

# the parser requires --text_encoder and --save_path; inject placeholders
# when unset so latent-only / non-saving requests still build a valid
# namespace (a real generation must set them)
_PLACEHOLDER_TEXT_ENCODER = "__unset_text_encoder__"
_PLACEHOLDER_SAVE_PATH = "__unset_save_path__"


@dataclass(frozen=True)
class GenerationRequest:
    """A single text-to-image generation request.

    Fields with concrete defaults mirror ``inference.parse_args`` and are always
    emitted; ``None``-valued fields are omitted from ``to_argv()`` so the parser's
    own default wins (no default duplicated in two places — the
    ``test_generation_request`` drift guard pins the ones that are).
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

    dit: Optional[str] = None
    vae: Optional[str] = None
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

    # verbatim CLI tokens for the long tail this dataclass doesn't model
    # (e.g. ["--spectrum", "--smc_cfg"]); overrides the same flag above
    extra_argv: Sequence[str] = field(default_factory=tuple)

    def to_argv(self) -> list[str]:
        """Render this request as a CLI token list for ``inference.parse_args``.

        Always-emitted scalar fields carry parser-matching defaults; ``None``
        fields are skipped (parser default wins); store_true flags emit only when
        ``True``. ``extra_argv`` is appended last so it can override anything.
        """
        argv: list[str] = []

        # Required by the parser — placeholder when unset so parse_args succeeds.
        argv += ["--text_encoder", self.text_encoder or _PLACEHOLDER_TEXT_ENCODER]
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

        Routes ``to_argv()`` through ``library.inference.args.build_default_args``
        (lazy-imported) — the same parser ``inference.parse_args`` now delegates
        to, so the request stays entirely inside ``library`` with no edge into
        the root ``inference`` entry-point script. Pass an explicit ``parse_args``
        to inject a different parser (the test suite does this). The parser fills
        every knob this dataclass doesn't model, and validates choices/requireds —
        so a request with no ``prompt`` raises here, the same as the CLI.
        """
        if parse_args is None:
            from library.inference.args import build_default_args

            parse_args = build_default_args
        return parse_args(self.to_argv())
