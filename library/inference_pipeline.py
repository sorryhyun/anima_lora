"""Reusable inference pipeline for Anima: model loading, generation, and output."""

import argparse
import datetime
import gc
import math
import os
import random
import time
from types import SimpleNamespace
from typing import Tuple, Optional, List, Any, Dict, Union

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

from library import (
    anima_models,
    anima_utils,
    inference_utils,
    qwen_image_autoencoder_kl,
    strategy_base,
)
from library.device_utils import clean_memory_on_device
from library.utils import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


# region Settings


class GenerationSettings:
    def __init__(
        self, device: torch.device, dit_weight_dtype: Optional[torch.dtype] = None
    ):
        self.device = device
        self.dit_weight_dtype = (
            dit_weight_dtype  # not used currently because model may be optimized
        )


def get_generation_settings(args: argparse.Namespace) -> GenerationSettings:
    device = torch.device(args.device)

    dit_weight_dtype = torch.bfloat16  # default

    logger.info(
        f"Using device: {device}, DiT weight weight precision: {dit_weight_dtype}"
    )

    gen_settings = GenerationSettings(device=device, dit_weight_dtype=dit_weight_dtype)
    return gen_settings


# endregion


# region Validation / helpers


def check_inputs(args: argparse.Namespace) -> Tuple[int, int]:
    """Validate image size.

    Returns:
        Tuple[int, int]: (height, width)
    """
    height = args.image_size[0]
    width = args.image_size[1]

    if height % 32 != 0 or width % 32 != 0:
        raise ValueError(
            f"`height` and `width` have to be divisible by 32 but are {height} and {width}."
        )

    return height, width


def process_escape(text: str) -> str:
    """Process escape sequences in text."""
    return text.encode("utf-8").decode("unicode_escape")


def get_time_flag():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S-%f")[
        :-3
    ]


# endregion


# region Model loading


def load_dit_model(
    args: argparse.Namespace,
    device: torch.device,
    dit_weight_dtype: Optional[torch.dtype] = None,
) -> anima_models.Anima:
    """Load DiT model with optional LoRA merge, P-GRAFT hooks, and torch.compile."""

    loading_device = "cpu"
    if not args.lycoris:
        loading_device = device

    # P-GRAFT: load without LoRA merge, attach dynamic hooks instead
    pgraft_mode = (
        getattr(args, "pgraft", False)
        and args.lora_weight is not None
        and len(args.lora_weight) > 0
    )

    # load LoRA weights (skip static merge for P-GRAFT)
    if (
        not pgraft_mode
        and not args.lycoris
        and args.lora_weight is not None
        and len(args.lora_weight) > 0
    ):
        lora_weights_list = []
        for lora_weight in args.lora_weight:
            logger.info(f"Loading LoRA weight from: {lora_weight}")
            lora_sd = load_file(lora_weight)  # load on CPU, dtype is as is
            lora_sd = {
                k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")
            }  # only keep unet lora weights
            lora_weights_list.append(lora_sd)
    else:
        lora_weights_list = None

    model = anima_utils.load_anima_model(
        device,
        args.dit,
        args.attn_mode,
        True,  # enable split_attn to trim masked tokens
        loading_device,
        dit_weight_dtype,
        lora_weights_list=lora_weights_list,
        lora_multipliers=args.lora_multiplier,
    )

    # Modulation guidance: load trained pooled_text_proj weights before .to()
    # (pooled_text_proj params are meta tensors when not in the pretrained checkpoint)
    pooled_text_proj_path = getattr(args, "pooled_text_proj", None)
    if pooled_text_proj_path is not None:
        anima_utils.load_pooled_text_proj(model, pooled_text_proj_path, "cpu")

    target_dtype = dit_weight_dtype
    if target_dtype is not None:
        logger.info(f"Convert model to {target_dtype}")
    logger.info(f"Move model to device: {device}")
    model.to(device, target_dtype)

    # model.to(device)
    model.to(device, dtype=torch.bfloat16)  # ensure model is in bfloat16 for inference

    model.eval().requires_grad_(False)

    # P-GRAFT: attach LoRA as dynamic hooks (can be toggled mid-denoising)
    if pgraft_mode:
        from networks import lora_anima

        logger.info("P-GRAFT: Loading LoRA as dynamic hooks (not static merge)")
        for lora_weight_path in args.lora_weight:
            lora_sd = load_file(lora_weight_path)
            lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")}

            multiplier = (
                args.lora_multiplier
                if isinstance(args.lora_multiplier, (int, float))
                else args.lora_multiplier[0]
            )
            network, weights_sd = lora_anima.create_network_from_weights(
                multiplier=multiplier,
                file=None,
                ae=None,
                text_encoders=[],
                unet=model,
                weights_sd=lora_sd,
                for_inference=True,
            )
            network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
            info = network.load_state_dict(weights_sd, strict=False)
            if info.unexpected_keys:
                logger.debug(
                    f"P-GRAFT: unexpected keys in LoRA state dict: {info.unexpected_keys[:5]}..."
                )
            network.to(device, dtype=torch.bfloat16)
            network.eval()
            model._pgraft_network = network
            logger.info(
                f"P-GRAFT: LoRA attached with cutoff_step={getattr(args, 'lora_cutoff_step', None)}"
            )

    if getattr(args, "compile", False):
        logger.info("Compiling DiT model with torch.compile...")
        model = torch.compile(model)

    clean_memory_on_device(device)

    return model


def load_text_encoder(
    args: argparse.Namespace,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    lora_weights_list = None
    if args.lora_weight is not None and len(args.lora_weight) > 0:
        lora_weights_list = []
        for lora_weight in args.lora_weight:
            logger.info(f"Loading LoRA weight from: {lora_weight}")
            lora_sd = load_file(lora_weight)  # load on CPU, dtype is as is
            lora_sd = {
                "model_" + k[len("lora_te_") :]: v
                for k, v in lora_sd.items()
                if k.startswith("lora_te_")
            }  # only keep Text Encoder lora weights, remove prefix "lora_te_" and add "model_" prefix
            lora_weights_list.append(lora_sd)

    lora_multipliers = args.lora_multiplier
    if lora_multipliers is not None and not isinstance(lora_multipliers, list):
        lora_multipliers = [lora_multipliers]
    text_encoder, _ = anima_utils.load_qwen3_text_encoder(
        args.text_encoder,
        dtype=dtype,
        device=device,
        lora_weights=lora_weights_list,
        lora_multipliers=lora_multipliers,
    )
    text_encoder.eval()
    return text_encoder


def load_shared_models(args: argparse.Namespace) -> Dict:
    """Load shared models for batch processing or interactive mode.
    Models are loaded to CPU to save memory. VAE is NOT loaded here.
    DiT model is also NOT loaded here, handled by process_batch_prompts or generate.
    """
    shared_models = {}
    text_encoder_dtype = torch.bfloat16
    text_encoder = load_text_encoder(
        args, dtype=text_encoder_dtype, device=torch.device("cpu")
    )
    shared_models["text_encoder"] = text_encoder
    return shared_models


# endregion


# region Text encoding


def prepare_text_inputs(
    args: argparse.Namespace,
    device: torch.device,
    anima: anima_models.Anima,
    shared_models: Optional[Dict] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Prepare text-related inputs for T2I: LLM encoding. Anima model is also needed for preprocessing."""

    # load text encoder: conds_cache holds cached encodings for prompts without padding
    conds_cache = {}
    text_encoder_device = torch.device("cpu") if args.text_encoder_cpu else device
    if shared_models is not None:
        text_encoder = shared_models.get("text_encoder")

        if "conds_cache" in shared_models:  # Use shared cache if available
            conds_cache = shared_models["conds_cache"]

        # text_encoder is on device (batched inference) or CPU (interactive inference)
    else:  # Load if not in shared_models
        text_encoder_dtype = torch.bfloat16  # Default dtype for Text Encoder
        text_encoder = load_text_encoder(
            args, dtype=text_encoder_dtype, device=text_encoder_device
        )
        text_encoder.eval()
        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
        # Store references so load_target_model can reuse them

    # Store original devices to move back later if they were shared.
    text_encoder_original_device = text_encoder.device if text_encoder else None

    if not text_encoder:
        raise ValueError("Text encoder is not loaded properly.")

    model_is_moved = False

    def move_models_to_device_if_needed():
        nonlocal model_is_moved
        nonlocal shared_models

        if model_is_moved:
            return
        model_is_moved = True

        logger.info(f"Moving Text Encoder to appropriate device: {text_encoder_device}")
        text_encoder.to(text_encoder_device)

    logger.info("Encoding prompt with Text Encoder")

    prompt = process_escape(args.prompt)
    cache_key = prompt
    if cache_key in conds_cache:
        embed = conds_cache[cache_key]
    else:
        move_models_to_device_if_needed()

        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
        encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

        with torch.no_grad():
            tokens = tokenize_strategy.tokenize(prompt)
            embed = encoding_strategy.encode_tokens(
                tokenize_strategy, [text_encoder], tokens
            )
            crossattn_emb, _ = anima._preprocess_text_embeds(
                source_hidden_states=embed[0].to(anima.device),
                target_input_ids=embed[2].to(anima.device),
                target_attention_mask=embed[3].to(anima.device),
                source_attention_mask=embed[1].to(anima.device),
            )
            crossattn_emb[~embed[3].bool()] = 0
            # Pad to 512 tokens (model expects fixed-length context)
            if crossattn_emb.shape[1] < 512:
                crossattn_emb = torch.nn.functional.pad(
                    crossattn_emb, (0, 0, 0, 512 - crossattn_emb.shape[1])
                )
            embed[0] = crossattn_emb
        embed[0] = embed[0].cpu()

        conds_cache[cache_key] = embed

    negative_prompt = process_escape(args.negative_prompt)
    cache_key = negative_prompt
    if cache_key in conds_cache:
        negative_embed = conds_cache[cache_key]
    else:
        move_models_to_device_if_needed()

        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
        encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

        with torch.no_grad():
            tokens = tokenize_strategy.tokenize(negative_prompt)
            negative_embed = encoding_strategy.encode_tokens(
                tokenize_strategy, [text_encoder], tokens
            )
            crossattn_emb, _ = anima._preprocess_text_embeds(
                source_hidden_states=negative_embed[0].to(anima.device),
                target_input_ids=negative_embed[2].to(anima.device),
                target_attention_mask=negative_embed[3].to(anima.device),
                source_attention_mask=negative_embed[1].to(anima.device),
            )
            crossattn_emb[~negative_embed[3].bool()] = 0
            # Pad to 512 tokens (model expects fixed-length context)
            if crossattn_emb.shape[1] < 512:
                crossattn_emb = torch.nn.functional.pad(
                    crossattn_emb, (0, 0, 0, 512 - crossattn_emb.shape[1])
                )
            negative_embed[0] = crossattn_emb
        negative_embed[0] = negative_embed[0].cpu()

        conds_cache[cache_key] = negative_embed

    if not (shared_models and "text_encoder" in shared_models):  # if loaded locally
        # There is a bug text_encoder is not freed from GPU memory when text encoder is fp8
        del text_encoder
        gc.collect()
    else:  # if shared, move back to original device (likely CPU)
        if text_encoder:
            text_encoder.to(text_encoder_original_device)

    clean_memory_on_device(device)

    arg_c = {"embed": embed, "prompt": prompt}
    arg_null = {"embed": negative_embed, "prompt": negative_prompt}

    return arg_c, arg_null


# endregion


# region Generation


def compute_tile_positions(
    h_latent: int, w_latent: int, tile_size: int, overlap: int
) -> List[Tuple[int, int]]:
    """Compute (y, x) start positions for overlapping tiles covering the full latent grid."""
    stride = tile_size - overlap
    positions = []
    y = 0
    while y < h_latent:
        if y + tile_size > h_latent:
            y = h_latent - tile_size  # clamp last row
        x = 0
        while x < w_latent:
            if x + tile_size > w_latent:
                x = w_latent - tile_size  # clamp last column
            positions.append((y, x))
            if x + tile_size >= w_latent:
                break
            x += stride
        if y + tile_size >= h_latent:
            break
        y += stride
    return positions


def create_tile_blend_weight(
    tile_h: int,
    tile_w: int,
    overlap: int,
    y: int,
    x: int,
    h_latent: int,
    w_latent: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a (1, 1, 1, tile_h, tile_w) blend weight with cosine ramps on overlapping edges."""
    weight = torch.ones(1, 1, 1, tile_h, tile_w, device=device, dtype=dtype)
    if overlap <= 0:
        return weight

    # Precompute cosine ramp: (1 - cos(pi * t)) / 2 for t in [0, 1]
    ramp = torch.linspace(0.0, 1.0, overlap, device=device, dtype=dtype)
    ramp = (1.0 - torch.cos(math.pi * ramp)) / 2.0

    # Top edge
    if y > 0:
        weight[:, :, :, :overlap, :] *= ramp[None, None, None, :, None]
    # Bottom edge
    if y + tile_h < h_latent:
        weight[:, :, :, -overlap:, :] *= ramp.flip(0)[None, None, None, :, None]
    # Left edge
    if x > 0:
        weight[:, :, :, :, :overlap] *= ramp[None, None, None, None, :]
    # Right edge
    if x + tile_w < w_latent:
        weight[:, :, :, :, -overlap:] *= ramp.flip(0)[None, None, None, None, :]

    return weight


def generate_body_tiled(
    args: Union[argparse.Namespace, SimpleNamespace],
    anima: anima_models.Anima,
    context: Dict[str, Any],
    context_null: Optional[Dict[str, Any]],
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """MultiDiffusion-style tiled denoising for high-resolution generation."""
    seed_g = torch.Generator(device="cpu")
    seed_g.manual_seed(seed)

    height, width = check_inputs(args)
    logger.info(
        f"Tiled diffusion: image size {height}x{width} (HxW), infer_steps: {args.infer_steps}"
    )

    tile_size = args.tile_size
    overlap = args.tile_overlap
    patch_spatial = anima.patch_spatial

    embed = context["embed"][0].to(device, dtype=torch.bfloat16)
    if context_null is None:
        context_null = context
    negative_embed = context_null["embed"][0].to(device, dtype=torch.bfloat16)

    # Prefix tuning: prepend learned vectors to cached adapter output
    prefix_weight = getattr(args, "prefix_weight", None)
    if prefix_weight is not None:
        from networks.postfix_anima import create_network_from_weights

        prefix_net, prefix_sd = create_network_from_weights(
            multiplier=1.0, file=prefix_weight, ae=None, text_encoders=None, unet=None
        )
        prefix_net.load_weights(prefix_weight)
        prefix_net.to(device, dtype=torch.bfloat16)
        embed = prefix_net.prepend_prefix(embed)
        negative_embed = prefix_net.prepend_prefix(negative_embed)
        logger.info(f"Prefix: prepended {prefix_net.num_postfix_tokens} tokens, embed shape now {embed.shape}")

    # Postfix tuning: append learned vectors after real text tokens
    postfix_weight = getattr(args, "postfix_weight", None)
    if postfix_weight is not None:
        from networks.postfix_anima import create_network_from_weights

        postfix_net, postfix_sd = create_network_from_weights(
            multiplier=1.0, file=postfix_weight, ae=None, text_encoders=None, unet=None
        )
        postfix_net.load_weights(postfix_weight)
        postfix_net.to(device, dtype=torch.bfloat16)
        embed_mask = context["embed"][3].to(device)
        embed_seqlens = embed_mask.sum(dim=-1).to(torch.int32)
        embed = postfix_net.append_postfix(embed, embed_seqlens)
        neg_mask = context_null["embed"][3].to(device)
        neg_seqlens = neg_mask.sum(dim=-1).to(torch.int32)
        negative_embed = postfix_net.append_postfix(negative_embed, neg_seqlens)
        logger.info(f"Postfix: appended {postfix_net.num_postfix_tokens} tokens after text")

    num_channels_latents = anima_models.Anima.LATENT_CHANNELS
    h_latent = height // 8
    w_latent = width // 8
    shape = (1, num_channels_latents, 1, h_latent, w_latent)
    latents = randn_tensor(shape, generator=seed_g, device=device, dtype=torch.bfloat16)

    # Compute tile positions and precompute blend weights
    positions = compute_tile_positions(h_latent, w_latent, tile_size, overlap)
    logger.info(
        f"Tiled diffusion: {len(positions)} tiles, tile_size={tile_size}, overlap={overlap}"
    )

    blend_weights = {}
    for y, x in positions:
        tile_h = min(tile_size, h_latent - y)
        tile_w = min(tile_size, w_latent - x)
        blend_weights[(y, x)] = create_tile_blend_weight(
            tile_h, tile_w, overlap, y, x, h_latent, w_latent, device, torch.bfloat16
        )

    embed = embed.to(torch.bfloat16)
    negative_embed = negative_embed.to(torch.bfloat16)

    timesteps, sigmas = inference_utils.get_timesteps_sigmas(
        args.infer_steps, args.flow_shift, device
    )
    timesteps /= 1000
    timesteps = timesteps.to(device, dtype=torch.bfloat16)

    # Create sampler
    er_sde = None
    if args.sampler == "er_sde":
        er_sde = inference_utils.ERSDESampler(sigmas, seed=args.seed, device=device)

    do_cfg = args.guidance_scale != 1.0
    autocast_enabled = args.fp8

    # P-GRAFT: get network reference for mid-denoising cutoff
    pgraft_network = getattr(anima, "_pgraft_network", None)
    lora_cutoff_step = getattr(args, "lora_cutoff_step", None)

    with tqdm(total=len(timesteps), desc="Denoising steps (tiled)") as pbar:
        for i, t in enumerate(timesteps):
            # P-GRAFT: disable LoRA at cutoff step
            if (
                pgraft_network is not None
                and lora_cutoff_step is not None
                and i == lora_cutoff_step
            ):
                pgraft_network.set_enabled(False)
                logger.info(f"P-GRAFT: Disabled LoRA at step {i}/{len(timesteps)}")

            t_expand = t.expand(latents.shape[0])

            noise_acc = torch.zeros_like(latents)
            weight_acc = torch.zeros(
                1, 1, 1, h_latent, w_latent, device=device, dtype=torch.bfloat16
            )

            if do_cfg:
                uncond_noise_acc = torch.zeros_like(latents)
                uncond_weight_acc = torch.zeros(
                    1, 1, 1, h_latent, w_latent, device=device, dtype=torch.bfloat16
                )

            for y, x in positions:
                tile_h = min(tile_size, h_latent - y)
                tile_w = min(tile_size, w_latent - x)
                tile_latent = latents[:, :, :, y : y + tile_h, x : x + tile_w]
                tile_padding_mask = torch.zeros(
                    1, 1, tile_h, tile_w, dtype=torch.bfloat16, device=device
                )

                h_off = y // patch_spatial
                w_off = x // patch_spatial

                bw = blend_weights[(y, x)]

                # Conditional pass
                if anima.blocks_to_swap:
                    anima.prepare_block_swap_before_forward()
                with (
                    torch.no_grad(),
                    torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=autocast_enabled,
                    ),
                ):
                    tile_pred = anima(
                        tile_latent,
                        t_expand,
                        embed,
                        padding_mask=tile_padding_mask,
                        h_offset=h_off,
                        w_offset=w_off,
                    )
                noise_acc[:, :, :, y : y + tile_h, x : x + tile_w] += tile_pred * bw
                weight_acc[:, :, :, y : y + tile_h, x : x + tile_w] += bw

                # Unconditional pass
                if do_cfg:
                    if anima.blocks_to_swap:
                        anima.prepare_block_swap_before_forward()
                    with (
                        torch.no_grad(),
                        torch.autocast(
                            device_type=device.type,
                            dtype=torch.bfloat16,
                            enabled=autocast_enabled,
                        ),
                    ):
                        uncond_tile_pred = anima(
                            tile_latent,
                            t_expand,
                            negative_embed,
                            padding_mask=tile_padding_mask,
                            h_offset=h_off,
                            w_offset=w_off,
                        )
                    uncond_noise_acc[:, :, :, y : y + tile_h, x : x + tile_w] += (
                        uncond_tile_pred * bw
                    )
                    uncond_weight_acc[:, :, :, y : y + tile_h, x : x + tile_w] += bw

            noise_pred = noise_acc / weight_acc
            if do_cfg:
                uncond_noise_pred = uncond_noise_acc / uncond_weight_acc
                noise_pred = uncond_noise_pred + args.guidance_scale * (
                    noise_pred - uncond_noise_pred
                )

            if er_sde is not None:
                denoised = latents.float() - sigmas[i] * noise_pred.float()
                latents = er_sde.step(latents, denoised, i).to(latents.dtype)
            else:
                latents = inference_utils.step(latents, noise_pred, sigmas, i).to(
                    latents.dtype
                )
            pbar.update()

    # P-GRAFT: restore LoRA for next generation
    if pgraft_network is not None and lora_cutoff_step is not None:
        pgraft_network.set_enabled(True)

    return latents


def generate_body(
    args: Union[argparse.Namespace, SimpleNamespace],
    anima: anima_models.Anima,
    context: Dict[str, Any],
    context_null: Optional[Dict[str, Any]],
    device: torch.device,
    seed: int,
) -> torch.Tensor:

    # set random generator
    seed_g = torch.Generator(device="cpu")
    seed_g.manual_seed(seed)

    height, width = check_inputs(args)

    # Dispatch to tiled diffusion if enabled and latent exceeds tile size
    h_latent = height // 8
    w_latent = width // 8
    if getattr(args, "tiled_diffusion", False) and (
        h_latent > args.tile_size or w_latent > args.tile_size
    ):
        return generate_body_tiled(args, anima, context, context_null, device, seed)

    logger.info(f"Image size: {height}x{width} (HxW), infer_steps: {args.infer_steps}")

    # image generation ######

    logger.info(f"Prompt: {context['prompt']}")

    embed = context["embed"][0].to(device, dtype=torch.bfloat16)
    if context_null is None:
        context_null = context  # dummy for unconditional
    negative_embed = context_null["embed"][0].to(device, dtype=torch.bfloat16)

    # Prefix/postfix tuning: inject learned vectors into cross-attention embeddings.
    # Pool text BEFORE injection so modulation guidance sees only real text tokens.
    _pooled_text_pos = None
    _pooled_text_neg = None

    prefix_weight = getattr(args, "prefix_weight", None)
    postfix_weight = getattr(args, "postfix_weight", None)
    if prefix_weight is not None or postfix_weight is not None:
        _pooled_text_pos = embed.max(dim=1).values  # (1, 1024)
        _pooled_text_neg = negative_embed.max(dim=1).values

    if prefix_weight is not None:
        from networks.postfix_anima import create_network_from_weights

        prefix_net, prefix_sd = create_network_from_weights(
            multiplier=1.0, file=prefix_weight, ae=None, text_encoders=None, unet=None
        )
        prefix_net.load_weights(prefix_weight)
        prefix_net.to(device, dtype=torch.bfloat16)
        embed = prefix_net.prepend_prefix(embed)
        negative_embed = prefix_net.prepend_prefix(negative_embed)
        logger.info(f"Prefix: prepended {prefix_net.num_postfix_tokens} tokens, embed shape now {embed.shape}")

    if postfix_weight is not None:
        from networks.postfix_anima import create_network_from_weights

        postfix_net, postfix_sd = create_network_from_weights(
            multiplier=1.0, file=postfix_weight, ae=None, text_encoders=None, unet=None
        )
        postfix_net.load_weights(postfix_weight)
        postfix_net.to(device, dtype=torch.bfloat16)
        # Compute seqlens from attention masks
        embed_mask = context["embed"][3].to(device)
        embed_seqlens = embed_mask.sum(dim=-1).to(torch.int32)
        embed = postfix_net.append_postfix(embed, embed_seqlens)
        neg_mask = context_null["embed"][3].to(device)
        neg_seqlens = neg_mask.sum(dim=-1).to(torch.int32)
        negative_embed = postfix_net.append_postfix(negative_embed, neg_seqlens)
        logger.info(f"Postfix: appended {postfix_net.num_postfix_tokens} tokens after text")

    # Prepare latent variables
    num_channels_latents = anima_models.Anima.LATENT_CHANNELS
    shape = (
        1,
        num_channels_latents,
        1,  # Frame dimension
        height // 8,
        width // 8,
    )
    latents = randn_tensor(shape, generator=seed_g, device=device, dtype=torch.bfloat16)

    # Create padding mask
    bs = latents.shape[0]
    h_latent = latents.shape[-2]
    w_latent = latents.shape[-1]
    padding_mask = torch.zeros(
        bs, 1, h_latent, w_latent, dtype=torch.bfloat16, device=device
    )

    logger.info(
        f"Embed: {embed.shape}, negative_embed: {negative_embed.shape}, latents: {latents.shape}"
    )
    embed = embed.to(torch.bfloat16)
    negative_embed = negative_embed.to(torch.bfloat16)

    # Prepare timesteps
    timesteps, sigmas = inference_utils.get_timesteps_sigmas(
        args.infer_steps, args.flow_shift, device
    )
    timesteps /= 1000  # scale to [0,1] range
    timesteps = timesteps.to(device, dtype=torch.bfloat16)

    # Create sampler
    er_sde = None
    if args.sampler == "er_sde":
        er_sde = inference_utils.ERSDESampler(sigmas, seed=args.seed, device=device)

    # Denoising loop
    do_cfg = args.guidance_scale != 1.0
    autocast_enabled = args.fp8

    # P-GRAFT: get network reference for mid-denoising cutoff
    pgraft_network = getattr(anima, "_pgraft_network", None)
    lora_cutoff_step = getattr(args, "lora_cutoff_step", None)

    if getattr(args, "spectrum", False):
        from networks.spectrum import spectrum_denoise

        latents = spectrum_denoise(
            anima,
            latents,
            timesteps,
            sigmas,
            embed,
            negative_embed,
            padding_mask,
            args.guidance_scale,
            er_sde,
            device,
            window_size=getattr(args, "spectrum_window_size", 2.0),
            flex_window=getattr(args, "spectrum_flex_window", 0.25),
            warmup_steps=getattr(args, "spectrum_warmup", 6),
            w=getattr(args, "spectrum_w", 0.3),
            m=getattr(args, "spectrum_m", 3),
            lam=getattr(args, "spectrum_lam", 0.1),
            stop_caching_step=getattr(args, "spectrum_stop_caching_step", -1),
            calibration_strength=getattr(args, "spectrum_calibration", 0.0),
            autocast_enabled=autocast_enabled,
            pgraft_network=pgraft_network,
            pooled_text_pos=_pooled_text_pos,
            pooled_text_neg=_pooled_text_neg,
            lora_cutoff_step=lora_cutoff_step,
        )
    else:
        with tqdm(total=len(timesteps), desc="Denoising steps") as pbar:
            for i, t in enumerate(timesteps):
                # P-GRAFT: disable LoRA at cutoff step (reference model takes over)
                if (
                    pgraft_network is not None
                    and lora_cutoff_step is not None
                    and i == lora_cutoff_step
                ):
                    pgraft_network.set_enabled(False)
                    logger.info(f"P-GRAFT: Disabled LoRA at step {i}/{len(timesteps)}")

                t_expand = t.expand(latents.shape[0])

                with (
                    torch.no_grad(),
                    torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=autocast_enabled,
                    ),
                ):
                    _pos_kw = {"pooled_text_override": _pooled_text_pos} if _pooled_text_pos is not None else {}
                    noise_pred = anima(
                        latents, t_expand, embed, padding_mask=padding_mask, **_pos_kw
                    )

                if do_cfg:
                    with (
                        torch.no_grad(),
                        torch.autocast(
                            device_type=device.type,
                            dtype=torch.bfloat16,
                            enabled=autocast_enabled,
                        ),
                    ):
                        _neg_kw = {"pooled_text_override": _pooled_text_neg} if _pooled_text_neg is not None else {}
                        uncond_noise_pred = anima(
                            latents, t_expand, negative_embed, padding_mask=padding_mask, **_neg_kw
                        )
                    noise_pred = uncond_noise_pred + args.guidance_scale * (
                        noise_pred - uncond_noise_pred
                    )

                # ensure latents dtype is consistent
                if er_sde is not None:
                    denoised = latents.float() - sigmas[i] * noise_pred.float()
                    latents = er_sde.step(latents, denoised, i).to(latents.dtype)
                else:
                    latents = inference_utils.step(latents, noise_pred, sigmas, i).to(
                        latents.dtype
                    )

                pbar.update()

        # P-GRAFT: restore LoRA for next generation
        if pgraft_network is not None and lora_cutoff_step is not None:
            pgraft_network.set_enabled(True)

    return latents


def _encode_prompt_for_mod(
    prompt: str,
    anima: anima_models.Anima,
    text_encoder,
    device: torch.device,
) -> torch.Tensor:
    """Encode a prompt and return its crossattn_emb (post-LLMAdapter, padded)."""
    prompt = process_escape(prompt)
    tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
    encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

    with torch.no_grad():
        tokens = tokenize_strategy.tokenize(prompt)
        embed = encoding_strategy.encode_tokens(
            tokenize_strategy, [text_encoder], tokens
        )
        crossattn_emb, _ = anima._preprocess_text_embeds(
            source_hidden_states=embed[0].to(anima.device),
            target_input_ids=embed[2].to(anima.device),
            target_attention_mask=embed[3].to(anima.device),
            source_attention_mask=embed[1].to(anima.device),
        )
        crossattn_emb[~embed[3].bool()] = 0
        if crossattn_emb.shape[1] < 512:
            crossattn_emb = torch.nn.functional.pad(
                crossattn_emb, (0, 0, 0, 512 - crossattn_emb.shape[1])
            )
    return crossattn_emb


def _setup_mod_guidance(
    args: argparse.Namespace,
    anima: anima_models.Anima,
    device: torch.device,
    shared_models: Optional[Dict] = None,
) -> None:
    """Compute Phase 2 modulation guidance delta and store on model.

    guidance_delta = w * (proj(pool(pos_crossattn)) - proj(pool(neg_crossattn)))
    This is added to t_embedding every forward step.
    """
    mod_w = args.mod_w
    mod_pos = args.mod_pos_prompt
    mod_neg = args.mod_neg_prompt

    # Load text encoder (reuse shared if available, otherwise load temporarily)
    if shared_models and "text_encoder" in shared_models:
        text_encoder = shared_models["text_encoder"]
        text_encoder.to(device)
        loaded_locally = False
    else:
        text_encoder_dtype = torch.bfloat16
        text_encoder = load_text_encoder(args, dtype=text_encoder_dtype, device=device)
        text_encoder.eval()
        loaded_locally = True

    logger.info(f"Computing modulation guidance delta (w={mod_w})")
    logger.info(f"  pos: {mod_pos}")
    logger.info(f"  neg: {mod_neg}")

    pos_crossattn = _encode_prompt_for_mod(mod_pos, anima, text_encoder, device)
    neg_crossattn = _encode_prompt_for_mod(mod_neg, anima, text_encoder, device)

    # Pool and project through trained pooled_text_proj
    with torch.no_grad():
        pos_pooled = pos_crossattn.max(dim=1).values  # (1, 1024)
        neg_pooled = neg_crossattn.max(dim=1).values
        proj_pos = anima.pooled_text_proj(pos_pooled.to(anima.pooled_text_proj[0].weight.dtype))
        proj_neg = anima.pooled_text_proj(neg_pooled.to(anima.pooled_text_proj[0].weight.dtype))
        delta = mod_w * (proj_pos - proj_neg)  # (1, model_channels)

    anima._mod_guidance_delta = delta.to(device, dtype=torch.bfloat16)
    logger.info(f"Modulation guidance delta set (norm={delta.norm().item():.4f})")

    if loaded_locally:
        del text_encoder
        gc.collect()
        clean_memory_on_device(device)
    else:
        text_encoder.to("cpu")


def generate(
    args: argparse.Namespace,
    gen_settings: GenerationSettings,
    shared_models: Optional[Dict] = None,
    precomputed_text_data: Optional[Dict] = None,
) -> torch.Tensor:
    """Main function for generation.

    Returns:
        torch.Tensor: generated latent
    """
    device, dit_weight_dtype = (gen_settings.device, gen_settings.dit_weight_dtype)

    # prepare seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    args.seed = seed  # set seed to args for saving

    if shared_models is None or "model" not in shared_models:
        # load DiT model
        anima = load_dit_model(args, device, dit_weight_dtype)

        if shared_models is not None:
            shared_models["model"] = anima
    else:
        # use shared model
        logger.info("Using shared DiT model.")
        anima: anima_models.Anima = shared_models["model"]

    if precomputed_text_data is not None:
        logger.info("Using precomputed text data.")
        context = precomputed_text_data["context"]
        context_null = precomputed_text_data["context_null"]

    else:
        logger.info("No precomputed data. Preparing image and text inputs.")
        context, context_null = prepare_text_inputs(args, device, anima, shared_models)

    # Phase 2 modulation guidance: compute guidance delta once
    if getattr(args, "pooled_text_proj", None) is not None and getattr(args, "mod_w", 0.0) != 0.0:
        _setup_mod_guidance(args, anima, device, shared_models)
    else:
        anima._mod_guidance_delta = None

    return generate_body(args, anima, context, context_null, device, seed)


# endregion


# region Output


def decode_latent(
    vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage,
    latent: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    logger.info(f"Decoding image. Latent shape {latent.shape}, device {device}")

    vae.to(device)
    with torch.no_grad():
        pixels = vae.decode_to_pixels(latent.to(device, dtype=vae.dtype))
    if (
        pixels.ndim == 5
    ):  # remove frame dimension if exists, [B, C, F, H, W] -> [B, C, H, W]
        pixels = pixels.squeeze(2)

    pixels = pixels.to(
        "cpu", dtype=torch.float32
    )  # move to CPU and convert to float32 (bfloat16 is not supported by numpy)
    vae.to("cpu")

    logger.info(f"Decoded. Pixel shape {pixels.shape}")
    return pixels[0]  # remove batch dimension


def save_latent(
    latent: torch.Tensor, args: argparse.Namespace, height: int, width: int
) -> str:
    """Save latent to file.

    Returns:
        str: Path to saved latent file
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = get_time_flag()

    seed = args.seed

    latent_path = f"{save_path}/{time_flag}_{seed}_latent.safetensors"

    if args.no_metadata:
        metadata = None
    else:
        metadata = {
            "seeds": f"{seed}",
            "prompt": f"{args.prompt}",
            "height": f"{height}",
            "width": f"{width}",
            "infer_steps": f"{args.infer_steps}",
            "guidance_scale": f"{args.guidance_scale}",
        }
        if args.negative_prompt is not None:
            metadata["negative_prompt"] = f"{args.negative_prompt}"

    sd = {"latent": latent.contiguous()}
    save_file(sd, latent_path, metadata=metadata)
    logger.info(f"Latent saved to: {latent_path}")

    return latent_path


def save_images(
    sample: torch.Tensor,
    args: argparse.Namespace,
    original_base_name: Optional[str] = None,
) -> str:
    """Save images to directory.

    Returns:
        str: Path to saved image
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = get_time_flag()

    seed = args.seed
    original_name = "" if original_base_name is None else f"_{original_base_name}"
    image_name = f"{time_flag}_{seed}{original_name}"

    x = torch.clamp(sample, -1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
    x = x.transpose(1, 2, 0)  # C, H, W -> H, W, C

    image = Image.fromarray(x)
    image.save(os.path.join(save_path, f"{image_name}.png"))

    logger.info(f"Sample images saved to: {save_path}/{image_name}")

    return f"{save_path}/{image_name}"


def save_output(
    args: argparse.Namespace,
    vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage,
    latent: torch.Tensor,
    device: torch.device,
    original_base_name: Optional[str] = None,
) -> None:
    """Decode latent and save output (latent/images based on args.output_type)."""
    height, width = latent.shape[-2], latent.shape[-1]  # BCTHW
    height *= 8
    width *= 8
    if args.output_type == "latent" or args.output_type == "latent_images":
        save_latent(latent, args, height, width)
    if args.output_type == "latent":
        return

    if vae is None:
        logger.error("VAE is None, cannot decode latents for saving video/images.")
        return

    if latent.ndim == 2:  # S,C. For packed latents from other inference scripts
        latent = latent.unsqueeze(0)
        height, width = check_inputs(args)
        latent = latent.view(
            1,
            vae.latent_channels,
            1,  # Frame dimension
            height // 8,
            width // 8,
        )

    image = decode_latent(vae, latent, device)

    if args.output_type == "images" or args.output_type == "latent_images":
        if original_base_name is None:
            original_name = ""
        else:
            original_name = f"_{original_base_name}"
        save_images(image, args, original_name)


# endregion
