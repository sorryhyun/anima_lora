"""Model loading for Anima inference: DiT, text encoder, shared model management."""

import argparse
import logging
from typing import Optional, Dict

import torch
from safetensors.torch import load_file

from library.anima import models as anima_models, weights as anima_utils
from library.runtime.device import clean_memory_on_device

logger = logging.getLogger(__name__)


def _is_hydra_moe(path: str) -> bool:
    """Cheap check: peek at the safetensors header for a `.lora_ups.` key.

    HydraLoRA moe files carry per-expert `lora_ups.N.weight` keys; regular
    LoRA files do not. Uses `safe_open` so only the header is read.
    """
    from safetensors import safe_open

    try:
        with safe_open(path, framework="pt") as f:
            return any(".lora_ups." in k for k in f.keys())
    except Exception:
        return False


def load_dit_model(
    args: argparse.Namespace,
    device: torch.device,
    dit_weight_dtype: Optional[torch.dtype] = None,
) -> anima_models.Anima:
    """Load DiT model with optional LoRA merge, P-GRAFT hooks, and torch.compile."""

    loading_device = "cpu"
    if not args.lycoris:
        loading_device = device

    # HydraLoRA moe: router-live inference can't go through static merge.
    # Detect early so we can skip the baked-down path and take the dynamic
    # hook route regardless of whether --pgraft is set.
    hydra_mode = False
    if args.lora_weight is not None and len(args.lora_weight) > 0:
        hydra_flags = [_is_hydra_moe(p) for p in args.lora_weight]
        if any(hydra_flags):
            if not all(hydra_flags):
                raise ValueError(
                    "Mixing HydraLoRA moe files with regular LoRA files in a "
                    "single --lora_weight list is not supported. The static "
                    "merge + dynamic hook interaction is untested. Pass them "
                    "in separate invocations."
                )
            hydra_mode = True

    # P-GRAFT: load without LoRA merge, attach dynamic hooks instead
    pgraft_mode = (
        getattr(args, "pgraft", False)
        and args.lora_weight is not None
        and len(args.lora_weight) > 0
    )

    # load LoRA weights (skip static merge for P-GRAFT and HydraLoRA moe)
    if (
        not pgraft_mode
        and not hydra_mode
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
    if pgraft_mode and not hydra_mode:
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

    # HydraLoRA moe: rehydrate the trained router-live network and attach it
    # as dynamic forward hooks, identical shape to the P-GRAFT path above.
    # The router runs per-sample on each adapted module, so the net stays in
    # eval mode with requires_grad_(False).
    if hydra_mode:
        from networks import lora_anima

        logger.info("HydraLoRA: loading moe file as router-live dynamic hooks")
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
                logger.warning(
                    f"HydraLoRA: unexpected keys in state dict: {info.unexpected_keys[:5]}..."
                )
            if info.missing_keys:
                logger.warning(
                    f"HydraLoRA: missing keys in state dict: {info.missing_keys[:5]}..."
                )
            network.to(device, dtype=torch.bfloat16)
            network.eval().requires_grad_(False)
            model._hydra_network = network
            # Reuse the P-GRAFT cutoff slot so existing toggle sites
            # (inference_pipeline loops + spectrum_denoise) honor
            # --lora_cutoff_step without further plumbing.
            model._pgraft_network = network
            logger.info(
                f"HydraLoRA: router-live attached "
                f"({len(network.unet_loras)} modules, "
                f"cutoff_step={getattr(args, 'lora_cutoff_step', None)})"
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
