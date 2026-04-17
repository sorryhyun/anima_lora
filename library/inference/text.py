"""Text encoding and preparation for Anima inference."""

import gc
import logging
from typing import Optional, Tuple, Any, Dict

import torch

from library import strategy_base
from library.anima import models as anima_models
from library.runtime.device import clean_memory_on_device
from library.inference.models import load_text_encoder

logger = logging.getLogger(__name__)


def process_escape(text: str) -> str:
    """Process escape sequences in text."""
    return text.encode("utf-8").decode("unicode_escape")


def prepare_text_inputs(
    args,
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
