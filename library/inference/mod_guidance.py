"""Modulation guidance setup for Anima inference."""

import argparse
import gc
import logging
from typing import Optional, Dict, List

import torch

from library import strategy_base
from library.anima import models as anima_models
from library.runtime.device import clean_memory_on_device
from library.inference.models import load_text_encoder
from library.inference.text import process_escape

logger = logging.getLogger(__name__)


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


def build_mod_schedule(args: argparse.Namespace, num_blocks: int) -> List[float]:
    """Build a per-block w(l) list from CLI args.

    Default flags reproduce the 'step_i8_skip27' ComfyUI preset -- protects
    tonal-DC blocks 0-7 and the compensation block 27, applying full w to 8-26.
    See docs/methods/mod-guidance.md for rationale.
    """
    w = float(args.mod_w)
    start = int(getattr(args, "mod_start_layer", 8))
    end_raw = int(getattr(args, "mod_end_layer", 27))
    end = num_blocks if end_raw < 0 else min(end_raw, num_blocks)
    start = max(0, min(start, end))
    taper = int(getattr(args, "mod_taper", 0))
    taper_scale = float(getattr(args, "mod_taper_scale", 0.25))

    sched = [0.0] * num_blocks
    for i in range(start, end):
        sched[i] = w
    if taper > 0 and end > start:
        taper_start = max(start, end - taper)
        taper_w = w * taper_scale
        for i in range(taper_start, end):
            sched[i] = taper_w
    return sched


def setup_mod_guidance(
    args: argparse.Namespace,
    anima: anima_models.Anima,
    device: torch.device,
    shared_models: Optional[Dict] = None,
) -> None:
    """Compute Phase 2 modulation guidance delta and per-block schedule.

    delta_unit  = proj(pool(pos_crossattn)) - proj(pool(neg_crossattn))
    schedule[l] = w(l) from --mod_start_layer / --mod_end_layer / --mod_taper
    final_w     = --mod_final_w (applied at the final_layer only)

    At inference each block l receives `t_emb + schedule[l] * delta_unit`;
    `final_layer` receives `t_emb + final_w * delta_unit`. See docs/methods/mod-guidance.md.
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

    # Pool and project through trained pooled_text_proj. Note: unit delta -- the
    # per-block weight comes from the schedule, not baked in here.
    with torch.no_grad():
        pos_pooled = pos_crossattn.max(dim=1).values  # (1, 1024)
        neg_pooled = neg_crossattn.max(dim=1).values
        proj_pos = anima.pooled_text_proj(
            pos_pooled.to(anima.pooled_text_proj[0].weight.dtype)
        )
        proj_neg = anima.pooled_text_proj(
            neg_pooled.to(anima.pooled_text_proj[0].weight.dtype)
        )
        delta_unit = proj_pos - proj_neg  # (1, model_channels)

    # Copy into the preregistered non-persistent buffers so the forward can
    # always index them without a None/getattr fallback. Dtype/device follow
    # the buffer (which moves with the model's .to(...)).
    delta_buf = anima._mod_guidance_delta
    delta_buf.copy_(delta_unit.to(delta_buf.device, dtype=delta_buf.dtype))

    num_blocks = len(anima.blocks)
    schedule = build_mod_schedule(args, num_blocks)
    schedule_buf = anima._mod_guidance_schedule
    schedule_buf.copy_(
        torch.tensor(schedule, device=schedule_buf.device, dtype=schedule_buf.dtype)
    )
    final_w = float(getattr(args, "mod_final_w", 0.0))
    anima._mod_guidance_final_w.fill_(final_w)

    active = [(i, w) for i, w in enumerate(schedule) if w != 0.0]
    logger.info(
        f"Modulation guidance: delta_norm={delta_unit.norm().item():.4f}, "
        f"active blocks={len(active)}/{num_blocks}, "
        f"final_w={final_w}"
    )
    if active:
        logger.info(
            f"  schedule: blocks {active[0][0]}..{active[-1][0]} @ w={active[0][1]} "
            f"(taper={int(getattr(args, 'mod_taper', 0))})"
        )

    if loaded_locally:
        del text_encoder
        gc.collect()
        clean_memory_on_device(device)
    else:
        text_encoder.to("cpu")
