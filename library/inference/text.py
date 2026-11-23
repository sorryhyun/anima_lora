"""Text encoding and preparation for Anima inference."""

import argparse
import gc
import logging
import re
from typing import Optional, Tuple, Any, Dict

import torch

from library.anima import models as anima_models, text_strategies
from library.runtime.device import clean_memory_on_device
from library.inference.models import load_text_encoder

logger = logging.getLogger(__name__)

# Anima's DiT treats zero-padded positions as attention sinks in the
# cross-attention softmax; trimming to text length produces black images.
# Every site that prepares crossattn embeds must pad to this length.
MAX_CROSSATTN_TOKENS = 512


_ESCAPE_RE = re.compile(r"\\(n|t|r|\\|u[0-9a-fA-F]{4})")


def _escape_sub(m: "re.Match[str]") -> str:
    g = m.group(1)
    simple = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\"}.get(g)
    return simple if simple is not None else chr(int(g[1:], 16))


def process_escape(text: str) -> str:
    """Process escape sequences (\\n, \\t, \\r, \\\\, \\uXXXX) in a prompt.

    Non-ASCII text passes through untouched; an ``unicode_escape`` round-trip
    would mojibake CJK prompts.
    """
    return _ESCAPE_RE.sub(_escape_sub, text)


def ensure_text_strategies(
    text_encoder_path: Optional[str],
    max_length: int = MAX_CROSSATTN_TOKENS,
    vocab_pack=None,
) -> Tuple["text_strategies.TokenizeStrategy", "text_strategies.TextEncodingStrategy"]:
    """Idempotently install (and return) the tokenize/encode strategy singletons.

    ``TokenizeStrategy`` and ``TextEncodingStrategy`` are process-global. The
    CLI sets them in ``inference.main``; an embedder calling ``generate()``
    directly gets ``'NoneType' object has no attribute 'tokenize'`` unless they
    are set. Installs whichever is unset (no-op when both are), building the
    tokenizer from ``text_encoder_path``; raises ``ValueError`` when one is
    missing and there is no path.

    ``vocab_pack``: a path prefix or loaded ``VocabPack``; ``None`` = the
    ``vocab_pack`` key in ``configs/base.toml`` (``ANIMA_VOCAB_PACK`` wins);
    ``""`` = off. Ignored when the tokenize strategy is already installed.
    """
    from library.anima import strategy as strategy_anima

    need_tok = text_strategies.TokenizeStrategy.get_strategy() is None
    need_enc = text_strategies.TextEncodingStrategy.get_strategy() is None

    if need_tok:
        if not text_encoder_path:
            raise ValueError(
                "Text strategies are not initialized and no text-encoder path was "
                "provided to initialize them. Either set them yourself "
                "(text_strategies.TokenizeStrategy.set_strategy(...) + "
                "TextEncodingStrategy.set_strategy(...)) or pass a text-encoder path."
            )
        from library.anima.vocab_pack import (
            default_vocab_pack,
            load_vocab_pack,
            make_tokenize_strategy,
        )

        pack = load_vocab_pack(
            default_vocab_pack() if vocab_pack is None else vocab_pack
        )
        text_strategies.TokenizeStrategy.set_strategy(
            make_tokenize_strategy(
                pack,
                qwen3_path=text_encoder_path,
                t5_tokenizer_path=None,
                qwen3_max_length=max_length,
                t5_max_length=max_length,
            )
        )
    if need_enc:
        text_strategies.TextEncodingStrategy.set_strategy(
            strategy_anima.AnimaTextEncodingStrategy()
        )

    return (
        text_strategies.TokenizeStrategy.get_strategy(),
        text_strategies.TextEncodingStrategy.get_strategy(),
    )


def prepare_text_inputs(
    args: Optional[argparse.Namespace] = None,
    device: Optional[torch.device] = None,
    anima: Optional[anima_models.Anima] = None,
    shared_models: Optional[Dict] = None,
    *,
    prompt: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    text_encoder_path: Optional[str] = None,
    text_encoder_cpu: Optional[bool] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Encode the prompt and negative prompt into Anima crossattn embeds.

    Reads ``prompt``, ``negative_prompt``, the text-encoder path and
    ``text_encoder_cpu``, as keywords or from ``args`` (keywords win)::

        prepare_text_inputs(
            device=device, anima=dit, prompt="a lighthouse at dusk",
            shared_models={"text_encoder": text_encoder},
        )

    ``device`` and ``anima`` are required (``None`` defaults only keep the
    positional order). ``shared_models["text_encoder"]`` is reused and parked
    back on its device afterwards; without it the encoder is loaded and freed
    locally. Strategy singletons are installed via ``ensure_text_strategies``.
    """
    if device is None or anima is None:
        raise ValueError("prepare_text_inputs requires `device` and `anima`.")

    prompt = prompt if prompt is not None else getattr(args, "prompt", None)
    if prompt is None:
        raise ValueError(
            "prepare_text_inputs needs a prompt (pass prompt=... or args.prompt)."
        )
    negative_prompt = (
        negative_prompt
        if negative_prompt is not None
        else (getattr(args, "negative_prompt", "") or "")
    )
    te_path = (
        text_encoder_path
        if text_encoder_path is not None
        else getattr(args, "text_encoder", None)
    )
    te_cpu = (
        text_encoder_cpu
        if text_encoder_cpu is not None
        else bool(getattr(args, "text_encoder_cpu", False))
    )

    # Vocab pack follows --vocab_pack / --no_vocab_pack, else the config default.
    from library.anima.vocab_pack import resolve_active_pack

    ensure_text_strategies(
        te_path,
        vocab_pack=(resolve_active_pack(args) or "") if args is not None else None,
    )

    # load text encoder: conds_cache holds cached encodings for prompts without padding
    conds_cache = {}
    text_encoder_device = torch.device("cpu") if te_cpu else device
    text_encoder = None
    if shared_models is not None:
        # text_encoder is on device (batched inference) or CPU (interactive inference)
        text_encoder = shared_models.get("text_encoder")
        if "conds_cache" in shared_models:  # Use shared cache if available
            conds_cache = shared_models["conds_cache"]

    loaded_locally = text_encoder is None
    if loaded_locally:  # no shared dict, or a shared dict without a text encoder
        text_encoder_dtype = torch.bfloat16
        # The namespace carries TE-side LoRA (lora_weight/lora_multiplier).
        if args is not None:
            text_encoder = load_text_encoder(
                args, dtype=text_encoder_dtype, device=text_encoder_device
            )
        else:
            text_encoder = load_text_encoder(
                text_encoder=te_path,
                dtype=text_encoder_dtype,
                device=text_encoder_device,
            )
        text_encoder.eval()

    text_encoder_original_device = text_encoder.device

    model_is_moved = False

    def move_models_to_device_if_needed():
        nonlocal model_is_moved
        nonlocal shared_models

        if model_is_moved:
            return
        model_is_moved = True

        logger.debug(f"Moving Text Encoder to appropriate device: {text_encoder_device}")
        text_encoder.to(text_encoder_device)

    logger.debug("Encoding prompt with Text Encoder")

    prompt = process_escape(prompt)
    cache_key = prompt
    if cache_key in conds_cache:
        embed = conds_cache[cache_key]
    else:
        move_models_to_device_if_needed()

        tokenize_strategy = text_strategies.TokenizeStrategy.get_strategy()
        encoding_strategy = text_strategies.TextEncodingStrategy.get_strategy()

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
            if crossattn_emb.shape[1] < MAX_CROSSATTN_TOKENS:
                crossattn_emb = torch.nn.functional.pad(
                    crossattn_emb,
                    (0, 0, 0, MAX_CROSSATTN_TOKENS - crossattn_emb.shape[1]),
                )
            embed[0] = crossattn_emb
        embed[0] = embed[0].cpu()

        conds_cache[cache_key] = embed

    negative_prompt = process_escape(negative_prompt)
    cache_key = negative_prompt
    if cache_key in conds_cache:
        negative_embed = conds_cache[cache_key]
    else:
        move_models_to_device_if_needed()

        tokenize_strategy = text_strategies.TokenizeStrategy.get_strategy()
        encoding_strategy = text_strategies.TextEncodingStrategy.get_strategy()

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
            if crossattn_emb.shape[1] < MAX_CROSSATTN_TOKENS:
                crossattn_emb = torch.nn.functional.pad(
                    crossattn_emb,
                    (0, 0, 0, MAX_CROSSATTN_TOKENS - crossattn_emb.shape[1]),
                )
            negative_embed[0] = crossattn_emb
        negative_embed[0] = negative_embed[0].cpu()

        conds_cache[cache_key] = negative_embed

    if loaded_locally:
        del text_encoder
        gc.collect()
    else:  # if shared, move back to original device (likely CPU)
        text_encoder.to(text_encoder_original_device)

    clean_memory_on_device(device)

    arg_c = {"embed": embed, "prompt": prompt}
    arg_null = {"embed": negative_embed, "prompt": negative_prompt}

    return arg_c, arg_null
