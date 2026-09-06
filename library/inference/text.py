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

# Anima's DiT expects a fixed-length cross-attention context. The pretrained
# model treats zero-padded positions as attention sinks in the cross-attention
# softmax — trimming to actual text length produces black images. All sites
# that prepare crossattn embeds (training, inference, CFG-uncond,
# DirectEdit, distillation) must pad to this exact length.
MAX_CROSSATTN_TOKENS = 512


_ESCAPE_RE = re.compile(r"\\(n|t|r|\\|u[0-9a-fA-F]{4})")


def _escape_sub(m: "re.Match[str]") -> str:
    g = m.group(1)
    simple = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\"}.get(g)
    return simple if simple is not None else chr(int(g[1:], 16))


def process_escape(text: str) -> str:
    """Process escape sequences (\\n, \\t, \\r, \\\\, \\uXXXX) in a prompt.

    Only the escape sequences themselves are rewritten — non-ASCII text passes
    through untouched. The previous ``encode("utf-8").decode("unicode_escape")``
    round-trip mojibaked every non-ASCII character (CJK prompts reached the
    tokenizer as latin-1 garbage).
    """
    return _ESCAPE_RE.sub(_escape_sub, text)


def ensure_text_strategies(
    text_encoder_path: Optional[str],
    max_length: int = MAX_CROSSATTN_TOKENS,
    vocab_pack=None,
) -> Tuple["text_strategies.TokenizeStrategy", "text_strategies.TextEncodingStrategy"]:
    """Idempotently install (and return) the tokenize/encode strategy singletons.

    Anima encodes prompts through two *process-global* singletons —
    ``TokenizeStrategy`` and ``TextEncodingStrategy`` (the strategy pattern in
    ``library/anima/strategy.py``). The CLI sets them in ``inference.main``; an
    embedder calling ``generate()`` / ``prepare_text_inputs()`` directly must too,
    or ``get_strategy()`` returns ``None`` and the first ``tokenize()`` call dies
    with a cryptic ``'NoneType' object has no attribute 'tokenize'``.

    This installs whichever singleton is still unset, building the tokenizer from
    ``text_encoder_path``. It is a **no-op when both are already installed**, so it
    composes with the CLI path (and is safe to call on every generation). If a
    strategy is missing *and* no path is available to build it, it raises a clear
    ``ValueError`` instead of failing later deep in the encode call.

    Returns the two live strategies (whether freshly installed or pre-existing) so
    a caller can use them directly instead of fishing them back out of the globals
    with ``get_strategy()``. They remain global — the return value is the same
    object the downstream encode path reads, not a private copy.

    ``vocab_pack`` selects the CJK vocab pack the tokenizer routes through: a
    path prefix or loaded ``VocabPack``; ``None`` = the ``vocab_pack`` key in
    ``configs/base.toml`` (``ANIMA_VOCAB_PACK`` wins); ``""`` = off. Only read
    when the tokenize strategy is actually installed here — an already-installed
    strategy (CLI path) is never swapped.
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
    """Prepare text-related inputs for T2I: LLM encoding. Anima model is also needed for preprocessing.

    Only four things are read from the request: ``prompt``, ``negative_prompt``,
    the text-encoder path, and whether to keep the encoder on CPU. Pass them as
    keywords for a self-documenting call (no ``inference.parse_args`` needed)::

        prepare_text_inputs(
            device=device, anima=dit, prompt="a lighthouse at dusk",
            shared_models={"text_encoder": text_encoder},
        )

    The legacy ``args`` namespace is still accepted as a positional fallback (the
    CLI and ``generate()`` pass one); explicit keywords win over it. ``device`` and
    ``anima`` are always required — they keep ``None`` defaults only so ``args`` can
    be omitted entirely without reordering positional callers.

    Text-encoder resolution: ``shared_models["text_encoder"]`` is reused when
    present (and parked back on its original device afterwards); otherwise — no
    shared dict, or a shared dict without one — the encoder is loaded locally
    and freed after encoding. A shared dict is never an obligation to carry the
    encoder; pass one holding only ``conds_cache`` and loading still just works.

    The tokenize/encode strategy singletons are lazily installed from the
    text-encoder path via ``ensure_text_strategies`` if a caller (an embedder
    driving ``generate()`` directly) hasn't set them — a no-op on the CLI path,
    which sets them in ``inference.main``.
    """
    if device is None or anima is None:
        raise ValueError("prepare_text_inputs requires `device` and `anima`.")

    # Explicit keyword wins; otherwise fall back to the namespace; otherwise default.
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

    # Install the global tokenize/encode strategies if the caller didn't (the
    # CLI does; a bare generate() embedder may not). No-op when already set.
    # The vocab pack follows the request's selection (--vocab_pack /
    # --no_vocab_pack), else the config default.
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
        # Pass the namespace through when we have one so TE-side LoRA
        # (lora_weight/lora_multiplier) still folds in; otherwise load from path.
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

    # Store original device to move back later if it was shared.
    text_encoder_original_device = text_encoder.device

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
