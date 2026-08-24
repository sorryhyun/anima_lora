"""T5("") unconditional cross-attention sidecar — Anima-domain helpers.

The unconditional text input every training / distill / inference path shares is
a single model-scoped file — ``post_image_dataset/_anima_uncond_te.safetensors``
— so the LoRA's CFG-uncond branch matches Anima's own inference path
(``library/inference/text.py:99-127``). This is paper-faithful (Starodubcev
et al., ICLR 2026, arXiv:2602.09268v1 §5) and avoids the
``torch.zeros_like(crossattn_emb)`` shortcut that would be neither.

This module owns the *Anima-domain* half: path/constants, encoding ``T5("")``
through Qwen3 + the LLM adapter, and loading/broadcasting the cached tensor.
The *produce-to-disk* half (writing the sidecar, staging, on-demand ensure)
lives in :mod:`library.preprocess.uncond`, which builds on these primitives.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file as _load_safetensors

logger = logging.getLogger(__name__)

UNCOND_TE_FILENAME = "_anima_uncond_te.safetensors"
# Matches ``library.inference.text.MAX_CROSSATTN_TOKENS``; defined locally so the
# anima/ domain layer never imports up into inference/. The 512 cap is an
# Anima-model fact (its CFG-uncond padding), so it rightly lives here.
DEFAULT_SEQ_LEN = 512

# The uncond sidecar is a model-scoped artifact, not a per-cache-dir one: it is a
# pure function of the base Qwen3 encoder + base-DiT LLM adapter + 512-pad, so it
# never varies by dataset / checkpoint / run. We therefore SHIP it as a bundled
# package asset (~1 MB) right next to this module — no preprocess step needed to
# materialise it. The path is package-relative (via ``__file__``) so it resolves
# from any CWD and ships with the install; staging/regeneration (e.g. after a
# base-model swap) overwrites this same file in place, keeping one source of
# truth. Override with an explicit path arg where a call site exposes one.
DEFAULT_UNCOND_DIR = Path(__file__).resolve().parent / "assets"


def default_uncond_path() -> Path:
    """Canonical sidecar path — the shipped, model-scoped package asset.
    Override via CLI flag where a call site exposes one."""
    return DEFAULT_UNCOND_DIR / UNCOND_TE_FILENAME


def encode_uncond_with_models(
    text_encoder,
    tokenize_strategy,
    encoding_strategy,
    llm_adapter,
    *,
    seq_len: int = DEFAULT_SEQ_LEN,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode ``T5("")`` using already-loaded models. Returns
    ``(crossattn_emb (seq_len, 1024), pooled (1024,))`` as bf16 CPU tensors.

    Use this from preprocess / training entry points where the text encoder
    and LLM adapter are already on device — avoids the second model-load cost
    of :func:`encode_uncond_crossattn`.
    """
    with torch.no_grad():
        tokens_and_masks = tokenize_strategy.tokenize([""])
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = (
            encoding_strategy.encode_tokens(
                tokenize_strategy, [text_encoder], tokens_and_masks
            )
        )
        crossattn_emb = llm_adapter(
            source_hidden_states=prompt_embeds,
            target_input_ids=t5_input_ids.to(device, dtype=torch.long),
            target_attention_mask=t5_attn_mask.to(device),
            source_attention_mask=attn_mask,
        )
        # Zero padding positions — attention sinks in cross-attention softmax.
        crossattn_emb[~t5_attn_mask.to(device).bool()] = 0

    cur_seq = crossattn_emb.shape[1]
    if cur_seq < seq_len:
        crossattn_emb = F.pad(crossattn_emb, (0, 0, 0, seq_len - cur_seq))
    elif cur_seq > seq_len:
        crossattn_emb = crossattn_emb[:, :seq_len, :]

    crossattn_emb = crossattn_emb.squeeze(0).to(dtype=torch.bfloat16).cpu()
    pooled = crossattn_emb.amax(dim=0)  # matches load_cached_text_features fallback
    return crossattn_emb, pooled


def encode_uncond_crossattn(
    qwen3_path: str,
    dit_path: str,
    *,
    t5_tokenizer_path: str | None = None,
    seq_len: int = DEFAULT_SEQ_LEN,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``T5("")`` through Qwen3 + LLM adapter, zero padding positions,
    pad/truncate to ``seq_len``. Returns ``(crossattn_emb, pooled)``, both bf16
    on CPU. Shape: ``(seq_len, 1024)`` and ``(1024,)``.

    Mirrors the negative-prompt path in ``library/inference/text.py:99-127``
    and the encode path in ``scripts/preprocess/cache_text_embeddings.py:71-105``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from library.anima import weights as anima_utils
    from library.anima.strategy import AnimaTextEncodingStrategy, AnimaTokenizeStrategy

    logger.info(f"Loading Qwen3 text encoder from {qwen3_path} ...")
    text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(
        qwen3_path, dtype=torch.bfloat16, device=str(device)
    )
    t5_tokenizer = anima_utils.load_t5_tokenizer(t5_tokenizer_path)

    logger.info(f"Loading LLM adapter from {dit_path} ...")
    llm_adapter = anima_utils.load_llm_adapter(
        dit_path, dtype=torch.bfloat16, device=str(device)
    )

    tokenize_strategy = AnimaTokenizeStrategy(
        qwen3_tokenizer=qwen3_tokenizer, t5_tokenizer=t5_tokenizer
    )
    encoding_strategy = AnimaTextEncodingStrategy()

    crossattn_emb, pooled = encode_uncond_with_models(
        text_encoder,
        tokenize_strategy,
        encoding_strategy,
        llm_adapter,
        seq_len=seq_len,
        device=device,
    )

    text_encoder.to("cpu")
    del text_encoder, llm_adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return crossattn_emb, pooled


def load_uncond_crossattn(path: str, device, dtype) -> torch.Tensor:
    """Load the ``T5("")`` sidecar staged by ``make preprocess-te`` and return a
    ``(1, seq, 1024)`` tensor on ``device`` in ``dtype``. Used as the student's
    unconditional cross-attention input; replaces ``torch.zeros_like(...)``,
    which is neither paper-faithful nor what Anima uses at CFG-uncond inference.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Unconditional TE sidecar not found at {path!r}. "
            f"Run `make preprocess-te` (or `python tasks.py preprocess-te`) first."
        )
    sd = _load_safetensors(path)
    uncond = sd.get("crossattn_emb")
    if uncond is None:
        raise KeyError(
            f"Expected key 'crossattn_emb' in {path!r}; got {list(sd.keys())}"
        )
    if uncond.dim() != 2:
        raise ValueError(
            f"Expected (seq, dim) tensor in {path!r}; got shape {tuple(uncond.shape)}"
        )
    return uncond.to(device=device, dtype=dtype).unsqueeze(0).contiguous()


def uncond_for_batch(uncond_1: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast ``uncond_1`` (1, S_u, D) to ``(B, S_ref, D)`` matching ``ref``.
    Pads with zeros (attention sinks) if ``S_u < S_ref``; truncates if larger.
    """
    B, S_ref, _D = ref.shape
    S_u = uncond_1.shape[1]
    if S_u < S_ref:
        uncond_1 = F.pad(uncond_1, (0, 0, 0, S_ref - S_u))
    elif S_u > S_ref:
        uncond_1 = uncond_1[:, :S_ref, :]
    return uncond_1.expand(B, -1, -1)
