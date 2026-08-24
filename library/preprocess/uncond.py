"""T5("") unconditional sidecar — produce-to-disk half.

Staging (encode → write the sidecar) and the load-or-stage ``ensure``
orchestration. Builds on the Anima-domain encode / load primitives in
:mod:`library.anima.uncond`.

The sidecar is a model-scoped, run-invariant artifact, so it ships as a bundled
package asset (``library/anima/assets/_anima_uncond_te.safetensors``) and is read
directly by ``make turbo``, the mod-guidance distiller
(``project/finished/mod_guidance/``), and training-time caption dropout. The staging here is now only the
*regeneration* path — it overwrites that same asset in place after a base-model
swap (or if the bundled copy is ever missing).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from safetensors.torch import save_file

from library.anima.uncond import (
    DEFAULT_SEQ_LEN,
    DEFAULT_UNCOND_DIR,
    UNCOND_TE_FILENAME,
    default_uncond_path,
    encode_uncond_crossattn,
    encode_uncond_with_models,
    load_uncond_crossattn,
)

logger = logging.getLogger(__name__)


def _write_sidecar(
    out_path: Path, crossattn_emb: torch.Tensor, pooled: torch.Tensor
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"crossattn_emb": crossattn_emb, "pooled": pooled}, str(out_path))
    logger.info(
        f"Wrote {out_path}  (crossattn_emb={tuple(crossattn_emb.shape)}, "
        f"pooled={tuple(pooled.shape)}, dtype={crossattn_emb.dtype})"
    )


def stage_uncond_sidecar(
    cache_dir: Path,
    qwen3_path: str,
    dit_path: str,
    *,
    t5_tokenizer_path: str | None,
    seq_len: int,
    overwrite: bool,
) -> Path:
    """Stand-alone entry point: loads models from disk, encodes, writes
    ``<cache_dir>/_anima_uncond_te.safetensors``.

    Use :func:`stage_uncond_sidecar_with_models` when models are already
    loaded (e.g. inside ``make preprocess-te``).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / UNCOND_TE_FILENAME

    if out_path.exists() and not overwrite:
        logger.info(
            f"Uncond sidecar already exists at {out_path}; pass --overwrite to regenerate."
        )
        return out_path

    crossattn_emb, pooled = encode_uncond_crossattn(
        qwen3_path,
        dit_path,
        t5_tokenizer_path=t5_tokenizer_path,
        seq_len=seq_len,
    )
    _write_sidecar(out_path, crossattn_emb, pooled)
    return out_path


def stage_uncond_sidecar_with_models(
    out_dir: Path,
    text_encoder,
    tokenize_strategy,
    encoding_strategy,
    llm_adapter,
    *,
    seq_len: int = DEFAULT_SEQ_LEN,
    device: torch.device,
    overwrite: bool = False,
) -> Path:
    """Stage the sidecar using already-loaded models. No-op when the file
    already exists unless ``overwrite=True``. Returns the sidecar path.

    Intended for ``scripts/preprocess/cache_text_embeddings.py`` and any other entry
    point that already has Qwen3 + LLM adapter on device — encoding ``T5("")``
    is one extra batch so the marginal cost is ~ms.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / UNCOND_TE_FILENAME
    if out_path.exists() and not overwrite:
        logger.info(f"Uncond sidecar already exists at {out_path}; skipping encode.")
        return out_path
    crossattn_emb, pooled = encode_uncond_with_models(
        text_encoder,
        tokenize_strategy,
        encoding_strategy,
        llm_adapter,
        seq_len=seq_len,
        device=device,
    )
    _write_sidecar(out_path, crossattn_emb, pooled)
    return out_path


def ensure_uncond_crossattn(
    *,
    qwen3_path: str,
    dit_path: str,
    t5_tokenizer_path: str | None = None,
    device,
    dtype: torch.dtype,
    existing: torch.Tensor | None = None,
    seq_len: int = DEFAULT_SEQ_LEN,
) -> torch.Tensor:
    """Return the ``T5("")`` crossattn sidecar as a ``(1, S, 1024)`` tensor.

    Idempotent: pass the previously-loaded tensor as ``existing`` and it's
    returned untouched. The sidecar normally ships as a bundled package asset
    (``library/anima/assets/_anima_uncond_te.safetensors``), so this just loads
    it; the on-demand staging below is a fallback that only fires if that asset
    is somehow missing (e.g. deleted, or a swapped base model). Caller owns where
    it stores the result (e.g. ``TrainState.uncond_crossattn_1``) — this stays
    ignorant of trainer state.
    """
    if existing is not None:
        return existing

    sidecar = default_uncond_path()
    if not sidecar.exists():
        logger.info(
            f"T5('') uncond sidecar missing at {sidecar} — staging "
            f"on demand (would normally be produced by `make preprocess-te`)."
        )
        stage_uncond_sidecar(
            DEFAULT_UNCOND_DIR,
            qwen3_path=qwen3_path,
            dit_path=dit_path,
            t5_tokenizer_path=t5_tokenizer_path,
            seq_len=seq_len,
            overwrite=False,
        )
    uncond = load_uncond_crossattn(str(sidecar), device=device, dtype=dtype)
    logger.info(f"caption dropout uncond loaded: {sidecar} shape={tuple(uncond.shape)}")
    return uncond
