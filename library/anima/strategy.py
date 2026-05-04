# Anima Strategy Classes

import os
import random
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from safetensors import safe_open as _safe_open
from safetensors.torch import (
    save_file as _save_safetensors,
)

from library.anima import weights as anima_utils
from library.datasets import base as _datasets_base
from library.io.cache import resolve_cache_path
from library.runtime.device import clean_memory_on_device
from library.anima.text_strategies import (
    LatentsCachingStrategy,
    TextEncodingStrategy,
    TokenizeStrategy,
    TextEncoderOutputsCachingStrategy,
)
from library.models import qwen_vae as qwen_image_autoencoder_kl

from library.log import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)

# Module-level latch so the legacy-cache warning fires once per process,
# not once per cache file.
_warned_legacy_variants_cache = False


class AnimaTokenizeStrategy(TokenizeStrategy):
    """Tokenize strategy for Anima: dual tokenization with Qwen3 + T5.

    Qwen3 tokens are used for the text encoder.
    T5 tokens are used as target input IDs for the LLM Adapter (NOT encoded by T5).

    Can be initialized with either pre-loaded tokenizer objects or paths to load from.
    """

    def __init__(
        self,
        qwen3_tokenizer=None,
        t5_tokenizer=None,
        qwen3_max_length: int = 512,
        t5_max_length: int = 512,
        qwen3_path: Optional[str] = None,
        t5_tokenizer_path: Optional[str] = None,
    ) -> None:
        # Load tokenizers from paths if not provided directly
        if qwen3_tokenizer is None:
            if qwen3_path is None:
                raise ValueError(
                    "Either qwen3_tokenizer or qwen3_path must be provided"
                )
            qwen3_tokenizer = anima_utils.load_qwen3_tokenizer(qwen3_path)
        if t5_tokenizer is None:
            t5_tokenizer = anima_utils.load_t5_tokenizer(t5_tokenizer_path)

        self.qwen3_tokenizer = qwen3_tokenizer
        self.qwen3_max_length = qwen3_max_length
        self.t5_tokenizer = t5_tokenizer
        self.t5_max_length = t5_max_length

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text

        # Tokenize with Qwen3
        qwen3_encoding = self.qwen3_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.qwen3_max_length,
        )
        qwen3_input_ids = qwen3_encoding["input_ids"]
        qwen3_attn_mask = qwen3_encoding["attention_mask"]

        # Tokenize with T5 (for LLM Adapter target tokens)
        t5_encoding = self.t5_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.t5_max_length,
        )
        t5_input_ids = t5_encoding["input_ids"]
        t5_attn_mask = t5_encoding["attention_mask"]
        return [qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask]


class AnimaTextEncodingStrategy(TextEncodingStrategy):
    """Text encoding strategy for Anima.

    Encodes Qwen3 tokens through the Qwen3 text encoder to get hidden states.
    T5 tokens are passed through unchanged (only used by LLM Adapter).
    """

    def __init__(self) -> None:
        super().__init__()

    def encode_tokens(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Encode Qwen3 tokens and return embeddings + T5 token IDs.

        Args:
            models: [qwen3_text_encoder]
            tokens: [qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask]

        Returns:
            [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]
        """
        # Do not handle dropout here; handled dataset-side or in apply_caption_dropout_inplace()

        qwen3_text_encoder = models[0]
        qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask = tokens

        encoder_device = qwen3_text_encoder.device

        qwen3_input_ids = qwen3_input_ids.to(encoder_device)
        qwen3_attn_mask = qwen3_attn_mask.to(encoder_device)
        outputs = qwen3_text_encoder(
            input_ids=qwen3_input_ids, attention_mask=qwen3_attn_mask
        )
        prompt_embeds = outputs.last_hidden_state

        prompt_embeds[~qwen3_attn_mask.bool()] = 0

        return [prompt_embeds, qwen3_attn_mask, t5_input_ids, t5_attn_mask]

    def apply_caption_dropout_inplace(
        self,
        caption_dropout_rates: torch.Tensor,
        *,
        prompt_embeds: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        t5_input_ids: Optional[torch.Tensor] = None,
        t5_attn_mask: Optional[torch.Tensor] = None,
        crossattn_emb: Optional[torch.Tensor] = None,
    ) -> None:
        """Zero per-sample text conditioning at the per-sample dropout rate.

        Operates in-place on whichever tensors are passed — caller must own
        them (e.g. fresh outputs of a `.to(device)` H2D copy that aren't
        aliased to the dataloader's CPU tensors). Pass only the tensors
        actually consumed downstream so the unused ones can stay on CPU.

        Replaces dropped items with the unconditional encoding (encoding "")
        to match diffusion-pipe-main behavior.
        """
        device_tensor = next(
            (
                t
                for t in (prompt_embeds, crossattn_emb, t5_attn_mask, attn_mask, t5_input_ids)
                if t is not None
            ),
            None,
        )
        if device_tensor is None:
            return
        device = device_tensor.device
        rates = caption_dropout_rates.to(device, non_blocking=True)
        # No `.any()` early-out: that would force a GPU sync. Indexed
        # assignment with an all-False mask is a cheap no-op on device.
        drop_mask = torch.rand(rates.shape[0], device=device) < rates

        if prompt_embeds is not None:
            prompt_embeds[drop_mask] = 0
        if attn_mask is not None:
            attn_mask[drop_mask] = 0
        if t5_input_ids is not None:
            t5_input_ids[drop_mask, 0] = 1  # </s> token ID
            t5_input_ids[drop_mask, 1:] = 0
        if t5_attn_mask is not None:
            t5_attn_mask[drop_mask, 0] = 1
            t5_attn_mask[drop_mask, 1:] = 0
        if crossattn_emb is not None:
            crossattn_emb[drop_mask] = 0


class AnimaTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    """Caching strategy for Anima text encoder outputs.

    Caches: prompt_embeds (bf16), attn_mask (int32), t5_input_ids (int64), t5_attn_mask (int32)
    """

    ANIMA_TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX = "_anima_te.safetensors"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
        cache_llm_adapter_outputs: bool = False,
        use_shuffled_caption_variants: bool = False,
    ) -> None:
        super().__init__(
            cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial
        )
        self.cache_llm_adapter_outputs = cache_llm_adapter_outputs
        self.use_shuffled_caption_variants = use_shuffled_caption_variants

    def get_outputs_npz_path(
        self, image_abs_path: str, cache_dir: Optional[str] = None
    ) -> str:
        return resolve_cache_path(
            image_abs_path,
            self.ANIMA_TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX,
            cache_dir=cache_dir,
        )

    def is_disk_cached_outputs_expected(self, cache_path: str) -> bool:
        if not self.cache_to_disk:
            return False
        if not os.path.exists(cache_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            with _safe_open(cache_path, framework="pt") as f:
                keys = set(f.keys())
                if "num_variants" in keys:
                    num_variants = int(f.get_tensor("num_variants"))
            if "num_variants" in keys:
                for vi in range(num_variants):
                    if f"prompt_embeds_v{vi}" not in keys:
                        return False
                    if f"attn_mask_v{vi}" not in keys:
                        return False
                    if f"t5_input_ids_v{vi}" not in keys:
                        return False
                    if f"t5_attn_mask_v{vi}" not in keys:
                        return False
                    if (
                        self.cache_llm_adapter_outputs
                        and f"crossattn_emb_v{vi}" not in keys
                    ):
                        return False
            else:
                for k in ("prompt_embeds", "attn_mask", "t5_input_ids", "t5_attn_mask"):
                    if k not in keys:
                        return False
                if self.cache_llm_adapter_outputs and "crossattn_emb" not in keys:
                    return False
            if "caption_dropout_rate" not in keys:
                return False
        except Exception as e:
            logger.error(f"Error loading file: {cache_path}")
            raise e

        return True

    def load_outputs_npz(self, cache_path: str) -> list:
        # Lazy per-tensor read via safe_open: when the cache holds N preprocessed
        # variants × cache_llm_adapter_outputs, the file has 5×N tensors but only
        # one variant is consumed per step. load_file() materializes everything
        # and starves the dataloader workers; safe_open + get_tensor pulls just
        # the chosen variant's bytes from the mmap.
        with _safe_open(cache_path, framework="pt") as f:
            keys = set(f.keys())
            has_variants = "num_variants" in keys
            if has_variants and self.use_shuffled_caption_variants:
                num_variants = int(f.get_tensor("num_variants"))
                v0_intact = "v0_intact" in keys
                if not v0_intact:
                    # Legacy cache: every variant is shuffled (no pristine v0).
                    # Fall back to uniform sampling and warn once so the user
                    # knows to re-cache for the 20%/80% weighted behavior.
                    global _warned_legacy_variants_cache
                    if not _warned_legacy_variants_cache:
                        logger.warning(
                            "Loaded a legacy multi-variant TE cache without the "
                            "`v0_intact` marker (e.g. %s). Sampling uniformly "
                            "across v0..v%d. Re-run `make preprocess-te` to "
                            "regenerate caches with v0=pristine and "
                            "20%%/80%% weighted sampling.",
                            cache_path,
                            num_variants - 1,
                        )
                        _warned_legacy_variants_cache = True
                    vi = random.randint(0, num_variants - 1)
                elif num_variants <= 1:
                    vi = 0
                else:
                    # 20% pristine v0, 80% uniform over v1..v{N-1}.
                    vi = (
                        0
                        if random.random() < 0.2
                        else random.randint(1, num_variants - 1)
                    )
                prompt_embeds = f.get_tensor(f"prompt_embeds_v{vi}")
                attn_mask = f.get_tensor(f"attn_mask_v{vi}")
                t5_input_ids = f.get_tensor(f"t5_input_ids_v{vi}")
                t5_attn_mask = f.get_tensor(f"t5_attn_mask_v{vi}")
                crossattn_key = f"crossattn_emb_v{vi}"
                crossattn_emb = (
                    f.get_tensor(crossattn_key)
                    if self.cache_llm_adapter_outputs and crossattn_key in keys
                    else None
                )
            elif has_variants:
                # Variants on disk but the user opted out — pin to v0 deterministically.
                prompt_embeds = f.get_tensor("prompt_embeds_v0")
                attn_mask = f.get_tensor("attn_mask_v0")
                t5_input_ids = f.get_tensor("t5_input_ids_v0")
                t5_attn_mask = f.get_tensor("t5_attn_mask_v0")
                crossattn_emb = (
                    f.get_tensor("crossattn_emb_v0")
                    if self.cache_llm_adapter_outputs and "crossattn_emb_v0" in keys
                    else None
                )
            else:
                # Single-variant cache. Loaded as-is whether or not the user
                # asked for shuffles — silent fallback so a bool flip doesn't
                # require re-preprocessing.
                prompt_embeds = f.get_tensor("prompt_embeds")
                attn_mask = f.get_tensor("attn_mask")
                t5_input_ids = f.get_tensor("t5_input_ids")
                t5_attn_mask = f.get_tensor("t5_attn_mask")
                crossattn_emb = (
                    f.get_tensor("crossattn_emb")
                    if self.cache_llm_adapter_outputs and "crossattn_emb" in keys
                    else None
                )

            caption_dropout_rate = f.get_tensor("caption_dropout_rate")
        if crossattn_emb is None:
            return [
                prompt_embeds,
                attn_mask,
                t5_input_ids,
                t5_attn_mask,
                caption_dropout_rate,
            ]
        return [
            prompt_embeds,
            attn_mask,
            t5_input_ids,
            t5_attn_mask,
            crossattn_emb,
            caption_dropout_rate,
        ]

    def _encode_to_tensors(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: "AnimaTextEncodingStrategy",
        captions: List[str],
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
    ]:
        """Tokenize, encode, and optionally run LLM adapter. Returns typed CPU tensors."""
        tokens_and_masks = tokenize_strategy.tokenize(captions)
        with torch.no_grad():
            prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = (
                text_encoding_strategy.encode_tokens(
                    tokenize_strategy, models, tokens_and_masks
                )
            )

        crossattn_emb = None
        if self.cache_llm_adapter_outputs:
            if len(models) < 2 or models[1] is None:
                raise ValueError(
                    "cache_llm_adapter_outputs requires llm_adapter model to be passed as models[1]"
                )
            llm_adapter = models[1]
            adapter_device = next(llm_adapter.parameters()).device
            prompt_embeds_for_adapter = prompt_embeds.to(adapter_device)
            attn_mask_for_adapter = (
                attn_mask.to(adapter_device) if attn_mask is not None else None
            )
            t5_input_ids_for_adapter = t5_input_ids.to(adapter_device, dtype=torch.long)
            t5_attn_mask_for_adapter = t5_attn_mask.to(adapter_device)
            with torch.no_grad():
                crossattn_emb = llm_adapter(
                    source_hidden_states=prompt_embeds_for_adapter,
                    target_input_ids=t5_input_ids_for_adapter,
                    target_attention_mask=t5_attn_mask_for_adapter,
                    source_attention_mask=attn_mask_for_adapter,
                )
                crossattn_emb[~t5_attn_mask_for_adapter.bool()] = 0

        # Convert to typed CPU tensors: bf16 for embeddings, int for IDs/masks
        prompt_embeds = prompt_embeds.to(dtype=torch.bfloat16).cpu()
        attn_mask = attn_mask.to(dtype=torch.int32).cpu()
        t5_input_ids = t5_input_ids.to(dtype=torch.long).cpu()
        t5_attn_mask = t5_attn_mask.to(dtype=torch.int32).cpu()
        if crossattn_emb is not None:
            crossattn_emb = crossattn_emb.to(dtype=torch.bfloat16).cpu()

        return prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, crossattn_emb

    @staticmethod
    def _trim_outputs(
        prompt_embeds_i: torch.Tensor,
        attn_mask_i: torch.Tensor,
        t5_input_ids_i: torch.Tensor,
        t5_attn_mask_i: torch.Tensor,
        crossattn_emb_i: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
    ]:
        """Keep max-padded outputs (pretrained model expects padding tokens in cross-attention)."""
        return (
            prompt_embeds_i,
            attn_mask_i,
            t5_input_ids_i,
            t5_attn_mask_i,
            crossattn_emb_i,
        )

    def cache_batch_outputs(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        infos: List,
    ):
        # Inline caching always writes a single variant. Multi-variant caches
        # are produced exclusively by `preprocess/cache_text_embeddings.py`.
        anima_text_encoding_strategy: AnimaTextEncodingStrategy = text_encoding_strategy
        self._cache_batch_outputs_single(
            tokenize_strategy, models, anima_text_encoding_strategy, infos
        )

    def _cache_batch_outputs_single(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: "AnimaTextEncodingStrategy",
        infos: List,
    ):
        """Original single-variant caching path."""
        captions = [info.caption for info in infos]
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, crossattn_emb = (
            self._encode_to_tensors(
                tokenize_strategy, models, text_encoding_strategy, captions
            )
        )

        for i, info in enumerate(infos):
            pe_i, am_i, t5_i, t5m_i, ce_i = self._trim_outputs(
                prompt_embeds[i],
                attn_mask[i],
                t5_input_ids[i],
                t5_attn_mask[i],
                crossattn_emb[i] if crossattn_emb is not None else None,
            )
            caption_dropout_rate = torch.tensor(
                info.caption_dropout_rate, dtype=torch.float32
            )

            if self.cache_to_disk:
                save_dict = {
                    "prompt_embeds": pe_i,
                    "attn_mask": am_i,
                    "t5_input_ids": t5_i,
                    "t5_attn_mask": t5m_i,
                    "caption_dropout_rate": caption_dropout_rate,
                }
                if ce_i is not None:
                    save_dict["crossattn_emb"] = ce_i
                _save_safetensors(save_dict, info.text_encoder_outputs_npz)
            else:
                if ce_i is None:
                    info.text_encoder_outputs = (
                        pe_i,
                        am_i,
                        t5_i,
                        t5m_i,
                        caption_dropout_rate,
                    )
                else:
                    info.text_encoder_outputs = (
                        pe_i,
                        am_i,
                        t5_i,
                        t5m_i,
                        ce_i,
                        caption_dropout_rate,
                    )

class AnimaLatentsCachingStrategy(LatentsCachingStrategy):
    """Latent caching strategy for Anima using WanVAE.

    WanVAE produces 16-channel latents with spatial downscale 8x.
    Latent shape for images: (B, 16, 1, H/8, W/8)
    """

    ANIMA_LATENTS_NPZ_SUFFIX = "_anima.npz"

    def __init__(
        self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)

    @property
    def cache_suffix(self) -> str:
        return self.ANIMA_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(
        self,
        absolute_path: str,
        image_size: Tuple[int, int],
        cache_dir: Optional[str] = None,
    ) -> str:
        suffix = (
            f"_{image_size[0]:04d}x{image_size[1]:04d}"
            + self.ANIMA_LATENTS_NPZ_SUFFIX
        )
        return resolve_cache_path(absolute_path, suffix, cache_dir=cache_dir)

    def is_disk_cached_latents_expected(
        self,
        bucket_reso: Tuple[int, int],
        npz_path: str,
        flip_aug: bool,
        alpha_mask: bool,
    ):
        return self._default_is_disk_cached_latents_expected(
            8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True
        )

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[List[int]],
        Optional[List[int]],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        return self._default_load_latents_from_disk(8, npz_path, bucket_reso)

    def cache_batch_latents(
        self,
        vae,
        image_infos: List,
        flip_aug: bool,
        alpha_mask: bool,
        random_crop: bool,
    ):
        """Cache batch of latents using Qwen Image VAE.

        vae is expected to be the Qwen Image VAE (AutoencoderKLQwenImage).
        The encoding function handles the mean/std normalization.
        """
        vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage = vae
        vae_device = vae.device
        vae_dtype = vae.dtype

        def encode_by_vae(img_tensor):
            """Encode image tensor to latents.

            img_tensor: (B, C, H, W) in [-1, 1] range (already normalized by IMAGE_TRANSFORMS)
            Qwen Image VAE accepts inputs in (B, C, H, W) or (B, C, 1, H, W) shape.
            Returns latents in (B, 16, 1, H/8, W/8) shape on CPU.
            """
            latents = vae.encode_pixels_to_latents(
                img_tensor
            )  # Keep 4D for input/output
            return latents.to("cpu")

        self._default_cache_batch_latents(
            encode_by_vae,
            vae_device,
            vae_dtype,
            image_infos,
            flip_aug,
            alpha_mask,
            random_crop,
            multi_resolution=True,
        )

        if not _datasets_base.HIGH_VRAM:
            clean_memory_on_device(vae_device)
