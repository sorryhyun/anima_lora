# Anima Strategy Classes

import os
import random
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from safetensors import safe_open as _safe_open
from safetensors.torch import (
    load_file as _load_safetensors,
    save_file as _save_safetensors,
)

from library import train_util
from library.anima import training as anima_train_utils, weights as anima_utils
from library.strategy_base import (
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
        # Do not handle dropout here; handled dataset-side or in drop_cached_text_encoder_outputs()

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

    def drop_cached_text_encoder_outputs(
        self,
        prompt_embeds: torch.Tensor,
        attn_mask: torch.Tensor,
        t5_input_ids: torch.Tensor,
        t5_attn_mask: torch.Tensor,
        crossattn_emb: Optional[torch.Tensor] = None,
        caption_dropout_rates: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Apply dropout to cached text encoder outputs.

        Called during training when using cached outputs.
        Replaces dropped items with pre-cached unconditional embeddings (from encoding "")
        to match diffusion-pipe-main behavior.
        """
        if caption_dropout_rates is None:
            outputs = [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]
            if crossattn_emb is not None:
                outputs.append(crossattn_emb)
            return outputs

        # Vectorized dropout: single RNG call, boolean-indexed assignment
        drop_mask = (
            torch.rand(prompt_embeds.shape[0], device=caption_dropout_rates.device)
            < caption_dropout_rates
        )
        if not drop_mask.any():
            outputs = [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]
            if crossattn_emb is not None:
                outputs.append(crossattn_emb)
            return outputs

        # Only clone tensors that will be modified
        prompt_embeds = prompt_embeds.clone()
        prompt_embeds[drop_mask] = 0
        if attn_mask is not None:
            attn_mask = attn_mask.clone()
            attn_mask[drop_mask] = 0
        if t5_input_ids is not None:
            t5_input_ids = t5_input_ids.clone()
            t5_input_ids[drop_mask, 0] = 1  # Set to </s> token ID
            t5_input_ids[drop_mask, 1:] = 0
        if t5_attn_mask is not None:
            t5_attn_mask = t5_attn_mask.clone()
            t5_attn_mask[drop_mask, 0] = 1
            t5_attn_mask[drop_mask, 1:] = 0
        if crossattn_emb is not None:
            crossattn_emb = crossattn_emb.clone()
            crossattn_emb[drop_mask] = 0

        outputs = [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]
        if crossattn_emb is not None:
            outputs.append(crossattn_emb)
        return outputs


class _VariantOutputs:
    """Wrapper that randomly selects a cached variant each time it's accessed as a sequence.

    Used for in-memory caption shuffle variants so that train_util.__getitem__() gets a
    different shuffled variant per epoch without modifying shared code.

    Picks a new random variant on __getitem__(0) (start of collation loop) and reuses
    the same variant for subsequent indices within the same collation round.
    """

    def __init__(self, variants: list, caption_dropout_rate):
        self._variants = variants  # list of tuples, each is (pe, am, t5, t5m) or (pe, am, t5, t5m, ce)
        self._caption_dropout_rate = caption_dropout_rate
        self._resolved = None

    def _resolve(self):
        v = self._variants[random.randint(0, len(self._variants) - 1)]
        self._resolved = (*v, self._caption_dropout_rate)
        return self._resolved

    def __len__(self):
        # Collation calls len() on the first element — resolve a new variant
        return len(self._resolve())

    def __getitem__(self, idx):
        if idx == 0 or self._resolved is None:
            self._resolve()
        return self._resolved[idx]

    def __iter__(self):
        return iter(self._resolve())


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
        caption_shuffle_variants: int = 0,
    ) -> None:
        super().__init__(
            cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial
        )
        self.cache_llm_adapter_outputs = cache_llm_adapter_outputs
        self.caption_shuffle_variants = caption_shuffle_variants

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return (
            os.path.splitext(image_abs_path)[0]
            + self.ANIMA_TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX
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
        data = _load_safetensors(cache_path)

        if "num_variants" in data:
            num_variants = int(data["num_variants"])
            vi = random.randint(0, num_variants - 1)
            prompt_embeds = data[f"prompt_embeds_v{vi}"]
            attn_mask = data[f"attn_mask_v{vi}"]
            t5_input_ids = data[f"t5_input_ids_v{vi}"]
            t5_attn_mask = data[f"t5_attn_mask_v{vi}"]
            crossattn_key = f"crossattn_emb_v{vi}"
            crossattn_emb = (
                data[crossattn_key]
                if self.cache_llm_adapter_outputs and crossattn_key in data
                else None
            )
        else:
            prompt_embeds = data["prompt_embeds"]
            attn_mask = data["attn_mask"]
            t5_input_ids = data["t5_input_ids"]
            t5_attn_mask = data["t5_attn_mask"]
            crossattn_emb = (
                data["crossattn_emb"]
                if self.cache_llm_adapter_outputs and "crossattn_emb" in data
                else None
            )

        caption_dropout_rate = data["caption_dropout_rate"]
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

    @staticmethod
    def _generate_shuffled_captions(caption: str, num_variants: int) -> List[str]:
        """Generate N shuffled caption variants using the shared smart shuffle logic."""
        tags = [t.strip() for t in caption.split(",")]
        variants = []
        for _ in range(num_variants):
            shuffled = anima_train_utils.anima_smart_shuffle_caption(tags.copy())
            variants.append(", ".join(shuffled))
        return variants

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
        anima_text_encoding_strategy: AnimaTextEncodingStrategy = text_encoding_strategy

        if self.caption_shuffle_variants > 0:
            self._cache_batch_outputs_with_variants(
                tokenize_strategy, models, anima_text_encoding_strategy, infos
            )
        else:
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

    def _cache_batch_outputs_with_variants(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: "AnimaTextEncodingStrategy",
        infos: List,
    ):
        """Cache N shuffled caption variants per image."""
        N = self.caption_shuffle_variants

        # For each info, generate N shuffled captions
        # We batch all variants across all infos for efficient encoding
        all_captions = []  # flat list of all variant captions
        variant_map = []  # (info_idx, variant_idx) for each entry in all_captions
        for info_idx, info in enumerate(infos):
            variants = self._generate_shuffled_captions(info.caption, N)
            for vi, caption in enumerate(variants):
                all_captions.append(caption)
                variant_map.append((info_idx, vi))

        # Encode all variants in one batch
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, crossattn_emb = (
            self._encode_to_tensors(
                tokenize_strategy, models, text_encoding_strategy, all_captions
            )
        )

        # Group results by info and save
        for i, info in enumerate(infos):
            caption_dropout_rate = torch.tensor(
                info.caption_dropout_rate, dtype=torch.float32
            )
            save_dict = {
                "num_variants": torch.tensor(N, dtype=torch.int64),
                "caption_dropout_rate": caption_dropout_rate,
            }

            for vi in range(N):
                flat_idx = i * N + vi
                pe_i, am_i, t5_i, t5m_i, ce_i = self._trim_outputs(
                    prompt_embeds[flat_idx],
                    attn_mask[flat_idx],
                    t5_input_ids[flat_idx],
                    t5_attn_mask[flat_idx],
                    crossattn_emb[flat_idx] if crossattn_emb is not None else None,
                )
                save_dict[f"prompt_embeds_v{vi}"] = pe_i
                save_dict[f"attn_mask_v{vi}"] = am_i
                save_dict[f"t5_input_ids_v{vi}"] = t5_i
                save_dict[f"t5_attn_mask_v{vi}"] = t5m_i
                if ce_i is not None:
                    save_dict[f"crossattn_emb_v{vi}"] = ce_i

            if self.cache_to_disk:
                _save_safetensors(save_dict, info.text_encoder_outputs_npz)
            else:
                # Build list of variant tuples for in-memory random selection
                variants = []
                for vi in range(N):
                    v = (
                        save_dict[f"prompt_embeds_v{vi}"],
                        save_dict[f"attn_mask_v{vi}"],
                        save_dict[f"t5_input_ids_v{vi}"],
                        save_dict[f"t5_attn_mask_v{vi}"],
                    )
                    if f"crossattn_emb_v{vi}" in save_dict:
                        v = (*v, save_dict[f"crossattn_emb_v{vi}"])
                    variants.append(v)
                info.text_encoder_outputs = _VariantOutputs(
                    variants, caption_dropout_rate
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
        self, absolute_path: str, image_size: Tuple[int, int]
    ) -> str:
        return (
            os.path.splitext(absolute_path)[0]
            + f"_{image_size[0]:04d}x{image_size[1]:04d}"
            + self.ANIMA_LATENTS_NPZ_SUFFIX
        )

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

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae_device)
