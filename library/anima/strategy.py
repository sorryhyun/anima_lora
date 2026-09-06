import os
import random
from dataclasses import dataclass
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
_warned_missing_randomized_cache = False


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
        uncond_crossattn_emb: Optional[torch.Tensor] = None,
    ) -> None:
        """Zero per-sample text conditioning at the per-sample dropout rate.

        Operates in-place on whichever tensors are passed — caller must own
        them (e.g. fresh outputs of a `.to(device)` H2D copy that aren't
        aliased to the dataloader's CPU tensors). Pass only the tensors
        actually consumed downstream so the unused ones can stay on CPU.

        ``uncond_crossattn_emb`` is the T5("") sidecar (shape ``(1, S, D)``
        staged by ``make preprocess-te`` — see
        ``library/preprocess/uncond.py``). When provided, dropped rows of
        ``crossattn_emb`` are replaced with it so the trained adapter sees
        the *same* unconditional embedding at CFG-uncond time that
        ``library/inference/text.py:99-127`` feeds at inference. When None,
        falls back to zeros — legacy behavior that drives the LoRA's
        learned CFG-uncond branch out of distribution.
        """
        device_tensor = next(
            (
                t
                for t in (
                    prompt_embeds,
                    crossattn_emb,
                    t5_attn_mask,
                    attn_mask,
                    t5_input_ids,
                )
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
            if uncond_crossattn_emb is not None:
                # uncond is (1, S_u, D); pad/truncate to crossattn S then
                # rely on (1, S, D) → (K, S, D) broadcast over drop_mask.
                S = crossattn_emb.shape[1]
                u = uncond_crossattn_emb
                if u.shape[1] < S:
                    u = torch.nn.functional.pad(u, (0, 0, 0, S - u.shape[1]))
                elif u.shape[1] > S:
                    u = u[:, :S, :]
                crossattn_emb[drop_mask] = u.to(crossattn_emb.dtype)
            else:
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
        use_shuffled_caption_variants_only: bool = False,
        use_randomized_caption_variants: bool = False,
        use_randomized_caption_variants_only: bool = False,
    ) -> None:
        super().__init__(
            cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial
        )
        self.cache_llm_adapter_outputs = cache_llm_adapter_outputs
        # "only" implies the base flag (it just changes which variants are eligible).
        self.use_randomized_caption_variants = (
            use_randomized_caption_variants or use_randomized_caption_variants_only
        )
        self.use_randomized_caption_variants_only = use_randomized_caption_variants_only
        # Randomized is a sampling MODE inside the variant-consuming branch, so it
        # implies the shuffled gate (and falls back to the v-family if a cache has
        # no r-family on disk — e.g. one preprocessed without a randomize rate).
        self.use_shuffled_caption_variants = (
            use_shuffled_caption_variants
            or use_shuffled_caption_variants_only
            or self.use_randomized_caption_variants
        )
        self.use_shuffled_caption_variants_only = use_shuffled_caption_variants_only

    def get_outputs_npz_path(
        self,
        image_abs_path: str,
        cache_dir: Optional[str] = None,
        image_dir: Optional[str] = None,
    ) -> str:
        return resolve_cache_path(
            image_abs_path,
            self.ANIMA_TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX,
            cache_dir=cache_dir,
            image_dir=image_dir,
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
                # Pack stamp: a cache encoded through a different vocab pack
                # (or none) than the active one still "exists", so say so
                # once — the ids inside no longer match the tokenizer.
                from library.anima.vocab_pack import check_cache_stamp, strategy_pack

                check_cache_stamp(
                    f.metadata(),
                    cache_path,
                    strategy_pack(TokenizeStrategy.get_strategy()),
                )
                if "num_variants" in keys:
                    num_variants = int(f.get_tensor("num_variants"))
            # Adapter-output caches prune the unused Qwen tensors and store only
            # crossattn_emb (+ t5_attn_mask); plain (no-adapter) caches store the
            # Qwen prompt_embeds tuple. Require whichever set the mode consumes.
            if self.cache_llm_adapter_outputs:
                required = ("crossattn_emb", "t5_attn_mask")
            else:
                required = (
                    "prompt_embeds",
                    "attn_mask",
                    "t5_input_ids",
                    "t5_attn_mask",
                )
            if "num_variants" in keys:
                for vi in range(num_variants):
                    for stem in required:
                        if f"{stem}_v{vi}" not in keys:
                            return False
            else:
                for stem in required:
                    if stem not in keys:
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
                has_randomized = "num_randomized" in keys
                # Randomized mode draws the identity-erased r-family when the cache
                # carries one; otherwise it silently degrades to the shuffled
                # v-family (warned once) so a knob flip never forces a re-cache.
                use_r = self.use_randomized_caption_variants and has_randomized
                if self.use_randomized_caption_variants and not has_randomized:
                    global _warned_missing_randomized_cache
                    if not _warned_missing_randomized_cache:
                        logger.warning(
                            "use_randomized_caption_variants is on but the TE cache "
                            "(e.g. %s) has no identity-randomized r-family. Falling "
                            "back to the shuffled v-family. Re-run preprocess with "
                            "--caption_tag_randomize_rate (or set "
                            "caption_tag_randomize_rate in preprocess.toml) to build "
                            "it.",
                            cache_path,
                        )
                        _warned_missing_randomized_cache = True

                if use_r:
                    num_randomized = int(f.get_tensor("num_randomized"))
                    # "only" (either axis) excludes the pristine v0 entirely.
                    only = (
                        self.use_randomized_caption_variants_only
                        or self.use_shuffled_caption_variants_only
                    )
                    if num_randomized < 1:
                        suffix = "v0"
                    elif only:
                        suffix = f"r{random.randint(1, num_randomized)}"
                    else:
                        # 20% pristine v0, 80% uniform over r1..r{num_randomized}.
                        suffix = (
                            "v0"
                            if random.random() < 0.2
                            else f"r{random.randint(1, num_randomized)}"
                        )
                elif num_variants <= 1:
                    suffix = "v0"
                elif self.use_shuffled_caption_variants_only:
                    # Exclude the pristine v0 entirely — uniform over the
                    # shuffled+tag-dropped v1..v{N-1}. (Legacy caches have no
                    # pristine v0 anyway, so v1.. is still the shuffled set.)
                    suffix = f"v{random.randint(1, num_variants - 1)}"
                elif not v0_intact:
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
                    suffix = f"v{random.randint(0, num_variants - 1)}"
                else:
                    # 20% pristine v0, 80% uniform over v1..v{N-1}.
                    suffix = (
                        "v0"
                        if random.random() < 0.2
                        else f"v{random.randint(1, num_variants - 1)}"
                    )
                sfx = f"_{suffix}"
            elif has_variants:
                # Variants on disk but the user opted out — pin to v0 deterministically.
                sfx = "_v0"
            else:
                # Single-variant cache. Loaded as-is whether or not the user
                # asked for shuffles — silent fallback so a bool flip doesn't
                # require re-preprocessing.
                sfx = ""

            # Serve whatever the cache actually holds, independent of this
            # run's cache_llm_adapter_outputs flag. A pruned adapter cache
            # stores only crossattn_emb (no prompt_embeds); gating the read on
            # the run flag hard-crashed in safetensors when preprocess (flag on
            # → wrote crossattn-only) and training (flag off → read
            # prompt_embeds) disagreed. The downstream consumer
            # (library/training/forward/text_conds.py) switches on tuple shape,
            # not the flag, so returning crossattn whenever the file carries it
            # is always correct — and it's the only readable path for a pruned
            # cache. The flag governs writing/encoding, never reading.
            crossattn_key = f"crossattn_emb{sfx}"
            crossattn_emb = (
                f.get_tensor(crossattn_key) if crossattn_key in keys else None
            )
            t5_attn_mask = (
                f.get_tensor(f"t5_attn_mask{sfx}")
                if f"t5_attn_mask{sfx}" in keys
                else None
            )
            if crossattn_emb is not None:
                # Adapter-output mode: the Qwen prompt_embeds / attn_mask /
                # t5_input_ids are unused downstream (see
                # library/training/forward/text_conds.py) — skip the read.
                # Force them to None even for legacy caches that still carry the
                # keys, so a batch mixing pruned + legacy files stacks to a
                # uniform all-None column instead of crashing in
                # none_or_stack_elements.
                prompt_embeds = attn_mask = t5_input_ids = None
            elif f"prompt_embeds{sfx}" in keys:
                prompt_embeds = f.get_tensor(f"prompt_embeds{sfx}")
                attn_mask = f.get_tensor(f"attn_mask{sfx}")
                t5_input_ids = f.get_tensor(f"t5_input_ids{sfx}")
            else:
                raise RuntimeError(
                    f"TE cache {cache_path!r} has neither 'crossattn_emb{sfx}' "
                    f"nor 'prompt_embeds{sfx}' — incompatible cache format "
                    f"(keys: {sorted(keys)}). Re-run `make preprocess-te` to "
                    f"regenerate it."
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
        # are produced exclusively by `scripts/preprocess/cache_text_embeddings.py`.
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
                    "t5_attn_mask": t5m_i,
                    "caption_dropout_rate": caption_dropout_rate,
                }
                if ce_i is not None:
                    # Adapter-output cache: only crossattn_emb is consumed at
                    # train time (+ t5_attn_mask for postfix). Prune the unused
                    # Qwen prompt_embeds / attn_mask / t5_input_ids (~half the
                    # file). Mirrors library/preprocess/text.py.
                    save_dict["crossattn_emb"] = ce_i
                else:
                    save_dict["prompt_embeds"] = pe_i
                    save_dict["attn_mask"] = am_i
                    save_dict["t5_input_ids"] = t5_i
                # Stamp the vocab pack the ids were routed through (None when
                # the stock tokenizer encoded them) so a later run with a
                # different pack state can warn instead of silently training
                # on mismatched ids. Mirrors library/preprocess/text.py.
                from library.anima.vocab_pack import strategy_pack

                pack = strategy_pack(tokenize_strategy)
                _save_safetensors(
                    save_dict,
                    info.text_encoder_outputs_npz,
                    metadata=pack.cache_metadata() if pack is not None else None,
                )
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
        image_dir: Optional[str] = None,
    ) -> str:
        suffix = (
            f"_{image_size[0]:04d}x{image_size[1]:04d}" + self.ANIMA_LATENTS_NPZ_SUFFIX
        )
        return resolve_cache_path(
            absolute_path, suffix, cache_dir=cache_dir, image_dir=image_dir
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

        if not _datasets_base.HIGH_VRAM:
            clean_memory_on_device(vae_device)


# --- Training-side strategy installation ------------------------------------
#
# Anima's tokenize / encode / cache strategies are *process-global* singletons
# (``set_strategy`` / ``get_strategy`` on the base classes in
# ``library.anima.text_strategies``). The inference side installs its pair via
# ``library.inference.text.ensure_text_strategies``; these two functions are
# the training-side counterpart, replacing the per-strategy ``get_*_strategy``
# factory hooks ``train.py`` inherited from the sd-scripts subclass-override
# design (one architecture now — the indirection bought nothing).


@dataclass
class TrainingStrategies:
    """Handles for the strategies :func:`setup_training_strategies` installed.

    The same objects the globals hold — returned so ``train()`` can use them
    directly instead of fishing them back out with ``get_strategy()``.
    """

    tokenize: AnimaTokenizeStrategy
    latents_caching: AnimaLatentsCachingStrategy
    text_encoding: AnimaTextEncodingStrategy


def setup_training_strategies(args) -> TrainingStrategies:
    """Build + install the arg-stable strategy singletons for a training run.

    Call BEFORE dataset construction — dataset init reads the tokenize and
    latents-caching strategies. The text-encoding strategy is stateless, so
    installing it here (earlier than its first use in the TE caching pass) is
    free and keeps every install in one place.

    The text-encoder-OUTPUTS caching strategy is deliberately NOT installed
    here: it reads ``args.cache_llm_adapter_outputs``, which
    ``assert_extra_args`` may still mutate (it auto-disables the flag when text
    caching is off) — install it after that via
    :func:`setup_text_encoder_outputs_caching_strategy`.
    """
    # A CJK vocab pack (``vocab_pack`` in the config chain, "" = off) swaps in
    # the pack-routing tokenizer so inline TE caching and sample prompts see the
    # same T5 ids the preprocess caches were built with. EN stays bit-exact.
    from library.anima.vocab_pack import load_vocab_pack, make_tokenize_strategy

    tokenize = make_tokenize_strategy(
        load_vocab_pack(getattr(args, "vocab_pack", None)),
        qwen3_path=args.qwen3,
        t5_tokenizer_path=args.t5_tokenizer_path,
        qwen3_max_length=args.qwen3_max_token_length,
        t5_max_length=args.t5_max_token_length,
    )
    TokenizeStrategy.set_strategy(tokenize)

    latents_caching = AnimaLatentsCachingStrategy(
        args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
    )
    LatentsCachingStrategy.set_strategy(latents_caching)

    text_encoding = AnimaTextEncodingStrategy()
    TextEncodingStrategy.set_strategy(text_encoding)

    return TrainingStrategies(
        tokenize=tokenize,
        latents_caching=latents_caching,
        text_encoding=text_encoding,
    )


def setup_text_encoder_outputs_caching_strategy(
    args,
) -> Optional[AnimaTextEncoderOutputsCachingStrategy]:
    """Build + install the TE-outputs caching strategy; ``None`` when caching is off.

    Split from :func:`setup_training_strategies` because it reads args that
    ``assert_extra_args`` may mutate (``cache_llm_adapter_outputs``) — call it
    after that, and before anything probes the TE cache for completeness.
    """
    if not args.cache_text_encoder_outputs:
        return None
    strategy = AnimaTextEncoderOutputsCachingStrategy(
        args.cache_text_encoder_outputs_to_disk,
        args.text_encoder_batch_size,
        args.skip_cache_check,
        False,
        cache_llm_adapter_outputs=getattr(args, "cache_llm_adapter_outputs", False),
        use_shuffled_caption_variants=getattr(
            args, "use_shuffled_caption_variants", False
        ),
        use_shuffled_caption_variants_only=getattr(
            args, "use_shuffled_caption_variants_only", False
        ),
        use_randomized_caption_variants=getattr(
            args, "use_randomized_caption_variants", False
        ),
        use_randomized_caption_variants_only=getattr(
            args, "use_randomized_caption_variants_only", False
        ),
    )
    TextEncoderOutputsCachingStrategy.set_strategy(strategy)
    return strategy
