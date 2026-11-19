"""Model loading + device helpers — ``anima_lora.models``.

| export | canonical home |
|--------|----------------|
| ``load_dit_model`` | ``library.inference.models`` |
| ``load_anima_model`` | ``library.anima.weights`` |
| ``load_vae`` | ``library.models.qwen_vae`` |
| ``default_checkpoints`` / ``DefaultCheckpoints`` | ``library.env`` |
| ``str_to_dtype`` | ``library.runtime.device`` |
| ``VocabPack`` / ``load_vocab_pack`` / ``attach_vocab_pack`` | ``library.anima.vocab_pack`` |
| ``PixelHead`` / ``load_pixel_head`` / ``PixelTap`` | ``library.anima.pixel_head`` |
"""

from __future__ import annotations

from anima_lora._lazy import attach

attach(
    globals(),
    {
        "load_dit_model": "library.inference.models",
        "load_anima_model": "library.anima.weights",
        "load_vae": "library.models.qwen_vae",
        "default_checkpoints": "library.env",
        "DefaultCheckpoints": "library.env",
        "str_to_dtype": "library.runtime.device",
        # CJK vocab pack primitives; GenerationRequest(vocab_pack=…) is the
        # request-driven path.
        "VocabPack": "library.anima.vocab_pack",
        "load_vocab_pack": "library.anima.vocab_pack",
        "attach_vocab_pack": "library.anima.vocab_pack",
        # pixel head: the DiT's last features → RGB in place of the VAE;
        # GenerationRequest(decoder="pixel_head") is the request-driven path.
        "PixelHead": "library.anima.pixel_head",
        "load_pixel_head": "library.anima.pixel_head",
        "PixelTap": "library.anima.pixel_head",
    },
)
