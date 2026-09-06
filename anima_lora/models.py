"""Model loading + device helpers — ``anima_lora.models``.

| export | canonical home |
|--------|----------------|
| ``load_dit_model`` | ``library.inference.models`` |
| ``load_anima_model`` | ``library.anima.weights`` |
| ``load_vae`` | ``library.models.qwen_vae`` |
| ``default_checkpoints`` / ``DefaultCheckpoints`` | ``library.env`` |
| ``str_to_dtype`` | ``library.runtime.device`` |
| ``VocabPack`` / ``load_vocab_pack`` / ``attach_vocab_pack`` | ``library.anima.vocab_pack`` |
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
        # CJK vocab pack (text-encoder asset, not a LoRA): load once, hook
        # onto a DiT / LLMAdapter. GenerationRequest(vocab_pack=…) is the
        # request-driven path; these are the primitives behind it.
        "VocabPack": "library.anima.vocab_pack",
        "load_vocab_pack": "library.anima.vocab_pack",
        "attach_vocab_pack": "library.anima.vocab_pack",
    },
)
