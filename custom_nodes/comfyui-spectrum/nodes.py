"""ComfyUI node definitions for Spectrum inference acceleration."""

import comfy.samplers
import folder_paths

from .mod_guidance import AUTO_ADAPTER_SENTINEL, setup_mod_guidance
from .spectrum import spectrum_sample


def _adapter_choices():
    return [AUTO_ADAPTER_SENTINEL] + folder_paths.get_filename_list("loras")

# ---------------------------------------------------------------------------
# Common input definitions
# ---------------------------------------------------------------------------

_KSAMPLER_INPUTS = {
    "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
    "seed": (
        "INT",
        {
            "default": 0,
            "min": 0,
            "max": 0xFFFFFFFFFFFFFFFF,
            "control_after_generate": True,
        },
    ),
    "steps": ("INT", {"default": 28, "min": 1, "max": 10000}),
    "cfg": (
        "FLOAT",
        {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01},
    ),
    "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
    "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
    "positive": ("CONDITIONING",),
    "negative": ("CONDITIONING",),
    "latent_image": ("LATENT",),
    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
}

_MOD_GUIDANCE_BASE_INPUTS = {
    "clip": ("CLIP", {"tooltip": "CLIP encoder for encoding positive quality tags."}),
    "quality_tags": (
        "STRING",
        {
            "default": "absurdres, highres, masterpiece, best quality, score_9, score_8, newest, year 2025, year 2024",
            "multiline": True,
            "dynamicPrompts": True,
            "tooltip": "Quality tags to steer generation toward via modulation.",
        },
    ),
    "mod_w": (
        "FLOAT",
        {
            "default": 3.0,
            "min": -20.0,
            "max": 20.0,
            "step": 0.1,
            "tooltip": "Modulation guidance strength. Steers t_emb toward quality tags.",
        },
    ),
}


def _mod_guidance_advanced_inputs():
    return {
        "clip": _MOD_GUIDANCE_BASE_INPUTS["clip"],
        "adapter": (
            _adapter_choices(),
            {
                "tooltip": (
                    "pooled_text_proj safetensors adapter. "
                    f"'{AUTO_ADAPTER_SENTINEL}' fetches the default ~12MB weight "
                    "from the anima_lora release page on first use."
                ),
            },
        ),
        "quality_tags": _MOD_GUIDANCE_BASE_INPUTS["quality_tags"],
        "mod_w": _MOD_GUIDANCE_BASE_INPUTS["mod_w"],
    }

_SPECTRUM_INPUTS = {
    "window_size": (
        "FLOAT",
        {
            "default": 2.0,
            "min": 1.0,
            "max": 10.0,
            "step": 0.25,
            "tooltip": "Initial caching window N — actual forward every floor(N) steps.",
        },
    ),
    "flex_window": (
        "FLOAT",
        {
            "default": 0.25,
            "min": 0.0,
            "max": 2.0,
            "step": 0.05,
            "tooltip": "Window growth rate — N increases by this after each actual forward.",
        },
    ),
    "warmup_steps": (
        "INT",
        {
            "default": 7,
            "min": 0,
            "max": 50,
            "tooltip": "Number of initial steps that always run actual forwards.",
        },
    ),
    "blend_w": (
        "FLOAT",
        {
            "default": 0.3,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "tooltip": "Chebyshev/Taylor blend weight (1.0 = pure Chebyshev).",
        },
    ),
    "cheby_degree": (
        "INT",
        {
            "default": 3,
            "min": 1,
            "max": 10,
            "tooltip": "Number of Chebyshev basis functions.",
        },
    ),
    "ridge_lambda": (
        "FLOAT",
        {
            "default": 0.1,
            "min": 0.001,
            "max": 10.0,
            "step": 0.01,
            "tooltip": "Ridge regression regularization strength.",
        },
    ),
}

_SPECTRUM_DEFAULTS = dict(
    window_size=2.0,
    flex_window=0.25,
    warmup_steps=7,
    blend_w=0.3,
    cheby_degree=3,
    ridge_lambda=0.1,
)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


class SpectrumKSampler:
    """Drop-in KSampler replacement with Spectrum acceleration using sensible defaults."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": dict(_KSAMPLER_INPUTS)}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = (
        "Spectrum-accelerated sampler. Drop-in KSampler replacement that "
        "skips transformer blocks on predicted steps via Chebyshev polynomial "
        "feature forecasting for ~2-3x speedup. Uses sensible defaults."
    )

    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=1.0,
    ):
        return spectrum_sample(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise,
            **_SPECTRUM_DEFAULTS,
        )


class SpectrumKSamplerModGuidance:
    """Spectrum sampler with modulation guidance — quality steering via learned projection."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {**_KSAMPLER_INPUTS, **_MOD_GUIDANCE_BASE_INPUTS}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = (
        "Spectrum-accelerated sampler with modulation guidance. "
        "Steers generation toward quality tags via a learned pooled-text "
        "projection into the AdaLN timestep embedding. The default ~12MB "
        "pooled_text_proj adapter is auto-downloaded on first use. Quality "
        "tags are encoded through the full CLIP + LLM adapter pipeline for "
        "correct post-adapter pooling. Uses sensible Spectrum defaults."
    )

    def sample(
        self,
        model,
        clip,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        quality_tags,
        mod_w,
        denoise=1.0,
    ):
        m = model.clone()
        setup_mod_guidance(m, clip, positive, negative, None, quality_tags, mod_w)
        return spectrum_sample(
            m,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise,
            **_SPECTRUM_DEFAULTS,
        )


class SpectrumKSamplerAdvanced:
    """Full Spectrum sampler with modulation guidance and tunable forecasting parameters."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **_KSAMPLER_INPUTS,
                **_mod_guidance_advanced_inputs(),
                **_SPECTRUM_INPUTS,
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = (
        "Spectrum-accelerated sampler with modulation guidance and full "
        "control over forecasting parameters. Combines quality steering "
        "via learned pooled-text projection with adjustable Chebyshev "
        "polynomial feature forecasting for tuned speed/quality tradeoff."
    )

    def sample(
        self,
        model,
        clip,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        adapter,
        quality_tags,
        mod_w,
        denoise=1.0,
        window_size=2.0,
        flex_window=0.25,
        warmup_steps=7,
        blend_w=0.3,
        cheby_degree=3,
        ridge_lambda=0.1,
    ):
        m = model.clone()
        setup_mod_guidance(m, clip, positive, negative, adapter, quality_tags, mod_w)
        return spectrum_sample(
            m,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise,
            window_size=window_size,
            flex_window=flex_window,
            warmup_steps=warmup_steps,
            blend_w=blend_w,
            cheby_degree=cheby_degree,
            ridge_lambda=ridge_lambda,
        )


NODE_CLASS_MAPPINGS = {
    "SpectrumKSampler": SpectrumKSampler,
    "SpectrumKSamplerModGuidance": SpectrumKSamplerModGuidance,
    "SpectrumKSamplerAdvanced": SpectrumKSamplerAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectrumKSampler": "KSampler (Spectrum)",
    "SpectrumKSamplerModGuidance": "KSampler (Spectrum + Mod Guidance)",
    "SpectrumKSamplerAdvanced": "KSampler (Spectrum + Mod Guidance Advanced)",
}
