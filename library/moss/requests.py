"""Typed requests for the MOSS-Image jobs: caching, generation, LoRA training.

Torch-free on purpose (a GUI form can be built from these without importing
torch). Each field carries its CLI help in ``metadata``, so the sidecar CLIs
(``scripts/moss/``) read one definition:

    req = GenerateRequest(prompt="…", steps=30)
    argv = req.to_argv()                  # → daemon job argv
    req = GenerateRequest.from_argv(argv) # ← sidecar CLI

``to_argv`` writes only fields that differ from their defaults.

Model directory: ``--model_dir`` if given, else ``ANIMA_MOSS_MODEL_DIR`` (env or
``.env``), else ``models/moss_image`` under the repo home. It holds one diffusers
folder per variant, ``MOSS-Image-Base/`` and ``MOSS-Image-SFT/`` (``transformer/``,
``text_encoder/``, ``vae/``, ``tokenizer/``, ``scheduler/``); SFT differs from Base
in ``transformer/`` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from library.downloads import MOSS_DIR as MODEL_DIR_NAME
from library.downloads import MOSS_VARIANT_DIRS as VARIANT_DIRS
from library.downloads import default_moss_dir
from library.env import resolve_under_home
from library.lines.requests import Request, env_model_dir
from library.lines.requests import request_field as _f

MODEL_DIR_ENV = "ANIMA_MOSS_MODEL_DIR"
VARIANTS = tuple(VARIANT_DIRS)
# LoRA alpha when the request leaves it unset (lokr / loha fall back to rank).
LORA_ALPHA = 128.0
# The reference sampler's defaults (``MOSSImagePipeline.__call__``).
DEFAULT_STEPS = 50
DEFAULT_CFG = 3.0
DEFAULT_UNCOND_SKIP = (22,)
DEFAULT_SHIFT = 5.0
# Qwen3.5-9B weights for the text pass (``library/moss/loader.py::TE_PRECISIONS``).
DEFAULT_TEXT_ENCODER = "int8"
TEXT_ENCODERS = ("int8", "fp8", "bf16")
_TE_HELP = (
    "int8 (default; INT8 W8A8 decoder linears, ~9.2 GB, fetched on first use), fp8 "
    "(float8_e4m3fn weight-only, ~9.2 GB, fetched on first use) or bf16 (the reference "
    "shards). Both quantized files embed ~0.10 rel. L2 from bf16; int8 encodes ~1.7x "
    "faster than fp8"
)


def default_model_dir() -> Path:
    """``ANIMA_MOSS_MODEL_DIR`` if set, else ``<repo home>/models/moss_image``."""
    return env_model_dir(MODEL_DIR_ENV, default_moss_dir())


def resolve_model_dir(model_dir: str | Path | None) -> Path:
    """An explicit ``model_dir`` wins; empty/None falls back to the default."""
    if model_dir:
        return resolve_under_home(model_dir)
    return default_model_dir()


def variant_dir(model_dir: str | Path | None, variant: str) -> Path:
    """The diffusers folder of one variant (``base`` / ``sft``)."""
    return resolve_model_dir(model_dir) / VARIANT_DIRS[variant]


_MODEL_DIR_HELP = f"folder holding MOSS-Image-Base/ and MOSS-Image-SFT/ (default: ${MODEL_DIR_ENV} or models/{MODEL_DIR_NAME})"


@dataclass
class CacheRequest(Request):
    """Precache Qwen3.5-9B text embeddings and FLUX.2 VAE latents for a training folder."""

    SCRIPT: ClassVar[str] = "scripts/moss/cache.py"

    src: str = _f(
        "post_image_dataset/resized",
        "folder of images + .txt captions, subfolders included (default: the "
        "Anima resized tree, whose captions are the revised ones; a caption there "
        "with no resized image uses its image_dataset/ original)",
    )
    out: str = _f("post_image_dataset/moss", "cache folder (two files per image, flat)")
    pairs: str = _f(
        "",
        "i2i pair list instead of src: an i2i_tasks staging tree's pairs.jsonl "
        "(rows stem / target / cond / caption, paths relative to the file). Each "
        "target is cached as usual and its cond as {stem}.moss_ref",
        advanced=True,
    )
    ref_scale: int = _f(
        1,
        "with pairs: the reference latent's sides are the target's divided by "
        "this, rounded to multiples of 16 (1 = full size)",
        advanced=True,
    )
    resolution: int = _f(
        1024,
        "pixel-area cap as a square edge: 1024 means at most ~1024^2 pixels at "
        "the image's own aspect, both sides multiples of 16; never upscaled",
    )
    caption_format: str = _f(
        "layout_json",
        "what the TE encodes: the Anima caption mapped onto MOSS's layout-JSON "
        "prompt (anime_tools.captions.layout_json; needs the danbooru tag CSV), or "
        "the .txt verbatim, or fixed = fixed_caption for every image (an i2i task "
        "whose edit the reference carries; generate with that caption). "
        "layout_json leaks the set's attributes onto other characters less than "
        "plain tags; generate with the same form",
        choices=("text", "layout_json", "fixed"),
    )
    fixed_caption: str = _f(
        "",
        "with caption_format fixed: the text every image encodes, verbatim "
        "('' = the empty prompt, which is also CFG's uncond)",
        advanced=True,
    )
    overwrite: bool = _f(False, "re-encode even when the cache is current")
    skip_text: bool = _f(False, "skip the text-encoder pass")
    skip_latents: bool = _f(False, "skip the VAE pass")
    save_crops: bool = _f(
        False, "also write the exact pixels each latent was encoded from", advanced=True
    )
    vae_dtype: str = _f(
        "fp32", "VAE encode dtype", choices=("fp32", "bf16"), advanced=True
    )
    max_text_tokens: int = _f(
        4096, "TE truncation length (the model's own limit)", advanced=True
    )
    text_encoder: str = _f(
        DEFAULT_TEXT_ENCODER,
        f"Qwen3.5-9B weights for the text pass: {_TE_HELP}. Switching re-encodes the TE cache",
        choices=TEXT_ENCODERS,
        advanced=True,
    )
    model_dir: str = _f("", _MODEL_DIR_HELP, advanced=True)


@dataclass
class ColorizeCacheRequest(Request):
    """Build the colorize cache: the B&W page is the VAE reference (and, with
    ``vision``, enters the TE through the vision tower); the colour original is the
    target."""

    SCRIPT: ClassVar[str] = "scripts/moss/colorize_cache.py"

    out: str = _f(
        "post_image_dataset/moss/i2i_colorize",
        "cache folder (flat; held-out pairs in <out>/holdout/ with holdout.json)",
    )
    n: int = _f(1667, "illustration pairs to train on (staging page + resized target)")
    manga: int = _f(1667, "manga_color pages to add (0 = illustrations only)")
    holdout: int = _f(8, "illustration pairs held out for the eval")
    manga_holdout: int = _f(
        4, "manga works held out, one page each; no edition of them trains"
    )
    resolution: int = _f(
        1024,
        "pixel-area cap as a square edge (native aspect, multiples of 16, never "
        "upscaled)",
    )
    ref_scale: int = _f(
        1,
        "the reference latent's sides are the target's divided by this, rounded to "
        "multiples of 16 (generate with the same value); v2 used 2",
    )
    variants: int = _f(
        3,
        "sparse captions + 1: embeds_v1 … embeds_v{variants-1} hold the OCR layout "
        "over the colour caption, odd k with tag_dropout of its tags dropped, even "
        "k with all of them (a comic page keeps one comic tag); at least 3, odd "
        "for an even split",
    )
    tag_dropout: float = _f(
        0.8, "share of the colour caption's tags an odd-k sparse caption drops"
    )
    vision: bool = _f(
        False,
        "the page enters the TE through the vision tower ahead of every caption; "
        "off = the captions alone, no embeds_plain",
    )
    seed: int = _f(0, "RNG seed of the pair draw, holdout and sparse captions")
    holdout_json: str = _f(
        "",
        "hold out the pages of this holdout.json (its stem / condition / target) "
        "instead of the drawn ones; their stems and manga works never train",
        advanced=True,
    )
    train_json: str = _f(
        "",
        "train on exactly the stems of this JSON (a list, or an object's keys: "
        "another cache's ocr.json) instead of drawing n + manga; holdout stems "
        "are left out",
        advanced=True,
    )
    staging: str = _f(
        "post_image_dataset/i2i/colorize/staging",
        "the mangafied pages (make i2i-stage TASK=colorize); each pairs with the same "
        "path under resized",
        advanced=True,
    )
    resized: str = _f(
        "post_image_dataset/resized",
        "colour targets with their .txt captions",
        advanced=True,
    )
    manga_dir: str = _f(
        "post_image_dataset/i2i/colorize/manga_color",
        "manga pages as {staging,resized}/<lang>/{anchor}_{lang}_{page}.png",
        advanced=True,
    )
    text_encoder: str = _f(
        "int8",
        f"Qwen3.5-9B weights for every encode, with the vision tower in bf16: {_TE_HELP}. "
        "v3 is int8 (v2 moss_colorize_3k_ocr was fp8); generate with the same",
        choices=TEXT_ENCODERS,
        advanced=True,
    )
    te_file: str = _f(
        "",
        "quantized TE file instead of the catalog's (fp8 / int8 only)",
        advanced=True,
    )
    model_dir: str = _f("", _MODEL_DIR_HELP, advanced=True)


@dataclass
class GenerateRequest(Request):
    """Render prompts with MOSS-Image (Base or SFT), optionally with a LoRA."""

    SCRIPT: ClassVar[str] = "scripts/moss/generate.py"

    prompt: str = _f(
        "An anime illustration of a girl with long silver hair and red eyes, "
        "wearing a black school uniform, sitting on a classroom windowsill at "
        "sunset, cherry blossom petals drifting outside the window, detailed cel "
        "shading, soft rim light.",
        "prompt (ignored when prompts_file or example is set)",
        multiline=True,
    )
    negative_prompt: str = _f("", "negative prompt (the pipeline default is empty)")
    prompt_format: str = _f(
        "layout_json",
        "layout_json = an Anima-format prompt mapped onto MOSS's layout JSON, as "
        "the cache's caption_format does; text = as typed, for a prompt already "
        "in JSON or a LoRA cached as text (not applied to example)",
        choices=("text", "layout_json"),
    )
    model: str = _f("base", "which DiT", choices=VARIANTS)
    lora: str = _f(
        "",
        "LoRA from scripts/moss/train.py (empty = none); a Base LoRA also loads on SFT",
    )
    lora_multiplier: float = _f(1.0, "LoRA strength; 0.0 is the base model exactly")
    width: int = _f(1024, "multiple of 16")
    height: int = _f(1024, "multiple of 16")
    steps: int = _f(
        DEFAULT_STEPS, "denoising steps (the snapped grid can return fewer)"
    )
    cfg: float = _f(DEFAULT_CFG, "CFG scale; 1.0 = no uncond forward")
    sampler: str = _f(
        "er_sde",
        "er_sde = ER-SDE-Solver-3 on the same σ grid, noise seeded by the prompt's "
        "seed (sharper line work, no extra forwards); euler = the pipeline's ODE",
        choices=("euler", "er_sde"),
    )
    uncond_skip: str = _f(
        ",".join(map(str, DEFAULT_UNCOND_SKIP)),
        "comma-separated DiT blocks the uncond branch skips ('' = plain CFG); "
        "the pipeline default replaces the negative with a block-22-skipped "
        "prediction, still two forwards per step",
    )
    seed: int = _f(
        42, "seed of the first prompt (+1 per prompt); fp32 noise from a CPU generator"
    )
    out_dir: str = _f("output/tests", "images + manifest.json land here")
    ref_image: str = _f(
        "",
        "i2i reference: an image, or a folder of images (one job per image and "
        "prompt). The output keeps its aspect, area capped at width x height; the "
        "LoRA's stamped ref_timestep / ref_isolate apply",
    )
    ref_scale: int = _f(
        1,
        "with ref_image: the reference latent's sides are the output's divided by "
        "this, multiples of 16. Match the cache the LoRA trained on (sanitize 1, "
        "colorize 2)",
    )
    ref_vision: bool = _f(
        False,
        "with ref_image: the reference also enters the TE through its vision tower, "
        "ahead of the prompt, as a colorize cache encodes the page. An empty prompt "
        "is then the page alone",
    )
    prompts_file: str = _f("", "one prompt per line; overrides prompt", advanced=True)
    example: str = _f(
        "",
        "comma-separated Space gallery parameters.json files to replay; each sets "
        "its own prompt / negative / size / cfg / seed / uncond skip",
        advanced=True,
    )
    batched: bool = _f(
        False,
        "cond + uncond in one masked forward instead of two: ~25%% slower at "
        "1024² (the key mask takes SDPA off flash) and matches the reference only "
        "within bf16 noise",
        advanced=True,
    )
    shift: float = _f(
        DEFAULT_SHIFT, "scheduler shift (pipeline default)", advanced=True
    )
    decode_only: bool = _f(
        False,
        "decode every latent already in <out_dir>/latents/ to PNG and exit (no "
        "text encoder, no DiT)",
        advanced=True,
    )
    compile: bool = _f(
        True,
        "torch.compile each block with dynamic shapes, after the LoRA is applied: "
        "1.37x at 1024² with or without Spectrum, a few seconds of compile once per "
        "job. --no-compile for eager",
        advanced=True,
    )
    stream: str = _f(
        "bf16",
        "residual stream precision: fp32 runs the stream, modulation, norms, "
        "embedders and head in fp32 around the bf16 block GEMMs (+1.5%% time, "
        "+0.6 GB; renders move a few /255 in detail, neither consistently better)",
        choices=("bf16", "fp32"),
        advanced=True,
    )
    blocks_to_swap: int = _f(
        0,
        "DiT blocks held on CPU and streamed in per forward (0: all resident); "
        "sets expandable_segments. An fp32 stream with a full-size i2i reference "
        "needs 10 (peak 13.5 GiB, +10%% time)",
        advanced=True,
    )
    spectrum: bool = _f(
        True,
        "Spectrum: forecast the last block's image tokens on most steps instead of "
        "running the 42 blocks (library/moss/spectrum.py); --no-spectrum runs every "
        "step in full",
    )
    spectrum_window_size: float = _f(
        2.0, "initial window N: a full forward every floor(N) steps", advanced=True
    )
    spectrum_flex_window: float = _f(
        0.5,
        "growth of N after each full forward: 0.25 / 0.5 / 1.0 = 22 / 19 / 16 full "
        "forwards of 50 steps",
        advanced=True,
    )
    spectrum_warmup: int = _f(6, "leading steps that always run full", advanced=True)
    spectrum_stop_caching_step: int = _f(
        -1, "full forwards from this step on (-1 = steps - 3)", advanced=True
    )
    spectrum_w: float = _f(
        0.3, "Chebyshev / Taylor blend (1.0 = pure Chebyshev)", advanced=True
    )
    spectrum_m: int = _f(3, "Chebyshev degree", advanced=True)
    spectrum_lam: float = _f(0.1, "ridge regularisation", advanced=True)
    dave: str = _f(
        "",
        "DAVE same-prompt diversity: attenuate the image tokens' per-channel mean on "
        "a block pool during the first steps. '' = off, 'auto' = the shipped MOSS "
        "mask, 'lo-hi' = blocks lo..hi, or a dave_alpha.npz path (library/moss/dave.py)",
    )
    dave_strength: float = _f(
        0.3, "fraction of the mean removed on a pooled block (0 = off, 1 = all)"
    )
    dave_tau: float = _f(
        0.10, "attenuate the first tau of the steps (at least one)", advanced=True
    )
    cns: str = _f(
        "",
        "er_sde only: CNS recolors the sampler's injected noise "
        "(docs/cns.md). 'auto' = the shipped γ for the variant at the "
        "calibrated cfg nearest the job's (3 / 4 / 5), else an npz path; '' = off. "
        "ComfyUI's er_sde_cns on MOSS is 'auto' at strength 1",
    )
    cns_strength: float = _f(1.0, "scale on the CNS γ", advanced=True)
    text_encoder: str = _f(
        DEFAULT_TEXT_ENCODER,
        f"Qwen3.5-9B weights for the prompts: {_TE_HELP}",
        choices=TEXT_ENCODERS,
        advanced=True,
    )
    model_dir: str = _f("", _MODEL_DIR_HELP, advanced=True)


@dataclass
class TrainRequest(Request):
    """Train a LoRA on a precached folder with flow matching."""

    SCRIPT: ClassVar[str] = "scripts/moss/train.py"
    # The block's six bias-free linears, as a fullmatch over qualified names.
    TARGETS: ClassVar[str] = r"layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)"

    cache: str = _f(
        "post_image_dataset/moss", "cache folder written by scripts/moss/cache.py"
    )
    output: str = _f(
        "output/ckpt_moss/moss_lora.safetensors",
        "LoRA output path (a .json run report lands beside it)",
    )
    model: str = _f("base", "which DiT to train on", choices=VARIANTS)
    epochs: int = _f(8, "passes over the cache")
    max_steps: int = _f(0, "stop after this many steps (0 = epochs decide)")
    algo: str = _f(
        "lora",
        "adapter: lora, or LyCORIS lokr (Kronecker product) / loha (Hadamard "
        "product of two rank-r pairs)",
        choices=("lora", "lokr", "loha"),
    )
    rank: int = _f(32, "LoRA rank (lokr: rank of the larger Kronecker factor)")
    alpha: float | None = _f(
        None,
        f"LoRA alpha; scale = alpha / rank. Default {LORA_ALPHA:g} for lora (scale 4 "
        "at rank 32), = rank for lokr / loha. lokr ignores it "
        "when the rank reaches half the larger factor's side and both factors are full",
    )
    mod_rank: int = _f(
        0,
        "> 0: also adapt shared_block_modulation (the timestep's shift / scale / gate, "
        "one Linear shared by all 42 blocks) at this rank, Anima's adaln_rank. 0 = off",
        advanced=True,
    )
    mod_alpha: float | None = _f(
        None,
        "alpha of the mod_rank adapter. Default alpha · sqrt(mod_rank / rank), "
        "Anima's adaln √r law (90.5 at rank 32 / alpha 128 / mod_rank 16)",
        advanced=True,
    )
    factor: int = _f(
        -1,
        "lokr only: cap on the small Kronecker factor's side (-1 = near-square "
        "split; LyCORIS's usual 4 / 8 / 16 give a larger full-rank w2)",
    )
    down_init: str = _f(
        "kaiming",
        "lora only, down-projection init: kaiming or weight_svd (the base weight's "
        "top-r right singular vectors / sqrt 3, Anima's SVD-Down). Kaiming is the "
        "default: the SVD basis feeds 4-6x more input DC into the adapter, and under "
        "muon the output DC shift is 0.43 with kaiming vs 1.24 "
        "(docs/moss/findings/muon_channel_init.md). weight_svd_nodc: the SVD basis "
        "of W with the calibrated input DC direction (channel_mean) projected out, "
        "so down · x̄ = 0 at init (docs/moss/proposal/lora_dcfree_svd_init.md)",
        choices=("kaiming", "weight_svd", "weight_svd_nodc"),
    )
    svd_slice: int = _f(
        0,
        "lora + down_init weight_svd[_nodc] only: seed down from the base weight's right singular "
        "vectors [k·r, (k+1)·r) instead of the top-r. Slices of one basis are "
        "mutually orthogonal, so LoRAs trained on different slices own disjoint "
        "input subspaces (Anima's svd_slice, docs/anima/methods/svd-down-lora.md). "
        "Stamped as ss_svd_slice. 0 = the top-r",
        advanced=True,
    )
    svd_slice_count: int = _f(
        0,
        "> 0 interleaves the slices: slice k takes singular vectors {k, k+N, "
        "k+2N, …}, so every slice gets one of the top-N directions instead of "
        "slice 0 holding the whole top of the spectrum. Needs svd_slice < N. "
        "0 = contiguous",
        advanced=True,
    )
    svd_drop_loud: int = _f(
        0,
        "lora + down_init weight_svd[_nodc] only: also project the k input channels "
        "with the largest calibrated mean|x| (channel_stats) out of the SVD basis, "
        "so their down columns start at zero. On MOSS k=1 is ch 42 into "
        "ffn_up_gate, a quarter of the nodc init's channel leverage there "
        "(docs/moss/proposal/lora_dcfree_svd_init.md). Stamped as svd_drop_loud",
        advanced=True,
    )
    channel_stats: str = _f(
        "library/moss/calibration/channel_stats.safetensors",
        "per-target mean|x| calibration for svd_drop_loud (from "
        "scripts/calibration/moss_channel_stats.py on Base)",
        advanced=True,
    )
    channel_mean: str = _f(
        "library/moss/calibration/channel_mean.safetensors",
        "per-target signed token-mean x̄ for down_init weight_svd_nodc (from "
        "scripts/calibration/moss_channel_stats.py on Base)",
        advanced=True,
    )
    lr: float = _f(
        5e-5,
        "learning rate (5e-5 for rank 32 / alpha 128 under muon)",
    )
    optimizer: str = _f(
        "muon",
        "muon: Newton–Schulz-orthogonalised updates on the adapter matrices "
        "(Moonlight lr scaling 0.2·sqrt(max(m, n)), so AdamW's lr carries over; "
        "MOSS was pretrained with Muon–AdamW; needs 2D adapter weights, so "
        "lokr / loha take adamw). adamw: the earlier default",
        choices=("adamw", "muon"),
        advanced=True,
    )
    warmup_ratio: float = _f(0.05, "linear warmup, as a fraction of all steps")
    grad_accum: int = _f(
        1,
        "samples per optimizer update (gradient accumulation; samples keep their "
        "own shapes, so this is the only way past batch 1). Steps, epochs and the "
        "loss log still count samples; warmup counts updates",
    )
    save_every_epochs: int = _f(
        0,
        "also save every N epochs into <output dir>/<name>/<name>_eN (0 = final only)",
    )
    caption_dropout: float = _f(
        0.0,
        "probability of training a step on the empty-prompt embedding (CFG's "
        "uncond), read from library/moss/asset/empty_prompt.safetensors",
    )
    plain_caption_rate: float = _f(
        0.2,
        "probability of training a step on the plain Anima caption instead of its "
        "layout JSON; only a caption_format=layout_json cache carries the plain one",
    )
    caption_variant_rate: float = _f(
        0.0,
        "probability of training a step on one of the TE file's tag-dropped "
        "caption variants (embeds_v1…, drawn uniformly) instead of its full "
        "caption. Drawn only when the plain caption was not, so on a sample with "
        "a plain caption the share of variant steps is (1 - plain_caption_rate) "
        "x this. Only a cache built with variants carries them",
        advanced=True,
    )
    delta_rows: str = _f(
        "all",
        "which joint-sequence rows the adapter delta applies to: all; text (the "
        "whole TE context); caption (the context minus the 3 prefix and 7 suffix "
        "template rows); trigger (the rows whose tokens overlap --trigger in the "
        "caption the step trains on, found with the tokenizer at start; an "
        "empty prompt has none). Anything but all is an attention-path "
        "constraint: the image is then reached only through its reads of those "
        "text rows (project/moss_te_sae). Stamped as delta_rows",
        choices=("all", "text", "caption", "trigger"),
        advanced=True,
    )
    trigger: str = _f(
        "",
        "the trigger string, as written in the captions (e.g. '@channel "
        "(caststation)'); needed by delta_rows=trigger and stamped with it",
        advanced=True,
    )
    preserve_weight: float = _f(
        0.0,
        "weight of the trigger-preservation term: on the same noisy latent and "
        "σ, the MSE between the LoRA's and the base model's prediction for the "
        "trigger-stripped caption (embeds / embeds_plain in "
        "{stem}.moss_te_neg.safetensors, written by "
        "project/moss_te_sae/src/trigger_neg_cache.py, which also re-encodes the "
        "full caption into embeds_pos / embeds_plain_pos so the pair shares one "
        "encoder; the main term then trains on those). Pulls the delta onto "
        "whatever distinguishes the two captions, i.e. the trigger rows. Costs "
        "one no-grad and one grad forward per step. 0 = off",
        advanced=True,
    )
    ref_timestep: float = _f(
        0.0,
        "timestep (1000·σ) whose block modulation the reference tokens get (a "
        "cache with *.moss_ref files); 0 = clean, as the reference is. Negative = "
        "the image's timestep. Stamped in the LoRA metadata",
        advanced=True,
    )
    ref_isolate: bool = _f(
        False,
        "the reference tokens attend only to themselves; image and text still "
        "attend to them. Stamped in the LoRA metadata",
        advanced=True,
    )
    region_weight: float = _f(
        1.0,
        "i2i: per-token loss multiplier on the edit region (the latent tokens "
        "where the target differs from its full-size reference; region_threshold "
        "and friends). The loss is the mean over all tokens of the multiplied "
        "error, not renormalised; region_weight = rest_weight = 1 is the plain MSE",
        advanced=True,
    )
    rest_weight: float = _f(
        1.0,
        "i2i: per-token loss multiplier outside the edit region",
        advanced=True,
    )
    region_weight_end: float | None = _f(
        None,
        "i2i: region_weight at the last step, reached linearly from region_weight "
        "at step 0. None = region_weight throughout",
        advanced=True,
    )
    region_threshold: float = _f(
        0.3,
        "per-token RMS of target − reference latent above which a token is "
        "edited. On swap_text the copied tokens sit near 0.06 (WebP noise) and "
        "the pasted lettering near 1",
        advanced=True,
    )
    region_min_tokens: int = _f(
        5,
        "edited components (8-connected) smaller than this many tokens are noise "
        "and dropped",
        advanced=True,
    )
    region_dilate: int = _f(
        1,
        "tokens the edit region grows by (3x3 steps), so its seam is in it",
        advanced=True,
    )
    tone_weight: float = _f(
        0.0,
        "i2i: weight of the tone term on the unedited tokens (outside the edit "
        "region grown by tone_dilate): the x̂₀ error's masked mean per tone_pool "
        "cell, projected on tone_dirs, squared. Needs a full-size reference. "
        "0 = off. docs/moss/tone_loss.md",
        advanced=True,
    )
    tone_log: bool = _f(
        False,
        "compute and log the tone term without adding it (to size tone_weight)",
        advanced=True,
    )
    tone_pool: int = _f(
        8,
        "tone cell side in latent tokens (8 = 128 px); 0 = one cell over the "
        "whole unedited area",
        advanced=True,
    )
    tone_dilate: int = _f(
        2,
        "extra tokens the edit region grows by for the tone term, beyond region_dilate",
        advanced=True,
    )
    tone_dirs: str = _f(
        "library/moss/calibration/tone_dirs.safetensors",
        "per-channel latent response to +1 Lab L / a / b "
        "(project/finished/moss_lora/src/tone_dirs.py); the term is then in Lab units "
        "squared. '' = the raw 128 channels",
        advanced=True,
    )
    tone_b_weight: float = _f(
        1.0,
        "scale on the b row of the Lab projection (b error weighs its square); "
        "Lab tone_dirs only",
        advanced=True,
    )
    tone_start: float = _f(
        0.0,
        "fraction of all steps before which the tone term is only logged, not added",
        advanced=True,
    )
    tone_b_start: float = _f(
        0.0,
        "fraction of all steps before which tone_b_weight is 1",
        advanced=True,
    )
    repa_weight: float = _f(
        0.0,
        "relational REPA (Anima's DoG-Gram term, docs/anima/methods/repa.md): weight of "
        "the Gram MSE between block repa_layer's image tokens and the image's "
        "PE-Spatial patch tokens. 0 = off",
        advanced=True,
    )
    repa_layer: int = _f(
        11,
        "block whose output REPA aligns; the gradient reaches the LoRA in blocks <= it",
        advanced=True,
    )
    repa_pe_root: str = _f(
        "post_image_dataset/lora",
        "root of the Anima PE-Spatial sidecars "
        "(<root>/<source dir>/{stem}_anima_pe_spatial.safetensors; "
        "make preprocess-pe-spatial writes them)",
        advanced=True,
    )
    int8: str = _f(
        "stream",
        "the frozen DiT's 252 block linears as int8 GEMMs: stream runs dX on a second "
        "int8 copy held in pinned host memory (4.4 GiB) and loaded block by block, "
        "4.2 GB off the peak; full keeps that copy on the card; fwd keeps the bf16 "
        "weight for dX; off is the bf16 base. int8 refuses blocks_to_swap and loha "
        "(configs/moss/lora_bf16.toml is the bf16 plan). Stamped as int8_base. "
        "docs/moss/optimization/int8.md",
        choices=("off", "fwd", "full", "stream"),
        advanced=True,
    )
    int8_file: str = _f(
        "",
        "dit_gptq_moss.py file with the calibrated forward copies. Empty: on Base the "
        "catalog's moss_dit_int8, fetched on first use; on SFT round to nearest. "
        "rtn = round to nearest. Stamped as int8_base_id",
        advanced=True,
    )
    stream: str = _f(
        "fp32",
        "residual stream precision (GenerateRequest.stream): fp32 runs the stream, "
        "modulation, norms, embedders and head in fp32 around the block GEMMs; the "
        "late blocks' LoRA gradients land 1.5-3x closer to fp32, and the int8 base's "
        "prediction error drops below bf16's. Stamped in the LoRA",
        choices=("bf16", "fp32"),
        advanced=True,
    )
    lora_dtype: str = _f(
        "bf16",
        "dtype the LoRA is saved in; training keeps fp32 masters and runs the "
        "rank GEMMs in bf16",
        choices=("bf16", "fp16", "fp32"),
    )
    targets: str = _f(TARGETS, "fullmatch regex over linear names", advanced=True)
    sigma_mu: float = _f(
        -1.09,
        "noise sampling: Z ~ N(mu, s^2), sigma = 1 - sigmoid(Z) (report App. A.2, "
        "final pretraining interval)",
        advanced=True,
    )
    sigma_s: float = _f(0.95, "noise sampling std of Z", advanced=True)
    sigma_cutoff: float = _f(
        0.95,
        "truncate sigma <= cutoff, then rescale onto the 1/1000 grid (1.0 = off)",
        advanced=True,
    )
    sigma_uniform: float = _f(
        0.01, "mixture weight of discrete-uniform sigma", advanced=True
    )
    max_grad_norm: float = _f(1.0, "gradient clip", advanced=True)
    ema_window: float = _f(
        0.1,
        "EMA of the adapter weights, its span as a fraction of the run's optimizer "
        "updates: decay 1 - 1/(ema_window x updates), so 0.1 is 0.958 at 240 "
        "updates and 0.983 at 600. Starts at the end of warmup (fp32 shadow, ~0.3 GB at r32; "
        "ema_card places it). Saves write the EMA; the final raw weights go to "
        "<output dir>/<name>/<name>_raw. 0 = off (docs/moss/findings/ema_lr1e4.md)",
        advanced=True,
    )
    ema_extra: str = _f(
        "",
        "more EMA spans tracked beside ema_window, comma-separated fractions "
        "(e.g. 0.05,0.3); each final EMA goes to <output dir>/<name>/<name>_ema<span>. "
        "One more shadow and lerp per update each",
        advanced=True,
    )
    ema_card: bool = _f(
        True,
        "keep the EMA shadows on the card (~0.28 GB each at r32) instead of copying "
        "the adapter to the host every update: 0.065 s/step (4.8%%) faster on the int8 "
        "plan. --no-ema_card for the CPU shadow",
        advanced=True,
    )
    grad_checkpointing: bool = _f(
        False,
        "per-block activation checkpointing. Off by default: the activation budget "
        "below is faster (1.45 vs 1.35 s/step on the int8 plan) for ~2.4 GB of peak. "
        "Turn it on with --no-compile, or for that peak",
    )
    activation_memory_budget: float = _f(
        0.2,
        "cap on the compiled blocks' saved-for-backward set (Inductor's min-cut "
        "partitioner recomputes the rest); applies with --compile and "
        "--no-grad_checkpointing, ignored otherwise. On the int8 plan speed is flat "
        "from 0.05 to 0.4 and the peak climbs ~1 GB per 0.1 (0.2: 12.04 GB; 0.5 OOMs "
        "on 16 GB). The bf16 plan runs 0.5 at swap 36. i2i samples carry reference "
        "tokens: the i2i configs run 0.175 at swap 38 with --swap_h2d_only (~9%% "
        "faster than 0.15 at swap 28, where 0.2 OOMs)",
        advanced=True,
    )
    blocks_to_swap: int = _f(
        0,
        "DiT blocks to swap to CPU (-1 = size from free VRAM); the int8 base refuses "
        "anything but 0. The bf16 plan runs 36, the room budget 0.5 needs at rank 32 on "
        "16 GB; swap 28-38 run at the same speed, past ~40 the PCIe time stops hiding "
        "(--swap_h2d_only narrows that: 8%% faster at 40). scripts/moss/tune_swap.py "
        "measures this machine's bf16 pair",
        advanced=True,
    )
    swap_h2d_only: bool = _f(
        True,
        "block swap copies host-to-device only: the CPU keeps a page-locked copy "
        "of every swapped block (~0.21 GB each, up to the whole DiT, 8.8 GB) and "
        "an evicted block is repointed at it instead of copied back. Same numerics "
        "and VRAM as the classic swap; on gen 4 x16 it ties up to swap 38 and is "
        "8%% faster at 40. --no-swap_h2d_only for the classic swap",
        advanced=True,
    )
    fit_report: bool = _f(
        False,
        "train the largest sample (longer caption) at step 1 and report its peak "
        "against free VRAM, with a --blocks_to_swap suggestion; off, epoch 1 is "
        "shuffled like the rest",
        advanced=True,
    )
    compile: bool = _f(
        True,
        "torch.compile each block once: dynamic=False with the joint token axis "
        "marked over the samples' range and FA2 attention, so one graph serves "
        "every caption length with no recompiles. 2.0 vs 3.1 s/step at ~4.5k "
        "joint tokens and 0.7 GB lower peak, for ~30 s of compile. --no-compile "
        "for a quick smoke",
        advanced=True,
    )
    expandable_segments: bool = _f(
        True,
        "set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before CUDA starts "
        "(an existing setting wins): lengths that vary per step, such as "
        "colorize's OCR layouts, fragment the allocator's pool without it. "
        "--no-expandable_segments to skip",
        advanced=True,
    )
    log_every: int = _f(10, "full log line every N steps", advanced=True)
    seed: int = _f(0, "RNG seed", advanced=True)
    model_dir: str = _f("", _MODEL_DIR_HELP, advanced=True)


@dataclass
class TuneSwapRequest(Request):
    """Find the --blocks_to_swap / --activation_memory_budget pair to train with on
    this machine: one arm per swap count at the largest budget that fits, timed
    on the largest sample."""

    SCRIPT: ClassVar[str] = "scripts/moss/tune_swap.py"

    cache: str = _f(
        "post_image_dataset/moss",
        "cache folder; its largest sample (image + text tokens) is the one timed",
    )
    sizes: str = _f(
        "",
        "synthetic WxH:text_tokens instead of the cache (e.g. 1024x1024:300)",
    )
    model: str = _f("base", "which DiT to time", choices=VARIANTS)
    rank: int = _f(32, "LoRA rank (the adapters and AdamW state take VRAM too)")
    swap_grid: str = _f(
        "8,12,16,20,24,28,32,36,40",
        "swap counts to try, ascending (at most num_layers - 2)",
    )
    patience: int = _f(
        2, "stop after this many arms in a row slower than the best so far"
    )
    margin_gb: float = _f(
        1.0,
        "free VRAM an arm must leave at its peak to be recommended (real runs "
        "vary in length and grow the allocator)",
    )
    tolerance: float = _f(
        0.02,
        "arms this close to the fastest count as tied; the least swap among "
        "them is recommended (the curve is flat below the PCIe knee)",
    )
    reference: bool = _f(
        True, "also time the checkpointed plan (--grad_checkpointing, swap 0)"
    )
    swap_h2d_only: bool = _f(
        True, "time the host-to-device-only swap (TrainRequest.swap_h2d_only)"
    )
    budget_step: float = _f(0.05, "budget grid", advanced=True)
    warmup: int = _f(2, "untimed steps per arm (compile lands here)", advanced=True)
    steps: int = _f(5, "timed steps per arm", advanced=True)
    output: str = _f(
        "output/ckpt_moss/tune/tune_swap.json", "JSON report path", advanced=True
    )
    seed: int = _f(0, "RNG seed", advanced=True)
    model_dir: str = _f("", _MODEL_DIR_HELP, advanced=True)


# i2i tasks with a MOSS recipe in ``configs/moss/i2i/<task>/`` (``cache.toml``,
# ``train.toml``, ``generate.toml``): the cache request per task; training and
# generation are the shared ``TrainRequest`` / ``GenerateRequest``.
I2I_CONFIG_DIR = "configs/moss/i2i"
I2I_CACHE: dict[str, type[Request]] = {
    "colorize": ColorizeCacheRequest,
    "sanitize": CacheRequest,
    "swap_text": CacheRequest,
}


def i2i_config(task: str, kind: str) -> Path:
    """``configs/moss/i2i/<task>/<kind>.toml`` under the repo home."""
    return resolve_under_home(f"{I2I_CONFIG_DIR}/{task}/{kind}.toml")
