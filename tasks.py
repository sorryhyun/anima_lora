#!/usr/bin/env python3
"""Cross-platform task runner — replaces Makefile for Windows compatibility.

Usage:
    python tasks.py <command> [extra args...]

Examples:
    python tasks.py lora
    python tasks.py lora --network_dim 32 --max_train_epochs 64
    python tasks.py test
    python tasks.py test-spectrum
    python tasks.py download-models
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def _preset(default: str = "default") -> str:
    return os.environ.get("PRESET", default)


def latest_output(prefix: str = "", exclude: str | None = None) -> Path:
    """Return the most recently modified .safetensors file in output/ckpt/ matching prefix.

    If `exclude` is given, any filename containing that substring is skipped. Useful
    to disambiguate overlapping prefixes (e.g. anima_postfix vs anima_postfix_exp).
    HydraLoRA multi-head sibling files (`*_moe.safetensors`) and backup files
    (containing `.bak.`) are always excluded.
    """
    outputs = sorted(
        (
            f
            for f in (ROOT / "output" / "ckpt").glob("*.safetensors")
            if f.name.startswith(prefix)
            and not f.name.endswith("_moe.safetensors")
            and ".bak." not in f.name
            and (exclude is None or exclude not in f.name)
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not outputs:
        label = f"'{prefix}*.safetensors'" if prefix else "*.safetensors"
        print(f"No {label} files found in output/ckpt/", file=sys.stderr)
        sys.exit(1)
    return outputs[0]


def latest_lora() -> Path:
    return latest_output()


def latest_hydra() -> Path:
    """Latest HydraLoRA multi-head file (`anima_hydra*_moe.safetensors`)."""
    outputs = sorted(
        (
            f
            for f in (ROOT / "output" / "ckpt").glob("anima_hydra*_moe.safetensors")
            if ".bak." not in f.name
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not outputs:
        print(
            "No 'anima_hydra*_moe.safetensors' files found in output/ckpt/ "
            "(enable the HydraLoRA block in configs/methods/lora.toml and run `make lora`)",
            file=sys.stderr,
        )
        sys.exit(1)
    return outputs[0]


def run(cmd: list[str], **kwargs):
    """Run a subprocess, exit on failure."""
    print(f"  > {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=kwargs.pop("cwd", ROOT), **kwargs)
    if result.returncode != 0:
        sys.exit(result.returncode)


def accelerate_launch(*args: str):
    """Launch training via accelerate with extra CLI args forwarded."""
    run(
        [
            "accelerate",
            "launch",
            "--num_cpu_threads_per_process",
            "3",
            "--mixed_precision",
            "bf16",
            "train.py",
            *args,
        ]
    )


def train(method: str, extra, preset: str | None = None, methods_subdir: str | None = None):
    """Launch training for a given method + preset (PRESET env overrides default).

    `methods_subdir` selects the folder under `configs/` that holds the method
    file (default ``"methods"``; pass ``"gui-methods"`` for the clean per-variant
    files used by the `lora-gui` path).

    ARTIST env var trains an artist-only LoRA — equivalent to passing
    `--artist_filter <name>` (filters dataset to `@<name>`-tagged captions and
    redirects output to `output/ckpt-artist/`).
    """
    args = ["--method", method, "--preset", preset or _preset()]
    if methods_subdir:
        args += ["--methods_subdir", methods_subdir]
    artist = os.environ.get("ARTIST")
    if artist and not any(a == "--artist_filter" for a in extra):
        args += ["--artist_filter", artist]
    accelerate_launch(*args, *extra)


# ── Training ──────────────────────────────────────────────────────────


def cmd_lora(extra):
    train("lora", extra)


def cmd_lora_fast(extra):
    train("lora", extra, preset=_preset("fast_16gb"))


def cmd_lora_low_vram(extra):
    train("lora", extra, preset=_preset("low_vram"))


def cmd_lora_gui(extra):
    """Train from configs/gui-methods/<variant>.toml.

    Variant is taken from GUI_PRESETS env var, falling back to the first
    positional extra arg (``python tasks.py lora-gui tlora ...``), then to
    ``lora`` (plain). Extra args after the variant are forwarded as usual.
    """
    variant = os.environ.get("GUI_PRESETS")
    if not variant and extra and not extra[0].startswith("-"):
        variant = extra[0]
        extra = extra[1:]
    variant = variant or "lora"

    expected = ROOT / "configs" / "gui-methods" / f"{variant}.toml"
    if not expected.exists():
        available = sorted(
            p.stem for p in (ROOT / "configs" / "gui-methods").glob("*.toml")
        )
        print(
            f"Unknown gui-methods variant: {variant!r}\n"
            f"Available: {', '.join(available)}",
            file=sys.stderr,
        )
        sys.exit(1)

    train(variant, extra, methods_subdir="gui-methods")


def cmd_apex(extra):
    train("apex", extra)


def cmd_postfix(extra):
    train("postfix", extra)


def cmd_ip_adapter(extra):
    train("ip_adapter", extra)


def cmd_ip_adapter_cache(extra):
    """Pre-cache PE-Core patch features for IP-Adapter training.

    Writes ``{stem}_anima_pe.safetensors`` files into
    ``post_image_dataset/ip-adapter/``. IP_ENCODER env var overrides the
    registry name (default ``pe``).
    """
    encoder = os.environ.get("IP_ENCODER", "pe")
    run(
        [
            PY,
            "preprocess/cache_pe_encoder.py",
            "--dir",
            "ip-adapter-dataset",
            "--cache_dir",
            "post_image_dataset/ip-adapter",
            "--encoder",
            encoder,
            *extra,
        ]
    )


def cmd_ip_adapter_preprocess(extra):
    """Full IP-Adapter preprocess: VAE latents + text-encoder outputs + PE features.

    Source: ``ip-adapter-dataset/``  Caches: ``post_image_dataset/ip-adapter/``.
    """
    encoder = os.environ.get("IP_ENCODER", "pe")
    src = "ip-adapter-dataset"
    dst = "post_image_dataset/ip-adapter"
    run(
        [
            PY,
            "preprocess/cache_latents.py",
            "--dir", src,
            "--cache_dir", dst,
            "--vae", "models/vae/qwen_image_vae.safetensors",
            "--batch_size", "4",
            "--chunk_size", "64",
        ]
    )
    run(
        [
            PY,
            "preprocess/cache_text_embeddings.py",
            "--dir", src,
            "--cache_dir", dst,
            "--qwen3", "models/text_encoders/qwen_3_06b_base.safetensors",
            "--dit", "models/diffusion_models/anima-preview3-base.safetensors",
            "--caption_shuffle_variants", "4",
        ]
    )
    run(
        [
            PY,
            "preprocess/cache_pe_encoder.py",
            "--dir", src,
            "--cache_dir", dst,
            "--encoder", encoder,
        ]
    )


def cmd_easycontrol_preprocess(extra):
    """Full EasyControl preprocess: VAE latents + text-encoder outputs.

    Source: ``easycontrol-dataset/``  Caches: ``post_image_dataset/easycontrol/``.
    """
    src = "easycontrol-dataset"
    dst = "post_image_dataset/easycontrol"
    run(
        [
            PY,
            "preprocess/cache_latents.py",
            "--dir", src,
            "--cache_dir", dst,
            "--vae", "models/vae/qwen_image_vae.safetensors",
            "--batch_size", "4",
            "--chunk_size", "64",
        ]
    )
    run(
        [
            PY,
            "preprocess/cache_text_embeddings.py",
            "--dir", src,
            "--cache_dir", dst,
            "--qwen3", "models/text_encoders/qwen_3_06b_base.safetensors",
            "--dit", "models/diffusion_models/anima-preview3-base.safetensors",
            "--caption_shuffle_variants", "4",
        ]
    )


def cmd_test_ip(extra):
    """Inference with latest IP-Adapter weight. First positional arg is the
    reference image path; everything else is forwarded to inference.py and
    overrides the defaults in INFERENCE_BASE (argparse takes the last value).

    Examples:
      python tasks.py test-ip ref.png --prompt "a girl in a coffee shop"
      python tasks.py test-ip ref.png --prompt "..." --ip_scale 0.8 --seed 7
    """
    if not extra:
        print(
            "Usage: python tasks.py test-ip <ref_image> [extra inference args...]",
            file=sys.stderr,
        )
        sys.exit(1)
    ref_image = extra[0]
    rest = list(extra[1:])
    run(
        [
            *INFERENCE_BASE,
            "--ip_adapter_weight",
            str(latest_output("anima_ip_adapter")),
            "--ip_image",
            ref_image,
            *rest,
        ]
    )


def cmd_easycontrol(extra):
    train("easycontrol", extra)


def cmd_test_easycontrol(extra):
    """Inference with latest EasyControl weight. First positional arg is the
    reference image path; everything else is forwarded to inference.py.

    Examples:
      python tasks.py test-easycontrol ref.png --prompt "a girl in a coffee shop"
      python tasks.py test-easycontrol ref.png --easycontrol_scale 0.8
    """
    if not extra:
        print(
            "Usage: python tasks.py test-easycontrol <ref_image> [extra inference args...]",
            file=sys.stderr,
        )
        sys.exit(1)
    ref_image = extra[0]
    rest = list(extra[1:])
    run(
        [
            *INFERENCE_BASE,
            "--easycontrol_weight",
            str(latest_output("anima_easycontrol")),
            "--easycontrol_image",
            ref_image,
            "--easycontrol_image_match_size",
            *rest,
        ]
    )


# ── img2emb ───────────────────────────────────────────────────────────
#
# TIPSv2-L/14 vision encoder only. Aspect-preserving bucketed preprocessing
# is always on — images are resized to the closest patch-14 bucket
# (~1024 tokens, aspects 1:2..2:1); tokens zero-padded to a single T_MAX so
# the cache stays a flat (N, T_MAX, D) tensor. See
# ``scripts/img2emb/{preprocess,pretrain,finetune,infer}.py``.


def cmd_img2emb(extra):
    """Usage: python tasks.py img2emb [preprocess|anchors|pretrain|finetune] [extra...]

    Bare `img2emb` runs pretrain + finetune (preprocess + anchors are a separate
    preprocessing step — `make preprocess-img2emb` / `python tasks.py preprocess-img2emb`).
    """
    stage = extra[0] if extra else None
    rest = extra[1:] if extra else []
    if stage is None:
        run([PY, "scripts/img2emb/pretrain.py"])
        run([PY, "scripts/img2emb/finetune.py"])
        return
    if stage == "anchors":
        run([PY, "scripts/img2emb/rebuild_anchor_artifacts.py", *rest])
        return
    if stage not in ("preprocess", "pretrain", "finetune"):
        print(
            f"Unknown img2emb stage '{stage}' "
            f"(use: preprocess | anchors | pretrain | finetune)",
            file=sys.stderr,
        )
        sys.exit(1)
    run([PY, f"scripts/img2emb/{stage}.py", *rest])


def cmd_img2emb_preprocess(extra):
    run([PY, "scripts/img2emb/preprocess.py", *extra])


def cmd_img2emb_anchors(extra):
    run([PY, "scripts/img2emb/rebuild_anchor_artifacts.py", *extra])


def cmd_img2emb_pretrain(extra):
    run([PY, "scripts/img2emb/pretrain.py", *extra])


def cmd_img2emb_finetune(extra):
    run([PY, "scripts/img2emb/finetune.py", *extra])


def cmd_preprocess_img2emb(extra):
    run([PY, "scripts/img2emb/preprocess.py", *extra])
    run([PY, "scripts/img2emb/rebuild_anchor_artifacts.py"])


def cmd_test_img2emb(extra):
    """Usage: python tasks.py test-img2emb <ref_image> [extra...]"""
    if not extra:
        print("Usage: python tasks.py test-img2emb <ref_image> [extra...]", file=sys.stderr)
        sys.exit(1)
    ref, *rest = extra
    run([PY, "scripts/img2emb/infer.py", "--ref_image", ref, *rest])


# ── Inference ─────────────────────────────────────────────────────────

INFERENCE_BASE = [
    PY,
    "inference.py",
    "--dit",
    "models/diffusion_models/anima-preview3-base.safetensors",
    "--text_encoder",
    "models/text_encoders/qwen_3_06b_base.safetensors",
    "--vae",
    "models/vae/qwen_image_vae.safetensors",
    "--vae_chunk_size",
    "64",
    "--vae_disable_cache",
    "--attn_mode",
    "flash",  # flash4 not supported yet (flash-attention-sm120 disabled)
    "--lora_multiplier",
    "1.0",
    "--prompt",
    "masterpiece, best quality, score_7, safe. An anime girl wearing a black tank-top"
    " and denim shorts is standing outdoors. She's holding a rectangular sign out in"
    ' front of her that reads "ANIMA". She\'s looking at the viewer with a smile. The'
    " background features some trees and blue sky with clouds.",
    "--negative_prompt",
    "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia",
    "--image_size",
    "1024",
    "1024",
    "--infer_steps",
    "30",
    "--flow_shift",
    "1.0",
    "--sampler",
    "er_sde",
    "--guidance_scale",
    "4.0",
    "--seed",
    "42",
    "--save_path",
    "output/tests",
]


def cmd_test(extra):
    run([*INFERENCE_BASE, "--lora_weight", str(latest_lora()), *extra])


def cmd_test_apex(extra):
    run(
        [
            *INFERENCE_BASE,
            "--lora_weight",
            str(latest_output("anima_apex")),
            "--infer_steps",
            "4",
            "--guidance_scale",
            "1.0",
            "--sampler",
            "euler",
            *extra,
        ]
    )


def cmd_test_hydra(extra):
    # Uses the moe sibling (router-live); static-merge is auto-skipped in
    # library/inference_pipeline.py:_is_hydra_moe detection.
    run([*INFERENCE_BASE, "--lora_weight", str(latest_hydra()), *extra])


def cmd_test_prefix(extra):
    run(
        [*INFERENCE_BASE, "--prefix_weight", str(latest_output("anima_prefix")), *extra]
    )


def cmd_test_ref(extra):
    # Reference-inversion prefixes ride the same loader as prefix-mode tuning;
    # the prefix loader at inference hard-prepends the K slots to crossattn_emb
    # (matches exactly how invert_reference.py assembled them at training time).
    run(
        [*INFERENCE_BASE, "--prefix_weight", str(latest_output("anima_ref")), *extra]
    )


def cmd_test_postfix(extra):
    # exclude both _exp and _func so the vanilla postfix target doesn't grab them
    outputs = sorted(
        (
            f
            for f in (ROOT / "output" / "ckpt").glob("anima_postfix*.safetensors")
            if "_exp" not in f.name and "_func" not in f.name
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not outputs:
        print("No 'anima_postfix*.safetensors' files found in output/ckpt/", file=sys.stderr)
        sys.exit(1)
    run(
        [
            *INFERENCE_BASE,
            "--postfix_weight",
            str(outputs[0]),
            *extra,
        ]
    )


def cmd_test_postfix_exp(extra):
    run(
        [
            *INFERENCE_BASE,
            "--postfix_weight",
            str(latest_output("anima_postfix_exp")),
            *extra,
        ]
    )


def cmd_test_postfix_func(extra):
    run(
        [
            *INFERENCE_BASE,
            "--postfix_weight",
            str(latest_output("anima_postfix_func")),
            *extra,
        ]
    )


def cmd_test_merge(extra):
    """Inference with a baked (merged) DiT from MODEL_DIR (default 'output_temp').

    MODEL_DIR accepts either a directory (picks the latest
    ``*_merged.safetensors`` inside) or a direct ``.safetensors`` path. The
    merged file is a standalone DiT (LoRA folded in), so no ``--lora_weight``
    is passed. The trailing ``--dit`` overrides the base one in
    ``INFERENCE_BASE`` (argparse keeps the last value).
    """
    target = Path(os.environ.get("MODEL_DIR", "output_temp"))
    if not target.is_absolute():
        target = ROOT / target
    if target.is_file():
        chosen = target
    elif target.is_dir():
        candidates = sorted(
            target.glob("*_merged.safetensors"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print(
                f"No '*_merged.safetensors' files found in {target}", file=sys.stderr
            )
            sys.exit(1)
        chosen = candidates[0]
    else:
        print(f"MODEL_DIR path not found: {target}", file=sys.stderr)
        sys.exit(1)
    run([*INFERENCE_BASE, "--dit", str(chosen), *extra])


def cmd_test_spectrum(extra):
    run(
        [
            *INFERENCE_BASE,
            "--lora_weight",
            str(latest_lora()),
            "--spectrum",
            "--spectrum_window_size",
            "2.0",
            "--spectrum_flex_window",
            "0.25",
            "--spectrum_warmup",
            "7",
            "--spectrum_w",
            "0.3",
            "--spectrum_m",
            "3",
            "--spectrum_lam",
            "0.1",
            "--spectrum_stop_caching_step",
            "29",
            "--spectrum_calibration",
            "0.0",
            *extra,
        ]
    )


# ── Utilities ─────────────────────────────────────────────────────────


def cmd_merge(extra):
    """Bake latest LoRA in ADAPTER_DIR (env, default 'output/ckpt') into the base DiT."""
    adapter_dir = os.environ.get("ADAPTER_DIR", "output/ckpt")
    multiplier = os.environ.get("MULTIPLIER", "1.0")
    run(
        [
            PY,
            "scripts/merge_to_dit.py",
            "--adapter_dir",
            adapter_dir,
            "--multiplier",
            multiplier,
            *extra,
        ]
    )


def cmd_preprocess_resize(extra):
    run(
        [
            PY,
            "preprocess/resize_images.py",
            "--src",
            "image_dataset",
            "--dst",
            "post_image_dataset/resized",
            "--no_copy_captions",
            *extra,
        ]
    )


def cmd_preprocess_vae(extra):
    run(
        [
            PY,
            "preprocess/cache_latents.py",
            "--dir",
            "post_image_dataset/resized",
            "--cache_dir",
            "post_image_dataset/lora",
            "--vae",
            "models/vae/qwen_image_vae.safetensors",
            "--batch_size",
            "4",
            "--chunk_size",
            "64",
            *extra,
        ]
    )


def cmd_preprocess_te(extra):
    run(
        [
            PY,
            "preprocess/cache_text_embeddings.py",
            "--dir",
            "image_dataset",
            "--cache_dir",
            "post_image_dataset/lora",
            "--qwen3",
            "models/text_encoders/qwen_3_06b_base.safetensors",
            "--dit",
            "models/diffusion_models/anima-preview3-base.safetensors",
            "--caption_shuffle_variants",
            "8",
            *extra,
        ]
    )


def cmd_preprocess(extra):
    cmd_preprocess_resize(extra)
    cmd_preprocess_vae(extra)
    cmd_preprocess_te(extra)


def cmd_comfy_batch(extra):
    workflow = extra[0] if extra else "workflows/lora-batch.json"
    remaining = extra[1:] if extra else []
    run([PY, "scripts/comfy_batch.py", workflow, *remaining])


# ── Downloads ─────────────────────────────────────────────────────────


def cmd_download_sam3(_extra):
    (ROOT / "models" / "sam3").mkdir(parents=True, exist_ok=True)
    run(["hf", "download", "facebook/sam3", "--local-dir", "models/sam3"])


def cmd_download_tipsv2(_extra):
    # TIPSv2 ships custom code consumed via trust_remote_code; grab the whole
    # repo so local-dir load works offline. See scripts/img2emb/preprocess.py.
    (ROOT / "models" / "tipsv2").mkdir(parents=True, exist_ok=True)
    run(["hf", "download", "google/tipsv2-l14", "--local-dir", "models/tipsv2"])


def cmd_download_pe(_extra):
    # PE-Core-L14-336 — only the .pt checkpoint is needed; vision tower is
    # vendored at library/models/pe.py (no perception_models clone required).
    (ROOT / "models" / "pe").mkdir(parents=True, exist_ok=True)
    run([
        "hf", "download",
        "facebook/PE-Core-L14-336", "PE-Core-L14-336.pt",
        "--local-dir", "models/pe",
    ])


def cmd_download_mit(_extra):
    (ROOT / "models" / "mit").mkdir(parents=True, exist_ok=True)
    run(
        [
            "hf",
            "download",
            "a-b-c-x-y-z/Manga-Text-Segmentation-2025",
            "model.pth",
            "--local-dir",
            "models/mit",
        ]
    )


def cmd_download_anima(_extra):
    for d in ["diffusion_models", "text_encoders", "vae"]:
        (ROOT / "models" / d).mkdir(parents=True, exist_ok=True)
    run(
        [
            "hf",
            "download",
            "circlestone-labs/Anima",
            "split_files/diffusion_models/anima-preview3-base.safetensors",
            "split_files/text_encoders/qwen_3_06b_base.safetensors",
            "split_files/vae/qwen_image_vae.safetensors",
            "--local-dir",
            "models",
            "--include",
            "split_files/*",
        ]
    )
    split = ROOT / "models" / "split_files"
    for subdir in ["diffusion_models", "text_encoders", "vae"]:
        src = split / subdir
        dst = ROOT / "models" / subdir
        if src.exists():
            for f in src.iterdir():
                shutil.move(str(f), str(dst / f.name))
    if split.exists():
        shutil.rmtree(split)


def cmd_download_models(_extra):
    cmd_download_anima(_extra)
    cmd_download_sam3(_extra)
    cmd_download_mit(_extra)
    cmd_download_tipsv2(_extra)


# ── Masking ───────────────────────────────────────────────────────────


def cmd_mask_sam(extra):
    run(
        [
            PY,
            "preprocess/generate_masks.py",
            "--config",
            "configs/sam_mask.yaml",
            "--image-dir",
            "post_image_dataset",
            "--mask-dir",
            "masks/sam",
            "--checkpoint",
            "models/sam3/sam3.pt",
            "--batch-size",
            "2",
            *extra,
        ]
    )


def cmd_mask_mit(extra):
    run(
        [
            PY,
            "preprocess/generate_masks_mit.py",
            "--image-dir",
            "post_image_dataset",
            "--mask-dir",
            "masks/mit",
            "--model-path",
            "models/mit/model.pth",
            *extra,
        ]
    )


def cmd_mask(extra):
    if not (ROOT / "masks" / "sam").is_dir():
        cmd_mask_sam([])
    if not (ROOT / "masks" / "mit").is_dir():
        cmd_mask_mit([])
    run(
        [
            PY,
            "preprocess/merge_masks.py",
            "masks/sam",
            "masks/mit",
            "--output-dir",
            "masks/merged",
            *extra,
        ]
    )


def cmd_mask_clean(_extra):
    p = ROOT / "masks"
    if p.exists():
        shutil.rmtree(p)
        print("  Removed masks/")


def cmd_gui(_extra):
    run([PY, "-m", "gui"])


def cmd_test_unit(extra):
    run([PY, "-m", "pytest", "-q", "tests/", *extra])


def cmd_export_logs(extra):
    """Dump TB scalar logs to JSON. RUN=<dir> (default output/logs), ALL=1, JSONL=1."""
    run_path = os.environ.get("RUN", "output/logs")
    cmd = [PY, "scripts/export_logs_json.py", run_path]
    if os.environ.get("ALL"):
        cmd.append("--all")
    if os.environ.get("JSONL"):
        cmd.append("--jsonl")
    run([*cmd, *extra])


def cmd_print_config(extra):
    method = os.environ.get("METHOD", "lora")
    preset = _preset()
    run(
        [
            PY,
            "train.py",
            "--method",
            method,
            "--preset",
            preset,
            "--print-config",
            "--no-config-snapshot",
            *extra,
        ]
    )


def cmd_invert_ref(extra):
    """Reference-image inversion (scripts/inversion/invert_reference.py).

    Image source:
        - If REF_IMAGE is set, use that file.
        - Otherwise pick a random file from REF_IMAGE_DIR (default post_image_dataset).
          Re-running picks a new random image each time.
    Optional env: REF_TEMPLATE (default "a photo"), REF_K (default 8),
                  REF_STEPS (default 100), REF_LR (default 0.01),
                  REF_NAME (default "latest" -> output/ckpt/anima_ref_latest.safetensors),
                  REF_SAVE_PATH (overrides REF_NAME),
                  REF_SWAP (blocks_to_swap, default 0).
    """
    import glob
    import random

    ref_image = os.environ.get("REF_IMAGE", "").strip()
    if not ref_image:
        ref_dir = os.environ.get("REF_IMAGE_DIR", "post_image_dataset")
        candidates = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            candidates.extend(
                glob.glob(os.path.join(ref_dir, "**", ext), recursive=True)
            )
        if not candidates:
            print(
                f"Error: no images found in REF_IMAGE_DIR={ref_dir}/ and REF_IMAGE not set.\n"
                "       Either set REF_IMAGE=path/to/ref.png or point REF_IMAGE_DIR at a "
                "directory with .png/.jpg/.webp files.",
                file=sys.stderr,
            )
            sys.exit(1)
        ref_image = random.choice(candidates)
        print(f"  > random pick from {ref_dir}/: {ref_image}")
    ref_template = os.environ.get("REF_TEMPLATE", "a photo")
    ref_k = os.environ.get("REF_K", "8")
    ref_steps = os.environ.get("REF_STEPS", "100")
    ref_lr = os.environ.get("REF_LR", "0.01")
    ref_name = os.environ.get("REF_NAME", "latest")
    ref_save_path = os.environ.get(
        "REF_SAVE_PATH", f"output/ckpt/anima_ref_{ref_name}.safetensors"
    )
    ref_swap = os.environ.get("REF_SWAP", "0")
    run(
        [
            PY,
            "scripts/inversion/invert_reference.py",
            "--image",
            ref_image,
            "--dit",
            "models/diffusion_models/anima-preview3-base.safetensors",
            "--vae",
            "models/vae/qwen_image_vae.safetensors",
            "--text_encoder",
            "models/text_encoders/qwen_3_06b_base.safetensors",
            "--attn_mode",
            "flash",
            "--template",
            ref_template,
            "--num_tokens",
            ref_k,
            "--steps",
            ref_steps,
            "--lr",
            ref_lr,
            "--save_path",
            ref_save_path,
            "--blocks_to_swap",
            ref_swap,
            "--verify",
            *extra,
        ]
    )


def cmd_invert(extra):
    run(
        [
            PY,
            "scripts/inversion/invert_embedding.py",
            "--dit",
            "models/diffusion_models/anima-preview3-base.safetensors",
            "--attn_mode",
            "flash",
            "--image_dir",
            "post_image_dataset",
            "--num_images",
            "10",
            "--shuffle",
            "--steps",
            "500",
            "--lr",
            "0.01",
            "--aggregate_by",
            "3",
            "--save_per_run",
            "--probe_functional",
            "--probe_blocks",
            "8,12,16,20",
            "--output_dir",
            "output/inversions",
            "--log_block_grads",
            *extra,
        ]
    )


# ── CLI ───────────────────────────────────────────────────────────────

COMMANDS = {
    "lora": (cmd_lora, "LoRA family (lora|tlora|tlora_rf|hydralora via configs/methods/lora.toml)"),
    "lora-fast": (cmd_lora_fast, "Fast LoRA training (16GB, no block swap)"),
    "lora-low-vram": (cmd_lora_low_vram, "LoRA training (low VRAM)"),
    "lora-gui": (
        cmd_lora_gui,
        "Train from a self-contained configs/gui-methods/<variant>.toml "
        "(variant from GUI_PRESETS env or 1st positional; e.g. tlora, hydralora, reft, postfix_exp).",
    ),
    "apex": (cmd_apex, "APEX distillation (condition-shift self-adversarial)"),
    "postfix": (cmd_postfix, "Postfix/prefix tuning (mode selected in configs/methods/postfix.toml)"),
    "ip-adapter": (cmd_ip_adapter, "IP-Adapter training (decoupled image cross-attention)"),
    "ip-adapter-cache": (
        cmd_ip_adapter_cache,
        "Pre-cache PE-Core patch features for IP-Adapter (writes into "
        "post_image_dataset/ip-adapter/). IP_ENCODER=pe|pe-g.",
    ),
    "ip-adapter-preprocess": (
        cmd_ip_adapter_preprocess,
        "Full IP-Adapter preprocess: latents + text emb + PE features. "
        "Source: ip-adapter-dataset/  Cache: post_image_dataset/ip-adapter/.",
    ),
    "test-ip": (cmd_test_ip, "Inference with latest IP-Adapter weight. Usage: test-ip <ref_image> [--prompt ... --ip_scale ...]"),
    "easycontrol": (cmd_easycontrol, "EasyControl training (extended self-attn KV with VAE-encoded reference)"),
    "easycontrol-preprocess": (
        cmd_easycontrol_preprocess,
        "Full EasyControl preprocess: latents + text emb. "
        "Source: easycontrol-dataset/  Cache: post_image_dataset/easycontrol/.",
    ),
    "test-easycontrol": (cmd_test_easycontrol, "Inference with latest EasyControl weight. Usage: test-easycontrol <ref_image> [--prompt ... --easycontrol_scale ...]"),
    "test": (cmd_test, "Inference with latest LoRA"),
    "test-apex": (cmd_test_apex, "Inference with latest APEX LoRA"),
    "test-hydra": (cmd_test_hydra, "Inference with latest HydraLoRA moe (router-live)"),
    "test-prefix": (cmd_test_prefix, "Inference with latest prefix weight"),
    "test-ref": (
        cmd_test_ref,
        "Inference with latest reference-inversion prefix (output/ckpt/anima_ref*.safetensors)",
    ),
    "test-postfix": (cmd_test_postfix, "Inference with latest postfix weight"),
    "test-postfix-exp": (
        cmd_test_postfix_exp,
        "Inference with latest postfix-exp weight",
    ),
    "test-postfix-func": (
        cmd_test_postfix_func,
        "Inference with latest postfix-func weight",
    ),
    "test-merge": (
        cmd_test_merge,
        "Inference with latest *_merged.safetensors (MODEL_DIR=..., default 'output_temp')",
    ),
    "test-spectrum": (cmd_test_spectrum, "Spectrum-accelerated inference"),
    "merge": (
        cmd_merge,
        "Bake latest LoRA (ADAPTER_DIR=..., default 'output/ckpt') into base DiT",
    ),
    "preprocess": (
        cmd_preprocess,
        "Full preprocessing (resize + VAE + text embeddings)",
    ),
    "preprocess-resize": (cmd_preprocess_resize, "Resize images to bucket resolutions"),
    "preprocess-vae": (cmd_preprocess_vae, "Cache VAE latents"),
    "preprocess-te": (cmd_preprocess_te, "Cache text encoder embeddings"),
    "img2emb": (
        cmd_img2emb,
        "Train img2emb resampler (pretrain + finetune). Optional stage arg: preprocess|anchors|pretrain|finetune",
    ),
    "img2emb-preprocess": (cmd_img2emb_preprocess, "img2emb: extract encoder features (stage 1)"),
    "img2emb-anchors": (cmd_img2emb_anchors, "img2emb: refresh people=* prototypes + phase1_positions (stage 1.5)"),
    "img2emb-pretrain": (cmd_img2emb_pretrain, "img2emb: resampler pretrain on cached targets (stage 2)"),
    "img2emb-finetune": (cmd_img2emb_finetune, "img2emb: flow-matching finetune through frozen DiT (stage 3)"),
    "preprocess-img2emb": (
        cmd_preprocess_img2emb,
        "img2emb preprocessing: extract features + rebuild anchor artifacts",
    ),
    "test-img2emb": (
        cmd_test_img2emb,
        "Generate an image from a ref image via the finetuned resampler. Usage: test-img2emb <ref_image>",
    ),
    "comfy-batch": (cmd_comfy_batch, "Run ComfyUI batch workflow"),
    "download-models": (cmd_download_models, "Download all models"),
    "download-anima": (cmd_download_anima, "Download Anima model"),
    "download-sam3": (cmd_download_sam3, "Download SAM3 model"),
    "download-mit": (cmd_download_mit, "Download MIT model"),
    "download-tipsv2": (cmd_download_tipsv2, "Download TIPSv2-L/14 (img2emb encoder)"),
    "download-pe": (cmd_download_pe, "Download PE-Core-L14-336 (img2emb encoder)"),
    "mask": (cmd_mask, "Generate SAM3 + MIT masks, then merge"),
    "mask-sam": (cmd_mask_sam, "Generate SAM3 masks only"),
    "mask-mit": (cmd_mask_mit, "Generate MIT masks only"),
    "mask-clean": (cmd_mask_clean, "Remove all generated masks"),
    "gui": (cmd_gui, "Launch PySide6 GUI"),
    "invert": (cmd_invert, "Embedding inversion (image → text embedding)"),
    "invert-ref": (
        cmd_invert_ref,
        "Reference inversion (REF_IMAGE=path/to/ref.png; see tasks.py::cmd_invert_ref for env vars)",
    ),
    "test-unit": (cmd_test_unit, "Run smoke/unit tests (pytest tests/)"),
    "export-logs": (
        cmd_export_logs,
        "Dump TB scalar logs to JSON (RUN=<dir>, ALL=1 for every subrun, JSONL=1 for line-delimited)",
    ),
    "print-config": (
        cmd_print_config,
        "Dump merged config (METHOD=<name> PRESET=<name>) with provenance",
    ),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python tasks.py <command> [extra args...]\n")
        print("Commands:")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:20s} {desc}")
        print("\nExtra arguments are forwarded to the underlying command.")
        print("Example: python tasks.py lora --network_dim 32 --max_train_epochs 64")
        sys.exit(0)

    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Run 'python tasks.py --help' for available commands", file=sys.stderr)
        sys.exit(1)

    extra = sys.argv[2:]
    fn, _ = COMMANDS[command]
    fn(extra)


if __name__ == "__main__":
    main()
