"""Inference entry-points (test / test-* commands).

All variants share ``INFERENCE_BASE`` from ``_common`` and add method-specific
flags. Reference-image variants (test-ip / test-easycontrol) accept REF_IMAGE
env or first positional arg, copy the ref alongside the generated output.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from ._common import (
    INFERENCE_BASE,
    ROOT,
    latest_hydra,
    latest_lora,
    latest_output,
    run,
)


def cmd_test(extra):
    run([*INFERENCE_BASE, "--lora_weight", str(latest_lora()), *extra])


def cmd_test_mod(extra):
    """Inference with the latest distilled pooled_text_proj MLP for modulation guidance."""
    run(
        [
            *INFERENCE_BASE,
            "--pooled_text_proj",
            str(latest_output("pooled_text_proj")),
            *extra,
        ]
    )


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
    run([*INFERENCE_BASE, "--prefix_weight", str(latest_output("anima_ref")), *extra])


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
        print(
            "No 'anima_postfix*.safetensors' files found in output/ckpt/",
            file=sys.stderr,
        )
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
            print(f"No '*_merged.safetensors' files found in {target}", file=sys.stderr)
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


def cmd_test_ip(extra):
    """Inference with latest IP-Adapter weight.

    Reference image is taken from REF_IMAGE env or the first positional arg.
    PROMPT, NEG, IP_SCALE env vars override defaults. Saves to output/tests/ip/
    and copies the ref image alongside the generated output as ``<name>_ref.png``.

    Examples:
      python tasks.py test-ip ref.png --prompt "a girl in a coffee shop"
      REF_IMAGE=ref.png IP_SCALE=0.8 python tasks.py test-ip
    """
    ref_image = os.environ.get("REF_IMAGE", "").strip()
    if not ref_image and extra and not extra[0].startswith("-"):
        ref_image = extra[0]
        extra = extra[1:]
    if not ref_image:
        print(
            "Usage: python tasks.py test-ip <ref_image> [extra...]\n"
            "   or: REF_IMAGE=path/to/ref.png python tasks.py test-ip [extra...]",
            file=sys.stderr,
        )
        sys.exit(1)

    save_dir = ROOT / "output" / "tests" / "ip"
    save_dir.mkdir(parents=True, exist_ok=True)

    args = [
        *INFERENCE_BASE,
        "--save_path",
        str(save_dir),
        "--ip_adapter_weight",
        str(latest_output("anima_ip_adapter")),
        "--ip_image",
        ref_image,
        "--ip_image_match_size",
    ]
    if scale := os.environ.get("IP_SCALE"):
        args += ["--ip_scale", scale]
    if prompt := os.environ.get("PROMPT"):
        args += ["--prompt", prompt]
    if neg := os.environ.get("NEG"):
        args += ["--negative_prompt", neg]
    args += list(extra)
    run(args)

    pngs = sorted(
        (p for p in save_dir.glob("*.png") if not p.name.endswith("_ref.png")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pngs:
        ref_dst = pngs[0].with_name(pngs[0].stem + "_ref.png")
        shutil.copy(ref_image, ref_dst)
        print(f"  > Ref pasted: {ref_dst}")


def cmd_test_easycontrol(extra):
    """Inference with latest EasyControl weight.

    Reference image is taken from REF_IMAGE env or the first positional arg.
    PROMPT, NEG, EC_SCALE env vars override defaults. Saves to
    output/tests/easycontrol/ and copies the ref image alongside the generated
    output as ``<name>_ref.png``.

    Examples:
      python tasks.py test-easycontrol ref.png --prompt "a girl in a coffee shop"
      REF_IMAGE=ref.png EC_SCALE=0.8 python tasks.py test-easycontrol
    """
    ref_image = os.environ.get("REF_IMAGE", "").strip()
    if not ref_image and extra and not extra[0].startswith("-"):
        ref_image = extra[0]
        extra = extra[1:]
    if not ref_image:
        print(
            "Usage: python tasks.py test-easycontrol <ref_image> [extra...]\n"
            "   or: REF_IMAGE=path/to/ref.png python tasks.py test-easycontrol [extra...]",
            file=sys.stderr,
        )
        sys.exit(1)

    save_dir = ROOT / "output" / "tests" / "easycontrol"
    save_dir.mkdir(parents=True, exist_ok=True)

    args = [
        *INFERENCE_BASE,
        "--save_path",
        str(save_dir),
        "--easycontrol_weight",
        str(latest_output("anima_easycontrol")),
        "--easycontrol_image",
        ref_image,
        "--easycontrol_image_match_size",
    ]
    if scale := os.environ.get("EC_SCALE"):
        args += ["--easycontrol_scale", scale]
    if prompt := os.environ.get("PROMPT"):
        args += ["--prompt", prompt]
    if neg := os.environ.get("NEG"):
        args += ["--negative_prompt", neg]
    args += list(extra)
    run(args)

    pngs = sorted(
        (p for p in save_dir.glob("*.png") if not p.name.endswith("_ref.png")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pngs:
        ref_dst = pngs[0].with_name(pngs[0].stem + "_ref.png")
        shutil.copy(ref_image, ref_dst)
        print(f"  > Ref pasted: {ref_dst}")
