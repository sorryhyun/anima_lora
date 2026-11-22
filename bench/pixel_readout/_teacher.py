"""The released ResShift ×2 student: arm A's target and the reference of the ×2 reads."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

TEACHER_REPO = "../comfy/custom_nodes/comfyui-distilled-resshift"
TEACHER_DIR = "../comfy/models/resshift"
TEACHER_FILE = "rsd_student_18k.safetensors"  # Hub sorryhyun/Distilled-ResShift-2x
VQGAN_FILE = "autoencoder_vq_f4.pth"


def load_teacher(repo: Path, weights: Path, device):
    """The released ResShift ×2 student as ``up2(img (3,H,W) in [-1,1], seed) → (3,2H,2W)``.

    Loaded the way the ComfyUI node loads it (student fp32 under bf16 autocast, VQGAN
    fp16) and run through the node's ``upscale`` at its defaults (chop 512, overlap 64,
    tile batch 4, shared noise)."""
    spec = importlib.util.spec_from_file_location(
        "resshift_infer", str(repo / "resshift_infer.py")
    )
    R = importlib.util.module_from_spec(spec)
    sys.modules["resshift_infer"] = R
    spec.loader.exec_module(R)
    cfg = R.load_configs(str(weights / VQGAN_FILE), config_path=R.CONFIGS["x2"])
    diff = R.build_diffusion(cfg)
    path = str(weights / TEACHER_FILE)
    with safe_open(path, "pt") as fh:
        meta = fh.metadata() or {}
    nc = meta.get("noise_channels")
    student = R.build_student(
        cfg,
        device,
        noise_mode=meta.get("noise_mode", "concat"),
        noise_channels=int(nc) if nc is not None else None,
    )
    R.load_student_weights(student, load_file(path))
    vqgan = R.build_autoencoder(cfg, device, dtype=torch.float16)
    align = R.swin_align(cfg)
    cond = meta.get("cond_lq", "latent")
    scale = int(cfg.diffusion.params.sf)
    assert scale == 2, f"teacher scale {scale}"

    @torch.no_grad()
    def up2(img: torch.Tensor, seed: int) -> torch.Tensor:
        return R.upscale(
            cfg,
            diff,
            vqgan,
            student,
            img[None].float().clamp(-1, 1),
            device,
            scale,
            align,
            overlap=64,
            chop=512,
            tile_batch=4,
            amp=True,
            seed=seed,
            shared_noise=True,
            cond_mode=cond,
        )[0].float()

    return up2
