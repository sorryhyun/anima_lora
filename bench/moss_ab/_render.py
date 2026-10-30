"""The render loop the probes here share: DiT + LoRA + block compile, ER-SDE with
Spectrum, latent cache keyed by its settings, decode to PNG, a labelled grid."""

import dataclasses
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file, save_file

from library.env import resolve_under_home
from library.moss.generate import build_spectrum, cached_embeds, denoise, initial_noise
from library.moss.loader import (
    compile_blocks,
    load_transformer,
    pad_embeds,
    stream_fp32_,
)
from library.moss.lora import load_for_inference
from library.moss.model import decode
from library.moss.requests import GenerateRequest

PROMPT_TE = "post_image_dataset/moss_prompt_te"


def lora_id(path: Path) -> str:
    """Path + mtime: a LoRA retrained under the same name is a new LoRA."""
    path = Path(path).resolve()
    return f"{path}@{path.stat().st_mtime_ns}"


def spectrum_settings(on: bool = True) -> dict | None:
    """The ``GenerateRequest`` Spectrum fields ``build_spectrum`` reads, as
    ``generate.py``'s own latent fingerprint records them."""
    if not on:
        return None
    req = dataclasses.asdict(GenerateRequest(spectrum=True))
    return {k: v for k, v in req.items() if k.startswith("spectrum")}


def settings_tag(**settings) -> str:
    """8-hex digest of whatever decides a latent, for its cache filename."""
    blob = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def uncond_embeds(device, texts=("",), precision="bf16"):
    """The empty prompt (plus ``texts``) through the on-disk prompt TE cache.
    Returns ``(embeds by text, padded uncond, uncond mask)``."""
    embeds = cached_embeds(
        "", ["", *texts], resolve_under_home(PROMPT_TE), precision=precision
    )
    uncond, mask = pad_embeds([embeds[""]], device)
    return embeds, uncond, mask


def load_dit(
    lora, device, variant: str = "base", compile: bool = True, stream: str = "bf16"
):
    """A fresh DiT with ``lora`` attached, blocks compiled after the attach (the
    compiled blocks trace the adapter, so one DiT per LoRA). ``stream`` ``fp32``:
    ``stream_fp32_`` before the compile."""
    tr = load_transformer("", variant, device=device)
    net = load_for_inference(tr, lora)
    if stream == "fp32":
        stream_fp32_(tr)
    if compile:
        compile_blocks(tr.layers)
    return tr, net


def render(
    tr,
    cond,
    uncond,
    uncond_mask,
    sigmas,
    *,
    seed: int,
    height: int,
    width: int,
    cfg: float,
    sampler: str = "er_sde",
    spectrum: bool = True,
    ref=None,
    cns=None,
) -> torch.Tensor:
    """One denoise from ``initial_noise(seed)``; ``cond`` is an unpadded embed."""
    device = uncond.device
    cond, cond_mask = pad_embeds([cond], device)
    return denoise(
        tr,
        initial_noise(seed, height, width).to(device),
        cond,
        cond_mask,
        uncond,
        uncond_mask,
        sigmas,
        cfg=cfg,
        sampler=sampler,
        seed=seed,
        cns=cns,
        spectrum=build_spectrum(GenerateRequest(spectrum=spectrum)),
        ref=None if ref is None else ref.to(device, torch.float32),
    )


def save_latent(z: torch.Tensor, path: Path) -> None:
    save_file({"latent": z.cpu().contiguous()}, str(path))


def decoded(lat_path: Path, vae, device, png: Path | None = None) -> Image.Image:
    """The PNG beside ``lat_path`` (or ``png``), decoding the latent first if missing."""
    png = png or lat_path.with_suffix(".png")
    if not png.exists():
        z = load_file(str(lat_path))["latent"].to(device)
        x = decode(z, vae)[0].float().add(1).mul(127.5).round().clamp(0, 255)
        Image.fromarray(x.permute(1, 2, 0).byte().cpu().numpy()).save(png)
    return Image.open(png).convert("RGB")


def grid(rows, cell_h: int = 640, label_h: int = 30) -> Image.Image:
    """``rows`` = ``[(row title, [(label, image), ...]), ...]``, each cell scaled to
    ``cell_h`` tall in a column as wide as the widest cell."""
    cw = max(
        round(im.width * cell_h / im.height) for _, cells in rows for _, im in cells
    )
    ncol = max(len(cells) for _, cells in rows)
    out = Image.new("RGB", (cw * ncol, (cell_h + label_h) * len(rows)), "white")
    d = ImageDraw.Draw(out)
    for ri, (title, cells) in enumerate(rows):
        y = ri * (cell_h + label_h)
        for ci, (label, im) in enumerate(cells):
            th = im.resize(
                (round(im.width * cell_h / im.height), cell_h), Image.LANCZOS
            )
            out.paste(th, (ci * cw, y + label_h))
            d.text((ci * cw + 6, y + 8), f"{title} {label}", fill="black")
    return out
