"""Model plumbing: checkpoints, generation, VAE, text encoding, the DiT forward.

Torch and the library are imported lazily so CPU-only stages can import this.
"""

from __future__ import annotations

import copy
from pathlib import Path

from .shapes import wh


def checkpoints():
    from library.env import default_checkpoints

    return default_checkpoints()


def gen_args(size, steps: int, cfg: int | float, save: Path, negative_prompt: str = ""):
    """``size``: int side or ``(W, H)``; the request wants (height, width)."""
    from anima_lora.inference import GenerationRequest

    W, H = wh(size)
    ck = checkpoints()
    req = GenerationRequest(
        prompt="",
        negative_prompt=negative_prompt,
        image_size=(H, W),
        infer_steps=steps,
        guidance_scale=cfg,
        seed=0,
        dit=ck.dit,
        vae=ck.vae,
        text_encoder=ck.text_encoder,
        attn_mode="flash",
        save_path=str(save),
    )
    return req.to_args()


def load_generator(size, steps: int, cfg: int | float, save: Path, negative_prompt: str = ""):
    """``(args, gen settings, device, shared models)`` ready for ``generate``;
    the DiT is ``shared['model']``."""
    import torch

    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model, load_shared_models

    args = gen_args(size, steps, cfg, save, negative_prompt)
    gen = get_generation_settings(args)
    device = gen.device
    shared = load_shared_models(args)
    shared["conds_cache"] = {}
    shared["model"] = load_dit_model(args, device, torch.bfloat16)
    return args, gen, device, shared


def generate_to(fn: Path, args, gen, shared, vae, device, prompt: str, seed: int):
    """Render ``prompt`` at ``seed`` to ``fn`` unless it already exists."""
    import torch

    from library.inference.generation import generate

    if fn.exists():
        return
    a2 = copy.deepcopy(args)
    a2.prompt = prompt
    a2.seed = seed
    with torch.no_grad():
        lat = generate(a2, gen, shared)
    decode_image(vae, lat, device).save(fn)


def load_vae(device):
    import torch

    from library.models import qwen_vae

    vae = qwen_vae.load_vae(
        checkpoints().vae,
        device="cpu",
        disable_mmap=True,
        disable_cache=True,
        vae_2d=True,
    )
    return vae.to(device, dtype=torch.bfloat16).eval()


def decode_image(vae, latent, device):
    import torch

    from library.inference.output import pixels_to_pil

    with torch.no_grad():
        px = vae.decode_to_pixels(latent.to(device, dtype=vae.dtype))
    if px.ndim == 5:
        px = px.squeeze(2)
    return pixels_to_pil(px[0].float().cpu())


def encode_images(vae, files, device, size=None):
    """VAE latents (float32, CPU) for image files, 8 per VAE call; ``size``
    ``(W, H)`` resizes first. Pixels go in at the IMAGE_TRANSFORMS range."""
    import numpy as np
    import torch
    from PIL import Image

    out = []
    with torch.no_grad():
        for i in range(0, len(files), 8):
            ims = [Image.open(f).convert("RGB") for f in files[i : i + 8]]
            px = np.stack([np.array(im.resize(size) if size else im) for im in ims])
            px = (
                torch.from_numpy(px)
                .permute(0, 3, 1, 2)
                .float()
                .div(127.5)
                .sub(1.0)
                .to(device)
            )
            out.append(vae.encode_pixels_to_latents(px).float().cpu())
    return torch.cat(out)


class _TextEntry:
    """One cached caption, indexed like the tuple ``(prompt_embeds, attn_mask,
    t5_ids, t5_mask)``; the max-padded embeds are row ``i`` of
    ``encode_captions``' mmapped spill file."""

    __slots__ = ("mm", "i", "rest")

    def __init__(self, mm, i, rest):
        self.mm, self.i, self.rest = mm, i, rest

    def __getitem__(self, k):
        return self.mm[self.i] if k == 0 else self.rest[k - 1]


def encode_captions(captions, device):
    """Unique captions → dict caption -> (prompt_embeds, attn_mask, t5_ids,
    t5_mask) on CPU. The embeds stream from an unlinked temp file, written
    max-padded as encoded (a row is 1 MB; 40 k captions do not fit in RAM —
    ``TMPDIR`` moves the file); the rest stays in memory."""
    import os
    import tempfile

    import torch

    from library.inference.models import load_text_encoder
    from library.inference.text import ensure_text_strategies

    tok, enc = ensure_text_strategies(checkpoints().text_encoder, vocab_pack=None)
    te = load_text_encoder(
        text_encoder=checkpoints().text_encoder, dtype=torch.bfloat16, device=device
    ).eval()
    uniq = sorted(set(captions))
    fd, spill = tempfile.mkstemp(prefix="wake_te_", suffix=".bf16")
    rests, shape = [], None
    with os.fdopen(fd, "wb") as f, torch.no_grad():
        for i in range(0, len(uniq), 16):
            chunk = uniq[i : i + 16]
            tokens = tok.tokenize(chunk)
            pe, am, t5, t5m = enc.encode_tokens(tok, [te], tokens)
            pe = pe.to(torch.bfloat16).cpu().contiguous()
            assert shape in (None, tuple(pe.shape[1:])), (shape, pe.shape)
            shape = tuple(pe.shape[1:])
            f.write(pe.view(torch.int16).numpy().tobytes())
            am, t5, t5m = am.cpu(), t5.long().cpu(), t5m.cpu()
            rests += [(am[j], t5[j], t5m[j]) for j in range(len(chunk))]
    te.to("cpu")
    del te
    torch.cuda.empty_cache()
    if not uniq:
        os.unlink(spill)
        return {}
    # mapped private and unlinked: pages are evictable, the file goes with the process
    n = len(uniq) * shape[0] * shape[1]
    mm = torch.from_file(spill, shared=False, size=n, dtype=torch.bfloat16)
    mm = mm.view(len(uniq), *shape)
    os.unlink(spill)
    return {c: _TextEntry(mm, i, rests[i]) for i, c in enumerate(uniq)}


def ext_ids_of(cache) -> set[int]:
    """Pack ext rows touched by the T5 ids of every cached caption."""
    from library.anima.ext_vocab import T5_TABLE_SIZE

    ids = set()
    for ent in cache.values():
        t5 = ent[2]
        ids.update(int(v) - T5_TABLE_SIZE for v in t5.tolist() if v >= T5_TABLE_SIZE)
    return ids


def dit_forward(anima, noisy, ts, cache, captions, device):
    """One DiT forward on 4D latents ``(B, C, H, W)`` conditioned on cached
    captions (one per batch row) → 4D prediction. The frame axis is dim 2."""
    import torch

    def stacked(k):
        return torch.stack([cache[c][k] for c in captions]).to(device)

    b = len(captions)
    return anima(
        noisy.unsqueeze(2),
        ts,
        stacked(0),
        padding_mask=torch.zeros(
            b, 1, *noisy.shape[-2:], dtype=torch.bfloat16, device=device
        ),
        target_input_ids=stacked(2),
        target_attention_mask=stacked(3),
        source_attention_mask=stacked(1),
    ).squeeze(2)


def load_trained(arm_dir: Path) -> dict:
    """An arm's ``trained.pt`` (our own file: full unpickling)."""
    import torch

    return torch.load(arm_dir / "trained.pt", weights_only=False)
