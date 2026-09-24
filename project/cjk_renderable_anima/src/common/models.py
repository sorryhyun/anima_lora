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
    args = req.to_args()
    args.compile_blocks = True
    return args


def load_generator(
    size, steps: int, cfg: int | float, save: Path, negative_prompt: str = ""
):
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


def encode_images(vae, files, device, size=None, out_file=None):
    """VAE latents (float32, CPU) for image files, 8 per VAE call; ``size``
    ``(W, H)`` resizes first. Pixels go in at the IMAGE_TRANSFORMS range.

    ``out_file`` (a ``.npy`` path): the latents are written into that file
    chunk by chunk instead of being collected in RAM, ``<out_file>.done``
    holds the number of finished items so an interrupted encode resumes, and
    the result is the file mapped copy-on-write (pages are evictable)."""
    import numpy as np
    import torch
    from PIL import Image

    def chunk(i):
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
        return vae.encode_pixels_to_latents(px).float().cpu()

    if out_file is None:
        with torch.no_grad():
            return torch.cat([chunk(i) for i in range(0, len(files), 8)])
    out_file = Path(out_file)
    mark = out_file.with_suffix(out_file.suffix + ".done")
    done = int(mark.read_text()) if mark.exists() and out_file.exists() else 0
    mm = np.load(out_file, mmap_mode="r+") if done else None
    with torch.no_grad():
        for i in range(done, len(files), 8):
            lat = chunk(i)
            if mm is None:
                mm = np.lib.format.open_memmap(
                    out_file, "w+", np.float32, (len(files), *lat.shape[1:])
                )
            mm[i : i + len(lat)] = lat.numpy()
            if (i // 8) % 250 == 249:  # every 2 000 items
                mm.flush()
                mark.write_text(str(i + len(lat)))
    if mm is not None:
        mm.flush()
        del mm
    mark.write_text(str(len(files)))
    return torch.from_numpy(np.load(out_file, mmap_mode="c"))


class _TextEntry:
    """One cached caption, indexed like the tuple ``(prompt_embeds, attn_mask,
    t5_ids, t5_mask)``: row ``i`` of ``encode_captions``' four arrays (the
    max-padded embeds are an mmapped file)."""

    __slots__ = ("arrs", "i")

    def __init__(self, arrs, i):
        self.arrs, self.i = arrs, i

    def __getitem__(self, k):
        return self.arrs[k][self.i]


_TE_REST = ("attn_mask", "t5_ids", "t5_mask")


def _te_key(uniq) -> str:
    """What a text cache is valid for: the captions, the text encoder file and
    the pack's routing json (the T5 ids come from it)."""
    import hashlib

    from library.anima.vocab_pack import resolve_pack_prefix

    ck = checkpoints()
    h = hashlib.sha256("\n".join(uniq).encode())
    h.update(str(ck.text_encoder).encode())
    pj = (
        Path(str(resolve_pack_prefix(ck.vocab_pack)) + ".json")
        if ck.vocab_pack
        else None
    )
    h.update(pj.read_bytes() if pj is not None and pj.exists() else b"no-pack")
    return h.hexdigest()[:16]


def encode_captions(captions, device, cache_dir=None):
    """Unique captions → dict caption -> (prompt_embeds, attn_mask, t5_ids,
    t5_mask) on CPU.

    Everything is written to disk chunk by chunk as it is encoded — the
    max-padded embeds as a raw bf16 file (a row is 1 MB), the three id / mask
    arrays as ``.npy`` — and read back mapped, so nothing accumulates in RAM.
    ``cache_dir``: keep the files there (``meta.json`` holds the key and the
    number of finished captions; an interrupted encode resumes, a finished one
    is reused without loading the text encoder). Without it they go to a temp
    dir (``TMPDIR`` picks the disk) that is removed once mapped."""
    import json
    import shutil
    import tempfile

    import numpy as np
    import torch

    uniq = sorted(set(captions))
    if not uniq:
        return {}
    key = _te_key(uniq) if cache_dir is not None else ""
    d = (
        Path(cache_dir)
        if cache_dir is not None
        else Path(tempfile.mkdtemp(prefix="wake_te_"))
    )
    d.mkdir(parents=True, exist_ok=True)
    meta_f, emb_f = d / "meta.json", d / "embeds.bf16"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    if meta.get("key") != key or meta.get("n") != len(uniq):
        meta = {"key": key, "n": len(uniq), "done": 0}
    done = meta["done"] if emb_f.exists() else 0
    if done < len(uniq):
        from library.inference.models import load_text_encoder
        from library.inference.text import ensure_text_strategies

        tok, enc = ensure_text_strategies(checkpoints().text_encoder, vocab_pack=None)
        te = load_text_encoder(
            text_encoder=checkpoints().text_encoder, dtype=torch.bfloat16, device=device
        ).eval()
        rest = None
        with open(emb_f, "r+b" if done else "wb") as f, torch.no_grad():
            for i in range(done, len(uniq), 16):
                pe, am, t5, t5m = enc.encode_tokens(
                    tok, [te], tok.tokenize(uniq[i : i + 16])
                )
                pe = pe.to(torch.bfloat16).cpu().contiguous()
                shape = tuple(pe.shape[1:])
                assert meta.setdefault("shape", list(shape)) == list(shape), shape
                f.seek(i * shape[0] * shape[1] * 2)
                f.write(memoryview(pe.view(torch.int16).numpy()))
                others = [o.cpu().numpy() for o in (am, t5.long(), t5m)]
                if rest is None:
                    rest = [
                        np.lib.format.open_memmap(
                            d / f"{n}.npy",
                            "r+" if done else "w+",
                            o.dtype,
                            (len(uniq), *o.shape[1:]),
                        )
                        for n, o in zip(_TE_REST, others)
                    ]
                for r, o in zip(rest, others):
                    r[i : i + len(o)] = o
                if (i // 16) % 250 == 249:  # every 4 000 captions
                    f.flush()
                    for r in rest:
                        r.flush()
                    meta["done"] = i + len(others[0])
                    meta_f.write_text(json.dumps(meta))
        for r in rest:
            r.flush()
        del rest
        meta["done"] = len(uniq)
        meta_f.write_text(json.dumps(meta))
        te.to("cpu")
        del te
        torch.cuda.empty_cache()
    shape = tuple(meta["shape"])
    n = len(uniq) * shape[0] * shape[1]
    mm = torch.from_file(str(emb_f), shared=False, size=n, dtype=torch.bfloat16)
    arrs = [mm.view(len(uniq), *shape)] + [
        torch.from_numpy(np.load(d / f"{nm}.npy", mmap_mode="c")) for nm in _TE_REST
    ]
    if cache_dir is None:  # mapped: the files go with the process
        shutil.rmtree(d, ignore_errors=True)
    return {c: _TextEntry(arrs, i) for i, c in enumerate(uniq)}


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
