"""A job's conditioning: TE outputs cached by prompt, reference latents."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from library.moss.loader import (
    MAX_SEQUENCE_LENGTH,
    empty_cache,
    encode_prompts,
    load_vae,
)


def cached_embeds(
    model_dir, texts, cache_dir: Path, precision="bf16"
) -> dict[str, torch.Tensor]:
    """TE outputs keyed by text (and TE precision), read from / written to
    ``cache_dir`` so a rerun of the same prompts skips the text encoder."""
    from safetensors.torch import load_file, save_file

    def path(text):
        key = f"{MAX_SEQUENCE_LENGTH}\0{text}"
        if precision != "bf16":
            key = f"{precision}\0{key}"
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        return cache_dir / f"{digest}.safetensors"

    out, missing = {}, []
    for text in dict.fromkeys(texts):
        if path(text).is_file():
            out[text] = load_file(path(text))["embeds"]
        else:
            missing.append(text)
    if missing:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for text, emb in encode_prompts(
            model_dir, missing, precision=precision
        ).items():
            save_file({"embeds": emb.contiguous()}, path(text))
            out[text] = emb
    print(f"[te] {len(out) - len(missing)} cached, {len(missing)} encoded", flush=True)
    return out


def vision_embeds(
    model_dir, jobs: list[dict], cache_dir: Path, precision: str
) -> dict[str, torch.Tensor]:
    """Per ``ref_vision`` job key, the TE output of its reference page (fitted to
    the job's size) followed by its prompt, through the vision tower; read from /
    written to ``cache_dir`` like ``cached_embeds``."""
    from safetensors.torch import load_file, save_file

    from library.moss.caching.images import fit, load_rgb

    def path(j):
        key = "\0".join(
            map(
                str,
                (
                    "vision",
                    precision,
                    j["ref"],
                    j["ref_mtime"],
                    j["width"],
                    j["height"],
                    j["prompt"],
                ),
            )
        )
        return (
            cache_dir / f"v_{hashlib.sha1(key.encode()).hexdigest()[:16]}.safetensors"
        )

    out, missing = {}, []
    for j in jobs:
        if path(j).is_file():
            out[j["key"]] = load_file(path(j))["embeds"]
        else:
            missing.append(j)
    if missing:
        from library.moss.loader import load_tokenizer
        from library.moss.vision_te import encode, load_te

        device = torch.device("cuda")
        tok = load_tokenizer(model_dir)
        te = load_te(tok, device, model_dir, precision=precision)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for j in missing:
            w, h = j["width"], j["height"]
            page = fit(load_rgb(Path(j["ref"])), w, h)
            emb, n_img = encode(tok, te, page, j["prompt"], w * h, device)
            emb = emb.to("cpu", torch.bfloat16).contiguous()
            save_file({"embeds": emb}, path(j))
            out[j["key"]] = emb
            print(
                f"[te] {j['key']} page {w}x{h}: {emb.shape[0]} rows ({n_img} image)",
                flush=True,
            )
        del te, tok
        empty_cache()
    print(
        f"[te] vision: {len(out) - len(missing)} cached, {len(missing)} encoded",
        flush=True,
    )
    return out


def ref_latents(model_dir, jobs: list[dict], device) -> dict[str, torch.Tensor]:
    """Per job key, its reference fitted to the job's size and then to
    ``ref_size`` (as ``cache.cache_refs`` builds ``*.moss_ref``), ``(1,128,h,w)``
    fp32 on the CPU."""
    from library.moss.caching.images import fit, load_rgb, ref_size, to_pixels
    from library.moss.model import encode

    vae = load_vae(model_dir, device=device)
    out = {}
    for j in jobs:
        w, h = j["width"], j["height"]
        rw, rh = ref_size(w, h, j["ref_scale"])
        image = fit(fit(load_rgb(Path(j["ref"])), w, h), rw, rh)
        out[j["key"]] = encode(to_pixels(image), vae).float().cpu()
        print(f"[ref] {j['key']}: {w}x{h} -> {rw}x{rh}", flush=True)
    del vae
    empty_cache()
    return out
