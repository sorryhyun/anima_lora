"""Reading a MOSS cache folder and rewriting a TE file's caption variants."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from library.moss.scan import REF_SUFFIX, TE_SUFFIX, caption_sha, latent_files

_VARIANT_RE = re.compile(r"embeds_v[1-9][0-9]*")
_VARIANT_META_RE = re.compile(r"caption_v[1-9][0-9]*|variant_prefix")


@dataclass
class CacheItem:
    stem: str
    latent_path: Path
    te_path: Path
    width: int
    height: int
    n_text: int
    n_text_plain: int = 0  # 0 = no plain-caption embedding in the TE file
    ref_path: Path | None = None  # clean reference latent (i2i probe)
    n_ref: int = 0
    n_variants: int = 0  # embeds_v1 … embeds_v{n} in the TE file
    n_text_variants: int = 0  # the longest variant, its shared prefix included
    n_text_min: int = 0  # the shortest of full, plain and every variant


def list_items(cache_dir) -> list[CacheItem]:
    """Every stem with both a latent and a TE file, sorted by stem."""
    cache_dir = Path(cache_dir)
    latents: dict[str, tuple[Path, int, int]] = {}
    for stem, w, h, p in latent_files(cache_dir):
        prev = latents.get(stem)
        if prev is None or p.stat().st_mtime > prev[0].stat().st_mtime:
            latents[stem] = (p, w, h)
    items = []
    for stem, (path, w, h) in sorted(latents.items()):
        te = cache_dir / f"{stem}{TE_SUFFIX}"
        if not te.exists():
            continue
        with safe_open(te, "pt") as f:
            n_text = f.get_slice("embeds").get_shape()[0]
            n_plain = (
                f.get_slice("embeds_plain").get_shape()[0]
                if "embeds_plain" in f.keys()
                else 0
            )
            tails = [
                f.get_slice(k).get_shape()[0]
                for k in f.keys()
                if _VARIANT_RE.fullmatch(k)
            ]
            prefix = int((f.metadata() or {}).get("variant_prefix", 0))
        ref, n_ref = cache_dir / f"{stem}{REF_SUFFIX}", 0
        if ref.exists():
            with safe_open(ref, "pt") as f:
                _, rh, rw = f.get_slice("latent").get_shape()
            n_ref = rh * rw
        items.append(
            CacheItem(
                stem,
                path,
                te,
                w,
                h,
                n_text,
                n_plain,
                ref if n_ref else None,
                n_ref,
                len(tails),
                prefix + max(tails) if tails else 0,
                min(
                    [n_text]
                    + ([n_plain] if n_plain else [])
                    + [prefix + t for t in tails]
                ),
            )
        )
    return items


def load_ref(item: CacheItem) -> torch.Tensor | None:
    """``ref (128, h, w)`` fp32 on CPU, or ``None``."""
    if item.ref_path is None:
        return None
    return load_file(item.ref_path)["latent"].float()


def load_item(item: CacheItem) -> tuple[torch.Tensor, torch.Tensor]:
    """``latent (128, H/16, W/16)`` fp32, ``embeds (S, 4096)`` bf16, both on CPU."""
    latent = load_file(item.latent_path)["latent"].float()
    embeds = read_te(item.te_path)
    return latent, embeds


def read_te(path, key: str = "embeds") -> torch.Tensor:
    """One embedding ``(S, 4096)`` bf16 from a TE file, only that key read; an
    ``embeds_v{k}`` tail comes back behind its shared ``embeds`` prefix."""
    with safe_open(str(path), "pt") as f:
        x = f.get_tensor(key)
        prefix = int((f.metadata() or {}).get("variant_prefix", 0))
        if _VARIANT_RE.fullmatch(key) and prefix:
            x = torch.cat((f.get_slice("embeds")[:prefix], x))
    return x.to(torch.bfloat16)


def load_variant(item: CacheItem, k: int) -> torch.Tensor:
    """``embeds_v{k} (S_k, 4096)`` bf16 on CPU, read from disk on each call."""
    return read_te(item.te_path, f"embeds_v{k}")


def load_plain(item: CacheItem) -> torch.Tensor | None:
    """``embeds_plain (S', 4096)`` bf16 on CPU, or ``None`` when the file has none."""
    if not item.n_text_plain:
        return None
    return read_te(item.te_path, "embeds_plain")


def variant_entries(
    variants: list[tuple[str, torch.Tensor]], prefix: int
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """TE-file tensors + metadata for ``[(caption, embeds (S_k, 4096)), …]``, each a
    full embedding sharing its first ``prefix`` rows with ``embeds``: only the
    rows after them are stored (``read_te`` puts the prefix back)."""
    if not variants:
        return {}, {}
    tensors, meta = {}, {"variant_prefix": str(prefix)}
    for k, (caption, embeds) in enumerate(variants, 1):
        tensors[f"embeds_v{k}"] = embeds[prefix:].to("cpu", torch.bfloat16).contiguous()
        meta[f"caption_v{k}"] = caption
    return tensors, meta


def add_variants(path, variants: list[tuple[str, torch.Tensor]], prefix: int) -> None:
    """Replace the caption variants of the TE file at ``path`` with ``variants``
    (``variant_entries``), in place through a temp file; a symlinked file is
    rewritten at its target."""
    real = Path(path).resolve()
    tensors, meta = _without_variants(real)
    new, new_meta = variant_entries(variants, prefix)
    tmp = real.with_name(real.name + ".tmp")
    save_file(tensors | new, str(tmp), metadata=meta | new_meta)
    tmp.replace(real)


def _without_variants(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    with safe_open(str(path), "pt") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys() if not _VARIANT_RE.fullmatch(k)}
        meta = f.metadata() or {}
    return tensors, {k: v for k, v in meta.items() if not _VARIANT_META_RE.fullmatch(k)}


def carried_variants(
    path: Path, caption: str
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """The caption variants (tensors, metadata) of the TE file a re-encode is about
    to replace, while its caption is still ``caption``. After a caption edit they
    are stale: dropped, with a warning."""
    if not path.exists():
        return {}, {}
    with safe_open(str(path), "pt") as f:
        keys = [k for k in f.keys() if _VARIANT_RE.fullmatch(k)]
        if not keys:
            return {}, {}
        meta = f.metadata() or {}
        if meta.get("caption_sha256") != caption_sha(caption):
            print(
                f"  warning: {path.name}: caption changed, dropping its "
                f"{len(keys)} stale caption variants",
                flush=True,
            )
            return {}, {}
        tensors = {k: f.get_tensor(k) for k in keys}
    return tensors, {k: v for k, v in meta.items() if _VARIANT_META_RE.fullmatch(k)}
