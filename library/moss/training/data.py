"""What one training step reads: samples, the caption choice, the σ draw."""

from __future__ import annotations

import functools
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from library.env import resolve_under_home
from library.moss.requests import TrainRequest

EMPTY_PROMPT_PATH = "library/moss/asset/empty_prompt.safetensors"
NUM_TRAIN_TIMESTEPS = 1000


@dataclass
class Sample:
    """One training pair, lazily loadable. ``n_img`` / ``n_text`` order the run."""

    stem: str
    n_img: int
    n_text: int
    load: Callable[
        [], tuple[torch.Tensor, torch.Tensor]
    ]  # latent (128,h,w), embeds (S,4096)
    load_plain: Callable[[], torch.Tensor] | None = None  # embeds_plain (S',4096)
    n_ref: int = 0
    load_ref: Callable[[], torch.Tensor] | None = None  # clean ref latent (128,h,w)
    latent_path: Path | None = None  # finds the REPA target (latent metadata)
    n_variants: int = 0
    load_variant: Callable[[int], torch.Tensor] | None = None  # embeds_v{k}, from disk
    latent_hw: tuple[int, int] | None = None  # (h, w) latent tokens, from the file name
    n_text_min: int = 0  # the shortest text it can train on; 0 = n_text
    te_path: Path | None = None  # its captions (metadata) and the neg sidecar beside it


def samples_from_cache(cache_dir: Path) -> list[Sample]:
    from library.moss import cache as moss_cache

    out = []
    for item in moss_cache.list_items(cache_dir):
        load_plain = (
            functools.partial(moss_cache.load_plain, item)
            if item.n_text_plain
            else None
        )
        out.append(
            Sample(
                item.stem,
                (item.width // 16) * (item.height // 16),
                max(item.n_text, item.n_text_plain, item.n_text_variants),
                functools.partial(moss_cache.load_item, item),
                load_plain,
                item.n_ref,
                functools.partial(moss_cache.load_ref, item) if item.n_ref else None,
                item.latent_path,
                item.n_variants,
                functools.partial(moss_cache.load_variant, item)
                if item.n_variants
                else None,
                (item.height // 16, item.width // 16),
                item.n_text_min,
                item.te_path,
            )
        )
    return out


def sample_captions(te_path: Path) -> dict[str, str]:
    """The captions a TE file's embeddings encode, by ``choose_embeds`` kind:
    ``full`` (metadata ``caption``), ``plain`` (the ``.txt`` beside ``source``,
    when the file has ``embeds_plain``), ``v{k}`` (``caption_v{k}``), ``empty``."""
    from safetensors import safe_open

    with safe_open(str(te_path), "pt") as f:
        meta = f.metadata() or {}
        has_plain = "embeds_plain" in f.keys()
    out = {"full": meta["caption"], "empty": ""}
    if has_plain:
        out["plain"] = Path(meta["source"]).with_suffix(".txt").read_text().strip()
    for k, v in meta.items():
        if k.startswith("caption_v"):
            out["v" + k[len("caption_v") :]] = v
    return out


TEMPLATE_PREFIX, TEMPLATE_SUFFIX = 3, 7  # chat-template rows around the caption


def delta_row_ids(delta_rows: str, n_text: int, trigger_rows: list[int]) -> list[int]:
    """The text rows (0-based in the context) ``delta_rows`` keeps."""
    if delta_rows == "text":
        return list(range(n_text))
    if delta_rows == "caption":
        return list(
            range(TEMPLATE_PREFIX, max(TEMPLATE_PREFIX, n_text - TEMPLATE_SUFFIX))
        )
    if delta_rows == "trigger":
        return [r for r in trigger_rows if r < n_text]
    raise ValueError(f"delta_rows={delta_rows!r} has no row set")


def row_mask(n_img: int, n_text: int, rows: list[int], n_ref: int = 0, device=None):
    """``(1, n_img + n_text + n_ref, 1)`` bf16 mask keeping the text ``rows``."""
    mask = torch.zeros(
        1, n_img + n_text + n_ref, 1, dtype=torch.bfloat16, device=device
    )
    if rows:
        mask[0, [n_img + r for r in rows], 0] = 1
    return mask


NEG_SUFFIX = ".moss_te_neg.safetensors"


def load_neg(te_path: Path) -> dict[str, torch.Tensor]:
    """The trigger-stripped pair beside ``te_path`` (``preserve_weight``):
    ``full`` / ``plain`` → ``(pos, neg)`` bf16 CPU, both from the sidecar's own
    encoder (``embeds_pos`` / ``embeds``, ``embeds_plain_pos`` / ``embeds_plain``).
    Refuses a sidecar whose ``caption_sha256`` is not the TE file's."""
    from safetensors import safe_open

    path = te_path.with_name(te_path.name.replace(".moss_te.safetensors", NEG_SUFFIX))
    if not path.exists():
        raise SystemExit(
            f"preserve_weight needs {path.name} (project/moss_te_sae/src/trigger_neg_cache.py)"
        )
    with safe_open(str(te_path), "pt") as f:
        sha = (f.metadata() or {}).get("caption_sha256")
    with safe_open(str(path), "pt") as f:
        meta = f.metadata() or {}
        if meta.get("caption_sha256") != sha:
            raise SystemExit(f"{path.name}: stale, its caption is not the TE file's")
        out = {"full": (f.get_tensor("embeds_pos"), f.get_tensor("embeds"))}
        if "embeds_plain" in f.keys():
            out["plain"] = (
                f.get_tensor("embeds_plain_pos"),
                f.get_tensor("embeds_plain"),
            )
    return out


def choose_embeds(
    embeds: torch.Tensor,
    plain: torch.Tensor | None = None,
    n_variants: int = 0,
    load_variant: Callable[[int], torch.Tensor] | None = None,
    empty: torch.Tensor | None = None,
    *,
    plain_rate: float = 0.0,
    variant_rate: float = 0.0,
    dropout: float = 0.0,
    plain_rng: random.Random | None = None,
    variant_rng: random.Random | None = None,
    drop_rng: random.Random | None = None,
    worst_case: bool = False,
) -> tuple[torch.Tensor, str]:
    """The text one step trains on and its kind (``full`` / ``plain`` / ``v{k}`` /
    ``empty``): the plain caption at ``plain_rate``, else a variant (loaded on
    demand) at ``variant_rate``, then the empty prompt at ``dropout``. A candidate
    that is None / 0 draws nothing from its RNG. ``worst_case`` (the fit-report
    step) draws nothing and returns the longest of full, plain and every variant."""
    if worst_case:
        cands = [("full", embeds)]
        if plain is not None:
            cands.append(("plain", plain))
        cands += [(f"v{k}", load_variant(k)) for k in range(1, n_variants + 1)]
        kind, x = max(cands, key=lambda c: c[1].shape[0])
        return x, kind
    kind, x = "full", embeds
    if plain is not None and plain_rng.random() < plain_rate:
        kind, x = "plain", plain
    elif n_variants and variant_rng.random() < variant_rate:
        k = variant_rng.randint(1, n_variants)
        kind, x = f"v{k}", load_variant(k)
    if empty is not None and drop_rng.random() < dropout:
        kind, x = "empty", empty
    return x, kind


def sample_sigma(
    gen: torch.Generator,
    mu: float,
    s: float,
    cutoff: float = 0.95,
    uniform: float = 0.01,
    total: int = NUM_TRAIN_TIMESTEPS,
) -> float:
    """One σ on the ``{1/T … 1}`` grid, from report eq. (4)."""
    if torch.rand((), generator=gen).item() < uniform:
        j = int(torch.randint(1, total + 1, (), generator=gen))
        return j / total
    while True:
        z = torch.randn((), generator=gen).item() * s + mu
        noise = 1.0 - 1.0 / (1.0 + math.exp(-z))
        if noise <= cutoff:
            break
    j = min(total, max(1, math.ceil(noise * total / cutoff)))
    return j / total


def empty_prompt_embeds(req: TrainRequest) -> torch.Tensor:
    """``(S, 4096)`` bf16 embedding of ``""`` (10 chat-template tokens)."""
    from safetensors.torch import load_file, save_file

    path = resolve_under_home(EMPTY_PROMPT_PATH)
    if not path.exists():
        from library.moss.loader import encode_prompts

        print(f"encoding the empty prompt -> {path}", flush=True)
        embeds = encode_prompts(req.model_dir, [""])[""].to(torch.bfloat16)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"embeds": embeds.contiguous()}, str(path))
    return load_file(str(path))["embeds"]
