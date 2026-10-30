"""Train a LoRA on a precached folder: flow matching, batch 1.

Reads only what ``cache.py`` wrote (latents + text embeddings), so neither the
Qwen3.5-9B TE nor the VAE is loaded. ``--caption_dropout`` reads the empty prompt
from the shipped ``library/moss/asset/empty_prompt.safetensors`` (key ``embeds``;
one file for every cache, re-encoded only if it is missing).

A layout-JSON cache also holds each image's plain Anima caption
(``embeds_plain``); a step trains on it with probability ``plain_caption_rate``,
else on one of the TE file's caption variants (``embeds_v{k}``) with probability
``caption_variant_rate``, then on the empty prompt with ``caption_dropout``
(``choose_embeds``). Each of the three draws has its own RNG, so the shuffle and
σ sequence match a run with none of them. Samples stay in RAM up to
``PRELOAD_LIMIT_GB``; a larger set is read from disk each step (variants always are).
With ``--fit_report``, step 1 is the largest sample with its longest caption (full,
plain or a variant) and no dropout, so the report bounds the worst case.

Objective (report App. A.2): ``z_σ = (1−σ)·z0 + σ·ε``, the DiT predicts
``ε − z0`` at timestep ``1000·σ``, MSE averaged over image tokens and channels
(text tokens never reach the head), with no σ/SNR weighting. With
``region_weight`` / ``rest_weight`` off 1 (i2i, full-size reference) each token's
error is multiplied before the mean: ``region_weight`` on the tokens the edit
changes (``edit_region``, moving linearly to ``region_weight_end`` by the last
step), ``rest_weight`` on the copied ones.

Two ways to push the training signal through the DiT's caption-row reads instead
of the prompt-independent shift a style set invites (``project/moss_te_sae``):
``delta_rows`` confines the adapter delta to the text / caption / trigger rows of
the joint sequence (``row_mask``, a per-step mask on ``LoRANetwork``; the trigger
rows are found with the tokenizer, ``loader.trigger_rows``), and
``preserve_weight`` adds, on the same noisy latent, the MSE between the LoRA's and
the base model's prediction for the trigger-stripped caption (``preserve_loss``;
the pair comes from ``{stem}.moss_te_neg.safetensors``, ``load_neg``).

σ follows the final Stage-3 interval exactly: ``Z ~ N(μ, s²)``, ``S = 1 − sigmoid(Z)``,
truncated to ``S ≤ cutoff`` and rescaled onto the grid ``{1/1000 … 1}``
(``σ = ⌈1000·S/cutoff⌉ / 1000``), mixed with 1 % discrete uniform. Shift 1, no
resolution shift: that interval trained at area ≤ 1024², the size the cache
targets.

Default memory plan: no checkpointing, 12 blocks swapped, and the compiled
blocks' saved set capped by ``activation_memory_budget`` (0.1), which is faster
than checkpointing on 16 GB.
``--grad_checkpointing`` wraps each block's ``forward`` (``checkpoint_blocks``;
not ``__call__``), so the block swapper's forward hooks fire once per step and
never again during the recompute.

    make daemon-run ARGS="scripts/moss/train.py --cache post_image_dataset/moss/<set> --epochs 8"
"""

from __future__ import annotations

import functools
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from library.env import resolve_under_home
from library.moss.loader import (
    compile_blocks,
    empty_cache,
    free_vram_gb,
    load_transformer,
    stream_fp32_,
)
from library.moss.lora import LoRANetwork, load_channel_stats
from library.moss.requests import LORA_ALPHA, TrainRequest
from library.training.progress import ProgressSink, command_job_path, run_scope

EMPTY_PROMPT_PATH = "library/moss/asset/empty_prompt.safetensors"
NUM_TRAIN_TIMESTEPS = 1000
PRELOAD_LIMIT_GB = 16  # above this the samples are read from disk each step
PROGRESS_EVERY = 2  # steps per "  progress i/n" line (the GUI bar reads them)


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


def preserve_loss(
    transformer, network, noisy, timestep, neg_embeds, ref=None, attached=None
):
    """MSE between the adapter's and the base model's prediction on ``noisy`` for
    the trigger-stripped ``neg_embeds (1, S', 4096)``: the base one under
    ``multiplier`` 0 without grad, the adapter's with. ``attached`` (the block
    swapper): the no-grad forward has no backward to walk the swapped tail back,
    so its layout is restored by hand before the next forward."""
    network.set_multiplier(0.0)
    with torch.no_grad():
        base = transformer(noisy, timestep, neg_embeds, return_dict=False, ref=ref)[0]
    if attached is not None:
        attached.prepare(free_cache=False)
    network.set_multiplier(1.0)
    pred = transformer(noisy, timestep, neg_embeds, return_dict=False, ref=ref)[0]
    return F.mse_loss(pred.float(), base.float())


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


def edit_region(
    latent: torch.Tensor,
    ref: torch.Tensor,
    threshold: float,
    min_tokens: int = 5,
    dilate: int = 1,
) -> torch.Tensor:
    """``(h, w)`` bool: the tokens where ``latent`` differs from a same-size
    ``ref`` by more than ``threshold`` (RMS over channels), components under
    ``min_tokens`` dropped, grown by ``dilate``."""
    import numpy as np
    from scipy import ndimage

    diff = (latent.float() - ref.float()).pow(2).mean(0).sqrt()
    m = (diff > threshold).numpy()
    eight = np.ones((3, 3), dtype=bool)
    labels, n = ndimage.label(m, structure=eight)
    sizes = ndimage.sum(m, labels, range(1, n + 1))
    keep = np.isin(labels, 1 + np.flatnonzero(np.asarray(sizes) >= min_tokens))
    if dilate and keep.any():
        keep = ndimage.binary_dilation(keep, structure=eight, iterations=dilate)
    return torch.from_numpy(keep)


def region_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    region: torch.Tensor,
    region_weight: float,
    rest_weight: float,
    parts: dict | None = None,
) -> torch.Tensor:
    """The mean over tokens of the per-token MSE times ``region_weight`` on
    ``region`` and ``rest_weight`` elsewhere (not renormalised). ``parts``
    receives the plain mean over each side (``region`` / ``rest``) when both are
    non-empty."""
    err = (pred.float() - target.float()).pow(2).mean(1)[0]
    if parts is not None and region.any() and not region.all():
        parts.update(region=err[region].mean().item(), rest=err[~region].mean().item())
    return (torch.where(region, region_weight, rest_weight) * err).mean()


def tone_projection(path, b_weight: float = 1.0) -> torch.Tensor:
    """``(3, C)`` least-squares map from a per-channel latent offset to Lab
    ``(L, a, b)``, from the per-unit directions in ``path``; the b row scaled by
    ``b_weight``."""
    from safetensors.torch import load_file

    dirs = load_file(str(path))
    proj = torch.linalg.pinv(
        torch.stack([dirs[k] for k in ("L", "a", "b")], 1).double()
    ).float()
    proj[2] *= b_weight
    return proj


def tone_loss(
    err: torch.Tensor, rest: torch.Tensor, pool: int, proj: torch.Tensor | None = None
) -> torch.Tensor:
    """``err (1,C,h,w)`` (x̂₀ − x₀) averaged over ``rest (h,w)`` per ``pool``×``pool``
    cell (``pool`` 0 = one cell), cells under half coverage skipped. With ``proj``
    the cell means map to Lab and the term is their squared norm (ΔE²); without it,
    the mean square over channels. Mean over cells; 0 when no cell qualifies."""
    m = rest[None, None].to(err.dtype)
    if pool == 0:
        num, cov = (err * m).mean((2, 3), keepdim=True), m.mean((2, 3), keepdim=True)
        ok = cov[0, 0] > 0
    else:
        num = F.avg_pool2d(err * m, pool, ceil_mode=True)
        cov = F.avg_pool2d(m, pool, ceil_mode=True)
        ok = cov[0, 0] >= 0.5
    if not ok.any():
        return err.sum() * 0
    cells = (num / cov.clamp(min=1e-6))[0][:, ok]
    if proj is None:
        return cells.pow(2).mean()
    return (proj.to(cells) @ cells).pow(2).sum(0).mean()


def training_loss(
    transformer,
    latent,
    embeds,
    sigma: float,
    gen=None,
    ref=None,
    region=None,
    region_weight: float = 1.0,
    rest_weight: float = 1.0,
    parts: dict | None = None,
    tone: dict | None = None,
    state: dict | None = None,
):
    """``latent (1,128,h,w)`` fp32, ``embeds (1,S,4096)`` at true length,
    ``ref (1,128,h',w')`` an optional clean reference latent (never noised),
    ``region (h,w)`` bool on the latent's device: :func:`region_mse` instead of
    the plain MSE. ``tone``: ``rest`` mask, ``pool``, ``proj``, ``weight`` for
    :func:`tone_loss` on ``x̂₀ − x₀ = σ·(target − pred)``, added at ``weight``
    and logged in ``parts`` as ``tone``. ``state`` receives ``noisy`` and
    ``timestep`` (:func:`preserve_loss` runs on the same ones)."""
    noise = torch.randn(
        latent.shape, generator=gen, device=latent.device, dtype=latent.dtype
    )
    noisy = (1.0 - sigma) * latent + sigma * noise
    target = noise - latent
    timestep = torch.tensor([NUM_TRAIN_TIMESTEPS * sigma], device=latent.device)
    if state is not None:
        state.update(noisy=noisy, timestep=timestep)
    pred = transformer(noisy, timestep, embeds, return_dict=False, ref=ref)[0]
    if region is not None:
        loss = region_mse(pred, target, region, region_weight, rest_weight, parts)
    else:
        loss = F.mse_loss(pred.float(), target.float())
    if tone is None:
        return loss
    term = tone_loss(
        sigma * (target.float() - pred.float()),
        tone["rest"],
        tone["pool"],
        tone["proj"],
    )
    if parts is not None:
        parts["tone"] = term.item()
        if pred.requires_grad:
            # Output-space gradient norms, the tone one at the configured weight
            # even before tone_start: one backward cannot split them per param.
            g_fm, g_tone = (
                torch.autograd.grad(x, pred, retain_graph=True)[0].float()
                for x in (loss, term)
            )
            g_tone = g_tone * tone.get("log_weight", tone["weight"])
            parts.update(
                g_fm=g_fm.norm().item(),
                g_tone=g_tone.norm().item(),
                g_cos=F.cosine_similarity(g_fm.flatten(), g_tone.flatten(), 0).item(),
            )
    return loss + tone["weight"] * term if tone["weight"] else loss


def ema_decay(span: float, updates: int) -> float:
    """Decay whose EMA averages ``span`` of the run's ``updates`` (1/(1 - decay))."""
    return 1.0 - 1.0 / max(1.0, span * updates)


def ema_step(
    shadows: list[list], staging: list, params: list, decays: list[float]
) -> None:
    """Copy ``params`` into the pinned CPU ``staging`` buffers once and lerp each
    fp32 shadow toward them at its decay; a shadow's first call (empty) seeds it."""
    with torch.no_grad():
        for buf, p in zip(staging, params):
            buf.copy_(p)
        for shadow, decay in zip(shadows, decays):
            if shadow:
                torch._foreach_lerp_(shadow, staging, 1.0 - decay)
            else:
                shadow.extend(buf.clone() for buf in staging)


def checkpoint_blocks(blocks) -> None:
    """Per-block activation checkpointing around ``block.forward``."""
    for block in blocks:
        inner = block.forward

        @functools.wraps(inner)
        def forward(*args, _inner=inner, **kwargs):
            if not torch.is_grad_enabled():
                return _inner(*args, **kwargs)
            return checkpoint(_inner, *args, use_reentrant=False, **kwargs)

        block.forward = forward


# Uncheckpointed, the compiled saved set is ~5.25 MB per joint token at budget
# 1.0 (128 KiB per block x 42); the constant covers AdamW state and allocator
# growth (fit to budgets 0.15-0.25 at 4.4k tokens, ~1 GiB above measured).
_BUDGET_BASE_GB = 3.5
_BUDGET_MB_PER_TOKEN = 6.0


def step_reserve_gb(
    joint: int, *, grad_checkpointing: bool, budget: float = 1.0
) -> float:
    """VRAM one training step needs beyond the placed weights, at ``joint``
    (image + text) tokens."""
    if grad_checkpointing:
        # ~0.6 MB per joint token (qwen21's measured slope, same hidden width
        # class) plus one recomputed block.
        return 1.5 + 0.6 * joint / 1024
    return _BUDGET_BASE_GB + budget * _BUDGET_MB_PER_TOKEN * joint / 1024


def budget_for_free(free_gb: float, joint: int, step: float = 0.05) -> float:
    """The largest budget on a ``step`` grid whose :func:`step_reserve_gb` fits
    ``free_gb``; 0.0 when not even one ``step`` fits."""
    room = (free_gb - _BUDGET_BASE_GB) / (_BUDGET_MB_PER_TOKEN * joint / 1024)
    return min(1.0, math.floor(max(0.0, room) / step + 1e-9) * step)


def set_activation_memory_budget(budget: float) -> None:
    """Inductor's partitioner reads this at compile time: set it before
    ``compile_blocks`` (and reset ``torch._dynamo`` to change it later)."""
    import torch._functorch.config as functorch_config

    functorch_config.activation_memory_budget = budget


def place(
    transformer,
    device,
    blocks_to_swap: int,
    reserve_gb: float,
    h2d_only: bool = False,
):
    """Everything on ``device``, or the block swapper for ``blocks_to_swap`` > 0
    (``-1`` sizes it from ``mem_get_info`` free). Returns the attachment or None."""
    from library.runtime.blockswap import place as place_blocks

    return place_blocks(
        transformer,
        "layers",
        device,
        blocks_to_swap=None if blocks_to_swap < 0 else blocks_to_swap,
        supports_backward=True,
        activation_reserve_gb=reserve_gb,
        label="transformer",
        h2d_only=h2d_only,
    )


def report_fit(transformer, attached, n_img: int, n_text: int) -> dict:
    """After step 1 (the largest sample): peak against what is actually free."""
    import library.runtime.blockswap as blockswap

    allocated = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    free = free_vram_gb()
    per = blockswap.block_size_gb(transformer.layers)
    swapped = attached.offloader.blocks_to_swap if attached else 0
    fit = {
        "image_tokens": n_img,
        "text_tokens": n_text,
        "peak_allocated_gb": round(allocated, 3),
        "peak_reserved_gb": round(reserved, 3),
        "free_gb": round(free, 3),
        "blocks_to_swap": swapped,
    }
    print(
        f"fit @ step 1 ({n_img} image + {n_text} text tokens, the largest sample): "
        f"peak {allocated:.2f} GB allocated / {reserved:.2f} GB reserved, "
        f"{free:.2f} GB actually free, swapping {swapped} at {per:.3f} GB each",
        flush=True,
    )
    if swapped:
        spare = max(0, int((free - per) // per))
        if spare:
            print(
                f"  -> rerun with --blocks_to_swap {max(0, swapped - spare)}",
                flush=True,
            )
    return fit


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


def run_train(req: TrainRequest) -> Path:
    cache_dir = resolve_under_home(req.cache)
    samples = samples_from_cache(cache_dir)
    if not samples:
        raise SystemExit(f"no cached pairs under {cache_dir}")
    empty = empty_prompt_embeds(req) if req.caption_dropout > 0 else None
    return train(req, samples, empty)


def train(
    req: TrainRequest, samples: list[Sample], empty: torch.Tensor | None = None
) -> Path:
    """The loop, over already-built ``samples`` (``run_train`` reads the cache;
    a fit smoke passes synthetic ones)."""
    torch.manual_seed(req.seed)
    random.seed(req.seed)
    gen = torch.Generator().manual_seed(req.seed)
    device = torch.device("cuda")

    img = sorted(s.n_img for s in samples)
    txt = sorted(s.n_text for s in samples)
    print(
        f"{len(samples)} samples, image {img[0]}-{img[-1]} tokens, text {txt[0]}-{txt[-1]} tokens",
        flush=True,
    )
    n_refs = sum(s.load_ref is not None for s in samples)
    if n_refs:
        ref_tok = sorted(s.n_ref for s in samples if s.load_ref is not None)
        print(
            f"reference latents: {n_refs}/{len(samples)} samples, "
            f"{ref_tok[0]}-{ref_tok[-1]} tokens",
            flush=True,
        )
    joint = max(s.n_img + s.n_ref + s.n_text for s in samples)
    use_plain = req.plain_caption_rate > 0
    n_plain = sum(s.load_plain is not None for s in samples) if use_plain else 0
    if n_plain:
        print(
            f"plain Anima captions: {n_plain}/{len(samples)} samples, "
            f"rate {req.plain_caption_rate}",
            flush=True,
        )

    def fetch(s):
        """``(latent, embeds, plain or None, ref or None)`` on CPU."""
        latent, embeds = s.load()
        alt = s.load_plain() if use_plain and s.load_plain is not None else None
        ref = s.load_ref() if s.load_ref is not None else None
        return latent, embeds, alt, ref

    # bf16 TE rows (S×4096×2 bytes) + fp32 latents (128×4 bytes per token).
    preload_bytes = sum(
        s.n_text * 8192 * (2 if use_plain and s.load_plain else 1)
        + (s.n_img + s.n_ref) * 512
        for s in samples
    )
    if preload_bytes <= PRELOAD_LIMIT_GB * 1024**3:
        resident = {s.stem: fetch(s) for s in samples}

        def get(s):
            return resident[s.stem]
    else:
        get = fetch
        print(
            f"data ~{preload_bytes / 1024**3:.0f} GB > {PRELOAD_LIMIT_GB} GB: "
            "read from disk each step",
            flush=True,
        )
    regions = {}
    region_end = (
        req.region_weight if req.region_weight_end is None else req.region_weight_end
    )
    if req.region_weight != 1.0 or region_end != 1.0 or req.rest_weight != 1.0:
        for s in samples:
            latent, _, _, ref = get(s)
            if ref is None or ref.shape != latent.shape:
                raise SystemExit(
                    f"region_weight / rest_weight need a full-size reference (ref_scale 1); "
                    f"{s.stem} has {None if ref is None else tuple(ref.shape)} "
                    f"for latent {tuple(latent.shape)}"
                )
            regions[s.stem] = edit_region(
                latent,
                ref,
                req.region_threshold,
                req.region_min_tokens,
                req.region_dilate,
            )
        area = sorted(r.float().mean().item() for r in regions.values())
        print(
            f"edit region: x{req.region_weight}"
            f"{f' -> x{region_end}' if region_end != req.region_weight else ''}"
            f", rest x{req.rest_weight}, threshold "
            f"{req.region_threshold}, min {req.region_min_tokens} tokens, dilate "
            f"{req.region_dilate}; area median {100 * area[len(area) // 2]:.2f}% "
            f"({100 * area[0]:.2f}-{100 * area[-1]:.2f}%), "
            f"{sum(a == 0 for a in area)} empty",
            flush=True,
        )
    rests, tone_proj, tone_proj_b = {}, None, None
    if req.tone_weight or req.tone_log:
        for s in samples:
            latent, _, _, ref = get(s)
            if ref is None or ref.shape != latent.shape:
                raise SystemExit(
                    f"tone_weight / tone_log need a full-size reference (ref_scale 1); "
                    f"{s.stem} has {None if ref is None else tuple(ref.shape)} "
                    f"for latent {tuple(latent.shape)}"
                )
            rest = ~edit_region(
                latent,
                ref,
                req.region_threshold,
                req.region_min_tokens,
                req.region_dilate + req.tone_dilate,
            )
            if rest.any():
                rests[s.stem] = rest
        if req.tone_dirs:
            tone_proj = tone_projection(resolve_under_home(req.tone_dirs)).to(device)
            tone_proj_b = tone_projection(
                resolve_under_home(req.tone_dirs), req.tone_b_weight
            ).to(device)
        share = sorted(r.float().mean().item() for r in rests.values())
        print(
            f"tone: x{req.tone_weight}{' (log only)' if not req.tone_weight else ''}, "
            f"pool {req.tone_pool}, dilate +{req.tone_dilate}, "
            f"{'Lab via ' + req.tone_dirs if req.tone_dirs else 'raw channels'}"
            f"{f' (b x{req.tone_b_weight} from {req.tone_b_start:g})' if req.tone_dirs and req.tone_b_weight != 1 else ''}"
            f"{f', added from {req.tone_start:g}' if req.tone_start else ''}; "
            f"{len(rests)}/{len(samples)} samples with unedited tokens, share median "
            f"{100 * share[len(share) // 2] if share else 0:.0f}%",
            flush=True,
        )
    use_variants = req.caption_variant_rate > 0
    n_variant_samples = sum(1 for s in samples if s.n_variants) if use_variants else 0
    if n_variant_samples:
        print(
            f"caption variants: {n_variant_samples}/{len(samples)} samples, "
            f"rate {req.caption_variant_rate}",
            flush=True,
        )
    rates = dict(
        plain_rate=req.plain_caption_rate,
        variant_rate=req.caption_variant_rate,
        dropout=req.caption_dropout,
        plain_rng=random.Random(req.seed + 1),
        drop_rng=random.Random(req.seed + 2),
        variant_rng=random.Random(req.seed + 3),
    )
    kinds = {"plain": 0, "variant": 0, "empty": 0, "preserve": 0}

    # delta_rows: the text rows (per sample and caption kind) the adapter may touch.
    rows_of: dict[tuple[str, str], list[int]] = {}
    if req.delta_rows != "all":
        if req.delta_rows == "trigger" and not req.trigger:
            raise SystemExit("delta_rows=trigger needs --trigger")
        if req.mod_rank:
            raise SystemExit("delta_rows applies to token rows; drop mod_rank")
        tok = None
        if req.delta_rows == "trigger":
            from library.moss.loader import load_tokenizer, trigger_rows

            tok = load_tokenizer(req.model_dir)
        found = {"full": 0, "plain": 0}
        for s in samples:
            if s.te_path is None:
                raise SystemExit(f"{s.stem}: delta_rows needs the cache's TE files")
            _, full, alt, _ = get(s)
            for kind, caption in sample_captions(s.te_path).items():
                if kind == "full":
                    n = full.shape[0]
                elif kind == "plain":
                    n = alt.shape[0] if alt is not None else s.load_plain().shape[0]
                elif kind == "empty":
                    n = TEMPLATE_PREFIX + TEMPLATE_SUFFIX
                else:
                    n = s.load_variant(int(kind[1:])).shape[0]
                hits = trigger_rows(tok, caption, req.trigger) if tok else []
                rows_of[s.stem, kind] = delta_row_ids(req.delta_rows, n, hits)
                if kind in found and hits:
                    found[kind] += 1
        if req.delta_rows == "trigger":
            print(
                f"delta rows: trigger {req.trigger!r} found in {found['full']}/{len(samples)} "
                f"full and {found['plain']}/{sum(s.load_plain is not None for s in samples)} "
                "plain captions",
                flush=True,
            )
            if not found["full"]:
                raise SystemExit("the trigger is in none of the full captions")
        else:
            print(f"delta rows: {req.delta_rows}", flush=True)
    negs: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    if req.preserve_weight > 0:
        for s in samples:
            if s.te_path is None:
                raise SystemExit(
                    f"{s.stem}: preserve_weight needs the cache's TE files"
                )
            negs[s.stem] = load_neg(s.te_path)
        print(
            f"preserve: x{req.preserve_weight:g} on the trigger-stripped caption "
            f"({NEG_SUFFIX} beside every TE file; full and plain steps)",
            flush=True,
        )

    transformer = load_transformer(req.model_dir, req.model)
    transformer.requires_grad_(False)
    if req.stream == "fp32":
        stream_fp32_(transformer)
    transformer.ref_timestep = req.ref_timestep if req.ref_timestep >= 0 else None
    transformer.ref_isolate = req.ref_isolate
    alpha = req.alpha if req.alpha is not None or req.algo != "lora" else LORA_ALPHA
    rank_pattern = None
    if req.mod_rank:
        if req.algo != "lora":
            raise SystemExit(f"mod_rank is lora only, not {req.algo}")
        mod_alpha = req.mod_alpha
        if mod_alpha is None:
            mod_alpha = alpha * math.sqrt(req.mod_rank / req.rank)
        rank_pattern = {"shared_block_modulation": (req.mod_rank, mod_alpha)}
    network = LoRANetwork(
        transformer,
        rank=req.rank,
        alpha=alpha,
        targets=req.targets,
        dtype=torch.float32,
        down_init=req.down_init,
        algo=req.algo,
        factor=req.factor,
        channel_stats=load_channel_stats(resolve_under_home(req.channel_stats))
        if req.algo == "lora" and req.svd_drop_loud
        else None,
        channel_mean=load_channel_stats(resolve_under_home(req.channel_mean))
        if req.algo == "lora" and req.down_init == "weight_svd_nodc"
        else None,
        svd_slice=req.svd_slice,
        svd_slice_count=req.svd_slice_count,
        svd_drop_loud=req.svd_drop_loud,
        rank_pattern=rank_pattern,
    )
    patched = network.apply_to()
    network.to(device)
    detail = {
        "lora": f"{req.down_init} down"
        + (
            f" slice {req.svd_slice}"
            + (f"/{req.svd_slice_count} interleaved" if req.svd_slice_count else "")
            if req.svd_slice or req.svd_slice_count
            else ""
        )
        + (
            f", {req.svd_drop_loud} loud channel(s) dropped"
            if req.svd_drop_loud
            else ""
        ),
        "lokr": f"factor {req.factor}",
        "loha": "hadamard",
    }[req.algo]
    for pat, (r, a) in network.rank_pattern.items():
        detail += f"; {pat} rank {r} alpha {a:g}"
    print(
        f"{req.algo}: rank {req.rank} alpha {network.alpha} ({detail}) on {patched} linears, "
        f"{network.num_parameters / 1e6:.1f}M params (fp32 masters, saved {req.lora_dtype})",
        flush=True,
    )

    budgeted = req.compile and not req.grad_checkpointing
    budget = req.activation_memory_budget if budgeted else 1.0
    reserve_gb = step_reserve_gb(
        joint, grad_checkpointing=req.grad_checkpointing, budget=budget
    )
    attached = place(
        transformer, device, req.blocks_to_swap, reserve_gb, req.swap_h2d_only
    )
    if budgeted and budget < 1.0:
        set_activation_memory_budget(budget)
        print(f"activation memory budget {budget}", flush=True)
    elif not req.compile and not req.grad_checkpointing:
        print(
            "warning: eager without checkpointing saves ~10 MB per joint token "
            "per step; add --grad_checkpointing unless the card has room",
            flush=True,
        )
    repa, repa_targets = None, {}
    if req.repa_weight > 0:
        from library.moss.repa import REPATerm, load_target, pe_sidecar

        pe_root = resolve_under_home(req.repa_pe_root)
        for s in samples:
            if s.latent_path is None or s.latent_hw is None:
                continue
            h, w = s.latent_hw
            path = pe_sidecar(s.latent_path, s.stem, w * 16, h * 16, pe_root)
            if path is not None:
                repa_targets[s.stem] = load_target(path, h, w)
        if not repa_targets:
            raise SystemExit(
                f"repa_weight {req.repa_weight} but no sample has a PE-Spatial sidecar "
                f"under {pe_root} (make preprocess-pe-spatial)"
            )
        repa = REPATerm(transformer, req.repa_layer)
        print(
            f"REPA: block {req.repa_layer}, weight {req.repa_weight:g}, DoG-Gram target "
            f"on {len(repa_targets)}/{len(samples)} samples",
            flush=True,
        )
    if req.compile:

        def shortest(s):
            n = s.n_text_min or s.n_text
            for _pos, neg in negs.get(s.stem, {}).values():
                n = min(n, neg.shape[0])
            return n if empty is None else min(n, len(empty))

        seq_range = (min(s.n_img + s.n_ref + shortest(s) for s in samples), joint)
        print(f"compile: token axis marked {seq_range[0]}-{seq_range[1]}", flush=True)
        compile_blocks(transformer.layers, seq_range)
    else:
        seq_range = None

    def set_rows(n_text: int, rows: list[int], s: Sample) -> None:
        """The step's row mask on the network; its token axis marked like the
        block inputs' (``compile_blocks``), or it would specialise against them."""
        mask = row_mask(s.n_img, n_text, rows, s.n_ref, device)
        if seq_range is not None and seq_range[0] <= mask.shape[1] <= seq_range[1]:
            torch._dynamo.mark_dynamic(mask, 1, min=seq_range[0], max=seq_range[1])
        network.set_row_mask(mask)

    if req.grad_checkpointing:
        checkpoint_blocks(transformer.layers)
    empty_cache()
    print(f"free VRAM before training: {free_vram_gb():.2f} GB", flush=True)

    params = list(network.parameters())
    optimizer_name = req.optimizer
    if optimizer_name == "muon" and req.algo != "lora":
        print(
            f"muon is measured on lora only; {req.algo} trains with adamw", flush=True
        )
        optimizer_name = "adamw"
    if optimizer_name == "muon":
        # weight_decay = torch's AdamW default, so only the update rule changes.
        optimizer = torch.optim.Muon(
            params, lr=req.lr, weight_decay=0.01, adjust_lr_fn="match_rms_adamw"
        )
        print(f"muon on {len(params)} matrices", flush=True)
    else:
        optimizer = torch.optim.AdamW(params, lr=req.lr)
    total = req.epochs * len(samples)
    if req.max_steps:
        total = min(total, req.max_steps) if req.epochs else req.max_steps
    if req.grad_accum < 1:
        raise SystemExit(f"grad_accum must be >= 1, got {req.grad_accum}")
    accum = req.grad_accum
    updates = math.ceil(total / accum)
    warmup = max(1, int(updates * req.warmup_ratio))
    lr_sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: min(1.0, (update + 1) / warmup)
    )
    epochs = math.ceil(total / len(samples))
    print(
        f"{total} steps ({epochs} epochs x {len(samples)}), "
        + (f"{updates} updates of {accum}, " if accum > 1 else "")
        + f"warmup {warmup}",
        flush=True,
    )
    try:
        extra = [float(x) for x in req.ema_extra.split(",") if x.strip()]
    except ValueError:
        raise SystemExit(
            f"ema_extra must be comma-separated floats: {req.ema_extra!r}"
        ) from None
    spans = ([req.ema_window] if req.ema_window else []) + extra
    if not all(0.0 < s <= 1.0 for s in spans):
        raise SystemExit(f"EMA spans must be in (0, 1], got {spans}")
    decays = [ema_decay(s, updates) for s in spans]
    shadows = [[] for _ in spans]
    main_shadow = shadows[0] if req.ema_window else None
    staging, n_updates = [], 0
    if spans:
        staging = [
            torch.empty(p.shape, dtype=torch.float32, pin_memory=True) for p in params
        ]
        print(
            f"EMA from update {warmup}, CPU shadow: "
            + ", ".join(
                f"span {s:g} = decay {d:.4f} (~{1 / (1 - d):.0f} updates)"
                for s, d in zip(spans, decays)
            ),
            flush=True,
        )

    out_path = resolve_under_home(req.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "base_variant": req.model,
        "lr": req.lr,
        "samples": len(samples),
        "steps": total,
        "image_tokens": f"{img[0]}-{img[-1]}",
        "sigma": f"mu={req.sigma_mu} s={req.sigma_s} cutoff={req.sigma_cutoff} "
        f"uniform={req.sigma_uniform}",
        "caption_dropout": req.caption_dropout,
        "plain_caption_rate": req.plain_caption_rate if n_plain else 0.0,
        "caption_variant_rate": req.caption_variant_rate if n_variant_samples else 0.0,
        "reference_latents": n_refs,
    }
    if accum > 1:
        meta["grad_accum"] = accum
    meta["optimizer"] = optimizer_name
    if req.stream != "bf16":
        meta["stream"] = req.stream
    if req.delta_rows != "all":
        meta["delta_rows"] = req.delta_rows
    if req.trigger:
        meta["trigger"] = req.trigger
    if req.preserve_weight:
        meta["preserve_weight"] = req.preserve_weight
    if n_refs:
        meta["ref_timestep"] = req.ref_timestep
        if req.ref_isolate:
            meta["ref_isolate"] = "1"
    if repa is not None:
        meta.update(repa_weight=req.repa_weight, repa_layer=req.repa_layer)
    if regions:
        meta.update(
            region_weight=req.region_weight,
            rest_weight=req.rest_weight,
            region_threshold=req.region_threshold,
            region_min_tokens=req.region_min_tokens,
            region_dilate=req.region_dilate,
        )
        if region_end != req.region_weight:
            meta["region_weight_end"] = region_end
    if req.tone_weight:
        meta.update(
            tone_weight=req.tone_weight,
            tone_pool=req.tone_pool,
            tone_dilate=req.tone_dilate,
            tone_dirs=req.tone_dirs or "raw",
        )
        if req.tone_dirs and req.tone_b_weight != 1:
            meta.update(tone_b_weight=req.tone_b_weight, tone_b_start=req.tone_b_start)
        if req.tone_start:
            meta.update(tone_start=req.tone_start)

    def save(tag: str | None = None, shadow: list | None = None) -> Path:
        """The final file at ``out_path``; an epoch save under ``<dir>/<stem>/``.
        A started EMA ``shadow`` is saved in place of the live weights, swapped in
        through the staging buffers."""
        if tag is None:
            path = out_path
        else:
            folder = out_path.parent / out_path.stem
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{out_path.stem}_{tag}{out_path.suffix}"
        if not shadow:
            network.save(path, req.lora_dtype, **meta)
            return path
        k = next(i for i, s in enumerate(shadows) if s is shadow)
        with torch.no_grad():
            for buf, s, p in zip(staging, shadow, params):
                buf.copy_(p)
                p.copy_(s)
            network.save(
                path,
                req.lora_dtype,
                **meta,
                ema_window=spans[k],
                ema_decay=round(decays[k], 6),
            )
            for buf, p in zip(staging, params):
                p.copy_(buf)
        return path

    sink_path = command_job_path(
        str(out_path.parent.parent / "logs" / f"{out_path.stem}.progress.jsonl")
    )
    sink = ProgressSink(sink_path, run=out_path.stem, method="moss", preset=None)
    sink.run_start(total_steps=total, total_epochs=epochs, pid=os.getpid())
    print(f"progress sink: {sink_path}", flush=True)

    largest = max(samples, key=lambda s: s.n_img + s.n_ref + s.n_text)
    history, losses_all, fit = [], [], None
    step, grad_norm = 0, float("nan")
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    with run_scope(sink, final_step=lambda: step):
        for epoch in range(epochs):
            order = random.sample(samples, len(samples))
            if epoch == 0 and req.fit_report:
                # Only the largest sample moves to step 1, the worst case the fit
                # report bounds; the rest of epoch 1 stays shuffled.
                order.remove(largest)
                order.insert(0, largest)
            losses, aligns, t_epoch = [], [], time.time()
            for s in order:
                if step >= total:
                    break
                latent, embeds, alt, ref = get(s)
                embeds, kind = choose_embeds(
                    embeds,
                    alt,
                    s.n_variants if use_variants else 0,
                    s.load_variant,
                    empty,
                    worst_case=step == 0 and req.fit_report,
                    **rates,
                )
                neg = negs.get(s.stem, {}).get(kind)
                if neg is not None:
                    embeds, neg = neg  # the sidecar's own encode of the full caption
                    kinds["preserve"] += 1
                text_rows = rows_of.get((s.stem, kind))
                kind = "variant" if kind.startswith("v") else kind
                if kind in kinds:
                    kinds[kind] += 1
                latent = latent.unsqueeze(0).to(device, torch.float32)
                embeds = embeds.unsqueeze(0).to(device)
                if ref is not None:
                    ref = ref.unsqueeze(0).to(device, torch.float32)
                if rows_of:
                    set_rows(embeds.shape[1], text_rows or [], s)
                sigma = sample_sigma(
                    gen, req.sigma_mu, req.sigma_s, req.sigma_cutoff, req.sigma_uniform
                )
                if repa is not None:
                    target, grid = repa_targets.get(s.stem, (None, None))
                    repa.prime(
                        None if target is None else target.to(device),
                        grid,
                        tuple(latent.shape[-2:]),
                    )
                parts, state = {}, {}
                tone = None
                if s.stem in rests:
                    tone = dict(
                        rest=rests[s.stem].to(device),
                        pool=req.tone_pool,
                        proj=tone_proj_b
                        if step >= total * req.tone_b_start
                        else tone_proj,
                        weight=req.tone_weight
                        if step >= total * req.tone_start
                        else 0.0,
                        log_weight=req.tone_weight,
                    )
                if s.stem in regions or tone is not None:
                    loss = training_loss(
                        transformer,
                        latent,
                        embeds,
                        sigma,
                        ref=ref,
                        region=regions[s.stem].to(device)
                        if s.stem in regions
                        else None,
                        region_weight=req.region_weight
                        + (region_end - req.region_weight) * step / max(total - 1, 1),
                        rest_weight=req.rest_weight,
                        parts=parts,
                        tone=tone,
                        state=state,
                    )
                else:
                    loss = training_loss(
                        transformer, latent, embeds, sigma, ref=ref, state=state
                    )
                align = repa.loss() if repa is not None else None
                if align is not None:
                    ((loss + req.repa_weight * align) / accum).backward()
                    aligns.append(align.item())
                else:
                    (loss / accum).backward()
                if neg is not None:
                    # Its own backward: the two graphs never sit on the card together.
                    neg = neg.unsqueeze(0).to(device)
                    if rows_of:
                        # The stripped caption has no trigger rows.
                        set_rows(
                            neg.shape[1],
                            delta_row_ids(req.delta_rows, neg.shape[1], []),
                            s,
                        )
                    pres = preserve_loss(
                        transformer,
                        network,
                        state["noisy"],
                        state["timestep"],
                        neg,
                        ref,
                        attached,
                    )
                    (req.preserve_weight * pres / accum).backward()
                    parts["pres"] = pres.item()
                step += 1
                updated = step % accum == 0 or step == total
                if updated:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        params, req.max_grad_norm
                    ).item()
                    optimizer.step()
                    lr_sched.step()
                    optimizer.zero_grad(set_to_none=True)
                    n_updates += 1
                    if shadows and n_updates >= warmup:
                        ema_step(shadows, staging, params, decays)
                value = loss.item()
                sink.log(
                    {
                        "loss": value,
                        "sigma": sigma,
                        "lr": lr_sched.get_last_lr()[0],
                    }
                    | ({"grad_norm": grad_norm} if updated else {})
                    | ({"repa": aligns[-1]} if align is not None else {})
                    | {f"loss_{k}": v for k, v in parts.items()},
                    global_step=step,
                    epoch=epoch + 1,
                )
                losses.append(value)
                losses_all.append(
                    {"step": step, "loss": value, "sigma": sigma}
                    | ({"repa": aligns[-1]} if align is not None else {})
                    | {f"loss_{k}": v for k, v in parts.items()}
                )
                if step == 1:
                    if req.fit_report:
                        fit = report_fit(
                            transformer, attached, s.n_img, embeds.shape[1]
                        )
                    t_start = time.time()  # s/step excludes step 1's allocator warm-up
                    t_window, step_window, per_step = t_start, 1, None
                eta_rate = (
                    (time.time() - t_start) / max(1, step - 1)
                    if per_step is None
                    else per_step
                )
                eta = int(eta_rate * (total - step))
                if step % PROGRESS_EVERY == 0 or step == total:
                    print(
                        f"  progress {step}/{total} epoch {epoch + 1}/{epochs} "
                        f"loss {sum(losses) / len(losses):.4f} "
                        f"eta {eta // 3600}:{eta % 3600 // 60:02d}:{eta % 60:02d}",
                        flush=True,
                    )
                if step % req.log_every == 0:
                    # s/step over the steps since the last log line, so a recompile
                    # stays in the window it landed in. Step 1 (compile) has none.
                    now = time.time()
                    if step > step_window:
                        per_step = (now - t_window) / (step - step_window)
                        t_window, step_window = now, step
                    rate = f" {per_step:.2f}s/step" if per_step is not None else ""
                    if align is not None:
                        rate += f" repa {aligns[-1]:.4f}"
                    rate += "".join(f" {k} {v:.4f}" for k, v in parts.items())
                    print(
                        f"  step {step}/{total} loss {value:.4f} |g| {grad_norm:.3f} "
                        f"sigma {sigma:.3f} lr {lr_sched.get_last_lr()[0]:.2e} "
                        f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
                        f"{rate}",
                        flush=True,
                    )
            if not losses:
                break
            mean = sum(losses) / len(losses)
            history.append(
                {"epoch": epoch + 1, "loss": mean, "seconds": time.time() - t_epoch}
                | ({"repa": sum(aligns) / len(aligns)} if aligns else {})
            )
            repa_mean = f" repa {sum(aligns) / len(aligns):.4f}" if aligns else ""
            print(
                f"epoch {epoch + 1}/{epochs}: loss {mean:.4f}{repa_mean} {time.time() - t_epoch:.0f}s "
                f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB",
                flush=True,
            )
            if (
                req.save_every_epochs
                and (epoch + 1) % req.save_every_epochs == 0
                and epoch + 1 < epochs
            ):
                saved = save(f"e{epoch + 1}", main_shadow)
                sink.ckpt(global_step=step, path=str(saved))
                print(f"  saved {saved}", flush=True)

        final = save(shadow=main_shadow)
        sink.ckpt(global_step=step, path=str(final))
        if any(shadows):
            print(f"  saved raw weights {save('raw')}", flush=True)
        for span, shadow in zip(spans, shadows):
            if shadow is not main_shadow and shadow:
                print(
                    f"  saved EMA {span:g} {save(f'ema{span:g}', shadow)}", flush=True
                )
        seconds = time.time() - t_start
        summary = {
            "output": str(final),
            "request": {k: getattr(req, k) for k in req.__dataclass_fields__},
            "fit": fit,
            "steps": step,
            "plain_steps": kinds["plain"],
            "variant_steps": kinds["variant"],
            "dropout_steps": kinds["empty"],
            "preserve_steps": kinds["preserve"],
            "minutes": seconds / 60,
            "s_per_step": seconds / max(1, step - 1),
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
            "history": history,
            "losses": losses_all,
        }
        report = out_path.with_suffix(".json")
        report.write_text(json.dumps(summary, indent=2))
        print(f"wrote {final} and {report} — {seconds / 60:.1f} min", flush=True)
    if attached is not None:
        attached.detach()
    return final
