#!/usr/bin/env python3
"""Real cross-attention probe queries for the set-level ``attn`` loss.

``attn_bank`` reads the K/V projections and the q/k RMSNorm gains straight out
of the DiT safetensors, but the *queries* cannot be read out of a checkpoint —
they are ``q_proj(image tokens)``, so they only exist during a forward. The
first G2 run substituted seeded random directions for them and the whole
readout space collapsed: a random query attends almost uniformly, so the
readout degenerates to a near-mean over the sequence. Measured consequence
(``_archive/cjk_aware_anima/reports/0816_phase2.md``): unrelated prompts sat at 0.374 and
*rose to 0.738 during training*, and two captions of the same image landed at
exactly 1.000 — the space was blind to wording, which is the one axis this line
exists to serve. That contaminated ``L_attn`` as an objective, not just as a
metric, and both G2 recovery numbers were withdrawn.

So: tap the real thing. Run a handful of DiT forwards on **cached** latents and
**cached** post-adapter contexts at 2-3 noise levels, hook each sampled block's
``cross_attn.q_norm`` (the queries as the attention actually sees them — cross
attention applies no RoPE, gated on ``is_selfattn`` in ``Attention.compute_qkv`` (``library/anima/models.py``), so the post-norm tensor is final),
and keep a pool of real token queries per block.

σ matters: the cross-attn input is adaLN-modulated by the timestep embedding,
so a query at σ=0.9 is not a query at σ=0.3. The pool spans the sampled σ set
and ``attn_bank`` draws its ``n_query`` probes from it.

Nothing here needs the training corpus, the Qwen encoder or the ext table —
only the DiT and the ordinary image caches:

    make daemon-run ARGS="scripts/distill_cjk/build_query_bank.py"

Output: ``bench/cjk_distill/assets/query_bank.safetensors`` (+ a ``.json``
sidecar with the provenance), read by ``attn_bank.build_bank``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from library.anima import weights as anima_weights  # noqa: E402

logger = logging.getLogger("build_query_bank")

DEFAULT_IMAGE_CACHE = REPO / "post_image_dataset" / "lora"
DEFAULT_OUT = REPO / "bench" / "cjk_distill" / "assets" / "query_bank.safetensors"
DEFAULT_SIGMAS = (0.3, 0.6, 0.9)
SEQ_TOTAL = 512
LATENT_KEY = re.compile(r"^latents_(\d+)x(\d+)$")


# --------------------------------------------------------------------------
# cached inputs
# --------------------------------------------------------------------------


def find_items(cache_root: Path, n: int, seed: int) -> list[tuple[Path, Path]]:
    """``(latents npz, te safetensors)`` pairs from the ordinary image caches.

    Caches are nested one directory per artist, hence the recursive glob. An
    image is usable only if both sidecars exist — TE caches are text-only and
    can lag a re-preprocess.
    """
    items: list[tuple[Path, Path]] = []
    for npz in sorted(cache_root.rglob("*_anima.npz")):
        # `{stem}_{WxH}_anima.npz` → `{stem}_anima_te.safetensors`
        stem = npz.name[: -len("_anima.npz")].rsplit("_", 1)[0]
        te = npz.parent / f"{stem}_anima_te.safetensors"
        if te.exists():
            items.append((npz, te))
    if not items:
        raise FileNotFoundError(
            f"no (latent, te) cache pairs under {cache_root} — run `make preprocess`"
        )
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n]


def load_latent(npz_path: Path) -> torch.Tensor:
    """Cached latent → ``(1, C, 1, h, w)``, the DiT's 5D layout (T at dim 2).

    The ``demoted_*`` keys (σ-demote siblings) are deliberately skipped: the
    probe wants the queries the model sees on a native-resolution forward.
    """
    data = np.load(npz_path)
    keys = [k for k in data.files if LATENT_KEY.match(k)]
    if not keys:
        raise KeyError(f"{npz_path}: no latents_* key (found {list(data.files)})")
    key = max(keys, key=lambda k: int(k.split("_")[1].split("x")[0]))
    lat = torch.from_numpy(np.asarray(data[key]))
    if lat.dim() == 4:  # (C,1,h,w) — cached with the temporal singleton
        lat = lat.squeeze(1)
    assert lat.dim() == 3, f"{npz_path}: unexpected latent shape {tuple(lat.shape)}"
    return lat[None, :, None]  # (1,C,1,h,w)


def load_context(te_path: Path) -> torch.Tensor:
    """Cached post-adapter ``crossattn_emb`` → ``(1, 512, D)``, pads zeroed.

    The padding invariant (CLAUDE.md): the context is max-padded to 512 and the
    pad rows are *zeroed*, never masked out. Multi-variant caches expose
    ``_v0`` (the pristine caption) — pin to it so the bank is deterministic.
    """
    from safetensors import safe_open

    with safe_open(str(te_path), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        suffix = "" if "crossattn_emb" in keys else "_v0"
        ce_key, mask_key = f"crossattn_emb{suffix}", f"t5_attn_mask{suffix}"
        if ce_key not in keys:
            raise KeyError(f"{te_path}: no {ce_key} (found {sorted(keys)})")
        ce = f.get_tensor(ce_key)
        mask = f.get_tensor(mask_key) if mask_key in keys else None
    ce = ce.unsqueeze(0).float()
    if ce.shape[1] < SEQ_TOTAL:
        ce = torch.nn.functional.pad(ce, (0, 0, 0, SEQ_TOTAL - ce.shape[1]))
    if mask is not None:
        ce[~mask.unsqueeze(0).bool()] = 0
    return ce


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------


def resolve_qnorm(dit, block: int):
    """The **DiT's** ``blocks.<i>.cross_attn.q_norm``, wherever it is nested.

    ``load_anima_model`` may or may not keep the checkpoint's ``net.`` prefix as
    a real submodule, so a suffix match is needed — but a bare suffix match is a
    trap: ``LLMAdapter`` has its own ``blocks.<i>.cross_attn.q_norm`` stack and
    for low indices it matches first. Those are the wrong queries entirely (the
    adapter attends text→text, and here it does not even run — the context comes
    pre-adapted out of the TE cache). So: exact name wins, and any candidate
    under ``llm_adapter`` is excluded outright.
    """
    want = f"blocks.{block}.cross_attn.q_norm"
    candidates = [
        (name, mod)
        for name, mod in dit.named_modules()
        if (name == want or name.endswith("." + want))
        and "llm_adapter" not in name.split(".")
    ]
    for name, mod in candidates:
        if name == want:
            return name, mod
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise KeyError(f"block {block}: no DiT-side {want} in the loaded model")
    raise KeyError(f"block {block}: ambiguous {want} — {[n for n, _ in candidates]}")


@torch.no_grad()
def collect(
    dit,
    items: list[tuple[Path, Path]],
    blocks: tuple[int, ...],
    sigmas: tuple[float, ...],
    *,
    per_forward: int,
    device,
    dtype,
    seed: int,
) -> dict[int, torch.Tensor]:
    """``{block: [pool, heads, head_dim]}`` of real, q_norm'ed token queries."""
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for b in blocks:
        name, mod = resolve_qnorm(dit, b)
        logger.info("hooking block %d at %s", b, name)

        def make_hook(block_id: int):
            def hook(_mod, _inp, out):
                captured[block_id] = out.detach()

            return hook

        handles.append(mod.register_forward_hook(make_hook(b)))

    pool: dict[int, list[torch.Tensor]] = {b: [] for b in blocks}
    g = torch.Generator(device="cpu").manual_seed(seed)
    if hasattr(dit, "prepare_block_swap_before_forward"):
        dit.prepare_block_swap_before_forward()
    try:
        for i, (npz, te) in enumerate(items):
            z0 = load_latent(npz).to(device, dtype)
            emb = load_context(te).to(device, dtype)
            h, w = z0.shape[-2], z0.shape[-1]
            pad = torch.zeros(1, 1, h, w, dtype=dtype, device=device)
            eps = torch.randn(z0.shape, generator=g).to(device, dtype)
            for s in sigmas:
                z_t = ((1.0 - s) * z0.float() + s * eps.float()).to(dtype)
                t_b = torch.full((1,), float(s), device=device, dtype=dtype)
                captured.clear()
                dit(z_t, t_b, emb, padding_mask=pad)
                if set(captured) != set(blocks):
                    raise RuntimeError(
                        f"hooks fired for {sorted(captured)}, expected {list(blocks)} "
                        "— the cross-attn path did not run for every sampled block"
                    )
                for b in blocks:
                    q = captured[b].float()
                    # (..., heads, head_dim) over image tokens, whatever the
                    # caller's leading layout (5D grid or flattened) happens to be.
                    q = q.reshape(-1, q.shape[-2], q.shape[-1])
                    idx = torch.randperm(q.shape[0], generator=g)[:per_forward]
                    pool[b].append(q[idx.to(q.device)].cpu())
            del z0, emb, eps, pad
            if (i + 1) % 8 == 0:
                logger.info("  %d/%d images", i + 1, len(items))
    finally:
        for hd in handles:
            hd.remove()
    return {b: torch.cat(v, dim=0) for b, v in pool.items()}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dit", default=None, help="DiT safetensors (default: canonical)")
    ap.add_argument("--image_cache", type=Path, default=DEFAULT_IMAGE_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--blocks",
        default="0,13,27",
        help="DiT blocks to tap — must match --attn_blocks of the runs that read this",
    )
    ap.add_argument("--sigmas", type=float, nargs="+", default=list(DEFAULT_SIGMAS))
    ap.add_argument("--n_images", type=int, default=32)
    ap.add_argument(
        "--per_forward", type=int, default=8, help="tokens kept per forward"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn_mode", default="torch")
    args = ap.parse_args()

    from anima_lora import default_checkpoints

    dit_path = args.dit or default_checkpoints().dit
    blocks = tuple(int(b) for b in str(args.blocks).split(",") if b.strip())
    sigmas = tuple(float(s) for s in args.sigmas)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    items = find_items(args.image_cache, args.n_images, args.seed)
    logger.info(
        "%d images x %d sigmas x %d tokens = %d queries/block from %s",
        len(items),
        len(sigmas),
        args.per_forward,
        len(items) * len(sigmas) * args.per_forward,
        dit_path,
    )

    dit = anima_weights.load_anima_model(
        device=device,
        dit_path=str(dit_path),
        attn_mode=args.attn_mode,
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    # Same step-2 the shared harness does (``runtime/harness.py::build_anima``).
    # The explicit ``.to()`` is load-bearing: ``load_anima_model``'s device
    # argument does not reach the runtime buffers, so the mod-guidance schedule
    # stays on CPU and ``_run_blocks`` adds it to a cuda timestep embedding.
    dit.to(device, dtype=dtype).requires_grad_(False)
    dit.reset_mod_guidance()
    dit.eval()

    pool = collect(
        dit,
        items,
        blocks,
        sigmas,
        per_forward=args.per_forward,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )

    from safetensors.torch import save_file

    tensors = {f"queries_block{b}": pool[b].contiguous() for b in blocks}
    meta = {
        "source": "real cross_attn q_norm outputs (DiT forwards on cached latents)",
        "dit": str(dit_path),
        "blocks": ",".join(str(b) for b in blocks),
        "sigmas": ",".join(f"{s:g}" for s in sigmas),
        "n_images": str(len(items)),
        "per_forward": str(args.per_forward),
        "seed": str(args.seed),
        "pool_per_block": str(next(iter(pool.values())).shape[0]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out), metadata=meta)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                **meta,
                "shapes": {k: list(v.shape) for k, v in tensors.items()},
                "images": [str(p.relative_to(REPO)) for p, _ in items],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for k, v in tensors.items():
        logger.info(
            "%s: %s  (rms %.4f)", k, tuple(v.shape), float(v.pow(2).mean() ** 0.5)
        )
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
