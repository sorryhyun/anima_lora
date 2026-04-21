"""Embedding inversion: find the optimal text embedding for a target image.

Given a target image (or a directory of preprocessed images), optimizes a text
embedding (crossattn_emb) in the post-T5, pre-DiT space to minimize flow-matching
loss through the frozen DiT. This reveals "how the DiT interprets the image" in
embedding space.

Usage:
    # Single image (encodes via VAE on the fly)
    python scripts/invert_embedding.py \
        --image path/to/image.png \
        --dit models/diffusion_models/anima-preview3-base.safetensors \
        --vae models/vae/qwen_image_vae.safetensors \
        --save_path output/inverted.safetensors

    # Batch from post_image_dataset/ (uses cached latents + cached TE for init)
    python scripts/invert_embedding.py \
        --image_dir post_image_dataset \
        --dit models/diffusion_models/anima-preview3-base.safetensors \
        --save_path output/inversions/ \
        --num_images 10 --steps 500
"""

import argparse
import csv
import json
import logging
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from library import strategy_base
from library.anima import (
    models as anima_models,
    weights as anima_utils,
    strategy as strategy_anima,
)
from library.models import qwen_vae as qwen_image_autoencoder_kl
from library.io.cache import (
    discover_cached_images,
    load_cached_crossattn_emb,
    load_cached_latents,
)
from library.datasets.image_utils import IMAGE_TRANSFORMS
from library.runtime.device import clean_memory_on_device
from library.log import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Embedding inversion for Anima DiT")

    # Model paths
    p.add_argument("--dit", type=str, required=True, help="DiT checkpoint path")
    p.add_argument(
        "--vae",
        type=str,
        default=None,
        help="VAE checkpoint path (only needed for --image mode)",
    )
    p.add_argument(
        "--text_encoder",
        type=str,
        default=None,
        help="Text encoder path (for --init_prompt)",
    )
    p.add_argument("--attn_mode", type=str, default="flash", help="Attention backend")

    # Target — single image or directory
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--image", type=str, default=None, help="Single target image path"
    )
    target.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory with preprocessed images (cached latents + TE outputs)",
    )
    p.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=None,
        help="Resize to H W (--image mode only)",
    )
    p.add_argument(
        "--num_images",
        type=int,
        default=None,
        help="Process N random images from --image_dir (default: all)",
    )
    p.add_argument(
        "--shuffle", action="store_true", help="Shuffle image order in --image_dir mode"
    )

    # Initialization
    p.add_argument(
        "--init_prompt",
        type=str,
        default=None,
        help="Initialize embedding from this prompt (requires --text_encoder)",
    )
    p.add_argument(
        "--init_embedding",
        type=str,
        default=None,
        help="Initialize from a saved embedding .safetensors",
    )
    p.add_argument(
        "--init_from_cache",
        action="store_true",
        default=True,
        help="Initialize from cached crossattn_emb in _anima_te.safetensors (default: True, --image_dir mode)",
    )
    p.add_argument(
        "--init_zeros",
        action="store_true",
        help="Initialize from zeros instead of cached embedding",
    )
    p.add_argument(
        "--init_jitter_std",
        type=float,
        default=0.15,
        help="Gaussian jitter std added to the init embedding. Required when "
        "--active_length<512: without jitter, zero-valued rows in the first "
        "active_length slots stay bit-identical during optimization (cross-attn "
        "has no positional asymmetry on the text side), collapsing the solution "
        "to a single broadcast vector. Default 0.15 matches the measured "
        "per-element std of active (non-padded) rows in cached crossattn_emb "
        "(mean≈0, std=0.149 over 31k rows from 200 random samples). Set to 0 "
        "to disable.",
    )

    # Optimization
    p.add_argument(
        "--steps", type=int, default=100, help="Optimization steps per image"
    )
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    p.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["cosine", "constant"],
        help="LR schedule",
    )
    p.add_argument(
        "--timesteps_per_step",
        type=int,
        default=1,
        help="Random timesteps sampled per optimization step (batched forward)",
    )
    p.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (total timesteps per update = timesteps_per_step × grad_accum)",
    )
    p.add_argument(
        "--sigma_sampling",
        type=str,
        default="uniform",
        choices=["uniform", "sigmoid"],
        help="Sigma sampling strategy",
    )
    p.add_argument(
        "--sigmoid_scale",
        type=float,
        default=1.0,
        help="Scale for sigmoid sigma sampling",
    )
    p.add_argument(
        "--active_length",
        type=int,
        default=128,
        help="Number of leading token positions that are trainable; the rest of "
        "the 512-length sequence is hard-zero every step. Matches the on-manifold "
        "layout the DiT was trained on (kv_proj has bias=False, so pad→K=V=0). "
        "Set to 512 to disable (old unconstrained behavior).",
    )

    # Multi-run token-aligned aggregation
    p.add_argument(
        "--aggregate_by",
        type=int,
        default=1,
        help="Run N independent inversions per image and aggregate via Hungarian token alignment + mean (default: 1 = single run)",
    )
    p.add_argument(
        "--aggregate_seed_base",
        type=int,
        default=42,
        help="Base RNG seed for multi-run aggregation (each run uses base + k*1000)",
    )
    p.add_argument(
        "--save_per_run",
        action="store_true",
        help="Also save each individual run's embedding alongside the aligned mean",
    )
    p.add_argument(
        "--probe_functional",
        action="store_true",
        help="After aggregation, probe per-run embeddings in functional space by forwarding the DiT and capturing cross-attention output. Writes pairwise cosines to the alignment json.",
    )
    p.add_argument(
        "--probe_samples",
        type=int,
        default=4,
        help="Number of (noise, sigma) probes used by --probe_functional",
    )
    p.add_argument(
        "--probe_blocks",
        type=str,
        default="0",
        help="Comma-separated block indices to capture for --probe_functional (e.g. '0,7,14,21,27')",
    )

    # Output
    p.add_argument(
        "--output_dir",
        type=str,
        default="inversions",
        help="Output directory (results/ and logs/ created inside)",
    )
    p.add_argument("--log_every", type=int, default=10, help="Log loss every N steps")
    p.add_argument(
        "--log_block_grads",
        action="store_true",
        help="Log per-block gradient norms (for analyzing block sensitivity to embedding)",
    )

    # Verification
    p.add_argument(
        "--verify",
        action="store_true",
        help="Generate an image from the inverted embedding after optimization",
    )
    p.add_argument(
        "--verify_steps", type=int, default=50, help="Inference steps for verification"
    )
    p.add_argument(
        "--verify_seed", type=int, default=42, help="Seed for verification generation"
    )
    p.add_argument(
        "--flow_shift",
        type=float,
        default=5.0,
        help="Flow shift for verification sampling",
    )

    # Device / VRAM
    p.add_argument(
        "--device", type=str, default=None, help="Device (default: cuda if available)"
    )
    p.add_argument(
        "--vae_chunk_size", type=int, default=64, help="VAE spatial chunk size"
    )
    p.add_argument(
        "--blocks_to_swap",
        type=int,
        default=0,
        help="Number of transformer blocks to swap to CPU (0 = use gradient checkpointing instead)",
    )

    args = p.parse_args()

    if args.image is not None and args.vae is None:
        p.error("--vae is required when using --image mode")

    # Setup output subdirectories
    args.results_dir = os.path.join(args.output_dir, "results")
    args.logs_dir = os.path.join(args.output_dir, "logs")

    return args


# region Data loading


def load_and_encode_image(args, device):
    """Load a single target image, encode to latents via VAE, then free VAE."""
    logger.info(f"Loading target image: {args.image}")
    img = Image.open(args.image).convert("RGB")

    if args.image_size is not None:
        h, w = args.image_size
    else:
        w, h = img.size
        h = (h // 32) * 32
        w = (w // 32) * 32

    if img.size != (w, h):
        img = img.resize((w, h), Image.LANCZOS)
        logger.info(f"Resized to {w}x{h}")

    logger.info("Loading VAE...")
    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=args.vae_chunk_size,
    )
    vae.to(device, dtype=torch.bfloat16)
    vae.eval()

    img_tensor = IMAGE_TRANSFORMS(img).unsqueeze(0).to(device)
    with torch.no_grad():
        latents = vae.encode_pixels_to_latents(img_tensor)
    latents = latents.to(torch.bfloat16)
    logger.info(f"Encoded latents shape: {latents.shape}")

    del vae
    clean_memory_on_device(device)

    return latents, h, w


# endregion


# region Embedding initialization


def create_initial_embedding(args, device, anima, te_path=None):
    """Create the initial embedding tensor to optimize.

    Priority: --init_embedding > --init_prompt > cached TE > zeros. A Gaussian
    jitter (--init_jitter_std) is added last to break the row-identical symmetry
    that would otherwise collapse --active_length<S runs to a rank-1 broadcast
    (cross-attention has no positional asymmetry on the text side, so identical
    rows receive identical gradients and stay tied forever).
    """
    embed = None

    if args.init_embedding is not None:
        logger.info(f"Init from saved embedding: {args.init_embedding}")
        sd = load_file(args.init_embedding)
        embed = sd["crossattn_emb"].to(device, dtype=torch.float32)
        if embed.ndim == 2:
            embed = embed.unsqueeze(0)
    elif args.init_prompt is not None and args.text_encoder is not None:
        logger.info(f"Init from prompt: {args.init_prompt}")
        embed = _encode_prompt(args, device, anima)
    elif not args.init_zeros and te_path is not None:
        cached = load_cached_crossattn_emb(te_path)
        if cached is not None:
            logger.info(f"Init from cached TE: {te_path}")
            embed = cached.unsqueeze(0).to(device, dtype=torch.float32)

    if embed is None:
        logger.info("Init from zeros")
        embed = torch.zeros(1, 512, 1024, dtype=torch.float32, device=device)

    if args.init_jitter_std > 0:
        noise = torch.randn_like(embed) * args.init_jitter_std
        embed = embed + noise
        logger.info(f"Added jitter: std={args.init_jitter_std}")

    return embed


def _encode_prompt(args, device, anima):
    """Encode a text prompt to crossattn_emb via text encoder + LLM adapter."""
    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_path=args.text_encoder,
        t5_tokenizer_path=None,
        qwen3_max_length=512,
        t5_max_length=512,
    )
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
    encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
    strategy_base.TextEncodingStrategy.set_strategy(encoding_strategy)

    text_encoder, _ = anima_utils.load_qwen3_text_encoder(
        args.text_encoder,
        dtype=torch.bfloat16,
        device=device,
    )
    text_encoder.eval()

    with torch.no_grad():
        tokens = tokenize_strategy.tokenize(args.init_prompt)
        embed = encoding_strategy.encode_tokens(
            tokenize_strategy, [text_encoder], tokens
        )
        crossattn_emb, _ = anima._preprocess_text_embeds(
            source_hidden_states=embed[0].to(device),
            target_input_ids=embed[2].to(device),
            target_attention_mask=embed[3].to(device),
            source_attention_mask=embed[1].to(device),
        )
        crossattn_emb[~embed[3].bool()] = 0
        if crossattn_emb.shape[1] < 512:
            crossattn_emb = F.pad(
                crossattn_emb, (0, 0, 0, 512 - crossattn_emb.shape[1])
            )

    del text_encoder
    clean_memory_on_device(device)

    return crossattn_emb.to(torch.float32)


# endregion


# region Optimization core


def sample_sigmas(args, batch_size, device):
    """Sample noise levels for the optimization step."""
    if args.sigma_sampling == "sigmoid":
        sigmas = torch.sigmoid(
            args.sigmoid_scale * torch.randn(batch_size, device=device)
        )
    else:
        sigmas = torch.rand(batch_size, device=device)
    return sigmas


def inversion_step(anima, latents, embed_bf16, sigmas, padding_mask):
    """One optimization step: compute flow-matching loss."""
    n_t = sigmas.shape[0]

    lat = latents.expand(n_t, -1, -1, -1)
    noise = torch.randn_like(lat)

    sigmas_view = sigmas.view(-1, 1, 1, 1)
    noisy = (1.0 - sigmas_view) * lat + sigmas_view * noise
    noisy_5d = noisy.to(torch.bfloat16).unsqueeze(2)

    timesteps = sigmas.to(torch.bfloat16)
    emb = embed_bf16.expand(n_t, -1, -1)
    pm = padding_mask.expand(n_t, -1, -1, -1)

    pred = anima(noisy_5d, timesteps, emb, padding_mask=pm)
    pred = pred.squeeze(2)

    target = noise - lat
    return F.mse_loss(pred.float(), target.float())


def _install_block_grad_hooks(anima):
    """Register hooks to capture gradient norms at each block's cross-attention output.

    Hooks the output_proj of each block's cross_attn module.
    Returns (handle_list, grad_norms_dict) where grad_norms_dict maps
    block index -> list of gradient norms (one per backward call).
    """
    handles = []
    grad_norms = {}
    num_blocks = len(anima.blocks)  # type: ignore[arg-type]

    for block_idx in range(num_blocks):
        grad_norms[block_idx] = []
        block = anima.blocks[block_idx]  # type: ignore[index]

        target_module = getattr(getattr(block, "cross_attn", None), "output_proj", None)
        if target_module is None:
            continue

        def make_hook(idx):
            def hook_fn(_module, _grad_input, grad_output):
                if grad_output[0] is not None:
                    grad_norms[idx].append(grad_output[0].norm().item())

            return hook_fn

        h = target_module.register_full_backward_hook(make_hook(block_idx))
        handles.append(h)

    return handles, grad_norms


def optimize_embedding(
    args, anima, latents, init_embed, device, log_path=None, seed=None
):
    """Run the optimization loop for a single image. Returns (best_embed, best_loss)."""
    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)

    S = init_embed.shape[1]
    L = args.active_length
    if L <= 0 or L > S:
        raise ValueError(f"--active_length must be in [1, {S}], got {L}")

    active = torch.nn.Parameter(init_embed[:, :L, :].clone())
    if L < S:
        pad_tail = torch.zeros(
            1, S - L, init_embed.shape[-1], dtype=torch.float32, device=device
        )
    else:
        pad_tail = None

    optimizer = torch.optim.AdamW([active], lr=args.lr, weight_decay=0.0)

    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.steps, eta_min=args.lr * 0.01
        )
    else:
        scheduler = None

    h_lat, w_lat = latents.shape[-2], latents.shape[-1]
    padding_mask = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)

    # Per-block gradient hooks
    block_grad_handles, block_grad_norms = [], {}
    if args.log_block_grads:
        block_grad_handles, block_grad_norms = _install_block_grad_hooks(anima)

    # CSV log
    csv_file = None
    csv_writer = None
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        csv_file = open(log_path, "w", newline="")
        fieldnames = ["step", "loss", "best_loss", "lr", "grad_norm"]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    best_loss = float("inf")
    best_embed = None
    grad_accum = args.grad_accum

    pbar = tqdm(range(args.steps), desc="Inverting", leave=False)
    for step in pbar:
        optimizer.zero_grad()

        accum_loss = 0.0
        for _ in range(grad_accum):
            sigmas = sample_sigmas(args, args.timesteps_per_step, device)
            if pad_tail is not None:
                embed_full = torch.cat([active, pad_tail], dim=1)
            else:
                embed_full = active
            embed_bf16 = embed_full.to(torch.bfloat16)

            loss = inversion_step(anima, latents, embed_bf16, sigmas, padding_mask)
            (loss / grad_accum).backward()
            accum_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_([active], max_norm=1.0).item()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_val = accum_loss / grad_accum
        if loss_val < best_loss:
            best_loss = loss_val
            with torch.no_grad():
                if pad_tail is not None:
                    best_embed = torch.cat([active.detach(), pad_tail], dim=1).clone()
                else:
                    best_embed = active.detach().clone()

        if step % args.log_every == 0 or step == args.steps - 1:
            lr_now = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{loss_val:.6f}", best=f"{best_loss:.6f}", lr=f"{lr_now:.2e}"
            )

        # Write every step to CSV
        if csv_writer is not None:
            csv_writer.writerow(
                {
                    "step": step,
                    "loss": f"{loss_val:.6f}",
                    "best_loss": f"{best_loss:.6f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "grad_norm": f"{grad_norm:.6f}",
                }
            )

    # Finalize logging
    if csv_file is not None:
        csv_file.close()

    # Save per-block gradient summary
    if args.log_block_grads and log_path is not None:
        block_summary = {}
        for idx, norms in block_grad_norms.items():
            if norms:
                block_summary[f"block_{idx:02d}"] = {
                    "mean": float(np.mean(norms)),
                    "std": float(np.std(norms)),
                    "max": float(np.max(norms)),
                    "min": float(np.min(norms)),
                }
        block_log_path = log_path.replace(".csv", "_block_grads.json")
        with open(block_log_path, "w") as f:
            json.dump(block_summary, f, indent=2)
        logger.info(f"  Block gradient summary: {block_log_path}")

    # Cleanup hooks
    for h in block_grad_handles:
        h.remove()

    return best_embed, best_loss


# endregion


# region Token-aligned aggregation


def _per_token_cosine(
    a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Row-wise cosine similarity between two (S, D) tensors. Returns (S,)."""
    a = a.float()
    b = b.float()
    num = (a * b).sum(dim=-1)
    denom = a.norm(dim=-1).clamp_min(eps) * b.norm(dim=-1).clamp_min(eps)
    return num / denom


def _pairwise_token_cosine(
    a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """All-pairs cosine similarity between rows of two (S, D) tensors. Returns (S, S)."""
    a = a.float()
    b = b.float()
    a_norm = a / a.norm(dim=-1, keepdim=True).clamp_min(eps)
    b_norm = b / b.norm(dim=-1, keepdim=True).clamp_min(eps)
    return a_norm @ b_norm.T


def align_and_aggregate(embeddings, ref_index=0):
    """Token-align a list of (1, S, D) embeddings via Hungarian matching, then mean.

    For each non-reference embedding, solves a 1:1 token-to-token assignment that
    maximizes per-row cosine similarity to the reference. Permutes rows to match,
    then averages across the aligned stack.

    Returns:
        aligned_mean: (1, S, D) float32
        diagnostics: dict with per-token cosine (before/after) and per-run permutations
    """
    assert len(embeddings) >= 1
    if len(embeddings) == 1:
        return embeddings[0].float().clone(), {
            "per_token_cos_before": 1.0,
            "per_token_cos_after": 1.0,
            "permutations": [list(range(embeddings[0].shape[-2]))],
        }

    stacked = [e.squeeze(0).float().cpu() for e in embeddings]  # list of (S, D)
    ref = stacked[ref_index]
    s = ref.shape[0]

    aligned = [None] * len(stacked)
    aligned[ref_index] = ref
    permutations = [None] * len(stacked)
    permutations[ref_index] = list(range(s))

    # Per-token cosine BEFORE alignment (mean over all non-ref runs, averaged over tokens)
    pre_sims = []
    for k, e in enumerate(stacked):
        if k == ref_index:
            continue
        pre_sims.append(_per_token_cosine(ref, e).mean().item())

    for k, e in enumerate(stacked):
        if k == ref_index:
            continue
        # Cost = negative cosine sim → Hungarian minimizes
        sim = _pairwise_token_cosine(ref, e).numpy()  # (S, S)
        row_ind, col_ind = linear_sum_assignment(-sim)
        # row_ind is 0..S-1 when the matrix is square; col_ind gives the
        # permutation that sends e's rows into alignment with ref's rows.
        perm = col_ind.tolist()
        permutations[k] = perm
        aligned[k] = e[col_ind]

    aligned_stack = torch.stack(aligned, dim=0)  # (N, S, D)
    aligned_mean = aligned_stack.mean(dim=0).unsqueeze(0)  # (1, S, D)

    # Per-token cosine AFTER alignment (aligned runs vs ref)
    post_sims = []
    for k in range(len(aligned)):
        if k == ref_index:
            continue
        post_sims.append(_per_token_cosine(ref, aligned[k]).mean().item())

    diagnostics = {
        "per_token_cos_before": float(np.mean(pre_sims)),
        "per_token_cos_after": float(np.mean(post_sims)),
        "per_token_cos_before_list": [float(x) for x in pre_sims],
        "per_token_cos_after_list": [float(x) for x in post_sims],
        "permutations": permutations,
        "ref_index": ref_index,
    }
    return aligned_mean, diagnostics


def probe_functional_space(
    args, anima, latents, labeled_embeddings, device, n_probes=4, block_idxs=(0,)
):
    """Forward each embedding through DiT at a fixed (noise, sigma) probe bank,
    capture cross_attn.output_proj output at each block in `block_idxs`, and
    return pairwise cosines per block.

    All hooks are installed before forwarding so a single pass populates every
    requested block's outputs.

    Args:
        labeled_embeddings: list of (label, embedding) tuples; embeddings are (1, S, D) float32
        block_idxs: iterable of block indices to probe. Scalar accepted for backwards compat.
    Returns:
        dict with 'per_block' (keyed by str(block_idx)) + top-level summary at block_idxs[0].
    """
    if isinstance(block_idxs, int):
        block_idxs = [block_idxs]
    block_idxs = list(block_idxs)

    # Resolve target modules for each requested block
    block_modules = {}
    for bi in block_idxs:
        m = getattr(getattr(anima.blocks[bi], "cross_attn", None), "output_proj", None)
        if m is None:
            logger.warning(
                f"  probe: block {bi} has no cross_attn.output_proj; skipping"
            )
            continue
        block_modules[bi] = m
    if not block_modules:
        return None

    # Per-block capture buffer — mutated by hooks during each forward
    captured = {bi: None for bi in block_modules}

    def make_hook(bi):
        def hook_fn(_module, _input, output):
            captured[bi] = output.detach().float().cpu()

        return hook_fn

    # Reproducible probe bank
    g = torch.Generator(device=device).manual_seed(20260412)
    sigmas = torch.rand(n_probes, generator=g, device=device)
    noises = [
        torch.randn(latents.shape, generator=g, device=device, dtype=torch.float32).to(
            torch.bfloat16
        )
        for _ in range(n_probes)
    ]

    h_lat, w_lat = latents.shape[-2], latents.shape[-1]
    padding_mask = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)

    if hasattr(anima, "prepare_block_swap_before_forward"):
        anima.prepare_block_swap_before_forward()

    # For each embedding, run all probes and stash per-block outputs
    # per_embed_outs[k][bi] = (T, D)
    per_embed_outs = []
    for label, emb in labeled_embeddings:
        emb_bf16 = emb.to(device=device, dtype=torch.bfloat16)
        if emb_bf16.ndim == 2:
            emb_bf16 = emb_bf16.unsqueeze(0)

        per_block_probe_tensors = {bi: [] for bi in block_modules}
        for i in range(n_probes):
            sigma = sigmas[i : i + 1]
            noise = noises[i]
            sv = sigma.view(-1, 1, 1, 1).to(torch.bfloat16)
            noisy = ((1.0 - sv) * latents + sv * noise).to(torch.bfloat16).unsqueeze(2)
            t = sigma.to(torch.bfloat16)

            handles = [
                block_modules[bi].register_forward_hook(make_hook(bi))
                for bi in block_modules
            ]
            try:
                with torch.no_grad():
                    anima(noisy, t, emb_bf16, padding_mask=padding_mask)
            finally:
                for h in handles:
                    h.remove()

            for bi in block_modules:
                out = captured[bi]
                if out is None:
                    continue
                per_block_probe_tensors[bi].append(out.reshape(-1, out.shape[-1]))
                captured[bi] = None

        per_embed_outs.append(
            {
                bi: (torch.cat(ts, dim=0) if ts else None)
                for bi, ts in per_block_probe_tensors.items()
            }
        )

    labels = [lbl for lbl, _ in labeled_embeddings]
    n = len(labels)

    def _cos_flat(a, b):
        return F.cosine_similarity(
            a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
        ).item()

    def _cos_per_token(a, b, eps=1e-8):
        num = (a * b).sum(dim=-1)
        denom = a.norm(dim=-1).clamp_min(eps) * b.norm(dim=-1).clamp_min(eps)
        return (num / denom).mean().item()

    # Raw-embedding cosines (block-independent) — computed once.
    # Callers may pass embeddings from mixed devices (e.g. CPU init_embed + CUDA
    # optimized best_embed), so pin both sides to CPU before the cosine.
    raw_pairwise_flat = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = f"{labels[i]}__{labels[j]}"
            a = labeled_embeddings[i][1].detach().float().cpu().flatten().unsqueeze(0)
            b = labeled_embeddings[j][1].detach().float().cpu().flatten().unsqueeze(0)
            raw_pairwise_flat[key] = F.cosine_similarity(a, b).item()

    per_block = {}
    for bi in block_modules:
        pairwise_flat = {}
        pairwise_per_token = {}
        for i in range(n):
            for j in range(i + 1, n):
                a = per_embed_outs[i][bi]
                b = per_embed_outs[j][bi]
                if a is None or b is None:
                    continue
                key = f"{labels[i]}__{labels[j]}"
                pairwise_flat[key] = _cos_flat(a, b)
                pairwise_per_token[key] = _cos_per_token(a, b)

        flat_vals = list(pairwise_flat.values())
        per_tok_vals = list(pairwise_per_token.values())
        per_block[str(bi)] = {
            "block_idx": bi,
            "pairwise_cos_flat": pairwise_flat,
            "pairwise_cos_per_token": pairwise_per_token,
            "summary": {
                "mean_functional_flat_cos": float(np.mean(flat_vals))
                if flat_vals
                else None,
                "mean_functional_per_token_cos": float(np.mean(per_tok_vals))
                if per_tok_vals
                else None,
            },
        }
        logger.info(
            f"  functional probe block {bi:02d}: "
            f"flat {float(np.mean(flat_vals)):.4f} per-tok {float(np.mean(per_tok_vals)):.4f}"
        )

    raw_mean = float(np.mean(list(raw_pairwise_flat.values())))
    logger.info(f"  raw flat cos (reference): {raw_mean:.4f}")

    return {
        "n_probes": n_probes,
        "block_idxs": list(block_modules.keys()),
        "labels": labels,
        "raw_pairwise_cos_flat": raw_pairwise_flat,
        "raw_summary": {"mean_raw_flat_cos": raw_mean},
        "per_block": per_block,
    }


def run_aggregated_inversion(args, anima, latents, init_embed, device, log_path=None):
    """Run N inversions with different seeds, align tokens, return aggregated mean.

    Returns:
        best_embed: aligned-mean embedding (1, S, D) float32
        best_loss: mean of per-run best losses
        per_run: list of dicts with 'embed', 'loss' for each run
        diagnostics: alignment diagnostics dict
    """
    n = max(1, args.aggregate_by)
    per_run = []

    for k in range(n):
        seed = args.aggregate_seed_base + k * 1000
        if n > 1:
            logger.info(f"  aggregate run {k + 1}/{n} (seed={seed})")
        run_log_path = None
        if log_path is not None:
            if n > 1:
                run_log_path = log_path.replace(".csv", f"_run{k}.csv")
            else:
                run_log_path = log_path
        embed, loss = optimize_embedding(
            args, anima, latents, init_embed, device, log_path=run_log_path, seed=seed
        )
        per_run.append({"embed": embed, "loss": loss})
        if n > 1:
            logger.info(f"    run {k} best_loss={loss:.6f}")

    if n == 1:
        return per_run[0]["embed"], per_run[0]["loss"], per_run, None

    aligned_mean, diagnostics = align_and_aggregate([r["embed"] for r in per_run])
    mean_loss = float(np.mean([r["loss"] for r in per_run]))

    logger.info(
        f"  alignment: per-token cos {diagnostics['per_token_cos_before']:.4f} "
        f"→ {diagnostics['per_token_cos_after']:.4f}"
    )

    if args.probe_functional:
        labeled = [(f"run{k}", per_run[k]["embed"]) for k in range(n)]
        labeled.append(("mean", aligned_mean))
        probe = probe_functional_space(
            args,
            anima,
            latents,
            labeled,
            device,
            n_probes=args.probe_samples,
            block_idxs=[int(s) for s in args.probe_blocks.split(",") if s.strip()],
        )
        if probe is not None:
            diagnostics["functional_probe"] = probe

    return aligned_mean, mean_loss, per_run, diagnostics


# endregion


# region Verification


def verify_embedding(args, anima, embed, h, w, device, save_path):
    """Generate an image from the inverted embedding to verify quality."""
    from library.inference import sampling as inference_utils

    logger.info("Generating verification image...")

    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=args.vae_chunk_size,
    )
    vae.to(device, dtype=torch.bfloat16)
    vae.eval()

    embed_bf16 = embed.to(device=device, dtype=torch.bfloat16)
    if embed_bf16.ndim == 2:
        embed_bf16 = embed_bf16.unsqueeze(0)

    h_lat, w_lat = h // 8, w // 8
    padding_mask = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)

    gen = torch.Generator(device=device).manual_seed(args.verify_seed)
    latents = torch.randn(
        1,
        anima_models.Anima.LATENT_CHANNELS,
        1,
        h_lat,
        w_lat,
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )

    timesteps, sigmas = inference_utils.get_timesteps_sigmas(
        args.verify_steps, args.flow_shift, device
    )
    timesteps = (timesteps / 1000).to(device, dtype=torch.bfloat16)

    if hasattr(anima, "switch_block_swap_for_inference"):
        anima.switch_block_swap_for_inference()

    with torch.no_grad():
        for step_i, t in enumerate(tqdm(timesteps, desc="Denoising", leave=False)):
            if hasattr(anima, "prepare_block_swap_before_forward"):
                anima.prepare_block_swap_before_forward()
            t_expand = t.unsqueeze(0)
            noise_pred = anima(latents, t_expand, embed_bf16, padding_mask=padding_mask)
            latents = inference_utils.step(latents, noise_pred, sigmas, step_i).to(
                torch.bfloat16
            )

    with torch.no_grad():
        pixels = vae.decode_to_pixels(latents.squeeze(2))
    pixels = (
        ((pixels + 1.0) / 2.0)
        .clamp(0, 1)
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .float()
        .numpy()
    )
    pixels = (pixels * 255).clip(0, 255).astype("uint8")
    Image.fromarray(pixels).save(save_path)
    logger.info(f"Saved verification image: {save_path}")

    del vae
    clean_memory_on_device(device)


# endregion


# region Main


def _save_aggregation_artifacts(args, stem, per_run, diagnostics):
    """Persist per-run embeddings and alignment diagnostics when --aggregate_by > 1."""
    if diagnostics is None:
        return
    diag_path = os.path.join(args.logs_dir, f"{stem}_alignment.json")
    payload = {
        "aggregate_by": args.aggregate_by,
        "per_run_losses": [r["loss"] for r in per_run],
        "per_token_cos_before": diagnostics["per_token_cos_before"],
        "per_token_cos_after": diagnostics["per_token_cos_after"],
        "per_token_cos_before_list": diagnostics["per_token_cos_before_list"],
        "per_token_cos_after_list": diagnostics["per_token_cos_after_list"],
        "permutations": diagnostics["permutations"],
        "ref_index": diagnostics["ref_index"],
    }
    if "functional_probe" in diagnostics:
        payload["functional_probe"] = diagnostics["functional_probe"]
    with open(diag_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"  Alignment diagnostics: {diag_path}")

    if args.save_per_run:
        for k, r in enumerate(per_run):
            run_path = os.path.join(
                args.results_dir, f"{stem}_inverted_run{k}.safetensors"
            )
            save_file(
                {"crossattn_emb": r["embed"].squeeze(0).to(torch.bfloat16)},
                run_path,
                metadata={
                    "source_image": stem,
                    "run_index": str(k),
                    "best_loss": f"{r['loss']:.6f}",
                    "seed": str(args.aggregate_seed_base + k * 1000),
                },
            )


def process_single(args, anima, device):
    """Process a single image (--image mode)."""
    stem = os.path.splitext(os.path.basename(args.image))[0]
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    latents, h, w = load_and_encode_image(args, device)
    init_embed = create_initial_embedding(args, device, anima)

    log_path = os.path.join(args.logs_dir, f"{stem}.csv")
    best_embed, best_loss, per_run, diagnostics = run_aggregated_inversion(
        args, anima, latents, init_embed, device, log_path=log_path
    )

    out_path = os.path.join(args.results_dir, f"{stem}_inverted.safetensors")
    save_dict = {"crossattn_emb": best_embed.squeeze(0).to(torch.bfloat16)}
    metadata = {
        "source_image": os.path.basename(args.image),
        "image_size": f"{h}x{w}",
        "steps": str(args.steps),
        "lr": str(args.lr),
        "best_loss": f"{best_loss:.6f}",
        "init_prompt": args.init_prompt or "",
        "aggregate_by": str(args.aggregate_by),
        "active_length": str(args.active_length),
    }
    if diagnostics is not None:
        metadata["per_token_cos_before"] = f"{diagnostics['per_token_cos_before']:.6f}"
        metadata["per_token_cos_after"] = f"{diagnostics['per_token_cos_after']:.6f}"
    save_file(save_dict, out_path, metadata=metadata)
    logger.info(f"Saved: {out_path} (loss={best_loss:.6f})")

    _save_aggregation_artifacts(args, stem, per_run, diagnostics)

    if args.verify and args.vae:
        verify_path = os.path.join(args.results_dir, f"{stem}_verify.png")
        verify_embedding(args, anima, best_embed, h, w, device, verify_path)


def process_batch(args, anima, device):
    """Process a directory of preprocessed images (--image_dir mode)."""
    images = discover_cached_images(args.image_dir)
    if not images:
        logger.error(f"No images with cached latents found in {args.image_dir}")
        return

    if args.shuffle:
        random.shuffle(images)

    if args.num_images is not None:
        images = images[: args.num_images]

    logger.info(f"Processing {len(images)} images from {args.image_dir}")
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    for i, img_info in enumerate(images):
        stem = img_info.stem
        logger.info(f"[{i + 1}/{len(images)}] {stem}")

        out_path = os.path.join(args.results_dir, f"{stem}_inverted.safetensors")
        if os.path.exists(out_path):
            logger.info(f"  Skipping (already exists): {out_path}")
            continue

        # Load cached latents
        lat, _res, orig_h, orig_w = load_cached_latents(img_info.npz_path)
        latents = lat.unsqueeze(0).to(device, dtype=torch.bfloat16)

        # Initialize from cached TE embedding
        init_embed = create_initial_embedding(
            args, device, anima, te_path=img_info.te_path
        )

        # Optimize (N runs + alignment if --aggregate_by > 1)
        log_path = os.path.join(args.logs_dir, f"{stem}.csv")
        best_embed, best_loss, per_run, diagnostics = run_aggregated_inversion(
            args, anima, latents, init_embed, device, log_path=log_path
        )

        # Save
        save_dict = {"crossattn_emb": best_embed.squeeze(0).to(torch.bfloat16)}
        metadata = {
            "source_image": stem,
            "image_size": f"{orig_h}x{orig_w}",
            "steps": str(args.steps),
            "lr": str(args.lr),
            "best_loss": f"{best_loss:.6f}",
            "aggregate_by": str(args.aggregate_by),
            "active_length": str(args.active_length),
        }
        if diagnostics is not None:
            metadata["per_token_cos_before"] = (
                f"{diagnostics['per_token_cos_before']:.6f}"
            )
            metadata["per_token_cos_after"] = (
                f"{diagnostics['per_token_cos_after']:.6f}"
            )
        save_file(save_dict, out_path, metadata=metadata)
        logger.info(f"  Saved: {out_path} (loss={best_loss:.6f})")

        _save_aggregation_artifacts(args, stem, per_run, diagnostics)

        if args.verify and args.vae:
            verify_path = os.path.join(args.results_dir, f"{stem}_verify.png")
            verify_embedding(
                args, anima, best_embed, orig_h, orig_w, device, verify_path
            )


def main():
    args = parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Device: {device}")

    # Load frozen DiT
    logger.info("Loading DiT model...")
    is_swapping = args.blocks_to_swap > 0
    grad_ckpt = args.blocks_to_swap < 0
    anima = anima_utils.load_anima_model(
        device="cpu" if is_swapping else device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        split_attn=True,
        loading_device="cpu" if is_swapping else device,
        dit_weight_dtype=torch.bfloat16,
    )
    anima.to(torch.bfloat16)
    anima.requires_grad_(False)
    anima.split_attn = False  # all batch elements share the same image; avoids data-dependent graph breaks

    if is_swapping:
        logger.info(f"Enabling block swap: {args.blocks_to_swap} blocks to CPU")
        anima.enable_block_swap(args.blocks_to_swap, device)
        anima.move_to_device_except_swap_blocks(device)
        anima.prepare_block_swap_before_forward()
    else:
        anima.to(device)
        if grad_ckpt:
            # Gradient checkpointing: recompute block activations during backward.
            # Blocks must be in train mode for checkpointing to activate.
            logger.info("Enabling gradient checkpointing")
            anima.enable_gradient_checkpointing()
            for block in anima.blocks:  # type: ignore[union-attr]
                block.train()

        logger.info("Compiling DiT with torch.compile...")
        anima = torch.compile(anima)

    if args.image_dir is not None:
        process_batch(args, anima, device)
    else:
        process_single(args, anima, device)

    logger.info("Done!")


# endregion


if __name__ == "__main__":
    main()
