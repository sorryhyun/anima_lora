"""
Modulation guidance distillation (Phase 1).

Trains `pooled_text_proj` to inject pooled text embedding into the AdaLN
modulation path.  The entire DiT backbone is frozen; only the small projection
MLP (~8M params) receives gradients.

Distillation setup (Starodubcev et al., ICLR 2026, Section 5):
  - Teacher: normal forward with real crossattn_emb, pooled_text_proj disabled.
  - Student: forward with zeroed crossattn_emb (unconditional cross-attention),
    but pooled_text_proj receives the real pooled text vector.
  - Loss: MSE(student_pred, teacher_pred).

This forces pooled_text_proj to encode text information through modulation,
complementing the cross-attention path.

Usage:
    python scripts/distill_modulation.py [--iterations 4000] [--lr 1e-4] [--batch_size 1]
"""

import argparse
import glob
import logging
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from library import anima_utils
from library.anima_models import Anima

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Dataset: load cached latents + crossattn_emb from disk
# ---------------------------------------------------------------------------

class CachedDataset(torch.utils.data.Dataset):
    """Loads pre-cached latents and text encoder outputs for distillation.

    Samples are grouped by latent resolution so that each batch has uniform
    spatial dimensions (matching the bucket-based batching used in training).
    """

    def __init__(self, data_dir: str, batch_size: int = 1):
        self.data_dir = data_dir
        # Find all text encoder cache files (one per image)
        te_files = sorted(glob.glob(os.path.join(data_dir, "*_anima_te.safetensors")))

        # Group samples by latent resolution
        buckets: dict[str, list[tuple[str, str]]] = {}
        for te_path in te_files:
            stem = te_path.replace("_anima_te.safetensors", "")
            latent_pattern = f"{stem}_*_anima.npz"
            latent_files = glob.glob(latent_pattern)
            if not latent_files:
                continue
            latent_path = latent_files[0]
            # Extract resolution from latent key (e.g. "latents_64x64")
            npz_keys = np.load(latent_path).files
            latent_key = [k for k in npz_keys if k.startswith("latents_")][0]
            res = latent_key.split("_", 1)[1]  # "64x64"
            buckets.setdefault(res, []).append((latent_path, te_path))

        # Flatten: emit only full batches from each bucket (drop per-bucket
        # remainders to avoid mixed-resolution batches during collation)
        self.samples = []
        for res, items in buckets.items():
            random.shuffle(items)
            full = (len(items) // batch_size) * batch_size
            self.samples.extend(items[:full])

        logger.info(f"Found {len(self.samples)} cached samples in {data_dir} "
                     f"across {len(buckets)} resolution buckets")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        latent_path, te_path = self.samples[idx]

        # Load latent
        npz = np.load(latent_path)
        # Key is latents_{H}x{W}
        latent_key = [k for k in npz.keys() if k.startswith("latents_")][0]
        latents = torch.from_numpy(npz[latent_key].copy()).float()  # (16, H, W)

        # Load text encoder outputs (load_file avoids mmap non-resizable storage)
        te_data = load_file(te_path)
        if "num_variants" in te_data:
            vi = random.randint(0, int(te_data["num_variants"]) - 1)
            crossattn_emb = te_data[f"crossattn_emb_v{vi}"]
        else:
            crossattn_emb = te_data["crossattn_emb"]

        return latents, crossattn_emb


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Modulation guidance distillation")
    parser.add_argument("--data_dir", type=str, default="post_image_dataset",
                        help="Directory with cached latents and text encoder outputs")
    parser.add_argument("--dit_path", type=str,
                        default="models/diffusion_models/anima-preview3-base.safetensors")
    parser.add_argument("--output_path", type=str, default="output/pooled_text_proj.safetensors",
                        help="Where to save the trained projection weights")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--blocks_to_swap", type=int, default=0,
                        help="Number of transformer blocks to offload to CPU")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--attn_mode", type=str, default="flash",
                        help="Attention mode (torch, flash). flash4 not supported yet.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigmoid_scale", type=float, default=1.0,
                        help="Scale for sigmoid timestep sampling")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a saved pooled_text_proj checkpoint")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--torch_compile", action="store_true", default=True,
                        help="Compile block._forward with torch.compile")
    parser.add_argument("--no_compile", dest="torch_compile", action="store_false",
                        help="Disable torch.compile")
    parser.add_argument("--warmup", type=float, default=0,
                        help="Warmup steps: int >= 1 for absolute steps, float < 1 for ratio of iterations")
    parser.add_argument("--dry_run", action="store_true",
                        help="Iterate entire DataLoader without training to test collation")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # --- Dry run: test DataLoader collation without loading the model ---
    if args.dry_run:
        dataset = CachedDataset(args.data_dir, batch_size=args.batch_size)

        def _collate_dry(batch):
            return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])

        dl = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=2, pin_memory=True, drop_last=True, collate_fn=_collate_dry,
        )
        total = len(dl)
        for i, (lat, te) in enumerate(tqdm(dl, desc="dry-run")):
            if (i + 1) % 200 == 0:
                logger.info(f"  batch {i+1}/{total}  latents={lat.shape}  te={te.shape}")
        logger.info(f"Dry run OK: {total} batches, no collation errors.")
        return

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # --- Load model ---
    logger.info("Loading DiT model...")
    model: Anima = anima_utils.load_anima_model(
        device,
        args.dit_path,
        attn_mode=args.attn_mode,
        split_attn=False,
        loading_device="cpu" if args.blocks_to_swap > 0 else device,
        dit_weight_dtype=dtype,
        fp8_scaled=False,
    )

    # pooled_text_proj isn't in the pretrained checkpoint, so its params are
    # still meta tensors after load_state_dict(assign=True). Materialize on CPU
    # before any .to(device) calls.
    model.pooled_text_proj.to_empty(device="cpu")
    nn.init.kaiming_uniform_(model.pooled_text_proj[0].weight, a=math.sqrt(5))
    nn.init.zeros_(model.pooled_text_proj[0].bias)
    nn.init.zeros_(model.pooled_text_proj[-1].weight)
    nn.init.zeros_(model.pooled_text_proj[-1].bias)

    # Resume from checkpoint if provided
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        from safetensors.torch import load_file
        state = load_file(args.resume)
        model.pooled_text_proj.load_state_dict(state)

    # Enable block swap for VRAM efficiency (two forwards per step)
    if args.blocks_to_swap > 0:
        model.enable_block_swap(args.blocks_to_swap, device)
        model.move_to_device_except_swap_blocks(device)
        model.switch_block_swap_for_training()  # forward+backward block movement
    else:
        model.to(device)

    # Static token count: pad all spatial sequences to 4096 tokens so
    # torch.compile sees a single shape across all bucket resolutions.
    model.set_static_token_count(4096)

    # Compile individual block._forward for speedup.
    # unsloth_checkpoint wraps Block.forward with @torch._disable_dynamo,
    # so we compile _forward (the inner computation) not forward.
    if args.torch_compile:
        model.compile_blocks()
        logger.info("Compiled block._forward with torch.compile")

    # Gradient checkpointing with CPU offload: recompute block activations
    # during backward, offloading saved tensors to CPU between forward/backward.
    # Essential for VRAM with two forwards per step on 16GB GPUs.
    # Note: must keep model in train() mode because Block.forward gates
    # checkpointing behind self.training.
    model.enable_gradient_checkpointing(unsloth_offload=True)
    model.train()

    # Freeze everything, then unfreeze pooled_text_proj
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.pooled_text_proj.parameters():
        param.requires_grad_(True)

    # Train pooled_text_proj in float32 for precision
    model.pooled_text_proj.to(dtype=torch.float32)

    trainable_params = sum(p.numel() for p in model.pooled_text_proj.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable_params:,} / {total_params:,} params "
                f"({trainable_params / total_params * 100:.4f}%)")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.pooled_text_proj.parameters(), lr=args.lr)

    # Warmup + cosine annealing
    warmup_steps = int(args.warmup) if args.warmup >= 1 else int(args.warmup * args.iterations)
    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6 / args.lr, total_iters=warmup_steps
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.iterations - warmup_steps, eta_min=args.lr * 0.1
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.iterations, eta_min=args.lr * 0.1
        )

    # --- Dataset ---
    dataset = CachedDataset(args.data_dir, batch_size=args.batch_size)
    # Custom collate to bypass collate_tensor_fn's _new_shared_filename_cpu
    # which creates non-resizable storage on some PyTorch/Python 3.13 builds.
    def _collate(batch):
        return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # dataset is pre-bucketed; shuffling would mix resolutions
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate,
    )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    # --- Training loop ---
    grad_accum = args.grad_accum
    logger.info(f"Starting distillation: {args.iterations} iterations, "
                f"grad_accum={grad_accum} (effective batch={args.batch_size * grad_accum})")

    data_iter = iter(dataloader)
    running_loss = 0.0
    log_interval = 50

    progress = tqdm(range(args.iterations), desc="distill")
    for step in progress:
        accum_loss = 0.0

        for accum_step in range(grad_accum):
            # Get batch (infinite cycling)
            try:
                latents, crossattn_emb = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                latents, crossattn_emb = next(data_iter)

            # latents: (B, 16, H, W), crossattn_emb: (B, seq, 1024)
            latents = latents.to(device, dtype=dtype)
            crossattn_emb = crossattn_emb.to(device, dtype=dtype)

            B = latents.shape[0]

            # Sample noise and timesteps (sigmoid sampling like training)
            noise = torch.randn_like(latents)
            sigmas = torch.sigmoid(args.sigmoid_scale * torch.randn(B, device=device))
            timesteps = sigmas  # [0, 1] range (model expects this)

            # Noisy input: (1-σ) * latents + σ * noise
            sigmas_expand = sigmas.view(B, 1, 1, 1)
            noisy_input = (1.0 - sigmas_expand) * latents + sigmas_expand * noise

            # Add temporal dim: (B, 16, H, W) -> (B, 16, 1, H, W)
            noisy_input = noisy_input.unsqueeze(2)

            # Padding mask (all zeros = no padding)
            padding_mask = torch.zeros(
                B, 1, latents.shape[-2], latents.shape[-1],
                dtype=dtype, device=device
            )

            # Pre-compute pooled text from real crossattn_emb
            pooled_text = crossattn_emb.max(dim=1).values  # (B, 1024)

            # --- Teacher forward: real crossattn, pooled_text_proj skipped ---
            if model.blocks_to_swap:
                model.prepare_block_swap_before_forward()
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                teacher_pred = model.forward_mini_train_dit(
                    noisy_input, timesteps, crossattn_emb,
                    padding_mask=padding_mask,
                    skip_pooled_text_proj=True,
                )

            # --- Student forward: zeroed crossattn, real pooled text through proj ---
            # requires_grad_ needed for gradient checkpointing
            noisy_input = noisy_input.requires_grad_()
            if model.blocks_to_swap:
                model.prepare_block_swap_before_forward()
            uncond_crossattn = torch.zeros_like(crossattn_emb)
            with torch.autocast("cuda", dtype=dtype):
                student_pred = model.forward_mini_train_dit(
                    noisy_input, timesteps, uncond_crossattn,
                    padding_mask=padding_mask,
                    pooled_text_override=pooled_text,
                )

            # --- MSE loss (scaled for accumulation) ---
            loss = nn.functional.mse_loss(student_pred.float(), teacher_pred.float())
            loss = loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        running_loss += accum_loss
        lr = scheduler.get_last_lr()[0]

        if (step + 1) % log_interval == 0:
            avg = running_loss / log_interval
            progress.set_postfix(loss=f"{avg:.6f}", lr=f"{lr:.2e}")
            running_loss = 0.0
        else:
            progress.set_postfix(loss=f"{accum_loss:.6f}", lr=f"{lr:.2e}")

        # Save checkpoint
        if (step + 1) % args.save_every == 0 or (step + 1) == args.iterations:
            save_path = args.output_path
            state = {
                k: v.to(torch.bfloat16)
                for k, v in model.pooled_text_proj.state_dict().items()
            }
            save_file(state, save_path)
            logger.info(f"Saved checkpoint at step {step + 1} -> {save_path}")

    logger.info("Distillation complete.")


if __name__ == "__main__":
    main()
