"""Reference inversion: learn K prefix tokens that represent a reference image.

Textual-inversion-style "referencer": freeze the DiT, freeze a user-provided
template prompt, and optimize only K consecutive token vectors in the
post-LLM-adapter (T5-compatible) space against the flow-matching loss for a
single reference image. The resulting K vectors capture the image's subject /
style within what the DiT already knows how to read through cross-attention.

The output file uses the same `prefix_embeds` key + metadata schema as
networks/postfix_anima.py's "prefix" mode, so it plugs directly into today's
inference path (`inference.py --prefix_weight`) with zero changes there: at
inference the K vectors get prepended to the first K positions of the user's
prompt's crossattn_emb (trimming K trailing padding zeros to keep total length
= 512). This mirrors exactly how the K slots are assembled at training time.

Metadata additionally records the template text and placeholder offset, so a
future placement-aware loader can splice the K vectors into the middle of a
user's prompt (where they wrote `<REF>`) instead of hard-prepending.

Usage:
    # Basic: K=8 slots, defaults (template "a photo")
    python scripts/inversion/invert_reference.py \
        --image path/to/ref.png \
        --dit models/diffusion_models/anima-preview3-base.safetensors \
        --vae models/vae/qwen_image_vae.safetensors \
        --text_encoder models/text_encoders/qwen_3_06b_base.safetensors \
        --save_path output/ref_cat.safetensors

    # Placement marker (metadata-only for now; runtime will front-prepend)
    python scripts/inversion/invert_reference.py \
        ... \
        --template "a photo of <REF> in a scene" \
        --num_tokens 8

    # Then at inference:
    python inference.py --prefix_weight output/ref_cat.safetensors \
        --prompt "a <REF> on a beach"   # <REF> is ignored by current loader;
                                        # the K vectors get prepended regardless.
"""

import argparse
import csv
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file
from tqdm import tqdm

from library import strategy_base
from library.anima import (
    models as anima_models,
    weights as anima_utils,
    strategy as strategy_anima,
)
from library.models import qwen_vae as qwen_image_autoencoder_kl
from library.datasets.image_utils import IMAGE_TRANSFORMS
from library.runtime.device import clean_memory_on_device
from library.log import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


PLACEHOLDER_MARKER = "<REF>"
MAX_SEQ_LEN = 512


def parse_args():
    p = argparse.ArgumentParser(description="Reference-image inversion (K-slot prefix)")

    p.add_argument("--image", type=str, required=True, help="Reference image path")
    p.add_argument("--dit", type=str, required=True, help="DiT checkpoint path")
    p.add_argument("--vae", type=str, required=True, help="VAE checkpoint path")
    p.add_argument(
        "--text_encoder", type=str, required=True, help="Qwen3 text encoder path"
    )
    p.add_argument("--attn_mode", type=str, default="flash", help="Attention backend")

    p.add_argument(
        "--template",
        type=str,
        default="a photo",
        help=f"Prompt template. Include '{PLACEHOLDER_MARKER}' to record a splice "
        f"offset in metadata (for a future placement-aware loader). Otherwise "
        f"the K slots go at the very front (matching today's --prefix_weight).",
    )
    p.add_argument(
        "--num_tokens",
        "-K",
        type=int,
        default=8,
        help="Number of trainable slot vectors (default: 8)",
    )
    p.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=None,
        help="Resize reference to H W (default: crop to multiple of 32)",
    )

    # Init
    p.add_argument(
        "--init_std",
        type=float,
        default=0.02,
        help="Gaussian std for slot init. 0 = zero-init (baseline-neutral start).",
    )
    p.add_argument(
        "--init_from_template",
        action="store_true",
        help="Init slots from the first K crossattn positions of the encoded template "
        "(includes BOS + first real tokens). Overrides --init_std.",
    )

    # Optimization
    p.add_argument("--steps", type=int, default=100, help="Optimization steps")
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
        help="Gradient accumulation steps (effective TS per update = timesteps_per_step × grad_accum)",
    )
    p.add_argument(
        "--sigma_sampling",
        type=str,
        default="uniform",
        choices=["uniform", "sigmoid"],
        help="Sigma sampling strategy",
    )
    p.add_argument("--sigmoid_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42, help="RNG seed")

    # Output
    p.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Output .safetensors path (prefix_embeds key + metadata)",
    )
    p.add_argument(
        "--log_csv",
        type=str,
        default=None,
        help="Optional CSV path for per-step loss log",
    )
    p.add_argument("--log_every", type=int, default=10, help="Log loss every N steps")

    # Verification
    p.add_argument(
        "--verify",
        action="store_true",
        help="Generate an image with the inverted prefix after optimization",
    )
    p.add_argument(
        "--verify_prompt",
        type=str,
        default=None,
        help="Prompt for verification. Defaults to --template (reproduces training "
        "conditions; output should resemble the reference).",
    )
    p.add_argument("--verify_steps", type=int, default=30)
    p.add_argument("--verify_seed", type=int, default=42)
    p.add_argument("--flow_shift", type=float, default=5.0)

    # Device / VRAM
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--vae_chunk_size", type=int, default=64)
    p.add_argument(
        "--blocks_to_swap",
        type=int,
        default=0,
        help="Number of transformer blocks to swap to CPU (0 = gradient checkpointing)",
    )

    args = p.parse_args()
    if args.num_tokens <= 0 or args.num_tokens >= MAX_SEQ_LEN:
        p.error(f"--num_tokens must be in [1, {MAX_SEQ_LEN - 1}]")
    return args


# region Data


def load_and_encode_image(args, device):
    logger.info(f"Loading reference image: {args.image}")
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
        latents = vae.encode_pixels_to_latents(img_tensor).to(torch.bfloat16)
    logger.info(f"Encoded latents shape: {latents.shape}")

    del vae
    clean_memory_on_device(device)
    return latents, h, w


# endregion


# region Template encoding


def _locate_placeholder_offset(template: str) -> int:
    """Return the placeholder offset for metadata purposes.

    For the runtime default (front-prepended K slots), this is just 0 — the K
    vectors sit at positions [0, K) of the final crossattn_emb. When the user
    supplies a template with '<REF>' we record its *character position* so a
    future placement-aware loader can re-derive the token offset at load time
    (token offsets depend on the tokenizer and can't be trusted across
    templates). Returns -1 if no marker is present.
    """
    idx = template.find(PLACEHOLDER_MARKER)
    return idx if idx >= 0 else -1


def encode_template(args, device, anima) -> torch.Tensor:
    """Encode the template (with <REF> stripped) through Qwen3 + LLM adapter.

    Returns template_emb: [1, MAX_SEQ_LEN, D] bf16 in T5-compatible space.
    """
    # Strip the marker — the K slots will be prepended at position 0 regardless.
    # The template's role is to give the DiT a sensible caption surrounding the
    # learned slots (e.g. "a photo"), not to dictate slot placement.
    text = args.template.replace(PLACEHOLDER_MARKER, "").strip()
    if not text:
        text = "a photo"
    logger.info(f"Encoding template text: {text!r}")

    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_path=args.text_encoder,
        t5_tokenizer_path=None,
        qwen3_max_length=MAX_SEQ_LEN,
        t5_max_length=MAX_SEQ_LEN,
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
        tokens = tokenize_strategy.tokenize(text)
        embed = encoding_strategy.encode_tokens(
            tokenize_strategy, [text_encoder], tokens
        )
        crossattn_emb, _ = anima._preprocess_text_embeds(
            source_hidden_states=embed[0].to(device),
            target_input_ids=embed[2].to(device),
            target_attention_mask=embed[3].to(device),
            source_attention_mask=embed[1].to(device),
        )
        crossattn_emb[~embed[3].bool().to(device)] = 0
        if crossattn_emb.shape[1] < MAX_SEQ_LEN:
            crossattn_emb = F.pad(
                crossattn_emb, (0, 0, 0, MAX_SEQ_LEN - crossattn_emb.shape[1])
            )
        elif crossattn_emb.shape[1] > MAX_SEQ_LEN:
            crossattn_emb = crossattn_emb[:, :MAX_SEQ_LEN]

    # Sanity: template must fit in the first MAX_SEQ_LEN - K positions so the
    # trimmed tail is pure padding. crossattn_seqlens = count of non-zero rows
    # via the mask is equivalent to t5_attn_mask.sum(-1).
    n_text = int(embed[3].to(device).sum().item())
    if n_text > MAX_SEQ_LEN - args.num_tokens:
        logger.warning(
            f"Template uses {n_text} tokens; with K={args.num_tokens} the last "
            f"{n_text - (MAX_SEQ_LEN - args.num_tokens)} will be trimmed. Consider "
            "shortening --template."
        )

    del text_encoder
    clean_memory_on_device(device)
    return crossattn_emb.to(torch.bfloat16), n_text


# endregion


# region Optimization


def _assemble_emb(slots: torch.Tensor, template_emb: torch.Tensor) -> torch.Tensor:
    """Build [1, MAX_SEQ_LEN, D] = [K slots ; first MAX_SEQ_LEN-K of template].

    Matches the exact layout produced by postfix_anima.PostfixNetwork.prepend_prefix
    at inference time, so what we train here IS what gets used there.
    """
    K = slots.shape[0]
    slots_bf16 = slots.to(template_emb.dtype).unsqueeze(0)  # [1, K, D]
    tail = template_emb[:, : template_emb.shape[1] - K, :]  # [1, S-K, D]
    return torch.cat([slots_bf16, tail], dim=1)


def sample_sigmas(args, batch_size, device):
    if args.sigma_sampling == "sigmoid":
        return torch.sigmoid(
            args.sigmoid_scale * torch.randn(batch_size, device=device)
        )
    return torch.rand(batch_size, device=device)


def step_loss(anima, latents, emb_full, sigmas, padding_mask):
    """One flow-matching loss evaluation — identical math to invert_embedding.py."""
    n_t = sigmas.shape[0]
    lat = latents.expand(n_t, -1, -1, -1)
    noise = torch.randn_like(lat)
    sv = sigmas.view(-1, 1, 1, 1)
    noisy = (1.0 - sv) * lat + sv * noise
    noisy_5d = noisy.to(torch.bfloat16).unsqueeze(2)
    emb = emb_full.expand(n_t, -1, -1)
    pm = padding_mask.expand(n_t, -1, -1, -1)
    timesteps = sigmas.to(torch.bfloat16)
    pred = anima(noisy_5d, timesteps, emb, padding_mask=pm).squeeze(2)
    target = noise - lat
    return F.mse_loss(pred.float(), target.float())


def init_slots(args, template_emb: torch.Tensor, device) -> torch.Tensor:
    """Create the initial [K, D] slot tensor (float32, on device)."""
    K = args.num_tokens
    D = template_emb.shape[-1]

    if args.init_from_template:
        # First K positions of the full template (includes BOS + leading words).
        init = template_emb[0, :K, :].detach().clone().to(torch.float32)
        logger.info(f"Init slots from template[:K] — K={K}, D={D}")
        return init

    if args.init_std <= 0.0:
        logger.info(f"Init slots = zeros — K={K}, D={D}")
        return torch.zeros(K, D, dtype=torch.float32, device=device)

    logger.info(f"Init slots = N(0, {args.init_std}^2) — K={K}, D={D}")
    return torch.randn(K, D, dtype=torch.float32, device=device) * args.init_std


def optimize(args, anima, latents, template_emb, device):
    """Returns (best_slots: [K, D] float32 on cpu, best_loss: float)."""
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    slots = torch.nn.Parameter(init_slots(args, template_emb, device))
    optimizer = torch.optim.AdamW([slots], lr=args.lr, weight_decay=0.0)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.steps, eta_min=args.lr * 0.01
        )
        if args.lr_schedule == "cosine"
        else None
    )

    h_lat, w_lat = latents.shape[-2], latents.shape[-1]
    padding_mask = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)

    csv_file = None
    csv_writer = None
    if args.log_csv is not None:
        os.makedirs(os.path.dirname(args.log_csv) or ".", exist_ok=True)
        csv_file = open(args.log_csv, "w", newline="")
        csv_writer = csv.DictWriter(
            csv_file, fieldnames=["step", "loss", "best_loss", "lr", "grad_norm"]
        )
        csv_writer.writeheader()

    best_loss = float("inf")
    best_slots = None

    pbar = tqdm(range(args.steps), desc="Inverting reference", leave=False)
    for step in pbar:
        optimizer.zero_grad()

        accum_loss = 0.0
        for _ in range(args.grad_accum):
            sigmas = sample_sigmas(args, args.timesteps_per_step, device)
            emb_full = _assemble_emb(slots, template_emb)
            loss = step_loss(anima, latents, emb_full, sigmas, padding_mask)
            (loss / args.grad_accum).backward()
            accum_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_([slots], max_norm=1.0).item()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_val = accum_loss / args.grad_accum
        if loss_val < best_loss:
            best_loss = loss_val
            best_slots = slots.detach().clone().cpu()

        if step % args.log_every == 0 or step == args.steps - 1:
            pbar.set_postfix(
                loss=f"{loss_val:.6f}",
                best=f"{best_loss:.6f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
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

    if csv_file is not None:
        csv_file.close()
    return best_slots, best_loss


# endregion


# region Save


def save_prefix(args, slots: torch.Tensor, n_text: int, best_loss: float):
    """Save K slot vectors using the same key + metadata schema as
    networks/postfix_anima.py prefix mode, so `inference.py --prefix_weight` works.
    """
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    K, D = slots.shape
    state_dict = {"prefix_embeds": slots.to(torch.bfloat16).contiguous()}

    metadata = {
        "ss_network_module": "networks.postfix_anima",
        "ss_network_spec": "postfix",
        "ss_mode": "prefix",
        "ss_splice_position": "end_of_sequence",  # unused by prefix mode; kept for schema parity
        "ss_num_postfix_tokens": str(K),
        "ss_embed_dim": str(D),
        # Reference-inversion specific fields
        "ss_source": "invert_reference",
        "ss_reference_image": os.path.basename(args.image),
        "ss_template": args.template,
        "ss_placeholder_char_offset": str(_locate_placeholder_offset(args.template)),
        "ss_template_token_count": str(n_text),
        "ss_steps": str(args.steps),
        "ss_lr": str(args.lr),
        "ss_init_std": str(args.init_std),
        "ss_init_from_template": str(args.init_from_template),
        "ss_best_loss": f"{best_loss:.6f}",
        "ss_seed": str(args.seed),
    }
    save_file(state_dict, args.save_path, metadata=metadata)
    logger.info(f"Saved: {args.save_path} (loss={best_loss:.6f}, K={K})")


# endregion


# region Verification


def verify(args, anima, slots: torch.Tensor, device):
    """Generate an image with the learned prefix to eyeball quality.

    Uses --verify_prompt (default: the training template) to produce the
    verification crossattn_emb, then prepends the K slots exactly as the
    runtime prefix loader would. If the verification roughly reproduces the
    reference image, the inversion worked.
    """
    from library.inference import sampling as inference_utils

    prompt = args.verify_prompt or args.template
    prompt = prompt.replace(PLACEHOLDER_MARKER, "").strip() or "a photo"
    logger.info(f"Verification prompt: {prompt!r}")

    # Encode prompt through Qwen3 + adapter (fresh encoder load)
    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_path=args.text_encoder,
        t5_tokenizer_path=None,
        qwen3_max_length=MAX_SEQ_LEN,
        t5_max_length=MAX_SEQ_LEN,
    )
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
    encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
    strategy_base.TextEncodingStrategy.set_strategy(encoding_strategy)

    text_encoder, _ = anima_utils.load_qwen3_text_encoder(
        args.text_encoder, dtype=torch.bfloat16, device=device
    )
    text_encoder.eval()
    with torch.no_grad():
        tokens = tokenize_strategy.tokenize(prompt)
        embed = encoding_strategy.encode_tokens(
            tokenize_strategy, [text_encoder], tokens
        )
        crossattn_emb, _ = anima._preprocess_text_embeds(
            source_hidden_states=embed[0].to(device),
            target_input_ids=embed[2].to(device),
            target_attention_mask=embed[3].to(device),
            source_attention_mask=embed[1].to(device),
        )
        crossattn_emb[~embed[3].bool().to(device)] = 0
        if crossattn_emb.shape[1] < MAX_SEQ_LEN:
            crossattn_emb = F.pad(
                crossattn_emb, (0, 0, 0, MAX_SEQ_LEN - crossattn_emb.shape[1])
            )
    del text_encoder
    clean_memory_on_device(device)

    # Splice: match prepend_prefix behavior exactly
    K = slots.shape[0]
    prefix_b = slots.to(device=device, dtype=torch.bfloat16).unsqueeze(0)
    full_emb = torch.cat(
        [prefix_b, crossattn_emb[:, : MAX_SEQ_LEN - K, :]], dim=1
    ).to(torch.bfloat16)

    # Use same resolution as the reference latent grid (square-ish)
    latents_shape_hw = getattr(args, "_verify_hw", None)
    if latents_shape_hw is None:
        h, w = 1024, 1024
    else:
        h, w = latents_shape_hw
    h_lat, w_lat = h // 8, w // 8

    # Decode via fresh VAE load
    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=args.vae_chunk_size,
    )
    vae.to(device, dtype=torch.bfloat16)
    vae.eval()

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
    padding_mask = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)

    if hasattr(anima, "switch_block_swap_for_inference"):
        anima.switch_block_swap_for_inference()

    with torch.no_grad():
        for step_i, t in enumerate(tqdm(timesteps, desc="Verify denoise", leave=False)):
            if hasattr(anima, "prepare_block_swap_before_forward"):
                anima.prepare_block_swap_before_forward()
            noise_pred = anima(
                latents, t.unsqueeze(0), full_emb, padding_mask=padding_mask
            )
            latents = inference_utils.step(latents, noise_pred, sigmas, step_i).to(
                torch.bfloat16
            )
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
    save_stem = os.path.splitext(args.save_path)[0]
    verify_path = f"{save_stem}_verify.png"
    Image.fromarray(pixels).save(verify_path)
    logger.info(f"Saved verification image: {verify_path}")

    del vae
    clean_memory_on_device(device)


# endregion


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Device: {device}")

    # 1. VAE encode the reference (VAE is freed before DiT loads)
    latents, h, w = load_and_encode_image(args, device)

    # 2. Load DiT (frozen)
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
    # All steps share the same image; avoid data-dependent graph breaks.
    anima.split_attn = False

    if is_swapping:
        logger.info(f"Enabling block swap: {args.blocks_to_swap} blocks to CPU")
        anima.enable_block_swap(args.blocks_to_swap, device)
        anima.move_to_device_except_swap_blocks(device)
        anima.prepare_block_swap_before_forward()
    else:
        anima.to(device)
        if grad_ckpt:
            logger.info("Enabling gradient checkpointing")
            anima.enable_gradient_checkpointing()
            for block in anima.blocks:  # type: ignore[union-attr]
                block.train()
        logger.info("Compiling DiT with torch.compile...")
        anima = torch.compile(anima)

    # 3. Encode template (runs Qwen3 + LLM adapter, then frees encoder)
    template_emb, n_text = encode_template(args, device, anima)

    # 4. Optimize K slot vectors
    best_slots, best_loss = optimize(args, anima, latents, template_emb, device)
    assert best_slots is not None, "optimize() produced no best_slots"

    # 5. Save in prefix-mode format (plugs into --prefix_weight today)
    save_prefix(args, best_slots, n_text, best_loss)

    # 6. Optional verification render
    if args.verify:
        args._verify_hw = (h, w)
        verify(args, anima, best_slots, device)

    logger.info("Done!")


if __name__ == "__main__":
    main()
