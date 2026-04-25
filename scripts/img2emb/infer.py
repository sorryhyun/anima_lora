#!/usr/bin/env python
"""Generate an image conditioned on a reference image via the img2emb resampler.

Pipeline:
  ref image → vision tower (bucketed) → AnchoredResampler (finetune ckpt)
  → per-group anchor injection at default slots → DiT cross-attention →
  flow-matching euler denoise → VAE decode → png.

Vision tower is selected via ``--encoder`` (``tipsv2`` or ``pe``), and must
match the encoder used at preprocess + finetune time.

Replaces the text-encoder path entirely — no prompt, no negative prompt.

Usage:
  python scripts/img2emb/infer.py --ref_image path/to/ref.png
  python scripts/img2emb/infer.py --ref_image ref.png --encoder pe
  python scripts/img2emb/infer.py --ref_image ref.png --resampler_ckpt other.safetensors
  python scripts/img2emb/infer.py --ref_image ref.png --slot_override rating=0,people_count=2
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.img2emb.anchors import (  # noqa: E402
    AnchorSpec,
    AnchoredResampler,
    inject_anchors,
    load_anchor_spec,
)
from scripts.img2emb.buckets import bucket_pixel_size, pick_bucket  # noqa: E402
from scripts.img2emb.encoders import (  # noqa: E402
    EncoderInfo,
    available_encoders,
    get_encoder_info,
)
from scripts.img2emb.finetune import finetune_ckpt_path  # noqa: E402
from scripts.img2emb.preprocess import DEFAULT_ENCODER  # noqa: E402
from library.anima import weights as anima_utils  # noqa: E402
from library.datasets.buckets import CONSTANT_TOKEN_BUCKETS  # noqa: E402
from library.inference.output import save_images  # noqa: E402
from library.inference.sampling import get_timesteps_sigmas, step as euler_step  # noqa: E402
from library.log import setup_logging  # noqa: E402
from library.models.qwen_vae import load_vae  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


DEFAULT_ANCHORS_YAML = Path(__file__).parent / "anchors.yaml"
DEFAULT_FINETUNE_DIR = REPO_ROOT / "output" / "img2embs" / "finetune"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--ref_image", required=True, help="Path to the reference PNG/JPG.")
    p.add_argument(
        "--encoder",
        default=DEFAULT_ENCODER,
        choices=available_encoders(),
        help="Vision encoder. Must match the encoder used at preprocess + finetune time.",
    )
    p.add_argument(
        "--encoder_model_id",
        default=None,
        help="Override the encoder weights path (defaults to the registered local path).",
    )
    p.add_argument(
        "--resampler_ckpt",
        default=None,
        help="Path to the img2emb resampler ckpt (finetune output). Defaults to "
             "output/img2embs/finetune/{encoder}_finetune.safetensors.",
    )
    p.add_argument(
        "--tag_slot_dir",
        default=str(REPO_ROOT / "output" / "img2embs" / "anchors"),
        help="Directory with class prototype tables (needed to rebuild AnchoredResampler).",
    )
    p.add_argument(
        "--anchors_yaml",
        default=str(DEFAULT_ANCHORS_YAML),
        help="YAML spec listing anchor groups + classes.",
    )
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument(
        "--n_slots",
        type=int,
        default=256,
        help="Resampler query count K (default 256). Output is zero-padded to "
             "512 before feeding to DiT cross-attn (the cached T5 crossattn_emb "
             "shape).",
    )
    p.add_argument(
        "--dit_slot_count",
        type=int,
        default=512,
        help="DiT cross-attn slot count (matches cached crossattn_emb shape).",
    )

    p.add_argument(
        "--slot_override",
        default="",
        help="Per-group slot override, e.g. 'rating=0,people_count=2'. "
             "Unspecified groups use default_slot from anchors.yaml.",
    )
    p.add_argument(
        "--skip_anchors",
        action="store_true",
        help="Skip anchor injection (debug: use raw resampler output).",
    )
    p.add_argument(
        "--use_cached_te",
        action="store_true",
        help="Diagnostic: bypass vision encoder + resampler entirely and feed the cached "
             "T5 crossattn_emb_v0 (from <ref_dir>/<stem>_anima_te.safetensors) as ctx. "
             "Verifies the DiT+VAE+pad pipeline independently of the resampler.",
    )
    p.add_argument(
        "--te_variant",
        type=int,
        default=0,
        help="With --use_cached_te, which variant (0..V-1) to load as ctx.",
    )

    # DiT / VAE
    p.add_argument("--dit", default="models/diffusion_models/anima-preview3-base.safetensors")
    p.add_argument("--vae", default="models/vae/qwen_image_vae.safetensors")
    p.add_argument("--attn_mode", default="flash")
    p.add_argument("--blocks_to_swap", type=int, default=0)

    # sampling — default (None) picks the constant-token bucket closest to the
    # ref image's aspect ratio; pass `--image_size H W` to force a size.
    p.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("H", "W"))
    p.add_argument("--infer_steps", type=int, default=30)
    p.add_argument("--flow_shift", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=4.0,
                   help="CFG scale; 1.0 disables CFG. Unconditional uses zero-ctx.")

    # output
    p.add_argument("--save_path", default="output/tests")
    p.add_argument("--no_metadata", action="store_true")
    p.add_argument("--output_type", default="images", choices=["images", "latent", "latent_images"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def _bucket_for_ref(ref_path: str) -> tuple[int, int]:
    """Pick the CONSTANT_TOKEN_BUCKETS entry whose aspect ratio is closest to
    the ref image. Returns (H, W) to match --image_size convention."""
    with Image.open(ref_path) as img:
        w, h = img.size
    target = w / h
    best = min(CONSTANT_TOKEN_BUCKETS, key=lambda wh: abs((wh[0] / wh[1]) - target))
    return (best[1], best[0])


def _parse_slot_override(raw: str) -> dict[str, int]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    out: dict[str, int] = {}
    for chunk in raw.split(","):
        if not chunk.strip():
            continue
        k, v = chunk.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def _encode_ref_image(
    path: str,
    info: EncoderInfo,
    model_id: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(tokens[1, T_MAX_TOKENS, D] bf16, pooled[1, D] f32)``.

    Picks the bucket closest to the ref's aspect ratio for the given encoder
    and encodes at that pixel size. Tokens are zero-padded to the encoder's
    ``t_max_tokens`` so the resampler sees the same KV length distribution it
    saw during training.
    """
    bucket_spec = info.bucket_spec
    T_MAX = bucket_spec.t_max_tokens

    img = Image.open(path).convert("RGB")
    bucket = pick_bucket(img.height, img.width, bucket_spec)
    Hp, Wp = bucket_pixel_size(bucket, bucket_spec)
    cls_extra = 1 if bucket_spec.use_cls else 0
    logger.info(
        f"[{info.name}] bucket pick: aspect h/w={img.height / img.width:.3f} → "
        f"({bucket[0]}x{bucket[1]}) = {Hp}x{Wp}px, "
        f"tokens={bucket[0] * bucket[1] + cls_extra} (padded to {T_MAX})"
    )

    model = info.loader(device, model_id)
    processor = info.processor_factory(image_size=(Hp, Wp))

    pixel_values = processor(images=img, return_tensors="pt")["pixel_values"].to(
        device, dtype=torch.bfloat16
    )
    with torch.no_grad():
        out = model(pixel_values=pixel_values)
    tokens = out.last_hidden_state.to(torch.bfloat16)
    if out.pooler_output is not None:
        pooled = out.pooler_output.float()
    else:
        pooled = out.last_hidden_state.float().mean(dim=1)

    T_cur = tokens.shape[1]
    if T_cur < T_MAX:
        pad = torch.zeros(
            (tokens.shape[0], T_MAX - T_cur, tokens.shape[2]),
            dtype=tokens.dtype,
            device=tokens.device,
        )
        tokens = torch.cat([tokens, pad], dim=1)
    elif T_cur > T_MAX:
        raise RuntimeError(
            f"Encoded tokens T={T_cur} exceed T_MAX_TOKENS={T_MAX}; "
            "bucket spec and preprocessing cache are out of sync."
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return tokens, pooled


def _load_resampler(
    args, spec: AnchorSpec, info: EncoderInfo, device: torch.device,
) -> AnchoredResampler:
    model = AnchoredResampler(
        spec=spec, d_enc=info.d_enc, d_pool=info.d_pool,
        d_model=args.d_model, n_heads=args.n_heads, n_slots=args.n_slots, n_layers=args.n_layers,
    ).to(device)

    sd = load_file(args.resampler_ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"resampler missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        logger.warning(f"resampler unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    model.eval()
    return model


def _build_ctx(
    model: AnchoredResampler,
    spec: AnchorSpec,
    tokens: torch.Tensor,
    pooled: torch.Tensor,
    args,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resampler + per-group anchor injection at default slots.

    Returns ``(ctx[1, S, 1024] bf16, pooled_text[1, 1024] bf16)``. ``pooled_text``
    is ``amax`` over every K content slot (matches the finetune-time pooling).
    """
    with torch.no_grad():
        fwd = model(tokens, pooled)
    pred = fwd["pred"]
    B, K = pred.shape[:2]
    slot_overrides = _parse_slot_override(args.slot_override)

    for g in spec.groups:
        lg = fwd["logits"][g.name][0]
        if g.mutex:
            idx = int(lg.argmax(dim=-1).item())
            name = g.classes[idx] if idx < g.n_classes else "<unknown>"
            logger.info(f"  [{g.name}] top1={name} (idx={idx})")
        else:
            probs = torch.sigmoid(lg[: g.n_classes])
            pos = [(g.classes[i], float(probs[i].item())) for i in range(g.n_classes) if probs[i].item() > 0.5]
            logger.info(f"  [{g.name}] positives={pos}")

    if not args.skip_anchors:
        for g in spec.groups:
            anchor_emb = fwd["anchor_emb"][g.name]
            if g.mutex:
                slot = slot_overrides.get(g.name, g.default_slot)
                slots = torch.full((B,), int(slot), dtype=torch.long, device=device)
                inject_anchors(pred, anchor_emb, slots, mutex=True, mode="replace")
            else:
                base_slot = slot_overrides.get(g.name)
                if base_slot is None:
                    per_class = torch.tensor(g.default_slots, dtype=torch.long, device=device)
                else:
                    per_class = torch.full((g.n_classes,), int(base_slot), dtype=torch.long, device=device)
                slots = per_class.unsqueeze(0).expand(B, -1).contiguous()
                logits = fwd["logits"][g.name][..., : g.n_classes]
                mask = (torch.sigmoid(logits) > 0.5)
                inject_anchors(pred, anchor_emb, slots, mutex=False, mode="replace", mask=mask)

    pooled_text = pred.float().amax(dim=1).to(torch.bfloat16)
    return _pad_to_dit(pred, args.dit_slot_count).to(torch.bfloat16), pooled_text


def _pad_to_dit(pred: torch.Tensor, pad_to: int) -> torch.Tensor:
    """Zero-pad (B, K, D) → (B, pad_to, D) to match DiT cross-attn slot count."""
    K = pred.shape[1]
    if K >= pad_to:
        return pred
    return F.pad(pred, (0, 0, 0, pad_to - K))


@torch.no_grad()
def _denoise(
    anima, ctx: torch.Tensor, pooled_text: torch.Tensor, args, device: torch.device
) -> torch.Tensor:
    """Flow-matching euler denoise with optional CFG. Returns (1, C, 1, H_lat, W_lat) bf16.

    CFG unconditional: zero-ctx, same pooled_text. Matches the cache-time
    zero-pad convention (crossattn_emb_v0 padding is zeros).
    """
    H, W = args.image_size
    h_lat, w_lat = H // 8, W // 8
    C = 16  # Anima.LATENT_CHANNELS

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    latents = torch.randn((1, C, 1, h_lat, w_lat), generator=g, dtype=torch.bfloat16).to(device)

    timesteps, sigmas = get_timesteps_sigmas(args.infer_steps, args.flow_shift, device)
    timesteps = (timesteps / 1000).to(dtype=torch.bfloat16)

    padding_mask = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)
    kwargs = {"padding_mask": padding_mask, "pooled_text_override": pooled_text}

    do_cfg = args.guidance_scale != 1.0
    uncond_ctx = torch.zeros_like(ctx) if do_cfg else None

    logger.info(
        f"denoising: {args.infer_steps} steps, {H}x{W}, flow_shift={args.flow_shift}, "
        f"cfg={args.guidance_scale}"
    )
    for i, t in enumerate(timesteps):
        t_expand = t.expand(1)
        noise_pred = anima(latents, t_expand, ctx, **kwargs)
        if do_cfg:
            uncond_pred = anima(latents, t_expand, uncond_ctx, **kwargs)
            noise_pred = uncond_pred + args.guidance_scale * (noise_pred - uncond_pred)
        latents = euler_step(latents, noise_pred, sigmas, i).to(torch.bfloat16)
    return latents


def infer(args: argparse.Namespace) -> None:
    """Run inference end-to-end. Importable from other tools if needed."""
    device = torch.device(args.device)
    os.makedirs(args.save_path, exist_ok=True)

    info = get_encoder_info(args.encoder)
    if args.resampler_ckpt is None:
        args.resampler_ckpt = str(finetune_ckpt_path(DEFAULT_FINETUNE_DIR, encoder=args.encoder))
    encoder_model_id = args.encoder_model_id or info.default_model_id()

    if args.image_size is None:
        args.image_size = list(_bucket_for_ref(args.ref_image))
        logger.info(f"image_size auto-picked from ref aspect: {tuple(args.image_size)} (HxW)")

    spec = load_anchor_spec(Path(args.anchors_yaml), Path(args.tag_slot_dir))

    if args.use_cached_te:
        ref_path = Path(args.ref_image)
        te_path = ref_path.with_name(f"{ref_path.stem}_anima_te.safetensors")
        if not te_path.exists():
            raise SystemExit(f"--use_cached_te: no TE cache found at {te_path}")
        sd = load_file(str(te_path))
        v = args.te_variant
        key = f"crossattn_emb_v{v}" if f"crossattn_emb_v{v}" in sd else "crossattn_emb"
        ctx = sd[key].to(device=device, dtype=torch.bfloat16).unsqueeze(0)
        pooled_text = ctx.float().amax(dim=1).to(torch.bfloat16)
        logger.info(
            f"using cached T5 {te_path.name}[{key}] as ctx; "
            f"shape={tuple(ctx.shape)}  norm={ctx.float().norm():.2f}"
        )
    else:
        logger.info(f"encoding ref image: {args.ref_image} (encoder={info.name})")
        tokens, pooled = _encode_ref_image(args.ref_image, info, encoder_model_id, device)
        logger.info(f"{info.name} tokens={tuple(tokens.shape)}  pooled={tuple(pooled.shape)}")

        logger.info(f"loading resampler ckpt: {args.resampler_ckpt}")
        resampler = _load_resampler(args, spec, info, device)
        ctx, pooled_text = _build_ctx(resampler, spec, tokens, pooled, args, device)
        logger.info(f"pooled_text norm={pooled_text.float().norm():.2f}")
        logger.info(f"ctx shape: {tuple(ctx.shape)}")
        del resampler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info(f"loading DiT: {args.dit}")
    is_swapping = args.blocks_to_swap > 0
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
    anima.eval()
    anima.split_attn = False
    if is_swapping:
        anima.enable_block_swap(args.blocks_to_swap, device)
        anima.move_to_device_except_swap_blocks(device)
        anima.prepare_block_swap_before_forward()
    else:
        anima.to(device)

    latents = _denoise(anima, ctx, pooled_text, args, device)
    del anima
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info(f"loading VAE: {args.vae}")
    vae = load_vae(args.vae, device=device, disable_cache=True)
    with torch.no_grad():
        pixels = vae.decode_to_pixels(latents.to(device, dtype=vae.dtype))
    if pixels.ndim == 5:
        pixels = pixels.squeeze(2)
    pixels = pixels[0].to("cpu", dtype=torch.float32)

    args.prompt = f"[img2emb:{Path(args.ref_image).name}]"
    args.negative_prompt = None
    out = save_images(pixels, args, original_base_name=Path(args.ref_image).stem)
    logger.info(f"saved: {out}.png")

    ref_copy = Path(f"{out}_ref{Path(args.ref_image).suffix}")
    shutil.copyfile(args.ref_image, ref_copy)
    logger.info(f"ref image copied to: {ref_copy}")


def main() -> None:
    infer(parse_args())


if __name__ == "__main__":
    main()
