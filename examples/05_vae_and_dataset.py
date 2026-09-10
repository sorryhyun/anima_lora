#!/usr/bin/env python3
"""Read the dataset cache and round-trip pixels through the VAE.

Two things a scripts/ author repeatedly needs:

  A. VAE round-trip — encode an image to a latent and decode it back. The clean
     helpers are encode_pixels_to_latents() / decode_to_pixels(), both expecting
     pixels in [-1, 1]. The image is first resized to its free-fit bucket (the
     same select_resize_bucket + resize_to_bucket preprocess uses) so the
     round-trip runs on a training-shaped latent, and the printout shows how
     little the native-aspect fit crops.

  B. Iterate the preprocessed cache that training actually consumes.
     CachedDataset (library/datasets/cache.py) yields
     (idx, latent, crossattn_emb, pooled_text) straight from the
     `{stem}_{WxH}_anima.npz` + `{stem}_anima_te.safetensors` files under
     post_image_dataset/lora/ — no DiT/encoder needed, it's all on disk.
     Samples are bucket-grouped so each batch is one resolution.

    python examples/05_vae_and_dataset.py --image some/photo.png      # part A
    python examples/05_vae_and_dataset.py --data_dir post_image_dataset/lora  # part B
"""

from __future__ import annotations

import argparse


import torch

from library.datasets.cache import CachedDataset
from library.env import default_checkpoints
from library.models import qwen_vae

# env ANIMA_VAE (incl. a project-root `.env`) → configs/base.toml → fallback.
VAE = default_checkpoints().vae


def _load_vae(device):
    return qwen_vae.load_vae(
        VAE, device=device, disable_mmap=True, dtype=torch.bfloat16, eval=True
    )


def _crop_fraction(w: int, h: int, bucket: tuple[int, int]) -> float:
    """Fraction of source pixels the cover-scale→crop to ``bucket`` discards."""
    bw, bh = bucket
    scale = max(bw / w, bh / h)  # cover-scale: smallest factor that covers bucket
    return 1.0 - (bw * bh) / (w * scale * h * scale)


def vae_roundtrip(image_path: str, out_path: str, device, *, target_res: int) -> None:
    """Part A — pixels → latent → pixels, resized to a real Anima bucket first.

    The VAE itself takes any H,W, but training never feeds it raw native pixels:
    a 4000px photo would blow the rope cap and is not a cached shape. Preprocessing
    resizes every image to its free-fit bucket. We do the same here so the
    round-trip is on a *training-shaped* latent — reusing the exact chooser
    (`select_resize_bucket`) and pixel geometry (`resize_to_bucket`) preprocess uses,
    not a hand-rolled resize.

    Free-fit is the only resize mode: it keeps the image's native aspect and
    lands its token count anywhere inside the tier's band, so the crop the
    printout reports is sub-patch.
    """
    from PIL import Image, ImageOps
    from torchvision import transforms

    from library.preprocess.images import resize_to_bucket
    from library.preprocess.resize_preview import select_resize_bucket

    vae = _load_vae(device)

    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    w, h = img.size

    edge, bucket = select_resize_bucket(w, h, target_res)
    print(
        f"native {w}x{h} (AR {w / h:.3f}) @ target_res {target_res}:\n"
        f"  free-fit → {bucket[0]}x{bucket[1]} (AR {bucket[0] / bucket[1]:.3f}, "
        f"tier {edge}, crop {_crop_fraction(w, h, bucket):.1%})"
    )

    img = resize_to_bucket(img, bucket)
    print(f"resized to {img.size[0]}x{img.size[1]}")

    to_tensor = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]  # → [-1, 1]
    )
    pixels = to_tensor(img).unsqueeze(0).to(device, dtype=torch.bfloat16)  # [1,3,H,W]

    with torch.no_grad():
        latent = vae.encode_pixels_to_latents(pixels)  # [1, 16, 1, H/8, W/8]
        recon = vae.decode_to_pixels(latent)  # [-1, 1]

    print(
        f"image {tuple(pixels.shape)} → latent {tuple(latent.shape)} → recon {tuple(recon.shape)}"
    )

    recon01 = (recon.squeeze(0).float().clamp(-1, 1) + 1) / 2  # → [0, 1]
    transforms.ToPILImage()(recon01.cpu()).save(out_path)
    print(f"saved reconstruction → {out_path}")


def iterate_cache(data_dir: str, device) -> None:
    """Part B — read the on-disk training cache and decode one cached latent."""
    dataset = CachedDataset(data_dir, batch_size=1)
    print(f"CachedDataset({data_dir}): {len(dataset.samples)} samples")
    if not dataset.samples:
        print("  (empty — run `make preprocess` first)")
        return

    idx, latent, crossattn_emb, pooled = dataset[0]
    print(
        f"  sample 0: latent={tuple(latent.shape)}  "
        f"crossattn_emb={tuple(crossattn_emb.shape)}  pooled={tuple(pooled.shape)}"
    )

    # Decode the cached latent back to an image to confirm it's the real thing.
    # decode_to_pil is the in-memory latent→PIL exit (VAE decode + the
    # [-1,1]→[0,255] + channel handling), so no temp PNG / hand-rolled
    # denormalization — just an Image you can .save(), composite, or score.
    from library.inference.output import decode_to_pil

    vae = _load_vae(device)
    out = "output/tests/example_05_cached_sample0.png"
    with torch.no_grad():
        decode_to_pil(vae, latent.unsqueeze(0), device).save(out)
    print(f"  decoded cached latent → {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", help="run part A (VAE round-trip) on this image")
    p.add_argument(
        "--data_dir",
        default="post_image_dataset/lora",
        help="run part B: iterate this cache dir",
    )
    p.add_argument("--out", default="output/tests/example_05_roundtrip.png")
    p.add_argument(
        "--target_res",
        type=int,
        default=1024,
        help="part A: tier to resize into (512/768/896/1024/1280/1536)",
    )
    opts = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if opts.image:
        vae_roundtrip(
            opts.image,
            opts.out,
            device,
            target_res=opts.target_res,
        )
    else:
        iterate_cache(opts.data_dir, device)


if __name__ == "__main__":
    main()
