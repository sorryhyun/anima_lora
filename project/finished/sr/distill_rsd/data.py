"""Art HR dataset + degradation for RSD distillation and the x2 finetune.

Yields (gt, lq) pairs in [-1,1], both at HR size (256²): gt = a random HR crop from
our art; lq = bicubic ×scale down + light JPEG/blur, then bicubic-upsampled back to HR
(ResShift feeds the LR upsampled-to-HR; the ×scale lives in the degradation). This
light degradation is art-appropriate (proposal §1b) and keeps the v2 teacher
in-distribution (Phase 0). Swap in ResShift's Real-ESRGAN pipeline later if needed.

Two optional (default-OFF) knobs added for the x2 teacher's text-fidelity pass —
random crops almost never land on text, so the model learns "tiny glyphs = texture"
and hallucinates strokes:
  - text_boxes/text_crop_prob: bias that fraction of samples to crops covering a CTD
    text box (precompute the json with sr/scripts/detect_text_boxes.py).
  - scale_jitter_prob/scale_jitter_max: degrade that fraction at a random scale in
    [scale, scale_jitter_max] instead of exactly `scale`, so sub-native text sizes
    appear WITH GT supervision (teaches graceful degradation instead of confident
    wrong strokes on inputs effectively downscaled more than ×scale).
"""

import io
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

REPO = Path(__file__).resolve().parents[4]
EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _jpeg(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class ArtSRDataset(Dataset):
    def __init__(
        self,
        src=None,
        gt_size=256,
        scale=4,
        length=20000,
        jpeg_range=(55, 95),
        blur_prob=0.3,
        text_boxes=None,
        text_crop_prob=0.0,
        scale_jitter_prob=0.0,
        scale_jitter_max=None,
    ):
        self.src = Path(src or (REPO / "image_dataset")).resolve()
        self.files = sorted(p for p in self.src.rglob("*") if p.suffix.lower() in EXTS)
        if not self.files:
            raise SystemExit(f"no images under {self.src}")
        self.gt_size = gt_size
        self.scale = scale
        self.length = length
        self.jpeg_range = jpeg_range
        self.blur_prob = blur_prob
        self.scale_jitter_prob = scale_jitter_prob
        self.scale_jitter_max = scale_jitter_max or 2 * scale
        self.text_crop_prob = text_crop_prob
        self.text_files = []
        if text_boxes and text_crop_prob > 0:
            tb_path = Path(text_boxes)
            if tb_path.exists():
                bx = json.loads(tb_path.read_text())["boxes"]
                self.text_files = [
                    (p, bx[rel])
                    for p in self.files
                    if (rel := str(p.relative_to(self.src))) in bx and bx[rel]
                ]
            if not self.text_files:
                print(
                    f"ArtSRDataset: WARNING no text boxes usable from {text_boxes} "
                    f"(missing file or no key overlap with {self.src}) — text "
                    f"oversampling DISABLED; run sr/scripts/detect_text_boxes.py"
                )
                self.text_crop_prob = 0.0
        print(
            f"ArtSRDataset: {len(self.files)} source images, virtual length {length}, "
            f"text crops p={self.text_crop_prob} ({len(self.text_files)} files w/ text), "
            f"scale jitter p={self.scale_jitter_prob} "
            f"[{self.scale}, {self.scale_jitter_max}]"
        )

    def __len__(self):
        return self.length

    def _fit_min(self, im: Image.Image):
        """Upscale so both sides >= gt_size; returns (im, sx, sy) box-coord scales."""
        w, h = im.size
        g = self.gt_size
        if w >= g and h >= g:
            return im, 1.0, 1.0
        s = g / min(w, h)
        im = im.resize((max(g, int(w * s + 1)), max(g, int(h * s + 1))), Image.LANCZOS)
        return im, im.size[0] / w, im.size[1] / h

    def _rand_crop(self, im: Image.Image) -> Image.Image:
        im, _, _ = self._fit_min(im)
        w, h = im.size
        g = self.gt_size
        x, y = random.randint(0, w - g), random.randint(0, h - g)
        return im.crop((x, y, x + g, y + g))

    def _text_crop(self, im: Image.Image, boxes) -> Image.Image:
        """Crop covering a random point inside a random text box (clamped in-bounds)."""
        im, sx, sy = self._fit_min(im)
        w, h = im.size
        g = self.gt_size
        x0, y0, x1, y1 = random.choice(boxes)
        cx = (x0 + random.random() * max(x1 - x0, 1)) * sx
        cy = (y0 + random.random() * max(y1 - y0, 1)) * sy
        x = int(min(max(cx - g / 2, 0), w - g))
        y = int(min(max(cy - g / 2, 0), h - g))
        return im.crop((x, y, x + g, y + g))

    def _degrade(self, gt: Image.Image) -> Image.Image:
        g = self.gt_size
        s = self.scale
        if self.scale_jitter_prob and random.random() < self.scale_jitter_prob:
            s = random.uniform(self.scale, self.scale_jitter_max)
        lr = gt.resize((max(1, round(g / s)), max(1, round(g / s))), Image.BICUBIC)
        if random.random() < self.blur_prob:
            from PIL import ImageFilter

            lr = lr.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
        lr = _jpeg(lr, random.randint(*self.jpeg_range))
        return lr.resize((g, g), Image.BICUBIC)  # upsample back to HR size

    def __getitem__(self, idx):
        use_text = bool(self.text_files) and random.random() < self.text_crop_prob
        im = boxes = None
        for _ in range(8):
            try:
                if use_text:
                    path, boxes = random.choice(self.text_files)
                else:
                    path = random.choice(self.files)
                im = Image.open(path).convert("RGB")
                break
            except Exception:  # noqa: BLE001
                continue
        gt = self._text_crop(im, boxes) if use_text else self._rand_crop(im)
        if random.random() < 0.5:
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
        lq = self._degrade(gt)

        def to_t(p):
            return torch.from_numpy(np.asarray(p, np.float32) / 127.5 - 1).permute(
                2, 0, 1
            )

        return {"gt": to_t(gt), "lq": to_t(lq)}
