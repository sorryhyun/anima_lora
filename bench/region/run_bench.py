#!/usr/bin/env python3
"""Region adapter bench — does the paint blob steer character position/size?

The region EasyControl adapter claims: a solid paint blob over a real
background image controls WHERE (and roughly how big) the character is
generated, while the caption owns identity and the cond owns the scene.
This bench measures that claim directly:

  Phase A0 (GPU) — generate one character-free background *plate* per prompt
      (scene-only ``no humans`` twin of each caption, base model, no adapter),
      decoded up front so conds can be painted onto it.
  Phase A (GPU) — generate a grid of images: ``prompts × layouts × seeds``
      with paint conds (soft gray blobs at known positions/sizes composited
      onto that prompt's plate), plus an unpainted-plate control arm per
      ``prompt × seed``. One shared DiT; each sample's EasyControl patch is
      removed after its generate() call (``network.remove_from()``) — a fresh
      network instance re-patches the shared model every call, so skipping
      the removal stacks wrappers and compounds the cond. Latents decode at
      the end with one VAE load, after the DiT is freed.
  Phase B (GPU) — SAM3-segment ``girl`` in every output via
      ``scripts/preprocess/generate_masks.py`` (focus mode, threshold 0.4 —
      the dataset-staging setting).
  Phase C — metrics against each sample's paint mask:
      * iou               — IoU(paint, generated-girl mask)
      * center_dist       — paint→girl centroid distance / image diagonal
      * girl_in_paint     — |girl ∩ paint| / |girl| (containment recall)
      * area_ratio        — girl area / paint area
      * center_corr_x/y   — Pearson r between paint centroid and girl centroid
                            across the cond arm (the "does it follow" score;
                            an inert adapter scores ~0)
      * chance_iou        — control-arm girls scored against every layout
                            (the no-paint baseline IoU); the headline lift is
                            ``mean_iou - chance_iou``
      * found_rate        — SAM found a girl at all (per arm)
      * bg_psnr           — PSNR between output and the plate outside
                            (dilated paint ∪ girl) — does the scene survive?

Run (GPU — submit through the daemon)::

    make daemon-run ARGS="--label region-bench bench/region/run_bench.py [--label v1]"

Reuses the newest ``anima_easycontrol_region*.safetensors`` under output/ckpt
unless ``--adapter`` is given. ``--easyedit_adapter <phash_edit.safetensors>``
adds a third ``easyedit`` arm on the SAME plates + paint masks: the archived
directedit_ec "EasyEdit" mask recipe (``scripts/edit.py --mask`` +
``--easycontrol_mask``, ``prompt_src=""``, ``prompt_tar=<girl caption>``) run
as a subprocess per sample, so the trained region adapter and the
zero-training hybrid are scored by identical paint metrics
(``metrics.arms.easyedit``). ``--skip_generate``/``--skip_segment`` resume a
run dir (pass ``--run_dir``) to iterate on metrics without re-rendering.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from bench._common import REPO_ROOT, make_run_dir, write_result  # noqa: E402

# Booru-style prompts matching the training caption register: generic solo
# girls with distinct scenes so placement isn't prompt-forced.
PROMPTS = [
    "1girl, solo, long hair, black hair, school uniform, classroom, standing, smile",
    "1girl, solo, silver hair, blue eyes, white dress, outdoors, garden, flowers",
    "1girl, solo, twintails, blonde hair, casual clothes, city street, walking",
    "1girl, solo, maid, maid headdress, indoors, cafe, holding tray",
]

# Character-free scene twin of each prompt — the background plate the paint
# blob is composited onto (index-paired with PROMPTS).
PLATE_PROMPTS = [
    "no humans, empty classroom, chalkboard, desks, school, indoors, scenery",
    "no humans, outdoors, garden, flowers, bushes, path, scenery",
    "no humans, city street, crosswalk, buildings, road, scenery",
    "no humans, cafe interior, tables, chairs, counter, indoors, scenery",
]

# Layouts as normalized ellipses (cx, cy, rx, ry) — position/size spread wide
# enough that following vs ignoring the paint separates cleanly.
LAYOUTS = {
    "left": (0.28, 0.55, 0.16, 0.35),
    "right": (0.72, 0.55, 0.16, 0.35),
    "top_left": (0.30, 0.32, 0.17, 0.22),
    "bottom_right": (0.70, 0.70, 0.18, 0.25),
    "small_center": (0.50, 0.50, 0.10, 0.15),
    "large_center": (0.50, 0.52, 0.30, 0.42),
}

PAINT_COLOR = (128, 128, 128)  # keep in sync with configs/easycontrol/region.toml


def _latest_region_adapter() -> Path:
    hits = sorted(
        (REPO_ROOT / "output" / "ckpt").glob("anima_easycontrol_region*.safetensors"),
        key=lambda p: p.stat().st_mtime,
    )
    if not hits:
        raise SystemExit(
            "No anima_easycontrol_region*.safetensors under output/ckpt — train with "
            "`make easycontrol EASYADAPTER=region` first, or pass --adapter."
        )
    return hits[-1]


def _draw_layout(size: tuple[int, int], spec, blur_frac: float = 0.012) -> np.ndarray:
    """Soft-edged ellipse blob (hand-paint look), binary uint8 {0,1}."""
    import cv2

    w, h = size
    cx, cy, rx, ry = spec
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    draw.ellipse(
        [
            (cx - rx) * w,
            (cy - ry) * h,
            (cx + rx) * w,
            (cy + ry) * h,
        ],
        fill=255,
    )
    m = np.array(canvas, dtype=np.float32) / 255.0
    m = cv2.GaussianBlur(m, (0, 0), max(h, w) * blur_frac)
    return (m >= 0.5).astype(np.uint8)


def _easyedit_arms(args) -> dict[str, Path]:
    """``--easyedit_adapter`` paths → ``easyedit_<tag>`` arm names."""
    arms = {}
    for path in getattr(args, "easyedit_adapter", None) or []:
        tag = Path(path).stem.replace("anima_easycontrol_", "").replace("anima_", "")
        arms[f"easyedit_{tag}"] = Path(path)
    return arms


def _is_easyedit(arm: str) -> bool:
    return arm.startswith("easyedit_")


def _samples(args) -> list[dict]:
    """The full sample plan: cond arm + unpainted-plate control arm."""
    plan = []
    n_prompts = min(len(PROMPTS), getattr(args, "max_prompts", None) or len(PROMPTS))
    for pi in range(n_prompts):
        for seed_i in range(args.seeds):
            for layout in LAYOUTS:
                plan.append(
                    {
                        "arm": "cond",
                        "prompt_idx": pi,
                        "layout": layout,
                        "seed": 1000 + 97 * pi + 31 * seed_i,
                        "name": f"cond_p{pi}_{layout}_s{seed_i}",
                    }
                )
                for tag, path in _easyedit_arms(args).items():
                    plan.append(
                        {
                            "arm": tag,
                            "adapter": str(path),
                            "prompt_idx": pi,
                            "layout": layout,
                            "seed": 1000 + 97 * pi + 31 * seed_i,
                            "name": f"{tag}_p{pi}_{layout}_s{seed_i}",
                        }
                    )
            plan.append(
                {
                    "arm": "control",
                    "prompt_idx": pi,
                    "layout": "plate",
                    "seed": 1000 + 97 * pi + 31 * seed_i,
                    "name": f"control_p{pi}_s{seed_i}",
                }
            )
    return plan


def _decode_pending(pending: list[tuple], out_dir: Path, device) -> None:
    """Decode queued (gen_args, latent, name) with one VAE load → {name}.png.

    save_images treats save_path as a DIRECTORY and invents a timestamped
    filename — decode into a per-sample temp dir, then move the single png
    to the plan name."""
    import shutil

    from anima_lora import load_vae, save_output
    from library.runtime.device import clean_memory_on_device

    if not pending:
        return
    vae = load_vae(
        pending[0][0].vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=pending[0][0].vae_chunk_size,
        disable_cache=pending[0][0].vae_disable_cache,
        dtype=torch.bfloat16,
        eval=True,
    )
    for gen_args, latent, name in pending:
        tmp_dir = out_dir / f"_tmp_{name}"
        gen_args.save_path = str(tmp_dir)
        save_output(gen_args, vae, latent.to(device), device)
        saved = sorted(tmp_dir.glob("*.png"))
        if not saved:
            raise RuntimeError(f"decode produced no png for {name}")
        saved[-1].replace(out_dir / f"{name}.png")
        shutil.rmtree(tmp_dir)
    del vae
    clean_memory_on_device(device)


def phase_generate(args, run_dir: Path, plan: list[dict]) -> None:
    from anima_lora import (
        GenerationRequest,
        default_checkpoints,
        generate,
        get_generation_settings,
    )
    from library.runtime.device import clean_memory_on_device

    ckpt = default_checkpoints()

    conds_dir = run_dir / "conds"
    images_dir = run_dir / "images"
    conds_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    size_wh = (args.size[1], args.size[0])  # (W, H)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_prompts = min(len(PROMPTS), getattr(args, "max_prompts", None) or len(PROMPTS))

    def _request(prompt: str, out: Path, seed: int, cond: Path | None):
        easy = (
            dict(
                easycontrol_weight=str(args.adapter),
                easycontrol_image=str(cond),
                extra_argv=(
                    ["--easycontrol_b_offset", str(args.b_offset)]
                    if args.b_offset is not None
                    else []
                ),
            )
            if cond is not None
            else {}
        )
        request = GenerationRequest(
            dit=ckpt.dit,
            vae=ckpt.vae,
            text_encoder=ckpt.text_encoder,
            prompt=prompt,
            save_path=str(out),
            infer_steps=args.steps,
            guidance_scale=args.cfg,
            image_size=tuple(args.size),  # (H, W)
            seed=seed,
            # flash → the LSE-decomposed extended-attention path; the default
            # 'torch' falls back to masked-SDPA (full [B,H,S_t,S_t+S_c] matrix
            # per block — slow, OOM risk at 1024²).
            attn_mode=args.attn_mode,
            **easy,
        )
        gen_args = request.to_args()
        gen_args.device = device
        return gen_args, get_generation_settings(gen_args)

    # ── Phase A0: character-free background plates (base model, no adapter).
    # Decoded up front — the paint conds are composited onto them — at the
    # cost of one extra DiT load when plates aren't already on disk.
    missing = [
        pi for pi in range(n_prompts) if not (conds_dir / f"plate_p{pi}.png").is_file()
    ]
    if missing:
        shared: dict = {}
        pending: list[tuple] = []
        for pi in missing:
            name = f"plate_p{pi}"
            gen_args, gen_settings = _request(
                PLATE_PROMPTS[pi], conds_dir / f"{name}.png", 7000 + pi, cond=None
            )
            print(f"[plate {pi + 1}/{len(missing)}] {name}")
            latent = generate(gen_args, gen_settings, shared_models=shared)
            pending.append((gen_args, latent.to("cpu"), name))
            clean_memory_on_device(device)
        shared.clear()
        clean_memory_on_device(device)  # free the DiT before the VAE decode
        _decode_pending(pending, conds_dir, device)

    # ── Paint the layout blobs onto each prompt's plate.
    plates = {
        pi: np.array(
            Image.open(conds_dir / f"plate_p{pi}.png").convert("RGB").resize(size_wh)
        )
        for pi in range(n_prompts)
    }
    for layout, spec in LAYOUTS.items():
        mask = _draw_layout(size_wh, spec)
        Image.fromarray(mask * 255).save(conds_dir / f"{layout}_mask.png")
        for pi in range(n_prompts):
            canvas = plates[pi].copy()
            canvas[mask > 0] = PAINT_COLOR
            Image.fromarray(canvas).save(conds_dir / f"cond_p{pi}_{layout}.png")

    # ── Phase A: cond arm (painted plate) + control arm (unpainted plate).
    shared_models: dict = {}
    pending = []
    for i, s in enumerate(plan):
        out = images_dir / f"{s['name']}.png"
        if _is_easyedit(s["arm"]):
            continue  # separate process below (edit.py owns its own model load)
        if out.is_file():
            print(f"[{i + 1}/{len(plan)}] skip {s['name']} (exists)")
            continue
        cond_name = (
            f"cond_p{s['prompt_idx']}_{s['layout']}.png"
            if s["arm"] == "cond"
            else f"plate_p{s['prompt_idx']}.png"
        )
        gen_args, gen_settings = _request(
            PROMPTS[s["prompt_idx"]], out, s["seed"], cond=conds_dir / cond_name
        )
        print(f"[{i + 1}/{len(plan)}] {s['name']}")
        latent = generate(gen_args, gen_settings, shared_models=shared_models)
        pending.append((gen_args, latent.to("cpu"), s["name"]))
        # A fresh network instance patched the shared DiT for this sample —
        # unpatch before the next one or the wrappers stack and the cond
        # compounds.
        anima = shared_models.get("model")
        network = getattr(anima, "_easycontrol_network", None)
        if network is not None:
            network.remove_from()
            anima._easycontrol_network = None
        clean_memory_on_device(device)

    shared_models.clear()
    clean_memory_on_device(device)
    _decode_pending(pending, images_dir, device)

    # ── Phase A' (optional): EasyEdit hybrid arm — edit.py on the plate with
    # the paint mask as both the Δz-anchor release (--mask) and the EC cond
    # gray-hole (--easycontrol_mask); delta grammar (empty ψ_src, the girl
    # caption as ψ_tar). One subprocess per sample: edit.py has no batch mode.
    easyedit = [s for s in plan if _is_easyedit(s["arm"])]
    for i, s in enumerate(easyedit):
        out = images_dir / f"{s['name']}.png"
        if out.is_file():
            print(f"[easyedit {i + 1}/{len(easyedit)}] skip {s['name']} (exists)")
            continue
        _run_easyedit(args, ckpt, s, conds_dir, images_dir)
        print(f"[easyedit {i + 1}/{len(easyedit)}] {s['name']}")


def _run_easyedit(args, ckpt, s: dict, conds_dir: Path, images_dir: Path) -> None:
    import shutil

    mask = conds_dir / f"{s['layout']}_mask.png"
    tmp_dir = images_dir / f"_tmp_{s['name']}"
    tmp_dir.mkdir(exist_ok=True)
    argv = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "edit.py"),
        "--dit",
        ckpt.dit,
        "--text_encoder",
        ckpt.text_encoder,
        "--vae",
        ckpt.vae,
        "--image",
        str(conds_dir / f"plate_p{s['prompt_idx']}.png"),
        # `=` form: delta-grammar ψ can start with a removal `-tag`.
        "--prompt_src=",
        f"--prompt_tar={PROMPTS[s['prompt_idx']]}",
        "--save_path",
        str(tmp_dir),
        "--seed",
        str(s["seed"]),
        "--infer_steps",
        str(args.steps),
        "--guidance_scale",
        str(args.cfg),
        "--attn_mode",
        args.attn_mode,
        "--t_inj",
        "0",
        "--no_compile_blocks",
        "--easycontrol_weight",
        s["adapter"],
        "--easycontrol_scale",
        "1.0",
        "--easycontrol_mask",
        str(mask),
        "--mask",
        str(mask),
    ]
    if args.b_offset is not None:
        argv += ["--easycontrol_b_offset", str(args.b_offset)]
    subprocess.run(argv, check=True, cwd=REPO_ROOT)
    saved = sorted(tmp_dir.glob("*.png"))
    if not saved:
        raise RuntimeError(f"edit.py produced no png for {s['name']}")
    saved[-1].replace(images_dir / f"{s['name']}.png")
    shutil.rmtree(tmp_dir)


def phase_segment(args, run_dir: Path) -> None:
    cfg = {
        "prompts": [],
        "focus_prompts": ["girl"],
        "threshold": 0.4,
        "dilate": 0,
        "path_pattern": "*",
    }
    cfg_path = run_dir / "sam_bench.yaml"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")  # JSON is valid YAML
    subprocess.run(
        [
            sys.executable,
            "scripts/preprocess/generate_masks.py",
            "--config",
            str(cfg_path),
            "--image-dir",
            str(run_dir / "images"),
            "--mask-dir",
            str(run_dir / "masks"),
            "--checkpoint",
            "models/sam3/sam3.pt",
            "--batch-size",
            "4",
            "--recursive",
        ],
        check=True,
        cwd=REPO_ROOT,
    )


def _load_mask(path: Path, size_wh: tuple[int, int]) -> np.ndarray | None:
    if not path.exists():
        return None
    import cv2

    m = (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)
    if m.shape != (size_wh[1], size_wh[0]):
        m = cv2.resize(m, size_wh, interpolation=cv2.INTER_NEAREST)
    return m


def _centroid(m: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(m)
    return float(xs.mean()), float(ys.mean())


def phase_metrics(args, run_dir: Path, plan: list[dict]) -> tuple[dict, list[str]]:
    size_wh = (args.size[1], args.size[0])
    diag = float(np.hypot(*size_wh))
    layout_masks = {
        name: _load_mask(run_dir / "conds" / f"{name}_mask.png", size_wh)
        for name in LAYOUTS
    }
    plates: dict[int, np.ndarray | None] = {}
    for s in plan:
        pi = s["prompt_idx"]
        if pi not in plates:
            p = run_dir / "conds" / f"plate_p{pi}.png"
            plates[pi] = (
                np.array(Image.open(p).convert("RGB").resize(size_wh), dtype=np.float32)
                if p.exists()
                else None
            )

    rows: list[dict] = []
    for s in plan:
        girl = _load_mask(run_dir / "masks" / f"{s['name']}_mask.png", size_wh)
        row = dict(s, found=girl is not None and bool(girl.any()))
        if row["found"] and (s["arm"] == "cond" or _is_easyedit(s["arm"])):
            paint = layout_masks[s["layout"]]
            inter = int((girl & paint).sum())
            union = int((girl | paint).sum())
            pcx, pcy = _centroid(paint)
            gcx, gcy = _centroid(girl)
            row.update(
                iou=inter / union,
                center_dist=float(np.hypot(pcx - gcx, pcy - gcy)) / diag,
                girl_in_paint=inter / int(girl.sum()),
                area_ratio=int(girl.sum()) / int(paint.sum()),
                paint_area_frac=int(paint.sum()) / (size_wh[0] * size_wh[1]),
                girl_area_frac=int(girl.sum()) / (size_wh[0] * size_wh[1]),
                paint_cx=pcx / size_wh[0],
                paint_cy=pcy / size_wh[1],
                girl_cx=gcx / size_wh[0],
                girl_cy=gcy / size_wh[1],
            )
            # Does the scene survive outside the character? PSNR vs the plate
            # over pixels away from both the paint and wherever the girl
            # actually landed (girl overflow is already scored by area/iou).
            plate = plates.get(s["prompt_idx"])
            out_p = run_dir / "images" / f"{s['name']}.png"
            if plate is not None and out_p.exists():
                import cv2

                k = max(3, int(0.02 * max(size_wh)) | 1)
                excl = (cv2.dilate(paint, np.ones((k, k), np.uint8)) > 0) | (girl > 0)
                keep = ~excl
                if keep.any():
                    out_img = np.array(
                        Image.open(out_p).convert("RGB").resize(size_wh), np.float32
                    )
                    mse = float(((out_img - plate)[keep] ** 2).mean())
                    row["bg_psnr"] = float(10 * np.log10(255.0**2 / max(mse, 1e-6)))
        elif row["found"] and s["arm"] == "control":
            # Chance baseline: how well does an UNsteered girl match each layout?
            girl_sum = int(girl.sum())
            ious = [
                int((girl & p).sum()) / int((girl | p).sum())
                for p in layout_masks.values()
            ]
            gcx, gcy = _centroid(girl)
            row.update(
                chance_iou=float(np.mean(ious)),
                girl_cx=gcx / size_wh[0],
                girl_cy=gcy / size_wh[1],
                area_frac=girl_sum / (size_wh[0] * size_wh[1]),
            )
        rows.append(row)

    csv_path = run_dir / "per_sample.csv"
    fields = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    control = [
        r for r in rows if r["arm"] == "control" and r.get("chance_iou") is not None
    ]
    chance_iou = (
        float(np.mean([r["chance_iou"] for r in control])) if control else float("nan")
    )

    def _corr(a: list[float], b: list[float]) -> float:
        if len(a) < 3 or np.std(a) < 1e-6 or np.std(b) < 1e-6:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def _arm_summary(arm: str) -> dict:
        cond = [r for r in rows if r["arm"] == arm and r.get("iou") is not None]
        per_layout = {}
        for name in LAYOUTS:
            sub = [r for r in cond if r["layout"] == name]
            if sub:
                per_layout[name] = {
                    "iou": float(np.mean([r["iou"] for r in sub])),
                    "center_dist": float(np.mean([r["center_dist"] for r in sub])),
                    "girl_in_paint": float(np.mean([r["girl_in_paint"] for r in sub])),
                    "area_ratio": float(np.mean([r["area_ratio"] for r in sub])),
                    "n": len(sub),
                }
        mean_iou = float(np.mean([r["iou"] for r in cond])) if cond else float("nan")
        return {
            "n_cond": len(cond),
            "found_rate_cond": (
                sum(r["found"] for r in rows if r["arm"] == arm)
                / max(1, sum(1 for r in rows if r["arm"] == arm))
            ),
            "mean_iou": mean_iou,
            "iou_lift": mean_iou - chance_iou,
            "mean_center_dist": (
                float(np.mean([r["center_dist"] for r in cond]))
                if cond
                else float("nan")
            ),
            "mean_girl_in_paint": (
                float(np.mean([r["girl_in_paint"] for r in cond]))
                if cond
                else float("nan")
            ),
            "center_corr_x": _corr(
                [r["paint_cx"] for r in cond], [r["girl_cx"] for r in cond]
            ),
            "center_corr_y": _corr(
                [r["paint_cy"] for r in cond], [r["girl_cy"] for r in cond]
            ),
            # Does girl size track paint size? Inert adapter → ~0.
            "area_corr": _corr(
                [r["paint_area_frac"] for r in cond],
                [r["girl_area_frac"] for r in cond],
            ),
            "mean_bg_psnr": (
                float(np.mean([r["bg_psnr"] for r in cond if "bg_psnr" in r]))
                if any("bg_psnr" in r for r in cond)
                else float("nan")
            ),
            "per_layout": per_layout,
        }

    # Top level = the region adapter (schema-compatible with earlier runs);
    # every paint-scored arm also lands under ``arms`` for side-by-side reads.
    metrics = {
        **_arm_summary("cond"),
        "n_control": len(control),
        "found_rate_control": (
            sum(r["found"] for r in rows if r["arm"] == "control")
            / max(1, sum(1 for r in rows if r["arm"] == "control"))
        ),
        "chance_iou": chance_iou,
    }
    arms = ["cond"] + sorted({r["arm"] for r in rows if _is_easyedit(r["arm"])})
    metrics["arms"] = {a: _arm_summary(a) for a in arms}

    artifacts = ["per_sample.csv"]
    for arm in arms:
        sheet = _contact_sheet(run_dir, plan, layout_masks, size_wh, arm=arm)
        if sheet:
            artifacts.append(sheet)
    return metrics, artifacts


def _contact_sheet(run_dir, plan, layout_masks, size_wh, arm="cond") -> str | None:
    """layouts × prompts grid (seed 0) with the paint contour overlaid."""
    import cv2

    cell = 232
    names = [
        [f"{arm}_p{pi}_{layout}_s0" for pi in range(len(PROMPTS))] for layout in LAYOUTS
    ]
    rows_img = []
    for (layout, _), row_names in zip(LAYOUTS.items(), names):
        cells = []
        for name in row_names:
            p = run_dir / "images" / f"{name}.png"
            if not p.exists():
                return None
            img = np.array(Image.open(p).convert("RGB").resize((cell, cell)))
            paint = cv2.resize(
                layout_masks[layout], (cell, cell), interpolation=cv2.INTER_NEAREST
            )
            contours, _ = cv2.findContours(
                paint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
            cells.append(img)
        rows_img.append(np.concatenate(cells, axis=1))
    sheet = np.concatenate(rows_img, axis=0)
    fname = "contact.png" if arm == "cond" else f"contact_{arm}.png"
    Image.fromarray(sheet).save(run_dir / fname)
    return fname


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="EasyControl region checkpoint (default: newest anima_easycontrol_region*)",
    )
    parser.add_argument(
        "--easyedit_adapter",
        type=Path,
        nargs="*",
        default=None,
        help="Add one `easyedit_<tag>` arm per path: the archived directedit_ec mask recipe "
        "(scripts/edit.py --mask + --easycontrol_mask, delta grammar) with this "
        "EC adapter (e.g. output/ckpt/anima_easycontrol_phash_edit.safetensors) "
        "on the same plates/paint masks, scored by the same paint metrics.",
    )
    parser.add_argument("--seeds", type=int, default=1, help="seeds per prompt")
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="use only the first N prompts (quick sweeps)",
    )
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--attn_mode", type=str, default="flash")
    parser.add_argument(
        "--b_offset",
        type=float,
        default=None,
        help="--easycontrol_b_offset at inference (gate-opening diagnostic)",
    )
    parser.add_argument(
        "--size", nargs=2, type=int, default=[1024, 1024], metavar=("H", "W")
    )
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument(
        "--run_dir", type=Path, default=None, help="resume an existing run dir"
    )
    parser.add_argument("--skip_generate", action="store_true")
    parser.add_argument("--skip_segment", action="store_true")
    args = parser.parse_args()

    if args.adapter is None:
        args.adapter = _latest_region_adapter()
    run_dir = args.run_dir or make_run_dir("region", label=args.label)
    print(
        f"run_dir: {run_dir}\nadapter: {args.adapter}\neasyedit: {args.easyedit_adapter}"
    )

    plan = _samples(args)
    if not args.skip_generate:
        phase_generate(args, run_dir, plan)
    if not args.skip_segment:
        phase_segment(args, run_dir)
    metrics, artifacts = phase_metrics(args, run_dir, plan)

    print(json.dumps({k: v for k, v in metrics.items() if k != "per_layout"}, indent=1))
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=artifacts,
    )
    print(f"result: {run_dir / 'result.json'}")


if __name__ == "__main__":
    main()
