#!/usr/bin/env python3
"""Batch MOSS i2i sanitize through a running ComfyUI, re-rolling the seed where text survives.

Each image goes through the toolkit's sanitize graph (``moss-i2i-sanitize.json``
over the API: MOSS-Image Loader → MOSS i2i Reference → Caption Helper ``sanitize``
→ MOSS Spectrum KSampler). The AnimeText detector then reads every output; an
output with a text block left in it is rendered again with the next seed, up to
``--retries`` times. The pick per image is the attempt with the fewest blocks
(the earlier seed on a tie), copied to ``<comfy>/output/<name>/best/{stem}.png``,
and every attempt's boxes go to ``<name>/report.jsonl``.

Attempts are saved as ``{stem}_s{seed}_00001_.png`` and an attempt already on
disk is not queued again, so a rerun resumes. The detector runs on the CPU by
default, beside ComfyUI's hold on the card.

    python scripts/toolkits/comfy_sanitize.py post_image_dataset/resized/sincos \
        --name sanitize_sincos_int8 --text_only
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

from library.env import resolve_under_home

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
LORA = "moss_i2i_sanitize_idt_w0003_late_lr5e5_e8.safetensors"
DIT_INT8 = "moss_image_base_int8_gptq.safetensors"


def build_prompt(args, image: str, seed: int, prefix: str) -> dict:
    return {
        "17": {
            "class_type": "MOSSImageLoader",
            "inputs": {
                "dit": args.dit,
                "weight_dtype": "default",
                "text_encoder": "int8/qwen35_9b_int8_convrot.safetensors",
                "vae": "MOSS-Image-Base",
                "text_device": "default",
                "int8_matmul": True,
                "stream": args.stream,
            },
        },
        "23": {"class_type": "LoadImage", "inputs": {"image": image}},
        "3": {
            "class_type": "MOSSi2iReference",
            "inputs": {
                "model": ["17", 0],
                "vae": ["17", 2],
                "image": ["23", 0],
                "lora_name": args.lora,
                "strength": 1.0,
                "resolution": args.resolution,
                "ref_scale": 1,
                "task": "auto",
                "ref_cache": "gpu",
                # Patching requantizes an INT8 DiT's linears and loses the GPTQ weights.
                "bypass": "int8" in args.dit,
            },
        },
        "22": {
            "class_type": "MOSSCaptionHelper",
            "inputs": {"caption": "", "task": "sanitize", "resolution": args.resolution},
        },
        "24": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["17", 1], "text": ["22", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["17", 1], "text": ""}},
        "5": {
            "class_type": "MossSpectrumKSampler",
            "inputs": {
                "model": ["3", 0],
                "seed": seed,
                "steps": args.steps,
                "cfg": args.cfg,
                "sampler_name": "er_sde_cns",
                "scheduler": "normal",
                "positive": ["24", 0],
                "negative": ["4", 0],
                "latent_image": ["3", 1],
                "denoise": 1.0,
                "shift": 5.0,
                "uncond_skip_layers": "22",
                "spectrum": True,
                "flex_window": 0.5,
                "compile": True,
            },
        },
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["17", 2]}},
        "6": {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": prefix}},
    }


def _post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    try:
        return json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"ComfyUI refused the prompt: {e.read().decode()[:2000]}") from e


def _finished(url: str, prompt_id: str) -> dict | None:
    with urllib.request.urlopen(f"{url}/history/{prompt_id}") as r:
        h = json.load(r).get(prompt_id)
    if h and h.get("status", {}).get("completed") is not None:
        if h["status"].get("status_str") == "error" or not h["status"]["completed"]:
            return {"error": h["status"].get("messages", [])[-1:]}
        return h
    return None


def select_images(src: Path, text_only: bool) -> list[Path]:
    images = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not text_only:
        return images
    from anime_tools.captions.position_clauses import has_text_clauses

    keep = []
    for p in images:
        txt = p.with_suffix(".txt")
        if txt.is_file() and has_text_clauses(txt.read_text(encoding="utf-8")):
            keep.append(p)
    return keep


def render_round(args, jobs: list[tuple[str, int]], out_dir: Path, input_sub: str) -> dict:
    """Queue the (stem, seed) attempts not yet on disk, wait for them, return {(stem, seed): path}."""
    done, pending = {}, {}
    for stem, seed in jobs:
        hit = sorted(out_dir.glob(f"{stem}_s{seed}_*.png"))
        if hit:
            done[(stem, seed)] = hit[0]
            continue
        ext = args.ext[stem]
        r = _post(
            f"{args.url}/prompt",
            {"prompt": build_prompt(args, f"{input_sub}/{stem}{ext}", seed, f"{args.name}/{stem}_s{seed}")},
        )
        pending[r["prompt_id"]] = (stem, seed)
    total = len(pending)
    if total:
        print(f"queued {total} renders ({len(done)} already on disk)", flush=True)
    t0 = time.time()
    while pending:
        time.sleep(2)
        for pid in list(pending):
            h = _finished(args.url, pid)
            if h is None:
                continue
            stem, seed = pending.pop(pid)
            if "error" in h:
                print(f"  {stem} s{seed}: failed {h['error']}", flush=True)
                continue
            img = h["outputs"]["6"]["images"][0]
            done[(stem, seed)] = out_dir.parent / img["subfolder"] / img["filename"]
            n = total - len(pending)
            print(f"  [{n}/{total}] {stem} s{seed}  {(time.time() - t0) / n:.1f} s/img", flush=True)
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images_dir", help="folder of pages (with .txt captions for --text_only)")
    ap.add_argument("--name", required=True, help="ComfyUI output subfolder")
    ap.add_argument("--text_only", action="store_true", help="only pages whose caption has an OCR text clause")
    ap.add_argument("--comfy_dir", default="../comfy")
    ap.add_argument("--url", default="http://localhost:8188")
    ap.add_argument("--lora", default=LORA)
    ap.add_argument("--dit", default=DIT_INT8, help="MOSS-Image-Base for the bf16 DiT")
    ap.add_argument("--stream", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--retries", type=int, default=1, help="extra seeds for an output that still has text")
    ap.add_argument("--conf", type=float, default=0.25, help="AnimeText score floor")
    ap.add_argument("--device", default="cpu", help="AnimeText device")
    args = ap.parse_args()

    src = resolve_under_home(args.images_dir)
    comfy = resolve_under_home(args.comfy_dir)
    images = select_images(src, args.text_only)
    if not images:
        raise SystemExit(f"no images selected in {src}")
    input_sub = f"{args.name}_in"
    in_dir = comfy / "input" / input_sub
    out_dir = comfy / "output" / args.name
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in images:
        if not (in_dir / p.name).exists():
            shutil.copy2(p, in_dir / p.name)
    args.ext = {p.stem: p.suffix for p in images}
    print(f"{len(images)} pages from {src}", flush=True)

    import cv2

    from anime_tools.ocr.animetext import AnimeTextDetector

    det = AnimeTextDetector.load(device=args.device, conf=args.conf)

    attempts: dict[str, list[dict]] = {p.stem: [] for p in images}
    todo = [(p.stem, args.seed) for p in images]
    for round_ in range(args.retries + 1):
        if not todo:
            break
        print(f"round {round_}: {len(todo)} pages", flush=True)
        done = render_round(args, todo, out_dir, input_sub)
        retry = []
        for stem, seed in todo:
            path = done.get((stem, seed))
            if path is None:
                continue
            boxes = det.detect_scored(cv2.imread(str(path)))
            attempts[stem].append(
                {"seed": seed, "file": path.name, "boxes": [[*map(int, b[:4]), round(float(b[4]), 3)] for b in boxes]}
            )
            if boxes:
                retry.append((stem, seed + 1))
        print(f"round {round_}: text left on {len(retry)} pages", flush=True)
        todo = retry

    best_dir = out_dir / "best"
    best_dir.mkdir(exist_ok=True)
    clean = 0
    with open(out_dir / "report.jsonl", "w", encoding="utf-8") as f:
        for stem, tries in attempts.items():
            if not tries:
                continue
            pick = min(tries, key=lambda t: len(t["boxes"]))
            clean += not pick["boxes"]
            shutil.copy2(out_dir / pick["file"], best_dir / f"{stem}.png")
            f.write(json.dumps({"stem": stem, "pick": pick["seed"], "attempts": tries}, ensure_ascii=False) + "\n")
    print(f"done: {clean}/{len(attempts)} pages with no text block left; picks in {best_dir}", flush=True)


if __name__ == "__main__":
    main()
