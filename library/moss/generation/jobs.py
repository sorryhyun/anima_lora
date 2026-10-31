"""What a generate request renders: its jobs, and each job's resume fingerprint."""

from __future__ import annotations

import dataclasses
import json
import math
import re
from pathlib import Path

from library.env import resolve_under_home
from library.moss.requests import GenerateRequest


def parse_skip(value) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return tuple(int(v) for v in str(value).split(",") if v.strip())


def build_jobs(req: GenerateRequest) -> list[dict]:
    base = dict(
        negative=req.negative_prompt,
        width=req.width,
        height=req.height,
        cfg=req.cfg,
        skip=parse_skip(req.uncond_skip),
    )
    if req.example:
        jobs = []
        for n, path in enumerate(
            filter(None, (p.strip() for p in req.example.split(",")))
        ):
            path = resolve_under_home(path)
            p = json.loads(path.read_text(encoding="utf-8"))
            p = p.get("parameters", p)
            jobs.append(
                dict(
                    # Gallery entries are <id>/parameters.json: key by the folder.
                    key=f"{n:03d}_space{path.parent.name if path.stem == 'parameters' else path.stem}",
                    prompt=p["prompt"],
                    negative=p.get("negative_prompt", ""),
                    width=int(p["width"]),
                    height=int(p["height"]),
                    cfg=float(p.get("cfg", p.get("guidance_scale", req.cfg))),
                    seed=int(p["seed"]),
                    skip=parse_skip(p.get("uncond_skip_layers", req.uncond_skip)),
                    steps=int(p.get("steps", p.get("num_inference_steps", req.steps))),
                    shift=float(p.get("shift", req.shift)),
                )
            )
        return jobs
    if req.prompts_file:
        source = resolve_under_home(req.prompts_file)
        prompts = [
            ln.strip()
            for ln in source.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    else:
        source, prompts = "prompt", [req.prompt.strip()] if req.prompt.strip() else []
    if not prompts:
        if not req.ref_image:
            raise SystemExit(f"no prompts in {source}")
        prompts = [""]  # the reference alone
    if req.prompt_format != "text":
        from library.moss.caching.sources import caption_converter

        convert = caption_converter(req.prompt_format)
        prompts = [convert(p) if p else "" for p in prompts]
    jobs = [
        dict(
            base,
            key=f"{i:03d}",
            prompt=p,
            seed=req.seed + i,
            steps=req.steps,
            shift=req.shift,
        )
        for i, p in enumerate(prompts)
    ]
    return _with_refs(req, jobs) if req.ref_image else jobs


def ref_images(spec: str) -> list[Path]:
    """``ref_image``: one image, or every image directly in a folder."""
    from library.moss.scan import IMAGE_EXTS

    path = resolve_under_home(spec)
    if path.is_dir():
        found = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not found:
            raise SystemExit(f"no images in {path}")
        return found
    if not path.is_file():
        raise SystemExit(f"no reference image at {path}")
    return [path]


def _with_refs(req: GenerateRequest, jobs: list[dict]) -> list[dict]:
    """Each job once per reference image, sized to it: its aspect, area capped at
    ``width × height``, multiples of 16."""
    from PIL import Image

    from library.moss.caching.images import target_size

    if req.example:
        raise SystemExit("ref_image does not combine with example")
    resolution = round(math.sqrt(req.width * req.height))
    out = []
    for ref in ref_images(req.ref_image):
        with Image.open(ref) as im:
            width, height = target_size(*im.size, resolution)
        tag = re.sub(r"[^\w.-]+", "_", ref.stem)[:48]
        for j in jobs:
            out.append(
                dict(
                    j,
                    key=f"{j['key']}_{tag}",
                    width=width,
                    height=height,
                    ref=str(ref),
                    ref_mtime=ref.stat().st_mtime_ns,
                    ref_scale=req.ref_scale,
                    ref_vision=req.ref_vision,
                )
            )
    return out


def fingerprint(req: GenerateRequest, job: dict) -> dict:
    """What a saved latent was made from; a resume reuses it only on a match."""
    spectrum = {
        k: v for k, v in dataclasses.asdict(req).items() if k.startswith("spectrum")
    }
    # Stamped only off bf16, so bf16 latents without the field still match.
    te = {} if req.text_encoder == "bf16" else {"text_encoder": req.text_encoder}
    return dict(
        {k: v for k, v in job.items() if k != "skip"},
        **te,
        uncond_skip=list(job["skip"]),
        model=req.model,
        lora=str(resolve_under_home(req.lora)) if req.lora else None,
        lora_multiplier=req.lora_multiplier if req.lora else None,
        batched=req.batched,
        sampler=req.sampler,
        spectrum=spectrum if req.spectrum else None,
        dave=(
            dict(spec=req.dave, strength=req.dave_strength, tau=req.dave_tau)
            if req.dave
            else None
        ),
        **({"cns": dict(spec=req.cns, strength=req.cns_strength)} if req.cns else {}),
        **({"stream": req.stream} if req.stream != "bf16" else {}),
        **({"int8": req.int8_file or req.int8} if req.int8 != "off" else {}),
    )
