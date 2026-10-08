#!/usr/bin/env python
"""Render the turbo pair-set + step-checkpoint family for read-back Phase 0a.

Produces the renders the real-data controls cannot: for gate (i) [teacher >
student on adherence] and gate (iii) [turbo checkpoint spread must not be noise].
All arms render the SAME prompts x seeds so read-back scores are directly
comparable across arms.

Arms:
  * teacher            — base DiT, no adapter, 28-step, cfg 4.0 (the fidelity ref)
  * cdm_{500,1k,2k,3500} — turbo cdm LoRA step checkpoints, 4-step, cfg 1.0
                          (student_steps=4, not per-step-expert). cdm_3500 is the
                          final student for the gate-(i) pair; the four together
                          are the gate-(iii) training-progress family.

Prompts are REAL captions (bench/turbo/real_prompts.txt — real captions are
mandatory for turbo evaluation). Each caption doubles as the
read-back target, so the manifest carries it verbatim.

Writes bench/readback/results/<run>/:
  * <arm>/p{pi}_s{si}.png          — the renders
  * render_manifest.json           — {prompts, seeds, arms:{name:{rank,family,
                                     step,renders:{p{pi}_s{si}:relpath}}}}

Then score with::

    uv run python bench/readback/run_bench.py --render_manifest \
        bench/readback/results/<run>/render_manifest.json --label turbo

Run from anima_lora/ (single GPU; ~10-15 min for the default 16 prompts x 2 seeds)::

    uv run python bench/readback/render_turbo.py --label turbo
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from anima_lora import GenerationRequest, generate, load_vae  # noqa: E402
from library.inference import get_generation_settings  # noqa: E402
from library.inference.output import decode_to_pil  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DIT = "models/diffusion_models/anima-base-v1.0.safetensors"
VAE = "models/vae/qwen_image_vae.safetensors"
TEXT_ENCODER = "models/text_encoders/qwen_3_06b_base.safetensors"
CDM_DIR = "output/ckpt/anima_turbo_cdm"

# (arm name, lora relpath|None, steps, cfg, rank-for-gate-i, family, step)
ARMS = [
    ("teacher", None, 28, 4.0, 0, None, None),
    ("cdm_500", f"{CDM_DIR}/anima_turbo_cdm_500.safetensors", 4, 1.0, None, "cdm", 500),
    ("cdm_1k", f"{CDM_DIR}/anima_turbo_cdm_1k.safetensors", 4, 1.0, None, "cdm", 1000),
    ("cdm_2k", f"{CDM_DIR}/anima_turbo_cdm_2k.safetensors", 4, 1.0, None, "cdm", 2000),
    ("cdm_3500", f"{CDM_DIR}/anima_turbo_cdm_3500.safetensors", 4, 1.0, 1, "cdm", 3500),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--prompts_file", default="bench/turbo/real_prompts.txt")
    p.add_argument("--num_prompts", type=int, default=16, help="Cap prompts (0=all).")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument(
        "--size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W")
    )
    p.add_argument("--label", default="turbo")
    p.add_argument(
        "--out_root",
        default="bench/readback/results",
        help="Renders land in <out_root>/<ts>-<label>/.",
    )
    return p.parse_args()


def _timestamp_dir(out_root: Path, label: str) -> Path:
    # Date.now() is fine here (a plain script, not a resumable workflow).
    import datetime

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    d = out_root / f"{ts}-{label}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompts: List[str] = [
        ln.strip()
        for ln in Path(args.prompts_file).read_text().splitlines()
        if ln.strip()
    ]
    if args.num_prompts:
        prompts = prompts[: args.num_prompts]
    seeds = list(args.seeds)
    run_dir = _timestamp_dir(Path(args.out_root), args.label)
    log.info(
        "rendering %d prompts x %d seeds x %d arms -> %s",
        len(prompts),
        len(seeds),
        len(ARMS),
        run_dir,
    )

    vae = load_vae(
        VAE,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=64,
        disable_cache=True,
        dtype=torch.bfloat16,
        eval=True,
    )

    manifest: Dict = {
        "prompts": {str(i): prompts[i] for i in range(len(prompts))},
        "seeds": seeds,
        "arms": {},
    }

    for arm_name, lora, steps, cfg, rank, family, step in ARMS:
        arm_dir = run_dir / arm_name
        arm_dir.mkdir(exist_ok=True)
        req = GenerationRequest(
            dit=DIT,
            vae=VAE,
            text_encoder=TEXT_ENCODER,
            prompt=prompts[0],  # placeholder; overridden per (prompt, seed) below
            image_size=(args.size[0], args.size[1]),
            infer_steps=steps,
            guidance_scale=cfg,
            sampler="euler",
            lora_weight=[lora] if lora else None,
            lora_multiplier=[1.0] if lora else None,
            save_path=str(arm_dir),  # placeholder; we save via decode_to_pil
        )
        base_args = req.to_args()
        base_args.device = device
        gen_settings = get_generation_settings(base_args)

        shared: Dict = {}
        latents: Dict[str, torch.Tensor] = {}
        for pi, caption in enumerate(prompts):
            for si, seed in enumerate(seeds):
                a = copy.copy(base_args)
                a.prompt = caption
                a.seed = seed
                latent = generate(a, gen_settings, shared_models=shared)
                latents[f"p{pi}_s{si}"] = latent.detach().to("cpu")
                log.info("  [%s] p%d s%d done", arm_name, pi, seed)
        # free the DiT before decoding (lazy-load discipline)
        shared.pop("model", None)
        clean_memory_on_device(device)

        renders: Dict[str, str] = {}
        for key, latent in latents.items():
            pil = decode_to_pil(vae, latent, device)
            rel = f"{arm_name}/{key}.png"
            pil.save(run_dir / rel)
            renders[key] = str((run_dir / rel).resolve())
        clean_memory_on_device(device)

        manifest["arms"][arm_name] = {
            "rank": rank,
            "family": family,
            "step": step,
            "steps": steps,
            "cfg": cfg,
            "lora": lora,
            "renders": renders,
        }
        (run_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2))
        log.info("[%s] %d renders saved", arm_name, len(renders))

    log.info("done -> %s/render_manifest.json", run_dir)


if __name__ == "__main__":
    main()
