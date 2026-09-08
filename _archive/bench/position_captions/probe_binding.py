#!/usr/bin/env python3
"""Probe A — does the base model hear positional caption clauses?

Renders base-model (no-LoRA) images from prompts written in the dataset's
positional-clause convention — flat tag bag first (both hair colors present,
unordered), then ``On the left, <color> hair. On the right, <color> hair.`` —
and scores binding by splitting each render into left/right halves and asking
the Anima Tagger which hair color wins on each side.

Clause order is counterbalanced within every color pair, so a position-deaf
model converges to 50% side accuracy. Note the two sides of one image are
anti-correlated (both colors usually render; only the assignment varies), so
side accuracy is the honest headline number.

Run through the daemon:
    make daemon-run ARGS="_archive/bench/position_captions/probe_binding.py --label binding"
"""

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from anima_lora import (  # noqa: E402
    GenerationRequest,
    decode_to_pil,
    generate,
    get_generation_settings,
    load_vae,
)
from bench._anima import DEFAULT_NEG, add_common_args, add_model_args  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402

COLOR_PAIRS = [
    ("blonde", "black"),
    ("red", "blue"),
    ("pink", "green"),
    ("white", "purple"),
]

FLAT = (
    "masterpiece, best quality, safe, 2girls, standing, looking at viewer, "
    "full body, simple background, white background"
)


def build_prompt(left: str, right: str) -> str:
    return (
        f"{FLAT}, {left} hair, {right} hair. "
        f"On the left, {left} hair. On the right, {right} hair."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        include_dtype=False,
        include_checkpointing=False,
        include_compile=False,
    )
    add_model_args(parser)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--infer_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    label = args.label or "binding"
    run_dir = make_run_dir(
        "position_captions",
        label=label,
        root="_archive/bench/position_captions/results",
    )
    renders = run_dir / "renders"
    renders.mkdir(exist_ok=True)

    cases = []
    for a, b in COLOR_PAIRS:
        for left, right in ((a, b), (b, a)):
            for seed in range(args.seed, args.seed + args.num_seeds):
                cases.append((f"{left}-{right}_s{seed}", left, right, seed))

    req = GenerationRequest(
        dit=args.dit,
        vae=args.vae,
        text_encoder=args.text_encoder,
        prompt="placeholder",  # overridden per case below
        negative_prompt=DEFAULT_NEG,
        image_size=(args.height, args.width),
        infer_steps=args.infer_steps,
        guidance_scale=args.guidance_scale,
        sampler="euler",
        attn_mode=getattr(args, "attn_mode", None) or "torch",
        lora_weight=None,  # base-model arm — the whole point of the probe
        save_path=str(renders),  # placeholder; we save via decode_to_pil
        device=args.device,
        seed=args.seed,
    )
    base_args = req.to_args()
    base_args.device = device
    gen_settings = get_generation_settings(base_args)

    shared: dict = {}
    latents: dict[str, torch.Tensor] = {}
    for i, (case_id, left, right, seed) in enumerate(cases, 1):
        a = copy.copy(base_args)
        a.prompt = build_prompt(left, right)
        a.seed = seed
        print(f"[{i}/{len(cases)}] gen {case_id}", flush=True)
        latents[case_id] = (
            generate(a, gen_settings, shared_models=shared).detach().to("cpu")
        )

    shared.clear()  # free DiT + text encoder before VAE decode / tagger
    clean_memory_on_device(device)

    vae = load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=64,
        disable_cache=True,
        dtype=torch.bfloat16,
        eval=True,
    )
    images = {}
    for case_id, latent in latents.items():
        pil = decode_to_pil(vae, latent, device)
        pil.save(renders / f"{case_id}.png")
        images[case_id] = pil
    del vae
    clean_memory_on_device(device)

    from anime_tools.tagger.tagger import (  # noqa: E402
        DEFAULT_TAGGER_DIR,
        AnimaTagger,
        ensure_tagger_checkpoint,
    )
    from library.env import resolve_under_home  # noqa: E402

    tagger = AnimaTagger(
        ensure_tagger_checkpoint(resolve_under_home(DEFAULT_TAGGER_DIR)),
        device=args.device,
    )

    rows = []
    for case_id, left, right, seed in cases:
        img = images[case_id]
        w, h = img.size
        halves = {
            "left": img.crop((0, 0, w // 2, h)),
            "right": img.crop((w // 2, 0, w, h)),
        }
        want = {"left": left, "right": right}
        other = {"left": right, "right": left}
        row = {"case": case_id, "left": left, "right": right, "seed": seed}
        for side, half in halves.items():
            scores = tagger.predict(half)["scores"]
            p_want = float(scores[f"{want[side]} hair"])
            p_other = float(scores[f"{other[side]} hair"])
            row[f"{side}_p_want"] = round(p_want, 4)
            row[f"{side}_p_other"] = round(p_other, 4)
            row[f"{side}_correct"] = p_want > p_other
        row["both_correct"] = row["left_correct"] and row["right_correct"]
        rows.append(row)
        print(
            f"{case_id}: left={'ok' if row['left_correct'] else 'MISS'} "
            f"right={'ok' if row['right_correct'] else 'MISS'}",
            flush=True,
        )

    n = len(rows)
    side_acc = sum(r["left_correct"] + r["right_correct"] for r in rows) / (2 * n)
    img_acc = sum(r["both_correct"] for r in rows) / n
    (run_dir / "per_image.json").write_text(json.dumps(rows, indent=2))
    metrics = {
        "n_images": n,
        "side_accuracy": round(side_acc, 4),
        "image_accuracy": round(img_acc, 4),
        "chance_side_accuracy": 0.5,
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=label,
        artifacts=["per_image.json", "renders"],
        device=str(device),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
