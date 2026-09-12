#!/usr/bin/env python3
"""E1 — blind A/B of the ``lora_down`` seed arms on GENERAL prompts.

The member-caption read (``e1_read.py``) shows every arm landing a *different*
image (arm-vs-arm PE cos 0.89–0.95 on a 0.73 floor) with CMMD below the
real-vs-real noise floor for all of them — "the seed picks a mode" is visible,
"which mode is better" is not. This renders the same arm checkpoints on a
12-row general prompt set (``@aak`` trigger, SFW-leaning, same shape as the
unmask eval rows) and composes them into the blind-pairs protocol
(``project/cjk_aware_anima/probes/blind_pairs.py``), one DIRECT pairing per
set against the shipped ``weight_svd`` control — never chain sets.

Sets (default): WSVD vs BASIS (load-bearing), WSVD vs GSVD, WSVD vs KAIMING
(does init matter at all), WSVD vs MINSNR — each at its own fresh seed pair, since
the control recurs in every set and a repeated image would unblind it. Grade in the private repo, then::

    .venv/bin/python project/cjk_aware_anima/probes/blind_pairs.py score --set <set>

Usage (one daemon command job; ~10 inference.py loads)::

    make daemon-run ARGS="--queue bench/grad_init/e1_blind.py --push"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
BLIND = REPO / "project" / "cjk_aware_anima" / "probes" / "blind_pairs.py"

ARMS = {
    "KAIMING": "output/ckpt/e1_grad_init/e1_kaiming.safetensors",
    "WSVD": "output/ckpt/e1_grad_init/e1_weightsvd.safetensors",
    "GSVD": "output/ckpt/e1_grad_init/e1_gradsvd.safetensors",
    "BASIS": "output/ckpt/e1_grad_init/e1_basisfile.safetensors",
    "MINSNR": "output/ckpt/e1_grad_init/e1_minsnr.safetensors",
}
CONTROL = "WSVD"
# Every set carries its OWN seeds: the control arm sits in all four pairings,
# and an image that recurs across sets would identify it (the re-blind rule —
# a grader who has seen an arm at a seed needs fresh seeds for any new pairing).
DEFAULT_SETS = [
    ("s22_E1_WSVD_vs_BASIS", ["WSVD", "BASIS"], [42, 7]),
    ("s23_E1_WSVD_vs_GSVD", ["WSVD", "GSVD"], [1, 2]),
    ("s24_E1_WSVD_vs_KAIMING", ["WSVD", "KAIMING"], [3, 4]),
    ("s25_E1_WSVD_vs_MINSNR", ["WSVD", "MINSNR"], [5, 6]),
]

# Same render recipe as project/cjk_aware_anima/run_unmask_r2.py so the
# recorded rungs / seed-twin floor of the blind protocol carry over.
INFER_BASE = [
    "inference.py",
    "--dit",
    "models/diffusion_models/anima-base-v1.0.safetensors",
    "--text_encoder",
    "models/text_encoders/qwen_3_06b_base.safetensors",
    "--vae",
    "models/vae/qwen_image_vae.safetensors",
    "--vae_chunk_size",
    "64",
    "--vae_disable_cache",
    "--attn_mode",
    "flash",
    "--lora_multiplier",
    "1.0",
    "--negative_prompt",
    "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia",
    "--image_size",
    "1024",
    "1024",
    "--infer_steps",
    "28",
    "--flow_shift",
    "3.0",
    "--sampler",
    "euler",
    "--guidance_scale",
    "4.0",
]


def run(tag: str, cmd: list[str]) -> None:
    print(f"\n=== {tag}:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def n_rows(prompts: Path) -> int:
    return len(
        [ln for ln in prompts.read_text(encoding="utf-8").splitlines() if ln.strip()]
    )


def rendered(eval_dir: Path, arm: str, seed: int, rows: int) -> bool:
    return len(list((eval_dir / f"arm{arm}_s{seed}").glob("*.png"))) == rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="override every set's seeds (default: each set's own fresh seeds)",
    )
    ap.add_argument("--prompts", default="bench/grad_init/e1_general_prompts.txt")
    ap.add_argument("--eval_dir", default="output/tests/e1_grad_init_eval")
    ap.add_argument(
        "--sets",
        nargs="*",
        default=None,
        help="set names to build (default: every DEFAULT_SETS whose arms were rendered)",
    )
    ap.add_argument("--skip_render", action="store_true")
    ap.add_argument("--skip_sets", action="store_true")
    ap.add_argument("--push", action="store_true")
    o = ap.parse_args()

    prompts = (REPO / o.prompts).resolve()
    eval_dir = (REPO / o.eval_dir).resolve()
    rows = n_rows(prompts)
    for arm in o.arms:
        if not (REPO / ARMS[arm]).exists():
            sys.exit(f"missing LoRA for arm {arm}: {ARMS[arm]}")

    wanted = (
        DEFAULT_SETS if o.sets is None else [s for s in DEFAULT_SETS if s[0] in o.sets]
    )
    wanted = [
        (name, arms, o.seeds or seeds)
        for name, arms, seeds in wanted
        if all(a in o.arms for a in arms)
    ]
    needed = sorted({(a, s) for _, arms, seeds in wanted for a in arms for s in seeds})

    if not o.skip_render:
        for arm, seed in needed:
            if rendered(eval_dir, arm, seed, rows):
                print(f"=== arm{arm} s{seed} already rendered, skip", flush=True)
                continue
            run(
                f"gen arm{arm} s{seed}",
                [
                    PY,
                    *INFER_BASE,
                    "--lora_weight",
                    ARMS[arm],
                    "--from_file",
                    str(prompts),
                    "--seed",
                    str(seed),
                    "--save_path",
                    str(eval_dir / f"arm{arm}_s{seed}"),
                ],
            )

    if o.skip_sets:
        return
    for name, arms, seeds in wanted:
        run(
            f"blind {name}",
            [
                PY,
                str(BLIND),
                "make",
                "--set",
                name,
                "--arms",
                *arms,
                "--seeds",
                *map(str, seeds),
                "--eval_dir",
                str(eval_dir),
                "--prompts",
                str(prompts),
                "--overwrite",
                *(["--push"] if o.push else []),
            ],
        )


if __name__ == "__main__":
    main()
