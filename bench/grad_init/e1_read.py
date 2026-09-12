#!/usr/bin/env python3
"""E1 — does the ``lora_down`` seed move the render?

The render-level half of ``docs/proposal/grad_basis_init.md`` §E1. Given the
paired arm checkpoints (same artist, same ``--seed``, ``--deterministic
--paired_step_rng``, one knob apart), this renders ONE prompt set with ONE seed
per row under every arm and reports two things:

1. **arm-vs-arm PE cos** — the ruler. Two arms that land the same image are not
   separated by their init, whatever their loss curves did. Read against the
   recorded rungs of this ruler (memory ``project_cjk_dit_line_2026_09_05``):
   unrelated images ≈ 0.935, LoRA-vs-base ≈ 0.984, α32-vs-α128 ≈ 0.982. This
   run measures its own unrelated floor (cos across *different* prompts) so the
   rungs are comparable on this prompt set.
2. **CMMD vs the artist's real images** (PE-Core pooled, ``library.training.cmmd``)
   — which arm sits closer to the artist's distribution, plus the real-vs-real
   noise floor below which no arm delta is interpretable.

**Caveat carried in the output**: E1's runs train on the artist's whole folder
(``sample_ratio = 1.0``), so every prompt and every reference image is a
*member*. The CMMD column therefore reads "reconstruction fidelity", NOT
generalization — for the latter see ``bench/memorization/generalize.py``, which
replays a real member/holdout split. The load-bearing E1 read is the pairwise
cos plus the contact sheet.

Usage::

    make daemon-run ARGS="bench/grad_init/e1_read.py \
        --adapters output/ckpt/e1_grad_init/e1_{kaiming,weightsvd,gradsvd,basisfile,minsnr}.safetensors \
        --artist aak --num_prompts 12 --label e1"
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from random import Random

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from anima_lora import default_checkpoints, load_vae  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from bench.memorization.eyeball import _caption_for, _pick  # noqa: E402
from bench.memorization.generalize import (  # noqa: E402
    Prompt,
    _pil_to_minus1to1,
    encode_pool,
    render_model,
)
from library.env import resolve_under_home  # noqa: E402
from library.inference.output import decode_latent  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402
from library.training.cmmd import cmmd_from_pools  # noqa: E402
from library.vision.encoder import load_pe_encoder  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
TILE_W = 384


def artist_images(resized_root: Path, artist: str) -> list[Path]:
    root = resized_root / artist
    if not root.is_dir():
        raise SystemExit(f"no resized tree for artist {artist!r} at {root}")
    paths = sorted(
        p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )
    if not paths:
        raise SystemExit(f"no images under {root}")
    return paths


def to_prompt(path: Path) -> Prompt | None:
    caption = _caption_for(path)
    if not caption:
        return None
    with Image.open(path) as im:
        w, h = im.size
    return Prompt(path.stem, caption, path, (h, w))


def contact_sheet(
    out_path: Path, labels: list[str], images_by_label: dict[str, list[Path]]
) -> None:
    """rows = prompts, cols = arms; each tile downscaled to TILE_W."""
    n_rows = len(next(iter(images_by_label.values())))
    tiles: dict[str, list[Image.Image]] = {}
    tile_h = 0
    for label in labels:
        col = []
        for p in images_by_label[label]:
            im = Image.open(p).convert("RGB")
            h = max(1, round(im.height * TILE_W / im.width))
            col.append(im.resize((TILE_W, h), Image.LANCZOS))
            tile_h = max(tile_h, h)
        tiles[label] = col
    sheet = Image.new("RGB", (TILE_W * len(labels), tile_h * n_rows), "white")
    for c, label in enumerate(labels):
        for r, im in enumerate(tiles[label]):
            sheet.paste(im, (c * TILE_W, r * tile_h))
    sheet.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--artist", required=True)
    ap.add_argument(
        "--resized_root", default="post_image_dataset/resized", help="repo-relative"
    )
    ap.add_argument("--with_base", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--num_prompts", type=int, default=12)
    ap.add_argument("--num_refs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--multiplier", type=float, default=1.0)
    ap.add_argument("--pick_seed", type=int, default=0)
    ap.add_argument(
        "--compile_blocks", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--label", default="e1")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = Random(args.pick_seed)

    paths = artist_images(Path(resolve_under_home(args.resized_root)), args.artist)
    prompts = [
        pr for pr in (to_prompt(p) for p in _pick(paths, args.num_prompts, rng)) if pr
    ]
    if len(prompts) < 2:
        raise SystemExit(
            f"{args.artist}: only {len(prompts)} captioned images among the picked "
            "set — the caption sidecars beside the resized images are what "
            "training read, so an empty set means a caption stage never ran."
        )
    used = {p.ref_path for p in prompts}
    refs = _pick([p for p in paths if p not in used], args.num_refs, rng)
    print(
        f"artist={args.artist} prompts={len(prompts)} refs={len(refs)} "
        f"(all members: E1 trains the whole folder)",
        flush=True,
    )

    models: list[tuple[str, str | None]] = []
    if args.with_base:
        models.append(("base", None))
    models += [(Path(a).stem, a) for a in args.adapters]

    # Phase 1 — DiT sampling, identical prompts and per-row seeds per arm.
    latents = {
        label: render_model(args, label, adapter, prompts, device)
        for label, adapter in models
    }

    # Phase 2 — one VAE pass, and keep the PNGs for the eyeball.
    run_dir = make_run_dir("grad_init", args.label)
    ckpt = default_checkpoints()
    vae = load_vae(
        ckpt.vae, device="cpu", disable_mmap=True, dtype=torch.bfloat16, eval=True
    )
    pixels: dict[str, list[torch.Tensor]] = {}
    png_paths: dict[str, list[Path]] = {}
    for label, lats in latents.items():
        out_dir = run_dir / "renders" / label
        out_dir.mkdir(parents=True, exist_ok=True)
        pxs, pngs = [], []
        for p, lat in zip(prompts, lats):
            px = decode_latent(vae, lat, device).to("cpu")
            pxs.append(px.unsqueeze(0) if px.dim() == 3 else px)
            arr = ((pxs[-1][0].float().clamp(-1, 1) + 1) * 127.5).round()
            img = Image.fromarray(
                arr.permute(1, 2, 0).to(torch.uint8).numpy(), mode="RGB"
            )
            path = out_dir / f"{p.stem}.png"
            img.save(path)
            pngs.append(path)
        pixels[label] = pxs
        png_paths[label] = pngs
    del vae, latents
    clean_memory_on_device(device)

    # Phase 3 — one PE-Core pass over every render plus the real reference pool.
    bundle = load_pe_encoder(device)
    pools = {label: encode_pool(bundle, pxs) for label, pxs in pixels.items()}
    ref_pool = encode_pool(bundle, [_pil_to_minus1to1(Image.open(p)) for p in refs])
    del bundle
    clean_memory_on_device(device)

    labels = [label for label, _a in models]

    # (1) the ruler: matched-row cos between arms, and this prompt set's own
    # unrelated floor (cos across different rows of the same arm).
    pairwise: dict[str, float] = {}
    for a, b in itertools.combinations(labels, 2):
        pairwise[f"{a}|{b}"] = float((pools[a] * pools[b]).sum(dim=1).mean())
    unrelated = []
    for label in labels:
        P = pools[label]
        G = P @ P.T
        n = G.shape[0]
        unrelated.append(float((G.sum() - G.diag().sum()) / (n * (n - 1))))
    floor_unrelated = sum(unrelated) / len(unrelated)

    # (2) distribution distance to the artist's real images + its noise floor.
    half = len(ref_pool) // 2
    cmmd_floor = cmmd_from_pools(ref_pool[:half], ref_pool[half:])
    cmmd = {label: cmmd_from_pools(ref_pool, pools[label]) for label in labels}

    contact_sheet(run_dir / "contact_sheet.png", labels, png_paths)

    lines = [
        "# E1 — does the lora_down seed move the render?",
        "",
        f"- artist `{args.artist}` · {len(prompts)} prompts (paired seeds) · "
        f"{len(refs)} real refs · {args.steps} steps · cfg {args.cfg} · "
        f"multiplier {args.multiplier}",
        "- **every prompt and ref is a training member** (E1 runs at "
        "sample_ratio=1.0), so `cmmd` reads reconstruction fidelity, not "
        "generalization.",
        "",
        "## arm-vs-arm PE cos (matched prompt + seed)",
        "",
        f"Unrelated-row floor on this prompt set: **{floor_unrelated:.4f}** "
        f"(recorded rungs: unrelated ≈ 0.935, LoRA-vs-base ≈ 0.984, "
        f"α32-vs-α128 ≈ 0.982).",
        "",
        "| pair | PE cos |",
        "|---|---|",
    ]
    for k, v in sorted(pairwise.items(), key=lambda kv: kv[1]):
        lines.append(f"| {k.replace('|', ' vs ')} | {v:.4f} |")
    lines += [
        "",
        "## CMMD vs the artist's real images",
        "",
        f"real-vs-real noise floor **{cmmd_floor:.4f}** — smaller deltas are noise.",
        "",
        "| arm | cmmd ↓ |",
        "|---|---|",
    ]
    for label, v in sorted(cmmd.items(), key=lambda kv: kv[1]):
        lines += [f"| {label} | {v:.4f} |"]
    lines += [
        "",
        f"Contact sheet (rows = prompts, cols = {', '.join(labels)}): "
        f"`{(run_dir / 'contact_sheet.png').relative_to(REPO_ROOT)}`",
        "",
    ]
    report = "\n".join(lines)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report, flush=True)

    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        artifacts=[run_dir / "contact_sheet.png", run_dir / "report.md"],
        device=device,
        metrics={
            "artist": args.artist,
            "labels": labels,
            "n_prompts": len(prompts),
            "n_refs": len(refs),
            "steps": args.steps,
            "cfg": args.cfg,
            "multiplier": args.multiplier,
            "pe_cos_pairwise": pairwise,
            "pe_cos_unrelated_floor": floor_unrelated,
            "cmmd": cmmd,
            "cmmd_real_vs_real_floor": cmmd_floor,
            "prompt_stems": [p.stem for p in prompts],
            "adapters": list(args.adapters),
        },
    )
    print(json.dumps({"run_dir": str(run_dir)}), flush=True)


if __name__ == "__main__":
    main()
