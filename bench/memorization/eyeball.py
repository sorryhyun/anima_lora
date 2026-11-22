#!/usr/bin/env python3
"""Eyeball contact sheet — the human validation layer over the loss-gap verdict.

``loss_gap.py`` scores member-specific signal in an adapter's WEIGHTS and
``probe.py`` scores feature-space snapping; this bench renders the grid a human
reads for "is this LoRA usable / is it visibly xeroxing":

  rows (default 6 prompts, all reused verbatim from training captions):
    * target artist, MEMBER caption   — the memorization check: does the
      render collapse back to its exact source frame? (compare col 0)
    * target artist, HOLDOUT caption  — same-artist caption the run never
      trained on: a healthy style LoRA renders a fresh plausible image.
    * 2 other artists x 2 captions    — never-seen content: does the adapter
      follow the caption (style transfer) or drag renders toward its own
      training frames (severe memorization / mode collapse)?
  cols: [source image | seed_0 | seed_1 | seed_2]

Membership is reconstructed with the same trainer-blueprint replay as
``loss_gap.py`` (snapshot knobs are authoritative), so MEMBER/HOLDOUT labels
match the MIA's definition exactly. Each prompt renders at its source image's
own bucket size (free-fit native shapes), so the reference column is directly
comparable.

Usage (from anima_lora/; ~18 generations, one DiT load)::

    .venv/bin/python bench/memorization/eyeball.py \
        --adapter output/ckpt/bench_sincos_half_plain.safetensors \
        [--other_artists aak abmayo] [--seeds 0 1 2] [--plan_only]

Artifacts: ``contact_sheet.png`` (the deliverable), ``samples/`` (individual
PNGs with generation metadata), ``summary.md``, ``result.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from random import Random


import torch
from PIL import Image, ImageDraw, ImageFont

from anima_lora import (
    GenerationRequest,
    decode_to_pil,
    default_checkpoints,
    generate,
    get_generation_settings,
    load_vae,
)
from bench._common import make_run_dir, write_result
from bench.memorization.loss_gap import (
    _read_snapshot_knobs,
    build_items,
    setup_strategies,
)
from bench.memorization.probe import build_training_args
from library.inference import load_shared_models
from library.runtime.device import clean_memory_on_device

REPO_ROOT = Path(__file__).resolve().parents[2]

RESIZED_DIR = REPO_ROOT / "post_image_dataset" / "resized"


# ---------------------------------------------------------------------------
# Prompt-row selection
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Row:
    artist: str
    stem: str
    tag: str  # MEMBER / HOLDOUT / OTHER
    caption: str
    ref_path: Path


def _caption_for(image_path: Path) -> str | None:
    """The canonical caption sidecar next to a resized image (not .variants)."""
    txt = image_path.with_suffix(".txt")
    if not txt.exists():
        return None
    caption = txt.read_text(encoding="utf-8").strip()
    return caption or None


def _pick(items: list, n: int, rng: Random) -> list:
    if len(items) <= n:
        return list(items)
    return [items[i] for i in sorted(rng.sample(range(len(items)), n))]


def target_rows(args, target: str, rng: Random) -> list[Row]:
    """MEMBER + same-artist HOLDOUT captions via the trainer-blueprint replay
    (mirrors loss_gap.py main: snapshot knobs are authoritative, including
    absences — never fall back to today's method TOML for a snapshotted run)."""
    snap = _read_snapshot_knobs(args.adapter)
    if snap:
        print(f"snapshot knobs: {snap}", flush=True)
    targs = build_training_args(args.method, args.preset, {})
    for k in ("validation_seed", "validation_split", "validation_split_num"):
        if k in snap:
            setattr(targs, k, snap[k])
    if snap:
        targs.path_pattern = snap.get("path_pattern")
        run_sr = snap.get("sample_ratio")
        run_shard = snap.get("artists_shard")
    else:
        run_sr = getattr(targs, "sample_ratio", None)
        run_shard = getattr(targs, "artists_shard", None)

    te_strategy = setup_strategies(targs)
    member_items = build_items(
        targs, te_strategy, sample_ratio=run_sr, artists_shard=run_shard
    )
    full_items = build_items(
        targs, te_strategy, sample_ratio=1.0, artists_shard=None, include_val=True
    )
    member_paths = {it[0].absolute_path for it in member_items}

    def in_target(it) -> bool:
        return Path(it[0].absolute_path).parent.name == target

    members = [it for it in member_items if in_target(it)]
    holdouts = [
        it
        for it in full_items
        if in_target(it) and it[0].absolute_path not in member_paths
    ]
    if not members:
        raise RuntimeError(
            f"no member items for artist {target!r} — check --target_artist "
            "against the run's path_pattern"
        )
    print(
        f"target {target!r}: members={len(members)} holdouts={len(holdouts)}",
        flush=True,
    )

    n = args.prompts_per_artist
    n_hold = min(len(holdouts), n // 2)
    picked = [(it, "MEMBER") for it in _pick(members, n - n_hold, rng)]
    picked += [(it, "HOLDOUT") for it in _pick(holdouts, n_hold, rng)]

    rows: list[Row] = []
    for it, tag in picked:
        path = Path(it[0].absolute_path)
        caption = _caption_for(path) or getattr(it[0], "caption", None)
        if not caption:
            raise RuntimeError(f"no caption sidecar for {path}")
        rows.append(Row(target, path.stem, tag, caption, path))
    return rows


def other_artist_rows(artists: list[str], n_per: int, rng: Random) -> list[Row]:
    """Captions the adapter never saw, read straight off the resized tree."""
    rows: list[Row] = []
    for artist in artists:
        adir = RESIZED_DIR / artist
        pairs = [
            (png, cap)
            for png in sorted(adir.glob("*.png"))
            if (cap := _caption_for(png))
        ]
        if len(pairs) < n_per:
            raise RuntimeError(f"artist dir {adir} has {len(pairs)} captioned images")
        for png, cap in _pick(pairs, n_per, rng):
            rows.append(Row(artist, png.stem, "OTHER", cap, png))
    return rows


def default_other_artists(target: str, n: int) -> list[str]:
    """Deterministic fallback: the n biggest artist dirs besides the target."""
    counts = [
        (len(list(d.glob("*.png"))), d.name)
        for d in sorted(RESIZED_DIR.iterdir())
        if d.is_dir() and d.name != target
    ]
    counts.sort(reverse=True)
    return [name for _c, name in counts[:n]]


# ---------------------------------------------------------------------------
# Generation (one shared DiT/TE load, VAE decode at the end)
# ---------------------------------------------------------------------------


def render_all(args, rows: list[Row], samples_dir: Path) -> list[tuple[str, Path]]:
    """(row_key, png_path) per generation, in row-major order."""
    ckpt = default_checkpoints()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_req = GenerationRequest(
        dit=ckpt.dit,
        vae=ckpt.vae,
        text_encoder=ckpt.text_encoder,
        prompt="placeholder",
        save_path=str(samples_dir / "placeholder.png"),
        infer_steps=args.steps,
        guidance_scale=args.cfg,
        seed=0,
        lora_weight=[args.adapter],
        lora_multiplier=[args.multiplier],
        # Block-compile is the blessed fast path; the bench renders ~6 distinct
        # bucket shapes x many steps each, so per-shape warmup amortizes (and
        # the inductor cache persists across runs).
        extra_argv=("--compile_blocks",) if args.compile_blocks else (),
    )
    base_args = base_req.to_args()
    base_args.device = device
    gen_settings = get_generation_settings(base_args)
    shared_models = load_shared_models(base_args)
    shared_models["conds_cache"] = {}

    pending: list[tuple[argparse.Namespace, torch.Tensor]] = []
    outputs: list[tuple[str, Path]] = []
    total = len(rows) * len(args.seeds)
    for row in rows:
        with Image.open(row.ref_path) as ref:
            w, h = ref.size
        for seed in args.seeds:
            out = samples_dir / f"{row.artist}_{row.stem}_{row.tag}_s{seed}.png"
            pa = dataclasses.replace(
                base_req,
                prompt=row.caption,
                seed=seed,
                image_size=(h, w),
                save_path=str(out),
            ).to_args()
            pa.device = device
            print(
                f"[{len(pending) + 1}/{total}] {row.artist}/{row.stem} "
                f"({row.tag}) seed={seed} {w}x{h}",
                flush=True,
            )
            latent = generate(pa, gen_settings, shared_models)
            pending.append((pa, latent.detach().to("cpu")))
            outputs.append((f"{row.artist}/{row.stem}", out))

    # Lazy-load discipline: drop DiT + text encoder before the VAE comes up.
    shared_models.clear()
    del gen_settings
    clean_memory_on_device(device)

    vae = load_vae(
        base_args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=base_args.vae_chunk_size,
        disable_cache=base_args.vae_disable_cache,
        dtype=torch.bfloat16,
        eval=True,
    )
    # decode_to_pil, not save_output — save_output treats save_path as a
    # DIRECTORY (inference.py batch semantics) and buries the PNG under a
    # timestamped name; the sheet builder needs exact per-sample paths.
    for pa, latent in pending:
        decode_to_pil(vae, latent, device).save(pa.save_path)
    del vae
    clean_memory_on_device(device)
    return outputs


# ---------------------------------------------------------------------------
# Contact sheet
# ---------------------------------------------------------------------------


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow
        return ImageFont.load_default()


def build_sheet(
    rows: list[Row], seeds: list[int], samples_dir: Path, out_path: Path, cell_h: int
) -> None:
    pad, strip_h, header_h = 8, 40, 28
    font, small = _font(18), _font(14)

    grid: list[tuple[Row, list[Image.Image]]] = []
    for row in rows:
        cells = [Image.open(row.ref_path).convert("RGB")]
        for seed in seeds:
            p = samples_dir / f"{row.artist}_{row.stem}_{row.tag}_s{seed}.png"
            cells.append(Image.open(p).convert("RGB"))
        cells = [
            c.resize((max(1, round(c.width * cell_h / c.height)), cell_h))
            for c in cells
        ]
        grid.append((row, cells))

    row_ws = [
        sum(c.width for c in cells) + pad * (len(cells) + 1) for _r, cells in grid
    ]
    sheet_w = max(row_ws)
    sheet_h = header_h + len(grid) * (strip_h + cell_h + pad) + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)

    draw.text(
        (pad, 6),
        "cols: source | " + " | ".join(f"seed {s}" for s in seeds),
        fill=(180, 180, 180),
        font=small,
    )
    y = header_h
    for row, cells in grid:
        label = f"{row.artist}/{row.stem}  [{row.tag}]  {row.caption}"
        draw.text((pad, y + 10), label[:220], fill=(235, 235, 235), font=font)
        y += strip_h
        x = pad
        for i, cell in enumerate(cells):
            sheet.paste(cell, (x, y))
            if i == 0:  # mark the reference column
                draw.rectangle(
                    [x, y, x + cell.width - 1, y + cell.height - 1],
                    outline=(255, 200, 60),
                    width=3,
                )
            x += cell.width + pad
        y += cell_h + pad
    sheet.save(out_path)
    print(f"contact sheet → {out_path}", flush=True)


def write_summary(run_dir: Path, args, rows: list[Row]) -> Path:
    lines = [
        "# Eyeball contact sheet",
        "",
        f"- adapter: `{args.adapter}`",
        f"- seeds {args.seeds} · steps {args.steps} · cfg {args.cfg} · "
        f"multiplier {args.multiplier}",
        "",
        "| row | artist/stem | tag | caption |",
        "|---|---|---|---|",
    ]
    for i, r in enumerate(rows):
        lines.append(f"| {i} | {r.artist}/{r.stem} | {r.tag} | {r.caption[:120]} |")
    lines += [
        "",
        "## How to read it",
        "",
        "- **MEMBER rows**: renders that collapse toward the source column "
        "(same composition/pose, not just style) = visible memorization — the "
        "eyeball confirmation of a high loss-gap AUC.",
        "- **HOLDOUT row**: should be a fresh, plausible image in the target "
        "style; snapping to some *other* training frame is a red flag.",
        "- **OTHER rows**: caption content should survive (style transfer is "
        "expected and fine); drifting toward target-artist training frames "
        "regardless of caption = mode collapse.",
        "- Seed columns near-identical to each other on a row = low diversity "
        "for that caption (memorization's usual companion).",
        "",
    ]
    out = run_dir / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--adapter", required=True, help="LoRA checkpoint to eyeball.")
    ap.add_argument("--method", default="lora")
    ap.add_argument("--preset", default="default")
    ap.add_argument(
        "--target_artist",
        default=None,
        help="Trained artist (default: derived from the snapshot's path_pattern).",
    )
    ap.add_argument(
        "--other_artists",
        nargs="+",
        default=None,
        help="Never-trained artists for the OTHER rows (default: 2 biggest dirs).",
    )
    ap.add_argument("--prompts_per_artist", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=3.5)
    ap.add_argument("--multiplier", type=float, default=1.0)
    ap.add_argument("--pick_seed", type=int, default=0, help="Caption-choice RNG.")
    ap.add_argument(
        "--compile_blocks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Block-compile the DiT for faster sampling (--no-compile_blocks "
        "to skip the first-run warmup).",
    )
    ap.add_argument("--cell_height", type=int, default=352)
    ap.add_argument("--label", default=None)
    ap.add_argument(
        "--plan_only",
        action="store_true",
        help="Print the prompt-row plan and exit (no GPU work).",
    )
    args = ap.parse_args()

    snap = _read_snapshot_knobs(args.adapter)
    target = args.target_artist
    if target is None:
        pattern = snap.get("path_pattern") or ""
        target = pattern.split("/")[0] if "/" in pattern else None
        if not target:
            ap.error("--target_artist required (snapshot has no path_pattern)")

    others = args.other_artists or default_other_artists(target, 2)
    print(f"target={target} others={others}", flush=True)

    rng = Random(args.pick_seed)
    rows = target_rows(args, target, rng)
    rows += other_artist_rows(others, args.prompts_per_artist, rng)
    for r in rows:
        print(f"  {r.tag:<8} {r.artist}/{r.stem}: {r.caption[:100]}", flush=True)
    if args.plan_only:
        return

    label = args.label or f"eyeball_{Path(args.adapter).stem}"
    run_dir = make_run_dir("memorization", label)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    render_all(args, rows, samples_dir)
    sheet = run_dir / "contact_sheet.png"
    build_sheet(rows, args.seeds, samples_dir, sheet, args.cell_height)
    summary = write_summary(run_dir, args, rows)

    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=label,
        metrics={
            "n_rows": len(rows),
            "n_samples": len(rows) * len(args.seeds),
            "target_artist": target,
            "other_artists": others,
        },
        artifacts=[sheet, summary, samples_dir],
        extra={
            "rows": [
                {
                    "artist": r.artist,
                    "stem": r.stem,
                    "tag": r.tag,
                    "caption": r.caption,
                    "ref": str(r.ref_path),
                }
                for r in rows
            ]
        },
    )
    print(f"done → {run_dir}", flush=True)


if __name__ == "__main__":
    main()
