"""Phase 2.5 — does the subject_edit descriptor follow edit instructions?

The subject probe (`run_subject_probe.py`) only exercises *retrieval*: cond =
image A, prompt = B's full caption. This probe replays the subject_edit
adapter's own training task instead:

    cond   = image A of a character          (--easycontrol_image)
    prompt = the mined TAG DELTA between A and B's captions
             (additions in B's order + `-`-prefixed removals)

The prompt is an instruction, not a description — and the character NAME tag
cancels out of it by construction, so identity in the render can only come
from the cond stream. Verdict is render-judged on two axes at once:

    1. do the *instructed* changes land (additions appear, removals vanish)?
    2. does identity/appearance hold (cond attributes not named in the delta
       carry over)?

Arms per pair:
    noec        delta prompt only, no adapter — control: what the fragmentary
                delta tags alone produce (no identity source at all)
    ec_b<off>   same seed/prompt + cond = A   — one arm per --b_offsets entry

The contact sheet is cond | real target (B = mining's ground-truth "edited
result") | noec | ec_*. Pairs come from the miner's manifest, so they are
TRAIN pairs: an upper bound on instruction-following — a failure here is
decisive, a success still owes a held-out check.

Default pair scope is same-artist (the miner's majority class): style held
constant means every visible change is attributable to the instruction. Use
--pair_scope to widen.

Usage:
    uv run python project/directedit_ec/bench/run_edit_probe.py --n_pairs 3
    uv run python project/directedit_ec/bench/run_edit_probe.py --b_offsets 0,2,4
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from library.env import default_checkpoints  # noqa: E402
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

PAIRS_JSON = (
    REPO_ROOT / "post_image_dataset" / "easycontrol" / "subject_edit" / "pairs.json"
)
RESIZED_DIR = REPO_ROOT / "post_image_dataset" / "resized"
DEFAULT_EC_WEIGHT = (
    REPO_ROOT / "output" / "ckpt" / "anima_easycontrol_subject_edit.safetensors"
)




def resolve(rel: str) -> Path:
    """'artist/stem' → the resized PNG (the miner's manifest is dir-relative)."""
    return RESIZED_DIR / f"{rel}.png"


def rating_of(rel: str) -> str:
    """First caption tag = the Anima rating band (safe/sensitive/nsfw/explicit)."""
    txt = RESIZED_DIR / f"{rel}.txt"
    if not txt.is_file():
        return ""
    return txt.read_text(encoding="utf-8").split(",", 1)[0].strip()


def pick_pairs(
    n: int, seed: int, scope: str, max_delta: int, ratings: set[str] | None
) -> list[dict]:
    manifest = json.loads(PAIRS_JSON.read_text(encoding="utf-8"))
    pairs = manifest["pairs"]
    if scope == "same_artist":
        pairs = [p for p in pairs if p["same_artist"]]
    elif scope == "cross_artist":
        pairs = [p for p in pairs if not p["same_artist"]]
    # Legibility cap: a 3-row render-judged sheet needs deltas a human can
    # actually check off, not the corpus median ~31 tags.
    pairs = [p for p in pairs if p["n_additions"] + p["n_removals"] <= max_delta]
    ok = []
    for p in pairs:
        t, c = resolve(p["target"]), resolve(p["cond"])
        if not (t.is_file() and c.is_file() and p.get("delta_caption")):
            continue
        if ratings and not (
            rating_of(p["target"]) in ratings and rating_of(p["cond"]) in ratings
        ):
            continue
        ok.append(p)
    if not ok:
        raise SystemExit(
            f"no resolvable pairs in {PAIRS_JSON} "
            f"(scope={scope}, max_delta={max_delta}, ratings={sorted(ratings or ())})"
        )
    # One pair per character, so n pairs = n distinct identities.
    by_char: dict[str, dict] = {}
    for p in ok:
        by_char.setdefault(p["character"], p)
    chosen = sorted(by_char.values(), key=lambda p: p["target"])
    random.Random(seed).shuffle(chosen)
    return chosen[:n]


def arm_argv(
    prompt: str,
    cond: Optional[Path],
    b_offset: Optional[float],
    out_dir: Path,
    args,
    ck,
) -> list[str]:
    argv = [
        sys.executable,
        str(REPO_ROOT / "inference.py"),
        "--dit",
        ck.dit,
        "--text_encoder",
        ck.text_encoder,
        "--vae",
        ck.vae,
        "--vae_chunk_size",
        "64",
        "--vae_disable_cache",
        "--attn_mode",
        "flash",
        "--prompt",
        prompt,
        "--negative_prompt",
        "worst quality, low quality, score_1, score_2, score_3, blurry, "
        "jpeg artifacts, sepia",
        "--image_size",
        str(args.height),
        str(args.width),
        "--infer_steps",
        str(args.infer_steps),
        "--flow_shift",
        "3.0",
        "--sampler",
        "euler",
        "--guidance_scale",
        str(args.guidance_scale),
        "--seed",
        str(args.seed),
        "--save_path",
        str(out_dir),
    ]
    if cond is not None:
        argv += [
            "--easycontrol_weight",
            str(args.ec_weight),
            "--easycontrol_image",
            str(cond),
            "--easycontrol_image_match_size",
        ]
        if b_offset:
            argv += ["--easycontrol_b_offset", str(b_offset)]
    return argv


def make_sheet(rows: list[tuple[str, list[tuple[str, Optional[Path]]]]], out: Path):
    from PIL import Image, ImageDraw

    thumb_h, label_h = 384, 22
    cols = max(len(r[1]) for r in rows)
    thumbs = []
    for _, cells in rows:
        row = []
        for label, p in cells:
            if p is not None and Path(p).is_file():
                im = Image.open(p).convert("RGB")
                w = int(im.size[0] * thumb_h / im.size[1])
                row.append((label, im.resize((w, thumb_h), Image.LANCZOS)))
            else:
                row.append((label, None))
        thumbs.append(row)
    cell_w = max(
        (im.size[0] for row in thumbs for _, im in row if im is not None),
        default=thumb_h,
    )
    canvas = Image.new(
        "RGB", (cell_w * cols, (thumb_h + label_h) * len(rows)), (24, 24, 24)
    )
    draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(thumbs):
        for c, (label, im) in enumerate(row):
            x, y = c * cell_w, r * (thumb_h + label_h)
            if im is not None:
                canvas.paste(im, (x + (cell_w - im.size[0]) // 2, y + label_h))
            draw.text((x + 4, y + 4), label, fill=(255, 255, 255))
    canvas.save(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--n_pairs", type=int, default=3)
    p.add_argument("--ec_weight", default=str(DEFAULT_EC_WEIGHT))
    p.add_argument(
        "--b_offsets",
        default="0,2,3,4",
        help="Comma-separated b_cond offsets; one ec_b<off> arm each. Trained "
        "b_cond_init is -4, so the subject-v2 retrieval band (+2/+3) is the "
        "expected engagement zone.",
    )
    p.add_argument("--infer_steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=1216)
    p.add_argument("--timeout", type=int, default=1200)
    p.add_argument(
        "--pair_scope",
        choices=("any", "same_artist", "cross_artist"),
        default="same_artist",
        help="same_artist (default) holds style constant so changes are "
        "attributable to the instruction",
    )
    p.add_argument(
        "--max_delta",
        type=int,
        default=20,
        help="only probe pairs whose instruction has at most this many tags "
        "(judgeability cap; the corpus median is ~31)",
    )
    p.add_argument(
        "--rating",
        default="any",
        help="comma-separated Anima rating bands (safe,sensitive,nsfw,explicit) "
        "both endpoints of a pair must fall in; 'any' disables the filter",
    )
    p.add_argument("--label", default="phase2p5-edit-probe")
    args = p.parse_args()

    if not Path(args.ec_weight).is_file():
        raise SystemExit(f"EC checkpoint not found: {args.ec_weight}")
    start_heartbeat()
    ck = default_checkpoints()
    offsets = [float(s) for s in args.b_offsets.split(",") if s.strip()]
    ratings = (
        None
        if args.rating == "any"
        else {r.strip() for r in args.rating.split(",") if r.strip()}
    )
    pairs = pick_pairs(
        args.n_pairs, args.seed, args.pair_scope, args.max_delta, ratings
    )

    run_dir = make_run_dir(
        "directedit_ec",
        root=Path(__file__).resolve().parent / "results",
        label=args.label,
    )
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    logger.info("Run dir: %s", run_dir)
    for pair in pairs:
        logger.info(
            "Pair %s (%s -> %s): %s",
            pair["character"],
            pair["cond"],
            pair["target"],
            pair["delta_caption"],
        )

    arms: list[tuple[str, Optional[float]]] = [("noec", None)]
    arms += [(f"ec_b{o:g}", o) for o in offsets]

    per_pair = []
    sheet_rows = []
    for pair in pairs:
        cond = resolve(pair["cond"])
        target = resolve(pair["target"])
        prompt = pair["delta_caption"]
        key = pair["character"].replace(" ", "_").replace("/", "_")
        rec = {
            "character": pair["character"],
            "cond": str(cond),
            "target": str(target),
            "same_artist": pair["same_artist"],
            "n_additions": pair["n_additions"],
            "n_removals": pair["n_removals"],
            "prompt": prompt,
            "arms": {},
        }
        cells: list[tuple[str, Optional[Path]]] = [
            ("cond (A)", cond),
            ("real target (B)", target),
        ]
        for name, off in arms:
            out_dir = run_dir / "renders" / key / name
            out_dir.mkdir(parents=True, exist_ok=True)
            argv = arm_argv(
                prompt,
                cond if name != "noec" else None,
                off,
                out_dir,
                args,
                ck,
            )
            log_path = logs_dir / f"{key}_{name}.log"
            logger.info("[%s/%s] running", key, name)
            t0 = time.time()
            try:
                with log_path.open("w") as lf:
                    lf.write(" ".join(argv) + "\n\n")
                    lf.flush()
                    proc = subprocess.run(
                        argv,
                        cwd=REPO_ROOT,
                        stdout=lf,
                        stderr=subprocess.STDOUT,
                        timeout=args.timeout,
                    )
                ok = proc.returncode == 0
            except subprocess.TimeoutExpired:
                ok = False
                logger.error("[%s/%s] TIMEOUT", key, name)
            pngs = sorted(q for q in out_dir.glob("*.png"))
            out_png = pngs[-1] if (ok and pngs) else None
            rec["arms"][name] = {
                "ok": out_png is not None,
                "wall_s": round(time.time() - t0, 1),
                "png": str(out_png.relative_to(run_dir)) if out_png else None,
            }
            cells.append((name, out_png))
            logger.info(
                "[%s/%s] %s wall=%.0fs",
                key,
                name,
                "ok" if out_png else "FAILED",
                time.time() - t0,
            )
        per_pair.append(rec)
        sheet_rows.append((key, cells))

    sheet = run_dir / "grid.png"
    make_sheet(sheet_rows, sheet)
    write_result(
        run_dir,
        script=str(Path(__file__).relative_to(REPO_ROOT)),
        args=args,
        metrics={
            "ec_weight": args.ec_weight,
            "arms": [a for a, _ in arms],
            "per_pair": per_pair,
            "note": "render-judged on (1) instructed changes landing and "
            "(2) identity holding; train-set pairs = upper bound",
        },
        artifacts=["grid.png"],
    )
    logger.info("Human verdict artifact: %s", sheet)
    print(json.dumps({"run_dir": str(run_dir)}))


if __name__ == "__main__":
    main()
